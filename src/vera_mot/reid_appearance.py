"""Explicit image-to-appearance features for weight-free and local-weight ReID.

Histogram is an unsupervised engineering baseline. ResNet18 uses generic
ImageNet classification features, NOT vehicle-ReID-trained weights.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def _validate_crop(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("expected uint8 BGR frame shaped (H, W, 3)")
    box = np.asarray(xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.all(np.isfinite(box)) or np.any(box[2:] <= box[:2]):
        raise ValueError("invalid xyxy box")
    height, width = frame.shape[:2]
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    x1, x2 = np.clip([x1, x2], 0, width)
    y1, y2 = np.clip([y1, y2], 0, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("appearance crop is empty or smaller than 2x2 pixels")
    return frame[y1:y2, x1:x2]


class HSVHistogramEmbedder:
    """Deterministic 5-part HSV histogram (global and 2x2 spatial cells)."""

    name = "hsv-spatial-histogram-v1"
    dimension = 640

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        features = []
        for box in boxes:
            hsv = cv2.cvtColor(_validate_crop(frame, box), cv2.COLOR_BGR2HSV)
            height, width = hsv.shape[:2]
            # Keep global appearance stable even for very small crops.
            slices = [(0, height, 0, width)]
            for y1, y2 in ((0, height // 2), (height // 2, height)):
                for x1, x2 in ((0, width // 2), (width // 2, width)):
                    slices.append((y1, y2, x1, x2))
            parts = []
            for y1, y2, x1, x2 in slices:
                region = hsv[y1:y2, x1:x2]
                hist = cv2.calcHist([region], [0, 1, 2], None, [8, 4, 4],
                                    [0, 180, 0, 256, 0, 256]).ravel()
                hist /= max(1.0, float(np.sum(hist)))
                parts.append(hist)
            vector = np.concatenate(parts).astype(np.float32)
            norm = np.linalg.norm(vector)
            if norm == 0 or not np.isfinite(norm):
                raise ValueError("invalid appearance histogram")
            features.append(vector / norm)
        return np.stack(features) if features else np.empty((0, self.dimension), dtype=np.float32)


class ResNet18Embedder:
    """Generic ImageNet-pretrained ResNet18 penultimate features (not trained ReID).

    Local, explicitly provided torchvision ResNet18 state_dict only: no silent
    download, random initialization or fallback. No image data leaves the host.
    """

    name = "resnet18-imagenet-generic-v1"
    dimension = 512

    def __init__(self, weights: Path, device: str = "cpu", expected_sha256: str | None = None):
        if not Path(weights).is_file():
            raise FileNotFoundError(f"local ResNet18 weights required: {weights}")
        import torch  # optional heavy dependency
        from torchvision.models import resnet18

        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        hasher = hashlib.sha256()
        with Path(weights).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
            raise ValueError("ResNet18 weight SHA-256 mismatch")
        try:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError("invalid local ResNet18 state dict") from exc
        if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
            raise ValueError("ResNet18 weights must be a state dictionary")
        model = resnet18(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()
        self._model = model.to(device).eval()
        self._torch = torch
        self._device = device
        self.weights_sha256 = digest

    def encode(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        torch = self._torch
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if not len(boxes):
            return np.empty((0, self.dimension), dtype=np.float32)
        # Matches torchvision ResNet18 ImageNet inference normalization.
        patches = []
        for box in boxes:
            bgr = _validate_crop(frame, box)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # torchvision ImageNet ResNet18 reference uses resize shorter side 256
            # then center crop 224; preserving that operation per bounding box.
            height, width = rgb.shape[:2]
            # Bound memory use even for extremely elongated detections.
            scale = min(256.0 / min(height, width), 1024.0 / max(height, width))
            resized = cv2.resize(rgb, (max(1, int(round(width * scale))),
                                       max(1, int(round(height * scale)))),
                                 interpolation=cv2.INTER_CUBIC)
            # For extreme aspect ratios, pad to a square before center crop.
            h, w = resized.shape[:2]
            pad_top, pad_bottom = max(0, (256 - h) // 2), max(0, 256 - h - max(0, (256 - h) // 2))
            pad_left, pad_right = max(0, (256 - w) // 2), max(0, 256 - w - max(0, (256 - w) // 2))
            resized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=(0, 0, 0))
            h, w = resized.shape[:2]
            y, x = (h - 224) // 2, (w - 224) // 2
            rgb = resized[y:y + 224, x:x + 224]
            tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255
            tensor = (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / (
                torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            )
            patches.append(tensor)
        with torch.inference_mode():
            features = self._model(torch.stack(patches).to(self._device))
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        vectors = features.cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(vectors)) or np.any(np.linalg.norm(vectors, axis=1) <= 0):
            raise ValueError("invalid neural embeddings")
        return vectors
