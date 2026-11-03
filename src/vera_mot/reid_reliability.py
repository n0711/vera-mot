"""Conservative, interpretable crop-quality gate for experimental vehicle ReID.

This is a *heuristic* reliability proxy, not a learned identity confidence.
Only high-score detections already selected for upstream appearance extraction
are assessed. No image data is retained after evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityGateConfig:
    """Fixed provisional gates; no thresholds are learned from test identities."""

    threshold: float = 0.5
    reference_area: float = 32.0 * 32.0
    reference_laplacian_variance: float = 120.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if not np.isfinite(self.reference_area) or self.reference_area <= 0:
            raise ValueError("reference_area must be positive and finite")
        if (
            not np.isfinite(self.reference_laplacian_variance)
            or self.reference_laplacian_variance <= 0
        ):
            raise ValueError("reference_laplacian_variance must be positive and finite")


@dataclass(frozen=True)
class CropQuality:
    """A record of *observable crop proxies*, not a true visibility label."""

    score: float
    area_score: float
    focus_score: float
    in_frame_fraction: float
    appearance_allowed: bool


class CropQualityGate:
    """Fail-closed crop scoring with observable counters and no image retention.

    For each box:
      q_size = clip(clipped_area / reference_area, 0, 1)
      q_focus = clip(variance(Laplacian(gray)) / reference_variance, 0, 1)
      q_visible = clip(clipped_area / original_box_area, 0, 1)
      q = min(q_size, q_focus, q_visible)
      allowed = q >= threshold

    In-frame fraction is a proxy for truncation only, NOT an occlusion
    annotation; texture is only a Laplacian-variance focus proxy.
    """

    name = "vera-crop-quality-gate-v0"
    def __init__(self, config: QualityGateConfig | None = None):
        self.config = config or QualityGateConfig()
        self.assessed = 0
        self.allowed_count = 0
        self.blocked_count = 0
        self.quality_total = 0.0
        self.last_observations: tuple[CropQuality, ...] = ()

    @staticmethod
    def _validated(frame: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a uint8 BGR image (H,W,3)")
        if frame.shape[0] < 1 or frame.shape[1] < 1:
            raise ValueError("frame has zero height or width")
        boxes = np.asarray(boxes, dtype=np.float64)
        if boxes.shape == (0,):
            boxes = np.empty((0, 4), dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must be shape (N,4) in xyxy coordinates")
        if not np.all(np.isfinite(boxes)):
            raise ValueError("boxes contain nonfinite coordinates")
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("boxes must have positive area")
        return frame, boxes

    def score_detections(self, frame: np.ndarray, boxes: np.ndarray) -> tuple[CropQuality, ...]:
        """Compute scores without updating state; an invalid box fails closed."""
        frame, boxes = self._validated(frame, boxes)
        height, width = frame.shape[:2]
        out = []
        for box in boxes:
            raw_area = float(np.prod(box[2:] - box[:2]))
            x1 = int(np.clip(np.floor(box[0]), 0, width))
            y1 = int(np.clip(np.floor(box[1]), 0, height))
            x2 = int(np.clip(np.ceil(box[2]), 0, width))
            y2 = int(np.clip(np.ceil(box[3]), 0, height))
            clipped_area = float(max(0, x2 - x1) * max(0, y2 - y1))
            in_frame = float(np.clip(clipped_area / max(raw_area, 1e-12), 0, 1))
            area = float(np.clip(clipped_area / self.config.reference_area, 0, 1))
            if x2 - x1 < 2 or y2 - y1 < 2:
                focus = 0.0
            else:
                gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                focus = float(np.clip(
                    variance / self.config.reference_laplacian_variance, 0, 1
                )) if np.isfinite(variance) else 0.0
            score = float(min(area, focus, in_frame))
            out.append(CropQuality(
                score=score, area_score=area, focus_score=focus,
                in_frame_fraction=in_frame,
                appearance_allowed=score >= self.config.threshold,
            ))
        return tuple(out)

    def allowed(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Return one Boolean per box, record aggregate evidence for provenance."""
        evidence = self.score_detections(frame, boxes)
        accepted = np.fromiter(
            (x.appearance_allowed for x in evidence),
            dtype=np.bool_, count=len(evidence)
        )
        self.last_observations = evidence
        self.assessed += len(evidence)
        self.allowed_count += int(accepted.sum())
        self.blocked_count += len(evidence) - int(accepted.sum())
        self.quality_total += sum(x.score for x in evidence)
        return accepted

    def summary(self) -> dict:
        """Metadata only; never stores crops or claimed ground-truth quality."""
        return {
            "method": self.name,
            "threshold": self.config.threshold,
            "reference_area": self.config.reference_area,
            "reference_laplacian_variance": self.config.reference_laplacian_variance,
            "assessed": self.assessed,
            "appearance_allowed": self.allowed_count,
            "appearance_blocked": self.blocked_count,
            "mean_heuristic_quality": (
                self.quality_total / self.assessed if self.assessed else None
            ),
            "scope": "unvalidated_image_quality_proxy_not_identity_reliability",
        }
