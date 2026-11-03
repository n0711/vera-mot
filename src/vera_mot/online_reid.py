"""CPU-only appearance-based online re-identification of vehicle detections.

Experimental baseline independent of YOLOX/BoT-SORT and their saved weights.
No ground truth, identity accuracy, or UAVDT performance is implied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class AssignedDetection:
    """Detection output with a persistent within-stream association ID."""

    track_id: int
    bbox: tuple[float, float, float, float]  # xyxy
    score: float
    matched_existing: bool


@dataclass
class _Track:
    track_id: int
    bbox: np.ndarray
    feature: np.ndarray
    last_frame: int


def _normalize(rows: np.ndarray) -> np.ndarray:
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        raise ValueError("appearance embeddings must be a finite 2D array")
    lengths = np.linalg.norm(rows, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("zero or non-finite appearance embedding")
    return rows / lengths[:, None]


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    left_top = np.maximum(first[:2], second[:2])
    right_bottom = np.minimum(first[2:], second[2:])
    side = np.maximum(0, right_bottom - left_top)
    intersection = float(side[0] * side[1])
    a = float(np.prod(first[2:] - first[:2]))
    b = float(np.prod(second[2:] - second[:2]))
    return intersection / (a + b - intersection) if a + b > intersection else 0.0


class OnlineReID:
    """Global assignment using normalized appearance, IoU, and displacement gates.

    Only matches within a single video stream. Association IDs are hypotheses,
    never verified ground-truth identities. Scores are not learned or calibrated.
    """

    def __init__(
        self,
        *,
        min_similarity: float = 0.75,
        max_cost: float = 0.55,
        appearance_weight: float = 0.85,
        max_age: int = 30,
        max_displacement: float = 5.0,
        min_score: float = 0.25,
        feature_momentum: float = 0.9,
    ) -> None:
        for name, value in (
            ("min_similarity", min_similarity), ("max_cost", max_cost),
            ("appearance_weight", appearance_weight), ("min_score", min_score),
            ("feature_momentum", feature_momentum),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if not isinstance(max_age, int) or isinstance(max_age, bool) or max_age < 1:
            raise ValueError("max_age must be a positive integer")
        if not np.isfinite(max_displacement) or max_displacement <= 0:
            raise ValueError("max_displacement must be positive")
        self.min_similarity = min_similarity
        self.max_cost = max_cost
        self.appearance_weight = appearance_weight
        self.max_age = max_age
        self.max_displacement = max_displacement
        self.min_score = min_score
        self.feature_momentum = feature_momentum
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self._frame = 0
        self._dimension: int | None = None

    @property
    def active_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def update(
        self,
        frame: int,
        boxes: np.ndarray,
        embeddings: np.ndarray,
        scores: np.ndarray,
    ) -> list[AssignedDetection]:
        """Associate xyxy boxes and embeddings, returning detections in input order.

        Empty frames must also call update to advance expiration. Lower-score
        detections are ignored. No mutation of caller arrays or past results.
        """
        if not isinstance(frame, int) or isinstance(frame, bool) or frame <= self._frame:
            raise ValueError("frame must be a strictly increasing positive integer")
        boxes = np.asarray(boxes, dtype=np.float64)
        embeddings = np.asarray(embeddings, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)
        if boxes.shape != (len(boxes), 4) or boxes.ndim != 2:
            raise ValueError("boxes must have shape (N, 4) in xyxy format")
        if embeddings.ndim != 2 or embeddings.shape[0] != len(boxes) or embeddings.shape[1] == 0:
            raise ValueError("embeddings must have shape (N, F) with F > 0")
        if scores.shape != (len(boxes),) or not np.all(np.isfinite(scores)):
            raise ValueError("scores must be finite with shape (N,)")
        if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("boxes must have finite positive width and height")
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("scores must be in [0, 1]")
        if self._dimension is not None and embeddings.shape[1] != self._dimension:
            raise ValueError("embedding dimension changed within a stream")
        # Only use features from accepted detections; rejected low-score entries
        # cannot corrupt the association state or embeddings.
        accepted = np.where(scores >= self.min_score)[0]
        features = _normalize(embeddings[accepted]) if accepted.size else np.empty((0, embeddings.shape[1]))
        if features.size and self._dimension is None:
            self._dimension = features.shape[1]
        # Validate everything before changing any state.
        self._frame = frame
        self._tracks = {key: value for key, value in self._tracks.items()
                        if frame - value.last_frame <= self.max_age}
        if not len(accepted):
            return []
        existing = [self._tracks[key] for key in sorted(self._tracks)]
        costs = np.full((len(existing), len(accepted)), 1e6, dtype=np.float64)
        for i, track in enumerate(existing):
            gap = frame - track.last_frame
            a_center = (track.bbox[:2] + track.bbox[2:]) / 2
            for j, original_index in enumerate(accepted):
                box = boxes[original_index]
                b_center = (box[:2] + box[2:]) / 2
                scale = max(*(track.bbox[2:] - track.bbox[:2]), *(box[2:] - box[:2]))
                if np.linalg.norm(a_center - b_center) > self.max_displacement * scale * gap:
                    continue
                similarity = float(np.clip(track.feature @ features[j], -1, 1))
                if similarity < self.min_similarity:
                    continue
                cost = self.appearance_weight * (1 - similarity) + (
                    1 - self.appearance_weight
                ) * (1 - _iou(track.bbox, box))
                if cost <= self.max_cost:
                    costs[i, j] = cost
        matches: dict[int, int] = {}
        if costs.size:
            rows, cols = linear_sum_assignment(costs)
            for i, j in zip(rows, cols):
                if costs[i, j] < 1e6:
                    matches[j] = existing[i].track_id
        result: list[AssignedDetection] = []
        for j, original_index in enumerate(accepted):
            box = boxes[original_index].copy()
            feature = features[j].copy()
            matched = j in matches
            if matched:
                track_id = matches[j]
                old = self._tracks[track_id]
                smoothed = self.feature_momentum * old.feature + (
                    1 - self.feature_momentum
                ) * feature
                norm = np.linalg.norm(smoothed)
                # A 180-degree feature reversal must not corrupt state.
                feature = smoothed / norm if norm > 1e-12 else feature
            else:
                track_id = self._next_id
                self._next_id += 1
            self._tracks[track_id] = _Track(track_id, box, feature, frame)
            result.append(AssignedDetection(track_id, tuple(float(v) for v in box),
                                            float(scores[original_index]), matched))
        return result
