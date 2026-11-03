"""Inject project-owned image embeddings into the PINNED BoT-SORT matcher.

The upstream tracker/assignment algorithm is unchanged. Runtime integration
requires the explicit local compatibility patch and pinned submodule; the CPU
contract functions can be tested without importing the upstream dependencies.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PINNED_BOTSORT_REVISION = "251985436d6712aaf682aaaf5f71edb4987224bd"


class BoTSORTAppearanceEncoder:
    """Adapter from project-owned .encode(frame, xyxy) to BoT-SORT .inference."""

    def __init__(self, embedder) -> None:
        if not callable(getattr(embedder, "encode", None)):
            raise TypeError("embedder must implement encode(frame, xyxy)")
        if not isinstance(getattr(embedder, "dimension", None), int) or embedder.dimension <= 0:
            raise ValueError("embedder requires a positive fixed dimension")
        self.embedder = embedder
        self.dimension = embedder.dimension
        self.calls = 0
        self.features_extracted = 0

    def inference(self, image: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """Return a finite normalized (N, F) matrix, including empty batches."""
        boxes = np.asarray(detections, dtype=np.float64)
        if boxes.shape == (0,):
            boxes = np.empty((0, 4), dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("BoT-SORT encoder expects xyxy detections (N,4)")
        if not np.all(np.isfinite(boxes)) or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("invalid appearance box from tracker")
        if not len(boxes):
            features = np.empty((0, self.dimension), dtype=np.float64)
        else:
            features = np.asarray(self.embedder.encode(image, boxes.copy()), dtype=np.float64)
        if features.shape != (len(boxes), self.dimension) or not np.all(np.isfinite(features)):
            raise ValueError("appearance encoder returned invalid shape or non-finite features")
        if len(features):
            norms = np.linalg.norm(features, axis=1)
            if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
                raise ValueError("appearance encoder produced zero/non-finite feature")
            features = features / norms[:, None]
        self.calls += 1
        self.features_extracted += len(boxes)
        return features.astype(np.float32, copy=True)


def make_tracker_args(
    *,
    name: str = "reid_test",
    cmc_method: str = "none",
    match_thresh: float = 0.8,
    track_high_thresh: float = 0.35,
    track_low_thresh: float = 0.10,
    new_track_thresh: float = 0.45,
    track_buffer: int = 30,
    proximity_thresh: float = 0.5,
    appearance_thresh: float = 0.25,
) -> SimpleNamespace:
    """Make the frozen baseline-like arguments without enabling ReID at init."""
    if cmc_method not in ("none", "sparseOptFlow", "orb", "ecc"):
        raise ValueError("unsupported GMC method")
    for key, val in (
        ("match_thresh", match_thresh), ("track_high_thresh", track_high_thresh),
        ("track_low_thresh", track_low_thresh), ("new_track_thresh", new_track_thresh),
        ("proximity_thresh", proximity_thresh), ("appearance_thresh", appearance_thresh)
    ):
        if not np.isfinite(val) or not 0 <= val <= 1:
            raise ValueError(f"{key} must be between 0 and 1")
    if not track_low_thresh < track_high_thresh <= new_track_thresh:
        raise ValueError("requires track_low < track_high <= new_track")
    if not isinstance(track_buffer, int) or track_buffer < 1:
        raise ValueError("track_buffer must be positive")
    return SimpleNamespace(
        track_high_thresh=track_high_thresh,
        track_low_thresh=track_low_thresh,
        new_track_thresh=new_track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        proximity_thresh=proximity_thresh,
        appearance_thresh=appearance_thresh,
        with_reid=False,
        fast_reid_config="",
        fast_reid_weights="",
        device="cpu",
        cmc_method=cmc_method,
        name=name,
        ablation=False,
        mot20=False,
    )


def attach_project_encoder(tracker, embedder) -> BoTSORTAppearanceEncoder:
    """Opt in to upstream ReID association only after tracker construction.

    Avoids the upstream FastReIDInterface initializer that otherwise requires
    unavailable pretrained model files. Does not override upstream matching.
    """
    if not hasattr(tracker, "args") or not hasattr(tracker.args, "with_reid"):
        raise TypeError("BoT-SORT tracker with args.with_reid required")
    if tracker.args.with_reid or hasattr(tracker, "encoder"):
        raise ValueError("tracker ReID encoder is already installed")
    encoder = BoTSORTAppearanceEncoder(embedder)
    tracker.encoder = encoder
    tracker.args.with_reid = True
    return encoder


def load_pinned_botsort(project_root: Path):
    """Load upstream BoT-SORT only after verifying the known ReID bug fixes."""
    root = Path(project_root).resolve() / "baselines/BoT-SORT"
    if not (root / ".git").exists():
        raise RuntimeError("BoT-SORT submodule not initialized: git submodule update --init --recursive")
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True
    ).stdout.strip()
    if revision != PINNED_BOTSORT_REVISION:
        raise RuntimeError(f"unexpected BoT-SORT revision: {revision}")
    # These source checks are required because unpatched STrack calls its
    # update_features() before deque initialization and uses removed np.float.
    bot_sort_text = (root / "tracker/bot_sort.py").read_text(encoding="utf-8")
    matching_text = (root / "tracker/matching.py").read_text(encoding="utf-8")
    if (
        "dtype=np.float" in bot_sort_text
        or "np.float" in matching_text
        or "self.features = deque([], maxlen=feat_history)" not in bot_sort_text
        or "self.update_features(feat)" not in bot_sort_text
        or bot_sort_text.index("self.features = deque([], maxlen=feat_history)")
        > bot_sort_text.index("self.update_features(feat)")
    ):
        raise RuntimeError(
            "BoT-SORT ReID compatibility fixes absent; run "
            "bash scripts/apply_botsort_reid_patch.sh after baseline compatibility patch"
        )
    # Import only inside explicit runtime path. Unit tests and CI do not need
    # torch, lap, cython-bbox, YOLOX, or upstream FastReID.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from tracker.bot_sort import BoTSORT
    except ImportError as exc:
        raise RuntimeError(
            "BoT-SORT runtime dependencies unavailable; install the pinned baseline "
            "environment and verify the upstream compatibility patch"
        ) from exc
    return BoTSORT


def create_tracker(
    project_root: Path, *, embedder=None, frame_rate: int = 30,
    name: str = "reid_demo", cmc_method: str = "none"
):
    """Return (BoTSORT, appearance adapter or None), with no tracker code edits."""
    if not isinstance(frame_rate, int) or frame_rate < 1:
        raise ValueError("frame_rate must be positive integer")
    bot_sort = load_pinned_botsort(project_root)
    args = make_tracker_args(name=name, cmc_method=cmc_method)
    tracker = bot_sort(args, frame_rate=frame_rate)
    adapter = attach_project_encoder(tracker, embedder) if embedder is not None else None
    return tracker, adapter


def detections_to_botsort(rows: np.ndarray) -> np.ndarray:
    """Convert a validated frame's xyxy+score rows into BoT-SORT (N,5).

    This uses BoT-SORT's documented special case for 5-column detections,
    avoiding detector-specific objectness/class multiplication.
    """
    data = np.asarray(rows, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError("detections require (N,5) xyxy+score")
    if not np.all(np.isfinite(data)):
        raise ValueError("non-finite detection")
    if len(data) and (
        np.any(data[:, 2:4] <= data[:, 0:2]) or
        np.any((data[:, 4] < 0) | (data[:, 4] > 1))
    ):
        raise ValueError("invalid detector bounding box or score")
    return data.copy()
