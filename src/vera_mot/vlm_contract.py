"""Offline ingestion contract for external VLM annotations over VERA-MOT frames.

This does NOT run a VLM. It validates a reproducible model-output manifest
and exports only bounded aggregate semantic labels, not images or captions.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

MAX_MANIFEST_BYTES = 1_000_000
MAX_FRAMES = 10_000
MAX_LABELS_PER_FRAME = 10
SHA = re.compile(r"[a-f0-9]{64}\Z")
IDENT = re.compile(r"[A-Za-z0-9_.\-/]{1,128}\Z")
LABEL = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z")


def _unique(pairs):
    obj = {}
    for key, val in pairs:
        if key in obj:
            raise ValueError("duplicate annotation JSON key: " + key)
        obj[key] = val
    return obj


def _invalid_constant(value):
    raise ValueError("invalid annotation number: " + value)


def summarise_vlm_manifest(raw: bytes):
    """Validate declared inference provenance; do not certify model performance."""
    if not isinstance(raw, bytes) or len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("VLM manifest exceeds 1 MB or is not bytes")
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                         parse_constant=_invalid_constant)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("invalid UTF-8 VLM JSON manifest") from exc
    required = {"annotation_version", "sequence", "model_id", "model_revision",
                "prompt_sha256", "frames"}
    if not isinstance(obj, dict) or set(obj) != required:
        raise ValueError("unexpected or missing VLM manifest fields")
    if type(obj["annotation_version"]) is not int or obj["annotation_version"] != 1:
        raise ValueError("annotation_version must be 1")
    for key in ("sequence", "model_id", "model_revision"):
        if not isinstance(obj[key], str) or not IDENT.fullmatch(obj[key]):
            raise ValueError("invalid " + key)
    if not isinstance(obj["prompt_sha256"], str) or not SHA.fullmatch(obj["prompt_sha256"]):
        raise ValueError("invalid prompt_sha256")
    frames = obj["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("frames must contain 1-10000 entries")
    previous = 0
    counts = Counter()
    for row in frames:
        if not isinstance(row, dict) or set(row) != {"frame", "image_sha256", "labels"}:
            raise ValueError("invalid VLM frame record fields")
        frame = row["frame"]
        if type(frame) is not int or frame <= previous:
            raise ValueError("VLM frames must be positive and strictly increasing")
        previous = frame
        if not isinstance(row["image_sha256"], str) or not SHA.fullmatch(row["image_sha256"]):
            raise ValueError("invalid frame image_sha256")
        tags = row["labels"]
        if not isinstance(tags, list) or len(tags) > MAX_LABELS_PER_FRAME or any(
                not isinstance(t, str) or not LABEL.fullmatch(t) for t in tags):
            raise ValueError("invalid VLM labels")
        if len(set(tags)) != len(tags):
            raise ValueError("duplicate labels in one frame")
        counts.update(tags)
    return {
        "vlm_report_version": 1,
        "source": "declared-external-vlm-output",
        "sequence": obj["sequence"],
        "model_id": obj["model_id"],
        "model_revision": obj["model_revision"],
        "prompt_sha256": obj["prompt_sha256"],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "annotated_frames": len(frames),
        "first_annotated_frame": frames[0]["frame"],
        "last_annotated_frame": previous,
        "label_counts": dict(sorted(counts.items())),
        "claims": {
            "model_executed_by_this_tool": False,
            "labels_ground_truth_verified": False,
            "frames_time_aligned_to_ros3": False,
            "confidence_calibrated": False,
        },
        "interpretation": (
            "Aggregated labels declared by an external VLM workflow, not measured "
            "ground truth or verified inference. Image hashes and model metadata "
            "are supplied by the caller; provenance is not authenticated."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="external VLM inference manifest JSON v1")
    args = parser.parse_args(argv)
    try:
        with Path(args.manifest).open("rb") as handle:
            raw = handle.read(MAX_MANIFEST_BYTES + 1)
        result = summarise_vlm_manifest(raw)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
