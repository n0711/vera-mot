"""Read-only, bounded aggregate evidence exporter for VERA-MOT track results.

The input is a single sequence's MOTChallenge 10-column text output.
No images, boxes, crop paths, or per-track identities are exported.
This is tracking-output evidence, not a vision-language model or ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

REPORT_VERSION = 1
MAX_LINES = 1_000_000
MAX_LINE_BYTES = 4096
MAX_TRACK_IDS = 100_000


def _parse_row(raw: bytes, line_number: int):
    try:
        fields = raw.decode("utf-8").strip().split(",")
        if len(fields) != 10:
            raise ValueError("expected exactly 10 columns")
        numbers = [float(field) for field in fields]
        if not all(math.isfinite(n) for n in numbers):
            raise ValueError("non-finite row value")
        frame, track = numbers[:2]
        if frame < 1 or track < 1 or not frame.is_integer() or not track.is_integer():
            raise ValueError("frame and track identifiers must be positive integers")
        if numbers[4] <= 0 or numbers[5] <= 0:
            raise ValueError("bounding box dimensions must be positive")
        return int(frame), int(track)
    except (ValueError, OverflowError, UnicodeError) as exc:
        raise ValueError(f"line {line_number}: {exc}") from exc


def build_track_evidence(binary_stream, *, sequence: str):
    """Stream one sequence, validating all rows before returning a summary."""
    if not isinstance(sequence, str) or not sequence or len(sequence) > 100 or not all(
            ch.isascii() and (ch.isalnum() or ch in "_-.") for ch in sequence):
        raise ValueError("sequence must be 1-100 ASCII letters, digits, _, - or .")
    digest = hashlib.sha256()
    count = 0
    first = last = None
    observed_frames = 0
    seen_in_frame = set()
    tracks = set()
    for index in range(1, MAX_LINES + 2):
        raw = binary_stream.readline(MAX_LINE_BYTES + 1)
        if not isinstance(raw, bytes):
            raise ValueError("input stream must be binary")
        if not raw:
            break
        if index > MAX_LINES:
            raise ValueError("MOT output exceeds supported row count")
        if len(raw) > MAX_LINE_BYTES:
            raise ValueError(f"line {index}: line exceeds {MAX_LINE_BYTES} bytes")
        digest.update(raw)
        if not raw.strip():
            raise ValueError(f"line {index}: blank MOT row")
        frame, track = _parse_row(raw, index)
        if last is not None and frame < last:
            raise ValueError(f"line {index}: frames must be non-decreasing")
        if frame != last:
            observed_frames += 1
            seen_in_frame.clear()
        if track in seen_in_frame:
            raise ValueError(f"line {index}: duplicate track id in frame")
        seen_in_frame.add(track)
        tracks.add(track)
        if len(tracks) > MAX_TRACK_IDS:
            raise ValueError("too many distinct track identifiers")
        if first is None:
            first = frame
        last = frame
        count += 1
    if not count:
        raise ValueError("MOT output has no tracked detections")
    return {
        "vision_report_version": REPORT_VERSION,
        "source": "vera-mot",
        "source_format": "motchallenge-10-column",
        "parsed_to_eof": True,
        "sequence": sequence,
        "mot_sha256": digest.hexdigest(),
        "track_observation_rows": count,
        "distinct_sequence_local_track_ids": len(tracks),
        "frames_with_output": observed_frames,
        "first_output_frame": first,
        "last_output_frame": last,
        "mean_rows_per_output_frame": count / observed_frames,
        "claims": {
            "ground_truth_identity_accuracy": False,
            "sensor_capture_freshness": False,
            "gnss_independence": False,
            "vision_language_inference": False,
            "clock_synchronization": False,
        },
        "interpretation": (
            "Read-only aggregate of tracker output; IDs are sequence-local and may "
            "switch. No image understanding, identity correctness, unseen-frame "
            "inference, or time synchronization is established."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mot_file", help="existing sequence-local MOT result text")
    parser.add_argument("--sequence", required=True)
    args = parser.parse_args(argv)
    try:
        with Path(args.mot_file).open("rb") as handle:
            result = build_track_evidence(handle, sequence=args.sequence)
        print(json.dumps(result, allow_nan=False, sort_keys=True, indent=2))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
