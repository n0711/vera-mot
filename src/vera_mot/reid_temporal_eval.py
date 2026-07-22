"""Deterministic, sequence-local temporal retrieval evaluation for UAVDT."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from vera_mot.reid_data import ManifestError


PROTOCOL_VERSION = "uavdt-temporal-reid-v1"
MINIMUM_OBSERVATIONS = 4


@dataclass(frozen=True)
class Observation:
    sequence: str
    identity: str
    frame: int
    crop_path: str


@dataclass(frozen=True)
class IdentitySplit:
    sequence: str
    identity: str
    observation_count: int
    k: int
    gallery: tuple[Observation, ...]
    query: tuple[Observation, ...]
    excluded: tuple[Observation, ...]


def read_registry(path: Path) -> list[Observation]:
    """Read only retrieval-relevant registry columns and reject duplicate crops/records."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"sequence", "identity", "frame", "crop_path"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ManifestError(f"registry lacks required columns: {sorted(required)}")
            rows = [Observation(row["sequence"], row["identity"], int(row["frame"]), row["crop_path"])
                    for row in reader]
    except (OSError, ValueError) as exc:
        raise ManifestError(f"cannot read registry {path}: {exc}") from exc
    if not rows:
        raise ManifestError(f"empty registry: {path}")
    crop_paths = [row.crop_path for row in rows]
    if len(crop_paths) != len(set(crop_paths)):
        raise ManifestError("duplicate crop records in registry")
    keys = [(row.sequence, row.identity, row.frame) for row in rows]
    if len(keys) != len(set(keys)):
        raise ManifestError("duplicate (sequence, identity, frame) records in registry")
    return rows


def build_temporal_splits(
    observations: Iterable[Observation], validation_sequences: Iterable[str],
    frozen_test_sequences: Iterable[str], *, max_identities: int | None = None,
) -> tuple[list[IdentitySplit], list[dict]]:
    """Validate the boundary and form earliest-gallery/latest-query identity splits."""
    validation = tuple(validation_sequences)
    allowed = set(validation)
    frozen = set(frozen_test_sequences)
    if not validation or len(validation) != len(allowed) or allowed & frozen:
        raise ManifestError("invalid validation/frozen-test sequence boundary")
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for row in observations:
        if row.sequence in frozen:
            raise ManifestError(f"frozen-test sequence in registry: {row.sequence}")
        if row.sequence not in allowed:
            raise ManifestError(f"registry sequence is not in validation split: {row.sequence}")
        if not row.identity.startswith(row.sequence + "_"):
            raise ManifestError(f"identity is not sequence-composite: {row.identity}")
        grouped[(row.sequence, row.identity)].append(row)
    keys = sorted(grouped)
    if max_identities is not None:
        if max_identities < 1:
            raise ManifestError("max_identities must be positive")
        keys = keys[:max_identities]
    splits, rejected = [], []
    for sequence, identity in keys:
        rows = sorted(grouped[(sequence, identity)],
                      key=lambda row: (row.sequence, row.identity, row.frame, row.crop_path))
        if len(rows) < MINIMUM_OBSERVATIONS:
            rejected.append({"sequence": sequence, "identity": identity,
                             "observation_count": len(rows), "reason": "fewer_than_four_observations"})
            continue
        k = min(5, len(rows) // 2)
        gallery, query = tuple(rows[:k]), tuple(rows[-k:])
        if set(gallery) & set(query):
            raise ManifestError(f"gallery/query overlap for {identity}")
        splits.append(IdentitySplit(sequence, identity, len(rows), k, gallery, query,
                                    tuple(rows[k:len(rows) - k])))
    return splits, rejected


def split_manifest(splits: Iterable[IdentitySplit], rejected: Iterable[dict]) -> dict:
    def encode(rows: tuple[Observation, ...]) -> list[dict]:
        return [asdict(row) for row in rows]
    return {
        "identities": [{
            "sequence": item.sequence, "identity": item.identity,
            "observation_count": item.observation_count, "k": item.k,
            "gallery": encode(item.gallery), "query": encode(item.query),
            "excluded_middle": encode(item.excluded),
        } for item in sorted(splits, key=lambda x: (x.sequence, x.identity))],
        "rejected_identities": sorted(rejected, key=lambda x: (x["sequence"], x["identity"])),
    }


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ManifestError("embeddings must be a finite two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ManifestError("zero-norm embedding")
    return values / norms


def cosine_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 1.0 - l2_normalize(left) @ l2_normalize(right).T


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _metrics(query_rows: list[dict]) -> dict:
    return {
        "queries": len(query_rows),
        "rank1": _mean([row["rank1"] for row in query_rows]),
        "rank5": _mean([row["rank5"] for row in query_rows]),
        "mAP": _mean([row["average_precision"] for row in query_rows]),
        "mean_positive_cosine_distance": _mean([row["mean_positive_cosine_distance"] for row in query_rows]),
        "median_positive_cosine_distance": _median([d for row in query_rows for d in row["positive_distances"]]),
        "mean_nearest_negative_cosine_distance": _mean([row["nearest_negative_cosine_distance"] for row in query_rows
                                                          if row["nearest_negative_cosine_distance"] is not None]),
        "mean_positive_negative_margin": _mean([row["positive_negative_margin"] for row in query_rows
                                                  if row["positive_negative_margin"] is not None]),
    }


def evaluate_sequence_local(splits: Iterable[IdentitySplit], embeddings: Mapping[str, np.ndarray]) -> dict:
    """Evaluate every query only against the gallery from its own sequence."""
    split_list = sorted(splits, key=lambda item: (item.sequence, item.identity))
    paths = [row.crop_path for item in split_list for row in (*item.gallery, *item.query)]
    missing = sorted(set(paths) - set(embeddings))
    if missing:
        raise ManifestError(f"missing embeddings, first: {missing[0]}")
    normalized = {path: l2_normalize(np.asarray(embeddings[path]).reshape(1, -1))[0] for path in paths}
    dimensions = {value.shape[0] for value in normalized.values()}
    if len(dimensions) != 1:
        raise ManifestError("embedding dimensions differ")
    by_sequence: dict[str, list[IdentitySplit]] = defaultdict(list)
    for item in split_list:
        by_sequence[item.sequence].append(item)
    query_rows, hardest, nearest = [], [], []
    per_sequence = []
    for sequence in sorted(by_sequence):
        items = by_sequence[sequence]
        gallery = [row for item in items for row in item.gallery]
        gallery_matrix = np.stack([normalized[row.crop_path] for row in gallery])
        for item in items:
            for query in item.query:
                distances = 1.0 - gallery_matrix @ normalized[query.crop_path]
                order = sorted(range(len(gallery)), key=lambda i: (float(distances[i]), gallery[i].crop_path))
                relevant = [gallery[i].identity == item.identity for i in order]
                positive_ranks = [rank for rank, match in enumerate(relevant, 1) if match]
                ap = sum(hit / rank for hit, rank in enumerate(positive_ranks, 1)) / len(positive_ranks)
                positive_indices = [i for i, row in enumerate(gallery) if row.identity == item.identity]
                negative_indices = [i for i, row in enumerate(gallery) if row.identity != item.identity]
                positive_distances = [float(distances[i]) for i in positive_indices]
                hardest_index = max(positive_indices, key=lambda i: (float(distances[i]), gallery[i].crop_path))
                nearest_index = min(negative_indices, key=lambda i: (float(distances[i]), gallery[i].crop_path)) if negative_indices else None
                nearest_distance = float(distances[nearest_index]) if nearest_index is not None else None
                mean_positive = float(np.mean(positive_distances))
                margin = nearest_distance - mean_positive if nearest_distance is not None else None
                row = {"sequence": sequence, "identity": item.identity, "query_path": query.crop_path,
                       "query_frame": query.frame, "gallery_samples": len(gallery),
                       "positive_gallery_samples": len(positive_indices), "rank1": int(positive_ranks[0] <= 1),
                       "rank5": int(positive_ranks[0] <= 5), "average_precision": float(ap),
                       "mean_positive_cosine_distance": mean_positive,
                       "positive_distances": positive_distances,
                       "nearest_negative_cosine_distance": nearest_distance,
                       "positive_negative_margin": margin}
                query_rows.append(row)
                hardest.append({"sequence": sequence, "identity": item.identity, "query_path": query.crop_path,
                                "gallery_path": gallery[hardest_index].crop_path,
                                "cosine_distance": float(distances[hardest_index])})
                if nearest_index is not None:
                    nearest.append({"sequence": sequence, "identity": item.identity, "query_path": query.crop_path,
                                    "negative_identity": gallery[nearest_index].identity,
                                    "gallery_path": gallery[nearest_index].crop_path,
                                    "cosine_distance": nearest_distance})
        sequence_queries = [row for row in query_rows if row["sequence"] == sequence]
        per_sequence.append({"sequence": sequence, "identities": len(items),
                             "gallery_samples": len(gallery), **_metrics(sequence_queries)})
    metric_names = ("rank1", "rank5", "mAP", "mean_positive_cosine_distance",
                    "median_positive_cosine_distance", "mean_nearest_negative_cosine_distance",
                    "mean_positive_negative_margin")
    macro = {name: _mean([row[name] for row in per_sequence if row[name] is not None]) for name in metric_names}
    macro.update({"sequences": len(per_sequence), "identities": sum(row["identities"] for row in per_sequence),
                  "queries": sum(row["queries"] for row in per_sequence),
                  "gallery_samples": sum(row["gallery_samples"] for row in per_sequence)})
    micro = _metrics(query_rows)
    micro.update({"sequences": len(per_sequence), "identities": sum(row["identities"] for row in per_sequence),
                  "gallery_samples": sum(row["gallery_samples"] for row in per_sequence)})
    clean_queries = [{key: value for key, value in row.items() if key != "positive_distances"} for row in query_rows]
    return {"per_sequence": per_sequence, "per_query": clean_queries, "hardest_positive_pairs": hardest,
            "nearest_negative_pairs": nearest, "macro_average": macro, "micro_average": micro}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_outputs(output_dir: Path, manifest: dict, result: dict, metadata: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "protocol_manifest.json", {**metadata, "split": manifest})
    write_json(output_dir / "summary.json", {**metadata, "macro_average": result["macro_average"],
                                               "micro_average": result["micro_average"]})
    metadata_columns = {key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                        for key, value in metadata.items()}
    for filename, key in (("per_sequence.csv", "per_sequence"), ("per_query.csv", "per_query"),
                          ("hardest_positive_pairs.csv", "hardest_positive_pairs"),
                          ("nearest_negative_pairs.csv", "nearest_negative_pairs")):
        rows = [{**metadata_columns, **row} for row in result[key]]
        write_csv(output_dir / filename, rows, list(metadata_columns) + (list(result[key][0]) if result[key] else []))
