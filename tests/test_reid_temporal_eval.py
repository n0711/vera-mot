import csv
from pathlib import Path

import numpy as np
import pytest

from vera_mot.reid_data import ManifestError
from vera_mot.reid_temporal_eval import (
    Observation, build_temporal_splits, cosine_distance, evaluate_sequence_local,
    read_registry, split_manifest, write_evaluation_outputs,
)


VAL = ["M0201", "M0202"]
FROZEN = ["M0203"]


def observations(sequence="M0201", identity="M0201_1", count=6):
    return [Observation(sequence, identity, frame, f"{identity}_{frame}.png") for frame in range(1, count + 1)]


def test_deterministic_temporal_split_and_middle_exclusion():
    rows = list(reversed(observations(count=11)))
    first, _ = build_temporal_splits(rows, VAL, FROZEN)
    second, _ = build_temporal_splits(rows, VAL, FROZEN)
    assert first == second
    assert [row.frame for row in first[0].gallery] == [1, 2, 3, 4, 5]
    assert [row.frame for row in first[0].query] == [7, 8, 9, 10, 11]
    assert [row.frame for row in first[0].excluded] == [6]


def test_four_observations_have_disjoint_two_by_two_split():
    splits, _ = build_temporal_splits(observations(count=4), VAL, FROZEN)
    assert splits[0].k == 2
    assert set(splits[0].gallery).isdisjoint(splits[0].query)


def test_short_identity_is_rejected():
    splits, rejected = build_temporal_splits(observations(count=3), VAL, FROZEN)
    assert splits == []
    assert rejected[0]["reason"] == "fewer_than_four_observations"


def test_frozen_test_sequence_is_rejected():
    with pytest.raises(ManifestError, match="frozen-test"):
        build_temporal_splits(observations("M0203", "M0203_1", 4), VAL, FROZEN)


def two_identity_case():
    rows = observations(identity="M0201_1", count=4) + observations(identity="M0201_2", count=4)
    splits, _ = build_temporal_splits(rows, VAL, FROZEN)
    vectors = {}
    for item in splits:
        vector = np.array([1.0, 0.0]) if item.identity.endswith("_1") else np.array([0.0, 1.0])
        for row in (*item.gallery, *item.query):
            vectors[row.crop_path] = vector
    return splits, vectors


def test_sequence_local_evaluation_excludes_cross_sequence_gallery():
    splits, vectors = two_identity_case()
    other, _ = build_temporal_splits(observations("M0202", "M0202_1", 4), VAL, FROZEN)
    for row in (*other[0].gallery, *other[0].query):
        vectors[row.crop_path] = np.array([1.0, 0.0])
    result = evaluate_sequence_local(splits + other, vectors)
    assert {row["gallery_samples"] for row in result["per_query"] if row["sequence"] == "M0201"} == {4}
    assert {row["gallery_samples"] for row in result["per_query"] if row["sequence"] == "M0202"} == {2}


def test_correct_rank1_rank5_and_map():
    splits, vectors = two_identity_case()
    result = evaluate_sequence_local(splits, vectors)
    assert result["micro_average"]["rank1"] == 1.0
    assert result["micro_average"]["rank5"] == 1.0
    assert result["micro_average"]["mAP"] == 1.0


def test_rank5_when_first_positive_is_fifth():
    rows = sum((observations(identity=f"M0201_{i}", count=4) for i in range(1, 4)), [])
    splits, _ = build_temporal_splits(rows, VAL, FROZEN)
    # Identity 1 query ranks after all four negatives, then its two positives.
    vectors = {row.crop_path: np.array([0.0, 1.0]) for item in splits for row in (*item.gallery, *item.query)}
    for row in splits[0].gallery:
        vectors[row.crop_path] = np.array([1.0, 0.0])
    for row in splits[0].query:
        vectors[row.crop_path] = np.array([-1.0, 0.0])
    result = evaluate_sequence_local(splits, vectors)
    target = [row for row in result["per_query"] if row["identity"] == "M0201_1"]
    assert {row["rank1"] for row in target} == {0}
    assert {row["rank5"] for row in target} == {1}


def test_average_precision_formula_with_interleaved_positive():
    splits, vectors = two_identity_case()
    a = splits[0]
    vectors[a.gallery[0].crop_path] = np.array([1.0, 0.0])
    vectors[a.gallery[1].crop_path] = np.array([0.6, 0.8])
    for row in a.query:
        vectors[row.crop_path] = np.array([1.0, 0.0])
    # Put exactly one negative between A's positives.
    b = splits[1]
    vectors[b.gallery[0].crop_path] = np.array([0.8, 0.6])
    vectors[b.gallery[1].crop_path] = np.array([0.0, 1.0])
    result = evaluate_sequence_local(splits, vectors)
    target = [row for row in result["per_query"] if row["identity"] == a.identity]
    assert target[0]["average_precision"] == pytest.approx((1 + 2 / 3) / 2)


def test_cosine_distance_and_normalization():
    distance = cosine_distance(np.array([[3.0, 0.0]]), np.array([[0.0, 4.0], [2.0, 0.0]]))
    assert np.allclose(distance, [[1.0, 0.0]])


def test_macro_and_micro_are_distinct_for_unequal_sequence_sizes():
    one, vectors_one = two_identity_case()
    # Sequence 2 has one identity and therefore no negatives; its retrieval is perfect.
    other, _ = build_temporal_splits(observations("M0202", "M0202_1", 4), VAL, FROZEN)
    vectors = dict(vectors_one)
    for row in (*other[0].gallery, *other[0].query):
        vectors[row.crop_path] = np.array([1.0, 0.0])
    # Break all four queries in the larger sequence by swapping query features.
    for item in one:
        for row in item.query:
            vectors[row.crop_path] = np.array([0.0, 1.0]) if item.identity.endswith("_1") else np.array([1.0, 0.0])
    result = evaluate_sequence_local(one + other, vectors)
    assert result["macro_average"]["rank1"] == pytest.approx(0.5)
    assert result["micro_average"]["rank1"] == pytest.approx(1 / 3)


def test_duplicate_crop_records_rejected(tmp_path):
    path = tmp_path / "registry.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "identity", "frame", "crop_path"])
        writer.writeheader()
        writer.writerow({"sequence": "M0201", "identity": "M0201_1", "frame": 1, "crop_path": "same.png"})
        writer.writerow({"sequence": "M0201", "identity": "M0201_1", "frame": 2, "crop_path": "same.png"})
    with pytest.raises(ManifestError, match="duplicate crop"):
        read_registry(path)


def test_output_determinism(tmp_path):
    splits, vectors = two_identity_case()
    result = evaluate_sequence_local(splits, vectors)
    manifest = split_manifest(splits, [])
    metadata = {"checkpoint_path": "c", "checkpoint_sha256": "1", "configuration_path": "x",
                "configuration_sha256": "2", "registry_path": "r", "registry_sha256": "3",
                "git_commit_sha": "g", "protocol_version": "v", "creation_timestamp": "fixed",
                "validation_sequences": VAL}
    left, right = tmp_path / "left", tmp_path / "right"
    write_evaluation_outputs(left, manifest, result, metadata)
    write_evaluation_outputs(right, manifest, result, metadata)
    assert {p.name: p.read_bytes() for p in left.iterdir()} == {p.name: p.read_bytes() for p in right.iterdir()}
