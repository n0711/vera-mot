"""CPU-only checks for read-only tracking and declared VLM output evidence."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile

import pytest

from vera_mot.vision_evidence import build_track_evidence, main as mot_main
from vera_mot.vlm_contract import summarise_vlm_manifest, main as vlm_main


def row(frame, track):
    return f"{frame},{track},3,4,10,15,0.75,-1,-1,-1\n".encode()


def annotation():
    return {
        "annotation_version": 1,
        "sequence": "M0203",
        "model_id": "research/example-vlm",
        "model_revision": "local-revision1",
        "prompt_sha256": "0" * 64,
        "frames": [
            {"frame": 1, "image_sha256": "1" * 64, "labels": ["vehicle"]},
            {"frame": 2, "image_sha256": "2" * 64, "labels": ["vehicle", "road"]},
        ],
    }


def test_mot_stream_summary_no_position_or_track_identifiers():
    data = row(1, 22) + row(1, 23) + row(3, 22)
    result = build_track_evidence(io.BytesIO(data), sequence="M0203")
    assert result["mot_sha256"] == hashlib.sha256(data).hexdigest()
    assert result["track_observation_rows"] == 3
    assert result["distinct_sequence_local_track_ids"] == 2
    assert result["frames_with_output"] == 2
    assert result["first_output_frame"] == 1
    assert result["last_output_frame"] == 3
    encoded = json.dumps(result)
    assert '"x"' not in encoded
    assert '"latitude"' not in encoded
    assert result["claims"]["vision_language_inference"] is False


@pytest.mark.parametrize("bad", [
    b"", b"NaN,1,0,0,1,1,1,-1,-1,-1\n",
    row(2, 1) + row(1, 2),
    row(1, 1) + row(1, 1),
    b"1,1,3,4,0,15,0.75,-1,-1,-1\n",
    b"1,1,3,4,10,15,0.75,-1,-1\n",
])
def test_mot_invalid_files_rejected(bad):
    with pytest.raises(ValueError):
        build_track_evidence(io.BytesIO(bad), sequence="M0203")


def test_vlm_summary_has_provenance_but_no_images_or_captions():
    raw = json.dumps(annotation()).encode()
    result = summarise_vlm_manifest(raw)
    assert result["manifest_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["label_counts"] == {"road": 1, "vehicle": 2}
    assert result["annotated_frames"] == 2
    assert result["claims"]["model_executed_by_this_tool"] is False
    assert '"image_sha256"' not in json.dumps(result)


@pytest.mark.parametrize("mutate", [
    lambda obj: obj.update(prompt_sha256="not-a-hash"),
    lambda obj: obj["frames"].append(obj["frames"][0]),
    lambda obj: obj["frames"][0].update(labels=["vehicle", "vehicle"]),
    lambda obj: obj.update(extra_field=True),
])
def test_vlm_bad_manifest_rejected(mutate):
    obj = annotation()
    mutate(obj)
    with pytest.raises(ValueError):
        summarise_vlm_manifest(json.dumps(obj).encode())


def test_duplicate_keys_and_nan_rejected():
    with pytest.raises(ValueError):
        summarise_vlm_manifest(b'{"annotation_version":1,"annotation_version":1}')
    with pytest.raises(ValueError):
        summarise_vlm_manifest(b'{"x":NaN}')


def test_cli_failures_emit_no_partial_report():
    with tempfile.TemporaryDirectory() as temp:
        file = Path(temp) / "bad.txt"
        file.write_text("not mot", encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            with pytest.raises(SystemExit):
                mot_main([str(file), "--sequence", "M0203"])
        assert stdout.getvalue() == ""
        file.write_text('{"bad":"annotation"}', encoding="utf-8")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            with pytest.raises(SystemExit):
                vlm_main([str(file)])
        assert stdout.getvalue() == ""
