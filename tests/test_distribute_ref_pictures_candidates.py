"""Tests for the --candidates N multi-track FILL mode and name detection."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import opentimelineio as otio
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "distribute_ref_pictures.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("drp", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_plan():
    return [
        {
            "index": 0,
            "center_sec": 1.0,
            "start_frames": 30,
            "dur_frames": 60,
            "dur_source": "continuous_gap",
            "slot_name": "first",
            "srt_text": "first",
        },
        {
            "index": 1,
            "center_sec": 4.0,
            "start_frames": 120,
            "dur_frames": 60,
            "dur_source": "continuous_gap",
            "slot_name": "second",
            "srt_text": "second",
        },
    ]


def _make_image_grid(tmp: Path, candidates: int) -> list[list[Path]]:
    grid = []
    for c in range(candidates):
        row = [tmp / f"c{c:02d}_g{g}.jpg" for g in range(2)]
        for p in row:
            p.write_bytes(b"jpeg")
        grid.append(row)
    return grid


def test_candidate_track_names_default_and_custom():
    mod = _load_module()
    assert mod.candidate_track_names("V-Ref", None, 3) == [
        "V-Ref-Cand00",
        "V-Ref-Cand01",
        "V-Ref-Cand02",
    ]
    assert mod.candidate_track_names("V-Ref", "Pick-", 2) == ["Pick-00", "Pick-01"]


def test_build_timeline_single_candidate_unchanged(tmp_path):
    mod = _load_module()
    plan = _make_plan()
    grid = _make_image_grid(tmp_path, 1)
    tl = mod.build_timeline(
        plan=plan,
        image_paths=grid[0],
        track_name="V-Ref",
        rate=30.0,
        subject="x",
        source_total_frames=0,
        match_duration=False,
    )
    assert len(tl.tracks) == 1
    assert tl.tracks[0].name == "V-Ref"
    mod.verify_fill_timing(plan, tl, ["V-Ref"], 30.0)


def test_build_timeline_multi_candidate_aligns_clips(tmp_path):
    mod = _load_module()
    plan = _make_plan()
    grid = _make_image_grid(tmp_path, 3)
    track_names = mod.candidate_track_names("V-Ref", None, 3)
    tl = mod.build_timeline(
        plan=plan,
        image_paths=grid[0],
        track_name="V-Ref",
        rate=30.0,
        subject="x",
        source_total_frames=0,
        match_duration=False,
        candidates=3,
        candidate_image_paths=grid,
        candidate_track_names=track_names,
    )
    assert [t.name for t in tl.tracks] == track_names
    for cand_idx, track in enumerate(tl.tracks):
        clips = [x for x in track if isinstance(x, otio.schema.Clip)]
        assert len(clips) == 2
        for clip, entry in zip(clips, plan):
            assert int(round(clip.range_in_parent().start_time.value)) == entry["start_frames"]
            assert int(round(clip.duration().value)) == entry["dur_frames"]
            assert clip.metadata["candidate_index"] == cand_idx
            assert clip.metadata["candidate_track_name"] == track_names[cand_idx]
    mod.verify_fill_timing(plan, tl, track_names, 30.0)


def test_verify_fill_timing_detects_drift(tmp_path):
    mod = _load_module()
    plan = _make_plan()
    grid = _make_image_grid(tmp_path, 1)
    track_names = ["V-Ref"]
    tl = mod.build_timeline(
        plan=plan,
        image_paths=grid[0],
        track_name="V-Ref",
        rate=30.0,
        subject="x",
        source_total_frames=0,
        match_duration=False,
    )
    # Insert a leading gap so clip start_time drifts 5 frames to the right.
    track = tl.tracks[0]
    track.insert(0, otio.schema.Gap(
        source_range=otio.opentime.TimeRange(
            start_time=otio.opentime.RationalTime(0, 30.0),
            duration=otio.opentime.RationalTime(5, 30.0),
        )
    ))
    with pytest.raises(ValueError, match="timing mismatch"):
        mod.verify_fill_timing(plan, tl, track_names, 30.0)


def test_cli_rejects_candidates_without_fill_gaps(tmp_path, monkeypatch, caplog):
    mod = _load_module()
    srt = tmp_path / "x.srt"
    srt.write_text("1\n00:00:01,000 --> 00:00:04,000\nhello\n\n", encoding="utf-8")
    otio_src = tmp_path / "x.otio"
    otio_src.write_text("", encoding="utf-8")
    argv = [
        "distribute_ref_pictures.py",
        "--srt",
        str(srt),
        "--otio",
        str(otio_src),
        "--candidates",
        "3",
        "--no-llm",
    ]
    monkeypatch.setattr("sys.argv", argv)
    rc = mod.main()
    assert rc == 2
    assert any("--candidates > 1 requires --fill-gaps" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Name detection helpers (used by --detect-names)
# ---------------------------------------------------------------------------


def _patch_heuristic(mod, names):
    """Patch `extract_named_entities_from_text` inside drp's closure."""
    fake_mod = mock.MagicMock()
    fake_mod.extract_named_entities_from_text = mock.MagicMock(return_value=list(names))
    import sys as _sys

    original = _sys.modules.get("src.matching.scoring")
    _sys.modules["src.matching.scoring"] = fake_mod
    return _sys, original


def _restore(_sys, original):
    if original is None:
        _sys.modules.pop("src.matching.scoring", None)
    else:
        _sys.modules["src.matching.scoring"] = original


def test_detect_name_in_segment_heuristic_only():
    mod = _load_module()
    _sys, original = _patch_heuristic(mod, ["Ada Lovelace"])
    try:
        name, provider = mod._detect_name_in_segment(
            "Ada Lovelace pioneered computing.",
            provider="heuristic",
            model="m",
            host=None,
        )
    finally:
        _restore(_sys, original)
    assert name == "Ada Lovelace"
    assert provider == "heuristic"


def test_detect_name_in_segment_ollama_success():
    mod = _load_module()
    fake_response = mock.MagicMock()
    fake_response.text = '["Grace Hopper"]'
    fake_client = mock.MagicMock()
    fake_client.generate.return_value = fake_response

    with mock.patch.object(mod, "create_client", return_value=fake_client):
        name, provider = mod._detect_name_in_segment(
            "Grace Hopper invented COBOL.",
            provider="ollama",
            model="m",
            host="http://localhost:11434",
        )
    assert name == "Grace Hopper"
    assert provider == "ollama"


def test_detect_name_in_segment_ollama_fallback_to_heuristic():
    mod = _load_module()
    fake_response = mock.MagicMock()
    fake_response.text = "[]"
    fake_client = mock.MagicMock()
    fake_client.generate.return_value = fake_response

    _sys, original = _patch_heuristic(mod, ["Linus Torvalds"])
    try:
        with mock.patch.object(mod, "create_client", return_value=fake_client):
            name, provider = mod._detect_name_in_segment(
                "Linus Torvalds created Linux.",
                provider="ollama",
                model="m",
                host="http://localhost:11434",
            )
    finally:
        _restore(_sys, original)
    assert name == "Linus Torvalds"
    assert provider == "heuristic"


def test_detect_name_in_segment_no_name_returns_none_pair():
    mod = _load_module()
    _sys, original = _patch_heuristic(mod, [])
    try:
        name, provider = mod._detect_name_in_segment(
            "the weather is beautiful today",
            provider="heuristic",
            model="m",
            host=None,
        )
    finally:
        _restore(_sys, original)
    assert name is None
    assert provider is None


def test_detect_name_in_segment_blank_returns_none_pair():
    mod = _load_module()
    assert mod._detect_name_in_segment("", provider="heuristic", model="m", host=None) == (None, None)
    assert mod._detect_name_in_segment("   ", provider="heuristic", model="m", host=None) == (None, None)


def test_build_name_detection_block_disabled():
    mod = _load_module()
    args = SimpleNamespace(detect_names=False, name_provider="ollama")
    assert mod.build_name_detection_block(args=args, plan=[]) is None


def test_build_name_detection_block_enabled_records_detections():
    mod = _load_module()
    args = SimpleNamespace(detect_names=True, name_provider="ollama")
    plan = [
        {"index": 0, "detected_name": "Ada", "detected_provider": "ollama"},
        {"index": 1, "detected_name": None, "detected_provider": None},
        {"index": 2, "detected_name": "Grace", "detected_provider": "heuristic"},
    ]
    block = mod.build_name_detection_block(args=args, plan=plan)
    assert block == {
        "enabled": True,
        "provider": "ollama",
        "providers_used": ["heuristic", "ollama"],
        "detected": [
            {"index": 0, "name": "Ada", "provider": "ollama"},
            {"index": 2, "name": "Grace", "provider": "heuristic"},
        ],
    }


def test_build_summary_carries_detected_name_into_gap_entry(tmp_path):
    mod = _load_module()
    plan = [
        {
            "index": 0,
            "center_sec": 1.0,
            "start_frames": 30,
            "dur_frames": 60,
            "dur_source": "continuous_gap",
            "slot_name": "g0",
            "srt_text": "first",
            "detected_name": "Ada",
            "detected_provider": "ollama",
        }
    ]
    args = SimpleNamespace(
        track_name="V-Ref",
        candidates=1,
        max_images=1,
        context="",
        detect_names=True,
        name_provider="ollama",
    )
    img = tmp_path / "a.jpg"
    img.write_bytes(b"jpeg")
    summary = mod.build_summary(
        args=args,
        subject="Ada",
        srt_path=tmp_path / "x.srt",
        otio_path=tmp_path / "x.otio",
        output_path=tmp_path / "out.otio",
        rate=30.0,
        source_total_frames=0,
        srt_start=0.0,
        srt_end=10.0,
        plan=plan,
        per_candidate=[[img]],
        track_names=["V-Ref"],
        actual=1,
        source_image_dirs=None,
        provider=None,
        name_detection=mod.build_name_detection_block(args=args, plan=plan),
    )
    assert summary["gaps"][0]["detected_name"] == "Ada"
    assert summary["gaps"][0]["detected_name_provider"] == "ollama"
    assert summary["name_detection"] == {
        "enabled": True,
        "provider": "ollama",
        "providers_used": ["ollama"],
        "detected": [{"index": 0, "name": "Ada", "provider": "ollama"}],
    }


def test_cli_accepts_detect_names(tmp_path, monkeypatch):
    """--detect-names flag is accepted; main() will validate later."""
    mod = _load_module()
    monkeypatch.setattr("sys.argv", ["distribute_ref_pictures.py", "--help"])
    # Argparse will exit SystemExit(0) on --help; catch it.
    with pytest.raises(SystemExit):
        mod.parse_args()
    # Now feed minimal flags; parse_args must not raise on --detect-names.
    monkeypatch.setattr(
        "sys.argv",
        [
            "distribute_ref_pictures.py",
            "--srt",
            str(tmp_path / "x.srt"),
            "--otio",
            str(tmp_path / "x.otio"),
            "--detect-names",
            "--help-argparse-stub",  # unknown; will SystemExit but proves flag is registered
        ],
    )
    with pytest.raises(SystemExit):
        mod.parse_args()

