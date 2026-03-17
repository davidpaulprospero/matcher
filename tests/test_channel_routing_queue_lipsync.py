"""Focused tests for label-based channel routing and queue lipsync audio selection."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEGOLD_DIR = PROJECT_ROOT / "Degold"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, DEGOLD_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from channel_routing import resolve_channels_for_board
import queue_lipsync as ql


@pytest.mark.fast
def test_label_override_wins_over_board_defaults():
    routing = resolve_channels_for_board(
        {"channel": "RRU", "lipsync_channel": "RRU"},
        "RRU",
        [{"name": "DeepSeaReports"}],
    )

    assert routing == {
        "project_channel": "DSR",
        "lipsync_channel": "DSR",
    }


@pytest.mark.fast
def test_resolve_local_project_audio_prefers_expected_channel_over_stale_state(tmp_path, monkeypatch):
    monkeypatch.setattr(ql, "LOCAL_PROJECTS_ROOT", tmp_path)

    rru_project = tmp_path / "RennReports" / "card123-Label_Override_Card__2026-03-12"
    rru_voiceover = rru_project / "voiceover"
    rru_voiceover.mkdir(parents=True)
    (rru_voiceover / "voiceover.mp3").write_bytes(b"rru audio")

    dsr_project = tmp_path / "DeepSeaReports" / "Label_Override_Card__2026-03-12"
    dsr_voiceover = dsr_project / "voiceover"
    dsr_voiceover.mkdir(parents=True)
    selected_audio = dsr_voiceover / "voiceover.mp3"
    selected_audio.write_bytes(b"dsr audio")

    state = {
        "pipelines": {
            "card123": {
                "project": {
                    "local_project_dirs": [str(rru_project)],
                }
            }
        }
    }

    result = ql.resolve_local_project_audio(
        "card123",
        state=state,
        staging_root=tmp_path / "staged",
        expected_channel="DSR",
        card_title="Label Override Card",
    )

    assert result is not None
    assert result["project_dir"] == str(dsr_project)
    assert result["original_audio_path"] == str(selected_audio)
    assert result["local_project_dirs"][0] == str(dsr_project)
    assert str(rru_project) in result["local_project_dirs"]


@pytest.mark.fast
def test_resolve_local_project_audio_prefers_richer_cross_channel_project(tmp_path, monkeypatch):
    monkeypatch.setattr(ql, "LOCAL_PROJECTS_ROOT", tmp_path)

    rru_project = tmp_path / "RennReports" / "card123-Label_Override_Card__2026-03-12"
    rru_voiceover = rru_project / "voiceover"
    rru_voiceover.mkdir(parents=True)
    selected_audio = rru_voiceover / "voiceover.mp3"
    selected_audio.write_bytes(b"rru audio")
    (rru_project / "logs").mkdir()
    (rru_project / "logs" / "run.log").write_text("started", encoding="utf-8")
    (rru_project / "output" / "20260312_070139").mkdir(parents=True)
    (rru_project / "output" / "20260312_070139" / "timeline.xml").write_text("ok", encoding="utf-8")
    (rru_project / "checkpoint.json").write_text("{}", encoding="utf-8")
    (rru_project / "trello_card.json").write_text(
        json.dumps({"card_id": "card123", "short_url": "https://trello.com/c/card123"}),
        encoding="utf-8",
    )

    dsr_project = tmp_path / "DeepSeaReports" / "Label_Override_Card__2026-03-12"
    dsr_voiceover = dsr_project / "voiceover"
    dsr_voiceover.mkdir(parents=True)
    (dsr_voiceover / "voiceover.mp3").write_bytes(b"dsr audio")

    state = {
        "pipelines": {
            "card123": {
                "project": {
                    "local_project_dirs": [str(rru_project)],
                }
            }
        }
    }

    result = ql.resolve_local_project_audio(
        "card123",
        state=state,
        staging_root=tmp_path / "staged",
        expected_channel="DSR",
        card_title="Label Override Card",
    )

    assert result is not None
    assert result["project_dir"] == str(rru_project)
    assert result["original_audio_path"] == str(selected_audio)
    assert result["local_project_dirs"][0] == str(rru_project)
