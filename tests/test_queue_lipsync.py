"""Tests for Degold lipsync MCP command generation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = PROJECT_ROOT / "clients" / "shared"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, SHARED_DIR, SCRIPTS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from channel_routing import default_channel_for_account, resolve_channels_for_board
import queue_lipsync as ql


@pytest.mark.fast
def test_generate_mcp_commands_v2_uses_direct_input_workflow():
    jobs = [
        {
            "title": "Test Title\nSecond Line",
            "channel": "RRU",
            "audio_path": "clients/degold/audio/sample.mp3",
        }
    ]
    channel_config = {
        "RRU": {
            "drive_folder": "drive-folder-123",
            "avatar_path": "clients/degold/avatars/rru_avatar.jpg",
        }
    }

    commands = ql.generate_mcp_commands_v2(jobs, channel_config)

    assert commands[0] == {"tool": "enable", "params": {"client_id": "lipsync-queue"}}
    assert all(command["tool"] != "browser_fill_form" for command in commands)

    fill_command = commands[3]
    assert fill_command["tool"] == "browser_interact"
    assert fill_command["params"]["actions"] == [
        {"type": "clear", "selector": "input[type='text'][name='field-0']"},
        {
            "type": "type",
            "selector": "input[type='text'][name='field-0']",
            "text": "Test Title Second Line",
        },
        {"type": "clear", "selector": "input[type='text'][name='field-4']"},
        {
            "type": "type",
            "selector": "input[type='text'][name='field-4']",
            "text": "drive-folder-123",
        },
    ]

    avatar_upload = commands[5]
    assert avatar_upload["tool"] == "browser_interact"
    assert avatar_upload["params"]["actions"] == [
        {
            "type": "file_upload",
            "selector": "input[type='file'][name='field-2']",
            "files": [str((PROJECT_ROOT / "clients" / "degold" / "avatars" / "rru_avatar.jpg").resolve()).replace("\\", "/")],
        }
    ]

    audio_upload = commands[6]
    assert audio_upload["tool"] == "browser_interact"
    assert audio_upload["params"]["actions"] == [
        {
            "type": "file_upload",
            "selector": "input[type='file'][name='field-3']",
            "files": [str((PROJECT_ROOT / "clients" / "degold" / "audio" / "sample.mp3").resolve()).replace("\\", "/")],
        }
    ]


@pytest.mark.fast
def test_generate_mcp_commands_v2_opens_one_new_tab_per_job():
    jobs = [
        {"title": "First Job", "channel": "RRU", "audio_path": "clients/degold/audio/first.mp3"},
        {"title": "Second Job", "channel": "RRU", "audio_path": "clients/degold/audio/second.mp3"},
    ]
    channel_config = {
        "RRU": {
            "drive_folder": "drive-folder-123",
            "avatar_path": "clients/degold/avatars/rru_avatar.jpg",
        }
    }

    commands = ql.generate_mcp_commands_v2(jobs, channel_config)

    new_tabs = [command for command in commands if command["tool"] == "browser_tabs"]
    assert new_tabs == [
        {"tool": "browser_tabs", "params": {"action": "new", "url": ql.FORM_URL, "activate": True}},
        {"tool": "browser_tabs", "params": {"action": "new", "url": ql.FORM_URL, "activate": True}},
    ]

    submit_actions = [
        command for command in commands
        if command["tool"] == "browser_interact"
        and command["params"]["actions"] == [{"type": "click", "selector": "button[type='submit']"}]
    ]
    assert len(submit_actions) == len(jobs)


@pytest.mark.fast
def test_get_local_project_dirs_for_card_matches_case_insensitively():
    state = {
        "pipelines": {
            "vP3PMq5T": {
                "project": {
                    "local_project_dirs": [
                        "E:/Edit Job/Degold/CardA",
                        "E:/Edit Job/Degold/CardA",
                        "E:/Edit Job/Degold/CardB",
                    ]
                }
            }
        }
    }

    assert ql.get_local_project_dirs_for_card("vp3PMq5T", state=state) == [
        "E:/Edit Job/Degold/CardA",
        "E:/Edit Job/Degold/CardB",
    ]


@pytest.mark.fast
def test_find_preferred_project_audio_prefers_nosilence(tmp_path):
    project_dir = tmp_path / "project"
    voiceover_dir = project_dir / "voiceover"
    voiceover_dir.mkdir(parents=True)
    voiceover = voiceover_dir / "voiceover.mp3"
    voiceover.write_bytes(b"regular voiceover")
    nosilence = voiceover_dir / "voiceover_nosilence.mp3"
    nosilence.write_bytes(b"nosilence voiceover")

    result = ql.find_preferred_project_audio(project_dir)

    assert result == {
        "audio_source": "local_project_nosilence",
        "audio_path": str(nosilence),
        "project_dir": str(project_dir),
    }


@pytest.mark.fast
def test_find_preferred_project_audio_falls_back_to_regular_voiceover(tmp_path):
    project_dir = tmp_path / "project"
    voiceover_dir = project_dir / "voiceover"
    voiceover_dir.mkdir(parents=True)
    voiceover = voiceover_dir / "voiceover.mp3"
    voiceover.write_bytes(b"regular voiceover")

    result = ql.find_preferred_project_audio(project_dir)

    assert result == {
        "audio_source": "local_project_voiceover",
        "audio_path": str(voiceover),
        "project_dir": str(project_dir),
    }


@pytest.mark.fast
def test_resolve_local_project_audio_stages_workspace_copy(tmp_path):
    project_dir = tmp_path / "external_project"
    voiceover_dir = project_dir / "voiceover"
    voiceover_dir.mkdir(parents=True)
    nosilence = voiceover_dir / "voiceover_nosilence.mp3"
    nosilence.write_bytes(b"preferred project audio")
    staging_root = tmp_path / "staged_audio"
    state = {
        "pipelines": {
            "card123": {
                "project": {
                    "local_project_dirs": [str(project_dir)]
                }
            }
        }
    }

    result = ql.resolve_local_project_audio("card123", state=state, staging_root=staging_root)

    assert result is not None
    assert result["audio_source"] == "local_project_nosilence"
    assert result["project_dir"] == str(project_dir)
    assert result["original_audio_path"] == str(nosilence)
    assert result["local_project_dirs"] == [str(project_dir)]
    staged_path = Path(result["audio_path"])
    assert staged_path == Path(result["staged_audio_path"])
    assert staged_path.parent == staging_root / "card123"
    assert staged_path.read_bytes() == nosilence.read_bytes()


@pytest.mark.fast
def test_generate_mcp_commands_legacy_opens_one_new_tab_per_job():
    jobs = [
        {"title": "First Job", "channel": "RRU", "audio_path": "clients/degold/audio/first.mp3"},
        {"title": "Second Job", "channel": "RRU", "audio_path": "clients/degold/audio/second.mp3"},
    ]
    channel_config = {
        "RRU": {
            "drive_folder": "drive-folder-123",
            "avatar_path": "clients/degold/avatars/rru_avatar.jpg",
        }
    }

    commands = ql.generate_mcp_commands(jobs, channel_config)

    new_tabs = [command for command in commands if command["tool"] == "browser_tabs"]
    assert new_tabs == [
        {"tool": "browser_tabs", "params": {"action": "new"}},
        {"tool": "browser_tabs", "params": {"action": "new"}},
    ]

    navigations = [command for command in commands if command["tool"] == "browser_navigate"]
    assert navigations == [
        {"tool": "browser_navigate", "params": {"url": ql.FORM_URL}},
        {"tool": "browser_navigate", "params": {"url": ql.FORM_URL}},
    ]


@pytest.mark.fast
def test_channel_routing_keeps_project_channel_but_uses_account_default_for_lipsync():
    routing = resolve_channels_for_board({"channel": "RRU"}, "DSR")

    assert routing == {
        "project_channel": "RRU",
        "lipsync_channel": "DSR",
    }


@pytest.mark.fast
def test_channel_routing_respects_explicit_lipsync_override():
    routing = resolve_channels_for_board({"channel": "RRU", "lipsync_channel": "DSR"}, "RRU")

    assert routing == {
        "project_channel": "RRU",
        "lipsync_channel": "DSR",
    }


@pytest.mark.fast
def test_account_default_channel_keeps_pamela_on_rru():
    assert default_channel_for_account("pamela") == "RRU"
    assert default_channel_for_account("stuart") == "DSR"
