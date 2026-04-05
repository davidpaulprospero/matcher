"""Smoke tests for the unified autorunner and clients/ reorganization.

Covers:
  - Board registry loading (valid, missing, bad YAML, filtering, disabled boards)
  - --board shorthand resolution in pipeline_queue_state.py
  - make_board_autorun_config bridge function
  - Unified CLI parsing (--unified, --board, --registry)
  - Unified --status output
  - Legacy mode backward compatibility
  - Path correctness after the clients/ reorganization
  - run_unified_cycle with mocked subprocesses
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SHARED_DIR = PROJECT_ROOT / "clients" / "shared"

for candidate in (PROJECT_ROOT, SCRIPTS_DIR, SHARED_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from degold_autorun import (
    AutorunConfig,
    BoardConfig,
    DEFAULT_BOARD_REGISTRY,
    DEFAULT_QUEUE_SCRIPT,
    DEFAULT_UNIFIED_AUTORUN_STATE_FILE,
    DEFAULT_UNIFIED_LOCK_FILE,
    DEFAULT_UNIFIED_LOG_FILE,
    DEFAULT_UNIFIED_STOP_FILE,
    build_parser,
    build_queue_snapshot,
    load_board_registry,
    load_json_file,
    make_board_autorun_config,
    normalize_card_ids,
    write_json_file,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_REGISTRY = textwrap.dedent("""\
    schema_version: "1.0.0"
    defaults:
      skip_discord_prepare: true
      refresh_lipsync: true
    boards:
      alpha:
        display_name: "Alpha Board"
        enabled: true
        priority: 10
        state_file: "clients/alpha/state.json"
        accounts_dir: "clients/alpha/accounts"
        board_map_file: "clients/alpha/board_channel_map.yaml"
        projects_root: null
        channels: ["A"]
        skip_discord_prepare: false
        refresh_lipsync: true
      beta:
        display_name: "Beta Board"
        enabled: true
        priority: 5
        state_file: "clients/beta/state.json"
        accounts_dir: null
        board_map_file: null
        projects_root: "E:/Edit Job/Beta"
        channels: ["B1", "B2"]
        skip_discord_prepare: true
        refresh_lipsync: false
      disabled:
        display_name: "Disabled Board"
        enabled: false
        priority: 1
        state_file: "clients/disabled/state.json"
        accounts_dir: null
        board_map_file: null
        projects_root: null
        channels: []
""")


def write_registry(tmp_path: Path, content: str = MINIMAL_REGISTRY) -> Path:
    registry = tmp_path / "board_registry.yaml"
    registry.write_text(content, encoding="utf-8")
    return registry


def make_fake_queue_state(
    ready_ids: list[str] | None = None,
    running_ids: list[str] | None = None,
) -> dict:
    return {
        "schema_version": "1.0.0",
        "generated_at": "2026-04-05T00:00:00+00:00",
        "summary": {
            "pending_not_started": 0,
            "ready": len(ready_ids or []),
            "blocked": 0,
            "completed": 0,
            "not_required": 0,
            "running": len(running_ids or []),
            "pipeline_actionable": len(ready_ids or []),
            "needs_submission_workflow": 0,
        },
        "queue": {
            "pending_not_started_card_ids": [],
            "ready_card_ids": ready_ids or [],
            "blocked_card_ids": [],
            "completed_card_ids": [],
        },
        "runtime_summary": {
            "running_count": len(running_ids or []),
            "running_card_ids": running_ids or [],
        },
        "pipelines": {},
    }


# ===================================================================
# 1. Board registry loading
# ===================================================================


class TestBoardRegistryLoading:
    def test_loads_enabled_boards_sorted_by_priority(self, tmp_path):
        registry = write_registry(tmp_path)
        boards = load_board_registry(registry)

        assert len(boards) == 2
        assert boards[0].key == "alpha"
        assert boards[0].priority == 10
        assert boards[1].key == "beta"
        assert boards[1].priority == 5

    def test_disabled_board_excluded(self, tmp_path):
        registry = write_registry(tmp_path)
        boards = load_board_registry(registry)
        keys = [b.key for b in boards]
        assert "disabled" not in keys

    def test_filter_board_returns_only_that_board(self, tmp_path):
        registry = write_registry(tmp_path)
        boards = load_board_registry(registry, filter_board="beta")

        assert len(boards) == 1
        assert boards[0].key == "beta"

    def test_filter_board_overrides_disabled(self, tmp_path):
        registry = write_registry(tmp_path)
        boards = load_board_registry(registry, filter_board="disabled")

        assert len(boards) == 1
        assert boards[0].key == "disabled"

    def test_filter_board_unknown_raises(self, tmp_path):
        registry = write_registry(tmp_path)
        with pytest.raises(SystemExit, match="No enabled boards"):
            load_board_registry(registry, filter_board="nonexistent")

    def test_missing_registry_file_raises(self, tmp_path):
        with pytest.raises(SystemExit, match="Board registry not found"):
            load_board_registry(tmp_path / "missing.yaml")

    def test_missing_boards_key_raises(self, tmp_path):
        registry = write_registry(tmp_path, "schema_version: '1.0.0'\nfoo: bar\n")
        with pytest.raises(SystemExit, match="missing 'boards' key"):
            load_board_registry(registry)

    def test_empty_boards_raises(self, tmp_path):
        registry = write_registry(tmp_path, "boards: {}\n")
        with pytest.raises(SystemExit, match="No enabled boards"):
            load_board_registry(registry)

    def test_all_disabled_raises(self, tmp_path):
        content = textwrap.dedent("""\
            boards:
              only:
                enabled: false
                state_file: "x.json"
        """)
        registry = write_registry(tmp_path, content)
        with pytest.raises(SystemExit, match="No enabled boards"):
            load_board_registry(registry)

    def test_board_fields_parsed_correctly(self, tmp_path):
        registry = write_registry(tmp_path)
        boards = load_board_registry(registry)
        alpha = boards[0]

        assert alpha.display_name == "Alpha Board"
        assert alpha.channels == ("A",)
        assert alpha.skip_discord_prepare is False
        assert alpha.refresh_lipsync is True
        assert alpha.state_file == PROJECT_ROOT / "clients" / "alpha" / "state.json"
        assert alpha.accounts_dir == PROJECT_ROOT / "clients" / "alpha" / "accounts"
        assert alpha.board_map_file == PROJECT_ROOT / "clients" / "alpha" / "board_channel_map.yaml"
        assert alpha.projects_root is None

    def test_beta_fields_parsed_correctly(self, tmp_path):
        registry = write_registry(tmp_path)
        boards = load_board_registry(registry)
        beta = [b for b in boards if b.key == "beta"][0]

        assert beta.channels == ("B1", "B2")
        assert beta.skip_discord_prepare is True
        assert beta.refresh_lipsync is False
        assert beta.projects_root == Path("E:/Edit Job/Beta")
        assert beta.accounts_dir is None
        assert beta.board_map_file is None

    def test_defaults_applied_when_board_omits_fields(self, tmp_path):
        content = textwrap.dedent("""\
            defaults:
              skip_discord_prepare: true
              refresh_lipsync: false
            boards:
              minimal:
                state_file: "s.json"
        """)
        registry = write_registry(tmp_path, content)
        boards = load_board_registry(registry)
        assert boards[0].skip_discord_prepare is True
        assert boards[0].refresh_lipsync is False


# ===================================================================
# 2. make_board_autorun_config bridge
# ===================================================================


class TestMakeBoardAutorunConfig:
    def _make_board(self, **overrides) -> BoardConfig:
        defaults = dict(
            key="test",
            display_name="Test",
            priority=5,
            state_file=Path("test/state.json"),
            accounts_dir=Path("test/accounts"),
            board_map_file=Path("test/board_map.yaml"),
            projects_root=Path("E:/Projects/Test"),
            channels=("CH1",),
            skip_discord_prepare=True,
            refresh_lipsync=False,
        )
        defaults.update(overrides)
        return BoardConfig(**defaults)

    def test_basic_bridge(self, tmp_path):
        board = self._make_board()
        config = make_board_autorun_config(
            board=board,
            autorun_state_file=tmp_path / "state.json",
            lock_file=tmp_path / "lock",
            stop_file=tmp_path / "stop",
            log_file=tmp_path / "log",
        )

        assert isinstance(config, AutorunConfig)
        assert config.queue_state_file == Path("test/state.json")
        assert config.accounts_dir == Path("test/accounts")
        assert config.board_map_file == Path("test/board_map.yaml")
        assert config.projects_root == Path("E:/Projects/Test")
        assert config.channels == ("CH1",)
        assert config.skip_discord_prepare is True
        assert config.refresh_lipsync is False

    def test_none_optional_paths(self, tmp_path):
        board = self._make_board(accounts_dir=None, board_map_file=None, projects_root=None)
        config = make_board_autorun_config(
            board=board,
            autorun_state_file=tmp_path / "s.json",
            lock_file=tmp_path / "l",
            stop_file=tmp_path / "st",
            log_file=tmp_path / "log",
        )

        assert config.accounts_dir is None
        assert config.board_map_file is None
        assert config.projects_root is None

    def test_shared_config_forwarded(self, tmp_path):
        board = self._make_board()
        config = make_board_autorun_config(
            board=board,
            autorun_state_file=tmp_path / "s.json",
            lock_file=tmp_path / "l",
            stop_file=tmp_path / "st",
            log_file=tmp_path / "log",
            once=True,
            dry_run=True,
            card_ids=("abc123",),
            verify=True,
            verify_threshold=0.3,
            pipeline_timeout_minutes=120,
        )

        assert config.once is True
        assert config.dry_run is True
        assert config.card_ids == ("abc123",)
        assert config.verify is True
        assert config.verify_threshold == 0.3
        assert config.pipeline_timeout_minutes == 120


# ===================================================================
# 3. Unified CLI parsing
# ===================================================================


class TestUnifiedCLIParsing:
    def test_unified_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["--unified", "--once"])
        assert args.unified is True
        assert args.once is True

    def test_registry_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["--unified", "--registry", "/custom/path.yaml"])
        assert args.registry == "/custom/path.yaml"

    def test_board_flag_parsed(self):
        parser = build_parser()
        args = parser.parse_args(["--unified", "--board", "stu"])
        assert args.board == "stu"

    def test_legacy_mode_no_unified(self):
        parser = build_parser()
        args = parser.parse_args(["--once"])
        assert args.unified is False
        assert args.board is None

    def test_unified_with_card_ids(self):
        parser = build_parser()
        args = parser.parse_args([
            "--unified", "--card-id", "abc123", "--card-id", "def456",
        ])
        assert args.card_id == ["abc123", "def456"]

    def test_unified_with_stop_when_idle(self):
        parser = build_parser()
        args = parser.parse_args(["--unified", "--stop-when-idle"])
        assert args.stop_when_idle is True


# ===================================================================
# 4. build_queue_snapshot
# ===================================================================


class TestBuildQueueSnapshot:
    def test_extracts_ready_card_ids(self):
        state = make_fake_queue_state(ready_ids=["aaa", "bbb"])
        snap = build_queue_snapshot(state)
        assert "aaa" in snap["ready_card_ids"]
        assert "bbb" in snap["ready_card_ids"]

    def test_extracts_running_card_ids(self):
        state = make_fake_queue_state(running_ids=["ccc"])
        snap = build_queue_snapshot(state)
        assert "ccc" in snap["running_card_ids"]

    def test_empty_state_returns_empty_lists(self):
        snap = build_queue_snapshot({})
        assert snap["ready_card_ids"] == []
        assert snap["running_card_ids"] == []
        assert snap["actionable_card_ids"] == []


# ===================================================================
# 5. Path correctness after clients/ reorganization
# ===================================================================


class TestPathCorrectness:
    def test_clients_degold_exists(self):
        assert (PROJECT_ROOT / "clients" / "degold").is_dir()

    def test_clients_stu_exists(self):
        assert (PROJECT_ROOT / "clients" / "stu").is_dir()

    def test_clients_samples_exists(self):
        assert (PROJECT_ROOT / "clients" / "samples").is_dir()

    def test_clients_shared_exists(self):
        assert (PROJECT_ROOT / "clients" / "shared").is_dir()

    def test_old_degold_dir_removed(self):
        assert not (PROJECT_ROOT / "Degold").exists()

    def test_old_stu_dir_removed(self):
        assert not (PROJECT_ROOT / "Stu").exists()

    def test_old_samples_dir_removed(self):
        assert not (PROJECT_ROOT / "Samples").exists()

    def test_shared_modules_present(self):
        shared = PROJECT_ROOT / "clients" / "shared"
        for module in [
            "channels.py",
            "channel_routing.py",
            "trello_to_lipsync.py",
            "queue_lipsync.py",
            "history_tracker.py",
            "local_project_selector.py",
        ]:
            assert (shared / module).is_file(), f"Missing {module} in clients/shared/"

    def test_degold_board_files_present(self):
        degold = PROJECT_ROOT / "clients" / "degold"
        assert (degold / "board_channel_map.yaml").is_file()
        assert (degold / "pipeline_queue_state.json").is_file()

    def test_stu_board_files_present(self):
        stu = PROJECT_ROOT / "clients" / "stu"
        assert (stu / "board_channel_map.yaml").is_file()
        assert (stu / "pipeline_queue_state.json").is_file()

    def test_samples_board_files_present(self):
        samples = PROJECT_ROOT / "clients" / "samples"
        assert (samples / "board_channel_map.yaml").is_file()
        assert (samples / "pipeline_queue_state.json").is_file()

    def test_board_registry_exists(self):
        assert (PROJECT_ROOT / "config" / "board_registry.yaml").is_file()

    def test_shared_importable(self):
        from channel_routing import default_channel_for_account
        assert callable(default_channel_for_account)

    def test_channels_importable(self):
        from channels import CHANNELS
        assert isinstance(CHANNELS, dict)
        assert "RRU" in CHANNELS or "DSR" in CHANNELS


# ===================================================================
# 6. Real board registry loads from actual config
# ===================================================================


class TestRealBoardRegistry:
    def test_loads_production_registry(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY)
        keys = [b.key for b in boards]
        assert "degold" in keys
        assert "stu" in keys

    def test_degold_priority_higher_than_stu(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY)
        degold = [b for b in boards if b.key == "degold"][0]
        stu = [b for b in boards if b.key == "stu"][0]
        assert degold.priority > stu.priority

    def test_degold_state_file_exists(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY)
        degold = [b for b in boards if b.key == "degold"][0]
        assert degold.state_file.exists(), f"State file not found: {degold.state_file}"

    def test_stu_state_file_exists(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY)
        stu = [b for b in boards if b.key == "stu"][0]
        assert stu.state_file.exists(), f"State file not found: {stu.state_file}"

    def test_degold_board_map_exists(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY)
        degold = [b for b in boards if b.key == "degold"][0]
        assert degold.board_map_file.exists()

    def test_stu_board_map_exists(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY)
        stu = [b for b in boards if b.key == "stu"][0]
        assert stu.board_map_file.exists()

    def test_filter_to_stu_only(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY, filter_board="stu")
        assert len(boards) == 1
        assert boards[0].key == "stu"
        assert boards[0].channels == ("STU",)

    def test_samples_disabled_by_default(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY)
        keys = [b.key for b in boards]
        assert "samples" not in keys

    def test_samples_accessible_via_filter(self):
        boards = load_board_registry(DEFAULT_BOARD_REGISTRY, filter_board="samples")
        assert len(boards) == 1
        assert boards[0].key == "samples"


# ===================================================================
# 7. --board shorthand in pipeline_queue_state.py
# ===================================================================


class TestBoardShorthand:
    def test_board_stu_resolves_state_file(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "pipeline_queue_state.py"),
             "--board", "stu", "show", "--compact"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "PIPELINE" in result.stdout

    def test_board_degold_resolves_state_file(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "pipeline_queue_state.py"),
             "--board", "degold", "show", "--compact"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "PIPELINE" in result.stdout

    def test_board_unknown_exits_with_error(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "pipeline_queue_state.py"),
             "--board", "nonexistent", "show", "--compact"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode != 0
        assert "Unknown board" in result.stderr or "Unknown board" in result.stdout


# ===================================================================
# 8. Unified --status via subprocess
# ===================================================================


class TestUnifiedStatus:
    def test_unified_status_shows_both_boards(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "degold_autorun.py"),
             "--unified", "--status"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "degold:" in result.stdout
        assert "stu:" in result.stdout
        assert "ready" in result.stdout

    def test_unified_status_board_filter(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "degold_autorun.py"),
             "--unified", "--board", "stu", "--status"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "stu:" in result.stdout
        assert "degold:" not in result.stdout

    def test_unified_status_includes_last_launch(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "degold_autorun.py"),
             "--unified", "--status"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "Last launch:" in result.stdout


# ===================================================================
# 9. Legacy mode backward compatibility
# ===================================================================


class TestLegacyMode:
    def test_legacy_status_still_works(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "degold_autorun.py"), "--status"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "Autorun Status:" in result.stdout

    def test_legacy_list_suppressed_works(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "degold_autorun.py"),
             "--list-suppressed"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert "suppressed" in result.stdout.lower()


# ===================================================================
# 10. Unified stop file management
# ===================================================================


class TestUnifiedStopFile:
    def test_unified_stop_creates_file(self, tmp_path):
        stop_file = tmp_path / "autorun.stop"
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "degold_autorun.py"),
             "--unified", "--stop",
             "--stop-file", str(stop_file),
             "--log-file", str(tmp_path / "log")],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(PROJECT_ROOT), timeout=30,
        )
        assert result.returncode == 0
        assert stop_file.exists()
        assert "stop signal" in result.stdout.lower()


# ===================================================================
# 11. run_unified_cycle with mocked subprocess
# ===================================================================


class TestRunUnifiedCycleMocked:
    def _make_boards(self, tmp_path):
        """Create board configs with tmp_path state files."""
        state_a = tmp_path / "state_a.json"
        state_b = tmp_path / "state_b.json"
        write_json_file(state_a, make_fake_queue_state(ready_ids=["card_A1"]))
        write_json_file(state_b, make_fake_queue_state(ready_ids=["card_B1", "card_B2"]))

        board_a = BoardConfig(
            key="board_a", display_name="Board A", priority=10,
            state_file=state_a, accounts_dir=None, board_map_file=None,
            projects_root=None, channels=(), skip_discord_prepare=True,
            refresh_lipsync=False,
        )
        board_b = BoardConfig(
            key="board_b", display_name="Board B", priority=5,
            state_file=state_b, accounts_dir=None, board_map_file=None,
            projects_root=None, channels=(), skip_discord_prepare=True,
            refresh_lipsync=False,
        )
        return [board_a, board_b]

    def _common_kwargs(self, tmp_path):
        return {
            "autorun_state_file": tmp_path / "autorun_state.json",
            "lock_file": tmp_path / "autorun.lock",
            "stop_file": tmp_path / "autorun.stop",
            "log_file": tmp_path / "autorun.log",
            "queue_script": DEFAULT_QUEUE_SCRIPT,
            "python_executable": sys.executable,
            "interval_minutes": 1.0,
            "failure_retry_minutes": 5.0,
            "once": True,
            "command_timeout_seconds": 30,
            "card_ids": (),
            "stop_when_idle": False,
            "verify": False,
            "verify_threshold": 0.5,
            "auto_watch": False,
            "clear_suppressed": False,
            "validate_card_ids": False,
            "dry_run": False,
            "verbose": False,
            "pipeline_timeout_minutes": 480,
            "auto_lipsync": False,
            "webhook_url": None,
        }

    @patch("degold_autorun.run_workflow_step")
    def test_syncs_all_boards_then_picks_highest_priority(self, mock_step, tmp_path):
        from degold_autorun import run_unified_cycle

        # Mock run_workflow_step to return success without calling subprocesses
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_step.return_value = (
            mock_result,
            {"name": "test", "status": "success", "args": [], "returncode": 0,
             "stdout_tail": "", "stderr_tail": "", "warning": "", "retry_args": []},
        )

        boards = self._make_boards(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        summary = run_unified_cycle(boards, kwargs)

        assert summary["board_key"] == "board_a"  # highest priority
        assert summary["status"] in ("success", "success_with_warnings")

        # Verify all boards were synced (archive-completed called for each)
        step_names = [call[0][1] for call in mock_step.call_args_list]
        assert step_names.count("archive-completed") == 2  # once per board

    @patch("degold_autorun.run_workflow_step")
    def test_board_queue_summary_populated(self, mock_step, tmp_path):
        from degold_autorun import run_unified_cycle

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_step.return_value = (
            mock_result,
            {"name": "test", "status": "success", "args": [], "returncode": 0,
             "stdout_tail": "", "stderr_tail": "", "warning": "", "retry_args": []},
        )

        boards = self._make_boards(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        summary = run_unified_cycle(boards, kwargs)

        bqs = summary["board_queue_summary"]
        assert "board_a" in bqs
        assert "board_b" in bqs
        assert bqs["board_a"]["ready"] == 1
        assert bqs["board_b"]["ready"] == 2

    @patch("degold_autorun.run_workflow_step")
    def test_card_id_filter_across_boards(self, mock_step, tmp_path):
        from degold_autorun import run_unified_cycle

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_step.return_value = (
            mock_result,
            {"name": "test", "status": "success", "args": [], "returncode": 0,
             "stdout_tail": "", "stderr_tail": "", "warning": "", "retry_args": []},
        )

        boards = self._make_boards(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        # Filter to a card that only exists on board_b
        kwargs["card_ids"] = ("card_B1",)

        summary = run_unified_cycle(boards, kwargs)

        # Should launch from board_b even though board_a has higher priority
        assert summary["board_key"] == "board_b"

    @patch("degold_autorun.run_workflow_step")
    def test_autorun_state_file_written(self, mock_step, tmp_path):
        from degold_autorun import run_unified_cycle

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_step.return_value = (
            mock_result,
            {"name": "test", "status": "success", "args": [], "returncode": 0,
             "stdout_tail": "", "stderr_tail": "", "warning": "", "retry_args": []},
        )

        boards = self._make_boards(tmp_path)
        kwargs = self._common_kwargs(tmp_path)
        run_unified_cycle(boards, kwargs)

        state_file = tmp_path / "autorun_state.json"
        assert state_file.exists()
        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["schema_version"] == "2.0.0"
        assert "boards" in state
        assert "board_a" in state["boards"]
        assert "board_b" in state["boards"]

    @patch("degold_autorun.run_workflow_step")
    def test_no_ready_cards_produces_not_launched(self, mock_step, tmp_path):
        from degold_autorun import run_unified_cycle

        # Create boards with empty queues
        state_a = tmp_path / "state_a.json"
        state_b = tmp_path / "state_b.json"
        write_json_file(state_a, make_fake_queue_state())
        write_json_file(state_b, make_fake_queue_state())

        board_a = BoardConfig(
            key="empty_a", display_name="Empty A", priority=10,
            state_file=state_a, accounts_dir=None, board_map_file=None,
            projects_root=None, channels=(), skip_discord_prepare=True,
            refresh_lipsync=False,
        )
        board_b = BoardConfig(
            key="empty_b", display_name="Empty B", priority=5,
            state_file=state_b, accounts_dir=None, board_map_file=None,
            projects_root=None, channels=(), skip_discord_prepare=True,
            refresh_lipsync=False,
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_step.return_value = (
            mock_result,
            {"name": "test", "status": "success", "args": [], "returncode": 0,
             "stdout_tail": "", "stderr_tail": "", "warning": "", "retry_args": []},
        )

        kwargs = self._common_kwargs(tmp_path)
        summary = run_unified_cycle([board_a, board_b], kwargs)

        assert summary["launch_outcome"] == "not_launched"
        assert summary["launched_card_id"] is None
        assert summary["board_key"] == ""


# ===================================================================
# 12. JSON state file helpers
# ===================================================================


class TestJsonHelpers:
    def test_write_and_load(self, tmp_path):
        path = tmp_path / "test.json"
        payload = {"key": "value", "nested": {"a": 1}}
        write_json_file(path, payload)

        loaded = load_json_file(path, {})
        assert loaded == payload

    def test_load_missing_returns_default(self, tmp_path):
        result = load_json_file(tmp_path / "missing.json", {"default": True})
        assert result == {"default": True}

    def test_load_corrupt_returns_default(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{{invalid json", encoding="utf-8")
        result = load_json_file(path, {"fallback": True})
        assert result == {"fallback": True}
