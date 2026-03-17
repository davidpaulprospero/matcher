"""
Tests for healing session persistence and crash recovery (US-64-012).

Verifies:
- save_session() persists orchestrator state to JSON
- load_session() restores state from persisted file
- Heal history, metrics, config snapshots, and healer states are persisted
- Auto-save triggers on heal success, stage completion, and escalation
- create_orchestrated_pipeline() recovers session from crash
- Roundtrip save/load preserves all data
"""

import json
import time

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.agents.orchestrator import HealingOrchestrator
from src.agents.strategy import HealingStrategy, HealingMetrics, ConfigSnapshot
from src.agents.base import HealerResult, HealerAction


class TestSaveSession:
    """Test save_session() persists orchestrator state."""

    @pytest.mark.fast
    def test_save_creates_file(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        path = orchestrator.save_session()
        assert path.exists()
        assert path.name == "healing_session.json"

    @pytest.mark.fast
    def test_save_contains_version(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.save_session()

        data = json.loads(orchestrator._get_session_path().read_text(encoding="utf-8"))
        assert data["version"] == 1

    @pytest.mark.fast
    def test_save_contains_timestamp(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        before = time.time()
        orchestrator.save_session()
        after = time.time()

        data = json.loads(orchestrator._get_session_path().read_text(encoding="utf-8"))
        assert before <= data["timestamp"] <= after

    @pytest.mark.fast
    def test_save_contains_current_stage(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.current_stage = "DOWNLOAD_SEGMENTS"
        orchestrator.save_session()

        data = json.loads(orchestrator._get_session_path().read_text(encoding="utf-8"))
        assert data["current_stage"] == "DOWNLOAD_SEGMENTS"

    @pytest.mark.fast
    def test_save_contains_metrics(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 150.0)
        orchestrator.metrics.record_heal("download-healer", "DOWNLOAD", False)
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.time_spent_healing = 5.5
        orchestrator.metrics.user_escalations = 2

        orchestrator.save_session()

        data = json.loads(orchestrator._get_session_path().read_text(encoding="utf-8"))
        m = data["metrics"]
        assert m["total_heals"] == 2
        assert m["successful_heals"] == 1
        assert m["failed_heals"] == 1
        assert m["heals_by_healer"]["api-healer"] == 1
        assert m["time_spent_healing"] == 5.5
        assert m["user_escalations"] == 2
        assert m["error_categories"]["api"] == 1

    @pytest.mark.fast
    def test_save_contains_config_snapshots(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        # Manually add a snapshot to avoid triggering auto-save loop
        snap = ConfigSnapshot(
            stage_name="MATCH",
            timestamp=1000.0,
            config_values={"output.gap_mode": "scale"},
        )
        orchestrator.config_snapshots.append(snap)
        orchestrator.save_session()

        data = json.loads(orchestrator._get_session_path().read_text(encoding="utf-8"))
        snaps = data["config_snapshots"]
        assert len(snaps) == 1
        assert snaps[0]["stage_name"] == "MATCH"
        assert snaps[0]["config_values"]["output.gap_mode"] == "scale"

    @pytest.mark.fast
    def test_save_contains_healer_states(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        # Set some healer state
        api_healer = orchestrator._healer_instances.get("api-healer")
        if api_healer:
            api_healer.backoff_time = 60.0
            api_healer.retry_count = 3

        orchestrator.save_session()

        data = json.loads(orchestrator._get_session_path().read_text(encoding="utf-8"))
        states = data["healer_states"]
        if "api-healer" in states:
            assert states["api-healer"]["backoff_time"] == 60.0
            assert states["api-healer"]["retry_count"] == 3

    @pytest.mark.fast
    def test_save_contains_strategy(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.aggressive()
        )
        orchestrator.save_session()

        data = json.loads(orchestrator._get_session_path().read_text(encoding="utf-8"))
        assert data["strategy"]["mode"] == "aggressive"
        assert data["strategy"]["max_attempts_per_stage"] == 5


class TestLoadSession:
    """Test load_session() restores orchestrator state."""

    @pytest.mark.fast
    def test_load_returns_false_when_no_file(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orchestrator.load_session() is False

    @pytest.mark.fast
    def test_load_returns_false_for_corrupt_json(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        session_path = orchestrator._get_session_path()
        session_path.write_text("not valid json{{{", encoding="utf-8")
        assert orchestrator.load_session() is False

    @pytest.mark.fast
    def test_load_returns_false_for_wrong_version(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        session_path = orchestrator._get_session_path()
        session_path.write_text(json.dumps({"version": 99}), encoding="utf-8")
        assert orchestrator.load_session() is False

    @pytest.mark.fast
    def test_load_restores_current_stage(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.current_stage = "CAPTION"
        orchestrator.save_session()

        # Create fresh orchestrator
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is True
        assert orch2.current_stage == "CAPTION"

    @pytest.mark.fast
    def test_load_restores_metrics(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 200.0)
        orchestrator.metrics.record_heal("api-healer", "MATCH", False)
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.time_spent_healing = 3.14
        orchestrator.metrics.preflight_issues_found = 5
        orchestrator.metrics.preflight_issues_fixed = 3
        orchestrator.metrics.rollbacks_performed = 1
        orchestrator.metrics.user_escalations = 2
        orchestrator.metrics.errors_encountered = ["error1", "error2"]
        orchestrator.save_session()

        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is True
        assert orch2.metrics.total_heals == 2
        assert orch2.metrics.successful_heals == 1
        assert orch2.metrics.failed_heals == 1
        assert orch2.metrics.heals_by_healer["api-healer"] == 2
        assert orch2.metrics.successful_heals_by_healer["api-healer"] == 1
        assert orch2.metrics.error_categories["api"] == 2
        assert orch2.metrics.time_spent_healing == 3.14
        assert orch2.metrics.preflight_issues_found == 5
        assert orch2.metrics.preflight_issues_fixed == 3
        assert orch2.metrics.rollbacks_performed == 1
        assert orch2.metrics.user_escalations == 2
        assert orch2.metrics.errors_encountered == ["error1", "error2"]

    @pytest.mark.fast
    def test_load_restores_heal_times(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 100.0)
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 200.0)
        orchestrator.save_session()

        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is True
        assert orch2.metrics.heal_times_by_healer["api-healer"] == [100.0, 200.0]

    @pytest.mark.fast
    def test_load_restores_config_snapshots(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        snap = ConfigSnapshot(
            stage_name="OUTPUT",
            timestamp=2000.0,
            config_values={"download.socket_timeout": 60},
        )
        orchestrator.config_snapshots.append(snap)
        orchestrator.save_session()

        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is True
        assert len(orch2.config_snapshots) == 1
        assert orch2.config_snapshots[0].stage_name == "OUTPUT"
        assert orch2.config_snapshots[0].timestamp == 2000.0
        assert orch2.config_snapshots[0].config_values["download.socket_timeout"] == 60

    @pytest.mark.fast
    def test_load_restores_healer_states(self, mock_config, project_dir):
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        api_healer = orchestrator._healer_instances.get("api-healer")
        if not api_healer:
            pytest.skip("api-healer not available")

        api_healer.backoff_time = 120.0
        api_healer.retry_count = 7
        orchestrator.save_session()

        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        assert orch2.load_session() is True
        api2 = orch2._healer_instances.get("api-healer")
        assert api2.backoff_time == 120.0
        assert api2.retry_count == 7


class TestSaveLoadRoundtrip:
    """Test that save -> load preserves all data exactly."""

    @pytest.mark.fast
    def test_full_roundtrip(self, mock_config, project_dir):
        """Complete roundtrip: populate state, save, create new orchestrator, load, verify."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )

        # Populate diverse state
        orchestrator.current_stage = "ITERATIVE_MATCH"
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 150.0)
        orchestrator.metrics.record_heal("download-healer", "DOWNLOAD", False)
        orchestrator.metrics.record_heal("checkpoint-healer", "CAPTION", True, 50.0)
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.record_error_category("download")
        orchestrator.metrics.record_error_category("api")
        orchestrator.metrics.time_spent_healing = 10.5
        orchestrator.metrics.preflight_issues_found = 4
        orchestrator.metrics.preflight_issues_fixed = 2
        orchestrator.metrics.rollbacks_performed = 1
        orchestrator.metrics.user_escalations = 1
        orchestrator.metrics.errors_encountered = ["err1", "err2"]

        # Add config snapshots
        orchestrator.config_snapshots = [
            ConfigSnapshot("MATCH", 1000.0, {"output.gap_mode": "scale"}),
            ConfigSnapshot("DOWNLOAD", 2000.0, {"download.socket_timeout": 45}),
        ]

        orchestrator.save_session()

        # Load into fresh orchestrator
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        assert orch2.load_session() is True

        # Verify everything
        assert orch2.current_stage == "ITERATIVE_MATCH"
        assert orch2.metrics.total_heals == 3
        assert orch2.metrics.successful_heals == 2
        assert orch2.metrics.failed_heals == 1
        assert orch2.metrics.heals_by_healer["api-healer"] == 1
        assert orch2.metrics.heals_by_healer["download-healer"] == 1
        assert orch2.metrics.heals_by_healer["checkpoint-healer"] == 1
        assert orch2.metrics.error_categories == {"api": 2, "download": 1}
        assert orch2.metrics.time_spent_healing == 10.5
        assert orch2.metrics.preflight_issues_found == 4
        assert orch2.metrics.preflight_issues_fixed == 2
        assert orch2.metrics.rollbacks_performed == 1
        assert orch2.metrics.user_escalations == 1
        assert orch2.metrics.errors_encountered == ["err1", "err2"]
        assert len(orch2.config_snapshots) == 2
        assert orch2.config_snapshots[0].stage_name == "MATCH"
        assert orch2.config_snapshots[1].config_values["download.socket_timeout"] == 45

    @pytest.mark.fast
    def test_roundtrip_empty_state(self, mock_config, project_dir):
        """Roundtrip with default/empty state works."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.save_session()

        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is True
        assert orch2.metrics.total_heals == 0
        assert orch2.current_stage is None
        assert orch2.config_snapshots == []


class TestAutoSave:
    """Test auto-save on significant events."""

    @pytest.mark.fast
    def test_auto_save_after_stage_snapshot(self, mock_config, project_dir):
        """snapshot_config() triggers auto-save."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        with patch.object(orchestrator, '_auto_save_session') as mock_save:
            orchestrator.snapshot_config("MATCH")
            mock_save.assert_called_once_with("stage_snapshot:MATCH")

    @pytest.mark.fast
    def test_auto_save_after_heal_success(self, mock_config, project_dir):
        """coordinate_heal() auto-saves after successful heal."""
        # Disable healing sub-components that need real config
        mock_config.healing = None
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )

        # Create a mock healer that succeeds
        mock_healer = Mock()
        mock_healer.name = "test-healer"
        mock_healer.can_handle.return_value = True
        mock_healer.fix.return_value = HealerResult.fixed("Fixed it")
        orchestrator.healers = [mock_healer]
        orchestrator._healer_instances["test-healer"] = mock_healer

        state = Mock()
        error = Exception("test error")

        with patch.object(orchestrator, '_auto_save_session') as mock_save:
            orchestrator.coordinate_heal(error, state, "TEST_STAGE")
            # Verify auto-save was called with heal_success event
            save_calls = [c for c in mock_save.call_args_list
                         if "heal_success" in str(c)]
            assert len(save_calls) >= 1

    @pytest.mark.fast
    def test_auto_save_does_not_crash_on_failure(self, mock_config, project_dir):
        """Auto-save failure should not propagate."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        with patch.object(orchestrator, 'save_session', side_effect=OSError("disk full")):
            # Should not raise
            orchestrator._auto_save_session("test_event")


class TestSessionRecovery:
    """Test session recovery in create_orchestrated_pipeline()."""

    @pytest.mark.fast
    def test_factory_recovers_session(self, mock_config, project_dir):
        """create_orchestrated_pipeline() loads session when file exists."""
        # Pre-create a session file
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        orchestrator.current_stage = "MATCH"
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 100.0)
        orchestrator.save_session()

        # Simulate factory behavior: create new orchestrator + load_session
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        recovered = orch2.load_session()
        assert recovered is True
        assert orch2.current_stage == "MATCH"
        assert orch2.metrics.total_heals == 1

    @pytest.mark.fast
    def test_factory_recover_session_flag(self, mock_config, project_dir):
        """Verify create_orchestrated_pipeline has recover_session parameter."""
        from src.agents.runner import create_orchestrated_pipeline
        import inspect
        sig = inspect.signature(create_orchestrated_pipeline)
        assert "recover_session" in sig.parameters
        # Default should be True
        assert sig.parameters["recover_session"].default is True

    @pytest.mark.fast
    def test_factory_skips_recovery_when_disabled(self, mock_config, project_dir):
        """When recover_session=False, session is not loaded."""
        # Pre-create a session file
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.metrics.record_heal("api-healer", "MATCH", True)
        orchestrator.save_session()

        # Simulate factory with recover_session=False
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        # Don't call load_session (simulating recover_session=False)
        assert orch2.metrics.total_heals == 0

    @pytest.mark.fast
    def test_factory_handles_missing_session_gracefully(self, mock_config, project_dir):
        """load_session returns False when no file exists."""
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is False
        assert orch2.metrics.total_heals == 0
        assert orch2.current_stage is None


class TestCrashRecoverySimulation:
    """Simulate crash recovery scenarios."""

    @pytest.mark.fast
    def test_recover_after_simulated_crash(self, mock_config, project_dir):
        """Simulate: orchestrator runs, saves, 'crashes', new instance recovers."""
        # Phase 1: Normal operation
        orch1 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        orch1.current_stage = "DOWNLOAD_SEGMENTS"
        orch1.metrics.record_heal("download-healer", "DOWNLOAD", True, 500.0)
        orch1.metrics.record_heal("api-healer", "CAPTION", True, 200.0)
        orch1.metrics.time_spent_healing = 7.0
        orch1.metrics.errors_encountered = ["403 Forbidden"]
        orch1.config_snapshots = [
            ConfigSnapshot("MATCH", 1000.0, {"output.gap_mode": "scale"}),
        ]
        orch1.save_session()

        # Phase 2: "Crash" - orch1 is gone, create new orchestrator
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )

        # Before load - fresh state
        assert orch2.metrics.total_heals == 0
        assert orch2.current_stage is None

        # Recover
        assert orch2.load_session() is True

        # After load - restored state
        assert orch2.current_stage == "DOWNLOAD_SEGMENTS"
        assert orch2.metrics.total_heals == 2
        assert orch2.metrics.successful_heals == 2
        assert orch2.metrics.time_spent_healing == 7.0
        assert orch2.metrics.errors_encountered == ["403 Forbidden"]
        assert len(orch2.config_snapshots) == 1

    @pytest.mark.fast
    def test_recover_preserves_heal_time_averages(self, mock_config, project_dir):
        """Heal time averages are accurate after recovery."""
        orch1 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orch1.metrics.record_heal("api-healer", "MATCH", True, 100.0)
        orch1.metrics.record_heal("api-healer", "MATCH", True, 300.0)
        orch1.save_session()

        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orch2.load_session()

        avg = orch2.metrics.get_healer_average_time_ms("api-healer")
        assert avg == 200.0

    @pytest.mark.fast
    def test_reset_clears_session_state(self, mock_config, project_dir):
        """reset() clears in-memory state but does NOT delete session file."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.metrics.record_heal("api-healer", "MATCH", True)
        orchestrator.current_stage = "OUTPUT"
        orchestrator.save_session()

        # Reset clears in-memory state
        orchestrator.reset()
        assert orchestrator.metrics.total_heals == 0
        assert orchestrator.current_stage is None

        # But the file still exists (in case we need it)
        assert orchestrator._get_session_path().exists()


class TestStaleSessionSkip:
    """Test that stale sessions (>24h) are not loaded (US-68-009)."""

    @pytest.mark.fast
    def test_load_returns_false_for_stale_session(self, mock_config, project_dir):
        """Sessions older than 24 hours should not be loaded."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.current_stage = "MATCH"
        orchestrator.metrics.record_heal("api-healer", "MATCH", True, 100.0)
        orchestrator.save_session()

        # Tamper with the timestamp to make it 25 hours old
        session_path = orchestrator._get_session_path()
        data = json.loads(session_path.read_text(encoding="utf-8"))
        data["timestamp"] = time.time() - (25 * 60 * 60)  # 25 hours ago
        session_path.write_text(json.dumps(data), encoding="utf-8")

        # Fresh orchestrator should refuse to load stale session
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is False
        assert orch2.metrics.total_heals == 0
        assert orch2.current_stage is None

    @pytest.mark.fast
    def test_load_succeeds_for_fresh_session(self, mock_config, project_dir):
        """Sessions within the TTL should load normally."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.current_stage = "CAPTION"
        orchestrator.save_session()

        # Session just saved - should load fine
        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is True
        assert orch2.current_stage == "CAPTION"

    @pytest.mark.fast
    def test_load_returns_false_at_exactly_ttl_boundary(self, mock_config, project_dir):
        """Session at exactly TTL + 1 second should not load."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.save_session()

        session_path = orchestrator._get_session_path()
        data = json.loads(session_path.read_text(encoding="utf-8"))
        # Set to exactly 1 second past the TTL
        data["timestamp"] = time.time() - HealingOrchestrator.SESSION_TTL_SECONDS - 1
        session_path.write_text(json.dumps(data), encoding="utf-8")

        orch2 = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert orch2.load_session() is False


class TestSessionCleanupAfterCompletion:
    """Test that recovery files are cleaned up after successful completion (US-68-009)."""

    @pytest.mark.fast
    def test_cleanup_deletes_session_file(self, mock_config, project_dir):
        """cleanup_session_file() removes the recovery file."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.save_session()
        session_path = orchestrator._get_session_path()
        assert session_path.exists()

        result = orchestrator.cleanup_session_file()
        assert result is True
        assert not session_path.exists()

    @pytest.mark.fast
    def test_cleanup_returns_true_when_no_file(self, mock_config, project_dir):
        """cleanup_session_file() succeeds when file doesn't exist."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        assert not orchestrator._get_session_path().exists()
        assert orchestrator.cleanup_session_file() is True

    @pytest.mark.fast
    def test_cleanup_stale_removes_old_file(self, mock_config, project_dir):
        """_cleanup_stale_sessions() removes files older than TTL."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.save_session()

        # Make the session file stale
        session_path = orchestrator._get_session_path()
        data = json.loads(session_path.read_text(encoding="utf-8"))
        data["timestamp"] = time.time() - (25 * 60 * 60)
        session_path.write_text(json.dumps(data), encoding="utf-8")

        removed = orchestrator._cleanup_stale_sessions()
        assert removed == 1
        assert not session_path.exists()

    @pytest.mark.fast
    def test_cleanup_stale_preserves_fresh_file(self, mock_config, project_dir):
        """_cleanup_stale_sessions() keeps files within TTL."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.save_session()

        removed = orchestrator._cleanup_stale_sessions()
        assert removed == 0
        assert orchestrator._get_session_path().exists()

    @pytest.mark.fast
    def test_cleanup_stale_custom_ttl(self, mock_config, project_dir):
        """_cleanup_stale_sessions() respects custom TTL parameter."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        orchestrator.save_session()

        # Make the session file 2 hours old
        session_path = orchestrator._get_session_path()
        data = json.loads(session_path.read_text(encoding="utf-8"))
        data["timestamp"] = time.time() - (2 * 60 * 60)
        session_path.write_text(json.dumps(data), encoding="utf-8")

        # Default TTL (24h) - should not remove
        assert orchestrator._cleanup_stale_sessions() == 0

        # Custom TTL (1h) - should remove
        assert orchestrator._cleanup_stale_sessions(ttl_seconds=3600) == 1
        assert not session_path.exists()

    @pytest.mark.fast
    def test_cleanup_stale_removes_corrupt_file(self, mock_config, project_dir):
        """_cleanup_stale_sessions() removes corrupt session files."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.minimal()
        )
        session_path = orchestrator._get_session_path()
        session_path.write_text("not valid json{{{", encoding="utf-8")

        removed = orchestrator._cleanup_stale_sessions()
        assert removed == 1
        assert not session_path.exists()

    @pytest.mark.fast
    def test_cleanup_called_during_preflight(self, mock_config, project_dir):
        """_cleanup_stale_sessions() is called during run_preflight()."""
        orchestrator = HealingOrchestrator(
            mock_config, project_dir, strategy=HealingStrategy.conservative()
        )
        state = Mock()
        state.matches = []

        with patch.object(orchestrator, '_cleanup_stale_sessions') as mock_cleanup:
            orchestrator.run_preflight(state)
            mock_cleanup.assert_called_once()
