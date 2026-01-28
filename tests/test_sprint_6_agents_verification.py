"""
Sprint 6 Agents Verification Tests.

Verifies that all agent modules import cleanly without errors.
This is a smoke test to ensure the self-healing agents infrastructure
is properly structured and has no import-time errors.
"""

import pytest


class TestAgentsMainImports:
    """Test that main agent modules import cleanly."""

    @pytest.mark.fast
    def test_agents_init_imports(self):
        """Test that src/agents/__init__.py imports cleanly."""
        # This should not raise any ImportError
        from src.agents import (
            # Base classes
            Healer,
            HealerResult,
            HealerAction,
            # Strategy
            HealingStrategy,
            HealingMode,
            HealingMetrics,
            ConfigSnapshot,
            # Orchestrator
            HealingOrchestrator,
            PreflightIssue,
            EscalationRequest,
            # Runner
            ResilientRunner,
            # Factory functions
            create_resilient_pipeline,
            create_orchestrated_pipeline,
            # Individual healers
            OTIOHealer,
            APIHealer,
            CheckpointHealer,
            DownloadHealer,
            DiskHealer,
            PathHealer,
            LLMHealer,
            HEALER_REGISTRY,
            # Two-tier LLM delegation
            WatcherAgent,
            ErrorClassification,
            HealingLogger,
            HealingLogEntry,
            FallbackChain,
            pattern_route,
            PATTERN_ROUTING,
        )

        # Verify __all__ contains expected exports
        from src import agents
        assert hasattr(agents, '__all__')
        assert 'Healer' in agents.__all__
        assert 'HealingOrchestrator' in agents.__all__
        assert 'ResilientRunner' in agents.__all__

    @pytest.mark.fast
    def test_base_module_imports(self):
        """Test that src/agents/base.py imports cleanly."""
        from src.agents.base import Healer, HealerResult, HealerAction

        # Verify HealerAction enum values
        assert HealerAction.RETRY is not None
        assert HealerAction.SKIP is not None
        assert HealerAction.MODIFY_CONFIG is not None
        assert HealerAction.RESTORE is not None
        assert HealerAction.ABORT is not None

    @pytest.mark.fast
    def test_strategy_module_imports(self):
        """Test that src/agents/strategy.py imports cleanly."""
        from src.agents.strategy import (
            HealingStrategy,
            HealingMode,
            HealingMetrics,
            ConfigSnapshot,
        )

        # Verify HealingMode enum values
        assert HealingMode.AGGRESSIVE is not None
        assert HealingMode.CONSERVATIVE is not None
        assert HealingMode.INTERACTIVE is not None
        assert HealingMode.MINIMAL is not None

        # Verify factory methods exist
        assert callable(HealingStrategy.aggressive)
        assert callable(HealingStrategy.conservative)
        assert callable(HealingStrategy.interactive)
        assert callable(HealingStrategy.minimal)

    @pytest.mark.fast
    def test_orchestrator_module_imports(self):
        """Test that src/agents/orchestrator.py imports cleanly."""
        from src.agents.orchestrator import (
            HealingOrchestrator,
            PreflightIssue,
            EscalationRequest,
            create_orchestrated_pipeline,
        )

        # Verify dataclass structure
        issue = PreflightIssue(
            category="test",
            severity="warning",
            message="Test message"
        )
        assert issue.category == "test"
        assert issue.auto_fixable is False  # default

        request = EscalationRequest(
            stage_name="TEST",
            error="Test error",
            options=["retry", "skip"]
        )
        assert request.stage_name == "TEST"
        assert request.recommendation is None  # default

    @pytest.mark.fast
    def test_runner_module_imports(self):
        """Test that src/agents/runner.py imports cleanly."""
        from src.agents.runner import (
            ResilientRunner,
            create_resilient_pipeline,
            create_orchestrated_pipeline,
        )

        # Verify class constants
        assert hasattr(ResilientRunner, 'MAX_HEAL_ATTEMPTS')
        assert hasattr(ResilientRunner, 'HEAL_DELAY_SECONDS')


class TestHealerImports:
    """Test that all healers in src/agents/healers/ import cleanly."""

    @pytest.mark.fast
    def test_healers_init_imports(self):
        """Test that src/agents/healers/__init__.py imports cleanly."""
        from src.agents.healers import (
            OTIOHealer,
            APIHealer,
            CheckpointHealer,
            DownloadHealer,
            DiskHealer,
            PathHealer,
            LLMHealer,
            HEALER_REGISTRY,
        )

        # Verify registry contains expected healers
        assert len(HEALER_REGISTRY) == 6
        assert CheckpointHealer in HEALER_REGISTRY
        assert APIHealer in HEALER_REGISTRY
        assert DownloadHealer in HEALER_REGISTRY
        assert DiskHealer in HEALER_REGISTRY
        assert PathHealer in HEALER_REGISTRY
        assert OTIOHealer in HEALER_REGISTRY
        # LLMHealer should NOT be in registry (invoked by orchestrator)
        assert LLMHealer not in HEALER_REGISTRY

    @pytest.mark.fast
    def test_api_healer_imports(self):
        """Test that src/agents/healers/api.py imports cleanly."""
        from src.agents.healers.api import APIHealer
        assert APIHealer.name == "api-healer"

    @pytest.mark.fast
    def test_checkpoint_healer_imports(self):
        """Test that src/agents/healers/checkpoint.py imports cleanly."""
        from src.agents.healers.checkpoint import CheckpointHealer
        assert CheckpointHealer.name == "checkpoint-healer"

    @pytest.mark.fast
    def test_disk_healer_imports(self):
        """Test that src/agents/healers/disk.py imports cleanly."""
        from src.agents.healers.disk import DiskHealer
        assert DiskHealer.name == "disk-healer"

    @pytest.mark.fast
    def test_download_healer_imports(self):
        """Test that src/agents/healers/download.py imports cleanly."""
        from src.agents.healers.download import DownloadHealer
        assert DownloadHealer.name == "download-healer"

    @pytest.mark.fast
    def test_path_healer_imports(self):
        """Test that src/agents/healers/path.py imports cleanly."""
        from src.agents.healers.path import PathHealer
        assert PathHealer.name == "path-healer"

    @pytest.mark.fast
    def test_otio_healer_imports(self):
        """Test that src/agents/healers/otio.py imports cleanly."""
        from src.agents.healers.otio import OTIOHealer
        assert OTIOHealer.name == "otio-healer"

    @pytest.mark.fast
    def test_llm_healer_imports(self):
        """Test that src/agents/healers/llm_healer.py imports cleanly."""
        from src.agents.healers.llm_healer import LLMHealer
        assert LLMHealer.name == "llm-healer"


class TestWatcherAndFallbackImports:
    """Test two-tier LLM delegation components import cleanly."""

    @pytest.mark.fast
    def test_watcher_module_imports(self):
        """Test that src/agents/watcher.py imports cleanly."""
        from src.agents.watcher import WatcherAgent, ErrorClassification

        # Verify ErrorClassification has expected fields
        classification = ErrorClassification(
            category="api",
            severity="recoverable",
            suggested_healer="api-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="test classification"
        )
        assert classification.category == "api"
        assert classification.confidence == 0.9

    @pytest.mark.fast
    def test_fallback_module_imports(self):
        """Test that src/agents/fallback.py imports cleanly."""
        from src.agents.fallback import (
            FallbackChain,
            pattern_route,
            PATTERN_ROUTING,
            PatternClassification,
        )

        # Verify PATTERN_ROUTING is a dict of regex patterns to (category, healer) tuples
        assert isinstance(PATTERN_ROUTING, dict)
        assert len(PATTERN_ROUTING) > 0

        # Verify pattern_route returns valid classification
        result = pattern_route("rate limit exceeded")
        assert hasattr(result, 'category')
        assert hasattr(result, 'suggested_healer')

    @pytest.mark.fast
    def test_healing_logger_imports(self):
        """Test that src/agents/healing_logger.py imports cleanly."""
        from src.agents.healing_logger import HealingLogger, HealingLogEntry
        from datetime import datetime

        # Verify HealingLogEntry has expected fields
        entry = HealingLogEntry(
            timestamp=datetime.now(),
            stage="TEST",
            component="healer",
            action="attempt",
            error_type="RateLimitError",
            error_message="Rate limit exceeded",
            result="success",
            duration_ms=100.0
        )
        assert entry.component == "healer"
        assert entry.action == "attempt"


class TestHealerBaseClass:
    """Test that Healer base class functionality works."""

    @pytest.mark.fast
    def test_healer_result_factory_methods(self):
        """Test HealerResult.fixed(), .failed(), .config_changed() methods."""
        from src.agents.base import HealerResult, HealerAction

        # Test fixed()
        result = HealerResult.fixed("Issue resolved")
        assert result.success is True
        assert result.action == HealerAction.RETRY
        assert result.message == "Issue resolved"

        # Test failed()
        result = HealerResult.failed("Cannot fix")
        assert result.success is False
        assert result.action == HealerAction.ABORT
        assert result.message == "Cannot fix"

        # Test config_changed()
        result = HealerResult.config_changed("Increased timeout")
        assert result.success is True
        assert result.action == HealerAction.MODIFY_CONFIG
        assert result.modified_config is True

    @pytest.mark.fast
    def test_healer_action_enum_complete(self):
        """Test all HealerAction enum values are present."""
        from src.agents.base import HealerAction

        actions = [action.value for action in HealerAction]
        assert "retry" in actions
        assert "skip" in actions
        assert "modify" in actions  # MODIFY_CONFIG uses "modify"
        assert "restore" in actions
        assert "abort" in actions


class TestHealingMetrics:
    """Test HealingMetrics tracking functionality."""

    @pytest.mark.fast
    def test_metrics_initialization(self):
        """Test HealingMetrics initializes with zeroed counters."""
        from src.agents.strategy import HealingMetrics

        metrics = HealingMetrics()
        assert metrics.total_heals == 0
        assert metrics.successful_heals == 0
        assert metrics.preflight_issues_found == 0
        assert metrics.preflight_issues_fixed == 0
        assert metrics.rollbacks_performed == 0
        assert metrics.user_escalations == 0
        assert len(metrics.heals_by_healer) == 0
        assert len(metrics.heals_by_stage) == 0

    @pytest.mark.fast
    def test_metrics_record_heal(self):
        """Test HealingMetrics.record_heal() updates counters."""
        from src.agents.strategy import HealingMetrics

        metrics = HealingMetrics()

        # Record successful heal
        metrics.record_heal("api-healer", "DOWNLOAD", success=True)
        assert metrics.total_heals == 1
        assert metrics.successful_heals == 1
        assert metrics.heals_by_healer.get("api-healer") == 1
        assert metrics.heals_by_stage.get("DOWNLOAD") == 1

        # Record failed heal
        metrics.record_heal("disk-healer", "OUTPUT", success=False)
        assert metrics.total_heals == 2
        assert metrics.successful_heals == 1  # unchanged
        assert metrics.heals_by_healer.get("disk-healer") == 1

    @pytest.mark.fast
    def test_metrics_summary(self):
        """Test HealingMetrics.summary() returns formatted string."""
        from src.agents.strategy import HealingMetrics

        metrics = HealingMetrics()
        metrics.record_heal("api-healer", "DOWNLOAD", success=True)

        summary = metrics.summary()
        assert isinstance(summary, str)
        assert "Total heals" in summary or "1" in summary


class TestConfigSnapshot:
    """Test ConfigSnapshot save/restore functionality."""

    @pytest.mark.fast
    def test_config_snapshot_creation(self):
        """Test ConfigSnapshot stores stage name and timestamp."""
        from src.agents.strategy import ConfigSnapshot
        import time

        snapshot = ConfigSnapshot(
            stage_name="TEST_STAGE",
            timestamp=time.time(),
            config_values={"output.max_clips": 100}
        )

        assert snapshot.stage_name == "TEST_STAGE"
        assert snapshot.timestamp > 0
        assert "output.max_clips" in snapshot.config_values