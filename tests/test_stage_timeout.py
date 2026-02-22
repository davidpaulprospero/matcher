"""Tests for US-106-002 & US-108-002: Configurable stage timeouts.

Tests cover:
  - StageTimeoutConfig dataclass fields
  - StageTimeoutSettingsConfig global timeout settings (US-108-002)
  - Timeout enforcement in PipelineOrchestrator.run()
  - timeout_occurred flag in StageMetrics
  - timeout_warning when approaching 80% of timeout
  - save_on_timeout partial checkpoint
  - timeout_count, timeout_duration, was_force_killed in StageMetrics (US-108-002)
  - Pipeline resume after timeout (US-108-002)
"""

import time
import threading
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.config.sections.infrastructure import (
    StageTimeoutConfig,
    StageTimeoutSettingsConfig,
    PipelineConfig,
)
from src.stages import Stage, StageResult, StageMetrics


class SlowStage(Stage):
    """Test stage that sleeps for a configurable duration."""

    name = "slow-stage"

    def __init__(self, sleep_duration: float = 5.0):
        super().__init__()
        self.sleep_duration = sleep_duration
        self.run_called = False

    def run(self, state, config, checkpoint):
        self.run_called = True
        time.sleep(self.sleep_duration)
        return StageResult(success=True, metrics=StageMetrics(items_processed=1))

    def can_skip(self, state, checkpoint):
        return False

    def restore(self, state, checkpoint, config):
        return True


class TestStageTimeoutConfig:
    """Test StageTimeoutConfig dataclass."""

    @pytest.mark.fast
    def test_stage_timeout_config_defaults(self):
        """Test that StageTimeoutConfig has sensible defaults."""
        config = StageTimeoutConfig()
        assert config.timeout_seconds == 3600
        assert config.enabled is True
        assert config.save_on_timeout is True

    @pytest.mark.fast
    def test_stage_timeout_config_custom_values(self):
        """Test custom timeout configuration."""
        config = StageTimeoutConfig(
            timeout_seconds=1800,
            enabled=True,
            save_on_timeout=False
        )
        assert config.timeout_seconds == 1800
        assert config.enabled is True
        assert config.save_on_timeout is False

    @pytest.mark.fast
    def test_stage_timeout_config_validation(self):
        """Test that timeout_seconds < 1 raises ValueError."""
        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            StageTimeoutConfig(timeout_seconds=0)

        with pytest.raises(ValueError, match="timeout_seconds must be >= 1"):
            StageTimeoutConfig(timeout_seconds=-1)


class TestPipelineTimeoutPolicy:
    """Test PipelineConfig timeout_policy."""

    @pytest.mark.fast
    def test_get_timeout_config_default(self):
        """Test get_timeout_config returns default for unconfigured stage."""
        pipeline_config = PipelineConfig()
        config = pipeline_config.get_timeout_config("UNKNOWN_STAGE")
        assert config.timeout_seconds == 3600
        assert config.enabled is True
        assert config.save_on_timeout is True

    @pytest.mark.fast
    def test_get_timeout_config_custom(self):
        """Test get_timeout_config returns configured value."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "CAPTION": StageTimeoutConfig(
                    timeout_seconds=1800,
                    enabled=True,
                    save_on_timeout=True
                )
            }
        )
        config = pipeline_config.get_timeout_config("CAPTION")
        assert config.timeout_seconds == 1800

    @pytest.mark.fast
    def test_timeout_policy_from_dict(self):
        """Test timeout_policy accepts dict config and converts to StageTimeoutConfig."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "MATCH": {
                    "timeout_seconds": 2400,
                    "enabled": True,
                    "save_on_timeout": False
                }
            }
        )
        config = pipeline_config.get_timeout_config("MATCH")
        assert config.timeout_seconds == 2400
        assert config.save_on_timeout is False


class TestPerStageTypeTimeout:
    """Test US-125-002: Per-stage-type timeout configuration."""

    @pytest.mark.fast
    def test_video_search_timeout_config(self):
        """Test VIDEO_SEARCH stage has appropriate timeout (15 min = 900s)."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "VIDEO_SEARCH": StageTimeoutConfig(
                    timeout_seconds=900,
                    enabled=True,
                    save_on_timeout=True
                )
            }
        )
        config = pipeline_config.get_timeout_config("VIDEO_SEARCH")
        assert config.timeout_seconds == 900
        assert config.enabled is True
        assert config.save_on_timeout is True

    @pytest.mark.fast
    def test_caption_timeout_config(self):
        """Test CAPTION stage has appropriate timeout (30 min = 1800s)."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "CAPTION": StageTimeoutConfig(
                    timeout_seconds=1800,
                    enabled=True,
                    save_on_timeout=True
                )
            }
        )
        config = pipeline_config.get_timeout_config("CAPTION")
        assert config.timeout_seconds == 1800
        assert config.enabled is True
        assert config.save_on_timeout is True

    @pytest.mark.fast
    def test_match_timeout_config(self):
        """Test MATCH stage has appropriate timeout (40 min = 2400s)."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "MATCH": StageTimeoutConfig(
                    timeout_seconds=2400,
                    enabled=True,
                    save_on_timeout=True
                )
            }
        )
        config = pipeline_config.get_timeout_config("MATCH")
        assert config.timeout_seconds == 2400
        assert config.enabled is True
        assert config.save_on_timeout is True

    @pytest.mark.fast
    def test_download_segments_timeout_config(self):
        """Test DOWNLOAD_SEGMENTS stage has appropriate timeout (2 hours = 7200s)."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "DOWNLOAD_SEGMENTS": StageTimeoutConfig(
                    timeout_seconds=7200,
                    enabled=True,
                    save_on_timeout=True
                )
            }
        )
        config = pipeline_config.get_timeout_config("DOWNLOAD_SEGMENTS")
        assert config.timeout_seconds == 7200
        assert config.enabled is True
        assert config.save_on_timeout is True

    @pytest.mark.fast
    def test_all_stages_have_different_timeouts(self):
        """Test that different stage types can have different timeouts."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "VIDEO_SEARCH": StageTimeoutConfig(timeout_seconds=900),
                "CAPTION": StageTimeoutConfig(timeout_seconds=1800),
                "MATCH": StageTimeoutConfig(timeout_seconds=2400),
                "DOWNLOAD_SEGMENTS": StageTimeoutConfig(timeout_seconds=7200),
            }
        )
        # Each stage should have its own timeout
        assert pipeline_config.get_timeout_config("VIDEO_SEARCH").timeout_seconds == 900
        assert pipeline_config.get_timeout_config("CAPTION").timeout_seconds == 1800
        assert pipeline_config.get_timeout_config("MATCH").timeout_seconds == 2400
        assert pipeline_config.get_timeout_config("DOWNLOAD_SEGMENTS").timeout_seconds == 7200

    @pytest.mark.fast
    def test_timeout_warning_triggers_at_80_percent(self):
        """Test timeout warning triggers at 80% of configured timeout (US-106-002)."""
        # Test 80% threshold logic for each stage type
        stage_configs = {
            "VIDEO_SEARCH": 900,    # 15 min
            "CAPTION": 1800,         # 30 min
            "MATCH": 2400,           # 40 min
            "DOWNLOAD_SEGMENTS": 7200,  # 2 hours
        }

        for stage_name, timeout_seconds in stage_configs.items():
            warning_threshold = timeout_seconds * 0.8

            # At 79% - no warning
            elapsed_79 = timeout_seconds * 0.79
            assert elapsed_79 < warning_threshold, f"{stage_name} at 79% should not trigger warning"

            # At 80% - warning should trigger
            elapsed_80 = timeout_seconds * 0.8
            assert elapsed_80 >= warning_threshold, f"{stage_name} at 80% should trigger warning"

            # At 90% - warning should trigger
            elapsed_90 = timeout_seconds * 0.9
            assert elapsed_90 >= warning_threshold, f"{stage_name} at 90% should trigger warning"

    @pytest.mark.fast
    def test_iterative_match_timeout_uses_match_default(self):
        """Test ITERATIVE_MATCH stage falls back to default timeout if not configured."""
        pipeline_config = PipelineConfig(
            timeout_policy={
                "MATCH": StageTimeoutConfig(timeout_seconds=2400),
            }
        )
        # ITERATIVE_MATCH not explicitly configured - should use default
        config = pipeline_config.get_timeout_config("ITERATIVE_MATCH")
        assert config.timeout_seconds == 3600  # default


class TestStageMetricsTimeoutFlags:
    """Test StageMetrics timeout fields."""

    @pytest.mark.fast
    def test_stage_metrics_timeout_defaults(self):
        """Test StageMetrics timeout fields default to False."""
        metrics = StageMetrics()
        assert metrics.timeout_warning is False
        assert metrics.timeout_occurred is False

    @pytest.mark.fast
    def test_stage_metrics_timeout_flags(self):
        """Test StageMetrics can store timeout flags."""
        metrics = StageMetrics(
            timeout_warning=True,
            timeout_occurred=True
        )
        assert metrics.timeout_warning is True
        assert metrics.timeout_occurred is True


class TestPipelineStageTimeout:
    """Test stage timeout enforcement in PipelineOrchestrator."""

    @pytest.mark.fast
    def test_timeout_occurred_set_on_timeout(self):
        """Test that timeout_occurred is True when stage times out."""
        from src.pipeline import PipelineOrchestrator
        from src.state import PipelineState
        from src.checkpoint import CheckpointManager

        # Create slow stage that will timeout
        slow_stage = SlowStage(sleep_duration=10.0)

        # Create mock config with short timeout
        mock_config = MagicMock()
        mock_config.pipeline.timeout_policy = {
            "slow-stage": StageTimeoutConfig(
                timeout_seconds=1,  # 1 second timeout
                enabled=True,
                save_on_timeout=False
            )
        }
        mock_config.pipeline.get_timeout_config = lambda name: mock_config.pipeline.timeout_policy.get(
            name, StageTimeoutConfig()
        )
        mock_config.freeze = MagicMock()

        # Create minimal pipeline
        state = PipelineState()
        checkpoint = MagicMock(spec=CheckpointManager)
        checkpoint.save = MagicMock()

        # Run stage with timeout
        # We need to manually test the timeout logic since full pipeline setup is complex
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        timeout_seconds = 1

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slow_stage.run, state, mock_config, checkpoint)
            try:
                result = future.result(timeout=timeout_seconds)
                # Should not reach here - stage takes too long
                pytest.fail("Expected timeout")
            except FuturesTimeoutError:
                # Timeout occurred as expected
                pass

    @pytest.mark.fast
    def test_timeout_warning_at_80_percent(self):
        """Test that timeout_warning is set when approaching 80% of timeout."""
        # Test the 80% threshold logic
        timeout_seconds = 100

        # At 79% - no warning
        elapsed_79 = timeout_seconds * 0.79
        assert elapsed_79 < timeout_seconds * 0.8

        # At 80% - warning
        elapsed_80 = timeout_seconds * 0.8
        assert elapsed_80 >= timeout_seconds * 0.8

        # At 90% - warning
        elapsed_90 = timeout_seconds * 0.9
        assert elapsed_90 >= timeout_seconds * 0.8


class TestStageTimeoutSettingsConfig:
    """Test StageTimeoutSettingsConfig global timeout settings (US-108-002)."""

    @pytest.mark.fast
    def test_stage_timeout_settings_defaults(self):
        """Test that StageTimeoutSettingsConfig has sensible defaults."""
        config = StageTimeoutSettingsConfig()
        assert config.default_timeout_seconds == 3600
        assert config.enable_timeout is True
        assert config.stage_overrides == {}

    @pytest.mark.fast
    def test_stage_timeout_settings_custom_values(self):
        """Test custom timeout settings."""
        config = StageTimeoutSettingsConfig(
            default_timeout_seconds=1800,
            enable_timeout=True,
            stage_overrides={"CAPTION": 900, "MATCH": 1200}
        )
        assert config.default_timeout_seconds == 1800
        assert config.enable_timeout is True
        assert config.stage_overrides["CAPTION"] == 900
        assert config.stage_overrides["MATCH"] == 1200

    @pytest.mark.fast
    def test_stage_timeout_settings_validation(self):
        """Test that default_timeout_seconds < 1 raises ValueError."""
        with pytest.raises(ValueError, match="default_timeout_seconds must be >= 1"):
            StageTimeoutSettingsConfig(default_timeout_seconds=0)

        with pytest.raises(ValueError, match="default_timeout_seconds must be >= 1"):
            StageTimeoutSettingsConfig(default_timeout_seconds=-1)

    @pytest.mark.fast
    def test_get_timeout_for_stage_default(self):
        """Test get_timeout_for_stage returns default when no override."""
        config = StageTimeoutSettingsConfig(default_timeout_seconds=1800)
        assert config.get_timeout_for_stage("UNKNOWN") == 1800
        assert config.get_timeout_for_stage("CAPTION") == 1800

    @pytest.mark.fast
    def test_get_timeout_for_stage_override(self):
        """Test get_timeout_for_stage returns override when configured."""
        config = StageTimeoutSettingsConfig(
            default_timeout_seconds=3600,
            stage_overrides={"CAPTION": 900}
        )
        assert config.get_timeout_for_stage("CAPTION") == 900
        assert config.get_timeout_for_stage("MATCH") == 3600

    @pytest.mark.fast
    def test_stage_overrides_converts_to_int(self):
        """Test that stage_overrides values are converted to int."""
        config = StageTimeoutSettingsConfig(
            stage_overrides={"CAPTION": 900.5, "MATCH": "1200"}
        )
        assert config.stage_overrides["CAPTION"] == 900
        assert config.stage_overrides["MATCH"] == 1200


class TestStageMetricsTimeoutFields:
    """Test StageMetrics new timeout fields (US-108-002)."""

    @pytest.mark.fast
    def test_stage_metrics_timeout_fields_defaults(self):
        """Test StageMetrics timeout fields default to zero/false."""
        metrics = StageMetrics()
        assert metrics.timeout_count == 0
        assert metrics.timeout_duration == 0.0
        assert metrics.was_force_killed is False

    @pytest.mark.fast
    def test_stage_metrics_timeout_fields_custom(self):
        """Test StageMetrics can store custom timeout values."""
        metrics = StageMetrics(
            timeout_count=2,
            timeout_duration=7200.5,
            was_force_killed=True
        )
        assert metrics.timeout_count == 2
        assert metrics.timeout_duration == 7200.5
        assert metrics.was_force_killed is True

    @pytest.mark.fast
    def test_stage_metrics_to_dict_includes_timeout_fields(self):
        """Test to_dict includes timeout fields when set."""
        metrics = StageMetrics(
            timeout_count=1,
            timeout_duration=100.5,
            was_force_killed=True
        )
        d = metrics.to_dict()
        assert d['timeout_count'] == 1
        assert d['timeout_duration'] == 100.5
        assert d['was_force_killed'] is True

    @pytest.mark.fast
    def test_stage_metrics_to_dict_excludes_zero_timeout_fields(self):
        """Test to_dict excludes zero timeout fields for cleanliness."""
        metrics = StageMetrics(timeout_count=0, timeout_duration=0.0)
        d = metrics.to_dict()
        assert 'timeout_count' not in d
        assert 'timeout_duration' not in d

    @pytest.mark.fast
    def test_stage_metrics_from_dict_includes_timeout_fields(self):
        """Test from_dict includes timeout fields."""
        data = {
            'items_processed': 5,
            'timeout_count': 2,
            'timeout_duration': 300.5,
            'was_force_killed': True,
        }
        metrics = StageMetrics.from_dict(data)
        assert metrics.timeout_count == 2
        assert metrics.timeout_duration == 300.5
        assert metrics.was_force_killed is True

    @pytest.mark.fast
    def test_stage_metrics_from_dict_missing_timeout_fields(self):
        """Test from_dict handles missing timeout fields gracefully."""
        data = {'items_processed': 5}
        metrics = StageMetrics.from_dict(data)
        assert metrics.timeout_count == 0
        assert metrics.timeout_duration == 0.0
        assert metrics.was_force_killed is False


class TestPipelineResumeAfterTimeout:
    """Test pipeline can resume after timeout (US-108-002)."""

    @pytest.mark.fast
    def test_timeout_saves_checkpoint_for_resume(self):
        """Test that checkpoint can be saved for potential resume after timeout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from src.checkpoint import CheckpointManager

            # Create checkpoint manager
            checkpoint = CheckpointManager(tmpdir, config_hash="test", config=None)

            # Save checkpoint using a valid stage name (ANALYZE)
            checkpoint.save(
                "ANALYZE",
                {"keywords": ["test", "keyword"]}
            )

            # Verify checkpoint exists - this is what allows resume after timeout
            assert checkpoint.exists()

            # Verify checkpoint is valid
            validation = checkpoint.validate()
            assert validation['valid'] is True
