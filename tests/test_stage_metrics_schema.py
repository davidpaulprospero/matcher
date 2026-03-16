"""
Tests for StageMetrics schema validation (US-85-010).

Verifies that:
- StageType enum exists and categorizes stages
- STAGE_METRICS_SCHEMA maps stage types to required fields
- StageMetrics.validate_metrics() detects missing required fields
- StageResult integrates validation and generates warnings
"""

import pytest
from src.stages import (
    StageType,
    STAGE_METRICS_SCHEMA,
    StageMetrics,
    StageResult,
)


class TestStageType:
    """Test StageType enum exists and has expected values."""

    @pytest.mark.fast
    def test_stage_type_enum_exists(self):
        """Verify StageType enum is defined."""
        assert hasattr(StageType, 'DOWNLOAD')
        assert hasattr(StageType, 'PROCESSING')
        assert hasattr(StageType, 'ANALYSIS')
        assert hasattr(StageType, 'OUTPUT')

    @pytest.mark.fast
    def test_stage_type_values(self):
        """Verify StageType enum values."""
        assert StageType.DOWNLOAD.value == "download"
        assert StageType.PROCESSING.value == "processing"
        assert StageType.ANALYSIS.value == "analysis"
        assert StageType.OUTPUT.value == "output"


class TestStageMetricsSchema:
    """Test STAGE_METRICS_SCHEMA mapping."""

    @pytest.mark.fast
    def test_download_requires_items_processed_and_failed(self):
        """Verify download stage type requires items_processed and items_failed."""
        required = STAGE_METRICS_SCHEMA[StageType.DOWNLOAD]
        assert 'items_processed' in required
        assert 'items_failed' in required

    @pytest.mark.fast
    def test_processing_requires_duration(self):
        """Verify processing stage type requires duration_seconds."""
        required = STAGE_METRICS_SCHEMA[StageType.PROCESSING]
        assert 'duration_seconds' in required

    @pytest.mark.fast
    def test_analysis_requires_items_processed(self):
        """Verify analysis stage type requires items_processed."""
        required = STAGE_METRICS_SCHEMA[StageType.ANALYSIS]
        assert 'items_processed' in required

    @pytest.mark.fast
    def test_output_requires_items_processed(self):
        """Verify output stage type requires items_processed."""
        required = STAGE_METRICS_SCHEMA[StageType.OUTPUT]
        assert 'items_processed' in required


class TestStageMetricsValidation:
    """Test StageMetrics.validate_metrics() method."""

    @pytest.mark.fast
    def test_download_missing_items_failed_triggers_warning(self):
        """Download-type stage missing items_failed should trigger a warning."""
        metrics = StageMetrics(items_processed=10, items_failed=0, duration_seconds=30.0)
        warnings = metrics.validate_metrics(StageType.DOWNLOAD)

        # Should have warning about items_failed (duration_seconds is present)
        assert len(warnings) == 1
        assert "items_failed" in warnings[0]

    @pytest.mark.fast
    def test_download_missing_items_processed_triggers_warning(self):
        """Download-type stage missing items_processed should trigger a warning."""
        metrics = StageMetrics(items_processed=0, items_failed=2, duration_seconds=30.0)
        warnings = metrics.validate_metrics(StageType.DOWNLOAD)

        # Should have warning about items_processed (duration_seconds is present)
        assert len(warnings) == 1
        assert "items_processed" in warnings[0]

    @pytest.mark.fast
    def test_download_with_all_required_fields_no_warning(self):
        """Download-type stage with all required fields should have no warnings."""
        metrics = StageMetrics(items_processed=10, items_failed=2, duration_seconds=30.0)
        warnings = metrics.validate_metrics(StageType.DOWNLOAD)

        assert warnings == []

    @pytest.mark.fast
    def test_processing_missing_duration_triggers_warning(self):
        """Processing-type stage missing duration_seconds should trigger a warning."""
        metrics = StageMetrics(duration_seconds=0.0)
        warnings = metrics.validate_metrics(StageType.PROCESSING)

        assert len(warnings) == 1
        assert "duration_seconds" in warnings[0]

    @pytest.mark.fast
    def test_processing_with_duration_no_warning(self):
        """Processing-type stage with duration_seconds should have no warnings."""
        metrics = StageMetrics(duration_seconds=30.5)
        warnings = metrics.validate_metrics(StageType.PROCESSING)

        assert warnings == []


class TestStageResultMetricsValidation:
    """Test StageResult integrates metrics validation."""

    @pytest.mark.fast
    def test_stage_result_with_download_metrics_missing_failed(self):
        """StageResult with download-type stage missing items_failed should add warning."""
        metrics = StageMetrics(items_processed=10, items_failed=0, duration_seconds=30.0)
        result = StageResult.ok(metrics=metrics, stage_type=StageType.DOWNLOAD)

        # Should have added warning about missing items_failed
        warning_found = any("items_failed" in w for w in result.warnings)
        assert warning_found

    @pytest.mark.fast
    def test_stage_result_without_stage_type_no_validation(self):
        """StageResult without stage_type should not validate metrics."""
        metrics = StageMetrics(items_processed=10, items_failed=0, duration_seconds=30.0)
        result = StageResult.ok(metrics=metrics, stage_type=None)

        # No warnings since no stage_type provided
        assert result.warnings == []

    @pytest.mark.fast
    def test_stage_result_ok_factory_with_validation(self):
        """StageResult.ok() factory should support stage_type parameter."""
        metrics = StageMetrics(items_processed=5, items_failed=1)
        result = StageResult.ok(metrics=metrics, stage_type=StageType.DOWNLOAD)

        assert result.success is True
        assert result.metrics == metrics

    @pytest.mark.fast
    def test_stage_result_fail_factory_with_validation(self):
        """StageResult.fail() factory should support stage_type parameter."""
        metrics = StageMetrics(items_processed=5, items_failed=1)
        result = StageResult.fail("Error", metrics=metrics, stage_type=StageType.DOWNLOAD)

        assert result.success is False
        assert result.error == "Error"
        assert result.metrics == metrics


class TestCheckpointMetricsValidation:
    """Test StageMetrics completeness validation in CheckpointManager.save() (US-106-005)."""

    @pytest.mark.fast
    def test_save_with_incomplete_download_metrics_logs_warning(self, tmp_path, caplog):
        """CheckpointManager.save() logs warning for incomplete download metrics."""
        import logging
        caplog.set_level(logging.WARNING)

        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path, config_hash="test_hash")

        # Incomplete metrics - missing items_failed (required for DOWNLOAD)
        incomplete_metrics = {
            'items_processed': 10,
            # items_failed is missing
            'duration_seconds': 30.0,
        }

        manager.save('DOWNLOAD_SEGMENTS', {'segments': []}, stage_metrics=incomplete_metrics)

        # Should log warning about missing items_failed
        warning_found = any('items_failed' in record.message for record in caplog.records)
        assert warning_found, "Expected warning about missing items_failed"

    @pytest.mark.fast
    def test_save_with_complete_download_metrics_no_warning(self, tmp_path, caplog):
        """CheckpointManager.save() does not log warning for complete download metrics."""
        import logging
        caplog.set_level(logging.WARNING)

        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path, config_hash="test_hash")

        # Complete metrics - all required fields for DOWNLOAD present
        complete_metrics = {
            'items_processed': 10,
            'items_failed': 2,
            'duration_seconds': 30.0,
        }

        manager.save('DOWNLOAD_SEGMENTS', {'segments': []}, stage_metrics=complete_metrics)

        # Should not log warning about missing fields
        warning_found = any('required field' in record.message for record in caplog.records)
        assert not warning_found, "Should not warn when all required fields present"

    @pytest.mark.fast
    def test_save_with_incomplete_processing_metrics_logs_warning(self, tmp_path, caplog):
        """CheckpointManager.save() logs warning for incomplete processing metrics."""
        import logging
        caplog.set_level(logging.WARNING)

        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path, config_hash="test_hash")

        # Incomplete metrics - missing duration_seconds (required for PROCESSING)
        incomplete_metrics = {
            'items_processed': 10,
            # duration_seconds is missing
        }

        manager.save('ANALYZE', {'keywords': []}, stage_metrics=incomplete_metrics)

        # Should log warning about missing duration_seconds
        warning_found = any('duration_seconds' in record.message for record in caplog.records)
        assert warning_found, "Expected warning about missing duration_seconds"

    @pytest.mark.fast
    def test_save_with_unknown_stage_no_validation(self, tmp_path, caplog):
        """CheckpointManager.save() skips validation for unknown stage names."""
        import logging
        caplog.set_level(logging.WARNING)

        from src.checkpoint import CheckpointManager

        manager = CheckpointManager(tmp_path, config_hash="test_hash")

        # Metrics with unknown stage - should not cause error
        metrics = {
            'items_processed': 10,
            'items_failed': 2,
        }

        # This should not raise - UNKNOWN_STAGE is not in _STAGE_NAME_TO_TYPE
        manager.save('UNKNOWN_STAGE', {'data': []}, stage_metrics=metrics)

        # Should not log validation warnings for unknown stage
        warning_found = any('required field' in record.message for record in caplog.records)
        assert not warning_found, "Unknown stages should be skipped"
