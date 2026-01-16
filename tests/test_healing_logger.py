"""
Tests for HealingLogger with focus on stage-agnostic behavior and CAPTION stage support.

Verifies:
- HealingLogger works with all pipeline stages including CAPTION
- Log entries correctly serialize stage information
- Report generation includes all stages
- Thread safety across different stages
"""

import json
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from src.agents.healing_logger import HealingLogger, HealingLogEntry
from src.agents.base import HealerResult, HealerAction


# All pipeline stages from checkpoint.py
ALL_STAGES = [
    "ANALYZE",
    "ENTITY_IMAGES",
    "ENTITY_VIDEOS",
    "DOWNLOAD",
    "CAPTION",  # New caption-first stage
    "STOCK",
    "BROLL_DOWNLOAD",
    "REMIX",
    "TRANSCRIBE",
    "SCENE_DETECTION",
    "MATCH",
    "BROLL_MATCH",
    "DOWNLOAD_SEGMENTS",
    "OUTPUT",
]


class TestHealingLoggerStageAgnostic:
    """Tests verifying HealingLogger is stage-agnostic."""

    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_logger(self) -> HealingLogger:
        """Create a HealingLogger for testing."""
        return HealingLogger(self.log_dir, json_log=True, console_format="minimal")

    def _create_test_entry(self, stage: str, msg: str = "test") -> HealingLogEntry:
        """Create a test log entry for a specific stage."""
        return HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage=stage,
            component="test",
            action="test",
            error_type="TestError",
            error_message=msg,
            result="success",
            duration_ms=1.0,
        )

    # --- Stage Acceptance Tests ---

    @pytest.mark.parametrize("stage", ALL_STAGES)
    def test_accepts_all_pipeline_stages(self, stage: str):
        """HealingLogger accepts all pipeline stages."""
        logger = self._create_logger()
        entry = self._create_test_entry(stage, f"Error in {stage}")
        logger._write(entry)

        data = json.loads(logger.json_file.read_text())
        assert len(data) == 1
        assert data[0]["stage"] == stage

    def test_accepts_caption_stage(self):
        """HealingLogger specifically accepts CAPTION stage."""
        logger = self._create_logger()
        entry = self._create_test_entry("CAPTION", "Caption fetch failed")
        logger._write(entry)

        data = json.loads(logger.json_file.read_text())
        assert len(data) == 1
        assert data[0]["stage"] == "CAPTION"
        assert data[0]["error_message"] == "Caption fetch failed"

    def test_accepts_arbitrary_stage_names(self):
        """HealingLogger accepts arbitrary stage names (future-proof)."""
        logger = self._create_logger()

        custom_stages = ["CUSTOM_STAGE", "FUTURE_FEATURE", "V2_PROCESSING"]
        for stage in custom_stages:
            entry = self._create_test_entry(stage, f"Error in {stage}")
            logger._write(entry)

        data = json.loads(logger.json_file.read_text())
        assert len(data) == len(custom_stages)
        stages_logged = [e["stage"] for e in data]
        assert set(stages_logged) == set(custom_stages)


class TestHealingLoggerCaptionStage:
    """Tests specifically for CAPTION stage logging."""

    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_logger(self) -> HealingLogger:
        """Create a HealingLogger for testing."""
        return HealingLogger(self.log_dir, json_log=True, console_format="minimal")

    def test_log_classification_caption_stage(self):
        """log_classification works with CAPTION stage."""
        logger = self._create_logger()

        # Mock classification
        @dataclass
        class MockClassification:
            category: str = "caption"
            severity: str = "medium"
            suggested_healer: str = "caption-healer"
            confidence: float = 0.85
            needs_llm_healer: bool = False
            reasoning: str = "Caption not available for video"

        classification = MockClassification()
        error = Exception("No captions found for video XYZ")

        logger.log_classification(
            stage="CAPTION",
            error=error,
            classification=classification,
            duration_ms=50.0
        )

        data = json.loads(logger.json_file.read_text())
        assert len(data) == 1
        assert data[0]["stage"] == "CAPTION"
        assert data[0]["component"] == "watcher"
        assert data[0]["details"]["category"] == "caption"
        assert data[0]["details"]["suggested_healer"] == "caption-healer"

    def test_log_healer_attempt_caption_stage(self):
        """log_healer_attempt works with CAPTION stage."""
        logger = self._create_logger()

        error = Exception("yt-dlp: No subtitles available")
        result = HealerResult.fixed(
            "Falling back to audio download",
            action=HealerAction.RETRY,
            fallback_triggered=True
        )

        logger.log_healer_attempt(
            stage="CAPTION",
            healer_name="caption-healer",
            error=error,
            result=result,
            duration_ms=100.0
        )

        data = json.loads(logger.json_file.read_text())
        assert len(data) == 1
        assert data[0]["stage"] == "CAPTION"
        assert data[0]["details"]["healer"] == "caption-healer"
        assert data[0]["result"] == "success"

    def test_log_fallback_caption_stage(self):
        """log_fallback works with CAPTION stage."""
        logger = self._create_logger()

        logger.log_fallback(
            stage="CAPTION",
            from_component="caption-healer",
            to_component="download-healer",
            reason="No captions, falling back to audio download"
        )

        data = json.loads(logger.json_file.read_text())
        assert len(data) == 1
        assert data[0]["stage"] == "CAPTION"
        assert data[0]["action"] == "fallback"
        assert "caption" in data[0]["error_message"].lower()

    def test_log_escalation_caption_stage(self):
        """log_escalation works with CAPTION stage."""
        logger = self._create_logger()
        error = Exception("Caption parsing failed with unknown format")

        logger.log_escalation(
            stage="CAPTION",
            error=error,
            reason="Unknown subtitle format, needs LLM analysis",
            to_user=False
        )

        data = json.loads(logger.json_file.read_text())
        assert len(data) == 1
        assert data[0]["stage"] == "CAPTION"
        assert data[0]["action"] == "escalate"


class TestHealingLoggerReport:
    """Tests for report generation with all stages."""

    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_logger(self) -> HealingLogger:
        """Create a HealingLogger for testing."""
        return HealingLogger(self.log_dir, json_log=True, console_format="minimal")

    def test_report_includes_caption_stage(self):
        """generate_report includes CAPTION stage statistics."""
        logger = self._create_logger()

        # Add entries for CAPTION stage
        entry1 = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="CAPTION",
            component="healer",
            action="attempt",
            error_type="CaptionError",
            error_message="No captions available",
            result="success",
            duration_ms=50.0
        )
        entry2 = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="CAPTION",
            component="healer",
            action="attempt",
            error_type="CaptionError",
            error_message="Caption parse failed",
            result="failed",
            duration_ms=30.0
        )
        logger._write(entry1)
        logger._write(entry2)

        report = logger.generate_report()
        assert "CAPTION" in report
        assert "1/2 healed" in report  # 1 success out of 2 attempts

    def test_report_includes_all_stages(self):
        """generate_report includes all pipeline stages with healing activity."""
        logger = self._create_logger()

        # Add entries for multiple stages
        stages_to_test = ["DOWNLOAD", "CAPTION", "TRANSCRIBE", "MATCH", "OUTPUT"]
        for stage in stages_to_test:
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage=stage,
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message=f"Error in {stage}",
                result="success",
                duration_ms=10.0
            )
            logger._write(entry)

        report = logger.generate_report()
        for stage in stages_to_test:
            assert stage in report, f"Stage {stage} not in report"

    def test_report_excludes_special_stages(self):
        """generate_report excludes SELF_HEAL and PREFLIGHT from stage breakdown."""
        logger = self._create_logger()

        # Add entries for special stages
        for stage in ["SELF_HEAL", "PREFLIGHT", "CAPTION"]:
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage=stage,
                component="healer",
                action="attempt",
                error_type="TestError",
                error_message=f"Error in {stage}",
                result="success",
                duration_ms=10.0
            )
            logger._write(entry)

        report = logger.generate_report()
        # CAPTION should appear in "By stage:" section
        assert "CAPTION:" in report
        # SELF_HEAL and PREFLIGHT should NOT appear in "By stage:" section
        lines = report.split("\n")
        stage_section = False
        for line in lines:
            if "By stage:" in line:
                stage_section = True
            if stage_section:
                assert "SELF_HEAL:" not in line
                assert "PREFLIGHT:" not in line


class TestHealingLoggerConcurrency:
    """Tests for thread-safe logging across stages."""

    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_logger(self) -> HealingLogger:
        """Create a HealingLogger for testing."""
        return HealingLogger(self.log_dir, json_log=True, console_format="minimal")

    def test_concurrent_writes_different_stages(self):
        """Concurrent writes from different stages are thread-safe."""
        logger = self._create_logger()
        num_writes_per_stage = 10
        stages = ["DOWNLOAD", "CAPTION", "TRANSCRIBE", "MATCH", "OUTPUT"]
        expected_total = len(stages) * num_writes_per_stage

        errors = []

        def write_for_stage(stage: str):
            try:
                for i in range(num_writes_per_stage):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=stage,
                        component="test",
                        action="test",
                        error_type="TestError",
                        error_message=f"{stage}_entry_{i}",
                        result="success",
                        duration_ms=1.0
                    )
                    logger._write(entry)
            except Exception as e:
                errors.append(f"Stage {stage}: {e}")

        threads = [threading.Thread(target=write_for_stage, args=(stage,))
                   for stage in stages]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"Errors during concurrent writes: {errors}"

        data = json.loads(logger.json_file.read_text())
        assert len(data) == expected_total

        # Verify all stages are represented
        stages_in_data = set(e["stage"] for e in data)
        assert stages_in_data == set(stages)


class TestHealingLogEntrySerialization:
    """Tests for HealingLogEntry serialization."""

    def test_entry_to_dict_preserves_stage(self):
        """to_dict preserves stage field."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="CAPTION",
            component="healer",
            action="fix",
            error_type="CaptionError",
            error_message="Test",
            result="success",
            duration_ms=10.0
        )

        d = entry.to_dict()
        assert d["stage"] == "CAPTION"
        assert d["component"] == "healer"
        assert d["action"] == "fix"

    def test_entry_with_caption_details(self):
        """Entry with caption-specific details serializes correctly."""
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="CAPTION",
            component="healer",
            action="fix",
            error_type="CaptionError",
            error_message="No captions available",
            result="success",
            duration_ms=10.0,
            details={
                "video_id": "abc123",
                "language": "en",
                "is_auto_generated": True,
                "fallback_triggered": True
            }
        )

        d = entry.to_dict()
        assert d["details"]["video_id"] == "abc123"
        assert d["details"]["language"] == "en"
        assert d["details"]["is_auto_generated"] == True
        assert d["details"]["fallback_triggered"] == True


class TestPreflightCheckLogging:
    """Tests for preflight check logging."""

    def setup_method(self):
        """Create temp directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_logger(self) -> HealingLogger:
        """Create a HealingLogger for testing."""
        return HealingLogger(self.log_dir, json_log=True, console_format="minimal")

    def test_log_preflight_check(self):
        """log_preflight_check works correctly."""
        logger = self._create_logger()

        # Log a caption-related preflight check
        logger.log_preflight_check(
            component="caption-healer",
            check_name="yt-dlp_available",
            passed=True,
            message="yt-dlp is installed and accessible"
        )

        data = json.loads(logger.json_file.read_text())
        assert len(data) == 1
        assert data[0]["stage"] == "PREFLIGHT"
        assert data[0]["component"] == "caption-healer"
        assert data[0]["details"]["check_name"] == "yt-dlp_available"
        assert data[0]["details"]["passed"] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
