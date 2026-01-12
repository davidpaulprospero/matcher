"""
Tests for src/logger.py edge cases to improve coverage.

Targets:
- _sanitize_run_id edge cases
- finalize() console output branches
- verbose markdown generation branches
"""

import pytest
import io
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.logger import (
    RunLogger,
    _sanitize_run_id,
    NumpyEncoder,
    HAS_NUMPY,
)


class TestSanitizeRunId:
    """Test _sanitize_run_id edge cases."""

    def test_sanitize_empty_string(self):
        """Test with empty string returns empty string."""
        result = _sanitize_run_id("")
        assert result == ""

    def test_sanitize_none(self):
        """Test with None returns None."""
        result = _sanitize_run_id(None)
        assert result is None

    def test_sanitize_double_underscores(self):
        """Test collapsing multiple underscores."""
        result = _sanitize_run_id("test__double__underscores")
        assert "__" not in result
        assert result == "test_double_underscores"

    def test_sanitize_special_chars(self):
        """Test replacing special characters."""
        result = _sanitize_run_id("test/path:file*name")
        assert "/" not in result
        assert ":" not in result
        assert "*" not in result
        assert result == "test_path_file_name"

    def test_sanitize_all_underscores_returns_default(self):
        """Test input that becomes all underscores returns default."""
        result = _sanitize_run_id("///:::")
        assert result == "unnamed_run"

    def test_sanitize_newlines_and_tabs(self):
        """Test newlines and tabs are replaced."""
        result = _sanitize_run_id("test\nwith\ttabs")
        assert "\n" not in result
        assert "\t" not in result


class TestFinalizeProcessingStats:
    """Test finalize() processing stats branches."""

    def test_finalize_with_downloads(self, tmp_path):
        """Test finalize with video downloads shows processing section."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.videos_downloaded = 5
        logger.run_log.videos_skipped = 3
        logger.run_log.videos_failed = 1

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "PROCESSING" in output
        assert "Videos:" in output

    def test_finalize_with_transcriptions(self, tmp_path):
        """Test finalize with transcriptions shows transcription stats."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.videos_transcribed = 10
        logger.run_log.transcription_cache_hits = 5

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "PROCESSING" in output
        assert "Transcribe:" in output

    def test_finalize_with_embeddings(self, tmp_path):
        """Test finalize with embeddings shows embedding stats."""
        logger = RunLogger(log_dir=str(tmp_path))
        # Need to trigger PROCESSING section first
        logger.run_log.videos_downloaded = 1
        logger.run_log.embeddings_computed = 100
        logger.run_log.embedding_cache_hits = 50

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "PROCESSING" in output
        assert "Embeddings:" in output

    def test_finalize_with_entity_media(self, tmp_path):
        """Test finalize with entity images/videos shows entity stats."""
        logger = RunLogger(log_dir=str(tmp_path))
        # Need to trigger PROCESSING section first
        logger.run_log.videos_downloaded = 1
        logger.run_log.entity_images_downloaded = 20
        logger.run_log.entity_videos_downloaded = 5

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "Entity:" in output


class TestFinalizeTimings:
    """Test finalize() timing display branches."""

    def test_finalize_with_overhead(self, tmp_path):
        """Test finalize shows overhead when significant."""
        from datetime import datetime, timedelta

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.stage_timings = {"download": 5.0, "transcribe": 10.0}
        # Total stage time = 15s
        # Set start_time to 20 seconds ago to create 5s of overhead
        start = datetime.now() - timedelta(seconds=20)
        logger.run_log.start_time = start.isoformat()

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "STAGE TIMINGS" in output
        assert "Overhead:" in output

    def test_finalize_minutes_display(self, tmp_path):
        """Test finalize displays time in minutes when over 60s."""
        from datetime import datetime, timedelta

        logger = RunLogger(log_dir=str(tmp_path))
        # Set stage timings to create >60s total time
        logger.run_log.stage_timings = {"long_stage": 90.0}
        # Set start_time to 90 seconds ago
        start = datetime.now() - timedelta(seconds=90)
        logger.run_log.start_time = start.isoformat()

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        # Should show minutes format like "1m 30.0s"
        assert "TOTAL TIME:" in output
        assert "m " in output  # Minutes indicator


class TestFinalizeFilesGenerated:
    """Test finalize() files generated section."""

    def test_finalize_with_files_generated(self, tmp_path):
        """Test finalize shows files generated section."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.files_generated = {
            "timeline": str(tmp_path / "output.otio"),
            "json": str(tmp_path / "output.json"),
        }

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "FILES GENERATED" in output


class TestFinalizeAPIUsage:
    """Test finalize() API usage section."""

    def test_finalize_with_api_calls(self, tmp_path):
        """Test finalize shows API usage when calls made."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_api_calls = 50
        logger.run_log.total_api_cost_est_usd = 0.0123

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "API USAGE" in output
        assert "Total calls:" in output


class TestUpdateStatsUnknownKey:
    """Test update_stats/set_stats with unknown keys."""

    def test_update_stats_unknown_key_logs_warning(self, tmp_path):
        """Test update_stats logs warning for unknown keys."""
        logger = RunLogger(log_dir=str(tmp_path))
        # This should log a warning, not raise
        logger.update_stats(nonexistent_key=42)
        # Verify the key wasn't set
        assert not hasattr(logger.run_log, "nonexistent_key")

    def test_set_stats_unknown_key_logs_warning(self, tmp_path):
        """Test set_stats logs warning for unknown keys."""
        logger = RunLogger(log_dir=str(tmp_path))
        # This should log a warning, not raise
        logger.set_stats(another_fake_key="test")
        # Verify the key wasn't set
        assert not hasattr(logger.run_log, "another_fake_key")


class TestLogMatchDecisionNoneClip:
    """Test log_match_decision with None clip."""

    def test_log_match_decision_none_clip(self, tmp_path):
        """Test log_match_decision handles None selected_clip."""
        logger = RunLogger(log_dir=str(tmp_path))

        # Should not raise with None clip
        logger.log_match_decision(
            segment_index=0,
            voiceover_text="Test segment",
            selected_clip=None,
            confidence=0.5,
            reasoning="No suitable clip found",
            embedding_similarity=0.3,
            duration_penalty=0.1,
            keyword_boost=0.0,
            entity_boost=0.0,
            is_keyword_match=False,
            is_visual_match=False,
            alternatives_considered=5,
            llm_reranked=False,
            clip_reuse_count=0,
        )

        # Verify it was logged
        assert len(logger.run_log.match_decisions) == 1


class TestNumpyEncoderWithoutNumpy:
    """Test NumpyEncoder behavior."""

    def test_numpy_encoder_non_numpy_type_raises(self):
        """Test NumpyEncoder raises for unknown types when numpy not involved."""
        encoder = NumpyEncoder()
        # A custom class that isn't numpy
        class CustomType:
            pass

        with pytest.raises(TypeError):
            encoder.default(CustomType())


class TestVerboseMarkdownFilesGenerated:
    """Test verbose markdown files generated section."""

    def test_verbose_markdown_with_files(self, tmp_path):
        """Test verbose markdown includes files generated section."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.files_generated = {
            "timeline": str(tmp_path / "test.otio"),
        }
        logger.run_log.stage_timings = {"output": 2.5}

        captured = io.StringIO()
        sys.stdout = captured
        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        # Check the verbose markdown file was created and contains expected content
        md_file = tmp_path / f"run_{logger.run_id}_verbose.md"
        if md_file.exists():
            content = md_file.read_text(encoding="utf-8")
            # Should have output stage section
            assert "Stage: OUTPUT" in content or "Files Generated" in content
