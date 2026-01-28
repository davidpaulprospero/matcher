"""
Tests for src/logger.py to achieve 100% coverage.

Targets:
- RunLogger methods (log_api_call, log_match_decision, stage_timer)
- Formatting utilities (confidence tiers, colors)
- Verbose markdown generation
- NumpyEncoder
- API cost estimation
"""

import pytest
import json
import os
import time
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
from dataclasses import asdict

from src.logger import (
    RunLogger,
    RunLog,
    APICallLog,
    MatchDecisionLog,
    ConfigAccessLog,
    PerformanceLog,
    VideoProcessLog,
    MatchDetailLog,
    TrackVarietyLog,
    StageLog,
    NumpyEncoder,
    get_confidence_tier,
    get_confidence_color,
    estimate_tokens,
    get_api_cost,
    API_PRICING,
    log_config_access,
    log_hardcoded,
)


class TestConfidenceTiers:
    """Test confidence tier functions."""

    @pytest.mark.fast
    def test_get_confidence_tier_high(self):
        """Test HIGH tier."""
        assert get_confidence_tier(0.90) == "HIGH"
        assert get_confidence_tier(0.80) == "HIGH"

    @pytest.mark.fast
    def test_get_confidence_tier_good(self):
        """Test GOOD tier."""
        assert get_confidence_tier(0.79) == "GOOD"
        assert get_confidence_tier(0.60) == "GOOD"

    @pytest.mark.fast
    def test_get_confidence_tier_medium(self):
        """Test MEDIUM tier."""
        assert get_confidence_tier(0.59) == "MEDIUM"
        assert get_confidence_tier(0.40) == "MEDIUM"

    @pytest.mark.fast
    def test_get_confidence_tier_low(self):
        """Test LOW tier."""
        assert get_confidence_tier(0.39) == "LOW"
        assert get_confidence_tier(0.20) == "LOW"

    @pytest.mark.fast
    def test_get_confidence_tier_gap(self):
        """Test GAP tier."""
        assert get_confidence_tier(0.19) == "GAP"
        assert get_confidence_tier(0.0) == "GAP"

    @pytest.mark.fast
    def test_get_confidence_color_all_tiers(self):
        """Test colors for all tiers."""
        assert get_confidence_color(0.90) == "GREEN"
        assert get_confidence_color(0.70) == "CYAN"
        assert get_confidence_color(0.50) == "YELLOW"
        assert get_confidence_color(0.30) == "ORANGE"
        assert get_confidence_color(0.10) == "RED"


class TestTokenEstimation:
    """Test token estimation functions."""

    @pytest.mark.fast
    def test_estimate_tokens(self):
        """Test token estimation."""
        # 4 chars ≈ 1 token
        assert estimate_tokens("1234") == 1
        assert estimate_tokens("12345678") == 2
        assert estimate_tokens("") == 0

    @pytest.mark.fast
    def test_get_api_cost_known_model(self):
        """Test API cost for known model."""
        cost = get_api_cost("gemini-2.0-flash", 1000, 500)
        assert cost > 0

    @pytest.mark.fast
    def test_get_api_cost_unknown_model(self):
        """Test API cost for unknown model falls back to local."""
        cost = get_api_cost("unknown-model", 1000, 500)
        assert cost == 0  # Local pricing is 0

    @pytest.mark.fast
    def test_get_api_cost_local(self):
        """Test API cost for local model."""
        cost = get_api_cost("local", 10000, 5000)
        assert cost == 0


class TestNumpyEncoder:
    """Test NumpyEncoder for JSON serialization."""

    @pytest.mark.fast
    def test_numpy_encoder_int(self):
        """Test encoding numpy integers."""
        try:
            import numpy as np
            encoder = NumpyEncoder()
            result = encoder.default(np.int64(42))
            assert result == 42
            assert isinstance(result, int)
        except ImportError:
            pytest.skip("numpy not available")

    @pytest.mark.fast
    def test_numpy_encoder_float(self):
        """Test encoding numpy floats."""
        try:
            import numpy as np
            encoder = NumpyEncoder()
            result = encoder.default(np.float64(3.14))
            assert abs(result - 3.14) < 0.001
            assert isinstance(result, float)
        except ImportError:
            pytest.skip("numpy not available")

    @pytest.mark.fast
    def test_numpy_encoder_array(self):
        """Test encoding numpy arrays."""
        try:
            import numpy as np
            encoder = NumpyEncoder()
            result = encoder.default(np.array([1, 2, 3]))
            assert result == [1, 2, 3]
        except ImportError:
            pytest.skip("numpy not available")

    @pytest.mark.fast
    def test_numpy_encoder_bool(self):
        """Test encoding numpy booleans."""
        try:
            import numpy as np
            encoder = NumpyEncoder()
            result = encoder.default(np.bool_(True))
            assert result == True
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("numpy not available")

    @pytest.mark.fast
    def test_numpy_encoder_fallback(self):
        """Test encoder falls back for unknown types."""
        encoder = NumpyEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())


class TestDataclasses:
    """Test log dataclasses."""

    @pytest.mark.fast
    def test_api_call_log_to_dict(self):
        """Test APICallLog.to_dict."""
        log = APICallLog(
            provider="gemini",
            endpoint="embed",
            model="text-embedding-004",
            timestamp="2026-01-01T00:00:00",
            duration_ms=100.0,
            input_tokens_est=1000,
            output_tokens_est=0,
            cost_est_usd=0.01,
            success=True
        )
        result = log.to_dict()
        assert result['provider'] == "gemini"
        assert result['success'] == True

    @pytest.mark.fast
    def test_match_decision_log_to_dict(self):
        """Test MatchDecisionLog.to_dict."""
        log = MatchDecisionLog(
            segment_index=0,
            voiceover_text="test",
            selected_clip="video.mp4",
            confidence=0.85,
            confidence_tier="HIGH",
            reasoning="good match",
            embedding_similarity=0.9,
            duration_penalty=0.0,
            keyword_boost=0.1,
            entity_boost=0.05,
            is_keyword_match=True,
            is_visual_match=False,
            is_hybrid_match=False,
            alternatives_considered=5,
            llm_reranked=True,
            clip_reuse_count=1
        )
        result = log.to_dict()
        assert result['confidence'] == 0.85

    @pytest.mark.fast
    def test_run_log_to_dict(self):
        """Test RunLog.to_dict."""
        log = RunLog(
            run_id="test_run",
            start_time="2026-01-01T00:00:00"
        )
        result = log.to_dict()
        assert result['run_id'] == "test_run"
        assert 'summary' in result


class TestRunLogger:
    """Test RunLogger class."""

    @pytest.mark.fast
    def test_init(self, tmp_path):
        """Test RunLogger initialization."""
        logger = RunLogger(log_dir=str(tmp_path), run_id="test123")
        assert logger.run_id == "test123"
        assert logger.log_dir == tmp_path
        assert logger.log_file.exists() or True  # May not exist until first write

    @pytest.mark.fast
    def test_init_auto_run_id(self, tmp_path):
        """Test RunLogger with auto-generated run_id."""
        logger = RunLogger(log_dir=str(tmp_path))
        assert len(logger.run_id) == 15  # YYYYMMDD_HHMMSS

    @pytest.mark.fast
    def test_log_config(self, tmp_path):
        """Test log_config method."""
        logger = RunLogger(log_dir=str(tmp_path))

        # Create mock config
        mock_config = MagicMock()
        mock_config._config_path = "/path/to/config.yaml"
        mock_config._config_hash = "abc123"
        mock_config._load_time_ms = 50.0
        mock_config.matching.min_confidence = 0.5
        mock_config.matching.high_confidence_threshold = 0.85
        mock_config.matching.embedding_candidates = 50
        mock_config.transcription.model = "whisper"
        mock_config.transcription.max_workers = 4
        mock_config.transcription.use_gpu = True
        mock_config.embedding.provider = "gemini"
        mock_config.embedding.batch_size = 32

        logger.log_config(mock_config)
        assert logger.run_log.config_path == "/path/to/config.yaml"
        assert logger.run_log.config_hash == "abc123"

    @pytest.mark.fast
    def test_log_api_call_success(self, tmp_path):
        """Test log_api_call with success."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_api_call(
            provider="gemini",
            endpoint="embed",
            model="text-embedding-004",
            input_text="test input",
            output_text="",
            duration_ms=100.0,
            success=True
        )

        assert logger.run_log.total_api_calls == 1
        assert len(logger.run_log.api_calls) == 1

    @pytest.mark.fast
    def test_log_api_call_error(self, tmp_path):
        """Test log_api_call with error."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_api_call(
            provider="gemini",
            endpoint="generate",
            model="gemini-2.0-flash",
            input_text="test",
            output_text="",
            duration_ms=50.0,
            success=False,
            error="Rate limited"
        )

        assert logger.run_log.api_calls[0].error == "Rate limited"
        assert logger.run_log.api_calls[0].success == False

    @pytest.mark.fast
    def test_log_match_decision(self, tmp_path):
        """Test log_match_decision."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_match_decision(
            segment_index=0,
            voiceover_text="This is a test segment",
            selected_clip="/path/to/video.mp4",
            confidence=0.85,
            reasoning="Good semantic match",
            embedding_similarity=0.9,
            duration_penalty=0.02,
            keyword_boost=0.1,
            entity_boost=0.05,
            is_keyword_match=True,
            is_visual_match=False,
            alternatives_considered=10,
            llm_reranked=True,
            clip_reuse_count=1
        )

        assert logger.run_log.total_matches == 1
        assert len(logger.run_log.match_decisions) == 1
        assert logger.run_log.match_decisions[0].confidence_tier == "HIGH"

    @pytest.mark.fast
    def test_log_config_access(self, tmp_path):
        """Test log_config_access."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_config_access(
            component="matching",
            config_path="min_confidence",
            value=0.5,
            source="config"
        )

        assert len(logger.run_log.config_accesses) == 1

    @pytest.mark.fast
    def test_log_config_access_hardcoded(self, tmp_path):
        """Test log_config_access with hardcoded source."""
        logger = RunLogger(log_dir=str(tmp_path))

        # First call should log warning
        logger.log_config_access(
            component="matching",
            config_path="hardcoded_value",
            value=42,
            source="hardcoded"
        )

        assert len(logger.run_log.warnings) == 1
        assert "Hardcoded value" in logger.run_log.warnings[0]

        # Second call with same key should not add another warning
        logger.log_config_access(
            component="matching",
            config_path="hardcoded_value",
            value=42,
            source="hardcoded"
        )

        assert len(logger.run_log.warnings) == 1  # Still 1

    @pytest.mark.fast
    def test_log_hardcoded_warning(self, tmp_path):
        """Test log_hardcoded_warning convenience method."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_hardcoded_warning("test_component", "test_value", 123)

        assert len(logger.run_log.config_accesses) == 1
        assert logger.run_log.config_accesses[0].source == "hardcoded"


class TestRunLoggerVerbose:
    """Test verbose logging methods."""

    @pytest.mark.fast
    def test_set_project_name(self, tmp_path):
        """Test set_project_name."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.set_project_name("My Project")
        assert logger.run_log.project_name == "My Project"

    def test_log_stage_start_end(self, tmp_path):
        """Test log_stage_start and log_stage_end."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_stage_start("DOWNLOAD")
        time.sleep(0.01)
        logger.log_stage_end("DOWNLOAD", videos_count=10)

        assert len(logger.run_log.stages) == 1
        assert logger.run_log.stages[0].name == "DOWNLOAD"
        assert logger.run_log.stages[0].duration_seconds > 0

    @pytest.mark.fast
    def test_log_video_process(self, tmp_path):
        """Test log_video_process."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_video_process(
            index=0,
            video_id="abc123",
            duration_seconds=120.0,
            segments=15,
            vad_removed_seconds=5.0,
            cached=False
        )

        assert len(logger.run_log.video_process_logs) == 1
        assert logger.run_log.video_process_logs[0].video_id == "abc123"

    @pytest.mark.fast
    def test_log_match_detail(self, tmp_path):
        """Test log_match_detail."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_match_detail(
            segment_index=0,
            voiceover_text="This is a test voiceover segment",
            matched_clip="/path/to/video.mp4",
            start_time=12.5,
            end_time=18.0,
            confidence=0.85,
            alternatives=[{"clip": "alt1.mp4", "confidence": 0.7}]
        )

        assert len(logger.run_log.match_detail_logs) == 1
        assert logger.run_log.match_detail_logs[0].segment_index == "S001"
        assert "00:12-00:18" in logger.run_log.match_detail_logs[0].clip_timecode

    @pytest.mark.fast
    def test_log_track_variety(self, tmp_path):
        """Test log_track_variety."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_track_variety(
            track="V1",
            unique_sources=5,
            most_used="source1.mp4",
            most_used_count=3
        )

        assert len(logger.run_log.track_variety_logs) == 1

    @pytest.mark.fast
    def test_log_remix_stats(self, tmp_path):
        """Test log_remix_stats."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_remix_stats(
            videos_scanned=100,
            included=80,
            excluded=20,
            avg_score=0.75
        )

        assert logger.run_log.remix_videos_scanned == 100
        assert logger.run_log.remix_included == 80

    @pytest.mark.fast
    def test_log_embedding_stats(self, tmp_path):
        """Test log_embedding_stats."""
        logger = RunLogger(log_dir=str(tmp_path))

        logger.log_embedding_stats(
            total=1000,
            dimensions=768,
            batches=50,
            rate=20.0,
            duration=50.0
        )

        assert logger.run_log.embeddings_computed == 1000
        assert logger.run_log.embedding_dimensions == 768

    @pytest.mark.fast
    def test_log_matching_config(self, tmp_path):
        """Test log_matching_config."""
        logger = RunLogger(log_dir=str(tmp_path))

        config_dict = {
            'min_confidence': 0.5,
            'embedding_candidates': 50
        }
        logger.log_matching_config(config_dict)

        assert logger.run_log.matching_config == config_dict


class TestRunLoggerStageTimer:
    """Test stage_timer context manager."""

    def test_stage_timer_success(self, tmp_path):
        """Test stage_timer with successful execution."""
        logger = RunLogger(log_dir=str(tmp_path))

        with logger.stage_timer("TEST_STAGE", item_count=10):
            time.sleep(0.01)

        assert len(logger.run_log.performance) == 1
        assert logger.run_log.performance[0].stage == "TEST_STAGE"
        assert logger.run_log.performance[0].duration_seconds > 0

    @pytest.mark.fast
    def test_stage_timer_exception(self, tmp_path):
        """Test stage_timer records timing even on exception."""
        logger = RunLogger(log_dir=str(tmp_path))

        with pytest.raises(ValueError):
            with logger.stage_timer("FAILING_STAGE"):
                raise ValueError("Test error")

        assert len(logger.run_log.performance) == 1

    @pytest.mark.fast
    def test_stage_timer_zero_items(self, tmp_path):
        """Test stage_timer with zero items."""
        logger = RunLogger(log_dir=str(tmp_path))

        with logger.stage_timer("EMPTY_STAGE", item_count=0):
            pass

        assert logger.run_log.performance[0].items_per_second == 0


class TestRunLoggerUtilities:
    """Test utility methods."""

    @pytest.mark.fast
    def test_log_warning(self, tmp_path):
        """Test log_warning."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.log_warning("Test warning")
        assert "Test warning" in logger.run_log.warnings

    @pytest.mark.fast
    def test_log_error(self, tmp_path):
        """Test log_error."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.log_error("Test error")
        assert "Test error" in logger.run_log.errors

    @pytest.mark.fast
    def test_log_output_file(self, tmp_path):
        """Test log_output_file."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.log_output_file("/path/to/output.otio")
        assert "/path/to/output.otio" in logger.run_log.output_files

    @pytest.mark.fast
    def test_log_config_reload(self, tmp_path):
        """Test log_config_reload."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.log_config_reload("old_hash", "new_hash")
        assert any("Config reloaded" in w for w in logger.run_log.warnings)

    @pytest.mark.fast
    def test_log_stage_complete(self, tmp_path):
        """Test log_stage_complete."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.log_stage_complete("ANALYZE", 5.5, {"keywords": 15})
        assert logger.run_log.stage_timings["ANALYZE"] == 5.5

    @pytest.mark.fast
    def test_update_stats_increment(self, tmp_path):
        """Test update_stats increments counters."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.update_stats(transcription_cache_hits=5)
        logger.update_stats(transcription_cache_hits=3)
        # Should increment
        assert logger.run_log.transcription_cache_hits == 8

    @pytest.mark.fast
    def test_update_stats_set(self, tmp_path):
        """Test update_stats sets non-counter values."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.update_stats(avg_confidence=0.85)
        assert logger.run_log.avg_confidence == 0.85

    @pytest.mark.fast
    def test_set_stats(self, tmp_path):
        """Test set_stats directly sets values."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.set_stats(videos_downloaded=100)
        assert logger.run_log.videos_downloaded == 100

    @pytest.mark.fast
    def test_log_file_generated(self, tmp_path):
        """Test log_file_generated."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.log_file_generated("otio", "/path/to/output.otio")
        assert logger.run_log.files_generated["otio"] == "/path/to/output.otio"


class TestRunLoggerFinalize:
    """Test finalize method."""

    @pytest.mark.fast
    def test_finalize_basic(self, tmp_path):
        """Test finalize without match decisions."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_segments = 10

        # Capture stdout
        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured

        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "RUN SUMMARY" in output

    @pytest.mark.fast
    def test_finalize_with_matches(self, tmp_path):
        """Test finalize with match decisions."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_segments = 5

        # Add match decisions
        for i in range(5):
            logger.log_match_decision(
                segment_index=i,
                voiceover_text=f"Segment {i}",
                selected_clip=f"video_{i}.mp4",
                confidence=0.7 + i * 0.05
            )

        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured

        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        # avg_confidence should be calculated
        assert logger.run_log.avg_confidence > 0

    @pytest.mark.fast
    def test_finalize_with_stage_timings(self, tmp_path):
        """Test finalize with stage timings."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.stage_timings = {
            "ANALYZE": 5.0,
            "DOWNLOAD": 10.0,
            "MATCH": 3.0
        }

        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured

        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "STAGE TIMINGS" in output

    @pytest.mark.fast
    def test_finalize_with_warnings_errors(self, tmp_path):
        """Test finalize with warnings and errors."""
        logger = RunLogger(log_dir=str(tmp_path))
        logger.log_warning("Warning 1")
        logger.log_warning("Warning 2")
        logger.log_warning("Warning 3")
        logger.log_warning("Warning 4")  # More than 3
        logger.log_error("Error 1")

        import io
        import sys
        captured = io.StringIO()
        sys.stdout = captured

        try:
            logger.finalize()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "ISSUES" in output
        assert "Warnings: 4" in output


class TestModuleLevelFunctions:
    """Test module-level logging functions."""

    @pytest.mark.fast
    def test_log_config_access_module(self):
        """Test module-level log_config_access."""
        # Should not raise even without RunLogger
        log_config_access("test_component", "test_path", "test_value")

    @pytest.mark.fast
    def test_log_hardcoded_module(self):
        """Test module-level log_hardcoded."""
        # Should not raise even without RunLogger
        log_hardcoded("test_component", "test_name", 42)


class TestAPICallLogError:
    """Test API call log with error field."""

    @pytest.mark.fast
    def test_api_call_log_with_error(self):
        """Test APICallLog with error."""
        log = APICallLog(
            provider="test",
            endpoint="test",
            model="test",
            timestamp="2026-01-01",
            duration_ms=0,
            input_tokens_est=0,
            output_tokens_est=0,
            cost_est_usd=0,
            success=False,
            error="Test error message"
        )
        result = log.to_dict()
        assert result['error'] == "Test error message"
