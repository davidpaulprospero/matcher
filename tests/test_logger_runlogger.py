"""
Comprehensive test suite for logger.py - RunLogger class.

Tests coverage for:
- src/logger.py (RunLogger initialization, logging methods, file generation)

Created: January 10, 2026
Session: 13 Phase 2
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import (
    RunLogger,
    APICallLog,
    MatchDecisionLog,
    ConfigAccessLog,
    PerformanceLog,
    RunLog,
    VideoProcessLog,
    MatchDetailLog,
    TrackVarietyLog,
    StageLog,
    set_global_logger,
    get_global_logger,
    log_config_access,
    log_hardcoded
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_log_dir():
    """Temporary log directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def run_logger(temp_log_dir):
    """Create a RunLogger instance"""
    logger = RunLogger(log_dir=temp_log_dir, run_id="test_run_123")
    yield logger
    # Cleanup - properly close handlers
    for handler in logger.file_logger.handlers[:]:
        handler.close()
        logger.file_logger.removeHandler(handler)


@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config._config_path = "config.yaml"
    config._config_hash = "abc123def456"
    config._load_time_ms = 42.5

    # Matching config
    config.matching = Mock()
    config.matching.min_confidence = 0.4
    config.matching.high_confidence_threshold = 0.8
    config.matching.embedding_candidates = 50

    # Transcription config
    config.transcription = Mock()
    config.transcription.model = "base"
    config.transcription.max_workers = 4
    config.transcription.use_gpu = True

    # Embedding config
    config.embedding = Mock()
    config.embedding.provider = "voyage"
    config.embedding.batch_size = 128

    return config


# ============================================================================
# Test RunLogger Initialization
# ============================================================================

class TestRunLoggerInit:
    """Test RunLogger initialization"""

    @pytest.mark.fast
    def test_init_creates_log_dir(self, temp_log_dir):
        """Test initialization creates log directory"""
        log_dir = Path(temp_log_dir) / "new_logs"
        assert not log_dir.exists()

        logger = RunLogger(log_dir=str(log_dir))
        assert log_dir.exists()

        # Cleanup
        for handler in logger.file_logger.handlers[:]:
            handler.close()
            logger.file_logger.removeHandler(handler)

    @pytest.mark.fast
    def test_init_with_custom_run_id(self, temp_log_dir):
        """Test initialization with custom run ID"""
        logger = RunLogger(log_dir=temp_log_dir, run_id="custom_id_123")

        assert logger.run_id == "custom_id_123"
        assert logger.log_file.name == "run_custom_id_123.log"
        assert logger.json_file.name == "run_custom_id_123.json"

        # Cleanup
        for handler in logger.file_logger.handlers[:]:
            handler.close()
            logger.file_logger.removeHandler(handler)

    @pytest.mark.fast
    def test_init_generates_run_id_if_none(self, temp_log_dir):
        """Test initialization generates run ID if not provided"""
        logger = RunLogger(log_dir=temp_log_dir)

        assert logger.run_id is not None
        assert len(logger.run_id) > 0
        # Should be timestamp format YYYYMMDD_HHMMSS
        assert len(logger.run_id) == 15

        # Cleanup
        for handler in logger.file_logger.handlers[:]:
            handler.close()
            logger.file_logger.removeHandler(handler)

    @pytest.mark.fast
    def test_init_creates_file_logger(self, run_logger):
        """Test initialization creates file logger"""
        assert run_logger.file_logger is not None
        assert len(run_logger.file_logger.handlers) > 0

    @pytest.mark.fast
    def test_init_creates_run_log(self, run_logger):
        """Test initialization creates RunLog instance"""
        assert run_logger.run_log is not None
        assert run_logger.run_log.run_id == "test_run_123"
        assert run_logger.run_log.start_time is not None


# ============================================================================
# Test Config Logging
# ============================================================================

class TestRunLoggerConfigLogging:
    """Test configuration logging"""

    @pytest.mark.fast
    def test_log_config_basic(self, run_logger, mock_config):
        """Test basic config logging"""
        run_logger.log_config(mock_config)

        assert run_logger.run_log.config_path == "config.yaml"
        assert run_logger.run_log.config_hash == "abc123def456"

    @pytest.mark.fast
    def test_log_config_without_attributes(self, run_logger):
        """Test config logging with minimal config object"""
        config = Mock(spec=[])  # Empty config
        run_logger.log_config(config)

        # Should not crash, just not set values
        assert run_logger.run_log.config_path == ""

    @pytest.mark.fast
    def test_log_config_access(self, run_logger):
        """Test logging config access"""
        run_logger.log_config_access(
            component="matcher",
            config_path="matching.min_confidence",
            value=0.4,
            source="config"
        )

        assert len(run_logger.run_log.config_accesses) == 1
        access = run_logger.run_log.config_accesses[0]
        assert access.component == "matcher"
        assert access.config_path == "matching.min_confidence"
        assert access.value == "0.4"
        assert access.source == "config"

    @pytest.mark.fast
    def test_log_hardcoded_warning(self, run_logger):
        """Test logging hardcoded value warnings"""
        run_logger.log_hardcoded_warning("downloader", "timeout_seconds", 300)

        assert len(run_logger.run_log.warnings) == 1
        assert "Hardcoded value" in run_logger.run_log.warnings[0]
        assert "timeout_seconds" in run_logger.run_log.warnings[0]

    @pytest.mark.fast
    def test_log_hardcoded_warning_deduplication(self, run_logger):
        """Test hardcoded warnings are deduplicated"""
        # Log same warning twice
        run_logger.log_hardcoded_warning("downloader", "timeout_seconds", 300)
        run_logger.log_hardcoded_warning("downloader", "timeout_seconds", 300)

        # Should only be logged once
        assert len(run_logger.run_log.warnings) == 1


# ============================================================================
# Test API Call Logging
# ============================================================================

class TestRunLoggerAPICallLogging:
    """Test API call logging"""

    @pytest.mark.fast
    def test_log_api_call_basic(self, run_logger):
        """Test basic API call logging"""
        run_logger.log_api_call(
            provider="gemini",
            endpoint="generate",
            model="gemini-2.0-flash",
            input_text="Test input " * 100,  # ~1000 chars
            output_text="Test output " * 50,  # ~500 chars
            duration_ms=150.5,
            success=True
        )

        assert len(run_logger.run_log.api_calls) == 1
        assert run_logger.run_log.total_api_calls == 1
        assert run_logger.run_log.total_api_cost_est_usd > 0

        call = run_logger.run_log.api_calls[0]
        assert call.provider == "gemini"
        assert call.endpoint == "generate"
        assert call.model == "gemini-2.0-flash"
        assert call.success is True
        assert call.duration_ms == 150.5

    @pytest.mark.fast
    def test_log_api_call_with_error(self, run_logger):
        """Test API call logging with error"""
        run_logger.log_api_call(
            provider="anthropic",
            endpoint="messages",
            model="claude-3-sonnet",
            input_text="test",
            output_text="",
            duration_ms=0,
            success=False,
            error="Rate limit exceeded"
        )

        call = run_logger.run_log.api_calls[0]
        assert call.success is False
        assert call.error == "Rate limit exceeded"

    @pytest.mark.fast
    def test_log_api_call_cost_accumulation(self, run_logger):
        """Test API call cost accumulation"""
        # Log multiple calls
        for i in range(5):
            run_logger.log_api_call(
                provider="gemini",
                endpoint="embed",
                model="text-embedding-004",
                input_text="x" * 1000,
                output_text="",
                duration_ms=50,
                success=True
            )

        assert run_logger.run_log.total_api_calls == 5
        assert run_logger.run_log.total_api_cost_est_usd > 0


# ============================================================================
# Test Match Decision Logging
# ============================================================================

class TestRunLoggerMatchDecisionLogging:
    """Test match decision logging"""

    @pytest.mark.fast
    def test_log_match_decision_basic(self, run_logger):
        """Test basic match decision logging"""
        run_logger.log_match_decision(
            segment_index=0,
            voiceover_text="This is the voiceover text",
            selected_clip="video1.mp4",
            confidence=0.85,
            reasoning="High semantic match",
            embedding_similarity=0.82,
            alternatives_considered=10
        )

        assert len(run_logger.run_log.match_decisions) == 1
        assert run_logger.run_log.total_matches == 1

        decision = run_logger.run_log.match_decisions[0]
        assert decision.segment_index == 0
        assert decision.selected_clip == "video1.mp4"
        assert decision.confidence == 0.85
        assert decision.confidence_tier == "HIGH"
        assert decision.embedding_similarity == 0.82

    @pytest.mark.fast
    def test_log_match_decision_with_boosts(self, run_logger):
        """Test match decision logging with various boosts"""
        run_logger.log_match_decision(
            segment_index=1,
            voiceover_text="Test",
            selected_clip="video2.mp4",
            confidence=0.65,
            keyword_boost=0.1,
            entity_boost=0.05,
            is_keyword_match=True,
            is_visual_match=True,
            llm_reranked=True,
            clip_reuse_count=2
        )

        decision = run_logger.run_log.match_decisions[0]
        assert decision.confidence_tier == "GOOD"
        assert decision.keyword_boost == 0.1
        assert decision.entity_boost == 0.05
        assert decision.is_keyword_match is True
        assert decision.is_visual_match is True
        assert decision.is_hybrid_match is True  # Both keyword and visual
        assert decision.llm_reranked is True
        assert decision.clip_reuse_count == 2

    @pytest.mark.fast
    def test_log_match_decision_long_text_truncation(self, run_logger):
        """Test voiceover text is truncated to 100 chars"""
        long_text = "A" * 200  # 200 chars
        run_logger.log_match_decision(
            segment_index=0,
            voiceover_text=long_text,
            selected_clip="video.mp4",
            confidence=0.5
        )

        decision = run_logger.run_log.match_decisions[0]
        assert len(decision.voiceover_text) == 100


# ============================================================================
# Test Stage Timing
# ============================================================================

class TestRunLoggerStageTiming:
    """Test stage timing and context manager"""

    def test_stage_timer_context_manager(self, run_logger):
        """Test stage_timer context manager"""
        import time

        with run_logger.stage_timer("test_stage", item_count=10):
            time.sleep(0.1)  # Simulate work

        assert len(run_logger.run_log.performance) == 1
        perf = run_logger.run_log.performance[0]
        assert perf.stage == "test_stage"
        assert perf.items_processed == 10
        assert perf.duration_seconds >= 0.1
        assert perf.items_per_second > 0

    @pytest.mark.fast
    def test_log_stage_complete(self, run_logger):
        """Test log_stage_complete"""
        run_logger.log_stage_complete(
            stage_name="download",
            duration_seconds=45.2,
            stats={'videos': 10, 'cache_hits': 5}
        )

        assert "download" in run_logger.run_log.stage_timings
        assert run_logger.run_log.stage_timings["download"] == 45.2

    def test_log_stage_start_end(self, run_logger):
        """Test verbose stage logging"""
        import time

        run_logger.log_stage_start("test_stage")
        time.sleep(0.05)
        run_logger.log_stage_end("test_stage", items=100, cache_hits=20)

        assert len(run_logger.run_log.stages) == 1
        stage = run_logger.run_log.stages[0]
        assert stage.name == "test_stage"
        assert stage.duration_seconds >= 0.05
        assert stage.details['items'] == 100
        assert stage.details['cache_hits'] == 20


# ============================================================================
# Test Stats Updates
# ============================================================================

class TestRunLoggerStatsUpdates:
    """Test statistics update methods"""

    @pytest.mark.fast
    def test_update_stats(self, run_logger):
        """Test update_stats increments counters"""
        run_logger.update_stats(videos_downloaded=5)
        assert run_logger.run_log.videos_downloaded == 5

        run_logger.update_stats(videos_downloaded=3)
        assert run_logger.run_log.videos_downloaded == 8  # Incremented

    @pytest.mark.fast
    def test_set_stats(self, run_logger):
        """Test set_stats sets values directly"""
        run_logger.set_stats(total_segments=10)
        assert run_logger.run_log.total_segments == 10

        run_logger.set_stats(total_segments=20)
        assert run_logger.run_log.total_segments == 20  # Replaced

    @pytest.mark.fast
    def test_update_stats_invalid_field(self, run_logger):
        """Test update_stats with invalid field name"""
        # Should not crash, just ignore
        run_logger.update_stats(invalid_field=100)


# ============================================================================
# Test Verbose Logging Methods
# ============================================================================

class TestRunLoggerVerboseLogging:
    """Test verbose logging methods"""

    @pytest.mark.fast
    def test_set_project_name(self, run_logger):
        """Test setting project name"""
        run_logger.set_project_name("MyProject")
        assert run_logger.run_log.project_name == "MyProject"

    @pytest.mark.fast
    def test_log_video_process(self, run_logger):
        """Test logging video processing"""
        run_logger.log_video_process(
            index=1,
            video_id="abc123",
            duration_seconds=120.5,
            segments=45,
            vad_removed_seconds=10.2,
            cached=True
        )

        assert len(run_logger.run_log.video_process_logs) == 1
        vp = run_logger.run_log.video_process_logs[0]
        assert vp.video_id == "abc123"
        assert vp.duration_seconds == 120.5
        assert vp.segments == 45
        assert vp.cached is True

    @pytest.mark.fast
    def test_log_match_detail(self, run_logger):
        """Test logging detailed match"""
        run_logger.log_match_detail(
            segment_index=5,
            voiceover_text="This is a very long voiceover text that should be truncated",
            matched_clip="/path/to/video.mp4",
            start_time=12.5,
            end_time=18.7,
            confidence=0.72,
            alternatives=[
                {'clip': 'alt1.mp4', 'confidence': 0.65},
                {'clip': 'alt2.mp4', 'confidence': 0.60}
            ]
        )

        assert len(run_logger.run_log.match_detail_logs) == 1
        md = run_logger.run_log.match_detail_logs[0]
        assert md.segment_index == "S006"  # 1-indexed, zero-padded
        assert md.clip_timecode == "00:12-00:18"
        # Text might have ellipsis added
        assert len(md.voiceover_text) <= 54  # 50 + "..."
        assert len(md.alternatives) == 2

    @pytest.mark.fast
    def test_log_track_variety(self, run_logger):
        """Test logging track variety"""
        run_logger.log_track_variety(
            track="V1",
            unique_sources=15,
            most_used="video1.mp4",
            most_used_count=3
        )

        assert len(run_logger.run_log.track_variety_logs) == 1
        tv = run_logger.run_log.track_variety_logs[0]
        assert tv.track == "V1"
        assert tv.unique_sources == 15

    @pytest.mark.fast
    def test_log_remix_stats(self, run_logger):
        """Test logging remix stats"""
        run_logger.log_remix_stats(
            videos_scanned=100,
            included=80,
            excluded=20,
            avg_score=0.65
        )

        assert run_logger.run_log.remix_videos_scanned == 100
        assert run_logger.run_log.remix_included == 80
        assert run_logger.run_log.remix_excluded == 20
        assert run_logger.run_log.remix_avg_score == 0.65

    @pytest.mark.fast
    def test_log_embedding_stats(self, run_logger):
        """Test logging embedding stats"""
        run_logger.log_embedding_stats(
            total=5000,
            dimensions=1024,
            batches=40,
            rate=125.0,
            duration=40.0
        )

        assert run_logger.run_log.embeddings_computed == 5000
        assert run_logger.run_log.embedding_dimensions == 1024
        assert run_logger.run_log.embedding_batches == 40
        assert run_logger.run_log.embedding_rate == 125.0

    @pytest.mark.fast
    def test_log_matching_config(self, run_logger):
        """Test logging matching config"""
        config_dict = {
            'embedding_candidates': 50,
            'llm_rerank_count': 10,
            'max_clip_reuse': 3
        }
        run_logger.log_matching_config(config_dict)

        assert run_logger.run_log.matching_config == config_dict


# ============================================================================
# Test Warnings and Errors
# ============================================================================

class TestRunLoggerWarningsErrors:
    """Test warning and error logging"""

    @pytest.mark.fast
    def test_log_warning(self, run_logger):
        """Test logging warnings"""
        run_logger.log_warning("This is a warning message")

        assert len(run_logger.run_log.warnings) == 1
        assert run_logger.run_log.warnings[0] == "This is a warning message"

    @pytest.mark.fast
    def test_log_error(self, run_logger):
        """Test logging errors"""
        run_logger.log_error("This is an error message")

        assert len(run_logger.run_log.errors) == 1
        assert run_logger.run_log.errors[0] == "This is an error message"

    @pytest.mark.fast
    def test_log_output_file(self, run_logger):
        """Test logging output files"""
        run_logger.log_output_file("/path/to/output.otio")

        assert len(run_logger.run_log.output_files) == 1
        assert run_logger.run_log.output_files[0] == "/path/to/output.otio"

    @pytest.mark.fast
    def test_log_file_generated(self, run_logger):
        """Test logging generated files"""
        run_logger.log_file_generated("otio", "/path/to/timeline.otio")

        assert "otio" in run_logger.run_log.files_generated
        assert run_logger.run_log.files_generated["otio"] == "/path/to/timeline.otio"
        assert "/path/to/timeline.otio" in run_logger.run_log.output_files

    @pytest.mark.fast
    def test_log_config_reload(self, run_logger):
        """Test logging config reload"""
        run_logger.log_config_reload("old_hash123", "new_hash456")

        assert len(run_logger.run_log.warnings) >= 1
        # Check warning contains both hashes
        reload_warning = [w for w in run_logger.run_log.warnings if "Config reloaded" in w][0]
        assert "old_hash123" in reload_warning
        assert "new_hash456" in reload_warning


# ============================================================================
# Test Global Logger Functions
# ============================================================================

class TestGlobalLoggerFunctions:
    """Test global logger functions"""

    @pytest.mark.fast
    def test_set_and_get_global_logger(self, run_logger):
        """Test setting and getting global logger"""
        set_global_logger(run_logger)
        assert get_global_logger() == run_logger

    @pytest.mark.fast
    def test_log_config_access_global(self, run_logger):
        """Test log_config_access with global logger"""
        set_global_logger(run_logger)

        log_config_access("component", "path.to.value", 42, "config")

        assert len(run_logger.run_log.config_accesses) == 1

    @pytest.mark.fast
    def test_log_config_access_no_global_logger(self):
        """Test log_config_access without global logger"""
        set_global_logger(None)

        # Should not crash
        log_config_access("component", "path", 42)

    @pytest.mark.fast
    def test_log_hardcoded_global(self, run_logger):
        """Test log_hardcoded with global logger"""
        set_global_logger(run_logger)

        log_hardcoded("downloader", "timeout", 300)

        assert len(run_logger.run_log.warnings) >= 1

    @patch('src.logger.logging.warning')
    @pytest.mark.fast
    def test_log_hardcoded_no_global_logger(self, mock_warning):
        """Test log_hardcoded without global logger falls back to logging"""
        set_global_logger(None)

        log_hardcoded("component", "value_name", 100)

        # Should use standard logging
        mock_warning.assert_called_once()
        assert "HARDCODED VALUE" in mock_warning.call_args[0][0]


# ============================================================================
# Test Finalization and File Generation
# ============================================================================

class TestRunLoggerFinalization:
    """Test finalization and file generation"""

    @pytest.mark.fast
    def test_finalize_creates_json_file(self, run_logger):
        """Test finalize creates JSON file"""
        run_logger.finalize()

        assert run_logger.json_file.exists()

        # Check JSON is valid
        with open(run_logger.json_file, 'r') as f:
            data = json.load(f)

        assert data['run_id'] == "test_run_123"
        assert data['start_time'] is not None
        assert data['end_time'] is not None

    @pytest.mark.fast
    def test_finalize_creates_summary_files(self, run_logger):
        """Test finalize creates markdown and text summaries"""
        run_logger.set_stats(total_segments=10, total_matches=8)
        run_logger.log_match_decision(
            segment_index=0,
            voiceover_text="test",
            selected_clip="video.mp4",
            confidence=0.75
        )

        run_logger.finalize()

        # Check summary files exist
        log_dir = Path(run_logger.json_file).parent
        json_stem = Path(run_logger.json_file).stem
        summary_md = log_dir / f"{json_stem}_summary.md"
        summary_txt = log_dir / f"{json_stem}_summary.txt"
        verbose_md = run_logger.verbose_md_file

        assert summary_md.exists()
        assert summary_txt.exists()
        assert verbose_md.exists()

    @pytest.mark.fast
    def test_finalize_calculates_avg_confidence(self, run_logger):
        """Test finalize calculates average confidence"""
        run_logger.log_match_decision(
            segment_index=0,
            voiceover_text="test1",
            selected_clip="v1.mp4",
            confidence=0.8
        )
        run_logger.log_match_decision(
            segment_index=1,
            voiceover_text="test2",
            selected_clip="v2.mp4",
            confidence=0.6
        )

        run_logger.finalize()

        # Average should be 0.7
        assert abs(run_logger.run_log.avg_confidence - 0.7) < 0.01

    @patch('builtins.print')
    @pytest.mark.fast
    def test_finalize_prints_summary(self, mock_print, run_logger):
        """Test finalize prints console summary"""
        run_logger.set_stats(total_segments=10, total_matches=8)
        run_logger.finalize()

        # Check print was called with summary
        assert mock_print.called
        # Look for key summary elements in print calls
        all_output = ' '.join(str(call[0][0]) for call in mock_print.call_args_list)
        assert "RUN SUMMARY" in all_output or "Segments matched" in all_output
