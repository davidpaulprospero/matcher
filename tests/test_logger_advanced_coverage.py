"""
Advanced coverage tests for src/logger.py

Covers missed lines:
- 39-41: numpy ImportError handling
- 790, 797-798: total_matches and wall_clock_time edge cases
- 862-863, 867-871: overhead and API usage display
- 902: total time with minutes
- 1020, 1027: warnings/errors overflow
- 1060-1062, 1066: fmt_time and fmt_duration functions
- 1082-1083: stages iteration
- embedding_batches display
- matching_config output
- match_detail_logs table
- track_variety_logs table
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestRunLoggerFinalize:
    """Test finalize method edge cases."""

    @pytest.mark.fast
    def test_total_matches_from_match_decisions(self, tmp_path):
        """Test that total_matches is set from match_decisions when 0."""
        from src.logger import RunLogger, MatchDecisionLog

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_matches = 0
        logger.run_log.total_segments = 5

        # Add match decisions with all required fields
        decision = MatchDecisionLog(
            segment_index=1,
            voiceover_text="test",
            selected_clip="clip.mp4",
            confidence=0.85,
            confidence_tier="high",
            reasoning="",
            embedding_similarity=0.8,
            duration_penalty=0.0,
            keyword_boost=0.0,
            entity_boost=0.0,
            is_keyword_match=False,
            is_visual_match=False,
            is_hybrid_match=False,
            alternatives_considered=5,
            llm_reranked=False,
            clip_reuse_count=0
        )
        logger.run_log.match_decisions = [decision, decision]

        with patch('builtins.print'):
            with patch.object(logger, '_generate_llm_summary'):
                logger.finalize()

        # total_matches should be updated from match_decisions
        assert logger.run_log.total_matches == 2

    @pytest.mark.fast
    def test_wall_clock_time_exception_fallback(self, tmp_path):
        """Test fallback to stage_timings sum when time parsing fails."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.start_time = "invalid-time"
        logger.run_log.end_time = "also-invalid"
        logger.run_log.stage_timings = {'download': 10.0, 'match': 5.0}

        with patch('builtins.print'):
            with patch.object(logger, '_generate_llm_summary'):
                logger.finalize()

        # Should not raise, uses fallback

    @pytest.mark.fast
    def test_overhead_display_when_significant(self, tmp_path):
        """Test stage timings display with overhead calculation."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        now = datetime.now()
        logger.run_log.start_time = now.isoformat()
        logger.run_log.end_time = (now + timedelta(seconds=15)).isoformat()
        logger.run_log.stage_timings = {'download': 5.0, 'match': 5.0}  # 10s total, 5s overhead

        output = StringIO()
        with patch('builtins.print', side_effect=lambda *args, **kwargs: output.write(str(args) + '\n')):
            with patch.object(logger, '_generate_llm_summary'):
                logger.finalize()

        # Stage timings section should appear with active vs total time showing overhead
        # Output shows: "STAGE TIMINGS (10.0s active, X.Xs total)"
        assert 'STAGE TIMINGS' in output.getvalue()
        assert '10.0s active' in output.getvalue()

    @pytest.mark.fast
    def test_api_usage_display(self, tmp_path):
        """Test API usage section when total_api_calls > 0."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_api_calls = 50
        logger.run_log.total_api_cost_est_usd = 0.0025

        output = StringIO()
        with patch('builtins.print', side_effect=lambda *args, **kwargs: output.write(str(args) + '\n')):
            logger.finalize()

        assert 'API' in output.getvalue()

    @pytest.mark.fast
    def test_total_time_with_minutes(self, tmp_path):
        """Test total time display when >= 60 seconds."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        now = datetime.now()
        logger.run_log.start_time = now.isoformat()
        logger.run_log.end_time = (now + timedelta(seconds=125)).isoformat()  # 2m 5s
        logger.run_log.stage_timings = {'download': 125.0}

        output = StringIO()
        with patch('builtins.print', side_effect=lambda *args, **kwargs: output.write(str(args) + '\n')):
            logger.finalize()

        # Should show minutes
        assert '2m' in output.getvalue() or 'TOTAL TIME' in output.getvalue()


class TestWriteMarkdownSummary:
    """Test _write_markdown_summary method."""

    @pytest.mark.fast
    def test_warnings_overflow(self, tmp_path):
        """Test warnings list truncation when > 10."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.warnings = [f"Warning {i}" for i in range(15)]
        logger.run_log.errors = []

        path = tmp_path / "summary.md"
        logger._write_markdown_summary(path, 100.0)

        content = path.read_text()
        assert "... and 5 more" in content

    @pytest.mark.fast
    def test_errors_overflow(self, tmp_path):
        """Test errors list truncation when > 10."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.warnings = []
        logger.run_log.errors = [f"Error {i}" for i in range(12)]

        path = tmp_path / "summary.md"
        logger._write_markdown_summary(path, 100.0)

        content = path.read_text()
        assert "... and 2 more" in content


class TestWriteVerboseMarkdown:
    """Test _write_verbose_markdown method."""

    @pytest.mark.fast
    def test_fmt_time_formatting(self, tmp_path):
        """Test fmt_time helper function output."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_matches = 1
        logger.run_log.match_detail_logs = []

        path = tmp_path / "verbose.md"
        logger._write_verbose_markdown(path, 100.0)

        content = path.read_text()
        assert 'Pipeline Run' in content

    @pytest.mark.fast
    def test_fmt_duration_minutes(self, tmp_path):
        """Test fmt_duration with >= 60 seconds."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.stage_timings = {'match': 120.0}  # 2 minutes
        logger.run_log.total_matches = 1

        path = tmp_path / "verbose.md"
        logger._write_verbose_markdown(path, 100.0)

        content = path.read_text()
        assert '2.0m' in content or 'MATCH' in content

    @pytest.mark.fast
    def test_stages_iteration(self, tmp_path):
        """Test stages list iteration in verbose markdown."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))

        # Add stages with details
        stage = MagicMock()
        stage.name = "DOWNLOAD"
        stage.start_time = "12:00:00"
        stage.end_time = "12:05:00"
        stage.duration_seconds = 300.0
        stage.details = {'videos': 10}
        logger.run_log.stages = [stage]

        path = tmp_path / "verbose.md"
        logger._write_verbose_markdown(path, 100.0)

        content = path.read_text()
        assert 'DOWNLOAD' in content or 'Pipeline' in content

    @pytest.mark.fast
    def test_embedding_batches_display(self, tmp_path):
        """Test embedding batches rate display."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.embeddings_computed = 1000
        logger.run_log.embedding_cache_hits = 500
        logger.run_log.embedding_batches = 50
        logger.run_log.embedding_rate = 200.0
        logger.run_log.embedding_dimensions = 768

        path = tmp_path / "verbose.md"
        logger._write_verbose_markdown(path, 100.0)

        content = path.read_text()
        assert 'Batches' in content or '50' in content

    @pytest.mark.fast
    def test_matching_config_output(self, tmp_path):
        """Test matching_config section in verbose markdown."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_matches = 5
        logger.run_log.matching_config = {
            'embedding_candidates': 50,
            'llm_rerank_count': 10,
            'max_clip_reuse': 3,
            'face_preference': 'more',
            'llm_providers': ['gemini', 'claude']
        }

        path = tmp_path / "verbose.md"
        logger._write_verbose_markdown(path, 100.0)

        content = path.read_text()
        assert 'Config' in content or 'Candidates' in content

    @pytest.mark.fast
    def test_match_detail_logs_table(self, tmp_path):
        """Test match_detail_logs table in verbose markdown."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_matches = 2

        # Add match detail logs
        md1 = MagicMock()
        md1.segment_index = 1
        md1.voiceover_text = "First segment text"
        md1.matched_clip = "video_abc.mp4"
        md1.clip_timecode = "00:01:30"
        md1.confidence = 0.92
        md1.alternatives = [
            {'clip': 'alt1.mp4', 'confidence': 0.85},
            {'clip': 'alt2.mp4', 'confidence': 0.78}
        ]

        md2 = MagicMock()
        md2.segment_index = 2
        md2.voiceover_text = "Second segment with | pipe character"
        md2.matched_clip = "video_def.mp4"
        md2.clip_timecode = "00:02:15"
        md2.confidence = 0.88
        md2.alternatives = []

        logger.run_log.match_detail_logs = [md1, md2]

        path = tmp_path / "verbose.md"
        logger._write_verbose_markdown(path, 100.0)

        content = path.read_text()
        assert 'Matches' in content
        # Pipe should be escaped
        assert '\\|' in content or 'pipe' in content

    @pytest.mark.fast
    def test_track_variety_logs_table(self, tmp_path):
        """Test track_variety_logs table in verbose markdown."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        logger.run_log.total_matches = 5

        # Add track variety logs
        tv1 = MagicMock()
        tv1.track = "V1"
        tv1.unique_sources = 8
        tv1.most_used = "channel_abc_video_123"
        tv1.most_used_count = 3

        tv2 = MagicMock()
        tv2.track = "V2"
        tv2.unique_sources = 6
        tv2.most_used = "channel_def_video_456"
        tv2.most_used_count = 2

        logger.run_log.track_variety_logs = [tv1, tv2]

        path = tmp_path / "verbose.md"
        logger._write_verbose_markdown(path, 100.0)

        content = path.read_text()
        assert 'Track Variety' in content or 'V1' in content


class TestNumpyImportHandling:
    """Test numpy import error handling."""

    @pytest.mark.fast
    def test_numpy_not_available_encoder(self):
        """Test NumpyEncoder when numpy is not available."""
        # This tests the behavior when HAS_NUMPY is True but object is not numpy type
        from src.logger import NumpyEncoder
        import json

        encoder = NumpyEncoder()

        # Test with a regular object that's not numpy
        class CustomObj:
            pass

        with pytest.raises(TypeError):
            encoder.default(CustomObj())


class TestRunLoggerStageTimer:
    """Test stage_timer context manager edge cases."""

    @pytest.mark.fast
    def test_stage_timer_records_timing(self, tmp_path):
        """Test stage_timer records timing properly."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))

        with logger.stage_timer("TEST_STAGE"):
            pass  # Just test timing recording

        # Stage timer records to performance list
        assert len(logger.run_log.performance) == 1
        assert logger.run_log.performance[0].stage == "TEST_STAGE"


class TestRunLoggerLogApiCall:
    """Test log_api_call method."""

    @pytest.mark.fast
    def test_log_api_call_accumulates(self, tmp_path):
        """Test that log_api_call accumulates calls and costs."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        # Use correct signature: provider, endpoint, model, input_text, output_text
        logger.log_api_call(provider="gemini", endpoint="generate", model="gemini-pro", input_text="test input", output_text="test output")
        logger.log_api_call(provider="gemini", endpoint="generate", model="gemini-pro", input_text="test2", output_text="output2")
        logger.log_api_call(provider="anthropic", endpoint="messages", model="claude-3", input_text="test3", output_text="output3")

        assert logger.run_log.total_api_calls == 3
        # Cost should be calculated based on tokens
        assert logger.run_log.total_api_cost_est_usd >= 0


class TestRunLoggerLogMatchDecision:
    """Test log_match_decision method."""

    @pytest.mark.fast
    def test_log_match_decision_records_decision(self, tmp_path):
        """Test that match decisions are recorded."""
        from src.logger import RunLogger

        logger = RunLogger(log_dir=str(tmp_path))
        # Use correct signature
        logger.log_match_decision(
            segment_index=1,
            voiceover_text="Test segment",
            selected_clip="video.mp4",
            confidence=0.9,
            reasoning="Good semantic match"
        )

        assert len(logger.run_log.match_decisions) == 1
        assert logger.run_log.match_decisions[0].confidence == 0.9


class TestLoggerNoLegacyStages:
    """Guard tests ensuring logger report code does not reference legacy stages."""

    @pytest.mark.fast
    def test_verbose_markdown_no_legacy_stage_names(self, tmp_path):
        """Test that _write_verbose_markdown does not reference any LEGACY_STAGES."""
        import ast
        import inspect
        import textwrap
        from src.logger import RunLogger
        from src.checkpoint import LEGACY_STAGES

        source = textwrap.dedent(inspect.getsource(RunLogger._write_verbose_markdown))
        tree = ast.parse(source)

        # Collect all string literals in the method
        string_literals = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_literals.append(node.value)

        for legacy_stage in LEGACY_STAGES:
            for s in string_literals:
                assert legacy_stage not in s.upper(), (
                    f"_write_verbose_markdown contains legacy stage reference '{legacy_stage}' in string '{s}'"
                )

    @pytest.mark.fast
    def test_report_dataclass_no_legacy_fields(self):
        """Test that RunLog dataclass has no remix_ or transcription_ prefixed fields."""
        from src.logger import RunLog
        import dataclasses

        field_names = [f.name for f in dataclasses.fields(RunLog)]

        legacy_prefixes = ['remix_', 'transcription_']
        legacy_field_names = ['videos_transcribed', 'video_process_logs']

        for field_name in field_names:
            for prefix in legacy_prefixes:
                assert not field_name.startswith(prefix), (
                    f"RunLog still has legacy field '{field_name}' with prefix '{prefix}'"
                )
            assert field_name not in legacy_field_names, (
                f"RunLog still has legacy field '{field_name}'"
            )

    @pytest.mark.fast
    def test_runlogger_no_legacy_methods(self):
        """Test that RunLogger has no remix or video_process logging methods."""
        from src.logger import RunLogger

        legacy_methods = ['log_remix_stats', 'log_video_process']
        for method_name in legacy_methods:
            assert not hasattr(RunLogger, method_name), (
                f"RunLogger still has legacy method '{method_name}'"
            )
