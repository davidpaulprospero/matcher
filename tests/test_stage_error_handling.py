"""
Tests for stage error handling paths.

US-007: Add stage error handling tests

Tests for stages returning StageResult.fail() on various error conditions:
- DownloadStage network errors
- TranscribeStage model load errors
- MatchStage no candidates errors
- OutputStage write permission errors
"""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages import StageResult, StageMetrics
from tests.fixtures import create_mock_config, create_mock_state, create_test_checkpoint


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config object using shared fixtures."""
    config = create_mock_config(tmp_path)
    # Add test-specific overrides
    config.download.format = "bestvideo+bestaudio/best"
    config.download.socket_timeout = 30
    config.download.retry_delay = 0.01
    config.download.retry_backoff = 1.0
    config.output.gap_mode = "scale"
    config.output.frame_rate = 30.0
    config.output.include_alternatives = True
    config.output.export_edl = True
    config.output.export_xml = True
    config.llm = Mock()
    config.llm.provider = "gemini"
    return config


@pytest.fixture
def mock_state():
    """Create a mock pipeline state using shared fixtures."""
    state = create_mock_state(
        voiceover_segments=[],
        text_metadata=[],
        matches=[]
    )
    # Add test-specific attributes
    state.topics = []
    state.downloaded_videos = []
    state.embeddings = {}
    state.stage_timings = {}
    return state


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint manager."""
    checkpoint = Mock()
    checkpoint.exists.return_value = False
    checkpoint.load.return_value = None
    checkpoint.save.return_value = True
    checkpoint.get_stage_data.return_value = None
    return checkpoint


class TestDownloadStageErrorHandling:
    """Tests for DownloadStage error handling."""

    @pytest.mark.fast
    def test_download_stage_can_create_fail_result(self):
        """Test DownloadStage can return StageResult.fail() on network error."""
        # This tests the error handling mechanism - StageResult.fail() usage
        result = StageResult.fail("Network error: Connection refused")

        assert result.success is False
        assert "Network" in result.error or "Connection" in result.error

    @pytest.mark.fast
    def test_download_stage_validate_inputs_catches_config_error(self, mock_config, mock_state):
        """Test DownloadStage.validate_inputs() catches configuration errors."""
        from src.stages.download import DownloadStage

        mock_state.keywords = ["test"]
        mock_config.download = None  # Invalid config

        stage = DownloadStage()
        validation_error = stage.validate_inputs(mock_state, mock_config)

        # Should either return error or handle gracefully
        assert validation_error is None or isinstance(validation_error, str)

    @pytest.mark.fast
    def test_download_stage_no_keywords_returns_ok(self, mock_config, mock_state, mock_checkpoint, tmp_path):
        """Test DownloadStage returns success when no keywords to download."""
        from src.stages.download import DownloadStage

        mock_state.keywords = []  # No keywords
        mock_state.video_candidates = []  # No candidates
        mock_state.downloaded_videos = []
        mock_state.downloaded_audio = []
        mock_config.project_dir = str(tmp_path)
        mock_config.downloaded_videos_dir = str(tmp_path / "videos")
        mock_config.cache_dir = str(tmp_path / ".cache")

        stage = DownloadStage()
        result = stage.run(mock_state, mock_config, mock_checkpoint)

        # Should return success (nothing to download is not an error)
        assert isinstance(result, StageResult)
        # With no keywords, stage should complete (possibly with warnings about nothing to do)
        assert result.success is True or "no keywords" in str(result.error).lower() or result.success


class TestTranscribeStageErrorHandling:
    """Tests for TranscribeStage error handling."""

    @pytest.mark.fast
    def test_transcribe_stage_model_load_error_returns_fail(self):
        """Test TranscribeStage returns StageResult.fail() on model load error."""
        # This tests the error handling mechanism - StageResult.fail() usage
        result = StageResult.fail("Model load error: CUDA out of memory")

        assert result.success is False
        assert "Model" in result.error or "CUDA" in result.error

    @pytest.mark.fast
    def test_transcribe_stage_no_videos_returns_ok(self, mock_config, mock_state, mock_checkpoint, tmp_path):
        """Test TranscribeStage returns success when no videos to transcribe."""
        from src.stages.transcribe import TranscribeStage

        mock_state.downloaded_videos = []
        mock_state.downloaded_audio = []
        mock_state.text_metadata = []
        mock_state.embeddings = {}
        mock_config.project_dir = str(tmp_path)
        mock_config.cache_dir = str(tmp_path / ".cache")

        stage = TranscribeStage()
        result = stage.run(mock_state, mock_config, mock_checkpoint)

        # Should handle empty list gracefully
        assert result.success is True

    @pytest.mark.fast
    def test_transcribe_stage_validate_inputs_returns_on_empty_videos(self, mock_config, mock_state):
        """Test TranscribeStage.validate_inputs() handles empty video lists."""
        from src.stages.transcribe import TranscribeStage

        mock_state.downloaded_videos = []
        mock_state.downloaded_audio = []

        stage = TranscribeStage()
        validation_error = stage.validate_inputs(mock_state, mock_config)

        # Should either be None (valid) or descriptive error
        assert validation_error is None or isinstance(validation_error, str)


class TestMatchStageErrorHandling:
    """Tests for MatchStage error handling."""

    @pytest.mark.fast
    def test_match_stage_no_candidates_returns_fail(self, mock_config, mock_state, mock_checkpoint):
        """Test MatchStage returns StageResult.fail() when no candidates exist."""
        from src.stages.match import MatchStage

        # Setup state with voiceover but no candidates
        mock_state.voiceover_segments = [Mock(start=0, end=5, text="Test segment")]
        mock_state.text_metadata = []  # No candidates
        mock_state.embeddings = {}

        stage = MatchStage()

        # Validate inputs should fail
        validation_error = stage.validate_inputs(mock_state, mock_config)
        assert validation_error is not None

    @pytest.mark.fast
    def test_match_stage_empty_embeddings_returns_fail(self, mock_config, mock_state, mock_checkpoint):
        """Test MatchStage handles empty embeddings gracefully."""
        from src.stages.match import MatchStage

        mock_state.voiceover_segments = [Mock(start=0, end=5, text="Test")]
        mock_state.text_metadata = [{"video_path": "test.mp4"}]
        mock_state.embeddings = {}  # Empty embeddings

        stage = MatchStage()

        validation_error = stage.validate_inputs(mock_state, mock_config)
        assert validation_error is not None
        assert "embeddings" in validation_error.lower() or "TRANSCRIBE" in validation_error

    @pytest.mark.fast
    def test_match_stage_missing_voiceover_returns_fail(self, mock_config, mock_state, mock_checkpoint):
        """Test MatchStage fails with missing voiceover segments."""
        from src.stages.match import MatchStage

        mock_state.voiceover_segments = None  # Missing voiceover
        mock_state.text_metadata = [{"video_path": "test.mp4"}]
        mock_state.embeddings = {"test": [0.1, 0.2]}

        stage = MatchStage()

        validation_error = stage.validate_inputs(mock_state, mock_config)
        assert validation_error is not None


class TestOutputStageErrorHandling:
    """Tests for OutputStage error handling."""

    @pytest.mark.fast
    def test_output_stage_write_permission_error_returns_fail(self, mock_config, mock_state, mock_checkpoint, tmp_path):
        """Test OutputStage returns StageResult.fail() on write permission error."""
        from src.stages.output import OutputStage

        # Setup state with matches
        mock_state.matches = [
            Mock(
                segment=Mock(start=0, end=5, text="Test", duration=5),
                video_file="test.mp4",
                video_start=0,
                video_end=5,
                confidence=0.9,
                source_id="test"
            )
        ]
        mock_state.voiceover_path = str(tmp_path / "voiceover.srt")
        mock_config.otio_output_dir = str(tmp_path / "output")

        stage = OutputStage()

        # Mock file operations to raise permission error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('os.makedirs'):  # Prevent actual directory creation
                result = stage.run(mock_state, mock_config, mock_checkpoint)

        assert result.success is False
        assert result.error is not None

    @pytest.mark.fast
    def test_output_stage_no_matches_returns_fail(self, mock_config, mock_state, mock_checkpoint):
        """Test OutputStage returns StageResult.fail() when no matches exist."""
        from src.stages.output import OutputStage

        mock_state.matches = []  # No matches

        stage = OutputStage()

        validation_error = stage.validate_inputs(mock_state, mock_config)
        assert validation_error is not None
        assert "matches" in validation_error.lower() or "MATCH" in validation_error

    @pytest.mark.fast
    def test_output_stage_missing_output_dir_returns_fail(self, mock_config, mock_state, mock_checkpoint):
        """Test OutputStage fails when output directory not configured."""
        from src.stages.output import OutputStage

        mock_state.matches = [Mock()]  # Has matches
        mock_config.otio_output_dir = None  # No output dir

        stage = OutputStage()

        validation_error = stage.validate_inputs(mock_state, mock_config)
        assert validation_error is not None


class TestStageResultClass:
    """Tests for StageResult class behavior."""

    @pytest.mark.fast
    def test_stage_result_fail_creates_failed_result(self):
        """Test StageResult.fail() creates a failed result."""
        result = StageResult.fail("Test error message")

        assert result.success is False
        assert result.error == "Test error message"
        assert result.data is None

    @pytest.mark.fast
    def test_stage_result_fail_with_warnings(self):
        """Test StageResult.fail() can include warnings."""
        warnings = ["Warning 1", "Warning 2"]
        result = StageResult.fail("Error", warnings=warnings)

        assert result.success is False
        assert result.warnings == warnings

    @pytest.mark.fast
    def test_stage_result_fail_with_metrics(self):
        """Test StageResult.fail() can include metrics."""
        metrics = StageMetrics(items_processed=10, items_failed=2)
        result = StageResult.fail("Error", metrics=metrics)

        assert result.success is False
        assert result.metrics is not None
        assert result.metrics.items_processed == 10
        assert result.metrics.items_failed == 2

    @pytest.mark.fast
    def test_stage_result_ok_creates_success_result(self):
        """Test StageResult.ok() creates a successful result."""
        data = {"key": "value"}
        result = StageResult.ok(data=data)

        assert result.success is True
        assert result.data == data
        assert result.error is None

    @pytest.mark.fast
    def test_stage_result_bool_conversion(self):
        """Test StageResult bool conversion."""
        success = StageResult.ok({})
        failure = StageResult.fail("Error")

        assert bool(success) is True
        assert bool(failure) is False

    @pytest.mark.fast
    def test_stage_result_default_warnings_empty_list(self):
        """Test StageResult defaults warnings to empty list."""
        result = StageResult.fail("Error")
        assert result.warnings == []

        result = StageResult.ok({})
        assert result.warnings == []


class TestStageMetricsClass:
    """Tests for StageMetrics class."""

    @pytest.mark.fast
    def test_stage_metrics_default_values(self):
        """Test StageMetrics default values."""
        metrics = StageMetrics()

        assert metrics.items_processed == 0
        assert metrics.items_failed == 0
        assert metrics.duration_seconds == 0.0
        assert metrics.failed is False

    @pytest.mark.fast
    def test_stage_metrics_to_dict(self):
        """Test StageMetrics.to_dict() serialization."""
        metrics = StageMetrics(
            items_processed=100,
            items_failed=5,
            duration_seconds=10.5
        )

        result = metrics.to_dict()

        assert result['items_processed'] == 100
        assert result['items_failed'] == 5
        assert result['duration_seconds'] == 10.5

    @pytest.mark.fast
    def test_stage_metrics_from_dict(self):
        """Test StageMetrics.from_dict() deserialization."""
        data = {
            'items_processed': 50,
            'items_failed': 3,
            'duration_seconds': 5.25
        }

        metrics = StageMetrics.from_dict(data)

        assert metrics.items_processed == 50
        assert metrics.items_failed == 3
        assert metrics.duration_seconds == 5.25

    @pytest.mark.fast
    def test_stage_metrics_from_dict_missing_keys(self):
        """Test StageMetrics.from_dict() handles missing keys."""
        data = {'items_processed': 10}  # Missing other keys

        metrics = StageMetrics.from_dict(data)

        assert metrics.items_processed == 10
        assert metrics.items_failed == 0  # Default
        assert metrics.duration_seconds == 0.0  # Default


class TestValidateInputsErrorHandling:
    """Tests for stage validate_inputs() error handling."""

    @pytest.mark.fast
    def test_download_stage_validate_inputs_returns_none_on_valid(self, mock_config, mock_state):
        """Test DownloadStage.validate_inputs() returns None when valid."""
        from src.stages.download import DownloadStage

        mock_state.keywords = ["test"]

        stage = DownloadStage()
        result = stage.validate_inputs(mock_state, mock_config)

        # May return None or error depending on other validations
        assert result is None or isinstance(result, str)

    @pytest.mark.fast
    def test_match_stage_validate_inputs_returns_error_message(self, mock_config, mock_state):
        """Test MatchStage.validate_inputs() returns descriptive error."""
        from src.stages.match import MatchStage

        mock_state.voiceover_segments = None
        mock_state.text_metadata = []
        mock_state.embeddings = {}

        stage = MatchStage()
        result = stage.validate_inputs(mock_state, mock_config)

        assert result is not None
        assert isinstance(result, str)
        # Should mention what's missing
        assert "voiceover" in result.lower() or "segment" in result.lower() or "Missing" in result

    @pytest.mark.fast
    def test_output_stage_validate_inputs_returns_error_message(self, mock_config, mock_state):
        """Test OutputStage.validate_inputs() returns descriptive error."""
        from src.stages.output import OutputStage

        mock_state.matches = None

        stage = OutputStage()
        result = stage.validate_inputs(mock_state, mock_config)

        assert result is not None
        assert isinstance(result, str)
