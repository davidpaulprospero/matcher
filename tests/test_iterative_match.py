"""
Tests for Iterative Match Stage (High Matches Mode)

Tests the iterative matching stage that orchestrates the
download → match → analyze cycle.
"""

import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import Mock, patch, MagicMock


class TestIterativeMatchStageBasics:
    """Basic tests for IterativeMatchStage"""

    def test_stage_name(self):
        """Test stage has correct name"""
        from src.stages.iterative_match import IterativeMatchStage

        stage = IterativeMatchStage()
        assert stage.name == "ITERATIVE_MATCH"

    def test_stage_description(self):
        """Test stage has description"""
        from src.stages.iterative_match import IterativeMatchStage

        stage = IterativeMatchStage()
        assert "iterative" in stage.description.lower()
        assert "coverage" in stage.description.lower()

    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage

        stage_class = get_stage("ITERATIVE_MATCH")
        assert stage_class is not None
        assert stage_class.name == "ITERATIVE_MATCH"


class TestIterativeMatchStageSkipConditions:
    """Tests for conditions when stage should skip"""

    @pytest.fixture
    def mock_state(self):
        """Create mock pipeline state"""
        state = Mock()
        state.project_dir = tempfile.mkdtemp()
        state.voiceover_segments = [Mock() for _ in range(10)]
        state.matches = [Mock() for _ in range(10)]
        state.downloaded_videos = [Mock() for _ in range(20)]
        state.keywords = ["test1", "test2"]
        state.video_candidates = [Mock() for _ in range(5)]
        state.text_metadata = []
        state.iterative_match_state = None
        return state

    @pytest.fixture
    def mock_config_disabled(self):
        """Create mock config with high matches mode disabled"""
        config = Mock()
        config.matching = Mock()
        config.matching.high_matches_mode = Mock()
        config.matching.high_matches_mode.enabled = False
        return config

    @pytest.fixture
    def mock_config_enabled(self):
        """Create mock config with high matches mode enabled"""
        config = Mock()
        config.matching = Mock()
        config.matching.high_matches_mode = Mock()
        config.matching.high_matches_mode.enabled = True
        config.matching.high_matches_mode.target_confidence = 0.90
        config.matching.high_matches_mode.coverage_target = 0.85
        config.matching.high_matches_mode.max_iterations = 5
        config.matching.high_matches_mode.videos_per_iteration = 10
        config.matching.high_matches_mode.keyword_strategy = "weak_segments"
        return config

    @pytest.fixture
    def mock_checkpoint(self):
        """Create mock checkpoint manager"""
        checkpoint = Mock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.save = Mock()
        checkpoint.get_stage_data.return_value = None
        return checkpoint

    def test_skips_when_disabled(self, mock_state, mock_config_disabled, mock_checkpoint):
        """Test stage skips when high matches mode is disabled"""
        from src.stages.iterative_match import IterativeMatchStage

        stage = IterativeMatchStage()
        result = stage.run(mock_state, mock_config_disabled, mock_checkpoint)

        assert result.success is True

    @patch("src.stages.iterative_match.HighMatchesLogger")
    @patch("src.stages.iterative_match.analyze_coverage")
    def test_exits_early_when_target_met(
        self, mock_analyze, mock_logger_class,
        mock_state, mock_config_enabled, mock_checkpoint
    ):
        """Test stage exits when coverage target already met"""
        from src.stages.iterative_match import IterativeMatchStage
        from src.matching.coverage_analyzer import CoverageReport

        # Mock coverage already at target
        mock_analyze.return_value = CoverageReport(
            total_segments=100,
            high_confidence=90,
            medium_confidence=5,
            low_confidence=5,
            coverage_ratio=0.90,  # >= 0.85 target
        )

        mock_logger_class.return_value = Mock()

        stage = IterativeMatchStage()
        result = stage.run(mock_state, mock_config_enabled, mock_checkpoint)

        assert result.success is True
        # Should not enter iteration loop
        assert mock_state.iterative_match_state.target_achieved is True


class TestIterativeMatchShouldStop:
    """Tests for _should_stop method"""

    @pytest.fixture
    def stage(self):
        """Create stage instance"""
        from src.stages.iterative_match import IterativeMatchStage
        return IterativeMatchStage()

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock()
        config.matching = Mock()
        config.matching.high_matches_mode = Mock()
        config.matching.high_matches_mode.coverage_target = 0.85
        config.matching.high_matches_mode.max_iterations = 5
        return config

    def test_stops_when_target_achieved(self, stage, mock_config):
        """Test stops when coverage target achieved"""
        from src.matching.coverage_analyzer import CoverageReport, WeakSegment
        from src.state import IterativeMatchState

        coverage = CoverageReport(
            total_segments=100,
            high_confidence=90,
            medium_confidence=5,
            low_confidence=5,
            coverage_ratio=0.90,  # >= 0.85 target
            weak_segments=[WeakSegment("S001", 0, "test", 0.5)],
        )
        iter_state = IterativeMatchState(iteration_count=2)

        result = stage._should_stop(coverage, iter_state, mock_config)
        assert result is True

    def test_stops_when_max_iterations_reached(self, stage, mock_config):
        """Test stops when max iterations reached"""
        from src.matching.coverage_analyzer import CoverageReport, WeakSegment
        from src.state import IterativeMatchState

        coverage = CoverageReport(
            total_segments=100,
            high_confidence=50,
            medium_confidence=30,
            low_confidence=20,
            coverage_ratio=0.50,
            weak_segments=[WeakSegment("S001", 0, "test", 0.5)],
        )
        iter_state = IterativeMatchState(iteration_count=5)  # = max

        result = stage._should_stop(coverage, iter_state, mock_config)
        assert result is True

    def test_stops_when_no_weak_segments(self, stage, mock_config):
        """Test stops when no weak segments remain"""
        from src.matching.coverage_analyzer import CoverageReport
        from src.state import IterativeMatchState

        coverage = CoverageReport(
            total_segments=100,
            high_confidence=70,
            medium_confidence=30,
            low_confidence=0,
            coverage_ratio=0.70,
            weak_segments=[],  # No weak segments
        )
        iter_state = IterativeMatchState(iteration_count=1)

        result = stage._should_stop(coverage, iter_state, mock_config)
        assert result is True

    def test_continues_when_conditions_not_met(self, stage, mock_config):
        """Test continues when stop conditions not met"""
        from src.matching.coverage_analyzer import CoverageReport, WeakSegment
        from src.state import IterativeMatchState

        coverage = CoverageReport(
            total_segments=100,
            high_confidence=60,
            medium_confidence=25,
            low_confidence=15,
            coverage_ratio=0.60,  # < 0.85 target
            weak_segments=[WeakSegment("S001", 0, "test", 0.5)],
        )
        iter_state = IterativeMatchState(iteration_count=2)  # < 5 max

        result = stage._should_stop(coverage, iter_state, mock_config)
        assert result is False


class TestIterativeMatchAnalyzeCoverage:
    """Tests for _analyze_coverage method"""

    @pytest.fixture
    def stage(self):
        """Create stage instance"""
        from src.stages.iterative_match import IterativeMatchStage
        return IterativeMatchStage()

    @pytest.fixture
    def mock_state(self):
        """Create mock state with matches and segments"""
        @dataclass
        class SimpleSegment:
            index: int
            text: str

        @dataclass
        class SimpleMatch:
            segment_index: int
            confidence: float
            video_file: str = ""

        state = Mock()
        state.voiceover_segments = [SimpleSegment(i, f"Segment {i}") for i in range(10)]
        state.matches = [SimpleMatch(i, 0.80 + i * 0.02) for i in range(10)]
        return state

    def test_analyzes_coverage_correctly(self, stage, mock_state):
        """Test coverage analysis returns valid report"""
        result = stage._analyze_coverage(mock_state, target_confidence=0.90)

        assert result.total_segments == 10
        assert result.high_confidence >= 0
        assert result.coverage_ratio >= 0


class TestIterativeMatchGetResultData:
    """Tests for _get_result_data method"""

    def test_returns_checkpoint_data(self):
        """Test result data for checkpointing"""
        from src.stages.iterative_match import IterativeMatchStage
        from src.state import IterativeMatchState

        stage = IterativeMatchStage()
        iter_state = IterativeMatchState(
            iteration_count=3,
            coverage_history=[0.50, 0.65, 0.75, 0.82],
            videos_added_per_iteration=[10, 8, 5],
            final_coverage=0.82,
            target_achieved=False,
            weak_segment_count=18,
        )

        data = stage._get_result_data(iter_state)

        assert data["iteration_count"] == 3
        assert data["coverage_history"] == [0.50, 0.65, 0.75, 0.82]
        assert data["videos_added_per_iteration"] == [10, 8, 5]
        assert data["final_coverage"] == 0.82
        assert data["target_achieved"] is False
        assert data["weak_segment_count"] == 18


class TestIterativeMatchCanSkip:
    """Tests for can_skip method"""

    def test_delegates_to_checkpoint(self):
        """Test can_skip delegates to checkpoint manager"""
        from src.stages.iterative_match import IterativeMatchStage

        stage = IterativeMatchStage()
        state = Mock()
        checkpoint = Mock()

        checkpoint.should_skip_stage.return_value = True
        assert stage.can_skip(state, checkpoint) is True

        checkpoint.should_skip_stage.return_value = False
        assert stage.can_skip(state, checkpoint) is False


class TestIterativeMatchRestore:
    """Tests for restore method"""

    @pytest.fixture
    def stage(self):
        """Create stage instance"""
        from src.stages.iterative_match import IterativeMatchStage
        return IterativeMatchStage()

    def test_restore_from_checkpoint(self, stage):
        """Test restoring state from checkpoint data"""
        from src.state import IterativeMatchState

        state = Mock()
        state.iterative_match_state = None
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = {
            "iteration_count": 2,
            "coverage_history": [0.5, 0.7],
            "videos_added_per_iteration": [10, 8],
            "final_coverage": 0.70,
            "target_achieved": False,
            "weak_segment_count": 30,
        }

        result = stage.restore(state, checkpoint)

        assert result is True
        assert state.iterative_match_state.iteration_count == 2
        assert state.iterative_match_state.coverage_history == [0.5, 0.7]
        assert state.iterative_match_state.final_coverage == 0.70

    def test_restore_no_data(self, stage):
        """Test restore returns False when no checkpoint data"""
        state = Mock()
        checkpoint = Mock()
        checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, checkpoint)

        assert result is False


class TestIterativeMatchState:
    """Tests for IterativeMatchState dataclass"""

    def test_default_values(self):
        """Test default values for IterativeMatchState"""
        from src.state import IterativeMatchState

        state = IterativeMatchState()

        assert state.iteration_count == 0
        assert state.coverage_history == []
        assert state.videos_added_per_iteration == []
        assert state.final_coverage == 0.0
        assert state.target_achieved is False
        assert state.weak_segment_count == 0

    def test_custom_values(self):
        """Test custom values for IterativeMatchState"""
        from src.state import IterativeMatchState

        state = IterativeMatchState(
            iteration_count=3,
            coverage_history=[0.5, 0.6, 0.7],
            videos_added_per_iteration=[10, 8, 5],
            final_coverage=0.70,
            target_achieved=True,
            weak_segment_count=20,
        )

        assert state.iteration_count == 3
        assert len(state.coverage_history) == 3
        assert sum(state.videos_added_per_iteration) == 23
        assert state.target_achieved is True


class TestIterativeMatchPrintReport:
    """Tests for _print_report method"""

    @patch("src.stages.iterative_match.logger")
    def test_prints_summary(self, mock_logger):
        """Test report prints iteration summary"""
        from src.stages.iterative_match import IterativeMatchStage
        from src.state import IterativeMatchState
        from src.matching.coverage_analyzer import CoverageReport

        stage = IterativeMatchStage()
        iter_state = IterativeMatchState(
            iteration_count=3,
            coverage_history=[0.50, 0.65, 0.75, 0.82],
            videos_added_per_iteration=[10, 8, 5],
            final_coverage=0.82,
            target_achieved=False,
            weak_segment_count=18,
        )
        coverage = CoverageReport(
            total_segments=100,
            high_confidence=82,
            medium_confidence=10,
            low_confidence=8,
            coverage_ratio=0.82,
        )

        config = Mock()
        config.matching.high_matches_mode.target_confidence = 0.90
        config.matching.high_matches_mode.coverage_target = 0.85

        stage._print_report(iter_state, coverage, config)

        # Verify logging was called
        assert mock_logger.info.called


class TestIterativeMatchDownloadPexels:
    """Tests for _download_pexels_videos method"""

    @pytest.fixture
    def stage(self):
        """Create stage instance"""
        from src.stages.iterative_match import IterativeMatchStage
        return IterativeMatchStage()

    def test_returns_zero_when_pexels_disabled(self, stage):
        """Test returns 0 when Pexels is not enabled"""
        state = Mock()
        config = Mock()
        config.pexels = None  # Pexels not configured

        result = stage._download_pexels_videos(state, config, ["ocean"])

        assert result == 0

    def test_returns_zero_when_pexels_not_enabled(self, stage):
        """Test returns 0 when Pexels explicitly disabled"""
        state = Mock()
        config = Mock()
        config.pexels = Mock()
        config.pexels.enabled = False

        result = stage._download_pexels_videos(state, config, ["ocean"])

        assert result == 0

    @patch.dict("os.environ", {"PEXELS_API_KEY": ""})
    def test_returns_zero_without_api_key(self, stage):
        """Test returns 0 when no API key"""
        state = Mock()
        config = Mock()
        config.pexels = Mock()
        config.pexels.enabled = True

        result = stage._download_pexels_videos(state, config, ["ocean"])

        assert result == 0


class TestIterativeMatchDownloadVideos:
    """Tests for _download_videos method"""

    @pytest.fixture
    def stage(self):
        """Create stage instance"""
        from src.stages.iterative_match import IterativeMatchStage
        return IterativeMatchStage()

    @pytest.fixture
    def mock_state(self):
        """Create mock state"""
        state = Mock()
        state.keywords = ["existing1", "existing2"]
        state.video_candidates = [Mock(video_id="vid1"), Mock(video_id="vid2")]
        state.downloaded_videos = []
        state.text_metadata = []
        return state

    def test_clears_candidates_before_search(self, stage, mock_state):
        """Test that video_candidates is cleared before searching"""
        from src.stages import StageResult

        config = Mock()
        config.download = Mock()
        config.download.search_pool_multiplier = 5
        config.download.max_search_pool = 100
        checkpoint = Mock()

        # Patch stages at the import location (inside the method)
        with patch("src.stages.video_metadata.VideoMetadataStage") as mock_vm, \
             patch("src.stages.caption.CaptionStage") as mock_cap, \
             patch("src.stages.download.DownloadStage") as mock_dl:

            mock_vm.return_value.run.return_value = StageResult.ok()
            mock_cap.return_value.run.return_value = StageResult.ok()
            mock_dl.return_value.run.return_value = StageResult.ok()

            stage._download_videos(mock_state, config, checkpoint, ["new_keyword"])

            # VideoMetadataStage should have been called
            mock_vm.return_value.run.assert_called_once()

    def test_handles_metadata_stage_failure(self, stage, mock_state):
        """Test handling of metadata stage failure"""
        from src.stages import StageResult

        config = Mock()
        config.download = Mock()
        config.download.search_pool_multiplier = 5
        config.download.max_search_pool = 100
        checkpoint = Mock()

        with patch("src.stages.video_metadata.VideoMetadataStage") as mock_vm:
            mock_vm.return_value.run.return_value = StageResult.fail("Test error")

            result = stage._download_videos(mock_state, config, checkpoint, ["keyword"])

            assert result is False


class TestIterativeMatchTranscribeNewVideos:
    """Tests for _transcribe_new_videos method"""

    @pytest.fixture
    def stage(self):
        """Create stage instance"""
        from src.stages.iterative_match import IterativeMatchStage
        return IterativeMatchStage()

    def test_runs_transcribe_stage(self, stage):
        """Test that TranscribeStage is run"""
        from src.stages import StageResult

        state = Mock()
        state.text_metadata = []
        config = Mock()
        checkpoint = Mock()

        with patch("src.stages.transcribe.TranscribeStage") as mock_transcribe:
            mock_transcribe.return_value.run.return_value = StageResult.ok()

            result = stage._transcribe_new_videos(state, config, checkpoint)

            assert result is True
            mock_transcribe.return_value.run.assert_called_once()


class TestIterativeMatchRematch:
    """Tests for _rematch method"""

    @pytest.fixture
    def stage(self):
        """Create stage instance"""
        from src.stages.iterative_match import IterativeMatchStage
        return IterativeMatchStage()

    def test_clears_matches_before_rematch(self, stage):
        """Test that matches are cleared before re-matching"""
        from src.stages import StageResult

        state = Mock()
        state.matches = [Mock() for _ in range(10)]
        config = Mock()
        checkpoint = Mock()

        with patch("src.stages.match.MatchStage") as mock_match:
            mock_match.return_value.run.return_value = StageResult.ok()

            stage._rematch(state, config, checkpoint)

            # clear_matches should be called
            state.clear_matches.assert_called_once()

    def test_runs_match_stage(self, stage):
        """Test that MatchStage is run"""
        from src.stages import StageResult

        state = Mock()
        state.matches = [{"confidence": 0.9}]  # Real list for len()
        config = Mock()
        checkpoint = Mock()

        with patch("src.stages.match.MatchStage") as mock_match:
            mock_match.return_value.run.return_value = StageResult.ok()

            result = stage._rematch(state, config, checkpoint)

            assert result is True
            mock_match.return_value.run.assert_called_once()


class TestIterativeMatchConfigConversion:
    """Tests for config dict-to-object conversion"""

    def test_converts_dict_config_to_object(self):
        """Test that dict config is converted to HighMatchesModeConfig"""
        from src.stages.iterative_match import IterativeMatchStage
        from src.matching.coverage_analyzer import CoverageReport

        with patch("src.stages.iterative_match.HighMatchesLogger") as mock_logger, \
             patch("src.stages.iterative_match.analyze_coverage") as mock_analyze:

            mock_analyze.return_value = CoverageReport(
                total_segments=100,
                high_confidence=90,
                medium_confidence=5,
                low_confidence=5,
                coverage_ratio=0.90,
            )
            mock_logger.return_value = Mock()

            stage = IterativeMatchStage()

            state = Mock()
            state.project_dir = tempfile.mkdtemp()
            # Use real lists instead of mocks for len() to work
            state.voiceover_segments = [{"index": i, "text": f"seg {i}"} for i in range(10)]
            state.matches = [{"segment_index": i, "confidence": 0.9} for i in range(10)]
            state.downloaded_videos = [{"file": f"v{i}.mp4"} for i in range(20)]
            state.keywords = ["kw1", "kw2"]
            state.video_candidates = []
            state.text_metadata = []
            state.iterative_match_state = None

            # Config as dict (simulating YAML load without __post_init__)
            config = Mock()
            config.matching = Mock()
            config.matching.high_matches_mode = {
                "enabled": True,
                "target_confidence": 0.90,
                "coverage_target": 0.85,
                "max_iterations": 5,
                "videos_per_iteration": 10,
                "keyword_strategy": "weak_segments",
            }

            checkpoint = Mock()
            checkpoint.save = Mock()

            result = stage.run(state, config, checkpoint)

            # Should succeed without error
            assert result.success is True
            # Config should have been converted
            from src.config.sections.matching import HighMatchesModeConfig
            assert isinstance(config.matching.high_matches_mode, HighMatchesModeConfig)
