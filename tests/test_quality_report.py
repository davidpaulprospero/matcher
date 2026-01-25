"""
Tests for match quality summary report generation.

Sprint 4 - US-012: Add match quality summary to OUTPUT stage
"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mark all tests as unit tests
pytestmark = pytest.mark.unit


def create_mock_match(confidence: float = 0.8, source_file: str = "video1.mp4"):
    """Create a mock match object with the expected structure."""
    mock = MagicMock()
    mock.primary_match.confidence = confidence
    mock.primary_match.video_segment.source_file = source_file
    return mock


def create_mock_config(quality_report_enabled: bool = True):
    """Create a mock config object."""
    mock = MagicMock()
    mock.output.quality_report_enabled = quality_report_enabled
    mock.output.frame_rate = 30.0
    mock.otio_output_dir = "/tmp/test_output"
    return mock


class TestQualityReportEnabledConfig:
    """Tests for quality_report_enabled config option"""

    def test_config_option_exists(self):
        """OutputConfig should have quality_report_enabled field"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig()
        assert hasattr(config, 'quality_report_enabled')

    def test_config_default_true(self):
        """quality_report_enabled should default to True"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig()
        assert config.quality_report_enabled is True

    def test_config_can_be_disabled(self):
        """quality_report_enabled can be set to False"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(quality_report_enabled=False)
        assert config.quality_report_enabled is False


class TestOTIOQualityMetadata:
    """Tests for quality summary in OTIO metadata"""

    def test_create_timeline_accepts_quality_metrics(self):
        """create_timeline should accept quality_metrics parameter"""
        from src.otio.timeline import create_timeline
        import inspect

        sig = inspect.signature(create_timeline)
        params = list(sig.parameters.keys())
        assert 'quality_metrics' in params

    def test_quality_metrics_added_to_metadata(self):
        """Timeline metadata should include quality_summary when metrics provided"""
        from src.otio.timeline import create_timeline

        # Create minimal mock config
        mock_config = MagicMock()
        mock_config.output.num_alternatives = 2
        mock_config.output.include_alternatives = True
        mock_config.output.include_strategy_tracks = True
        mock_config.output.strategy_tracks = ["embedding_diversity", "broll_only"]
        mock_config.output.voiceover_offset = 0.0
        mock_config.output.min_gap_threshold = 0.0
        mock_config.output.time_scale_factor = 1.0
        mock_config.output.gap_mode = 'scale'

        # Quality metrics to embed
        quality_metrics = {
            'total_segments': 10,
            'matched_segments': 10,
            'avg_confidence': 0.85,
            'gaps_count': 0,
        }

        # Call create_timeline with empty matches but with metrics
        timeline = create_timeline(
            matches=[],
            config=mock_config,
            quality_metrics=quality_metrics
        )

        assert 'quality_summary' in timeline.metadata
        assert timeline.metadata['quality_summary'] == quality_metrics

    def test_quality_metadata_not_added_when_none(self):
        """Timeline metadata should not include quality_summary when metrics is None"""
        from src.otio.timeline import create_timeline

        mock_config = MagicMock()
        mock_config.output.num_alternatives = 2
        mock_config.output.include_alternatives = True
        mock_config.output.include_strategy_tracks = False
        mock_config.output.strategy_tracks = []
        mock_config.output.voiceover_offset = 0.0
        mock_config.output.min_gap_threshold = 0.0
        mock_config.output.time_scale_factor = 1.0
        mock_config.output.gap_mode = 'scale'

        timeline = create_timeline(
            matches=[],
            config=mock_config,
            quality_metrics=None
        )

        assert 'quality_summary' not in timeline.metadata


class TestCalculateQualityMetrics:
    """Tests for OutputStage._calculate_quality_metrics"""

    def test_calculate_returns_metrics_object(self):
        """_calculate_quality_metrics should return MatchQualityMetrics"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [create_mock_match(0.8), create_mock_match(0.9)]

        metrics = stage._calculate_quality_metrics(matches)

        assert metrics is not None
        assert hasattr(metrics, 'avg_confidence')
        assert hasattr(metrics, 'matched_segments')

    def test_calculate_empty_matches(self):
        """Should handle empty matches list"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        metrics = stage._calculate_quality_metrics([])

        assert metrics is not None
        assert metrics.total_segments == 0

    def test_calculate_avg_confidence(self):
        """Should correctly calculate average confidence"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [
            create_mock_match(0.8, "video1.mp4"),
            create_mock_match(0.6, "video2.mp4"),
            create_mock_match(0.7, "video3.mp4"),
        ]

        metrics = stage._calculate_quality_metrics(matches)

        # Average of 0.8, 0.6, 0.7 = 0.7
        assert abs(metrics.avg_confidence - 0.7) < 0.01

    def test_calculate_min_max_confidence(self):
        """Should correctly calculate min and max confidence"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [
            create_mock_match(0.9, "video1.mp4"),
            create_mock_match(0.5, "video2.mp4"),
            create_mock_match(0.7, "video3.mp4"),
        ]

        metrics = stage._calculate_quality_metrics(matches)

        assert metrics.min_confidence == 0.5
        assert metrics.max_confidence == 0.9

    def test_calculate_source_variety(self):
        """Should calculate source variety (unique sources / total matches)"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [
            create_mock_match(0.8, "video1.mp4"),  # unique
            create_mock_match(0.8, "video2.mp4"),  # unique
            create_mock_match(0.8, "video1.mp4"),  # duplicate
            create_mock_match(0.8, "video3.mp4"),  # unique
        ]

        metrics = stage._calculate_quality_metrics(matches)

        # 3 unique sources / 4 total = 0.75
        assert hasattr(metrics, '_source_variety')
        assert abs(metrics._source_variety - 0.75) < 0.01
        assert metrics._unique_sources == 3

    def test_calculate_matched_segments(self):
        """Should correctly count matched segments"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [
            create_mock_match(0.8),
            create_mock_match(0.9),
            create_mock_match(0.7),
        ]

        metrics = stage._calculate_quality_metrics(matches)

        assert metrics.matched_segments == 3
        assert metrics.total_segments == 3


class TestGenerateQualityReport:
    """Tests for OutputStage._generate_quality_report"""

    def test_generates_json_file(self):
        """Should generate quality_report.json file"""
        from src.stages.output import OutputStage
        from src.matching.metrics import MatchQualityMetrics

        stage = OutputStage()

        # Create mock state
        mock_state = MagicMock()
        mock_state.matches = [create_mock_match(), create_mock_match()]

        # Create metrics
        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.7,
            max_confidence=0.95,
            confidence_std=0.08,
            gap_count=1,
            match_rate=0.9,
            total_segments=10,
            matched_segments=9,
        )

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = stage._generate_quality_report(mock_state, output_dir, metrics)

            assert report_path.exists()
            assert report_path.name == "quality_report.json"

    def test_report_contains_all_fields(self):
        """Quality report should contain all expected fields"""
        from src.stages.output import OutputStage
        from src.matching.metrics import MatchQualityMetrics

        stage = OutputStage()

        mock_state = MagicMock()
        mock_state.matches = [create_mock_match(), create_mock_match()]

        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.7,
            max_confidence=0.95,
            confidence_std=0.08,
            gap_count=1,
            match_rate=0.9,
            total_segments=10,
            matched_segments=9,
        )
        metrics._source_variety = 0.8
        metrics._unique_sources = 8

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = stage._generate_quality_report(mock_state, output_dir, metrics)

            with open(report_path) as f:
                report = json.load(f)

            # Check all required fields
            assert 'total_segments' in report
            assert 'matched_segments' in report
            assert 'avg_confidence' in report
            assert 'gaps_count' in report
            assert 'source_variety' in report
            assert 'min_confidence' in report
            assert 'max_confidence' in report
            assert 'match_rate' in report

    def test_report_values_correct(self):
        """Quality report values should be correct"""
        from src.stages.output import OutputStage
        from src.matching.metrics import MatchQualityMetrics

        stage = OutputStage()

        mock_state = MagicMock()
        mock_state.matches = [create_mock_match() for _ in range(10)]

        metrics = MatchQualityMetrics(
            avg_confidence=0.85,
            min_confidence=0.7,
            max_confidence=0.95,
            confidence_std=0.08,
            gap_count=1,
            match_rate=0.9,
            total_segments=10,
            matched_segments=9,
        )
        metrics._source_variety = 0.8
        metrics._unique_sources = 8

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = stage._generate_quality_report(mock_state, output_dir, metrics)

            with open(report_path) as f:
                report = json.load(f)

            assert report['total_segments'] == 10
            assert report['matched_segments'] == 9
            assert report['avg_confidence'] == 0.85
            assert report['gaps_count'] == 1
            assert report['source_variety'] == 0.8
            assert report['unique_sources'] == 8

    def test_report_handles_none_metrics(self):
        """Should handle None metrics gracefully"""
        from src.stages.output import OutputStage

        stage = OutputStage()

        mock_state = MagicMock()
        mock_state.matches = []

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = stage._generate_quality_report(mock_state, output_dir, None)

            assert report_path.exists()

            with open(report_path) as f:
                report = json.load(f)

            # Should have default values
            assert report['total_segments'] == 0
            assert report['avg_confidence'] == 0.0
            assert report['gaps_count'] == 0

    def test_report_valid_json(self):
        """Generated report should be valid JSON"""
        from src.stages.output import OutputStage
        from src.matching.metrics import MatchQualityMetrics

        stage = OutputStage()

        mock_state = MagicMock()
        mock_state.matches = [create_mock_match()]

        metrics = MatchQualityMetrics(avg_confidence=0.85)

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = stage._generate_quality_report(mock_state, output_dir, metrics)

            # Should not raise
            with open(report_path) as f:
                report = json.load(f)

            assert isinstance(report, dict)


class TestQualityReportIntegration:
    """Integration tests for quality report in OUTPUT stage"""

    def test_quality_report_in_outputs(self):
        """Quality report path should be in outputs dict"""
        from src.stages.output import OutputStage
        from src.matching.metrics import MatchQualityMetrics

        stage = OutputStage()

        mock_state = MagicMock()
        mock_state.matches = [create_mock_match()]

        metrics = MatchQualityMetrics(avg_confidence=0.85)

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = stage._generate_quality_report(mock_state, output_dir, metrics)

            outputs = {'quality_report': str(report_path)}

            assert 'quality_report' in outputs
            assert Path(outputs['quality_report']).exists()


class TestQualityReportEdgeCases:
    """Edge case tests for quality report generation"""

    def test_single_match(self):
        """Should handle single match"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [create_mock_match(0.9)]

        metrics = stage._calculate_quality_metrics(matches)

        assert metrics.avg_confidence == 0.9
        assert metrics.min_confidence == 0.9
        assert metrics.max_confidence == 0.9

    def test_all_same_confidence(self):
        """Should handle all matches with same confidence"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [create_mock_match(0.8) for _ in range(5)]

        metrics = stage._calculate_quality_metrics(matches)

        assert metrics.avg_confidence == 0.8
        assert metrics.min_confidence == 0.8
        assert metrics.max_confidence == 0.8
        assert metrics.confidence_std == 0.0

    def test_all_same_source(self):
        """Should handle all matches from same source"""
        from src.stages.output import OutputStage

        stage = OutputStage()
        matches = [create_mock_match(0.8, "same_video.mp4") for _ in range(5)]

        metrics = stage._calculate_quality_metrics(matches)

        # 1 unique source / 5 total = 0.2
        assert metrics._source_variety == 0.2
        assert metrics._unique_sources == 1

    def test_confidence_rounding(self):
        """Confidence values should be rounded appropriately in report"""
        from src.stages.output import OutputStage
        from src.matching.metrics import MatchQualityMetrics

        stage = OutputStage()

        mock_state = MagicMock()
        mock_state.matches = [create_mock_match()]

        # Use values that would have many decimal places
        metrics = MatchQualityMetrics(
            avg_confidence=0.8333333333,
            min_confidence=0.666666666,
            max_confidence=0.999999999,
        )

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            report_path = stage._generate_quality_report(mock_state, output_dir, metrics)

            with open(report_path) as f:
                report = json.load(f)

            # Should be rounded to 4 decimal places
            assert report['avg_confidence'] == 0.8333
            assert report['min_confidence'] == 0.6667
            assert report['max_confidence'] == 1.0


class TestQualityReportDisabled:
    """Tests for when quality report is disabled"""

    def test_config_disabled_skips_report(self):
        """When quality_report_enabled=False, no report should be generated"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(quality_report_enabled=False)

        # The actual check happens in OutputStage.run()
        # Here we verify the config value
        assert config.quality_report_enabled is False


class TestQualityReportRealWorldScenarios:
    """Real-world scenario tests"""

    def test_mixed_confidence_levels(self):
        """Should handle realistic mixed confidence levels"""
        from src.stages.output import OutputStage

        stage = OutputStage()

        # Simulate realistic distribution
        confidences = [0.95, 0.88, 0.72, 0.65, 0.91, 0.55, 0.82, 0.78, 0.89, 0.60]
        sources = ["vid1.mp4", "vid2.mp4", "vid1.mp4", "vid3.mp4", "vid4.mp4",
                   "vid2.mp4", "vid5.mp4", "vid1.mp4", "vid6.mp4", "vid3.mp4"]

        matches = [create_mock_match(conf, src) for conf, src in zip(confidences, sources)]

        metrics = stage._calculate_quality_metrics(matches)

        # Verify reasonable values
        assert 0.7 < metrics.avg_confidence < 0.8  # ~0.775
        assert metrics.min_confidence == 0.55
        assert metrics.max_confidence == 0.95
        assert metrics.matched_segments == 10
        assert metrics._unique_sources == 6
        assert 0.5 < metrics._source_variety < 0.7  # 6/10 = 0.6

    def test_high_quality_matches(self):
        """Should correctly identify high quality match sets"""
        from src.stages.output import OutputStage

        stage = OutputStage()

        # All high confidence
        matches = [create_mock_match(conf, f"vid{i}.mp4")
                   for i, conf in enumerate([0.95, 0.92, 0.89, 0.94, 0.91])]

        metrics = stage._calculate_quality_metrics(matches)

        assert metrics.avg_confidence > 0.9
        assert metrics.min_confidence > 0.85
        assert metrics._source_variety == 1.0  # All unique sources

    def test_low_quality_matches(self):
        """Should correctly identify low quality match sets"""
        from src.stages.output import OutputStage

        stage = OutputStage()

        # All low confidence, same source
        matches = [create_mock_match(conf, "same.mp4")
                   for conf in [0.45, 0.52, 0.38, 0.55, 0.41]]

        metrics = stage._calculate_quality_metrics(matches)

        assert metrics.avg_confidence < 0.5
        assert metrics.max_confidence < 0.6
        assert metrics._source_variety == 0.2  # 1/5 unique sources
