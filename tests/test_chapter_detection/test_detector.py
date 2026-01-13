"""Tests for EnhancedChapterDetector orchestrator."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.chapter_detection.detector import EnhancedChapterDetector
from src.chapter_detection.models import ChapterCandidate, DetectionResult


class TestEnhancedChapterDetector:
    """Test the main detector class."""

    def test_init_with_config(self, mock_config):
        """Test detector initialization."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            assert detector.config == mock_config

    def test_detect_chapters_empty(self, mock_config):
        """Test empty segments return empty list."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            result = detector.detect_chapters([])
            assert result == []

    def test_detect_chapters_full_empty(self, mock_config):
        """Test detect_chapters_full with empty segments."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            result = detector.detect_chapters_full([])

            assert isinstance(result, DetectionResult)
            assert result.chapters == []
            assert result.total_segments == 0

    def test_fallback_when_no_llm_client(self, mock_config, sample_segments):
        """Test fallback result when no LLM client."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = None

            result = detector.detect_chapters_full(sample_segments)

            assert result.fallback_used
            assert len(result.chapters) == 1
            assert result.chapters[0].detection_strategy == 'fallback'

    def test_fallback_with_topic(self, mock_config, sample_segments):
        """Test fallback uses overall_topic."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = None

            result = detector.detect_chapters_full(
                sample_segments,
                overall_topic="Paris Travel Guide"
            )

            assert result.fallback_used
            assert result.chapters[0].title == "Paris Travel Guide"

    @patch('src.chapter_detection.detector.run_initial_detection')
    @patch('src.chapter_detection.detector.run_coverage_resolution')
    def test_runs_passes(
        self,
        mock_coverage,
        mock_initial,
        mock_config,
        sample_segments,
    ):
        """Test detector runs all passes."""
        # Setup mocks
        mock_chapter = ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=len(sample_segments) - 1,
            title="Test Chapter",
        )
        mock_initial.return_value = [mock_chapter]
        mock_coverage.return_value = [mock_chapter]

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = Mock()

            # Disable refinement and validation for simpler test
            with patch.object(detector, '_should_run_refinement', return_value=False):
                with patch.object(detector, '_should_run_validation', return_value=False):
                    result = detector.detect_chapters_full(sample_segments)

        assert 'initial' in result.detection_passes_run
        assert 'coverage' in result.detection_passes_run
        assert not result.fallback_used

    def test_detect_chapters_returns_dicts(self, mock_config, sample_segments):
        """Test detect_chapters returns list of dicts."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = None

            result = detector.detect_chapters(sample_segments)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], dict)
            assert 'title' in result[0]

    def test_content_type_mapping(self, mock_config):
        """Test content type to strategy mapping."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)

            assert detector._map_content_type_to_strategy('topic') == 'topic'
            assert detector._map_content_type_to_strategy('location') == 'location'
            assert detector._map_content_type_to_strategy('travel') == 'location'
            assert detector._map_content_type_to_strategy('educational') == 'topic'
            assert detector._map_content_type_to_strategy('unknown') == 'topic'


class TestDetectorWithMockedPasses:
    """Test detector with all passes mocked."""

    @patch('src.chapter_detection.detector.run_initial_detection')
    @patch('src.chapter_detection.detector.run_boundary_refinement')
    @patch('src.chapter_detection.detector.compute_voiceover_embeddings')
    @patch('src.chapter_detection.detector.run_validation')
    @patch('src.chapter_detection.detector.run_coverage_resolution')
    def test_full_pipeline(
        self,
        mock_coverage,
        mock_validation,
        mock_embeddings,
        mock_refinement,
        mock_initial,
        mock_config,
        sample_segments,
    ):
        """Test full pipeline with all passes."""
        # Setup mock chain
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=5,
                title="Chapter 1",
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=6,
                end_segment_idx=9,
                title="Chapter 2",
            ),
        ]

        mock_initial.return_value = chapters.copy()
        mock_refinement.return_value = chapters.copy()
        mock_validation.return_value = chapters.copy()
        mock_coverage.return_value = chapters
        mock_embeddings.return_value = None  # Embeddings can be None

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = Mock()

            result = detector.detect_chapters_full(sample_segments)

        assert len(result.chapters) == 2
        assert 'initial' in result.detection_passes_run
        assert 'refinement' in result.detection_passes_run
        assert 'validation' in result.detection_passes_run
        assert 'coverage' in result.detection_passes_run


class TestDetectorConfigAccess:
    """Test config value access."""

    def test_get_config_value_from_chapter_detection(self):
        """Test reading config from chapter_detection section."""
        config = Mock()
        config.matching.chapter_detection.min_chapter_segments = 5

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(config)
            value = detector._get_config_value('min_chapter_segments', 3)

        assert value == 5

    def test_get_config_value_default(self):
        """Test default value when config missing."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(None)
            value = detector._get_config_value('some_key', 'default')

        assert value == 'default'

    def test_get_config_value_dict_config(self):
        """Test reading from dict-style config."""
        config = Mock()
        config.matching.chapter_detection = {
            'use_validation_pass': False,
            'min_chapter_segments': 4,
        }

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(config)
            value = detector._get_config_value('use_validation_pass', True)

        assert value is False


class TestDetectorLocationResolution:
    """Test location resolution integration."""

    def test_resolves_locations(self, mock_config):
        """Test location resolution is called when service provided."""
        location_service = Mock()
        geo_location = Mock()
        geo_location.name = "Paris"
        geo_location.country_name = "France"
        geo_location.to_dict.return_value = {"name": "Paris", "country": "France"}
        location_service.disambiguate.return_value = geo_location

        # Create chapter with location that needs resolution
        chapter = ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=9,
            title="Paris Tour",
            location_name="Paris",
            visual_keywords=["Eiffel Tower"],
            context_keywords=["romantic"],
        )

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)

            # Call the private method directly
            result = detector._resolve_chapter_locations([chapter], location_service)

        # Location should be resolved
        location_service.disambiguate.assert_called_once()
        assert result[0].location_data == {"name": "Paris", "country": "France"}
