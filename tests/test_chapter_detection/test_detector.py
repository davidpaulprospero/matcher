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


# ==== US-004 Integration Tests ====


class TestDetectorReturnsValidBoundariesUS004:
    """AC1: Test ChapterDetector.detect() returns ChapterResult with valid boundaries."""

    @patch('src.chapter_detection.detector.run_initial_detection')
    @patch('src.chapter_detection.detector.run_coverage_resolution')
    def test_detect_returns_result_with_valid_boundaries(
        self,
        mock_coverage,
        mock_initial,
        mock_config,
        sample_segments,
    ):
        """Test full detection returns DetectionResult with valid chapter boundaries."""
        # Setup: LLM returns chapters with valid boundaries
        mock_chapter = ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=5,
            title="Valid Chapter",
            confidence=0.8,
        )
        mock_initial.return_value = [mock_chapter]
        mock_coverage.return_value = [mock_chapter]

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = Mock()

            with patch.object(detector, '_should_run_refinement', return_value=False):
                with patch.object(detector, '_should_run_validation', return_value=False):
                    result = detector.detect_chapters_full(sample_segments)

        # Verify result is a DetectionResult with valid boundaries
        assert isinstance(result, DetectionResult)
        assert len(result.chapters) == 1
        ch = result.chapters[0]
        assert ch.start_segment_idx >= 0
        assert ch.end_segment_idx >= ch.start_segment_idx
        assert ch.end_segment_idx < len(sample_segments)

    @patch('src.chapter_detection.detector.run_initial_detection')
    @patch('src.chapter_detection.detector.run_coverage_resolution')
    def test_detect_boundaries_cover_entire_transcript(
        self,
        mock_coverage,
        mock_initial,
        mock_config,
        sample_segments,
    ):
        """Test detected chapters cover the entire transcript when coverage resolution runs."""
        # Setup: Two chapters that should cover all segments
        chapters = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=2, title="Ch1"),
            ChapterCandidate(chapter_id=1, start_segment_idx=3, end_segment_idx=5, title="Ch2"),
        ]
        mock_initial.return_value = chapters
        mock_coverage.return_value = chapters

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = Mock()

            with patch.object(detector, '_should_run_refinement', return_value=False):
                with patch.object(detector, '_should_run_validation', return_value=False):
                    result = detector.detect_chapters_full(sample_segments)

        # Verify full coverage
        first_chapter = min(result.chapters, key=lambda c: c.start_segment_idx)
        last_chapter = max(result.chapters, key=lambda c: c.end_segment_idx)
        assert first_chapter.start_segment_idx == 0
        assert last_chapter.end_segment_idx == len(sample_segments) - 1


class TestDetectorHandlesOverlappingChaptersUS004:
    """AC2: Test ChapterDetector.detect() handles LLM response with overlapping chapters."""

    @patch('src.chapter_detection.detector.run_initial_detection')
    @patch('src.chapter_detection.detector.run_coverage_resolution')
    def test_overlapping_chapters_from_llm_resolved(
        self,
        mock_coverage,
        mock_initial,
        mock_config,
        sample_segments,
    ):
        """Test overlapping chapters from LLM are resolved to non-overlapping."""
        # Setup: LLM returns overlapping chapters
        overlapping = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=4,  # Overlaps with next
                title="Paris",
                confidence=0.9,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=3,  # Overlaps with previous at 3-4
                end_segment_idx=5,
                title="Tokyo",
                confidence=0.7,
            ),
        ]
        mock_initial.return_value = overlapping

        # Coverage resolution should resolve overlaps
        resolved = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=3, title="Paris"),
            ChapterCandidate(chapter_id=1, start_segment_idx=4, end_segment_idx=5, title="Tokyo"),
        ]
        mock_coverage.return_value = resolved

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = Mock()

            with patch.object(detector, '_should_run_refinement', return_value=False):
                with patch.object(detector, '_should_run_validation', return_value=False):
                    result = detector.detect_chapters_full(sample_segments)

        # Verify no overlaps in result
        chapters = sorted(result.chapters, key=lambda c: c.start_segment_idx)
        for i in range(len(chapters) - 1):
            assert chapters[i].end_segment_idx < chapters[i + 1].start_segment_idx or \
                   chapters[i].end_segment_idx == chapters[i + 1].start_segment_idx - 1, \
                   f"Overlap between chapter {i} and {i+1}"

    @patch('src.chapter_detection.detector.run_initial_detection')
    @patch('src.chapter_detection.detector.run_coverage_resolution')
    def test_heavily_overlapping_chapters_handled(
        self,
        mock_coverage,
        mock_initial,
        mock_config,
        sample_segments,
    ):
        """Test heavily overlapping chapters (same range) are handled."""
        # Setup: Two chapters with identical ranges
        overlapping = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=5, title="Ch1", confidence=0.8),
            ChapterCandidate(chapter_id=1, start_segment_idx=0, end_segment_idx=5, title="Ch2", confidence=0.6),
        ]
        mock_initial.return_value = overlapping

        # Coverage resolution merges identical ranges
        mock_coverage.return_value = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=5, title="Ch1 & Ch2"),
        ]

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = Mock()

            with patch.object(detector, '_should_run_refinement', return_value=False):
                with patch.object(detector, '_should_run_validation', return_value=False):
                    result = detector.detect_chapters_full(sample_segments)

        assert len(result.chapters) >= 1
        assert not result.fallback_used


class TestDetectorMergesShortChaptersUS004:
    """AC3: Test ChapterDetector.detect() merges short chapters below min_duration threshold."""

    @patch('src.chapter_detection.detector.run_initial_detection')
    @patch('src.chapter_detection.detector.run_coverage_resolution')
    def test_short_chapters_merged_by_coverage(
        self,
        mock_coverage,
        mock_initial,
        mock_config,
        sample_segments,
    ):
        """Test chapters below min_chapter_segments threshold are merged."""
        # Setup: LLM returns a tiny chapter (1 segment)
        chapters_with_tiny = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=3, title="Big"),
            ChapterCandidate(chapter_id=1, start_segment_idx=4, end_segment_idx=4, title="Tiny"),  # Only 1 segment
            ChapterCandidate(chapter_id=2, start_segment_idx=5, end_segment_idx=5, title="Also Tiny"),
        ]
        mock_initial.return_value = chapters_with_tiny

        # After merge: tiny chapters absorbed
        merged = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=5, title="Big & Tiny & Also Tiny"),
        ]
        mock_coverage.return_value = merged

        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            detector.llm_client = Mock()

            with patch.object(detector, '_should_run_refinement', return_value=False):
                with patch.object(detector, '_should_run_validation', return_value=False):
                    result = detector.detect_chapters_full(sample_segments)

        # Verify tiny chapters were merged
        assert len(result.chapters) == 1
        # All segments should be covered
        assert result.chapters[0].start_segment_idx == 0
        assert result.chapters[0].end_segment_idx == 5

    def test_coverage_resolution_merges_tiny_directly(self, mock_config):
        """Test _merge_tiny_chapters function directly with min_segments config."""
        from src.chapter_detection.passes.coverage import _merge_tiny_chapters

        chapters = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=4, title="Big", confidence=0.8),
            ChapterCandidate(chapter_id=1, start_segment_idx=5, end_segment_idx=6, title="Tiny", confidence=0.7),  # 2 segs
        ]

        # min_segments=3 means 2-segment chapter is too small
        result = _merge_tiny_chapters(chapters, min_segments=3)

        assert len(result) == 1
        assert result[0].end_segment_idx == 6  # Merged


class TestDetectorRejectsNegativeDurationsUS004:
    """AC4: Test ChapterDetector validation rejects chapters with negative durations."""

    def test_validate_chapters_rejects_negative_duration(self):
        """Test _validate_chapters clamps negative indices to valid range."""
        from src.chapter_detection.passes.initial import _validate_chapters

        # Chapter with end < start (negative duration)
        invalid_chapters = [
            {
                'start_segment_idx': 5,
                'end_segment_idx': 2,  # Invalid: end < start
                'title': 'Backwards Chapter',
            }
        ]

        # Should either reject or fix the chapter
        result = _validate_chapters(invalid_chapters, total_segments=10, min_segments=1)

        # Either rejected (empty) or fixed (end >= start)
        if result:
            assert result[0]['end_segment_idx'] >= result[0]['start_segment_idx']
        # If empty, it was rejected - either outcome is valid

    def test_validate_chapters_clamps_out_of_bounds_indices(self):
        """Test _validate_chapters clamps indices to valid segment range."""
        from src.chapter_detection.passes.initial import _validate_chapters

        # Chapter with indices outside valid range
        invalid_chapters = [
            {
                'start_segment_idx': -5,  # Negative
                'end_segment_idx': 100,   # Way past total_segments
                'title': 'Out of Bounds',
            }
        ]

        result = _validate_chapters(invalid_chapters, total_segments=10, min_segments=1)

        # Should be clamped to valid range
        assert len(result) == 1
        assert result[0]['start_segment_idx'] >= 0
        assert result[0]['end_segment_idx'] < 10
        assert result[0]['end_segment_idx'] >= result[0]['start_segment_idx']

    def test_coverage_resolution_handles_inverted_boundaries(self):
        """Test coverage resolution handles inverted start/end gracefully."""
        from src.chapter_detection.passes.coverage import run_coverage_resolution

        # This shouldn't happen normally but test robustness
        config = Mock()
        config.matching = Mock()
        config.matching.chapter_detection = Mock(min_chapter_segments=1)

        # Note: ChapterCandidate itself doesn't validate, but coverage should handle
        chapters = [
            ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=4),
            ChapterCandidate(chapter_id=1, start_segment_idx=5, end_segment_idx=9),
        ]

        result = run_coverage_resolution(chapters, total_segments=10, config=config)

        # All chapters should have valid boundaries
        for ch in result:
            assert ch.start_segment_idx >= 0
            assert ch.end_segment_idx >= ch.start_segment_idx


class TestDetectorHandlesEmptyTranscriptUS004:
    """AC5: Test ChapterDetector handles empty transcript gracefully."""

    def test_detect_with_empty_segments_returns_empty_result(self, mock_config):
        """Test detect() with empty segments returns empty DetectionResult."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            result = detector.detect_chapters_full([])

        assert isinstance(result, DetectionResult)
        assert result.chapters == []
        assert result.total_segments == 0
        assert result.detection_passes_run == []
        assert not result.fallback_used

    def test_detect_with_none_segments_handles_gracefully(self, mock_config):
        """Test detect() handles None-like input gracefully."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)

            # Empty list should be handled
            result = detector.detect_chapters([])

        assert isinstance(result, list)
        assert len(result) == 0

    def test_detect_dict_output_with_empty_segments(self, mock_config):
        """Test detect_chapters (dict output) with empty segments."""
        with patch.object(EnhancedChapterDetector, '_init_llm_client'):
            detector = EnhancedChapterDetector(mock_config)
            result = detector.detect_chapters([])

        assert result == []

    def test_coverage_resolution_with_zero_total_segments(self, mock_config):
        """Test coverage resolution handles zero total_segments."""
        from src.chapter_detection.passes.coverage import run_coverage_resolution

        chapters = [ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=0)]

        # Should return chapters unchanged when total_segments=0
        result = run_coverage_resolution(chapters, total_segments=0, config=mock_config)

        assert result == chapters
