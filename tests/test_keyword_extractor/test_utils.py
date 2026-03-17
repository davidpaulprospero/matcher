"""
Tests for keyword_extractor.utils

Tests cover:
- extract_keywords_from_srt() SRT file parsing
- extract_keyword_per_segment_from_srt() per-segment extraction
- find_keyword_matches() keyword overlap calculation
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from src.keyword_extractor.utils import (
    extract_keywords_from_srt,
    extract_keyword_per_segment_from_srt,
    find_keyword_matches
)


class TestFindKeywordMatches:
    """Test find_keyword_matches() function"""

    @pytest.mark.fast
    def test_find_keyword_matches_no_overlap(self):
        """Test with no keyword overlap"""
        vo_keywords = ["mountain", "snow", "ice"]
        video_keywords = ["ocean", "waves", "beach"]

        boost, is_keyword_match, is_visual_match = find_keyword_matches(
            vo_keywords, video_keywords
        )

        assert boost == 0.0
        assert is_keyword_match is False
        assert is_visual_match is False

    @pytest.mark.fast
    def test_find_keyword_matches_text_overlap(self):
        """Test with text keyword overlap"""
        vo_keywords = ["mountain", "snow", "ice"]
        video_keywords = ["mountain", "forest", "river"]

        boost, is_keyword_match, is_visual_match = find_keyword_matches(
            vo_keywords, video_keywords,
            keyword_boost=0.05
        )

        assert boost == 0.05  # 1 match * 0.05
        assert is_keyword_match is True
        assert is_visual_match is False

    @pytest.mark.fast
    def test_find_keyword_matches_multiple_text_overlap(self):
        """Test with multiple text keyword overlaps"""
        vo_keywords = ["mountain", "snow", "ice"]
        video_keywords = ["mountain", "snow", "ocean"]

        boost, is_keyword_match, is_visual_match = find_keyword_matches(
            vo_keywords, video_keywords,
            keyword_boost=0.05
        )

        assert boost == 0.10  # 2 matches * 0.05
        assert is_keyword_match is True

    @pytest.mark.fast
    def test_find_keyword_matches_visual_overlap(self):
        """Test with visual keyword overlap"""
        vo_keywords = ["mountain", "snow"]
        video_keywords = ["ocean"]
        visual_keywords = ["mountain", "ice"]

        boost, is_keyword_match, is_visual_match = find_keyword_matches(
            vo_keywords, video_keywords, visual_keywords,
            visual_boost=0.03
        )

        assert boost == 0.03  # 1 visual match * 0.03
        assert is_keyword_match is False
        assert is_visual_match is True

    @pytest.mark.fast
    def test_find_keyword_matches_both_overlap(self):
        """Test with both text and visual keyword overlap"""
        vo_keywords = ["mountain", "snow"]
        video_keywords = ["mountain"]
        visual_keywords = ["snow"]

        boost, is_keyword_match, is_visual_match = find_keyword_matches(
            vo_keywords, video_keywords, visual_keywords,
            keyword_boost=0.05,
            visual_boost=0.03
        )

        assert boost == 0.08  # 0.05 + 0.03
        assert is_keyword_match is True
        assert is_visual_match is True

    @pytest.mark.fast
    def test_find_keyword_matches_max_boost_cap(self):
        """Test boost score is capped at max_boost"""
        vo_keywords = ["kw1", "kw2", "kw3", "kw4", "kw5"]
        video_keywords = ["kw1", "kw2", "kw3", "kw4", "kw5"]

        boost, _, _ = find_keyword_matches(
            vo_keywords, video_keywords,
            keyword_boost=0.05,
            max_boost=0.2
        )

        # 5 matches * 0.05 = 0.25, but capped at 0.2
        assert boost == 0.2

    @pytest.mark.fast
    def test_find_keyword_matches_case_insensitive(self):
        """Test matching is case-insensitive"""
        vo_keywords = ["MOUNTAIN", "Snow"]
        video_keywords = ["mountain", "SNOW"]

        boost, is_keyword_match, _ = find_keyword_matches(
            vo_keywords, video_keywords,
            keyword_boost=0.05
        )

        assert boost == 0.10  # 2 matches
        assert is_keyword_match is True

    @pytest.mark.fast
    def test_find_keyword_matches_empty_keywords(self):
        """Test with empty keyword lists"""
        boost, is_keyword_match, is_visual_match = find_keyword_matches(
            [], [], []
        )

        assert boost == 0.0
        assert is_keyword_match is False
        assert is_visual_match is False

    @pytest.mark.fast
    def test_find_keyword_matches_custom_boosts(self):
        """Test with custom boost values"""
        vo_keywords = ["mountain"]
        video_keywords = ["mountain"]

        boost, _, _ = find_keyword_matches(
            vo_keywords, video_keywords,
            keyword_boost=0.10,  # Custom
            max_boost=0.5
        )

        assert boost == 0.10


class TestExtractKeywordsFromSrt:
    """Test extract_keywords_from_srt() function"""

    @pytest.mark.integration
    def test_extract_keywords_from_srt_basic(self):
        """Test basic SRT file keyword extraction"""
        # Create temporary SRT file
        srt_content = """1
00:00:00,000 --> 00:00:05,000
This is about mountain climbing.

2
00:00:05,000 --> 00:00:10,000
We scaled the highest peaks.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            # Create mock config
            mock_config = Mock()
            mock_config.keyword = Mock()
            mock_config.keyword.max_keywords = 30

            # Mock LLM client
            with patch('src.keyword_extractor.utils.LLMKeywordExtractor') as MockExtractor:
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_keywords = Mock(return_value=Mock(
                    keywords=["mountain climbing", "peaks"],
                    segments_analyzed=2,
                    extraction_method="llm"
                ))
                MockExtractor.return_value = mock_extractor_instance

                result = extract_keywords_from_srt(srt_path, mock_config)

                assert result.keywords == ["mountain climbing", "peaks"]
                assert result.segments_analyzed == 2

        finally:
            os.unlink(srt_path)

    @pytest.mark.integration
    def test_extract_keywords_from_srt_returns_all_keywords(self):
        """Test SRT extraction returns all keywords without limit"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Sample text.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            mock_config = Mock()
            mock_config.keyword = Mock()

            with patch('src.keyword_extractor.utils.LLMKeywordExtractor') as MockExtractor:
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_keywords = Mock(return_value=Mock(
                    keywords=["kw1", "kw2", "kw3"],
                    segments_analyzed=1,
                    extraction_method="llm"
                ))
                MockExtractor.return_value = mock_extractor_instance

                # Call function
                result = extract_keywords_from_srt(srt_path, mock_config)

                # Verify extract_keywords was called
                mock_extractor_instance.extract_keywords.assert_called_once()
                assert result.keywords == ["kw1", "kw2", "kw3"]

        finally:
            os.unlink(srt_path)


class TestExtractKeywordPerSegmentFromSrt:
    """Test extract_keyword_per_segment_from_srt() function"""

    @pytest.mark.integration
    def test_extract_keyword_per_segment_from_srt_basic(self):
        """Test per-segment SRT extraction"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Mountain climbing is dangerous.

2
00:00:05,000 --> 00:00:10,000
Ocean diving requires training.

3
00:00:10,000 --> 00:00:15,000
Forest hiking is peaceful.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            mock_config = Mock()

            with patch('src.keyword_extractor.utils.LLMKeywordExtractor') as MockExtractor:
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_keyword_per_segment = Mock(
                    return_value=["mountain climbing", "ocean diving", "forest hiking"]
                )
                MockExtractor.return_value = mock_extractor_instance

                result = extract_keyword_per_segment_from_srt(srt_path, mock_config)

                assert len(result) == 3
                assert result == ["mountain climbing", "ocean diving", "forest hiking"]

        finally:
            os.unlink(srt_path)

    @pytest.mark.integration
    def test_extract_keyword_per_segment_from_srt_with_topic(self):
        """Test per-segment SRT extraction with explicit topic"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Sample text.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            mock_config = Mock()

            with patch('src.keyword_extractor.utils.LLMKeywordExtractor') as MockExtractor:
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_keyword_per_segment = Mock(
                    return_value=["keyword"]
                )
                MockExtractor.return_value = mock_extractor_instance

                result = extract_keyword_per_segment_from_srt(
                    srt_path, mock_config, topic="Explicit Topic"
                )

                # Verify topic was passed
                mock_extractor_instance.extract_keyword_per_segment.assert_called_once()
                call_kwargs = mock_extractor_instance.extract_keyword_per_segment.call_args[1]
                assert call_kwargs.get('topic') == "Explicit Topic"

        finally:
            os.unlink(srt_path)

    @pytest.mark.integration
    def test_extract_keyword_per_segment_from_srt_topic_detection(self):
        """Test per-segment SRT extraction with topic auto-detection"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
This is about Mountains.

2
00:00:05,000 --> 00:00:10,000
Mountains are tall.

3
00:00:10,000 --> 00:00:15,000
More about Mountains.
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            mock_config = Mock()

            with patch('src.keyword_extractor.utils.LLMKeywordExtractor') as MockExtractor:
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_keyword_per_segment = Mock(
                    return_value=["kw1", "kw2", "kw3"]
                )
                MockExtractor.return_value = mock_extractor_instance

                # Call without explicit topic
                result = extract_keyword_per_segment_from_srt(srt_path, mock_config)

                # Topic should be auto-detected (likely "Mountains")
                call_kwargs = mock_extractor_instance.extract_keyword_per_segment.call_args[1]
                topic = call_kwargs.get('topic', '')
                # Should have detected something from capitalized words
                assert isinstance(topic, str)

        finally:
            os.unlink(srt_path)


class TestUtilsEdgeCases:
    """Test edge cases for utils functions"""

    @pytest.mark.integration
    def test_extract_keywords_from_empty_srt(self):
        """Test extracting keywords from empty SRT file"""
        srt_content = ""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as f:
            f.write(srt_content)
            srt_path = f.name

        try:
            mock_config = Mock()
            mock_config.keyword = Mock()
            mock_config.keyword.max_keywords = 30

            with patch('src.keyword_extractor.utils.LLMKeywordExtractor') as MockExtractor:
                mock_extractor_instance = Mock()
                mock_extractor_instance.extract_keywords = Mock(return_value=Mock(
                    keywords=[],
                    segments_analyzed=0,
                    extraction_method="none"
                ))
                MockExtractor.return_value = mock_extractor_instance

                result = extract_keywords_from_srt(srt_path, mock_config)

                assert result.keywords == []

        finally:
            os.unlink(srt_path)

    @pytest.mark.fast
    def test_find_keyword_matches_with_none_visual(self):
        """Test find_keyword_matches with None visual_keywords"""
        vo_keywords = ["mountain"]
        video_keywords = ["ocean"]

        boost, is_keyword_match, is_visual_match = find_keyword_matches(
            vo_keywords, video_keywords, visual_keywords=None
        )

        assert is_visual_match is False
        assert boost == 0.0
