"""
Edge Case Tests - Extreme Scenarios and Boundary Conditions

Tests for edge cases, boundary conditions, and error scenarios across
multiple modules to ensure robustness and prevent unexpected failures.

Categories:
1. Empty/Null Inputs
2. Extreme Values (very large, very small, zero)
3. Malformed Data
4. Unicode and Special Characters
5. Concurrent Operations
6. Resource Limits
7. Error Recovery
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import modules under test
from src.utils import SRTSegment, Match, MatchResult
from src.config import Config
from src.state import PipelineState


# =============================================================================
# CATEGORY 1: EMPTY/NULL INPUTS
# =============================================================================

class TestEmptyInputs:
    """Test handling of empty inputs across modules"""

    @pytest.mark.fast
    def test_empty_voiceover_text(self):
        """Test SRTSegment with empty text"""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="",  # Empty text
            source_file="test.srt"
        )

        assert segment.text == ""
        assert segment.duration == 1.0

    @pytest.mark.fast
    def test_empty_keywords_list(self):
        """Test Match with empty keywords"""
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "v.mp4", keywords=[])

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.8,
            reasoning="test"
        )

        assert match.video_segment.keywords == []

    @pytest.mark.fast
    def test_empty_match_result_list(self):
        """Test MatchResult with empty alternatives and secondaries"""
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "v.mp4")
        primary = Match(vo_seg, vid_seg, None, 0.8, "test")

        result = MatchResult(
            primary_match=primary,
            alternatives=[],
            secondary_matches=[],
            strategy_matches=[]
        )

        assert len(result.alternatives) == 0
        assert len(result.secondary_matches) == 0
        assert len(result.strategy_matches) == 0

    @pytest.mark.fast
    def test_pipeline_state_with_no_data(self):
        """Test PipelineState initialization with minimal data"""
        state = PipelineState()

        # Test default values
        assert state.voiceover_segments == []
        assert state.transcripts == {}  # Dict, not list
        assert state.embeddings == []   # List, not None or dict
        assert state.matches == []
        assert state.downloaded_videos == []


# =============================================================================
# CATEGORY 2: EXTREME VALUES
# =============================================================================

class TestExtremeValues:
    """Test handling of extreme values"""

    @pytest.mark.fast
    def test_very_long_text(self):
        """Test SRTSegment with extremely long text (10,000 chars)"""
        long_text = "A" * 10000
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=100.0,
            text=long_text,
            source_file="test.srt"
        )

        assert len(segment.text) == 10000
        assert segment.text == long_text

    @pytest.mark.fast
    def test_very_short_duration(self):
        """Test segment with extremely short duration (0.001 seconds)"""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=0.001,
            text="Quick",
            source_file="test.srt"
        )

        assert segment.duration == 0.001

    @pytest.mark.fast
    def test_very_long_duration(self):
        """Test segment with very long duration (10 hours)"""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=36000.0,  # 10 hours
            text="Long segment",
            source_file="test.srt"
        )

        assert segment.duration == 36000.0

    @pytest.mark.fast
    def test_zero_confidence_match(self):
        """Test Match with zero confidence"""
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "v.mp4")

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.0,  # Zero confidence
            reasoning="No match found"
        )

        assert match.confidence == 0.0

    @pytest.mark.fast
    def test_very_high_segment_index(self):
        """Test SRTSegment with very high index (10000)"""
        segment = SRTSegment(
            index=10000,
            start_time=0.0,
            end_time=1.0,
            text="High index",
            source_file="test.srt"
        )

        assert segment.index == 10000

    @pytest.mark.fast
    def test_large_number_of_keywords(self):
        """Test segment with 1000 keywords"""
        keywords = [f"keyword_{i}" for i in range(1000)]
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test",
            source_file="test.srt",
            keywords=keywords
        )

        assert len(segment.keywords) == 1000


# =============================================================================
# CATEGORY 3: MALFORMED DATA
# =============================================================================

class TestMalformedData:
    """Test handling of malformed or invalid data"""

    @pytest.mark.fast
    def test_negative_duration(self):
        """Test SRTSegment with negative duration (end < start)"""
        segment = SRTSegment(
            index=0,
            start_time=5.0,
            end_time=3.0,  # End before start
            text="Invalid",
            source_file="test.srt"
        )

        # Duration should be negative (catches bug)
        assert segment.duration == -2.0

    @pytest.mark.fast
    def test_confidence_above_one(self):
        """Test Match with confidence > 1.0 (invalid but possible)"""
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "v.mp4")

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=1.5,  # > 1.0
            reasoning="test"
        )

        # Should accept but may need validation
        assert match.confidence == 1.5

    @pytest.mark.fast
    def test_negative_confidence(self):
        """Test Match with negative confidence"""
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "v.mp4")

        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=-0.5,  # Negative
            reasoning="test"
        )

        assert match.confidence == -0.5

    @pytest.mark.fast
    def test_segment_with_none_source_file(self):
        """Test SRTSegment with None as source_file"""
        # This should work since source_file has default ""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test"
        )

        assert segment.source_file == ""


# =============================================================================
# CATEGORY 4: UNICODE AND SPECIAL CHARACTERS
# =============================================================================

class TestUnicodeAndSpecialChars:
    """Test handling of unicode and special characters"""

    @pytest.mark.fast
    def test_unicode_text(self):
        """Test SRTSegment with unicode text"""
        unicode_text = "日本語テキスト 中文文本 العربية Ελληνικά 🎬🎥📹"
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text=unicode_text,
            source_file="test.srt"
        )

        assert segment.text == unicode_text

    @pytest.mark.fast
    def test_special_xml_characters(self):
        """Test text with XML special characters"""
        special_text = "Text with <tags> & \"quotes\" and 'apostrophes'"
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text=special_text,
            source_file="test.srt"
        )

        assert segment.text == special_text

    @pytest.mark.fast
    def test_emoji_in_text(self):
        """Test segment with emojis"""
        emoji_text = "Video about 🎬 movies 🎥 and 📹 cameras"
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text=emoji_text,
            source_file="test.srt"
        )

        assert "🎬" in segment.text
        assert "🎥" in segment.text

    @pytest.mark.fast
    def test_newlines_and_tabs_in_text(self):
        """Test segment with newlines and tabs"""
        text_with_whitespace = "Line 1\nLine 2\tTabbed"
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text=text_with_whitespace,
            source_file="test.srt"
        )

        assert "\n" in segment.text
        assert "\t" in segment.text

    @pytest.mark.fast
    def test_path_with_unicode(self):
        """Test file path with unicode characters"""
        unicode_path = "/videos/日本語/ビデオ.mp4"
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test",
            source_file=unicode_path
        )

        assert segment.source_file == unicode_path


# =============================================================================
# CATEGORY 5: BOUNDARY CONDITIONS
# =============================================================================

class TestBoundaryConditions:
    """Test boundary conditions at limits"""

    @pytest.mark.fast
    def test_segment_at_zero_time(self):
        """Test segment starting at exactly 0.0"""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="Start",
            source_file="test.srt"
        )

        assert segment.start_time == 0.0

    @pytest.mark.fast
    def test_exact_frame_boundary(self):
        """Test segment duration that's exactly one frame at 30fps"""
        frame_duration = 1.0 / 30.0  # 0.0333... seconds
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=frame_duration,
            text="One frame",
            source_file="test.srt"
        )

        assert abs(segment.duration - frame_duration) < 0.0001

    @pytest.mark.fast
    def test_confidence_at_boundaries(self):
        """Test Match with confidence at 0.0 and 1.0 boundaries"""
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "v.mp4")

        match_zero = Match(vo_seg, vid_seg, None, 0.0, "min")
        match_one = Match(vo_seg, vid_seg, None, 1.0, "max")

        assert match_zero.confidence == 0.0
        assert match_one.confidence == 1.0

    @pytest.mark.fast
    def test_single_character_text(self):
        """Test segment with single character"""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="A",
            source_file="test.srt"
        )

        assert len(segment.text) == 1


# =============================================================================
# CATEGORY 6: DATA TYPE EDGE CASES
# =============================================================================

class TestDataTypeEdgeCases:
    """Test edge cases related to data types"""

    @pytest.mark.fast
    def test_float_precision_in_timestamps(self):
        """Test that float precision is maintained"""
        precise_time = 1.123456789
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=precise_time,
            text="Precise",
            source_file="test.srt"
        )

        # Python floats have ~15 decimal digits of precision
        assert abs(segment.end_time - precise_time) < 1e-10

    @pytest.mark.fast
    def test_integer_as_float_time(self):
        """Test that integer times work (implicit conversion)"""
        segment = SRTSegment(
            index=0,
            start_time=0,  # Integer
            end_time=5,    # Integer
            text="Integer times",
            source_file="test.srt"
        )

        assert segment.start_time == 0.0
        assert segment.end_time == 5.0

    @pytest.mark.fast
    def test_very_large_index(self):
        """Test with maximum safe integer for index"""
        large_index = 2**31 - 1  # Max 32-bit signed int
        segment = SRTSegment(
            index=large_index,
            start_time=0.0,
            end_time=1.0,
            text="Large index",
            source_file="test.srt"
        )

        assert segment.index == large_index


# =============================================================================
# CATEGORY 7: LIST/COLLECTION EDGE CASES
# =============================================================================

class TestCollectionEdgeCases:
    """Test edge cases with lists and collections"""

    @pytest.mark.fast
    def test_single_item_alternatives(self):
        """Test MatchResult with single alternative"""
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "v.mp4")
        primary = Match(vo_seg, vid_seg, None, 0.9, "primary")
        alt = Match(vo_seg, vid_seg, None, 0.7, "alt")

        result = MatchResult(
            primary_match=primary,
            alternatives=[alt],
            secondary_matches=[],
            strategy_matches=[]
        )

        assert len(result.alternatives) == 1

    @pytest.mark.fast
    def test_duplicate_keywords(self):
        """Test segment with duplicate keywords"""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test",
            source_file="test.srt",
            keywords=["word", "word", "word"]  # Duplicates
        )

        assert len(segment.keywords) == 3  # Duplicates preserved

    @pytest.mark.fast
    def test_keywords_with_whitespace(self):
        """Test keywords containing whitespace"""
        segment = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="test",
            source_file="test.srt",
            keywords=["multi word", "  spaces  ", "\ttabs\t"]
        )

        assert "multi word" in segment.keywords


# =============================================================================
# CATEGORY 8: STATE TRANSITIONS
# =============================================================================

class TestStateTransitions:
    """Test edge cases in state transitions"""

    @pytest.mark.fast
    def test_pipeline_state_stage_progression(self):
        """Test PipelineState through all stages"""
        state = PipelineState()

        # Initial state
        assert state.transcripts == {}

        # Add transcripts (dict format)
        state.transcripts["video1"] = [
            {"text": "test", "start": 0.0, "end": 1.0}
        ]
        assert len(state.transcripts) == 1

        # Add matches
        vo_seg = SRTSegment(0, 0.0, 1.0, "test", "vo.srt")
        vid_seg = SRTSegment(0, 0.0, 1.0, "video", "test.mp4")
        match = Match(vo_seg, vid_seg, None, 0.8, "test")
        state.matches = [MatchResult(match, [], [], [])]

        assert len(state.matches) == 1


# =============================================================================
# SUMMARY
# =============================================================================

if __name__ == "__main__":
    # Can run directly for quick validation
    pytest.main([__file__, "-v"])
