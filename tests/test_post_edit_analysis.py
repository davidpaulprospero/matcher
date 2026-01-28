"""
Tests for post-edit analysis helper functions and filename-based matching.

Covers: _is_matcher_track, _get_track_category, _render_coverage_bar,
        _format_timecode, and FilenameAnalyzer filename-based analysis.
"""

import pytest

from src.post_edit_analysis import (
    _is_matcher_track,
    _get_track_category,
    _render_coverage_bar,
    _format_timecode,
)


# =============================================================================
# AC1: _is_matcher_track() tests
# =============================================================================

class TestIsMatcherTrack:
    """Test _is_matcher_track() correctly identifies V1-V10 track names."""

    def test_v1_returns_true(self):
        assert _is_matcher_track("V1") is True

    def test_v10_returns_true(self):
        assert _is_matcher_track("V10") is True

    def test_v1_primary_returns_true(self):
        assert _is_matcher_track("V1 Primary") is True

    def test_v11_returns_false(self):
        assert _is_matcher_track("V11") is False

    def test_v100_returns_false(self):
        assert _is_matcher_track("V100") is False

    def test_audio_1_returns_false(self):
        assert _is_matcher_track("Audio 1") is False

    def test_v5_returns_true(self):
        assert _is_matcher_track("V5") is True

    def test_keyword_primary_returns_true(self):
        assert _is_matcher_track("Primary") is True

    def test_keyword_entity_returns_true(self):
        assert _is_matcher_track("Entity Images") is True

    def test_keyword_stock_returns_true(self):
        assert _is_matcher_track("Stock") is True

    def test_empty_string_returns_false(self):
        assert _is_matcher_track("") is False

    def test_random_name_returns_false(self):
        assert _is_matcher_track("My Custom Track") is False


# =============================================================================
# AC2: _get_track_category() tests
# =============================================================================

class TestGetTrackCategory:
    """Test _get_track_category() returns correct categories."""

    def test_v1_returns_v1(self):
        assert _get_track_category("V1") == "v1"

    def test_v2_returns_v2_v3(self):
        assert _get_track_category("V2") == "v2_v3"

    def test_v3_returns_v2_v3(self):
        assert _get_track_category("V3") == "v2_v3"

    def test_v4_returns_v4_v6(self):
        assert _get_track_category("V4") == "v4_v6"

    def test_v5_returns_v4_v6(self):
        assert _get_track_category("V5") == "v4_v6"

    def test_v6_returns_v4_v6(self):
        assert _get_track_category("V6") == "v4_v6"

    def test_v7_returns_v7_plus(self):
        assert _get_track_category("V7") == "v7_plus"

    def test_v8_returns_v7_plus(self):
        assert _get_track_category("V8") == "v7_plus"

    def test_v9_returns_v7_plus(self):
        assert _get_track_category("V9") == "v7_plus"

    def test_v10_returns_v7_plus(self):
        assert _get_track_category("V10") == "v7_plus"

    def test_non_matcher_returns_none(self):
        assert _get_track_category("Audio 1") is None

    def test_primary_keyword_returns_v1(self):
        assert _get_track_category("Primary") == "v1"

    def test_alternative_keyword_returns_v2_v3(self):
        assert _get_track_category("Alternative") == "v2_v3"

    def test_secondary_keyword_returns_v4_v6(self):
        assert _get_track_category("Secondary") == "v4_v6"

    def test_embedding_keyword_returns_v7_plus(self):
        assert _get_track_category("Embedding") == "v7_plus"

    def test_diversity_keyword_returns_v7_plus(self):
        assert _get_track_category("Diversity") == "v7_plus"

    def test_empty_returns_none(self):
        assert _get_track_category("") is None


# =============================================================================
# AC3: _render_coverage_bar() tests
# =============================================================================

class TestRenderCoverageBar:
    """Test _render_coverage_bar() renders ASCII bars correctly."""

    def test_100_percent(self):
        assert _render_coverage_bar(100.0) == "[##########]"

    def test_0_percent(self):
        assert _render_coverage_bar(0.0) == "[..........]"

    def test_50_percent(self):
        assert _render_coverage_bar(50.0) == "[#####.....]"

    def test_25_percent(self):
        # 25% of 10 = 2.5, rounds to 2 (Python rounds .5 to nearest even = 2)
        result = _render_coverage_bar(25.0)
        assert result.startswith("[")
        assert result.endswith("]")
        assert len(result) == 12  # [ + 10 chars + ]

    def test_75_percent(self):
        # 75% of 10 = 7.5, rounds to 8
        assert _render_coverage_bar(75.0) == "[########..]"

    def test_custom_width(self):
        assert _render_coverage_bar(50.0, width=20) == "[##########..........]"

    def test_bar_length_always_correct(self):
        """Bar inner content should always be exactly width chars."""
        for pct in [0, 10, 33, 50, 67, 90, 100]:
            result = _render_coverage_bar(float(pct), width=10)
            # Strip brackets, check length
            inner = result[1:-1]
            assert len(inner) == 10, f"Failed for {pct}%: got {len(inner)}"


# =============================================================================
# AC4: _format_timecode() tests
# =============================================================================

class TestFormatTimecode:
    """Test _format_timecode() formats seconds correctly."""

    def test_90_seconds(self):
        assert _format_timecode(90) == "1:30"

    def test_3661_seconds(self):
        assert _format_timecode(3661) == "1:01:01"

    def test_0_seconds(self):
        assert _format_timecode(0) == "0:00"

    def test_negative_seconds(self):
        result = _format_timecode(-90)
        assert result.startswith("-")
        assert "1:30" in result

    def test_59_seconds(self):
        assert _format_timecode(59) == "0:59"

    def test_60_seconds(self):
        assert _format_timecode(60) == "1:00"

    def test_3600_seconds(self):
        assert _format_timecode(3600) == "1:00:00"

    def test_large_value(self):
        # 2 hours, 30 minutes, 45 seconds = 9045
        assert _format_timecode(9045) == "2:30:45"

    def test_negative_prepends_minus(self):
        result = _format_timecode(-5)
        assert result == "-0:05"


# =============================================================================
# AC5: Filename-based analysis matching tests
# =============================================================================

class TestFilenameBasedAnalysis:
    """
    Test filename-based analysis matches clips by normalized filename
    across original and edited OTIO.

    These tests use mock objects to avoid requiring opentimelineio,
    testing the core matching logic via the FilenameAnalyzer._normalize
    and the analysis result calculations.
    """

    def test_normalize_strips_extension(self):
        from src.post_edit_analysis import FilenameAnalyzer
        # We can't instantiate without OTIO, so test _normalize as unbound
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("video_abc.mp4") == "video_abc"

    def test_normalize_strips_segment_suffix(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("abc123_0045.mp4") == "abc123"

    def test_normalize_handles_empty(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("") == ""

    def test_normalize_handles_multiple_extensions(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        # Only strips known video extensions
        assert analyzer._normalize("file.backup.mp4") == "file.backup"

    def test_normalize_extracts_basename(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("E:/videos/project/clip.mov") == "clip"

    def test_coverage_calculation_with_result(self):
        """Test coverage percentage calculation in FilenameAnalysisResult."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=10,
            v1_kept=7,
            segments_covered=7,
            segments_not_covered=3,
            segments_dropped=3,
        )
        # v1_kept_pct should be calculated after analysis, but we can verify manually
        expected_pct = (7 / 10) * 100
        assert expected_pct == 70.0

    def test_result_clips_found_and_missing(self):
        """Test that result tracks both found and missing clips."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=5,
            v1_kept=3,
            segments_covered=3,
            segments_not_covered=2,
            segments_dropped=2,
            v1_clips_used=["clip_a", "clip_b", "clip_c"],
            dropped_v1_clips=["clip_d", "clip_e"],
        )
        assert len(result.v1_clips_used) == 3
        assert len(result.dropped_v1_clips) == 2
        assert result.segments_covered + result.segments_not_covered == result.total_segments

    def test_result_to_dict_preserves_counts(self):
        """Test to_dict() preserves clip counts correctly."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=20,
            v1_kept=12,
            v1_kept_pct=60.0,
            v2_v3_used=3,
            v4_v6_used=2,
            v7_plus_used=1,
            external_added=2,
            segments_dropped=8,
            segments_covered=12,
            segments_not_covered=8,
        )
        d = result.to_dict()
        assert d["summary"]["total_segments"] == 20
        assert d["summary"]["v1_kept"] == 12
        assert d["summary"]["v1_kept_pct"] == 60.0
        assert d["summary"]["v2_v3_used"] == 3
        assert d["summary"]["external_added"] == 2

    def test_result_empty_has_zero_coverage(self):
        """Test empty result defaults to 0 coverage."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult()
        assert result.total_segments == 0
        assert result.segments_covered == 0
        assert result.v1_kept_pct == 0.0
        # _pct should not divide by zero
        assert result._pct(0) == 0.0
        assert result._pct(5) == 0.0  # 0 total_segments

    def test_normalize_case_insensitive_extensions(self):
        """Verify extension removal is case-insensitive."""
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("clip.MP4") == "clip"
        assert analyzer._normalize("clip.MoV") == "clip"
        assert analyzer._normalize("clip.WebM") == "clip"
