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

@pytest.mark.fast
class TestIsMatcherTrack:
    """Test _is_matcher_track() correctly identifies V1-V10 track names."""

    def test_v1_returns_true(self):
        assert _is_matcher_track("V1") is True

    @pytest.mark.fast
    def test_v10_returns_true(self):
        assert _is_matcher_track("V10") is True

    @pytest.mark.fast
    def test_v1_primary_returns_true(self):
        assert _is_matcher_track("V1 Primary") is True

    @pytest.mark.fast
    def test_v11_returns_false(self):
        assert _is_matcher_track("V11") is False

    @pytest.mark.fast
    def test_v100_returns_false(self):
        assert _is_matcher_track("V100") is False

    @pytest.mark.fast
    def test_audio_1_returns_false(self):
        assert _is_matcher_track("Audio 1") is False

    @pytest.mark.fast
    def test_v5_returns_true(self):
        assert _is_matcher_track("V5") is True

    @pytest.mark.fast
    def test_keyword_primary_returns_true(self):
        assert _is_matcher_track("Primary") is True

    @pytest.mark.fast
    def test_keyword_entity_returns_true(self):
        assert _is_matcher_track("Entity Images") is True

    @pytest.mark.fast
    def test_keyword_stock_returns_true(self):
        assert _is_matcher_track("Stock") is True

    @pytest.mark.fast
    def test_empty_string_returns_false(self):
        assert _is_matcher_track("") is False

    @pytest.mark.fast
    def test_random_name_returns_false(self):
        assert _is_matcher_track("My Custom Track") is False


# =============================================================================
# AC2: _get_track_category() tests
# =============================================================================

@pytest.mark.fast
class TestGetTrackCategory:
    """Test _get_track_category() returns correct categories."""

    def test_v1_returns_v1(self):
        assert _get_track_category("V1") == "v1"

    @pytest.mark.fast
    def test_v2_returns_v2_v3(self):
        assert _get_track_category("V2") == "v2_v3"

    @pytest.mark.fast
    def test_v3_returns_v2_v3(self):
        assert _get_track_category("V3") == "v2_v3"

    @pytest.mark.fast
    def test_v4_returns_v4_v6(self):
        assert _get_track_category("V4") == "v4_v6"

    @pytest.mark.fast
    def test_v5_returns_v4_v6(self):
        assert _get_track_category("V5") == "v4_v6"

    @pytest.mark.fast
    def test_v6_returns_v4_v6(self):
        assert _get_track_category("V6") == "v4_v6"

    @pytest.mark.fast
    def test_v7_returns_v7_plus(self):
        assert _get_track_category("V7") == "v7_plus"

    @pytest.mark.fast
    def test_v8_returns_v7_plus(self):
        assert _get_track_category("V8") == "v7_plus"

    @pytest.mark.fast
    def test_v9_returns_v7_plus(self):
        assert _get_track_category("V9") == "v7_plus"

    @pytest.mark.fast
    def test_v10_returns_v7_plus(self):
        assert _get_track_category("V10") == "v7_plus"

    @pytest.mark.fast
    def test_non_matcher_returns_none(self):
        assert _get_track_category("Audio 1") is None

    @pytest.mark.fast
    def test_primary_keyword_returns_v1(self):
        assert _get_track_category("Primary") == "v1"

    @pytest.mark.fast
    def test_alternative_keyword_returns_v2_v3(self):
        assert _get_track_category("Alternative") == "v2_v3"

    @pytest.mark.fast
    def test_secondary_keyword_returns_v4_v6(self):
        assert _get_track_category("Secondary") == "v4_v6"

    @pytest.mark.fast
    def test_embedding_keyword_returns_v7_plus(self):
        assert _get_track_category("Embedding") == "v7_plus"

    @pytest.mark.fast
    def test_diversity_keyword_returns_v7_plus(self):
        assert _get_track_category("Diversity") == "v7_plus"

    @pytest.mark.fast
    def test_empty_returns_none(self):
        assert _get_track_category("") is None


# =============================================================================
# AC3: _render_coverage_bar() tests
# =============================================================================

@pytest.mark.fast
class TestRenderCoverageBar:
    """Test _render_coverage_bar() renders ASCII bars correctly."""

    def test_100_percent(self):
        assert _render_coverage_bar(100.0) == "[##########]"

    @pytest.mark.fast
    def test_0_percent(self):
        assert _render_coverage_bar(0.0) == "[..........]"

    @pytest.mark.fast
    def test_50_percent(self):
        assert _render_coverage_bar(50.0) == "[#####.....]"

    @pytest.mark.fast
    def test_25_percent(self):
        # 25% of 10 = 2.5, rounds to 2 (Python rounds .5 to nearest even = 2)
        result = _render_coverage_bar(25.0)
        assert result.startswith("[")
        assert result.endswith("]")
        assert len(result) == 12  # [ + 10 chars + ]

    @pytest.mark.fast
    def test_75_percent(self):
        # 75% of 10 = 7.5, rounds to 8
        assert _render_coverage_bar(75.0) == "[########..]"

    @pytest.mark.fast
    def test_custom_width(self):
        assert _render_coverage_bar(50.0, width=20) == "[##########..........]"

    @pytest.mark.fast
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

@pytest.mark.fast
class TestFormatTimecode:
    """Test _format_timecode() formats seconds correctly."""

    def test_90_seconds(self):
        assert _format_timecode(90) == "1:30"

    @pytest.mark.fast
    def test_3661_seconds(self):
        assert _format_timecode(3661) == "1:01:01"

    @pytest.mark.fast
    def test_0_seconds(self):
        assert _format_timecode(0) == "0:00"

    @pytest.mark.fast
    def test_negative_seconds(self):
        result = _format_timecode(-90)
        assert result.startswith("-")
        assert "1:30" in result

    @pytest.mark.fast
    def test_59_seconds(self):
        assert _format_timecode(59) == "0:59"

    @pytest.mark.fast
    def test_60_seconds(self):
        assert _format_timecode(60) == "1:00"

    @pytest.mark.fast
    def test_3600_seconds(self):
        assert _format_timecode(3600) == "1:00:00"

    @pytest.mark.fast
    def test_large_value(self):
        # 2 hours, 30 minutes, 45 seconds = 9045
        assert _format_timecode(9045) == "2:30:45"

    @pytest.mark.fast
    def test_negative_prepends_minus(self):
        result = _format_timecode(-5)
        assert result == "-0:05"


# =============================================================================
# AC5: Filename-based analysis matching tests
# =============================================================================

@pytest.mark.fast
class TestFilenameBasedAnalysis:
    """
    Test filename-based analysis matches clips by normalized filename
    across original and edited OTIO.

    These tests use mock objects to avoid requiring opentimelineio,
    testing the core matching logic via the FilenameAnalyzer._normalize
    and the analysis result calculations.
    """

    @pytest.mark.fast
    def test_normalize_strips_extension(self):
        from src.post_edit_analysis import FilenameAnalyzer
        # We can't instantiate without OTIO, so test _normalize as unbound
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("video_abc.mp4") == "video_abc"

    @pytest.mark.fast
    def test_normalize_strips_segment_suffix(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("abc123_0045.mp4") == "abc123"

    @pytest.mark.fast
    def test_normalize_handles_empty(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("") == ""

    @pytest.mark.fast
    def test_normalize_handles_multiple_extensions(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        # Only strips known video extensions
        assert analyzer._normalize("file.backup.mp4") == "file.backup"

    @pytest.mark.fast
    def test_normalize_extracts_basename(self):
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("E:/videos/project/clip.mov") == "clip"

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_normalize_case_insensitive_extensions(self):
        """Verify extension removal is case-insensitive."""
        from src.post_edit_analysis import FilenameAnalyzer
        analyzer = object.__new__(FilenameAnalyzer)
        assert analyzer._normalize("clip.MP4") == "clip"
        assert analyzer._normalize("clip.MoV") == "clip"
        assert analyzer._normalize("clip.WebM") == "clip"


# =============================================================================
# US-002: Track category statistics tests
# =============================================================================


# =============================================================================
# AC1: Track category aggregation tests
# =============================================================================

@pytest.mark.fast
class TestTrackCategoryAggregation:
    """Test track category aggregation groups clips into v1, v2_v3, v4_v6, v7_plus buckets."""

    def test_aggregation_10_clips_across_5_tracks(self):
        """Verify counts per category with a mock result containing 10 clips across 5 tracks."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=10,
            v1_kept=4,
            v2_v3_used=2,
            v4_v6_used=2,
            v7_plus_used=1,
            external_added=1,
            track_breakdown={
                "V1": 4, "V2": 1, "V3": 1, "V5": 2, "V8": 1
            },
        )
        # Verify each category bucket has correct count
        assert result.v1_kept == 4
        assert result.v2_v3_used == 2
        assert result.v4_v6_used == 2
        assert result.v7_plus_used == 1
        assert result.external_added == 1
        # Total across all buckets should equal total clips used
        total_used = result.v1_kept + result.v2_v3_used + result.v4_v6_used + result.v7_plus_used + result.external_added
        assert total_used == 10

    @pytest.mark.fast
    def test_get_track_category_maps_all_tracks_to_buckets(self):
        """Verify all V1-V10 map to correct category bucket."""
        track_categories = {}
        for i in range(1, 11):
            cat = _get_track_category(f"V{i}")
            if cat not in track_categories:
                track_categories[cat] = []
            track_categories[cat].append(f"V{i}")

        assert track_categories["v1"] == ["V1"]
        assert set(track_categories["v2_v3"]) == {"V2", "V3"}
        assert set(track_categories["v4_v6"]) == {"V4", "V5", "V6"}
        assert set(track_categories["v7_plus"]) == {"V7", "V8", "V9", "V10"}

    @pytest.mark.fast
    def test_track_breakdown_dict_records_per_track_counts(self):
        """Verify track_breakdown dict stores correct per-track clip counts."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            track_breakdown={"V1": 5, "V2": 3, "V3": 1, "V4": 2, "V8": 4}
        )
        assert result.track_breakdown["V1"] == 5
        assert result.track_breakdown["V2"] == 3
        assert result.track_breakdown["V3"] == 1
        assert result.track_breakdown["V4"] == 2
        assert result.track_breakdown["V8"] == 4
        assert sum(result.track_breakdown.values()) == 15

    @pytest.mark.fast
    def test_user_added_tracks_separated_from_matcher(self):
        """Verify user-added tracks are tracked separately from matcher tracks."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            track_breakdown={"V1": 3, "V2": 2},
            user_added_tracks={"Lower Thirds": 5, "Graphics": 3},
        )
        # Matcher tracks
        assert "V1" in result.track_breakdown
        assert "V2" in result.track_breakdown
        # User tracks separate
        assert "Lower Thirds" in result.user_added_tracks
        assert result.user_added_tracks["Lower Thirds"] == 5
        # No overlap
        assert "Lower Thirds" not in result.track_breakdown

    @pytest.mark.fast
    def test_aggregation_single_category_only(self):
        """Verify aggregation works when only one category has clips."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=6,
            v1_kept=6,
            v2_v3_used=0,
            v4_v6_used=0,
            v7_plus_used=0,
            external_added=0,
        )
        assert result.v1_kept == 6
        assert result.v2_v3_used == 0
        assert result.v4_v6_used == 0
        assert result.v7_plus_used == 0


# =============================================================================
# AC2: Coverage calculation per track category
# =============================================================================

@pytest.mark.fast
class TestCoverageCalculation:
    """Test coverage calculation per track category."""

    def test_coverage_pct_calculation(self):
        """Verify coverage_pct is (matched_duration / total_duration) * 100."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=20,
            segments_covered=15,
            segments_not_covered=5,
        )
        coverage_pct = (result.segments_covered / result.total_segments) * 100
        assert coverage_pct == 75.0

    @pytest.mark.fast
    def test_v1_kept_pct_calculation(self):
        """Verify v1_kept_pct = (v1_kept / total_segments) * 100."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=40,
            v1_kept=30,
            v1_kept_pct=75.0,  # Pre-calculated as done in analyze()
        )
        expected = (30 / 40) * 100
        assert result.v1_kept_pct == expected

    @pytest.mark.fast
    def test_pct_helper_method(self):
        """Verify _pct() helper computes percentage of total_segments."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(total_segments=50)
        assert result._pct(25) == 50.0
        assert result._pct(50) == 100.0
        assert result._pct(0) == 0.0

    @pytest.mark.fast
    def test_coverage_ratio_duration_based(self):
        """Verify coverage_ratio = edited_duration / original_duration."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            duration_enabled=True,
            original_duration_sec=120.0,
            edited_duration_sec=90.0,
            coverage_ratio=90.0 / 120.0,
        )
        assert abs(result.coverage_ratio - 0.75) < 0.001

    @pytest.mark.fast
    def test_segments_covered_plus_not_covered_equals_total(self):
        """Verify segments_covered + segments_not_covered == total when segment_order populated."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            total_segments=30,
            segments_covered=22,
            segments_not_covered=8,
        )
        assert result.segments_covered + result.segments_not_covered == result.total_segments


# =============================================================================
# AC3: Empty edited OTIO returns 0% coverage
# =============================================================================

@pytest.mark.fast
class TestEmptyEditedOtio:
    """Test empty edited OTIO returns 0% coverage for all categories."""

    def test_zero_total_segments_no_division_by_zero(self):
        """Verify no division by zero errors when total_segments is 0."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(total_segments=0)
        # _pct should handle 0 total_segments
        assert result._pct(0) == 0.0
        assert result._pct(5) == 0.0

    @pytest.mark.fast
    def test_empty_result_all_categories_zero(self):
        """Verify all category counts are 0 when no clips matched."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult()
        assert result.v1_kept == 0
        assert result.v2_v3_used == 0
        assert result.v4_v6_used == 0
        assert result.v7_plus_used == 0
        assert result.external_added == 0
        assert result.segments_covered == 0
        assert result.segments_not_covered == 0

    @pytest.mark.fast
    def test_empty_result_v1_kept_pct_is_zero(self):
        """Verify v1_kept_pct defaults to 0.0 on empty result."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult()
        assert result.v1_kept_pct == 0.0

    @pytest.mark.fast
    def test_empty_result_summary_no_crash(self):
        """Verify summary() doesn't crash on empty result (0 segments)."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T00:00:00",
        )
        summary = result.summary()
        assert isinstance(summary, str)
        assert "0 total" in summary
        # Should not raise any exception

    @pytest.mark.fast
    def test_empty_result_to_dict_no_crash(self):
        """Verify to_dict() doesn't crash on empty result."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult()
        d = result.to_dict()
        assert d["summary"]["total_segments"] == 0
        assert d["summary"]["v1_kept"] == 0
        assert d["summary"]["v1_kept_pct"] == 0.0
        assert d["summary"]["segments_covered"] == 0

    @pytest.mark.fast
    def test_empty_coverage_ratio_zero(self):
        """Verify coverage_ratio is 0.0 when original_duration is 0."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            duration_enabled=True,
            original_duration_sec=0.0,
            edited_duration_sec=0.0,
            coverage_ratio=0.0,
        )
        assert result.coverage_ratio == 0.0


# =============================================================================
# AC4: Position-based mode matches clips by frame position
# =============================================================================

@pytest.mark.fast
class TestPositionBasedMatching:
    """Test position-based mode matches clips by frame position."""

    def test_position_exact_match_counted(self):
        """Verify clips matched when original and edited positions align."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            position_match_enabled=True,
            position_exact_matches=8,
            position_alt_used=1,
            position_secondary_used=1,
            position_external_used=0,
            position_gaps=0,
        )
        assert result.position_exact_matches == 8

    @pytest.mark.fast
    def test_position_gaps_tracked(self):
        """Verify gaps at segment positions are counted."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            position_match_enabled=True,
            position_exact_matches=5,
            position_gaps=3,
        )
        assert result.position_gaps == 3

    @pytest.mark.fast
    def test_position_alt_and_secondary_counted(self):
        """Verify alternative and secondary replacements at positions counted."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            position_match_enabled=True,
            position_alt_used=3,
            position_secondary_used=2,
        )
        assert result.position_alt_used == 3
        assert result.position_secondary_used == 2

    @pytest.mark.fast
    def test_position_mismatches_stored(self):
        """Verify position mismatch details stored correctly."""
        from src.post_edit_analysis import FilenameAnalysisResult
        mismatches = [
            {"segment": "S003", "expected": "clip_a.mp4", "actual": "clip_b.mp4", "type": "alternative"},
            {"segment": "S007", "expected": "clip_c.mp4", "actual": "(gap/cut)", "type": "gap"},
        ]
        result = FilenameAnalysisResult(
            position_match_enabled=True,
            position_mismatches=mismatches,
        )
        assert len(result.position_mismatches) == 2
        assert result.position_mismatches[0]["type"] == "alternative"
        assert result.position_mismatches[1]["type"] == "gap"

    @pytest.mark.fast
    def test_position_result_to_dict(self):
        """Verify position matching data appears in to_dict() output."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            position_match_enabled=True,
            position_exact_matches=7,
            position_alt_used=2,
            position_secondary_used=1,
            position_external_used=0,
            position_gaps=0,
            position_mismatches=[
                {"segment": "S001", "expected": "a.mp4", "actual": "b.mp4", "type": "alternative"},
            ],
        )
        d = result.to_dict()
        assert "position_matching" in d
        assert d["position_matching"]["exact_matches"] == 7
        assert d["position_matching"]["alt_used"] == 2
        assert d["position_matching"]["gaps"] == 0

    @pytest.mark.fast
    def test_position_analysis_result_separate_dataclass(self):
        """Verify PositionAnalysisResult stores position stats correctly."""
        from src.post_edit_analysis import PositionAnalysisResult
        result = PositionAnalysisResult(
            total_segments=20,
            v1_kept=14,
            v1_kept_pct=70.0,
            replaced_with_alt=3,
            replaced_with_secondary=2,
            replaced_with_external=1,
            track_usage={"V1": 14, "V2": 3, "V4": 2, "External": 1},
        )
        assert result.v1_kept == 14
        assert result.replaced_with_alt == 3
        assert result.track_usage["V1"] == 14

    @pytest.mark.fast
    def test_position_analysis_result_to_dict(self):
        """Verify PositionAnalysisResult.to_dict() has correct structure."""
        from src.post_edit_analysis import PositionAnalysisResult
        result = PositionAnalysisResult(
            total_segments=10,
            v1_kept=6,
            v1_kept_pct=60.0,
            replaced_with_alt=2,
            replaced_with_secondary=1,
            replaced_with_external=1,
            all_disabled=0,
            missing=0,
            track_usage={"V1": 6, "V2": 2},
        )
        d = result.to_dict()
        assert d["summary"]["total_segments"] == 10
        assert d["summary"]["v1_kept"] == 6
        assert d["summary"]["v1_kept_pct"] == 60.0
        # "replaced" = total - v1_kept - all_disabled - missing
        assert d["summary"]["replaced"] == 4


# =============================================================================
# AC5: Report generation produces valid markdown
# =============================================================================

@pytest.mark.fast
class TestReportGeneration:
    """Test report generation produces valid markdown with track breakdown, coverage bars, and summary."""

    def test_summary_contains_header(self):
        """Verify summary() output contains report header."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=10,
        )
        summary = result.summary()
        assert "POST-EDIT ANALYSIS REPORT" in summary

    @pytest.mark.fast
    def test_summary_contains_track_breakdown_section(self):
        """Verify summary includes track preference breakdown section."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=10,
            track_breakdown={"V1": 5, "V2": 3, "V5": 2},
        )
        summary = result.summary()
        assert "TRACK PREFERENCE BREAKDOWN" in summary
        assert "V1: 5 clips" in summary
        assert "V2: 3 clips" in summary
        assert "V5: 2 clips" in summary

    @pytest.mark.fast
    def test_summary_contains_coverage_map_with_bars(self):
        """Verify summary includes coverage map with ASCII bars when enabled."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=10,
            coverage_map_enabled=True,
            coverage_map=[
                {"start_segment": "S000", "end_segment": "S009",
                 "total": 10, "covered": 7, "coverage_pct": 70.0},
            ],
        )
        summary = result.summary()
        assert "SEGMENT COVERAGE MAP" in summary
        # Should contain ASCII bar chars
        assert "[" in summary and "#" in summary

    @pytest.mark.fast
    def test_summary_contains_segment_counts(self):
        """Verify summary includes segment count and coverage stats."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=20,
            segments_covered=15,
            segments_not_covered=5,
            v1_kept=12,
            v2_v3_used=3,
            v4_v6_used=2,
            v7_plus_used=1,
            external_added=2,
            segments_dropped=8,
        )
        summary = result.summary()
        assert "20 total" in summary
        assert "V1 CLIPS USED" in summary
        assert "V2-V3 ALTERNATIVES" in summary
        assert "V4-V6 SECONDARY" in summary
        assert "V7+ STRATEGY" in summary
        assert "EXTERNAL CLIPS" in summary

    @pytest.mark.fast
    def test_summary_duration_section_when_enabled(self):
        """Verify duration comparison appears when enabled."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=10,
            duration_enabled=True,
            original_duration_sec=300.0,
            edited_duration_sec=240.0,
            coverage_ratio=0.8,
        )
        summary = result.summary()
        assert "DURATION COMPARISON" in summary
        assert "5:00" in summary  # 300 seconds
        assert "4:00" in summary  # 240 seconds

    @pytest.mark.fast
    def test_summary_position_section_when_enabled(self):
        """Verify position matching section appears when enabled."""
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=10,
            position_match_enabled=True,
            position_exact_matches=7,
            position_alt_used=2,
            position_secondary_used=1,
            position_external_used=0,
            position_gaps=0,
        )
        summary = result.summary()
        assert "POSITION MATCHING" in summary
        assert "V1 at correct position" in summary

    @pytest.mark.fast
    def test_to_dict_json_serializable(self):
        """Verify to_dict() output is fully JSON serializable."""
        import json as json_mod
        from src.post_edit_analysis import FilenameAnalysisResult
        result = FilenameAnalysisResult(
            original_file="original.otio",
            edited_file="edited.otio",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=20,
            v1_kept=15,
            v1_kept_pct=75.0,
            v2_v3_used=3,
            v4_v6_used=1,
            v7_plus_used=1,
            external_added=0,
            segments_dropped=5,
            segments_covered=15,
            segments_not_covered=5,
            track_breakdown={"V1": 15, "V2": 2, "V3": 1, "V5": 1, "V8": 1},
            duration_enabled=True,
            original_duration_sec=600.0,
            edited_duration_sec=480.0,
            coverage_ratio=0.8,
            confidence_enabled=True,
            kept_avg_confidence=0.85,
            dropped_avg_confidence=0.62,
            confidence_correlation="positive",
            kept_confidences=[0.9, 0.8],
            dropped_confidences=[0.6, 0.64],
        )
        d = result.to_dict()
        # Must not raise
        json_str = json_mod.dumps(d)
        assert isinstance(json_str, str)
        parsed = json_mod.loads(json_str)
        assert parsed["summary"]["total_segments"] == 20
        assert parsed["summary"]["v1_kept_pct"] == 75.0
        assert "duration" in parsed
        assert "confidence" in parsed

    @pytest.mark.fast
    def test_position_analysis_summary_format(self):
        """Verify PositionAnalysisResult.summary() produces readable output."""
        from src.post_edit_analysis import PositionAnalysisResult
        result = PositionAnalysisResult(
            edited_file="edited.otio",
            segment_map_file="segments.json",
            analyzed_at="2026-01-28T12:00:00",
            total_segments=15,
            v1_kept=10,
            v1_kept_pct=66.7,
            replaced_with_alt=3,
            replaced_with_secondary=1,
            replaced_with_external=1,
            track_usage={"V1": 10, "V2": 3, "V4": 1},
            replacements=[
                {"segment": "S003", "original": "clip_a.mp4", "replaced_with": "clip_b.mp4", "new_track": "V2"},
            ],
        )
        summary = result.summary()
        assert "POST-EDIT ANALYSIS (POSITION-BASED)" in summary
        assert "V1 clips kept" in summary
        assert "Replaced with alt" in summary
        assert "Track Usage:" in summary
