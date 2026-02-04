"""
Tests for source diversity metrics (US-53-005).

Covers:
- Per-track source concentration ratio computation
- WARNING when any track has >60% clips from one source
- Temporal clustering detection (3+ consecutive same-source)
- Diversity metrics included in stage completion log
- Edge cases: empty results, single segment, all unique sources
"""

import pytest
import logging
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from src.matching.metrics import (
    compute_diversity_metrics,
    log_diversity_metrics,
    _extract_source,
    _get_track_sources,
    _compute_concentration,
    _detect_temporal_clusters,
    TrackDiversityMetrics,
    DiversityReport,
)

pytestmark = pytest.mark.unit


def _make_match_result(
    primary_source: str = "",
    alt_sources: Optional[List[str]] = None,
    secondary_sources: Optional[List[str]] = None,
    strategy_sources: Optional[List[tuple]] = None,
):
    """Create a mock MatchResult with specified sources per track.

    Args:
        primary_source: source_file for V1 primary match.
        alt_sources: source_files for V2-V3 alternatives.
        secondary_sources: source_files for V4-V6 secondary matches.
        strategy_sources: list of (source_file, strategy_name) for V7/V8.
    """
    result = MagicMock()

    # V1 - primary match
    if primary_source:
        pm = MagicMock()
        pm.video_segment = MagicMock()
        pm.video_segment.source_file = primary_source
        result.primary_match = pm
    else:
        result.primary_match = None

    # V2-V3 - alternatives
    alts = []
    for src in (alt_sources or []):
        alt = MagicMock()
        alt.video_segment = MagicMock()
        alt.video_segment.source_file = src
        alts.append(alt)
    result.alternatives = alts

    # V4-V6 - secondary matches
    secs = []
    for src in (secondary_sources or []):
        sec = MagicMock()
        sec.video_segment = MagicMock()
        sec.video_segment.source_file = src
        secs.append(sec)
    result.secondary_matches = secs

    # V7/V8 - strategy matches
    strats = []
    for src, strategy in (strategy_sources or []):
        sm = MagicMock()
        sm.video_segment = MagicMock()
        sm.video_segment.source_file = src
        sm.strategy = strategy
        strats.append(sm)
    result.strategy_matches = strats

    return result


class TestExtractSource:
    """Test _extract_source helper."""

    @pytest.mark.fast
    def test_extracts_stem(self):
        assert _extract_source("videos/my_video.mp4") == "my_video"

    @pytest.mark.fast
    def test_empty_string(self):
        assert _extract_source("") == ""

    @pytest.mark.fast
    def test_video_id(self):
        assert _extract_source("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    @pytest.mark.fast
    def test_truncates_long_name(self):
        long_name = "a" * 50 + ".mp4"
        result = _extract_source(long_name)
        assert len(result) <= 30


class TestComputeConcentration:
    """Test _compute_concentration helper."""

    @pytest.mark.fast
    def test_empty_list(self):
        ratio, src, count, unique, total = _compute_concentration([])
        assert ratio == 0.0
        assert total == 0

    @pytest.mark.fast
    def test_all_same(self):
        ratio, src, count, unique, total = _compute_concentration(["a", "a", "a"])
        assert ratio == 100.0
        assert src == "a"
        assert count == 3
        assert unique == 1

    @pytest.mark.fast
    def test_all_unique(self):
        ratio, src, count, unique, total = _compute_concentration(["a", "b", "c", "d"])
        assert ratio == 25.0
        assert unique == 4

    @pytest.mark.fast
    def test_ignores_empty_strings(self):
        ratio, src, count, unique, total = _compute_concentration(["a", "", "a", ""])
        assert total == 2
        assert ratio == 100.0

    @pytest.mark.fast
    def test_majority_source(self):
        sources = ["a", "a", "a", "b", "c"]
        ratio, src, count, unique, total = _compute_concentration(sources)
        assert ratio == 60.0
        assert src == "a"
        assert count == 3


class TestDetectTemporalClusters:
    """Test _detect_temporal_clusters."""

    @pytest.mark.fast
    def test_no_clusters(self):
        sources = ["a", "b", "a", "b", "a"]
        assert _detect_temporal_clusters(sources) == 0

    @pytest.mark.fast
    def test_single_cluster_of_3(self):
        sources = ["a", "a", "a", "b", "c"]
        assert _detect_temporal_clusters(sources) == 1

    @pytest.mark.fast
    def test_single_cluster_of_5(self):
        # One long run should count as 1 cluster, not multiple
        sources = ["a", "a", "a", "a", "a", "b"]
        assert _detect_temporal_clusters(sources) == 1

    @pytest.mark.fast
    def test_two_clusters(self):
        sources = ["a", "a", "a", "b", "b", "b"]
        assert _detect_temporal_clusters(sources) == 2

    @pytest.mark.fast
    def test_empty_breaks_cluster(self):
        sources = ["a", "a", "", "a", "a", "a"]
        assert _detect_temporal_clusters(sources) == 1

    @pytest.mark.fast
    def test_empty_list(self):
        assert _detect_temporal_clusters([]) == 0

    @pytest.mark.fast
    def test_all_empty(self):
        assert _detect_temporal_clusters(["", "", ""]) == 0

    @pytest.mark.fast
    def test_run_of_2_no_cluster(self):
        sources = ["a", "a", "b", "b"]
        assert _detect_temporal_clusters(sources) == 0


class TestComputeDiversityMetrics:
    """Test compute_diversity_metrics main function."""

    @pytest.mark.fast
    def test_empty_results(self):
        report = compute_diversity_metrics([])
        assert len(report.track_metrics) == 0
        assert len(report.concentration_warnings) == 0

    @pytest.mark.fast
    def test_all_same_v1_source_warns(self):
        """WARNING fires when >60% of V1 clips from one source."""
        results = [
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
        ]
        report = compute_diversity_metrics(results)

        v1 = report.track_metrics['V1']
        assert v1.total_clips == 5
        assert v1.unique_sources == 2
        assert v1.max_concentration_ratio == 80.0

        # Should have a concentration warning for V1
        assert len(report.concentration_warnings) >= 1
        assert any("V1" in w for w in report.concentration_warnings)

    @pytest.mark.fast
    def test_diverse_sources_no_warning(self):
        """No warning when sources are well-distributed."""
        results = [
            _make_match_result(primary_source=f"video_{i}.mp4")
            for i in range(10)
        ]
        report = compute_diversity_metrics(results)

        v1 = report.track_metrics['V1']
        assert v1.max_concentration_ratio <= 10.0
        # No concentration warnings
        v1_warnings = [w for w in report.concentration_warnings if "V1" in w]
        assert len(v1_warnings) == 0

    @pytest.mark.fast
    def test_temporal_clustering_detected(self):
        """Detects 3+ consecutive segments using same source."""
        results = [
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
            _make_match_result(primary_source="video_C.mp4"),
        ]
        report = compute_diversity_metrics(results)

        v1 = report.track_metrics['V1']
        assert v1.temporal_cluster_count == 1
        assert len(report.temporal_warnings) >= 1
        assert any("V1" in w for w in report.temporal_warnings)

    @pytest.mark.fast
    def test_no_temporal_clustering(self):
        """No temporal warning when sources alternate."""
        results = [
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
        ]
        report = compute_diversity_metrics(results)

        v1 = report.track_metrics['V1']
        assert v1.temporal_cluster_count == 0
        assert len(report.temporal_warnings) == 0

    @pytest.mark.fast
    def test_secondary_track_concentration(self):
        """Concentration computed for V4-V6 tracks."""
        results = [
            _make_match_result(
                primary_source="video_A.mp4",
                secondary_sources=["video_X.mp4", "video_X.mp4", "video_X.mp4"]
            ),
            _make_match_result(
                primary_source="video_B.mp4",
                secondary_sources=["video_X.mp4", "video_Y.mp4", "video_Z.mp4"]
            ),
        ]
        report = compute_diversity_metrics(results)

        # V4 should have 100% concentration (both segments use video_X)
        v4 = report.track_metrics['V4']
        assert v4.total_clips == 2
        assert v4.max_concentration_ratio == 100.0

    @pytest.mark.fast
    def test_v7_strategy_matches(self):
        """Strategy matches (non-broll) appear in V7."""
        results = [
            _make_match_result(
                primary_source="video_A.mp4",
                strategy_sources=[("video_S.mp4", "embedding_diversity")]
            ),
            _make_match_result(
                primary_source="video_B.mp4",
                strategy_sources=[("video_S.mp4", "embedding_diversity")]
            ),
        ]
        report = compute_diversity_metrics(results)

        v7 = report.track_metrics['V7']
        assert v7.total_clips == 2
        assert v7.max_concentration_ratio == 100.0

    @pytest.mark.fast
    def test_v8_broll_matches(self):
        """B-roll strategy matches appear in V8."""
        results = [
            _make_match_result(
                primary_source="video_A.mp4",
                strategy_sources=[("broll_1.mp4", "broll_only")]
            ),
            _make_match_result(
                primary_source="video_B.mp4",
                strategy_sources=[("broll_2.mp4", "broll_only")]
            ),
        ]
        report = compute_diversity_metrics(results)

        v8 = report.track_metrics['V8']
        assert v8.total_clips == 2
        assert v8.unique_sources == 2

    @pytest.mark.fast
    def test_custom_threshold(self):
        """Custom concentration threshold works."""
        results = [
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
        ]
        # ~66.7% concentration
        report_strict = compute_diversity_metrics(results, concentration_threshold=50.0)
        report_lenient = compute_diversity_metrics(results, concentration_threshold=70.0)

        assert len(report_strict.concentration_warnings) >= 1
        assert len([w for w in report_lenient.concentration_warnings if "V1" in w]) == 0

    @pytest.mark.fast
    def test_to_dict_serializable(self):
        """DiversityReport.to_dict() produces a JSON-serializable dict."""
        import json
        results = [
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
        ]
        report = compute_diversity_metrics(results)
        d = report.to_dict()
        # Should be JSON-serializable
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert 'V1' in d


class TestLogDiversityMetrics:
    """Test log_diversity_metrics logging output."""

    @pytest.mark.fast
    def test_logs_warnings(self, caplog):
        """WARNING-level logs appear for high concentration."""
        results = [
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
        ]
        report = compute_diversity_metrics(results)

        with caplog.at_level(logging.WARNING, logger="src.matching.metrics"):
            log_diversity_metrics(report)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("HIGH CONCENTRATION" in m for m in warning_messages)

    @pytest.mark.fast
    def test_logs_temporal_warnings(self, caplog):
        """WARNING-level logs appear for temporal clustering."""
        results = [
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_A.mp4"),
            _make_match_result(primary_source="video_B.mp4"),
        ]
        report = compute_diversity_metrics(results)

        with caplog.at_level(logging.WARNING, logger="src.matching.metrics"):
            log_diversity_metrics(report)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("TEMPORAL CLUSTERING" in m for m in warning_messages)

    @pytest.mark.fast
    def test_no_warnings_diverse(self, caplog):
        """No warnings for well-distributed sources."""
        results = [
            _make_match_result(primary_source=f"video_{i}.mp4")
            for i in range(10)
        ]
        report = compute_diversity_metrics(results)

        with caplog.at_level(logging.WARNING, logger="src.matching.metrics"):
            log_diversity_metrics(report)

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_messages) == 0


class TestGetTrackSources:
    """Test _get_track_sources extraction."""

    @pytest.mark.fast
    def test_extracts_all_tracks(self):
        """All 8 tracks have entries for each result."""
        results = [
            _make_match_result(
                primary_source="v1.mp4",
                alt_sources=["v2.mp4", "v3.mp4"],
                secondary_sources=["v4.mp4", "v5.mp4", "v6.mp4"],
                strategy_sources=[("v7.mp4", "embedding_diversity"), ("v8.mp4", "broll_only")],
            ),
        ]
        tracks = _get_track_sources(results)
        assert len(tracks) == 8
        assert tracks['V1'] == ['v1']
        assert tracks['V2'] == ['v2']
        assert tracks['V3'] == ['v3']
        assert tracks['V4'] == ['v4']
        assert tracks['V5'] == ['v5']
        assert tracks['V6'] == ['v6']
        assert tracks['V7'] == ['v7']
        assert tracks['V8'] == ['v8']

    @pytest.mark.fast
    def test_missing_tracks_get_empty(self):
        """Missing alternatives/secondaries produce empty strings."""
        results = [_make_match_result(primary_source="v1.mp4")]
        tracks = _get_track_sources(results)
        assert tracks['V1'] == ['v1']
        assert tracks['V2'] == ['']
        assert tracks['V4'] == ['']
        assert tracks['V7'] == ['']
        assert tracks['V8'] == ['']
