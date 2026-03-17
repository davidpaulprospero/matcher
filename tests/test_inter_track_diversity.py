"""
Tests for inter-track embedding diversity metric (US-84-009).

Covers:
- Pairwise cosine distance between V1, V2, V3 per segment
- Track-level diversity score (mean pairwise distance across segments)
- Warning when avg_embedding_distance < configurable threshold (default 0.15)
- Quality report includes track_diversity_score and low_diversity_segments_count
- Low diversity detected for similar embeddings vs high diversity for different ones
- Edge cases: missing alternatives, single track, empty results
"""

import math
import logging
import pytest
from unittest.mock import MagicMock
from typing import Dict, List, Optional, Tuple

from src.matching.alternative_selection import (
    cosine_distance,
    compute_inter_track_embedding_diversity,
    log_inter_track_diversity,
    InterTrackDiversityResult,
    TrackEmbeddingDiversityReport,
)
from src.matching.metrics import MatchQualityMetrics

pytestmark = pytest.mark.unit


# =============================================================================
# HELPERS
# =============================================================================

def _make_segment(source_file: str, start_time: float = 0.0):
    """Create a mock SRTSegment-like object."""
    seg = MagicMock()
    seg.source_file = source_file
    seg.start_time = start_time
    return seg


def _make_match_result(
    v1_source: str = "",
    v1_start: float = 0.0,
    v2_source: str = "",
    v2_start: float = 0.0,
    v3_source: str = "",
    v3_start: float = 0.0,
):
    """Create a mock MatchResult with V1/V2/V3 segments."""
    result = MagicMock()

    # V1 - primary match
    if v1_source:
        pm = MagicMock()
        pm.video_segment = _make_segment(v1_source, v1_start)
        result.primary_match = pm
    else:
        result.primary_match = None

    # V2, V3 - alternatives
    alts = []
    for src, start in [(v2_source, v2_start), (v3_source, v3_start)]:
        if src:
            alt = MagicMock()
            alt.video_segment = _make_segment(src, start)
            alts.append(alt)
    result.alternatives = alts

    return result


def _embedding_lookup(embeddings_map: Dict[Tuple[str, float], List[float]]):
    """Create an embedding lookup function from a dict."""
    def get_embedding(source_file: str, start_time: float) -> Optional[List[float]]:
        return embeddings_map.get((source_file, start_time))
    return get_embedding


# =============================================================================
# COSINE DISTANCE TESTS
# =============================================================================

class TestCosineDistance:
    """Test the cosine_distance utility function."""

    def test_identical_vectors_zero_distance(self):
        vec = [1.0, 0.0, 0.0]
        assert cosine_distance(vec, vec) == pytest.approx(0.0, abs=1e-9)

    def test_orthogonal_vectors_distance_one(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_opposite_vectors_distance_two(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-9)

    def test_similar_vectors_small_distance(self):
        a = [1.0, 0.1, 0.0]
        b = [1.0, 0.2, 0.0]
        dist = cosine_distance(a, b)
        assert 0.0 < dist < 0.1  # Very similar

    def test_zero_vector_returns_one(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert cosine_distance(a, b) == 1.0

    def test_both_zero_vectors_returns_one(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_distance(a, b) == 1.0


# =============================================================================
# INTER-TRACK DIVERSITY COMPUTATION TESTS
# =============================================================================

class TestComputeInterTrackDiversity:
    """Test compute_inter_track_embedding_diversity function."""

    def test_low_diversity_similar_cityscape_variants(self):
        """V1-V3 are all cityscape variants: similar embeddings = low diversity."""
        # All three embeddings are nearly identical (cityscape variants)
        emb_map = {
            ("cityscape_day.mp4", 0.0): [0.9, 0.1, 0.0, 0.05],
            ("cityscape_sunset.mp4", 0.0): [0.88, 0.12, 0.01, 0.04],
            ("cityscape_aerial.mp4", 0.0): [0.91, 0.09, 0.0, 0.06],
        }
        results = [_make_match_result(
            v1_source="cityscape_day.mp4",
            v2_source="cityscape_sunset.mp4",
            v3_source="cityscape_aerial.mp4",
        )]
        report = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.15
        )
        assert report.track_diversity_score < 0.15
        assert report.low_diversity_segments_count == 1

    def test_high_diversity_different_scene_types(self):
        """V1=cityscape, V2=forest, V3=underwater: different embeddings = high diversity."""
        emb_map = {
            ("cityscape.mp4", 0.0): [0.9, 0.1, 0.0, 0.0],
            ("forest.mp4", 0.0): [0.0, 0.9, 0.1, 0.0],
            ("underwater.mp4", 0.0): [0.0, 0.0, 0.9, 0.1],
        }
        results = [_make_match_result(
            v1_source="cityscape.mp4",
            v2_source="forest.mp4",
            v3_source="underwater.mp4",
        )]
        report = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.15
        )
        assert report.track_diversity_score > 0.15
        assert report.low_diversity_segments_count == 0

    def test_multiple_segments_mixed_diversity(self):
        """Two segments: one low diversity, one high diversity."""
        emb_map = {
            # Segment 0: low diversity (similar)
            ("city_a.mp4", 0.0): [0.9, 0.1, 0.0],
            ("city_b.mp4", 0.0): [0.88, 0.12, 0.0],
            ("city_c.mp4", 0.0): [0.91, 0.09, 0.0],
            # Segment 1: high diversity (different)
            ("desert.mp4", 10.0): [0.9, 0.0, 0.1],
            ("ocean.mp4", 10.0): [0.0, 0.9, 0.1],
            ("mountain.mp4", 10.0): [0.1, 0.0, 0.9],
        }
        results = [
            _make_match_result(
                v1_source="city_a.mp4",
                v2_source="city_b.mp4",
                v3_source="city_c.mp4",
            ),
            _make_match_result(
                v1_source="desert.mp4", v1_start=10.0,
                v2_source="ocean.mp4", v2_start=10.0,
                v3_source="mountain.mp4", v3_start=10.0,
            ),
        ]
        report = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.15
        )
        assert report.total_segments == 2
        # One segment low, one high
        assert report.low_diversity_segments_count == 1
        # Overall score should be between the two
        assert len(report.segment_results) == 2

    def test_empty_results_returns_zero(self):
        """Empty results list returns zero metrics."""
        report = compute_inter_track_embedding_diversity(
            [], lambda s, t: None, min_distance_threshold=0.15
        )
        assert report.track_diversity_score == 0.0
        assert report.low_diversity_segments_count == 0
        assert report.total_segments == 0

    def test_single_track_no_pairwise(self):
        """Only V1, no V2/V3: can't compute pairwise distance."""
        emb_map = {("video.mp4", 0.0): [1.0, 0.0, 0.0]}
        results = [_make_match_result(v1_source="video.mp4")]
        report = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.15
        )
        assert report.track_diversity_score == 0.0
        assert len(report.segment_results) == 0

    def test_two_tracks_v1_v2_only(self):
        """V1 and V2 only (no V3): computes V1-V2 distance."""
        emb_map = {
            ("video_a.mp4", 0.0): [1.0, 0.0, 0.0],
            ("video_b.mp4", 0.0): [0.0, 1.0, 0.0],
        }
        results = [_make_match_result(
            v1_source="video_a.mp4",
            v2_source="video_b.mp4",
        )]
        report = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.15
        )
        assert report.track_diversity_score > 0.5
        assert len(report.segment_results) == 1
        assert "V1-V2" in report.segment_results[0].pairwise_distances

    def test_missing_embedding_skips_segment(self):
        """When embedding lookup returns None, segment is skipped."""
        results = [_make_match_result(
            v1_source="video_a.mp4",
            v2_source="video_b.mp4",
            v3_source="video_c.mp4",
        )]
        # No embeddings available
        report = compute_inter_track_embedding_diversity(
            results, lambda s, t: None, min_distance_threshold=0.15
        )
        assert len(report.segment_results) == 0

    def test_pairwise_distances_keys(self):
        """Check that pairwise distance keys are correct for V1-V2-V3."""
        emb_map = {
            ("a.mp4", 0.0): [1.0, 0.0, 0.0],
            ("b.mp4", 0.0): [0.0, 1.0, 0.0],
            ("c.mp4", 0.0): [0.0, 0.0, 1.0],
        }
        results = [_make_match_result(
            v1_source="a.mp4", v2_source="b.mp4", v3_source="c.mp4"
        )]
        report = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.15
        )
        keys = set(report.segment_results[0].pairwise_distances.keys())
        assert keys == {"V1-V2", "V1-V3", "V2-V3"}

    def test_configurable_threshold(self):
        """Threshold is respected: same data, different threshold changes low_diversity flag."""
        emb_map = {
            ("a.mp4", 0.0): [0.9, 0.1, 0.0],
            ("b.mp4", 0.0): [0.85, 0.15, 0.0],
            ("c.mp4", 0.0): [0.88, 0.12, 0.0],
        }
        results = [_make_match_result(
            v1_source="a.mp4", v2_source="b.mp4", v3_source="c.mp4"
        )]
        # With very low threshold, same data counts as diverse
        report_low = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.001
        )
        assert report_low.low_diversity_segments_count == 0

        # With high threshold, same data counts as low diversity
        report_high = compute_inter_track_embedding_diversity(
            results, _embedding_lookup(emb_map), min_distance_threshold=0.99
        )
        assert report_high.low_diversity_segments_count == 1

    def test_report_to_dict(self):
        """TrackEmbeddingDiversityReport serializes correctly."""
        report = TrackEmbeddingDiversityReport(
            segment_results=[],
            track_diversity_score=0.42,
            low_diversity_segments_count=3,
            total_segments=10,
        )
        d = report.to_dict()
        assert d['track_diversity_score'] == pytest.approx(0.42)
        assert d['low_diversity_segments_count'] == 3
        assert d['total_segments'] == 10


# =============================================================================
# LOGGING TESTS
# =============================================================================

class TestLogInterTrackDiversity:
    """Test log_inter_track_diversity logging output."""

    def test_logs_warning_when_low_diversity(self, caplog):
        """Warning logged when overall diversity score < threshold."""
        report = TrackEmbeddingDiversityReport(
            segment_results=[],
            track_diversity_score=0.10,
            low_diversity_segments_count=5,
            total_segments=10,
        )
        with caplog.at_level(logging.WARNING):
            log_inter_track_diversity(report, min_distance_threshold=0.15)
        assert any("LOW TRACK DIVERSITY" in r.message for r in caplog.records)

    def test_no_warning_when_high_diversity(self, caplog):
        """No warning when diversity is above threshold."""
        report = TrackEmbeddingDiversityReport(
            segment_results=[],
            track_diversity_score=0.50,
            low_diversity_segments_count=0,
            total_segments=10,
        )
        with caplog.at_level(logging.WARNING):
            log_inter_track_diversity(report, min_distance_threshold=0.15)
        assert not any("LOW TRACK DIVERSITY" in r.message for r in caplog.records)

    def test_logs_info_metrics(self, caplog):
        """INFO level logs include diversity score and low segment count."""
        report = TrackEmbeddingDiversityReport(
            segment_results=[],
            track_diversity_score=0.30,
            low_diversity_segments_count=2,
            total_segments=8,
        )
        with caplog.at_level(logging.INFO):
            log_inter_track_diversity(report, min_distance_threshold=0.15)
        info_messages = " ".join(r.message for r in caplog.records)
        assert "0.300" in info_messages
        assert "2/8" in info_messages


# =============================================================================
# QUALITY METRICS INTEGRATION TESTS
# =============================================================================

class TestQualityMetricsIntegration:
    """Test MatchQualityMetrics includes track diversity fields."""

    def test_metrics_has_diversity_fields(self):
        """MatchQualityMetrics has track_diversity_score and low_diversity_segments_count."""
        metrics = MatchQualityMetrics()
        assert hasattr(metrics, 'track_diversity_score')
        assert hasattr(metrics, 'low_diversity_segments_count')
        assert metrics.track_diversity_score == 0.0
        assert metrics.low_diversity_segments_count == 0

    def test_metrics_to_dict_includes_diversity(self):
        """to_dict includes track_diversity_score and low_diversity_segments_count."""
        metrics = MatchQualityMetrics(
            track_diversity_score=0.35,
            low_diversity_segments_count=4,
        )
        d = metrics.to_dict()
        assert 'track_diversity_score' in d
        assert d['track_diversity_score'] == pytest.approx(0.35)
        assert d['low_diversity_segments_count'] == 4

    def test_metrics_from_dict_includes_diversity(self):
        """from_dict deserializes track diversity fields."""
        data = {
            'track_diversity_score': 0.28,
            'low_diversity_segments_count': 7,
        }
        metrics = MatchQualityMetrics.from_dict(data)
        assert metrics.track_diversity_score == pytest.approx(0.28)
        assert metrics.low_diversity_segments_count == 7

    def test_metrics_from_dict_defaults(self):
        """from_dict uses defaults when diversity fields missing (backward compat)."""
        metrics = MatchQualityMetrics.from_dict({})
        assert metrics.track_diversity_score == 0.0
        assert metrics.low_diversity_segments_count == 0


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Test MatchingScoringConfig includes min_track_diversity_distance."""

    def test_scoring_config_has_field(self):
        from src.config.sections.matching import MatchingScoringConfig
        config = MatchingScoringConfig()
        assert hasattr(config, 'min_track_diversity_distance')
        assert config.min_track_diversity_distance == 0.15
