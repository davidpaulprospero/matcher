"""
Match quality metrics tracking and logging.

Provides MatchQualityMetrics dataclass for tracking:
- Confidence score distribution (avg, min, max, std)
- Gap count and match rate
- Quality summary logging
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchQualityMetrics:
    """
    Tracks quality metrics for a matching run.

    Attributes:
        avg_confidence: Average confidence score across all matches
        min_confidence: Minimum confidence score
        max_confidence: Maximum confidence score
        confidence_std: Standard deviation of confidence scores
        gap_count: Number of segments with no match (gaps)
        match_rate: Ratio of matched segments to total segments (0.0-1.0)
        total_segments: Total number of voiceover segments
        matched_segments: Number of segments with matches
    """
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0
    confidence_std: float = 0.0
    gap_count: int = 0
    match_rate: float = 0.0
    total_segments: int = 0
    matched_segments: int = 0
    uncertain_matches_count: int = 0  # US-84-008: segments with ambiguous candidate pools
    track_diversity_score: float = 0.0  # US-84-009: mean pairwise cosine distance V1/V2/V3
    low_diversity_segments_count: int = 0  # US-84-009: segments where V1/V2/V3 too similar

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary for checkpoint storage."""
        # Convert all float values to Python float to avoid numpy.float32 incompatibility
        # with OpenTimelineIO's AnyDictionary which only accepts specific types
        return {
            'avg_confidence': float(self.avg_confidence),
            'min_confidence': float(self.min_confidence),
            'max_confidence': float(self.max_confidence),
            'confidence_std': float(self.confidence_std),
            'gap_count': self.gap_count,
            'match_rate': float(self.match_rate),
            'total_segments': self.total_segments,
            'matched_segments': self.matched_segments,
            'uncertain_matches_count': self.uncertain_matches_count,
            'track_diversity_score': float(self.track_diversity_score),
            'low_diversity_segments_count': self.low_diversity_segments_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchQualityMetrics':
        """Deserialize metrics from dictionary."""
        return cls(
            avg_confidence=data.get('avg_confidence', 0.0),
            min_confidence=data.get('min_confidence', 0.0),
            max_confidence=data.get('max_confidence', 0.0),
            confidence_std=data.get('confidence_std', 0.0),
            gap_count=data.get('gap_count', 0),
            match_rate=data.get('match_rate', 0.0),
            total_segments=data.get('total_segments', 0),
            matched_segments=data.get('matched_segments', 0),
            uncertain_matches_count=data.get('uncertain_matches_count', 0),
            track_diversity_score=data.get('track_diversity_score', 0.0),
            low_diversity_segments_count=data.get('low_diversity_segments_count', 0),
        )


def calculate_match_quality_metrics(
    matches: List[Any],
    total_segments: int
) -> MatchQualityMetrics:
    """
    Calculate quality metrics from a list of matches.

    Args:
        matches: List of match objects (MatchResult or Match)
        total_segments: Total number of voiceover segments

    Returns:
        MatchQualityMetrics with calculated values
    """
    if not matches or total_segments <= 0:
        return MatchQualityMetrics(
            total_segments=total_segments,
            match_rate=0.0 if total_segments > 0 else 0.0
        )

    # Extract confidence scores from matches
    confidences = []
    gap_count = 0
    uncertain_count = 0

    for m in matches:
        # Handle MatchResult structure (has primary_match)
        if hasattr(m, 'primary_match') and m.primary_match:
            conf = getattr(m.primary_match, 'confidence', 0.0)
            confidences.append(conf)
            # Check for gap
            if getattr(m, 'has_gap', False):
                gap_count += 1
            # US-84-008: Count ambiguous pool matches
            if getattr(m, 'ambiguous_pool', False):
                uncertain_count += 1
        # Handle direct Match structure
        elif hasattr(m, 'confidence'):
            conf = m.confidence
            confidences.append(conf)
            # Check for gap in Match object
            if getattr(m, 'has_gap', False):
                gap_count += 1
        else:
            # Treat as gap if no valid match
            gap_count += 1

    matched_segments = len(confidences)
    match_rate = matched_segments / total_segments if total_segments > 0 else 0.0

    if not confidences:
        return MatchQualityMetrics(
            total_segments=total_segments,
            gap_count=gap_count,
            match_rate=0.0,
            matched_segments=0,
            uncertain_matches_count=uncertain_count,
        )

    # Convert all confidence values to Python float to avoid numpy.float32 issues
    # numpy.float32 is not JSON serializable and causes errors in output stage
    confidences = [float(c) for c in confidences]

    avg_confidence = sum(confidences) / len(confidences)
    min_confidence = min(confidences)
    max_confidence = max(confidences)

    # Calculate standard deviation (need at least 2 values)
    if len(confidences) >= 2:
        confidence_std = statistics.stdev(confidences)
    else:
        confidence_std = 0.0

    return MatchQualityMetrics(
        avg_confidence=avg_confidence,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
        confidence_std=confidence_std,
        gap_count=gap_count,
        match_rate=match_rate,
        total_segments=total_segments,
        matched_segments=matched_segments,
        uncertain_matches_count=uncertain_count,
    )


def log_quality_summary(metrics: MatchQualityMetrics) -> None:
    """
    Log a summary of match quality metrics.

    Logs at INFO level with formatted output.

    Args:
        metrics: MatchQualityMetrics to log
    """
    logger.info("=== Match Quality Summary ===")
    logger.info(f"  Total segments: {metrics.total_segments}")
    logger.info(f"  Matched segments: {metrics.matched_segments}")
    logger.info(f"  Match rate: {metrics.match_rate:.1%}")
    logger.info(f"  Gap count: {metrics.gap_count}")
    logger.info(f"  Avg confidence: {metrics.avg_confidence:.3f}")
    logger.info(f"  Min confidence: {metrics.min_confidence:.3f}")
    logger.info(f"  Max confidence: {metrics.max_confidence:.3f}")
    logger.info(f"  Confidence std: {metrics.confidence_std:.3f}")
    if metrics.uncertain_matches_count > 0:
        logger.info(f"  Uncertain matches (ambiguous pool): {metrics.uncertain_matches_count}")
    if metrics.track_diversity_score > 0:
        logger.info(f"  Track diversity score (V1/V2/V3): {metrics.track_diversity_score:.3f}")
        logger.info(f"  Low diversity segments: {metrics.low_diversity_segments_count}")
    logger.info("=============================")


def log_confidence_histogram(confidences: List[float], bar_width: int = 30) -> str:
    """
    Generate and log an ASCII histogram of confidence distribution.

    Buckets: 0-0.5, 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0

    Args:
        confidences: List of confidence scores (0.0-1.0)
        bar_width: Maximum width of histogram bars (default: 30)

    Returns:
        The histogram string (for testing purposes)
    """
    # Define buckets with labels
    buckets = [
        (0.0, 0.5, "0.0-0.5"),
        (0.5, 0.6, "0.5-0.6"),
        (0.6, 0.7, "0.6-0.7"),
        (0.7, 0.8, "0.7-0.8"),
        (0.8, 0.9, "0.8-0.9"),
        (0.9, 1.0, "0.9-1.0"),
    ]

    # Count confidences per bucket
    counts = {label: 0 for _, _, label in buckets}
    for conf in confidences:
        for low, high, label in buckets:
            # Include upper bound for last bucket (0.9-1.0)
            if label == "0.9-1.0":
                if low <= conf <= high:
                    counts[label] += 1
                    break
            else:
                if low <= conf < high:
                    counts[label] += 1
                    break

    # Find max count for scaling
    max_count = max(counts.values()) if counts.values() else 1
    total = len(confidences)

    # Build histogram lines
    lines = []
    lines.append("=== Confidence Distribution ===")
    lines.append("")

    for _, _, label in buckets:
        count = counts[label]
        # Scale bar length proportionally
        if max_count > 0:
            bar_len = int((count / max_count) * bar_width)
        else:
            bar_len = 0
        bar = "█" * bar_len
        # Calculate percentage
        pct = (count / total * 100) if total > 0 else 0
        # Format: "0.5-0.6 |████████████████     |  12 ( 24.0%)"
        lines.append(f"  {label} |{bar:<{bar_width}}| {count:4d} ({pct:5.1f}%)")

    lines.append("")
    lines.append(f"  Total segments: {total}")
    lines.append("===============================")

    # Log each line at INFO level
    histogram_str = "\n".join(lines)
    for line in lines:
        logger.info(line)

    return histogram_str


# =============================================================================
# DIVERSITY METRICS (US-53-005)
# =============================================================================

@dataclass
class TrackDiversityMetrics:
    """Diversity metrics for a single track."""
    track_name: str
    total_clips: int = 0
    unique_sources: int = 0
    max_concentration_ratio: float = 0.0  # Percentage of clips from most-used source
    most_used_source: str = ""
    most_used_count: int = 0
    temporal_cluster_count: int = 0  # Number of 3+ consecutive same-source runs


@dataclass
class DiversityReport:
    """Complete diversity report across all tracks."""
    track_metrics: Dict[str, TrackDiversityMetrics] = field(default_factory=dict)
    concentration_warnings: List[str] = field(default_factory=list)
    temporal_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for checkpoint storage."""
        result = {}
        for track_name, tm in self.track_metrics.items():
            result[track_name] = {
                'total_clips': tm.total_clips,
                'unique_sources': tm.unique_sources,
                'max_concentration_ratio': float(tm.max_concentration_ratio),
                'most_used_source': tm.most_used_source,
                'most_used_count': tm.most_used_count,
                'temporal_cluster_count': tm.temporal_cluster_count,
            }
        return result


def _extract_source(source_file: str) -> str:
    """Extract a short source identifier from a source_file path or video ID."""
    if not source_file:
        return ""
    from pathlib import Path
    # Use stem (filename without extension), truncated for readability
    return Path(source_file).stem[:30]


def _get_track_sources(results: List[Any]) -> Dict[str, List[str]]:
    """Extract ordered source lists per track from match results.

    Returns dict mapping track name to list of source identifiers (in segment order).
    """
    tracks: Dict[str, List[str]] = {
        'V1': [], 'V2': [], 'V3': [],
        'V4': [], 'V5': [], 'V6': [],
        'V7': [], 'V8': [],
    }

    for r in results:
        # V1 - primary match
        if hasattr(r, 'primary_match') and r.primary_match:
            src = getattr(r.primary_match.video_segment, 'source_file', '')
            tracks['V1'].append(_extract_source(src))
        else:
            tracks['V1'].append('')

        # V2-V3 - alternatives
        alts = getattr(r, 'alternatives', []) or []
        for i, track in enumerate(['V2', 'V3']):
            if i < len(alts):
                src = getattr(alts[i].video_segment, 'source_file', '')
                tracks[track].append(_extract_source(src))
            else:
                tracks[track].append('')

        # V4-V6 - secondary matches
        secs = getattr(r, 'secondary_matches', []) or []
        for i, track in enumerate(['V4', 'V5', 'V6']):
            if i < len(secs):
                src = getattr(secs[i].video_segment, 'source_file', '')
                tracks[track].append(_extract_source(src))
            else:
                tracks[track].append('')

        # V7-V8 - strategy matches
        strats = getattr(r, 'strategy_matches', []) or []
        v7_src = ''
        v8_src = ''
        for sm in strats:
            strategy = getattr(sm, 'strategy', '')
            src = _extract_source(getattr(sm.video_segment, 'source_file', ''))
            if 'broll' in strategy.lower() or 'b_roll' in strategy.lower():
                if not v8_src:
                    v8_src = src
            else:
                if not v7_src:
                    v7_src = src
        tracks['V7'].append(v7_src)
        tracks['V8'].append(v8_src)

    return tracks


def _compute_concentration(sources: List[str]) -> tuple:
    """Compute concentration ratio for a list of sources.

    Returns (max_concentration_ratio, most_used_source, most_used_count, unique_count, total).
    """
    # Filter out empty strings (no match for that track/segment)
    non_empty = [s for s in sources if s]
    if not non_empty:
        return (0.0, '', 0, 0, 0)

    from collections import Counter
    counts = Counter(non_empty)
    most_common_source, most_common_count = counts.most_common(1)[0]
    total = len(non_empty)
    ratio = most_common_count / total * 100.0

    return (ratio, most_common_source, most_common_count, len(counts), total)


def _detect_temporal_clusters(sources: List[str], min_run: int = 3) -> int:
    """Detect temporal clustering: runs of min_run+ consecutive same source.

    Args:
        sources: Ordered list of source identifiers per segment.
        min_run: Minimum consecutive count to be considered a cluster.

    Returns:
        Number of temporal clusters found.
    """
    cluster_count = 0
    if not sources:
        return 0

    current_src = None
    run_length = 0

    for src in sources:
        if not src:
            # Empty = no match, breaks the run
            current_src = None
            run_length = 0
            continue

        if src == current_src:
            run_length += 1
            if run_length == min_run:
                cluster_count += 1
            # Don't double-count: extending a cluster beyond min_run doesn't add more
        else:
            current_src = src
            run_length = 1

    return cluster_count


def compute_diversity_metrics(
    results: List[Any],
    concentration_threshold: float = 60.0,
) -> DiversityReport:
    """Compute source diversity metrics for all tracks (V1-V8).

    Args:
        results: List of MatchResult objects.
        concentration_threshold: Warn when any track exceeds this % from one source.

    Returns:
        DiversityReport with per-track metrics and warnings.
    """
    report = DiversityReport()

    if not results:
        return report

    track_sources = _get_track_sources(results)

    for track_name, sources in track_sources.items():
        ratio, top_src, top_count, unique, total = _compute_concentration(sources)
        temporal_clusters = _detect_temporal_clusters(sources)

        metrics = TrackDiversityMetrics(
            track_name=track_name,
            total_clips=total,
            unique_sources=unique,
            max_concentration_ratio=ratio,
            most_used_source=top_src,
            most_used_count=top_count,
            temporal_cluster_count=temporal_clusters,
        )
        report.track_metrics[track_name] = metrics

        # Concentration warning
        if total > 0 and ratio > concentration_threshold:
            warning = (
                f"{track_name}: {ratio:.1f}% of clips from '{top_src}' "
                f"({top_count}/{total} clips) exceeds {concentration_threshold:.0f}% threshold"
            )
            report.concentration_warnings.append(warning)

        # Temporal clustering warning
        if temporal_clusters > 0:
            warning = (
                f"{track_name}: {temporal_clusters} temporal cluster(s) detected "
                f"(3+ consecutive segments using same source)"
            )
            report.temporal_warnings.append(warning)

    return report


def log_diversity_metrics(report: DiversityReport) -> None:
    """Log diversity metrics and warnings.

    Logs per-track stats at INFO level and warnings at WARNING level.
    """
    logger.info("=== Source Diversity Metrics ===")

    for track_name in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8']:
        tm = report.track_metrics.get(track_name)
        if not tm or tm.total_clips == 0:
            continue
        logger.info(
            f"  {track_name}: {tm.unique_sources} unique sources, "
            f"{tm.total_clips} clips, "
            f"max concentration {tm.max_concentration_ratio:.1f}% ('{tm.most_used_source}'), "
            f"temporal clusters: {tm.temporal_cluster_count}"
        )

    # Log warnings
    for warning in report.concentration_warnings:
        logger.warning(f"HIGH CONCENTRATION: {warning}")

    for warning in report.temporal_warnings:
        logger.warning(f"TEMPORAL CLUSTERING: {warning}")

    logger.info("================================")


# =============================================================================
# MATCH QUALITY TREND LOGGING (US-63-010)
# =============================================================================

@dataclass
class ChunkTrendMetrics:
    """Metrics for a 10% chunk of voiceover segments during matching."""
    chunk_index: int
    chunk_start_idx: int  # Starting segment index
    chunk_end_idx: int    # Ending segment index (exclusive)
    segment_count: int
    rolling_avg_confidence: float  # Rolling average up to this chunk
    chunk_avg_confidence: float    # Average for just this chunk
    low_confidence_segments: List[int] = field(default_factory=list)  # Segments >0.2 below avg
    # US-84-010: Per-chunk diagnostics
    avg_candidates_considered: float = 0.0  # Mean candidate pool size in this chunk
    dominant_negative_adjustment: str = ""  # Most common negative adjustment type in chunk
    negative_adjustment_counts: Dict[str, int] = field(default_factory=dict)  # Counts per adj type


@dataclass
class MatchQualityTrend:
    """Complete trend data for a matching run across all chunks.

    Tracks confidence trends across 10% chunks of voiceover segments,
    identifying problematic sections that consistently match poorly.

    US-63-010: Helps identify voiceover sections (e.g., technical jargon,
    abstract concepts) that need targeted query refinement in ITERATIVE_MATCH.
    """
    chunk_metrics: List[ChunkTrendMetrics] = field(default_factory=list)
    trend_direction: str = "stable"  # 'improving', 'stable', 'degrading'
    trend_slope: float = 0.0  # Linear regression slope of chunk averages
    overall_avg_confidence: float = 0.0
    low_confidence_threshold: float = 0.2  # Delta below running avg to flag
    total_low_confidence_segments: int = 0
    # US-84-010: Diagnostics for degradation cause analysis
    cause_summary: str = ""  # Human-readable cause when trend is DEGRADING
    trend_diagnostics: List[Dict[str, Any]] = field(default_factory=list)  # Per-chunk detail

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for checkpoint storage."""
        result = {
            'chunk_metrics': [
                {
                    'chunk_index': cm.chunk_index,
                    'chunk_start_idx': cm.chunk_start_idx,
                    'chunk_end_idx': cm.chunk_end_idx,
                    'segment_count': cm.segment_count,
                    'rolling_avg_confidence': float(cm.rolling_avg_confidence),
                    'chunk_avg_confidence': float(cm.chunk_avg_confidence),
                    'low_confidence_segments': cm.low_confidence_segments,
                    'avg_candidates_considered': float(cm.avg_candidates_considered),
                    'dominant_negative_adjustment': cm.dominant_negative_adjustment,
                    'negative_adjustment_counts': cm.negative_adjustment_counts,
                }
                for cm in self.chunk_metrics
            ],
            'trend_direction': self.trend_direction,
            'trend_slope': float(self.trend_slope),
            'overall_avg_confidence': float(self.overall_avg_confidence),
            'low_confidence_threshold': float(self.low_confidence_threshold),
            'total_low_confidence_segments': self.total_low_confidence_segments,
            'cause_summary': self.cause_summary,
        }
        if self.trend_diagnostics:
            result['trend_diagnostics'] = self.trend_diagnostics
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MatchQualityTrend':
        """Deserialize from checkpoint data."""
        trend = cls()
        trend.trend_direction = data.get('trend_direction', 'stable')
        trend.trend_slope = data.get('trend_slope', 0.0)
        trend.overall_avg_confidence = data.get('overall_avg_confidence', 0.0)
        trend.low_confidence_threshold = data.get('low_confidence_threshold', 0.2)
        trend.total_low_confidence_segments = data.get('total_low_confidence_segments', 0)
        trend.cause_summary = data.get('cause_summary', '')
        trend.trend_diagnostics = data.get('trend_diagnostics', [])

        for cm_data in data.get('chunk_metrics', []):
            trend.chunk_metrics.append(ChunkTrendMetrics(
                chunk_index=cm_data.get('chunk_index', 0),
                chunk_start_idx=cm_data.get('chunk_start_idx', 0),
                chunk_end_idx=cm_data.get('chunk_end_idx', 0),
                segment_count=cm_data.get('segment_count', 0),
                rolling_avg_confidence=cm_data.get('rolling_avg_confidence', 0.0),
                chunk_avg_confidence=cm_data.get('chunk_avg_confidence', 0.0),
                low_confidence_segments=cm_data.get('low_confidence_segments', []),
                avg_candidates_considered=cm_data.get('avg_candidates_considered', 0.0),
                dominant_negative_adjustment=cm_data.get('dominant_negative_adjustment', ''),
                negative_adjustment_counts=cm_data.get('negative_adjustment_counts', {}),
            ))

        return trend


def calculate_confidence_trend(
    matches: List[Any],
    chunk_count: int = 10,
    low_conf_threshold: float = 0.2,
) -> MatchQualityTrend:
    """
    Calculate confidence trend across voiceover chunks.

    Divides matches into chunks (default 10% each) and calculates:
    - Rolling average confidence up to each chunk
    - Segments with confidence significantly below rolling average
    - Overall trend direction (improving/stable/degrading)

    Args:
        matches: List of match objects from matching stage
        chunk_count: Number of chunks to divide segments into (default 10 for 10% each)
        low_conf_threshold: Delta below running avg to flag as low confidence

    Returns:
        MatchQualityTrend with per-chunk metrics and trend analysis
    """
    trend = MatchQualityTrend(low_confidence_threshold=low_conf_threshold)

    if not matches:
        return trend

    # Extract confidence scores
    confidences: List[float] = []
    for m in matches:
        if hasattr(m, 'primary_match') and m.primary_match:
            conf = getattr(m.primary_match, 'confidence', 0.0)
        elif hasattr(m, 'confidence'):
            conf = m.confidence
        else:
            conf = 0.0
        confidences.append(float(conf))

    if not confidences:
        return trend

    total_segments = len(confidences)
    chunk_size = max(1, total_segments // chunk_count)

    # Track cumulative stats for rolling average
    cumulative_sum = 0.0
    cumulative_count = 0
    chunk_averages: List[float] = []
    total_low_conf_segments = 0

    for chunk_idx in range(chunk_count):
        start_idx = chunk_idx * chunk_size
        # Last chunk takes remaining segments
        if chunk_idx == chunk_count - 1:
            end_idx = total_segments
        else:
            end_idx = start_idx + chunk_size

        if start_idx >= total_segments:
            break

        chunk_confidences = confidences[start_idx:end_idx]
        if not chunk_confidences:
            continue

        chunk_avg = sum(chunk_confidences) / len(chunk_confidences)
        chunk_averages.append(chunk_avg)

        # Update cumulative for rolling average
        cumulative_sum += sum(chunk_confidences)
        cumulative_count += len(chunk_confidences)
        rolling_avg = cumulative_sum / cumulative_count

        # Find low confidence segments (>threshold below rolling avg)
        low_conf_segments = []
        for i, conf in enumerate(chunk_confidences):
            segment_idx = start_idx + i
            if rolling_avg - conf > low_conf_threshold:
                low_conf_segments.append(segment_idx)

        total_low_conf_segments += len(low_conf_segments)

        chunk_metrics = ChunkTrendMetrics(
            chunk_index=chunk_idx,
            chunk_start_idx=start_idx,
            chunk_end_idx=end_idx,
            segment_count=len(chunk_confidences),
            rolling_avg_confidence=rolling_avg,
            chunk_avg_confidence=chunk_avg,
            low_confidence_segments=low_conf_segments,
        )
        trend.chunk_metrics.append(chunk_metrics)

    trend.total_low_confidence_segments = total_low_conf_segments
    trend.overall_avg_confidence = cumulative_sum / cumulative_count if cumulative_count > 0 else 0.0

    # Calculate trend direction using linear regression on chunk averages
    if len(chunk_averages) >= 2:
        trend.trend_slope = _calculate_trend_slope(chunk_averages)

        # Classify trend direction based on slope
        # Use threshold of 0.02 to determine significance
        if trend.trend_slope > 0.02:
            trend.trend_direction = "improving"
        elif trend.trend_slope < -0.02:
            trend.trend_direction = "degrading"
        else:
            trend.trend_direction = "stable"

    return trend


def _calculate_trend_slope(values: List[float]) -> float:
    """
    Calculate linear regression slope for trend detection.

    Uses simple linear regression: slope = sum((x-x_mean)(y-y_mean)) / sum((x-x_mean)^2)

    Args:
        values: List of chunk average confidences

    Returns:
        Slope of the linear fit (positive = improving, negative = degrading)
    """
    n = len(values)
    if n < 2:
        return 0.0

    # x values are chunk indices (0, 1, 2, ...)
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n

    numerator = 0.0
    denominator = 0.0

    for i, y in enumerate(values):
        x_diff = i - x_mean
        y_diff = y - y_mean
        numerator += x_diff * y_diff
        denominator += x_diff * x_diff

    if denominator == 0:
        return 0.0

    return numerator / denominator


def populate_chunk_diagnostics(
    trend: MatchQualityTrend,
    segment_adjustments: Optional[List[Dict[str, float]]] = None,
    segment_candidates_considered: Optional[List[int]] = None,
) -> None:
    """
    Populate per-chunk diagnostics and generate cause summary for degrading trends.

    US-84-010: Adds actionable diagnostics to each chunk and identifies the dominant
    negative adjustment driving quality degradation.

    Args:
        trend: MatchQualityTrend with chunk_metrics already populated
        segment_adjustments: Per-segment dict of {adjustment_name: value} (negative = penalty).
            Index aligns with segment index in the match list.
        segment_candidates_considered: Per-segment count of candidates in the pool.
            Index aligns with segment index in the match list.
    """
    if not trend.chunk_metrics:
        return

    for cm in trend.chunk_metrics:
        start = cm.chunk_start_idx
        end = cm.chunk_end_idx

        # Populate avg_candidates_considered
        if segment_candidates_considered:
            chunk_candidates = segment_candidates_considered[start:end]
            if chunk_candidates:
                cm.avg_candidates_considered = sum(chunk_candidates) / len(chunk_candidates)

        # Populate negative adjustment counts
        if segment_adjustments:
            chunk_adjustments = segment_adjustments[start:end]
            neg_counts: Dict[str, int] = {}
            for adj_dict in chunk_adjustments:
                if not isinstance(adj_dict, dict):
                    continue
                for adj_name, adj_value in adj_dict.items():
                    if adj_value < 0:
                        neg_counts[adj_name] = neg_counts.get(adj_name, 0) + 1
            cm.negative_adjustment_counts = neg_counts
            if neg_counts:
                cm.dominant_negative_adjustment = max(neg_counts, key=neg_counts.get)

    # Build trend_diagnostics array and cause_summary for DEGRADING trends
    if trend.trend_direction == "degrading":
        _build_degradation_diagnostics(trend)


def _build_degradation_diagnostics(trend: MatchQualityTrend) -> None:
    """
    Build trend_diagnostics array and cause_summary for a degrading trend.

    Identifies which chunks are degrading (below overall avg) and what
    negative adjustment dominates those chunks.
    """
    overall_avg = trend.overall_avg_confidence
    degrading_chunks = []
    global_neg_counts: Dict[str, int] = {}

    for cm in trend.chunk_metrics:
        diag: Dict[str, Any] = {
            'chunk_index': cm.chunk_index,
            'avg_confidence': float(cm.chunk_avg_confidence),
            'avg_candidates_considered': float(cm.avg_candidates_considered),
            'dominant_negative_adjustment': cm.dominant_negative_adjustment,
        }
        trend.trend_diagnostics.append(diag)

        # Track chunks below overall average as degrading
        if cm.chunk_avg_confidence < overall_avg:
            degrading_chunks.append(cm)
            for adj_name, count in cm.negative_adjustment_counts.items():
                global_neg_counts[adj_name] = global_neg_counts.get(adj_name, 0) + count

    # Generate cause summary
    if degrading_chunks and global_neg_counts:
        total_neg = sum(global_neg_counts.values())
        top_adj = max(global_neg_counts, key=global_neg_counts.get)
        top_pct = (global_neg_counts[top_adj] / total_neg * 100) if total_neg > 0 else 0

        # Build chunk range string (e.g., "8-10")
        chunk_indices = [cm.chunk_index for cm in degrading_chunks]
        first_chunk = min(chunk_indices)
        last_chunk = max(chunk_indices)
        chunk_range = f"{first_chunk}-{last_chunk}" if first_chunk != last_chunk else str(first_chunk)

        trend.cause_summary = (
            f"Degradation in chunks {chunk_range} due to: "
            f"{top_pct:.0f}% of negative adjustments were {top_adj}"
        )
    elif degrading_chunks:
        chunk_indices = [cm.chunk_index for cm in degrading_chunks]
        first_chunk = min(chunk_indices)
        last_chunk = max(chunk_indices)
        chunk_range = f"{first_chunk}-{last_chunk}" if first_chunk != last_chunk else str(first_chunk)
        trend.cause_summary = (
            f"Degradation in chunks {chunk_range} due to: "
            f"low confidence with no dominant adjustment identified"
        )


def log_chunk_trend(chunk: ChunkTrendMetrics, total_segments: int) -> None:
    """
    Log metrics for a single chunk after matching.

    Called after each 10% chunk is matched to provide real-time feedback.

    Args:
        chunk: ChunkTrendMetrics for the completed chunk
        total_segments: Total number of segments being matched
    """
    pct_complete = (chunk.chunk_end_idx / total_segments) * 100

    logger.info(
        f"  Chunk {chunk.chunk_index + 1} ({pct_complete:.0f}%): "
        f"avg={chunk.chunk_avg_confidence:.3f}, "
        f"rolling_avg={chunk.rolling_avg_confidence:.3f}, "
        f"low_conf_count={len(chunk.low_confidence_segments)}"
    )

    # Log warning for chunks with significant low-confidence segments
    if chunk.low_confidence_segments:
        logger.warning(
            f"    Low confidence segments in chunk {chunk.chunk_index + 1}: "
            f"{chunk.low_confidence_segments[:5]}{'...' if len(chunk.low_confidence_segments) > 5 else ''} "
            f"(>0.2 below rolling avg of {chunk.rolling_avg_confidence:.3f})"
        )


def log_trend_summary(trend: MatchQualityTrend) -> None:
    """
    Log summary of confidence trend at end of MATCH stage.

    Args:
        trend: Complete MatchQualityTrend from matching run
    """
    logger.info("=== Match Quality Trend Summary ===")
    logger.info(f"  Overall avg confidence: {trend.overall_avg_confidence:.3f}")
    logger.info(f"  Trend direction: {trend.trend_direction.upper()}")
    logger.info(f"  Trend slope: {trend.trend_slope:+.4f}")
    logger.info(f"  Low confidence segments: {trend.total_low_confidence_segments}")

    # Print trend indicator
    if trend.trend_direction == "improving":
        indicator = "↑ IMPROVING - quality increased over voiceover duration"
    elif trend.trend_direction == "degrading":
        indicator = "↓ DEGRADING - quality decreased over voiceover duration"
    else:
        indicator = "→ STABLE - consistent quality throughout"

    logger.info(f"  {indicator}")

    # Log chunk-by-chunk summary
    if trend.chunk_metrics:
        logger.info("  Chunk breakdown:")
        for cm in trend.chunk_metrics:
            marker = ""
            if cm.low_confidence_segments:
                marker = f" [!{len(cm.low_confidence_segments)} low]"
            logger.info(
                f"    {cm.chunk_index + 1:2d}: avg={cm.chunk_avg_confidence:.3f}, "
                f"rolling={cm.rolling_avg_confidence:.3f}{marker}"
            )

    logger.info("===================================")

    # Additional warning for degrading trend
    if trend.trend_direction == "degrading":
        logger.warning(
            "TREND WARNING: Match confidence degraded over voiceover duration. "
            "Later segments may benefit from ITERATIVE_MATCH refinement."
        )
        # US-84-010: Log cause summary and diagnostics
        if trend.cause_summary:
            logger.warning(f"CAUSE: {trend.cause_summary}")
        if trend.trend_diagnostics:
            logger.info("  Trend diagnostics (per-chunk):")
            for diag in trend.trend_diagnostics:
                adj_info = f", dominant_adj={diag['dominant_negative_adjustment']}" if diag.get('dominant_negative_adjustment') else ""
                logger.info(
                    f"    Chunk {diag['chunk_index']}: "
                    f"avg_conf={diag['avg_confidence']:.3f}, "
                    f"avg_candidates={diag['avg_candidates_considered']:.1f}"
                    f"{adj_info}"
                )
