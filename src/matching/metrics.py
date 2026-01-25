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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary for checkpoint storage."""
        return {
            'avg_confidence': self.avg_confidence,
            'min_confidence': self.min_confidence,
            'max_confidence': self.max_confidence,
            'confidence_std': self.confidence_std,
            'gap_count': self.gap_count,
            'match_rate': self.match_rate,
            'total_segments': self.total_segments,
            'matched_segments': self.matched_segments,
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

    for m in matches:
        # Handle MatchResult structure (has primary_match)
        if hasattr(m, 'primary_match') and m.primary_match:
            conf = getattr(m.primary_match, 'confidence', 0.0)
            confidences.append(conf)
            # Check for gap
            if getattr(m, 'has_gap', False):
                gap_count += 1
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
        )

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
    logger.info("=============================")
