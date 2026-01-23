"""Coverage analysis for high matches mode.

Analyzes match results to identify segments needing improvement
and track coverage progress through iterations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class WeakSegment:
    """A segment that needs better matching."""
    segment_id: str
    segment_index: int
    text: str
    current_confidence: float
    current_match: Optional[str] = None
    suggested_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'segment_id': self.segment_id,
            'segment_index': self.segment_index,
            'text': self.text,
            'current_confidence': self.current_confidence,
            'current_match': self.current_match,
            'suggested_keywords': self.suggested_keywords,
        }


@dataclass
class CoverageReport:
    """Report on match coverage quality."""
    total_segments: int
    high_confidence: int  # >= target_confidence
    medium_confidence: int  # >= 0.7 but < target
    low_confidence: int  # < 0.7
    coverage_ratio: float  # high_confidence / total
    weak_segments: List[WeakSegment] = field(default_factory=list)
    target_confidence: float = 0.90

    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_segments': self.total_segments,
            'high_confidence': self.high_confidence,
            'medium_confidence': self.medium_confidence,
            'low_confidence': self.low_confidence,
            'coverage_ratio': self.coverage_ratio,
            'target_confidence': self.target_confidence,
            'weak_segment_count': len(self.weak_segments),
        }

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Coverage: {self.coverage_ratio:.1%} "
            f"({self.high_confidence}/{self.total_segments} at {self.target_confidence:.0%}+)\n"
            f"  High: {self.high_confidence}, Medium: {self.medium_confidence}, "
            f"Low: {self.low_confidence}"
        )


def analyze_coverage(
    matches: List[Any],
    voiceover_segments: List[Any],
    target_confidence: float = 0.90,
    medium_threshold: float = 0.70,
) -> CoverageReport:
    """Analyze match results and identify segments needing improvement.

    Args:
        matches: List of Match objects from matching stage
        voiceover_segments: List of VoiceoverSegment objects
        target_confidence: Threshold for "high confidence" (default 0.90)
        medium_threshold: Threshold for "medium confidence" (default 0.70)

    Returns:
        CoverageReport with breakdown and list of weak segments
    """
    logger.info(f"[coverage_analyzer] Starting coverage analysis")

    # Handle None inputs
    if matches is None:
        matches = []
    if voiceover_segments is None:
        voiceover_segments = []

    logger.info(f"[coverage_analyzer] Input: {len(matches)} matches, {len(voiceover_segments)} segments")
    logger.info(f"[coverage_analyzer] Thresholds: high={target_confidence:.0%}, medium={medium_threshold:.0%}")

    total = len(voiceover_segments)
    if total == 0:
        logger.warning("[coverage_analyzer] No voiceover segments to analyze")
        return CoverageReport(
            total_segments=0,
            high_confidence=0,
            medium_confidence=0,
            low_confidence=0,
            coverage_ratio=0.0,
            weak_segments=[],
            target_confidence=target_confidence,
        )

    # Build match lookup by segment index
    # Handle different match object types:
    # 1. MatchResult (from utils.py) - segment_index in primary_match.voiceover_segment.index
    # 2. MatchResultWrapper - segment_index in _simple_match.segment_index
    # 3. Dict (serialized) - segment_index key
    # 4. Simple Match (state.py) - segment_index attribute
    match_by_segment: Dict[int, Any] = {}
    for i, m in enumerate(matches):
        idx = -1

        # Try different ways to get segment index
        if hasattr(m, 'primary_match') and m.primary_match:
            # MatchResult or MatchResultWrapper
            pm = m.primary_match
            if hasattr(pm, 'voiceover_segment') and pm.voiceover_segment:
                idx = getattr(pm.voiceover_segment, 'index', -1)
            if idx < 0:
                # Fallback to _simple_match for MatchResultWrapper
                simple = getattr(m, '_simple_match', None)
                if simple:
                    idx = getattr(simple, 'segment_index', -1)
        elif isinstance(m, dict):
            idx = m.get('segment_index', -1)
        else:
            idx = getattr(m, 'segment_index', -1)

        # If we still don't have an index, use position in list
        if idx < 0:
            idx = i

        if idx >= 0:
            match_by_segment[idx] = m

    logger.debug(f"[coverage_analyzer] Built match lookup with {len(match_by_segment)} entries")

    high_count = 0
    medium_count = 0
    low_count = 0
    weak_segments: List[WeakSegment] = []
    confidence_values: List[float] = []

    for seg in voiceover_segments:
        seg_idx = getattr(seg, 'index', seg.get('index', -1) if isinstance(seg, dict) else -1)
        seg_text = getattr(seg, 'text', seg.get('text', '') if isinstance(seg, dict) else '')

        match = match_by_segment.get(seg_idx)
        if match:
            # Handle different match object types:
            # 1. MatchResultWrapper (has primary_match.confidence)
            # 2. Simple Match dataclass (has direct confidence)
            # 3. Dict (from serialization)
            if hasattr(match, 'primary_match') and match.primary_match:
                # MatchResultWrapper - get confidence from primary_match
                confidence = getattr(match.primary_match, 'confidence', 0.0)
                video_segment = getattr(match.primary_match, 'video_segment', None)
                video_file = getattr(video_segment, 'source_file', '') if video_segment else ''
            elif isinstance(match, dict):
                confidence = match.get('confidence', 0.0)
                video_file = match.get('video_file', '')
            else:
                # Simple Match dataclass
                confidence = getattr(match, 'confidence', 0.0)
                video_file = getattr(match, 'video_file', '')
        else:
            confidence = 0.0
            video_file = None

        confidence_values.append(confidence)

        if confidence >= target_confidence:
            high_count += 1
        elif confidence >= medium_threshold:
            medium_count += 1
            # Medium confidence segments could still be improved
            weak_segments.append(WeakSegment(
                segment_id=f"S{seg_idx:03d}",
                segment_index=seg_idx,
                text=seg_text[:200],  # Truncate for reporting
                current_confidence=confidence,
                current_match=video_file,
            ))
        else:
            low_count += 1
            weak_segments.append(WeakSegment(
                segment_id=f"S{seg_idx:03d}",
                segment_index=seg_idx,
                text=seg_text[:200],
                current_confidence=confidence,
                current_match=video_file,
            ))

    coverage_ratio = high_count / total if total > 0 else 0.0

    # Log confidence distribution
    if confidence_values:
        avg_conf = sum(confidence_values) / len(confidence_values)
        min_conf = min(confidence_values)
        max_conf = max(confidence_values)
        logger.info(f"[coverage_analyzer] Confidence stats: avg={avg_conf:.2f}, min={min_conf:.2f}, max={max_conf:.2f}")

    # Sort weak segments by confidence (lowest first)
    weak_segments.sort(key=lambda w: w.current_confidence)

    report = CoverageReport(
        total_segments=total,
        high_confidence=high_count,
        medium_confidence=medium_count,
        low_confidence=low_count,
        coverage_ratio=coverage_ratio,
        weak_segments=weak_segments,
        target_confidence=target_confidence,
    )

    logger.info(f"[coverage_analyzer] === COVERAGE ANALYSIS COMPLETE ===")
    logger.info(f"[coverage_analyzer] Total segments: {total}")
    logger.info(f"[coverage_analyzer] High confidence (>={target_confidence:.0%}): {high_count} ({high_count/total*100:.1f}%)")
    logger.info(f"[coverage_analyzer] Medium confidence ({medium_threshold:.0%}-{target_confidence:.0%}): {medium_count} ({medium_count/total*100:.1f}%)")
    logger.info(f"[coverage_analyzer] Low confidence (<{medium_threshold:.0%}): {low_count} ({low_count/total*100:.1f}%)")
    logger.info(f"[coverage_analyzer] Coverage ratio: {coverage_ratio:.1%}")
    logger.info(f"[coverage_analyzer] Weak segments needing improvement: {len(weak_segments)}")

    if weak_segments:
        logger.info(f"[coverage_analyzer] Weakest 5 segments:")
        for ws in weak_segments[:5]:
            logger.info(f"[coverage_analyzer]   {ws.segment_id}: {ws.current_confidence:.2f} - '{ws.text[:50]}...'")

    return report


def get_improvement_delta(
    previous_coverage: float,
    current_coverage: float,
    min_improvement: float = 0.02,
) -> tuple[float, bool]:
    """Calculate improvement between iterations.

    Args:
        previous_coverage: Coverage ratio from previous iteration
        current_coverage: Coverage ratio from current iteration
        min_improvement: Minimum improvement to consider progress

    Returns:
        Tuple of (delta, is_improving)
    """
    delta = current_coverage - previous_coverage
    is_improving = delta >= min_improvement
    return delta, is_improving
