"""Shared match serialization logic for checkpoint data.

Extracts common match-to-dict logic used by both MATCH and ITERATIVE_MATCH
stages, eliminating duplication and ensuring consistent checkpoint formats.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _extract_match_core(match: Any, index: int) -> Dict[str, Any]:
    """Extract core fields from a Match or MatchResult into a flat dict.

    Handles both:
    - MatchResult (has primary_match with video_segment)
    - Direct Match (has video_segment directly)
    - Gap/fallback matches (empty source_file)

    Returns a dict with normalized keys:
        source_file, video_start, video_end, confidence,
        confidence_variance, matched_keywords, confidence_breakdown,
        strategy, reason, face_score, segment_index
    """
    source_file = ''
    video_start = 0.0
    video_end = 0.0
    confidence = 0.0
    confidence_variance = 0.0
    matched_keywords = []
    confidence_breakdown = []
    strategy = ''
    reason = ''
    face_score = 0.5

    # Path 1: MatchResult with primary_match
    if hasattr(match, 'primary_match') and match.primary_match:
        pm = match.primary_match
        if hasattr(pm, 'video_segment') and pm.video_segment:
            source_file = getattr(pm.video_segment, 'source_file', '') or ''
            video_start = getattr(pm.video_segment, 'start_time', 0.0)
            video_end = getattr(pm.video_segment, 'end_time', 0.0)

        confidence = getattr(pm, 'confidence', 0.0)
        confidence_variance = getattr(match, 'confidence_variance', 0.0)
        matched_keywords = getattr(match, 'matched_keywords', [])
        confidence_breakdown = getattr(match, 'confidence_breakdown', [])
        strategy = getattr(match, 'strategy', getattr(pm, 'reasoning', ''))
        reason = getattr(pm, 'reasoning', '')
        face_score = getattr(match, 'face_score', 0.5)

    # Path 2: Direct Match with video_segment
    elif hasattr(match, 'video_segment') and match.video_segment:
        source_file = getattr(match.video_segment, 'source_file', '') or ''
        video_start = getattr(match.video_segment, 'start_time', 0.0)
        video_end = getattr(match.video_segment, 'end_time', 0.0)
        confidence = getattr(match, 'confidence', 0.0)

    # Path 3: Flat match (e.g., state.Match with video_file directly)
    else:
        source_file = getattr(match, 'video_file', '') or ''
        video_start = getattr(match, 'video_start', 0.0)
        video_end = getattr(match, 'video_end', 0.0)
        confidence = getattr(match, 'confidence', 0.0)

        # Try drilling into primary_match.video_segment as fallback
        if not source_file:
            pm = getattr(match, 'primary_match', None)
            if pm is not None:
                vs = getattr(pm, 'video_segment', None)
                if vs is not None:
                    source_file = getattr(vs, 'source_file', '') or ''
                    video_start = getattr(vs, 'start_time', video_start)
                    video_end = getattr(vs, 'end_time', video_end)
                confidence = getattr(pm, 'confidence', confidence)

        strategy = getattr(match, 'strategy', '')
        reason = getattr(match, 'reason', '')
        face_score = getattr(match, 'face_score', 0.5)

    return {
        'source_file': source_file,
        'video_start': float(video_start),
        'video_end': float(video_end),
        'confidence': float(confidence),
        'confidence_variance': float(confidence_variance),
        'matched_keywords': list(matched_keywords) if matched_keywords else [],
        'confidence_breakdown': list(confidence_breakdown) if confidence_breakdown else [],
        'strategy': strategy,
        'reason': reason,
        'face_score': float(face_score),
        'segment_index': getattr(match, 'segment_index', index),
    }


def _extract_multi_track_data(match: Any) -> Dict[str, Any]:
    """Extract multi-track data (V2-V8) from a MatchResult.

    Returns serialized alternatives, secondary_matches, strategy_matches,
    has_gap, and gap_reason. Returns empty lists for non-MatchResult objects
    (backward compatible).
    """
    result = {
        'alternatives': [],
        'secondary_matches': [],
        'strategy_matches': [],
        'has_gap': False,
        'gap_reason': '',
    }

    # Only MatchResult has these fields
    if not hasattr(match, 'alternatives'):
        return result

    for alt in getattr(match, 'alternatives', []) or []:
        try:
            result['alternatives'].append(alt.to_dict())
        except Exception as e:
            logger.warning(f"Failed to serialize alternative: {e}")

    for sec in getattr(match, 'secondary_matches', []) or []:
        try:
            result['secondary_matches'].append(sec.to_dict())
        except Exception as e:
            logger.warning(f"Failed to serialize secondary match: {e}")

    for strat in getattr(match, 'strategy_matches', []) or []:
        try:
            result['strategy_matches'].append(strat.to_dict())
        except Exception as e:
            logger.warning(f"Failed to serialize strategy match: {e}")

    result['has_gap'] = bool(getattr(match, 'has_gap', False))
    result['gap_reason'] = getattr(match, 'gap_reason', '') or ''

    return result


def serialize_match_for_match_stage(match: Any, index: int) -> Dict[str, Any]:
    """Serialize a match for MATCH stage checkpoint format.

    Output keys: segment_index, source_file, start_time, confidence,
                 confidence_variance, matched_keywords, confidence_breakdown
    """
    core = _extract_match_core(match, index)
    result = {
        'segment_index': index,
        'source_file': core['source_file'],
        'start_time': core['video_start'],
        'confidence': core['confidence'],
        'confidence_variance': core['confidence_variance'],
        'matched_keywords': core['matched_keywords'],
        'confidence_breakdown': core['confidence_breakdown'],
    }
    result.update(_extract_multi_track_data(match))
    return result


def serialize_match_for_iterative_stage(match: Any, index: int) -> Dict[str, Any]:
    """Serialize a match for ITERATIVE_MATCH stage checkpoint format.

    Output keys: segment_index, video_file, video_start, video_end,
                 confidence, strategy, reason, face_score
    """
    core = _extract_match_core(match, index)
    result = {
        'segment_index': core['segment_index'],
        'video_file': core['source_file'],
        'video_start': core['video_start'],
        'video_end': core['video_end'],
        'confidence': core['confidence'],
        'strategy': core['strategy'],
        'reason': core['reason'],
        'face_score': core['face_score'],
    }
    result.update(_extract_multi_track_data(match))
    return result


def is_empty_source(match: Any, index: int) -> bool:
    """Check if a match has an empty source file (useful for logging)."""
    core = _extract_match_core(match, index)
    return not core['source_file']
