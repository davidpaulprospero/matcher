"""
Match Normalizer - Converts checkpoint match data to MatchResult format.

This module handles the normalization of simple Match objects (restored from
checkpoint) to MatchResult objects expected by create_timeline.

When matches are restored from checkpoint, they're simple Match objects from
state.py with fields: segment_index, video_file, video_start, etc.

create_timeline expects MatchResult objects from utils.py with fields:
primary_match (containing voiceover_segment, video_segment), alternatives, etc.
"""

import logging
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from ..state import PipelineState

from ..utils import Match as UtilsMatch, MatchResult, SRTSegment, AlternativeMatch, StrategyMatch


logger = logging.getLogger(__name__)


class MatchNormalizer:
    """
    Normalizes checkpoint match data to MatchResult format.

    Handles conversion of simple Match objects (restored from checkpoint)
    to MatchResult objects expected by timeline creation functions.
    """

    def normalize(self, state: 'PipelineState') -> List[MatchResult]:
        """
        Normalize matches to MatchResult format.

        When matches are restored from checkpoint, they're simple Match objects
        from state.py with fields: segment_index, video_file, video_start, etc.

        create_timeline expects MatchResult objects from utils.py with fields:
        primary_match (containing voiceover_segment, video_segment), alternatives, etc.

        This method converts simple Match objects to MatchResult objects.

        Args:
            state: PipelineState containing matches to normalize

        Returns:
            List of MatchResult objects
        """
        if not state.matches:
            return []

        # Check if matches are already MatchResult objects
        first_match = state.matches[0]
        if hasattr(first_match, 'primary_match'):
            # Already MatchResult format
            return state.matches

        # Need to convert simple Match objects to MatchResult
        logger.info("Converting checkpoint matches to MatchResult format")

        # Get raw checkpoint dicts stashed during restore (for multi-track data)
        raw_dicts = getattr(state, '_raw_match_dicts', None) or []
        normalized = []
        multi_track_stats = {'alternatives': 0, 'secondary': 0, 'strategy': 0}

        for i, match in enumerate(state.matches):
            if not match:
                continue

            # Get segment_index - simple Match uses segment_index field
            segment_index = getattr(match, 'segment_index', 0)

            # Get voiceover segment from state
            if segment_index < len(state.voiceover_segments):
                vo_seg = state.voiceover_segments[segment_index]
            else:
                # Create minimal voiceover segment
                vo_seg = SRTSegment(
                    index=segment_index,
                    start_time=0.0,
                    end_time=1.0,
                    text="",
                    source_file=""
                )

            # Create video segment from simple Match fields
            video_file = getattr(match, 'video_file', '')
            video_start = getattr(match, 'video_start', 0.0)
            video_end = getattr(match, 'video_end', video_start + 1.0)

            video_seg = SRTSegment(
                index=segment_index,
                start_time=video_start,
                end_time=video_end,
                text="",  # Not preserved in checkpoint
                source_file=video_file
            )

            # Create Match (from utils.py) with the segments
            confidence = getattr(match, 'confidence', 0.5)
            reasoning = getattr(match, 'reason', '')

            utils_match = UtilsMatch(
                voiceover_segment=vo_seg,
                video_segment=video_seg,
                video_scene=None,
                confidence=confidence,
                reasoning=reasoning,
                embedding_similarity=float(getattr(match, 'embedding_similarity', 0.0)),
                is_keyword_match=bool(getattr(match, 'is_keyword_match', False)),
                is_visual_match=bool(getattr(match, 'is_visual_match', False)),
                clip_reuse_count=int(getattr(match, 'clip_reuse_count', 0)),
                match_type=getattr(match, 'match_type', ''),
            )

            # Restore multi-track data from raw checkpoint dicts
            alternatives = []
            secondary_matches = []
            strategy_matches = []
            has_gap = False
            gap_reason = ''

            if i < len(raw_dicts):
                raw = raw_dicts[i]
                for alt_data in raw.get('alternatives', []):
                    try:
                        alternatives.append(AlternativeMatch.from_dict(alt_data))
                    except Exception as e:
                        logger.debug(f"Match {i}: failed to restore alternative: {e}")
                for sec_data in raw.get('secondary_matches', []):
                    try:
                        secondary_matches.append(AlternativeMatch.from_dict(sec_data))
                    except Exception as e:
                        logger.debug(f"Match {i}: failed to restore secondary match: {e}")
                for strat_data in raw.get('strategy_matches', []):
                    try:
                        strategy_matches.append(StrategyMatch.from_dict(strat_data))
                    except Exception as e:
                        logger.debug(f"Match {i}: failed to restore strategy match: {e}")
                has_gap = bool(raw.get('has_gap', False))
                gap_reason = raw.get('gap_reason', '') or ''

            multi_track_stats['alternatives'] += len(alternatives)
            multi_track_stats['secondary'] += len(secondary_matches)
            multi_track_stats['strategy'] += len(strategy_matches)

            # Wrap in MatchResult
            match_result = MatchResult(
                primary_match=utils_match,
                alternatives=alternatives,
                secondary_matches=secondary_matches,
                strategy_matches=strategy_matches,
                has_gap=has_gap,
                gap_reason=gap_reason,
            )

            normalized.append(match_result)

        # Clean up stashed raw dicts
        if hasattr(state, '_raw_match_dicts'):
            del state._raw_match_dicts

        # Log multi-track restoration stats
        total_extras = sum(multi_track_stats.values())
        if total_extras > 0:
            logger.info(
                f"Restored multi-track data: "
                f"{multi_track_stats['alternatives']} alternatives (V2-V3), "
                f"{multi_track_stats['secondary']} secondary (V4-V6), "
                f"{multi_track_stats['strategy']} strategy (V7-V8)"
            )
        else:
            logger.info("No multi-track data in checkpoint (old format or no extras)")

        logger.info(f"Converted {len(normalized)} matches to MatchResult format")
        return normalized
