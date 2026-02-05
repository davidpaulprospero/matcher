"""
Iterative Match Stage - Multi-Pass Gap Filling

Stage 5 of the simplified 7-stage pipeline. Runs after MATCH to achieve
90%+ confidence across all segments while enforcing source spacing rules.

Features:
- Identifies gaps (low confidence or source spacing violation)
- Locks high-confidence matches for preservation
- Multi-strategy query generation (voiceover text, similar-to-locked, entity/topic)
- Progressive query refinement on subsequent passes
- Gap pattern analysis for smarter search
- Query learning database for cross-project improvement
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from . import Stage, StageResult, StageMetrics, register_stage, validate_required_state_attrs
from ..downloader.search_cache import SearchResultsCache
from ..matching.serialization import serialize_match_for_iterative_stage, is_empty_source
from ..utils import extract_video_id

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, Match

logger = logging.getLogger(__name__)


@dataclass
class LockedMatch:
    """A segment with a locked (high-confidence, spacing-valid) match."""
    segment_index: int
    video_id: str
    confidence: float
    position: float  # Timeline position (start time)
    title: str = ""
    source_file: str = ""


@dataclass
class GapSegment:
    """A segment that needs a better match."""
    segment_index: int
    confidence: float
    voiceover_text: str
    position: float
    current_video_id: str = ""
    reason: str = ""  # Why it's a gap: 'low_confidence', 'spacing_violation'


@dataclass
class PassMetrics:
    """Metrics for a single iterative pass."""
    pass_number: int
    initial_gaps: int
    final_gaps: int
    gaps_filled: int
    new_videos_found: int
    queries_executed: int
    duration_seconds: float = 0.0
    strategy_breakdown: Dict[str, int] = field(default_factory=dict)


@register_stage
class IterativeMatchStage(Stage):
    """
    Fill matching gaps with iterative search passes.

    Inputs:
        - state.matches: Initial matches from MATCH stage
        - state.voiceover_segments: Voiceover segments to match
        - state.text_metadata: Video segment metadata

    Outputs:
        - state.matches: Updated matches with gaps filled
        - Checkpoint data with iteration metrics

    Configuration (config.iterative_matching):
        - enabled: Enable/disable stage
        - target_confidence: Confidence threshold for locking (0.90)
        - source_spacing_seconds: Minimum seconds between same source (300)
        - max_iterations: Maximum passes to attempt (5)
        - min_gap_percentage: Stop early if below this (0.05)
    """

    name = "ITERATIVE_MATCH"
    description = "Fill matching gaps with iterative search passes"

    def __init__(self):
        """Initialize the stage with cookie rotator."""
        super().__init__()
        self._cookie_rotator = None
        # Cross-pass tracking (reset per run)
        self._fetched_video_ids: Set[str] = set()
        self._used_queries: Set[str] = set()

    def _get_cookie_rotator(self, config: 'Config'):
        """Get or initialize the cookie rotator for YouTube authentication."""
        if self._cookie_rotator is None:
            from ..downloader.cookie_rotator import CookieRotator
            cookie_rotation_config = getattr(config.download, 'cookie_rotation', None)
            if cookie_rotation_config and getattr(cookie_rotation_config, 'enabled', False):
                self._cookie_rotator = CookieRotator(cookie_rotation_config)
        return self._cookie_rotator

    def _get_cookie_args(self, config: 'Config', log_usage: bool = False) -> List[str]:
        """
        Get yt-dlp cookie arguments for YouTube authentication.

        Uses CookieRotator if enabled (for rotation), otherwise falls back
        to static cookie configuration from get_cookies_args.

        Args:
            config: Pipeline config
            log_usage: If True, log which cookie source is being used

        Returns:
            List of cookie arguments for yt-dlp (empty if no cookies configured)
        """
        # Try cookie rotator first (for rotation support)
        rotator = self._get_cookie_rotator(config)
        if rotator and rotator.is_enabled:
            current_cookie = rotator.get_current_cookie()
            if current_cookie:
                if log_usage:
                    from pathlib import Path
                    cookie_name = Path(current_cookie).stem
                    logger.info(f"Using rotated cookie: {cookie_name}")
                return ['--cookies', current_cookie]

        # Fall back to static cookie configuration
        from ..downloader.utils import get_cookies_args
        static_args = get_cookies_args(config)
        if log_usage:
            if static_args:
                logger.info(f"Using static cookie config")
            else:
                logger.debug("No cookies configured for iterative match")
        return static_args

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the iterative matching stage.

        US-39-009: Validates state type and converts legacy objects if needed.
        US-40-008: Validates required state attributes exist.
        """
        # US-39-009: Validate state type at stage entry
        state = self._validate_state_type(state)

        # US-40-008: Validate required attributes exist
        validate_required_state_attrs(
            state,
            ['matches', 'text_metadata'],
            self.name
        )

        # US-40-009: Pre-check candidate pool size
        candidate_count = len(state.text_metadata) if state.text_metadata else 0
        logger.info(f"IterativeMatch starting with {candidate_count} candidates")

        if candidate_count == 0:
            logger.warning("No candidates available in text_metadata - skipping iterative matching")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_candidates',
                'candidate_count': 0
            })

        warnings = []

        # Get config (with fallback defaults)
        iter_config = getattr(config, 'iterative_matching', None)
        if not iter_config:
            logger.info("iterative_matching config not found, using defaults")
            from ..config.sections.iterative_matching import IterativeMatchingConfig
            iter_config = IterativeMatchingConfig()

        # Check if enabled
        if not getattr(iter_config, 'enabled', True):
            logger.info("Iterative matching disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'disabled'
            })

        # Validate inputs
        if not state.matches:
            logger.warning("No matches to iterate on")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_matches'
            })

        if not state.voiceover_segments:
            logger.warning("No voiceover segments")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_voiceover'
            })

        print(f"\n  ─── Stage: ITERATIVE MATCHING ───")

        try:
            # Get thresholds from config
            target_conf = getattr(iter_config, 'target_confidence', 0.90)
            source_spacing = getattr(iter_config, 'source_spacing_seconds', 300.0)
            max_iterations = getattr(iter_config, 'max_iterations', 5)
            min_gap_pct = getattr(iter_config, 'min_gap_percentage', 0.05)

            print(f"  Target confidence: {target_conf:.0%}")
            print(f"  Source spacing: {source_spacing:.0f}s")
            print(f"  Max iterations: {max_iterations}")

            # Initialize learning DB
            learning_db = None
            if getattr(iter_config, 'enable_query_learning', True):
                from ..iterative_match import QueryLearningDB
                db_path = getattr(iter_config, 'learning_db_path', '.cache/query_learning.json')
                learning_db = QueryLearningDB(str(Path(state.voiceover_path).parent / db_path))

            # Run iterative passes
            all_pass_metrics = []
            all_gap_pattern_logs = []  # US-63-012: Track gap patterns across passes
            total_start = time.time()
            gap_analysis = None  # Initialize before loop for early break case

            # Reset cross-pass tracking for this run
            self._fetched_video_ids = set()
            self._used_queries = set()

            # Initialize local embedding storage (no longer stored on PipelineState)
            self._embeddings = None
            self._embedding_index = None

            for pass_num in range(1, max_iterations + 1):
                pass_start = time.time()
                print(f"\n  Pass {pass_num}/{max_iterations}...")

                # 1. Identify gaps and locks
                locked, gaps = self._identify_gaps_and_locks(
                    state.matches,
                    state.voiceover_segments,
                    target_conf,
                    source_spacing
                )

                gap_count = len(gaps)
                total_count = len(state.voiceover_segments)
                gap_pct = gap_count / total_count if total_count > 0 else 0

                print(f"    Locked: {len(locked)} | Gaps: {gap_count} ({gap_pct:.1%})")

                # 2. Check stop conditions
                if gap_count == 0:
                    print(f"  ✓ No gaps remaining, stopping early")
                    break

                if gap_pct < min_gap_pct:
                    print(f"  ✓ Below {min_gap_pct:.0%} threshold, stopping")
                    break

                # 3. Analyze gap patterns
                gap_analysis = None
                gap_pattern_log = None
                if getattr(iter_config, 'analyze_gap_patterns', True):
                    from ..iterative_match import (
                        analyze_gaps,
                        GapSegment as GapSeg,
                        analyze_gap_patterns_for_logging,
                        log_gap_pattern_analysis,
                    )
                    gap_segments = [
                        GapSeg(
                            segment_index=g.segment_index,
                            confidence=g.confidence,
                            voiceover_text=g.voiceover_text,
                            position=g.position
                        )
                        for g in gaps
                    ]
                    gap_analysis = analyze_gaps(
                        gap_segments,
                        state,
                        state.extracted_entities
                    )
                    dominant = gap_analysis.get_dominant_pattern()
                    print(f"    Dominant gap pattern: {dominant}")

                    # US-63-012: Detailed gap pattern analysis logging
                    total_duration = 0.0
                    if state.voiceover_segments:
                        # Get total duration from last segment
                        last_seg = state.voiceover_segments[-1]
                        total_duration = getattr(last_seg, 'end', 0.0) or getattr(last_seg, 'end_time', 0.0)

                    gap_pattern_log = analyze_gap_patterns_for_logging(
                        gap_segments,
                        pass_num,
                        total_duration
                    )
                    log_gap_pattern_analysis(gap_pattern_log, logger)

                    # Print query hints to console
                    if gap_pattern_log.query_hints:
                        for hint in gap_pattern_log.query_hints[:3]:  # Limit to 3 hints
                            print(f"    💡 {hint}")

                    # US-63-012: Track gap pattern log for checkpoint storage
                    all_gap_pattern_logs.append(gap_pattern_log)

                # 4. Generate search queries
                queries = self._generate_multi_strategy_queries(
                    gaps, locked, state, iter_config, pass_num, gap_analysis
                )

                if not queries:
                    print(f"    No queries generated, stopping")
                    warnings.append(f"Pass {pass_num}: No queries generated")
                    break

                print(f"    Generated {len(queries)} search queries"
                      f" (excluding {len(self._fetched_video_ids)} videos already fetched)")

                # 5. Apply progressive refinement on subsequent passes
                if pass_num > 1 and getattr(iter_config, 'enable_progressive_refinement', True):
                    queries = self._refine_queries_progressive(
                        queries, pass_num, all_pass_metrics, learning_db
                    )

                # 6. Execute searches with streaming/batched caption fetching
                # Process captions in batches to avoid overwhelming the pipeline
                batch_size = getattr(iter_config, 'caption_batch_size', 10)
                fetch_delay = getattr(iter_config, 'caption_fetch_delay', 0.5)
                
                all_new_candidates = []
                gaps_filled_total = 0
                
                # Search first to get video IDs
                video_ids = self._search_youtube_for_videos(
                    queries, state, config, iter_config
                )
                
                if video_ids:
                    print(f"    Found {len(video_ids)} new video candidates")
                    
                    # Process captions in batches
                    for batch_start in range(0, len(video_ids), batch_size):
                        batch_end = min(batch_start + batch_size, len(video_ids))
                        batch_ids = video_ids[batch_start:batch_end]
                        
                        print(f"    Fetching captions batch {batch_start//batch_size + 1} "
                              f"({len(batch_ids)} videos)...")
                        
                        batch_candidates = self._fetch_captions_for_videos(
                            batch_ids, config, iter_config
                        )
                        
                        if batch_candidates:
                            all_new_candidates.extend(batch_candidates)
                            
                            # Try to fill gaps with this batch before continuing
                            gaps_filled = self._rematch_gaps(
                                gaps, locked, batch_candidates, state, config
                            )
                            gaps_filled_total += gaps_filled
                            
                            if gaps_filled > 0:
                                print(f"      Filled {gaps_filled} gaps with this batch")
                            
                            # Early exit if all gaps are filled
                            remaining_gaps = len(gaps) - gaps_filled_total
                            if remaining_gaps <= 0:
                                print(f"    All gaps filled, stopping caption fetch early")
                                break
                        
                        # Small delay between batches to avoid rate limiting
                        if batch_end < len(video_ids) and fetch_delay > 0:
                            time.sleep(fetch_delay)
                
                new_candidates = all_new_candidates
                gaps_filled = gaps_filled_total
                
                # 7. Log final results
                if gaps_filled > 0:
                    print(f"    Filled {gaps_filled} gaps with new matches")

                # 8. Update learning DB
                if learning_db and gap_analysis:
                    self._update_query_learning(
                        queries, gaps_filled, gap_analysis, learning_db
                    )

                # Record pass metrics
                pass_duration = time.time() - pass_start
                pass_metrics = PassMetrics(
                    pass_number=pass_num,
                    initial_gaps=gap_count,
                    final_gaps=gap_count - gaps_filled,
                    gaps_filled=gaps_filled,
                    new_videos_found=len(new_candidates),
                    queries_executed=len(queries),
                    duration_seconds=pass_duration
                )
                all_pass_metrics.append(pass_metrics)

                # Check if no progress
                if gaps_filled == 0 and pass_num >= 2:
                    print(f"  ✓ No progress in pass {pass_num}, stopping")
                    break

            # Save learning DB
            if learning_db:
                learning_db.save()

            # Final summary
            total_duration = time.time() - total_start

            # Calculate final stats
            final_locked, final_gaps = self._identify_gaps_and_locks(
                state.matches,
                state.voiceover_segments,
                target_conf,
                source_spacing
            )

            initial_gaps = all_pass_metrics[0].initial_gaps if all_pass_metrics else 0
            final_gap_count = len(final_gaps)
            total_filled = initial_gaps - final_gap_count

            print(f"\n  ─── Iterative Matching Complete ───")
            print(f"  Passes run: {len(all_pass_metrics)}")
            print(f"  Gaps filled: {total_filled}")
            print(f"  Final gaps: {final_gap_count} ({final_gap_count / len(state.voiceover_segments):.1%})")
            print(f"  Total time: {total_duration:.1f}s")

            # Serialize matches for checkpoint (preserves iterative improvements)
            # Carry forward multi-track data (V2-V8) from _raw_match_dicts since
            # flat Match objects don't store alternatives/secondary/strategy data
            raw_match_dicts = getattr(state, '_raw_match_dicts', None) or []
            serialized_matches = []
            empty_video_file_count = 0
            multi_track_carried = 0
            for i, match in enumerate(state.matches):
                try:
                    if is_empty_source(match, i):
                        logger.warning(
                            f"Match {i}: empty video_file after all fallback attempts "
                            f"(type={type(match).__name__})"
                        )
                        empty_video_file_count += 1

                    serialized = serialize_match_for_iterative_stage(match, i)

                    # If serialization produced empty multi-track data but we have
                    # raw dicts from a prior stage, carry forward the multi-track data
                    if (i < len(raw_match_dicts)
                            and not serialized.get('alternatives')
                            and not serialized.get('secondary_matches')):
                        raw = raw_match_dicts[i]
                        for key in ('alternatives', 'secondary_matches', 'strategy_matches',
                                    'has_gap', 'gap_reason'):
                            if key in raw and raw[key]:
                                serialized[key] = raw[key]
                        if raw.get('alternatives') or raw.get('secondary_matches'):
                            multi_track_carried += 1

                    serialized_matches.append(serialized)
                except Exception as e:
                    logger.warning(f"Failed to serialize match {i}: {e}")

            logger.info(
                f"Serialized {len(serialized_matches)} matches, "
                f"{empty_video_file_count} had empty video_file (skipped), "
                f"{multi_track_carried} carried forward multi-track data from prior stage"
            )

            # Build checkpoint data
            checkpoint_data = {
                'passes_completed': len(all_pass_metrics),
                'initial_gaps': initial_gaps,
                'final_gaps': final_gap_count,
                'total_gaps_filled': total_filled,
                'total_duration_seconds': total_duration,
                'matches': serialized_matches,  # Save modified matches
                'pass_metrics': [
                    {
                        'pass_number': pm.pass_number,
                        'gaps_filled': pm.gaps_filled,
                        'queries_executed': pm.queries_executed,
                        'duration_seconds': pm.duration_seconds,
                    }
                    for pm in all_pass_metrics
                ],
            }

            # Add gap analysis if available
            if gap_analysis:
                checkpoint_data['gap_analysis'] = gap_analysis.to_dict()

            # US-63-012: Store gap pattern logs for cross-run learning
            if all_gap_pattern_logs:
                checkpoint_data['gap_pattern_history'] = [
                    log.to_dict() for log in all_gap_pattern_logs
                ]

            # Stage metrics
            metrics = StageMetrics(
                items_processed=len(state.voiceover_segments),
                items_failed=final_gap_count,
                duration_seconds=total_duration
            )

            return StageResult.ok(checkpoint_data, warnings, metrics)

        except Exception as e:
            logger.error(f"Iterative matching failed: {e}", exc_info=True)
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if this stage can be skipped."""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore iterative match stage from checkpoint."""
        from ..state import restore_matches_from_dicts

        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            # US-51-008: Validate checkpoint data schema before restoring
            if not isinstance(data, dict):
                logger.warning(f"ITERATIVE_MATCH restore: expected dict, got {type(data).__name__}")
                return False

            if 'matches' in data and not isinstance(data['matches'], list):
                logger.warning(f"ITERATIVE_MATCH restore: 'matches' expected list, got {type(data['matches']).__name__}")
                return False

            # Restore matches using shared helper (same validation as MATCH stage)
            matches_data = data.get('matches', [])
            if matches_data:
                restored_matches = restore_matches_from_dicts(
                    matches_data, default_strategy='iterative_restored', logger_instance=logger
                )
                if restored_matches is None:
                    return False

                if restored_matches:
                    state.matches = restored_matches

                    # Merge multi-track data from MATCH checkpoint if iterative
                    # checkpoint has empty multi-track (legacy checkpoints before
                    # carry-forward fix didn't preserve V2-V8 data)
                    has_multi_track = any(
                        m.get('alternatives') or m.get('secondary_matches')
                        for m in matches_data[:5]  # Check first few
                    )
                    if not has_multi_track:
                        match_stage_data = checkpoint.get_stage_data('MATCH')
                        if match_stage_data and isinstance(match_stage_data, dict):
                            match_dicts = match_stage_data.get('matches', [])
                            if match_dicts and len(match_dicts) == len(matches_data):
                                # Merge multi-track keys from MATCH into iterative dicts
                                merged_count = 0
                                for iter_d, match_d in zip(matches_data, match_dicts):
                                    for key in ('alternatives', 'secondary_matches',
                                                'strategy_matches', 'has_gap', 'gap_reason'):
                                        if key in match_d and match_d[key] and not iter_d.get(key):
                                            iter_d[key] = match_d[key]
                                    if match_d.get('alternatives') or match_d.get('secondary_matches'):
                                        merged_count += 1
                                if merged_count:
                                    logger.info(
                                        f"Merged multi-track data from MATCH checkpoint "
                                        f"into {merged_count} iterative matches"
                                    )

                    state._raw_match_dicts = matches_data
                    logger.info(
                        f"Restored {self.name}: {len(restored_matches)} matches, "
                        f"{data.get('passes_completed', 0)} passes, "
                        f"{data.get('total_gaps_filled', 0)} gaps filled"
                    )
                    return True

            # Fallback: no matches in iterative data, rely on MATCH stage
            logger.info(
                f"Restored {self.name} metadata (no matches data): "
                f"{data.get('passes_completed', 0)} passes"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to restore {self.name}: {e}", exc_info=True)
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running."""
        if not state.matches:
            return "No matches from MATCH stage. Run MATCH stage first."
        if not state.voiceover_segments:
            return "No voiceover segments. Run ANALYZE stage first."
        return None

    # =========================================================================
    # Core Algorithm Methods
    # =========================================================================

    def _identify_gaps_and_locks(
        self,
        matches: List['Match'],
        vo_segments: List[Any],
        target_conf: float,
        source_spacing: float
    ) -> Tuple[List[LockedMatch], List[GapSegment]]:
        """
        Classify segments as locked or gap based on confidence + source spacing.

        A segment is LOCKED if:
        - Confidence >= target_conf AND
        - Same video source not used within source_spacing seconds

        A segment is a GAP if:
        - Confidence < target_conf OR
        - Source spacing violated

        Args:
            matches: List of Match objects from MATCH stage
            vo_segments: List of VoiceoverSegment objects
            target_conf: Confidence threshold for locking
            source_spacing: Minimum seconds between same source

        Returns:
            Tuple of (locked_list, gaps_list)
        """
        # Track video source usage by timeline position
        source_timeline: Dict[str, List[float]] = defaultdict(list)

        locked: List[LockedMatch] = []
        gaps: List[GapSegment] = []

        # Build segment lookup for text access
        segment_texts = {}
        for seg in vo_segments:
            idx = getattr(seg, 'index', None) or getattr(seg, 'segment_index', 0)
            text = getattr(seg, 'text', '')
            start = getattr(seg, 'start', 0.0)
            segment_texts[idx] = {'text': text, 'start': start}

        for i, match in enumerate(matches):
            # Extract match details
            segment_idx = i
            if hasattr(match, 'segment_index'):
                segment_idx = match.segment_index

            # Get confidence
            conf = 0.0
            if hasattr(match, 'primary_match') and match.primary_match:
                conf = getattr(match.primary_match, 'confidence', 0.0)
            elif hasattr(match, 'confidence'):
                conf = match.confidence

            # Get video source
            video_id = self._extract_video_id(match)

            # Get position from voiceover segment
            seg_info = segment_texts.get(segment_idx, {'text': '', 'start': 0.0})
            position = seg_info['start']
            vo_text = seg_info['text']

            # Check for source spacing violation
            spacing_violation = False
            if video_id:
                for prev_pos in source_timeline[video_id]:
                    if abs(position - prev_pos) < source_spacing:
                        spacing_violation = True
                        break

            # Classify
            if conf >= target_conf and not spacing_violation:
                # LOCKED
                locked.append(LockedMatch(
                    segment_index=segment_idx,
                    video_id=video_id,
                    confidence=conf,
                    position=position,
                    source_file=self._get_source_file(match)
                ))
                # Record position for spacing check
                source_timeline[video_id].append(position)
            else:
                # GAP
                reason = 'spacing_violation' if spacing_violation else 'low_confidence'
                gaps.append(GapSegment(
                    segment_index=segment_idx,
                    confidence=conf,
                    voiceover_text=vo_text,
                    position=position,
                    current_video_id=video_id,
                    reason=reason
                ))

        return locked, gaps

    def _extract_video_id(self, match: Any) -> str:
        """Extract YouTube video ID from match object."""
        source_file = ""

        if hasattr(match, 'primary_match') and match.primary_match:
            pm = match.primary_match
            if hasattr(pm, 'video_segment') and pm.video_segment:
                source_file = getattr(pm.video_segment, 'source_file', '')
        elif hasattr(match, 'video_file'):
            source_file = match.video_file

        if not source_file:
            return ""

        return extract_video_id(source_file) or Path(source_file).stem

    def _get_source_file(self, match: Any) -> str:
        """Get source file path from match object."""
        if hasattr(match, 'primary_match') and match.primary_match:
            pm = match.primary_match
            if hasattr(pm, 'video_segment') and pm.video_segment:
                return getattr(pm.video_segment, 'source_file', '')
        elif hasattr(match, 'video_file'):
            return match.video_file
        return ""

    def _generate_multi_strategy_queries(
        self,
        gaps: List[GapSegment],
        locked: List[LockedMatch],
        state: 'PipelineState',
        config: Any,
        pass_num: int,
        gap_analysis: Any = None
    ) -> List[Dict[str, Any]]:
        """
        Generate queries using multiple strategies.

        Strategies:
        1. Voiceover text keywords - extract keywords from gap text
        2. Similar-to-locked - find videos like successful matches
        3. Entity + topic queries - use extracted entities

        Args:
            gaps: Gap segments to generate queries for
            locked: Locked matches (for similar-to-locked strategy)
            state: Pipeline state
            config: Iterative matching config
            pass_num: Current pass number
            gap_analysis: Optional gap pattern analysis

        Returns:
            List of query dictionaries
        """
        queries = []

        # Strategy 1: Voiceover text keywords
        if getattr(config, 'use_voiceover_text_queries', True):
            from ..iterative_match.gap_analyzer import extract_keywords_for_gap
            from ..iterative_match import GapSegment as GapSeg

            for gap in gaps[:20]:  # Limit to avoid too many queries
                gap_obj = GapSeg(
                    segment_index=gap.segment_index,
                    confidence=gap.confidence,
                    voiceover_text=gap.voiceover_text,
                    position=gap.position,
                    pattern_type=gap_analysis.clustered_gaps.get(gap.segment_index, 'other')
                    if gap_analysis else 'other'
                )
                keywords = extract_keywords_for_gap(gap_obj, max_keywords=5)
                if keywords:
                    queries.append({
                        'query': ' '.join(keywords),
                        'strategy': 'voiceover',
                        'gap_indices': [gap.segment_index],
                        'priority': 2
                    })

        # Strategy 2: Similar-to-locked
        if getattr(config, 'use_similar_to_locked', True) and locked:
            for gap in gaps[:15]:
                # Find nearest locked match
                nearest = self._find_nearest_locked(gap, locked)
                if nearest and nearest.video_id:
                    queries.append({
                        'query': f'similar:{nearest.video_id}',
                        'strategy': 'similar_locked',
                        'gap_indices': [gap.segment_index],
                        'seed_video_id': nearest.video_id,
                        'priority': 3  # Higher priority
                    })

        # Strategy 3: Entity + topic queries
        if getattr(config, 'use_entity_topic_queries', True):
            entities = state.extracted_entities or []
            for gap in gaps[:15]:
                # Find entities mentioned in gap text
                gap_text_lower = gap.voiceover_text.lower()
                relevant_entities = []
                for entity in entities:
                    name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
                    if name and name.lower() in gap_text_lower:
                        relevant_entities.append(name)

                if relevant_entities:
                    queries.append({
                        'query': f'{relevant_entities[0]} footage video',
                        'strategy': 'entity',
                        'gap_indices': [gap.segment_index],
                        'priority': 2
                    })

        # Deduplicate by query string AND exclude already-used queries
        seen_queries = set()
        unique_queries = []
        skipped_as_used = 0
        for q in queries:
            query_key = q['query'].lower().strip()
            if query_key in self._used_queries:
                skipped_as_used += 1
            elif query_key not in seen_queries:
                seen_queries.add(query_key)
                unique_queries.append(q)

        # On subsequent passes, generate diverse variants of remaining gaps
        if pass_num > 1 and len(unique_queries) < 10:
            variant_queries = self._generate_query_variants(gaps, pass_num)
            for vq in variant_queries:
                vq_key = vq['query'].lower().strip()
                if vq_key not in seen_queries and vq_key not in self._used_queries:
                    seen_queries.add(vq_key)
                    unique_queries.append(vq)

        # Sort by priority
        unique_queries.sort(key=lambda x: x.get('priority', 0), reverse=True)

        # Limit total queries
        max_queries = getattr(config, 'max_new_videos_per_pass', 50) // 2
        final_queries = unique_queries[:max_queries]

        # Log deduplication stats if any queries were skipped
        if skipped_as_used > 0:
            logger.info(f"Skipped {skipped_as_used} queries already used in previous passes")

        # Track these queries as used for future passes
        for q in final_queries:
            self._used_queries.add(q['query'].lower().strip())

        return final_queries

    def _find_nearest_locked(
        self,
        gap: GapSegment,
        locked: List[LockedMatch],
        max_distance: float = 60.0
    ) -> Optional[LockedMatch]:
        """Find the locked match nearest to a gap in timeline."""
        nearest = None
        min_distance = float('inf')

        for lock in locked:
            distance = abs(gap.position - lock.position)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest = lock

        return nearest

    def _generate_query_variants(
        self,
        gaps: List[GapSegment],
        pass_num: int
    ) -> List[Dict[str, Any]]:
        """
        Generate diverse query variants for subsequent passes.

        Uses different framings and vocabulary to search for the same concepts
        in different ways, increasing the chance of finding useful footage.

        Args:
            gaps: Gap segments to generate variants for
            pass_num: Current pass number (affects variant strategy)

        Returns:
            List of variant query dictionaries
        """
        import re
        variants = []

        # Different framing suffixes per pass
        pass_suffixes = {
            2: ['documentary', 'stock footage', 'archive'],
            3: ['b-roll', 'cinematic', 'film'],
            4: ['historical', 'raw footage', 'clip'],
            5: ['4K', 'HD footage', 'professional'],
        }
        suffixes = pass_suffixes.get(pass_num, ['footage'])

        # Extract nouns and key phrases from gaps
        for gap in gaps[:15]:
            text = gap.voiceover_text
            if not text or len(text) < 10:
                continue

            # Extract capitalized words (likely proper nouns/entities)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

            # Extract longer words (likely meaningful content words)
            words = text.split()
            content_words = [w.strip('.,!?;:') for w in words if len(w) > 5]

            # Strategy varies by pass
            if pass_num == 2:
                # Focus on proper nouns with different framing
                for noun in proper_nouns[:2]:
                    for suffix in suffixes[:2]:
                        query = f'{noun} {suffix}'
                        if query.lower().strip() not in self._used_queries:
                            variants.append({
                                'query': query,
                                'strategy': 'variant_noun',
                                'gap_indices': [gap.segment_index],
                                'priority': 1
                            })

            elif pass_num == 3:
                # Use content words with visual descriptors
                if content_words:
                    for word in content_words[:3]:
                        for suffix in suffixes[:2]:
                            query = f'{word} {suffix}'
                            if query.lower().strip() not in self._used_queries:
                                variants.append({
                                    'query': query,
                                    'strategy': 'variant_content',
                                    'gap_indices': [gap.segment_index],
                                    'priority': 1
                                })

            else:  # pass 4+
                # Broader searches with just key terms
                all_terms = list(set(proper_nouns + content_words[:3]))
                if all_terms:
                    for term in all_terms[:2]:
                        query = f'{term} {suffixes[0] if suffixes else "footage"}'
                        if query.lower().strip() not in self._used_queries:
                            variants.append({
                                'query': query,
                                'strategy': 'variant_broad',
                                'gap_indices': [gap.segment_index],
                                'priority': 1
                            })

        return variants

    def _refine_queries_progressive(
        self,
        queries: List[Dict[str, Any]],
        pass_num: int,
        previous_metrics: List[PassMetrics],
        learning_db: Any
    ) -> List[Dict[str, Any]]:
        """
        Modify failing queries based on pass number and learning.

        Refinements:
        - Pass 2+: Add synonyms to keywords
        - Pass 3+: Broaden scope (remove specific terms)
        - Use learned successful templates

        Args:
            queries: Queries to refine
            pass_num: Current pass number
            previous_metrics: Metrics from prior passes
            learning_db: Query learning database

        Returns:
            Refined query list
        """
        if pass_num < 2:
            return queries

        refined = []
        for q in queries:
            query_text = q['query']

            # Simple refinements
            if pass_num >= 2:
                # Add 'footage' or 'video' if not present
                if 'footage' not in query_text.lower() and 'video' not in query_text.lower():
                    query_text = f"{query_text} footage"

            if pass_num >= 3:
                # Broaden: remove very specific terms (short words)
                words = query_text.split()
                words = [w for w in words if len(w) > 3 or w.lower() in {'the', 'and', 'for'}]
                if len(words) >= 2:
                    query_text = ' '.join(words)

            # Apply learning if available
            if learning_db:
                refined_template = learning_db.get_refined_template(query_text, pass_num)
                if refined_template:
                    query_text = refined_template

            refined.append({
                **q,
                'query': query_text,
                'refined': True
            })

        return refined

    def _search_youtube_for_videos(
        self,
        queries: List[Dict[str, Any]],
        state: 'PipelineState',
        config: 'Config',
        iter_config: Any
    ) -> List[str]:
        """
        Search YouTube for videos matching the queries.

        Returns a list of video IDs without fetching captions.
        This allows for batched/streamed caption fetching later.

        Args:
            queries: Search queries to execute
            state: Pipeline state
            config: Main config
            iter_config: Iterative matching config

        Returns:
            List of YouTube video IDs found
        """
        import subprocess
        import json

        # Get existing video IDs to avoid duplicates
        existing_ids: Set[str] = set()

        # From state.video_ids (simplified pipeline)
        if hasattr(state, 'video_ids') and state.video_ids:
            existing_ids.update(state.video_ids)

        # From text_metadata (caption-first mode)
        for meta in (state.text_metadata or []):
            if isinstance(meta, dict):
                vid_path = meta.get('video_path', '')
                vid_id = self._extract_video_id_from_path(vid_path)
                if vid_id:
                    existing_ids.add(vid_id)

        # Backward compatibility: check legacy downloaded_videos
        if hasattr(state, 'downloaded_videos'):
            for vid in state.downloaded_videos:
                vid_file = vid.file if hasattr(vid, 'file') else vid.get('file', '')
                vid_id = self._extract_video_id_from_path(vid_file)
                if vid_id:
                    existing_ids.add(vid_id)

        # Include videos already fetched in previous passes (cross-pass dedup)
        existing_ids.update(self._fetched_video_ids)

        # Search settings
        results_per_query = getattr(iter_config, 'search_results_per_gap', 10)
        max_new_videos = getattr(iter_config, 'max_new_videos_per_pass', 50)

        # Duration filter (typical for documentary footage)
        min_duration = 30
        max_duration = 600  # 10 minutes max

        # Initialize search cache
        cache_ttl = getattr(iter_config, 'search_cache_ttl_hours', 24)
        search_cache = SearchResultsCache(ttl_hours=cache_ttl)

        # Collect unique video IDs from search
        new_video_ids: Set[str] = set()
        query_video_map: Dict[str, List[str]] = {}  # Track which query found which videos
        cache_hits = 0
        cache_misses = 0

        # Log cookie being used for YouTube searches
        self._get_cookie_args(config, log_usage=True)

        print(f"    Searching YouTube ({len(queries)} queries, {results_per_query} results each)...")

        for q in queries:
            if len(new_video_ids) >= max_new_videos:
                break

            query_text = q['query']
            if not query_text or query_text.startswith('similar:'):
                # Skip empty or similar-to queries (not supported in basic search)
                continue

            try:
                # Check search cache first
                cached_videos = search_cache.get_search_result(
                    keyword=query_text,
                    tier='iterative',
                    search_pool=results_per_query
                )

                query_vids = []

                if cached_videos is not None:
                    # Use cached results
                    cache_hits += 1
                    for vid_info in cached_videos:
                        vid_id = vid_info.get('id', '')
                        if vid_id and vid_id not in existing_ids and vid_id not in new_video_ids:
                            new_video_ids.add(vid_id)
                            query_vids.append(vid_id)
                            if len(new_video_ids) >= max_new_videos:
                                break
                else:
                    # Cache miss - search YouTube
                    cache_misses += 1

                    # Use yt-dlp to search YouTube (metadata only, no download)
                    cmd = [
                        'yt-dlp',
                        '--ignore-config',
                        f'ytsearch{results_per_query}:{query_text}',
                        '--dump-json',
                        '--flat-playlist',
                        '--no-download',
                        '--match-filter', f'duration>{min_duration} & duration<{max_duration} & !is_live & !was_live',
                    ]

                    # Add cookie arguments for YouTube authentication (same as pipeline)
                    cookie_args = self._get_cookie_args(config)
                    if cookie_args:
                        cmd.extend(cookie_args)

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        timeout=60
                    )

                    if result.returncode != 0:
                        logger.debug(f"Search failed for '{query_text}': {result.stderr[:200]}")
                        continue

                    # Parse JSON lines output and collect for caching
                    search_results_for_cache = []
                    for line in result.stdout.strip().split('\n'):
                        if not line:
                            continue
                        try:
                            info = json.loads(line)
                            vid_id = info.get('id', '')
                            if vid_id:
                                # Store minimal info for cache
                                search_results_for_cache.append({
                                    'id': vid_id,
                                    'title': info.get('title', ''),
                                    'duration': info.get('duration', 0)
                                })
                                if vid_id not in existing_ids and vid_id not in new_video_ids:
                                    new_video_ids.add(vid_id)
                                    query_vids.append(vid_id)
                                    if len(new_video_ids) >= max_new_videos:
                                        break
                        except json.JSONDecodeError:
                            continue

                    # Cache the search results
                    if search_results_for_cache:
                        search_cache.set_search_result(
                            keyword=query_text,
                            tier='iterative',
                            search_pool=results_per_query,
                            videos=search_results_for_cache
                        )

                if query_vids:
                    query_video_map[query_text] = query_vids
                    q['videos_found'] = len(query_vids)

            except subprocess.TimeoutExpired:
                logger.warning(f"Search timeout for '{query_text}'")
            except Exception as e:
                logger.debug(f"Search error for '{query_text}': {e}")

        # Report cache stats
        total_queries = cache_hits + cache_misses
        if total_queries > 0:
            hit_rate = (cache_hits / total_queries) * 100
            print(f"    Search cache: {cache_hits}/{total_queries} hits ({hit_rate:.0f}%)")

        if not new_video_ids:
            logger.info("No new videos found from search queries")
            return []

        # Track fetched videos for cross-pass deduplication
        self._fetched_video_ids.update(new_video_ids)

        return list(new_video_ids)

    def _search_and_fetch_captions(
        self,
        queries: List[Dict[str, Any]],
        state: 'PipelineState',
        config: 'Config',
        iter_config: Any
    ) -> List[Dict[str, Any]]:
        """
        Execute YouTube searches and fetch captions for new videos.

        DEPRECATED: Use _search_youtube_for_videos + _fetch_captions_for_videos
        with batching for better streaming behavior.

        Args:
            queries: Search queries to execute
            state: Pipeline state
            config: Main config
            iter_config: Iterative matching config

        Returns:
            List of new video candidate dictionaries with caption segments
        """
        # Search for videos first
        video_ids = self._search_youtube_for_videos(queries, state, config, iter_config)

        if not video_ids:
            return []

        print(f"    Found {len(video_ids)} new video candidates")

        # Fetch captions for new videos
        caption_timeout = getattr(iter_config, 'caption_timeout', 30)
        new_candidates = self._fetch_captions_for_videos(
            video_ids,
            config,
            iter_config,
            caption_timeout
        )

        return new_candidates

    def _fetch_captions_for_videos(
        self,
        video_ids: List[str],
        config: 'Config',
        iter_config: Any,
        timeout: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Fetch captions for a list of video IDs.

        Args:
            video_ids: YouTube video IDs to fetch captions for
            config: Main config
            iter_config: Iterative matching config
            timeout: Timeout per video

        Returns:
            List of caption result dictionaries
        """
        from ..caption_fetcher import (
            CaptionFetcher,
            CaptionFetchError,
            CaptionUnavailableError,
            CaptionCache,
            determine_caption_quality
        )

        # Get caption config
        caption_config = getattr(config.download, 'caption_first', None)
        preferred_lang = 'en'
        if caption_config:
            preferred_lang = getattr(caption_config, 'preferred_language', 'en')

        # Get cookie args (with rotation if enabled) and log
        cookie_args = self._get_cookie_args(config, log_usage=True)

        # Initialize fetcher and cache with rotated cookies
        cache = CaptionCache(caption_config)
        fetcher = CaptionFetcher(config=config, cookie_args=cookie_args, caption_cache=cache)

        candidates = []
        success_count = 0
        fail_count = 0
        cache_hits = 0

        print(f"    Fetching captions for {len(video_ids)} videos...")

        for i, video_id in enumerate(video_ids):
            try:
                # Check cache first
                cached = cache.get_caption(video_id, preferred_lang) if cache.enabled else None

                if cached:
                    # Use cached result
                    result = cached
                    cache_hits += 1
                else:
                    # Fetch fresh
                    result = fetcher.fetch_captions(video_id, language=preferred_lang)
                    # Save to cache
                    if result and result.segments and cache.enabled:
                        cache.store(result)

                if result and result.segments:
                    # Segments may be CaptionSegment objects (fresh fetch) or dicts (from cache)
                    first_seg = result.segments[0]
                    segments_are_dicts = isinstance(first_seg, dict)

                    # Determine caption quality
                    if segments_are_dicts:
                        total_duration = sum(
                            (s['end'] - s['start']) for s in result.segments
                        )
                    else:
                        total_duration = sum(
                            (s.end_time - s.start_time) for s in result.segments
                        )
                    quality = determine_caption_quality(
                        is_auto_generated=result.is_auto_generated,
                        segment_count=len(result.segments),
                        total_duration=total_duration
                    )

                    # Convert segments to dict format
                    if segments_are_dicts:
                        segments = [
                            {
                                'text': seg.get('text', ''),
                                'start': seg.get('start', seg.get('start_time', 0)),
                                'end': seg.get('end', seg.get('end_time', 0)),
                            }
                            for seg in result.segments
                        ]
                    else:
                        segments = [
                            {
                                'text': seg.text,
                                'start': seg.start_time,
                                'end': seg.end_time,
                            }
                            for seg in result.segments
                        ]

                    candidates.append({
                        'video_id': video_id,
                        'segments': segments,
                        'language': result.language,
                        'is_auto_generated': result.is_auto_generated,
                        'caption_quality': quality,
                    })
                    success_count += 1

                    # Log progress every 10 videos
                    if (i + 1) % 10 == 0:
                        print(f"      [{i + 1}/{len(video_ids)}] Fetched {success_count} captions...")

            except CaptionUnavailableError:
                fail_count += 1
                logger.debug(f"No captions for {video_id}")
            except CaptionFetchError as e:
                fail_count += 1
                logger.debug(f"Caption fetch error for {video_id}: {e}")
            except Exception as e:
                fail_count += 1
                logger.debug(f"Unexpected error fetching {video_id}: {e}")

        print(f"    Captions: {success_count} fetched ({cache_hits} from cache), {fail_count} unavailable")

        return candidates

    def _extract_video_id_from_path(self, path: str) -> str:
        """Extract video ID from file path."""
        return extract_video_id(path) or ""

    def _rematch_gaps(
        self,
        gaps: List[GapSegment],
        locked: List[LockedMatch],
        new_candidates: List[Dict[str, Any]],
        state: 'PipelineState',
        config: 'Config'
    ) -> int:
        """
        Re-match gap segments with expanded candidate pool.

        Adds new caption data to state, computes embeddings, and re-matches gaps.

        Args:
            gaps: Gap segments to re-match
            locked: Locked matches (preserved)
            new_candidates: New video candidates from search
            state: Pipeline state
            config: Main config

        Returns:
            Number of gaps successfully filled
        """
        if not new_candidates:
            return 0

        # 1. Add new caption segments to text_metadata
        original_metadata_count = len(state.text_metadata) if state.text_metadata else 0
        new_segment_count = 0

        for candidate in new_candidates:
            video_id = candidate['video_id']
            segments = candidate.get('segments', [])
            language = candidate.get('language', 'en')
            is_auto = candidate.get('is_auto_generated', False)
            quality = candidate.get('caption_quality', 'medium')

            for seg in segments:
                state.text_metadata.append({
                    'text': seg.get('text', ''),
                    'video_path': video_id,
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'source_file': video_id,
                    'caption_source': 'youtube_iterative',
                    'caption_language': language,
                    'caption_auto_generated': is_auto,
                    'caption_quality': quality,
                    'iterative_pass': True,  # Mark as from iterative matching
                })
                new_segment_count += 1

        if new_segment_count == 0:
            return 0

        logger.info(f"Added {new_segment_count} caption segments from {len(new_candidates)} videos")

        # 2. Compute embeddings for new segments
        new_embeddings = self._compute_embeddings_for_new_segments(
            state, config, original_metadata_count
        )

        if new_embeddings is None or len(new_embeddings) == 0:
            logger.warning("Failed to compute embeddings for new segments")
            return 0

        # 3. Re-match gap segments using embedding similarity
        gaps_filled = self._match_gaps_to_new_candidates(
            gaps, locked, state, config, original_metadata_count
        )

        return gaps_filled

    def _compute_embeddings_for_new_segments(
        self,
        state: 'PipelineState',
        config: 'Config',
        start_index: int
    ) -> Optional[Any]:
        """
        Compute embeddings for newly added text_metadata segments.

        Args:
            state: Pipeline state
            config: Main config
            start_index: Index in text_metadata where new segments start

        Returns:
            New embeddings array or None on failure
        """
        try:
            from ..embeddings import compute_embeddings, get_embedding_provider
            from ..utils import CacheManager
            import numpy as np

            # Get new segment texts
            new_metadata = state.text_metadata[start_index:]
            if not new_metadata:
                return None

            texts = [m.get('text', '') for m in new_metadata if m.get('text')]
            if not texts:
                return None

            # Get embedding provider
            provider = get_embedding_provider(config)
            cache = CacheManager(config.cache.cache_dir)

            # Compute embeddings for new texts
            new_embeddings = compute_embeddings(
                texts=texts,
                provider=provider,
                cache=cache,
                cache_key="iterative_segments",
                show_progress=False,
                config=config
            )

            # Append to local embedding storage (no longer stored on PipelineState)
            if self._embeddings is not None and len(self._embeddings) > 0:
                self._embeddings = np.vstack([self._embeddings, new_embeddings])
            else:
                self._embeddings = new_embeddings

            # Rebuild embedding index with all vectors
            from ..embeddings import build_embedding_index
            self._embedding_index = build_embedding_index(self._embeddings, config)

            logger.info(f"Computed {len(new_embeddings)} new embeddings, total now {len(self._embeddings)}")
            return new_embeddings

        except Exception as e:
            logger.error(f"Failed to compute embeddings: {e}", exc_info=True)
            return None

    def _match_gaps_to_new_candidates(
        self,
        gaps: List[GapSegment],
        locked: List[LockedMatch],
        state: 'PipelineState',
        config: 'Config',
        new_segment_start: int
    ) -> int:
        """
        Match gap segments against new video candidates using embedding similarity.

        Pre-computes voiceover embeddings for all gaps at once, then iterates
        through gaps for matching. This avoids per-gap embedding API calls.

        Args:
            gaps: Gap segments to fill
            locked: Locked matches (for source spacing check)
            state: Pipeline state with updated embeddings
            config: Main config
            new_segment_start: Index where new segments start in text_metadata

        Returns:
            Number of gaps successfully filled
        """
        if not gaps or self._embedding_index is None:
            return 0

        try:
            from ..embeddings import get_embedding_provider
            import numpy as np

            provider = get_embedding_provider(config)

            # PRE-COMPUTE: Batch compute all voiceover embeddings at pass start
            precomputed_embeddings = self._precompute_voiceover_embeddings(
                gaps, state, provider
            )
            logger.info(f"Pre-computed {len(precomputed_embeddings)} voiceover embeddings for gap matching")

            # Get locked video IDs for source spacing
            locked_sources: Dict[str, List[float]] = defaultdict(list)
            for lock in locked:
                locked_sources[lock.video_id].append(lock.position)

            # Get source spacing threshold
            iter_config = getattr(config, 'iterative_matching', None)
            source_spacing = 300.0
            target_conf = 0.90
            if iter_config:
                source_spacing = getattr(iter_config, 'source_spacing_seconds', 300.0)
                target_conf = getattr(iter_config, 'target_confidence', 0.90)

            gaps_filled = 0

            for gap in gaps:
                # Get voiceover segment embedding (from precomputed cache)
                vo_embedding = self._get_voiceover_embedding(
                    gap.segment_index, gap.voiceover_text, state, provider,
                    precomputed=precomputed_embeddings
                )
                if vo_embedding is None:
                    continue

                # Search only in new segments (indices >= new_segment_start)
                # Use FAISS index for similarity search
                k = min(20, len(self._embeddings) - new_segment_start)
                if k <= 0:
                    continue

                # Query FAISS index
                distances, indices = self._embedding_index.search(
                    np.array([vo_embedding]).astype('float32'), k * 2
                )

                # Filter to only new segments and check source spacing
                best_match = None
                best_conf = 0.0

                for dist, idx in zip(distances[0], indices[0]):
                    if idx < new_segment_start:
                        continue  # Skip old segments

                    meta = state.text_metadata[idx]
                    video_id = self._extract_video_id_from_path(meta.get('video_path', ''))

                    # Check source spacing
                    if video_id in locked_sources:
                        violates_spacing = any(
                            abs(gap.position - pos) < source_spacing
                            for pos in locked_sources[video_id]
                        )
                        if violates_spacing:
                            continue

                    # Convert distance to confidence (FAISS returns L2 distance)
                    # Smaller distance = higher similarity
                    confidence = max(0, 1.0 - (dist / 2.0))

                    if confidence > best_conf and confidence >= target_conf:
                        best_conf = confidence
                        best_match = {
                            'index': idx,
                            'video_id': video_id,
                            'confidence': confidence,
                            'meta': meta
                        }

                # Update match if we found a good one
                if best_match:
                    self._update_match_for_gap(gap, best_match, state)
                    gaps_filled += 1

                    # Track this source for future spacing checks
                    locked_sources[best_match['video_id']].append(gap.position)

            return gaps_filled

        except Exception as e:
            logger.error(f"Error matching gaps: {e}", exc_info=True)
            return 0

    def _precompute_voiceover_embeddings(
        self,
        gaps: List[GapSegment],
        state: 'PipelineState',
        provider: Any
    ) -> Dict[int, Any]:
        """
        Pre-compute embeddings for all gap voiceover segments.

        Batches embedding requests to avoid per-segment API calls,
        significantly reducing latency in the gap matching loop.

        Args:
            gaps: Gap segments that need embeddings
            state: Pipeline state (may have cached embeddings)
            provider: Embedding provider

        Returns:
            Dict mapping segment_index to embedding vector
        """
        embeddings_map: Dict[int, Any] = {}

        # First, collect from cached voiceover embeddings
        if hasattr(state, 'voiceover_embeddings') and state.voiceover_embeddings is not None:
            for gap in gaps:
                if gap.segment_index < len(state.voiceover_embeddings):
                    embeddings_map[gap.segment_index] = state.voiceover_embeddings[gap.segment_index]

        # Find gaps that still need embeddings
        gaps_needing_embed = [g for g in gaps if g.segment_index not in embeddings_map]

        if not gaps_needing_embed:
            logger.debug(f"All {len(gaps)} voiceover embeddings found in cache")
            return embeddings_map

        # Batch compute missing embeddings
        logger.info(f"Pre-computing {len(gaps_needing_embed)} voiceover embeddings (batch)")
        try:
            texts = [g.voiceover_text for g in gaps_needing_embed]
            computed = provider.embed(texts)

            if computed is not None:
                for gap, emb in zip(gaps_needing_embed, computed):
                    embeddings_map[gap.segment_index] = emb

                logger.debug(f"Pre-computed {len(computed)} voiceover embeddings")

        except Exception as e:
            logger.warning(f"Batch embedding failed, will compute individually: {e}")

        return embeddings_map

    def _get_voiceover_embedding(
        self,
        segment_index: int,
        text: str,
        state: 'PipelineState',
        provider: Any,
        precomputed: Optional[Dict[int, Any]] = None
    ) -> Optional[Any]:
        """
        Get embedding for a voiceover segment.

        Checks precomputed cache first, then state cache, then computes.

        Args:
            segment_index: Index of the segment
            text: Segment text
            state: Pipeline state
            provider: Embedding provider
            precomputed: Optional dict of precomputed embeddings

        Returns:
            Embedding vector or None
        """
        try:
            # Check precomputed first (from batch computation)
            if precomputed and segment_index in precomputed:
                return precomputed[segment_index]

            # Check if we have cached voiceover embeddings
            if hasattr(state, 'voiceover_embeddings') and state.voiceover_embeddings is not None:
                if segment_index < len(state.voiceover_embeddings):
                    return state.voiceover_embeddings[segment_index]

            # Compute embedding for this text (fallback)
            logger.debug(f"Computing single voiceover embedding for segment {segment_index}")
            embeddings = provider.embed([text])
            return embeddings[0] if embeddings is not None else None

        except Exception as e:
            logger.debug(f"Failed to get voiceover embedding: {e}")
            return None

    def _update_match_for_gap(
        self,
        gap: GapSegment,
        best_match: Dict[str, Any],
        state: 'PipelineState'
    ):
        """Update the match for a gap segment with a new candidate."""
        try:
            # Find the match object for this segment
            if gap.segment_index < len(state.matches):
                match = state.matches[gap.segment_index]

                # Update the primary match
                if hasattr(match, 'primary_match') and match.primary_match:
                    pm = match.primary_match
                    if hasattr(pm, 'confidence'):
                        pm.confidence = best_match['confidence']
                    if hasattr(pm, 'video_segment'):
                        # Update video segment with new source
                        vs = pm.video_segment
                        if hasattr(vs, 'source_file'):
                            vs.source_file = best_match['video_id']
                        if hasattr(vs, 'text'):
                            vs.text = best_match['meta'].get('text', '')

                    logger.debug(
                        f"Updated gap {gap.segment_index} with {best_match['video_id']} "
                        f"(conf: {best_match['confidence']:.2f})"
                    )

        except Exception as e:
            logger.debug(f"Failed to update match for gap: {e}")

    def _update_query_learning(
        self,
        queries: List[Dict[str, Any]],
        gaps_filled: int,
        gap_analysis: Any,
        learning_db: Any
    ):
        """
        Update query learning database with results.

        Args:
            queries: Queries that were executed
            gaps_filled: Total gaps filled this pass
            gap_analysis: Gap pattern analysis
            learning_db: Learning database instance
        """
        if not learning_db or not gap_analysis:
            return

        from ..iterative_match import QueryResult

        # Create results for each query
        # (In practice, would track per-query success)
        avg_fill = gaps_filled / len(queries) if queries else 0

        for q in queries:
            gap_indices = q.get('gap_indices', [])
            if not gap_indices:
                continue

            # Determine pattern for this gap
            pattern = 'other'
            for p, indices in gap_analysis.clustered_gaps.items():
                if gap_indices[0] in indices:
                    pattern = p
                    break

            result = QueryResult(
                query=q['query'],
                strategy=q.get('strategy', 'unknown'),
                gap_indices=gap_indices,
                videos_found=q.get('videos_found', 0),
                gaps_filled=1 if avg_fill > 0.5 else 0,  # Simplified
                avg_confidence_improvement=0.1 if gaps_filled > 0 else 0.0
            )

            learning_db.record_result(result, pattern)
