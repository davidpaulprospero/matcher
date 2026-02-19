"""
Match Stage - Voiceover to Video Matching

Stage 4 of the simplified 7-stage pipeline:
- Matches voiceover segments to video clips
- Uses caption text similarity and LLM reranking
- Supports delta matching for incremental updates
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageMetrics, StageResult, register_stage, validate_required_state_attrs
from ..logger import get_global_logger
from ..matching.scoring import get_multimodal_tracker, aggregate_chapter_diagnostics, log_chapter_diagnostics
from ..matching.serialization import serialize_match_for_match_stage
from ..utils import is_embeddings_empty

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, Match

logger = logging.getLogger(__name__)


@register_stage
class MatchStage(Stage):
    """
    Matches voiceover segments to video clips.

    In the simplified 7-stage pipeline, receives caption data from CAPTION stage.

    Inputs:
        - state.voiceover_segments: List of VoiceoverSegment
        - state.caption_results: Dict of video_id -> caption data (from CAPTION stage)
        - (Optional) state.text_metadata: For embedding-based matching

    Outputs:
        - state.matches: List of Match objects
        - state.alternatives: Dict of alternative matches
    """

    name = "MATCH"
    description = "Match voiceover segments to video clips"
    DEPENDS_ON = ['ANALYZE', 'CAPTION']
    PRODUCES = ['matches', 'alternatives']

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the match stage.

        US-39-009: Validates state type and converts legacy objects if needed.
        US-40-008: Validates required state attributes exist.
        """
        # US-39-009: Validate state type at stage entry
        state = self._validate_state_type(state)

        # US-40-008: Validate required attributes exist
        validate_required_state_attrs(
            state,
            ['text_metadata', 'caption_results', 'voiceover_segments'],
            self.name
        )

        warnings = []

        try:
            if config.pipeline.skip_matching:
                print("  >> Skipping matching (config: skip_matching=true)")
                logger.info("Skipping MATCH stage (config: skip_matching=true)")
                return StageResult.ok({'skipped': True}, warnings)

            print(f"\n  --- Stage 4: MATCH FOOTAGE ---")

            # Validate we have data
            if not state.voiceover_segments:
                print("  ! No voiceover segments to match")
                warnings.append("No voiceover segments")
                return StageResult.ok({'matches': []}, warnings)

            # US-71-002: Detect listicle structure in voiceover segments
            self._detect_and_store_listicle_groups(state)

            # US-71-010: Bridge listicle groups to chapter structure for unified handling
            self._build_unified_chapters(state)

            # US-105-011: Apply fallback when no chapters detected
            chapter_fallback = self._apply_no_chapter_fallback(state, config)
            if chapter_fallback:
                warnings.extend(chapter_fallback)

            # In simplified pipeline, text_metadata comes from CAPTION stage
            # Embeddings are optional for caption-first matching

            # US-39-012/US-40-007: Fallback recovery if text_metadata is empty but caption_results exists
            if not state.text_metadata and getattr(state, 'caption_results', None):
                logger.warning(
                    "text_metadata empty, attempting recovery from caption_results"
                )
                warnings.append("Recovered text_metadata from caption_results (CAPTION stage partial failure)")
                self._recover_text_metadata_from_captions(state)

            # US-40-007: Return error if both text_metadata and caption_results are unavailable
            if not state.text_metadata:
                error_msg = "No captions available: text_metadata empty and caption_results has no usable data. Run CAPTION stage first."
                print(f"  ! {error_msg}")
                logger.error(error_msg)
                return StageResult.fail(error_msg, warnings)

            # Print settings
            self._print_settings(config)

            # Prepare segments
            vo_segments, video_segments, all_video_paths = self._prepare_segments(state)

            # US-111-002: Enrich voiceover segments with topic extraction
            try:
                from ..matching.voiceover_topics import enrich_voiceover_segments_with_topics
                vo_segments = enrich_voiceover_segments_with_topics(vo_segments, config)
            except Exception as e:
                logger.warning(f"Voiceover topic extraction failed: {e}")
                # Fallback to original behavior - don't fail the stage

            # Check delta matching
            force_rematch = getattr(config.matching, 'force_rematch', False)
            delta_enabled = getattr(config.matching, 'delta_matching_enabled', True)

            # US-53-009: Reset multimodal scoring tracker before matching run
            multimodal_tracker = get_multimodal_tracker()
            multimodal_tracker.reset()

            # Run matching (US-81-007: time the matching for throughput metrics)
            match_start_time = time.monotonic()
            matches = self._run_matching(
                vo_segments, video_segments, all_video_paths,
                state, config, delta_enabled, force_rematch,
                checkpoint=checkpoint
            )
            match_duration = time.monotonic() - match_start_time

            state.matches = matches

            # Calculate stats
            confidences = []
            for m in matches:
                if m and hasattr(m, 'primary_match') and m.primary_match:
                    confidences.append(m.primary_match.confidence)
                elif m and hasattr(m, 'confidence'):
                    confidences.append(m.confidence)

            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            print(f"\n  + Matched {len(matches)} segments")
            print(f"  Average confidence: {avg_conf:.1%}")

            # Calculate and log quality metrics
            from ..matching.metrics import (
                calculate_match_quality_metrics, log_quality_summary,
                log_confidence_histogram, compute_diversity_metrics,
                log_diversity_metrics,
                calculate_confidence_trend, log_trend_summary,
            )
            quality_metrics = calculate_match_quality_metrics(
                matches=matches,
                total_segments=len(state.voiceover_segments)
            )
            log_quality_summary(quality_metrics)

            # Log confidence distribution histogram
            if confidences:
                log_confidence_histogram(confidences)

            # Compute and log diversity metrics (US-53-005)
            diversity_report = compute_diversity_metrics(matches)
            log_diversity_metrics(diversity_report)

            # US-63-010: Calculate and log confidence trend across voiceover chunks
            confidence_trend = calculate_confidence_trend(matches)
            log_trend_summary(confidence_trend)

            # US-53-009: Log multimodal scoring summary
            multimodal_tracker.log_summary()

            # Update logger stats for match-only mode
            run_logger = get_global_logger()
            if run_logger:
                run_logger.set_stats(
                    total_segments=len(state.voiceover_segments) if hasattr(state, 'voiceover_segments') else len(matches),
                    total_matches=len(matches)
                )

            # Serialize essential match data for checkpoint
            # This enables DOWNLOAD_SEGMENTS to validate it has matches available
            serialized_matches = []
            for i, m in enumerate(matches):
                try:
                    serialized_matches.append(serialize_match_for_match_stage(m, i))
                except Exception as e:
                    logger.warning(f"Failed to serialize match {i}: {e}")

            # US-71-009: Serialize chapter/listicle data for checkpoint persistence
            chapter_data = self._serialize_chapter_data(state)

            # US-76-006: Aggregate and log cross-chapter coherence diagnostics
            chapter_diagnostics = aggregate_chapter_diagnostics(
                matches=matches,
                chapters=getattr(state, 'location_chapters', None),
            )
            log_chapter_diagnostics(chapter_diagnostics)

            checkpoint_data = {
                'match_count': len(matches),
                'avg_confidence': avg_conf,
                'matches': serialized_matches,  # Essential match data for validation
                'quality_metrics': quality_metrics.to_dict(),  # Quality metrics for analysis
                'diversity_metrics': diversity_report.to_dict(),  # US-53-005: Source diversity per track
                'trend_data': confidence_trend.to_dict(),  # US-63-010: Confidence trend for post-run analysis
                'chapter_data': chapter_data,  # US-71-009: Chapter/listicle detection results
                'chapter_diagnostics': chapter_diagnostics,  # US-76-006: Cross-chapter coherence diagnostics
            }

            # US-81-007: Throughput metrics for match stage
            stage_metrics = StageMetrics(
                items_processed=len(matches),
                duration_seconds=match_duration,
            )
            # Match processes all segments as a batch; compute overall throughput
            if match_duration > 0 and len(matches) > 0:
                overall_rate = len(matches) / match_duration
                stage_metrics.items_per_second = overall_rate
                stage_metrics.peak_items_per_second = overall_rate

            return StageResult.ok(checkpoint_data, warnings, stage_metrics)

        except Exception as e:
            logger.exception(f"Match stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if match stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """
        Restore match stage from checkpoint.

        Validates match objects before restoring:
        - segment_index must be a valid integer
        - video_file must be a non-empty string
        - confidence must be a valid float between 0 and 1

        Returns False on validation failure (not exception).
        """
        from ..state import restore_matches_from_dicts

        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}: checkpoint returned None")
                return False

            # US-51-008: Validate checkpoint data schema before restoring
            if not isinstance(data, dict):
                logger.warning(f"MATCH restore: expected dict, got {type(data).__name__}")
                return False

            if 'matches' in data and not isinstance(data['matches'], list):
                logger.warning(f"MATCH restore: 'matches' expected list, got {type(data['matches']).__name__}")
                return False

            # Load matches from checkpoint using shared helper
            matches_data = data.get('matches', [])
            if matches_data:
                restored_matches = restore_matches_from_dicts(
                    matches_data, default_strategy='restored', logger_instance=logger
                )
                if restored_matches is None:
                    return False

                state.matches = restored_matches
                state._raw_match_dicts = matches_data
                logger.info(f"Restored MATCH: {len(restored_matches)} matches from checkpoint")
            else:
                logger.info(f"Restored MATCH metadata from checkpoint (no matches data)")

            # US-71-009: Restore chapter/listicle data from checkpoint
            self._restore_chapter_data(state, data)

            return True

        except Exception as e:
            logger.warning(f"Failed to restore MATCH: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """
        Validate inputs before running.

        Returns specific missing field names and suggestions for which stage to run.
        """
        if not state.voiceover_segments:
            return "Missing voiceover_segments. Suggestion: run ANALYZE stage first"

        # In simplified pipeline, text_metadata is populated by CAPTION stage
        # Embeddings are optional (caption-first uses text matching)
        has_text_data = bool(state.text_metadata) or bool(getattr(state, 'caption_results', None))

        if not has_text_data:
            return "Missing text_metadata or caption_results. Suggestion: run CAPTION stage first"

        # US-44-010: Warn if caption_results is empty when text_metadata is populated
        caption_results = getattr(state, 'caption_results', {})
        if state.text_metadata and not caption_results:
            logger.warning(
                "text_metadata is populated but caption_results is empty — "
                "data may have been loaded from a legacy checkpoint"
            )

        # US-44-010: Warn on video count drift between video_ids and caption_results
        video_ids = getattr(state, 'video_ids', [])
        if video_ids and caption_results:
            vid_count = len(video_ids)
            cap_count = len(caption_results)
            if vid_count > 0:
                drift_pct = abs(vid_count - cap_count) / vid_count
                if drift_pct > 0.10:
                    logger.warning(
                        f"Video count drift detected: video_ids={vid_count}, "
                        f"caption_results={cap_count} "
                        f"(drift={drift_pct:.0%}, threshold=10%%). "
                        f"Expected drops from failed captions or filtered videos, "
                        f"but large drift may indicate silent data loss between stages."
                    )

        return None

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        # Count inputs
        input_count = len(state.voiceover_segments) if state.voiceover_segments else 0
        if hasattr(state, 'caption_results'):
            input_count += len(state.caption_results)

        # Count outputs (matches)
        output_count = None
        if hasattr(state, 'matches') and state.matches:
            output_count = len(state.matches)

        return {
            'inputs': 'voiceover segments + captions',
            'outputs': 'matches',
            'input_count': input_count,
            'output_count': output_count,
        }

    def get_api_estimates(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get API call estimates for dry-run preview"""
        # Count voiceover segments
        segment_count = len(state.voiceover_segments) if state.voiceover_segments else 0

        # Count caption results (videos)
        video_count = len(state.caption_results) if hasattr(state, 'caption_results') and state.caption_results else 0

        # Estimate embedding calls: 1 per voiceover segment + 1 per video
        embedding_calls = segment_count + video_count

        # Get embedding config for cost estimation
        embedding_config = getattr(config.matching, 'embeddings', {}) if hasattr(config, 'matching') else {}
        if not isinstance(embedding_config, dict):
            embedding_config = {}

        # Estimate cost: ~$0.0001 per embedding call (using ada-002 pricing as baseline)
        embedding_cost = embedding_calls * 0.0001

        # Estimate LLM calls for matching (if using LLM reranker)
        llm_calls = 0
        if hasattr(config.matching, 'use_llm_reranker') and config.matching.use_llm_reranker:
            # Estimate: ~1 LLM call per 10 segments for reranking
            llm_calls = max(1, segment_count // 10)

        # LLM cost estimate: ~$0.01 per call (GPT-4o mini as baseline)
        llm_cost = llm_calls * 0.01

        # Total cost
        total_cost = embedding_cost + llm_cost

        # Estimate duration: ~0.1s per embedding + ~1s per LLM call
        estimated_duration = embedding_calls * 0.1 + llm_calls * 1.0

        estimates = {
            'embedding_calls': embedding_calls,
            'estimated_cost_usd': round(total_cost, 4),
            'estimated_duration_seconds': round(estimated_duration, 1),
        }

        if llm_calls > 0:
            estimates['llm_calls'] = llm_calls

        return estimates

    # === Helper Methods ===

    def _recover_text_metadata_from_captions(self, state: 'PipelineState') -> None:
        """Recover text_metadata from caption_results when CAPTION stage had partial failure.

        US-39-012: Provides graceful degradation when caption stage partial failure
        leaves caption_results populated but text_metadata empty.

        Uses the same logic as CaptionStage._populate_text_metadata() to convert
        caption_results into text_metadata format expected by matching.

        Args:
            state: PipelineState with caption_results but empty text_metadata.
        """
        caption_results = getattr(state, 'caption_results', {})
        if not caption_results:
            logger.warning("Cannot recover: caption_results is empty")
            return

        # Ensure text_metadata exists
        if not hasattr(state, 'text_metadata'):
            state.text_metadata = []

        text_metadata = []

        for video_id, result in caption_results.items():
            # Skip unavailable/errored captions
            if result.get('unavailable') or result.get('error') or result.get('skipped'):
                continue

            segments = result.get('segments', [])
            language = result.get('language', 'en')
            is_auto = result.get('is_auto_generated', False)
            caption_quality = result.get('caption_quality', 'medium')
            timing_penalty = result.get('timing_penalty', 1.0)

            for seg in segments:
                text_metadata.append({
                    'text': seg.get('text', ''),
                    'video_path': video_id,  # In caption-first mode, this is video ID
                    'start_time': seg.get('start', 0),
                    'end_time': seg.get('end', 0),
                    'source_file': video_id,
                    # Caption-specific metadata
                    'caption_source': 'youtube',
                    'caption_language': language,
                    'caption_auto_generated': is_auto,
                    'caption_quality': caption_quality,
                    'timing_penalty': timing_penalty,
                })

        state.text_metadata.extend(text_metadata)
        logger.info(f"Recovered {len(text_metadata)} text_metadata entries from caption_results")

    def _detect_and_store_listicle_groups(self, state: 'PipelineState') -> None:
        """Detect listicle structure in voiceover segments and store on state.

        US-71-002: Runs listicle detection before scoring so downstream stages
        can use the group information for scoring adjustments.
        """
        from ..chapter_detection.listicle_detector import detect_listicle_groups

        try:
            groups = detect_listicle_groups(state.voiceover_segments)
            state.listicle_groups = groups

            if groups:
                labels = [g.item_label for g in groups]
                logger.info(
                    f"Listicle structure detected: {len(groups)} groups "
                    f"(labels: {', '.join(labels)})"
                )
                print(f"  Listicle structure detected: {len(groups)} groups")
            else:
                logger.debug("No listicle structure detected in voiceover segments")
        except Exception as e:
            logger.warning(f"Listicle detection failed (non-fatal): {e}")
            state.listicle_groups = []

    def _build_unified_chapters(self, state: 'PipelineState') -> None:
        """Bridge listicle groups into chapter structure for unified scoring.

        US-71-010: Converts listicle groups to ChapterCandidate objects and merges
        them with existing YouTube/location chapters. YouTube chapters take
        precedence for overlapping segment ranges. The unified list is stored
        on state.location_chapters so all chapter-aware scoring adjustments
        work uniformly.
        """
        from ..chapter_detection.bridge import build_unified_chapters

        listicle_groups = getattr(state, 'listicle_groups', []) or []
        location_chapters = getattr(state, 'location_chapters', []) or []

        if not listicle_groups and not location_chapters:
            return

        try:
            unified = build_unified_chapters(location_chapters, listicle_groups)
            state.location_chapters = unified
            if unified:
                logger.info(
                    f"US-71-010 unified chapters: {len(unified)} "
                    f"(from {len(location_chapters)} YouTube + {len(listicle_groups)} listicle)"
                )
        except Exception as e:
            logger.warning(f"Unified chapter bridge failed (non-fatal): {e}")

    def _apply_no_chapter_fallback(
        self, state: 'PipelineState', config: 'Config'
    ) -> List[str]:
        """US-105-011: Apply fallback behavior when no chapters are detected.

        When no chapters are found after building unified chapters, applies the
        configured fallback strategy:
        - 'global': Disable chapter-based boosting and use standard global matching
        - 'segment': Proceed with segment-level matching (no chapter grouping)

        Returns a list of warning messages if fallback was applied.
        """
        warnings: List[str] = []

        # Check if any chapters exist (from either YouTube or listicle detection)
        chapters = getattr(state, 'location_chapters', None) or []
        listicle_groups = getattr(state, 'listicle_groups', None) or []

        if not chapters and not listicle_groups:
            # No chapters detected - apply fallback strategy
            fallback_strategy = getattr(
                config.matching, 'no_chapter_fallback_strategy', 'global'
            )

            if fallback_strategy == 'global':
                # Disable chapter-based matching features
                logger.info(
                    "US-105-011: No chapters detected, using 'global' fallback strategy"
                )
                warnings.append(
                    "No chapters detected - using global matching (chapter features disabled)"
                )

                # Disable chapter features in config (will be restored on next run with chapters)
                # Store original values in state for restoration
                state._chapter_config_backup = {
                    'chapter_matching_enabled': getattr(
                        config.matching, 'chapter_matching_enabled', True
                    ),
                    'enforce_chapter_boundaries': getattr(
                        config.matching, 'enforce_chapter_boundaries', False
                    ),
                    'prefer_chapter_aligned_segments': getattr(
                        config.matching, 'prefer_chapter_aligned_segments', True
                    ),
                    'chapter_alignment_boost': getattr(
                        config.matching, 'chapter_alignment_boost', 0.05
                    ),
                }

                # Disable chapter features
                config.matching.chapter_matching_enabled = False
                config.matching.enforce_chapter_boundaries = False
                config.matching.prefer_chapter_aligned_segments = False
                config.matching.chapter_alignment_boost = 0.0

                logger.debug(
                    f"Chapter features disabled for global fallback: "
                    f"{list(state._chapter_config_backup.keys())}"
                )

            elif fallback_strategy == 'segment':
                logger.info(
                    "US-105-011: No chapters detected, using 'segment' fallback strategy"
                )
                warnings.append(
                    "No chapters detected - using segment-level matching"
                )
                # 'segment' strategy just proceeds without chapter grouping
                # Chapter features are disabled but listicle groups still work
                config.matching.chapter_matching_enabled = False

            else:
                logger.warning(
                    f"Unknown no_chapter_fallback_strategy: '{fallback_strategy}'. "
                    f"Using 'global' as default."
                )
                # Recursively apply global fallback
                config.matching.no_chapter_fallback_strategy = 'global'
                return self._apply_no_chapter_fallback(state, config)

        return warnings

    def _serialize_chapter_data(self, state: 'PipelineState') -> Dict[str, Any]:
        """US-71-009: Serialize chapter/listicle data for checkpoint persistence.

        Returns a dict with 'chapters' and 'listicle_groups' lists of dicts.
        """
        chapters = []
        for ch in getattr(state, 'location_chapters', []) or []:
            if isinstance(ch, dict):
                chapters.append(ch)
            elif hasattr(ch, 'to_dict'):
                chapters.append(ch.to_dict())

        listicle_groups = []
        for lg in getattr(state, 'listicle_groups', []) or []:
            if isinstance(lg, dict):
                listicle_groups.append(lg)
            elif hasattr(lg, 'to_dict'):
                listicle_groups.append(lg.to_dict())

        return {
            'chapters': chapters,
            'listicle_groups': listicle_groups,
        }

    def _restore_chapter_data(self, state: 'PipelineState', data: Dict[str, Any]) -> None:
        """US-71-009: Restore chapter/listicle data from checkpoint.

        Loads serialized chapter/listicle dicts from checkpoint and restores
        them as dataclass instances on state. Falls back to empty lists if
        the checkpoint has no chapter_data (backward compatibility).
        """
        chapter_data = data.get('chapter_data', {})
        if not isinstance(chapter_data, dict):
            chapter_data = {}

        # Restore chapters (as ChapterCandidate objects)
        raw_chapters = chapter_data.get('chapters', [])
        if raw_chapters:
            try:
                from ..chapter_detection.models import ChapterCandidate
                state.location_chapters = [
                    ChapterCandidate.from_dict(ch) if isinstance(ch, dict) else ch
                    for ch in raw_chapters
                ]
                logger.info(f"Restored {len(state.location_chapters)} chapters from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to restore chapters from checkpoint: {e}")
                state.location_chapters = []
        else:
            state.location_chapters = getattr(state, 'location_chapters', []) or []

        # Restore listicle groups (as ListicleGroup objects)
        raw_groups = chapter_data.get('listicle_groups', [])
        if raw_groups:
            try:
                from ..chapter_detection.models import ListicleGroup
                state.listicle_groups = [
                    ListicleGroup.from_dict(lg) if isinstance(lg, dict) else lg
                    for lg in raw_groups
                ]
                logger.info(f"Restored {len(state.listicle_groups)} listicle groups from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to restore listicle groups from checkpoint: {e}")
                state.listicle_groups = []
        else:
            state.listicle_groups = getattr(state, 'listicle_groups', []) or []

    def _build_video_metadata(self, state: 'PipelineState') -> Dict[str, Dict[str, Any]]:
        """Build video_id-to-metadata lookup for efficient context access during matching.

        US-75-009: Merges data from video_search_results and caption_results into a single
        dict keyed by video_id. Each entry contains 'title', 'description', 'tags', and
        'chapters'. Missing fields default to empty (empty string / empty list), never None.

        Args:
            state: PipelineState with video_search_results and caption_results.

        Returns:
            Dict mapping video_id to metadata dict with keys:
            title (str), description (str), tags (List[str]), chapters (List[dict]).
        """
        video_metadata: Dict[str, Dict[str, Any]] = {}

        # Seed from video_search_results (title, description, tags)
        for vsr in getattr(state, 'video_search_results', []) or []:
            channel = ''
            view_count = None
            subscriber_count = None
            if isinstance(vsr, dict):
                vid_id = vsr.get('video_id', '')
                title = vsr.get('title', '')
                desc = vsr.get('description', '')
                tags = vsr.get('video_tags', [])
                channel = vsr.get('channel', '')
                view_count = vsr.get('view_count')  # US-111-005
                subscriber_count = vsr.get('subscriber_count')  # US-111-005
            else:
                vid_id = getattr(vsr, 'video_id', '')
                title = getattr(vsr, 'title', '')
                desc = getattr(vsr, 'description', '')
                tags = getattr(vsr, 'video_tags', [])
                channel = getattr(vsr, 'channel', '')
                view_count = getattr(vsr, 'view_count', None)  # US-111-005
                subscriber_count = getattr(vsr, 'subscriber_count', None)  # US-111-005
            if vid_id:
                video_metadata[vid_id] = {
                    'title': title or '',
                    'description': desc or '',
                    'tags': tags or [],
                    'chapters': [],
                    'channel': channel or '',
                    'view_count': view_count,  # US-111-005: For channel reputation scoring
                    'subscriber_count': subscriber_count,  # US-111-005
                    'channel_subscriber_count': subscriber_count,  # Alias for tiered_matcher lookup
                }

        # Enrich from caption_results (tags, chapters — may have data VSR lacks)
        # US-134-007: Also include transcript segments for LLM reranker context
        caption_results = getattr(state, 'caption_results', {}) or {}
        for video_id, result in caption_results.items():
            if not isinstance(result, dict):
                continue
            cr_tags = result.get('video_tags', []) or []
            cr_chapters = result.get('video_chapters', []) or []
            # US-134-007: Get transcript segments for context enrichment
            cr_segments = result.get('segments', []) or []

            if video_id in video_metadata:
                # Merge: prefer non-empty caption_results data over empty VSR data
                entry = video_metadata[video_id]
                if not entry['tags'] and cr_tags:
                    entry['tags'] = cr_tags
                if cr_chapters:
                    entry['chapters'] = cr_chapters
                # US-134-007: Add transcript segments
                if cr_segments:
                    entry['transcript_segments'] = cr_segments
            else:
                # Video exists in caption_results but not in video_search_results
                video_metadata[video_id] = {
                    'title': '',
                    'description': '',
                    'tags': cr_tags,
                    'chapters': cr_chapters,
                    'channel': '',
                    'transcript_segments': cr_segments,  # US-134-007
                }

        return video_metadata

    def _print_settings(self, config: 'Config'):
        """Print matching settings"""
        print(f"  Matching settings (from config):")
        print(f"    - Min confidence: {config.matching.min_confidence}")
        print(f"    - High confidence threshold: {config.matching.high_confidence_threshold}")
        print(f"    - Embedding candidates: {config.matching.embedding_candidates}")
        print(f"    - LLM rerank candidates: {config.matching.llm_rerank_candidates}")
        print(f"    - Max clip reuse: {config.matching.max_clip_reuse}")

    def _prepare_segments(
        self,
        state: 'PipelineState'
    ) -> tuple:
        """Prepare voiceover and video segments for matching"""
        from ..utils import SRTSegment

        print(f"\n  Preparing voiceover segments...")
        vo_segments = []
        for i, seg in enumerate(state.voiceover_segments):
            if hasattr(seg, 'text'):
                # VoiceoverSegment object
                vo_segment = SRTSegment(
                    index=seg.index,
                    start_time=seg.start,
                    end_time=seg.end,
                    text=seg.text,
                    source_file='',
                    keywords=[],
                    entities=[],
                    topics=[]
                )
            elif isinstance(seg, dict):
                vo_segment = SRTSegment(
                    index=seg.get('index', i),
                    start_time=seg.get('start_time', seg.get('start', 0)),
                    end_time=seg.get('end_time', seg.get('end', 0)),
                    text=seg.get('text', ''),
                    source_file=seg.get('source_file', ''),
                    keywords=seg.get('keywords', []),
                    entities=seg.get('entities', []),
                    topics=seg.get('chapter_topics', [])
                )
            else:
                vo_segment = seg

            vo_segments.append(vo_segment)

        print(f"  Preparing video segments...")
        video_segments = []
        video_paths_set = set()

        # DEBUG: Count B-roll entries in text_metadata
        broll_count = sum(1 for m in state.text_metadata if isinstance(m, dict) and m.get('is_broll'))
        logger.info(f"text_metadata has {broll_count}/{len(state.text_metadata)} entries with is_broll=True")

        broll_segments_created = 0
        caption_quality_count = {'high': 0, 'medium': 0, 'low': 0}
        for i, meta in enumerate(state.text_metadata):
            if isinstance(meta, dict):
                vid_segment = SRTSegment(
                    index=i,
                    start_time=meta.get('start_time', 0),
                    end_time=meta.get('end_time', 0),
                    text=meta.get('text', ''),
                    source_file=meta.get('video_path', ''),
                )
                if meta.get('source'):
                    vid_segment.source = meta['source']
                if meta.get('face_score') is not None:
                    vid_segment.face_score = meta['face_score']
                if meta.get('is_broll') is not None:
                    vid_segment.is_broll = meta['is_broll']
                    if meta['is_broll']:
                        broll_segments_created += 1
                if meta.get('scene_index') is not None:
                    vid_segment.scene_index = meta['scene_index']
                # US-007: Caption quality for confidence adjustment
                if meta.get('caption_quality') is not None:
                    vid_segment.caption_quality = meta['caption_quality']
                    if meta['caption_quality'] in caption_quality_count:
                        caption_quality_count[meta['caption_quality']] += 1
                # US-008 Sprint 7: Timing penalty for confidence adjustment
                if meta.get('timing_penalty') is not None:
                    vid_segment.timing_penalty = meta['timing_penalty']
                # US-70-008: Title-enriched embedding text
                if meta.get('embedding_text'):
                    vid_segment.embedding_text = meta['embedding_text']
                video_paths_set.add(meta.get('video_path', ''))
            else:
                vid_segment = meta
                video_paths_set.add(getattr(meta, 'source_file', ''))

            video_segments.append(vid_segment)

        logger.info(f"Created {broll_segments_created} video_segments with is_broll=True")
        # US-007: Log caption quality distribution
        if any(caption_quality_count.values()):
            logger.info(f"Caption quality distribution: {caption_quality_count['high']} high, "
                       f"{caption_quality_count['medium']} medium, {caption_quality_count['low']} low")

        return vo_segments, video_segments, list(video_paths_set)

    def _run_matching(
        self,
        vo_segments: List[Any],
        video_segments: List[Any],
        all_video_paths: List[str],
        state: 'PipelineState',
        config: 'Config',
        delta_enabled: bool,
        force_rematch: bool,
        checkpoint: 'CheckpointManager' = None
    ) -> List[Any]:
        """Run the actual matching algorithm.

        US-85-005: Supports within-stage resumption via intermediate checkpointing.
        If partial match results exist in checkpoint, resumes from last checkpointed segment.
        """
        from ..matching import match_all_segments
        from ..utils import CacheManager
        from ..embeddings import compute_embeddings, get_embedding_provider, EmbeddingCache

        cache_dir = config.cache.cache_dir if hasattr(config.cache, 'cache_dir') else ".cache"
        provider = get_embedding_provider(config)
        cache = CacheManager(cache_dir)

        # Compute voiceover embeddings
        # US-111-002: Include extracted topics in embedding text for better context-aware matching
        print(f"  Computing voiceover embeddings...")
        vo_texts = []
        for seg in vo_segments:
            text = seg.text
            # Add extracted topics to embedding text if available
            if hasattr(seg, 'topics') and seg.topics:
                topics_str = ' '.join(seg.topics)
                text = f"{text} {topics_str}"
            vo_texts.append(text)

        vo_embeddings = compute_embeddings(
            texts=vo_texts,
            provider=provider,
            cache=cache,
            cache_key="voiceover",
            embed_mode="query"
        )

        if vo_embeddings is None or len(vo_embeddings) == 0:
            print("  ! Failed to compute voiceover embeddings")
            return []

        # Compute video embeddings locally (no longer stored on PipelineState)
        # US-70-008: Use embedding_text (title-enriched) when available, fall back to text
        print(f"  Computing video embeddings...")
        vid_texts = [getattr(seg, 'embedding_text', seg.text) for seg in video_segments]

        video_embeddings = compute_embeddings(
            texts=vid_texts,
            provider=provider,
            cache=cache,
            cache_key="video_segments",
            embed_mode="document"
        )

        # Log embedding cache hit/miss rate (use provider-qualified keys)
        embedding_cache = EmbeddingCache(cache_dir)
        provider_tag = type(provider).__name__.lower().replace("embeddings", "")
        vo_cached, vo_uncached, _ = embedding_cache.get_cached_embeddings(
            [t.strip() if t else "[silence]" for t in vo_texts], f"{provider_tag}_voiceover"
        )
        vid_cached, vid_uncached, _ = embedding_cache.get_cached_embeddings(
            [t.strip() if t else "[silence]" for t in vid_texts], f"{provider_tag}_video_segments"
        )
        vo_total = len(vo_texts) if vo_texts else 1
        vid_total = len(vid_texts) if vid_texts else 1
        logger.info(
            f"Embedding cache stats — "
            f"voiceover: {len(vo_cached)}/{vo_total} hits ({len(vo_cached)*100//vo_total}%%), "
            f"video: {len(vid_cached)}/{vid_total} hits ({len(vid_cached)*100//vid_total}%%)"
        )

        if video_embeddings is None or len(video_embeddings) == 0:
            logger.warning("Failed to compute video embeddings, matching will rely on text-only strategies")

        # Run matching
        print(f"  Running two-stage matching...")

        # US-72-006 / US-75-009: Build video_metadata dict for context enrichment
        video_metadata = self._build_video_metadata(state)
        if video_metadata:
            logger.info(f"US-75-009: Built video_metadata for {len(video_metadata)} videos")

        # US-75-010: Pass listicle_groups from state to match_all_segments
        listicle_groups = getattr(state, 'listicle_groups', None)

        # US-85-005: Restore partial matches from checkpoint for within-stage resumption
        start_index = 0
        prior_results = None
        if checkpoint is not None:
            start_index, prior_results = self._restore_partial_matches(checkpoint)

        # US-85-005: Build progress callback for intermediate checkpointing
        checkpoint_interval = getattr(
            config.matching, 'intermediate_checkpoint_interval', 25
        )
        progress_callback = None
        if checkpoint is not None and checkpoint_interval > 0:
            progress_callback = self._make_checkpoint_callback(
                checkpoint, checkpoint_interval
            )

        matches = match_all_segments(
            voiceover_segments=vo_segments,
            video_segments=video_segments,
            voiceover_embeddings=vo_embeddings,
            video_embeddings=video_embeddings,
            scenes=None,
            config=config,
            cache=cache,
            embedding_index=None,
            face_preference=state.face_preference,
            video_topics=None,
            location_chapters=getattr(state, 'location_chapters', None),
            video_locations=None,
            video_metadata=video_metadata,
            listicle_groups=listicle_groups,
            progress_callback=progress_callback,
            start_index=start_index,
            prior_results=prior_results,
        )

        # US-85-005: Clear partial_matches from checkpoint on successful completion
        if checkpoint is not None:
            self._clear_partial_matches(checkpoint)

        return matches

    def _restore_partial_matches(
        self,
        checkpoint: 'CheckpointManager'
    ) -> tuple:
        """Restore partial match results from checkpoint for within-stage resumption.

        US-85-005: If the MATCH stage was interrupted mid-way, partial results
        are stored under a 'partial_matches' key. On resume, already-matched
        segments are skipped.

        Returns:
            (start_index, prior_results): Index to resume from and pre-populated results list.
            Returns (0, None) if no partial matches found.
        """
        from ..state import restore_matches_from_dicts

        try:
            data = checkpoint.get_stage_data(self.name)
            if not data or not isinstance(data, dict):
                return 0, None

            partial = data.get('partial_matches')
            if not partial or not isinstance(partial, dict):
                return 0, None

            serialized = partial.get('matches', [])
            last_index = partial.get('last_completed_index', -1)

            if not serialized or last_index < 0:
                return 0, None

            restored = restore_matches_from_dicts(
                serialized, default_strategy='partial_resume', logger_instance=logger
            )
            if restored is None:
                logger.warning("Failed to deserialize partial matches — starting fresh")
                return 0, None

            start_index = last_index + 1
            logger.info(
                f"US-85-005: Restored {len(restored)} partial matches from checkpoint, "
                f"resuming from segment {start_index}"
            )
            return start_index, restored

        except Exception as e:
            logger.warning(f"Failed to restore partial matches: {e}")
            return 0, None

    def _make_checkpoint_callback(
        self,
        checkpoint: 'CheckpointManager',
        interval: int
    ):
        """Create a progress callback that saves intermediate match checkpoints.

        US-85-005: Returns a callable(index, results) that saves every `interval` segments.
        """
        from ..matching.serialization import serialize_match_for_match_stage

        def _callback(index: int, results: List[Any]) -> None:
            # Only checkpoint every N segments
            if (index + 1) % interval != 0:
                return

            try:
                serialized = []
                for i, m in enumerate(results):
                    try:
                        serialized.append(serialize_match_for_match_stage(m, i))
                    except Exception:
                        pass  # Skip unserializable matches

                partial_data = {
                    'partial_matches': {
                        'last_completed_index': index,
                        'match_count': len(serialized),
                        'matches': serialized,
                    }
                }
                checkpoint.save_intermediate(self.name, partial_data)
                logger.debug(
                    f"US-85-005: Saved intermediate checkpoint at segment {index} "
                    f"({len(serialized)} matches)"
                )
            except Exception as e:
                logger.debug(f"Intermediate match checkpoint failed: {e}")

        return _callback

    def _clear_partial_matches(self, checkpoint: 'CheckpointManager') -> None:
        """Remove partial_matches from checkpoint data after successful completion.

        US-85-005: Prevents stale partial data from being picked up on future runs.
        """
        try:
            data = checkpoint.get_stage_data(self.name)
            if data and isinstance(data, dict) and 'partial_matches' in data:
                del data['partial_matches']
                checkpoint.save_intermediate(self.name, data)
                logger.debug("US-85-005: Cleared partial_matches from checkpoint")
        except Exception as e:
            logger.debug(f"Failed to clear partial matches: {e}")
