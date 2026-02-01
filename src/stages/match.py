"""
Match Stage - Voiceover to Video Matching

Stage 4 of the simplified 7-stage pipeline:
- Matches voiceover segments to video clips
- Uses caption text similarity and LLM reranking
- Supports delta matching for incremental updates
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage
from ..logger import get_global_logger
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
        - (Optional) state.embeddings, state.text_metadata: For embedding-based matching

    Outputs:
        - state.matches: List of Match objects
        - state.alternatives: Dict of alternative matches
    """

    name = "MATCH"
    description = "Match voiceover segments to video clips"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the match stage"""
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

            # In simplified pipeline, text_metadata comes from CAPTION stage
            # Embeddings are optional for caption-first matching
            if not state.text_metadata:
                print("  ! No video data to match against")
                warnings.append("No video text metadata (run CAPTION stage)")
                return StageResult.ok({'matches': []}, warnings)

            # Print settings
            self._print_settings(config)

            # Prepare segments
            vo_segments, video_segments, all_video_paths = self._prepare_segments(state)

            # Check delta matching
            force_rematch = getattr(config.matching, 'force_rematch', False)
            delta_enabled = getattr(config.matching, 'delta_matching_enabled', True)

            # Run matching
            matches = self._run_matching(
                vo_segments, video_segments, all_video_paths,
                state, config, delta_enabled, force_rematch
            )

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
            from ..matching.metrics import calculate_match_quality_metrics, log_quality_summary, log_confidence_histogram
            quality_metrics = calculate_match_quality_metrics(
                matches=matches,
                total_segments=len(state.voiceover_segments)
            )
            log_quality_summary(quality_metrics)

            # Log confidence distribution histogram
            if confidences:
                log_confidence_histogram(confidences)

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
                    # Handle MatchResult structure (has primary_match)
                    if hasattr(m, 'primary_match') and m.primary_match:
                        pm = m.primary_match
                        source_file = ''
                        start_time = 0.0
                        conf = 0.0

                        if hasattr(pm, 'video_segment') and pm.video_segment:
                            source_file = getattr(pm.video_segment, 'source_file', '')
                            start_time = getattr(pm.video_segment, 'start_time', 0.0)

                        conf = getattr(pm, 'confidence', 0.0)
                        # Get confidence variance from MatchResult
                        conf_variance = getattr(m, 'confidence_variance', 0.0)
                        # Get matched keywords from MatchResult
                        matched_kws = getattr(m, 'matched_keywords', [])

                        serialized_matches.append({
                            'segment_index': i,
                            'source_file': source_file,
                            'start_time': float(start_time),
                            'confidence': float(conf),
                            'confidence_variance': float(conf_variance),
                            'matched_keywords': list(matched_kws) if matched_kws else []
                        })
                    # Handle direct Match structure (no confidence_variance/matched_keywords available)
                    elif hasattr(m, 'video_segment'):
                        source_file = getattr(m.video_segment, 'source_file', '')
                        start_time = getattr(m.video_segment, 'start_time', 0.0)
                        conf = getattr(m, 'confidence', 0.0)

                        serialized_matches.append({
                            'segment_index': i,
                            'source_file': source_file,
                            'start_time': float(start_time),
                            'confidence': float(conf),
                            'confidence_variance': 0.0,  # Not available for direct Match
                            'matched_keywords': []  # Not available for direct Match
                        })
                except Exception as e:
                    logger.warning(f"Failed to serialize match {i}: {e}")

            checkpoint_data = {
                'match_count': len(matches),
                'avg_confidence': avg_conf,
                'matches': serialized_matches,  # Essential match data for validation
                'quality_metrics': quality_metrics.to_dict(),  # Quality metrics for analysis
            }

            return StageResult.ok(checkpoint_data, warnings)

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
        from ..state import Match

        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}: checkpoint returned None")
                return False

            # Load matches from checkpoint
            matches_data = data.get('matches', [])
            if matches_data:
                # Validate matches_data is a list
                if not isinstance(matches_data, list):
                    logger.warning(f"Invalid checkpoint data for {self.name}: 'matches' is not a list (got {type(matches_data).__name__})")
                    return False

                restored_matches = []
                validation_errors = []

                for i, m in enumerate(matches_data):
                    # Validate each match object is a dict
                    if not isinstance(m, dict):
                        validation_errors.append(f"match[{i}] is not a dict")
                        continue

                    # Handle both old format (source_file) and new format (video_file)
                    video_file = m.get('video_file') or m.get('source_file', '')

                    # Validate video_file is a non-empty string
                    if not video_file or not isinstance(video_file, str):
                        validation_errors.append(f"match[{i}] has invalid video_file: {repr(video_file)}")
                        continue

                    # Validate segment_index is an integer
                    segment_index = m.get('segment_index', 0)
                    if not isinstance(segment_index, (int, float)):
                        validation_errors.append(f"match[{i}] has invalid segment_index: {repr(segment_index)}")
                        continue
                    segment_index = int(segment_index)

                    # Validate confidence is a valid float
                    confidence = m.get('confidence', 0.0)
                    if not isinstance(confidence, (int, float)):
                        validation_errors.append(f"match[{i}] has invalid confidence: {repr(confidence)}")
                        continue
                    confidence = float(confidence)
                    if not (0.0 <= confidence <= 1.0):
                        logger.debug(f"match[{i}] confidence {confidence} out of range [0, 1], clamping")
                        confidence = max(0.0, min(1.0, confidence))

                    # Estimate video_end if not provided (old checkpoints)
                    video_start = m.get('video_start', m.get('start_time', 0.0))
                    video_end = m.get('video_end', video_start + 10.0)  # Default 10s clip

                    match = Match(
                        segment_index=segment_index,
                        video_file=video_file,
                        video_start=float(video_start),
                        video_end=float(video_end),
                        confidence=confidence,
                        strategy=m.get('strategy', 'restored'),
                        reason=m.get('reason', ''),
                        face_score=m.get('face_score', 0.5)
                    )
                    restored_matches.append(match)

                # Log validation errors but don't fail if we got some valid matches
                if validation_errors:
                    logger.warning(f"Match validation errors during restore: {validation_errors[:5]}")
                    if len(validation_errors) > 5:
                        logger.warning(f"... and {len(validation_errors) - 5} more validation errors")

                # Return False if no valid matches were restored from non-empty data
                if not restored_matches and matches_data:
                    logger.warning(f"No valid matches restored from {len(matches_data)} checkpoint entries")
                    return False

                state.matches = restored_matches
                logger.info(f"Restored MATCH: {len(restored_matches)} matches from checkpoint")
            else:
                logger.info(f"Restored MATCH metadata from checkpoint (no matches data)")

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

        return None

    # === Helper Methods ===

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
        force_rematch: bool
    ) -> List[Any]:
        """Run the actual matching algorithm"""
        from ..matching import match_all_segments
        from ..utils import CacheManager
        from ..embeddings import compute_embeddings, get_embedding_provider

        cache_dir = config.cache.cache_dir if hasattr(config.cache, 'cache_dir') else ".cache"
        provider = get_embedding_provider(config)
        cache = CacheManager(cache_dir)

        # Compute voiceover embeddings
        print(f"  Computing voiceover embeddings...")
        vo_texts = [seg.text for seg in vo_segments]

        vo_embeddings = compute_embeddings(
            texts=vo_texts,
            provider=provider,
            cache=cache,
            cache_key="voiceover"
        )

        if vo_embeddings is None or len(vo_embeddings) == 0:
            print("  ! Failed to compute voiceover embeddings")
            return []

        # Run matching
        print(f"  Running two-stage matching...")

        matches = match_all_segments(
            voiceover_segments=vo_segments,
            video_segments=video_segments,
            voiceover_embeddings=vo_embeddings,
            video_embeddings=state.embeddings,
            scenes=None,
            config=config,
            cache=cache,
            embedding_index=state.embedding_index,
            face_preference=state.face_preference,
            video_topics=None,
            location_chapters=state.location_chapters or None,
            video_locations=None
        )

        return matches
