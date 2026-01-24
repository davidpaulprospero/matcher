"""
Match Stage - Voiceover to Video Matching

Stage 4 of the video matching pipeline:
- Matches voiceover segments to video clips
- Uses embedding similarity and LLM reranking
- Supports delta matching for incremental updates
- Handles chapter and location-aware matching
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

    Inputs:
        - state.voiceover_segments: List of VoiceoverSegment
        - state.embeddings: Video embeddings
        - state.text_metadata: Video segment metadata
        - state.embedding_index: FAISS index

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

            if not state.text_metadata or is_embeddings_empty(state.embeddings):
                print("  ! No video data to match against")
                warnings.append("No video embeddings")
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

                        serialized_matches.append({
                            'segment_index': i,
                            'source_file': source_file,
                            'start_time': float(start_time),
                            'confidence': float(conf)
                        })
                    # Handle direct Match structure
                    elif hasattr(m, 'video_segment'):
                        source_file = getattr(m.video_segment, 'source_file', '')
                        start_time = getattr(m.video_segment, 'start_time', 0.0)
                        conf = getattr(m, 'confidence', 0.0)

                        serialized_matches.append({
                            'segment_index': i,
                            'source_file': source_file,
                            'start_time': float(start_time),
                            'confidence': float(conf)
                        })
                except Exception as e:
                    logger.warning(f"Failed to serialize match {i}: {e}")

            checkpoint_data = {
                'match_count': len(matches),
                'avg_confidence': avg_conf,
                'matches': serialized_matches,  # Essential match data for validation
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
        """Restore match stage from checkpoint"""
        from ..state import Match

        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # Load matches from checkpoint
            matches_data = data.get('matches', [])
            if matches_data:
                restored_matches = []
                for m in matches_data:
                    # Handle both old format (source_file) and new format (video_file)
                    video_file = m.get('video_file') or m.get('source_file', '')

                    # Estimate video_end if not provided (old checkpoints)
                    video_start = m.get('video_start', m.get('start_time', 0.0))
                    video_end = m.get('video_end', video_start + 10.0)  # Default 10s clip

                    match = Match(
                        segment_index=m.get('segment_index', 0),
                        video_file=video_file,
                        video_start=video_start,
                        video_end=video_end,
                        confidence=m.get('confidence', 0.0),
                        strategy=m.get('strategy', 'restored'),
                        reason=m.get('reason', ''),
                        face_score=m.get('face_score', 0.5)
                    )
                    restored_matches.append(match)

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
        missing_fields = []

        if not state.voiceover_segments:
            missing_fields.append("voiceover_segments")

        if is_embeddings_empty(state.embeddings):
            missing_fields.append("embeddings")

        if not state.text_metadata:
            missing_fields.append("text_metadata")

        if missing_fields:
            fields_str = ", ".join(missing_fields)
            suggestions = []

            if "voiceover_segments" in missing_fields:
                suggestions.append("run ANALYZE stage first to extract voiceover segments")

            if "embeddings" in missing_fields or "text_metadata" in missing_fields:
                suggestions.append("run TRANSCRIBE stage first to generate video embeddings and metadata")

            suggestion_str = "; ".join(suggestions) if suggestions else "check pipeline configuration"
            return f"Missing required fields: {fields_str}. Suggestion: {suggestion_str}"

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
                video_paths_set.add(meta.get('video_path', ''))
            else:
                vid_segment = meta
                video_paths_set.add(getattr(meta, 'source_file', ''))

            video_segments.append(vid_segment)

        logger.info(f"Created {broll_segments_created} video_segments with is_broll=True")

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
