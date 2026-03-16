"""
Matching strategies for alternative track generation.

Strategies:
- embedding_diversity: Find maximally different clips from V1-V3
- broll_only: Silent footage only (no faces/speech)
- source_rotation: Cycle through sources for maximum variety

Created during strategy track refactoring (Jan 2026).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Any
from pathlib import Path

from ..embeddings import cosine_similarity
from .types import StrategyMatch, VideoScene

if TYPE_CHECKING:
    from ..config import Config
    from ..state import SRTSegment

logger = logging.getLogger(__name__)


class StrategyMatcher:
    """
    Generates alternative matches using different strategies.

    Strategies provide variety for editors by selecting clips based on
    different criteria than the primary match.
    """

    def __init__(self, config: 'Config'):
        self.config = config
        self.mc = config.matching
        # Cached instances (lazy-initialized)
        self._global_cache = None
        self._clip_id_cache: dict = {}  # segment id -> clip_id

    def get_clip_id(self, seg: SRTSegment) -> str:
        """Generate unique clip ID for deduplication (cached)."""
        seg_id = id(seg)
        if seg_id not in self._clip_id_cache:
            self._clip_id_cache[seg_id] = f"{seg.source_file}:{seg.start_time}-{seg.end_time}"
        return self._clip_id_cache[seg_id]

    def _get_global_cache(self):
        """Get or create global video cache instance (lazy singleton)."""
        if self._global_cache is None:
            try:
                from ..global_cache import GlobalCacheManager
                gc_config = getattr(self.config, 'global_cache', None)
                if gc_config:
                    cache_dir = getattr(gc_config, 'cache_dir', None)
                    if cache_dir:
                        self._global_cache = GlobalCacheManager(cache_dir)
            except Exception:
                pass
        return self._global_cache

    def _get_scene_for_segment(self, segment: SRTSegment) -> Optional['VideoScene']:
        """Get scene data for a segment if available (uses cached global cache)."""
        scene_index = getattr(segment, 'scene_index', None)
        if scene_index is None:
            return None

        # Use cached global cache instance
        try:
            cache = self._get_global_cache()
            if cache is None:
                return None

            video_name = Path(segment.source_file).stem
            entry = cache.get_video_entry(video_name)
            if entry and hasattr(entry, 'scenes') and entry.scenes and scene_index < len(entry.scenes):
                from .types import VideoScene
                scene_data = entry.scenes[scene_index]
                return VideoScene(
                    start_time=scene_data.get('start_time', 0),
                    end_time=scene_data.get('end_time', 0),
                    description=scene_data.get('description', ''),
                    visual_keywords=scene_data.get('visual_keywords', []),
                    is_broll=scene_data.get('is_broll', False),
                    face_score=scene_data.get('face_score', 0.5)
                )
        except Exception:
            pass

        return None

    def match_embedding_diversity(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]],
        candidate_embeddings: Dict[str, List[float]],
        vo_embedding: List[float]
    ) -> Optional[StrategyMatch]:
        """
        Strategy: Embedding Diversity (V7 track).

        Finds clips that are maximally different from already-selected clips
        while still being relevant to the voiceover. Uses embedding distance
        to ensure visual variety.

        Scoring:
        - High relevance to voiceover (embedding similarity)
        - High distance from existing matches (embedding diversity)
        - Prefers clips from different source files

        Args:
            vo_segment: Voiceover segment to match.
            all_candidates: List of (video_segment, text_similarity) tuples.
            existing_matches: Already-selected clips (to be different from).
            existing_embeddings: Embeddings of existing matches.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.
            vo_embedding: Embedding vector of the voiceover segment.

        Returns:
            StrategyMatch with strategy="embedding_diversity", or None if
            no suitable diverse candidate found.
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))

        # Get used sources and clips
        used_sources = set(seg.source_file for seg in existing_matches)
        used_clips = set(self.get_clip_id(seg) for seg in existing_matches)

        best_candidate = None
        best_score = -1

        for seg, text_sim in all_candidates:
            cand_id = self.get_clip_id(seg)

            # Skip already-used clips
            if cand_id in used_clips:
                continue

            # Get embedding
            cand_emb = candidate_embeddings.get(cand_id)
            if not _has_emb(cand_emb):
                continue

            # Calculate relevance to voiceover
            if _has_emb(vo_embedding):
                vo_relevance = cosine_similarity(cand_emb, vo_embedding)
            else:
                vo_relevance = text_sim

            # Calculate diversity from existing matches
            diversity_score = 0.0
            if existing_embeddings:
                diversities = []
                for existing_emb in existing_embeddings:
                    if _has_emb(existing_emb):
                        # Distance = 1 - similarity
                        dist = 1.0 - cosine_similarity(cand_emb, existing_emb)
                        diversities.append(dist)
                if diversities:
                    diversity_score = sum(diversities) / len(diversities)

            # Combined score: relevance * diversity
            # This rewards clips that are relevant but visually different
            combined_score = vo_relevance * (0.5 + 0.5 * diversity_score)

            # Boost for different source
            if seg.source_file not in used_sources:
                combined_score *= 1.1

            if combined_score > best_score:
                best_score = combined_score
                best_candidate = seg

        if best_candidate:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=best_score,
                reasoning="Embedding diversity match",
                strategy="embedding_diversity"
            )

        return None

    def match_broll_only(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]],
        candidate_embeddings: Dict[str, List[float]],
        vo_embedding: List[float]
    ) -> Optional[StrategyMatch]:
        """
        Strategy: B-roll Only matching (V8 track).

        Finds clips that are EXCLUSIVELY B-roll (silent footage with no
        speech/faces). Provides editors with a guaranteed silent footage
        option that won't have audio conflicts with the voiceover.

        B-roll detection sources:
        1. Face detection: face_score < 0.3 during SceneDetectionStage
        2. Silent detection: word_count < broll.min_words_threshold (default: 10)
           during BrollMatchStage

        Scoring:
        - Returns highest-relevance B-roll clip (embedding similarity to
            voiceover or original text_similarity as fallback).
        - Enforces different source from existing matches for variety.

        Args:
            vo_segment: Voiceover segment to match.
            all_candidates: List of (video_segment, text_similarity) tuples.
                Only candidates with is_broll=True are considered.
            existing_matches: Already-selected clips (excluded for variety).
            existing_embeddings: Embeddings of existing matches.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.
            vo_embedding: Embedding vector of the voiceover segment.

        Returns:
            StrategyMatch with strategy="broll_only", or None if no B-roll
            candidates available.

        Notes:
            - Requires is_broll=True attribute on video segments.
            - B-roll flag is set by SceneDetectionStage and BrollMatchStage.
            - Config: broll.enabled must be true for B-roll detection.
            - Logs count of B-roll candidates for debugging.
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))

        # Get used sources and clips to enforce variety
        used_sources = set(seg.source_file for seg in existing_matches)
        used_clips = set(self.get_clip_id(seg) for seg in existing_matches)

        best_candidate = None
        best_score = -1

        # DEBUG: Count total candidates and B-roll candidates
        broll_count = sum(1 for seg, _ in all_candidates if getattr(seg, 'is_broll', False))
        logger.info(f"B-roll strategy: {broll_count}/{len(all_candidates)} candidates have is_broll=True")

        # First pass: find best B-roll candidate with strict variety rules
        for seg, text_sim in all_candidates:
            # CRITICAL: Only consider B-roll segments
            if not getattr(seg, 'is_broll', False):
                continue

            cand_id = self.get_clip_id(seg)

            # Exclude already-used clips (same clip can't appear on multiple tracks)
            if cand_id in used_clips:
                continue

            # Prefer different source files for variety
            if seg.source_file in used_sources:
                continue

            # Score by embedding similarity to voiceover (relevance)
            cand_emb = candidate_embeddings.get(cand_id)
            if _has_emb(cand_emb) and _has_emb(vo_embedding):
                vo_relevance = cosine_similarity(cand_emb, vo_embedding)
            else:
                vo_relevance = text_sim

            if vo_relevance > best_score:
                best_score = vo_relevance
                best_candidate = seg

        # Second pass: relax source variety requirement if no match found
        if not best_candidate:
            logger.info(f"B-roll strategy: no match with strict variety, relaxing source requirement")
            for seg, text_sim in all_candidates:
                if not getattr(seg, 'is_broll', False):
                    continue

                cand_id = self.get_clip_id(seg)
                if cand_id in used_clips:
                    continue

                # Allow same source, just exclude exact same clip
                cand_emb = candidate_embeddings.get(cand_id)
                if _has_emb(cand_emb) and _has_emb(vo_embedding):
                    vo_relevance = cosine_similarity(cand_emb, vo_embedding)
                else:
                    vo_relevance = text_sim

                if vo_relevance > best_score:
                    best_score = vo_relevance
                    best_candidate = seg

        # Third pass: allow used clips if still no match (different time range)
        if not best_candidate:
            logger.info(f"B-roll strategy: no match with relaxed rules, allowing used clips")
            for seg, text_sim in all_candidates:
                if not getattr(seg, 'is_broll', False):
                    continue

                cand_emb = candidate_embeddings.get(self.get_clip_id(seg))
                if _has_emb(cand_emb) and _has_emb(vo_embedding):
                    vo_relevance = cosine_similarity(cand_emb, vo_embedding)
                else:
                    vo_relevance = text_sim

                if vo_relevance > best_score:
                    best_score = vo_relevance
                    best_candidate = seg

        # Final fallback: use any candidate with low face_score (likely B-roll even if not flagged)
        if not best_candidate:
            logger.info(f"B-roll strategy: no flagged B-roll, trying face_score fallback")
            for seg, text_sim in all_candidates:
                face_score = getattr(seg, 'face_score', 0.5)
                if face_score < 0.5:  # Likely B-roll even if not explicitly flagged
                    cand_emb = candidate_embeddings.get(self.get_clip_id(seg))
                    if _has_emb(cand_emb) and _has_emb(vo_embedding):
                        vo_relevance = cosine_similarity(cand_emb, vo_embedding)
                    else:
                        vo_relevance = text_sim

                    if vo_relevance > best_score:
                        best_score = vo_relevance
                        best_candidate = seg

        if best_candidate:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=best_score,
                reasoning="B-roll only match (silent footage)",
                strategy="broll_only"
            )

        return None

    def get_secondary_matches_diversity(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        primary_match: SRTSegment,
        secondary_matches: List[SRTSegment],
        candidate_embeddings: Dict[str, List[float]],
        vo_embedding: List[float],
        num_matches: int = 3
    ) -> List[StrategyMatch]:
        """
        Get secondary matches using embedding diversity.

        Secondary matches (V4-V6) use different source videos from V1-V3
        to provide editors with alternative footage options.

        Args:
            vo_segment: Voiceover segment to match
            all_candidates: All candidate clips
            primary_match: The primary match (V1)
            secondary_matches: Already selected secondary matches
            candidate_embeddings: Embedding lookup
            vo_embedding: Voiceover embedding
            num_matches: Number of secondary matches to return

        Returns:
            List of StrategyMatch objects
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))

        results = []

        # Get sources used in primary and existing secondary matches
        used_sources = {primary_match.source_file}
        for sm in secondary_matches:
            used_sources.add(sm.source_file)

        used_clips = {self.get_clip_id(primary_match)}
        for sm in secondary_matches:
            used_clips.add(self.get_clip_id(sm))

        # Get embeddings for diversity calculation
        existing_embs = []
        for seg in [primary_match] + secondary_matches:
            emb = candidate_embeddings.get(self.get_clip_id(seg))
            if _has_emb(emb):
                existing_embs.append(emb)

        # Find candidates from different sources
        for seg, text_sim in all_candidates:
            cand_id = self.get_clip_id(seg)

            # Skip if already used
            if cand_id in used_clips:
                continue

            # Prefer different source
            if seg.source_file in used_sources:
                continue

            # Get embedding
            cand_emb = candidate_embeddings.get(cand_id)
            if not _has_emb(cand_emb):
                continue

            # Calculate relevance
            if _has_emb(vo_embedding):
                vo_relevance = cosine_similarity(cand_emb, vo_embedding)
            else:
                vo_relevance = text_sim

            # Calculate diversity from existing matches
            diversity_score = 0.0
            if existing_embs:
                diversities = [1.0 - cosine_similarity(cand_emb, e) for e in existing_embs]
                diversity_score = sum(diversities) / len(diversities)

            # Combined score
            combined_score = vo_relevance * (0.5 + 0.5 * diversity_score)

            results.append(StrategyMatch(
                video_segment=seg,
                video_scene=self._get_scene_for_segment(seg),
                confidence=combined_score,
                reasoning="Secondary diversity match",
                strategy="secondary_diversity"
            ))

            used_sources.add(seg.source_file)
            used_clips.add(cand_id)
            existing_embs.append(cand_emb)

            if len(results) >= num_matches:
                break

        return results

    def match_source_rotation(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]],
        candidate_embeddings: Dict[str, List[float]],
        vo_embedding: List[float],
        segment_index: int
    ) -> Optional[StrategyMatch]:
        """
        Strategy: Source Rotation.

        Cycles through different source videos in round-robin fashion.
        Ensures no single source dominates the timeline.

        Args:
            vo_segment: Voiceover segment to match.
            all_candidates: List of (video_segment, text_similarity) tuples.
            existing_matches: Already-selected clips.
            existing_embeddings: Embeddings of existing matches.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.
            vo_embedding: Embedding vector of the voiceover segment.
            segment_index: Index of current segment (for rotation).

        Returns:
            StrategyMatch with strategy="source_rotation", or None.
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))

        # Get unique sources
        sources = list(set(seg.source_file for seg, _ in all_candidates))
        if not sources:
            return None

        # Rotate through sources based on segment index
        source_idx = segment_index % len(sources)
        target_source = sources[source_idx]

        # Find best candidate from target source
        best_candidate = None
        best_score = -1

        for seg, text_sim in all_candidates:
            if seg.source_file != target_source:
                continue

            cand_id = self.get_clip_id(seg)
            cand_emb = candidate_embeddings.get(cand_id)

            if _has_emb(cand_emb) and _has_emb(vo_embedding):
                score = cosine_similarity(cand_emb, vo_embedding)
            else:
                score = text_sim

            if score > best_score:
                best_score = score
                best_candidate = seg

        if best_candidate:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=best_score,
                reasoning=f"Source rotation: {Path(target_source).name}",
                strategy="source_rotation"
            )

        return None

    def get_strategy_matches(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        primary_match: SRTSegment,
        secondary_matches: List[SRTSegment],
        candidate_embeddings: Dict[str, List[float]],
        vo_embedding: List[float],
        segment_index: int
    ) -> List['StrategyMatch']:
        """
        Generate all strategy matches for a voiceover segment.

        Args:
            vo_segment: Voiceover segment to match
            all_candidates: All candidate clips with scores
            primary_match: Primary match (V1)
            secondary_matches: Secondary matches (V4-V6)
            candidate_embeddings: Embedding lookup dict
            vo_embedding: Voiceover embedding
            segment_index: Current segment index

        Returns:
            List of StrategyMatch objects for strategy tracks
        """
        strategy_matches = []

        # Get embeddings for existing matches
        existing_embs = []
        for seg in [primary_match] + secondary_matches:
            emb = candidate_embeddings.get(self.get_clip_id(seg))
            if emb is not None and len(emb) > 0:
                existing_embs.append(emb)

        all_existing = [primary_match] + secondary_matches

        # Get configured strategies
        strategies = getattr(self.config.output, 'strategy_tracks', [])

        for strategy in strategies:
            match = None
            if strategy == "embedding_diversity":
                match = self.match_embedding_diversity(
                    vo_segment, all_candidates, all_existing,
                    existing_embs, candidate_embeddings, vo_embedding
                )
            elif strategy == "source_rotation":
                match = self.match_source_rotation(
                    vo_segment, all_candidates, all_existing,
                    existing_embs, candidate_embeddings, vo_embedding,
                    segment_index
                )
            elif strategy == "broll_only":
                match = self.match_broll_only(
                    vo_segment, all_candidates, all_existing,
                    existing_embs, candidate_embeddings, vo_embedding
                )

            if match:
                strategy_matches.append(match)

        return strategy_matches


class FallbackMatchStrategy:
    """
    Fallback matching strategy for edge cases when primary matching fails.

    Provides 3 fallback levels, each with a confidence ceiling:
    - Level 1: Keyword-only matching (ceiling: 0.7)
    - Level 2: Visual-description matching (ceiling: 0.5)
    - Level 3: Generic B-roll matching (ceiling: 0.3)

    Triggered when primary matching returns confidence < trigger_threshold (default: 0.4)
    """

    # Confidence ceilings for each fallback level
    KEYWORD_ONLY_CEILING = 0.7
    VISUAL_DESCRIPTION_CEILING = 0.5
    GENERIC_BROLL_CEILING = 0.3

    def __init__(self, config: Config) -> None:
        """
        Initialize FallbackMatchStrategy.

        Args:
            config: Configuration object with matching settings
        """
        self.config = config
        mc = config.matching
        self.enabled = getattr(mc, 'fallback_matching_enabled', True)
        self.trigger_threshold = getattr(mc, 'fallback_trigger_threshold', 0.4)

    def should_trigger(self, primary_confidence: float) -> bool:
        """
        Check if fallback matching should be triggered.

        Args:
            primary_confidence: Confidence score from primary matching

        Returns:
            True if fallback should be triggered
        """
        if not self.enabled:
            return False
        return primary_confidence < self.trigger_threshold

    def match_keyword_only(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]]
    ) -> Optional[Tuple[SRTSegment, float, str]]:
        """
        Level 1 Fallback: Return best keyword match.

        Uses the text_similarity score already computed during candidate
        selection. This is the highest-confidence fallback.

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (segment, text_similarity) tuples

        Returns:
            Tuple of (best_segment, adjusted_confidence, reasoning) or None
        """
        if not candidates:
            return None

        # Find best by text similarity
        best_seg, best_sim = max(candidates, key=lambda x: x[1])

        # Apply ceiling
        confidence = min(best_sim * self.KEYWORD_ONLY_CEILING, self.KEYWORD_ONLY_CEILING)

        return (best_seg, confidence, "Fallback L1: keyword-only match")

    def match_visual_description(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]]
    ) -> Optional[Tuple[SRTSegment, float, str]]:
        """
        Level 2 Fallback: Return best visual-description match.

        Attempts to match based on visual keywords from scene detection.
        Lower ceiling than keyword-only.

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (segment, text_similarity) tuples

        Returns:
            Tuple of (best_segment, adjusted_confidence, reasoning) or None
        """
        if not candidates:
            return None

        # For now, use same as keyword-only but with lower ceiling
        # TODO: Implement visual keyword matching
        best_seg, best_sim = max(candidates, key=lambda x: x[1])
        confidence = min(best_sim * self.VISUAL_DESCRIPTION_CEILING, self.VISUAL_DESCRIPTION_CEILING)

        return (best_seg, confidence, "Fallback L2: visual-description match")

    def match_generic_broll(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]]
    ) -> Optional[Tuple[SRTSegment, float, str]]:
        """
        Level 3 Fallback: Return best available B-roll/generic footage.

        Last resort fallback that returns any silent/B-roll footage,
        or the best embedding match if no B-roll available.
        Has the lowest confidence ceiling (0.3).

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (segment, text_similarity) tuples

        Returns:
            Tuple of (best_segment, adjusted_confidence, reasoning) or None
        """
        if not candidates:
            return None

        # First try: find B-roll candidates
        broll_candidates = [
            (seg, sim) for seg, sim in candidates
            if getattr(seg, 'is_broll', False)
        ]

        if broll_candidates:
            # Return best B-roll by embedding similarity
            best_seg, best_sim = max(broll_candidates, key=lambda x: x[1])
            confidence = min(best_sim * self.GENERIC_BROLL_CEILING, self.GENERIC_BROLL_CEILING)
            return (best_seg, confidence, "Fallback L3: generic B-roll")

        # Second try: find short/silent segments (likely B-roll even if not flagged)
        silent_candidates = []
        for seg, sim in candidates:
            # Check if transcript is very short (likely silent/B-roll)
            word_count = len(seg.text.split()) if seg.text else 0
            if word_count < 10:
                silent_candidates.append((seg, sim))

        if silent_candidates:
            best_seg, best_sim = max(silent_candidates, key=lambda x: x[1])
            confidence = min(best_sim * self.GENERIC_BROLL_CEILING, self.GENERIC_BROLL_CEILING)
            return (best_seg, confidence, "Fallback L3: silent segment")

        # Last resort: return best overall candidate with low ceiling
        best_seg, best_sim = max(candidates, key=lambda x: x[1])
        confidence = min(best_sim * self.GENERIC_BROLL_CEILING, self.GENERIC_BROLL_CEILING)
        return (best_seg, confidence, "Fallback L3: best available")

    def get_fallback_match(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]],
        primary_confidence: float
    ) -> Optional[Tuple[SRTSegment, float, str]]:
        """
        Get fallback match if primary matching failed.

        Tries 3 levels of fallback in order:
        1. Keyword-only matching (ceiling: 0.7)
        2. Visual-description matching (ceiling: 0.5)
        3. Generic B-roll matching (ceiling: 0.3)

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of candidate segments
            primary_confidence: Confidence from primary matching

        Returns:
            Tuple of (segment, confidence, reasoning) or None if all fail
        """
        if not self.should_trigger(primary_confidence):
            return None

        # Level 1: Keyword-only
        result = self.match_keyword_only(vo_segment, candidates)
        if result:
            return result

        # Level 2: Visual description
        result = self.match_visual_description(vo_segment, candidates)
        if result:
            return result

        # Level 3: Generic B-roll
        result = self.match_generic_broll(vo_segment, candidates)
        if result:
            return result

        return None
