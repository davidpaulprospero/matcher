"""
Alternative matching strategies for V4-V10 tracks.

Migrated from matching.py lines 1668-2483.

Provides 6 different matching strategies:
- visual_first: Prioritize scene descriptions over transcripts
- different_source: Force selection from different source video
- keyword_only: Match purely on keyword overlap
- embedding_diversity: Find maximally different clips from V1-V3
- broll_only: Silent footage only (no faces/speech)
- source_rotation: Cycle through sources for maximum variety

Also includes get_secondary_matches_diversity for V4-V6 tracks.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict

from ..config import Config
from ..utils import SRTSegment, SceneInfo, StrategyMatch, AlternativeMatch
from ..embeddings import cosine_similarity
from ..keyword_extractor import find_keyword_matches

logger = logging.getLogger(__name__)


def calculate_source_diversity_score(
    candidate_source: str,
    candidate_keywords: Set[str],
    v1_v3_sources: Set[str],
    v1_v3_keywords: Set[str]
) -> float:
    """
    Calculate source diversity score for V4-V6 tracks.

    Scores how different the candidate is from V1-V3 matches based on:
    - Unique source file (different video = higher score)
    - Unique keywords (different content = higher score)

    Args:
        candidate_source: Source file path of the candidate
        candidate_keywords: Keywords from the candidate video segment
        v1_v3_sources: Set of source files used in V1-V3
        v1_v3_keywords: Set of keywords from V1-V3 matches

    Returns:
        Diversity score from 0.0 (same source/keywords) to 1.0 (completely different)
    """
    # Source diversity: 0.6 weight (most important for V4-V6)
    source_score = 1.0 if candidate_source not in v1_v3_sources else 0.0

    # Keyword diversity: 0.4 weight
    if not candidate_keywords:
        # No keywords to compare - neutral score
        keyword_score = 0.5
    elif not v1_v3_keywords:
        # No V1-V3 keywords - full diversity
        keyword_score = 1.0
    else:
        # Calculate keyword overlap ratio
        overlap = candidate_keywords & v1_v3_keywords
        overlap_ratio = len(overlap) / len(candidate_keywords) if candidate_keywords else 0.0
        # Invert: high overlap = low diversity
        keyword_score = 1.0 - overlap_ratio

    # Combined score: 60% source, 40% keywords
    diversity_score = source_score * 0.6 + keyword_score * 0.4

    return round(diversity_score, 2)


class StrategyMatcher:
    """
    Provides alternative matching strategies for variety tracks V4-V10.

    Strategies:
    - visual_first: Prioritize scene descriptions over transcripts
    - different_source: Force selection from different source video
    - keyword_only: Match purely on keyword overlap
    - embedding_diversity: Find maximally different clips from V1-V3
    - broll_only: Silent footage only (requires scene detection)
    - source_rotation: Round-robin through source videos
    """

    def __init__(self, config: Config, scenes: Optional[Dict[str, List[SceneInfo]]]) -> None:
        """
        Initialize the StrategyMatcher with configuration and scene data.

        Args:
            config: Configuration object containing output.variety settings
                that control clip exclusion rules (same_clip, different_source,
                time_distance, embedding_distance, timeline_variety).
            scenes: Dictionary mapping video file paths to lists of SceneInfo
                objects from scene detection. Used for visual matching strategies
                and B-roll detection.

        Notes:
            The variety_config is extracted from config.output.variety and
            handles both dict and object access patterns for flexibility.
        """
        self.config = config
        self.scenes = scenes or {}

        # Handle variety_config as either object or dict
        vc = config.output.variety
        if isinstance(vc, dict):
            # Convert dict to object-like accessor
            class VarietyWrapper:
                def __init__(self, d):
                    self.exclude_same_clip = d.get('exclude_same_clip', True)
                    self.require_different_source = d.get('require_different_source', True)
                    self.min_time_distance = d.get('min_time_distance', 10.0)
                    self.min_embedding_distance = d.get('min_embedding_distance', 0.3)
                    self.enforce_timeline_variety = d.get('enforce_timeline_variety', True)
                    self.timeline_variety_window = d.get('timeline_variety_window', 600.0)
                    self.max_source_repeats_in_window = d.get('max_source_repeats_in_window', 1)
            self.variety_config = VarietyWrapper(vc)
        else:
            self.variety_config = vc

    def get_clip_id(self, segment: SRTSegment) -> str:
        """
        Generate a unique identifier for a video clip.

        The clip ID combines the source file path with the time range,
        creating a unique key that can identify duplicate clips across
        different matching operations.

        Args:
            segment: Video segment to generate ID for.

        Returns:
            String in format "source_file:start_time-end_time" with times
            formatted to 2 decimal places (e.g., "video.mp4:10.50-15.75").
        """
        return f"{segment.source_file}:{segment.start_time:.2f}-{segment.end_time:.2f}"

    def is_clip_excluded(
        self,
        candidate: SRTSegment,
        existing_matches: List[SRTSegment],
        existing_embeddings: Optional[List[List[float]]] = None,
        candidate_embedding: Optional[List[float]] = None,
        force_different_source: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if a candidate clip violates variety enforcement rules.

        Applies four exclusion rules in order:
        1. Same clip exclusion - prevents exact clip reuse
        2. Same source exclusion - optionally prevents clips from same video
        3. Time window exclusion - prevents clips too close in time
        4. Embedding distance - prevents semantically similar clips

        Args:
            candidate: Video segment to check for exclusion.
            existing_matches: List of already-selected video segments to
                compare against.
            existing_embeddings: Optional list of embeddings for existing
                matches, used for semantic similarity checking.
            candidate_embedding: Optional embedding for the candidate clip.
            force_different_source: If True, always reject clips from the
                same source file as any existing match. Used for alternative
                tracks (V4-V10) to ensure variety.

        Returns:
            Tuple of (is_excluded, reason):
            - is_excluded: True if the candidate should be excluded
            - reason: Human-readable explanation of why (empty if not excluded)

        Notes:
            Variety rules are controlled by config.output.variety settings:
            - exclude_same_clip: Prevent exact clip reuse
            - require_different_source: Require different video files
            - min_time_distance: Minimum seconds between clips from same source
            - min_embedding_distance: Minimum cosine distance (0.0-2.0 range)
        """
        vc = self.variety_config

        # Rule 1: Exclude same clip
        if vc.exclude_same_clip:
            cand_id = self.get_clip_id(candidate)
            for existing in existing_matches:
                if self.get_clip_id(existing) == cand_id:
                    return True, "Same clip already used"

        # Rule 2: Require different source (config-based OR forced)
        if vc.require_different_source or force_different_source:
            for existing in existing_matches:
                if candidate.source_file == existing.source_file:
                    # When forcing different source, always reject same source
                    if force_different_source:
                        return True, "Same source file (different source required per track)"
                    # Check time distance within same source
                    if vc.min_time_distance > 0:
                        time_dist = abs(candidate.start_time - existing.start_time)
                        if time_dist < vc.min_time_distance:
                            return True, f"Too close in time ({time_dist:.1f}s < {vc.min_time_distance}s)"
                    else:
                        return True, "Same source file"

        # Rule 3: Time window exclusion (for any source) - only if NOT forcing different source
        if not force_different_source and vc.min_time_distance > 0:
            for existing in existing_matches:
                if candidate.source_file == existing.source_file:
                    time_dist = abs(candidate.start_time - existing.start_time)
                    if time_dist < vc.min_time_distance:
                        return True, f"Within time window ({time_dist:.1f}s)"

        # Rule 4: Minimum embedding distance
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))
        if vc.min_embedding_distance > 0 and _has_emb(candidate_embedding) and _has_emb(existing_embeddings):
            for existing_emb in existing_embeddings:
                similarity = cosine_similarity(candidate_embedding, existing_emb)
                distance = 1.0 - similarity
                if distance < vc.min_embedding_distance:
                    return True, f"Too similar (dist={distance:.2f})"

        return False, ""

    def match_visual_first(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: Optional[List[List[float]]] = None,
        candidate_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy A: Visual-First matching.

        Prioritizes scene descriptions and visual keywords over transcript text,
        finding clips that visually match the voiceover content. Falls back to
        filename/path matching when scene data is unavailable.

        Scoring weights:
        - Visual score (70% weight):
            - Scene description word overlap: up to 0.30 (0.05 per word)
            - Visual keyword overlap: 0.15 per keyword match
            - Filename term matches: 0.10 per term (fallback)
            - Text visual hints: 0.08 per match (fallback)
        - Text similarity (30% weight): Original embedding similarity * 0.3

        Args:
            vo_segment: Voiceover segment to match.
            all_candidates: List of (video_segment, text_similarity) tuples,
                pre-sorted by embedding similarity.
            existing_matches: Already-selected clips for this voiceover
                (used for variety exclusion).
            existing_embeddings: Embeddings of existing matches for
                similarity-based exclusion.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.

        Returns:
            StrategyMatch with strategy="visual_first" and confidence score,
            or None if no suitable match found.

        Notes:
            Visual terms checked: earthquake, tsunami, flood, storm, fire,
            volcano, disaster, building, city, water, wave, destruction,
            damage, rescue, people, crowd, evacuation, explosion, collapse,
            rubble.
        """
        vo_text = vo_segment.text.lower()
        vo_kw = getattr(vo_segment, 'keywords', None)
        vo_keywords = set(vo_kw) if vo_kw else set()

        # Extract key terms from voiceover for visual matching
        visual_terms = {'earthquake', 'tsunami', 'flood', 'storm', 'fire', 'volcano', 'disaster',
                       'building', 'city', 'water', 'wave', 'destruction', 'damage', 'rescue',
                       'people', 'crowd', 'evacuation', 'explosion', 'collapse', 'rubble'}
        vo_visual_hints = set(w.lower() for w in vo_text.split() if w.lower() in visual_terms)

        best_candidate = None
        best_score = -1
        best_reason = ""

        for seg, text_sim in all_candidates:
            # Check variety exclusion (force different source for strategy tracks)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, reason = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=True)
            if is_excluded:
                continue

            # Get scene for this segment
            scene = self._get_scene_for_segment(seg)

            visual_score = 0.0
            text_score = text_sim * 0.3  # Weight text at 30%

            if scene and scene.description:
                # Score based on scene description
                desc = scene.description.lower()
                visual_keywords = set(scene.visual_keywords) if scene.visual_keywords else set()

                # Keyword overlap with visual keywords
                vo_visual_overlap = len(vo_keywords & visual_keywords)
                visual_score += vo_visual_overlap * 0.15

                # Check if voiceover words appear in scene description
                vo_words = set(vo_text.split())
                desc_words = set(desc.split())
                word_overlap = len(vo_words & desc_words)
                visual_score += min(word_overlap * 0.05, 0.3)
            else:
                # Fallback: Use filename/path hints for visual matching
                source_name = Path(seg.source_file).stem.lower() if seg.source_file else ""

                # Check if visual terms from voiceover appear in filename
                filename_matches = sum(1 for term in vo_visual_hints if term in source_name)
                visual_score += filename_matches * 0.1

                # Check if voiceover words appear in video transcript
                seg_text = seg.text.lower()
                vo_visual_in_seg = sum(1 for term in vo_visual_hints if term in seg_text)
                visual_score += vo_visual_in_seg * 0.08

            total_score = visual_score * 0.7 + text_score  # Weight visual at 70%

            if total_score > best_score:
                best_score = total_score
                best_candidate = seg
                best_reason = f"Visual match (visual={visual_score:.2f}, text={text_score:.2f})"

        if best_candidate and best_score > 0:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=min(best_score, 1.0),
                reasoning=best_reason,
                strategy="visual_first"
            )
        return None

    def match_different_source(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: Optional[List[List[float]]] = None,
        candidate_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy B: Different Source Video matching.

        Forces selection from a different video file than existing matches,
        ensuring visual variety by using footage from different source videos.
        This is useful when multiple YouTube videos cover similar content.

        Scoring:
        - Primary: Returns first candidate from unused source with highest
            embedding similarity (confidence = original similarity).
        - Fallback: If no different source available, returns best match
            from any source with 0.8x confidence penalty.

        Args:
            vo_segment: Voiceover segment to match.
            all_candidates: List of (video_segment, text_similarity) tuples,
                pre-sorted by embedding similarity.
            existing_matches: Already-selected clips; their source files
                are excluded from selection.
            existing_embeddings: Embeddings of existing matches (used for
                variety exclusion checks).
            candidate_embeddings: Dict mapping clip_id to embedding vectors.

        Returns:
            StrategyMatch with strategy="different_source", or None if no
            candidates available. Fallback matches have reduced confidence.

        Notes:
            Source file uniqueness is enforced strictly first. Only if no
            unique source exists does it fall back to allowing same sources.
        """
        used_sources = set(seg.source_file for seg in existing_matches)

        for seg, sim in all_candidates:
            # Must be from different source
            if seg.source_file in used_sources:
                continue

            # Check other variety rules (force different source)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, reason = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=True)
            if is_excluded and "source" not in reason.lower():
                continue

            return StrategyMatch(
                video_segment=seg,
                video_scene=self._get_scene_for_segment(seg),
                confidence=sim,
                reasoning=f"Different source: {Path(seg.source_file).stem}",
                strategy="different_source"
            )

        # Fallback: if no different source available, get best that passes other rules
        for seg, sim in all_candidates:
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, _ = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=False)
            if not is_excluded:
                return StrategyMatch(
                    video_segment=seg,
                    video_scene=self._get_scene_for_segment(seg),
                    confidence=sim * 0.8,  # Penalty for not being different source
                    reasoning=f"Fallback (no different source available)",
                    strategy="different_source"
                )

        return None

    def match_keyword_only(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: Optional[List[List[float]]] = None,
        candidate_embeddings: Optional[Dict[str, List[float]]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy C: Keyword-Only matching.

        Matches purely based on keyword and entity overlap, ignoring embedding
        similarity. Useful when semantic embeddings miss domain-specific terms
        or proper nouns. Falls back to significant word overlap if keywords
        aren't populated.

        Scoring weights:
        - Keyword/entity overlap: 0.2 per matching term
        - Text word matches: 0.1 per voiceover keyword found in video text

        Args:
            vo_segment: Voiceover segment to match. Uses .keywords and
                .entities attributes if available.
            all_candidates: List of (video_segment, text_similarity) tuples.
                Similarity score is ignored in favor of keyword matching.
            existing_matches: Already-selected clips (used for variety
                exclusion with force_different_source=True).
            existing_embeddings: Embeddings of existing matches for
                variety checking.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.

        Returns:
            StrategyMatch with strategy="keyword_only" and reasoning showing
            matched keywords, or None if no overlap found.

        Notes:
            - Entities are handled as either strings or dicts with 'text' key.
            - Stop words (this, that, with, from, etc.) are excluded.
            - Minimum word length is 4 characters.
            - Force_different_source=True ensures variety from previous tracks.
        """
        vo_kw = getattr(vo_segment, 'keywords', None) or []
        vo_ent = getattr(vo_segment, 'entities', None) or []
        vo_keywords = set(k.lower() for k in vo_kw)

        # Handle entities as either strings or dicts with 'text' key
        vo_entities = set()
        for e in vo_ent:
            if isinstance(e, dict):
                text = e.get('text', '')
                if text:
                    vo_entities.add(text.lower())
            elif isinstance(e, str):
                vo_entities.add(e.lower())

        vo_all = vo_keywords | vo_entities

        # Fallback: Extract important words from voiceover text if no keywords
        if not vo_all:
            # Extract words 4+ chars, not common stop words
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'where', 'which', 'their', 'there', 'would', 'could', 'should', 'about', 'after', 'before', 'being', 'other', 'these', 'those', 'through', 'during', 'between'}
            vo_words = set(w.lower() for w in vo_segment.text.split() if len(w) >= 4 and w.lower() not in stop_words)
            vo_all = vo_words

        if not vo_all:
            return None

        best_candidate = None
        best_score = 0
        best_overlap = []

        for seg, _ in all_candidates:
            # Check variety exclusion (force different source for strategy tracks)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, _ = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=True)
            if is_excluded:
                continue

            seg_kw = getattr(seg, 'keywords', None) or []
            seg_ent = getattr(seg, 'entities', None) or []
            seg_keywords = set(k.lower() for k in seg_kw)

            # Handle entities as either strings or dicts
            seg_entities = set()
            for e in seg_ent:
                if isinstance(e, dict):
                    text = e.get('text', '')
                    if text:
                        seg_entities.add(text.lower())
                elif isinstance(e, str):
                    seg_entities.add(e.lower())

            seg_all = seg_keywords | seg_entities

            # Also check text for keyword presence
            seg_text = seg.text.lower()
            text_matches = sum(1 for kw in vo_all if kw in seg_text)

            # Extract words from video transcript if no keywords
            if not seg_all:
                seg_words = set(w.lower() for w in seg_text.split() if len(w) >= 4)
                seg_all = seg_words

            overlap = vo_all & seg_all
            score = len(overlap) * 0.2 + text_matches * 0.1

            if score > best_score:
                best_score = score
                best_candidate = seg
                best_overlap = list(overlap)[:5]

        if best_candidate and best_score > 0:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=min(best_score, 1.0),
                reasoning=f"Keyword match: {', '.join(best_overlap) if best_overlap else 'text overlap'}",
                strategy="keyword_only"
            )

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
        Strategy D: Embedding Diversity matching (V7 track).

        Finds clips that are semantically relevant to the voiceover but
        maximally DIFFERENT from V1-V3 matches. Creates variety by selecting
        clips that cover similar topics from different perspectives.

        Scoring weights:
        - Voiceover relevance (40% weight): Cosine similarity to vo_embedding.
            Minimum threshold of 0.3 required.
        - Diversity from existing (60% weight): Average cosine distance from
            all existing match embeddings. Higher = more different.

        Combined score = (vo_relevance * 0.4) + (avg_diversity * 0.6)

        Args:
            vo_segment: Voiceover segment to match.
            all_candidates: List of (video_segment, text_similarity) tuples.
            existing_matches: V1-V3 matches whose embeddings define what
                to be different from.
            existing_embeddings: Embedding vectors of V1-V3 matches.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.
            vo_embedding: Embedding vector of the voiceover segment.

        Returns:
            StrategyMatch with strategy="embedding_diversity" and diversity
            score in reasoning, or None if no candidates with embeddings.

        Notes:
            - Requires existing_embeddings and candidate_embeddings.
            - Enforces different source from V1-V3.
            - Minimum relevance threshold (0.3) prevents returning irrelevant clips.
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))
        if not _has_emb(existing_embeddings) or not candidate_embeddings:
            return None

        # Get used sources to enforce different source
        used_sources = set(seg.source_file for seg in existing_matches)

        best_candidate = None
        best_score = -1
        best_diversity = 0

        for seg, text_sim in all_candidates:
            cand_id = self.get_clip_id(seg)
            cand_emb = candidate_embeddings.get(cand_id)

            if not _has_emb(cand_emb):
                continue

            # Check basic exclusion (same clip)
            if self.variety_config.exclude_same_clip:
                if any(self.get_clip_id(existing) == cand_id for existing in existing_matches):
                    continue

            # Require different source from V1-V3
            if seg.source_file in used_sources:
                continue

            # Calculate diversity: average distance from existing matches
            distances = []
            for existing_emb in existing_embeddings:
                sim = cosine_similarity(cand_emb, existing_emb)
                distances.append(1.0 - sim)

            avg_diversity = sum(distances) / len(distances) if distances else 0

            # Relevance to voiceover (must still be relevant)
            vo_relevance = cosine_similarity(cand_emb, vo_embedding) if _has_emb(vo_embedding) else text_sim

            # Combined score: want high relevance AND high diversity
            # Diversity weighted more heavily
            combined_score = vo_relevance * 0.4 + avg_diversity * 0.6

            if combined_score > best_score and vo_relevance > 0.3:  # Min relevance threshold
                best_score = combined_score
                best_candidate = seg
                best_diversity = avg_diversity

        if best_candidate:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=best_score,
                reasoning=f"Diverse match (diversity={best_diversity:.2f})",
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
        alternatives: List[SRTSegment],
        vo_embedding: List[float],
        candidate_embeddings: Dict[str, List[float]],
        global_used_clips: Optional[Set[str]] = None
    ) -> List[AlternativeMatch]:
        """
        Get secondary matches (V4-V6) using diversity scoring.

        Provides three additional alternative matches beyond V1-V3, each
        strictly from a different source video. Uses the same diversity
        algorithm as match_embedding_diversity (V7) but applied iteratively.

        STRICT enforcement: Each track MUST use a different source video
        from V1-V3 AND from each other.

        Scoring weights (same as embedding_diversity):
        - Voiceover relevance (40%): Cosine similarity to vo_embedding.
        - Diversity from existing (60%): Average cosine distance from ALL
            previously selected clips (V1-V3 + previous V4-V6).

        Combined score = (vo_relevance * 0.4) + (avg_diversity * 0.6)

        Relevance thresholds:
        - Primary: 0.3 minimum (same as V7)
        - Fallback: 0.2 if no candidates meet primary threshold

        Args:
            vo_segment: The voiceover segment being matched.
            all_candidates: All candidate clips with (segment, similarity) tuples.
            primary_match: V1 match segment (excluded from selection).
            alternatives: V2-V3 match segments (excluded from selection).
            vo_embedding: Embedding vector of the voiceover segment.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.
            global_used_clips: Set of clip IDs already used in previous
                voiceover segments (cross-segment deduplication).

        Returns:
            List of up to 3 AlternativeMatch objects for V4, V5, V6.
            Each includes diversity_score calculated via
            calculate_source_diversity_score().

        Notes:
            - Labels: "Secondary Primary", "Secondary Alt 1", "Secondary Alt 2"
            - Source exclusion is cumulative (V5 excludes V1-V4 sources, etc.)
            - Falls back to relaxed threshold if strict matching fails.
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))

        if not candidate_embeddings:
            return []

        # Collect V1-V3 source files, embeddings, and keywords
        v1_v3_sources = {primary_match.source_file}
        v1_v3_embeddings = []
        v1_v3_keywords: Set[str] = set()

        # Get V1 embedding and keywords
        v1_id = self.get_clip_id(primary_match)
        if v1_id in candidate_embeddings:
            v1_v3_embeddings.append(candidate_embeddings[v1_id])
        v1_kw = getattr(primary_match, 'keywords', None) or []
        v1_v3_keywords.update(k.lower() for k in v1_kw)

        # Get V2-V3 embeddings and keywords
        for alt_seg in alternatives:
            if alt_seg:
                v1_v3_sources.add(alt_seg.source_file)
                alt_id = self.get_clip_id(alt_seg)
                if alt_id in candidate_embeddings:
                    v1_v3_embeddings.append(candidate_embeddings[alt_id])
                alt_kw = getattr(alt_seg, 'keywords', None) or []
                v1_v3_keywords.update(k.lower() for k in alt_kw)

        secondary_matches = []
        used_sources = set(v1_v3_sources)  # Start with V1-V3 sources excluded
        existing_embeddings = list(v1_v3_embeddings)

        labels = ["Secondary Primary", "Secondary Alt 1", "Secondary Alt 2"]

        for track_idx in range(3):  # V4, V5, V6
            best_candidate = None
            best_score = -1
            best_diversity = 0
            best_emb = None

            for seg, text_sim in all_candidates:
                # STRICT: Must be from a different source than V1-V3 AND previous secondary tracks
                if seg.source_file in used_sources:
                    continue

                cand_id = self.get_clip_id(seg)

                # Global deduplication: skip clips already used in any previous segment
                if global_used_clips and cand_id in global_used_clips:
                    continue

                cand_emb = candidate_embeddings.get(cand_id)

                if not _has_emb(cand_emb):
                    continue

                # Skip if same exact clip already used in this segment's secondary matches
                if any(self.get_clip_id(sm.video_segment) == cand_id for sm in secondary_matches):
                    continue

                # Calculate diversity: average distance from ALL existing matches (V1-V3 + previous V4-V6)
                if existing_embeddings:
                    distances = []
                    for existing_emb in existing_embeddings:
                        if _has_emb(existing_emb):
                            sim = cosine_similarity(cand_emb, existing_emb)
                            distances.append(1.0 - sim)
                    avg_diversity = sum(distances) / len(distances) if distances else 0
                else:
                    avg_diversity = 0.5  # Default if no embeddings

                # Relevance to voiceover
                vo_relevance = cosine_similarity(cand_emb, vo_embedding) if _has_emb(vo_embedding) else text_sim

                # Combined score: 40% relevance + 60% diversity (same as V7)
                combined_score = vo_relevance * 0.4 + avg_diversity * 0.6

                # Must meet minimum relevance threshold
                if vo_relevance < 0.3:
                    continue

                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = seg
                    best_diversity = avg_diversity
                    best_emb = cand_emb

            # FALLBACK: If no candidate found with strict threshold, try relaxed threshold
            if best_candidate is None:
                # Try with relaxed relevance threshold (0.2 instead of 0.3)
                for seg, text_sim in all_candidates:
                    if seg.source_file in used_sources:
                        continue

                    cand_id = self.get_clip_id(seg)
                    if global_used_clips and cand_id in global_used_clips:
                        continue

                    cand_emb = candidate_embeddings.get(cand_id)
                    if not _has_emb(cand_emb):
                        continue

                    if any(self.get_clip_id(sm.video_segment) == cand_id for sm in secondary_matches):
                        continue

                    # Calculate scores (same as above)
                    if existing_embeddings:
                        distances = []
                        for existing_emb in existing_embeddings:
                            if _has_emb(existing_emb):
                                sim = cosine_similarity(cand_emb, existing_emb)
                                distances.append(1.0 - sim)
                        avg_diversity = sum(distances) / len(distances) if distances else 0
                    else:
                        avg_diversity = 0.5

                    vo_relevance = cosine_similarity(cand_emb, vo_embedding) if _has_emb(vo_embedding) else text_sim
                    combined_score = vo_relevance * 0.4 + avg_diversity * 0.6

                    # Relaxed threshold: 0.2 instead of 0.3
                    if vo_relevance < 0.2:
                        continue

                    if combined_score > best_score:
                        best_score = combined_score
                        best_candidate = seg
                        best_diversity = avg_diversity
                        best_emb = cand_emb

            if best_candidate:
                scene = self._get_scene_for_segment(best_candidate)

                # Calculate source diversity score
                cand_kw = getattr(best_candidate, 'keywords', None) or []
                cand_keywords = set(k.lower() for k in cand_kw)
                source_diversity = calculate_source_diversity_score(
                    candidate_source=best_candidate.source_file,
                    candidate_keywords=cand_keywords,
                    v1_v3_sources=v1_v3_sources,
                    v1_v3_keywords=v1_v3_keywords
                )

                # Log diversity score for this segment
                logger.debug(
                    f"V{4 + track_idx} source diversity: {source_diversity:.2f} "
                    f"(source: {Path(best_candidate.source_file).stem}, "
                    f"unique_file: {best_candidate.source_file not in v1_v3_sources})"
                )

                secondary_matches.append(AlternativeMatch(
                    video_segment=best_candidate,
                    video_scene=scene,
                    confidence=best_score,
                    reasoning=f"{labels[track_idx]} (diversity={best_diversity:.2f}, source: {Path(best_candidate.source_file).stem})",
                    diversity_score=source_diversity
                ))

                # Add to exclusion for next track
                used_sources.add(best_candidate.source_file)
                if _has_emb(best_emb):
                    existing_embeddings.append(best_emb)

        return secondary_matches

    def _get_scene_for_segment(self, segment: SRTSegment) -> Optional[SceneInfo]:
        """
        Find the scene containing a video segment.

        Looks up the scene from scene detection that contains the segment's
        start time. Used to retrieve scene descriptions, visual keywords,
        and B-roll flags for matching.

        Args:
            segment: Video segment to find the scene for.

        Returns:
            SceneInfo object if found, None if no scene contains this segment.
            Scenes are matched by: scene.start_time <= segment.start_time < scene.end_time
        """
        video_scenes = self.scenes.get(segment.source_file, [])

        for scene in video_scenes:
            if scene.start_time <= segment.start_time < scene.end_time:
                return scene

        return None

    def match_source_rotation(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: Optional[List[List[float]]] = None,
        candidate_embeddings: Optional[Dict[str, List[float]]] = None,
        segment_index: int = 0
    ) -> Optional[StrategyMatch]:
        """
        Strategy: Source Rotation matching.

        Cycles through all source videos systematically using round-robin
        assignment for maximum variety. Ensures every downloaded video gets
        used across the timeline.

        Assignment logic:
        - For segment N with S source videos: assigned_source = sources[N % S]
        - Sources are sorted alphabetically for consistent ordering.
        - Falls back to next source in rotation if assigned source fails.

        Scoring:
        - Returns highest embedding similarity clip from assigned source.
        - Full confidence if from assigned source.
        - Falls back through rotation order if assigned source unavailable.

        Args:
            vo_segment: Voiceover segment to match.
            all_candidates: List of (video_segment, text_similarity) tuples.
            existing_matches: Already-selected clips (used for variety
                exclusion, but source requirement may override).
            existing_embeddings: Embeddings of existing matches.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.
            segment_index: Index of the voiceover segment in the timeline.
                Used for round-robin source assignment (segment_index % num_sources).

        Returns:
            StrategyMatch with strategy="source_rotation" and reasoning
            showing assigned source name and index, or None if no sources.

        Notes:
            - Creates predictable variety across long timelines.
            - Useful when you have many source videos covering similar content.
            - Fallback tries each source in rotation order until one works.
        """
        # Get all unique source videos from candidates
        source_videos = list(set(seg.source_file for seg, _ in all_candidates))

        if not source_videos:
            return None

        # Sort for consistent ordering
        source_videos.sort()

        # Determine which source to use for this segment (round-robin)
        assigned_source_idx = segment_index % len(source_videos)
        assigned_source = source_videos[assigned_source_idx]

        # Find best clip from assigned source
        best_candidate = None
        best_score = -1

        for seg, sim in all_candidates:
            # Must be from assigned source
            if seg.source_file != assigned_source:
                continue

            # Check variety exclusion (except source requirement since we're forcing it)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, reason = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb)
            if is_excluded and "source" not in reason.lower():
                continue

            if sim > best_score:
                best_score = sim
                best_candidate = seg

        # If no clip from assigned source passes filters, try next sources in rotation
        if best_candidate is None:
            for offset in range(1, len(source_videos)):
                fallback_idx = (assigned_source_idx + offset) % len(source_videos)
                fallback_source = source_videos[fallback_idx]

                for seg, sim in all_candidates:
                    if seg.source_file != fallback_source:
                        continue

                    cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
                    is_excluded, _ = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb)
                    if is_excluded:
                        continue

                    if sim > best_score:
                        best_score = sim
                        best_candidate = seg
                        assigned_source = fallback_source
                        break

                if best_candidate:
                    break

        if best_candidate:
            source_name = Path(assigned_source).stem
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=best_score,
                reasoning=f"Source rotation: {source_name} (idx {segment_index % len(source_videos)})",
                strategy="source_rotation"
            )

        return None

    def get_strategy_matches(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        primary_match: SRTSegment,
        alternatives: List[SRTSegment],
        vo_embedding: List[float],
        candidate_embeddings: Dict[str, List[float]],
        segment_index: int = 0,
        global_used_clips: Optional[Set[str]] = None
    ) -> List[StrategyMatch]:
        """
        Get all strategy matches for a voiceover segment.

        Orchestrates the configured strategy tracks (V7+), applying each
        strategy in sequence while maintaining variety across all tracks.
        Each strategy adds its match to the exclusion list before the next
        strategy runs.

        Strategy execution order (from config.output.strategy_tracks):
        - visual_first: Scene descriptions over transcript
        - different_source: Force unique video file
        - keyword_only: Keyword/entity overlap only
        - embedding_diversity: Maximize semantic distance from V1-V3
        - source_rotation: Round-robin through sources
        - broll_only: Silent footage only

        Args:
            vo_segment: Voiceover segment being matched.
            all_candidates: List of (video_segment, text_similarity) tuples.
            primary_match: V1 match segment.
            alternatives: V2-V3 match segments.
            vo_embedding: Embedding vector of the voiceover segment.
            candidate_embeddings: Dict mapping clip_id to embedding vectors.
            segment_index: Index for source_rotation strategy.
            global_used_clips: Set of clip IDs already used in previous
                segments. These are pre-filtered from candidates.

        Returns:
            List of StrategyMatch objects, one per configured strategy that
            found a match. List may be shorter than strategy_tracks if some
            strategies found no candidates.

        Notes:
            - Requires config.output.include_strategy_tracks=True.
            - Each strategy match is added to exclusion before next strategy.
            - Empty embeddings are filtered safely (handles numpy arrays).
        """
        # Pre-filter candidates by global used clips
        if global_used_clips:
            all_candidates = [
                (seg, dist) for seg, dist in all_candidates
                if self.get_clip_id(seg) not in global_used_clips
            ]
        strategies = self.config.output.strategy_tracks

        if not self.config.output.include_strategy_tracks:
            return []

        # Collect existing matches (V1 + V2-V3)
        existing_matches = [primary_match] + alternatives

        # Collect embeddings of existing matches
        existing_embeddings = []
        for seg in existing_matches:
            seg_id = self.get_clip_id(seg)
            if seg_id in candidate_embeddings:
                existing_embeddings.append(candidate_embeddings[seg_id])

        strategy_matches = []

        for strategy in strategies:
            # Add previous strategy matches to exclusion list
            all_existing = existing_matches + [sm.video_segment for sm in strategy_matches]
            all_existing_embs = existing_embeddings + [
                candidate_embeddings.get(self.get_clip_id(sm.video_segment), [])
                for sm in strategy_matches
            ]
            # Filter empty - handle both lists and numpy arrays
            def _has_content(e):
                if e is None:
                    return False
                if hasattr(e, '__len__'):
                    return len(e) > 0
                return bool(e)
            all_existing_embs = [e for e in all_existing_embs if _has_content(e)]

            match = None

            if strategy == "visual_first":
                match = self.match_visual_first(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings
                )
            elif strategy == "different_source":
                match = self.match_different_source(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings
                )
            elif strategy == "keyword_only":
                match = self.match_keyword_only(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings
                )
            elif strategy == "embedding_diversity":
                match = self.match_embedding_diversity(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings, vo_embedding
                )
            elif strategy == "source_rotation":
                match = self.match_source_rotation(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings, segment_index
                )
            elif strategy == "broll_only":
                match = self.match_broll_only(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings, vo_embedding
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
        Level 1 Fallback: Match purely based on keyword overlap.

        This is the highest quality fallback, using keyword extraction
        to find semantically relevant videos even when embedding similarity
        is low.

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (video_segment, similarity) tuples

        Returns:
            Tuple of (best_segment, confidence, reasoning) or None
        """
        # Extract voiceover keywords
        vo_kw = getattr(vo_segment, 'keywords', None) or []
        vo_keywords = set(k.lower() for k in vo_kw if k)

        # Also extract significant words from text if no keywords
        if not vo_keywords and vo_segment.text:
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'what',
                          'when', 'where', 'which', 'their', 'there', 'would', 'could',
                          'should', 'about', 'after', 'before', 'being', 'other'}
            vo_keywords = set(
                w.lower() for w in vo_segment.text.split()
                if len(w) >= 4 and w.lower() not in stop_words
            )

        if not vo_keywords:
            return None

        best_segment = None
        best_score = 0.0
        best_keywords = []

        for seg, emb_sim in candidates:
            # Get video keywords
            seg_kw = getattr(seg, 'keywords', None) or []
            seg_keywords = set(k.lower() for k in seg_kw if k)

            # Also check video text
            if seg.text:
                seg_text_words = set(
                    w.lower() for w in seg.text.split()
                    if len(w) >= 4
                )
                seg_keywords |= seg_text_words

            if not seg_keywords:
                continue

            # Calculate keyword overlap
            overlap = vo_keywords & seg_keywords
            if not overlap:
                continue

            # Score based on overlap ratio
            overlap_ratio = len(overlap) / len(vo_keywords)
            score = overlap_ratio * self.KEYWORD_ONLY_CEILING

            if score > best_score:
                best_score = score
                best_segment = seg
                best_keywords = list(overlap)[:5]

        if best_segment and best_score > 0:
            confidence = min(best_score, self.KEYWORD_ONLY_CEILING)
            reasoning = f"Fallback L1: keyword match ({', '.join(best_keywords)})"
            return (best_segment, confidence, reasoning)

        return None

    def match_visual_description(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]] = None
    ) -> Optional[Tuple[SRTSegment, float, str]]:
        """
        Level 2 Fallback: Match based on visual descriptions from scenes.

        Uses scene descriptions and visual keywords from scene detection
        to find visually relevant videos.

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (video_segment, similarity) tuples
            scenes: Dict mapping video paths to scene info lists

        Returns:
            Tuple of (best_segment, confidence, reasoning) or None
        """
        # Extract visual hints from voiceover text
        visual_terms = {
            'earthquake', 'tsunami', 'flood', 'storm', 'fire', 'volcano', 'disaster',
            'building', 'city', 'water', 'wave', 'destruction', 'damage', 'rescue',
            'people', 'crowd', 'evacuation', 'explosion', 'collapse', 'rubble',
            'mountain', 'ocean', 'forest', 'river', 'sky', 'sunset', 'sunrise',
            'car', 'plane', 'boat', 'train', 'road', 'street', 'bridge'
        }

        vo_text = vo_segment.text.lower() if vo_segment.text else ""
        vo_visual_hints = set(w for w in vo_text.split() if w in visual_terms)

        if not vo_visual_hints and not scenes:
            return None

        best_segment = None
        best_score = 0.0
        best_reason = ""

        for seg, emb_sim in candidates:
            score = 0.0

            # Check scene descriptions
            if scenes:
                video_scenes = scenes.get(seg.source_file, [])
                for scene in video_scenes:
                    if scene.start_time <= seg.start_time < scene.end_time:
                        # Found matching scene
                        if scene.description:
                            desc_lower = scene.description.lower()
                            matches = sum(1 for hint in vo_visual_hints if hint in desc_lower)
                            score += matches * 0.1

                        if scene.visual_keywords:
                            vis_kw = set(k.lower() for k in scene.visual_keywords)
                            keyword_overlap = len(vo_visual_hints & vis_kw)
                            score += keyword_overlap * 0.15
                        break

            # Check filename for visual hints
            if seg.source_file:
                filename = Path(seg.source_file).stem.lower()
                filename_matches = sum(1 for hint in vo_visual_hints if hint in filename)
                score += filename_matches * 0.1

            # Check video text for visual terms
            if seg.text:
                seg_text = seg.text.lower()
                text_matches = sum(1 for hint in vo_visual_hints if hint in seg_text)
                score += text_matches * 0.05

            if score > best_score:
                best_score = score
                best_segment = seg
                best_reason = f"visual hints: {', '.join(list(vo_visual_hints)[:3])}"

        if best_segment and best_score > 0:
            confidence = min(best_score, self.VISUAL_DESCRIPTION_CEILING)
            reasoning = f"Fallback L2: {best_reason}"
            return (best_segment, confidence, reasoning)

        return None

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
            candidates: List of (video_segment, similarity) tuples

        Returns:
            Tuple of (best_segment, confidence, reasoning) or None
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
            return (best_seg, confidence, "Fallback L3: short/silent segment")

        # Last resort: return top embedding match with very low confidence
        best_seg, best_sim = candidates[0]
        confidence = min(best_sim * 0.5, self.GENERIC_BROLL_CEILING)
        return (best_seg, confidence, "Fallback L3: best available (low confidence)")

    def apply_fallback(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]],
        primary_confidence: float,
        scenes: Optional[Dict[str, List[SceneInfo]]] = None
    ) -> Optional[Tuple[SRTSegment, float, str, int]]:
        """
        Apply fallback matching strategy.

        Tries fallback levels in order until one succeeds:
        1. Keyword-only matching (ceiling: 0.7)
        2. Visual-description matching (ceiling: 0.5)
        3. Generic B-roll matching (ceiling: 0.3)

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (video_segment, similarity) tuples
            primary_confidence: Confidence from primary matching
            scenes: Optional scene info for visual matching

        Returns:
            Tuple of (segment, confidence, reasoning, fallback_level) or None
            fallback_level: 1=keyword, 2=visual, 3=generic
        """
        if not self.should_trigger(primary_confidence):
            return None

        if not candidates:
            return None

        logger.info(
            f"Fallback triggered: primary_confidence={primary_confidence:.3f} < "
            f"threshold={self.trigger_threshold}"
        )

        # Level 1: Keyword-only
        result = self.match_keyword_only(vo_segment, candidates)
        if result:
            seg, conf, reason = result
            logger.info(f"Fallback L1 success: {reason}, confidence={conf:.3f}")
            return (seg, conf, reason, 1)

        # Level 2: Visual-description
        result = self.match_visual_description(vo_segment, candidates, scenes)
        if result:
            seg, conf, reason = result
            logger.info(f"Fallback L2 success: {reason}, confidence={conf:.3f}")
            return (seg, conf, reason, 2)

        # Level 3: Generic B-roll
        result = self.match_generic_broll(vo_segment, candidates)
        if result:
            seg, conf, reason = result
            logger.info(f"Fallback L3 success: {reason}, confidence={conf:.3f}")
            return (seg, conf, reason, 3)

        logger.warning("All fallback levels failed")
        return None
