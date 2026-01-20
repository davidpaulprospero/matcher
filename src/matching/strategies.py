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
from .tracking import extract_video_id

logger = logging.getLogger(__name__)


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

    def __init__(self, config: Config, scenes: Optional[Dict[str, List[SceneInfo]]]):
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
        """Generate unique clip ID"""
        return f"{segment.source_file}:{segment.start_time:.2f}-{segment.end_time:.2f}"

    def is_clip_excluded(
        self,
        candidate: SRTSegment,
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]] = None,
        candidate_embedding: List[float] = None,
        force_different_source: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if candidate violates variety enforcement rules.
        Returns (is_excluded, reason)

        Args:
            force_different_source: If True, always require different source file
                                   (used for alternative tracks)
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
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy A: Visual-First
        Prioritize scene descriptions and visual keywords over transcript text.
        Falls back to filename/path matching if no scene data.
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
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy B: Different Source Video
        Force selection from a different video file than existing matches.
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
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy C: Keyword-Only
        Match purely based on keyword and entity overlap, ignore embeddings.
        Falls back to text word overlap if keywords aren't populated.
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
        Strategy D: Embedding Diversity
        Find clips that are semantically relevant but maximally DIFFERENT from V1-V3.
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
        Strategy: B-roll Only
        Find clips that are EXCLUSIVELY B-roll (no speech/faces).
        Provides editors with a guaranteed silent footage option.

        B-roll is detected during scene analysis via face_score < threshold.
        This track only considers candidates with is_broll=True.
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
        global_used_clips: Set[str] = None,
        global_used_video_ids: Set[str] = None
    ) -> List[AlternativeMatch]:
        """
        Get secondary matches (V4-V6) using diversity scoring.

        STRICT enforcement: Each track MUST use a different source video.
        Uses same diversity algorithm as V7 (embedding_diversity).

        Args:
            vo_segment: The voiceover segment
            all_candidates: All candidate clips with similarity scores
            primary_match: V1 match segment
            alternatives: V2-V3 match segments
            vo_embedding: Voiceover embedding
            candidate_embeddings: Dict of clip_id -> embedding
            global_used_clips: Set of clip IDs already used globally (cross-segment dedup)
            global_used_video_ids: Set of video IDs already used globally (video-level dedup)

        Returns:
            List of up to 3 AlternativeMatch objects for V4, V5, V6
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))

        if not candidate_embeddings:
            return []

        # Collect V1-V3 source files and embeddings
        v1_v3_sources = {primary_match.source_file}
        v1_v3_embeddings = []

        # Get V1 embedding
        v1_id = self.get_clip_id(primary_match)
        if v1_id in candidate_embeddings:
            v1_v3_embeddings.append(candidate_embeddings[v1_id])

        # Get V2-V3 embeddings
        for alt_seg in alternatives:
            if alt_seg:
                v1_v3_sources.add(alt_seg.source_file)
                alt_id = self.get_clip_id(alt_seg)
                if alt_id in candidate_embeddings:
                    v1_v3_embeddings.append(candidate_embeddings[alt_id])

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

                # Video-level deduplication: skip videos already used
                if global_used_video_ids:
                    video_id = extract_video_id(seg.source_file)
                    if video_id in global_used_video_ids:
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

                    # Video-level deduplication: skip videos already used
                    if global_used_video_ids:
                        video_id = extract_video_id(seg.source_file)
                        if video_id in global_used_video_ids:
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
                secondary_matches.append(AlternativeMatch(
                    video_segment=best_candidate,
                    video_scene=scene,
                    confidence=best_score,
                    reasoning=f"{labels[track_idx]} (diversity={best_diversity:.2f}, source: {Path(best_candidate.source_file).stem})"
                ))

                # Add to exclusion for next track
                used_sources.add(best_candidate.source_file)
                if _has_emb(best_emb):
                    existing_embeddings.append(best_emb)

        return secondary_matches

    def _get_scene_for_segment(self, segment: SRTSegment) -> Optional[SceneInfo]:
        """Find the scene containing this segment"""
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
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None,
        segment_index: int = 0
    ) -> Optional[StrategyMatch]:
        """
        Strategy: Source Rotation
        Cycles through all source videos systematically for maximum variety.

        For segment N, picks the best matching clip from source video (N % num_sources).
        This ensures every source gets used and creates visual variety.
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
        global_used_clips: Set[str] = None,
        global_used_video_ids: Set[str] = None
    ) -> List[StrategyMatch]:
        """
        Get all strategy matches for a voiceover segment.
        Ensures variety across all tracks.

        Args:
            global_used_clips: Set of clip IDs already used globally (cross-segment dedup)
            global_used_video_ids: Set of video IDs already used globally (video-level dedup)
        """
        # Pre-filter candidates by global used clips
        if global_used_clips:
            all_candidates = [
                (seg, dist) for seg, dist in all_candidates
                if self.get_clip_id(seg) not in global_used_clips
            ]
        # Pre-filter by video ID for video-level deduplication
        if global_used_video_ids:
            all_candidates = [
                (seg, dist) for seg, dist in all_candidates
                if extract_video_id(seg.source_file) not in global_used_video_ids
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
