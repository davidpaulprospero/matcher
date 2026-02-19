"""
Tiered Matcher - Two-stage matching with LLM reranking.

Refactored architecture (Feb 2026):
- Uses composition with extracted modules for single responsibility
- EmbeddingSearch (in main.py): Embedding similarity search
- LLMReranker: LLM-based candidate reranking
- AlternativeSelector: V2-V6 track selection
- LocationMatcher: Geographic filtering
- CandidateFilter: Face/location/reuse filtering (US-33-006)
- scoring.py: Confidence adjustments and penalties

Core functionality:
- Two-stage matching: embedding similarity → LLM reranking
- Smart reuse prevention and confidence adjustment
- Location-aware filtering (delegates to LocationMatcher)
- Candidate filtering (delegates to CandidateFilter - face, location, reuse)
- Topic-based penalty for chapter matching (uses scoring.py)
- B-roll boost and project boost (uses scoring.py)
- Alternative and secondary match generation (delegates to AlternativeSelector)
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from .voiceover_topics import (
    calculate_topic_similarity,
    calculate_segment_coherence,
    get_topic_enriched_text,
)

# Refactored modules - REUSE instead of duplicating (406 lines saved)
from .scoring import (
    MatchScoring,  # US-33-005: Composition class for scoring
    apply_topic_penalty,
    apply_broll_boost,
    apply_caption_quality_adjustment,  # US-007
    apply_timing_penalty,  # US-008 Sprint 7
    apply_current_project_boost,
    apply_consecutive_source_penalty,  # US-63-009
    apply_title_relevance_adjustment,  # US-75-002
    apply_description_relevance_adjustment,  # US-75-003
    apply_tag_keyword_boost,  # US-75-004
    compute_tag_relevance_score,  # US-141-007
    apply_chapter_topic_match,  # US-75-005
    apply_chapter_source_consistency,  # US-75-005
    apply_source_channel_consistency,  # US-95-006
    apply_topic_alignment_boost,  # US-95-007
    apply_chapter_coherence_penalty,  # US-75-006
    apply_cross_chapter_relevance_boost,  # US-75-006
    apply_chapter_boundary_penalty,  # US-95-004
    apply_listicle_consistency,  # US-75-007
    apply_entity_match_boost,  # US-77-011
    apply_context_richness_calibration,  # US-95-010
    compute_chapter_alignment_boost,  # US-95-011
    compute_semantic_coherence,  # US-77-002
    compute_temporal_coherence,  # US-77-003
    apply_source_stutter_penalty,  # US-84-004
    compute_thematic_consistency,  # US-134-010
    check_consecutive_source_hard_cap,  # US-63-009
    calculate_adaptive_threshold,
    _extract_entity_texts,
    compute_multimodal_score,
    calculate_keyword_overlap_score,
    calculate_entity_match_score,
    calculate_visual_description_score,
    DEFAULT_MULTIMODAL_WEIGHTS,
    validate_multimodal_weights,
)
from .location_matching import LocationMatcher
from .llm_providers import GeminiMatcher, ClaudeMatcher, LocalLLMMatcher, validate_explanation_confidence
from .llm_reranker import LLMReranker, LLMRerankerConfig, validate_context_consistency
from .alternative_selection import AlternativeSelector, AlternativeSelectionConfig
from .candidate_filter import CandidateFilter  # US-33-006: Extracted filtering
from .similarity_cache import (
    get_keyword_cache, text_hash as compute_text_hash, log_all_cache_stats
)

# Utils and data structures
from ..utils import (
    SRTSegment, SceneInfo, Match, AlternativeMatch, MatchResult,
    CacheManager, ReuseTracker
)
from ..topic_extraction import VideoTopics, LocationChapter
from ..location_service import GeoLocation, create_location_service
from ..keyword_extractor import find_keyword_matches
from ..face_detection import apply_face_preference
from ..logger import get_global_logger
from ..config import get_config

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)

# Confidence assigned to gap matches where no suitable video was found.
# Downstream stages use this to identify unmatched segments for gap-filling.
DEFAULT_GAP_CONFIDENCE = 0.0


def create_gap_match(vo_segment: 'SRTSegment', reason: str) -> Match:
    """Factory function to create a gap match with consistent attributes.

    Gap matches represent segments where no suitable video match was found.
    The video_segment is set to the voiceover_segment itself (self-reference pattern),
    confidence is 0.0, and match_type is set to 'gap' for downstream identification.

    Args:
        vo_segment: The voiceover segment that could not be matched.
        reason: Human-readable explanation of why no match was found.

    Returns:
        A Match instance marked as a gap match.
    """
    return Match(
        voiceover_segment=vo_segment,
        video_segment=vo_segment,
        video_scene=None,
        confidence=DEFAULT_GAP_CONFIDENCE,
        reasoning=reason,
        match_type='gap',
    )


def _record_breakdown(breakdown: list, component: str, before: float, after: float, reason: str):
    """Record a scoring adjustment in the confidence breakdown list with structured logging."""
    adjustment = after - before
    if reason:
        breakdown.append({
            'component': component,
            'adjustment': round(adjustment, 4),
            'reason': reason,
        })
        # Log with appropriate level: debug for minor (<=0.05), info for major (>0.05)
        if abs(adjustment) > 0.05:
            logger.info(f"Confidence adjustment: {component}: {before:.2f} -> {after:.2f} ({adjustment:+.4f}) - {reason}")
        else:
            logger.debug(f"Confidence adjustment: {component}: {before:.2f} -> {after:.2f} ({adjustment:+.4f}) - {reason}")


def compute_scoring_audit_summary(results: List[Any], context_cache_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Aggregate confidence_breakdown data from all match results into an audit summary.

    Collects all confidence_breakdown dicts across all segments, computes:
    - Total segments matched and average confidence
    - Count of each scoring adjustment that fired
    - Average magnitude per adjustment
    - Top-3 most impactful adjustments by average absolute magnitude
    - Adjustments that never fired (reported as 'unused')
    - Context cache statistics (US-134-011)

    Args:
        results: List of MatchResult objects from the matching loop.
        context_cache_stats: Optional context cache statistics from LLMReranker.

    Returns:
        Dict with keys: total_segments, avg_confidence, adjustment_counts,
        avg_magnitudes, top_3_impactful, unused_adjustments, context_cache_stats.
    """
    # All known adjustment components (must match _record_breakdown calls)
    ALL_KNOWN_ADJUSTMENTS = {
        'topic_penalty', 'broll_boost', 'caption_quality', 'timing_penalty',
        'project_boost', 'consecutive_source_penalty', 'title_relevance',
        'description_relevance', 'tag_keyword_boost', 'chapter_topic_match',
        'chapter_source_consistency', 'chapter_coherence_penalty',
        'cross_chapter_relevance', 'chapter_boundary_penalty', 'listicle_consistency',
        'semantic_coherence', 'temporal_coherence', 'source_stutter_penalty',
        'explanation_validation', 'entity_match_boost', 'diversity_recheck',
        'chapter_alignment_boost', 'topic_alignment_boost',  # US-95-011, US-95-007
        'channel_reputation', 'source_channel_coherence',  # US-111-005
        'llm_reasoning_penalty', 'secondary_llm_decay',  # US-84-001
        'obvious_match_min_confidence',  # US-127-009
    }

    confidences: List[float] = []
    # adjustment_name -> list of adjustment values
    adjustment_values: Dict[str, List[float]] = {}

    for result in results:
        if not result or not hasattr(result, 'primary_match') or not result.primary_match:
            continue
        confidences.append(result.primary_match.confidence)

        breakdown = getattr(result, 'confidence_breakdown', None) or []
        for entry in breakdown:
            comp = entry.get('component', '')
            adj = entry.get('adjustment', 0.0)
            if comp:
                if comp not in adjustment_values:
                    adjustment_values[comp] = []
                adjustment_values[comp].append(adj)

    total_segments = len(confidences)
    avg_confidence = sum(confidences) / total_segments if total_segments else 0.0

    # Count how many times each adjustment fired
    adjustment_counts = {name: len(vals) for name, vals in adjustment_values.items()}

    # Average magnitude (absolute value) per adjustment
    avg_magnitudes = {}
    for name, vals in adjustment_values.items():
        avg_magnitudes[name] = sum(abs(v) for v in vals) / len(vals) if vals else 0.0

    # Top-3 most impactful by average absolute magnitude
    sorted_by_impact = sorted(avg_magnitudes.items(), key=lambda x: -x[1])
    top_3_impactful = sorted_by_impact[:3]

    # Unused adjustments: known adjustments that never appeared
    fired_adjustments = set(adjustment_values.keys())
    unused_adjustments = sorted(ALL_KNOWN_ADJUSTMENTS - fired_adjustments)

    # US-134-011: Include context cache stats if provided
    return {
        'total_segments': total_segments,
        'avg_confidence': avg_confidence,
        'adjustment_counts': adjustment_counts,
        'avg_magnitudes': avg_magnitudes,
        'top_3_impactful': top_3_impactful,
        'unused_adjustments': unused_adjustments,
        'context_cache_stats': context_cache_stats,
    }


def log_scoring_audit_summary(summary: Dict[str, Any]) -> None:
    """Log the scoring audit summary at INFO level.

    Args:
        summary: Dict returned by compute_scoring_audit_summary.
    """
    total = summary['total_segments']
    avg_conf = summary['avg_confidence']
    counts = summary['adjustment_counts']
    magnitudes = summary['avg_magnitudes']
    top3 = summary['top_3_impactful']
    unused = summary['unused_adjustments']
    context_cache_stats = summary.get('context_cache_stats')

    logger.info("=" * 60)
    logger.info("SCORING ADJUSTMENT AUDIT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total segments matched: {total}")
    logger.info(f"  Average confidence: {avg_conf:.3f}")

    if counts:
        logger.info(f"  Active adjustments ({len(counts)}):")
        for name in sorted(counts.keys()):
            logger.info(
                f"    {name}: fired {counts[name]}x, avg magnitude {magnitudes.get(name, 0):.4f}"
            )

    if top3:
        logger.info(f"  Top-3 most impactful (by avg magnitude):")
        for rank, (name, mag) in enumerate(top3, 1):
            logger.info(f"    #{rank}: {name} (avg |adjustment| = {mag:.4f})")

    if unused:
        logger.info(f"  Unused adjustments ({len(unused)}): {', '.join(unused)}")
    else:
        logger.info(f"  All known adjustments fired at least once")

    # US-134-011: Log context cache statistics
    if context_cache_stats:
        logger.info("  Context Cache Statistics:")
        logger.info(f"    Size: {context_cache_stats.get('size', 0)}/{context_cache_stats.get('max_size', 0)}")
        logger.info(f"    Hit rate: {context_cache_stats.get('hit_rate', 0):.1%} ({context_cache_stats.get('hits', 0)} hits, {context_cache_stats.get('misses', 0)} misses)")
        logger.info(f"    TTL: {context_cache_stats.get('ttl_seconds', 0)}s")
        logger.info(f"    Evictions: {context_cache_stats.get('evictions', 0)}, Expirations: {context_cache_stats.get('expirations', 0)}")

    logger.info("=" * 60)


def report_low_confidence_segments(
    results: List[Any],
    threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    """Report segments with confidence below LOW_CONFIDENCE_WARNING_THRESHOLD.

    Collects segments below the threshold, logs each at WARNING with:
    - Segment index, voiceover text preview (first 50 chars), confidence
    - Top 3 negative adjustments from confidence_breakdown

    When >20% of segments are low-confidence, logs a summary WARNING
    suggesting enabling iterative matching or adjusting scoring weights.

    Args:
        results: List of MatchResult objects from the matching loop.
        threshold: Confidence threshold (default 0.15 = LOW_CONFIDENCE_WARNING_THRESHOLD).

    Returns:
        List of dicts describing each low-confidence segment (for testing).
    """
    if not results:
        return []

    low_segments: List[Dict[str, Any]] = []

    for i, result in enumerate(results):
        if not result or not hasattr(result, 'primary_match') or not result.primary_match:
            continue
        match = result.primary_match
        conf = match.confidence
        if conf >= threshold:
            continue

        # Voiceover text preview
        vo_text = ''
        vo_seg = getattr(match, 'voiceover_segment', None)
        if vo_seg:
            vo_text = getattr(vo_seg, 'text', '') or ''
        text_preview = vo_text[:50]

        # Top 3 negative adjustments from confidence_breakdown
        breakdown = getattr(result, 'confidence_breakdown', None) or []
        negative_adjustments = sorted(
            [e for e in breakdown if e.get('adjustment', 0) < 0],
            key=lambda e: e.get('adjustment', 0),
        )[:3]

        neg_summary = '; '.join(
            f"{e.get('component', '?')}={e.get('adjustment', 0):+.4f}"
            for e in negative_adjustments
        ) if negative_adjustments else 'none'

        seg_info = {
            'segment_index': i,
            'text_preview': text_preview,
            'confidence': conf,
            'top_negative_adjustments': negative_adjustments,
        }
        low_segments.append(seg_info)

        logger.warning(
            f"Low-confidence segment #{i}: conf={conf:.3f}, "
            f"text=\"{text_preview}\", "
            f"top negative adjustments: [{neg_summary}]"
        )

    # Summary warning when >20% are low-confidence
    total = len([r for r in results if r and hasattr(r, 'primary_match') and r.primary_match])
    if total > 0 and len(low_segments) / total > 0.20:
        pct = len(low_segments) / total * 100
        logger.warning(
            f"Low-confidence summary: {len(low_segments)}/{total} segments ({pct:.0f}%) "
            f"below {threshold}. Consider enabling iterative matching (--high-matches) "
            f"or adjusting scoring weights."
        )

    return low_segments


class TieredMatcher:
    """
    Two-stage matcher with embedding search + LLM reranking.

    Uses composition with refactored modules:
    - scoring.py for confidence adjustments
    - location_matching.py for geographic filtering
    - llm_providers.py for LLM interactions
    """

    def __init__(self, config: Optional['Config'] = None, cache: Optional[CacheManager] = None, video_topics: Optional[Dict[str, VideoTopics]] = None,
                 video_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                 relevance_matrix: Optional[List[List[float]]] = None,
                 listicle_groups: Optional[List] = None):
        """
        Initialize TieredMatcher.

        Args:
            config: Configuration object (uses get_config() if None)
            cache: Cache manager for LLM responses
            video_topics: Dict mapping video paths to VideoTopics for chapter matching
            video_metadata: Optional dict mapping source_file (video ID) to
                {"title": str, "description": str, "tags": List[str], "chapters": List[dict]}
            relevance_matrix: 2D list [vo_chapter][vid_chapter] of relevance scores
                for cross-chapter relevance boost (US-75-006)
            listicle_groups: Optional list of ListicleGroup objects for
                listicle consistency boost (US-75-007)
        """
        self.config = config or get_config()
        self.cache = cache
        self.video_topics = video_topics or {}
        self.video_metadata = video_metadata or {}
        self.relevance_matrix = relevance_matrix
        self.listicle_groups = listicle_groups or []

        # US-75-006: Track unique video sources per voiceover chapter for coherence penalty
        self._chapter_source_counts: Dict[int, set] = {}
        # US-75-010: Per-segment chapter index mapping (set by match_all_segments)
        self.segment_chapter_map: Dict[int, int] = {}
        # US-76-012: Chapter confidence map (chapter_index -> confidence score)
        self._chapter_confidence_map: Dict[int, float] = {}
        # US-77-002: Semantic coherence - store previous match embedding for topic flow
        self._previous_match_embedding: Optional[Any] = None
        # US-77-003: Temporal coherence - store previous match segment for source continuity
        self._previous_match_segment: Optional[SRTSegment] = None
        # US-84-004: Source stutter penalty - store 2-segments-back match for A-B-A detection
        self._prev_prev_match_segment: Optional[SRTSegment] = None
        # US-134-010: Thematic consistency - store matched segments within window
        self._matched_segments_list: List[SRTSegment] = []
        self._embedding_lookup: Dict[int, int] = {}  # id(segment) -> index in video_embeddings
        self._video_embeddings: Optional[List] = None  # Reference to video embeddings list
        mc = self.config.matching

        # Matching thresholds
        self.primary_model = mc.gemini_model
        self.min_confidence = mc.min_confidence
        self.embedding_candidates = mc.embedding_candidates
        self.high_conf_threshold = mc.high_confidence_threshold
        self.low_conf_threshold = mc.low_confidence_threshold
        self.max_clip_reuse = mc.max_clip_reuse
        self.reuse_penalty = mc.reuse_penalty

        # Chapter-based matching
        self.chapter_matching_enabled = getattr(mc, 'chapter_matching_enabled', False)
        self.enforce_chapter_boundaries = getattr(mc, 'enforce_chapter_boundaries', False)  # US-95-004
        self.cross_chapter_penalty = getattr(mc, 'cross_chapter_penalty', 0.05)  # US-95-004
        self.topic_mismatch_penalty = getattr(mc, 'topic_mismatch_penalty', 0.15)
        self.topic_alignment_weight = getattr(mc, 'topic_alignment_weight', 0.1)  # US-95-007

        # Location-aware matching (compose LocationMatcher)
        self.location_matching_config = getattr(mc, 'location_matching', None)
        self.location_matching_enabled = False
        self.location_service = None
        self.location_matcher = None  # Will be LocationMatcher instance

        if self.location_matching_config:
            if isinstance(self.location_matching_config, dict):
                self.location_matching_enabled = self.location_matching_config.get('enabled', False)
            else:
                self.location_matching_enabled = getattr(self.location_matching_config, 'enabled', False)

            if self.location_matching_enabled:
                try:
                    self.location_service = create_location_service(self.config)
                    self.location_matcher = LocationMatcher(self.location_service)
                    logger.info("Location-aware matching enabled")
                except Exception as e:
                    logger.warning(f"Could not initialize location service: {e}")
                    self.location_matching_enabled = False

        # Initialize reuse tracker
        self.reuse_tracker = ReuseTracker(
            max_reuse=mc.max_clip_reuse,
            reuse_penalty=mc.reuse_penalty,
            max_source_file_reuse=getattr(mc, 'max_source_file_reuse', 0),
            source_file_penalty=getattr(mc, 'source_file_penalty', 0.05)
        )

        # Face preference (for face detection filtering)
        self.face_preference = getattr(mc, 'face_preference', 'neutral')

        # Initialize LLM providers
        self.primary_provider = None
        self.secondary_provider = None
        self.local_provider = None
        self._init_providers()

        # Initialize LLM reranker (delegates LLM selection logic)
        self.llm_reranker = LLMReranker(
            config=LLMRerankerConfig(
                ambiguous_threshold=getattr(mc, 'ambiguous_threshold', 0.65),
                cache_llm_responses=getattr(mc, 'cache_llm_responses', True)
            ),
            cache=cache
        )

        # Initialize alternative selector (delegates V2-V6 selection)
        self.alt_selector = AlternativeSelector.from_output_config(self.config.output)

        # Initialize scoring module (US-33-005: composition for scoring logic)
        self.scoring = MatchScoring(self.config)

        # Initialize candidate filter (US-33-006: composition for filtering logic)
        self.candidate_filter = CandidateFilter(
            config=self.config,
            location_matcher=self.location_matcher
        )

        # US-63-009: Track recent matches for consecutive source penalty
        # Stores the most recent N matches (most recent first) for diversity checking
        self._recent_matches: List[Match] = []
        self._max_recent_matches = getattr(mc, 'max_consecutive_same_source', 3) + 1

    def _init_providers(self):
        """Initialize LLM providers based on config"""
        mc = self.config.matching

        # Get model names
        gemini_model = getattr(mc, 'gemini_model', 'gemini-2.0-flash')
        anthropic_model = getattr(mc, 'anthropic_model', 'claude-3-haiku-20240307')
        ollama_model = getattr(mc, 'ollama_model', 'llama3.2')
        ollama_host = getattr(mc, 'ollama_host', 'http://localhost:11434')

        # Primary provider
        if mc.primary_provider == "gemini" and self.config.gemini_api_key:
            self.primary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Primary LLM: Gemini ({gemini_model})")
        elif mc.primary_provider == "anthropic" and self.config.anthropic_api_key:
            self.primary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Primary LLM: Claude ({anthropic_model})")
        elif self.config.gemini_api_key:
            self.primary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Primary LLM: Gemini ({gemini_model}) (auto)")
        elif self.config.anthropic_api_key:
            self.primary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Primary LLM: Claude ({anthropic_model}) (auto)")

        # Secondary provider
        if mc.secondary_provider == "anthropic" and self.config.anthropic_api_key:
            self.secondary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Secondary LLM: Claude ({anthropic_model})")
        elif mc.secondary_provider == "gemini" and self.config.gemini_api_key:
            self.secondary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Secondary LLM: Gemini ({gemini_model})")

        # Local provider
        if mc.use_local_for_review:
            try:
                self.local_provider = LocalLLMMatcher(ollama_model, ollama_host)
                import requests
                requests.get(f"{ollama_host}/api/tags", timeout=2)
                logger.info(f"Local LLM: Ollama ({ollama_model})")
            except:
                logger.info("Local LLM: Not available (Ollama not running)")
                self.local_provider = None

    # Location methods - delegate to LocationMatcher
    def set_location_chapters(self, location_chapters: List[LocationChapter]):
        """Set location chapters for location-aware matching"""
        if self.location_matcher:
            self.location_matcher.set_location_chapters(location_chapters)
            logger.info(f"Set {len(location_chapters)} location chapters")

    def set_video_locations(self, video_locations: Dict[str, GeoLocation]):
        """Set video location data for location-aware matching"""
        if self.location_matcher:
            self.location_matcher.set_video_locations(video_locations)
            logger.info(f"Set locations for {len(video_locations)} videos")

    # Utility methods
    def _calculate_confidence_variance(self, candidates: List[Tuple[SRTSegment, float]], top_n: int = 5) -> float:
        """
        Calculate standard deviation of top-N candidate similarity scores.

        High variance (> 0.1) indicates uncertain match - multiple candidates have
        similar scores, making the selection less definitive.
        Low variance indicates clear winner with others scoring much lower.

        US-53-011: Uses top-10 candidates when pool >= 10 (unless top_n explicitly
        overridden), applies NaN/Inf guard, and logs variance computation details.

        Args:
            candidates: List of (video_segment, similarity) tuples
            top_n: Number of top candidates to consider (default: 5,
                   auto-scales to 10 when pool >= 10)

        Returns:
            Standard deviation of top-N similarity scores (0.0 if < 2 candidates)
        """
        if len(candidates) < 2:
            return 0.0

        # Auto-scale to top-10 when pool is large enough and default top_n used
        effective_top_n = 10 if (top_n == 5 and len(candidates) >= 10) else top_n

        # Get top-N similarity scores (convert to Python float to avoid numpy coercion error)
        top_scores = [float(sim) for _, sim in candidates[:effective_top_n]]

        if len(top_scores) < 2:
            return 0.0

        try:
            result = statistics.stdev(top_scores)
        except statistics.StatisticsError:
            return 0.0

        # Guard against NaN/Inf from degenerate inputs
        if math.isnan(result) or math.isinf(result):
            logger.debug(f"Variance computation returned {result}, treating as 0.0")
            return 0.0

        logger.debug(
            f"Confidence variance: candidate_count={len(candidates)}, "
            f"top_n_used={len(top_scores)}, computed_variance={result:.4f}"
        )

        return result

    def _extract_matched_keywords(
        self,
        vo_segment: SRTSegment,
        video_segment: SRTSegment
    ) -> List[str]:
        """
        Extract common keywords between voiceover and video transcript.

        Uses caching to avoid re-extracting keywords from the same text
        across multiple candidate comparisons.

        Finds keywords that appear in both the voiceover segment and the selected
        video's transcript/keywords. Returns unique matched keywords sorted by
        frequency of occurrence.

        Args:
            vo_segment: Voiceover segment with text and optional keywords
            video_segment: Selected video segment with text and optional keywords

        Returns:
            List of matched keywords (lowercase, deduplicated)
        """
        # Get cached keyword extraction function
        vo_keywords = self._get_keywords_for_segment(vo_segment)
        video_keywords = self._get_keywords_for_segment(video_segment)

        # Find intersection
        matched = vo_keywords & video_keywords

        # Filter out very short keywords and return sorted list
        result = sorted([kw for kw in matched if len(kw) >= 3])

        return result

    def _get_keywords_for_segment(self, segment: SRTSegment) -> set:
        """
        Extract keywords from segment with caching.

        Caches extracted keywords by text hash to avoid recomputation
        when the same segment is compared against multiple candidates.

        Args:
            segment: SRTSegment with text and optional keywords

        Returns:
            Set of lowercase keywords
        """
        # Generate cache key from segment text
        text = segment.text or ""
        cache_key = compute_text_hash(text)

        # Check cache
        cache = get_keyword_cache()
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        # Extract keywords
        keywords = set()

        # Get keywords from segment's keywords attribute
        if hasattr(segment, 'keywords') and segment.keywords:
            keywords.update(kw.lower().strip() for kw in segment.keywords if kw)

        # Extract significant words from text (>= 4 chars, not common words)
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
                        'can', 'her', 'was', 'one', 'our', 'out', 'has', 'have',
                        'been', 'from', 'this', 'that', 'with', 'they', 'what',
                        'will', 'there', 'their', 'about', 'would', 'which', 'into'}
        if text:
            words = text.lower().split()
            keywords.update(
                w.strip('.,!?:;"\'()[]{}') for w in words
                if len(w) >= 4 and w.lower() not in common_words
            )

        # Cache and return
        cache.put(cache_key, keywords)
        return keywords

    def _compute_multimodal_confidence(
        self,
        vo_segment: SRTSegment,
        video_segment: SRTSegment,
        embedding_similarity: float,
        scene: Optional['SceneInfo'] = None
    ) -> Tuple[float, str, dict]:
        """
        Compute multimodal confidence score combining multiple similarity signals.

        Instead of simple additive boosting, this method uses weighted fusion:
        - text_embedding: Raw embedding similarity (40%)
        - keyword_overlap: Matched keywords (25%)
        - entity_match: Named entity overlap (20%)
        - visual_description: Scene description similarity (15%)

        Args:
            vo_segment: Voiceover segment
            video_segment: Video segment candidate
            embedding_similarity: Raw embedding similarity score
            scene: Optional scene info with visual keywords

        Returns:
            Tuple of (multimodal_confidence, reason_string, component_scores)
        """
        mc = self.config.matching

        # Check if multimodal scoring is enabled
        multimodal_enabled = getattr(mc, 'multimodal_enabled', True)
        raw_weights = getattr(mc, 'multimodal_weights', None)

        # Validate and normalize weights before use
        multimodal_weights = validate_multimodal_weights(raw_weights) if raw_weights else None

        # If disabled, return embedding similarity as-is
        if not multimodal_enabled:
            return embedding_similarity, "multimodal_disabled", {
                'embedding_similarity': embedding_similarity
            }

        # Get voiceover keywords and entities
        vo_keywords = getattr(vo_segment, 'keywords', []) or []
        vo_entities = _extract_entity_texts(vo_segment)

        # Get video keywords and entities
        video_keywords = getattr(video_segment, 'keywords', []) or []
        video_entities = _extract_entity_texts(video_segment)

        # Calculate keyword overlap score (normalized 0-1)
        keyword_score, matched_keywords = calculate_keyword_overlap_score(
            vo_keywords, video_keywords
        )

        # Calculate entity match score (normalized 0-1)
        entity_score, matched_entities = calculate_entity_match_score(
            vo_entities, video_entities
        )

        # Calculate visual description score
        video_description = video_segment.text if video_segment.text else ""
        visual_keywords = scene.visual_keywords if scene and hasattr(scene, 'visual_keywords') else []
        visual_score = calculate_visual_description_score(
            vo_segment.text,
            video_description,
            visual_keywords
        )

        # Compute multimodal score
        multimodal_conf, reason, components = compute_multimodal_score(
            embedding_similarity=embedding_similarity,
            keyword_overlap_score=keyword_score,
            entity_match_score=entity_score,
            visual_description_score=visual_score,
            weights=multimodal_weights,
            multimodal_enabled=True
        )

        # Add matched items to components for logging
        components['matched_keywords'] = matched_keywords
        components['matched_entities'] = matched_entities

        logger.debug(
            f"Multimodal confidence: emb={embedding_similarity:.3f} -> mm={multimodal_conf:.3f} "
            f"(kw={keyword_score:.2f}, ent={entity_score:.2f}, vis={visual_score:.2f})"
        )

        return multimodal_conf, reason, components

    def check_obvious_match(
        self,
        vo_segment: SRTSegment,
        video_segment: SRTSegment,
        similarity: float,
        matched_keywords: List[str]
    ) -> Optional[Tuple[float, str, List[str]]]:
        """
        Check if this is an obvious high-confidence match that can skip LLM.

        An obvious match requires ALL THREE conditions:
        1. Embedding similarity > config threshold (default 0.9)
        2. At least N keywords match (default 3)
        3. At least one named entity appears in both voiceover and video

        Args:
            vo_segment: Voiceover segment
            video_segment: Video segment candidate
            similarity: Embedding similarity score
            matched_keywords: List of keywords matching between voiceover and video

        Returns:
            None if not an obvious match, otherwise tuple of:
            (boosted_confidence, reasoning, matched_entities)
        """
        mc = self.config.matching

        # Check if obvious match is enabled
        obvious_match_enabled = getattr(mc, 'obvious_match_enabled', True)
        if not obvious_match_enabled:
            return None

        # Get thresholds from config
        min_similarity = getattr(mc, 'obvious_match_min_similarity', 0.9)
        min_keywords = getattr(mc, 'obvious_match_min_keywords', 3)
        min_confidence = getattr(mc, 'obvious_match_min_confidence', 0.92)

        # Condition 1: Embedding similarity must be very high
        if similarity < min_similarity:
            return None

        # Condition 2: Must have enough matched keywords
        if len(matched_keywords) < min_keywords:
            return None

        # Condition 3: Must have at least one matching named entity
        vo_entities = _extract_entity_texts(vo_segment)
        video_entities = _extract_entity_texts(video_segment)

        if not vo_entities or not video_entities:
            return None

        # Find matching entities (case-insensitive)
        vo_lower = {e.lower() for e in vo_entities}
        video_lower = {e.lower() for e in video_entities}
        matching_entities = vo_lower & video_lower

        if not matching_entities:
            return None

        # All conditions met - this is an obvious match
        # Get the original-case entity names for display
        matched_entity_names = [e for e in vo_entities if e.lower() in matching_entities]

        # Calculate boosted confidence with cap at 0.98 and boost of +0.03
        # The boost provides a small bonus above the raw similarity
        raw_confidence = max(min_confidence, similarity)
        boosted_confidence = min(0.98, raw_confidence + 0.03)

        # Build reasoning string
        reasoning = (
            f"obvious_match_early_termination: "
            f"sim={similarity:.3f}, "
            f"keywords={len(matched_keywords)}, "
            f"entities={matched_entity_names[:3]}"
        )

        logger.info(
            f"  Obvious match detected: {reasoning} "
            f"(skipping LLM, confidence={boosted_confidence:.3f})"
        )

        return boosted_confidence, reasoning, matched_entity_names

    # Note: _get_cache_key, _get_cached_response, _cache_response moved to LLMReranker

    def _update_recent_matches(self, match: Match) -> None:
        """
        Update recent matches list after a successful match (US-63-009).

        Keeps the most recent N matches for consecutive source penalty calculation.
        Most recent match is at index 0.
        """
        self._recent_matches.insert(0, match)
        # Trim to max size
        if len(self._recent_matches) > self._max_recent_matches:
            self._recent_matches = self._recent_matches[:self._max_recent_matches]

    def _update_chapter_source_counts(self, vo_segment: SRTSegment, video_segment: SRTSegment) -> None:
        """
        Track unique video sources per voiceover chapter (US-75-006).

        Used by chapter_coherence_penalty to detect excessive source diversity.
        """
        chapter_idx = getattr(vo_segment, 'chapter_index', None)
        if chapter_idx is None or chapter_idx < 0:
            return
        source = getattr(video_segment, 'source_file', None)
        if not source:
            return
        if chapter_idx not in self._chapter_source_counts:
            self._chapter_source_counts[chapter_idx] = set()
        self._chapter_source_counts[chapter_idx].add(source)

    def set_chapter_confidence_map(self, chapters) -> None:
        """
        Build chapter confidence map from chapter candidates (US-76-012).

        Args:
            chapters: List of ChapterCandidate objects or dicts with chapter_id and confidence
        """
        self._chapter_confidence_map = {}
        if not chapters:
            return
        for ch in chapters:
            if isinstance(ch, dict):
                ch_id = ch.get('chapter_id', ch.get('chapter_index'))
                conf = ch.get('confidence', 0.8)
            else:
                ch_id = getattr(ch, 'chapter_id', None)
                conf = getattr(ch, 'confidence', 0.8)
            if ch_id is not None:
                self._chapter_confidence_map[ch_id] = conf

    def _get_vo_chapter_confidence(self, vo_segment) -> float:
        """Look up chapter detection confidence for a voiceover segment (US-76-012)."""
        chapter_idx = getattr(vo_segment, 'chapter_index', None)
        if chapter_idx is None:
            return 1.0
        return self._chapter_confidence_map.get(chapter_idx, 1.0)

    def _is_within_chapter_boundary(self, vo_segment: SRTSegment) -> bool:
        """
        Check if current segment is within the same chapter as the previous match (US-127-005).

        Used to suppress consecutive_source_penalty within chapter boundaries since
        source consistency is desirable within a coherent chapter.

        Args:
            vo_segment: Current voiceover segment being matched

        Returns:
            True if current segment is in the same chapter as the previous match
        """
        # Need chapter grouping enabled
        mc = getattr(self.config, 'matching', None)
        if mc is None:
            return False
        cg = getattr(mc, 'chapter_grouping', None)
        if cg is None or not getattr(cg, 'enabled', True):
            return False

        # Get current chapter index
        current_chapter_idx = getattr(vo_segment, 'chapter_index', None)
        if current_chapter_idx is None or current_chapter_idx < 0:
            return False

        # Check if there's a previous match in the same chapter
        if not self._recent_matches:
            return False

        prev_match = self._recent_matches[0]
        if prev_match is None or prev_match.voiceover_segment is None:
            return False

        prev_chapter_idx = getattr(prev_match.voiceover_segment, 'chapter_index', None)
        return prev_chapter_idx == current_chapter_idx and prev_chapter_idx >= 0

    def reset_recent_matches(self) -> None:
        """
        Reset recent matches list (US-63-009).

        Call this at the start of a new matching session to clear state.
        """
        self._recent_matches = []

    def set_embedding_lookup(self, video_segments: List[SRTSegment], video_embeddings: List) -> None:
        """
        Build embedding lookup from video segments and their embeddings (US-77-002).

        Creates a mapping from segment identity to embedding index so we can
        retrieve the embedding of a matched video segment for semantic coherence scoring.
        """
        self._video_embeddings = video_embeddings
        self._embedding_lookup = {id(seg): i for i, seg in enumerate(video_segments)}

    def _get_segment_embedding(self, segment: SRTSegment) -> Optional[Any]:
        """Look up embedding vector for a video segment (US-77-002)."""
        if self._video_embeddings is None:
            return None
        idx = self._embedding_lookup.get(id(segment))
        if idx is not None and idx < len(self._video_embeddings):
            return self._video_embeddings[idx]
        return None

    def _apply_semantic_coherence(
        self, adjusted_confidence: float, best_seg: SRTSegment,
        confidence_breakdown: list
    ) -> Tuple[float, str]:
        """
        Apply semantic coherence scoring and update previous embedding (US-77-002).

        Returns (adjusted_confidence, reason).
        """
        mc = self.config.matching
        semantic_enabled = getattr(mc, 'semantic_coherence_enabled', True)
        current_embedding = self._get_segment_embedding(best_seg)

        prev = adjusted_confidence
        adjustment, reason = compute_semantic_coherence(
            current_embedding, self._previous_match_embedding,
            semantic_coherence_enabled=semantic_enabled,
            scoring_config=mc,
        )
        adjusted_confidence += adjustment
        _record_breakdown(confidence_breakdown, 'semantic_coherence', prev, adjusted_confidence, reason)

        # Update previous embedding for next segment
        if current_embedding is not None:
            self._previous_match_embedding = current_embedding

        return adjusted_confidence, reason

    def _apply_temporal_coherence(
        self, adjusted_confidence: float, best_seg: SRTSegment,
        confidence_breakdown: list
    ) -> Tuple[float, str]:
        """
        Apply temporal coherence scoring and update previous match segment (US-77-003).

        Returns (adjusted_confidence, reason).
        """
        prev = adjusted_confidence
        adjusted_confidence, reason = compute_temporal_coherence(
            adjusted_confidence, best_seg, self._previous_match_segment,
            None, self.config
        )
        _record_breakdown(confidence_breakdown, 'temporal_coherence', prev, adjusted_confidence, reason)

        # Update previous match segments for next iteration (shift window)
        self._prev_prev_match_segment = self._previous_match_segment
        self._previous_match_segment = best_seg

        return adjusted_confidence, reason

    def _apply_thematic_consistency(
        self, adjusted_confidence: float, vo_segment: SRTSegment,
        best_seg: SRTSegment, confidence_breakdown: list
    ) -> Tuple[float, str]:
        """
        Apply thematic consistency scoring (US-134-010).

        When matched videos within the window share common themes/topics with
        the voiceover, apply a confidence boost.

        Returns (adjusted_confidence, reason).
        """
        # Get window size from config
        mc = self.config.matching
        window_size = getattr(mc, 'thematic_consistency_window', 3)

        # Get matched segments within the window (last N segments)
        window_segments = self._matched_segments_list[-window_size:] if self._matched_segments_list else []

        prev = adjusted_confidence
        adjusted_confidence, reason = compute_thematic_consistency(
            adjusted_confidence, vo_segment, best_seg, window_segments, self.config
        )
        _record_breakdown(confidence_breakdown, 'thematic_consistency', prev, adjusted_confidence, reason)

        # Update matched segments list (add current match)
        self._matched_segments_list.append(best_seg)

        # Keep list size bounded to window_size * 2 (enough for sliding window)
        if len(self._matched_segments_list) > window_size * 2:
            self._matched_segments_list = self._matched_segments_list[-(window_size * 2):]

        return adjusted_confidence, reason

    def _apply_channel_reputation(
        self, adjusted_confidence: float, best_seg: SRTSegment,
        confidence_breakdown: list
    ) -> Tuple[float, str]:
        """
        Apply channel reputation and engagement boost (US-111-005).

        Boosts confidence for videos from high-subscriber channels and/or
        high view count videos. Provides small but meaningful signal for
        content quality.

        Returns (adjusted_confidence, reason).
        """
        # Check if channel reputation scoring is enabled
        matching_config = getattr(self.config, 'matching', None)
        if not matching_config:
            return adjusted_confidence, "no_config"

        enabled = getattr(matching_config, 'channel_reputation_enabled', True)
        if not enabled:
            return adjusted_confidence, "disabled"

        # Get channel data from video_metadata
        channel_data = self._get_video_channel_data(best_seg)
        if not channel_data:
            return adjusted_confidence, "no_channel_data"

        subscriber_count = channel_data.get('subscriber_count')
        view_count = channel_data.get('view_count')

        boost_amount = 0.0
        reasons = []

        # Apply subscriber count boost
        threshold = getattr(matching_config, 'channel_reputation_threshold', 100000)
        max_boost = getattr(matching_config, 'channel_reputation_boost', 0.02)

        if subscriber_count is not None and subscriber_count > 0:
            if subscriber_count >= threshold:
                # Full boost for high-subscriber channels
                boost_amount = max_boost
                reasons.append(f"high_subscribers({subscriber_count:,})")
            elif subscriber_count >= threshold // 10:
                # Partial boost for moderate channels (10% of threshold)
                boost_amount = max_boost * 0.3
                reasons.append(f"moderate_subscribers({subscriber_count:,})")

        # Apply view count boost (US-134-009: enhanced with log scaling and quality threshold)
        view_weight = getattr(matching_config, 'view_count_context_weight', 0.02)
        if view_weight > 0 and view_count is not None and view_count > 0:
            quality_threshold = getattr(matching_config, 'view_count_quality_threshold', 1000)
            if view_count < quality_threshold:
                # Below quality threshold - no boost (low view count = unreliable signal)
                pass
            else:
                use_log_scale = getattr(matching_config, 'view_count_log_scale', True)
                if use_log_scale:
                    # Log scale: reduce outlier impact, scale boost based on magnitude
                    # log10(1000)=3, log10(1M)=6, log10(100M)=8
                    # Normalize: (log - log_threshold) / (log_max - log_threshold)
                    import math
                    log_views = math.log10(view_count)
                    log_threshold = math.log10(quality_threshold)
                    log_max = math.log10(getattr(matching_config, 'view_count_boost_threshold', 1000000))
                    # Scale: 0 at threshold, 1.0 at 1M views
                    scale_factor = min(1.0, max(0.0, (log_views - log_threshold) / (log_max - log_threshold)))
                    view_boost = view_weight * scale_factor
                else:
                    # Binary: only boost if above threshold (legacy behavior)
                    view_threshold = getattr(matching_config, 'view_count_boost_threshold', 1000000)
                    view_boost = view_weight if view_count >= view_threshold else 0

                if view_boost > 0:
                    boost_amount += view_boost
                    reasons.append(f"views({view_count:,},log_scale={use_log_scale})")

        if boost_amount > 0:
            prev = adjusted_confidence
            adjusted_confidence = min(1.0, adjusted_confidence + boost_amount)
            reason = ", ".join(reasons) if reasons else "channel_reputation"
            _record_breakdown(confidence_breakdown, 'channel_reputation', prev, adjusted_confidence, reason)
            return adjusted_confidence, reason

        return adjusted_confidence, "no_boost_triggers"

    def _apply_source_stutter_penalty(
        self, adjusted_confidence: float, best_seg: SRTSegment,
        confidence_breakdown: list
    ) -> Tuple[float, str]:
        """
        Apply A-B-A source stutter penalty (US-84-004).

        Detects when current source matches 2-segments-ago but differs from previous,
        creating a jarring visual ping-pong pattern.

        Note: Must be called BEFORE _apply_temporal_coherence, which shifts the
        segment window. At call time, _previous_match_segment is 1-back and
        _prev_prev_match_segment is 2-back relative to best_seg.

        Returns (adjusted_confidence, reason).
        """
        current_source = getattr(best_seg, 'source_file', None)
        previous_source = getattr(self._previous_match_segment, 'source_file', None) if self._previous_match_segment else None
        prev_prev_source = getattr(self._prev_prev_match_segment, 'source_file', None) if self._prev_prev_match_segment else None

        prev = adjusted_confidence
        adjusted_confidence, reason = apply_source_stutter_penalty(
            adjusted_confidence, current_source, previous_source,
            prev_prev_source, self.config
        )
        _record_breakdown(confidence_breakdown, 'source_stutter_penalty', prev, adjusted_confidence, reason)

        return adjusted_confidence, reason

    def _get_video_title(self, segment: SRTSegment) -> Optional[str]:
        """Resolve video title from video_metadata using segment's source_file."""
        if not self.video_metadata:
            return None
        meta = self.video_metadata.get(segment.source_file)
        if isinstance(meta, dict):
            return meta.get('title')
        return None

    def _get_video_description(self, segment: SRTSegment) -> Optional[str]:
        """Resolve video description from video_metadata using segment's source_file."""
        if not self.video_metadata:
            return None
        meta = self.video_metadata.get(segment.source_file)
        if isinstance(meta, dict):
            return meta.get('description')
        return None

    def _get_video_tags(self, segment: SRTSegment) -> Optional[List[str]]:
        """Resolve video tags from video_metadata using segment's source_file."""
        if not self.video_metadata:
            return None
        meta = self.video_metadata.get(segment.source_file)
        if isinstance(meta, dict):
            return meta.get('tags')
        return None

    def _get_video_chapters(self, segment: SRTSegment) -> List[dict]:
        """Resolve video chapters from video_metadata using segment's source_file.

        US-75-009: Returns chapter list from the video_metadata lookup.
        """
        if not self.video_metadata:
            return []
        meta = self.video_metadata.get(segment.source_file)
        if isinstance(meta, dict):
            return meta.get('chapters', []) or []
        return []

    def _get_video_channel(self, segment: SRTSegment) -> Optional[str]:
        """Resolve video channel from video_metadata using segment's source_file.

        US-95-006: Returns channel name from the video_metadata lookup.
        Used for source channel consistency scoring.
        """
        if not self.video_metadata:
            return None
        meta = self.video_metadata.get(segment.source_file)
        if isinstance(meta, dict):
            return meta.get('channel') or None
        return None

    def _get_video_channel_data(self, segment: SRTSegment) -> Optional[Dict[str, Any]]:
        """Resolve channel metrics (subscriber_count, view_count) from video_metadata.

        US-111-005: Returns channel data dict with subscriber_count and view_count.
        Used for channel reputation scoring.
        """
        if not self.video_metadata:
            return None
        meta = self.video_metadata.get(segment.source_file)
        if isinstance(meta, dict):
            subscriber_count = meta.get('channel_subscriber_count') or meta.get('subscriber_count')
            view_count = meta.get('view_count')
            if subscriber_count is not None or view_count is not None:
                return {
                    'subscriber_count': int(subscriber_count) if subscriber_count else None,
                    'view_count': int(view_count) if view_count else None,
                }
        return None

    def _get_chapter_title(self, segment: SRTSegment) -> Optional[str]:
        """Resolve chapter title from the video segment's chapter_title attribute."""
        return getattr(segment, 'chapter_title', None) or None

    def _get_scene_for_segment(
        self,
        segment: SRTSegment,
        scenes: Optional[Dict[str, List[SceneInfo]]]
    ) -> Optional[SceneInfo]:
        """Find the scene containing this segment"""
        if not scenes:
            return None

        video_scenes = scenes.get(segment.source_file, [])

        for scene in video_scenes:
            if scene.start_time <= segment.start_time < scene.end_time:
                return scene

        return None

    def _build_context(
        self,
        context_before: Optional[List[SRTSegment]],
        context_after: Optional[List[SRTSegment]]
    ) -> Optional[str]:
        """Build context string from surrounding segments with topic-enriched text (US-111-003)"""
        if not context_before and not context_after:
            return None

        # Get topic coherence config
        vt_config = getattr(self.config.matching, 'voiceover_topic', None)
        use_topics = getattr(vt_config, 'topic_coherence_enabled', True) if vt_config else True

        parts = []

        if context_before:
            # US-111-003: Use topic-enriched text when available
            if use_topics:
                before_texts = [get_topic_enriched_text(s, max_text_length=40) for s in context_before[-2:]]
                before_text = " | ".join(before_texts)
            else:
                before_text = " | ".join(s.text[:50] for s in context_before[-2:])
            parts.append(f"Before: {before_text}")

        if context_after:
            # US-111-003: Use topic-enriched text when available
            if use_topics:
                after_texts = [get_topic_enriched_text(s, max_text_length=40) for s in context_after[:2]]
                after_text = " | ".join(after_texts)
            else:
                after_text = " | ".join(s.text[:50] for s in context_after[:2])
            parts.append(f"After: {after_text}")

        return " || ".join(parts)

    def match_segment(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]] = None,
        context_before: Optional[List[SRTSegment]] = None,
        context_after: Optional[List[SRTSegment]] = None,
        segment_idx: int = 0
    ) -> MatchResult:
        """
        Match a single voiceover segment.
        Returns MatchResult with primary match and alternatives.

        Args:
            vo_segment: Voiceover segment to match
            candidates: List of (video_segment, similarity) tuples
            scenes: Optional scene info for videos
            context_before: Previous voiceover segments for context
            context_after: Following voiceover segments for context
            segment_idx: Index of voiceover segment (for location chapter lookup)
        """
        mc = self.config.matching

        # US-134-010: Reset thematic consistency tracking at start of each segment
        # This ensures clean state for each new match run
        self._matched_segments_list = []

        segment_start_time = time.time()
        logger.info(f"  match_segment: entering for '{vo_segment.text[:30]}...'")

        # Guard: return gap if no candidates
        if not candidates:
            logger.warning(f"  match_segment: no candidates for '{vo_segment.text[:30]}...'")
            return MatchResult(
                primary_match=create_gap_match(vo_segment, "No video candidates available"),
                has_gap=True, gap_reason="No candidates"
            )

        # Apply all candidate filters (US-33-006: delegated to CandidateFilter)
        filter_start_time = time.time()
        cache_dir = self.cache.cache_dir if hasattr(self.cache, 'cache_dir') else None

        filter_result = self.candidate_filter.apply_all_filters(
            vo_segment=vo_segment,
            candidates=candidates,
            segment_idx=segment_idx,
            reuse_tracker=self.reuse_tracker,
            cache_dir=cache_dir
        )
        valid_candidates = filter_result.candidates

        filter_elapsed = time.time() - filter_start_time
        if filter_elapsed > 1.0:
            logger.info(f"  match_segment: filtering took {filter_elapsed:.2f}s")

        if filter_result.face_filter_applied:
            logger.debug(f"  Face preference applied: {self.face_preference}")
        if filter_result.location_filter_applied:
            logger.debug(f"  Location filter: {filter_result.location_reason}")

        if not valid_candidates:
            logger.warning(f"  match_segment: no valid candidates after filtering")
            return MatchResult(
                primary_match=create_gap_match(vo_segment, "No valid candidates after filtering"),
                has_gap=True, gap_reason="All candidates filtered"
            )

        # Check for high-confidence embedding match
        top_similarity = valid_candidates[0][1] if valid_candidates else 0

        # Calculate adaptive threshold if enabled
        adaptive_threshold_enabled = getattr(mc, 'adaptive_threshold_enabled', True)
        if adaptive_threshold_enabled:
            skip_threshold, threshold_reason = calculate_adaptive_threshold(
                base_threshold=mc.skip_llm_threshold,
                voiceover_text=vo_segment.text,
                candidates=valid_candidates,
                config=self.config
            )
            # US-84-008: Detect ambiguous pool from threshold reason
            is_ambiguous_pool = 'ambiguous_pool' in threshold_reason
            logger.info(f"  match_segment: top_sim={top_similarity:.3f}, adaptive_threshold={skip_threshold:.3f} ({threshold_reason})")
        else:
            skip_threshold = mc.skip_llm_threshold
            is_ambiguous_pool = False
            logger.info(f"  match_segment: top_sim={top_similarity:.3f}, skip_threshold={skip_threshold}")

        # Check for obvious match (early termination before LLM)
        # This bypasses both the threshold check and LLM when match is obviously good
        best_seg = valid_candidates[0][0]
        matched_keywords_for_check = self._extract_matched_keywords(vo_segment, best_seg)
        obvious_result = self.check_obvious_match(
            vo_segment, best_seg, top_similarity, matched_keywords_for_check
        )

        if obvious_result:
            boosted_confidence, obvious_reasoning, matched_entities = obvious_result
            self.reuse_tracker.record_usage(best_seg)

            scene = self._get_scene_for_segment(best_seg, scenes)

            # Apply scoring adjustments (same as normal path for consistency)
            confidence_breakdown = []
            prev = boosted_confidence
            adjusted_confidence, topic_penalty_reason = apply_topic_penalty(
                boosted_confidence, vo_segment, best_seg,
                video_topics=self.video_topics,
                chapter_matching_enabled=self.chapter_matching_enabled,
                topic_mismatch_penalty=self.topic_mismatch_penalty
            )
            _record_breakdown(confidence_breakdown, 'topic_penalty', prev, adjusted_confidence, topic_penalty_reason)

            # US-95-007: Topic alignment boost
            prev = adjusted_confidence
            adjusted_confidence, topic_align_reason = apply_topic_alignment_boost(
                adjusted_confidence, vo_segment, best_seg,
                video_topics=self.video_topics,
                topic_alignment_weight=self.topic_alignment_weight
            )
            _record_breakdown(confidence_breakdown, 'topic_alignment_boost', prev, adjusted_confidence, topic_align_reason)

            prev = adjusted_confidence
            adjusted_confidence, broll_reason = apply_broll_boost(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'broll_boost', prev, adjusted_confidence, broll_reason)

            # US-95-011: Apply chapter alignment boost
            prev = adjusted_confidence
            video_chapters = self._get_video_chapters(best_seg)
            chapter_boost, chapter_align_reason = compute_chapter_alignment_boost(
                best_seg, video_chapters, self.config.matching
            )
            adjusted_confidence += chapter_boost
            _record_breakdown(confidence_breakdown, 'chapter_alignment_boost', prev, adjusted_confidence, chapter_align_reason)

            # US-007: Apply caption quality adjustment
            prev = adjusted_confidence
            adjusted_confidence, caption_quality_reason = apply_caption_quality_adjustment(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'caption_quality', prev, adjusted_confidence, caption_quality_reason)

            # US-008 Sprint 7: Apply timing penalty for poor caption timing
            prev = adjusted_confidence
            adjusted_confidence, timing_penalty_reason = apply_timing_penalty(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'timing_penalty', prev, adjusted_confidence, timing_penalty_reason)

            prev = adjusted_confidence
            adjusted_confidence, project_reason = apply_current_project_boost(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'project_boost', prev, adjusted_confidence, project_reason)

            # US-63-009: Apply consecutive source penalty (US-127-005: suppress within chapter)
            prev = adjusted_confidence
            suppress_in_chapter = self._is_within_chapter_boundary(vo_segment)
            adjusted_confidence, consecutive_reason = apply_consecutive_source_penalty(
                adjusted_confidence, best_seg, self._recent_matches, self.config,
                suppress_in_chapter=suppress_in_chapter
            )
            _record_breakdown(confidence_breakdown, 'consecutive_source_penalty', prev, adjusted_confidence, consecutive_reason)

            # US-75-002: Apply title relevance adjustment
            prev = adjusted_confidence
            video_title = self._get_video_title(best_seg)
            adjusted_confidence, title_relevance_reason = apply_title_relevance_adjustment(
                adjusted_confidence, vo_segment, video_title
            )
            _record_breakdown(confidence_breakdown, 'title_relevance', prev, adjusted_confidence, title_relevance_reason)

            # US-75-003: Apply description relevance adjustment
            prev = adjusted_confidence
            video_desc = self._get_video_description(best_seg)
            adjusted_confidence, desc_relevance_reason = apply_description_relevance_adjustment(
                adjusted_confidence, vo_segment, video_desc, self.config
            )
            _record_breakdown(confidence_breakdown, 'description_relevance', prev, adjusted_confidence, desc_relevance_reason)

            # US-75-004: Apply tag keyword boost
            prev = adjusted_confidence
            video_tags = self._get_video_tags(best_seg)
            adjusted_confidence, tag_boost_reason = apply_tag_keyword_boost(
                adjusted_confidence, vo_segment, video_tags
            )
            _record_breakdown(confidence_breakdown, 'tag_keyword_boost', prev, adjusted_confidence, tag_boost_reason)

            # US-141-007: Apply tag relevance scoring with position/frequency weighting
            prev = adjusted_confidence
            vo_keywords = list(self._get_keywords_for_segment(vo_segment))
            mc = self.config.matching
            context_config = getattr(mc, 'context', None)
            position_decay = getattr(context_config, 'tag_position_decay', 0.9) if context_config else 0.9
            frequency_weight = getattr(context_config, 'tag_frequency_weight', 0.15) if context_config else 0.15
            tag_relevance = compute_tag_relevance_score(
                tags=video_tags,
                vo_keywords=vo_keywords,
                position_decay=position_decay,
                frequency_weight=frequency_weight,
            )
            # Apply as a boost: scale relevance to a reasonable boost (max +0.08)
            tag_relevance_boost = tag_relevance * 0.08
            if tag_relevance_boost > 0:
                adjusted_confidence = min(1.0, adjusted_confidence + tag_relevance_boost)
                tag_relevance_reason = f"tag relevance +{tag_relevance_boost:.3f} (pos_decay={position_decay}, freq_w={frequency_weight})"
            else:
                tag_relevance_reason = ""
            _record_breakdown(confidence_breakdown, 'tag_relevance', prev, adjusted_confidence, tag_relevance_reason)

            # US-75-005: Apply chapter topic match
            prev = adjusted_confidence
            chapter_title = self._get_chapter_title(best_seg)
            _ch_conf = self._get_vo_chapter_confidence(vo_segment)
            adjusted_confidence, chapter_topic_reason = apply_chapter_topic_match(
                adjusted_confidence, vo_segment, chapter_title,
                chapter_matching_enabled=self.chapter_matching_enabled,
                chapter_confidence=_ch_conf,
            )
            _record_breakdown(confidence_breakdown, 'chapter_topic_match', prev, adjusted_confidence, chapter_topic_reason)

            # US-75-005: Apply chapter source consistency
            prev = adjusted_confidence
            adjusted_confidence, chapter_source_reason = apply_chapter_source_consistency(
                adjusted_confidence, best_seg, vo_segment, self._recent_matches,
                chapter_matching_enabled=self.chapter_matching_enabled,
                chapter_confidence=_ch_conf,
            )
            _record_breakdown(confidence_breakdown, 'chapter_source_consistency', prev, adjusted_confidence, chapter_source_reason)

            # US-95-006: Apply source channel consistency boost
            current_channel = self._get_video_channel(best_seg)
            if current_channel:
                prev = adjusted_confidence
                adjusted_confidence, channel_reason = apply_source_channel_consistency(
                    adjusted_confidence, vo_segment, best_seg,
                    recent_matches=self._recent_matches,
                    current_channel=current_channel,
                    config=self.config,
                )
                _record_breakdown(confidence_breakdown, 'source_channel_coherence', prev, adjusted_confidence, channel_reason)

            # US-75-006: Apply chapter coherence penalty
            prev = adjusted_confidence
            adjusted_confidence, coherence_reason = apply_chapter_coherence_penalty(
                adjusted_confidence, vo_segment,
                chapter_source_counts=self._chapter_source_counts,
                chapter_matching_enabled=self.chapter_matching_enabled,
                chapter_confidence=_ch_conf,
            )
            _record_breakdown(confidence_breakdown, 'chapter_coherence_penalty', prev, adjusted_confidence, coherence_reason)

            # US-75-006: Apply cross-chapter relevance boost
            prev = adjusted_confidence
            adjusted_confidence, cross_chapter_reason = apply_cross_chapter_relevance_boost(
                adjusted_confidence, vo_segment, best_seg,
                relevance_matrix=self.relevance_matrix,
                chapter_matching_enabled=self.chapter_matching_enabled
            )
            _record_breakdown(confidence_breakdown, 'cross_chapter_relevance', prev, adjusted_confidence, cross_chapter_reason)

            # US-95-004: Apply chapter boundary penalty
            prev = adjusted_confidence
            adjusted_confidence, chapter_boundary_reason = apply_chapter_boundary_penalty(
                adjusted_confidence, vo_segment, best_seg,
                enforce_boundaries=self.enforce_chapter_boundaries,
                penalty=self.cross_chapter_penalty,
            )
            _record_breakdown(confidence_breakdown, 'chapter_boundary_penalty', prev, adjusted_confidence, chapter_boundary_reason)

            # US-75-007: Apply listicle consistency boost
            prev = adjusted_confidence
            adjusted_confidence, listicle_reason = apply_listicle_consistency(
                adjusted_confidence, vo_segment, best_seg,
                listicle_groups=self.listicle_groups,
                recent_matches=self._recent_matches
            )
            _record_breakdown(confidence_breakdown, 'listicle_consistency', prev, adjusted_confidence, listicle_reason)

            # US-77-011: Apply entity match boost
            prev = adjusted_confidence
            adjusted_confidence, entity_boost_reason, _entity_matched = apply_entity_match_boost(
                adjusted_confidence, vo_segment, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'entity_match_boost', prev, adjusted_confidence, entity_boost_reason)

            # US-77-002: Apply semantic coherence (topic flow between adjacent matches)
            adjusted_confidence, semantic_coherence_reason = self._apply_semantic_coherence(
                adjusted_confidence, best_seg, confidence_breakdown
            )

            # US-84-004: Apply source stutter penalty (A-B-A pattern detection)
            adjusted_confidence, stutter_reason = self._apply_source_stutter_penalty(
                adjusted_confidence, best_seg, confidence_breakdown
            )

            # US-77-003: Apply temporal coherence (source continuity between adjacent matches)
            adjusted_confidence, temporal_coherence_reason = self._apply_temporal_coherence(
                adjusted_confidence, best_seg, confidence_breakdown
            )

            # US-134-010: Apply thematic consistency (theme alignment between matched videos)
            adjusted_confidence, thematic_consistency_reason = self._apply_thematic_consistency(
                adjusted_confidence, vo_segment, best_seg, confidence_breakdown
            )

            # US-95-010: Apply context richness calibration
            # Get metadata for calibration
            video_title = self._get_video_title(best_seg)
            video_desc = self._get_video_description(best_seg)
            video_tags = self._get_video_tags(best_seg)
            video_chapter = self._get_chapter_title(best_seg)

            # Get calibration settings from config
            calibration_enabled = getattr(mc, 'context_richness_calibration', True)
            calibration_boost = getattr(mc, 'context_richness_boost_max', 0.08)
            calibration_penalty = getattr(mc, 'context_richness_penalty_max', 0.05)

            # US-111-011: Get signal weights from config
            title_weight = getattr(mc, 'context_richness_title_weight', 0.25)
            description_weight = getattr(mc, 'context_richness_description_weight', 0.25)
            tags_weight = getattr(mc, 'context_richness_tags_weight', 0.25)
            chapters_weight = getattr(mc, 'context_richness_chapters_weight', 0.25)

            prev = adjusted_confidence
            adjusted_confidence, context_richness_reason = apply_context_richness_calibration(
                adjusted_confidence,
                video_title=video_title,
                video_description=video_desc,
                video_tags=video_tags,
                video_chapter=video_chapter,
                enabled=calibration_enabled,
                boost_max=calibration_boost,
                penalty_max=calibration_penalty,
                title_weight=title_weight,
                description_weight=description_weight,
                tags_weight=tags_weight,
                chapters_weight=chapters_weight,
            )
            _record_breakdown(confidence_breakdown, 'context_richness_calibration', prev, adjusted_confidence, context_richness_reason)

            # US-75-006: Update chapter source tracking
            self._update_chapter_source_counts(vo_segment, best_seg)

            # Ensure we don't drop below minimum confidence after adjustments
            min_confidence = getattr(mc, 'obvious_match_min_confidence', 0.92)
            if adjusted_confidence < min_confidence:
                prev = adjusted_confidence
                adjusted_confidence = min_confidence
                _record_breakdown(confidence_breakdown, 'obvious_match_min_confidence', prev, adjusted_confidence,
                                  f'confidence floor enforced: {min_confidence}')

            final_reasoning = obvious_reasoning
            if topic_penalty_reason:
                final_reasoning += f" [{topic_penalty_reason}]"
            if broll_reason:
                final_reasoning += f" [{broll_reason}]"
            if caption_quality_reason:
                final_reasoning += f" [{caption_quality_reason}]"
            if timing_penalty_reason:
                final_reasoning += f" [{timing_penalty_reason}]"
            if project_reason:
                final_reasoning += f" [{project_reason}]"
            if consecutive_reason:
                final_reasoning += f" [{consecutive_reason}]"
            if title_relevance_reason:
                final_reasoning += f" [{title_relevance_reason}]"
            if desc_relevance_reason:
                final_reasoning += f" [{desc_relevance_reason}]"
            if tag_boost_reason:
                final_reasoning += f" [{tag_boost_reason}]"
            if chapter_topic_reason:
                final_reasoning += f" [{chapter_topic_reason}]"
            if chapter_source_reason:
                final_reasoning += f" [{chapter_source_reason}]"
            if listicle_reason:
                final_reasoning += f" [{listicle_reason}]"
            if entity_boost_reason:
                final_reasoning += f" [{entity_boost_reason}]"
            if context_richness_reason:
                final_reasoning += f" [{context_richness_reason}]"

            # US-63-007: Store confidence breakdown on Match object
            match = Match(
                voiceover_segment=vo_segment,
                video_segment=best_seg,
                video_scene=scene,
                confidence=adjusted_confidence,
                reasoning=final_reasoning,
                embedding_similarity=top_similarity,
                clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg),
                confidence_breakdown=confidence_breakdown,
            )

            alternatives = self.alt_selector.get_alternatives(
                valid_candidates[1:4], scenes, best_seg, self._get_scene_for_segment
            )

            used_video_files = {best_seg.source_file}
            alt_segments = []
            for alt in alternatives:
                used_video_files.add(alt.video_segment.source_file)
                alt_segments.append(alt.video_segment)

            secondary_matches = self.alt_selector.get_secondary_matches(
                valid_candidates, scenes, used_video_files,
                primary_segment=best_seg, alt_segments=alt_segments,
                get_scene_fn=self._get_scene_for_segment
            )

            confidence_variance = self._calculate_confidence_variance(valid_candidates)

            # Log confidence breakdown at DEBUG level (US-63-007)
            if confidence_breakdown:
                parts = [f"{b['component']}: {b['adjustment']:+.2f}" for b in confidence_breakdown]
                logger.debug(f"US-63-007 confidence breakdown: {boosted_confidence:.2f} -> {adjusted_confidence:.2f} ({', '.join(parts)})")

            # US-63-009: Update recent matches for consecutive source tracking
            self._update_recent_matches(match)

            return MatchResult(
                primary_match=match,
                alternatives=alternatives,
                secondary_matches=secondary_matches,
                confidence_variance=confidence_variance,
                matched_keywords=matched_keywords_for_check,
                confidence_breakdown=confidence_breakdown,
                ambiguous_pool=is_ambiguous_pool,
            )

        if top_similarity >= skip_threshold:
            best_seg = valid_candidates[0][0]
            self.reuse_tracker.record_usage(best_seg)

            scene = self._get_scene_for_segment(best_seg, scenes)

            # Apply scoring adjustments using refactored functions
            confidence_breakdown = []
            prev = top_similarity
            adjusted_confidence, topic_penalty_reason = apply_topic_penalty(
                top_similarity, vo_segment, best_seg,
                video_topics=self.video_topics,
                chapter_matching_enabled=self.chapter_matching_enabled,
                topic_mismatch_penalty=self.topic_mismatch_penalty
            )
            _record_breakdown(confidence_breakdown, 'topic_penalty', prev, adjusted_confidence, topic_penalty_reason)

            # US-95-007: Topic alignment boost
            prev = adjusted_confidence
            adjusted_confidence, topic_align_reason = apply_topic_alignment_boost(
                adjusted_confidence, vo_segment, best_seg,
                video_topics=self.video_topics,
                topic_alignment_weight=self.topic_alignment_weight
            )
            _record_breakdown(confidence_breakdown, 'topic_alignment_boost', prev, adjusted_confidence, topic_align_reason)

            prev = adjusted_confidence
            adjusted_confidence, broll_reason = apply_broll_boost(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'broll_boost', prev, adjusted_confidence, broll_reason)

            # US-95-011: Apply chapter alignment boost
            prev = adjusted_confidence
            video_chapters = self._get_video_chapters(best_seg)
            chapter_boost, chapter_align_reason = compute_chapter_alignment_boost(
                best_seg, video_chapters, self.config.matching
            )
            adjusted_confidence += chapter_boost
            _record_breakdown(confidence_breakdown, 'chapter_alignment_boost', prev, adjusted_confidence, chapter_align_reason)

            # US-007: Apply caption quality adjustment
            prev = adjusted_confidence
            adjusted_confidence, caption_quality_reason = apply_caption_quality_adjustment(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'caption_quality', prev, adjusted_confidence, caption_quality_reason)

            # US-008 Sprint 7: Apply timing penalty for poor caption timing
            prev = adjusted_confidence
            adjusted_confidence, timing_penalty_reason = apply_timing_penalty(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'timing_penalty', prev, adjusted_confidence, timing_penalty_reason)

            prev = adjusted_confidence
            adjusted_confidence, project_reason = apply_current_project_boost(
                adjusted_confidence, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'project_boost', prev, adjusted_confidence, project_reason)

            # US-63-009: Apply consecutive source penalty (US-127-005: suppress within chapter)
            prev = adjusted_confidence
            suppress_in_chapter = self._is_within_chapter_boundary(vo_segment)
            adjusted_confidence, consecutive_reason = apply_consecutive_source_penalty(
                adjusted_confidence, best_seg, self._recent_matches, self.config,
                suppress_in_chapter=suppress_in_chapter
            )
            _record_breakdown(confidence_breakdown, 'consecutive_source_penalty', prev, adjusted_confidence, consecutive_reason)

            # US-75-002: Apply title relevance adjustment
            prev = adjusted_confidence
            video_title = self._get_video_title(best_seg)
            adjusted_confidence, title_relevance_reason = apply_title_relevance_adjustment(
                adjusted_confidence, vo_segment, video_title
            )
            _record_breakdown(confidence_breakdown, 'title_relevance', prev, adjusted_confidence, title_relevance_reason)

            # US-75-003: Apply description relevance adjustment
            prev = adjusted_confidence
            video_desc = self._get_video_description(best_seg)
            adjusted_confidence, desc_relevance_reason = apply_description_relevance_adjustment(
                adjusted_confidence, vo_segment, video_desc, self.config
            )
            _record_breakdown(confidence_breakdown, 'description_relevance', prev, adjusted_confidence, desc_relevance_reason)

            # US-75-004: Apply tag keyword boost
            prev = adjusted_confidence
            video_tags = self._get_video_tags(best_seg)
            adjusted_confidence, tag_boost_reason = apply_tag_keyword_boost(
                adjusted_confidence, vo_segment, video_tags
            )
            _record_breakdown(confidence_breakdown, 'tag_keyword_boost', prev, adjusted_confidence, tag_boost_reason)

            # US-141-007: Apply tag relevance scoring with position/frequency weighting
            prev = adjusted_confidence
            vo_keywords = list(self._get_keywords_for_segment(vo_segment))
            mc = self.config.matching
            context_config = getattr(mc, 'context', None)
            position_decay = getattr(context_config, 'tag_position_decay', 0.9) if context_config else 0.9
            frequency_weight = getattr(context_config, 'tag_frequency_weight', 0.15) if context_config else 0.15
            tag_relevance = compute_tag_relevance_score(
                tags=video_tags,
                vo_keywords=vo_keywords,
                position_decay=position_decay,
                frequency_weight=frequency_weight,
            )
            # Apply as a boost: scale relevance to a reasonable boost (max +0.08)
            tag_relevance_boost = tag_relevance * 0.08
            if tag_relevance_boost > 0:
                adjusted_confidence = min(1.0, adjusted_confidence + tag_relevance_boost)
                tag_relevance_reason = f"tag relevance +{tag_relevance_boost:.3f} (pos_decay={position_decay}, freq_w={frequency_weight})"
            else:
                tag_relevance_reason = ""
            _record_breakdown(confidence_breakdown, 'tag_relevance', prev, adjusted_confidence, tag_relevance_reason)

            # US-75-005: Apply chapter topic match
            prev = adjusted_confidence
            chapter_title = self._get_chapter_title(best_seg)
            _ch_conf = self._get_vo_chapter_confidence(vo_segment)
            adjusted_confidence, chapter_topic_reason = apply_chapter_topic_match(
                adjusted_confidence, vo_segment, chapter_title,
                chapter_matching_enabled=self.chapter_matching_enabled,
                chapter_confidence=_ch_conf,
            )
            _record_breakdown(confidence_breakdown, 'chapter_topic_match', prev, adjusted_confidence, chapter_topic_reason)

            # US-75-005: Apply chapter source consistency
            prev = adjusted_confidence
            adjusted_confidence, chapter_source_reason = apply_chapter_source_consistency(
                adjusted_confidence, best_seg, vo_segment, self._recent_matches,
                chapter_matching_enabled=self.chapter_matching_enabled,
                chapter_confidence=_ch_conf,
            )
            _record_breakdown(confidence_breakdown, 'chapter_source_consistency', prev, adjusted_confidence, chapter_source_reason)

            # US-95-006: Apply source channel consistency boost
            current_channel = self._get_video_channel(best_seg)
            if current_channel:
                prev = adjusted_confidence
                adjusted_confidence, channel_reason = apply_source_channel_consistency(
                    adjusted_confidence, vo_segment, best_seg,
                    recent_matches=self._recent_matches,
                    current_channel=current_channel,
                    config=self.config,
                )
                _record_breakdown(confidence_breakdown, 'source_channel_coherence', prev, adjusted_confidence, channel_reason)

            # US-75-006: Apply chapter coherence penalty
            prev = adjusted_confidence
            adjusted_confidence, coherence_reason = apply_chapter_coherence_penalty(
                adjusted_confidence, vo_segment,
                chapter_source_counts=self._chapter_source_counts,
                chapter_matching_enabled=self.chapter_matching_enabled,
                chapter_confidence=_ch_conf,
            )
            _record_breakdown(confidence_breakdown, 'chapter_coherence_penalty', prev, adjusted_confidence, coherence_reason)

            # US-75-006: Apply cross-chapter relevance boost
            prev = adjusted_confidence
            adjusted_confidence, cross_chapter_reason = apply_cross_chapter_relevance_boost(
                adjusted_confidence, vo_segment, best_seg,
                relevance_matrix=self.relevance_matrix,
                chapter_matching_enabled=self.chapter_matching_enabled
            )
            _record_breakdown(confidence_breakdown, 'cross_chapter_relevance', prev, adjusted_confidence, cross_chapter_reason)

            # US-95-004: Apply chapter boundary penalty
            prev = adjusted_confidence
            adjusted_confidence, chapter_boundary_reason = apply_chapter_boundary_penalty(
                adjusted_confidence, vo_segment, best_seg,
                enforce_boundaries=self.enforce_chapter_boundaries,
                penalty=self.cross_chapter_penalty,
            )
            _record_breakdown(confidence_breakdown, 'chapter_boundary_penalty', prev, adjusted_confidence, chapter_boundary_reason)

            # US-75-007: Apply listicle consistency boost
            prev = adjusted_confidence
            adjusted_confidence, listicle_reason = apply_listicle_consistency(
                adjusted_confidence, vo_segment, best_seg,
                listicle_groups=self.listicle_groups,
                recent_matches=self._recent_matches
            )
            _record_breakdown(confidence_breakdown, 'listicle_consistency', prev, adjusted_confidence, listicle_reason)

            # US-77-011: Apply entity match boost
            prev = adjusted_confidence
            adjusted_confidence, entity_boost_reason, _entity_matched = apply_entity_match_boost(
                adjusted_confidence, vo_segment, best_seg, self.config
            )
            _record_breakdown(confidence_breakdown, 'entity_match_boost', prev, adjusted_confidence, entity_boost_reason)

            # US-77-002: Apply semantic coherence (topic flow between adjacent matches)
            adjusted_confidence, semantic_coherence_reason = self._apply_semantic_coherence(
                adjusted_confidence, best_seg, confidence_breakdown
            )

            # US-84-004: Apply source stutter penalty (A-B-A pattern detection)
            adjusted_confidence, stutter_reason = self._apply_source_stutter_penalty(
                adjusted_confidence, best_seg, confidence_breakdown
            )

            # US-77-003: Apply temporal coherence (source continuity between adjacent matches)
            adjusted_confidence, temporal_coherence_reason = self._apply_temporal_coherence(
                adjusted_confidence, best_seg, confidence_breakdown
            )

            # US-134-010: Apply thematic consistency (theme alignment between matched videos)
            adjusted_confidence, thematic_consistency_reason = self._apply_thematic_consistency(
                adjusted_confidence, vo_segment, best_seg, confidence_breakdown
            )

            # US-95-010: Apply context richness calibration
            # Get metadata for calibration
            video_title = self._get_video_title(best_seg)
            video_desc = self._get_video_description(best_seg)
            video_tags = self._get_video_tags(best_seg)
            video_chapter = self._get_chapter_title(best_seg)

            # Get calibration settings from config
            calibration_enabled = getattr(mc, 'context_richness_calibration', True)
            calibration_boost = getattr(mc, 'context_richness_boost_max', 0.08)
            calibration_penalty = getattr(mc, 'context_richness_penalty_max', 0.05)

            # US-111-011: Get signal weights from config
            title_weight = getattr(mc, 'context_richness_title_weight', 0.25)
            description_weight = getattr(mc, 'context_richness_description_weight', 0.25)
            tags_weight = getattr(mc, 'context_richness_tags_weight', 0.25)
            chapters_weight = getattr(mc, 'context_richness_chapters_weight', 0.25)

            prev = adjusted_confidence
            adjusted_confidence, context_richness_reason = apply_context_richness_calibration(
                adjusted_confidence,
                video_title=video_title,
                video_description=video_desc,
                video_tags=video_tags,
                video_chapter=video_chapter,
                enabled=calibration_enabled,
                boost_max=calibration_boost,
                penalty_max=calibration_penalty,
                title_weight=title_weight,
                description_weight=description_weight,
                tags_weight=tags_weight,
                chapters_weight=chapters_weight,
            )
            _record_breakdown(confidence_breakdown, 'context_richness_calibration', prev, adjusted_confidence, context_richness_reason)

            # US-75-006: Update chapter source tracking
            self._update_chapter_source_counts(vo_segment, best_seg)

            reasoning = f"High embedding similarity ({top_similarity:.2f})"
            if topic_penalty_reason:
                reasoning += f" [{topic_penalty_reason}]"
            if broll_reason:
                reasoning += f" [{broll_reason}]"
            if caption_quality_reason:
                reasoning += f" [{caption_quality_reason}]"
            if timing_penalty_reason:
                reasoning += f" [{timing_penalty_reason}]"
            if project_reason:
                reasoning += f" [{project_reason}]"
            if consecutive_reason:
                reasoning += f" [{consecutive_reason}]"
            if title_relevance_reason:
                reasoning += f" [{title_relevance_reason}]"
            if desc_relevance_reason:
                reasoning += f" [{desc_relevance_reason}]"
            if tag_boost_reason:
                reasoning += f" [{tag_boost_reason}]"
            if chapter_topic_reason:
                reasoning += f" [{chapter_topic_reason}]"
            if chapter_source_reason:
                reasoning += f" [{chapter_source_reason}]"
            if listicle_reason:
                reasoning += f" [{listicle_reason}]"
            if entity_boost_reason:
                reasoning += f" [{entity_boost_reason}]"
            if context_richness_reason:
                reasoning += f" [{context_richness_reason}]"

            # US-63-007: Store confidence breakdown on Match object
            match = Match(
                voiceover_segment=vo_segment,
                video_segment=best_seg,
                video_scene=scene,
                confidence=adjusted_confidence,
                reasoning=reasoning,
                embedding_similarity=top_similarity,
                clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg),
                confidence_breakdown=confidence_breakdown,
            )

            alternatives = self.alt_selector.get_alternatives(
                valid_candidates[1:4], scenes, best_seg, self._get_scene_for_segment
            )

            used_video_files = {best_seg.source_file}
            alt_segments = []
            for alt in alternatives:
                used_video_files.add(alt.video_segment.source_file)
                alt_segments.append(alt.video_segment)

            secondary_matches = self.alt_selector.get_secondary_matches(
                valid_candidates, scenes, used_video_files,
                primary_segment=best_seg, alt_segments=alt_segments,
                get_scene_fn=self._get_scene_for_segment
            )

            # Calculate confidence variance for top candidates
            confidence_variance = self._calculate_confidence_variance(valid_candidates)

            # Extract matched keywords between voiceover and selected video
            matched_keywords = self._extract_matched_keywords(vo_segment, best_seg)

            # Log confidence breakdown at DEBUG level (US-63-007)
            if confidence_breakdown:
                parts = [f"{b['component']}: {b['adjustment']:+.2f}" for b in confidence_breakdown]
                logger.debug(f"US-63-007 confidence breakdown: {top_similarity:.2f} -> {adjusted_confidence:.2f} ({', '.join(parts)})")

            # US-63-009: Update recent matches for consecutive source tracking
            self._update_recent_matches(match)

            return MatchResult(
                primary_match=match,
                alternatives=alternatives,
                secondary_matches=secondary_matches,
                confidence_variance=confidence_variance,
                matched_keywords=matched_keywords,
                confidence_breakdown=confidence_breakdown,
                ambiguous_pool=is_ambiguous_pool,
            )

        # US-141-004: Cross-signal validation - apply BEFORE LLM reranking
        # Filter/penalize candidates with inconsistent signals before LLM evaluation
        cross_signal_enabled = getattr(self.llm_reranker.config, 'cross_signal_validation_enabled', True)
        penalty_max = getattr(self.llm_reranker.config, 'consistency_penalty_max', 0.05)

        if cross_signal_enabled and self.video_metadata:
            validated_candidates = []
            for seg, sim in valid_candidates[:5]:
                source_file = seg.source_file
                video_meta = self.video_metadata.get(source_file, {})
                if video_meta:
                    title = video_meta.get('title', '')
                    description = video_meta.get('description', '')
                    tags = video_meta.get('tags', [])
                    consistency_score, consistency_penalty = validate_context_consistency(
                        title, description, tags, penalty_max
                    )
                    # Apply penalty to similarity before LLM
                    adjusted_sim = max(0.0, sim - consistency_penalty)
                    validated_candidates.append((seg, adjusted_sim, consistency_penalty))
                else:
                    validated_candidates.append((seg, sim, 0.0))
            # Pass validation info to LLM reranker
            candidate_penalties = {seg.source_file: pen for seg, sim, pen in validated_candidates}
        else:
            validated_candidates = [(seg, sim, 0.0) for seg, sim in valid_candidates[:5]]
            candidate_penalties = {}

        # Build context and call LLMReranker (handles caching internally)
        llm_start_time = time.time()
        context = self._build_context(context_before, context_after)
        negative_rules = self.config.negative_matching.rules if self.config.negative_matching.enabled else None

        # Use LLMReranker for candidate selection (with cross-signal penalties applied)
        logger.info(f"  match_segment: calling LLMReranker.rerank()...")
        # Extract (seg, sim) pairs from validated_candidates for LLM
        llm_candidates = [(seg, sim) for seg, sim, _ in validated_candidates]
        rerank_result = self.llm_reranker.rerank(
            voiceover_text=vo_segment.text,
            candidates=llm_candidates,
            primary_provider=self.primary_provider,
            secondary_provider=self.secondary_provider,
            context=context,
            negative_rules=negative_rules,
            video_metadata=self.video_metadata
        )
        selected_idx = rerank_result.selected_idx
        confidence = rerank_result.confidence
        reasoning = rerank_result.reasoning
        logger.info(f"  match_segment: LLM reranker returned results")

        llm_elapsed = time.time() - llm_start_time
        if llm_elapsed > 1.0:
            logger.info(f"  match_segment: LLM call took {llm_elapsed:.2f}s")

        # Build result
        selected_idx = min(selected_idx, len(valid_candidates) - 1)
        best_seg = valid_candidates[selected_idx][0]
        self.reuse_tracker.record_usage(best_seg)

        scene = self._get_scene_for_segment(best_seg, scenes)

        # Check for keyword/visual matches (for is_kw_match/is_vis_match flags)
        vo_keywords = getattr(vo_segment, 'keywords', []) or []
        seg_keywords = getattr(best_seg, 'keywords', []) or []
        keyword_boost, is_kw_match, is_vis_match = find_keyword_matches(
            vo_keywords,
            seg_keywords,
            scene.visual_keywords if scene else None
        )

        # Use multimodal scoring instead of simple additive boosting
        # This replaces: base_confidence = min(1.0, confidence + keyword_boost)
        embedding_sim = valid_candidates[selected_idx][1]
        multimodal_conf, multimodal_reason, mm_components = self._compute_multimodal_confidence(
            vo_segment, best_seg, embedding_sim, scene
        )

        # Use multimodal confidence as the base, but blend with LLM confidence
        # LLM provides contextual understanding, multimodal provides signal fusion
        mc = self.config.matching
        multimodal_enabled = getattr(mc, 'multimodal_enabled', True)
        if multimodal_enabled:
            # Blend: 60% multimodal + 40% LLM confidence for semantic understanding
            base_confidence = multimodal_conf * 0.6 + confidence * 0.4
        else:
            # Fall back to original additive boosting
            base_confidence = min(1.0, confidence + keyword_boost)

        # Apply remaining scoring adjustments (penalties/boosts not captured by multimodal)
        confidence_breakdown = []

        # US-84-006: Record LLM reasoning quality penalty in breakdown
        if rerank_result.llm_reasoning_quality == 0:
            penalty = getattr(self.llm_reranker.config, 'low_quality_reasoning_penalty', 0.05)
            _record_breakdown(
                confidence_breakdown, 'llm_reasoning_penalty',
                base_confidence + penalty, base_confidence,
                f"Low-quality LLM reasoning: -{penalty}"
            )

        # US-84-006: Record secondary LLM decay in breakdown
        if rerank_result.used_secondary:
            decay = getattr(self.llm_reranker.config, 'secondary_llm_decay', 0.9)
            # The decay was applied as confidence * decay inside reranker
            # Record the effective adjustment: conf_before_decay - conf_after_decay
            undecayed = base_confidence / decay if decay > 0 else base_confidence
            _record_breakdown(
                confidence_breakdown, 'secondary_llm_decay',
                undecayed, base_confidence,
                f"Secondary LLM fallback decay: {decay}x"
            )

        # US-141-004: Record cross-signal penalty from pre-LLM validation
        if cross_signal_enabled and candidate_penalties:
            source_file = best_seg.source_file
            applied_penalty = candidate_penalties.get(source_file, 0.0)
            if applied_penalty > 0:
                # The penalty was already applied to similarity before LLM
                # Record it in the breakdown
                _record_breakdown(
                    confidence_breakdown, 'cross_signal_penalty',
                    base_confidence + applied_penalty, base_confidence,
                    f"Cross-signal inconsistency penalty: -{applied_penalty:.3f}"
                )

        prev = base_confidence
        adjusted_confidence, topic_penalty_reason = apply_topic_penalty(
            base_confidence, vo_segment, best_seg,
            video_topics=self.video_topics,
            chapter_matching_enabled=self.chapter_matching_enabled,
            topic_mismatch_penalty=self.topic_mismatch_penalty
        )
        _record_breakdown(confidence_breakdown, 'topic_penalty', prev, adjusted_confidence, topic_penalty_reason)

        prev = adjusted_confidence
        adjusted_confidence, broll_reason = apply_broll_boost(
            adjusted_confidence, best_seg, self.config
        )
        _record_breakdown(confidence_breakdown, 'broll_boost', prev, adjusted_confidence, broll_reason)

        # US-95-011: Apply chapter alignment boost
        prev = adjusted_confidence
        video_chapters = self._get_video_chapters(best_seg)
        chapter_boost, chapter_align_reason = compute_chapter_alignment_boost(
            best_seg, video_chapters, self.config.matching
        )
        adjusted_confidence += chapter_boost
        _record_breakdown(confidence_breakdown, 'chapter_alignment_boost', prev, adjusted_confidence, chapter_align_reason)

        # US-007: Apply caption quality adjustment
        prev = adjusted_confidence
        adjusted_confidence, caption_quality_reason = apply_caption_quality_adjustment(
            adjusted_confidence, best_seg, self.config
        )
        _record_breakdown(confidence_breakdown, 'caption_quality', prev, adjusted_confidence, caption_quality_reason)

        # US-008 Sprint 7: Apply timing penalty for poor caption timing
        prev = adjusted_confidence
        adjusted_confidence, timing_penalty_reason = apply_timing_penalty(
            adjusted_confidence, best_seg, self.config
        )
        _record_breakdown(confidence_breakdown, 'timing_penalty', prev, adjusted_confidence, timing_penalty_reason)

        prev = adjusted_confidence
        adjusted_confidence, project_reason = apply_current_project_boost(
            adjusted_confidence, best_seg, self.config
        )
        _record_breakdown(confidence_breakdown, 'project_boost', prev, adjusted_confidence, project_reason)

        # US-63-009: Apply consecutive source penalty (US-127-005: suppress within chapter)
        prev = adjusted_confidence
        suppress_in_chapter = self._is_within_chapter_boundary(vo_segment)
        adjusted_confidence, consecutive_reason = apply_consecutive_source_penalty(
            adjusted_confidence, best_seg, self._recent_matches, self.config,
            suppress_in_chapter=suppress_in_chapter
        )
        _record_breakdown(confidence_breakdown, 'consecutive_source_penalty', prev, adjusted_confidence, consecutive_reason)

        # US-75-002: Apply title relevance adjustment
        prev = adjusted_confidence
        video_title = self._get_video_title(best_seg)
        adjusted_confidence, title_relevance_reason = apply_title_relevance_adjustment(
            adjusted_confidence, vo_segment, video_title
        )
        _record_breakdown(confidence_breakdown, 'title_relevance', prev, adjusted_confidence, title_relevance_reason)

        # US-75-003: Apply description relevance adjustment
        prev = adjusted_confidence
        video_desc = self._get_video_description(best_seg)
        adjusted_confidence, desc_relevance_reason = apply_description_relevance_adjustment(
            adjusted_confidence, vo_segment, video_desc
        )
        _record_breakdown(confidence_breakdown, 'description_relevance', prev, adjusted_confidence, desc_relevance_reason)

        # US-75-004: Apply tag keyword boost
        prev = adjusted_confidence
        video_tags = self._get_video_tags(best_seg)
        adjusted_confidence, tag_boost_reason = apply_tag_keyword_boost(
            adjusted_confidence, vo_segment, video_tags
        )
        _record_breakdown(confidence_breakdown, 'tag_keyword_boost', prev, adjusted_confidence, tag_boost_reason)

        # US-141-007: Apply tag relevance scoring with position/frequency weighting
        prev = adjusted_confidence
        vo_keywords = list(self._get_keywords_for_segment(vo_segment))
        mc = self.config.matching
        context_config = getattr(mc, 'context', None)
        position_decay = getattr(context_config, 'tag_position_decay', 0.9) if context_config else 0.9
        frequency_weight = getattr(context_config, 'tag_frequency_weight', 0.15) if context_config else 0.15
        tag_relevance = compute_tag_relevance_score(
            tags=video_tags,
            vo_keywords=vo_keywords,
            position_decay=position_decay,
            frequency_weight=frequency_weight,
        )
        # Apply as a boost: scale relevance to a reasonable boost (max +0.08)
        tag_relevance_boost = tag_relevance * 0.08
        if tag_relevance_boost > 0:
            adjusted_confidence = min(1.0, adjusted_confidence + tag_relevance_boost)
            tag_relevance_reason = f"tag relevance +{tag_relevance_boost:.3f} (pos_decay={position_decay}, freq_w={frequency_weight})"
        else:
            tag_relevance_reason = ""
        _record_breakdown(confidence_breakdown, 'tag_relevance', prev, adjusted_confidence, tag_relevance_reason)

        # US-75-005: Apply chapter topic match
        prev = adjusted_confidence
        chapter_title = self._get_chapter_title(best_seg)
        _ch_conf = self._get_vo_chapter_confidence(vo_segment)
        adjusted_confidence, chapter_topic_reason = apply_chapter_topic_match(
            adjusted_confidence, vo_segment, chapter_title,
            chapter_matching_enabled=self.chapter_matching_enabled,
            chapter_confidence=_ch_conf,
        )
        _record_breakdown(confidence_breakdown, 'chapter_topic_match', prev, adjusted_confidence, chapter_topic_reason)

        # US-75-005: Apply chapter source consistency
        prev = adjusted_confidence
        adjusted_confidence, chapter_source_reason = apply_chapter_source_consistency(
            adjusted_confidence, best_seg, vo_segment, self._recent_matches,
            chapter_matching_enabled=self.chapter_matching_enabled,
            chapter_confidence=_ch_conf,
        )
        _record_breakdown(confidence_breakdown, 'chapter_source_consistency', prev, adjusted_confidence, chapter_source_reason)

        # US-75-006: Apply chapter coherence penalty
        prev = adjusted_confidence
        adjusted_confidence, coherence_reason = apply_chapter_coherence_penalty(
            adjusted_confidence, vo_segment,
            chapter_source_counts=self._chapter_source_counts,
            chapter_matching_enabled=self.chapter_matching_enabled,
            chapter_confidence=_ch_conf,
        )
        _record_breakdown(confidence_breakdown, 'chapter_coherence_penalty', prev, adjusted_confidence, coherence_reason)

        # US-75-006: Apply cross-chapter relevance boost
        prev = adjusted_confidence
        adjusted_confidence, cross_chapter_reason = apply_cross_chapter_relevance_boost(
            adjusted_confidence, vo_segment, best_seg,
            relevance_matrix=self.relevance_matrix,
            chapter_matching_enabled=self.chapter_matching_enabled
        )
        _record_breakdown(confidence_breakdown, 'cross_chapter_relevance', prev, adjusted_confidence, cross_chapter_reason)

        # US-75-007: Apply listicle consistency boost
        prev = adjusted_confidence
        adjusted_confidence, listicle_reason = apply_listicle_consistency(
            adjusted_confidence, vo_segment, best_seg,
            listicle_groups=self.listicle_groups,
            recent_matches=self._recent_matches
        )
        _record_breakdown(confidence_breakdown, 'listicle_consistency', prev, adjusted_confidence, listicle_reason)

        # US-77-011: Apply entity match boost
        prev = adjusted_confidence
        adjusted_confidence, entity_boost_reason, _entity_matched = apply_entity_match_boost(
            adjusted_confidence, vo_segment, best_seg, self.config
        )
        _record_breakdown(confidence_breakdown, 'entity_match_boost', prev, adjusted_confidence, entity_boost_reason)

        # US-77-002: Apply semantic coherence (topic flow between adjacent matches)
        adjusted_confidence, semantic_coherence_reason = self._apply_semantic_coherence(
            adjusted_confidence, best_seg, confidence_breakdown
        )

        # US-84-004: Apply source stutter penalty (A-B-A pattern detection)
        adjusted_confidence, stutter_reason = self._apply_source_stutter_penalty(
            adjusted_confidence, best_seg, confidence_breakdown
        )

        # US-77-003: Apply temporal coherence (source continuity between adjacent matches)
        adjusted_confidence, temporal_coherence_reason = self._apply_temporal_coherence(
            adjusted_confidence, best_seg, confidence_breakdown
        )

        # US-134-010: Apply thematic consistency (theme alignment between matched videos)
        adjusted_confidence, thematic_consistency_reason = self._apply_thematic_consistency(
            adjusted_confidence, vo_segment, best_seg, confidence_breakdown
        )

        # US-95-010: Apply context richness calibration
        # Get metadata for calibration
        video_title = self._get_video_title(best_seg)
        video_desc = self._get_video_description(best_seg)
        video_tags = self._get_video_tags(best_seg)
        video_chapter = self._get_chapter_title(best_seg)

        # Get calibration settings from config
        calibration_enabled = getattr(mc, 'context_richness_calibration', True)
        calibration_boost = getattr(mc, 'context_richness_boost_max', 0.08)
        calibration_penalty = getattr(mc, 'context_richness_penalty_max', 0.05)

        prev = adjusted_confidence
        adjusted_confidence, context_richness_reason = apply_context_richness_calibration(
            adjusted_confidence,
            video_title=video_title,
            video_description=video_desc,
            video_tags=video_tags,
            video_chapter=video_chapter,
            enabled=calibration_enabled,
            boost_max=calibration_boost,
            penalty_max=calibration_penalty,
        )
        _record_breakdown(confidence_breakdown, 'context_richness_calibration', prev, adjusted_confidence, context_richness_reason)

        # US-77-004: Apply explanation validation (verify LLM reasoning keywords)
        explanation_validation_reason = ""
        explanation_validation_enabled = getattr(mc, 'explanation_validation_enabled', True)
        if explanation_validation_enabled:
            prev = adjusted_confidence
            video_text = getattr(best_seg, 'text', None) or None
            explanation_result = validate_explanation_confidence(
                explanation=reasoning,
                voiceover_text=vo_segment.text,
                video_text=video_text,
            )
            if not explanation_result.is_valid:
                adjusted_confidence = max(0.0, adjusted_confidence - explanation_result.confidence_penalty)
                explanation_validation_reason = (
                    f"Explanation validation failed: {explanation_result.verification_ratio:.0%} "
                    f"keywords verified, -{explanation_result.confidence_penalty} penalty"
                )
            _record_breakdown(confidence_breakdown, 'explanation_validation', prev, adjusted_confidence, explanation_validation_reason)

        # US-75-006: Update chapter source tracking
        self._update_chapter_source_counts(vo_segment, best_seg)

        final_reasoning = reasoning
        if multimodal_enabled:
            final_reasoning += f" [{multimodal_reason}]"
        if topic_penalty_reason:
            final_reasoning += f" [{topic_penalty_reason}]"
        if broll_reason:
            final_reasoning += f" [{broll_reason}]"
        if caption_quality_reason:
            final_reasoning += f" [{caption_quality_reason}]"
        if timing_penalty_reason:
            final_reasoning += f" [{timing_penalty_reason}]"
        if project_reason:
            final_reasoning += f" [{project_reason}]"
        if consecutive_reason:
            final_reasoning += f" [{consecutive_reason}]"
        if title_relevance_reason:
            final_reasoning += f" [{title_relevance_reason}]"
        if desc_relevance_reason:
            final_reasoning += f" [{desc_relevance_reason}]"
        if tag_boost_reason:
            final_reasoning += f" [{tag_boost_reason}]"
        if chapter_topic_reason:
            final_reasoning += f" [{chapter_topic_reason}]"
        if chapter_source_reason:
            final_reasoning += f" [{chapter_source_reason}]"
        if listicle_reason:
            final_reasoning += f" [{listicle_reason}]"
        if entity_boost_reason:
            final_reasoning += f" [{entity_boost_reason}]"
        if context_richness_reason:
            final_reasoning += f" [{context_richness_reason}]"
        if explanation_validation_reason:
            final_reasoning += f" [{explanation_validation_reason}]"
        if rerank_result.llm_reasoning_quality == 0:
            final_reasoning += f" [llm_reasoning_quality=low]"
        if rerank_result.used_secondary:
            final_reasoning += f" [secondary_llm_decay]"

        # US-63-007: Store confidence breakdown on Match object
        match = Match(
            voiceover_segment=vo_segment,
            video_segment=best_seg,
            video_scene=scene,
            confidence=adjusted_confidence,
            reasoning=final_reasoning,
            is_keyword_match=is_kw_match,
            is_visual_match=is_vis_match,
            embedding_similarity=valid_candidates[selected_idx][1],
            clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg),
            confidence_breakdown=confidence_breakdown,
        )

        # Log match decision
        run_logger = get_global_logger()
        if run_logger:
            run_logger.log_match_decision(
                segment_index=segment_idx,
                voiceover_text=vo_segment.text,
                selected_clip=best_seg.source_file,
                confidence=adjusted_confidence,
                reasoning=final_reasoning,
                embedding_similarity=valid_candidates[selected_idx][1],
                duration_penalty=0.0,
                keyword_boost=keyword_boost,
                entity_boost=0.0,
                is_keyword_match=is_kw_match,
                is_visual_match=is_vis_match,
                alternatives_considered=len(candidates),
                llm_reranked=True if self.primary_provider else False,
                clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg)
            )

        # Get alternatives and secondary matches
        alternatives = self.alt_selector.get_alternatives(
            [c for i, c in enumerate(valid_candidates[:4]) if i != selected_idx],
            scenes, best_seg, self._get_scene_for_segment
        )

        used_video_files = {best_seg.source_file}
        alt_segments = []
        for alt in alternatives:
            used_video_files.add(alt.video_segment.source_file)
            alt_segments.append(alt.video_segment)

        secondary_matches = self.alt_selector.get_secondary_matches(
            valid_candidates, scenes, used_video_files,
            primary_segment=best_seg, alt_segments=alt_segments,
            get_scene_fn=self._get_scene_for_segment
        )

        # Check for gap
        has_gap = confidence < self.config.matching.confidence_threshold
        gap_reason = f"Low confidence ({confidence:.2f})" if has_gap else ""

        logger.debug(f"Match decision for segment: '{vo_segment.text[:50]}...'")
        logger.debug(f"  Video: {Path(best_seg.source_file).name} @ {best_seg.start_time:.1f}s")
        logger.debug(f"  Confidence: {confidence:.2f}")
        logger.debug(f"  Reason: {reasoning[:100]}...")
        logger.debug(f"  Alternatives: {len(alternatives)}, Has gap: {has_gap}")

        # Calculate confidence variance for top candidates
        confidence_variance = self._calculate_confidence_variance(valid_candidates)

        # Extract matched keywords between voiceover and selected video
        matched_keywords = self._extract_matched_keywords(vo_segment, best_seg)

        total_elapsed = time.time() - segment_start_time
        if total_elapsed > 2.0:
            logger.info(f"  match_segment: TOTAL time for segment was {total_elapsed:.2f}s")

        # Log confidence breakdown at DEBUG level (US-63-007)
        if confidence_breakdown:
            parts = [f"{b['component']}: {b['adjustment']:+.2f}" for b in confidence_breakdown]
            logger.debug(f"US-63-007 confidence breakdown: {base_confidence:.2f} -> {adjusted_confidence:.2f} ({', '.join(parts)})")

        # US-63-009: Update recent matches for consecutive source tracking
        self._update_recent_matches(match)

        return MatchResult(
            primary_match=match,
            alternatives=alternatives,
            secondary_matches=secondary_matches,
            has_gap=has_gap,
            gap_reason=gap_reason,
            confidence_variance=confidence_variance,
            matched_keywords=matched_keywords,
            confidence_breakdown=confidence_breakdown,
            ambiguous_pool=is_ambiguous_pool,
        )

    def review_with_local_llm(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Use local LLM to review and potentially adjust low-confidence matches"""
        if not self.local_provider:
            return matches

        low_confidence = [
            (i, m) for i, m in enumerate(matches)
            if m.primary_match.confidence < self.config.matching.ambiguous_threshold
        ]

        if not low_confidence:
            return matches

        logger.info(f"Reviewing {len(low_confidence)} low-confidence matches with local LLM...")

        for i, (idx, match_result) in enumerate(low_confidence):
            primary = match_result.primary_match

            # Log progress every match
            logger.info(f"  Local LLM review: {i+1}/{len(low_confidence)} - '{primary.voiceover_segment.text[:40]}...'")

            candidates = [
                (primary.video_segment, primary.confidence),
                *[(alt.video_segment, alt.confidence) for alt in match_result.alternatives]
            ]

            try:
                results = self.local_provider.match_batch(
                    [(primary.voiceover_segment.text, candidates)]
                )
                new_idx, new_conf, new_reason, _ = results[0]

                if new_conf > primary.confidence:
                    logger.debug(f"Local LLM improved match: {primary.confidence:.2f} -> {new_conf:.2f}")

                    if new_idx == 0:
                        primary.confidence = new_conf
                        primary.reasoning = f"(local refined) {new_reason}"
                    else:
                        new_seg = candidates[new_idx][0]
                        match_result.primary_match = Match(
                            voiceover_segment=primary.voiceover_segment,
                            video_segment=new_seg,
                            video_scene=match_result.alternatives[new_idx-1].video_scene if new_idx <= len(match_result.alternatives) else None,
                            confidence=new_conf,
                            reasoning=f"(local selected) {new_reason}",
                            embedding_similarity=candidates[new_idx][1]
                        )
            except Exception as e:
                logger.debug(f"Local LLM review failed: {e}")

        return matches

    def enforce_chapter_source_diversity(self, matches: List[MatchResult]) -> List[MatchResult]:
        """Post-processing pass to enforce minimum source diversity per chapter (US-77-007).

        After initial matching, checks if any voiceover chapter uses only a single
        video source for all segments. When a chapter has >4 segments all from the
        same source, the lowest-confidence segments are swapped to their best
        alternative from a different source.

        Args:
            matches: List of MatchResult from the main match loop

        Returns:
            The same list, mutated in-place where diversity swaps occurred
        """
        chapter_grouping = getattr(self.config.matching, 'chapter_grouping', None)
        if not chapter_grouping:
            return matches
        if not getattr(chapter_grouping, 'enabled', True):
            return matches

        min_diversity = getattr(chapter_grouping, 'min_source_diversity', 2)
        if min_diversity <= 1:
            return matches

        # Group match indices by voiceover chapter
        chapter_segments: Dict[int, List[int]] = {}
        for idx, result in enumerate(matches):
            vo_seg = result.primary_match.voiceover_segment
            ch_idx = getattr(vo_seg, 'chapter_index', None)
            if ch_idx is None or ch_idx < 0:
                continue
            if ch_idx not in chapter_segments:
                chapter_segments[ch_idx] = []
            chapter_segments[ch_idx].append(idx)

        swapped = 0
        for ch_idx, seg_indices in chapter_segments.items():
            if len(seg_indices) <= 4:
                continue

            # Count unique sources in this chapter
            sources = {}
            for idx in seg_indices:
                src = matches[idx].primary_match.video_segment.source_file
                if src not in sources:
                    sources[src] = []
                sources[src].append(idx)

            if len(sources) >= min_diversity:
                continue

            # All segments (or nearly all) from a single source — need diversity
            # Sort by confidence ascending to find lowest-confidence candidates
            candidates_for_swap = sorted(
                seg_indices,
                key=lambda i: matches[i].primary_match.confidence
            )

            for idx in candidates_for_swap:
                result = matches[idx]
                current_source = result.primary_match.video_segment.source_file

                # Try to find an alternative from a different source
                best_alt = None
                best_alt_conf = -1.0
                for alt in result.alternatives:
                    if alt.video_segment.source_file != current_source and alt.confidence > best_alt_conf:
                        best_alt = alt
                        best_alt_conf = alt.confidence

                if best_alt is None:
                    continue

                # Swap: replace primary with the alternative
                old_confidence = result.primary_match.confidence
                result.primary_match = Match(
                    voiceover_segment=result.primary_match.voiceover_segment,
                    video_segment=best_alt.video_segment,
                    video_scene=best_alt.video_scene,
                    confidence=best_alt.confidence,
                    reasoning=f"(diversity swap) {best_alt.reasoning}",
                    embedding_similarity=getattr(best_alt, 'embedding_similarity', 0.0),
                    confidence_breakdown=list(getattr(result.primary_match, 'confidence_breakdown', []))
                )

                # Add diversity_recheck entry to confidence_breakdown
                diversity_entry = {
                    'component': 'diversity_recheck',
                    'adjustment': round(best_alt.confidence - old_confidence, 4),
                    'reason': f"diversity_recheck: swapped source in chapter {ch_idx} (was {current_source})"
                }
                result.primary_match.confidence_breakdown.append(diversity_entry)
                result.confidence_breakdown.append(diversity_entry)
                swapped += 1

                # Re-check: did we reach diversity target?
                new_sources = set()
                for i in seg_indices:
                    new_sources.add(matches[i].primary_match.video_segment.source_file)
                if len(new_sources) >= min_diversity:
                    break

        if swapped > 0:
            logger.info(f"US-77-007 diversity enforcement: swapped {swapped} segment(s) across chapters")

        return matches


__all__ = ['TieredMatcher', 'compute_scoring_audit_summary', 'log_scoring_audit_summary']
