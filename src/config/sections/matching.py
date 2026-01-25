"""Matching configuration: Matching engine, location matching, negative matching.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

__all__ = [
    'LocationMatchingConfig',
    'NegativeMatchingConfig',
    'ChapterDetectionConfig',
    'MatchingConfig',
]


@dataclass
class LocationMatchingConfig:
    """Location-aware matching settings for geographic content.

    When chapters focus on specific locations (cities, countries, landmarks),
    this enables geographic filtering and scoring to ensure videos match
    the correct geographic context.

    Features:
    - Hard filtering by country (prevents Paris, TX matching Paris, FR content)
    - Soft penalty fallback when filter too strict
    - Hierarchy bonus (France video can match Paris chapter)
    - GeoNames API integration for geocoding
    """
    enabled: bool = True

    # GeoNames API (free account at geonames.org)
    geonames_username: str = ""  # Required for API calls

    # Filtering level: how strict the hard filter should be
    # Options: "city" (strictest), "state", "country", "continent" (loosest)
    hard_filter_level: str = "city"

    # Scoring adjustments
    geographic_penalty: float = 0.4     # Penalty for wrong location (soft mode fallback)
    hierarchy_bonus: float = 0.15       # Bonus for parent/child match (France for Paris)
    landmark_bonus: float = 0.2         # Bonus when landmark detected in video

    # Disambiguation
    use_llm_disambiguation: bool = True  # Use LLM to resolve ambiguous locations

    # Caching
    cache_dir: str = ".cache/locations"

    def __post_init__(self):
        # Read from environment variable if not set in config
        if not self.geonames_username:
            self.geonames_username = os.getenv("GEONAMES_USERNAME", "")


@dataclass
class NegativeMatchingConfig:
    """Negative matching rules - what NOT to match"""
    enabled: bool = True
    rules: List[str] = field(default_factory=lambda: [
        "Don't match talking head shots to action narration",
        "Don't match static images to dynamic narration",
        "Avoid matching unrelated b-roll to specific statements"
    ])


@dataclass
class ChapterDetectionConfig:
    """Enhanced chapter detection settings.

    Multi-pass chapter detection with:
    - Pass 1: Initial topic/location detection
    - Pass 2: Boundary refinement using embeddings
    - Pass 3: Cross-validation with LLM
    - Pass 4: Gap/overlap resolution
    """
    enabled: bool = True  # Enable enhanced multi-pass detection

    # Pass controls
    use_validation_pass: bool = True  # Pass 3: Cross-validate chapters
    use_boundary_refinement: bool = True  # Pass 2: Refine with embeddings

    # Strategy selection
    default_strategy: str = 'topic'  # 'topic' or 'location'
    auto_detect_content_type: bool = True  # Auto-switch strategy based on content

    # Chunking for long transcripts
    max_chunk_chars: int = 6000  # Max chars per LLM call
    chunk_overlap_segments: int = 5  # Segments to overlap between chunks

    # Chapter constraints
    min_chapter_confidence: float = 0.5  # Filter low-confidence chapters
    min_chapter_segments: int = 3  # Minimum segments per chapter
    max_chapters: int = 20  # Maximum chapters to detect


@dataclass
class MatchingConfig:
    """Matching engine settings

    Chain-of-thought: Two-stage matching optimizes cost vs quality
    Reasoning: Embedding search cheap (FAISS), LLM expensive
    Decision: Retrieve 20 candidates → rerank top 5 with LLM
    """
    # Confidence thresholds
    min_confidence: float = 0.7
    high_confidence_threshold: float = 0.85  # Skip LLM if above
    skip_llm_threshold: float = 0.85  # Alias for high_confidence_threshold (for matching.py compatibility)
    low_confidence_threshold: float = 0.5    # Use secondary LLM if below
    ambiguous_threshold: float = 0.6  # Use secondary LLM if confidence < this
    confidence_threshold: float = 0.5  # Legacy alias for min_confidence

    # Adaptive threshold (adjusts skip_llm_threshold based on voiceover characteristics)
    # Short voiceover (<20 chars): +0.05 threshold (harder to match, require higher confidence)
    # Low candidate variance (<0.05): -0.05 threshold (clear winner, can accept lower)
    adaptive_threshold_enabled: bool = True

    # Negative sampling for LLM reranking
    # When enabled, includes a "unlikely match" sample from bottom 25% of candidates
    # This helps LLMs calibrate confidence by showing what a poor match looks like
    negative_sampling_enabled: bool = True

    # Clip reuse prevention
    max_clip_reuse: int = 1
    reuse_penalty: float = 0.5
    smart_reuse: bool = True
    sequential_when_reuse: bool = True

    # Global clip deduplication (hard block mode)
    # When True, same clip can NEVER appear twice anywhere in timeline (P1 requirement)
    clip_hard_block: bool = True

    # Source file reuse prevention (limits how many times any segment from same video can be used)
    max_source_file_reuse: int = 3  # 0 = unlimited, 3 = max 3 clips from same video
    source_file_penalty: float = 0.05  # Penalty per use after reaching half the max

    # Two-stage matching optimization
    embedding_candidates: int = 50  # Retrieve from FAISS (need 50+ for V1-V6)
    llm_rerank_candidates: int = 5  # Send to LLM
    top_k_candidates: int = 10  # Legacy alias

    # Context window
    context_window: int = 2  # Consider N segments before/after

    # Duration scoring
    duration_scoring_enabled: bool = True
    ideal_speed_min: float = 0.85
    ideal_speed_max: float = 1.15
    soft_speed_min: float = 0.70
    soft_speed_max: float = 1.50
    duration_penalty_factor: float = 0.1

    # Tuple versions for matching.py compatibility
    @property
    def ideal_speed_range(self) -> tuple:
        return (self.ideal_speed_min, self.ideal_speed_max)

    @property
    def soft_penalty_range(self) -> tuple:
        return (self.soft_speed_min, self.soft_speed_max)

    # Keyword/entity boosting
    keyword_boost: float = 0.05
    entity_boost: float = 0.08

    # LLM providers (tiered: primary → secondary → local)
    primary_provider: str = "gemini"
    secondary_provider: str = "anthropic"
    local_provider: str = "ollama"
    use_local_for_review: bool = True

    # Model names
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.2"
    local_llm_model: str = "llama3.2"  # Alias for ollama_model
    ollama_host: str = "http://localhost:11434"  # Ollama API host

    # Caching
    cache_llm_responses: bool = True
    cache_ttl_hours: int = 24

    # Delta matching (only match new videos)
    delta_matching_enabled: bool = True  # Enable delta-aware matching
    force_rematch: bool = False  # Force rematch all videos (CLI override)
    rematch_improvement_threshold: float = 0.05  # Re-evaluate if new video > existing + this

    # Chapter/topic matching
    chapter_matching_enabled: bool = True  # Enable chapter-based topic filtering
    topic_mismatch_penalty: float = 0.15  # Confidence penalty for topic mismatch
    extract_video_topics: bool = True  # Extract topics from video transcripts
    min_topic_overlap: int = 1  # Minimum topic keywords that must match

    # B-roll boost (silent videos are valuable)
    # Boosts confidence for: 1) silent/B-roll videos, 2) scenes without faces when topic matches
    broll_boost: float = 0.2  # Confidence boost for B-roll videos (0.0-0.3)

    # Caption quality boost/penalty (US-007)
    # Adjusts confidence based on caption quality: high, medium, low
    # High quality (human captions): +boost, Medium (auto): no change, Low (fallback): -penalty
    caption_quality_adjustment_enabled: bool = True  # Enable caption quality confidence adjustment
    caption_quality_high_boost: float = 0.05  # Boost for high-quality human captions
    caption_quality_low_penalty: float = 0.1  # Penalty for low-quality/fallback captions

    # B-roll preference (scene-level face detection)
    # When topic matches, prefer scenes without faces (B-roll) over talking heads
    prefer_broll_when_topic_matches: bool = True  # Enable B-roll preference
    broll_face_threshold: float = 0.3  # face_score < this = B-roll (no faces)

    # Temporal coherence (adjacent segment scoring)
    # Encourages visual continuity by scoring clips based on similarity to adjacent segments
    temporal_coherence_enabled: bool = True  # Enable temporal coherence scoring
    temporal_coherence_same_source_boost: float = 0.05  # Boost for clips from same source as adjacent
    temporal_coherence_context_switch_penalty: float = 0.05  # Penalty for jarring context switches

    # Fallback matching (for edge cases when primary matching fails)
    # Triggers when primary matching returns confidence < fallback_trigger_threshold
    # Provides 3 levels: keyword-only (0.7), visual-description (0.5), generic-broll (0.3)
    fallback_matching_enabled: bool = True
    fallback_trigger_threshold: float = 0.4  # Trigger fallback when confidence below this

    # Transcript quality scoring (adjust embedding weight based on transcript quality)
    # Low-quality transcripts (short, incoherent, repetitive) are less reliable for matching
    # When enabled, reduces embedding weight by 20% for low-quality transcripts
    transcript_quality_weight: bool = True  # Enable transcript quality-based weight adjustment

    # Pool size normalization (adjusts confidence based on candidate pool size)
    # Small pools (<10): Boost confidence when top match is clear
    # Large pools (>100): Reduce confidence when margins are tight
    # Formula: sqrt(pool_size/50) capped at [0.8, 1.2]
    pool_normalization_enabled: bool = True  # Enable confidence normalization by pool size

    # Chain-of-thought prompting for LLM matching
    # When enabled, uses structured 4-step reasoning prompt:
    # 1. Identify voiceover themes, 2. List video elements
    # 3. Evaluate rubric (visual 30%, topic 40%, keyword 20%, flow 10%)
    # 4. Compute weighted final score
    # Produces more reliable and explainable match decisions
    chain_of_thought_enabled: bool = True  # Enable chain-of-thought structured prompting

    # Early termination for obvious high-confidence matches
    # When enabled, skips LLM when match is obviously good:
    # - Embedding similarity > 0.9
    # - At least 3 keywords match
    # - Same named entity found in both voiceover and video
    # Returns boosted confidence (min 0.92) with reason 'obvious_match_early_termination'
    obvious_match_enabled: bool = True  # Enable early termination for obvious matches
    obvious_match_min_similarity: float = 0.9  # Minimum embedding similarity for obvious match
    obvious_match_min_keywords: int = 3  # Minimum matched keywords for obvious match
    obvious_match_min_confidence: float = 0.92  # Minimum confidence for obvious matches

    # Multi-modal similarity weighting
    # When enabled, combines multiple similarity signals with weighted fusion:
    # - text_embedding (40%): Semantic similarity from embeddings
    # - keyword_overlap (25%): Matched keywords between voiceover and video
    # - entity_match (20%): Named entity overlap (people, places, organizations)
    # - visual_description (15%): Visual scene description similarity
    # Replaces simple additive boosting with weighted multi-modal scoring
    multimodal_enabled: bool = True  # Enable multi-modal similarity weighting
    multimodal_weights: dict = None  # Custom weights dict (defaults used if None)

    # Semantic coherence (topic flow between adjacent segments)
    # When enabled, evaluates embedding similarity between current match candidate
    # and the previous segment's match to ensure smooth topic transitions.
    # - Smooth flow (similarity > 0.6): +0.03 boost (good continuity)
    # - Abrupt flow (similarity < 0.3): -0.05 penalty (jarring transition)
    # - Neutral (0.3 - 0.6): no adjustment
    semantic_coherence_enabled: bool = True  # Enable semantic coherence scoring

    # Explanation confidence validation
    # When enabled, cross-checks that LLM explanation keywords appear in actual
    # voiceover/video text. When less than 50% of keywords are verifiable,
    # confidence is downgraded by 0.1 to penalize "hallucinated" explanations.
    explanation_validation_enabled: bool = True  # Enable explanation keyword verification

    # Location-aware matching (for travel/location content)
    location_matching: LocationMatchingConfig = None

    # Enhanced chapter detection
    chapter_detection: ChapterDetectionConfig = None

    def __post_init__(self):
        if self.location_matching is None:
            self.location_matching = LocationMatchingConfig()
        elif isinstance(self.location_matching, dict):
            self.location_matching = LocationMatchingConfig(**self.location_matching)

        if self.chapter_detection is None:
            self.chapter_detection = ChapterDetectionConfig()
        elif isinstance(self.chapter_detection, dict):
            self.chapter_detection = ChapterDetectionConfig(**self.chapter_detection)
