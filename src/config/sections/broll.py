"""B-roll configuration: B-roll download and matching settings.

Extracted during B-roll improvement refactoring (Jan 2026).

BrollDownloadStage: Downloads videos specifically searched as B-roll.
BrollMatchStage: Matches silent scenes to voiceover for V8 track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

__all__ = [
    'BrollSourceBoostConfig',
    'BrollConfig',
]


@dataclass
class BrollSourceBoostConfig:
    """Score boosts for different B-roll sources."""
    youtube: float = 0.1   # YouTube B-roll preferred (more relevant to keywords)
    pexels: float = 0.05   # Stock footage secondary
    pixabay: float = 0.0   # No boost for Pixabay


@dataclass
class BrollConfig:
    """Configuration for B-roll download and matching stages.

    Two new pipeline stages:
    1. BrollDownloadStage: Downloads videos using B-roll-specific search terms
    2. BrollMatchStage: Matches silent scenes to voiceover for V8 track

    B-roll detection is scene-level (not video-level):
    - A video can have SOME scenes with speech and SOME silent
    - Scenes with < min_words_threshold are considered B-roll
    - Both face-detection B-roll and silent B-roll appear in V8
    """
    # Master enable
    enabled: bool = True

    # === Download stage (BrollDownloadStage) ===
    download_enabled: bool = True
    downloads_per_term: int = 3          # Videos per search term
    max_total_downloads: int = 30        # Cap on total B-roll downloads
    duration_tier: str = "short"         # "short" (<3min) or "medium" (3-10min)
    include_generic_searches: bool = True  # Include "stock footage compilation" etc.
    search_suffixes: List[str] = field(default_factory=lambda: [
        "b-roll",
        "footage",
        "cinematic"
    ])

    # === Detection (scene-level) ===
    min_words_threshold: int = 10        # Scenes with fewer words = B-roll
    ignore_markers: List[str] = field(default_factory=lambda: [
        "[Music]",
        "[Applause]",
        "[Silence]"
    ])

    # === Vision API enrichment ===
    vision_enabled: bool = True          # Auto-disabled in audio-first mode
    vision_prompt: str = "detailed"      # "brief", "detailed", "keywords_only"
    frames_per_scene: int = 3            # Sample start/mid/end frames
    cache_descriptions: bool = True      # Cache in .cache/broll_vision/
    vision_fallback_to_keywords: bool = True  # Use filename keywords if Vision fails

    # === Scoring weights (should sum to 1.0) ===
    embedding_weight: float = 0.4        # Vision/filename description similarity
    keyword_weight: float = 0.35         # Keyword overlap score
    entity_weight: float = 0.25          # Entity name matching

    # === Source priority ===
    source_boost: BrollSourceBoostConfig = field(default_factory=BrollSourceBoostConfig)

    # === Matching behavior ===
    min_match_score: float = 0.3         # Threshold for "good" match
    always_match: bool = True            # Always pick best B-roll (V8 never empty)
    max_matches_per_segment: int = 3     # Top N options per voiceover segment

    # === Long video handling ===
    long_video_threshold: int = 600      # Videos > 10 min use sampling
    sample_interval_seconds: int = 120   # 1 scene per 2 minutes
    max_scenes_per_video: int = 15       # Cap on sampled scenes

    def __post_init__(self):
        """Convert nested dicts to dataclasses."""
        if isinstance(self.source_boost, dict):
            self.source_boost = BrollSourceBoostConfig(**self.source_boost)
