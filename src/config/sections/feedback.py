"""Feedback system configuration.

Controls rejection learning, channel reputation scoring, and editorial feedback integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ChannelCategoriesConfig:
    """Channel categorization for filtering.

    trusted: Channels that are always accepted (high-quality sources)
    blocked: Channels that are always rejected (trainers, movies, etc.)
    """

    trusted: List[str] = field(default_factory=lambda: [
        # Major documentary/news networks
        "BBC",
        "National Geographic",
        "PBS",
        "Discovery",
        "Smithsonian Channel",
        "NOVA PBS",
        # Professional wildlife channels
        "Nat Geo WILD",
        "Wildlife Films",
    ])

    blocked: List[str] = field(default_factory=lambda: [
        # Dog trainers
        "Cesar Millan",
        "Zak George's Dog Training Revolution",
        "McCann Dog Training",
        "Victoria Stilwell",
        "Kikopup",
        "Will Atherton Canine Training",
        # Movie/entertainment
        "Movieclips",
        "Netflix",
        "Warner Bros. Pictures",
        "Universal Pictures",
    ])


@dataclass
class ChannelScoringConfig:
    """Channel reputation scoring settings."""

    enabled: bool = True

    # Minimum score to accept videos from a channel (0.0-1.0)
    min_score: float = 0.3

    # Score bonuses for subscriber counts
    subscriber_bonus_1m: float = 0.1   # Channels with >1M subscribers
    subscriber_bonus_100k: float = 0.05  # Channels with >100K subscribers
    subscriber_bonus_10k: float = 0.02   # Channels with >10K subscribers

    # Weight for acceptance rate in score calculation
    acceptance_weight: float = 0.3

    # Weight for rejection penalty (applied at 2x rejection rate)
    rejection_weight: float = 0.4

    # Base score for unknown channels
    base_score: float = 0.5

    # Fetch metadata from YouTube API (requires YOUTUBE_API_KEY)
    fetch_youtube_metadata: bool = True


@dataclass
class CrossProjectConfig:
    """Cross-project learning settings."""

    enabled: bool = True

    # Client ID for grouping projects (e.g., "theresa", "stu")
    client_id: str = ""

    # Share rejections across projects for same client
    share_rejections: bool = True

    # Apply learnings from other projects by same client
    apply_learnings: bool = True

    # Auto-evolve preset after each project (update learned patterns)
    auto_evolve_preset: bool = True

    # Merge client rejections to global database after project
    merge_to_global: bool = True

    # Use evolved preset thresholds (duration, min_score, etc.)
    use_evolved_thresholds: bool = True


@dataclass
class FeedbackConfig:
    """Main feedback system configuration."""

    enabled: bool = True

    # Learn from rejected videos (add to global database)
    learn_from_rejections: bool = True

    # Rejection weight in scoring (higher = more penalty for rejected channels)
    rejection_weight: float = 2.0

    # Channel scoring
    channel_scoring: ChannelScoringConfig = field(default_factory=ChannelScoringConfig)

    # Channel categories (trusted/blocked lists)
    channel_categories: ChannelCategoriesConfig = field(default_factory=ChannelCategoriesConfig)

    # Cross-project learning
    cross_project: CrossProjectConfig = field(default_factory=CrossProjectConfig)

    # DaVinci Resolve integration
    import_davinci_markers: bool = True
    davinci_marker_prefix: str = "REJECT:"
    auto_import_path: str = ""  # Path to auto-import marker CSV on pipeline start

    def __post_init__(self):
        """Convert dicts to dataclasses if needed (Rule 2)."""
        if isinstance(self.channel_scoring, dict):
            self.channel_scoring = ChannelScoringConfig(**self.channel_scoring)
        if isinstance(self.channel_categories, dict):
            self.channel_categories = ChannelCategoriesConfig(**self.channel_categories)
        if isinstance(self.cross_project, dict):
            self.cross_project = CrossProjectConfig(**self.cross_project)
