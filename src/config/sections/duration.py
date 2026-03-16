"""Duration tier configuration: Video length tiers and stock footage settings.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    'DurationTierConfig',
    'DurationTiersConfig',
    'StockFootageConfig',
]


@dataclass
class DurationTierConfig:
    """Single duration tier configuration"""
    min_seconds: int = 0
    max_seconds: int = 120
    videos_per_keyword: int = 5
    max_total: int = 0  # 0 = no limit, >0 = project-level cap


@dataclass
class DurationTiersConfig:
    """Duration tiers for downloading"""
    short: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(20, 120, 8, 0))
    medium: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(120, 600, 8, 0))
    long: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(600, 1500, 5, 0))
    longer: DurationTierConfig = field(default_factory=lambda: DurationTierConfig(1500, 3000, 1, 1))  # 1 per project

    def __post_init__(self):
        """Convert dict fields to DurationTierConfig instances and validate."""
        import logging
        logger = logging.getLogger(__name__)

        for tier_name in ['short', 'medium', 'long', 'longer']:
            tier = getattr(self, tier_name)
            if isinstance(tier, dict):
                # Map various key formats to internal format
                min_sec = tier.get('min_seconds', tier.get('min', 0))
                max_sec = tier.get('max_seconds', tier.get('max', 0))
                count = tier.get('videos_per_keyword', tier.get('count', tier.get('per_keyword', 5)))
                max_total = tier.get('max_total', 0)

                setattr(self, tier_name, DurationTierConfig(
                    min_seconds=min_sec,
                    max_seconds=max_sec,
                    videos_per_keyword=count,
                    max_total=max_total
                ))
                tier = getattr(self, tier_name)

            # Validate min/max on all tiers (dict-converted or existing dataclass)
            if isinstance(tier, DurationTierConfig):
                if tier.min_seconds > 0 and tier.max_seconds > 0:
                    if tier.min_seconds >= tier.max_seconds:
                        logger.warning(
                            f"duration_tiers.{tier_name}: min >= max ({tier.min_seconds} >= {tier.max_seconds}), values may need adjustment"
                        )


@dataclass
class StockFootageConfig:
    """Stock footage API settings (Pexels, Pixabay)"""
    enabled: bool = True
    pexels_enabled: bool = True
    pixabay_enabled: bool = True
    per_keyword: int = 3
    # Cadence rule for V10 generic stock placement:
    # every N voiceover segments, select one segment for stock footage.
    segment_interval: int = 3
    # Segment selection mode for each N-segment block.
    # Supported: best_in_block, first_in_block, rotate_in_block
    selection_mode: str = "best_in_block"
    # Max stock clips to download for each selected segment.
    max_clips_per_selected_segment: int = 1
    min_duration: int = 5
    max_duration: int = 60
    min_height: int = 720
    prefer_landscape: bool = True
    request_interval: float = 0.5

    def __post_init__(self):
        if self.min_duration <= 0:
            raise ValueError(
                f"StockFootageConfig.min_duration must be positive "
                f"(seconds), got {self.min_duration}"
            )
        if self.max_duration <= 0:
            raise ValueError(
                f"StockFootageConfig.max_duration must be positive "
                f"(seconds), got {self.max_duration}"
            )
        if self.min_duration > self.max_duration:
            raise ValueError(
                f"StockFootageConfig.min_duration must be <= max_duration, "
                f"got min_duration={self.min_duration} > max_duration={self.max_duration}"
            )
        if self.segment_interval <= 0:
            raise ValueError(
                f"StockFootageConfig.segment_interval must be >= 1, got {self.segment_interval}"
            )
        if self.max_clips_per_selected_segment <= 0:
            raise ValueError(
                "StockFootageConfig.max_clips_per_selected_segment must be >= 1, "
                f"got {self.max_clips_per_selected_segment}"
            )
        valid_modes = {"best_in_block", "first_in_block", "rotate_in_block"}
        if self.selection_mode not in valid_modes:
            raise ValueError(
                f"StockFootageConfig.selection_mode must be one of {sorted(valid_modes)}, "
                f"got {self.selection_mode!r}"
            )
