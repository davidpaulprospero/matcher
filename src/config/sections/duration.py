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
        """Convert dict fields to DurationTierConfig instances."""
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

                # Validate min/max
                if min_sec > 0 and max_sec > 0:
                    if min_sec >= max_sec:
                        logger.warning(
                            f"duration_tiers.{tier_name}: min >= max ({min_sec} >= {max_sec}), values may need adjustment"
                        )

                setattr(self, tier_name, DurationTierConfig(
                    min_seconds=min_sec,
                    max_seconds=max_sec,
                    videos_per_keyword=count,
                    max_total=max_total
                ))


@dataclass
class StockFootageConfig:
    """Stock footage API settings (Pexels, Pixabay)"""
    enabled: bool = True
    pexels_enabled: bool = True
    pixabay_enabled: bool = True
    per_keyword: int = 3
    min_duration: int = 5
    max_duration: int = 60
    min_height: int = 720
    prefer_landscape: bool = True
    request_interval: float = 0.5
