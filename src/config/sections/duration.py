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
