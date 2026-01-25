"""
Checkpoint and duration tier management.

Migrated from VideoDownloader.__init__ and checkpoint methods (lines 46-387).
Handles checkpoint save/load for resuming downloads and duration tier configuration.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from ..config import Config
    from ..state import DownloadedVideo

logger = logging.getLogger(__name__)


@dataclass
class DownloadCheckpoint:
    """Checkpoint for resuming downloads.

    Migrated from downloader.py lines 46-61.
    """
    completed_keywords: List[str]
    completed_videos: List[str]
    failed_keywords: List[str]
    current_keyword: Optional[str]
    current_tier: Optional[str]
    timestamp: str
    # Speed tracker state for adaptive timeout persistence (US-005)
    speed_tracker_state: Optional[Dict] = None
    # Rate limit tracking for cross-session cooldown (US-008)
    # ISO timestamp of last rate limit event
    last_rate_limit_timestamp: Optional[str] = None
    # Count of rate limit events in current/last session
    rate_limit_event_count: int = 0
    # Rate limit metrics for cross-session analysis (US-010)
    rate_limit_metrics: Optional[Dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DownloadCheckpoint":
        # Handle checkpoints created before speed_tracker_state was added
        if 'speed_tracker_state' not in data:
            data['speed_tracker_state'] = None
        # Handle checkpoints created before rate limit tracking was added (US-008)
        if 'last_rate_limit_timestamp' not in data:
            data['last_rate_limit_timestamp'] = None
        if 'rate_limit_event_count' not in data:
            data['rate_limit_event_count'] = 0
        # Handle checkpoints created before rate limit metrics was added (US-010)
        if 'rate_limit_metrics' not in data:
            data['rate_limit_metrics'] = None
        return cls(**data)


class CheckpointManager:
    """Manages download checkpoints and duration tier configuration.

    Migrated from VideoDownloader checkpoint methods and __init__.
    """

    def __init__(self, config: 'Config', checkpoint_file: Path, sources_file: Path):
        """
        Initialize CheckpointManager.

        Args:
            config: Config object with duration_tiers
            checkpoint_file: Path to checkpoint JSON file
            sources_file: Path to sources.json file
        """
        self.config = config
        self.checkpoint_file = checkpoint_file
        self.sources_file = sources_file
        self.duration_tiers = self._load_duration_tiers()

    def _load_duration_tiers(self) -> Dict[str, Dict]:
        """
        Convert DurationTiersConfig dataclass to dict format.

        Migrated from downloader.py lines 248-264.

        Maps config.duration_tiers (from YAML) to the dict format expected by
        the rest of the downloader code.

        Returns:
            Dict mapping tier names to {min, max, per_keyword, max_total}
        """
        # Default tier presets
        default_tiers = {
            'short': {'min': 20, 'max': 120, 'per_keyword': 8, 'max_total': 0},
            'medium': {'min': 120, 'max': 600, 'per_keyword': 8, 'max_total': 0},
            'long': {'min': 600, 'max': 1500, 'per_keyword': 5, 'max_total': 0},
            'longer': {'min': 1500, 'max': 3000, 'per_keyword': 1, 'max_total': 1}
        }

        # Check if config has duration_tiers
        if not hasattr(self.config, 'duration_tiers') or not self.config.duration_tiers:
            return default_tiers

        duration_tiers_config = self.config.duration_tiers
        tiers = {}
        for tier_name in ['short', 'medium', 'long', 'longer']:
            tier_config = getattr(duration_tiers_config, tier_name, None)
            if tier_config:
                tiers[tier_name] = {
                    'min': getattr(tier_config, 'min_seconds', 0),
                    'max': getattr(tier_config, 'max_seconds', 120),
                    'per_keyword': getattr(tier_config, 'videos_per_keyword', 5),
                    'max_total': getattr(tier_config, 'max_total', 0)
                }
        return tiers if tiers else default_tiers

    def get_tier_value(self, tier: str, key: str, default: int = 0) -> int:
        """
        Get tier config value, handling both dict and dataclass formats.

        Migrated from downloader.py lines 266-283.

        Args:
            tier: Tier name (short, medium, long, longer)
            key: Config key (min, max, per_keyword, max_total)
            default: Default value if not found

        Returns:
            Config value or default
        """
        tier_config = self.duration_tiers.get(tier, {})

        if isinstance(tier_config, dict):
            value = tier_config.get(key, default)
            return value if value is not None else default
        else:
            # Dataclass format - try different attribute names
            if key == 'min':
                value = getattr(tier_config, 'min_seconds', getattr(tier_config, 'min', default))
            elif key == 'max':
                value = getattr(tier_config, 'max_seconds', getattr(tier_config, 'max', default))
            elif key == 'per_keyword':
                value = getattr(tier_config, 'videos_per_keyword', getattr(tier_config, 'per_keyword', default))
            else:
                value = getattr(tier_config, key, default)
            return value if value is not None else default

    def load_checkpoint(self) -> Optional[DownloadCheckpoint]:
        """
        Load checkpoint for resume.

        Migrated from downloader.py lines 361-370.

        Returns:
            DownloadCheckpoint if found, None otherwise
        """
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return DownloadCheckpoint.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return None

    def save_checkpoint(self, checkpoint: DownloadCheckpoint):
        """
        Save checkpoint to file.

        Migrated from downloader.py lines 372-379.

        Args:
            checkpoint: DownloadCheckpoint to save
        """
        from datetime import datetime

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.timestamp = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

    def clear_checkpoint(self):
        """
        Clear checkpoint after successful completion.

        Migrated from downloader.py lines 381-385.
        """
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

    def load_sources(self) -> List['DownloadedVideo']:
        """
        Load existing sources.json.

        Migrated from downloader.py lines 342-352.

        Returns:
            List of DownloadedVideo objects
        """
        from ..state import DownloadedVideo

        if self.sources_file.exists():
            try:
                with open(self.sources_file, 'r') as f:
                    data = json.load(f)
                    sources = [DownloadedVideo(**v) for v in data]
                logger.info(f"Loaded {len(sources)} existing source records")
                return sources
            except Exception as e:
                logger.warning(f"Could not load sources.json: {e}")
                return []
        return []

    def save_sources(self, sources: List['DownloadedVideo']):
        """
        Save sources.json.

        Migrated from downloader.py lines 354-359.

        Args:
            sources: List of DownloadedVideo objects to save
        """
        from dataclasses import asdict

        self.sources_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.sources_file, 'w') as f:
            json.dump([asdict(s) for s in sources], f, indent=2)
