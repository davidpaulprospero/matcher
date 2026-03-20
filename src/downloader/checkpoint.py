"""
Checkpoint and duration tier management.

Migrated from VideoDownloader.__init__ and checkpoint methods (lines 46-387).
Handles checkpoint save/load for resuming downloads and duration tier configuration.
"""

from __future__ import annotations

import gzip
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
    # Cross-keyword rate limit budget (US-004)
    rate_limit_budget: Optional[Dict] = None
    # VPN manager state for switch count persistence (US-005)
    vpn_manager_state: Optional[Dict] = None
    # MullvadVPN state for rotation limit persistence (US-129-007)
    mullvad_vpn_state: Optional[Dict] = None
    # Escalation manager state for resume support (Sprint 10 US-007)
    escalation_state: Optional[Dict] = None
    # Per-tier backoff state for resume support (Sprint 12 US-003)
    tier_backoff_state: Optional[Dict] = None
    # US-129-002: Retry budget state for per-video retry limits
    retry_budget_state: Optional[Dict] = None
    # US-136-007: Region success tracking state for dynamic region backoff
    region_success_state: Optional[Dict] = None
    # Impersonation manager state for resume support
    impersonation_state: Optional[Dict] = None
    # Schema version for detecting checkpoint format drift (US-58-002)
    schema_version: int = 1

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
        # Handle checkpoints created before rate limit budget was added (US-004)
        if 'rate_limit_budget' not in data:
            data['rate_limit_budget'] = None
        # Handle checkpoints created before VPN manager state was added (US-005)
        if 'vpn_manager_state' not in data:
            data['vpn_manager_state'] = None
        # Handle checkpoints created before MullvadVPN state was added (US-129-007)
        if 'mullvad_vpn_state' not in data:
            data['mullvad_vpn_state'] = None
        # Handle checkpoints created before escalation state was added (Sprint 10 US-007)
        if 'escalation_state' not in data:
            data['escalation_state'] = None
        # Handle checkpoints created before tier backoff state was added (Sprint 12 US-003)
        if 'tier_backoff_state' not in data:
            data['tier_backoff_state'] = None
        # Handle checkpoints created before retry budget state was added (US-129-002)
        if 'retry_budget_state' not in data:
            data['retry_budget_state'] = None
        # Handle checkpoints created before region success state was added (US-136-007)
        if 'region_success_state' not in data:
            data['region_success_state'] = None
        # Handle checkpoints created before impersonation state was added
        if 'impersonation_state' not in data:
            data['impersonation_state'] = None
        # Handle checkpoints created before schema_version was added (US-58-002)
        if 'schema_version' not in data:
            data['schema_version'] = 1
        # Warn if loaded checkpoint has a different schema version than current
        current_version = 1
        if data['schema_version'] != current_version:
            logger.warning(
                "Checkpoint schema_version mismatch: loaded v%d, current v%d. "
                "Checkpoint may contain unexpected fields or missing defaults.",
                data['schema_version'], current_version
            )
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
        # US-114-009: Extract compression settings
        self._compression_enabled = True
        self._compression_level = 6
        if hasattr(config, 'checkpoint_compression') and config.checkpoint_compression:
            self._compression_enabled = config.checkpoint_compression.enabled
            self._compression_level = config.checkpoint_compression.compression_level

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
        US-114-009: Added gzip decompression support for backward compatibility.

        Returns:
            DownloadCheckpoint if found, None otherwise
        """
        if self.checkpoint_file.exists():
            try:
                # US-114-009: Try gzip first if compression enabled, then try uncompressed
                # This ensures backward compatibility with uncompressed checkpoints
                data = None
                if self._compression_enabled:
                    # Try compressed first
                    try:
                        with gzip.open(self.checkpoint_file, 'rt', encoding='utf-8') as f:
                            data = json.load(f)
                        logger.debug(f"Loaded compressed checkpoint from {self.checkpoint_file}")
                    except gzip.BadGzipFile:
                        # Not compressed, try plain JSON
                        pass
                # Fall back to uncompressed if not compressed or compression disabled
                if data is None:
                    with open(self.checkpoint_file, 'r') as f:
                        data = json.load(f)
                    logger.debug(f"Loaded uncompressed checkpoint from {self.checkpoint_file}")
                return DownloadCheckpoint.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return None

    def save_checkpoint(self, checkpoint: DownloadCheckpoint):
        """
        Save checkpoint to file.

        Migrated from downloader.py lines 372-379.
        US-114-009: Added gzip compression support.

        Args:
            checkpoint: DownloadCheckpoint to save
        """
        from datetime import datetime

        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.timestamp = datetime.now().isoformat()
        data = json.dumps(checkpoint.to_dict(), indent=2)

        # US-114-009: Compress if enabled
        if self._compression_enabled:
            with gzip.open(self.checkpoint_file, 'wt', encoding='utf-8', compresslevel=self._compression_level) as f:
                f.write(data)
            logger.debug(f"Saved compressed checkpoint to {self.checkpoint_file} (level={self._compression_level})")
        else:
            with open(self.checkpoint_file, 'w') as f:
                f.write(data)
            logger.debug(f"Saved uncompressed checkpoint to {self.checkpoint_file}")

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
        US-114-009: Added gzip decompression support for backward compatibility.

        Returns:
            List of DownloadedVideo objects
        """
        from ..state import DownloadedVideo

        if self.sources_file.exists():
            try:
                # US-114-009: Try gzip first if compression enabled, then try uncompressed
                data = None
                if self._compression_enabled:
                    try:
                        with gzip.open(self.sources_file, 'rt', encoding='utf-8') as f:
                            data = json.load(f)
                        logger.debug(f"Loaded compressed sources from {self.sources_file}")
                    except gzip.BadGzipFile:
                        pass
                if data is None:
                    with open(self.sources_file, 'r') as f:
                        data = json.load(f)
                    logger.debug(f"Loaded uncompressed sources from {self.sources_file}")
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
        US-114-009: Added gzip compression support.

        Args:
            sources: List of DownloadedVideo objects to save
        """
        from dataclasses import asdict

        self.sources_file.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps([asdict(s) for s in sources], indent=2)

        # US-114-009: Compress if enabled
        if self._compression_enabled:
            with gzip.open(self.sources_file, 'wt', encoding='utf-8', compresslevel=self._compression_level) as f:
                f.write(data)
            logger.debug(f"Saved compressed sources to {self.sources_file}")
        else:
            with open(self.sources_file, 'w') as f:
                f.write(data)
            logger.debug(f"Saved uncompressed sources to {self.sources_file}")
