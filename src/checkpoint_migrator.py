"""
Checkpoint Migrator for Versioned Checkpoint Upgrades

Extracts migration logic from CheckpointManager into a dedicated class
using the Strategy pattern: one function per version transition.

To add a future migration (e.g., v2.0 -> v3.0):
  1. Add a migration function: _migrate_2_0_to_3_0(self, data: dict) -> dict
  2. Register it in MIGRATIONS: ("2.0", "3.0"): _migrate_2_0_to_3_0
  3. Update CURRENT_CHECKPOINT_VERSION in checkpoint.py to "3.0"
  That's it — the migrate() method chains migrations automatically.
"""

import re
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Tuple

logger = logging.getLogger(__name__)


class CheckpointMigrator:
    """Handles versioned checkpoint data migrations.

    Each migration is a function that transforms a raw dict from one version
    to the next.  The migrate() method chains them so that a v0.9 checkpoint
    can reach v2.0 in one call (0.9 -> 1.0 -> 2.0).

    To add a v2.0 -> v3.0 migration, just add one function and register it
    in MIGRATIONS.  See module docstring for step-by-step instructions.
    """

    # Stage order (duplicated here to avoid circular import with checkpoint.py)
    STAGE_ORDER = [
        "ANALYZE", "VIDEO_SEARCH", "CAPTION", "MATCH",
        "ITERATIVE_MATCH", "DOWNLOAD_SEGMENTS", "OUTPUT",
    ]

    # Map old stages to their nearest new equivalent for last_completed_stage
    STAGE_REMAP = {
        'DOWNLOAD': 'VIDEO_SEARCH',
        'STOCK': 'VIDEO_SEARCH',
        'BROLL_DOWNLOAD': 'VIDEO_SEARCH',
        'REMIX': 'CAPTION',
        'TRANSCRIBE': 'CAPTION',
        'SCENE_DETECTION': 'MATCH',
        'BROLL_MATCH': 'MATCH',
    }

    def __init__(self):
        # Registry: (from_version, to_version) -> migration function
        self._migrations: Dict[Tuple[str, str], Callable[[dict], dict]] = {
            ("0.9", "1.0"): self._migrate_0_9_to_1_0,
            ("1.0", "2.0"): self._migrate_1_0_to_2_0,
        }
        # Ordered version chain for chaining migrations
        self._version_chain = ["0.9", "1.0", "2.0"]

    def needs_migration(self, data: dict, target_version: str) -> bool:
        """Check if a checkpoint dict needs migration to reach target_version."""
        current = data.get('version', '0.9')
        return current != target_version

    def migrate(self, data: dict, from_version: str, to_version: str) -> dict:
        """Migrate checkpoint data from from_version to to_version.

        Chains intermediate migrations automatically.
        E.g. migrate(data, "0.9", "2.0") runs 0.9->1.0 then 1.0->2.0.

        Args:
            data: Raw checkpoint dict (not a CheckpointData instance).
            from_version: Current version of the data.
            to_version: Target version to migrate to.

        Returns:
            Migrated dict with version set to to_version.

        Raises:
            ValueError: If no migration path exists between versions.
        """
        if from_version == to_version:
            return data

        # Find migration path
        try:
            start_idx = self._version_chain.index(from_version)
            end_idx = self._version_chain.index(to_version)
        except ValueError as e:
            raise ValueError(
                f"Unknown version in migration path: {e}. "
                f"Known versions: {self._version_chain}"
            )

        if start_idx >= end_idx:
            raise ValueError(
                f"Cannot migrate backwards: {from_version} -> {to_version}"
            )

        # Chain migrations
        result = dict(data)  # shallow copy to avoid mutating input
        for i in range(start_idx, end_idx):
            v_from = self._version_chain[i]
            v_to = self._version_chain[i + 1]
            migration_fn = self._migrations.get((v_from, v_to))
            if migration_fn is None:
                raise ValueError(
                    f"No migration registered for {v_from} -> {v_to}"
                )
            logger.info(f"Running checkpoint migration: v{v_from} -> v{v_to}")
            result = migration_fn(result)

        return result

    def _migrate_0_9_to_1_0(self, data: dict) -> dict:
        """Migrate v0.9 -> v1.0: Normalize stage keys to lowercase."""
        result = dict(data)
        result['version'] = '1.0'

        # Uppercase stage keys -> lowercase
        for key in list(result.keys()):
            if key.isupper() and key.lower() != key:
                lower_key = key.lower()
                if lower_key not in result:
                    result[lower_key] = result[key]
                del result[key]

        return result

    def _migrate_1_0_to_2_0(self, data: dict) -> dict:
        """Migrate v1.0 -> v2.0: 13-stage to 7-stage simplified pipeline."""
        migrated = {
            'version': '2.0',
            'created_at': data.get('created_at', datetime.now().isoformat()),
            'updated_at': data.get('updated_at', datetime.now().isoformat()),
            'config_hash': data.get('config_hash', ''),
            'voiceover_path': data.get('voiceover_path', ''),
            'voiceover_hash': data.get('voiceover_hash', ''),
        }

        # Migrate last_completed_stage
        old_stage = data.get('last_completed_stage', '')
        if old_stage in self.STAGE_REMAP:
            migrated['last_completed_stage'] = self.STAGE_REMAP[old_stage]
            logger.info(f"Remapped stage {old_stage} -> {self.STAGE_REMAP[old_stage]}")
        elif old_stage in self.STAGE_ORDER:
            migrated['last_completed_stage'] = old_stage
        else:
            migrated['last_completed_stage'] = ''
            if old_stage:
                logger.warning(f"Unknown stage '{old_stage}' in old checkpoint, resetting")

        # Copy existing stage data for stages that still exist
        for stage_key in ['analyze', 'caption', 'match', 'iterative_match', 'download_segments']:
            stage_data = data.get(stage_key.upper()) or data.get(stage_key) or {}
            migrated[stage_key] = stage_data

        # Migrate DOWNLOAD -> video_search (extract video IDs)
        download_data = data.get('DOWNLOAD') or data.get('download') or {}
        if download_data:
            video_ids = []
            for vid in download_data.get('downloaded_videos', []):
                if isinstance(vid, dict):
                    url = vid.get('url', '')
                    if 'youtube.com' in url or 'youtu.be' in url:
                        match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url)
                        if match:
                            video_ids.append(match.group(1))
            migrated['video_search'] = {
                'video_ids': video_ids,
                'migrated_from_download': True,
            }

        # Preserve fields that exist in both versions
        for field in ['chapter_data', 'stage_metrics', 'transcription_metrics']:
            if field in data:
                migrated[field] = data[field]

        return migrated
