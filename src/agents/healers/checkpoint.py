"""
Checkpoint Healer - Checkpoint corruption and recovery.

Handles:
- JSON parse errors in checkpoint.json
- Missing or corrupted checkpoint data
- Config hash mismatches
- Backup restoration
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..base import Healer, HealerResult, HealerAction

if TYPE_CHECKING:
    from ...config import Config
    from ...state import PipelineState

logger = logging.getLogger(__name__)


class CheckpointHealer(Healer):
    """
    Heals checkpoint-related errors.

    Recovery strategies:
    1. JSON parse error: Restore from backup
    2. Missing fields: Rebuild from cache
    3. Hash mismatch: Offer to continue or restart
    4. Corrupted backup: Start fresh with warning
    """

    name = "checkpoint-healer"
    description = "Recover from checkpoint corruption"

    error_patterns = [
        "json",
        "decode",
        "checkpoint",
        "parse",
        "corrupt",
        "invalid",
        "missing",
        "hash",
    ]

    exception_types = [
        json.JSONDecodeError,
    ]

    CHECKPOINT_FILE = "checkpoint.json"
    BACKUP_FILE = "checkpoint.backup.json"

    def fix(
        self,
        error: Exception,
        state: 'PipelineState',
        stage_name: str
    ) -> HealerResult:
        """Attempt to fix checkpoint-related errors."""
        error_str = str(error).lower()

        # JSON decode error - try backup
        if isinstance(error, json.JSONDecodeError) or "json" in error_str or "decode" in error_str:
            return self._restore_from_backup(error, state)

        # Corrupt or invalid checkpoint
        if "corrupt" in error_str or "invalid" in error_str:
            return self._restore_from_backup(error, state)

        # Missing fields - try to rebuild
        if "missing" in error_str or "keyerror" in error_str:
            return self._rebuild_checkpoint(error, state)

        # Hash mismatch (config changed)
        if "hash" in error_str or "mismatch" in error_str:
            return self._handle_hash_mismatch(error, state)

        # Generic checkpoint error
        return self._restore_from_backup(error, state)

    def _restore_from_backup(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Restore checkpoint from backup file."""
        self.log_attempt("Attempting to restore from checkpoint backup...")

        checkpoint_path = Path(self.project_dir) / self.CHECKPOINT_FILE
        backup_path = Path(self.project_dir) / self.BACKUP_FILE

        if not backup_path.exists():
            self.log_failure("No backup checkpoint found")
            return self._start_fresh(error, state)

        try:
            # Validate backup is valid JSON
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            # Backup is valid - restore it
            shutil.copy(backup_path, checkpoint_path)

            self.log_success("Restored checkpoint from backup")
            return HealerResult.fixed(
                "Restored checkpoint from backup file",
                action=HealerAction.RESTORE,
                backup_stages=list(backup_data.get('stages', {}).keys())
            )

        except json.JSONDecodeError:
            self.log_failure("Backup checkpoint is also corrupted")
            return self._start_fresh(error, state)

        except Exception as e:
            self.log_failure(f"Failed to restore backup: {e}")
            return self._start_fresh(error, state)

    def _rebuild_checkpoint(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Try to rebuild checkpoint from cached data."""
        self.log_attempt("Attempting to rebuild checkpoint from cache...")

        cache_dir = Path(self.project_dir) / ".cache"
        if not cache_dir.exists():
            return self._start_fresh(error, state)

        # Look for cached stage data matching the current 7-stage pipeline
        rebuilt_stages = []

        # Check for captions cache (CAPTION stage)
        caption_cache = cache_dir / "captions"
        if caption_cache.exists() and any(caption_cache.iterdir()):
            rebuilt_stages.append("CAPTION")

        # Check for LLM response cache (used by ANALYZE, MATCH stages)
        llm_cache = cache_dir / "llm_responses"
        if llm_cache.exists() and any(llm_cache.iterdir()):
            rebuilt_stages.append("ANALYZE")

        # Check for embeddings cache (used by MATCH, not a stage itself)
        embed_cache = cache_dir / "embeddings"
        if embed_cache.exists() and any(embed_cache.iterdir()):
            rebuilt_stages.append("MATCH")

        if rebuilt_stages:
            self.log_success(f"Found cached data for stages: {rebuilt_stages}")
            # Note: actual rebuild would need the pipeline to re-load caches
            return HealerResult.fixed(
                f"Cache data available for: {', '.join(rebuilt_stages)}. Partial resume possible.",
                action=HealerAction.RETRY,
                cached_stages=rebuilt_stages
            )

        return self._start_fresh(error, state)

    def _handle_hash_mismatch(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Handle config hash mismatch."""
        self.log_attempt("Config changed since last run...")

        # Could prompt user, but for auto-healing we continue with current config
        self.log_success("Continuing with current config (ignoring hash mismatch)")

        return HealerResult.fixed(
            "Config changed since checkpoint was created. Continuing with current config.",
            action=HealerAction.RETRY,
            hash_mismatch_ignored=True
        )

    def _start_fresh(self, error: Exception, state: 'PipelineState') -> HealerResult:
        """Start fresh by removing corrupted checkpoint."""
        self.log_attempt("Starting fresh (removing corrupted checkpoint)...")

        checkpoint_path = Path(self.project_dir) / self.CHECKPOINT_FILE

        try:
            if checkpoint_path.exists():
                # Keep a copy for debugging
                corrupt_path = checkpoint_path.with_suffix('.corrupted.json')
                shutil.move(checkpoint_path, corrupt_path)
                self.log_attempt(f"Moved corrupted checkpoint to {corrupt_path.name}")

            self.log_success("Ready to start fresh")
            return HealerResult.fixed(
                "Checkpoint was corrupted. Starting fresh run.",
                action=HealerAction.RETRY,
                fresh_start=True
            )

        except Exception as e:
            self.log_failure(f"Could not remove corrupted checkpoint: {e}")
            return HealerResult.failed(f"Cannot recover checkpoint: {e}")

    def create_backup(self) -> bool:
        """Create a backup of the current checkpoint."""
        checkpoint_path = Path(self.project_dir) / self.CHECKPOINT_FILE
        backup_path = Path(self.project_dir) / self.BACKUP_FILE

        if not checkpoint_path.exists():
            return False

        try:
            shutil.copy(checkpoint_path, backup_path)
            return True
        except Exception:
            return False
