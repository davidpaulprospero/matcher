"""
Checkpoint metadata indexing for fast queries without loading full checkpoint data.

US-130-009: Add checkpoint metadata indexing
- Creates checkpoint.index file with metadata entries
- Index tracks: timestamp, stage, size, config_hash, voiceover_hash
- Provides query_by_date_range(start, end) and query_by_stage(stage_name) methods
- Index updated atomically with checkpoint saves
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

INDEX_FILENAME = "checkpoint.index"


@dataclass
class IndexEntry:
    """Single metadata entry in the checkpoint index."""
    timestamp: str  # ISO format
    stage: str
    size: int  # bytes
    config_hash: str
    voiceover_hash: str
    checkpoint_file: str = "checkpoint.json"  # Relative path to checkpoint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "size": self.size,
            "config_hash": self.config_hash,
            "voiceover_hash": self.voiceover_hash,
            "checkpoint_file": self.checkpoint_file,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IndexEntry:
        return cls(
            timestamp=data.get("timestamp", ""),
            stage=data.get("stage", ""),
            size=data.get("size", 0),
            config_hash=data.get("config_hash", ""),
            voiceover_hash=data.get("voiceover_hash", ""),
            checkpoint_file=data.get("checkpoint_file", "checkpoint.json"),
        )


@dataclass
class CheckpointIndex:
    """Index for checkpoint metadata - enables fast queries without loading full checkpoint."""

    project_dir: Path
    entries: List[IndexEntry] = field(default_factory=list)
    _lock_file: Optional[Path] = None

    @property
    def index_path(self) -> Path:
        return self.project_dir / INDEX_FILENAME

    @property
    def lock_path(self) -> Path:
        return self.project_dir / f"{INDEX_FILENAME}.lock"

    def load(self) -> None:
        """Load index from disk."""
        if not self.index_path.exists():
            self.entries = []
            return

        try:
            with open(self.index_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            entries_list = data.get("entries", [])
            self.entries = [IndexEntry.from_dict(e) for e in entries_list]
            logger.debug(f"Loaded {len(self.entries)} index entries from {self.index_path}")

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt index file, starting fresh: {e}")
            self.entries = []
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            self.entries = []

    def _acquire_lock(self) -> bool:
        """Acquire exclusive lock for atomic writes."""
        self._lock_file = self.lock_path

        try:
            # Use O_CREAT | O_EXCL for atomic create-if-not-exists
            # O_BINARY is Windows-only, use 0 for other platforms
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if sys.platform == 'win32':
                flags |= os.O_BINARY

            fd = os.open(str(self._lock_file), flags)
            os.close(fd)
            return True
        except FileExistsError:
            return False

    def _release_lock(self) -> None:
        """Release exclusive lock."""
        if self._lock_file and self._lock_file.exists():
            try:
                self._lock_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to release index lock: {e}")
            finally:
                self._lock_file = None

    def _atomic_save(self) -> None:
        """Atomically write index to disk using temp file + rename."""
        if not self.entries:
            # Empty index - remove the file
            if self.index_path.exists():
                try:
                    self.index_path.unlink()
                except Exception:
                    pass
            return

        # Create temp file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=str(self.project_dir),
            prefix='.checkpoint.index.',
            suffix='.tmp'
        )

        try:
            data = {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "entries": [e.to_dict() for e in self.entries]
            }

            with os.fdopen(temp_fd, 'wb') as f:
                f.write(json.dumps(data, indent=2).encode('utf-8'))

            # Atomic rename
            os.replace(temp_path, self.index_path)
            logger.debug(f"Saved {len(self.entries)} entries to index")

        except Exception as e:
            # Clean up temp file on failure
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise e

    def add_entry(
        self,
        stage: str,
        size: int,
        config_hash: str,
        voiceover_hash: str,
        checkpoint_file: str = "checkpoint.json"
    ) -> None:
        """Add a new index entry (does not save to disk)."""
        entry = IndexEntry(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            size=size,
            config_hash=config_hash,
            voiceover_hash=voiceover_hash,
            checkpoint_file=checkpoint_file,
        )
        self.entries.append(entry)

    def add_entry_and_save(
        self,
        stage: str,
        size: int,
        config_hash: str,
        voiceover_hash: str,
        checkpoint_file: str = "checkpoint.json"
    ) -> None:
        """Add a new index entry and atomically save to disk."""
        # Acquire lock
        if not self._acquire_lock():
            logger.warning("Could not acquire index lock - skipping index update")
            return

        try:
            # Load current entries
            self.load()

            # Add new entry
            self.add_entry(stage, size, config_hash, voiceover_hash, checkpoint_file)

            # Atomic save
            self._atomic_save()

        finally:
            self._release_lock()

    def query_by_date_range(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None
    ) -> List[IndexEntry]:
        """Query entries within a date range.

        Args:
            start: Start datetime (inclusive). If None, no lower bound.
            end: End datetime (inclusive). If None, no upper bound.

        Returns:
            List of matching index entries
        """
        if not self.entries:
            return []

        results = []
        for entry in self.entries:
            try:
                entry_dt = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                continue

            if start and entry_dt < start:
                continue
            if end and entry_dt > end:
                continue

            results.append(entry)

        return results

    def query_by_stage(self, stage_name: str) -> List[IndexEntry]:
        """Query entries by stage name.

        Args:
            stage_name: Stage name to filter by (exact match)

        Returns:
            List of matching index entries
        """
        if not self.entries:
            return []

        return [e for e in self.entries if e.stage == stage_name]

    def query_by_config_hash(self, config_hash: str) -> List[IndexEntry]:
        """Query entries by config hash.

        Args:
            config_hash: Config hash to filter by

        Returns:
            List of matching index entries
        """
        if not self.entries:
            return []

        return [e for e in self.entries if e.config_hash == config_hash]

    def get_latest(self) -> Optional[IndexEntry]:
        """Get the most recent index entry."""
        if not self.entries:
            return None
        return self.entries[-1]

    def get_all(self) -> List[IndexEntry]:
        """Get all index entries."""
        return self.entries.copy()

    def clear(self) -> None:
        """Clear all index entries (does not save to disk)."""
        self.entries = []

    def clear_and_save(self) -> None:
        """Clear all index entries and save to disk."""
        if not self._acquire_lock():
            logger.warning("Could not acquire index lock - skipping index clear")
            return

        try:
            self.clear()
            self._atomic_save()
        finally:
            self._release_lock()


def get_or_create_index(project_dir: Path) -> CheckpointIndex:
    """Get or create a checkpoint index for the given project directory."""
    index = CheckpointIndex(project_dir=project_dir)
    index.load()
    return index
