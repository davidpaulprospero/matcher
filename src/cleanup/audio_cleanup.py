"""Audio file cleanup service."""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

from .file_deleter import FileDeleter

logger = logging.getLogger(__name__)


@dataclass
class AudioCleanupResult:
    """Result of audio cleanup operation."""
    files_deleted: int = 0
    directories_deleted: int = 0
    files_failed: List[Path] = field(default_factory=list)
    cache_entries_cleaned: int = 0
    dry_run: bool = False


class AudioCleanupService:
    """Service for cleaning up audio files after video segment download.

    Responsibilities:
    - Delete audio files listed in state
    - Delete orphaned *_audio directories
    - Update checkpoint to prevent stale references
    - Clean orphaned transcription cache entries

    Design decisions:
    - Checkpoint updated BEFORE file deletion (atomic consistency)
    - Uses set() for deduplication
    - Resolves paths for symlink handling
    - Retries on Windows file locking
    """

    def __init__(
        self,
        file_deleter: FileDeleter,
        checkpoint: 'CheckpointManager',
        config: 'Config',
        dry_run: bool = False
    ):
        """Initialize AudioCleanupService.

        Args:
            file_deleter: FileDeleter instance for low-level file operations
            checkpoint: CheckpointManager for updating checkpoint state
            config: Config object with download and cache settings
            dry_run: If True, only log what would be deleted without actually deleting
        """
        self.file_deleter = file_deleter
        self.checkpoint = checkpoint
        self.config = config
        self.dry_run = dry_run

    def should_cleanup(self) -> bool:
        """Check if cleanup is enabled, with backward compatibility.

        Returns:
            True if delete_audio_after_video is enabled in config
        """
        try:
            audio_config = getattr(self.config.download, 'audio_first', None)
            if audio_config is None:
                return False  # Old config without audio_first
            return bool(getattr(audio_config, 'delete_audio_after_video', False))
        except AttributeError:
            return False  # Config structure doesn't match expected

    def cleanup(self, state: 'PipelineState') -> AudioCleanupResult:
        """Perform audio cleanup with all safety measures.

        Order of operations (for safety):
        1. Collect files to delete
        2. Update checkpoint (if fails, files remain)
        3. Clear state.downloaded_audio
        4. Delete files from state
        5. Delete orphaned directories
        6. Clean transcription cache

        Args:
            state: Pipeline state containing downloaded_audio list

        Returns:
            AudioCleanupResult with counts of deleted items
        """
        result = AudioCleanupResult(dry_run=self.dry_run)

        if not self.should_cleanup():
            return result

        # Even if no audio files in state, still clean orphaned directories
        if not state.downloaded_audio:
            result.directories_deleted = self._delete_orphaned_directories()
            return result

        # 1. Collect files
        files_to_delete = self._collect_files(state)
        if not files_to_delete and not self.dry_run:
            # No files to delete, but still check for orphaned directories
            result.directories_deleted = self._delete_orphaned_directories()
            return result

        # 2. Update checkpoint FIRST (atomic safety)
        if not self.dry_run:
            self._update_checkpoint()

        # 3. Clear state
        if not self.dry_run:
            state.downloaded_audio = []

        # 4. Delete files
        for file_path in files_to_delete:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would delete: {file_path}")
                result.files_deleted += 1
            elif self.file_deleter.delete_file(file_path):
                logger.info(f"Deleted audio file: {file_path.name}")
                result.files_deleted += 1
            else:
                result.files_failed.append(file_path)

        # 5. Delete orphaned directories
        result.directories_deleted = self._delete_orphaned_directories()

        # 6. Clean transcription cache
        result.cache_entries_cleaned = self._clean_transcription_cache()

        if result.files_deleted > 0 or result.directories_deleted > 0:
            print(f"  ✓ Cleaned up {result.files_deleted} audio files, "
                  f"{result.directories_deleted} directories")
            if not self.dry_run:
                logger.warning(
                    "Audio files deleted. Future --match-only runs will require "
                    "re-downloading or set delete_audio_after_video: false"
                )

        return result

    def _collect_files(self, state: 'PipelineState') -> Set[Path]:
        """Collect unique files to delete, resolved and verified.

        Args:
            state: Pipeline state with downloaded_audio list

        Returns:
            Set of resolved Path objects for existing files
        """
        files = set()
        for ad in state.downloaded_audio:
            audio_path = ad.file if hasattr(ad, 'file') else ad.get('file', '')
            if audio_path:
                try:
                    audio_file = Path(audio_path).resolve()
                    if audio_file.exists():
                        files.add(audio_file)
                except (OSError, ValueError) as e:
                    logger.debug(f"Could not resolve path {audio_path}: {e}")
        return files

    def _update_checkpoint(self) -> None:
        """Update DOWNLOAD checkpoint to clear audio_downloads."""
        download_data = self.checkpoint.get_stage_data('DOWNLOAD') or {}
        download_data['audio_downloads'] = []
        download_data['audio_deleted'] = True
        self.checkpoint.save('DOWNLOAD', download_data)

    def _delete_orphaned_directories(self) -> int:
        """Delete all audio-first directories in videos dir.

        Uses stricter pattern *_{s,m,l}_audio to match only known tier suffixes
        (s=short, m=medium, l=long/longer) and avoid matching unrelated dirs.

        Returns:
            Number of directories deleted
        """
        deleted = 0
        video_dir_str = getattr(self.config, 'downloaded_videos_dir', None)
        if not video_dir_str:
            logger.debug("downloaded_videos_dir not set, skipping orphan cleanup")
            return 0

        video_dir = Path(video_dir_str)
        if not video_dir.exists():
            return 0

        # Match only known tier patterns: s, m, l (from 'short', 'medium', 'long'/'longer')
        # Note: 'long' and 'longer' both use 'l' suffix
        deleted_dirs: Set[Path] = set()  # Track to avoid re-processing
        max_dirs = 1000  # Safety limit to prevent runaway iteration

        for tier in ['s', 'm', 'l']:
            try:
                for i, audio_dir in enumerate(video_dir.glob(f'*_{tier}_audio')):
                    if i >= max_dirs:
                        logger.warning(f"Reached max directory limit ({max_dirs}), stopping cleanup")
                        break

                    # Skip if already processed or is a symlink (avoid loops)
                    if audio_dir in deleted_dirs:
                        continue
                    if audio_dir.is_symlink():
                        logger.debug(f"Skipping symlink: {audio_dir}")
                        continue
                    if audio_dir.is_dir():
                        if self.dry_run:
                            logger.info(f"[DRY RUN] Would remove: {audio_dir}")
                            deleted += 1
                            deleted_dirs.add(audio_dir)
                        elif self.file_deleter.delete_directory(audio_dir):
                            logger.info(f"Removed audio directory: {audio_dir.name}")
                            deleted += 1
                            deleted_dirs.add(audio_dir)
            except OSError as e:
                logger.warning(f"Error scanning for {tier} tier directories: {e}")

        return deleted

    def _clean_transcription_cache(self) -> int:
        """Clean orphaned transcription cache entries.

        Returns:
            Number of cache entries cleaned
        """
        if self.dry_run:
            return 0
        try:
            from ..transcription.cache import TranscriptCache
            cache_dir = Path(getattr(self.config, 'cache_dir', '.cache')) / 'transcriptions'
            if not cache_dir.exists():
                return 0
            # Use compress_cache from config if available, default to True (US-110-008)
            compress_cache = getattr(self.config, 'transcription', {}).get('compress_cache', True) if hasattr(self.config, 'transcription') else True
            cache = TranscriptCache(cache_dir, compress_cache=compress_cache)
            return cache.cleanup_orphaned()
        except ImportError:
            logger.debug("TranscriptCache not available, skipping cache cleanup")
            return 0
        except Exception as e:
            logger.debug(f"Transcription cache cleanup failed: {e}")
            return 0
