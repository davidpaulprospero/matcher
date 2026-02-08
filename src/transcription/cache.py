"""
Transcript caching with multi-strategy lookup.

Provides caching for transcription results with multiple lookup strategies:
- Source file path matching
- Video ID matching (for segment files)
- Hash-based matching (path + size)

NOTE: Does NOT use BaseCache due to complex multi-strategy lookup
and dynamic index building by scanning cache directory.
"""

import json
import hashlib
import logging
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional

from src.utils import normalize_path
from .utils import extract_video_id

logger = logging.getLogger(__name__)


class TranscriptCache:
    """
    Multi-strategy transcript cache.

    Lookup strategies:
    1. Source file path (most reliable)
    2. Video ID matching (for segment files in audio-first mode)
    3. Hash-based (path + size fallback)
    """

    def __init__(self, cache_dir: str):
        """
        Initialize transcript cache.

        Args:
            cache_dir: Base cache directory (will use transcriptions/ subdirectory)
        """
        base_dir = Path(cache_dir)

        # Check both possible folder names
        self.cache_dir = base_dir / "transcriptions"
        self.alt_cache_dir = base_dir / "transcripts"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._source_map: Dict[str, Path] = {}
        self._video_id_map: Dict[str, Path] = {}
        self._legacy_migrated: bool = False

        logger.debug(f"Primary cache: {self.cache_dir}")
        logger.debug(f"Alt cache: {self.alt_cache_dir}")

        # Warn about and migrate legacy cache directory
        self._check_legacy_cache()

        # Build reverse lookup by reading source_file from each cache file
        self._build_source_map()

    def _check_legacy_cache(self):
        """Check for legacy transcripts/ directory and warn + migrate if it has files."""
        if not self.alt_cache_dir.exists():
            return

        legacy_files = list(self.alt_cache_dir.glob("*.json"))
        if not legacy_files:
            return

        logger.warning(
            f"Found {len(legacy_files)} cached file(s) in legacy directory '{self.alt_cache_dir}'. "
            f"Move files from transcripts/ to transcriptions/ and delete transcripts/"
        )
        self._migrate_legacy_cache()

    def _migrate_legacy_cache(self):
        """Copy entries from legacy transcripts/ to transcriptions/ (once per session)."""
        if self._legacy_migrated:
            return

        self._legacy_migrated = True

        if not self.alt_cache_dir.exists():
            return

        migrated = 0
        for cache_file in self.alt_cache_dir.glob("*.json"):
            dest = self.cache_dir / cache_file.name
            if not dest.exists():
                try:
                    shutil.copy2(cache_file, dest)
                    migrated += 1
                except (OSError, PermissionError) as e:
                    logger.debug(f"Could not migrate {cache_file.name}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected {type(e).__name__} migrating {cache_file.name}: {e}")

        if migrated > 0:
            logger.info(f"Migrated {migrated} transcript cache file(s) from transcripts/ to transcriptions/")

    def _build_source_map(self):
        """
        Build mapping from video path to cache file.

        Reads source_file from each cached transcript to build index.
        """
        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue

            cache_files = list(folder.glob("*.json"))
            logger.debug(f"Scanning {len(cache_files)} cache files in {folder.name}...")

            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract source_file from the data
                    source_file = None

                    if isinstance(data, list) and len(data) > 0:
                        # List of segments - get source_file from first segment
                        source_file = data[0].get('source_file', '')
                    elif isinstance(data, dict):
                        # Dict format - check various keys
                        source_file = data.get('source_file', data.get('video', data.get('video_path', '')))
                        if not source_file and 'segments' in data:
                            segs = data['segments']
                            if segs and len(segs) > 0:
                                source_file = segs[0].get('source_file', '')

                    if source_file:
                        # Normalize path for matching (consistent across platforms)
                        normalized = normalize_path(source_file)
                        self._source_map[normalized] = cache_file

                        # Also add just the filename as key for partial matching
                        filename = Path(source_file).name.lower()
                        self._source_map[filename] = cache_file

                        # Extract video ID for segment matching (audio-first mode support)
                        video_id = extract_video_id(filename)
                        if video_id and video_id not in self._video_id_map:
                            self._video_id_map[video_id] = cache_file

                except (json.JSONDecodeError, OSError, PermissionError, KeyError, TypeError):
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected {type(e).__name__} reading cache file {cache_file.name}: {e}")
                    continue

        logger.debug(f"Built source map with {len(self._source_map)} entries, {len(self._video_id_map)} video IDs")

    def _get_video_hash(self, video_path: str) -> str:
        """
        Get hash for a video file based on path and size.

        Args:
            video_path: Path to video file

        Returns:
            MD5 hash string
        """
        path = Path(video_path)
        size = path.stat().st_size if path.exists() else 0
        key = f"{path.name}:{size}"
        return hashlib.md5(key.encode()).hexdigest()

    def validate_entry(self, segments: List[dict], video_id: str = "") -> Optional[str]:
        """
        Validate a cache entry's segments for integrity.

        Args:
            segments: List of segment dicts with 'start', 'end', 'text'
            video_id: Optional identifier for logging

        Returns:
            None if valid, or a string describing the validation failure
        """
        if not segments:
            return "empty segments list"

        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                return f"segment {i} is not a dict"
            text = seg.get('text', '')
            if not text or not str(text).strip():
                return f"segment {i} has empty text"
            start = seg.get('start', seg.get('start_time'))
            end = seg.get('end', seg.get('end_time'))
            if start is None or end is None:
                return f"segment {i} missing start_time or end_time"
            try:
                start_f = float(start)
                end_f = float(end)
            except (TypeError, ValueError):
                return f"segment {i} has non-numeric start/end times"
            if end_f < start_f:
                return f"segment {i} has end_time ({end_f}) < start_time ({start_f})"

        return None

    def get(self, video_path: str) -> Optional[List[dict]]:
        """
        Get cached transcript for a video.

        Tries multiple strategies:
        1. Source path matching (normalized)
        2. Filename matching
        3. Video ID matching (for segments)
        4. Hash-based matching (fallback)

        Args:
            video_path: Path to video file

        Returns:
            List of segment dicts with 'start', 'end', 'text', or None if not cached
        """
        # Normalize path for consistent matching
        normalized_path = normalize_path(video_path)
        video_name = Path(video_path).name.lower()

        # Try source map lookup first (most reliable)
        cache_file = None
        if normalized_path in self._source_map:
            cache_file = self._source_map[normalized_path]
        elif video_name in self._source_map:
            cache_file = self._source_map[video_name]

        # Fallback to video ID lookup (for segment files matching audio transcripts)
        if not cache_file:
            video_id = extract_video_id(video_name)
            if video_id and video_id in self._video_id_map:
                cache_file = self._video_id_map[video_id]

        # Fallback to hash-based lookup
        if not cache_file:
            video_hash = self._get_video_hash(video_path)
            for folder in [self.cache_dir, self.alt_cache_dir]:
                if folder and folder.exists():
                    exact = folder / f"{video_hash}.json"
                    if exact.exists():
                        cache_file = exact
                        break

        if not cache_file or not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Handle different cache formats
            if isinstance(data, list):
                segments = data
            elif isinstance(data, dict):
                segments = data.get('segments', data.get('transcripts', []))
            else:
                return None

            # Normalize segment format
            normalized = []
            for seg in segments:
                if isinstance(seg, dict):
                    normalized.append({
                        'start': seg.get('start', seg.get('start_time', 0)),
                        'end': seg.get('end', seg.get('end_time', 0)),
                        'text': seg.get('text', '')
                    })

            if not normalized:
                return None

            # Validate entry integrity
            reason = self.validate_entry(normalized, video_id=Path(video_path).name)
            if reason:
                logger.warning(f"Discarding invalid cache entry for {Path(video_path).name}: {reason}")
                return None

            return normalized

        except json.JSONDecodeError as e:
            logger.warning(f"Corrupt cache file {cache_file}: {e}")
            return None
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.debug(f"Cache read error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected {type(e).__name__} reading cache {cache_file}: {e}")
            return None

    def set(self, video_path: str, segments: List[dict]):
        """
        Cache transcript for a video.

        Args:
            video_path: Path to video file
            segments: List of segment dicts with 'start', 'end', 'text'
        """
        video_hash = self._get_video_hash(video_path)
        cache_file = self.cache_dir / f"{video_hash}.json"

        # Convert to format matching existing cache
        cache_data = []
        for i, seg in enumerate(segments):
            cache_data.append({
                'index': i + 1,
                'start_time': seg.get('start', seg.get('start_time', 0)),
                'end_time': seg.get('end', seg.get('end_time', 0)),
                'text': seg.get('text', ''),
                'source_file': str(video_path)
            })

        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)

            # Update source map
            video_path_resolved = str(Path(video_path).resolve())
            self._source_map[video_path_resolved] = cache_file
            self._source_map[Path(video_path).name] = cache_file

        except (OSError, PermissionError) as e:
            logger.debug(f"Could not cache transcript: {e}")
        except Exception as e:
            logger.warning(f"Unexpected {type(e).__name__} caching transcript: {e}")

    def cleanup_orphaned(self) -> int:
        """
        Remove cache entries for videos that no longer exist.

        Returns:
            Number of entries removed
        """
        removed = 0

        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue

            for cache_file in list(folder.glob("*.json")):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Get source file from cache data
                    source_file = None
                    if isinstance(data, list) and len(data) > 0:
                        source_file = data[0].get('source_file', '')
                    elif isinstance(data, dict):
                        source_file = data.get('source_file', '')
                        if not source_file and 'segments' in data:
                            segs = data['segments']
                            if segs and len(segs) > 0:
                                source_file = segs[0].get('source_file', '')

                    # Check if source file exists
                    if source_file and not Path(source_file).exists():
                        cache_file.unlink()
                        removed += 1

                except (json.JSONDecodeError, OSError, PermissionError, FileNotFoundError):
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected {type(e).__name__} during orphan cleanup of {cache_file.name}: {e}")
                    continue

        if removed > 0:
            logger.info(f"Cleaned up {removed} orphaned transcript cache entries")
            # Rebuild source map after cleanup
            self._source_map.clear()
            self._video_id_map.clear()
            self._build_source_map()

        return removed

    def cleanup_stale_entries(self, max_age_days: int = 30) -> int:
        """
        Remove cache entries older than max_age_days.

        Args:
            max_age_days: Maximum age in days before an entry is considered stale

        Returns:
            Number of entries removed
        """
        removed = 0
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)

        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue

            for cache_file in list(folder.glob("*.json")):
                try:
                    # Check file modification time
                    mtime = cache_file.stat().st_mtime
                    if mtime < cutoff_time:
                        cache_file.unlink()
                        removed += 1
                except (OSError, FileNotFoundError):
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected {type(e).__name__} during stale cleanup of {cache_file.name}: {e}")
                    continue

        if removed > 0:
            logger.info(f"Cleaned up {removed} stale transcript cache entries (older than {max_age_days} days)")
            # Rebuild source map after cleanup
            self._source_map.clear()
            self._video_id_map.clear()
            self._build_source_map()

        return removed

    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dict with total_entries, total_size_mb, cache_dir
        """
        total_entries = 0
        total_size = 0

        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue

            for cache_file in folder.glob("*.json"):
                total_entries += 1
                try:
                    total_size += cache_file.stat().st_size
                except OSError:
                    pass

        return {
            "total_entries": total_entries,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "source_map_size": len(self._source_map),
            "video_id_map_size": len(self._video_id_map)
        }

    def warmup_from_project(self, project_dir: str) -> int:
        """
        Import transcript cache entries from a project directory into global cache.

        Scans the project's .cache/transcriptions folder and imports entries
        that don't already exist in the global cache, enabling cross-project
        transcript reuse.

        Args:
            project_dir: Path to project directory containing .cache/transcriptions

        Returns:
            Number of entries imported (excluding duplicates)
        """
        project_path = Path(project_dir)
        project_cache_dir = project_path / ".cache" / "transcriptions"
        alt_project_cache_dir = project_path / ".cache" / "transcripts"

        imported_count = 0
        skipped_duplicates = 0
        skipped_invalid = 0

        # Collect existing source files to avoid duplicates
        existing_sources = set(self._source_map.keys())

        for source_cache_dir in [project_cache_dir, alt_project_cache_dir]:
            if not source_cache_dir.exists():
                continue

            cache_files = list(source_cache_dir.glob("*.json"))
            logger.debug(f"Scanning {len(cache_files)} cache files in {source_cache_dir}")

            for cache_file in cache_files:
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Extract segments for validation
                    if isinstance(data, list):
                        raw_segments = data
                    elif isinstance(data, dict):
                        raw_segments = data.get('segments', data.get('transcripts', []))
                    else:
                        raw_segments = []

                    # Validate entry integrity before importing
                    reason = self.validate_entry(raw_segments, video_id=cache_file.stem)
                    if reason:
                        skipped_invalid += 1
                        continue

                    # Extract source_file from the data
                    source_file = None

                    if isinstance(data, list) and len(data) > 0:
                        source_file = data[0].get('source_file', '')
                    elif isinstance(data, dict):
                        source_file = data.get('source_file', data.get('video', data.get('video_path', '')))
                        if not source_file and 'segments' in data:
                            segs = data['segments']
                            if segs and len(segs) > 0:
                                source_file = segs[0].get('source_file', '')

                    if not source_file:
                        continue

                    # Check for duplicates using normalized path and filename
                    normalized = normalize_path(source_file)
                    filename = Path(source_file).name.lower()

                    if normalized in existing_sources or filename in existing_sources:
                        skipped_duplicates += 1
                        continue

                    # Copy to global cache
                    dest_file = self.cache_dir / cache_file.name
                    if dest_file.exists():
                        skipped_duplicates += 1
                        continue

                    # Write to global cache
                    with open(dest_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)

                    # Update source map
                    self._source_map[normalized] = dest_file
                    self._source_map[filename] = dest_file
                    existing_sources.add(normalized)
                    existing_sources.add(filename)

                    # Extract video ID for segment matching
                    video_id = extract_video_id(filename)
                    if video_id and video_id not in self._video_id_map:
                        self._video_id_map[video_id] = dest_file

                    imported_count += 1

                except (json.JSONDecodeError, OSError, PermissionError) as e:
                    logger.debug(f"Could not import {cache_file}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Unexpected {type(e).__name__} importing {cache_file}: {e}")
                    continue

        if skipped_invalid > 0:
            logger.warning(f"Transcript cache warmup: skipped {skipped_invalid} invalid entries from {project_dir}")

        if imported_count > 0 or skipped_duplicates > 0:
            logger.info(f"Transcript cache warmup: imported {imported_count} entries, "
                       f"skipped {skipped_duplicates} duplicates from {project_dir}")

        return imported_count
