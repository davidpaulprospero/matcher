"""
Transcript caching with multi-strategy lookup.

Provides caching for transcription results with multiple lookup strategies:
- Source file path matching
- Video ID matching (for segment files)
- Hash-based matching (path + size)

NOTE: Does NOT use BaseCache due to complex multi-strategy lookup
and dynamic index building by scanning cache directory.
"""

import gzip
import json
import hashlib
import logging
import mmap
import shutil
import time
from pathlib import Path
from typing import List, Dict, Optional, Any

from src.utils import normalize_path
from .utils import extract_video_id

logger = logging.getLogger(__name__)

# Size threshold for using mmap (1MB)
MMAP_THRESHOLD_BYTES = 1024 * 1024


class TranscriptCache:
    """
    Multi-strategy transcript cache.

    Lookup strategies:
    1. Source file path (most reliable)
    2. Video ID matching (for segment files in audio-first mode)
    3. Hash-based (path + size fallback)
    """

    def __init__(self, cache_dir: str, compress_cache: bool = True, min_segment_words: int = 3,
                 force_rebuild: bool = False):
        """
        Initialize transcript cache.

        Args:
            cache_dir: Base cache directory (will use transcriptions/ subdirectory)
            compress_cache: Whether to compress cache files with gzip (default: True)
            min_segment_words: Minimum words required per segment (US-110-009), default 3
            force_rebuild: If True, rebuild source_map from all files (US-124-008)
        """
        self.compress_cache = compress_cache
        self.min_segment_words = min_segment_words  # US-110-009
        base_dir = Path(cache_dir)

        # Check both possible folder names
        self.cache_dir = base_dir / "transcriptions"
        self.alt_cache_dir = base_dir / "transcripts"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._source_map: Dict[str, Path] = {}
        self._video_id_map: Dict[str, Path] = {}
        self._legacy_migrated: bool = False

        # US-124-008: Track current cache files for incremental rebuild
        self._current_files: Dict[str, float] = {}

        logger.debug(f"Primary cache: {self.cache_dir}")
        logger.debug(f"Alt cache: {self.alt_cache_dir}")

        # Warn about and migrate legacy cache directory
        self._check_legacy_cache()

        # Build reverse lookup by reading source_file from each cache file
        # US-124-008: Use incremental rebuild unless force_rebuild is True
        self._build_source_map(force_rebuild=force_rebuild)

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

    def _read_file_with_mmap(self, file_path: Path, benchmark: bool = False) -> Optional[bytes]:
        """
        Read file contents using memory mapping for large files.

        Uses mmap for files > MMAP_THRESHOLD_BYTES for improved performance.
        Falls back to standard read if mmap fails or file is small.

        Args:
            file_path: Path to file to read
            benchmark: If True, also time full read for comparison (US-124-004)

        Returns:
            File contents as bytes, or None if read failed
        """
        try:
            file_size = file_path.stat().st_size

            # Use full read for small files
            if file_size < MMAP_THRESHOLD_BYTES:
                with open(file_path, 'rb') as f:
                    return f.read()

            # Try mmap for large files
            mmap_time = None
            try:
                mmap_start = time.perf_counter()
                with open(file_path, 'rb') as f:
                    # Use mmap for large files
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped:
                        data = mmapped[:]
                        mmap_time = time.perf_counter() - mmap_start
                        logger.debug(
                            f"mmap read {file_path.name} ({file_size / 1024 / 1024:.2f} MB) "
                            f"in {mmap_time*1000:.2f}ms"
                        )
            except (OSError, ValueError, mmap.error) as mmap_err:
                # Fallback to full read if mmap fails
                logger.debug(f"mmap failed for {file_path.name}, falling back to full read: {mmap_err}")
                with open(file_path, 'rb') as f:
                    return f.read()

            # Benchmark: also time full read for comparison (US-124-004)
            if benchmark and mmap_time is not None:
                full_read_start = time.perf_counter()
                with open(file_path, 'rb') as f:
                    _ = f.read()
                full_read_time = time.perf_counter() - full_read_start

                improvement = ((full_read_time - mmap_time) / full_read_time) * 100 if full_read_time > 0 else 0
                logger.info(
                    f"BENCHMARK {file_path.name}: mmap={mmap_time*1000:.2f}ms, "
                    f"full_read={full_read_time*1000:.2f}ms, improvement={improvement:.1f}%"
                )

            return data

        except (OSError, PermissionError, FileNotFoundError) as e:
            logger.debug(f"Could not read {file_path}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Unexpected {type(e).__name__} reading {file_path}: {e}")
            return None

    def _build_source_map(self, force_rebuild: bool = False):
        """
        Build mapping from video path to cache file.

        Reads source_file from each cached transcript to build index.
        Supports both .json and .json.gz files (US-110-008).

        US-124-008: Uses incremental rebuild when possible - only processes
 changed files if not force        new or_rebuild.

        Args:
            force_rebuild: If True, rebuild from all files ignoring incremental state
        """
        # US-124-008: Track files for incremental rebuild
        scan_start = time.perf_counter()

        # Collect all current cache files and their mtimes
        all_cache_files = []
        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue
            cache_files = list(folder.glob("*.json")) + list(folder.glob("*.json.gz"))
            for cf in cache_files:
                try:
                    mtime = cf.stat().st_mtime
                    self._current_files[str(cf)] = mtime
                    all_cache_files.append((cf, mtime))
                except OSError:
                    continue

        # US-124-008: Incremental rebuild - only process changed files
        if not force_rebuild:
            # Get delta index for change tracking
            try:
                from .delta_index import DeltaAwareIndex
                delta_index = DeltaAwareIndex(str(self.cache_dir.parent))
                current_mtimes = {str(cf): mt for cf, mt in all_cache_files}

                # Find new and changed files
                new_files, changed_files = delta_index.get_changed_files(current_mtimes)
                removed_files = delta_index.get_removed_files(current_mtimes)

                # Remove entries for deleted files
                for removed_path in removed_files:
                    self._remove_from_source_map(removed_path)
                    delta_index.remove_file(removed_path)

                # If no changes, use cached data
                if not new_files and not changed_files:
                    logger.debug(
                        f"Incremental rebuild: no changes detected, using cached source map "
                        f"({len(self._source_map)} entries)"
                    )
                    return

                # Only process new and changed files
                files_to_process = [cf for cf, mt in all_cache_files
                                   if str(cf) in new_files or str(cf) in changed_files]

                logger.info(
                    f"Incremental rebuild: {len(new_files)} new, {len(changed_files)} changed, "
                    f"{len(removed_files)} removed files"
                )

            except Exception as e:
                # Fall back to full rebuild if delta index fails
                logger.debug(f"Delta index unavailable, using full rebuild: {e}")
                files_to_process = [cf for cf, _ in all_cache_files]
        else:
            # Force rebuild - clear maps and process all files
            logger.info("Force rebuild requested, processing all cache files")
            self._source_map.clear()
            self._video_id_map.clear()
            files_to_process = [cf for cf, _ in all_cache_files]

        # Process only the files that need updating
        for cache_file in files_to_process:
            try:
                # Read file using mmap for large files (US-124-004)
                # Handle both gzip compressed and uncompressed
                file_data = self._read_file_with_mmap(cache_file)
                if file_data is None:
                    continue

                # Try to detect if file is gzip compressed
                if len(file_data) >= 2 and file_data[0] == 0x1f and file_data[1] == 0x8b:
                    # Decompress gzip
                    decompressed = gzip.decompress(file_data)
                    data = json.loads(decompressed.decode('utf-8'))
                else:
                    # Regular JSON file
                    data = json.loads(file_data.decode('utf-8'))

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

                # US-124-008: Update mtime tracking
                try:
                    mtime = cache_file.stat().st_mtime
                    self._current_files[str(cache_file)] = mtime
                except OSError:
                    pass

            except (json.JSONDecodeError, OSError, PermissionError, KeyError, TypeError):
                continue
            except Exception as e:
                logger.warning(f"Unexpected {type(e).__name__} reading cache file {cache_file.name}: {e}")
                continue

        scan_time = time.perf_counter() - scan_start
        logger.debug(
            f"Built source map with {len(self._source_map)} entries, "
            f"{len(self._video_id_map)} video IDs in {scan_time*1000:.1f}ms"
        )

    def _remove_from_source_map(self, cache_file_path: str):
        """
        Remove a cache file from the source maps.

        Args:
            cache_file_path: Path to the cache file to remove
        """
        # Find and remove entries pointing to this cache file
        keys_to_remove = []
        for key, cf in self._source_map.items():
            if str(cf) == cache_file_path:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._source_map.pop(key, None)

        # Also check video_id_map
        vids_to_remove = []
        for vid, cf in self._video_id_map.items():
            if str(cf) == cache_file_path:
                vids_to_remove.append(vid)

        for vid in vids_to_remove:
            self._video_id_map.pop(vid, None)

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

    def _get_video_metadata_hash(self, video_path: str) -> str:
        """
        Get hash for video metadata including mtime for change detection (US-137-012).

        Args:
            video_path: Path to video file

        Returns:
            MD5 hash string based on path, size, and mtime
        """
        path = Path(video_path)
        if path.exists():
            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        else:
            size = 0
            mtime = 0
        key = f"{path.name}:{size}:{mtime}"
        return hashlib.md5(key.encode()).hexdigest()

    def filter_low_quality_segments(
        self, segments: List[dict], video_path: str = ""
    ) -> tuple[List[dict], int]:
        """
        Filter out segments with fewer than min_segment_words (US-110-009).

        Args:
            segments: List of segment dicts with 'text'
            video_path: Optional path for logging warnings

        Returns:
            Tuple of (filtered_segments, rejected_count)
        """
        if self.min_segment_words <= 0:
            return segments, 0

        filtered = []
        rejected = 0
        for seg in segments:
            text = seg.get('text', '')
            word_count = len(str(text).split())
            if word_count >= self.min_segment_words:
                filtered.append(seg)
            else:
                rejected += 1
                logger.debug(
                    f"Filtering segment with {word_count} word(s) (min: {self.min_segment_words})"
                )

        if rejected > 0:
            logger.warning(
                f"Filtered {rejected} low-quality segment(s) from {Path(video_path).name if video_path else 'unknown'} "
                f"(fewer than {self.min_segment_words} words)"
            )

        return filtered, rejected

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

        # Fallback to hash-based lookup (US-110-008: check both .json.gz and .json)
        if not cache_file:
            video_hash = self._get_video_hash(video_path)
            for folder in [self.cache_dir, self.alt_cache_dir]:
                if folder and folder.exists():
                    # Try compressed first, then uncompressed (backward compatibility)
                    compressed = folder / f"{video_hash}.json.gz"
                    uncompressed = folder / f"{video_hash}.json"

                    if compressed.exists():
                        cache_file = compressed
                        break
                    elif uncompressed.exists():
                        cache_file = uncompressed
                        break

        if not cache_file or not cache_file.exists():
            return None

        try:
            # Read file using mmap for large files (US-124-004)
            # Handle both gzip compressed and uncompressed
            file_data = self._read_file_with_mmap(cache_file)
            if file_data is None:
                return None

            # Try to detect if file is gzip compressed
            # Gzip files start with magic bytes 0x1f 0x8b
            if len(file_data) >= 2 and file_data[0] == 0x1f and file_data[1] == 0x8b:
                # Decompress gzip
                decompressed = gzip.decompress(file_data)
                data = json.loads(decompressed.decode('utf-8'))
            else:
                # Regular JSON file
                data = json.loads(file_data.decode('utf-8'))

            # Handle different cache formats
            if isinstance(data, list):
                segments = data
                # Extract language info from top level if present (US-110-003)
                top_level_language = None
                top_level_confidence = None
                top_level_source = None
            elif isinstance(data, dict):
                segments = data.get('segments', data.get('transcripts', []))
                # Extract language info from top level (US-110-003)
                top_level_language = data.get('language')
                top_level_confidence = data.get('language_confidence')
                top_level_source = data.get('language_source')
            else:
                return None

            # US-110-011: Detect legacy format (missing _version field)
            legacy_format_detected = False
            if segments and isinstance(segments[0], dict):
                # Check if any segment has the version field
                first_seg = segments[0]
                cache_version = first_seg.get('_version')
                if cache_version is None:
                    legacy_format_detected = True
                    logger.info(
                        f"Legacy transcript cache format detected for {Path(video_path).name}, "
                        "migrating to current format"
                    )

            # Normalize segment format
            normalized = []
            for seg in segments:
                if isinstance(seg, dict):
                    seg_data = {
                        'start': seg.get('start', seg.get('start_time', 0)),
                        'end': seg.get('end', seg.get('end_time', 0)),
                        'text': seg.get('text', '')
                    }
                    # Include language info - prefer segment-level, fall back to top-level (US-110-003)
                    seg_data['language'] = seg.get('language') or top_level_language or 'en'
                    seg_data['language_confidence'] = seg.get('language_confidence') or top_level_confidence or 1.0
                    seg_data['language_source'] = seg.get('language_source') or top_level_source or 'whisper'
                    # US-110-011: Always include source_file in normalized output
                    seg_data['source_file'] = seg.get('source_file', str(video_path))
                    normalized.append(seg_data)

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
            segments: List of segment dicts with 'start', 'end', 'text',
                and optionally 'language', 'language_confidence', 'language_source'
        """
        # Filter low-quality segments before caching (US-110-009)
        original_count = len(segments)
        segments, rejected_count = self.filter_low_quality_segments(segments, video_path)

        if not segments:
            logger.warning(f"All segments filtered for {Path(video_path).name}, not caching empty transcript")
            return

        if rejected_count > 0:
            logger.info(
                f"Caching transcript for {Path(video_path).name}: "
                f"{len(segments)} segments (filtered {rejected_count} of {original_count})"
            )

        video_hash = self._get_video_hash(video_path)

        # Determine cache file path based on compression setting
        if self.compress_cache:
            cache_file = self.cache_dir / f"{video_hash}.json.gz"
        else:
            cache_file = self.cache_dir / f"{video_hash}.json"

        # Extract language detection info from first segment (US-110-003)
        language_info = {}
        if segments:
            first_seg = segments[0]
            language_info = {
                'language': first_seg.get('language', 'en'),
                'language_confidence': first_seg.get('language_confidence', 1.0),
                'language_source': first_seg.get('language_source', 'whisper')
            }

        # Convert to format matching existing cache (list format for backward compatibility)
        # US-110-011: Add cache version for format migration tracking
        cache_version = 1
        # US-137-012: Add video metadata hash for change detection
        video_metadata_hash = self._get_video_metadata_hash(video_path)
        cache_data = []
        for i, seg in enumerate(segments):
            seg_data = {
                'index': i + 1,
                'start_time': seg.get('start', seg.get('start_time', 0)),
                'end_time': seg.get('end', seg.get('end_time', 0)),
                'text': seg.get('text', ''),
                'source_file': str(video_path),
                '_version': cache_version,  # US-110-011: Track cache format version
                '_video_metadata_hash': video_metadata_hash,  # US-137-012: Track video metadata for invalidation
            }
            # Include language info in each segment (US-110-003)
            # Only add if present to maintain backward compatibility with old cache readers
            if 'language' in seg or language_info:
                seg_data['language'] = seg.get('language') or language_info.get('language', 'en')
            if 'language_confidence' in seg or language_info:
                seg_data['language_confidence'] = seg.get('language_confidence') or language_info.get('language_confidence', 1.0)
            if 'language_source' in seg or language_info:
                seg_data['language_source'] = seg.get('language_source') or language_info.get('language_source', 'whisper')
            cache_data.append(seg_data)

        # Serialize to JSON
        json_data = json.dumps(cache_data, indent=2)
        json_bytes = json_data.encode('utf-8')

        try:
            if self.compress_cache:
                # Compress with gzip and measure compression ratio
                compressed_data = gzip.compress(json_bytes)
                compression_ratio = len(compressed_data) / len(json_bytes) if json_bytes else 1.0

                with open(cache_file, 'wb') as f:
                    f.write(compressed_data)

                logger.debug(f"Cached transcript (compressed {compression_ratio:.2%} of original size): {cache_file.name}")
            else:
                # Write uncompressed JSON
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write(json_data)

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

            # Scan both .json and .json.gz files (US-110-008)
            for cache_file in list(folder.glob("*.json")) + list(folder.glob("*.json.gz")):
                try:
                    # Read file - handle both gzip compressed and uncompressed
                    with open(cache_file, 'rb') as f:
                        file_data = f.read()

                    # Try to detect if file is gzip compressed
                    if len(file_data) >= 2 and file_data[0] == 0x1f and file_data[1] == 0x8b:
                        decompressed = gzip.decompress(file_data)
                        data = json.loads(decompressed.decode('utf-8'))
                    else:
                        data = json.loads(file_data.decode('utf-8'))

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

            # Scan both .json and .json.gz files (US-110-008)
            for cache_file in list(folder.glob("*.json")) + list(folder.glob("*.json.gz")):
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

    def invalidate_if_stale(self, video_path: str) -> bool:
        """
        Check if cache entry is stale based on video file mtime vs cache mtime (US-137-012).

        Args:
            video_path: Path to video file

        Returns:
            True if cache entry was invalidated (was stale), False otherwise
        """
        path = Path(video_path)
        if not path.exists():
            logger.debug(f"Video file does not exist, skipping stale check: {video_path}")
            return False

        # Get current video metadata hash
        current_hash = self._get_video_metadata_hash(video_path)

        # Find cache file for this video
        cache_file = self._source_map.get(str(Path(video_path).resolve()))
        if not cache_file:
            cache_file = self._source_map.get(path.name)
        if not cache_file or not cache_file.exists():
            logger.debug(f"No cache entry found for {path.name}, skipping stale check")
            return False

        try:
            # Read cache entry
            with open(cache_file, 'rb') as f:
                file_data = f.read()

            if len(file_data) >= 2 and file_data[0] == 0x1f and file_data[1] == 0x8b:
                decompressed = gzip.decompress(file_data)
                data = json.loads(decompressed.decode('utf-8'))
            else:
                data = json.loads(file_data.decode('utf-8'))

            # Get cached video metadata hash
            cached_hash = None
            if isinstance(data, list) and len(data) > 0:
                cached_hash = data[0].get('_video_metadata_hash')
            elif isinstance(data, dict):
                cached_hash = data.get('_video_metadata_hash')

            if cached_hash is None:
                # No metadata hash in cache - treat as stale for old entries
                logger.info(f"Cache entry for {path.name} has no metadata hash, invalidating (US-137-012)")
                cache_file.unlink()
                self._remove_from_source_map(str(cache_file))
                return True

            if cached_hash != current_hash:
                # Video has changed - invalidate
                logger.info(f"Cache entry for {path.name} is stale (metadata changed), invalidating (US-137-012)")
                cache_file.unlink()
                self._remove_from_source_map(str(cache_file))
                return True

            logger.debug(f"Cache entry for {path.name} is valid (metadata hash matches)")
            return False

        except (json.JSONDecodeError, OSError, FileNotFoundError, PermissionError) as e:
            logger.warning(f"Error checking stale cache for {path.name}: {e}")
            return False
        except Exception as e:
            logger.warning(f"Unexpected {type(e).__name__} checking stale cache for {path.name}: {e}")
            return False

    def invalidate_by_video_id(self, video_id: str) -> int:
        """
        Invalidate cache entries by video ID (US-137-012).

        Args:
            video_id: Video ID to invalidate

        Returns:
            Number of entries invalidated
        """
        # Find cache file by video ID
        cache_file = self._video_id_map.get(video_id)
        if not cache_file or not cache_file.exists():
            logger.debug(f"No cache entry found for video_id: {video_id}")
            return 0

        try:
            cache_file.unlink()
            self._remove_from_source_map(str(cache_file))
            logger.info(f"Invalidated cache entry for video_id: {video_id} (US-137-012)")
            return 1
        except (OSError, FileNotFoundError, PermissionError) as e:
            logger.warning(f"Error invalidating cache for video_id {video_id}: {e}")
            return 0
        except Exception as e:
            logger.warning(f"Unexpected {type(e).__name__} invalidating cache for {video_id}: {e}")
            return 0

    def get_stats(self) -> Dict[str, any]:
        """
        Get cache statistics.

        Returns:
            Dict with total_entries, total_size_mb, cache_dir
        """
        total_entries = 0
        total_size = 0
        compressed_entries = 0

        for folder in [self.cache_dir, self.alt_cache_dir]:
            if not folder or not folder.exists():
                continue

            # Scan both .json and .json.gz files (US-110-008)
            for cache_file in list(folder.glob("*.json")) + list(folder.glob("*.json.gz")):
                total_entries += 1
                if cache_file.suffix == '.gz':
                    compressed_entries += 1
                try:
                    total_size += cache_file.stat().st_size
                except OSError:
                    pass

        return {
            "total_entries": total_entries,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "source_map_size": len(self._source_map),
            "video_id_map_size": len(self._video_id_map),
            "compressed_entries": compressed_entries
        }

    def force_rebuild(self) -> Dict[str, int]:
        """
        Force a full rebuild of the source map from all cache files.

        US-124-008: Provides CLI-accessible way to trigger full rebuild.

        Returns:
            Dict with 'entries_indexed' count
        """
        logger.info("Force rebuilding transcript cache source map...")
        self._source_map.clear()
        self._video_id_map.clear()
        self._current_files.clear()

        # Clear delta index mtimes for clean rebuild
        try:
            from .delta_index import DeltaAwareIndex
            delta_index = DeltaAwareIndex(str(self.cache_dir.parent))
            delta_index.clear_file_mtimes()
        except Exception as e:
            logger.debug(f"Could not clear delta index: {e}")

        # Do full rebuild
        self._build_source_map(force_rebuild=True)

        return {"entries_indexed": len(self._source_map)}

    def benchmark_startup(self) -> Dict[str, float]:
        """
        Benchmark incremental vs full rebuild startup time.

        US-124-008: Measures and reports startup time improvement.

        Returns:
            Dict with benchmark results: incremental_time_ms, full_time_ms, improvement_percent
        """
        # First, measure incremental rebuild time
        self._source_map.clear()
        self._video_id_map.clear()
        self._current_files.clear()

        # Clear mtimes to simulate "no changes"
        try:
            from .delta_index import DeltaAwareIndex
            delta_index = DeltaAwareIndex(str(self.cache_dir.parent))
            delta_index.clear_file_mtimes()
        except Exception:
            pass

        start_inc = time.perf_counter()
        self._build_source_map(force_rebuild=False)
        incremental_time = time.perf_counter() - start_inc

        # Now measure full rebuild time
        self._source_map.clear()
        self._video_id_map.clear()
        self._current_files.clear()

        try:
            from .delta_index import DeltaAwareIndex
            delta_index = DeltaAwareIndex(str(self.cache_dir.parent))
            delta_index.clear_file_mtimes()
        except Exception:
            pass

        start_full = time.perf_counter()
        self._build_source_map(force_rebuild=True)
        full_time = time.perf_counter() - start_full

        # Calculate improvement
        if full_time > 0:
            improvement = ((full_time - incremental_time) / full_time) * 100
        else:
            improvement = 0

        result = {
            "incremental_time_ms": round(incremental_time * 1000, 2),
            "full_time_ms": round(full_time * 1000, 2),
            "improvement_percent": round(improvement, 1),
            "entries_indexed": len(self._source_map)
        }

        logger.info(
            f"BENCHMARK startup: incremental={result['incremental_time_ms']}ms, "
            f"full={result['full_time_ms']}ms, improvement={result['improvement_percent']}%"
        )

        return result

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

            # Scan both .json and .json.gz files (US-110-008)
            cache_files = list(source_cache_dir.glob("*.json")) + list(source_cache_dir.glob("*.json.gz"))
            logger.debug(f"Scanning {len(cache_files)} cache files in {source_cache_dir}")

            for cache_file in cache_files:
                try:
                    # Read file - handle both gzip compressed and uncompressed
                    with open(cache_file, 'rb') as f:
                        file_data = f.read()

                    # Try to detect if file is gzip compressed
                    if len(file_data) >= 2 and file_data[0] == 0x1f and file_data[1] == 0x8b:
                        decompressed = gzip.decompress(file_data)
                        data = json.loads(decompressed.decode('utf-8'))
                    else:
                        data = json.loads(file_data.decode('utf-8'))

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

                    # Determine destination filename based on compression setting (US-110-008)
                    if self.compress_cache and cache_file.suffix != '.gz':
                        # Source is uncompressed, destination is compressed
                        dest_name = cache_file.stem + '.json.gz'
                    elif not self.compress_cache and cache_file.suffix == '.gz':
                        # Source is compressed, destination is uncompressed
                        dest_name = cache_file.stem.replace('.json', '') + '.json'
                    else:
                        # Same format
                        dest_name = cache_file.name

                    dest_file = self.cache_dir / dest_name
                    if dest_file.exists():
                        skipped_duplicates += 1
                        continue

                    # Write to global cache (respecting compression setting)
                    json_data = json.dumps(data, indent=2)
                    if self.compress_cache:
                        with open(dest_file, 'wb') as f:
                            f.write(gzip.compress(json_data.encode('utf-8')))
                    else:
                        with open(dest_file, 'w', encoding='utf-8') as f:
                            f.write(json_data)

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

    def warmup_from_video_ids(self, video_ids: List[str], global_cache_dir: str = None) -> int:
        """
        Import transcript cache entries from video IDs by scanning global cache.

        Checks the global cache directory for transcripts matching the given video IDs
        and adds them to this cache instance's lookup maps for faster future access.

        Args:
            video_ids: List of YouTube video IDs to warm up
            global_cache_dir: Path to global cache directory (default: ~/.matcher_global_cache)

        Returns:
            Number of entries warmed up (found and added to lookup)
        """
        if global_cache_dir is None:
            global_cache_dir = str(Path.home() / ".matcher_global_cache")

        global_path = Path(global_cache_dir)
        global_transcripts_dir = global_path / "transcriptions"

        if not global_transcripts_dir.exists():
            logger.debug(f"Global transcripts directory not found: {global_transcripts_dir}")
            return 0

        warmed_count = 0

        for video_id in video_ids:
            if not video_id:
                continue

            # Skip if already in our cache
            if video_id in self._video_id_map:
                continue

            # Look for transcript file with this video ID
            # Try various patterns: video_id.json, video_id.json.gz
            for suffix in ['.json', '.json.gz']:
                transcript_file = global_transcripts_dir / f"{video_id}{suffix}"
                if transcript_file.exists():
                    # Validate the entry before adding
                    try:
                        with open(transcript_file, 'rb') as f:
                            file_data = f.read()

                        if len(file_data) >= 2 and file_data[0] == 0x1f and file_data[1] == 0x8b:
                            decompressed = gzip.decompress(file_data)
                            data = json.loads(decompressed.decode('utf-8'))
                        else:
                            data = json.loads(file_data.decode('utf-8'))

                        # Extract segments for validation
                        if isinstance(data, list):
                            raw_segments = data
                        elif isinstance(data, dict):
                            raw_segments = data.get('segments', data.get('transcripts', []))
                        else:
                            raw_segments = []

                        reason = self.validate_entry(raw_segments, video_id=video_id)
                        if reason:
                            logger.debug(f"Skipping invalid global cache entry {video_id}: {reason}")
                            break

                        # Add to our lookup maps
                        self._video_id_map[video_id] = transcript_file
                        warmed_count += 1
                        logger.debug(f"Warmed transcript cache with {video_id} from global cache")
                        break

                    except (json.JSONDecodeError, OSError, PermissionError) as e:
                        logger.debug(f"Could not read global cache entry {video_id}: {e}")
                        break
                    except Exception as e:
                        logger.warning(f"Unexpected {type(e).__name__} reading global cache {video_id}: {e}")
                        break

        if warmed_count > 0:
            logger.info(f"Transcript cache warmup: warmed {warmed_count} entries from video IDs")

        return warmed_count

    def predict_cache_warm(self, video_ids: List[str], global_cache_dir: str = None) -> Dict[str, Any]:
        """
        Predictively warm transcription cache from VIDEO_SEARCH stage video IDs.

        This method is called after VIDEO_SEARCH stage but before CAPTION stage to:
        1. Check if videos already have transcriptions in global cache (transcription warming)
        2. Check if videos exist in global downloaded videos cache (prefetch logic)

        This enables cache hits for videos that will be downloaded, reducing unnecessary
        transcription work.

        Args:
            video_ids: List of YouTube video IDs from VIDEO_SEARCH stage
            global_cache_dir: Path to global cache directory (default: ~/.matcher_global_cache)

        Returns:
            Dict with warming results:
            - transcript_warmed: Number of transcripts added to cache
            - videos_found: Number of videos found in global downloaded cache
            - video_ids: The video IDs that were processed
        """
        if not video_ids:
            return {
                'transcript_warmed': 0,
                'videos_found': 0,
                'video_ids': []
            }

        # Warm transcription cache from global cache
        transcript_warmed = self.warmup_from_video_ids(video_ids, global_cache_dir)

        # Check if videos exist in global downloaded videos cache
        videos_found = 0
        if global_cache_dir is None:
            global_cache_dir = str(Path.home() / ".matcher_global_cache")

        global_path = Path(global_cache_dir)
        downloaded_videos_dir = global_path / "downloaded_videos"

        if downloaded_videos_dir.exists():
            for video_id in video_ids:
                # Check if video file exists (various extensions)
                for ext in ['.mp4', '.mkv', '.webm', '.m4a', '.mp3', '.wav']:
                    if (downloaded_videos_dir / f"{video_id}{ext}").exists():
                        videos_found += 1
                        break

        result = {
            'transcript_warmed': transcript_warmed,
            'videos_found': videos_found,
            'video_ids': video_ids
        }

        if transcript_warmed > 0 or videos_found > 0:
            logger.info(f"Predictive cache warming: {transcript_warmed} transcripts, {videos_found} videos found "
                       f"from {len(video_ids)} VIDEO_SEARCH IDs")

        return result
