"""
Global Cache Module

Enables cross-project video reuse by maintaining a central cache of:
- Video metadata and download provenance
- Transcripts, embeddings, scenes
- Topic/keyword indices for relevance matching

Key features:
- Pre-download optimization: Find relevant videos before downloading
- Smart re-download: Re-add deleted videos to download queue
- Priority system: Current project videos always take precedence

Directory Structure:
    ~/.matcher_global_cache/
    ├── video_registry/          # Individual video metadata (JSON files named by hash)
    │   └── {video_hash}.json    # VideoRegistryEntry for each video
    ├── transcripts/             # Cached transcript data
    │   └── {video_hash}.json    # Transcript JSON
    ├── embeddings/              # Cached embedding vectors
    │   └── {video_hash}.npy     # NumPy array embeddings
    ├── scenes/                 # Cached scene detection results
    │   └── {video_hash}.json    # Scene JSON
    ├── topics/                  # Topic index for fast lookup
    │   └── topic_index.json     # {topic: [video_hash, ...]}
    └── keywords/                # Keyword index for fast lookup
        └── keyword_index.json   # {keyword: [video_hash, ...]}

Video Source Priority System:
    CURRENT_PROJECT (1)       - Highest: Videos from current project being processed
    GLOBAL_HIGH_RELEVANCE (2) - Topic overlap >= 0.7 (exact/close keyword match)
    GLOBAL_MEDIUM_RELEVANCE (3) - Topic overlap 0.3-0.7 (partial keyword match)
    GLOBAL_LOW_RELEVANCE (4)  - Topic overlap < 0.3 (distant/weak match)

API Reference:

    query(keywords, topics=None, min_relevance=0.3, max_results=50) -> GlobalCacheQueryResult
        Alias for find_videos_for_keywords(). Searches the global cache for videos
        matching the given keywords and topics, returning reusable videos and keywords
        that need re-downloading.

    add_video(video_path, download_keyword="", topics=None, youtube_id="",
              youtube_url="", original_title="", project_id="", duration=0.0) -> VideoRegistryEntry
        Alias for register_video(). Adds a video to the global cache, computing a
        content-based hash for deduplication.

    find_videos_for_keywords(keywords, topics=None, min_relevance=0.3, max_results=50)
        -> GlobalCacheQueryResult
        Primary query method. Searches keyword and topic indices, computes relevance
        scores (60% keyword / 40% topic weights), returns categorized results.

    register_video(video_path, download_keyword="", topics=None, youtube_id="",
                   youtube_url="", original_title="", project_id="", duration=0.0)
        -> VideoRegistryEntry
        Registers a new video or updates existing entry. Creates entry file in
        video_registry/, updates topic_index.json and keyword_index.json.

    prioritize_project_videos(videos, project_id) -> List[Tuple[VideoRegistryEntry, float]]
        Reorders video list to prioritize videos from the specified project.
        Videos from the current project get priority boost (+0.2) in relevance scoring.

    get_video_entry(video_hash) -> Optional[VideoRegistryEntry]
        Retrieve metadata for a specific video by its content hash.

    update_video_topics(video_hash, topics)
        Update topics for a video after transcription/analysis.

    mark_video_processed(video_hash, has_transcript=False, has_embeddings=False,
                        has_scenes=False, face_score=None, broll_scenes=None)
        Mark processed artifacts (transcript, embeddings, scenes).

    copy_transcript_to_global(video_hash, transcript_data)
        Copy transcript data to global cache for sharing.

    get_transcript_from_global(video_hash) -> Optional[dict]
        Retrieve cached transcript for a video.

    copy_scenes_to_global(video_hash, scene_data)
        Copy scene detection data to global cache.

    get_scenes_from_global(video_hash) -> Optional[dict]
        Retrieve cached scene data for a video.

    get_stats() -> dict
        Get cache statistics: total_videos, total_topics, total_keywords, cache_dir,
        cache_size_mb.

    cleanup_orphaned_entries() -> int
        Remove entries for videos that no longer exist on disk. Returns count of
        removed entries.

    evict_videos(target_size_mb, strategy="lru") -> dict
        Evict videos to reach target cache size. Strategy can be 'lru' (least
        recently used) or 'oldest'. Returns eviction results with entries_removed,
        bytes_freed, final_size_mb, evicted_hashes.

    check_size_limit(max_size_mb=None, threshold=0.9) -> bool
        Check if cache exceeds size threshold. Returns True if size exceeds threshold.

Usage:
    # Initialize cache manager
    cache = GlobalCacheManager()

    # Query for relevant videos (using query alias)
    result = cache.query(
        keywords=["python tutorial", "coding"],
        topics=["programming"],
        min_relevance=0.3,
        max_results=50
    )

    # Or use the full method name
    result = cache.find_videos_for_keywords(
        keywords=["python tutorial", "coding"],
        topics=["programming"]
    )

    # Register a video (using add_video alias)
    entry = cache.add_video(
        video_path="/path/to/video.mp4",
        download_keyword="python tutorial",
        topics=["programming", "tutorial"],
        youtube_id="abc123",
        project_id="my-project"
    )

    # Or use the full method name
    entry = cache.register_video(...)

    # Prioritize project videos
    prioritized = cache.prioritize_project_videos(
        result.reuse_videos,
        project_id="my-project"
    )

    # Reuse cached videos
    for entry, relevance in prioritized:
        print(f"Reuse: {entry.filename} (relevance: {relevance})")

    # Get cache statistics
    stats = cache.get_stats()
    print(f"Total videos: {stats['total_videos']}")
    print(f"Cache size: {stats['cache_size_mb']:.1f} MB")
"""

import os
import json
import logging
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# Default global cache location
DEFAULT_GLOBAL_CACHE_DIR = Path.home() / ".matcher_global_cache"


class VideoSource(Enum):
    """Priority levels for video sources in matching"""
    CURRENT_PROJECT = 1  # Highest priority
    GLOBAL_HIGH_RELEVANCE = 2  # Topic overlap >= 0.7
    GLOBAL_MEDIUM_RELEVANCE = 3  # Topic overlap 0.3-0.7
    GLOBAL_LOW_RELEVANCE = 4  # Topic overlap < 0.3


@dataclass
class DownloadInfo:
    """Information needed to re-download a video"""
    keyword: str
    youtube_id: str = ""
    youtube_url: str = ""
    original_title: str = ""
    downloaded_at: str = ""
    source: str = "youtube"  # youtube, pexels, pixabay

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DownloadInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class VideoRegistryEntry:
    """Complete metadata for a video in the global cache"""
    video_hash: str
    filename: str
    file_size: int
    duration: float = 0.0

    # Path tracking
    original_paths: List[str] = field(default_factory=list)
    current_path: str = ""
    file_exists: bool = True

    # Download provenance
    download_info: Optional[DownloadInfo] = None

    # Content metadata
    topics: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    transcript_preview: str = ""

    # Cache status
    has_transcript: bool = False
    has_embeddings: bool = False
    has_scenes: bool = False
    face_score: float = 0.5
    broll_scene_indices: List[int] = field(default_factory=list)

    # Usage tracking
    projects_used_in: List[str] = field(default_factory=list)
    usage_count: int = 0
    avg_match_confidence: float = 0.0
    first_seen: str = ""
    last_used: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.download_info:
            d['download_info'] = self.download_info.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "VideoRegistryEntry":
        download_info = None
        if data.get('download_info'):
            download_info = DownloadInfo.from_dict(data['download_info'])

        # Filter to valid fields
        valid_fields = {k: v for k, v in data.items()
                       if k in cls.__dataclass_fields__ and k != 'download_info'}

        return cls(download_info=download_info, **valid_fields)


@dataclass
class GlobalCacheQueryResult:
    """Result of querying global cache for videos"""
    # Videos that exist and can be reused
    reuse_videos: List[Tuple[VideoRegistryEntry, float]] = field(default_factory=list)  # (entry, relevance)

    # Keywords for deleted videos that should be re-downloaded
    redownload_keywords: List[str] = field(default_factory=list)

    # Keywords with no cache matches
    uncovered_keywords: List[str] = field(default_factory=list)

    # Statistics
    total_cached_matches: int = 0
    files_exist_count: int = 0
    files_deleted_count: int = 0


class GlobalCacheManager:
    """
    Manages the global video cache across projects.

    Features:
    - Register videos when downloaded/processed
    - Query for relevant videos before downloading
    - Track file existence and enable re-downloading
    - Share transcripts, embeddings, scenes across projects

    API Methods:
        register_video(video_path, ...) -> VideoRegistryEntry
            Register a new video or update existing entry in the global cache.

        find_videos_for_keywords(keywords, ...) -> GlobalCacheQueryResult
            Query the cache for relevant videos matching keywords/topics.

        get_video_entry(video_hash) -> Optional[VideoRegistryEntry]
            Retrieve metadata for a specific video by its content hash.

        update_video_topics(video_hash, topics)
            Update topics for a video after transcription/analysis.

        mark_video_processed(video_hash, ...)
            Mark processed artifacts (transcript, embeddings, scenes).

        copy_transcript_to_global(video_hash, transcript_data)
            Copy transcript data to global cache for sharing.

        get_transcript_from_global(video_hash) -> Optional[dict]
            Retrieve cached transcript for a video.

        copy_scenes_to_global(video_hash, scene_data)
            Copy scene detection data to global cache.

        get_scenes_from_global(video_hash) -> Optional[dict]
            Retrieve cached scene data for a video.

        get_stats() -> dict
            Get cache statistics (total videos, topics, keywords, size).

        cleanup_orphaned_entries() -> int
            Remove entries for videos that no longer exist on disk.

        evict_videos(target_size_mb, strategy) -> dict
            Evict videos to reach target cache size (LRU or oldest).

        prioritize_project_videos(videos, project_id) -> List[Tuple[VideoRegistryEntry, float]]
            Reorder video results to prioritize videos from a specific project.
            Adds a +0.2 relevance boost to videos that have been used in the
            specified project before.

        check_size_limit(max_size_mb, threshold) -> bool
            Check if cache exceeds size threshold.

    Aliases:
        - query() -> find_videos_for_keywords()
        - add_video() -> register_video()

    Example:
        >>> cache = GlobalCacheManager()
        >>> entry = cache.register_video(
        ...     video_path="video.mp4",
        ...     download_keyword="tutorial",
        ...     project_id="project-123"
        ... )
        >>> print(f"Registered: {entry.video_hash[:8]}")
        >>> result = cache.find_videos_for_keywords(["tutorial"])
        >>> print(f"Found {len(result.reuse_videos)} reusable videos")
    """

    def __init__(self, cache_dir: str = None, config = None):
        """
        Initialize the global cache manager.

        Args:
            cache_dir: Path to global cache directory (default: ~/.matcher_global_cache)
            config: Optional config object with global_cache settings
        """
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_GLOBAL_CACHE_DIR
        self.config = config

        # Cache subdirectories
        self.video_registry_dir = self.cache_dir / "video_registry"
        self.transcripts_dir = self.cache_dir / "transcripts"
        self.embeddings_dir = self.cache_dir / "embeddings"
        self.scenes_dir = self.cache_dir / "scenes"
        self.topics_dir = self.cache_dir / "topics"
        self.keywords_dir = self.cache_dir / "keywords"

        # Index files
        self.registry_index_path = self.video_registry_dir / "index.json"
        self.topic_index_path = self.topics_dir / "topic_index.json"
        self.keyword_index_path = self.keywords_dir / "keyword_index.json"
        self.youtube_id_index_path = self.video_registry_dir / "youtube_id_index.json"

        # In-memory indices (loaded on demand)
        self._registry_index: Dict[str, str] = {}  # video_hash -> entry_file
        self._topic_index: Dict[str, List[str]] = {}  # topic -> [video_hashes]
        self._keyword_index: Dict[str, List[str]] = {}  # keyword -> [video_hashes]
        self._youtube_id_index: Dict[str, str] = {}  # youtube_id -> video_hash
        self._loaded = False

        # Relevance score cache (avoids recomputation across matching passes)
        self._relevance_cache: Dict[str, float] = {}

        # Operation counter for periodic stats logging (every 100 operations)
        self._operation_count: int = 0
        self._stats_log_interval: int = 100

        # Initialize directories
        self._init_directories()

    def _init_directories(self):
        """Create cache directory structure"""
        for dir_path in [
            self.cache_dir,
            self.video_registry_dir,
            self.transcripts_dir,
            self.embeddings_dir,
            self.scenes_dir,
            self.topics_dir,
            self.keywords_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def _load_indices(self):
        """Load indices from disk"""
        if self._loaded:
            return

        # Load registry index
        if self.registry_index_path.exists():
            try:
                with open(self.registry_index_path, 'r') as f:
                    self._registry_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry index: {e}")
                self._registry_index = {}

        # Load topic index
        if self.topic_index_path.exists():
            try:
                with open(self.topic_index_path, 'r') as f:
                    self._topic_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load topic index: {e}")
                self._topic_index = {}

        # Load keyword index
        if self.keyword_index_path.exists():
            try:
                with open(self.keyword_index_path, 'r') as f:
                    self._keyword_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load keyword index: {e}")
                self._keyword_index = {}

        # Load YouTube ID index
        if self.youtube_id_index_path.exists():
            try:
                with open(self.youtube_id_index_path, 'r') as f:
                    self._youtube_id_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load YouTube ID index: {e}")
                self._youtube_id_index = {}

        self._loaded = True
        logger.debug(f"Global cache loaded: {len(self._registry_index)} videos, "
                    f"{len(self._topic_index)} topics, {len(self._keyword_index)} keywords, "
                    f"{len(self._youtube_id_index)} YouTube IDs")

    def _save_indices(self):
        """Save indices to disk"""
        try:
            with open(self.registry_index_path, 'w') as f:
                json.dump(self._registry_index, f, indent=2)
            with open(self.topic_index_path, 'w') as f:
                json.dump(self._topic_index, f, indent=2)
            with open(self.keyword_index_path, 'w') as f:
                json.dump(self._keyword_index, f, indent=2)
            with open(self.youtube_id_index_path, 'w') as f:
                json.dump(self._youtube_id_index, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save global cache indices: {e}")

    def _log_stats_if_needed(self, operation: str):
        """Log cache stats every 100 operations for monitoring.

        Args:
            operation: Description of the operation being performed
        """
        self._operation_count += 1
        if self._operation_count % self._stats_log_interval == 0:
            stats = self.get_stats()
            logger.info(
                f"[CACHE] Global cache stats (ops={self._operation_count}): "
                f"videos={stats['total_videos']}, topics={stats['total_topics']}, "
                f"keywords={stats['total_keywords']}, size={stats['cache_size_mb']:.2f}MB"
            )

    def _compute_content_hash(self, video_path: str) -> str:
        """
        Compute a content-based hash for video deduplication.
        Uses file size + first/last 1MB of content for speed.
        """
        path = Path(video_path)
        if not path.exists():
            # Fall back to filename-based hash
            return hashlib.md5(path.name.encode()).hexdigest()

        try:
            file_size = path.stat().st_size

            # Read first and last 1MB for fingerprinting
            chunk_size = min(1024 * 1024, file_size // 2)  # 1MB or half file

            with open(path, 'rb') as f:
                first_chunk = f.read(chunk_size)
                if file_size > chunk_size * 2:
                    f.seek(-chunk_size, 2)  # Seek from end
                    last_chunk = f.read(chunk_size)
                else:
                    last_chunk = b''

            hash_input = f"{file_size}:{first_chunk[:1000]}:{last_chunk[:1000]}"
            return hashlib.md5(hash_input.encode('latin-1')).hexdigest()

        except Exception as e:
            logger.debug(f"Content hash failed for {video_path}: {e}")
            return hashlib.md5(f"{path.name}:{path.stat().st_size}".encode()).hexdigest()

    def register_video(
        self,
        video_path: str,
        download_keyword: str = "",
        topics: List[str] = None,
        youtube_id: str = "",
        youtube_url: str = "",
        original_title: str = "",
        project_id: str = "",
        duration: float = 0.0
    ) -> VideoRegistryEntry:
        """
        Register a video in the global cache.

        This method adds a video to the global cache, computing a content-based
        hash for deduplication. If the video already exists (same content hash),
        it updates the existing entry with new paths and metadata.

        Args:
            video_path: Path to the video file (required)
            download_keyword: Keyword used to download this video
            topics: List of topic keywords extracted from video
            youtube_id: YouTube video ID (for re-downloading)
            youtube_url: Full YouTube URL
            original_title: Original video title
            project_id: Current project identifier
            duration: Video duration in seconds

        Returns:
            VideoRegistryEntry for the registered video

        Side Effects:
            - Creates entry file in video_registry/
            - Updates topic_index.json if topics provided
            - Updates keyword_index.json if keywords provided

        Example:
            >>> entry = cache.register_video(
            ...     video_path="E:/Videos/tutorial.mp4",
            ...     download_keyword="python tutorial",
            ...     topics=["programming", "python"],
            ...     youtube_id="dQw4w9WgXcQ",
            ...     project_id="my-documentary"
            ... )
            >>> print(f"Video hash: {entry.video_hash}")
        """
        self._load_indices()
        self._log_stats_if_needed("register_video")

        path = Path(video_path)
        video_hash = self._compute_content_hash(video_path)

        # Check if already registered
        existing_entry = self.get_video_entry(video_hash)
        if existing_entry:
            # Update existing entry
            if video_path not in existing_entry.original_paths:
                existing_entry.original_paths.append(video_path)
            existing_entry.current_path = video_path
            existing_entry.file_exists = path.exists()
            existing_entry.last_used = datetime.now().isoformat()
            existing_entry.usage_count += 1
            if project_id and project_id not in existing_entry.projects_used_in:
                existing_entry.projects_used_in.append(project_id)
            if topics:
                for topic in topics:
                    if topic not in existing_entry.topics:
                        existing_entry.topics.append(topic)
            self._save_video_entry(existing_entry)
            return existing_entry

        # Create new entry
        now = datetime.now().isoformat()
        download_info = None
        if download_keyword:
            download_info = DownloadInfo(
                keyword=download_keyword,
                youtube_id=youtube_id,
                youtube_url=youtube_url,
                original_title=original_title,
                downloaded_at=now
            )

        entry = VideoRegistryEntry(
            video_hash=video_hash,
            filename=path.name,
            file_size=path.stat().st_size if path.exists() else 0,
            duration=duration,
            original_paths=[video_path],
            current_path=video_path,
            file_exists=path.exists(),
            download_info=download_info,
            topics=topics or [],
            keywords=[download_keyword] if download_keyword else [],
            projects_used_in=[project_id] if project_id else [],
            usage_count=1,
            first_seen=now,
            last_used=now
        )

        # US-129-011: Update indices BEFORE saving to ensure youtube_id_index is populated
        self._update_indices(entry)
        self._save_video_entry(entry)

        logger.info(f"[CACHE] Video registered in global cache: {path.name} ({video_hash[:8]})")
        logger.debug(f"Registered video in global cache: {path.name} ({video_hash[:8]})")
        return entry

    def _save_video_entry(self, entry: VideoRegistryEntry):
        """Save a video registry entry to disk"""
        entry_path = self.video_registry_dir / f"{entry.video_hash}.json"
        try:
            with open(entry_path, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)
            self._registry_index[entry.video_hash] = str(entry_path)
            self._save_indices()
        except Exception as e:
            logger.error(f"Failed to save video entry {entry.video_hash}: {e}")

    def _update_indices(self, entry: VideoRegistryEntry):
        """Update topic and keyword indices for a video entry"""
        # Update topic index
        for topic in entry.topics:
            topic_lower = topic.lower()
            if topic_lower not in self._topic_index:
                self._topic_index[topic_lower] = []
            if entry.video_hash not in self._topic_index[topic_lower]:
                self._topic_index[topic_lower].append(entry.video_hash)

        # Update keyword index
        for keyword in entry.keywords:
            kw_lower = keyword.lower()
            if kw_lower not in self._keyword_index:
                self._keyword_index[kw_lower] = []
            if entry.video_hash not in self._keyword_index[kw_lower]:
                self._keyword_index[kw_lower].append(entry.video_hash)

        # Also index the download keyword
        if entry.download_info and entry.download_info.keyword:
            kw_lower = entry.download_info.keyword.lower()
            if kw_lower not in self._keyword_index:
                self._keyword_index[kw_lower] = []
            if entry.video_hash not in self._keyword_index[kw_lower]:
                self._keyword_index[kw_lower].append(entry.video_hash)

        # Update YouTube ID index for fast lookup (US-129-011)
        if entry.download_info and entry.download_info.youtube_id:
            self._youtube_id_index[entry.download_info.youtube_id] = entry.video_hash

    def get_video_entry(self, video_hash: str) -> Optional[VideoRegistryEntry]:
        """Get a video registry entry by hash"""
        self._load_indices()
        self._log_stats_if_needed("get_video_entry")

        entry_path = self.video_registry_dir / f"{video_hash}.json"
        if not entry_path.exists():
            logger.warning(f"[CACHE] Video entry MISS (not found): {video_hash[:8]}")
            return None

        try:
            with open(entry_path, 'r') as f:
                data = json.load(f)
            entry = VideoRegistryEntry.from_dict(data)
            logger.info(f"[CACHE] Video entry HIT: {video_hash[:8]} ({entry.filename})")
            return entry
        except json.JSONDecodeError as e:
            logger.warning(f"[CACHE] Cache CORRUPTION detected: {video_hash[:8]}, error: {e}")
            # Mark entry as corrupted - remove invalid file
            try:
                entry_path.unlink()
                logger.info(f"[CACHE] Removed corrupted entry: {video_hash[:8]}")
            except Exception:
                pass
            return None
        except Exception as e:
            logger.warning(f"[CACHE] Video entry MISS (invalid): {video_hash[:8]}, error: {e}")
            return None

    def find_by_youtube_id(self, youtube_id: str) -> Optional[VideoRegistryEntry]:
        """
        Find a video in the global cache by YouTube ID.

        US-129-011: Enables download deduplication - checks if a video
        has already been downloaded globally before re-downloading.

        Args:
            youtube_id: YouTube video ID to search for

        Returns:
            VideoRegistryEntry if found and file exists, None otherwise
        """
        self._load_indices()

        # Look up video_hash by youtube_id
        video_hash = self._youtube_id_index.get(youtube_id)
        if not video_hash:
            logger.warning(f"[CACHE] YouTube ID MISS (not indexed): {youtube_id}")
            return None

        # Get the entry and verify file exists
        entry = self.get_video_entry(video_hash)
        if not entry:
            logger.warning(f"[CACHE] YouTube ID MISS (entry invalid): {youtube_id}")
            return None

        # Verify the file actually exists
        if not self._check_file_exists(entry):
            logger.warning(f"[CACHE] YouTube ID MISS (file deleted): {youtube_id}")
            return None

        logger.info(f"[CACHE] YouTube ID HIT: {youtube_id} ({entry.filename})")
        return entry

    def copy_to_project(
        self,
        video_hash: str,
        dest_dir: Path,
        dest_filename: str = None
    ) -> Optional[str]:
        """
        Copy a cached video file to a project directory.

        US-129-011: Used for cross-project deduplication - when a video
        already exists in global cache, copy it to the project directory
        instead of re-downloading.

        Args:
            video_hash: Content hash of the video in global cache
            dest_dir: Destination directory (project's downloaded_videos_dir)
            dest_filename: Optional custom filename, defaults to original filename

        Returns:
            Path to copied file if successful, None if failed
        """
        entry = self.get_video_entry(video_hash)
        if not entry:
            logger.warning(f"Cannot copy: video hash {video_hash[:8]} not found in global cache")
            return None

        # Find the actual source file path
        source_path = None
        if entry.current_path and Path(entry.current_path).exists():
            source_path = Path(entry.current_path)
        else:
            for path_str in entry.original_paths:
                if Path(path_str).exists():
                    source_path = Path(path_str)
                    break

        if not source_path:
            logger.warning(f"Cannot copy: no existing file found for {entry.filename}")
            return None

        # Prepare destination
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        if dest_filename:
            dest_path = dest_dir / dest_filename
        else:
            dest_path = dest_dir / entry.filename

        try:
            # Copy the file
            shutil.copy2(source_path, dest_path)

            # Update usage tracking
            entry.usage_count += 1
            entry.last_used = datetime.now().isoformat()
            self._save_video_entry(entry)

            logger.info(f"Copied cached video to project: {source_path.name} -> {dest_path}")
            return str(dest_path)

        except Exception as e:
            logger.error(f"Failed to copy cached video: {e}")
            return None

    def find_videos_for_keywords(
        self,
        keywords: List[str],
        topics: List[str] = None,
        min_relevance: float = 0.3,
        max_results: int = 50
    ) -> GlobalCacheQueryResult:
        """
        Find relevant videos in the global cache for given keywords/topics.

        This is the primary query method for the global cache. It searches
        both the keyword index and topic index to find relevant videos, then
        computes relevance scores and returns categorized results.

        Args:
            keywords: Download keywords to search for (e.g., ["python tutorial"])
            topics: Topic keywords for relevance matching (e.g., ["programming"])
            min_relevance: Minimum relevance score (0-1). Videos below this
                threshold are excluded. Default: 0.3
            max_results: Maximum videos to return per category. Default: 50

        Returns:
            GlobalCacheQueryResult containing:
            - reuse_videos: List[Tuple[VideoRegistryEntry, float]] - Existing files
            - redownload_keywords: List[str] - Keywords for deleted videos
            - uncovered_keywords: List[str] - Keywords with no matches
            - total_cached_matches: int - Total videos found
            - files_exist_count: int - Videos available for reuse
            - files_deleted_count: int - Videos that need re-downloading

        Relevance Scoring:
            - Keyword matching: 60% weight (exact/partial match)
            - Topic matching: 40% weight (topic overlap)
            - Score = (0.6 * keyword_ratio) + (0.4 * topic_ratio)

        Example:
            >>> result = cache.find_videos_for_keywords(
            ...     keywords=["python tutorial", "coding basics"],
            ...     topics=["programming", "education"],
            ...     min_relevance=0.5
            ... )
            >>> for entry, score in result.reuse_videos:
            ...     print(f"{entry.filename}: {score:.2f}")
        """
        self._load_indices()
        self._log_stats_if_needed("find_videos_for_keywords")

        result = GlobalCacheQueryResult()
        matched_hashes: Set[str] = set()
        keyword_matches: Dict[str, Set[str]] = {}  # keyword -> matched hashes

        # Search by keywords
        for keyword in keywords:
            kw_lower = keyword.lower()
            keyword_matches[keyword] = set()

            # Exact match
            if kw_lower in self._keyword_index:
                for video_hash in self._keyword_index[kw_lower]:
                    matched_hashes.add(video_hash)
                    keyword_matches[keyword].add(video_hash)

            # Partial match (keyword contains or is contained in index key)
            for idx_keyword, hashes in self._keyword_index.items():
                if kw_lower in idx_keyword or idx_keyword in kw_lower:
                    for video_hash in hashes:
                        matched_hashes.add(video_hash)
                        keyword_matches[keyword].add(video_hash)

        # Search by topics if provided
        if topics:
            for topic in topics:
                topic_lower = topic.lower()
                if topic_lower in self._topic_index:
                    for video_hash in self._topic_index[topic_lower]:
                        matched_hashes.add(video_hash)

        result.total_cached_matches = len(matched_hashes)

        # Check each matched video
        for video_hash in matched_hashes:
            entry = self.get_video_entry(video_hash)
            if not entry:
                continue

            # Check file existence
            file_exists = self._check_file_exists(entry)
            entry.file_exists = file_exists

            # Calculate relevance score
            relevance = self._compute_relevance(entry, keywords, topics)

            if relevance < min_relevance:
                continue

            if file_exists:
                result.reuse_videos.append((entry, relevance))
                result.files_exist_count += 1
            else:
                # File deleted - add to re-download queue
                if entry.download_info and entry.download_info.keyword:
                    if entry.download_info.keyword not in result.redownload_keywords:
                        result.redownload_keywords.append(entry.download_info.keyword)
                result.files_deleted_count += 1

        # Find uncovered keywords
        for keyword in keywords:
            if not keyword_matches.get(keyword):
                result.uncovered_keywords.append(keyword)

        # Sort by relevance and limit
        result.reuse_videos.sort(key=lambda x: x[1], reverse=True)
        result.reuse_videos = result.reuse_videos[:max_results]

        return result

    def _check_file_exists(self, entry: VideoRegistryEntry) -> bool:
        """Check if any known path for this video exists"""
        # Check current path first
        if entry.current_path and Path(entry.current_path).exists():
            return True

        # Check all original paths
        for path in entry.original_paths:
            if Path(path).exists():
                entry.current_path = path  # Update current path
                return True

        return False

    def _compute_relevance(
        self,
        entry: VideoRegistryEntry,
        keywords: List[str],
        topics: List[str] = None
    ) -> float:
        """
        Compute relevance score for a video entry with caching.

        Uses a cache keyed by (video_hash, keywords_hash, topics_hash) to avoid
        recomputation for the same video/query combinations across matching passes.

        Returns score from 0.0 to 1.0.
        """
        # Check cache
        cache_key = self._make_relevance_cache_key(entry.video_hash, keywords, topics)
        if cache_key in self._relevance_cache:
            return self._relevance_cache[cache_key]

        score = 0.0
        max_score = 0.0

        # Keyword matching (weight: 0.6)
        if keywords:
            keyword_matches = 0
            for keyword in keywords:
                kw_lower = keyword.lower()
                for entry_kw in entry.keywords:
                    if kw_lower in entry_kw.lower() or entry_kw.lower() in kw_lower:
                        keyword_matches += 1
                        break
                # Check download keyword
                if entry.download_info:
                    dl_kw = entry.download_info.keyword.lower()
                    if kw_lower in dl_kw or dl_kw in kw_lower:
                        keyword_matches += 1

            score += 0.6 * (keyword_matches / len(keywords))
            max_score += 0.6

        # Topic matching (weight: 0.4)
        if topics and entry.topics:
            topic_matches = 0
            for topic in topics:
                topic_lower = topic.lower()
                for entry_topic in entry.topics:
                    if topic_lower in entry_topic.lower() or entry_topic.lower() in topic_lower:
                        topic_matches += 1
                        break

            score += 0.4 * (topic_matches / len(topics))
            max_score += 0.4
        elif not topics:
            max_score += 0.4  # Give full score if no topics to match
            score += 0.4

        result = score / max_score if max_score > 0 else 0.0

        # Cache the result (limit cache size)
        if len(self._relevance_cache) < 10000:
            self._relevance_cache[cache_key] = result

        return result

    def _make_relevance_cache_key(
        self,
        video_hash: str,
        keywords: List[str],
        topics: List[str] = None
    ) -> str:
        """
        Create cache key for relevance score lookup.

        Args:
            video_hash: Video identifier
            keywords: Query keywords
            topics: Optional query topics

        Returns:
            Cache key string
        """
        kw_sorted = sorted(k.lower() for k in keywords) if keywords else []
        topic_sorted = sorted(t.lower() for t in topics) if topics else []
        key_str = f"{video_hash}|{'|'.join(kw_sorted)}|{'|'.join(topic_sorted)}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def update_video_topics(self, video_hash: str, topics: List[str]):
        """Update topics for a video after transcription/analysis"""
        entry = self.get_video_entry(video_hash)
        if not entry:
            return

        for topic in topics:
            if topic not in entry.topics:
                entry.topics.append(topic)

        self._save_video_entry(entry)
        self._update_indices(entry)

    def mark_video_processed(
        self,
        video_hash: str,
        has_transcript: bool = False,
        has_embeddings: bool = False,
        has_scenes: bool = False,
        face_score: float = None,
        broll_scenes: List[int] = None
    ):
        """Mark a video as having been processed with various analyses"""
        entry = self.get_video_entry(video_hash)
        if not entry:
            return

        if has_transcript:
            entry.has_transcript = True
        if has_embeddings:
            entry.has_embeddings = True
        if has_scenes:
            entry.has_scenes = True
        if face_score is not None:
            entry.face_score = face_score
        if broll_scenes:
            entry.broll_scene_indices = broll_scenes

        self._save_video_entry(entry)

    def copy_transcript_to_global(self, video_hash: str, transcript_data: dict):
        """Copy transcript data to global cache"""
        transcript_path = self.transcripts_dir / f"{video_hash}.json"
        try:
            with open(transcript_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            logger.info(f"[CACHE] Transcript written to global cache: {video_hash[:8]}")
        except Exception as e:
            logger.error(f"Failed to save transcript to global cache: {e}")

    def get_transcript_from_global(self, video_hash: str) -> Optional[dict]:
        """Get transcript data from global cache"""
        transcript_path = self.transcripts_dir / f"{video_hash}.json"
        if not transcript_path.exists():
            logger.warning(f"[CACHE] Transcript MISS (not found): {video_hash[:8]}")
            return None

        try:
            with open(transcript_path, 'r') as f:
                data = json.load(f)
            logger.info(f"[CACHE] Transcript HIT: {video_hash[:8]}")
            return data
        except Exception as e:
            logger.warning(f"[CACHE] Transcript MISS (invalid): {video_hash[:8]}, error: {e}")
            return None

    def copy_scenes_to_global(self, video_hash: str, scene_data: dict):
        """Copy scene detection data to global cache"""
        scene_path = self.scenes_dir / f"{video_hash}.json"
        try:
            with open(scene_path, 'w') as f:
                json.dump(scene_data, f, indent=2)
            logger.info(f"[CACHE] Scenes written to global cache: {video_hash[:8]}")
        except Exception as e:
            logger.error(f"Failed to save scenes to global cache: {e}")

    def get_scenes_from_global(self, video_hash: str) -> Optional[dict]:
        """Get scene data from global cache"""
        scene_path = self.scenes_dir / f"{video_hash}.json"
        if not scene_path.exists():
            logger.warning(f"[CACHE] Scenes MISS (not found): {video_hash[:8]}")
            return None

        try:
            with open(scene_path, 'r') as f:
                data = json.load(f)
            logger.info(f"[CACHE] Scenes HIT: {video_hash[:8]}")
            return data
        except Exception as e:
            logger.warning(f"[CACHE] Scenes MISS (invalid): {video_hash[:8]}, error: {e}")
            return None

    def get_stats(self) -> dict:
        """Get global cache statistics"""
        self._load_indices()

        return {
            "total_videos": len(self._registry_index),
            "total_topics": len(self._topic_index),
            "total_keywords": len(self._keyword_index),
            "cache_dir": str(self.cache_dir),
            "cache_size_mb": self._get_cache_size_mb()
        }

    def _get_cache_size_mb(self) -> float:
        """Calculate total cache size in MB"""
        total_size = 0
        for root, dirs, files in os.walk(self.cache_dir):
            for file in files:
                total_size += os.path.getsize(os.path.join(root, file))
        return total_size / (1024 * 1024)

    def get_video_hash(self, video_path: str) -> str:
        """Get content hash for a video file"""
        return self._compute_content_hash(video_path)

    def touch_video(self, video_hash: str):
        """Update last_used timestamp for a video (LRU tracking)"""
        entry = self.get_video_entry(video_hash)
        if entry:
            entry.last_used = datetime.now().isoformat()
            self._save_video_entry(entry)

    def cleanup_orphaned_entries(self) -> int:
        """Remove entries for videos that no longer exist on disk"""
        self._load_indices()
        removed = 0
        corrupted = 0

        for video_hash in list(self._registry_index.keys()):
            entry = self.get_video_entry(video_hash)
            if entry is None:
                # get_video_entry already logged the error and removed corrupted entries
                corrupted += 1
                # Remove from index if still present
                if video_hash in self._registry_index:
                    del self._registry_index[video_hash]
                removed += 1
            elif not self._check_file_exists(entry):
                # Remove the entry file
                entry_path = self.video_registry_dir / f"{video_hash}.json"
                if entry_path.exists():
                    entry_path.unlink()
                # Remove from index
                del self._registry_index[video_hash]
                removed += 1

        if removed > 0:
            self._save_indices()
            logger.info(f"[CACHE] Cleanup complete: removed {removed} orphaned entries")
            if corrupted > 0:
                logger.warning(f"[CACHE] Cleanup detected {corrupted} corrupted entries")

        return removed

    def count_orphaned_entries(self) -> int:
        """Count entries for videos that no longer exist on disk"""
        self._load_indices()
        orphaned = 0

        for video_hash in self._registry_index.keys():
            entry = self.get_video_entry(video_hash)
            if entry and not self._check_file_exists(entry):
                orphaned += 1

        return orphaned

    def evict_videos(
        self,
        target_size_mb: float,
        strategy: str = "lru"
    ) -> dict:
        """
        Evict videos to reach target size.

        Args:
            target_size_mb: Target cache size in MB
            strategy: 'lru' (least recently used) or 'oldest'

        Returns:
            Dict with eviction results
        """
        self._load_indices()
        current_size = self._get_cache_size_mb()

        if current_size <= target_size_mb:
            return {
                "entries_removed": 0,
                "bytes_freed": 0,
                "final_size_mb": current_size,
                "evicted_hashes": []
            }

        # Get all entries with timestamps
        entries_with_time = []
        for video_hash in self._registry_index.keys():
            entry = self.get_video_entry(video_hash)
            if entry:
                # Parse ISO timestamp to float
                try:
                    if strategy == "lru":
                        ts = datetime.fromisoformat(entry.last_used).timestamp() if entry.last_used else 0
                    else:
                        ts = datetime.fromisoformat(entry.first_seen).timestamp() if entry.first_seen else 0
                except (ValueError, OSError) as e:
                    # ValueError: invalid ISO format, OSError: timestamp out of range
                    logger.debug(f"Could not parse timestamp for {video_hash}: {e}")
                    ts = 0
                entries_with_time.append((video_hash, entry, ts))

        # Sort by timestamp (oldest/least-recently-used first)
        entries_with_time.sort(key=lambda x: x[2])

        evicted_hashes = []
        bytes_freed = 0
        entries_removed = 0

        for video_hash, entry, _ in entries_with_time:
            if self._get_cache_size_mb() <= target_size_mb:
                break

            # Calculate entry size
            entry_size = entry.file_size

            # Remove entry file
            entry_path = self.video_registry_dir / f"{video_hash}.json"
            if entry_path.exists():
                entry_size += entry_path.stat().st_size
                entry_path.unlink()

            # Remove associated files (transcripts, scenes, etc.)
            for subdir in [self.transcripts_dir, self.scenes_dir, self.embeddings_dir]:
                for ext in ['.json', '.npy']:
                    file_path = subdir / f"{video_hash}{ext}"
                    if file_path.exists():
                        entry_size += file_path.stat().st_size
                        file_path.unlink()

            # Remove from index
            if video_hash in self._registry_index:
                del self._registry_index[video_hash]

            evicted_hashes.append(video_hash)
            bytes_freed += entry_size
            entries_removed += 1

        if entries_removed > 0:
            self._save_indices()
            logger.info(
                f"[CACHE] Eviction complete: removed {entries_removed} videos, "
                f"freed {bytes_freed / (1024*1024):.2f}MB, "
                f"final size: {self._get_cache_size_mb():.2f}MB"
            )

        return {
            "entries_removed": entries_removed,
            "bytes_freed": bytes_freed,
            "final_size_mb": self._get_cache_size_mb(),
            "evicted_hashes": evicted_hashes
        }

    def check_size_limit(self, max_size_mb: float = None, threshold: float = 0.9) -> bool:
        """
        Check if cache size exceeds threshold of max size.

        Args:
            max_size_mb: Maximum size in MB (uses config if not provided)
            threshold: Fraction of max_size to trigger (default 0.9)

        Returns:
            True if size exceeds threshold
        """
        if max_size_mb is None:
            max_size_mb = getattr(self.config, 'max_size_mb', 0) if self.config else 0

        if max_size_mb <= 0:
            return False

        current_size = self._get_cache_size_mb()
        exceeds = current_size >= (max_size_mb * threshold)

        if exceeds:
            logger.warning(
                f"[CACHE] Size limit exceeded: {current_size:.2f}MB > {max_size_mb * threshold:.2f}MB "
                f"(threshold: {threshold:.0%}, max: {max_size_mb:.2f}MB)"
            )

        return exceeds

    # Alias methods for API consistency

    def query(
        self,
        keywords: List[str],
        topics: List[str] = None,
        min_relevance: float = 0.3,
        max_results: int = 50
    ) -> GlobalCacheQueryResult:
        """
        Query the global cache for relevant videos.

        Alias for find_videos_for_keywords() for API consistency.

        Args:
            keywords: Download keywords to search for
            topics: Topic keywords for relevance matching
            min_relevance: Minimum relevance score (0-1)
            max_results: Maximum videos to return

        Returns:
            GlobalCacheQueryResult with reuse_videos, redownload_keywords, etc.
        """
        return self.find_videos_for_keywords(keywords, topics, min_relevance, max_results)

    def add_video(
        self,
        video_path: str,
        download_keyword: str = "",
        topics: List[str] = None,
        youtube_id: str = "",
        youtube_url: str = "",
        original_title: str = "",
        project_id: str = "",
        duration: float = 0.0
    ) -> VideoRegistryEntry:
        """
        Add a video to the global cache.

        Alias for register_video() for API consistency.

        Args:
            video_path: Path to the video file
            download_keyword: Keyword used to download this video
            topics: List of topic keywords
            youtube_id: YouTube video ID
            youtube_url: Full YouTube URL
            original_title: Original video title
            project_id: Current project identifier
            duration: Video duration in seconds

        Returns:
            VideoRegistryEntry for the registered video
        """
        return self.register_video(
            video_path=video_path,
            download_keyword=download_keyword,
            topics=topics,
            youtube_id=youtube_id,
            youtube_url=youtube_url,
            original_title=original_title,
            project_id=project_id,
            duration=duration
        )

    def prioritize_project_videos(
        self,
        videos: List[Tuple[VideoRegistryEntry, float]],
        project_id: str
    ) -> List[Tuple[VideoRegistryEntry, float]]:
        """
        Reorder video results to prioritize videos from a specific project.

        Videos that have been used in the specified project receive a +0.2
        relevance boost and are moved to the front of the result list.

        Args:
            videos: List of (VideoRegistryEntry, relevance) tuples
            project_id: Project ID to prioritize

        Returns:
            Reordered list with project videos first, boosted by +0.2
        """
        if not videos or not project_id:
            return videos

        project_videos = []
        other_videos = []

        for entry, relevance in videos:
            if project_id in entry.projects_used_in:
                # Boost relevance for project videos
                boosted_relevance = min(1.0, relevance + 0.2)
                project_videos.append((entry, boosted_relevance))
            else:
                other_videos.append((entry, relevance))

        # Sort each group by relevance (descending)
        project_videos.sort(key=lambda x: x[1], reverse=True)
        other_videos.sort(key=lambda x: x[1], reverse=True)

        return project_videos + other_videos


def prompt_global_cache_reuse(
    query_result: GlobalCacheQueryResult,
    keywords: List[str]
) -> Tuple[bool, List[str], List[str]]:
    """
    Prompt user about reusing videos from global cache.

    Returns:
        Tuple of (use_cache, reuse_paths, download_keywords)
    """
    print("\n" + "=" * 60)
    print("  GLOBAL CACHE CHECK")
    print("=" * 60)

    if not query_result.reuse_videos and not query_result.redownload_keywords:
        print("\n  No relevant videos found in global cache.")
        print(f"  Will download {len(keywords)} keywords fresh.\n")
        return False, [], keywords

    print(f"\n  Extracted {len(keywords)} keywords from voiceover.")
    print(f"  Checking global cache... found {query_result.total_cached_matches} relevant videos:\n")

    # Show reusable videos
    if query_result.reuse_videos:
        print("  ✓ REUSABLE (files exist):")
        for i, (entry, relevance) in enumerate(query_result.reuse_videos[:5], 1):
            topics_str = ", ".join(entry.topics[:3]) if entry.topics else "no topics"
            print(f"    {i}. {entry.filename} ({entry.duration:.0f}s)")
            print(f"       Topics: {topics_str}")
            print(f"       Location: {entry.current_path}")
        if len(query_result.reuse_videos) > 5:
            print(f"    ... and {len(query_result.reuse_videos) - 5} more")
        print()

    # Show re-download keywords
    if query_result.redownload_keywords:
        print("  ↻ NEED RE-DOWNLOAD (files deleted):")
        for i, keyword in enumerate(query_result.redownload_keywords[:5], 1):
            print(f"    {i}. \"{keyword}\"")
        if len(query_result.redownload_keywords) > 5:
            print(f"    ... and {len(query_result.redownload_keywords) - 5} more")
        print()

    # Show uncovered keywords
    if query_result.uncovered_keywords:
        print("  + NEW KEYWORDS (not in cache):")
        for i, keyword in enumerate(query_result.uncovered_keywords[:5], 1):
            print(f"    {i}. \"{keyword}\"")
        if len(query_result.uncovered_keywords) > 5:
            print(f"    ... and {len(query_result.uncovered_keywords) - 5} more")
        print()

    # Summary counts
    reuse_count = len(query_result.reuse_videos)
    redownload_count = len(query_result.redownload_keywords)
    new_count = len(query_result.uncovered_keywords)

    print("-" * 60)
    print("  Options:")
    print(f"    [R] Reuse {reuse_count} cached + download {redownload_count} deleted + {new_count} new (recommended)")
    print(f"    [D] Download all {len(keywords)} fresh (ignore cache)")
    print(f"    [Q] Quit")
    print("-" * 60)

    while True:
        try:
            choice = input("  Choice [R/D/Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            choice = 'Q'

        if choice == 'R':
            reuse_paths = [entry.current_path for entry, _ in query_result.reuse_videos]
            download_kws = query_result.redownload_keywords + query_result.uncovered_keywords
            print(f"\n  → Reusing {len(reuse_paths)} videos, downloading {len(download_kws)} keywords\n")
            return True, reuse_paths, download_kws

        elif choice == 'D':
            print(f"\n  → Downloading all {len(keywords)} keywords fresh\n")
            return False, [], keywords

        elif choice == 'Q':
            print("\n  → Exiting\n")
            return False, [], []

        else:
            print("  Invalid choice. Please enter R, D, or Q.")
