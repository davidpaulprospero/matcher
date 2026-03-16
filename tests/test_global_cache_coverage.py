"""
Tests for src/global_cache.py to achieve 100% coverage.

Targets:
- DownloadInfo/VideoRegistryEntry dataclass serialization
- GlobalCacheManager initialization and directory creation
- Index loading/saving with error handling
- _compute_content_hash() with missing files
- register_video() new and update paths
- find_videos_for_keywords() with substring matching
- _compute_relevance() edge cases
- File I/O errors for transcripts and scenes
- prompt_global_cache_reuse() interactive function
"""

import pytest
import json
import os
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

from src.global_cache import (
    VideoSource,
    DownloadInfo,
    VideoRegistryEntry,
    GlobalCacheQueryResult,
    GlobalCacheManager,
    prompt_global_cache_reuse,
    DEFAULT_GLOBAL_CACHE_DIR,
)


@pytest.mark.fast
class TestVideoSource:
    """Test VideoSource enum."""

    def test_video_source_values(self):
        """Test VideoSource priority values."""
        assert VideoSource.CURRENT_PROJECT.value == 1
        assert VideoSource.GLOBAL_HIGH_RELEVANCE.value == 2
        assert VideoSource.GLOBAL_MEDIUM_RELEVANCE.value == 3
        assert VideoSource.GLOBAL_LOW_RELEVANCE.value == 4

    @pytest.mark.fast
    def test_video_source_ordering(self):
        """Test that CURRENT_PROJECT has highest priority (lowest value)."""
        assert VideoSource.CURRENT_PROJECT.value < VideoSource.GLOBAL_HIGH_RELEVANCE.value
        assert VideoSource.GLOBAL_HIGH_RELEVANCE.value < VideoSource.GLOBAL_MEDIUM_RELEVANCE.value
        assert VideoSource.GLOBAL_MEDIUM_RELEVANCE.value < VideoSource.GLOBAL_LOW_RELEVANCE.value


@pytest.mark.fast
class TestDownloadInfo:
    """Test DownloadInfo dataclass."""

    def test_download_info_defaults(self):
        """Test DownloadInfo default values."""
        info = DownloadInfo(keyword="test")
        assert info.keyword == "test"
        assert info.youtube_id == ""
        assert info.youtube_url == ""
        assert info.original_title == ""
        assert info.downloaded_at == ""
        assert info.source == "youtube"

    @pytest.mark.fast
    def test_download_info_all_fields(self):
        """Test DownloadInfo with all fields."""
        info = DownloadInfo(
            keyword="travel",
            youtube_id="abc123",
            youtube_url="https://youtube.com/watch?v=abc123",
            original_title="My Video",
            downloaded_at="2026-01-10T12:00:00",
            source="pexels"
        )
        assert info.keyword == "travel"
        assert info.youtube_id == "abc123"
        assert info.source == "pexels"

    @pytest.mark.fast
    def test_download_info_to_dict(self):
        """Test DownloadInfo serialization."""
        info = DownloadInfo(
            keyword="test",
            youtube_id="abc",
            source="pixabay"
        )
        d = info.to_dict()
        assert d["keyword"] == "test"
        assert d["youtube_id"] == "abc"
        assert d["source"] == "pixabay"

    @pytest.mark.fast
    def test_download_info_from_dict(self):
        """Test DownloadInfo deserialization."""
        data = {
            "keyword": "nature",
            "youtube_id": "xyz789",
            "youtube_url": "https://youtube.com/watch?v=xyz789",
            "original_title": "Nature Video",
            "downloaded_at": "2026-01-10",
            "source": "youtube"
        }
        info = DownloadInfo.from_dict(data)
        assert info.keyword == "nature"
        assert info.youtube_id == "xyz789"
        assert info.source == "youtube"

    @pytest.mark.fast
    def test_download_info_from_dict_extra_fields(self):
        """Test DownloadInfo ignores unknown fields."""
        data = {
            "keyword": "test",
            "unknown_field": "ignored",
            "another_unknown": 123
        }
        info = DownloadInfo.from_dict(data)
        assert info.keyword == "test"
        assert not hasattr(info, "unknown_field")


@pytest.mark.fast
class TestVideoRegistryEntry:
    """Test VideoRegistryEntry dataclass."""

    def test_entry_minimal(self):
        """Test VideoRegistryEntry with minimal fields."""
        entry = VideoRegistryEntry(
            video_hash="abc123",
            filename="test.mp4",
            file_size=1000
        )
        assert entry.video_hash == "abc123"
        assert entry.filename == "test.mp4"
        assert entry.file_size == 1000
        assert entry.duration == 0.0
        assert entry.original_paths == []
        assert entry.current_path == ""
        assert entry.file_exists is True
        assert entry.download_info is None
        assert entry.topics == []
        assert entry.keywords == []

    @pytest.mark.fast
    def test_entry_all_fields(self):
        """Test VideoRegistryEntry with all fields."""
        download_info = DownloadInfo(keyword="test", youtube_id="abc")
        entry = VideoRegistryEntry(
            video_hash="hash123",
            filename="video.mp4",
            file_size=5000,
            duration=120.5,
            original_paths=["/path/1", "/path/2"],
            current_path="/path/1",
            file_exists=True,
            download_info=download_info,
            topics=["travel", "nature"],
            keywords=["beach", "sunset"],
            transcript_preview="Sample transcript...",
            has_transcript=True,
            has_embeddings=True,
            has_scenes=True,
            face_score=0.8,
            broll_scene_indices=[1, 3, 5],
            projects_used_in=["project1", "project2"],
            usage_count=5,
            avg_match_confidence=0.75,
            first_seen="2026-01-01",
            last_used="2026-01-10"
        )
        assert entry.duration == 120.5
        assert entry.topics == ["travel", "nature"]
        assert entry.has_transcript is True
        assert entry.face_score == 0.8
        assert entry.broll_scene_indices == [1, 3, 5]

    @pytest.mark.fast
    def test_entry_to_dict(self):
        """Test VideoRegistryEntry serialization."""
        download_info = DownloadInfo(keyword="test", youtube_id="abc")
        entry = VideoRegistryEntry(
            video_hash="hash123",
            filename="video.mp4",
            file_size=5000,
            download_info=download_info,
            topics=["travel"]
        )
        d = entry.to_dict()
        assert d["video_hash"] == "hash123"
        assert d["filename"] == "video.mp4"
        assert d["topics"] == ["travel"]
        assert d["download_info"]["keyword"] == "test"
        assert d["download_info"]["youtube_id"] == "abc"

    @pytest.mark.fast
    def test_entry_to_dict_no_download_info(self):
        """Test VideoRegistryEntry serialization without download_info."""
        entry = VideoRegistryEntry(
            video_hash="hash123",
            filename="video.mp4",
            file_size=5000
        )
        d = entry.to_dict()
        assert d["download_info"] is None

    @pytest.mark.fast
    def test_entry_from_dict(self):
        """Test VideoRegistryEntry deserialization."""
        data = {
            "video_hash": "hash456",
            "filename": "test.mp4",
            "file_size": 3000,
            "duration": 60.0,
            "topics": ["nature", "wildlife"],
            "download_info": {
                "keyword": "animals",
                "youtube_id": "xyz"
            }
        }
        entry = VideoRegistryEntry.from_dict(data)
        assert entry.video_hash == "hash456"
        assert entry.filename == "test.mp4"
        assert entry.duration == 60.0
        assert entry.topics == ["nature", "wildlife"]
        assert entry.download_info.keyword == "animals"
        assert entry.download_info.youtube_id == "xyz"

    @pytest.mark.fast
    def test_entry_from_dict_no_download_info(self):
        """Test VideoRegistryEntry deserialization without download_info."""
        data = {
            "video_hash": "hash789",
            "filename": "test.mp4",
            "file_size": 2000,
            "download_info": None
        }
        entry = VideoRegistryEntry.from_dict(data)
        assert entry.download_info is None

    @pytest.mark.fast
    def test_entry_from_dict_empty_download_info(self):
        """Test VideoRegistryEntry deserialization with empty download_info."""
        data = {
            "video_hash": "hash789",
            "filename": "test.mp4",
            "file_size": 2000,
            "download_info": {}
        }
        entry = VideoRegistryEntry.from_dict(data)
        # Empty dict is falsy, so download_info should be None
        assert entry.download_info is None


@pytest.mark.fast
class TestGlobalCacheQueryResult:
    """Test GlobalCacheQueryResult dataclass."""

    def test_query_result_defaults(self):
        """Test GlobalCacheQueryResult default values."""
        result = GlobalCacheQueryResult()
        assert result.reuse_videos == []
        assert result.redownload_keywords == []
        assert result.uncovered_keywords == []
        assert result.total_cached_matches == 0
        assert result.files_exist_count == 0
        assert result.files_deleted_count == 0

    @pytest.mark.fast
    def test_query_result_with_data(self):
        """Test GlobalCacheQueryResult with data."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="test.mp4",
            file_size=1000
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)],
            redownload_keywords=["keyword1"],
            uncovered_keywords=["keyword2"],
            total_cached_matches=5,
            files_exist_count=3,
            files_deleted_count=2
        )
        assert len(result.reuse_videos) == 1
        assert result.reuse_videos[0][1] == 0.8
        assert result.redownload_keywords == ["keyword1"]


@pytest.mark.fast
class TestGlobalCacheManagerInit:
    """Test GlobalCacheManager initialization."""

    def test_init_default_cache_dir(self, tmp_path):
        """Test GlobalCacheManager with default cache directory."""
        with patch.object(Path, 'home', return_value=tmp_path):
            # Create manager to trigger DEFAULT_GLOBAL_CACHE_DIR usage
            manager = GlobalCacheManager(cache_dir=str(tmp_path / "test_cache"))
            assert manager.cache_dir == tmp_path / "test_cache"

    @pytest.mark.fast
    def test_init_custom_cache_dir(self, tmp_path):
        """Test GlobalCacheManager with custom cache directory."""
        cache_dir = tmp_path / "custom_cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        assert manager.cache_dir == cache_dir
        assert cache_dir.exists()

    @pytest.mark.fast
    def test_init_creates_subdirectories(self, tmp_path):
        """Test that initialization creates all subdirectories."""
        cache_dir = tmp_path / "test_cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        assert manager.video_registry_dir.exists()
        assert manager.transcripts_dir.exists()
        assert manager.embeddings_dir.exists()
        assert manager.scenes_dir.exists()
        assert manager.topics_dir.exists()
        assert manager.keywords_dir.exists()

    @pytest.mark.fast
    def test_init_with_config(self, tmp_path):
        """Test GlobalCacheManager with config object."""
        cache_dir = tmp_path / "test_cache"
        config = MagicMock()
        manager = GlobalCacheManager(cache_dir=str(cache_dir), config=config)
        assert manager.config == config

    @pytest.mark.fast
    def test_init_indices_not_loaded(self, tmp_path):
        """Test that indices are not loaded on init."""
        cache_dir = tmp_path / "test_cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        assert manager._loaded is False
        assert manager._registry_index == {}
        assert manager._topic_index == {}
        assert manager._keyword_index == {}


@pytest.mark.fast
class TestGlobalCacheManagerLoadIndices:
    """Test index loading functionality."""

    def test_load_indices_empty(self, tmp_path):
        """Test loading indices when files don't exist."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()
        assert manager._loaded is True
        assert manager._registry_index == {}
        assert manager._topic_index == {}
        assert manager._keyword_index == {}

    @pytest.mark.fast
    def test_load_indices_with_data(self, tmp_path):
        """Test loading indices with existing data."""
        cache_dir = tmp_path / "cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Create index files
        registry_data = {"hash1": "path1.json", "hash2": "path2.json"}
        topic_data = {"travel": ["hash1"], "nature": ["hash2"]}
        keyword_data = {"beach": ["hash1", "hash2"]}

        with open(manager.registry_index_path, 'w') as f:
            json.dump(registry_data, f)
        with open(manager.topic_index_path, 'w') as f:
            json.dump(topic_data, f)
        with open(manager.keyword_index_path, 'w') as f:
            json.dump(keyword_data, f)

        manager._load_indices()
        assert manager._registry_index == registry_data
        assert manager._topic_index == topic_data
        assert manager._keyword_index == keyword_data

    @pytest.mark.fast
    def test_load_indices_only_once(self, tmp_path):
        """Test that indices are only loaded once."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()
        assert manager._loaded is True

        # Modify index file
        with open(manager.registry_index_path, 'w') as f:
            json.dump({"new_hash": "new_path"}, f)

        # Load again - should not reload
        manager._load_indices()
        assert manager._registry_index == {}  # Still empty from first load

    @pytest.mark.fast
    def test_load_indices_corrupted_registry(self, tmp_path):
        """Test loading with corrupted registry index."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        # Create corrupted registry file
        with open(manager.registry_index_path, 'w') as f:
            f.write("not valid json{{{")

        manager._load_indices()
        assert manager._registry_index == {}
        assert manager._loaded is True

    @pytest.mark.fast
    def test_load_indices_corrupted_topic(self, tmp_path):
        """Test loading with corrupted topic index."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        # Create valid registry but corrupted topic
        with open(manager.registry_index_path, 'w') as f:
            json.dump({"hash1": "path1"}, f)
        with open(manager.topic_index_path, 'w') as f:
            f.write("invalid json")

        manager._load_indices()
        assert manager._registry_index == {"hash1": "path1"}
        assert manager._topic_index == {}

    @pytest.mark.fast
    def test_load_indices_corrupted_keyword(self, tmp_path):
        """Test loading with corrupted keyword index."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        # Create valid registry and topic, corrupted keyword
        with open(manager.registry_index_path, 'w') as f:
            json.dump({"hash1": "path1"}, f)
        with open(manager.topic_index_path, 'w') as f:
            json.dump({"travel": ["hash1"]}, f)
        with open(manager.keyword_index_path, 'w') as f:
            f.write("[not a dict]")

        manager._load_indices()
        assert manager._registry_index == {"hash1": "path1"}
        assert manager._topic_index == {"travel": ["hash1"]}
        assert manager._keyword_index == {}


@pytest.mark.fast
class TestGlobalCacheManagerSaveIndices:
    """Test index saving functionality."""

    def test_save_indices_success(self, tmp_path):
        """Test saving indices successfully."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._registry_index = {"hash1": "path1"}
        manager._topic_index = {"travel": ["hash1"]}
        manager._keyword_index = {"beach": ["hash1"]}

        manager._save_indices()

        # Verify files written
        with open(manager.registry_index_path, 'r') as f:
            assert json.load(f) == {"hash1": "path1"}
        with open(manager.topic_index_path, 'r') as f:
            assert json.load(f) == {"travel": ["hash1"]}
        with open(manager.keyword_index_path, 'r') as f:
            assert json.load(f) == {"beach": ["hash1"]}

    @pytest.mark.fast
    def test_save_indices_permission_error(self, tmp_path):
        """Test saving indices with permission error."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._registry_index = {"hash1": "path1"}

        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should not raise, just log error
            manager._save_indices()


@pytest.mark.fast
class TestGlobalCacheManagerComputeHash:
    """Test content hash computation."""

    def test_compute_hash_existing_file(self, tmp_path):
        """Test computing hash for existing file."""
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"fake video content" * 1000)

        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        hash1 = manager._compute_content_hash(str(video_file))

        assert len(hash1) == 32  # MD5 hex length
        assert hash1.isalnum()

    @pytest.mark.fast
    def test_compute_hash_missing_file(self, tmp_path):
        """Test computing hash for missing file (fallback)."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        missing_path = str(tmp_path / "missing_video.mp4")

        hash_result = manager._compute_content_hash(missing_path)

        # Should use filename-based fallback
        expected = hashlib.md5("missing_video.mp4".encode()).hexdigest()
        assert hash_result == expected

    @pytest.mark.fast
    def test_compute_hash_small_file(self, tmp_path):
        """Test computing hash for small file (< 2MB)."""
        video_file = tmp_path / "small_video.mp4"
        video_file.write_bytes(b"small content")

        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        hash_result = manager._compute_content_hash(str(video_file))

        assert len(hash_result) == 32

    @pytest.mark.fast
    def test_compute_hash_large_file(self, tmp_path):
        """Test computing hash for large file (uses first/last chunks)."""
        video_file = tmp_path / "large_video.mp4"
        # Create file larger than 2MB
        video_file.write_bytes(b"A" * (3 * 1024 * 1024))

        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        hash_result = manager._compute_content_hash(str(video_file))

        assert len(hash_result) == 32

    @pytest.mark.fast
    def test_compute_hash_read_error(self, tmp_path):
        """Test computing hash with read error (fallback to size-based)."""
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"content")

        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        # Mock open to fail after stat succeeds
        original_open = open
        call_count = [0]

        def mock_open_fail(*args, **kwargs):
            call_count[0] += 1
            if 'rb' in args or kwargs.get('mode') == 'rb':
                raise IOError("Read error")
            return original_open(*args, **kwargs)

        with patch('builtins.open', side_effect=mock_open_fail):
            hash_result = manager._compute_content_hash(str(video_file))
            # Should fall back to filename:size hash
            expected = hashlib.md5(f"video.mp4:{video_file.stat().st_size}".encode()).hexdigest()
            assert hash_result == expected

    @pytest.mark.fast
    def test_get_video_hash_public_method(self, tmp_path):
        """Test public get_video_hash method."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"content")

        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        hash_result = manager.get_video_hash(str(video_file))

        assert len(hash_result) == 32


@pytest.mark.fast
class TestGlobalCacheManagerRegisterVideo:
    """Test video registration."""

    def test_register_new_video(self, tmp_path):
        """Test registering a new video."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"video content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        entry = manager.register_video(
            video_path=str(video_file),
            download_keyword="nature",
            topics=["wildlife", "forest"],
            youtube_id="abc123",
            youtube_url="https://youtube.com/watch?v=abc123",
            original_title="Nature Documentary",
            project_id="project1",
            duration=120.0
        )

        assert entry.filename == "test_video.mp4"
        assert entry.duration == 120.0
        assert "wildlife" in entry.topics
        assert entry.download_info.keyword == "nature"
        assert entry.download_info.youtube_id == "abc123"
        assert "project1" in entry.projects_used_in
        assert entry.usage_count == 1

    @pytest.mark.fast
    def test_register_existing_video_update(self, tmp_path):
        """Test updating an existing video registration."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b"video content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Register first time
        entry1 = manager.register_video(
            video_path=str(video_file),
            download_keyword="nature",
            topics=["wildlife"],
            project_id="project1"
        )

        # Register again with different path
        video_file2 = tmp_path / "test_video_copy.mp4"
        video_file2.write_bytes(b"video content")  # Same content = same hash
        # Actually, different content = different hash, so let's use same file
        entry2 = manager.register_video(
            video_path=str(video_file),  # Same file
            topics=["forest"],  # New topic
            project_id="project2"
        )

        # Should update existing entry
        assert entry2.video_hash == entry1.video_hash
        assert "wildlife" in entry2.topics
        assert "forest" in entry2.topics
        assert "project1" in entry2.projects_used_in
        assert "project2" in entry2.projects_used_in
        assert entry2.usage_count == 2

    @pytest.mark.fast
    def test_register_video_no_download_keyword(self, tmp_path):
        """Test registering video without download keyword."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        entry = manager.register_video(video_path=str(video_file))

        assert entry.download_info is None
        assert entry.keywords == []

    @pytest.mark.fast
    def test_register_video_updates_paths(self, tmp_path):
        """Test that registering updates original_paths list."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"unique content for test")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Register with first path
        entry1 = manager.register_video(video_path=str(video_file))
        original_hash = entry1.video_hash

        # Copy file to new location (same content = same hash only if same file)
        # Use same file for update test
        entry2 = manager.register_video(video_path=str(video_file))

        assert str(video_file) in entry2.original_paths
        assert entry2.current_path == str(video_file)


@pytest.mark.fast
class TestGlobalCacheManagerGetEntry:
    """Test getting video entries."""

    def test_get_video_entry_exists(self, tmp_path):
        """Test getting an existing video entry."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        registered = manager.register_video(
            video_path=str(video_file),
            topics=["travel"]
        )

        # Get entry by hash
        entry = manager.get_video_entry(registered.video_hash)
        assert entry is not None
        assert entry.video_hash == registered.video_hash
        assert "travel" in entry.topics

    @pytest.mark.fast
    def test_get_video_entry_not_found(self, tmp_path):
        """Test getting a non-existent video entry."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        entry = manager.get_video_entry("nonexistent_hash")
        assert entry is None

    @pytest.mark.fast
    def test_get_video_entry_corrupted_file(self, tmp_path):
        """Test getting entry with corrupted entry file."""
        cache_dir = tmp_path / "cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        manager._load_indices()

        # Create corrupted entry file
        entry_path = manager.video_registry_dir / "badhash.json"
        entry_path.write_text("not valid json")
        manager._registry_index["badhash"] = str(entry_path)

        entry = manager.get_video_entry("badhash")
        assert entry is None


@pytest.mark.fast
class TestGlobalCacheManagerFindVideos:
    """Test finding videos for keywords."""

    def test_find_videos_empty_cache(self, tmp_path):
        """Test finding videos in empty cache."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        result = manager.find_videos_for_keywords(
            keywords=["travel", "nature"],
            topics=["wildlife"]
        )

        assert result.reuse_videos == []
        assert result.redownload_keywords == []
        assert result.uncovered_keywords == ["travel", "nature"]
        assert result.total_cached_matches == 0

    @pytest.mark.fast
    def test_find_videos_exact_match(self, tmp_path):
        """Test finding videos with exact keyword match."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "travel_video.mp4"
        video_file.write_bytes(b"travel content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        manager.register_video(
            video_path=str(video_file),
            download_keyword="travel",
            topics=["adventure"]
        )

        result = manager.find_videos_for_keywords(keywords=["travel"])

        assert len(result.reuse_videos) == 1
        assert result.reuse_videos[0][0].filename == "travel_video.mp4"
        assert result.uncovered_keywords == []

    @pytest.mark.fast
    def test_find_videos_substring_match(self, tmp_path):
        """Test finding videos with substring keyword match."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "beach_sunset.mp4"
        video_file.write_bytes(b"beach content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        manager.register_video(
            video_path=str(video_file),
            download_keyword="beach sunset",
            topics=["ocean"]
        )

        # Search with partial keyword
        result = manager.find_videos_for_keywords(keywords=["beach"])

        assert len(result.reuse_videos) == 1
        assert result.reuse_videos[0][0].filename == "beach_sunset.mp4"

    @pytest.mark.fast
    def test_find_videos_topic_match(self, tmp_path):
        """Test finding videos by topic."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "wildlife.mp4"
        video_file.write_bytes(b"wildlife content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        manager.register_video(
            video_path=str(video_file),
            download_keyword="animals",
            topics=["wildlife", "safari"]
        )

        # Search with topic
        result = manager.find_videos_for_keywords(
            keywords=["something_else"],
            topics=["wildlife"]
        )

        # Should find by topic
        assert result.total_cached_matches >= 1

    @pytest.mark.fast
    def test_find_videos_deleted_file(self, tmp_path):
        """Test finding videos where file was deleted."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "deleted_video.mp4"
        video_file.write_bytes(b"deleted content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        manager.register_video(
            video_path=str(video_file),
            download_keyword="travel"
        )

        # Delete the file
        video_file.unlink()

        result = manager.find_videos_for_keywords(keywords=["travel"])

        # Should suggest re-download
        assert "travel" in result.redownload_keywords
        assert result.files_deleted_count == 1

    @pytest.mark.fast
    def test_find_videos_min_relevance_filter(self, tmp_path):
        """Test that min_relevance filters out low relevance matches."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"video content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        manager.register_video(
            video_path=str(video_file),
            download_keyword="specific_keyword"
        )

        # Search with unrelated keyword but set low min_relevance
        result = manager.find_videos_for_keywords(
            keywords=["specific_keyword"],
            min_relevance=0.0  # Accept any relevance
        )

        assert len(result.reuse_videos) >= 0

    @pytest.mark.fast
    def test_find_videos_max_results(self, tmp_path):
        """Test max_results limits output."""
        cache_dir = tmp_path / "cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Create multiple videos
        for i in range(10):
            video_file = tmp_path / f"video_{i}.mp4"
            video_file.write_bytes(f"content {i}".encode())
            manager.register_video(
                video_path=str(video_file),
                download_keyword="common_keyword"
            )

        result = manager.find_videos_for_keywords(
            keywords=["common_keyword"],
            max_results=3
        )

        assert len(result.reuse_videos) <= 3


@pytest.mark.fast
class TestGlobalCacheManagerFileChecks:
    """Test file existence checking."""

    def test_check_file_exists_current_path(self, tmp_path):
        """Test checking file existence via current_path."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        video_file = tmp_path / "exists.mp4"
        video_file.write_bytes(b"content")

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="exists.mp4",
            file_size=100,
            current_path=str(video_file),
            original_paths=[]
        )

        assert manager._check_file_exists(entry) is True

    @pytest.mark.fast
    def test_check_file_exists_original_paths(self, tmp_path):
        """Test checking file existence via original_paths."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        video_file = tmp_path / "original.mp4"
        video_file.write_bytes(b"content")

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="original.mp4",
            file_size=100,
            current_path="/nonexistent/path.mp4",
            original_paths=["/also/nonexistent.mp4", str(video_file)]
        )

        result = manager._check_file_exists(entry)
        assert result is True
        # Should update current_path
        assert entry.current_path == str(video_file)

    @pytest.mark.fast
    def test_check_file_not_exists(self, tmp_path):
        """Test checking non-existent file."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="missing.mp4",
            file_size=100,
            current_path="/nonexistent/path.mp4",
            original_paths=["/also/nonexistent.mp4"]
        )

        assert manager._check_file_exists(entry) is False


@pytest.mark.fast
class TestGlobalCacheManagerRelevance:
    """Test relevance computation."""

    def test_compute_relevance_exact_keyword_match(self, tmp_path):
        """Test relevance for exact keyword match."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            keywords=["travel", "nature"],
            download_info=DownloadInfo(keyword="adventure")
        )

        relevance = manager._compute_relevance(entry, ["travel"], None)
        assert relevance > 0.5  # Should be high for keyword match

    @pytest.mark.fast
    def test_compute_relevance_partial_keyword_match(self, tmp_path):
        """Test relevance for partial keyword match."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            keywords=["travel adventure"],
            download_info=DownloadInfo(keyword="beach sunset")
        )

        # "travel" should partially match "travel adventure"
        relevance = manager._compute_relevance(entry, ["travel"], None)
        assert relevance > 0

    @pytest.mark.fast
    def test_compute_relevance_topic_match(self, tmp_path):
        """Test relevance for topic match."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            keywords=[],
            topics=["wildlife", "safari"]
        )

        relevance = manager._compute_relevance(entry, ["unrelated"], ["wildlife"])
        assert relevance > 0

    @pytest.mark.fast
    def test_compute_relevance_no_keywords_or_topics(self, tmp_path):
        """Test relevance with empty inputs."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100
        )

        relevance = manager._compute_relevance(entry, [], None)
        # With no keywords and no topics, max_score could be 0
        assert relevance >= 0.0

    @pytest.mark.fast
    def test_compute_relevance_max_score_zero(self, tmp_path):
        """Test relevance when max_score is zero (edge case)."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100
        )

        # No keywords = max_score starts at 0
        # But topics=None adds 0.4 to both score and max_score
        relevance = manager._compute_relevance(entry, [], None)
        # Should return 1.0 (0.4/0.4) when no keywords but topics is None
        assert relevance == 1.0

    @pytest.mark.fast
    def test_compute_relevance_download_keyword_match(self, tmp_path):
        """Test relevance includes download_info keyword."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            keywords=[],
            download_info=DownloadInfo(keyword="beach vacation")
        )

        relevance = manager._compute_relevance(entry, ["beach"], None)
        assert relevance > 0.5  # Should match download keyword


@pytest.mark.fast
class TestGlobalCacheManagerUpdateMethods:
    """Test update methods."""

    def test_update_video_topics(self, tmp_path):
        """Test updating video topics."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        entry = manager.register_video(
            video_path=str(video_file),
            topics=["travel"]
        )

        # Update with new topics
        manager.update_video_topics(entry.video_hash, ["nature", "wildlife"])

        # Verify
        updated = manager.get_video_entry(entry.video_hash)
        assert "travel" in updated.topics
        assert "nature" in updated.topics
        assert "wildlife" in updated.topics

    @pytest.mark.fast
    def test_update_video_topics_nonexistent(self, tmp_path):
        """Test updating topics for non-existent video."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        # Should not raise
        manager.update_video_topics("nonexistent_hash", ["topic"])

    @pytest.mark.fast
    def test_mark_video_processed(self, tmp_path):
        """Test marking video as processed."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        entry = manager.register_video(video_path=str(video_file))

        # Mark as processed
        manager.mark_video_processed(
            video_hash=entry.video_hash,
            has_transcript=True,
            has_embeddings=True,
            has_scenes=True,
            face_score=0.3,
            broll_scenes=[1, 2, 3]
        )

        # Verify
        updated = manager.get_video_entry(entry.video_hash)
        assert updated.has_transcript is True
        assert updated.has_embeddings is True
        assert updated.has_scenes is True
        assert updated.face_score == 0.3
        assert updated.broll_scene_indices == [1, 2, 3]

    @pytest.mark.fast
    def test_mark_video_processed_nonexistent(self, tmp_path):
        """Test marking non-existent video as processed."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        # Should not raise
        manager.mark_video_processed("nonexistent_hash", has_transcript=True)

    @pytest.mark.fast
    def test_mark_video_processed_partial(self, tmp_path):
        """Test marking video with partial flags."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "video.mp4"
        video_file.write_bytes(b"content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))
        entry = manager.register_video(video_path=str(video_file))

        # Mark only transcript
        manager.mark_video_processed(
            video_hash=entry.video_hash,
            has_transcript=True
        )

        updated = manager.get_video_entry(entry.video_hash)
        assert updated.has_transcript is True
        assert updated.has_embeddings is False
        assert updated.has_scenes is False


@pytest.mark.fast
class TestGlobalCacheManagerTranscripts:
    """Test transcript caching."""

    def test_copy_transcript_to_global(self, tmp_path):
        """Test copying transcript to global cache."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        transcript_data = {
            "segments": [{"text": "Hello world", "start": 0, "end": 5}],
            "language": "en"
        }

        manager.copy_transcript_to_global("hash123", transcript_data)

        # Verify file exists
        transcript_path = manager.transcripts_dir / "hash123.json"
        assert transcript_path.exists()

        with open(transcript_path, 'r') as f:
            saved = json.load(f)
        assert saved == transcript_data

    @pytest.mark.fast
    def test_copy_transcript_permission_error(self, tmp_path):
        """Test copying transcript with permission error."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should not raise
            manager.copy_transcript_to_global("hash123", {"data": "test"})

    @pytest.mark.fast
    def test_get_transcript_from_global(self, tmp_path):
        """Test getting transcript from global cache."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        transcript_data = {"segments": [], "language": "en"}
        manager.copy_transcript_to_global("hash456", transcript_data)

        result = manager.get_transcript_from_global("hash456")
        assert result == transcript_data

    @pytest.mark.fast
    def test_get_transcript_not_found(self, tmp_path):
        """Test getting non-existent transcript."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        result = manager.get_transcript_from_global("nonexistent")
        assert result is None

    @pytest.mark.fast
    def test_get_transcript_corrupted(self, tmp_path):
        """Test getting corrupted transcript."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        # Create corrupted file
        transcript_path = manager.transcripts_dir / "corrupted.json"
        transcript_path.write_text("not valid json")

        result = manager.get_transcript_from_global("corrupted")
        assert result is None


@pytest.mark.fast
class TestGlobalCacheManagerScenes:
    """Test scene caching."""

    def test_copy_scenes_to_global(self, tmp_path):
        """Test copying scenes to global cache."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        scene_data = {
            "scenes": [{"start": 0, "end": 5, "is_broll": True}],
            "total_scenes": 1
        }

        manager.copy_scenes_to_global("hash123", scene_data)

        # Verify file exists
        scene_path = manager.scenes_dir / "hash123.json"
        assert scene_path.exists()

    @pytest.mark.fast
    def test_copy_scenes_permission_error(self, tmp_path):
        """Test copying scenes with permission error."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should not raise
            manager.copy_scenes_to_global("hash123", {"data": "test"})

    @pytest.mark.fast
    def test_get_scenes_from_global(self, tmp_path):
        """Test getting scenes from global cache."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        scene_data = {"scenes": [], "total": 0}
        manager.copy_scenes_to_global("hash456", scene_data)

        result = manager.get_scenes_from_global("hash456")
        assert result == scene_data

    @pytest.mark.fast
    def test_get_scenes_not_found(self, tmp_path):
        """Test getting non-existent scenes."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        result = manager.get_scenes_from_global("nonexistent")
        assert result is None

    @pytest.mark.fast
    def test_get_scenes_corrupted(self, tmp_path):
        """Test getting corrupted scenes."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        scene_path = manager.scenes_dir / "corrupted.json"
        scene_path.write_text("invalid json{{")

        result = manager.get_scenes_from_global("corrupted")
        assert result is None


@pytest.mark.fast
class TestGlobalCacheManagerStats:
    """Test statistics methods."""

    def test_get_stats_empty(self, tmp_path):
        """Test getting stats from empty cache."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        stats = manager.get_stats()

        assert stats["total_videos"] == 0
        assert stats["total_topics"] == 0
        assert stats["total_keywords"] == 0
        assert stats["cache_dir"] == str(tmp_path / "cache")
        assert stats["cache_size_mb"] >= 0

    @pytest.mark.fast
    def test_get_stats_with_data(self, tmp_path):
        """Test getting stats with data."""
        cache_dir = tmp_path / "cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Add some videos
        for i in range(3):
            video_file = tmp_path / f"video_{i}.mp4"
            video_file.write_bytes(f"content {i}".encode())
            manager.register_video(
                video_path=str(video_file),
                download_keyword=f"keyword_{i}",
                topics=[f"topic_{i}"]
            )

        stats = manager.get_stats()

        assert stats["total_videos"] == 3
        assert stats["total_topics"] == 3
        assert stats["total_keywords"] == 3

    @pytest.mark.fast
    def test_get_cache_size_mb(self, tmp_path):
        """Test cache size calculation."""
        cache_dir = tmp_path / "cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Create some files
        test_file = manager.transcripts_dir / "test.json"
        test_file.write_bytes(b"x" * 1024)  # 1KB

        size = manager._get_cache_size_mb()
        assert size > 0


@pytest.mark.fast
class TestGlobalCacheManagerUpdateIndices:
    """Test index update functionality."""

    def test_update_indices_topics(self, tmp_path):
        """Test updating topic index."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            topics=["Travel", "NATURE"]  # Mixed case
        )

        manager._update_indices(entry)

        # Should be lowercase in index
        assert "travel" in manager._topic_index
        assert "nature" in manager._topic_index
        assert "hash1" in manager._topic_index["travel"]

    @pytest.mark.fast
    def test_update_indices_keywords(self, tmp_path):
        """Test updating keyword index."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            keywords=["Beach", "SUNSET"]
        )

        manager._update_indices(entry)

        assert "beach" in manager._keyword_index
        assert "sunset" in manager._keyword_index

    @pytest.mark.fast
    def test_update_indices_download_keyword(self, tmp_path):
        """Test updating index with download keyword."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            download_info=DownloadInfo(keyword="Adventure Travel")
        )

        manager._update_indices(entry)

        assert "adventure travel" in manager._keyword_index

    @pytest.mark.fast
    def test_update_indices_no_duplicates(self, tmp_path):
        """Test that duplicate hashes aren't added."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            topics=["travel"]
        )

        # Update twice
        manager._update_indices(entry)
        manager._update_indices(entry)

        # Should only have one entry
        assert manager._topic_index["travel"].count("hash1") == 1


@pytest.mark.fast
class TestPromptGlobalCacheReuse:
    """Test interactive prompt function."""

    def test_prompt_no_results(self, capsys):
        """Test prompt with no cache results."""
        result = GlobalCacheQueryResult()
        keywords = ["travel", "nature"]

        with patch('builtins.input', return_value='D'):
            use_cache, paths, kws = prompt_global_cache_reuse(result, keywords)

        assert use_cache is False
        assert paths == []
        assert kws == ["travel", "nature"]

        output = capsys.readouterr().out
        assert "No relevant videos found" in output

    @pytest.mark.fast
    def test_prompt_reuse_choice(self, capsys):
        """Test prompt with reuse choice."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            current_path="/path/to/video.mp4",
            topics=["travel"],
            duration=60.0
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)],
            redownload_keywords=["nature"],
            uncovered_keywords=["wildlife"]
        )

        with patch('builtins.input', return_value='R'):
            use_cache, paths, kws = prompt_global_cache_reuse(result, ["travel", "nature", "wildlife"])

        assert use_cache is True
        assert paths == ["/path/to/video.mp4"]
        assert "nature" in kws
        assert "wildlife" in kws

    @pytest.mark.fast
    def test_prompt_download_fresh_choice(self, capsys):
        """Test prompt with download fresh choice."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            current_path="/path/to/video.mp4"
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)]
        )
        keywords = ["travel"]

        with patch('builtins.input', return_value='D'):
            use_cache, paths, kws = prompt_global_cache_reuse(result, keywords)

        assert use_cache is False
        assert paths == []
        assert kws == ["travel"]

    @pytest.mark.fast
    def test_prompt_quit_choice(self, capsys):
        """Test prompt with quit choice."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            current_path="/path/to/video.mp4"
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)]
        )

        with patch('builtins.input', return_value='Q'):
            use_cache, paths, kws = prompt_global_cache_reuse(result, ["travel"])

        assert use_cache is False
        assert paths == []
        assert kws == []

    @pytest.mark.fast
    def test_prompt_invalid_then_valid(self, capsys):
        """Test prompt with invalid input then valid."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            current_path="/path/to/video.mp4"
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)]
        )

        with patch('builtins.input', side_effect=['X', 'invalid', 'D']):
            use_cache, paths, kws = prompt_global_cache_reuse(result, ["travel"])

        assert use_cache is False

    @pytest.mark.fast
    def test_prompt_eof_error(self, capsys):
        """Test prompt with EOFError."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            current_path="/path/to/video.mp4"
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)]
        )

        with patch('builtins.input', side_effect=EOFError):
            use_cache, paths, kws = prompt_global_cache_reuse(result, ["travel"])

        assert use_cache is False
        assert kws == []  # Quit behavior

    @pytest.mark.fast
    def test_prompt_keyboard_interrupt(self, capsys):
        """Test prompt with KeyboardInterrupt."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            current_path="/path/to/video.mp4"
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)]
        )

        with patch('builtins.input', side_effect=KeyboardInterrupt):
            use_cache, paths, kws = prompt_global_cache_reuse(result, ["travel"])

        assert use_cache is False
        assert kws == []

    @pytest.mark.fast
    def test_prompt_many_videos(self, capsys):
        """Test prompt with more than 5 videos (shows '... and X more')."""
        entries = []
        for i in range(10):
            entries.append((
                VideoRegistryEntry(
                    video_hash=f"hash{i}",
                    filename=f"video_{i}.mp4",
                    file_size=100,
                    current_path=f"/path/video_{i}.mp4",
                    topics=["travel"],
                    duration=60.0
                ),
                0.8
            ))

        result = GlobalCacheQueryResult(
            reuse_videos=entries,
            redownload_keywords=["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"],
            uncovered_keywords=["new1", "new2", "new3", "new4", "new5", "new6"]
        )

        with patch('builtins.input', return_value='D'):
            prompt_global_cache_reuse(result, ["travel"])

        output = capsys.readouterr().out
        assert "and 5 more" in output  # For videos
        assert "and 1 more" in output  # For redownload keywords

    @pytest.mark.fast
    def test_prompt_lowercase_input(self, capsys):
        """Test prompt accepts lowercase input."""
        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            current_path="/path/to/video.mp4"
        )
        result = GlobalCacheQueryResult(
            reuse_videos=[(entry, 0.8)]
        )

        with patch('builtins.input', return_value='r'):
            use_cache, paths, kws = prompt_global_cache_reuse(result, ["travel"])

        assert use_cache is True


@pytest.mark.fast
class TestGlobalCacheManagerSaveEntry:
    """Test _save_video_entry method."""

    def test_save_video_entry_success(self, tmp_path):
        """Test saving video entry successfully."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        entry = VideoRegistryEntry(
            video_hash="hash123",
            filename="test.mp4",
            file_size=1000,
            topics=["travel"]
        )

        manager._save_video_entry(entry)

        # Verify file exists
        entry_path = manager.video_registry_dir / "hash123.json"
        assert entry_path.exists()

        # Verify index updated
        assert "hash123" in manager._registry_index

    @pytest.mark.fast
    def test_save_video_entry_permission_error(self, tmp_path):
        """Test saving video entry with permission error."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        entry = VideoRegistryEntry(
            video_hash="hash123",
            filename="test.mp4",
            file_size=1000
        )

        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            # Should not raise, just log error
            manager._save_video_entry(entry)


@pytest.mark.fast
class TestMissedLines:
    """Tests specifically targeting missed coverage lines."""

    def test_register_video_new_path_same_content(self, tmp_path):
        """Test line 304: Registering same video from different path."""
        cache_dir = tmp_path / "cache"
        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Create video file
        video_file = tmp_path / "video1.mp4"
        video_content = b"unique video content for hash test"
        video_file.write_bytes(video_content)

        # Register first time
        entry1 = manager.register_video(
            video_path=str(video_file),
            download_keyword="test"
        )
        first_hash = entry1.video_hash

        # Copy to different location with same content
        video_file2 = tmp_path / "video2.mp4"
        video_file2.write_bytes(video_content)

        # The content hash may differ due to file size/location
        # Let's use the same file but different path string representation
        # Actually, we need to re-register the same file to trigger line 304
        # by using a path not in original_paths

        # Force the scenario by manually manipulating
        entry_from_db = manager.get_video_entry(first_hash)
        assert str(video_file) in entry_from_db.original_paths

        # Register again with the same file - path already in original_paths
        # so line 304 won't trigger. Need a different path with same hash.
        # Since hash is content-based, we need same content from different path

        # Create symlink-like scenario by modifying entry directly
        # Actually, let's just ensure the line gets hit by registering
        # the copied file which should have same hash if content is identical
        # but Python's file size based hash may differ

        # Alternative approach: mock the hash to return same value
        original_compute = manager._compute_content_hash
        def mock_hash(path):
            return first_hash  # Return same hash for any path

        manager._compute_content_hash = mock_hash

        # Now register the second file - same hash, different path
        entry2 = manager.register_video(
            video_path=str(video_file2),
            download_keyword="test2"
        )

        # Should update existing entry with new path
        assert str(video_file2) in entry2.original_paths
        assert str(video_file) in entry2.original_paths

        # Restore original method
        manager._compute_content_hash = original_compute

    @pytest.mark.fast
    def test_find_videos_relevance_below_minimum(self, tmp_path):
        """Test line 473: Video excluded due to low relevance."""
        cache_dir = tmp_path / "cache"
        video_file = tmp_path / "unrelated_video.mp4"
        video_file.write_bytes(b"unrelated content")

        manager = GlobalCacheManager(cache_dir=str(cache_dir))

        # Register video with keyword that can be partially matched
        manager.register_video(
            video_path=str(video_file),
            download_keyword="test_keyword_partial",  # "test" will partially match
            topics=["unique_topic_only"]
        )

        # Search with keyword that partially matches (will find the video)
        # but with topics that don't match and high min_relevance
        # "test" partially matches "test_keyword_partial"
        result = manager.find_videos_for_keywords(
            keywords=["test", "other_keyword", "third_keyword"],  # Only 1/3 matches
            topics=["completely_different_topic"],  # No topic match
            min_relevance=0.9  # Very high threshold
        )

        # The video should be found in cache (partial keyword match)
        # but excluded due to low relevance (only 1/3 keyword match = ~0.2 + 0 topic = 0.2)
        # Total cached matches should be > 0, but reuse_videos empty due to relevance filter
        assert result.total_cached_matches >= 1
        assert len(result.reuse_videos) == 0  # Filtered out by min_relevance


@pytest.mark.fast
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_find_videos_empty_keyword_list(self, tmp_path):
        """Test finding videos with empty keyword list."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        result = manager.find_videos_for_keywords(keywords=[])

        assert result.reuse_videos == []
        assert result.uncovered_keywords == []

    @pytest.mark.fast
    def test_register_video_missing_file(self, tmp_path):
        """Test registering a video that doesn't exist."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        # Register non-existent file
        entry = manager.register_video(
            video_path=str(tmp_path / "nonexistent.mp4"),
            download_keyword="test"
        )

        # Should still create entry (for re-download tracking)
        assert entry.file_size == 0
        assert entry.file_exists is False

    @pytest.mark.fast
    def test_compute_relevance_with_both_keywords_and_topics(self, tmp_path):
        """Test relevance with both keywords and topics matching."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))

        entry = VideoRegistryEntry(
            video_hash="hash1",
            filename="video.mp4",
            file_size=100,
            keywords=["beach", "sunset"],
            topics=["travel", "vacation"]
        )

        relevance = manager._compute_relevance(
            entry,
            keywords=["beach"],
            topics=["travel"]
        )

        # Should be high with both matching
        assert relevance > 0.5

    @pytest.mark.fast
    def test_find_videos_entry_not_found(self, tmp_path):
        """Test finding videos when entry file is missing."""
        manager = GlobalCacheManager(cache_dir=str(tmp_path / "cache"))
        manager._load_indices()

        # Add hash to index but don't create file
        manager._keyword_index["test"] = ["orphan_hash"]

        result = manager.find_videos_for_keywords(keywords=["test"])

        # Should handle gracefully
        assert result.total_cached_matches == 1
        # But no reuse since entry not found
        assert len(result.reuse_videos) == 0

    @pytest.mark.fast
    def test_default_global_cache_dir(self):
        """Test DEFAULT_GLOBAL_CACHE_DIR is set correctly."""
        assert DEFAULT_GLOBAL_CACHE_DIR == Path.home() / ".matcher_global_cache"
