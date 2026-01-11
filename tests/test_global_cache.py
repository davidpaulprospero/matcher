"""
Comprehensive tests for global cache module.

Covers:
- GlobalCacheManager initialization
- Video registration and tracking
- Content-based video hashing
- Cross-project video indexing
- Topic and keyword indices
- Video query and relevance scoring
- File existence checking
- Transcript/scene data sharing
- Cache statistics
- DownloadInfo and VideoRegistryEntry dataclasses

Created: 2026-01-09 (Phase 4.1)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import shutil
import json
import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.global_cache import (
    GlobalCacheManager,
    VideoRegistryEntry,
    DownloadInfo,
    GlobalCacheQueryResult,
    VideoSource,
    DEFAULT_GLOBAL_CACHE_DIR
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def cache_manager(temp_dir):
    """Create a GlobalCacheManager with temp directory"""
    return GlobalCacheManager(cache_dir=str(temp_dir))


@pytest.fixture
def sample_video_file(temp_dir):
    """Create a sample video file"""
    video_path = temp_dir / "test_video.mp4"
    video_path.write_bytes(b"fake video content for testing")
    return str(video_path)


@pytest.fixture
def sample_download_info():
    """Create sample DownloadInfo"""
    return DownloadInfo(
        keyword="travel vlog",
        youtube_id="abc123",
        youtube_url="https://youtube.com/watch?v=abc123",
        original_title="Amazing Travel Video",
        downloaded_at="2026-01-09T10:00:00"
    )


@pytest.fixture
def sample_registry_entry(sample_download_info):
    """Create sample VideoRegistryEntry"""
    return VideoRegistryEntry(
        video_hash="abcd1234",
        filename="video.mp4",
        file_size=1024000,
        duration=120.0,
        original_paths=["/path/to/video.mp4"],
        current_path="/path/to/video.mp4",
        file_exists=True,
        download_info=sample_download_info,
        topics=["travel", "vlog", "adventure"],
        keywords=["travel", "vlog"],
        projects_used_in=["project1"],
        usage_count=1,
        first_seen="2026-01-09T10:00:00",
        last_used="2026-01-09T10:00:00"
    )


# ============================================================================
# Test GlobalCacheManager Initialization
# ============================================================================

class TestGlobalCacheManagerInit:
    """Test GlobalCacheManager initialization"""

    def test_init_default_directory(self):
        """Test initialization with default directory"""
        manager = GlobalCacheManager()

        assert manager.cache_dir == DEFAULT_GLOBAL_CACHE_DIR
        assert manager.video_registry_dir.exists()
        assert manager.transcripts_dir.exists()
        assert manager.embeddings_dir.exists()

    def test_init_custom_directory(self, temp_dir):
        """Test initialization with custom directory"""
        manager = GlobalCacheManager(cache_dir=str(temp_dir))

        assert manager.cache_dir == temp_dir
        assert manager.video_registry_dir.exists()

    def test_init_creates_subdirectories(self, temp_dir):
        """Test all required subdirectories are created"""
        manager = GlobalCacheManager(cache_dir=str(temp_dir))

        assert (temp_dir / "video_registry").exists()
        assert (temp_dir / "transcripts").exists()
        assert (temp_dir / "embeddings").exists()
        assert (temp_dir / "scenes").exists()
        assert (temp_dir / "topics").exists()
        assert (temp_dir / "keywords").exists()

    def test_init_with_config(self, temp_dir):
        """Test initialization with config object"""
        config = Mock()
        manager = GlobalCacheManager(cache_dir=str(temp_dir), config=config)

        assert manager.config == config


# ============================================================================
# Test DownloadInfo Dataclass
# ============================================================================

class TestDownloadInfo:
    """Test DownloadInfo dataclass"""

    def test_download_info_creation(self):
        """Test creating DownloadInfo"""
        info = DownloadInfo(
            keyword="nature",
            youtube_id="xyz789",
            youtube_url="https://youtube.com/watch?v=xyz789"
        )

        assert info.keyword == "nature"
        assert info.youtube_id == "xyz789"
        assert info.source == "youtube"  # Default

    def test_download_info_to_dict(self, sample_download_info):
        """Test DownloadInfo serialization"""
        data = sample_download_info.to_dict()

        assert data['keyword'] == "travel vlog"
        assert data['youtube_id'] == "abc123"
        assert data['source'] == "youtube"

    def test_download_info_from_dict(self):
        """Test DownloadInfo deserialization"""
        data = {
            "keyword": "cooking",
            "youtube_id": "cook123",
            "youtube_url": "https://youtube.com/watch?v=cook123",
            "original_title": "Cooking Show",
            "source": "youtube"
        }

        info = DownloadInfo.from_dict(data)

        assert info.keyword == "cooking"
        assert info.youtube_id == "cook123"


# ============================================================================
# Test VideoRegistryEntry Dataclass
# ============================================================================

class TestVideoRegistryEntry:
    """Test VideoRegistryEntry dataclass"""

    def test_registry_entry_to_dict(self, sample_registry_entry):
        """Test VideoRegistryEntry serialization"""
        data = sample_registry_entry.to_dict()

        assert data['video_hash'] == "abcd1234"
        assert data['filename'] == "video.mp4"
        assert len(data['topics']) == 3
        assert 'download_info' in data
        assert data['download_info']['keyword'] == "travel vlog"

    def test_registry_entry_from_dict(self, sample_registry_entry):
        """Test VideoRegistryEntry deserialization"""
        data = sample_registry_entry.to_dict()
        restored = VideoRegistryEntry.from_dict(data)

        assert restored.video_hash == sample_registry_entry.video_hash
        assert restored.filename == sample_registry_entry.filename
        assert len(restored.topics) == len(sample_registry_entry.topics)
        assert restored.download_info.keyword == "travel vlog"


# ============================================================================
# Test Content Hashing
# ============================================================================

class TestContentHashing:
    """Test content-based video hashing"""

    def test_compute_content_hash_small_file(self, cache_manager, temp_dir):
        """Test hashing a small video file"""
        video_path = temp_dir / "small.mp4"
        video_path.write_bytes(b"small video content")

        hash1 = cache_manager._compute_content_hash(str(video_path))
        hash2 = cache_manager._compute_content_hash(str(video_path))

        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 32  # MD5 hex digest

    def test_compute_content_hash_large_file(self, cache_manager, temp_dir):
        """Test hashing a larger file (uses chunking)"""
        video_path = temp_dir / "large.mp4"
        # Create file > 2MB
        video_path.write_bytes(b"x" * (3 * 1024 * 1024))

        hash_val = cache_manager._compute_content_hash(str(video_path))

        assert len(hash_val) == 32

    def test_compute_content_hash_nonexistent_file(self, cache_manager, temp_dir):
        """Test hashing non-existent file falls back to filename"""
        video_path = temp_dir / "nonexistent.mp4"

        hash_val = cache_manager._compute_content_hash(str(video_path))

        # Should still return a hash (filename-based)
        assert len(hash_val) == 32

    def test_get_video_hash_public_method(self, cache_manager, sample_video_file):
        """Test public get_video_hash method"""
        hash_val = cache_manager.get_video_hash(sample_video_file)

        assert len(hash_val) == 32
        assert isinstance(hash_val, str)


# ============================================================================
# Test Video Registration
# ============================================================================

class TestVideoRegistration:
    """Test registering videos in global cache"""

    def test_register_video_basic(self, cache_manager, sample_video_file):
        """Test basic video registration"""
        entry = cache_manager.register_video(
            sample_video_file,
            download_keyword="test",
            project_id="test_project"
        )

        assert entry.filename == "test_video.mp4"
        assert entry.file_exists is True
        assert entry.projects_used_in == ["test_project"]
        assert entry.usage_count == 1

    def test_register_video_with_metadata(self, cache_manager, sample_video_file):
        """Test registering video with full metadata"""
        entry = cache_manager.register_video(
            sample_video_file,
            download_keyword="travel",
            topics=["adventure", "nature"],
            youtube_id="abc123",
            youtube_url="https://youtube.com/watch?v=abc123",
            original_title="Travel Video",
            project_id="proj1",
            duration=120.5
        )

        assert entry.topics == ["adventure", "nature"]
        assert entry.duration == 120.5
        assert entry.download_info.youtube_id == "abc123"
        assert entry.download_info.keyword == "travel"

    def test_register_video_twice_updates_entry(self, cache_manager, sample_video_file):
        """Test registering same video twice updates existing entry"""
        entry1 = cache_manager.register_video(
            sample_video_file,
            download_keyword="test1",
            project_id="proj1"
        )

        entry2 = cache_manager.register_video(
            sample_video_file,
            download_keyword="test2",
            project_id="proj2"
        )

        # Should be same hash
        assert entry1.video_hash == entry2.video_hash
        # Usage count should increase
        assert entry2.usage_count == 2
        # Projects should accumulate
        assert "proj1" in entry2.projects_used_in
        assert "proj2" in entry2.projects_used_in

    def test_register_video_saves_to_disk(self, cache_manager, sample_video_file):
        """Test video registration saves to disk"""
        entry = cache_manager.register_video(sample_video_file, download_keyword="test")

        # Check registry file exists
        entry_file = cache_manager.video_registry_dir / f"{entry.video_hash}.json"
        assert entry_file.exists()

        # Verify can load from disk
        with open(entry_file, 'r') as f:
            data = json.load(f)
        assert data['video_hash'] == entry.video_hash


# ============================================================================
# Test Index Management
# ============================================================================

class TestIndexManagement:
    """Test topic and keyword index management"""

    def test_load_indices_creates_empty_indices(self, cache_manager):
        """Test loading indices when none exist"""
        cache_manager._load_indices()

        assert cache_manager._registry_index == {}
        assert cache_manager._topic_index == {}
        assert cache_manager._keyword_index == {}
        assert cache_manager._loaded is True

    def test_update_indices_adds_topics(self, cache_manager, sample_registry_entry):
        """Test updating indices adds topic entries"""
        cache_manager._loaded = True
        cache_manager._update_indices(sample_registry_entry)

        # Topics should be indexed
        assert "travel" in cache_manager._topic_index
        assert "vlog" in cache_manager._topic_index
        assert sample_registry_entry.video_hash in cache_manager._topic_index["travel"]

    def test_update_indices_adds_keywords(self, cache_manager, sample_registry_entry):
        """Test updating indices adds keyword entries"""
        cache_manager._loaded = True
        cache_manager._update_indices(sample_registry_entry)

        # Keywords should be indexed
        assert "travel" in cache_manager._keyword_index
        assert sample_registry_entry.video_hash in cache_manager._keyword_index["travel"]

    def test_save_indices_creates_files(self, cache_manager):
        """Test saving indices creates JSON files"""
        cache_manager._loaded = True
        cache_manager._registry_index = {"hash1": "path1"}
        cache_manager._topic_index = {"travel": ["hash1"]}
        cache_manager._keyword_index = {"keyword": ["hash1"]}

        cache_manager._save_indices()

        assert cache_manager.registry_index_path.exists()
        assert cache_manager.topic_index_path.exists()
        assert cache_manager.keyword_index_path.exists()


# ============================================================================
# Test Video Query
# ============================================================================

class TestVideoQuery:
    """Test querying global cache for videos"""

    def test_find_videos_for_keywords_empty_cache(self, cache_manager):
        """Test query on empty cache"""
        result = cache_manager.find_videos_for_keywords(
            keywords=["travel", "nature"]
        )

        assert result.total_cached_matches == 0
        assert len(result.reuse_videos) == 0
        assert len(result.uncovered_keywords) == 2

    def test_find_videos_for_keywords_with_matches(self, cache_manager, sample_video_file):
        """Test query finds registered videos"""
        # Register a video
        cache_manager.register_video(
            sample_video_file,
            download_keyword="travel",
            topics=["adventure", "nature"],
            project_id="proj1"
        )

        result = cache_manager.find_videos_for_keywords(
            keywords=["travel"],
            topics=["adventure"]
        )

        assert result.total_cached_matches > 0
        assert len(result.reuse_videos) > 0
        assert len(result.uncovered_keywords) == 0

    def test_find_videos_relevance_filtering(self, cache_manager, sample_video_file):
        """Test query filters by minimum relevance"""
        cache_manager.register_video(
            sample_video_file,
            download_keyword="cooking",
            topics=["food"],
            project_id="proj1"
        )

        # Query with unrelated keywords
        result = cache_manager.find_videos_for_keywords(
            keywords=["travel", "nature"],
            min_relevance=0.5
        )

        # Should not match due to low relevance
        assert len(result.reuse_videos) == 0

    def test_find_videos_partial_keyword_match(self, cache_manager, sample_video_file):
        """Test partial keyword matching"""
        cache_manager.register_video(
            sample_video_file,
            download_keyword="travel vlog",
            project_id="proj1"
        )

        result = cache_manager.find_videos_for_keywords(
            keywords=["travel"],  # Partial match
            min_relevance=0.1
        )

        # Should match via partial keyword
        assert len(result.reuse_videos) > 0


# ============================================================================
# Test File Existence Checking
# ============================================================================

class TestFileExistence:
    """Test file existence checking"""

    def test_check_file_exists_current_path(self, cache_manager, sample_video_file):
        """Test checking file exists at current path"""
        entry = VideoRegistryEntry(
            video_hash="test",
            filename="test.mp4",
            file_size=1000,
            current_path=sample_video_file
        )

        exists = cache_manager._check_file_exists(entry)

        assert exists is True

    def test_check_file_exists_fallback_to_original(self, cache_manager, sample_video_file):
        """Test fallback to original paths when current path missing"""
        entry = VideoRegistryEntry(
            video_hash="test",
            filename="test.mp4",
            file_size=1000,
            current_path="/nonexistent/path.mp4",
            original_paths=[sample_video_file]
        )

        exists = cache_manager._check_file_exists(entry)

        assert exists is True
        assert entry.current_path == sample_video_file  # Should update

    def test_check_file_exists_all_missing(self, cache_manager):
        """Test when all paths are missing"""
        entry = VideoRegistryEntry(
            video_hash="test",
            filename="test.mp4",
            file_size=1000,
            current_path="/nonexistent1.mp4",
            original_paths=["/nonexistent2.mp4"]
        )

        exists = cache_manager._check_file_exists(entry)

        assert exists is False


# ============================================================================
# Test Relevance Scoring
# ============================================================================

class TestRelevanceScoring:
    """Test relevance score computation"""

    def test_compute_relevance_exact_match(self, cache_manager, sample_registry_entry):
        """Test relevance with exact keyword match"""
        score = cache_manager._compute_relevance(
            sample_registry_entry,
            keywords=["travel", "vlog"],
            topics=["travel", "adventure"]
        )

        # Should have high relevance
        assert score > 0.8

    def test_compute_relevance_partial_match(self, cache_manager, sample_registry_entry):
        """Test relevance with partial match"""
        score = cache_manager._compute_relevance(
            sample_registry_entry,
            keywords=["vlog"],  # Matches "vlog" in keywords and "travel vlog" in download_info
            topics=["adventure"]  # Only one of three topics
        )

        # Should have good relevance (matches keyword + some topics)
        assert score >= 0.5

    def test_compute_relevance_no_match(self, cache_manager, sample_registry_entry):
        """Test relevance with no match"""
        score = cache_manager._compute_relevance(
            sample_registry_entry,
            keywords=["cooking", "recipe"],
            topics=["food"]
        )

        # Should have low relevance
        assert score < 0.3


# ============================================================================
# Test Transcript/Scene Data Sharing
# ============================================================================

class TestDataSharing:
    """Test sharing transcript and scene data"""

    def test_copy_transcript_to_global(self, cache_manager):
        """Test copying transcript to global cache"""
        transcript_data = {
            "segments": [{"text": "Hello world", "start": 0.0, "end": 2.0}]
        }

        cache_manager.copy_transcript_to_global("hash123", transcript_data)

        transcript_path = cache_manager.transcripts_dir / "hash123.json"
        assert transcript_path.exists()

    def test_get_transcript_from_global(self, cache_manager):
        """Test retrieving transcript from global cache"""
        transcript_data = {"segments": []}

        cache_manager.copy_transcript_to_global("hash456", transcript_data)
        retrieved = cache_manager.get_transcript_from_global("hash456")

        assert retrieved == transcript_data

    def test_get_transcript_nonexistent(self, cache_manager):
        """Test retrieving non-existent transcript"""
        result = cache_manager.get_transcript_from_global("nonexistent")

        assert result is None

    def test_copy_scenes_to_global(self, cache_manager):
        """Test copying scenes to global cache"""
        scene_data = {"scenes": [{"start": 0.0, "end": 5.0}]}

        cache_manager.copy_scenes_to_global("hash789", scene_data)

        scene_path = cache_manager.scenes_dir / "hash789.json"
        assert scene_path.exists()

    def test_get_scenes_from_global(self, cache_manager):
        """Test retrieving scenes from global cache"""
        scene_data = {"scenes": []}

        cache_manager.copy_scenes_to_global("hash999", scene_data)
        retrieved = cache_manager.get_scenes_from_global("hash999")

        assert retrieved == scene_data


# ============================================================================
# Test Video Processing Status
# ============================================================================

class TestVideoProcessing:
    """Test marking videos as processed"""

    def test_mark_video_processed(self, cache_manager, sample_video_file):
        """Test marking video with processing flags"""
        entry = cache_manager.register_video(sample_video_file, download_keyword="test")

        cache_manager.mark_video_processed(
            entry.video_hash,
            has_transcript=True,
            has_embeddings=True,
            has_scenes=True,
            face_score=0.8
        )

        # Reload entry
        updated = cache_manager.get_video_entry(entry.video_hash)

        assert updated.has_transcript is True
        assert updated.has_embeddings is True
        assert updated.has_scenes is True
        assert updated.face_score == 0.8

    def test_update_video_topics(self, cache_manager, sample_video_file):
        """Test updating topics for a video"""
        entry = cache_manager.register_video(
            sample_video_file,
            download_keyword="test",
            topics=["travel"]
        )

        cache_manager.update_video_topics(entry.video_hash, ["nature", "adventure"])

        # Reload entry
        updated = cache_manager.get_video_entry(entry.video_hash)

        assert "travel" in updated.topics
        assert "nature" in updated.topics
        assert "adventure" in updated.topics


# ============================================================================
# Test Cache Statistics
# ============================================================================

class TestCacheStatistics:
    """Test cache statistics"""

    def test_get_stats_empty_cache(self, cache_manager):
        """Test stats on empty cache"""
        stats = cache_manager.get_stats()

        assert stats['total_videos'] == 0
        assert stats['total_topics'] == 0
        assert stats['total_keywords'] == 0
        assert 'cache_size_mb' in stats

    def test_get_stats_with_videos(self, cache_manager, sample_video_file):
        """Test stats with registered videos"""
        cache_manager.register_video(
            sample_video_file,
            download_keyword="travel",
            topics=["adventure", "nature"]
        )

        stats = cache_manager.get_stats()

        assert stats['total_videos'] == 1
        assert stats['total_topics'] == 2
        assert stats['total_keywords'] >= 1

    def test_get_cache_size_mb(self, cache_manager, sample_video_file):
        """Test cache size calculation"""
        cache_manager.register_video(sample_video_file, download_keyword="test")

        size_mb = cache_manager._get_cache_size_mb()

        assert size_mb >= 0.0
        assert isinstance(size_mb, float)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_get_video_entry_nonexistent(self, cache_manager):
        """Test getting non-existent video entry"""
        entry = cache_manager.get_video_entry("nonexistent_hash")

        assert entry is None

    def test_register_video_with_no_metadata(self, cache_manager, sample_video_file):
        """Test registering video with minimal metadata"""
        entry = cache_manager.register_video(sample_video_file)

        assert entry.filename == "test_video.mp4"
        assert entry.topics == []
        assert entry.keywords == []

    def test_find_videos_with_empty_keywords(self, cache_manager):
        """Test query with empty keyword list"""
        result = cache_manager.find_videos_for_keywords(keywords=[])

        # Should handle gracefully
        assert result.total_cached_matches == 0

    def test_update_topics_nonexistent_video(self, cache_manager):
        """Test updating topics for non-existent video"""
        # Should not crash
        cache_manager.update_video_topics("nonexistent", ["topic1"])

    def test_mark_processed_nonexistent_video(self, cache_manager):
        """Test marking non-existent video as processed"""
        # Should not crash
        cache_manager.mark_video_processed("nonexistent", has_transcript=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
