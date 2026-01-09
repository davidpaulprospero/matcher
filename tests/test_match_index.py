"""
Comprehensive tests for match index module.

Covers:
- MatchedVideoInfo dataclass
- MatchAwareIndex initialization
- Video tracking (new, modified, deleted)
- Voiceover and config change detection
- Match result caching
- Index persistence and loading
- Batch operations

Created: 2026-01-09 (Phase 8.1)
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
import json
import time
import tempfile
from pathlib import Path

from src.match_index import MatchedVideoInfo, MatchAwareIndex


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_project_dir():
    """Create temporary project directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def match_index(temp_project_dir):
    """Create MatchAwareIndex instance"""
    return MatchAwareIndex(temp_project_dir)


@pytest.fixture
def sample_video_paths(temp_project_dir):
    """Create sample video files"""
    video_dir = Path(temp_project_dir) / "videos"
    video_dir.mkdir(exist_ok=True)

    videos = []
    for i in range(3):
        video_path = video_dir / f"video{i}.mp4"
        video_path.write_bytes(b"fake video content")
        videos.append(str(video_path))

    return videos


# ============================================================================
# Test MatchedVideoInfo Dataclass
# ============================================================================

class TestMatchedVideoInfo:
    """Test MatchedVideoInfo dataclass"""

    def test_init(self):
        """Test initialization"""
        info = MatchedVideoInfo(
            video_path="/path/to/video.mp4",
            video_hash="abc123",
            matched_at=1234567890.0,
            segment_count=5
        )

        assert info.video_path == "/path/to/video.mp4"
        assert info.video_hash == "abc123"
        assert info.matched_at == 1234567890.0
        assert info.segment_count == 5

    def test_to_dict(self):
        """Test conversion to dict"""
        info = MatchedVideoInfo(
            video_path="/path/to/video.mp4",
            video_hash="abc123",
            matched_at=1234567890.0,
            segment_count=5
        )

        result = info.to_dict()

        assert result['video_path'] == "/path/to/video.mp4"
        assert result['video_hash'] == "abc123"
        assert result['matched_at'] == 1234567890.0
        assert result['segment_count'] == 5

    def test_from_dict(self):
        """Test creation from dict"""
        data = {
            'video_path': "/path/to/video.mp4",
            'video_hash': "abc123",
            'matched_at': 1234567890.0,
            'segment_count': 5
        }

        info = MatchedVideoInfo.from_dict(data)

        assert info.video_path == "/path/to/video.mp4"
        assert info.video_hash == "abc123"
        assert info.matched_at == 1234567890.0
        assert info.segment_count == 5

    def test_from_dict_missing_fields(self):
        """Test from_dict with missing fields"""
        data = {}  # Empty dict

        info = MatchedVideoInfo.from_dict(data)

        # Should use defaults
        assert info.video_path == ''
        assert info.video_hash == ''
        assert info.matched_at == 0.0
        assert info.segment_count == 0


# ============================================================================
# Test Initialization
# ============================================================================

class TestMatchAwareIndexInit:
    """Test MatchAwareIndex initialization"""

    def test_init_creates_cache_dir(self, temp_project_dir):
        """Test initialization creates cache directory"""
        index = MatchAwareIndex(temp_project_dir)

        cache_dir = Path(temp_project_dir) / ".cache"
        assert cache_dir.exists()
        assert index.cache_dir == cache_dir

    def test_init_empty_index(self, match_index):
        """Test initialization with no existing index"""
        assert len(match_index.matched_videos) == 0
        assert match_index.voiceover_hash == ""
        assert match_index.config_hash == ""

    def test_init_loads_existing_index(self, temp_project_dir):
        """Test initialization loads existing index"""
        # Create an index file
        cache_dir = Path(temp_project_dir) / ".cache"
        cache_dir.mkdir(exist_ok=True)

        index_data = {
            'version': '1.0',
            'updated_at': time.time(),
            'voiceover_hash': 'voiceover123',
            'config_hash': 'config456',
            'matched_videos': {
                'video1.mp4': {
                    'video_path': 'video1.mp4',
                    'video_hash': 'hash1',
                    'matched_at': 1234567890.0,
                    'segment_count': 3
                }
            }
        }

        with open(cache_dir / "match_index.json", 'w') as f:
            json.dump(index_data, f)

        # Initialize index
        index = MatchAwareIndex(temp_project_dir)

        assert len(index.matched_videos) == 1
        assert 'video1.mp4' in index.matched_videos
        assert index.voiceover_hash == 'voiceover123'
        assert index.config_hash == 'config456'

    def test_init_handles_corrupted_index(self, temp_project_dir):
        """Test initialization handles corrupted index file"""
        cache_dir = Path(temp_project_dir) / ".cache"
        cache_dir.mkdir(exist_ok=True)

        # Write invalid JSON
        with open(cache_dir / "match_index.json", 'w') as f:
            f.write("invalid json{")

        # Should reset to empty index
        index = MatchAwareIndex(temp_project_dir)

        assert len(index.matched_videos) == 0


# ============================================================================
# Test Video Tracking
# ============================================================================

class TestVideoTracking:
    """Test video tracking functionality"""

    def test_mark_matched(self, match_index, sample_video_paths):
        """Test marking a video as matched"""
        video_path = sample_video_paths[0]

        match_index.mark_matched(video_path, segment_count=5)

        assert match_index.is_matched(video_path)
        info = match_index.matched_videos[video_path]
        assert info.segment_count == 5
        assert info.video_hash != ""

    def test_mark_matched_batch(self, match_index, sample_video_paths):
        """Test marking multiple videos as matched"""
        segment_counts = {
            sample_video_paths[0]: 3,
            sample_video_paths[1]: 5
        }

        match_index.mark_matched_batch(sample_video_paths[:2], segment_counts)

        assert match_index.is_matched(sample_video_paths[0])
        assert match_index.is_matched(sample_video_paths[1])
        assert match_index.matched_videos[sample_video_paths[0]].segment_count == 3
        assert match_index.matched_videos[sample_video_paths[1]].segment_count == 5

    def test_is_matched(self, match_index, sample_video_paths):
        """Test checking if video is matched"""
        video_path = sample_video_paths[0]

        assert not match_index.is_matched(video_path)

        match_index.mark_matched(video_path)

        assert match_index.is_matched(video_path)

    def test_get_new_videos(self, match_index, sample_video_paths):
        """Test getting list of new (unmatched) videos"""
        # Mark first video as matched
        match_index.mark_matched(sample_video_paths[0])

        new_videos = match_index.get_new_videos(sample_video_paths)

        assert len(new_videos) == 2
        assert sample_video_paths[1] in new_videos
        assert sample_video_paths[2] in new_videos

    def test_is_video_modified(self, match_index, sample_video_paths):
        """Test detecting modified videos"""
        video_path = sample_video_paths[0]

        # Mark as matched
        match_index.mark_matched(video_path)
        assert not match_index.is_video_modified(video_path)

        # Modify the video
        Path(video_path).write_bytes(b"modified content")

        # Should detect modification
        assert match_index.is_video_modified(video_path)

    def test_get_modified_videos(self, match_index, sample_video_paths):
        """Test getting list of modified videos"""
        # Mark all as matched
        match_index.mark_matched_batch(sample_video_paths)

        # Modify first video
        Path(sample_video_paths[0]).write_bytes(b"modified content")

        modified = match_index.get_modified_videos(sample_video_paths)

        assert len(modified) == 1
        assert sample_video_paths[0] in modified

    def test_get_deleted_videos(self, match_index, sample_video_paths):
        """Test getting list of deleted videos"""
        # Mark all as matched
        match_index.mark_matched_batch(sample_video_paths)

        # Delete first video
        Path(sample_video_paths[0]).unlink()

        # Get deleted videos (pass only remaining videos)
        deleted = match_index.get_deleted_videos(sample_video_paths[1:])

        assert len(deleted) == 1
        assert sample_video_paths[0] in deleted

    def test_remove_videos(self, match_index, sample_video_paths):
        """Test removing videos from index"""
        # Mark all as matched
        match_index.mark_matched_batch(sample_video_paths)

        # Remove first video
        match_index.remove_videos([sample_video_paths[0]])

        assert not match_index.is_matched(sample_video_paths[0])
        assert match_index.is_matched(sample_video_paths[1])


# ============================================================================
# Test Voiceover & Config Tracking
# ============================================================================

class TestVoiceoverConfigTracking:
    """Test voiceover and config change detection"""

    def test_set_voiceover_hash(self, match_index, temp_project_dir):
        """Test setting voiceover hash"""
        voiceover_path = Path(temp_project_dir) / "voiceover.mp3"
        voiceover_path.write_bytes(b"voiceover audio")

        match_index.set_voiceover_hash(str(voiceover_path))

        assert match_index.voiceover_hash != ""

    def test_is_voiceover_changed(self, match_index, temp_project_dir):
        """Test detecting voiceover changes"""
        voiceover_path = Path(temp_project_dir) / "voiceover.mp3"
        voiceover_path.write_bytes(b"voiceover audio")

        # Set initial hash
        match_index.set_voiceover_hash(str(voiceover_path))
        assert not match_index.is_voiceover_changed(str(voiceover_path))

        # Modify voiceover
        voiceover_path.write_bytes(b"modified voiceover")

        # Should detect change
        assert match_index.is_voiceover_changed(str(voiceover_path))

    def test_is_voiceover_changed_first_run(self, match_index, temp_project_dir):
        """Test voiceover change detection on first run"""
        voiceover_path = Path(temp_project_dir) / "voiceover.mp3"
        voiceover_path.write_bytes(b"voiceover audio")

        # First run - no hash set yet
        assert not match_index.is_voiceover_changed(str(voiceover_path))

    def test_set_config_hash(self, match_index):
        """Test setting config hash"""
        config_hash = "config123abc"

        match_index.set_config_hash(config_hash)

        assert match_index.config_hash == config_hash

    def test_is_config_changed(self, match_index):
        """Test detecting config changes"""
        match_index.set_config_hash("config123")

        assert not match_index.is_config_changed("config123")
        assert match_index.is_config_changed("config456")

    def test_is_config_changed_first_run(self, match_index):
        """Test config change detection on first run"""
        # First run - no hash set yet
        assert not match_index.is_config_changed("config123")


# ============================================================================
# Test Match Result Caching
# ============================================================================

class TestMatchResultCaching:
    """Test match result caching"""

    def test_save_and_load_matches(self, match_index):
        """Test saving and loading match results"""
        matches = [
            {'segment_id': 1, 'video': 'video1.mp4', 'confidence': 0.9},
            {'segment_id': 2, 'video': 'video2.mp4', 'confidence': 0.8}
        ]

        match_index.save_matches(matches)

        loaded = match_index.get_cached_matches()

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0]['segment_id'] == 1

    def test_save_matches_creates_timestamped_copy(self, match_index, temp_project_dir):
        """Test that save_matches creates timestamped archive"""
        matches = [{'segment_id': 1, 'video': 'video1.mp4'}]

        match_index.save_matches(matches)

        # Check for timestamped file
        cache_dir = Path(temp_project_dir) / ".cache"
        timestamped_files = list(cache_dir.glob("matches_*.json"))

        assert len(timestamped_files) > 0

    def test_get_cached_matches_no_cache(self, match_index):
        """Test loading matches when no cache exists"""
        result = match_index.get_cached_matches()

        assert result is None

    def test_get_cached_matches_corrupted(self, match_index, temp_project_dir):
        """Test loading corrupted match cache"""
        cache_dir = Path(temp_project_dir) / ".cache"

        # Write invalid JSON
        with open(cache_dir / "cached_matches.json", 'w') as f:
            f.write("invalid json{")

        result = match_index.get_cached_matches()

        assert result is None

    def test_clear_matches(self, match_index):
        """Test clearing cached matches"""
        matches = [{'segment_id': 1, 'video': 'video1.mp4'}]
        match_index.save_matches(matches)

        assert match_index.get_cached_matches() is not None

        match_index.clear_matches()

        assert match_index.get_cached_matches() is None


# ============================================================================
# Test Persistence
# ============================================================================

class TestPersistence:
    """Test index persistence"""

    def test_index_persists_across_instances(self, temp_project_dir, sample_video_paths):
        """Test that index persists across instances"""
        # Create index and mark videos
        index1 = MatchAwareIndex(temp_project_dir)
        index1.mark_matched(sample_video_paths[0])

        # Create new instance
        index2 = MatchAwareIndex(temp_project_dir)

        # Should load persisted data
        assert index2.is_matched(sample_video_paths[0])

    def test_save_creates_index_file(self, match_index, temp_project_dir, sample_video_paths):
        """Test that saving creates index file"""
        match_index.mark_matched(sample_video_paths[0])

        index_path = Path(temp_project_dir) / ".cache" / "match_index.json"
        assert index_path.exists()

    def test_atomic_write(self, match_index, sample_video_paths):
        """Test atomic write using temp file"""
        match_index.mark_matched(sample_video_paths[0])

        # Temp file should not exist after save
        temp_path = match_index.index_path.with_suffix('.tmp')
        assert not temp_path.exists()


# ============================================================================
# Test Utility Methods
# ============================================================================

class TestUtilityMethods:
    """Test utility methods"""

    def test_clear(self, match_index, sample_video_paths):
        """Test clearing entire index"""
        # Add some data
        match_index.mark_matched(sample_video_paths[0])
        match_index.set_voiceover_hash("voiceover.mp3")
        match_index.save_matches([{'segment_id': 1}])

        match_index.clear()

        # Everything should be cleared
        assert len(match_index.matched_videos) == 0
        assert match_index.voiceover_hash == ""
        assert match_index.get_cached_matches() is None

    def test_get_stats(self, match_index, sample_video_paths):
        """Test getting index statistics"""
        match_index.mark_matched_batch(sample_video_paths)

        stats = match_index.get_stats()

        assert stats['matched_video_count'] == 3
        assert 'has_cached_matches' in stats
        assert 'updated_at' in stats

    def test_repr(self, match_index, sample_video_paths):
        """Test string representation"""
        match_index.mark_matched(sample_video_paths[0])

        repr_str = repr(match_index)

        assert 'MatchAwareIndex' in repr_str
        assert 'videos=1' in repr_str


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_mark_matched_with_no_segment_count(self, match_index, sample_video_paths):
        """Test marking video without segment count"""
        match_index.mark_matched(sample_video_paths[0])

        info = match_index.matched_videos[sample_video_paths[0]]
        assert info.segment_count == 0

    def test_get_file_hash_nonexistent_file(self, match_index):
        """Test getting hash of nonexistent file"""
        hash_result = match_index._get_file_hash("/nonexistent/file.mp4")

        assert hash_result == ""

    def test_is_video_modified_unmatched_video(self, match_index, sample_video_paths):
        """Test checking modification of unmatched video"""
        result = match_index.is_video_modified(sample_video_paths[0])

        assert result is False

    def test_save_matches_with_voiceover_path(self, match_index, temp_project_dir):
        """Test saving matches with voiceover metadata"""
        voiceover_path = Path(temp_project_dir) / "voiceover.mp3"
        matches = [{'segment_id': 1}]

        match_index.save_matches(matches, voiceover_path=str(voiceover_path))

        # Load and verify metadata
        with open(match_index.matches_path, 'r') as f:
            data = json.load(f)

        assert data['voiceover_path'] == str(voiceover_path)

    def test_empty_video_list_operations(self, match_index):
        """Test operations with empty video lists"""
        assert match_index.get_new_videos([]) == []
        assert match_index.get_modified_videos([]) == []
        assert match_index.get_deleted_videos([]) == []

    def test_mark_matched_batch_empty_segment_counts(self, match_index, sample_video_paths):
        """Test batch marking without segment counts"""
        match_index.mark_matched_batch(sample_video_paths)

        # Should work with default counts
        for video_path in sample_video_paths:
            assert match_index.is_matched(video_path)
            assert match_index.matched_videos[video_path].segment_count == 0
