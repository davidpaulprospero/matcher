"""
Comprehensive tests for deduplication module.

Covers:
- VideoDeduplicator initialization
- Perceptual hash computation
- Frame extraction from videos
- Duplicate detection
- Hash distance calculation
- Auto-delete functionality
- Deduplication reports
- Stats preview
- Convenience functions

Created: 2026-01-09 (Phase 4.3)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile
import shutil
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deduplication import (
    VideoDeduplicator,
    DuplicateGroup,
    DeduplicationReport,
    deduplicate_videos,
    VIDEO_EXTENSIONS
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
def mock_imagehash():
    """Mock imagehash library"""
    mock_hash = Mock()
    mock_hash.hex_to_hash = Mock(side_effect=lambda h: Mock(__sub__=lambda s, o: int(h, 16) ^ int(o, 16)))
    mock_hash.phash = Mock(return_value=Mock(__str__=lambda s: "abc123"))
    return mock_hash


@pytest.fixture
def deduplicator():
    """Create VideoDeduplicator with imagehash mocked"""
    with patch.dict('sys.modules', {'imagehash': MagicMock(), 'PIL': MagicMock()}):
        dedup = VideoDeduplicator(threshold=10)
        dedup._available = True
        dedup.imagehash = MagicMock()
        dedup.Image = MagicMock()
        return dedup


@pytest.fixture
def sample_videos(temp_dir):
    """Create sample video files"""
    videos = []
    for i in range(3):
        video_path = temp_dir / f"video{i}.mp4"
        video_path.write_bytes(b"fake video content")
        videos.append(str(video_path))
    return videos


# ============================================================================
# Test VideoDeduplicator Initialization
# ============================================================================

class TestVideoDeduplicatorInit:
    """Test VideoDeduplicator initialization"""

    @pytest.mark.fast
    def test_init_default_threshold(self):
        """Test initialization with default threshold"""
        dedup = VideoDeduplicator()

        assert dedup.threshold == VideoDeduplicator.DEFAULT_THRESHOLD
        assert dedup.frame_timeout == VideoDeduplicator.DEFAULT_FRAME_TIMEOUT

    @pytest.mark.fast
    def test_init_custom_threshold(self):
        """Test initialization with custom threshold"""
        dedup = VideoDeduplicator(threshold=5)

        assert dedup.threshold == 5

    @pytest.mark.fast
    def test_init_with_config(self):
        """Test initialization with config object"""
        config = Mock()
        config.deduplication = Mock()
        config.deduplication.hash_threshold = 8
        config.deduplication.frame_timeout = 60
        config.deduplication.auto_delete = False

        dedup = VideoDeduplicator(config)

        assert dedup.threshold == 8
        assert dedup.frame_timeout == 60
        assert dedup.auto_delete is False

    @pytest.mark.fast
    def test_init_checks_imagehash_availability(self):
        """Test initialization checks for imagehash"""
        with patch.dict('sys.modules', {'imagehash': None, 'PIL': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                dedup = VideoDeduplicator()

                assert dedup.is_available() is False


# ============================================================================
# Test Frame Extraction
# ============================================================================

class TestFrameExtraction:
    """Test video frame extraction"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_extract_first_frame_success(self, mock_run, deduplicator, temp_dir):
        """Test successful frame extraction"""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"video")
        frame_path = temp_dir / "test.thumb.jpg"
        frame_path.write_bytes(b"frame")

        mock_run.return_value = Mock(returncode=0)

        result = deduplicator._extract_first_frame(str(video_path))

        assert result == str(frame_path)

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_extract_first_frame_gpu_fallback(self, mock_run, deduplicator, temp_dir):
        """Test fallback to CPU when GPU extraction fails"""
        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"video")
        frame_path = temp_dir / "test.thumb.jpg"

        # First call (GPU) fails, second call (CPU) succeeds
        def side_effect(*args, **kwargs):
            if '-hwaccel' in args[0]:
                return Mock(returncode=1)
            else:
                frame_path.write_bytes(b"frame")
                return Mock(returncode=0)

        mock_run.side_effect = side_effect

        result = deduplicator._extract_first_frame(str(video_path))

        assert result == str(frame_path)
        assert mock_run.call_count == 2

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_extract_first_frame_failure(self, mock_run, deduplicator, temp_dir):
        """Test frame extraction failure"""
        mock_run.side_effect = Exception("FFmpeg error")

        video_path = temp_dir / "test.mp4"
        video_path.write_bytes(b"video")

        result = deduplicator._extract_first_frame(str(video_path))

        assert result is None


# ============================================================================
# Test Hash Computation
# ============================================================================

class TestHashComputation:
    """Test perceptual hash computation"""

    @patch('src.deduplication.VideoDeduplicator._extract_first_frame')
    @pytest.mark.fast
    def test_compute_hash_success(self, mock_extract, deduplicator):
        """Test successful hash computation"""
        mock_extract.return_value = "/tmp/frame.jpg"
        mock_img = MagicMock()
        deduplicator.Image.open.return_value = mock_img
        deduplicator.imagehash.phash.return_value = Mock(__str__=lambda s: "abc123")

        hash_val = deduplicator._compute_hash("/path/to/video.mp4")

        assert hash_val == "abc123"

    @patch('src.deduplication.VideoDeduplicator._extract_first_frame')
    @pytest.mark.fast
    def test_compute_hash_uses_cache(self, mock_extract, deduplicator):
        """Test hash computation uses cache"""
        mock_extract.return_value = "/tmp/frame.jpg"
        deduplicator.Image.open.return_value = MagicMock()
        deduplicator.imagehash.phash.return_value = Mock(__str__=lambda s: "abc123")

        hash1 = deduplicator._compute_hash("/path/to/video.mp4")
        hash2 = deduplicator._compute_hash("/path/to/video.mp4")

        assert hash1 == hash2
        # Should only extract frame once
        assert mock_extract.call_count == 1

    @patch('src.deduplication.VideoDeduplicator._extract_first_frame')
    @pytest.mark.fast
    def test_compute_hash_extraction_failure(self, mock_extract, deduplicator):
        """Test hash computation when frame extraction fails"""
        mock_extract.return_value = None

        hash_val = deduplicator._compute_hash("/path/to/video.mp4")

        assert hash_val is None

    @patch('src.deduplication.VideoDeduplicator._extract_first_frame')
    @pytest.mark.fast
    def test_compute_hash_cleans_up_frame(self, mock_extract, deduplicator, temp_dir):
        """Test hash computation cleans up temporary frame"""
        frame_path = temp_dir / "frame.jpg"
        frame_path.write_bytes(b"frame")
        mock_extract.return_value = str(frame_path)

        deduplicator.Image.open.return_value = MagicMock()
        deduplicator.imagehash.phash.return_value = Mock(__str__=lambda s: "abc123")

        deduplicator._compute_hash("/path/to/video.mp4")

        # Frame should be deleted
        assert not frame_path.exists()


# ============================================================================
# Test Hash Distance
# ============================================================================

class TestHashDistance:
    """Test hash distance calculation"""

    @pytest.mark.fast
    def test_hash_distance_identical(self, deduplicator):
        """Test distance between identical hashes"""
        deduplicator.imagehash.hex_to_hash = lambda h: Mock(__sub__=lambda s, o: 0)

        distance = deduplicator._hash_distance("abc123", "abc123")

        assert distance == 0

    @pytest.mark.fast
    def test_hash_distance_different(self, deduplicator):
        """Test distance between different hashes"""
        def mock_hex_to_hash(h):
            return Mock(__sub__=lambda s, o: 5)

        deduplicator.imagehash.hex_to_hash = mock_hex_to_hash

        distance = deduplicator._hash_distance("abc123", "def456")

        assert distance == 5

    @pytest.mark.fast
    def test_hash_distance_when_unavailable(self):
        """Test hash distance when imagehash unavailable"""
        dedup = VideoDeduplicator()
        dedup._available = False

        distance = dedup._hash_distance("abc", "def")

        assert distance == 999  # Error value


# ============================================================================
# Test Duplicate Finding
# ============================================================================

class TestDuplicateFinding:
    """Test finding duplicate videos"""

    @patch('src.deduplication.VideoDeduplicator._compute_hash')
    @pytest.mark.fast
    def test_find_duplicates_empty_directory(self, mock_compute, deduplicator, temp_dir):
        """Test finding duplicates in empty directory"""
        groups = deduplicator.find_duplicates(str(temp_dir))

        assert len(groups) == 0

    @patch('src.deduplication.VideoDeduplicator._compute_hash')
    @patch('src.deduplication.VideoDeduplicator._hash_distance')
    @pytest.mark.fast
    def test_find_duplicates_with_duplicates(self, mock_distance, mock_compute, deduplicator, sample_videos):
        """Test finding duplicate videos"""
        # Mock hashes - all three videos hash successfully
        mock_compute.side_effect = ["hash1", "hash1", "hash2"]  # First two are duplicates

        # Mock distances
        def distance_func(h1, h2):
            if h1 == h2:
                return 0
            return 5 if (h1 == "hash1" or h2 == "hash1") else 20

        mock_distance.side_effect = distance_func

        groups = deduplicator.find_duplicates(str(Path(sample_videos[0]).parent))

        # With Union-Find, all 3 videos get grouped because:
        # video0 (hash1) - video1 (hash1): distance 0 (duplicates)
        # video0 (hash1) - video2 (hash2): distance 5 (duplicates, threshold=10)
        # So video1 and video2 are both duplicates of video0
        assert len(groups) == 1
        assert len(groups[0].duplicates) == 2  # Both video1 and video2

    @patch('src.deduplication.VideoDeduplicator._compute_hash')
    @pytest.mark.fast
    def test_find_duplicates_no_duplicates(self, mock_compute, deduplicator, sample_videos):
        """Test finding when no duplicates exist"""
        # All different hashes
        mock_compute.side_effect = ["hash1", "hash2", "hash3"]

        deduplicator._hash_distance = lambda h1, h2: 20  # All different

        groups = deduplicator.find_duplicates(str(Path(sample_videos[0]).parent))

        assert len(groups) == 0

    @pytest.mark.fast
    def test_find_duplicates_when_unavailable(self, temp_dir):
        """Test finding duplicates when imagehash unavailable"""
        dedup = VideoDeduplicator()
        dedup._available = False

        groups = dedup.find_duplicates(str(temp_dir))

        assert groups == []


# ============================================================================
# Test Deduplication
# ============================================================================

class TestDeduplication:
    """Test full deduplication process"""

    @patch('src.deduplication.VideoDeduplicator.find_duplicates')
    @pytest.mark.fast
    def test_deduplicate_with_no_duplicates(self, mock_find, deduplicator, sample_videos):
        """Test deduplication when no duplicates found"""
        mock_find.return_value = []

        report = deduplicator.deduplicate(str(Path(sample_videos[0]).parent))

        assert report.duplicates_found == 0
        assert report.duplicates_deleted == 0
        assert report.space_saved_mb == 0

    @patch('src.deduplication.VideoDeduplicator.find_duplicates')
    @pytest.mark.fast
    def test_deduplicate_with_duplicates_auto_delete(self, mock_find, deduplicator, sample_videos):
        """Test deduplication with auto-delete enabled"""
        # Create duplicate group
        group = DuplicateGroup(
            keep=sample_videos[0],
            keep_size=1000,
            duplicates=[sample_videos[1]],
            similarity=0.95,
            hash_distance=3
        )
        mock_find.return_value = [group]

        report = deduplicator.deduplicate(str(Path(sample_videos[0]).parent), auto_delete=True)

        assert report.duplicates_found == 1
        assert report.duplicates_deleted == 1
        # File should be deleted
        assert not Path(sample_videos[1]).exists()

    @patch('src.deduplication.VideoDeduplicator.find_duplicates')
    @pytest.mark.fast
    def test_deduplicate_no_auto_delete(self, mock_find, deduplicator, sample_videos):
        """Test deduplication with auto-delete disabled"""
        group = DuplicateGroup(
            keep=sample_videos[0],
            keep_size=1000,
            duplicates=[sample_videos[1]],
            similarity=0.95,
            hash_distance=3
        )
        mock_find.return_value = [group]

        report = deduplicator.deduplicate(str(Path(sample_videos[0]).parent), auto_delete=False)

        assert report.duplicates_found == 1
        assert report.duplicates_deleted == 0
        # File should still exist
        assert Path(sample_videos[1]).exists()

    @patch('src.deduplication.VideoDeduplicator.find_duplicates')
    @pytest.mark.fast
    def test_deduplicate_saves_report(self, mock_find, deduplicator, sample_videos, temp_dir):
        """Test deduplication saves JSON report"""
        mock_find.return_value = []
        report_path = temp_dir / "dedup_report.json"

        deduplicator.deduplicate(
            str(Path(sample_videos[0]).parent),
            report_path=str(report_path)
        )

        assert report_path.exists()
        with open(report_path, 'r') as f:
            data = json.load(f)
        assert 'total_videos' in data


# ============================================================================
# Test DeduplicationReport
# ============================================================================

class TestDeduplicationReport:
    """Test DeduplicationReport dataclass"""

    @pytest.mark.fast
    def test_report_creation(self):
        """Test creating DeduplicationReport"""
        report = DeduplicationReport(
            total_videos=10,
            unique_videos=8,
            duplicates_found=2,
            duplicates_deleted=2,
            space_saved_mb=50.5,
            duplicate_groups=[],
            timestamp="2026-01-09T10:00:00"
        )

        assert report.total_videos == 10
        assert report.unique_videos == 8
        assert report.duplicates_found == 2

    @pytest.mark.fast
    def test_report_to_dict(self):
        """Test report serialization"""
        report = DeduplicationReport(
            total_videos=10,
            unique_videos=8,
            duplicates_found=2,
            duplicates_deleted=2,
            space_saved_mb=50.5,
            duplicate_groups=[],
            timestamp="2026-01-09T10:00:00"
        )

        data = report.to_dict()

        assert data['total_videos'] == 10
        assert data['space_saved_mb'] == 50.5
        assert 'timestamp' in data


# ============================================================================
# Test Stats Preview
# ============================================================================

class TestStatsPreview:
    """Test stats preview functionality"""

    @patch('src.deduplication.VideoDeduplicator.find_duplicates')
    @pytest.mark.fast
    def test_get_stats_preview(self, mock_find, deduplicator, sample_videos):
        """Test getting stats preview"""
        group = DuplicateGroup(
            keep=sample_videos[0],
            keep_size=1000,
            duplicates=[sample_videos[1], sample_videos[2]],
            similarity=0.95,
            hash_distance=3
        )
        mock_find.return_value = [group]

        stats = deduplicator.get_stats_preview(str(Path(sample_videos[0]).parent))

        assert stats['duplicate_groups'] == 1
        assert stats['total_duplicates'] == 2
        assert 'potential_savings_mb' in stats


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions"""

    @patch('src.deduplication.VideoDeduplicator.deduplicate')
    @pytest.mark.fast
    def test_deduplicate_videos_function(self, mock_deduplicate, temp_dir):
        """Test deduplicate_videos convenience function"""
        mock_report = DeduplicationReport(
            total_videos=5,
            unique_videos=4,
            duplicates_found=1,
            duplicates_deleted=1,
            space_saved_mb=10.0,
            duplicate_groups=[],
            timestamp=datetime.now().isoformat()
        )
        mock_deduplicate.return_value = mock_report

        with patch('src.deduplication.VideoDeduplicator.is_available', return_value=True):
            result = deduplicate_videos(str(temp_dir), auto_delete=True, threshold=10)

        assert result.duplicates_found == 1

    @pytest.mark.fast
    def test_deduplicate_videos_when_unavailable(self, temp_dir):
        """Test convenience function when imagehash unavailable"""
        with patch('src.deduplication.VideoDeduplicator.is_available', return_value=False):
            result = deduplicate_videos(str(temp_dir))

        assert result.total_videos == 0
        assert result.duplicates_found == 0


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    @patch('src.deduplication.VideoDeduplicator._hash_distance')
    @patch('src.deduplication.VideoDeduplicator._compute_hash')
    @pytest.mark.fast
    def test_find_duplicates_with_hash_failures(self, mock_compute, mock_distance, deduplicator, sample_videos):
        """Test handling when some videos fail to hash"""
        # First hash succeeds, second fails, third succeeds
        mock_compute.side_effect = ["hash1", None, "hash2"]

        # Mock distance for the two successful hashes
        mock_distance.return_value = 20  # Not duplicates

        groups = deduplicator.find_duplicates(str(Path(sample_videos[0]).parent))

        # Should handle gracefully, skipping the failed hash
        assert isinstance(groups, list)

    @patch('src.deduplication.VideoDeduplicator.find_duplicates')
    @pytest.mark.fast
    def test_deduplicate_handles_delete_errors(self, mock_find, deduplicator, temp_dir):
        """Test deduplication handles file deletion errors"""
        # Create group with non-existent duplicate
        group = DuplicateGroup(
            keep="/keep/video.mp4",
            keep_size=1000,
            duplicates=["/nonexistent/duplicate.mp4"],
            similarity=0.95,
            hash_distance=3
        )
        mock_find.return_value = [group]

        # Should not crash
        report = deduplicator.deduplicate(str(temp_dir), auto_delete=True)

        assert report.duplicates_deleted == 0

    @pytest.mark.fast
    def test_video_extensions_constant(self):
        """Test VIDEO_EXTENSIONS constant is comprehensive"""
        assert '.mp4' in VIDEO_EXTENSIONS
        assert '.mkv' in VIDEO_EXTENSIONS
        assert '.avi' in VIDEO_EXTENSIONS
        assert len(VIDEO_EXTENSIONS) >= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
