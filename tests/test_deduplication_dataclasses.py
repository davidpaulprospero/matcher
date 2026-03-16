"""
Tests for deduplication dataclasses in src/deduplication.py

Tests dataclass initialization and serialization for:
- DuplicateGroup - Group of duplicate videos
- DeduplicationReport - Summary of deduplication results
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.deduplication import DuplicateGroup, DeduplicationReport


class TestDuplicateGroup:
    """Test DuplicateGroup dataclass"""

    @pytest.mark.fast
    def test_duplicate_group_initialization(self):
        """Test basic initialization"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=1024000,
            duplicates=["video2.mp4", "video3.mp4"],
            similarity=0.95,
            hash_distance=3
        )

        assert group.keep == "video1.mp4"
        assert group.keep_size == 1024000
        assert group.duplicates == ["video2.mp4", "video3.mp4"]
        assert group.similarity == 0.95
        assert group.hash_distance == 3

    @pytest.mark.fast
    def test_duplicate_group_empty_duplicates(self):
        """Test with empty duplicates list"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=1024000,
            duplicates=[],
            similarity=1.0,
            hash_distance=0
        )

        assert group.duplicates == []
        assert group.similarity == 1.0

    @pytest.mark.fast
    def test_duplicate_group_single_duplicate(self):
        """Test with single duplicate"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=2048000,
            duplicates=["video2.mp4"],
            similarity=0.99,
            hash_distance=1
        )

        assert len(group.duplicates) == 1
        assert group.duplicates[0] == "video2.mp4"

    @pytest.mark.fast
    def test_duplicate_group_multiple_duplicates(self):
        """Test with multiple duplicates"""
        duplicates = [f"video{i}.mp4" for i in range(2, 11)]  # 9 duplicates
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=5000000,
            duplicates=duplicates,
            similarity=0.85,
            hash_distance=8
        )

        assert len(group.duplicates) == 9
        assert group.duplicates[0] == "video2.mp4"
        assert group.duplicates[-1] == "video10.mp4"

    @pytest.mark.fast
    def test_duplicate_group_high_similarity(self):
        """Test with high similarity (near identical)"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=1024000,
            duplicates=["video2.mp4"],
            similarity=0.999,
            hash_distance=0
        )

        assert group.similarity >= 0.99
        assert group.hash_distance == 0

    @pytest.mark.fast
    def test_duplicate_group_low_similarity(self):
        """Test with lower similarity (threshold edge case)"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=1024000,
            duplicates=["video2.mp4"],
            similarity=0.75,
            hash_distance=10
        )

        assert group.similarity == 0.75
        assert group.hash_distance == 10

    @pytest.mark.fast
    def test_duplicate_group_path_formats(self):
        """Test with different path formats"""
        group = DuplicateGroup(
            keep="/path/to/video1.mp4",
            keep_size=1024000,
            duplicates=["C:/Windows/path/video2.mp4", "./relative/video3.mp4"],
            similarity=0.90,
            hash_distance=5
        )

        assert group.keep == "/path/to/video1.mp4"
        assert "C:/Windows/path/video2.mp4" in group.duplicates
        assert "./relative/video3.mp4" in group.duplicates

    @pytest.mark.fast
    def test_duplicate_group_large_file_size(self):
        """Test with large file sizes"""
        group = DuplicateGroup(
            keep="large_video.mp4",
            keep_size=5000000000,  # 5 GB
            duplicates=["dup1.mp4"],
            similarity=0.95,
            hash_distance=3
        )

        assert group.keep_size == 5000000000
        assert group.keep_size > 1024 * 1024 * 1024  # > 1 GB


class TestDeduplicationReport:
    """Test DeduplicationReport dataclass"""

    @pytest.mark.fast
    def test_deduplication_report_initialization(self):
        """Test basic initialization"""
        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=100,
            unique_videos=85,
            duplicates_found=15,
            duplicates_deleted=15,
            space_saved_mb=450.5,
            duplicate_groups=[],
            timestamp=timestamp
        )

        assert report.total_videos == 100
        assert report.unique_videos == 85
        assert report.duplicates_found == 15
        assert report.duplicates_deleted == 15
        assert report.space_saved_mb == 450.5
        assert report.duplicate_groups == []
        assert report.timestamp == timestamp

    @pytest.mark.fast
    def test_deduplication_report_with_groups(self):
        """Test with duplicate groups"""
        group1 = DuplicateGroup(
            keep="video1.mp4",
            keep_size=1024000,
            duplicates=["video2.mp4"],
            similarity=0.95,
            hash_distance=3
        )
        group2 = DuplicateGroup(
            keep="video3.mp4",
            keep_size=2048000,
            duplicates=["video4.mp4", "video5.mp4"],
            similarity=0.90,
            hash_distance=5
        )

        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=5,
            unique_videos=2,
            duplicates_found=3,
            duplicates_deleted=3,
            space_saved_mb=100.0,
            duplicate_groups=[group1, group2],
            timestamp=timestamp
        )

        assert len(report.duplicate_groups) == 2
        assert report.duplicate_groups[0] == group1
        assert report.duplicate_groups[1] == group2

    @pytest.mark.fast
    def test_deduplication_report_to_dict(self):
        """Test to_dict() method"""
        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=100,
            unique_videos=85,
            duplicates_found=15,
            duplicates_deleted=15,
            space_saved_mb=450.5,
            duplicate_groups=[],
            timestamp=timestamp
        )

        result = report.to_dict()

        assert result['total_videos'] == 100
        assert result['unique_videos'] == 85
        assert result['duplicates_found'] == 15
        assert result['duplicates_deleted'] == 15
        assert result['space_saved_mb'] == 450.5
        assert result['duplicate_groups'] == []
        assert result['timestamp'] == timestamp

    @pytest.mark.fast
    def test_deduplication_report_to_dict_with_groups(self):
        """Test to_dict() with duplicate groups"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=1024000,
            duplicates=["video2.mp4"],
            similarity=0.95,
            hash_distance=3
        )

        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=2,
            unique_videos=1,
            duplicates_found=1,
            duplicates_deleted=1,
            space_saved_mb=50.0,
            duplicate_groups=[group],
            timestamp=timestamp
        )

        result = report.to_dict()

        assert len(result['duplicate_groups']) == 1
        group_dict = result['duplicate_groups'][0]
        assert group_dict['keep'] == "video1.mp4"
        assert group_dict['keep_size'] == 1024000
        assert group_dict['duplicates'] == ["video2.mp4"]
        assert group_dict['similarity'] == 0.95
        assert group_dict['hash_distance'] == 3

    @pytest.mark.fast
    def test_deduplication_report_space_saved_rounding(self):
        """Test space_saved_mb rounding in to_dict()"""
        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=10,
            unique_videos=8,
            duplicates_found=2,
            duplicates_deleted=2,
            space_saved_mb=123.456789,  # Many decimal places
            duplicate_groups=[],
            timestamp=timestamp
        )

        result = report.to_dict()

        # Should be rounded to 2 decimal places
        assert result['space_saved_mb'] == 123.46

    @pytest.mark.fast
    def test_deduplication_report_no_duplicates(self):
        """Test report with no duplicates found"""
        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=50,
            unique_videos=50,
            duplicates_found=0,
            duplicates_deleted=0,
            space_saved_mb=0.0,
            duplicate_groups=[],
            timestamp=timestamp
        )

        assert report.duplicates_found == 0
        assert report.duplicates_deleted == 0
        assert report.space_saved_mb == 0.0
        assert len(report.duplicate_groups) == 0

    @pytest.mark.fast
    def test_deduplication_report_partial_deletion(self):
        """Test report where not all duplicates were deleted"""
        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=20,
            unique_videos=18,
            duplicates_found=5,
            duplicates_deleted=3,  # Some failed to delete
            space_saved_mb=75.5,
            duplicate_groups=[],
            timestamp=timestamp
        )

        assert report.duplicates_found == 5
        assert report.duplicates_deleted == 3
        assert report.duplicates_deleted < report.duplicates_found


class TestDeduplicationEdgeCases:
    """Test edge cases for deduplication dataclasses"""

    @pytest.mark.fast
    def test_duplicate_group_zero_hash_distance(self):
        """Test with zero hash distance (identical frames)"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=1024000,
            duplicates=["video2.mp4"],
            similarity=1.0,
            hash_distance=0
        )

        assert group.hash_distance == 0
        assert group.similarity == 1.0

    @pytest.mark.fast
    def test_duplicate_group_zero_file_size(self):
        """Test with zero file size (corrupted/empty file)"""
        group = DuplicateGroup(
            keep="video1.mp4",
            keep_size=0,
            duplicates=["video2.mp4"],
            similarity=0.95,
            hash_distance=3
        )

        assert group.keep_size == 0

    @pytest.mark.fast
    def test_deduplication_report_large_numbers(self):
        """Test with large numbers of videos"""
        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=100000,
            unique_videos=95000,
            duplicates_found=5000,
            duplicates_deleted=5000,
            space_saved_mb=50000.0,  # 50 GB
            duplicate_groups=[],
            timestamp=timestamp
        )

        assert report.total_videos == 100000
        assert report.duplicates_found == 5000
        assert report.space_saved_mb == 50000.0

    @pytest.mark.fast
    def test_deduplication_report_multiple_groups_serialization(self):
        """Test serialization with multiple groups"""
        groups = [
            DuplicateGroup(
                keep=f"keep{i}.mp4",
                keep_size=1024000 * i,
                duplicates=[f"dup{i}_1.mp4", f"dup{i}_2.mp4"],
                similarity=0.90 + (i * 0.01),
                hash_distance=5 - i
            )
            for i in range(1, 6)
        ]

        timestamp = datetime.now().isoformat()
        report = DeduplicationReport(
            total_videos=15,
            unique_videos=5,
            duplicates_found=10,
            duplicates_deleted=10,
            space_saved_mb=500.0,
            duplicate_groups=groups,
            timestamp=timestamp
        )

        result = report.to_dict()

        assert len(result['duplicate_groups']) == 5
        # Verify first group
        assert result['duplicate_groups'][0]['keep'] == "keep1.mp4"
        assert result['duplicate_groups'][0]['similarity'] == 0.91
        # Verify last group
        assert result['duplicate_groups'][4]['keep'] == "keep5.mp4"


# ============================================================================
# Test VideoDeduplicator coverage gaps
# ============================================================================

class TestVideoDeduplicatorCoverage:
    """Test VideoDeduplicator edge cases for coverage"""

    @pytest.mark.fast
    def test_compute_hash_not_available_returns_none_line_177(self):
        """Test line 177: _compute_hash returns None when not available"""
        from src.deduplication import VideoDeduplicator

        detector = VideoDeduplicator()
        # Force unavailable
        detector._available = False

        result = detector._compute_hash("/path/to/video.mp4")
        assert result is None

    @pytest.mark.integration
    def test_extract_frame_empty_file_returns_none_line_165(self, tmp_path):
        """Test line 165: _extract_first_frame returns None for empty file"""
        from src.deduplication import VideoDeduplicator
        from unittest.mock import patch, MagicMock

        detector = VideoDeduplicator()

        if not detector._available:
            pytest.skip("Pillow/imagehash not available")

        # Create a temporary video path
        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        # Mock subprocess.run to succeed but create an empty frame
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # The frame_path won't exist after the mock, so it should return None
            result = detector._extract_first_frame(str(video_path))

            # Should return None because no frame was actually created
            assert result is None

    @pytest.mark.fast
    def test_compute_hash_exception_returns_none_lines_201_203(self, tmp_path):
        """Test lines 201-203: _compute_hash returns None on exception"""
        from src.deduplication import VideoDeduplicator
        from unittest.mock import patch, MagicMock

        detector = VideoDeduplicator()

        if not detector._available:
            pytest.skip("Pillow/imagehash not available")

        # Mock _extract_first_frame to return a fake path
        with patch.object(detector, '_extract_first_frame') as mock_extract:
            mock_extract.return_value = str(tmp_path / "fake_frame.jpg")

            # Mock Image.open to raise exception
            with patch.object(detector.Image, 'open', side_effect=Exception("Image load failed")):
                result = detector._compute_hash("/path/to/video.mp4")

                # Should return None due to exception
                assert result is None

    @pytest.mark.fast
    def test_hash_distance_not_available_returns_999_line_218(self):
        """Test line 218 (not explicitly listed but related): returns 999 when not available"""
        from src.deduplication import VideoDeduplicator

        detector = VideoDeduplicator()
        detector._available = False

        result = detector._hash_distance("abc123", "def456")
        assert result == 999

    @pytest.mark.fast
    def test_hash_distance_exception_returns_999_lines_224_225(self):
        """Test lines 224-225: _hash_distance returns 999 on exception"""
        from src.deduplication import VideoDeduplicator
        from unittest.mock import patch

        detector = VideoDeduplicator()

        if not detector._available:
            pytest.skip("Pillow/imagehash not available")

        # Mock hex_to_hash to raise exception
        with patch.object(detector.imagehash, 'hex_to_hash', side_effect=Exception("Invalid hash")):
            result = detector._hash_distance("invalid_hash1", "invalid_hash2")

            # Should return 999 due to exception
            assert result == 999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
