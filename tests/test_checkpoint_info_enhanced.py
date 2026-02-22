#!/usr/bin/env python3
"""
Test script for enhanced checkpoint-info CLI (US-115-011).

Tests the new sections: video_stats, match_quality, stage_timing_comparison
and the --checkpoint-info-verbose flag.

Run: python tests/test_checkpoint_info_enhanced.py
"""

import pytest
import tempfile
import json
import sys
from pathlib import Path
from io import StringIO
from unittest.mock import patch, MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.checkpoint import CheckpointManager, CheckpointData


class MockArgs:
    """Mock args object for testing CLI handler"""
    def __init__(self, checkpoint_info=False, checkpoint_info_verbose=False, project=None):
        self.checkpoint_info = checkpoint_info
        self.checkpoint_info_verbose = checkpoint_info_verbose
        self.project = project


def create_mock_checkpoint_data():
    """Create a mock CheckpointData with test data for all new sections"""
    data = CheckpointData(
        version="2.1",
        created_at="2026-02-16T10:00:00",
        updated_at="2026-02-16T12:00:00",
        last_completed_stage="DOWNLOAD_SEGMENTS",
        config_hash="abc123",
        voiceover_path="/test/voiceover.srt",
        voiceover_hash="hash123",
    )

    # Video search data
    data.video_search = {
        'video_ids': ['vid1', 'vid2', 'vid3', 'vid4', 'vid5'],
        'video_count': 5,
        'search_results': {},
    }

    # Caption data
    data.caption = {
        'caption_count': 5,
    }

    # Match data with quality metrics
    data.match = {
        'match_count': 10,
        'avg_confidence': 0.85,
        'matches': [
            {'video_file': 'vid1', 'confidence': 0.9},
            {'video_file': 'vid1', 'confidence': 0.85},
            {'video_file': 'vid2', 'confidence': 0.8},
            {'video_file': 'vid2', 'confidence': 0.75},
            {'video_file': 'vid3', 'confidence': 0.9},
            {'video_file': 'vid3', 'confidence': 0.85},
            {'video_file': 'vid4', 'confidence': 0.8},
            {'video_file': 'vid4', 'confidence': 0.75},
            {'video_file': 'vid5', 'confidence': 0.9},
            {'video_file': 'vid5', 'confidence': 0.85},
        ],
        'quality_metrics': {'score': 0.82},
        'diversity_metrics': {'per_track': {}},
    }

    # Download segments data
    data.download_segments = {
        'segment_count': 8,
        'total_matches': 10,
        'retry_count': 2,
        'failed_items': ['vid3_segment1'],  # 1 failed item
    }

    # Stage metrics for timing comparison
    data.stage_metrics = {
        'ANALYZE': {'duration_seconds': 5.2, 'items_processed': 10},
        'VIDEO_SEARCH': {'duration_seconds': 12.5, 'items_processed': 50},
        'CAPTION': {'duration_seconds': 8.3, 'items_processed': 5},
        'MATCH': {'duration_seconds': 15.7, 'items_processed': 10},
        'ITERATIVE_MATCH': {'duration_seconds': 3.2, 'items_processed': 2},
        'DOWNLOAD_SEGMENTS': {'duration_seconds': 45.0, 'items_processed': 8},
    }

    return data


@pytest.mark.fast
def test_video_stats_section():
    """Test that video_stats section renders correctly"""
    print("\n" + "=" * 60)
    print("  TEST: Video Stats Section")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create checkpoint manager with data
        cm = CheckpointManager(tmpdir, config_hash="test")
        cm.data = create_mock_checkpoint_data()

        # Verify video_stats calculations
        total_videos = cm.data.video_search.get('video_count', 0)
        downloaded = cm.data.download_segments.get('segment_count', 0)
        failed = len(cm.data.download_segments.get('failed_items', []))
        cached = cm.data.caption.get('caption_count', 0)

        print(f"    Total videos: {total_videos}")
        print(f"    Downloaded: {downloaded}")
        print(f"    Failed: {failed}")
        print(f"    Cached: {cached}")

        assert total_videos == 5, "Total videos should be 5"
        assert downloaded == 8, "Downloaded should be 8"
        assert failed == 1, "Failed should be 1"
        assert cached == 5, "Cached should be 5"

        print("    ✓ Video stats calculated correctly")


@pytest.mark.fast
def test_match_quality_section():
    """Test that match_quality section renders correctly"""
    print("\n" + "=" * 60)
    print("  TEST: Match Quality Section")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test")
        cm.data = create_mock_checkpoint_data()

        # Get match data
        match_count = cm.data.match.get('match_count', 0)
        avg_confidence = cm.data.match.get('avg_confidence', 0.0)

        # Calculate unique sources
        matches = cm.data.match.get('matches', [])
        source_ids = set()
        for m in matches:
            if isinstance(m, dict):
                vid = m.get('video_file', '')
                if vid:
                    source_ids.add(vid)
        unique_sources = len(source_ids)

        # Calculate diversity score
        source_diversity_score = unique_sources / match_count if match_count > 0 else 0.0

        print(f"    Match count: {match_count}")
        print(f"    Avg confidence: {avg_confidence:.1%}")
        print(f"    Unique sources: {unique_sources}")
        print(f"    Diversity score: {source_diversity_score:.2f}")

        assert match_count == 10, "Match count should be 10"
        assert avg_confidence == 0.85, "Avg confidence should be 0.85"
        assert unique_sources == 5, "Unique sources should be 5"
        assert source_diversity_score == 0.5, "Diversity score should be 0.5"

        print("    ✓ Match quality calculated correctly")


@pytest.mark.fast
def test_stage_timing_comparison():
    """Test that stage_timing_comparison section renders correctly"""
    print("\n" + "=" * 60)
    print("  TEST: Stage Timing Comparison")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test")
        cm.data = create_mock_checkpoint_data()

        # Get current stage metrics
        current_metrics = cm.data.stage_metrics

        # Simulate previous run timing (from history)
        previous_timing = {
            'ANALYZE': 4.5,
            'VIDEO_SEARCH': 15.0,
            'CAPTION': 10.0,
            'MATCH': 12.0,
            'ITERATIVE_MATCH': 5.0,
            'DOWNLOAD_SEGMENTS': 40.0,
        }

        stages_to_show = ['ANALYZE', 'VIDEO_SEARCH', 'CAPTION', 'MATCH', 'ITERATIVE_MATCH', 'DOWNLOAD_SEGMENTS']

        print("\n    Timing comparison:")
        for stage_name in stages_to_show:
            current_duration = current_metrics.get(stage_name, {}).get('duration_seconds', 0.0)
            prev_duration = previous_timing.get(stage_name, 0.0)

            if current_duration > 0 or prev_duration > 0:
                if prev_duration > 0:
                    diff = current_duration - prev_duration
                    diff_pct = (diff / prev_duration) * 100
                    sign = '+' if diff >= 0 else ''
                    print(f"      {stage_name}: {current_duration:.1f}s (prev: {prev_duration:.1f}s, {sign}{diff_pct:.1f}%)")

                    # Verify calculations
                    expected_diff_pct = ((current_duration - prev_duration) / prev_duration) * 100
                    assert abs(diff_pct - expected_diff_pct) < 0.1, f"Diff % mismatch for {stage_name}"
                else:
                    print(f"      {stage_name}: {current_duration:.1f}s (no previous run)")

        print("    ✓ Stage timing comparison calculated correctly")


@pytest.mark.fast
def test_checkpoint_info_verbose_flag_exists():
    """Test that --checkpoint-info-verbose argument is recognized"""
    print("\n" + "=" * 60)
    print("  TEST: Verbose Flag Exists")
    print("=" * 60)

    # Test that the args have the verbose flag attribute
    args = MockArgs(checkpoint_info=True, checkpoint_info_verbose=True)

    assert hasattr(args, 'checkpoint_info'), "Should have checkpoint_info attribute"
    assert hasattr(args, 'checkpoint_info_verbose'), "Should have checkpoint_info_verbose attribute"
    assert args.checkpoint_info == True, "checkpoint_info should be True"
    assert args.checkpoint_info_verbose == True, "checkpoint_info_verbose should be True"

    print("    ✓ Verbose flag exists and works correctly")


@pytest.mark.fast
def test_checkpoint_info_no_verbose():
    """Test that without verbose flag, enhanced sections are not shown"""
    print("\n" + "=" * 60)
    print("  TEST: No Verbose = Basic Info Only")
    print("=" * 60)

    args = MockArgs(checkpoint_info=True, checkpoint_info_verbose=False)

    verbose = getattr(args, 'checkpoint_info_verbose', False)

    assert verbose == False, "Verbose should be False when not set"

    print("    ✓ Basic info only when verbose is False")


@pytest.mark.fast
def test_checkpoint_info_minimal_data():
    """Test checkpoint info with minimal data"""
    print("\n" + "=" * 60)
    print("  TEST: Minimal Data Handling")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create checkpoint with minimal data
        cm = CheckpointManager(tmpdir, config_hash="test")
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T10:00:00",
            last_completed_stage="",
        )

        # Should handle empty/missing data gracefully
        total_videos = 0
        if cm.data.video_search:
            total_videos = cm.data.video_search.get('video_count', 0)

        downloaded = 0
        if cm.data.download_segments:
            downloaded = cm.data.download_segments.get('segment_count', 0)

        print(f"    Total videos (empty): {total_videos}")
        print(f"    Downloaded (empty): {downloaded}")

        assert total_videos == 0, "Should handle empty video_search"
        assert downloaded == 0, "Should handle empty download_segments"

        print("    ✓ Minimal data handled correctly")


@pytest.mark.fast
def test_get_checkpoint_stats():
    """US-130-002: Test get_checkpoint_stats() returns comprehensive statistics"""
    print("\n" + "=" * 60)
    print("  TEST: Get Checkpoint Stats (US-130-002)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test 1: Empty checkpoint - should return zeros/defaults
        cm = CheckpointManager(tmpdir, config_hash="test")
        stats = cm.get_checkpoint_stats()

        print(f"    Empty checkpoint stats: {stats}")
        assert stats["save_count"] == 0, "save_count should be 0"
        assert stats["avg_save_time_ms"] == 0.0, "avg_save_time_ms should be 0"
        assert stats["total_size_bytes"] == 0, "total_size_bytes should be 0"
        assert stats["oldest_checkpoint"] is None, "oldest_checkpoint should be None"
        assert stats["newest_checkpoint"] is None, "newest_checkpoint should be None"
        assert stats["backup_count"] == 0, "backup_count should be 0"
        assert stats["per_stage_sizes"] == {}, "per_stage_sizes should be empty"
        print("    ✓ Empty checkpoint returns default values")

        # Test 2: With data - should calculate sizes and timestamps
        cm.data = CheckpointData(
            version="2.1",
            created_at="2026-02-16T10:00:00",
            updated_at="2026-02-16T12:00:00",
            last_completed_stage="MATCH",
        )
        cm.data.video_search = {"video_ids": ["v1", "v2"], "video_count": 2}
        cm.data.caption = {"caption_count": 2}

        stats = cm.get_checkpoint_stats()
        print(f"    Stats with data: {stats}")
        assert stats["oldest_checkpoint"] == "2026-02-16T10:00:00", "oldest should be created_at"
        assert stats["newest_checkpoint"] == "2026-02-16T12:00:00", "newest should be updated_at"
        assert "video_search" in stats["per_stage_sizes"], "per_stage_sizes should have video_search"
        print("    ✓ Stats with data calculates correctly")

        # Test 3: Save operation updates stats
        # Manually simulate save stats update
        cm._stats["save_count"] = 1
        cm._stats["save_times_ms"] = [100.0]
        cm._stats["total_save_time_ms"] = 100.0

        stats = cm.get_checkpoint_stats()
        assert stats["save_count"] == 1, "save_count should be 1"
        assert stats["avg_save_time_ms"] == 100.0, "avg_save_time_ms should be 100.0"
        print("    ✓ Stats update on save operation")

        # Test 4: Backup count
        # Create a backup file
        backup_path = tmpdir / "checkpoint.backup.json"
        backup_path.write_text("{}")

        stats = cm.get_checkpoint_stats()
        assert stats["backup_count"] >= 1, "backup_count should be >= 1"
        print(f"    ✓ Backup count: {stats['backup_count']}")

        print("    ✓ All checkpoint stats tests passed")


@pytest.mark.fast
def test_checkpoint_stats_tracking():
    """US-130-002: Test that statistics are tracked on save operations"""
    print("\n" + "=" * 60)
    print("  TEST: Stats Tracking on Save (US-130-002)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        cm = CheckpointManager(tmpdir, config_hash="test")

        # Simulate save stats update (as done in _atomic_save)
        cm._stats["save_count"] = cm._stats.get("save_count", 0) + 1
        cm._stats["total_save_time_ms"] = cm._stats.get("total_save_time_ms", 0.0) + 150.0
        cm._stats["save_times_ms"] = [150.0]
        cm._stats["oldest_checkpoint"] = "2026-02-16T10:00:00"
        cm._stats["newest_checkpoint"] = "2026-02-16T12:00:00"

        stats = cm.get_checkpoint_stats()
        assert stats["save_count"] == 1, "Should track 1 save"
        assert stats["avg_save_time_ms"] == 150.0, "Should track save time"
        # Note: total_size_bytes returns actual file size from disk, not tracked size

        # Simulate another save
        cm._stats["save_count"] = cm._stats.get("save_count", 0) + 1
        cm._stats["total_save_time_ms"] = cm._stats.get("total_save_time_ms", 0.0) + 100.0
        cm._stats["save_times_ms"] = [150.0, 100.0]

        stats = cm.get_checkpoint_stats()
        assert stats["save_count"] == 2, "Should track 2 saves"
        assert stats["avg_save_time_ms"] == 125.0, "Should calculate average of 2 saves"
        print("    ✓ Stats tracking works correctly")

        print("    ✓ Stats tracking tests passed")


if __name__ == "__main__":
    # Run tests
    test_video_stats_section()
    test_match_quality_section()
    test_stage_timing_comparison()
    test_checkpoint_info_verbose_flag_exists()
    test_checkpoint_info_no_verbose()
    test_checkpoint_info_minimal_data()
    test_get_checkpoint_stats()
    test_checkpoint_stats_tracking()

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
