"""
Advanced tests for keyword_remix.py - Process Directory & LLM Methods

Targets uncovered lines:
- Lines 456-580: process_directory() with parallel/sequential scoring
- Lines 619-681: KeywordRemixer LLM-based keyword generation
- Lines 704-773: Batch remixing
- Lines 1112-1157: Convenience functions with detailed paths
- Lines 1161-1212: Audio file remix

Created: 2026-01-10 (Session 10)
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, mock_open
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.keyword_remix import (
    VideoScore,
    RemixResult,
    RemixConfig,
    KeywordRemixProcessor,
    KeywordRemixer,
    KeywordRemixResult,
    remix_downloaded_videos,
    remix_audio_files
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def remix_config():
    """Create basic remix config"""
    return RemixConfig(
        enabled=True,
        min_relevance_score=0.1,
        max_files_to_process=100,
        max_files_to_include=50,
        parallel_scoring=False,
        log_file_processing=False
    )


@pytest.fixture
def sample_keywords():
    """Sample keywords for testing"""
    return ["travel", "beach", "vacation", "sunset"]


# ============================================================================
# Test Process Directory (Lines 456-580)
# ============================================================================

class TestProcessVideos:
    """Test process_videos method with various scenarios"""

    @pytest.mark.fast
    def test_process_videos_empty(self, temp_dir, remix_config, sample_keywords):
        """Test process_videos with empty directory"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        result = processor.process_videos(temp_dir, show_progress=False)

        assert result.total_files == 0
        assert result.included_files == 0
        assert result.excluded_files == 0
        assert result.avg_match_score == 0.0

    @pytest.mark.fast
    def test_process_videos_with_files(self, temp_dir, remix_config, sample_keywords):
        """Test process_videos with actual video files"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        # Create test files with relevant names
        (temp_dir / "beach_vacation.mp4").write_text("fake video")
        (temp_dir / "travel_vlog.mp4").write_text("fake video")
        (temp_dir / "random_file.mp4").write_text("fake video")

        result = processor.process_videos(temp_dir, show_progress=False)

        assert result.total_files == 3
        assert result.included_files >= 2  # At least 2 should match keywords
        assert result.processing_time_seconds >= 0

    @pytest.mark.fast
    def test_process_videos_parallel_scoring(self, temp_dir, sample_keywords):
        """Test parallel scoring with many files"""
        config = RemixConfig(
            parallel_scoring=True,
            max_workers=2,
            log_file_processing=False,
            interactive_curation=False,  # Disable interactive prompts
            min_relevance_score=0.1  # Lower threshold to allow more matches
        )
        processor = KeywordRemixProcessor(config, sample_keywords)

        # Create 15 files to trigger parallel processing (>10)
        # Use filenames that contain keywords for scoring
        keywords_cycle = ['travel', 'beach', 'vacation']
        for i in range(15):
            keyword = keywords_cycle[i % len(keywords_cycle)]
            (temp_dir / f"{keyword}_video{i}.mp4").write_text(f"{keyword} video {i}")

        result = processor.process_videos(temp_dir, show_progress=False)

        assert result.total_files == 15
        assert result.included_files > 0

    @pytest.mark.fast
    def test_process_videos_sequential_scoring(self, temp_dir, sample_keywords):
        """Test sequential scoring with few files"""
        config = RemixConfig(
            parallel_scoring=False,
            log_file_processing=False,
            interactive_curation=False,  # Disable interactive prompts
            min_relevance_score=0.1  # Lower threshold to allow more matches
        )
        processor = KeywordRemixProcessor(config, sample_keywords)

        # Create 5 files (sequential path)
        # Use filenames that contain keywords for scoring
        for i in range(5):
            (temp_dir / f"beach_vacation_{i}.mp4").write_text(f"beach vacation {i}")

        result = processor.process_videos(temp_dir, show_progress=False)

        assert result.total_files == 5
        assert result.included_files > 0

    @pytest.mark.fast
    def test_process_videos_with_progress(self, temp_dir, remix_config, sample_keywords, capsys):
        """Test show_progress output"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        # Create files
        for i in range(3):
            (temp_dir / f"travel{i}.mp4").write_text("travel video")

        result = processor.process_videos(temp_dir, show_progress=True)

        captured = capsys.readouterr()
        assert "Scoring" in captured.out
        assert "videos" in captured.out

    @pytest.mark.fast
    def test_process_videos_scoring_error(self, temp_dir, remix_config, sample_keywords):
        """Test handling of scoring errors during processing"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        # Create test file
        (temp_dir / "test.mp4").write_text("test")

        # Mock score_video to raise exception
        with patch.object(processor, 'score_video', side_effect=Exception("Mock error")):
            result = processor.process_videos(temp_dir, show_progress=False)

        # Should handle error gracefully
        assert result.total_files == 1
        assert result.included_files == 0  # No successful scores
        assert processor.metrics['scoring_errors'] >= 1

    @pytest.mark.fast
    def test_process_videos_filtering(self, temp_dir, sample_keywords):
        """Test relevance filtering in process_videos"""
        config = RemixConfig(
            min_relevance_score=0.5,
            log_file_processing=False
        )
        processor = KeywordRemixProcessor(config, sample_keywords)

        # Create files with varying relevance
        (temp_dir / "travel_beach_vacation.mp4").write_text("travel beach")  # High score
        (temp_dir / "random_video.mp4").write_text("random")  # Low score

        result = processor.process_videos(temp_dir, show_progress=False)

        # High relevance file should be included
        assert result.included_files >= 1
        assert all(v.match_score >= 0.5 for v in result.included_videos)

    @pytest.mark.fast
    def test_process_videos_max_files_limit(self, temp_dir, sample_keywords):
        """Test max_files_to_include limit in process_videos"""
        config = RemixConfig(
            max_files_to_include=2,
            log_file_processing=False
        )
        processor = KeywordRemixProcessor(config, sample_keywords)

        # Create 5 highly relevant files
        for i in range(5):
            (temp_dir / f"travel_beach_vacation{i}.mp4").write_text("travel beach")

        result = processor.process_videos(temp_dir, show_progress=False)

        # Should only include 2 (max limit)
        assert result.included_files == 2
        assert result.excluded_files == 3

    @pytest.mark.fast
    def test_process_videos_sort_by_score(self, temp_dir, remix_config, sample_keywords):
        """Test that videos are sorted by score (highest first)"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        # Create files with different relevance
        (temp_dir / "travel.mp4").write_text("travel")  # 1 keyword
        (temp_dir / "travel_beach.mp4").write_text("travel beach")  # 2 keywords
        (temp_dir / "travel_beach_vacation.mp4").write_text("travel beach vacation")  # 3 keywords

        result = processor.process_videos(temp_dir, show_progress=False)

        # Included videos should be sorted by score (descending)
        scores = [v.match_score for v in result.included_videos]
        assert scores == sorted(scores, reverse=True)


# ============================================================================
# Test KeywordRemixer Initialization
# ============================================================================

class TestKeywordRemixerInit:
    """Test KeywordRemixer initialization (simpler, no LLM mocking)"""

    @pytest.mark.fast
    def test_init_with_none_config(self):
        """Test initialization with None config"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="travel videos",
            cache_dir=None
        )

        assert remixer.topic_context == "travel videos"
        assert remixer.stats['total_remixes'] == 0
        assert remixer.stats['successful_remixes'] == 0

    @pytest.mark.fast
    def test_init_with_topic_context(self):
        """Test initialization with topic context"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="beach vacation content",
            cache_dir=None
        )

        assert "beach vacation" in remixer.topic_context

    @pytest.mark.fast
    def test_stats_initialization(self):
        """Test stats dictionary is properly initialized"""
        remixer = KeywordRemixer(
            config=None,
            topic_context="test",
            cache_dir=None
        )

        assert 'total_remixes' in remixer.stats
        assert 'successful_remixes' in remixer.stats
        assert 'cache_hits' in remixer.stats
        assert 'api_calls' in remixer.stats  # Not 'failed_remixes'


# ============================================================================
# Test Convenience Functions (Lines 1112-1157, 1161-1212)
# ============================================================================

class TestConvenienceFunctionsAdvanced:
    """Test convenience functions with detailed code paths"""

    @pytest.mark.fast
    def test_remix_downloaded_videos_with_config(self, temp_dir):
        """Test remix_downloaded_videos with full config"""
        config = RemixConfig(
            enabled=True,
            min_relevance_score=0.3,
            log_file_processing=False,
            interactive_curation=False,  # Disable interactive prompts
            auto_accept_filter="filtered"  # Auto-accept filtered results
        )

        # Create video files
        (temp_dir / "travel_video.mp4").write_text("travel")
        (temp_dir / "beach_vacation.mp4").write_text("beach")

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=["travel", "beach"],
            config=config,
            show_progress=False
        )

        assert result is not None
        assert result.total_files == 2
        assert result.included_files >= 1
        assert len(selected) >= 1

    @pytest.mark.fast
    def test_remix_downloaded_videos_disabled(self, temp_dir):
        """Test remix_downloaded_videos when disabled"""
        config = RemixConfig(enabled=False)

        # Create a video file
        (temp_dir / "test.mp4").write_text("test")

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=["travel"],
            config=config
        )

        # When disabled, returns all videos with None result
        assert result is None
        assert len(selected) >= 1  # Should return all videos

    @pytest.mark.fast
    def test_remix_downloaded_videos_no_keywords(self, temp_dir):
        """Test remix_downloaded_videos with empty keywords"""
        # Create a video file
        (temp_dir / "test.mp4").write_text("test")

        config = RemixConfig(
            interactive_curation=False,
            auto_accept_filter="filtered"
        )

        selected, result = remix_downloaded_videos(
            video_dir=temp_dir,
            keywords=[],
            config=config,
            interactive=False,  # Disable prompts
            show_progress=False
        )

        # With empty keywords, all videos excluded (score=0)
        assert result is not None
        assert result.total_files == 1
        assert result.excluded_files == 1  # All excluded due to no keywords

    @pytest.mark.fast
    def test_remix_audio_files_basic(self, temp_dir):
        """Test remix_audio_files with audio files"""
        # Create audio files
        audio1 = temp_dir / "travel_podcast.mp3"
        audio2 = temp_dir / "beach_sounds.m4a"
        audio3 = temp_dir / "random.wav"

        audio1.write_text("travel")
        audio2.write_text("beach")
        audio3.write_text("random")

        audio_files = [str(audio1), str(audio2), str(audio3)]

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=["travel", "beach"],
            config=None,
            interactive=False,  # Disable prompts
            show_progress=False
        )

        assert result is not None
        assert result.total_files == 3
        assert len(selected) >= 2  # travel and beach files should match

    @pytest.mark.fast
    def test_remix_audio_files_with_filtering(self, temp_dir):
        """Test remix_audio_files with relevance filtering"""
        config = RemixConfig(
            min_relevance_score=0.5,
            log_file_processing=False
        )

        # Create audio files
        audio1 = temp_dir / "travel_beach_vacation.mp3"
        audio2 = temp_dir / "random.mp3"

        audio1.write_text("travel beach")
        audio2.write_text("random")

        audio_files = [str(audio1), str(audio2)]

        selected, result = remix_audio_files(
            audio_files=audio_files,
            keywords=["travel", "beach", "vacation"],
            config=config,
            interactive=False,
            show_progress=False
        )

        assert result.included_files >= 1
        assert all(v.match_score >= 0.5 for v in result.included_videos)


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class TestProcessVideosEdgeCases:
    """Test edge cases in process_videos"""

    @pytest.mark.fast
    def test_process_videos_parallel_with_errors(self, temp_dir, sample_keywords):
        """Test parallel scoring with some files causing errors"""
        config = RemixConfig(
            parallel_scoring=True,
            max_workers=2,
            log_file_processing=False
        )
        processor = KeywordRemixProcessor(config, sample_keywords)

        # Create many files to trigger parallel
        for i in range(15):
            (temp_dir / f"video{i}.mp4").write_text("travel")

        # Mock score_video to fail for some files
        original_score = processor.score_video
        call_count = [0]

        def mock_score_video(file_path):
            call_count[0] += 1
            if call_count[0] % 3 == 0:  # Every 3rd call fails
                raise Exception("Mock error")
            return original_score(file_path)

        with patch.object(processor, 'score_video', side_effect=mock_score_video):
            result = processor.process_videos(temp_dir, show_progress=False)

        # Should handle errors and process successfully scoring files
        assert result.total_files == 15
        assert processor.metrics['scoring_errors'] > 0

    @pytest.mark.fast
    def test_process_videos_all_excluded(self, temp_dir, sample_keywords):
        """Test when all files are excluded by filtering"""
        config = RemixConfig(
            min_relevance_score=0.9,  # Very high threshold
            log_file_processing=False
        )
        processor = KeywordRemixProcessor(config, sample_keywords)

        # Create files with low relevance
        (temp_dir / "random1.mp4").write_text("random")
        (temp_dir / "random2.mp4").write_text("other")

        result = processor.process_videos(temp_dir, show_progress=False)

        assert result.included_files == 0
        assert result.excluded_files == 2
        assert result.avg_match_score == 0.0

    @pytest.mark.fast
    def test_process_videos_metrics_tracking(self, temp_dir, remix_config, sample_keywords):
        """Test that metrics are properly tracked"""
        processor = KeywordRemixProcessor(remix_config, sample_keywords)

        # Create test files
        (temp_dir / "travel.mp4").write_text("travel")
        (temp_dir / "beach.mp4").write_text("beach")

        result = processor.process_videos(temp_dir, show_progress=False)

        assert processor.metrics['files_scored'] == 2
        assert processor.metrics['start_time'] > 0
        assert processor.metrics['end_time'] > processor.metrics['start_time']
        assert result.processing_time_seconds > 0
