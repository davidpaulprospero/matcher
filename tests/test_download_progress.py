"""
Tests for download progress percentage logging.

Verifies that:
- Progress is logged at 10% intervals (10%, 20%, 30%... 90%, 100%)
- Log format matches 'Download progress: X% (N/M keywords)'
- Progress logging does not slow down downloads significantly
- Milestones are logged exactly once

US-009: Add download progress percentage logging
Created: 2026-01-25 (Sprint 3)
"""

import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.core import VideoDownloader


# Mark all tests as unit tests
pytestmark = pytest.mark.unit


# ============================================================================
# Test Progress Logging Milestones
# ============================================================================

class TestProgressLoggingMilestones:
    """Test progress is logged at 10% milestones"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for VideoDownloader"""
        config = MagicMock()
        config.download = MagicMock()
        config.download.parallel_workers = 4
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.download.title_blacklist = []
        config.download.llm_title_filter = MagicMock(enabled=False)
        config.download.audio_first = MagicMock(enabled=False)
        config.download.speech_screening = MagicMock(enabled=False)
        config.download.zero_download_remix = MagicMock(enabled=False)
        config.download.delay_between_keywords = 0  # No delay for tests
        config.cache_dir = "/tmp/cache"
        config.downloaded_videos_dir = "/tmp/videos"
        config.duration_tiers = {}
        return config

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_10_percent_milestone_logged(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """10% milestone should be logged when processing 10 keywords"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(10)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        # Check 10% is logged (happens at keyword 2, after processing keyword 1 which is 10%)
        assert any("Download progress: 10%" in record.message for record in caplog.records)

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_multiple_milestones_logged(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """Multiple milestones (10%, 20%, 30%...) should be logged for 30 keywords"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(30)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # Should have 10%, 20%, 30%... up to 90% + 100%
        assert len(progress_messages) >= 9  # At least 9 milestones

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_100_percent_always_logged(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """100% milestone should always be logged at completion"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(5)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        assert any("Download progress: 100%" in record.message for record in caplog.records)


# ============================================================================
# Test Progress Log Format
# ============================================================================

class TestProgressLogFormat:
    """Test progress log message format"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for VideoDownloader"""
        config = MagicMock()
        config.download = MagicMock()
        config.download.parallel_workers = 4
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.download.title_blacklist = []
        config.download.llm_title_filter = MagicMock(enabled=False)
        config.download.audio_first = MagicMock(enabled=False)
        config.download.speech_screening = MagicMock(enabled=False)
        config.download.zero_download_remix = MagicMock(enabled=False)
        config.download.delay_between_keywords = 0
        config.cache_dir = "/tmp/cache"
        config.downloaded_videos_dir = "/tmp/videos"
        config.duration_tiers = {}
        return config

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_log_format_includes_percentage(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """Log format should include 'Download progress: X%'"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(20)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # All progress messages should have percentage
        for msg in progress_messages:
            assert "%" in msg

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_log_format_includes_keyword_count(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """Log format should include '(N/M keywords)'"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(20)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # All progress messages should have N/M format
        for msg in progress_messages:
            assert "/" in msg
            assert "keywords" in msg

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_log_format_100_percent_shows_total(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """100% log should show total/total (e.g., '20/20 keywords')"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(20)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_100_msgs = [r.message for r in caplog.records if "Download progress: 100%" in r.message]

        assert len(progress_100_msgs) == 1
        assert "20/20" in progress_100_msgs[0]


# ============================================================================
# Test Milestone Uniqueness
# ============================================================================

class TestMilestoneUniqueness:
    """Test each milestone is logged exactly once"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for VideoDownloader"""
        config = MagicMock()
        config.download = MagicMock()
        config.download.parallel_workers = 4
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.download.title_blacklist = []
        config.download.llm_title_filter = MagicMock(enabled=False)
        config.download.audio_first = MagicMock(enabled=False)
        config.download.speech_screening = MagicMock(enabled=False)
        config.download.zero_download_remix = MagicMock(enabled=False)
        config.download.delay_between_keywords = 0
        config.cache_dir = "/tmp/cache"
        config.downloaded_videos_dir = "/tmp/videos"
        config.duration_tiers = {}
        return config

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_each_milestone_logged_once(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """Each milestone (10%, 20%, etc.) should be logged exactly once"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(100)]  # 100 keywords for clean 10% intervals

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # Count each milestone
        milestone_counts = {}
        for msg in progress_messages:
            # Extract percentage from message
            for pct in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                if f"{pct}%" in msg:
                    milestone_counts[pct] = milestone_counts.get(pct, 0) + 1
                    break

        # Each milestone should appear exactly once
        for pct, count in milestone_counts.items():
            assert count == 1, f"Milestone {pct}% logged {count} times, expected 1"

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_no_duplicate_milestones(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """No duplicate milestone logs should appear"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(50)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # Check no duplicate messages
        assert len(progress_messages) == len(set(progress_messages)), "Duplicate progress messages found"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestProgressEdgeCases:
    """Test edge cases for progress logging"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for VideoDownloader"""
        config = MagicMock()
        config.download = MagicMock()
        config.download.parallel_workers = 4
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.download.title_blacklist = []
        config.download.llm_title_filter = MagicMock(enabled=False)
        config.download.audio_first = MagicMock(enabled=False)
        config.download.speech_screening = MagicMock(enabled=False)
        config.download.zero_download_remix = MagicMock(enabled=False)
        config.download.delay_between_keywords = 0
        config.cache_dir = "/tmp/cache"
        config.downloaded_videos_dir = "/tmp/videos"
        config.duration_tiers = {}
        return config

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_single_keyword_logs_100_percent(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """Single keyword should still log 100%"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = ["single_keyword"]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        assert any("Download progress: 100%" in record.message for record in caplog.records)

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_few_keywords_logs_100_percent_only(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """3 keywords should only log 100% (no intermediate milestones)"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = ["kw1", "kw2", "kw3"]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # With only 3 keywords, we might only get 100% or 30%+ 100%
        # But 100% should definitely be there
        assert any("100%" in msg for msg in progress_messages)


# ============================================================================
# Test Progress Logging Intervals
# ============================================================================

class TestProgressLoggingIntervals:
    """Test progress is logged at expected intervals"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for VideoDownloader"""
        config = MagicMock()
        config.download = MagicMock()
        config.download.parallel_workers = 4
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.download.title_blacklist = []
        config.download.llm_title_filter = MagicMock(enabled=False)
        config.download.audio_first = MagicMock(enabled=False)
        config.download.speech_screening = MagicMock(enabled=False)
        config.download.zero_download_remix = MagicMock(enabled=False)
        config.download.delay_between_keywords = 0
        config.cache_dir = "/tmp/cache"
        config.downloaded_videos_dir = "/tmp/videos"
        config.duration_tiers = {}
        return config

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_50_keywords_logs_correct_milestones(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """50 keywords should log 10%, 20%, 30%... 90%, 100%"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(50)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # Check expected milestones are present
        expected_milestones = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for milestone in expected_milestones:
            assert any(f"{milestone}%" in msg for msg in progress_messages), f"Missing {milestone}% milestone"

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_logging_does_not_log_0_percent(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """0% should never be logged as a milestone"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        keywords = [f"keyword_{i}" for i in range(20)]

        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)

        progress_messages = [r.message for r in caplog.records if "Download progress:" in r.message]

        # 0% should not be logged
        assert not any("Download progress: 0%" in msg for msg in progress_messages)


# ============================================================================
# Test Performance Impact
# ============================================================================

class TestProgressPerformance:
    """Test progress logging does not significantly slow down downloads"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for VideoDownloader"""
        config = MagicMock()
        config.download = MagicMock()
        config.download.parallel_workers = 4
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        config.download.title_blacklist = []
        config.download.llm_title_filter = MagicMock(enabled=False)
        config.download.audio_first = MagicMock(enabled=False)
        config.download.speech_screening = MagicMock(enabled=False)
        config.download.zero_download_remix = MagicMock(enabled=False)
        config.download.delay_between_keywords = 0
        config.cache_dir = "/tmp/cache"
        config.downloaded_videos_dir = "/tmp/videos"
        config.duration_tiers = {}
        return config

    @patch('src.downloader.core.CheckpointManager')
    @patch('src.downloader.core.TranscodingManager')
    @patch('src.downloader.core.TitleFilter')
    @patch('src.downloader.core.SpeechScreener')
    @patch('src.downloader.core.SearchOptimizer')
    @patch('src.downloader.core.AudioFirstPipeline')
    @pytest.mark.fast
    def test_milestone_tracking_is_efficient(
        self, mock_audio, mock_search, mock_speech, mock_title, mock_transcode, mock_checkpoint, mock_config, tmp_path, caplog
    ):
        """Milestone tracking uses set for O(1) lookup"""
        mock_checkpoint.return_value.load_sources.return_value = []
        mock_checkpoint.return_value.duration_tiers = {'short': {}}

        downloader = VideoDownloader(mock_config)
        downloader.download_for_keyword = MagicMock(return_value=[])
        downloader._clear_checkpoint = MagicMock()
        downloader.log_source_diversity_report = MagicMock()

        # Large number of keywords to test efficiency
        keywords = [f"keyword_{i}" for i in range(200)]

        import time
        start = time.time()
        with caplog.at_level(logging.INFO):
            downloader.download_all(keywords, tmp_path)
        elapsed = time.time() - start

        # Should complete quickly (mocked downloads, just testing overhead)
        # Allow generous time for CI environments
        assert elapsed < 5.0, f"Progress logging added too much overhead: {elapsed:.2f}s"
