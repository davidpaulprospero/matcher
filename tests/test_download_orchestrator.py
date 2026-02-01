"""
Tests for DownloadOrchestrator - batch coordination logic.

Tests cover:
- Batch processing with multiple keywords
- Retry queue coordination
- Progress tracking and checkpointing
- Result aggregation
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

from src.downloader.orchestrator import DownloadOrchestrator
from src.downloader.checkpoint import DownloadCheckpoint
from src.state import DownloadedVideo


class TestDownloadOrchestratorInit:
    """Tests for DownloadOrchestrator initialization."""

    def test_init_with_downloader(self):
        """Orchestrator initializes with a VideoDownloader instance."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)
        assert orchestrator.downloader is mock_downloader


class TestCountExistingVideos:
    """Tests for _count_existing_videos method."""

    def test_count_empty_directory(self, tmp_path):
        """Returns 0 for empty directory."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        count = orchestrator._count_existing_videos(tmp_path)
        assert count == 0

    def test_count_videos_in_subdirs(self, tmp_path):
        """Counts videos in subdirectories."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        # Create subdirs with videos
        subdir1 = tmp_path / "keyword1_s"
        subdir1.mkdir()
        (subdir1 / "video1.mp4").touch()
        (subdir1 / "video2.mp4").touch()

        subdir2 = tmp_path / "keyword2_m"
        subdir2.mkdir()
        (subdir2 / "video3.mkv").touch()
        (subdir2 / "video4.webm").touch()
        (subdir2 / "info.json").touch()  # Not a video

        count = orchestrator._count_existing_videos(tmp_path)
        assert count == 4

    def test_count_nonexistent_directory(self, tmp_path):
        """Returns 0 for non-existent directory."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        nonexistent = tmp_path / "nonexistent"
        count = orchestrator._count_existing_videos(nonexistent)
        assert count == 0


class TestSetupCheckpoint:
    """Tests for _setup_checkpoint method."""

    def test_fresh_checkpoint_created(self):
        """Creates fresh checkpoint when not resuming."""
        mock_downloader = MagicMock()
        mock_downloader.checkpoint = None
        mock_downloader._load_checkpoint.return_value = None

        orchestrator = DownloadOrchestrator(mock_downloader)
        keywords = ["keyword1", "keyword2"]

        result = orchestrator._setup_checkpoint(keywords, resume=False)

        assert result == keywords
        assert mock_downloader.checkpoint is not None
        assert mock_downloader.checkpoint.completed_keywords == []

    def test_resume_filters_completed_keywords(self):
        """Resume filters out completed keywords."""
        mock_downloader = MagicMock()

        # Create a mock checkpoint with completed keywords
        checkpoint = DownloadCheckpoint(
            completed_keywords=["keyword1"],
            completed_videos=[],
            failed_keywords=[],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            speed_tracker_state=None,
            last_rate_limit_timestamp=None,
            rate_limit_event_count=0
        )
        mock_downloader._load_checkpoint.return_value = checkpoint
        mock_downloader.checkpoint = None
        mock_downloader.speed_tracker = MagicMock()
        mock_downloader._check_rate_limit_cooldown.return_value = False
        mock_downloader._share_budget_across_keywords = False
        mock_downloader.vpn_manager = None
        mock_downloader.escalation_manager = None
        mock_downloader._per_tier_isolation = False

        orchestrator = DownloadOrchestrator(mock_downloader)
        keywords = ["keyword1", "keyword2", "keyword3"]

        result = orchestrator._setup_checkpoint(keywords, resume=True)

        # keyword1 should be filtered out
        assert result == ["keyword2", "keyword3"]
        assert mock_downloader.checkpoint is checkpoint


class TestLogMilestone:
    """Tests for _log_milestone method."""

    def test_logs_at_10_percent_intervals(self):
        """Logs at 10% milestone intervals."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        logged = set()

        # At 10% (10 of 100)
        orchestrator._log_milestone(11, 100, logged)
        assert 10 in logged

        # At 20% (20 of 100)
        orchestrator._log_milestone(21, 100, logged)
        assert 20 in logged

    def test_does_not_log_zero_percent(self):
        """Does not log at 0% milestone."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        logged = set()
        orchestrator._log_milestone(1, 100, logged)
        assert 0 not in logged

    def test_does_not_relog_same_milestone(self):
        """Does not log the same milestone twice."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        logged = {10}  # Already logged 10%

        # Should not add 10 again
        orchestrator._log_milestone(15, 100, logged)
        assert logged == {10}


class TestProcessRetryQueue:
    """Tests for _process_retry_queue method."""

    def test_empty_queue_returns_empty(self):
        """Returns empty lists when queue is empty."""
        mock_downloader = MagicMock()
        mock_downloader.retry_queue.is_enabled = True
        mock_downloader.retry_queue.has_pending.return_value = False

        orchestrator = DownloadOrchestrator(mock_downloader)
        recovered, failed = orchestrator._process_retry_queue(Path("/tmp"), "topic")

        assert recovered == []
        assert failed == []

    def test_disabled_queue_returns_empty(self):
        """Returns empty lists when queue is disabled."""
        mock_downloader = MagicMock()
        mock_downloader.retry_queue.is_enabled = False

        orchestrator = DownloadOrchestrator(mock_downloader)
        recovered, failed = orchestrator._process_retry_queue(Path("/tmp"), "topic")

        assert recovered == []
        assert failed == []

    def test_processes_pending_items(self):
        """Processes pending items in retry queue."""
        mock_downloader = MagicMock()
        mock_downloader.retry_queue.is_enabled = True
        mock_downloader._share_budget_across_keywords = False

        # Mock pending items
        mock_item = MagicMock()
        mock_item.keyword = "test_keyword"
        mock_item.tier = "medium"
        mock_item.video_id = "test_keyword|medium"

        # First call returns True (has pending), second returns True (still processing),
        # third returns False (done processing)
        mock_downloader.retry_queue.has_pending.side_effect = [True, True, False]
        mock_downloader.retry_queue.start_retry_pass.return_value = 1
        mock_downloader.retry_queue.get_pending_items.return_value = [mock_item]
        mock_downloader.retry_queue.get_stats.return_value = {'failed': 0, 'max_passes': 3}

        # Mock successful download
        downloaded_video = DownloadedVideo(
            file="test.mp4",
            url="https://youtube.com/watch?v=123",
            title="Test Video",
            channel="Test Channel",
            upload_date="2024-01-01",
            duration=60,
            duration_tier="medium",
            keyword="test_keyword",
            download_date="2024-01-15",
            license="Unknown"
        )
        mock_downloader._download_single.return_value = [downloaded_video]
        mock_downloader._lock = MagicMock()
        mock_downloader.tier_download_counts = {}
        mock_downloader.sources = []

        orchestrator = DownloadOrchestrator(mock_downloader)
        recovered, failed = orchestrator._process_retry_queue(Path("/tmp"), "topic")

        assert len(recovered) == 1
        assert recovered[0] == downloaded_video
        mock_downloader.retry_queue.mark_success.assert_called_with("test_keyword|medium")


class TestCoordinateRetries:
    """Tests for coordinate_retries method."""

    def test_retries_failed_items(self):
        """Retries a list of failed items."""
        mock_downloader = MagicMock()
        mock_downloader._lock = MagicMock()
        mock_downloader.tier_download_counts = {}
        mock_downloader.sources = []

        downloaded_video = DownloadedVideo(
            file="test.mp4",
            url="https://youtube.com/watch?v=123",
            title="Test Video",
            channel="Test Channel",
            upload_date="2024-01-01",
            duration=60,
            duration_tier="medium",
            keyword="keyword1",
            download_date="2024-01-15",
            license="Unknown"
        )
        mock_downloader._download_single.return_value = [downloaded_video]

        orchestrator = DownloadOrchestrator(mock_downloader)
        failed_items = [
            {'keyword': 'keyword1', 'tier': 'medium', 'video_id': 'keyword1|medium'}
        ]

        recovered, still_failed = orchestrator.coordinate_retries(
            failed_items, Path("/tmp"), "topic"
        )

        assert len(recovered) == 1
        assert len(still_failed) == 0

    def test_tracks_still_failed(self):
        """Tracks items that still fail after retry."""
        mock_downloader = MagicMock()
        mock_downloader._lock = MagicMock()
        mock_downloader.tier_download_counts = {}
        mock_downloader.sources = []
        mock_downloader._download_single.return_value = []  # Download fails

        orchestrator = DownloadOrchestrator(mock_downloader)
        failed_items = [
            {'keyword': 'keyword1', 'tier': 'medium', 'video_id': 'keyword1|medium'}
        ]

        recovered, still_failed = orchestrator.coordinate_retries(
            failed_items, Path("/tmp"), "topic"
        )

        assert len(recovered) == 0
        assert len(still_failed) == 1
        assert still_failed[0]['keyword'] == 'keyword1'


class TestAggregateResults:
    """Tests for aggregate_results method."""

    def test_aggregates_multiple_batches(self):
        """Aggregates results from multiple download batches."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        video1 = DownloadedVideo(
            file="v1.mp4", url="url1", title="V1", channel="C1",
            upload_date="2024-01-01", duration=60, duration_tier="short",
            keyword="k1", download_date="2024-01-15", license="Unknown"
        )
        video2 = DownloadedVideo(
            file="v2.mp4", url="url2", title="V2", channel="C2",
            upload_date="2024-01-01", duration=120, duration_tier="medium",
            keyword="k2", download_date="2024-01-15", license="Unknown"
        )

        results = [
            ([video1], ["failed1"]),
            ([video2], ["failed2", "failed3"]),
        ]

        all_downloaded, all_failed = orchestrator.aggregate_results(results)

        assert len(all_downloaded) == 2
        assert len(all_failed) == 3
        assert video1 in all_downloaded
        assert video2 in all_downloaded

    def test_aggregates_empty_results(self):
        """Handles empty result batches."""
        mock_downloader = MagicMock()
        orchestrator = DownloadOrchestrator(mock_downloader)

        results = [
            ([], []),
            ([], ["failed1"]),
        ]

        all_downloaded, all_failed = orchestrator.aggregate_results(results)

        assert len(all_downloaded) == 0
        assert len(all_failed) == 1


class TestGetBatchStats:
    """Tests for get_batch_stats method."""

    def test_returns_batch_statistics(self):
        """Returns comprehensive batch statistics."""
        mock_downloader = MagicMock()
        mock_downloader.sources = [MagicMock(), MagicMock()]
        mock_downloader.tier_download_counts = {'short': 5, 'medium': 10}
        mock_downloader.retry_queue.get_stats.return_value = {'pending': 0, 'failed': 2}

        checkpoint = DownloadCheckpoint(
            completed_keywords=["k1", "k2"],
            completed_videos=["v1", "v2", "v3"],
            failed_keywords=["f1"],
            current_keyword=None,
            current_tier=None,
            timestamp=datetime.now().isoformat(),
            speed_tracker_state=None,
            last_rate_limit_timestamp=None,
            rate_limit_event_count=0
        )
        mock_downloader.checkpoint = checkpoint

        orchestrator = DownloadOrchestrator(mock_downloader)
        stats = orchestrator.get_batch_stats()

        assert stats['total_sources'] == 2
        assert stats['tier_counts'] == {'short': 5, 'medium': 10}
        assert stats['checkpoint']['completed_keywords'] == 2
        assert stats['checkpoint']['failed_keywords'] == 1
        assert stats['checkpoint']['completed_videos'] == 3


class TestDownloadAllDelegation:
    """Tests for download_all delegation to orchestrator."""

    @patch('src.downloader.orchestrator.DownloadOrchestrator._count_existing_videos')
    def test_download_all_integrates_with_downloader(self, mock_count, tmp_path):
        """Tests full download_all flow with mocked downloader."""
        mock_count.return_value = 0

        mock_downloader = MagicMock()
        mock_downloader.checkpoint = None
        mock_downloader._load_checkpoint.return_value = None
        mock_downloader.download_config.parallel_workers = 2
        mock_downloader.download_config.delay_between_keywords = 0
        mock_downloader.download_config.title_blacklist = []
        mock_downloader.download_config.llm_title_filter = None

        # Mock download_for_keyword to return videos
        video = DownloadedVideo(
            file="test.mp4", url="url", title="Test", channel="Ch",
            upload_date="2024-01-01", duration=60, duration_tier="medium",
            keyword="keyword1", download_date="2024-01-15", license="Unknown"
        )
        mock_downloader.download_for_keyword.return_value = [video]

        mock_downloader.retry_queue.is_enabled = False
        mock_downloader.title_filter = None

        orchestrator = DownloadOrchestrator(mock_downloader)
        downloaded, failed = orchestrator.download_all(
            keywords=["keyword1"],
            output_dir=tmp_path,
            resume=False,
            topic="test"
        )

        assert len(downloaded) == 1
        assert len(failed) == 0
        mock_downloader.download_for_keyword.assert_called_once()
        mock_downloader._clear_checkpoint.assert_called_once()


class TestOrchestratorImport:
    """Tests for orchestrator import and export."""

    def test_can_import_from_downloader_package(self):
        """DownloadOrchestrator can be imported from downloader package."""
        from src.downloader import DownloadOrchestrator
        assert DownloadOrchestrator is not None

    def test_in_all_exports(self):
        """DownloadOrchestrator is in __all__."""
        from src.downloader import __all__
        assert 'DownloadOrchestrator' in __all__
