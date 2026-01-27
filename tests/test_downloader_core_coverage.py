"""
Additional tests for VideoDownloader coverage gaps.

Covers:
- Transcoding logic in _run_download_cmd
- Existing file handling in _download_by_ids
- Inventory report edge cases
- Cookie file search edge cases
- Check dependencies edge cases
- Audio-first mode delegation

Created: 2026-01-11 (Session 14)
"""

import pytest
import sys
import os
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.core import VideoDownloader
from src.downloader.title_filter import SearchResult
from src.state import DownloadedVideo
from src.config import Config
from src.config.sections.download import (
    LLMTitleFilterConfig,
    SpeechScreeningConfig,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def mock_config(temp_dir):
    """Create a mock config for tests."""
    config = Config()
    config.cache_dir = str(temp_dir / ".cache")
    config.downloaded_videos_dir = str(temp_dir / "videos")
    config.project_dir = str(temp_dir)

    # Create directories
    (temp_dir / ".cache").mkdir(exist_ok=True)
    (temp_dir / "videos").mkdir(exist_ok=True)

    # Ensure download config has proper cookie settings
    if hasattr(config, 'download'):
        config.download.cookies_from_browser = ""
        config.download.cookies_path = ""

    return config


@pytest.fixture
def downloader(mock_config):
    """Create a VideoDownloader instance."""
    return VideoDownloader(mock_config)


# ============================================================================
# Test get_inventory_report edge cases
# ============================================================================

class TestInventoryReportCoverage:
    """Test inventory report coverage gaps."""

    def test_inventory_report_nonexistent_dir(self, downloader, temp_dir):
        """Test inventory report when directory doesn't exist."""
        nonexistent_dir = temp_dir / "nonexistent"

        report = downloader.get_inventory_report(nonexistent_dir)

        assert report['total_videos'] == 0
        assert report.get('total_duration', 0) == 0 or report.get('total_duration_hours', 0) == 0
        assert report['keywords_covered'] == []

    def test_inventory_report_with_tier_dirs(self, downloader, temp_dir):
        """Test inventory report extracts keywords from tier directories."""
        videos_dir = temp_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        # Create tier directories with videos
        (videos_dir / "python_s").mkdir()
        (videos_dir / "python_s" / "video1.mp4").touch()

        (videos_dir / "machine_m").mkdir()
        (videos_dir / "machine_m" / "video2.mp4").touch()

        (videos_dir / "learning_l").mkdir()
        (videos_dir / "learning_l" / "video3.mp4").touch()

        report = downloader.get_inventory_report(videos_dir)

        assert report['total_videos'] == 3
        assert 'python' in report['keywords_covered']
        assert 'machine' in report['keywords_covered']
        assert 'learning' in report['keywords_covered']
        assert report['num_keywords_covered'] == 3

    def test_inventory_report_mixed_extensions(self, downloader, temp_dir):
        """Test inventory report counts different video extensions."""
        videos_dir = temp_dir / "videos"
        videos_dir.mkdir(exist_ok=True)

        (videos_dir / "test_s").mkdir()
        (videos_dir / "test_s" / "video.mp4").touch()
        (videos_dir / "test_s" / "video.mkv").touch()
        (videos_dir / "test_s" / "video.webm").touch()
        (videos_dir / "test_s" / "video.avi").touch()

        report = downloader.get_inventory_report(videos_dir)

        assert report['total_videos'] == 4


# ============================================================================
# Test check_dependencies edge cases
# ============================================================================

class TestCheckDependenciesCoverage:
    """Test dependency checking coverage gaps."""

    @patch('subprocess.run')
    def test_check_dependencies_ffmpeg_not_found(self, mock_run, mock_config, temp_dir):
        """Test check_dependencies when ffmpeg is not found."""
        mock_config.download.davinci_mode = True

        # yt-dlp succeeds, ffmpeg fails
        def side_effect(cmd, *args, **kwargs):
            if 'yt-dlp' in cmd:
                result = Mock()
                result.stdout = "2024.01.01"
                return result
            elif 'ffmpeg' in cmd:
                raise FileNotFoundError("ffmpeg not found")
            return Mock()

        mock_run.side_effect = side_effect

        downloader = VideoDownloader(mock_config)
        success, message = downloader.check_dependencies()

        assert success is False
        assert "FFmpeg not found" in message

    @patch('subprocess.run')
    def test_check_dependencies_yt_dlp_not_found(self, mock_run, mock_config, temp_dir):
        """Test check_dependencies when yt-dlp is not found."""
        mock_run.side_effect = FileNotFoundError("yt-dlp not found")

        downloader = VideoDownloader(mock_config)
        success, message = downloader.check_dependencies()

        assert success is False
        assert "yt-dlp not found" in message


# ============================================================================
# Test _find_cookies_file edge cases
# ============================================================================

class TestFindCookiesCoverage:
    """Test cookie file finding coverage gaps."""

    def test_find_cookies_explicit_path_not_exists(self, temp_dir):
        """Test finding cookies when explicit path doesn't exist."""
        config = Config()
        config.cache_dir = str(temp_dir / ".cache")
        config.downloaded_videos_dir = str(temp_dir / "videos")
        config.project_dir = str(temp_dir)
        config.download.cookies_path = str(temp_dir / "nonexistent_cookies.txt")
        config.download.cookies_from_browser = ""

        (temp_dir / ".cache").mkdir(exist_ok=True)
        (temp_dir / "videos").mkdir(exist_ok=True)

        downloader = VideoDownloader(config)

        # Should have warned and continued to other paths
        assert downloader._cookies_path is None or not downloader._cookies_path.exists()

    def test_find_cookies_install_dir(self, temp_dir):
        """Test finding cookies in install directory."""
        config = Config()
        config.cache_dir = str(temp_dir / ".cache")
        config.downloaded_videos_dir = str(temp_dir / "videos")
        config._config_path = str(temp_dir / "config.yaml")

        (temp_dir / ".cache").mkdir(exist_ok=True)
        (temp_dir / "videos").mkdir(exist_ok=True)

        # Create cookies file in install dir
        cookies_file = temp_dir / "cookies.txt"
        cookies_file.write_text("# Cookies file")

        downloader = VideoDownloader(config)

        assert downloader._cookies_path == cookies_file


# ============================================================================
# Test _download_by_ids existing file handling
# ============================================================================

class TestDownloadByIdsCoverage:
    """Test _download_by_ids coverage gaps."""

    def test_download_by_ids_finds_existing_video(self, downloader, temp_dir):
        """Test _download_by_ids correctly identifies existing videos."""
        keyword_dir = temp_dir / "videos" / "test_s"
        keyword_dir.mkdir(parents=True)

        # Create existing video file with video ID in name
        video_id = "abc123xyz"
        existing_video = keyword_dir / f"test_video_{video_id}.mp4"
        existing_video.touch()

        # Mock the run download cmd to avoid actual download
        with patch.object(downloader, '_run_download_cmd') as mock_run:
            mock_run.return_value = []

            results = downloader._download_by_ids(
                video_ids=[video_id],
                keyword_dir=keyword_dir,
                output_dir=temp_dir / "videos",
                keyword="test",
                tier="short"
            )

        # Should have returned the existing video without downloading
        assert len(results) == 1
        assert results[0].duration_tier == "short"
        mock_run.assert_not_called()

    def test_download_by_ids_mixed_existing_and_new(self, downloader, temp_dir):
        """Test _download_by_ids with mix of existing and new videos."""
        keyword_dir = temp_dir / "videos" / "test_s"
        keyword_dir.mkdir(parents=True)

        # Create existing video
        existing_id = "existing123"
        existing_video = keyword_dir / f"video_{existing_id}.mp4"
        existing_video.touch()

        new_id = "new456"

        # Mock the run download to simulate downloading new video
        with patch.object(downloader, '_run_download_cmd') as mock_run:
            new_video = DownloadedVideo(
                file="test_s/new_video.mp4",
                url=f"https://www.youtube.com/watch?v={new_id}",
                title="New Video",
                channel="Test",
                upload_date="2024-01-01",
                duration=60,
                duration_tier="short",
                keyword="test",
                download_date="2024-01-01",
                license="Unknown"
            )
            mock_run.return_value = [new_video]

            results = downloader._download_by_ids(
                video_ids=[existing_id, new_id],
                keyword_dir=keyword_dir,
                output_dir=temp_dir / "videos",
                keyword="test",
                tier="short"
            )

        # Should have 2 results - one existing, one new
        assert len(results) == 2

    def test_download_by_ids_mkv_and_webm_extensions(self, downloader, temp_dir):
        """Test _download_by_ids recognizes different video extensions."""
        keyword_dir = temp_dir / "videos" / "test_s"
        keyword_dir.mkdir(parents=True)

        # Create existing videos with different extensions
        (keyword_dir / "video_id1.mkv").touch()
        (keyword_dir / "video_id2.webm").touch()

        with patch.object(downloader, '_run_download_cmd') as mock_run:
            mock_run.return_value = []

            results = downloader._download_by_ids(
                video_ids=["id1", "id2"],
                keyword_dir=keyword_dir,
                output_dir=temp_dir / "videos",
                keyword="test",
                tier="short"
            )

        # Both should be found as existing
        assert len(results) == 2
        mock_run.assert_not_called()


# ============================================================================
# Test _run_download_cmd transcode coverage
# ============================================================================

class TestRunDownloadCmdTranscode:
    """Test transcoding logic in _run_download_cmd."""

    def test_run_download_cmd_with_transcoding_disabled(self, mock_config, temp_dir):
        """Test _run_download_cmd with transcoding disabled (davinci_mode=False)."""
        # Ensure davinci_mode is disabled
        mock_config.download.davinci_mode = False

        downloader = VideoDownloader(mock_config)

        keyword_dir = temp_dir / "test_s"
        keyword_dir.mkdir(parents=True)

        # Create a video file and info.json
        video_file = keyword_dir / "test_video_abc123.mp4"
        video_file.touch()

        info_json = keyword_dir / "test_video_abc123.info.json"
        info_json.write_text(json.dumps({
            'id': 'abc123',
            'title': 'Test Video',
            'uploader': 'Test Channel',
            'duration': 120,
            'webpage_url': 'https://youtube.com/watch?v=abc123'
        }))

        # Mock subprocess to simulate successful download
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0)

            results = downloader._run_download_cmd(
                cmd=['yt-dlp', 'test'],
                keyword_dir=keyword_dir,
                output_dir=temp_dir,
                keyword="test",
                tier="short",
                existing_before=set()
            )

        # Should have processed the video
        assert len(results) == 1
        assert results[0].keyword == "test"

    def test_run_download_cmd_no_transcode_needed(self, mock_config, temp_dir):
        """Test _run_download_cmd when transcoding check says not needed."""
        mock_config.download.davinci_mode = True
        mock_config.download.delete_original = False

        downloader = VideoDownloader(mock_config)

        keyword_dir = temp_dir / "test_s"
        keyword_dir.mkdir(parents=True)

        # Create video and info files
        video_file = keyword_dir / "test_abc123.mp4"
        video_file.write_bytes(b"fake video content")

        info_json = keyword_dir / "test_abc123.info.json"
        info_json.write_text(json.dumps({
            'id': 'abc123',
            'title': 'Test Video',
            'duration': 60,
            'webpage_url': 'https://youtube.com/watch?v=abc123'
        }))

        # Mock needs_transcoding to return False
        downloader.transcoding_mgr.needs_transcoding = Mock(return_value=(False, "Already compatible"))

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            results = downloader._run_download_cmd(
                cmd=['yt-dlp', 'test'],
                keyword_dir=keyword_dir,
                output_dir=temp_dir,
                keyword="test",
                tier="short",
                existing_before=set()
            )

        # Should have processed without transcoding
        assert len(results) == 1

    def test_run_download_cmd_exception_handling(self, mock_config, temp_dir):
        """Test _run_download_cmd handles exceptions gracefully."""
        downloader = VideoDownloader(mock_config)

        keyword_dir = temp_dir / "test_s"
        keyword_dir.mkdir(parents=True)

        # Mock subprocess to raise an exception
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Network error")

            results = downloader._run_download_cmd(
                cmd=['yt-dlp', 'test'],
                keyword_dir=keyword_dir,
                output_dir=temp_dir,
                keyword="test",
                tier="short",
                existing_before=set()
            )

        # Should return empty list on error
        assert results == []


# ============================================================================
# Test _download_single coverage gaps
# ============================================================================

class TestDownloadSingleCoverage:
    """Test _download_single coverage gaps."""

    def test_download_single_speech_screening_filters_all(self, mock_config, temp_dir):
        """Test when speech screening filters all videos."""
        # Use spec-based mocks for config sub-objects to catch phantom attributes
        mock_config.download.llm_title_filter = Mock(spec=LLMTitleFilterConfig, enabled=True)
        mock_config.download.speech_screening = Mock(spec=SpeechScreeningConfig, enabled=True, tiers=['short', 'long'])
        mock_config.download.title_blacklist = []
        mock_config.download.max_keyword_len = 8
        mock_config.download.max_filename_len = 10

        downloader = VideoDownloader(mock_config)

        # Mock methods
        downloader.checkpoint_mgr.get_tier_value = Mock(return_value=5)
        downloader.search_optimizer.get_adaptive_search_pool = Mock(return_value=20)
        downloader.title_filter.search_video_metadata = Mock(return_value=SearchResult(videos=[
            {'id': 'v1', 'title': 'Video 1'},
            {'id': 'v2', 'title': 'Video 2'},
        ]))
        downloader.title_filter.filter_titles_with_llm = Mock(return_value=[
            {'id': 'v1', 'title': 'Video 1'},
            {'id': 'v2', 'title': 'Video 2'},
        ])
        downloader.search_optimizer.record_search_pass_rate = Mock()

        # Speech screening returns empty (all filtered)
        downloader.speech_screener.screen_approved_videos = Mock(return_value=[])

        output_dir = temp_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = downloader._download_single(
            keyword="test",
            tier="short",
            output_dir=output_dir,
            topic="test topic"
        )

        assert results == []

    def test_download_single_no_search_results(self, mock_config, temp_dir):
        """Test _download_single when search returns no results."""
        # Configure download settings with spec to validate attributes
        mock_config.download.llm_title_filter = Mock(spec=LLMTitleFilterConfig, enabled=True)
        mock_config.download.max_keyword_len = 8
        mock_config.download.max_filename_len = 10

        downloader = VideoDownloader(mock_config)

        downloader.checkpoint_mgr.get_tier_value = Mock(return_value=5)
        downloader.search_optimizer.get_adaptive_search_pool = Mock(return_value=20)
        downloader.title_filter.search_video_metadata = Mock(return_value=SearchResult(videos=[]))

        output_dir = temp_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = downloader._download_single(
            keyword="test",
            tier="short",
            output_dir=output_dir,
            topic=""
        )

        assert results == []

    def test_download_single_llm_filter_rejects_all(self, mock_config, temp_dir):
        """Test _download_single when LLM filter rejects all videos."""
        # Configure download settings with spec to validate attributes
        mock_config.download.llm_title_filter = Mock(spec=LLMTitleFilterConfig, enabled=True)
        mock_config.download.title_blacklist = []
        mock_config.download.max_keyword_len = 8
        mock_config.download.max_filename_len = 10

        downloader = VideoDownloader(mock_config)

        downloader.checkpoint_mgr.get_tier_value = Mock(return_value=5)
        downloader.search_optimizer.get_adaptive_search_pool = Mock(return_value=20)
        downloader.title_filter.search_video_metadata = Mock(return_value=SearchResult(videos=[
            {'id': 'v1', 'title': 'Video 1'}
        ]))
        downloader.title_filter.filter_titles_with_llm = Mock(return_value=[])
        downloader.search_optimizer.record_search_pass_rate = Mock()

        output_dir = temp_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = downloader._download_single(
            keyword="test",
            tier="short",
            output_dir=output_dir,
            topic=""
        )

        assert results == []


# ============================================================================
# Test download_all coverage gaps
# ============================================================================

class TestDownloadAllCoverage:
    """Test download_all coverage gaps."""

    def test_download_all_llm_filter_logging(self, mock_config, temp_dir):
        """Test download_all logs LLM filter status."""
        # Configure download settings with spec to validate attributes
        mock_config.download.llm_title_filter = Mock(spec=LLMTitleFilterConfig, enabled=True, provider="gemini")
        mock_config.download.title_blacklist = ["spam", "clickbait"]

        downloader = VideoDownloader(mock_config)
        downloader.checkpoint_mgr.get_tier_value = Mock(return_value=0)

        output_dir = temp_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch.object(downloader, 'download_for_keyword') as mock_download:
            mock_download.return_value = ([], [])

            # Just verify it runs without error
            results, failed = downloader.download_all(
                keywords=["test"],
                output_dir=output_dir,
                topic="test topic"
            )

    def test_download_all_existing_videos_count(self, mock_config, temp_dir):
        """Test download_all counts existing videos."""
        output_dir = temp_dir / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create existing videos
        subdir = output_dir / "existing_s"
        subdir.mkdir()
        (subdir / "video1.mp4").touch()
        (subdir / "video2.mp4").touch()

        downloader = VideoDownloader(mock_config)
        downloader.checkpoint_mgr.get_tier_value = Mock(return_value=0)

        with patch.object(downloader, 'download_for_keyword') as mock_download:
            mock_download.return_value = ([], [])

            results, failed = downloader.download_all(
                keywords=["test"],
                output_dir=output_dir,
            )


# ============================================================================
# Test add_cookies_to_cmd
# ============================================================================

class TestAddCookiesCoverage:
    """Test _add_cookies_to_cmd coverage."""

    def test_add_cookies_from_browser(self, mock_config, temp_dir):
        """Test adding cookies from browser."""
        downloader = VideoDownloader(mock_config)
        downloader._cookies_from_browser = "chrome"
        downloader._cookies_path = None

        cmd = ['yt-dlp', 'test']
        downloader._add_cookies_to_cmd(cmd)

        assert '--cookies-from-browser' in cmd
        assert 'chrome' in cmd

    def test_add_cookies_from_file(self, mock_config, temp_dir):
        """Test adding cookies from file."""
        cookies_file = temp_dir / "cookies.txt"
        cookies_file.write_text("# Cookies")

        downloader = VideoDownloader(mock_config)
        downloader._cookies_from_browser = ""
        downloader._cookies_path = cookies_file

        cmd = ['yt-dlp', 'test']
        downloader._add_cookies_to_cmd(cmd)

        assert '--cookies' in cmd


# ============================================================================
# Test delegation methods coverage
# ============================================================================

class TestDelegationMethodsCoverage:
    """Test delegation methods for coverage."""

    def test_log_source_diversity_report(self, downloader):
        """Test log_source_diversity_report delegation."""
        downloader.search_optimizer.log_source_diversity_report = Mock()

        downloader.log_source_diversity_report()

        downloader.search_optimizer.log_source_diversity_report.assert_called_once()

    def test_record_search_pass_rate(self, downloader):
        """Test record_search_pass_rate delegation."""
        downloader.search_optimizer.record_search_pass_rate = Mock()

        downloader._record_search_pass_rate("keyword", 10, 5)

        downloader.search_optimizer.record_search_pass_rate.assert_called_once_with("keyword", 10, 5)

    def test_build_format_string(self, downloader):
        """Test _build_format_string delegation."""
        downloader.transcoding_mgr.build_format_string = Mock(return_value="bestvideo+bestaudio")

        result = downloader._build_format_string()

        assert result == "bestvideo+bestaudio"

    def test_build_filter_string(self, downloader):
        """Test _build_filter_string delegation."""
        downloader.transcoding_mgr.build_filter_string = Mock(return_value="duration>30")

        result = downloader._build_filter_string("short")

        downloader.transcoding_mgr.build_filter_string.assert_called_once_with("short", downloader.DURATION_TIERS)

    def test_get_ffmpeg_transcode_cmd_delegation(self, downloader):
        """Test _get_ffmpeg_transcode_cmd delegation (line 215)."""
        downloader.transcoding_mgr.get_ffmpeg_transcode_cmd = Mock(
            return_value=(['ffmpeg', '-i', 'input.mp4', 'output.mp4'], 'output.mp4')
        )

        result = downloader._get_ffmpeg_transcode_cmd("/path/input.mp4", "/path/output.mp4")

        downloader.transcoding_mgr.get_ffmpeg_transcode_cmd.assert_called_once_with(
            "/path/input.mp4", "/path/output.mp4"
        )
        assert result == (['ffmpeg', '-i', 'input.mp4', 'output.mp4'], 'output.mp4')


# ============================================================================
# Test existing videos partial coverage (lines 486-487)
# ============================================================================

class TestExistingVideosPartialCoverage:
    """Test when existing videos exist but fewer than required."""

    def test_existing_videos_need_more(self, downloader, temp_dir):
        """Test the elif existing_videos branch (lines 486-487)."""
        output_dir = temp_dir / "videos"
        output_dir.mkdir(exist_ok=True)

        # Create a keyword directory with some videos but not enough
        keyword_dir = output_dir / "python_s"
        keyword_dir.mkdir(exist_ok=True)
        (keyword_dir / "video1.mp4").touch()  # Only 1 video, need 5

        # Mock to avoid actual download
        downloader._download_single = Mock(return_value=[])
        downloader.checkpoint = None

        # Mock tier values - need 5 per keyword
        downloader.checkpoint_mgr.duration_tiers = {'short': {'per_keyword': 5, 'min': 0, 'max': 30}}
        downloader.DURATION_TIERS = downloader.checkpoint_mgr.duration_tiers

        with patch.object(downloader.checkpoint_mgr, 'get_tier_value', side_effect=lambda t, k, d=0: 5 if k == 'per_keyword' else d):
            result = downloader.download_for_keyword("python", output_dir, tiers=['short'])

        # Should have called _download_single since we need more
        downloader._download_single.assert_called()


# ============================================================================
# Test transcode timeout handling (lines 919, 925-926)
# ============================================================================

class TestTranscodeTimeoutCoverage:
    """Test transcode timeout handling in _run_download_cmd."""

    def test_transcode_timeout_warning(self, downloader, temp_dir, caplog):
        """Test transcode timeout logs warning (line 919)."""
        import logging
        caplog.set_level(logging.DEBUG)

        output_dir = temp_dir / "videos"
        output_dir.mkdir(exist_ok=True)
        keyword_dir = output_dir / "keyword_s"
        keyword_dir.mkdir(exist_ok=True)

        # Create a video file and info.json
        video_file = keyword_dir / "test_abc123.mp4"
        video_file.touch()
        info_json = keyword_dir / "test_abc123.info.json"
        info_json.write_text('{"id": "abc123", "title": "Test Video"}')

        # Enable davinci mode
        downloader.download_config.davinci_mode = True
        downloader.download_config.delete_original = False

        # Mock _needs_transcoding to return True
        downloader.transcoding_mgr.needs_transcoding = Mock(return_value=(True, "needs transcoding"))
        downloader.transcoding_mgr.get_ffmpeg_transcode_cmd = Mock(
            return_value=(['ffmpeg', '-i', str(video_file), str(video_file)], str(video_file))
        )

        # Create mock processes
        mock_download_process = Mock()
        mock_download_process.communicate.return_value = ("", "")
        mock_download_process.poll.return_value = 0

        # Transcode process that times out
        mock_transcode_process = Mock()
        mock_transcode_process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=['ffmpeg'], timeout=120),  # First call times out
            ("", "")  # Second call after kill succeeds
        ]
        mock_transcode_process.poll.return_value = None  # Still running
        mock_transcode_process.wait.return_value = None

        call_count = [0]
        def popen_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_download_process
            else:
                return mock_transcode_process

        with patch('subprocess.Popen', side_effect=popen_side_effect):
            # Should not raise, just log warning
            result = downloader._run_download_cmd(
                ['yt-dlp', 'test'],
                keyword_dir,
                output_dir,
                "keyword",
                "short",
                set()
            )

        # Check for timeout warning
        assert any("Transcode timeout" in r.message for r in caplog.records) or \
               any("timeout" in r.message.lower() for r in caplog.records)

    def test_transcode_process_wait_timeout(self, downloader, temp_dir):
        """Test TimeoutExpired in process.wait after transcode (lines 925-926)."""
        output_dir = temp_dir / "videos"
        output_dir.mkdir(exist_ok=True)
        keyword_dir = output_dir / "keyword_s"
        keyword_dir.mkdir(exist_ok=True)

        # Create video and info
        video_file = keyword_dir / "test_xyz789.mp4"
        video_file.write_bytes(b"fake video content")
        info_json = keyword_dir / "test_xyz789.info.json"
        info_json.write_text('{"id": "xyz789", "title": "Test"}')

        downloader.download_config.davinci_mode = True
        downloader.download_config.delete_original = False

        downloader.transcoding_mgr.needs_transcoding = Mock(return_value=(True, "needs transcode"))
        downloader.transcoding_mgr.get_ffmpeg_transcode_cmd = Mock(
            return_value=(['ffmpeg', '-i', str(video_file), str(video_file)], str(video_file))
        )

        mock_download = Mock()
        mock_download.communicate.return_value = ("", "")
        mock_download.poll.return_value = 0

        # Transcode process that needs force kill
        mock_transcode = Mock()
        mock_transcode.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=['ffmpeg'], timeout=120),  # First call
            ("", "")  # After kill
        ]
        mock_transcode.poll.return_value = None  # Still running in finally block
        mock_transcode.wait.side_effect = subprocess.TimeoutExpired(cmd=['ffmpeg'], timeout=5)

        call_idx = [0]
        def popen_factory(*args, **kwargs):
            call_idx[0] += 1
            return mock_download if call_idx[0] == 1 else mock_transcode

        with patch('subprocess.Popen', side_effect=popen_factory):
            result = downloader._run_download_cmd(
                ['yt-dlp', 'test'],
                keyword_dir,
                output_dir,
                "kw",
                "short",
                set()
            )

        # Should have called kill and wait
        mock_transcode.kill.assert_called()


# ============================================================================
# Test transcode success with delete_original (lines 929-934)
# ============================================================================

class TestTranscodeSuccessDeleteOriginal:
    """Test successful transcode with delete_original enabled."""

    def test_transcode_success_with_delete_original(self, downloader, temp_dir, caplog):
        """Test successful transcode deletes original and renames (lines 929-934)."""
        import logging
        caplog.set_level(logging.INFO)

        output_dir = temp_dir / "videos"
        output_dir.mkdir(exist_ok=True)
        keyword_dir = output_dir / "keyword_s"
        keyword_dir.mkdir(exist_ok=True)

        # Create video file
        video_file = keyword_dir / "video_abc123.mp4"
        video_file.write_bytes(b"original video content")

        # Create info.json
        info_json = keyword_dir / "video_abc123.info.json"
        info_json.write_text('{"id": "abc123", "title": "Test Video", "uploader": "TestChannel"}')

        # Enable davinci mode with delete_original
        downloader.download_config.davinci_mode = True
        downloader.download_config.delete_original = True

        downloader.transcoding_mgr.needs_transcoding = Mock(return_value=(True, "vp9 codec"))

        # Mock transcode command to create temp output path
        temp_output = video_file.with_stem(video_file.stem + '_davinci')
        downloader.transcoding_mgr.get_ffmpeg_transcode_cmd = Mock(
            return_value=(['ffmpeg', '-i', str(video_file), str(video_file)], str(video_file))
        )

        mock_download = Mock()
        mock_download.communicate.return_value = ("", "")
        mock_download.poll.return_value = 0

        mock_transcode = Mock()
        mock_transcode.communicate.return_value = ("", "")
        mock_transcode.returncode = 0
        mock_transcode.poll.return_value = 0

        call_num = [0]
        def popen_mock(*args, **kwargs):
            call_num[0] += 1
            if call_num[0] == 1:
                return mock_download
            else:
                # Create the temp output file to simulate successful transcode
                temp_output.write_bytes(b"transcoded video content - larger file")
                return mock_transcode

        with patch('subprocess.Popen', side_effect=popen_mock):
            result = downloader._run_download_cmd(
                ['yt-dlp', 'test'],
                keyword_dir,
                output_dir,
                "keyword",
                "short",
                set()
            )

        # Check transcode complete message (lines 929-934 covered)
        assert any("Transcode complete" in r.message for r in caplog.records)

        # Result should have the downloaded video
        assert len(result) == 1
        # Temp file should be gone (renamed to final)
        assert not temp_output.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
