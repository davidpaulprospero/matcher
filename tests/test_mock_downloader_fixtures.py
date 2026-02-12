"""
Quick verification tests for mock downloader fixtures (US-86-006).
"""

from tests.conftest import MockYTDL, MockDownloader


def test_mock_youtube_dl_video_info(mock_youtube_dl):
    """Test MockYTDL returns configured video info."""
    info = mock_youtube_dl.extract_info("dQw4w9WgXcQ", download=False)
    assert info is not None
    assert info["title"] == "Never Gonna Give You Up"
    assert mock_youtube_dl.get_call_log()


def test_mock_youtube_dl_search(mock_youtube_dl):
    """Test MockYTDL returns search results."""
    results = mock_youtube_dl.extract_info("ytsearch2:python tutorial", download=False)
    assert results is not None
    assert "entries" in results
    assert len(results["entries"]) >= 1


def test_mock_downloader_search(mock_downloader):
    """Test MockDownloader search."""
    results = mock_downloader.search_videos("python", max_results=5)
    assert len(results) >= 1


def test_mock_downloader_download(mock_downloader):
    """Test MockDownloader download."""
    result = mock_downloader.download_video("dQw4w9WgXcQ", "videos/test.mp4")
    assert result is not None
    assert len(mock_downloader.get_downloads()) == 1


def test_tmp_video_dir(tmp_video_dir):
    """Test tmp_video_dir fixture creates proper structure."""
    videos_dir = tmp_video_dir / "videos"
    assert videos_dir.exists()
