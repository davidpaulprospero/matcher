"""
Tests for US-48-012: Socket timeout and stall detection parity in audio_first pipeline.

Verifies:
- All yt-dlp subprocess command lists contain --socket-timeout 10
- All yt-dlp subprocess command lists contain --retries 10 and --fragment-retries 10
- Subprocess calls use encoding='utf-8' and errors='replace' per CLAUDE.md Rule 27
- Stall detection via _wait_for_process_with_progress is used for segment/fallback downloads
"""

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.audio_first import AudioFirstPipeline
from src.downloader.title_filter import SearchResult
from src.downloader.types import MergedSegment, MatchedSegment


def _sr(videos):
    """Wrap video list in SearchResult."""
    return SearchResult(videos=videos)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config():
    """Create mock config matching audio_first test patterns."""
    config = Mock()

    audio_first = Mock()
    audio_first.enabled = True
    audio_first.buffer_seconds = 30.0
    audio_first.merge_gap_seconds = 15.0
    audio_first.min_segment_duration = 5.0
    audio_first.max_segment_duration = 600.0
    audio_first.fallback_full_video = True
    audio_first.audio_quality = 5

    download = Mock()
    download.audio_first = audio_first
    download.download_timeouts = {'short': 60, 'medium': 120, 'long': 300}
    download.max_keyword_len = 50
    download.ffmpeg_location = ''
    download.llm_title_filter = None
    download.cookies_from_browser = ''
    download.cookies_path = ''
    download.max_retries = 1  # Single attempt for fast tests
    download.retry_delay = 0
    download.stall_timeout = 60
    download.checkpoint_interval = 10

    config.download = download
    return config


@pytest.fixture
def pipeline(mock_config):
    """Create AudioFirstPipeline with mocked dependencies."""
    lock = threading.Lock()
    return AudioFirstPipeline(
        config=mock_config,
        get_tier_value_func=Mock(side_effect=lambda tier, key, default: {
            'max_total': 0,
            'per_keyword': 5,
            'min': 20,
            'max': 600,
        }.get(key, default)),
        search_metadata_func=Mock(return_value=[]),
        filter_titles_func=Mock(side_effect=lambda videos, *args: videos),
        cleanup_partial_func=Mock(),
        tier_download_counts={},
        lock=lock
    )


def _make_merged_segment(video_id='abc123', start=0.0, end=30.0, keyword='test'):
    """Create a MergedSegment for testing."""
    match = MatchedSegment(
        video_id=video_id,
        video_url=f'https://www.youtube.com/watch?v={video_id}',
        start_time=start,
        end_time=end,
        track='V1',
        voiceover_segment_idx=0,
        keyword=keyword
    )
    return MergedSegment(
        video_id=video_id,
        video_url=f'https://www.youtube.com/watch?v={video_id}',
        start_time=start,
        end_time=end,
        original_matches=[match],
        keyword=keyword
    )


# ============================================================================
# Test: --socket-timeout present in all yt-dlp commands
# ============================================================================

class TestSocketTimeoutPresence:
    """Verify --socket-timeout 10 is in all yt-dlp subprocess command lists."""

    @pytest.mark.fast
    def test_audio_download_has_socket_timeout(self, pipeline, temp_dir):
        """Audio download yt-dlp command includes --socket-timeout 10."""
        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 1
            result.stderr = 'error'
            result.stdout = ''
            return result

        pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Test', 'duration': 60, 'webpage_url': 'https://youtube.com/watch?v=vid1'}
        ]))

        with patch('subprocess.run', side_effect=capture_run):
            pipeline.download_audio_for_keyword('test', temp_dir, 'short')

        assert len(captured_cmds) >= 1, "Expected at least one subprocess.run call"
        cmd = captured_cmds[0]
        assert '--socket-timeout' in cmd, f"--socket-timeout missing from audio download cmd: {cmd}"
        idx = cmd.index('--socket-timeout')
        assert cmd[idx + 1] == '10', f"--socket-timeout value should be '10', got '{cmd[idx + 1]}'"

    @pytest.mark.fast
    def test_segment_download_has_socket_timeout(self, pipeline, temp_dir):
        """Segment download yt-dlp command includes --socket-timeout 10."""
        captured_cmds = []

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)  # Process finished immediately
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        segments = [_make_merged_segment()]

        # Disable fallback so we don't get a second call
        pipeline.download_config.audio_first.fallback_full_video = False

        with patch('subprocess.Popen', side_effect=capture_popen):
            pipeline.download_video_segments(segments, temp_dir)

        assert len(captured_cmds) >= 1, "Expected at least one subprocess.Popen call"
        cmd = captured_cmds[0]
        assert '--socket-timeout' in cmd, f"--socket-timeout missing from segment download cmd: {cmd}"
        idx = cmd.index('--socket-timeout')
        assert cmd[idx + 1] == '10'

    @pytest.mark.fast
    def test_fallback_download_has_socket_timeout(self, pipeline, temp_dir):
        """Full video fallback yt-dlp command includes --socket-timeout 10."""
        captured_cmds = []

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        segments = [_make_merged_segment()]

        with patch('subprocess.Popen', side_effect=capture_popen):
            pipeline._download_full_video_fallback(
                video_id='abc123',
                video_url='https://youtube.com/watch?v=abc123',
                video_dir=temp_dir,
                segments=segments,
                keyword='test',
                timeout=120
            )

        assert len(captured_cmds) >= 1, "Expected at least one subprocess.Popen call for fallback"
        cmd = captured_cmds[0]
        assert '--socket-timeout' in cmd, f"--socket-timeout missing from fallback cmd: {cmd}"
        idx = cmd.index('--socket-timeout')
        assert cmd[idx + 1] == '10'


# ============================================================================
# Test: --retries and --fragment-retries present
# ============================================================================

class TestRetriesPresence:
    """Verify --retries 10 and --fragment-retries 10 are in all yt-dlp commands."""

    @pytest.mark.fast
    def test_audio_download_has_retries(self, pipeline, temp_dir):
        """Audio download yt-dlp command includes --retries 10 and --fragment-retries 10."""
        captured_cmds = []

        def capture_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            result = Mock()
            result.returncode = 1
            result.stderr = 'error'
            result.stdout = ''
            return result

        pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Test', 'duration': 60, 'webpage_url': 'https://youtube.com/watch?v=vid1'}
        ]))

        with patch('subprocess.run', side_effect=capture_run):
            pipeline.download_audio_for_keyword('test', temp_dir, 'short')

        assert len(captured_cmds) >= 1
        cmd = captured_cmds[0]

        assert '--retries' in cmd, f"--retries missing from audio cmd"
        idx = cmd.index('--retries')
        assert cmd[idx + 1] == '10'

        assert '--fragment-retries' in cmd, f"--fragment-retries missing from audio cmd"
        idx = cmd.index('--fragment-retries')
        assert cmd[idx + 1] == '10'

    @pytest.mark.fast
    def test_segment_download_has_retries(self, pipeline, temp_dir):
        """Segment download yt-dlp command includes --retries 10 and --fragment-retries 10."""
        captured_cmds = []

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        segments = [_make_merged_segment()]
        pipeline.download_config.audio_first.fallback_full_video = False

        with patch('subprocess.Popen', side_effect=capture_popen):
            pipeline.download_video_segments(segments, temp_dir)

        assert len(captured_cmds) >= 1
        cmd = captured_cmds[0]

        assert '--retries' in cmd
        assert cmd[cmd.index('--retries') + 1] == '10'
        assert '--fragment-retries' in cmd
        assert cmd[cmd.index('--fragment-retries') + 1] == '10'

    @pytest.mark.fast
    def test_fallback_download_has_retries(self, pipeline, temp_dir):
        """Full video fallback includes --retries 10 and --fragment-retries 10."""
        captured_cmds = []

        def capture_popen(cmd, **kwargs):
            captured_cmds.append(cmd)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        segments = [_make_merged_segment()]

        with patch('subprocess.Popen', side_effect=capture_popen):
            pipeline._download_full_video_fallback(
                video_id='abc123',
                video_url='https://youtube.com/watch?v=abc123',
                video_dir=temp_dir,
                segments=segments,
                keyword='test',
                timeout=120
            )

        assert len(captured_cmds) >= 1
        cmd = captured_cmds[0]
        assert '--retries' in cmd
        assert cmd[cmd.index('--retries') + 1] == '10'
        assert '--fragment-retries' in cmd
        assert cmd[cmd.index('--fragment-retries') + 1] == '10'


# ============================================================================
# Test: encoding='utf-8' and errors='replace' (CLAUDE.md Rule 27)
# ============================================================================

class TestSubprocessEncoding:
    """Verify all subprocess calls include encoding='utf-8' and errors='replace'."""

    @pytest.mark.fast
    def test_audio_download_uses_utf8_encoding(self, pipeline, temp_dir):
        """Audio download subprocess.run uses encoding='utf-8', errors='replace'."""
        captured_kwargs = []

        def capture_run(cmd, **kwargs):
            captured_kwargs.append(kwargs)
            result = Mock()
            result.returncode = 1
            result.stderr = 'error'
            result.stdout = ''
            return result

        pipeline._search_video_metadata = Mock(return_value=_sr([
            {'id': 'vid1', 'title': 'Test', 'duration': 60, 'webpage_url': 'https://youtube.com/watch?v=vid1'}
        ]))

        with patch('subprocess.run', side_effect=capture_run):
            pipeline.download_audio_for_keyword('test', temp_dir, 'short')

        assert len(captured_kwargs) >= 1
        kw = captured_kwargs[0]
        assert kw.get('encoding') == 'utf-8', f"encoding should be 'utf-8', got {kw.get('encoding')}"
        assert kw.get('errors') == 'replace', f"errors should be 'replace', got {kw.get('errors')}"

    @pytest.mark.fast
    def test_segment_download_uses_utf8_encoding(self, pipeline, temp_dir):
        """Segment download subprocess.Popen uses encoding='utf-8', errors='replace'."""
        captured_kwargs = []

        def capture_popen(cmd, **kwargs):
            captured_kwargs.append(kwargs)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        segments = [_make_merged_segment()]
        pipeline.download_config.audio_first.fallback_full_video = False

        with patch('subprocess.Popen', side_effect=capture_popen):
            pipeline.download_video_segments(segments, temp_dir)

        assert len(captured_kwargs) >= 1
        kw = captured_kwargs[0]
        assert kw.get('encoding') == 'utf-8', f"encoding should be 'utf-8', got {kw.get('encoding')}"
        assert kw.get('errors') == 'replace', f"errors should be 'replace', got {kw.get('errors')}"

    @pytest.mark.fast
    def test_fallback_download_uses_utf8_encoding(self, pipeline, temp_dir):
        """Full video fallback subprocess.Popen uses encoding='utf-8', errors='replace'."""
        captured_kwargs = []

        def capture_popen(cmd, **kwargs):
            captured_kwargs.append(kwargs)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        segments = [_make_merged_segment()]

        with patch('subprocess.Popen', side_effect=capture_popen):
            pipeline._download_full_video_fallback(
                video_id='abc123',
                video_url='https://youtube.com/watch?v=abc123',
                video_dir=temp_dir,
                segments=segments,
                keyword='test',
                timeout=120
            )

        assert len(captured_kwargs) >= 1
        kw = captured_kwargs[0]
        assert kw.get('encoding') == 'utf-8'
        assert kw.get('errors') == 'replace'


# ============================================================================
# Test: Stall detection via _wait_for_process_with_progress
# ============================================================================

class TestStallDetection:
    """Verify stall detection is used for segment and fallback downloads."""

    @pytest.mark.fast
    def test_segment_download_uses_popen_not_run(self, pipeline, temp_dir):
        """Segment download uses subprocess.Popen (stall detection), not subprocess.run."""
        popen_calls = []
        run_calls = []

        def capture_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        def capture_run(cmd, **kwargs):
            run_calls.append(cmd)
            result = Mock()
            result.returncode = 1
            result.stderr = ''
            result.stdout = ''
            return result

        segments = [_make_merged_segment()]
        pipeline.download_config.audio_first.fallback_full_video = False

        with patch('subprocess.Popen', side_effect=capture_popen), \
             patch('subprocess.run', side_effect=capture_run):
            pipeline.download_video_segments(segments, temp_dir)

        assert len(popen_calls) >= 1, "Segment download should use subprocess.Popen"
        # subprocess.run should NOT be called for segment downloads
        yt_dlp_run_calls = [c for c in run_calls if c[0] == 'yt-dlp']
        assert len(yt_dlp_run_calls) == 0, "Segment download should NOT use subprocess.run for yt-dlp"

    @pytest.mark.fast
    def test_fallback_download_uses_popen_not_run(self, pipeline, temp_dir):
        """Full video fallback uses subprocess.Popen (stall detection), not subprocess.run."""
        popen_calls = []

        def capture_popen(cmd, **kwargs):
            popen_calls.append(cmd)
            proc = Mock()
            proc.pid = 12345
            proc.returncode = 1
            proc.poll = Mock(return_value=0)
            proc.wait = Mock()
            proc.stdout = Mock()
            proc.stdout.readline = Mock(return_value='')
            proc.stdout.closed = False
            proc.stdout.close = Mock()
            proc.stderr = Mock()
            proc.stderr.readline = Mock(return_value='')
            proc.stderr.closed = False
            proc.stderr.close = Mock()
            return proc

        segments = [_make_merged_segment()]

        with patch('subprocess.Popen', side_effect=capture_popen):
            pipeline._download_full_video_fallback(
                video_id='abc123',
                video_url='https://youtube.com/watch?v=abc123',
                video_dir=temp_dir,
                segments=segments,
                keyword='test',
                timeout=120
            )

        assert len(popen_calls) >= 1, "Fallback download should use subprocess.Popen"

    @pytest.mark.fast
    def test_wait_for_process_stall_detection_kills_stalled(self, pipeline):
        """_wait_for_process_with_progress kills process after stall timeout."""
        proc = Mock()
        proc.pid = 99999
        proc.poll = Mock(return_value=None)  # Process keeps running
        proc.kill = Mock()
        proc.wait = Mock()

        # stdout/stderr that produce nothing (stall)
        proc.stdout = Mock()
        proc.stdout.readline = Mock(return_value='')
        proc.stdout.closed = False
        proc.stdout.close = Mock()
        proc.stderr = Mock()
        proc.stderr.readline = Mock(return_value='')
        proc.stderr.closed = False
        proc.stderr.close = Mock()

        # Very short stall timeout for fast test
        stdout, stderr, timeout_type = pipeline._wait_for_process_with_progress(
            proc, stall_timeout=1, max_timeout=10, video_id='test_vid'
        )

        assert timeout_type == 'stall', f"Expected 'stall' timeout, got {timeout_type}"
        proc.kill.assert_called_once()

    @pytest.mark.fast
    def test_wait_for_process_normal_completion(self, pipeline):
        """_wait_for_process_with_progress returns None timeout_type on normal exit."""
        proc = Mock()
        proc.pid = 99999
        proc.returncode = 0
        # Process finishes immediately
        proc.poll = Mock(return_value=0)
        proc.wait = Mock()

        proc.stdout = Mock()
        proc.stdout.readline = Mock(return_value='')
        proc.stdout.closed = False
        proc.stdout.close = Mock()
        proc.stderr = Mock()
        proc.stderr.readline = Mock(return_value='')
        proc.stderr.closed = False
        proc.stderr.close = Mock()

        stdout, stderr, timeout_type = pipeline._wait_for_process_with_progress(
            proc, stall_timeout=60, max_timeout=120, video_id='test_vid'
        )

        assert timeout_type is None, f"Expected no timeout, got {timeout_type}"
        proc.kill.assert_not_called()
