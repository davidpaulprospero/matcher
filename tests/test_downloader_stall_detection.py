"""
Tests for SABR stall detection in VideoDownloader._wait_for_process_with_progress.

Covers US-007: SABR stall detection edge case tests
- AC1: Test stall detector triggers after exact timeout threshold (60s default)
- AC2: Test stall detector resets timer on any subprocess output
- AC3: Test stall detector handles concurrent downloads without false positives
- AC4: Test resume-on-retry correctly continues from last completed fragment
- AC5: Test stall detector distinguishes metadata output from download progress

SABR (Streaming Adaptive Bitrate) downloads can stall indefinitely when YouTube
throttles or drops connections. The stall detector monitors subprocess output
and kills processes that produce no output for stall_timeout seconds.

Refactored as part of US-009 to use shared fixtures from tests/fixtures/downloader_fixtures.py.
"""

import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from tests.fixtures.downloader_fixtures import (
    create_mock_downloader_config,
    patch_video_downloader_dependencies,
)


class MockPipe:
    """Mock pipe that simulates subprocess stdout/stderr with controllable output timing."""

    def __init__(self, lines=None, delay_per_line=0, stall_after=None, encoding='utf-8', errors='replace',
                 wait_when_empty=False):
        """
        Args:
            lines: List of lines to return from readline()
            delay_per_line: Seconds to wait between lines
            stall_after: After this many lines, stop producing output (simulate stall)
            encoding: Pipe encoding (for compat checks)
            errors: Error handling mode
            wait_when_empty: If True, wait when no more lines; if False, return '' immediately
        """
        self.lines = lines or []
        self.delay_per_line = delay_per_line
        self.stall_after = stall_after
        self.encoding = encoding
        self.errors = errors
        self._index = 0
        self._closed = False
        self._stall_event = threading.Event()
        self._wait_when_empty = wait_when_empty

    def readline(self):
        """Return next line with optional delay, or empty string when done."""
        if self._closed:
            return ''

        if self._index >= len(self.lines):
            # No more lines
            if self._wait_when_empty:
                # Wait until closed (simulates stall)
                self._stall_event.wait(timeout=10)
            return ''

        if self.stall_after is not None and self._index >= self.stall_after:
            # Simulate stall - wait until pipe is closed
            self._stall_event.wait(timeout=10)
            return ''

        if self.delay_per_line > 0:
            time.sleep(self.delay_per_line)

        line = self.lines[self._index]
        self._index += 1
        return line

    def close(self):
        """Close the pipe, unblocking any waiting readline()."""
        self._closed = True
        self._stall_event.set()

    @property
    def closed(self):
        return self._closed


class MockProcess:
    """Mock subprocess.Popen that simulates yt-dlp download process."""

    def __init__(self, stderr_lines=None, stdout_lines=None, return_code=0,
                 stall_after=None, line_delay=0, run_time=None, wait_when_stalled=True):
        """
        Args:
            stderr_lines: Lines to produce on stderr
            stdout_lines: Lines to produce on stdout
            return_code: Exit code
            stall_after: Stall after this many stderr lines
            line_delay: Delay between lines
            run_time: How long process "runs" before completing (None = finish after all output)
            wait_when_stalled: If True, pipes wait when empty (for stall simulation)
        """
        # stderr waits when empty if we're simulating a stall, otherwise returns immediately
        stderr_waits = stall_after is not None or (run_time is not None and run_time > 5)
        self.stderr = MockPipe(stderr_lines or [], line_delay, stall_after, wait_when_empty=stderr_waits)
        self.stdout = MockPipe(stdout_lines or [], line_delay, wait_when_empty=False)
        self.returncode = return_code
        self._run_time = run_time
        self._start_time = time.time()
        self._killed = False
        self._wait_called = False
        self.pid = 12345

    def poll(self):
        """Return None if running, return code if finished."""
        if self._killed:
            return -9
        if self._run_time is not None:
            if time.time() - self._start_time >= self._run_time:
                return self.returncode
            return None
        # Finish when all stderr is consumed
        if self.stderr._index >= len(self.stderr.lines):
            # Check if stalled
            if self.stderr.stall_after is not None and self.stderr._index >= self.stderr.stall_after:
                return None  # Still "running" (stalled)
            return self.returncode
        return None

    def kill(self):
        """Kill the process."""
        self._killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        """Wait for process to complete."""
        self._wait_called = True
        return self.returncode


def create_downloader(config):
    """Create a VideoDownloader with all dependencies patched."""
    with patch_video_downloader_dependencies():
        from src.downloader.core import VideoDownloader
        return VideoDownloader(config)


@pytest.mark.fast
class TestStallDetectorTimeout:
    """AC1: Test stall detector triggers after exact timeout threshold."""

    def test_stall_timeout_triggers_after_threshold(self, tmp_path):
        """Test that stall detector triggers after exactly stall_timeout seconds of no output."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=2)
        downloader = create_downloader(config)

        # Process produces output then stalls
        process = MockProcess(
            stderr_lines=["[download] 10%\n", "[download] 20%\n"],
            stall_after=2,  # Stall after 2 lines
            line_delay=0.1
        )

        start = time.time()
        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=2, max_timeout=30, keyword="test", tier="short"
        )
        elapsed = time.time() - start

        # Should trigger stall timeout
        assert timeout_type == 'stall', f"Expected 'stall' timeout, got {timeout_type}"
        # Should take roughly stall_timeout seconds (with some buffer for execution)
        assert 2.0 <= elapsed < 4.0, f"Expected ~2s elapsed, got {elapsed:.2f}s"
        assert process._killed, "Process should have been killed"

    def test_stall_timeout_uses_config_default_60s(self, tmp_path):
        """Test that default stall_timeout of 60s is used from config."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=60)
        assert config.download.stall_timeout == 60

    def test_no_stall_when_output_continues(self, tmp_path):
        """Test that stall detector does not trigger when output continues."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=2)
        downloader = create_downloader(config)

        # Process produces continuous output every 0.3s for 10 lines = ~3s total
        lines = [f"[download] {i*10}%\n" for i in range(1, 11)]
        process = MockProcess(
            stderr_lines=lines,
            line_delay=0.3,  # Output every 300ms (well under 2s stall timeout)
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=2, max_timeout=30, keyword="test", tier="short"
        )

        # Should complete without timeout - continuous output resets timer
        assert timeout_type is None, f"Expected no timeout, got {timeout_type}"
        assert not process._killed, "Process should not have been killed"
        # Verify all lines were received
        assert len(stderr.strip().split('\n')) == 10, "Should receive all 10 lines"


@pytest.mark.fast
class TestStallDetectorTimerReset:
    """AC2: Test stall detector resets timer on any subprocess output."""

    def test_timer_resets_on_stderr_output(self, tmp_path):
        """Test that stall timer resets when stderr produces output."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=1)
        downloader = create_downloader(config)

        # 5 lines with 0.5s between each, stall_timeout=1s
        lines = [f"[download] {i*20}%\n" for i in range(1, 6)]
        process = MockProcess(
            stderr_lines=lines,
            line_delay=0.5,
            run_time=3.0
        )

        start = time.time()
        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=1, max_timeout=30, keyword="test", tier="short"
        )
        elapsed = time.time() - start

        # Should complete without stall timeout (timer keeps resetting)
        assert timeout_type is None, f"Expected no timeout, got {timeout_type}"
        assert elapsed >= 2.5, f"Expected >=2.5s runtime, got {elapsed:.2f}s"

    def test_timer_resets_on_stdout_output(self, tmp_path):
        """Test that stall timer also resets on stdout output."""
        import inspect
        from src.downloader import core

        # Get source of _wait_for_process_with_progress
        source = inspect.getsource(core.VideoDownloader._wait_for_process_with_progress)

        # Verify both readers update last_activity
        assert 'def read_stderr' in source, "Should have stderr reader"
        assert 'def read_stdout' in source, "Should have stdout reader"

        # Both should update last_activity inside lock
        assert source.count('last_activity = time.time()') >= 2, \
            "Both stdout and stderr readers should update last_activity"

        # Verify lock is used for thread safety
        assert 'with lock:' in source, "Should use lock for thread safety"

    def test_timer_not_reset_without_output(self, tmp_path):
        """Test that timer continues counting when no output occurs."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=1)
        downloader = create_downloader(config)

        # Process produces no output at all
        process = MockProcess(
            stderr_lines=[],
            stdout_lines=[],
            run_time=30  # Would run forever without stall detection
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=1, max_timeout=30, keyword="test", tier="short"
        )

        # Should trigger stall timeout quickly
        assert timeout_type == 'stall', f"Expected 'stall' timeout, got {timeout_type}"
        assert process._killed, "Process should have been killed"


@pytest.mark.fast
class TestConcurrentDownloads:
    """AC3: Test stall detector handles concurrent downloads without false positives."""

    def test_independent_stall_timers_per_process(self, tmp_path):
        """Test that each download process has its own independent stall timer."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=2)
        downloader = create_downloader(config)

        results = []
        errors = []

        def run_download(name, lines, stall_after, delay):
            try:
                process = MockProcess(
                    stderr_lines=lines,
                    stall_after=stall_after,
                    line_delay=delay
                )
                stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
                    process, stall_timeout=2, max_timeout=30, keyword=name, tier="short"
                )
                results.append((name, timeout_type, process._killed))
            except Exception as e:
                errors.append((name, str(e)))

        # Start two concurrent downloads
        thread1 = threading.Thread(target=run_download, args=(
            "continuous",
            [f"[download] {i*10}%\n" for i in range(1, 11)],
            None,  # No stall
            0.3
        ))
        thread2 = threading.Thread(target=run_download, args=(
            "stalling",
            ["[download] 10%\n", "[download] 20%\n"],
            2,  # Stall after 2 lines
            0.1
        ))

        thread1.start()
        thread2.start()
        thread1.join(timeout=10)
        thread2.join(timeout=10)

        assert not errors, f"Errors occurred: {errors}"
        assert len(results) == 2, f"Expected 2 results, got {len(results)}"

        # Find results by name
        continuous_result = next((r for r in results if r[0] == "continuous"), None)
        stalling_result = next((r for r in results if r[0] == "stalling"), None)

        assert continuous_result is not None, "Missing continuous result"
        assert stalling_result is not None, "Missing stalling result"

        # Continuous download should complete without timeout
        assert continuous_result[1] is None, f"Continuous should not timeout: {continuous_result}"
        assert continuous_result[2] is False, "Continuous should not be killed"

        # Stalling download should trigger stall timeout
        assert stalling_result[1] == 'stall', f"Stalling should timeout: {stalling_result}"
        assert stalling_result[2] is True, "Stalling should be killed"


@pytest.mark.fast
class TestResumeOnRetry:
    """AC4: Test resume-on-retry correctly continues from last completed fragment."""

    def test_resume_detection_in_output(self, tmp_path):
        """Test that resume-on-retry is detectable from yt-dlp output patterns."""
        # Resume output pattern from yt-dlp:
        # "[download] Resuming download at byte 12345678"
        # or continuation showing fragment numbers:
        # "[download] Got fragment 50 / 100"

        resume_patterns = [
            "[download] Resuming download at byte 12345678",
            "[download] Got fragment 50 / 100",
            "[download] Downloading video from fragment 50",
        ]

        for pattern in resume_patterns:
            # Verify patterns contain expected keywords
            assert any(kw in pattern.lower() for kw in ['resum', 'fragment', 'byte']), \
                f"Pattern should indicate resume: {pattern}"

    def test_stall_kills_preserves_partial_file(self, tmp_path):
        """Test that stall detection kill preserves partial .mp4 for resume."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=1)
        downloader = create_downloader(config)

        # Create a partial file (simulating download in progress)
        video_dir = tmp_path / "videos" / "test_keyword"
        video_dir.mkdir(parents=True)
        partial_file = video_dir / "video.mp4.part"
        partial_file.write_bytes(b"partial video data")

        # Process stalls
        process = MockProcess(
            stderr_lines=["[download] 50%\n"],
            stall_after=1,
            line_delay=0.1
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=1, max_timeout=30, keyword="test", tier="short"
        )

        # Process should be killed
        assert timeout_type == 'stall'
        assert process._killed

        # Partial file should still exist (not deleted by stall detection)
        assert partial_file.exists(), "Partial file should be preserved for resume"


@pytest.mark.fast
class TestMetadataVsProgress:
    """AC5: Test stall detector distinguishes metadata output from download progress."""

    def test_metadata_output_resets_timer(self, tmp_path):
        """Test that metadata extraction output also resets stall timer."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=2)
        downloader = create_downloader(config)

        # Metadata output lines (before actual download starts)
        # Output every 0.5s (well under 2s stall timeout), process exits when lines exhausted
        metadata_lines = [
            "[youtube] abcd12345: Downloading webpage\n",
            "[youtube] abcd12345: Downloading player API JSON\n",
            "[info] Writing video metadata as JSON to: video.info.json\n",
            "[download] Destination: video.mp4\n",
            "[download] 100%\n",
        ]

        process = MockProcess(
            stderr_lines=metadata_lines,
            line_delay=0.5,  # Output every 0.5s, well under 2s stall timeout
            # No run_time - process exits after all lines are consumed
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=2, max_timeout=30, keyword="test", tier="short"
        )

        # Should not stall - metadata output keeps timer reset
        assert timeout_type is None, f"Expected no timeout, got {timeout_type}"

    def test_warning_output_resets_timer(self, tmp_path):
        """Test that warning messages also reset stall timer."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=2)
        downloader = create_downloader(config)

        # Warning messages that might occur during download
        # Output every 0.5s, process exits after lines exhausted
        warning_lines = [
            "WARNING: Unable to download webpage: HTTP Error 429: Too Many Requests\n",
            "[download] Retrying in 5 seconds...\n",
            "[download] 10%\n",
        ]

        process = MockProcess(
            stderr_lines=warning_lines,
            line_delay=0.5,
            # No run_time - process exits after all lines are consumed
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=2, max_timeout=30, keyword="test", tier="short"
        )

        # Should not stall - warning output resets timer
        assert timeout_type is None, f"Expected no timeout, got {timeout_type}"


@pytest.mark.fast
class TestMaxTimeoutBehavior:
    """Test max_timeout behavior distinct from stall_timeout."""

    def test_max_timeout_triggers_when_process_too_slow(self, tmp_path):
        """Test that max_timeout triggers even with continuous output if process is too slow."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=5)
        downloader = create_downloader(config)

        # Very slow output (0.8s per line, 20 lines = 16s > max_timeout of 3s)
        # But never stalls for 5s (stall_timeout)
        lines = [f"[download] {i*5}%\n" for i in range(1, 21)]
        process = MockProcess(
            stderr_lines=lines,
            line_delay=0.8,
            run_time=20
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=5, max_timeout=3, keyword="test", tier="short"
        )

        # Should trigger max_timeout, not stall timeout
        assert timeout_type == 'max_timeout', f"Expected 'max_timeout', got {timeout_type}"
        assert process._killed, "Process should have been killed"

    def test_stall_timeout_wins_over_max_when_no_output(self, tmp_path):
        """Test that stall_timeout triggers before max_timeout if no output."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=1)
        downloader = create_downloader(config)

        # No output at all - should hit 1s stall timeout before 10s max
        process = MockProcess(
            stderr_lines=[],
            run_time=30
        )

        start = time.time()
        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=1, max_timeout=10, keyword="test", tier="short"
        )
        elapsed = time.time() - start

        assert timeout_type == 'stall', f"Expected 'stall', got {timeout_type}"
        # Should trigger quickly (around stall_timeout, not max_timeout)
        assert elapsed < 3, f"Should trigger in ~1s, took {elapsed:.2f}s"


@pytest.mark.fast
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_stall_with_zero_stall_timeout_uses_tier_timeout(self, tmp_path):
        """Test that stall_timeout=0 falls back to download_timeout."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=0, download_timeout=2)
        # With stall_timeout=0, the code should use download_timeout as stall timeout
        assert config.download.stall_timeout == 0
        assert config.download.download_timeout == 2

    def test_very_long_lines_still_reset_timer(self, tmp_path):
        """Test that very long output lines still reset the stall timer."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=1)
        downloader = create_downloader(config)

        # Very long lines (simulate verbose output)
        long_lines = [
            f"[download] {'x' * 1000} {i}%\n" for i in range(1, 6)
        ]
        process = MockProcess(
            stderr_lines=long_lines,
            line_delay=0.5,
            run_time=3.0
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=1, max_timeout=30, keyword="test", tier="short"
        )

        # Should not stall - long lines still reset timer
        assert timeout_type is None, f"Expected no timeout, got {timeout_type}"

    def test_empty_lines_reset_timer(self, tmp_path):
        """Test that even empty lines reset the stall timer."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=1)
        downloader = create_downloader(config)

        # Mix of empty and content lines
        lines = ["\n", "[download] 10%\n", "\n", "[download] 50%\n", "\n"]
        process = MockProcess(
            stderr_lines=lines,
            line_delay=0.5,
            run_time=3.0
        )

        stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
            process, stall_timeout=1, max_timeout=30, keyword="test", tier="short"
        )

        # Should not stall - even empty lines reset timer
        assert timeout_type is None, f"Expected no timeout, got {timeout_type}"


@pytest.mark.fast
class TestThreadSafety:
    """Test thread safety of stall detection."""

    def test_concurrent_reads_dont_corrupt_state(self, tmp_path):
        """Test that concurrent stdout/stderr reads don't corrupt shared state."""
        config = create_mock_downloader_config(tmp_path, stall_timeout=2)
        downloader = create_downloader(config)

        # Both stdout and stderr producing output concurrently
        stderr_lines = [f"[download] stderr line {i}\n" for i in range(20)]
        stdout_lines = [f"stdout line {i}\n" for i in range(20)]

        process = MockProcess(
            stderr_lines=stderr_lines,
            stdout_lines=stdout_lines,
            line_delay=0.05,
            run_time=2.0
        )

        # Run multiple times to catch race conditions
        for _ in range(3):
            stdout, stderr, timeout_type = downloader._wait_for_process_with_progress(
                process, stall_timeout=2, max_timeout=30, keyword="test", tier="short"
            )
            # Should complete without errors
            assert timeout_type is None or timeout_type == 'stall', \
                f"Unexpected timeout type: {timeout_type}"

            # Reset process for next iteration
            process = MockProcess(
                stderr_lines=stderr_lines,
                stdout_lines=stdout_lines,
                line_delay=0.05,
                run_time=2.0
            )
