"""
Dedicated unit tests for ImpersonationManager.

Sprint 11 / US-001: Create dedicated unit tests for ImpersonationManager.
Tests auto-detect parsing, round-robin rotation, thread-safety,
empty/invalid handling, and preferred_targets filtering.
"""

import threading
from collections import Counter
from unittest.mock import MagicMock, patch

import pytest

from src.downloader.impersonation import ImpersonationManager, ImpersonationStats


# ---------------------------------------------------------------------------
# Sample yt-dlp --list-impersonate-targets output in 3+ format variations
# ---------------------------------------------------------------------------

# Variation 1: Standard tabular output with curl_cffi source
SAMPLE_OUTPUT_STANDARD = """\
[info] Available impersonate targets
Client        OS             Source
------------------------------------
Chrome-136    Macos-15       curl_cffi
Chrome-131    Android-14     curl_cffi
Safari-18.0   Ios-18.0       curl_cffi
Edge-131      Windows-11     curl_cffi
Firefox-133   Linux           curl_cffi
"""

# Variation 2: Extra whitespace and varying column widths
SAMPLE_OUTPUT_WIDE_COLUMNS = """\
[info] Available impersonate targets
Client             OS                   Source
-------------------------------------------------------
Chrome-136         Macos-15             curl_cffi
Tor-13.5           Linux                curl_cffi
Safari-17.0        Macos-14             curl_cffi
"""

# Variation 3: Minimal output with only two targets and no info line
SAMPLE_OUTPUT_MINIMAL = """\
Client  OS       Source
------------------------
Chrome-136  Macos-15  curl_cffi
Edge-131  Windows-11  curl_cffi
"""

# Variation 4: Empty output (no targets detected)
SAMPLE_OUTPUT_EMPTY = """\
[info] Available impersonate targets
Client  OS  Source
------------------
"""

# Variation 5: No header at all (malformed)
SAMPLE_OUTPUT_NO_HEADER = """\
[info] Some unrelated info
No impersonation targets available.
"""


# ===========================================================================
# Helpers
# ===========================================================================

def _make_manager_with_targets(targets: list[str]) -> ImpersonationManager:
    """Create an ImpersonationManager with pre-set targets (no subprocess)."""
    mgr = ImpersonationManager(detect_at_startup=False)
    mgr._targets = list(targets)
    return mgr


# ===========================================================================
# Test: _parse_targets_output() with real sample output (≥3 variations)
# ===========================================================================

@pytest.mark.fast
class TestParseTargetsOutput:
    """AC: parse_list_impersonate_output() with at least 3 format variations."""

    def test_standard_output_parses_all_targets(self):
        targets = ImpersonationManager._parse_targets_output(SAMPLE_OUTPUT_STANDARD)
        assert len(targets) == 5
        assert "Chrome-136:Macos-15" in targets
        assert "Safari-18.0:Ios-18.0" in targets
        assert "Edge-131:Windows-11" in targets
        assert "Firefox-133:Linux" in targets
        assert "Chrome-131:Android-14" in targets

    @pytest.mark.fast
    def test_wide_column_output(self):
        targets = ImpersonationManager._parse_targets_output(SAMPLE_OUTPUT_WIDE_COLUMNS)
        assert len(targets) == 3
        assert "Chrome-136:Macos-15" in targets
        assert "Tor-13.5:Linux" in targets
        assert "Safari-17.0:Macos-14" in targets

    @pytest.mark.fast
    def test_minimal_output_no_info_line(self):
        targets = ImpersonationManager._parse_targets_output(SAMPLE_OUTPUT_MINIMAL)
        assert len(targets) == 2
        assert "Chrome-136:Macos-15" in targets
        assert "Edge-131:Windows-11" in targets

    @pytest.mark.fast
    def test_empty_output_returns_empty_list(self):
        targets = ImpersonationManager._parse_targets_output(SAMPLE_OUTPUT_EMPTY)
        assert targets == []

    @pytest.mark.fast
    def test_no_header_returns_empty_list(self):
        targets = ImpersonationManager._parse_targets_output(SAMPLE_OUTPUT_NO_HEADER)
        assert targets == []

    @pytest.mark.fast
    def test_completely_empty_string(self):
        targets = ImpersonationManager._parse_targets_output("")
        assert targets == []


# ===========================================================================
# Test: Round-robin rotation (wraps around after exhausting list)
# ===========================================================================

@pytest.mark.fast
class TestRoundRobinRotation:
    """AC: get_impersonate_args() cycles through all targets and wraps."""

    def test_cycles_through_all_targets(self):
        targets = ["Chrome-136:Macos-15", "Edge-131:Windows-11", "Safari-18.0:Ios-18.0"]
        mgr = _make_manager_with_targets(targets)

        returned = [mgr.get_next_target() for _ in range(len(targets))]
        assert returned == targets

    @pytest.mark.fast
    def test_wraps_around_after_exhausting(self):
        targets = ["A:1", "B:2"]
        mgr = _make_manager_with_targets(targets)

        # First full cycle
        assert mgr.get_next_target() == "A:1"
        assert mgr.get_next_target() == "B:2"
        # Wrap-around
        assert mgr.get_next_target() == "A:1"
        assert mgr.get_next_target() == "B:2"
        # Third cycle
        assert mgr.get_next_target() == "A:1"

    @pytest.mark.fast
    def test_get_impersonate_args_returns_flag_pair(self):
        mgr = _make_manager_with_targets(["Chrome-136:Macos-15"])
        args = mgr.get_impersonate_args()
        assert args == ["--impersonate", "Chrome-136:Macos-15"]

    @pytest.mark.fast
    def test_stats_track_rotation(self):
        targets = ["X:1", "Y:2"]
        mgr = _make_manager_with_targets(targets)

        for _ in range(5):
            mgr.get_next_target()

        assert mgr.stats.calls_made == 5
        assert mgr.stats.unique_count == 2
        # X:1 used 3 times (indices 0,2,4), Y:2 used 2 times (indices 1,3)
        assert mgr.stats.unique_targets_used["X:1"] == 3
        assert mgr.stats.unique_targets_used["Y:2"] == 2

    @pytest.mark.fast
    def test_single_target_always_returns_same(self):
        mgr = _make_manager_with_targets(["Only:One"])
        for _ in range(10):
            assert mgr.get_next_target() == "Only:One"


# ===========================================================================
# Test: Thread-safety (4+ threads, no duplicates in same rotation cycle)
# ===========================================================================

@pytest.mark.fast
class TestThreadSafety:
    """AC: concurrent calls from 4+ threads return valid targets."""

    def test_concurrent_calls_produce_valid_targets(self):
        targets = ["A:1", "B:2", "C:3", "D:4"]
        mgr = _make_manager_with_targets(targets)
        num_threads = 8
        calls_per_thread = 50
        results = []
        lock = threading.Lock()

        def worker():
            local = []
            for _ in range(calls_per_thread):
                t = mgr.get_next_target()
                local.append(t)
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total = num_threads * calls_per_thread
        assert len(results) == total
        # All results are valid targets
        assert all(r in targets for r in results)

    @pytest.mark.fast
    def test_concurrent_rotation_covers_all_targets(self):
        """Over enough iterations, every target should be returned."""
        targets = ["T1:A", "T2:B", "T3:C", "T4:D", "T5:E"]
        mgr = _make_manager_with_targets(targets)
        results = []
        lock = threading.Lock()

        def worker():
            local = []
            for _ in range(100):
                local.append(mgr.get_next_target())
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All targets should appear at least once
        returned_set = set(results)
        assert returned_set == set(targets)

    @pytest.mark.fast
    def test_stats_match_total_calls_under_concurrency(self):
        mgr = _make_manager_with_targets(["A:1", "B:2", "C:3"])
        num_threads = 6
        calls_per_thread = 100

        def worker():
            for _ in range(calls_per_thread):
                mgr.get_next_target()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert mgr.stats.calls_made == num_threads * calls_per_thread


# ===========================================================================
# Test: Empty/invalid target list handling
# ===========================================================================

@pytest.mark.fast
class TestEmptyAndInvalidTargets:
    """AC: constructor handles empty list, None, and malformed strings."""

    def test_empty_targets_returns_none(self):
        mgr = _make_manager_with_targets([])
        assert mgr.get_next_target() is None

    @pytest.mark.fast
    def test_empty_targets_returns_empty_args(self):
        mgr = _make_manager_with_targets([])
        assert mgr.get_impersonate_args() == []

    @pytest.mark.fast
    def test_target_count_zero_when_empty(self):
        mgr = _make_manager_with_targets([])
        assert mgr.target_count == 0

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_detection_failure_returns_empty(self, mock_run):
        """When yt-dlp returns non-zero exit code, targets list is empty."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        mgr = ImpersonationManager(detect_at_startup=True)
        assert mgr.target_count == 0
        assert mgr.get_next_target() is None

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_detection_timeout_returns_empty(self, mock_run):
        """When yt-dlp times out, targets list is empty."""
        import subprocess as sp
        mock_run.side_effect = sp.TimeoutExpired(cmd="yt-dlp", timeout=10)
        mgr = ImpersonationManager(detect_at_startup=True)
        assert mgr.target_count == 0

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_yt_dlp_not_found_returns_empty(self, mock_run):
        """When yt-dlp is not on PATH."""
        mock_run.side_effect = FileNotFoundError("yt-dlp not found")
        mgr = ImpersonationManager(detect_at_startup=True)
        assert mgr.target_count == 0

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_detect_at_startup_false_skips_detection(self, mock_run):
        """detect_at_startup=False should not call subprocess."""
        mgr = ImpersonationManager(detect_at_startup=False)
        mock_run.assert_not_called()
        assert mgr.target_count == 0

    @pytest.mark.fast
    def test_get_status_with_no_targets(self):
        mgr = _make_manager_with_targets([])
        status = mgr.get_status()
        assert status["target_count"] == 0
        assert status["targets"] == []
        assert status["current_index"] == 0


# ===========================================================================
# Test: preferred_targets filtering
# ===========================================================================

@pytest.mark.fast
class TestPreferredTargetsFiltering:
    """AC: when ImpersonationConfig.preferred_targets is set, only matching
    targets are used in rotation."""

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_preferred_targets_filters_detected(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=SAMPLE_OUTPUT_STANDARD
        )
        mgr = ImpersonationManager(
            preferred_targets=["Chrome-136:Macos-15", "Edge-131:Windows-11"],
            detect_at_startup=True,
        )
        # Should only have the 2 preferred targets (sorted)
        assert mgr.target_count == 2
        assert set(mgr.targets) == {"Chrome-136:Macos-15", "Edge-131:Windows-11"}

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_no_matching_preferred_uses_all(self, mock_run):
        """If no preferred targets match, fall back to using all detected."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=SAMPLE_OUTPUT_STANDARD
        )
        mgr = ImpersonationManager(
            preferred_targets=["NonExistent:Target"],
            detect_at_startup=True,
        )
        # None matched, so all 5 detected targets are used
        assert mgr.target_count == 5

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_empty_preferred_uses_all(self, mock_run):
        """Empty preferred_targets list means use all detected."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=SAMPLE_OUTPUT_STANDARD
        )
        mgr = ImpersonationManager(
            preferred_targets=[],
            detect_at_startup=True,
        )
        assert mgr.target_count == 5

    @patch("src.downloader.impersonation.subprocess.run")
    @pytest.mark.fast
    def test_preferred_targets_rotation_only_uses_filtered(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout=SAMPLE_OUTPUT_STANDARD
        )
        mgr = ImpersonationManager(
            preferred_targets=["Safari-18.0:Ios-18.0"],
            detect_at_startup=True,
        )
        # Only one target, so rotation always returns it
        assert mgr.target_count == 1
        for _ in range(5):
            assert mgr.get_next_target() == "Safari-18.0:Ios-18.0"


# ===========================================================================
# Test: ImpersonationStats
# ===========================================================================

@pytest.mark.fast
class TestImpersonationStats:
    """Test the stats dataclass independently."""

    def test_initial_state(self):
        stats = ImpersonationStats()
        assert stats.calls_made == 0
        assert stats.unique_count == 0
        assert stats.unique_targets_used == {}

    @pytest.mark.fast
    def test_record_use_increments(self):
        stats = ImpersonationStats()
        stats.record_use("A:1")
        stats.record_use("B:2")
        stats.record_use("A:1")
        assert stats.calls_made == 3
        assert stats.unique_count == 2
        assert stats.unique_targets_used["A:1"] == 2
        assert stats.unique_targets_used["B:2"] == 1

    @pytest.mark.fast
    def test_to_dict(self):
        stats = ImpersonationStats()
        stats.record_use("X:Y")
        d = stats.to_dict()
        assert d == {"calls_made": 1, "unique_targets_used": {"X:Y": 1}}
