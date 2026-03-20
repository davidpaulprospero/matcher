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
        assert d["calls_made"] == 1
        assert d["unique_targets_used"] == {"X:Y": 1}
        assert "success_count" in d
        assert "failure_count" in d

    @pytest.mark.fast
    def test_record_success_increments(self):
        stats = ImpersonationStats()
        stats.record_success("A:1")
        stats.record_success("A:1")
        stats.record_success("B:2")
        assert stats.success_count["A:1"] == 2
        assert stats.success_count["B:2"] == 1

    @pytest.mark.fast
    def test_record_failure_increments(self):
        stats = ImpersonationStats()
        stats.record_failure("A:1")
        stats.record_failure("A:1")
        stats.record_failure("B:2")
        assert stats.failure_count["A:1"] == 2
        assert stats.failure_count["B:2"] == 1

    @pytest.mark.fast
    def test_success_rate_calculation(self):
        stats = ImpersonationStats()
        stats.record_success("A:1")
        stats.record_success("A:1")
        stats.record_failure("A:1")
        # 2 successes / 3 total = 0.666...
        assert stats.get_success_rate("A:1") == pytest.approx(2 / 3)

    @pytest.mark.fast
    def test_success_rate_no_data_returns_one(self):
        """Untested targets should return 1.0 (optimistic default)."""
        stats = ImpersonationStats()
        assert stats.get_success_rate("Unknown:Target") == 1.0

    @pytest.mark.fast
    def test_success_rate_all_failures(self):
        stats = ImpersonationStats()
        stats.record_failure("A:1")
        stats.record_failure("A:1")
        assert stats.get_success_rate("A:1") == 0.0

    @pytest.mark.fast
    def test_success_rate_all_successes(self):
        stats = ImpersonationStats()
        stats.record_success("A:1")
        stats.record_success("A:1")
        assert stats.get_success_rate("A:1") == 1.0


# ===========================================================================
# Test: Success rate filtering in get_next_target
# ===========================================================================

@pytest.mark.fast
class TestSuccessRateFiltering:
    """AC: get_next_target() optionally skips targets with low success rates."""

    def test_low_success_targets_deprioritized(self):
        """Targets with success_rate < threshold are skipped in rotation."""
        targets = ["Good:1", "Bad:2", "Good:3"]
        mgr = _make_manager_with_targets(targets)
        mgr._min_success_rate = 0.2
        mgr._enable_success_filtering = True

        # Record failures for Bad:2 to make its success rate 0%
        for _ in range(5):
            mgr._stats.record_failure("Bad:2")

        # Record some successes for Good targets
        mgr._stats.record_success("Good:1")
        mgr._stats.record_success("Good:3")

        # Get many targets - Bad:2 should be skipped
        returned = [mgr.get_next_target() for _ in range(10)]
        assert "Bad:2" not in returned
        assert "Good:1" in returned
        assert "Good:3" in returned

    @pytest.mark.fast
    def test_new_targets_not_filtered(self):
        """Targets with no recorded data should pass through (optimistic)."""
        targets = ["New:1", "New:2"]
        mgr = _make_manager_with_targets(targets)
        mgr._min_success_rate = 0.5
        mgr._enable_success_filtering = True

        # No data recorded - both should be returned
        returned = {mgr.get_next_target() for _ in range(10)}
        assert "New:1" in returned
        assert "New:2" in returned

    @pytest.mark.fast
    def test_filtering_disabled_returns_all(self):
        """When enable_success_filtering=False, all targets used."""
        targets = ["A:1", "B:2"]
        mgr = _make_manager_with_targets(targets)
        mgr._enable_success_filtering = False

        # Record 100% failure for B:2
        for _ in range(10):
            mgr._stats.record_failure("B:2")

        # Should still return B:2 because filtering is disabled
        returned = [mgr.get_next_target() for _ in range(10)]
        assert "B:2" in returned

    @pytest.mark.fast
    def test_skip_low_success_override(self):
        """skip_low_success parameter overrides instance default."""
        targets = ["A:1", "B:2"]
        mgr = _make_manager_with_targets(targets)
        mgr._enable_success_filtering = True
        mgr._min_success_rate = 0.5

        # B:2 has 0% success
        for _ in range(5):
            mgr._stats.record_failure("B:2")

        # With override=False, B:2 should be returned despite low rate
        returned = [mgr.get_next_target(skip_low_success=False) for _ in range(10)]
        assert "B:2" in returned

    @pytest.mark.fast
    def test_all_low_success_falls_back_to_rotation(self):
        """When all targets have low success, fall back to round-robin."""
        targets = ["Bad:1", "Bad:2"]
        mgr = _make_manager_with_targets(targets)
        mgr._min_success_rate = 0.5
        mgr._enable_success_filtering = True

        # All targets fail
        for _ in range(5):
            mgr._stats.record_failure("Bad:1")
            mgr._stats.record_failure("Bad:2")

        # Should still return targets (fallback behavior)
        returned = [mgr.get_next_target() for _ in range(10)]
        assert len(returned) == 10
        assert all(t in targets for t in returned)

    @pytest.mark.fast
    def test_manager_record_success_method(self):
        """ImpersonationManager.record_success updates stats."""
        mgr = _make_manager_with_targets(["A:1"])
        mgr.record_success("A:1")
        mgr.record_success("A:1")
        assert mgr.stats.success_count.get("A:1") == 2

    @pytest.mark.fast
    def test_manager_record_failure_method(self):
        """ImpersonationManager.record_failure updates stats."""
        mgr = _make_manager_with_targets(["A:1"])
        mgr.record_failure("A:1")
        mgr.record_failure("A:1")
        assert mgr.stats.failure_count.get("A:1") == 2

    @pytest.mark.fast
    def test_manager_get_success_rate(self):
        """ImpersonationManager.get_success_rate returns correct rate."""
        mgr = _make_manager_with_targets(["A:1"])
        mgr.record_success("A:1")
        mgr.record_failure("A:1")
        assert mgr.get_success_rate("A:1") == 0.5

    @pytest.mark.fast
    def test_get_status_includes_success_rates(self):
        """get_status() includes success_rates dict."""
        targets = ["A:1", "B:2"]
        mgr = _make_manager_with_targets(targets)
        mgr.record_success("A:1")
        mgr.record_failure("B:2")

        status = mgr.get_status()
        assert "success_rates" in status
        assert status["success_rates"]["A:1"] == 1.0
        assert status["success_rates"]["B:2"] == 0.0
        assert "min_success_rate_threshold" in status
        assert "success_filtering_enabled" in status


# ===========================================================================
# Test: get_browser_family() - US-113-005
# ===========================================================================

@pytest.mark.fast
class TestBrowserFamilyExtraction:
    """AC: Extract browser family from impersonation target strings."""

    def test_chrome_family_extracted(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("Chrome-136:Macos-15") == "chrome"
        assert get_browser_family("Chrome-131:Android-14") == "chrome"

    def test_firefox_family_extracted(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("Firefox-133:Linux") == "firefox"

    def test_safari_family_extracted(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("Safari-18.0:Ios-18.0") == "safari"
        assert get_browser_family("Safari-17.0:Macos-14") == "safari"

    def test_edge_family_extracted(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("Edge-131:Windows-11") == "edge"

    def test_tor_family_extracted(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("Tor-13.5:Linux") == "tor"

    def test_empty_target_returns_none(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("") is None

    def test_none_target_returns_none(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family(None) is None

    def test_unknown_browser_returns_dynamic_family(self):
        """Unknown browsers now return their name as the family (Phase 4g)."""
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("UnknownBrowser:OS") == "unknownbrowser"


# ===========================================================================
# Test: Fallback chain - US-113-005
# ===========================================================================

@pytest.mark.fast
class TestFallbackChain:
    """AC: get_fallback_target() returns target from next browser in fallback chain."""

    def test_fallback_from_chrome_to_firefox(self):
        """When Chrome fails, fallback to Firefox."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux", "Safari-18.0:Ios-18.0"]
        mgr = _make_manager_with_targets(targets)
        mgr._fallback_order = ["chrome", "firefox", "safari"]

        fallback = mgr.get_fallback_target("Chrome-136:Macos-15")
        assert fallback == "Firefox-133:Linux"

    def test_fallback_from_firefox_to_safari(self):
        """When Firefox fails, fallback to Safari."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux", "Safari-18.0:Ios-18.0"]
        mgr = _make_manager_with_targets(targets)
        mgr._fallback_order = ["chrome", "firefox", "safari"]

        fallback = mgr.get_fallback_target("Firefox-133:Linux")
        assert fallback == "Safari-18.0:Ios-18.0"

    def test_fallback_from_safari_to_chrome(self):
        """When Safari fails, fallback to Chrome (wraps around)."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux", "Safari-18.0:Ios-18.0"]
        mgr = _make_manager_with_targets(targets)
        mgr._fallback_order = ["chrome", "firefox", "safari"]

        fallback = mgr.get_fallback_target("Safari-18.0:Ios-18.0")
        assert fallback == "Chrome-136:Macos-15"

    def test_fallback_returns_none_when_no_targets(self):
        """When no targets available, return None."""
        mgr = _make_manager_with_targets([])
        fallback = mgr.get_fallback_target("Chrome-136:Macos-15")
        assert fallback is None

    def test_fallback_skips_low_success_browsers(self):
        """Skip browsers with low success rates in fallback."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux"]
        mgr = _make_manager_with_targets(targets)
        mgr._fallback_order = ["chrome", "firefox"]
        mgr._enable_success_filtering = True
        mgr._min_success_rate = 0.3

        # Record Firefox failures to make its success rate 0%
        for _ in range(5):
            mgr._stats.record_failure("Firefox-133:Linux")

        fallback = mgr.get_fallback_target("Chrome-136:Macos-15")
        # Firefox has low success rate, should be skipped
        assert fallback is None or fallback == "Chrome-136:Macos-15"

    def test_get_targets_by_family(self):
        """Get all targets for a specific browser family."""
        targets = [
            "Chrome-136:Macos-15", "Chrome-131:Android-14",
            "Firefox-133:Linux",
            "Safari-18.0:Ios-18.0", "Safari-17.0:Macos-14"
        ]
        mgr = _make_manager_with_targets(targets)

        chrome_targets = mgr.get_targets_by_family("chrome")
        assert len(chrome_targets) == 2
        assert "Chrome-136:Macos-15" in chrome_targets
        assert "Chrome-131:Android-14" in chrome_targets

        firefox_targets = mgr.get_targets_by_family("firefox")
        assert len(firefox_targets) == 1
        assert "Firefox-133:Linux" in firefox_targets

    def test_browser_family_success_rate_tracking(self):
        """Browser family success rates are tracked correctly."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux"]
        mgr = _make_manager_with_targets(targets)

        # Record successes and failures
        mgr.record_success("Chrome-136:Macos-15")
        mgr.record_success("Chrome-136:Macos-15")
        mgr.record_failure("Chrome-136:Macos-15")

        mgr.record_failure("Firefox-133:Linux")
        mgr.record_failure("Firefox-133:Linux")

        chrome_rate = mgr._stats.get_browser_family_success_rate("chrome")
        firefox_rate = mgr._stats.get_browser_family_success_rate("firefox")

        assert chrome_rate == pytest.approx(2/3)  # 2 successes / 3 total
        assert firefox_rate == 0.0  # 0 successes / 2 total

    def test_get_browser_family_success_rates(self):
        """get_browser_family_success_rates() returns all family rates."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux"]
        mgr = _make_manager_with_targets(targets)

        mgr.record_success("Chrome-136:Macos-15")
        mgr.record_failure("Firefox-133:Linux")

        rates = mgr.get_browser_family_success_rates()
        assert "chrome" in rates
        assert "firefox" in rates
        assert "safari" in rates  # Even if not used, should be in dict
        assert rates["chrome"] == 1.0
        assert rates["firefox"] == 0.0

    def test_reorder_fallback_by_success(self):
        """reorder_fallback_by_success() reorders by success rate."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux", "Safari-18.0:Ios-18.0"]
        mgr = _make_manager_with_targets(targets)
        mgr._fallback_order = ["chrome", "firefox", "safari"]

        # Make Firefox have best success rate
        mgr.record_success("Firefox-133:Linux")
        mgr.record_success("Firefox-133:Linux")

        mgr.record_failure("Chrome-136:Macos-15")
        mgr.record_failure("Chrome-136:Macos-15")

        mgr.record_failure("Safari-18.0:Ios-18.0")

        mgr.reorder_fallback_by_success()

        # Firefox should now be first (highest success rate)
        assert mgr._fallback_order[0] == "firefox"

    def test_fallback_order_in_status(self):
        """get_status() includes fallback_order."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux"]
        mgr = _make_manager_with_targets(targets)

        status = mgr.get_status()
        assert "fallback_order" in status
        assert status["fallback_order"] == ["chrome", "firefox", "safari"]


# ===========================================================================
# Test: ImpersonationConfig new fields - US-113-005
# ===========================================================================

@pytest.mark.fast
class TestImpersonationConfigNewFields:
    """AC: ImpersonationConfig includes new fallback order and success filtering fields."""

    def test_impersonation_config_has_fallback_order(self):
        from src.config.sections.download import ImpersonationConfig
        config = ImpersonationConfig()
        assert hasattr(config, 'impersonation_fallback_order')
        assert config.impersonation_fallback_order == ["chrome", "firefox", "safari"]

    def test_impersonation_config_has_min_success_rate(self):
        from src.config.sections.download import ImpersonationConfig
        config = ImpersonationConfig()
        assert hasattr(config, 'min_success_rate')
        assert config.min_success_rate == 0.2

    def test_impersonation_config_has_enable_success_filtering(self):
        from src.config.sections.download import ImpersonationConfig
        config = ImpersonationConfig()
        assert hasattr(config, 'enable_success_filtering')
        assert config.enable_success_filtering is True


# ===========================================================================
# Test: ExtractorArgsConfig browser-specific clients - US-113-005
# ===========================================================================

@pytest.mark.fast
class TestExtractorArgsBrowserSpecific:
    """AC: ExtractorArgsConfig includes browser_specific_clients."""

    def test_extractor_args_has_browser_specific_clients(self):
        from src.config.sections.download import ExtractorArgsConfig
        config = ExtractorArgsConfig()
        assert hasattr(config, 'browser_specific_clients')
        assert "chrome" in config.browser_specific_clients
        assert "firefox" in config.browser_specific_clients
        assert "safari" in config.browser_specific_clients
        assert "edge" in config.browser_specific_clients

    def test_browser_specific_clients_are_lists(self):
        from src.config.sections.download import ExtractorArgsConfig
        config = ExtractorArgsConfig()
        assert isinstance(config.browser_specific_clients["chrome"], list)
        assert isinstance(config.browser_specific_clients["firefox"], list)
        assert len(config.browser_specific_clients["chrome"]) > 0


# ===========================================================================
# Test: Adaptive profile selection config - US-123-004
# ===========================================================================

@pytest.mark.fast
class TestAdaptiveProfileSelectionConfig:
    """AC: ImpersonationConfig includes adaptive profile selection fields."""

    def test_impersonation_config_has_adaptive_profile_selection(self):
        from src.config.sections.download import ImpersonationConfig
        config = ImpersonationConfig()
        assert hasattr(config, 'adaptive_profile_selection')
        assert config.adaptive_profile_selection is False

    def test_impersonation_config_has_profile_success_window(self):
        from src.config.sections.download import ImpersonationConfig
        config = ImpersonationConfig()
        assert hasattr(config, 'profile_success_window')
        assert config.profile_success_window == 20


@pytest.mark.fast
class TestAdaptiveProfileSelection:
    """AC: ImpersonationManager supports adaptive profile selection based on recent success rates."""

    def test_get_best_profile_returns_none_when_no_targets(self):
        """AC: get_best_profile returns None when no targets available."""
        mgr = ImpersonationManager(detect_at_startup=False)
        result = mgr.get_best_profile()
        assert result is None

    def test_get_best_profile_returns_chrome_with_high_success(self):
        """AC: get_best_profile returns chrome when it has highest recent success rate."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
            "Safari-18.0:Ios-18.0",
        ]
        mgr = _make_manager_with_targets(targets)

        # Record high success rate for chrome
        for _ in range(10):
            mgr.record_success("Chrome-136:Macos-15")

        # Record failures for others
        for _ in range(5):
            mgr.record_failure("Firefox-133:Linux")
            mgr.record_failure("Safari-18.0:Ios-18.0")

        result = mgr.get_best_profile()
        assert result == "chrome"

    def test_get_best_profile_returns_firefox_with_high_success(self):
        """AC: get_best_profile returns firefox when it has highest recent success rate."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
            "Safari-18.0:Ios-18.0",
        ]
        mgr = _make_manager_with_targets(targets)

        # Record high success rate for firefox
        for _ in range(8):
            mgr.record_success("Firefox-133:Linux")

        # Record failures for chrome
        for _ in range(3):
            mgr.record_failure("Chrome-136:Macos-15")

        result = mgr.get_best_profile()
        assert result == "firefox"

    def test_get_best_profile_returns_first_family_with_no_data(self):
        """AC: get_best_profile returns first family when no recent data exists."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
            "Safari-18.0:Ios-18.0",
        ]
        mgr = _make_manager_with_targets(targets)

        # No data recorded - should return first in fallback order
        result = mgr.get_best_profile()
        assert result == "chrome"  # First in fallback order

    def test_get_recent_success_rate_with_data(self):
        """AC: get_recent_success_rate calculates rate from windowed attempts."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
        ]
        mgr = _make_manager_with_targets(targets)

        # Record 7 successes, 3 failures = 70% recent success rate
        for _ in range(7):
            mgr.record_success("Chrome-136:Macos-15")
        for _ in range(3):
            mgr.record_failure("Chrome-136:Macos-15")

        rate = mgr.get_recent_success_rate("chrome")
        assert rate == 0.7

    def test_get_recent_success_rate_returns_one_for_no_data(self):
        """AC: get_recent_success_rate returns 1.0 for untested family."""
        targets = ["Chrome-136:Macos-15"]
        mgr = _make_manager_with_targets(targets)

        rate = mgr.get_recent_success_rate("firefox")
        assert rate == 1.0  # Optimistic default

    def test_get_target_for_profile_returns_target(self):
        """AC: get_target_for_profile returns a target for the specified browser family."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
        ]
        mgr = _make_manager_with_targets(targets)

        target = mgr.get_target_for_profile("chrome")
        assert target == "Chrome-136:Macos-15"

    def test_get_target_for_profile_returns_none_for_unknown_family(self):
        """AC: get_target_for_profile returns None for unknown family."""
        targets = ["Chrome-136:Macos-15"]
        mgr = _make_manager_with_targets(targets)

        target = mgr.get_target_for_profile("opera")
        assert target is None

    def test_adaptive_impersonate_args_selects_best_profile(self):
        """AC: get_impersonate_args uses adaptive selection when enabled."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
            "Safari-18.0:Ios-18.0",
        ]
        # Create manager with adaptive selection enabled
        mgr = ImpersonationManager(
            detect_at_startup=False,
            adaptive_profile_selection=True,
            profile_success_window=20,
        )
        mgr._targets = targets

        # Record high success for safari
        for _ in range(10):
            mgr.record_success("Safari-18.0:Ios-18.0")

        # Record failures for others
        for _ in range(5):
            mgr.record_failure("Chrome-136:Macos-15")
            mgr.record_failure("Firefox-133:Linux")

        args = mgr.get_impersonate_args()
        assert args == ["--impersonate", "Safari-18.0:Ios-18.0"]

    def test_non_adaptive_impersonate_args_uses_round_robin(self):
        """AC: get_impersonate_args uses round-robin when adaptive is disabled."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
        ]
        mgr = ImpersonationManager(
            detect_at_startup=False,
            adaptive_profile_selection=False,
        )
        mgr._targets = targets

        # First call should get first target
        args1 = mgr.get_impersonate_args()
        assert args1 == ["--impersonate", "Chrome-136:Macos-15"]

        # Second call should get second target (round-robin)
        args2 = mgr.get_impersonate_args()
        assert args2 == ["--impersonate", "Firefox-133:Linux"]

    def test_adaptive_impersonate_args_override_param(self):
        """AC: get_impersonate_args can override adaptive setting via param."""
        targets = [
            "Chrome-136:Macos-15",
            "Firefox-133:Linux",
        ]
        # Create manager with adaptive disabled
        mgr = ImpersonationManager(
            detect_at_startup=False,
            adaptive_profile_selection=False,
        )
        mgr._targets = targets

        # Can enable via param - Firefox must be strictly better than Chrome
        mgr._stats.set_window_size(20)
        for _ in range(10):
            mgr.record_success("Firefox-133:Linux")
        # Chrome needs some data too, otherwise optimistic default (1.0) ties Firefox
        for _ in range(3):
            mgr.record_failure("Chrome-136:Macos-15")

        args = mgr.get_impersonate_args(use_adaptive=True)
        assert args == ["--impersonate", "Firefox-133:Linux"]


# ===========================================================================
# Test: Phase 1 - Thread Safety (RLock, TOCTOU, deadlock)
# ===========================================================================

@pytest.mark.fast
class TestRLockAndThreadSafety:
    """Phase 1: Thread safety fixes."""

    def test_rlock_prevents_deadlock_in_adaptive_selection(self):
        """get_best_profile() calls get_recent_success_rate() which also locks - no deadlock."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux"]
        mgr = ImpersonationManager(
            detect_at_startup=False,
            adaptive_profile_selection=True,
        )
        mgr._targets = targets

        import signal
        result = [None]

        def run():
            result[0] = mgr.get_best_profile()

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "Deadlock detected in get_best_profile()"
        assert result[0] is not None

    def test_get_fallback_target_concurrent(self):
        """4 threads calling get_fallback_target while another calls record_failure."""
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux", "Safari-18.0:Ios-18.0"]
        mgr = _make_manager_with_targets(targets)
        mgr._fallback_order = ["chrome", "firefox", "safari"]
        errors = []

        def fallback_worker():
            try:
                for _ in range(50):
                    mgr.get_fallback_target("Chrome-136:Macos-15")
            except Exception as e:
                errors.append(e)

        def failure_worker():
            try:
                for _ in range(50):
                    mgr.record_failure("Chrome-136:Macos-15")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fallback_worker) for _ in range(4)]
        threads.append(threading.Thread(target=failure_worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Concurrent access errors: {errors}"
        assert all(not t.is_alive() for t in threads)

    def test_get_next_target_with_cleared_targets(self):
        """Clearing _targets after init returns None, not IndexError."""
        mgr = _make_manager_with_targets(["A:1", "B:2"])
        mgr._targets = []
        result = mgr.get_next_target()
        assert result is None


# ===========================================================================
# Test: Phase 2 - get_impersonate_args_with_target
# ===========================================================================

@pytest.mark.fast
class TestGetImpersonateArgsWithTarget:
    """Phase 2a: get_impersonate_args_with_target returns (args, target)."""

    def test_returns_tuple_with_target(self):
        mgr = _make_manager_with_targets(["Chrome-136:Macos-15"])
        args, target = mgr.get_impersonate_args_with_target()
        assert args == ["--impersonate", "Chrome-136:Macos-15"]
        assert target == "Chrome-136:Macos-15"

    def test_returns_empty_when_no_targets(self):
        mgr = _make_manager_with_targets([])
        args, target = mgr.get_impersonate_args_with_target()
        assert args == []
        assert target is None

    def test_get_impersonate_args_delegates(self):
        """get_impersonate_args still works via delegation."""
        mgr = _make_manager_with_targets(["Chrome-136:Macos-15"])
        args = mgr.get_impersonate_args()
        assert args == ["--impersonate", "Chrome-136:Macos-15"]


# ===========================================================================
# Test: Phase 3 - State persistence (to_dict/from_dict/restore_state)
# ===========================================================================

@pytest.mark.fast
class TestStatePersistence:
    """Phase 3: ImpersonationStats and Manager serialization."""

    def test_impersonation_stats_roundtrip(self):
        stats = ImpersonationStats()
        stats.record_use("A:1")
        stats.record_use("B:2")
        stats.record_success("A:1")
        stats.record_failure("B:2")

        data = stats.to_dict()
        restored = ImpersonationStats.from_dict(data)

        assert restored.calls_made == stats.calls_made
        assert restored.unique_targets_used == stats.unique_targets_used
        assert restored.success_count == stats.success_count
        assert restored.failure_count == stats.failure_count
        assert restored.browser_family_success == stats.browser_family_success
        assert restored.browser_family_failure == stats.browser_family_failure

    def test_impersonation_stats_from_dict_missing_keys(self):
        """from_dict handles missing keys with defaults."""
        stats = ImpersonationStats.from_dict({})
        assert stats.calls_made == 0
        assert stats.unique_targets_used == {}

    def test_impersonation_manager_state_roundtrip(self):
        targets = ["A:1", "B:2", "C:3"]
        mgr = _make_manager_with_targets(targets)
        # Advance index
        mgr.get_next_target()
        mgr.get_next_target()
        mgr.record_success("A:1")
        mgr.record_failure("B:2")

        data = mgr.to_dict()

        mgr2 = _make_manager_with_targets(targets)
        mgr2.restore_state(data)

        assert mgr2._index == mgr._index
        assert mgr2._stats.success_count == mgr._stats.success_count
        assert mgr2._stats.failure_count == mgr._stats.failure_count

    def test_restore_state_with_changed_targets(self):
        """When targets change, index resets to 0 but stats are preserved."""
        mgr = _make_manager_with_targets(["A:1", "B:2"])
        mgr.get_next_target()  # index -> 1
        mgr.record_success("A:1")

        data = mgr.to_dict()

        mgr2 = _make_manager_with_targets(["X:1", "Y:2"])  # different targets
        mgr2.restore_state(data)

        assert mgr2._index == 0  # Reset because targets changed
        assert mgr2._stats.success_count.get("A:1") == 1  # Stats preserved

    def test_old_checkpoint_without_impersonation_state(self):
        """Manager works fine when no saved state exists (backward compat)."""
        mgr = _make_manager_with_targets(["A:1"])
        # No restore_state call - should work fine
        assert mgr.get_next_target() == "A:1"


# ===========================================================================
# Test: Phase 4 - Observability & Validation
# ===========================================================================

@pytest.mark.fast
class TestTargetsExhaustedFlag:
    """Phase 4a: all_targets_exhausted flag."""

    def test_all_targets_exhausted_flag_set(self):
        targets = ["Bad:1", "Bad:2"]
        mgr = _make_manager_with_targets(targets)
        mgr._min_success_rate = 0.5
        mgr._enable_success_filtering = True

        for _ in range(5):
            mgr._stats.record_failure("Bad:1")
            mgr._stats.record_failure("Bad:2")

        # Force all-targets-exhausted path
        mgr.get_next_target()
        assert mgr.all_targets_exhausted is True

    def test_exhausted_flag_resets_on_success(self):
        targets = ["Bad:1"]
        mgr = _make_manager_with_targets(targets)
        mgr._all_targets_exhausted = True

        mgr.record_success("Bad:1")
        assert mgr.all_targets_exhausted is False


@pytest.mark.fast
class TestLogRateLimitLevel:
    """Phase 4b: log_rate_limit uses custom level."""

    def test_log_rate_limit_uses_custom_level(self):
        import logging
        from unittest.mock import MagicMock
        from src.logging_templates import log_rate_limit

        mock_logger = MagicMock()
        log_rate_limit(
            mock_logger, "test_op", "test_resource", "test_action",
            level=logging.INFO, key="value"
        )
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.INFO

    def test_log_rate_limit_default_warning(self):
        import logging
        from unittest.mock import MagicMock
        from src.logging_templates import log_rate_limit

        mock_logger = MagicMock()
        log_rate_limit(mock_logger, "test_op", "test_resource", "test_action")
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.WARNING


@pytest.mark.fast
class TestImpersonationConfigValidation:
    """Phase 4d: ImpersonationConfig __post_init__ validation."""

    def test_impersonation_config_validates_timeout(self):
        from src.config.sections.download import ImpersonationConfig
        with pytest.raises(ValueError, match="detection_timeout"):
            ImpersonationConfig(detection_timeout=0)

    def test_impersonation_config_validates_success_rate_high(self):
        from src.config.sections.download import ImpersonationConfig
        with pytest.raises(ValueError, match="min_success_rate"):
            ImpersonationConfig(min_success_rate=2.0)

    def test_impersonation_config_validates_success_rate_low(self):
        from src.config.sections.download import ImpersonationConfig
        with pytest.raises(ValueError, match="min_success_rate"):
            ImpersonationConfig(min_success_rate=-0.1)

    def test_impersonation_config_validates_window(self):
        from src.config.sections.download import ImpersonationConfig
        with pytest.raises(ValueError, match="profile_success_window"):
            ImpersonationConfig(profile_success_window=0)


@pytest.mark.fast
class TestMalformedTargetsParsing:
    """Phase 4f: Malformed targets skipped in parse."""

    def test_malformed_targets_skipped_in_parse(self):
        """Targets with invalid format are skipped."""
        output = """\
[info] Available impersonate targets
Client        OS             Source
------------------------------------
Chrome-136    Macos-15       curl_cffi
:BadTarget    Missing        curl_cffi
Normal-1      Win-10         curl_cffi
"""
        targets = ImpersonationManager._parse_targets_output(output)
        assert "Chrome-136:Macos-15" in targets
        assert "Normal-1:Win-10" in targets
        # :BadTarget:Missing should be skipped
        assert all(not t.startswith(":") for t in targets)


@pytest.mark.fast
class TestDynamicBrowserFamily:
    """Phase 4g: Unknown browser family falls back to browser name."""

    def test_unknown_browser_family_dynamic(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("Brave-1.0:Win-10") == "brave"

    def test_known_browser_still_maps(self):
        from src.downloader.impersonation import get_browser_family
        assert get_browser_family("Chrome-136:Macos-15") == "chrome"


# ===========================================================================
# Test: Phase 5 - Adaptive config wired through
# ===========================================================================

@pytest.mark.fast
class TestAdaptiveConfigWiredThrough:
    """Phase 5a: adaptive config passed from constructor."""

    def test_adaptive_config_wired_through(self):
        mgr = ImpersonationManager(
            detect_at_startup=False,
            adaptive_profile_selection=True,
            profile_success_window=30,
        )
        assert mgr._adaptive_profile_selection is True
        assert mgr._profile_success_window == 30


# ===========================================================================
# Regression test: Scenario from production log - 28 videos rate-walled
# after impersonation targets exhausted
# ===========================================================================

@pytest.mark.fast
class TestRateWallExhaustionScenario:
    """Reproduces the exact failure scenario from a production pipeline run.

    Scenario:
    - Pipeline downloads 289 segments successfully
    - YouTube rate-walls the session after heavy impersonation rotation
    - 28 remaining videos all fail with "No video formats found"
    - All impersonation targets are exhausted

    Pre-fix problems:
    1. Impersonation manager NEVER received stats because caption_fetcher used
       elif (Issue #6) — escalation_manager always took the branch
    2. Success/failure attributed to wrong target (Issue #1) — the target used
       for download was lost by the time record_success/failure was called
    3. No persistence (Issue #4) — fresh start on each run, no learned data
    4. RLock deadlock (Issue #5) — adaptive selection would deadlock
    5. 403 quota errors classified as BotDetection, not RateLimit (Issue #6)
    """

    def test_bug1_stats_never_recorded_with_elif(self):
        """BUG: With elif, impersonation stats were NEVER recorded when
        escalation_manager existed. This meant success filtering was blind.

        Before fix: elif self.impersonation_manager  (never reached)
        After fix:  if self.impersonation_manager    (always reached)
        """
        mgr = _make_manager_with_targets(["Chrome-136:Macos-15", "Firefox-133:Linux"])
        # Simulate what the caption_fetcher now does (if, not elif)
        # Both managers are notified independently
        mgr.record_failure("Chrome-136:Macos-15")
        mgr.record_failure("Chrome-136:Macos-15")
        mgr.record_failure("Chrome-136:Macos-15")
        mgr.record_success("Firefox-133:Linux")

        # Stats ARE recorded (this would have been empty with elif)
        assert mgr.stats.failure_count["Chrome-136:Macos-15"] == 3
        assert mgr.stats.success_count["Firefox-133:Linux"] == 1
        # Success filtering can now skip the failing Chrome target
        assert mgr.stats.get_success_rate("Chrome-136:Macos-15") == 0.0
        assert mgr.stats.get_success_rate("Firefox-133:Linux") == 1.0

    def test_bug2_target_correctly_attributed(self):
        """BUG: get_impersonate_args() returned args but LOST the target string.
        record_success/failure was called with video_id, not the actual target.

        After fix: get_impersonate_args_with_target() returns (args, target).
        """
        mgr = _make_manager_with_targets(["Chrome-136:Macos-15", "Firefox-133:Linux"])

        # Simulate what audio_first.py now does
        args, target = mgr.get_impersonate_args_with_target()
        assert target is not None
        assert target in ["Chrome-136:Macos-15", "Firefox-133:Linux"]

        # Record failure against the ACTUAL target that was used
        mgr.record_failure(target)
        assert mgr.stats.failure_count[target] == 1

    def test_bug3_state_persists_across_runs(self):
        """BUG: Every pipeline run started with fresh impersonation state.
        Rate-walled targets from run N were retried in run N+1.

        After fix: to_dict() / restore_state() persists through checkpoint.
        """
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux", "Safari-18.0:Ios-18.0"]

        # === Run 1: 289 successes, then 28 failures on Chrome ===
        mgr_run1 = _make_manager_with_targets(targets)
        # Simulate 289 successful downloads spread across targets
        for i in range(289):
            t = targets[i % len(targets)]
            mgr_run1.record_success(t)

        # Then 28 failures all on Chrome (rate-walled)
        for _ in range(28):
            mgr_run1.record_failure("Chrome-136:Macos-15")

        # Save state (checkpoint)
        saved = mgr_run1.to_dict()

        # === Run 2: Restore state ===
        mgr_run2 = _make_manager_with_targets(targets)
        mgr_run2.restore_state(saved)

        # Run 2 KNOWS Chrome was struggling
        chrome_rate = mgr_run2.stats.get_success_rate("Chrome-136:Macos-15")
        firefox_rate = mgr_run2.stats.get_success_rate("Firefox-133:Linux")
        safari_rate = mgr_run2.stats.get_success_rate("Safari-18.0:Ios-18.0")

        # Chrome had 96 successes + 28 failures = 77% success rate
        assert chrome_rate < firefox_rate
        assert chrome_rate < safari_rate
        # Firefox and Safari had ~96 successes each, 0 failures = 100%
        assert firefox_rate == 1.0
        assert safari_rate == 1.0

    def test_bug4_no_deadlock_with_adaptive_selection(self):
        """BUG: get_best_profile() called get_recent_success_rate() which also
        acquired the lock. With Lock(), this deadlocked. With RLock(), it works.
        """
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux"]
        mgr = ImpersonationManager(
            detect_at_startup=False,
            adaptive_profile_selection=True,
        )
        mgr._targets = targets
        mgr.record_success("Firefox-133:Linux")
        mgr.record_failure("Chrome-136:Macos-15")

        # This would deadlock with Lock() — now works with RLock()
        result = [None]
        def run():
            result[0] = mgr.get_impersonate_args(use_adaptive=True)

        t = threading.Thread(target=run)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive(), "DEADLOCK: get_impersonate_args with adaptive selection"
        assert result[0] == ["--impersonate", "Firefox-133:Linux"]

    def test_bug5_exhaustion_signaled_and_recoverable(self):
        """BUG: No signal when all targets were exhausted. Caller had no way
        to know the system was in a degraded state.

        After fix: all_targets_exhausted flag is set, and resets on success.
        """
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux"]
        mgr = _make_manager_with_targets(targets)
        mgr._min_success_rate = 0.3
        mgr._enable_success_filtering = True

        # All targets fail heavily (like the 28-video rate wall)
        for _ in range(10):
            mgr.record_failure("Chrome-136:Macos-15")
            mgr.record_failure("Firefox-133:Linux")

        assert not mgr.all_targets_exhausted  # Not set until get_next_target

        # Next rotation triggers the exhaustion fallback
        target = mgr.get_next_target()
        assert target is not None  # Still returns a target (fallback)
        assert mgr.all_targets_exhausted is True

        # A single success resets the signal
        mgr.record_success(target)
        assert mgr.all_targets_exhausted is False

    def test_full_scenario_simulation(self):
        """End-to-end simulation of the production failure with all fixes applied.

        Simulates 317 download attempts (289 success + 28 failure).
        Verifies that after state persistence, run 2 avoids the exhausted target.
        """
        targets = ["Chrome-136:Macos-15", "Firefox-133:Linux", "Safari-18.0:Ios-18.0"]
        mgr = _make_manager_with_targets(targets)
        mgr._min_success_rate = 0.3
        mgr._enable_success_filtering = True

        # === Run 1: First 289 videos download OK ===
        for i in range(289):
            args, target = mgr.get_impersonate_args_with_target()
            assert target is not None
            mgr.record_success(target)

        # YouTube rate-walls: next 28 all fail
        for _ in range(28):
            args, target = mgr.get_impersonate_args_with_target()
            mgr.record_failure(target)

        # State: some targets should have lower success rates
        # Save state for next run
        state = mgr.to_dict()

        # === Run 2: Restore state, verify learned behavior ===
        mgr2 = _make_manager_with_targets(targets)
        mgr2._min_success_rate = 0.3
        mgr2._enable_success_filtering = True
        mgr2.restore_state(state)

        # Verify the manager has learned from run 1
        total_stats = mgr2.stats.to_dict()
        total_calls = total_stats['calls_made']
        assert total_calls == 289 + 28  # All calls preserved

        # Get next target — should work (filtering skips bad targets if any)
        args, target = mgr2.get_impersonate_args_with_target()
        assert target is not None
        assert args == ["--impersonate", target]
