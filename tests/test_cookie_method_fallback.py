"""
Unit tests for CookieMethodFallback.

US-002: Create dedicated unit tests for CookieMethodFallback.
US-005: Add chain validation on init, get_chain_health(), os.access checks.
Tests the ordered fallback chain, advance/reset lifecycle, yt-dlp arg generation,
and file-based method handling (path resolution, missing files, readability).
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.downloader.cookie_method_fallback import CookieMethod, CookieMethodFallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_download_config(
    browser: str = "",
    cookie_files: list = None,
    cookies_path: str = "",
    files_exist: set = None,
):
    """Build a mock DownloadConfig for CookieMethodFallback.

    Args:
        browser: cookies_from_browser value (e.g. "firefox").
        cookie_files: list of file paths for cookie_rotation.cookie_files.
        cookies_path: static cookies_path value.
        files_exist: set of path strings that Path.exists() should return True for.
    """
    files_exist = files_exist or set()

    config = Mock()
    config.cookies_from_browser = browser
    config.cookies_path = cookies_path

    if cookie_files is not None:
        config.cookie_rotation = Mock(cookie_files=cookie_files)
    else:
        config.cookie_rotation = None

    return config, files_exist


def _build_fallback(browser="", cookie_files=None, cookies_path="", files_exist=None,
                    files_readable=None):
    """Convenience: build config + CookieMethodFallback in one call.

    Args:
        files_readable: set of path strings for which os.access(R_OK) returns True.
                        If None, defaults to same as files_exist (all existing files readable).
    """
    config, exist_set = _make_download_config(
        browser=browser,
        cookie_files=cookie_files,
        cookies_path=cookies_path,
        files_exist=files_exist,
    )
    readable_set = files_readable if files_readable is not None else exist_set

    def fake_exists(self):
        return str(self) in exist_set

    def fake_access(path, mode):
        return path in readable_set or str(Path(path)) in readable_set

    with patch.object(Path, "exists", fake_exists), \
         patch("src.downloader.cookie_method_fallback.os.access", fake_access):
        fb = CookieMethodFallback(config)
    return fb


# ===========================================================================
# Test: CookieMethod dataclass
# ===========================================================================


@pytest.mark.fast
class TestCookieMethod:
    """Tests for the CookieMethod dataclass and get_cmd_args()."""

    def test_browser_method_args(self):
        m = CookieMethod(kind="browser", value="firefox", label="browser:firefox")
        assert m.get_cmd_args() == ["--cookies-from-browser", "firefox"]

    def test_file_method_args(self):
        m = CookieMethod(kind="file", value="cookies/main.txt", label="file:main.txt")
        assert m.get_cmd_args() == ["--cookies", "cookies/main.txt"]

    def test_none_method_args(self):
        m = CookieMethod(kind="none", value="", label="no-cookies")
        assert m.get_cmd_args() == []

    def test_unknown_kind_returns_empty(self):
        m = CookieMethod(kind="unknown", value="x", label="unknown")
        assert m.get_cmd_args() == []


# ===========================================================================
# Test: Fallback chain ordering
# ===========================================================================


@pytest.mark.fast
class TestFallbackChainOrdering:
    """Verify the exact sequence: browser -> file(s) -> none."""

    def test_full_chain_order(self):
        """browser:firefox -> file:main.txt -> file:backup1.txt -> no-cookies."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt", "cookies/backup1.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt",
                         "cookies\\backup1.txt", "cookies/backup1.txt"},
        )
        labels = [m.label for m in fb._chain]
        assert labels == [
            "browser:firefox",
            "file:main.txt",
            "file:backup1.txt",
            "no-cookies",
        ], f"Chain order mismatch: {labels}"

    def test_no_browser_starts_with_files(self):
        fb = _build_fallback(
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        assert fb._chain[0].kind == "file"
        assert fb._chain[-1].kind == "none"

    def test_no_files_or_browser_only_none(self):
        fb = _build_fallback()
        assert len(fb._chain) == 1
        assert fb._chain[0].kind == "none"

    def test_cookies_path_appended_after_rotation_files(self):
        fb = _build_fallback(
            cookie_files=["cookies/main.txt"],
            cookies_path="cookies/static.txt",
            files_exist={"cookies\\main.txt", "cookies/main.txt",
                         "cookies\\static.txt", "cookies/static.txt"},
        )
        labels = [m.label for m in fb._chain]
        assert labels == ["file:main.txt", "file:static.txt", "no-cookies"]

    def test_duplicate_file_deduplication(self):
        """Same file in rotation_files and cookies_path is not duplicated."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt"],
            cookies_path="cookies/main.txt",
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        file_methods = [m for m in fb._chain if m.kind == "file"]
        assert len(file_methods) == 1, f"Expected 1 file method, got {len(file_methods)}"

    def test_none_always_last(self):
        fb = _build_fallback(browser="chrome")
        assert fb._chain[-1].kind == "none"


# ===========================================================================
# Test: advance() through the chain
# ===========================================================================


@pytest.mark.fast
class TestAdvance:
    """Test advance() walks through chain and reports exhaustion."""

    def test_advance_returns_true_until_exhausted(self):
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        # Chain: browser:firefox, file:main.txt, no-cookies (3 methods)
        assert fb.methods_remaining == 3
        assert fb.advance() is True   # -> file:main.txt
        assert fb.advance() is True   # -> no-cookies
        assert fb.advance() is False  # exhausted

    def test_is_exhausted_after_advancing_past_end(self):
        fb = _build_fallback()  # only no-cookies
        assert fb.is_exhausted is False
        fb.advance()
        assert fb.is_exhausted is True

    def test_methods_remaining_decreases(self):
        fb = _build_fallback(browser="firefox")
        # Chain: browser:firefox, no-cookies
        assert fb.methods_remaining == 2
        fb.advance()
        assert fb.methods_remaining == 1
        fb.advance()
        assert fb.methods_remaining == 0

    def test_current_method_changes_on_advance(self):
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        assert fb.current_method.label == "browser:firefox"
        fb.advance()
        assert fb.current_method.label == "file:main.txt"
        fb.advance()
        assert fb.current_method.label == "no-cookies"


# ===========================================================================
# Test: reset_for_next_download()
# ===========================================================================


@pytest.mark.fast
class TestReset:
    """Test reset_for_next_download() returns to correct starting position."""

    def test_reset_without_prior_success_goes_to_zero(self):
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        fb.advance()
        fb.advance()
        fb.reset_for_next_download()
        assert fb.current_method.label == "browser:firefox"

    def test_reset_after_success_goes_to_rotated_method(self):
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        # Succeed on first method (browser:firefox), rotation advances to file:main.txt
        fb.mark_success()
        fb.reset_for_next_download()
        assert fb.current_method.label == "file:main.txt"

    def test_reset_after_exhaustion_restores_position(self):
        fb = _build_fallback()  # only no-cookies
        fb.advance()  # exhausted
        assert fb.is_exhausted
        fb.reset_for_next_download()
        assert fb.is_exhausted is False
        assert fb.current_method.label == "no-cookies"


# ===========================================================================
# Test: get_cmd_args() integration
# ===========================================================================


@pytest.mark.fast
class TestGetCmdArgs:
    """Test get_cmd_args() returns correct yt-dlp args for each method type."""

    def test_browser_args(self):
        fb = _build_fallback(browser="firefox")
        assert fb.get_cmd_args() == ["--cookies-from-browser", "firefox"]

    def test_file_args(self):
        fb = _build_fallback(
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        assert fb.get_cmd_args() == ["--cookies", "cookies/main.txt"]

    def test_none_args(self):
        fb = _build_fallback()
        assert fb.get_cmd_args() == []

    def test_exhausted_returns_empty(self):
        fb = _build_fallback()
        fb.advance()
        assert fb.is_exhausted
        assert fb.get_cmd_args() == []

    def test_args_change_after_advance(self):
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        assert fb.get_cmd_args() == ["--cookies-from-browser", "firefox"]
        fb.advance()
        assert fb.get_cmd_args() == ["--cookies", "cookies/main.txt"]
        fb.advance()
        assert fb.get_cmd_args() == []  # no-cookies


# ===========================================================================
# Test: File-based method handling
# ===========================================================================


@pytest.mark.fast
class TestFileBasedMethods:
    """Test cookie file path resolution and missing file handling."""

    def test_missing_cookie_file_skipped(self):
        """Files that don't exist are excluded from the chain."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt", "cookies/missing.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        labels = [m.label for m in fb._chain]
        assert "file:missing.txt" not in labels
        assert "file:main.txt" in labels

    def test_all_files_missing_falls_back_to_none(self):
        fb = _build_fallback(
            cookie_files=["cookies/missing1.txt", "cookies/missing2.txt"],
            files_exist=set(),
        )
        assert len(fb._chain) == 1
        assert fb._chain[0].kind == "none"

    def test_cookies_path_missing_not_added(self):
        fb = _build_fallback(
            cookies_path="cookies/static.txt",
            files_exist=set(),
        )
        labels = [m.label for m in fb._chain]
        assert "file:static.txt" not in labels

    def test_file_path_preserved_in_args(self):
        """get_cmd_args() uses the original path string, not resolved."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        args = fb.get_cmd_args()
        assert args == ["--cookies", "cookies/main.txt"]

    def test_multiple_existing_files_all_included(self):
        fb = _build_fallback(
            cookie_files=["cookies/main.txt", "cookies/backup1.txt", "cookies/backup2.txt"],
            files_exist={
                "cookies\\main.txt", "cookies/main.txt",
                "cookies\\backup1.txt", "cookies/backup1.txt",
                "cookies\\backup2.txt", "cookies/backup2.txt",
            },
        )
        file_methods = [m for m in fb._chain if m.kind == "file"]
        assert len(file_methods) == 3


# ===========================================================================
# Test: mark_success() and rotation
# ===========================================================================


@pytest.mark.fast
class TestMarkSuccess:
    """Test proactive rotation on success."""

    def test_mark_success_rotates_to_next_authenticated(self):
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        # Chain: browser:firefox, file:main.txt, no-cookies
        fb.mark_success()
        fb.reset_for_next_download()
        assert fb.current_method.label == "file:main.txt"

    def test_mark_success_skips_none_method(self):
        """Rotation skips 'no-cookies' since rotating to no-auth is counterproductive."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        # Advance to file:main.txt, succeed there
        fb.advance()
        fb.mark_success()
        fb.reset_for_next_download()
        # Should rotate to browser:firefox (wrapping around, skipping no-cookies)
        assert fb.current_method.label == "browser:firefox"

    def test_mark_success_single_authenticated_stays(self):
        """With only one authenticated method, rotation stays put."""
        fb = _build_fallback(browser="firefox")
        # Chain: browser:firefox, no-cookies
        fb.mark_success()
        fb.reset_for_next_download()
        assert fb.current_method.label == "browser:firefox"

    def test_mark_success_no_op_when_exhausted(self):
        fb = _build_fallback()
        fb.advance()
        assert fb.is_exhausted
        fb.mark_success()  # should not raise


# ===========================================================================
# Test: get_status()
# ===========================================================================


@pytest.mark.fast
class TestGetStatus:
    """Test status reporting for logging/debugging."""

    def test_status_shows_chain_labels(self):
        fb = _build_fallback(browser="firefox")
        status = fb.get_status()
        assert status["chain"] == ["browser:firefox", "no-cookies"]

    def test_status_shows_current_method(self):
        fb = _build_fallback(browser="firefox")
        assert fb.get_status()["current_method"] == "browser:firefox"

    def test_status_shows_exhausted(self):
        fb = _build_fallback()
        fb.advance()
        assert fb.get_status()["is_exhausted"] is True
        assert fb.get_status()["current_method"] == "exhausted"

    def test_status_shows_last_success(self):
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        assert fb.get_status()["last_success"] is None
        fb.mark_success()
        assert fb.get_status()["last_success"] is not None


# ===========================================================================
# Test: Chain validation on init (US-005)
# ===========================================================================


@pytest.mark.fast
class TestChainValidation:
    """Test init-time chain validation: file existence, readability, health reporting."""

    def test_mix_existing_and_missing_files(self):
        """Only valid (existing + readable) files appear in the chain."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt", "cookies/missing.txt", "cookies/backup1.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt",
                         "cookies\\backup1.txt", "cookies/backup1.txt"},
        )
        labels = [m.label for m in fb._chain]
        assert "file:main.txt" in labels
        assert "file:backup1.txt" in labels
        assert "file:missing.txt" not in labels

    def test_all_files_invalid_degrades_to_browser_and_none(self):
        """If all file methods invalid, chain is [browser, none]."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/missing1.txt", "cookies/missing2.txt"],
            files_exist=set(),
        )
        labels = [m.label for m in fb._chain]
        assert labels == ["browser:firefox", "no-cookies"]

    def test_all_files_invalid_no_browser_degrades_to_none(self):
        """If all file methods invalid and no browser, chain is [none]."""
        fb = _build_fallback(
            cookie_files=["cookies/missing1.txt", "cookies/missing2.txt"],
            files_exist=set(),
        )
        labels = [m.label for m in fb._chain]
        assert labels == ["no-cookies"]

    def test_file_exists_but_not_readable_skipped(self):
        """A file that exists but isn't readable (R_OK) is skipped."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt", "cookies/locked.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt",
                         "cookies\\locked.txt", "cookies/locked.txt"},
            # locked.txt exists but is NOT in the readable set
            files_readable={"cookies\\main.txt", "cookies/main.txt"},
        )
        labels = [m.label for m in fb._chain]
        assert "file:main.txt" in labels
        assert "file:locked.txt" not in labels

    def test_get_chain_health_all_valid(self):
        """get_chain_health() reports correct counts when all files valid."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        health = fb.get_chain_health()
        # Chain: browser:firefox, file:main.txt, no-cookies = 3 valid, 0 skipped
        assert health["valid_methods"] == 3
        assert health["total_methods"] == 3
        assert health["skipped_files"] == []

    def test_get_chain_health_with_skipped(self):
        """get_chain_health() reports skipped files."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt", "cookies/missing.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        health = fb.get_chain_health()
        # Chain: file:main.txt, no-cookies = 2 valid; missing.txt = 1 skipped
        assert health["valid_methods"] == 2
        assert health["total_methods"] == 3
        assert len(health["skipped_files"]) == 1
        assert "cookies/missing.txt" in health["skipped_files"]

    def test_get_chain_health_all_files_skipped(self):
        """get_chain_health() when all file methods are invalid."""
        fb = _build_fallback(
            cookie_files=["cookies/a.txt", "cookies/b.txt"],
            cookies_path="cookies/c.txt",
            files_exist=set(),
        )
        health = fb.get_chain_health()
        # Chain: no-cookies = 1 valid; a.txt, b.txt, c.txt = 3 skipped
        assert health["valid_methods"] == 1
        assert health["total_methods"] == 4
        assert len(health["skipped_files"]) == 3

    def test_skipped_files_logged_as_warnings(self):
        """Each skipped file produces a logger.warning call."""
        config, exist_set = _make_download_config(
            cookie_files=["cookies/missing1.txt", "cookies/missing2.txt"],
            files_exist=set(),
        )

        def fake_exists(self):
            return str(self) in exist_set

        def fake_access(path, mode):
            return False

        with patch.object(Path, "exists", fake_exists), \
             patch("src.downloader.cookie_method_fallback.os.access", fake_access), \
             patch("src.downloader.cookie_method_fallback.logger") as mock_logger:
            CookieMethodFallback(config)

        # Two missing files = two warning calls
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "Cookie file not found, skipping:" in str(call)
        ]
        assert len(warning_calls) == 2

    def test_health_summary_logged_at_info(self):
        """Chain health summary is logged at INFO level on init."""
        config, exist_set = _make_download_config(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )

        def fake_exists(self):
            return str(self) in exist_set

        def fake_access(path, mode):
            return True

        with patch.object(Path, "exists", fake_exists), \
             patch("src.downloader.cookie_method_fallback.os.access", fake_access), \
             patch("src.downloader.cookie_method_fallback.logger") as mock_logger:
            CookieMethodFallback(config)

        # Look for the health summary INFO log
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        health_log = [c for c in info_calls if "methods available" in c]
        assert len(health_log) >= 1, f"Expected health summary log, got: {info_calls}"


# ===========================================================================
# Test: Cross-keyword exhaustion (US-003)
# ===========================================================================


@pytest.mark.fast
class TestCrossKeywordExhaustion:
    """
    US-003: Test CookieMethodFallback cross-keyword exhaustion paths.

    Covers full-chain advance, mark_success rotation, reset after exhaustion,
    get_status during partial traversal, and duplicate path deduplication.
    """

    # ---- AC1: Full chain advance with type verification ----

    def test_advance_full_chain_returns_correct_types(self):
        """Advance through browser -> file:main -> file:backup1 -> file:backup2 -> no-cookies.

        Each transition must return True and produce the correct CookieMethod kind.
        """
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt", "cookies/backup1.txt", "cookies/backup2.txt"],
            files_exist={
                "cookies\\main.txt", "cookies/main.txt",
                "cookies\\backup1.txt", "cookies/backup1.txt",
                "cookies\\backup2.txt", "cookies/backup2.txt",
            },
        )
        # Chain: browser:firefox, file:main.txt, file:backup1.txt, file:backup2.txt, no-cookies
        assert len(fb._chain) == 5

        # Start: browser
        assert fb.current_method.kind == "browser"
        assert fb.current_method.label == "browser:firefox"

        # Advance 1: -> file:main.txt
        assert fb.advance() is True
        assert fb.current_method.kind == "file"
        assert fb.current_method.label == "file:main.txt"

        # Advance 2: -> file:backup1.txt
        assert fb.advance() is True
        assert fb.current_method.kind == "file"
        assert fb.current_method.label == "file:backup1.txt"

        # Advance 3: -> file:backup2.txt
        assert fb.advance() is True
        assert fb.current_method.kind == "file"
        assert fb.current_method.label == "file:backup2.txt"

        # Advance 4: -> no-cookies
        assert fb.advance() is True
        assert fb.current_method.kind == "none"
        assert fb.current_method.label == "no-cookies"

        # Advance 5: exhausted
        assert fb.advance() is False
        assert fb.is_exhausted is True

    # ---- AC2: mark_success() proactive rotation ----

    def test_mark_success_on_method_0_advances_to_method_1(self):
        """After success on method 0 (browser), last_success_index points to method 1 (file).

        This ensures load balancing: the next download starts at the next
        authenticated method, not the same one.
        """
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt", "cookies/backup1.txt"],
            files_exist={
                "cookies\\main.txt", "cookies/main.txt",
                "cookies\\backup1.txt", "cookies/backup1.txt",
            },
        )
        # Chain: browser:firefox (0), file:main.txt (1), file:backup1.txt (2), no-cookies (3)
        assert fb._current_index == 0
        assert fb.current_method.label == "browser:firefox"

        fb.mark_success()

        # _last_success_index should point to next authenticated method (index 1)
        assert fb._last_success_index == 1
        assert fb._chain[fb._last_success_index].label == "file:main.txt"

    def test_mark_success_rotation_wraps_around(self):
        """Success on last authenticated method wraps to first authenticated method."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        # Chain: browser:firefox (0), file:main.txt (1), no-cookies (2)
        fb.advance()  # -> file:main.txt (index 1)
        fb.mark_success()

        # Next authenticated after file:main.txt: skip no-cookies (index 2), wrap to browser (index 0)
        assert fb._last_success_index == 0
        assert fb._chain[fb._last_success_index].label == "browser:firefox"

    # ---- AC3: reset_for_next_download() after exhaustion ----

    def test_reset_after_exhaustion_uses_last_success_index(self):
        """After exhaustion, reset goes to last_success_index, not always 0."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt", "cookies/backup1.txt"],
            files_exist={
                "cookies\\main.txt", "cookies/main.txt",
                "cookies\\backup1.txt", "cookies/backup1.txt",
            },
        )
        # Chain: browser:firefox (0), file:main.txt (1), file:backup1.txt (2), no-cookies (3)

        # Advance to file:main.txt and succeed there
        fb.advance()
        fb.mark_success()
        # _last_success_index should be file:backup1.txt (next authenticated after file:main.txt)
        success_idx = fb._last_success_index
        expected_label = fb._chain[success_idx].label

        # Exhaust remaining
        fb.advance()
        fb.advance()
        fb.advance()
        assert fb.is_exhausted

        # Reset should go to last_success_index, not 0
        fb.reset_for_next_download()
        assert fb.is_exhausted is False
        assert fb.current_method.label == expected_label
        assert fb._current_index == success_idx

    def test_reset_without_success_goes_to_zero_after_exhaustion(self):
        """After exhaustion with no prior success, reset goes to index 0."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        # Exhaust without ever calling mark_success
        fb.advance()
        fb.advance()
        fb.advance()
        assert fb.is_exhausted

        fb.reset_for_next_download()
        assert fb._current_index == 0
        assert fb.current_method.label == "browser:firefox"

    # ---- AC4: get_status() after partial traversal ----

    def test_get_status_remaining_methods_after_partial_traversal(self):
        """get_status() reports accurate remaining_methods and current_method mid-chain."""
        fb = _build_fallback(
            browser="firefox",
            cookie_files=["cookies/main.txt", "cookies/backup1.txt"],
            files_exist={
                "cookies\\main.txt", "cookies/main.txt",
                "cookies\\backup1.txt", "cookies/backup1.txt",
            },
        )
        # Chain has 4 methods: browser, main, backup1, no-cookies

        # Before any advance
        assert fb.methods_remaining == 4
        status = fb.get_status()
        assert status["current_method"] == "browser:firefox"
        assert status["current_index"] == 0

        # After 1 advance
        fb.advance()
        assert fb.methods_remaining == 3
        status = fb.get_status()
        assert status["current_method"] == "file:main.txt"
        assert status["current_index"] == 1

        # After 2 advances
        fb.advance()
        assert fb.methods_remaining == 2
        status = fb.get_status()
        assert status["current_method"] == "file:backup1.txt"
        assert status["current_index"] == 2

        # After 3 advances (at no-cookies)
        fb.advance()
        assert fb.methods_remaining == 1
        status = fb.get_status()
        assert status["current_method"] == "no-cookies"
        assert status["current_index"] == 3

        # Exhausted
        fb.advance()
        assert fb.methods_remaining == 0
        status = fb.get_status()
        assert status["current_method"] == "exhausted"
        assert status["is_exhausted"] is True

    # ---- AC5: Duplicate cookie file deduplication ----

    def test_duplicate_paths_in_cookie_files_deduplicated(self):
        """Duplicate entries within cookie_rotation.cookie_files are deduplicated."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt", "cookies/main.txt", "cookies/main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        file_methods = [m for m in fb._chain if m.kind == "file"]
        assert len(file_methods) == 1, (
            f"Expected 1 file method after dedup, got {len(file_methods)}: "
            f"{[m.label for m in file_methods]}"
        )

    def test_duplicate_resolved_paths_deduplicated(self):
        """Paths that resolve to the same file are deduplicated (forward vs back slash)."""
        fb = _build_fallback(
            cookie_files=["cookies/main.txt", "cookies\\main.txt"],
            files_exist={"cookies\\main.txt", "cookies/main.txt"},
        )
        file_methods = [m for m in fb._chain if m.kind == "file"]
        assert len(file_methods) == 1, (
            f"Expected 1 file method after path normalization dedup, got {len(file_methods)}: "
            f"{[m.label for m in file_methods]}"
        )
