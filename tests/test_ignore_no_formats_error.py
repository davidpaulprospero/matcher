"""
Tests for --ignore-no-formats-error fix in caption_fetcher.py.

When escalation args (--extractor-args youtube:player_client=...) are applied
to subtitle-only yt-dlp commands, video format resolution can fail with
"Requested format is not available" BEFORE subtitle extraction occurs — even
though --skip-download is set. Adding --ignore-no-formats-error lets yt-dlp
continue past video format failures to subtitle download/listing.

Verifies:
1. --ignore-no-formats-error present in subtitle fetch command
2. --ignore-no-formats-error present in list-subs command
3. Mutation tests: removing the flag would be caught
"""

import pytest
import tempfile
import textwrap
from pathlib import Path
from unittest.mock import Mock, patch

from src.caption_fetcher import CaptionFetcher


@pytest.mark.fast
class TestIgnoreNoFormatsErrorInSubtitleFetch:
    """--ignore-no-formats-error must be in the subtitle download command."""

    def _make_fetcher(self):
        return CaptionFetcher()

    def test_subtitle_fetch_cmd_includes_ignore_no_formats_error(self):
        """_fetch_subtitle_with_format command includes --ignore-no-formats-error."""
        fetcher = self._make_fetcher()
        captured_cmd = None

        def capture_subprocess_run(cmd, *args, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            return Mock(stdout="", stderr="", returncode=0)

        with patch('subprocess.run', side_effect=capture_subprocess_run):
            with tempfile.TemporaryDirectory() as td:
                try:
                    fetcher._fetch_subtitle_with_format(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "dQw4w9WgXcQ",
                        Path(td),
                        "en",
                        True,
                        "json3"
                    )
                except Exception:
                    pass  # May fail on missing file, we only need the command

        assert captured_cmd is not None, "subprocess.run was not called"
        assert '--ignore-no-formats-error' in captured_cmd, (
            f"--ignore-no-formats-error missing from subtitle fetch command: {captured_cmd}"
        )

    def test_subtitle_fetch_flag_coexists_with_skip_download(self):
        """Both --skip-download and --ignore-no-formats-error must be present."""
        fetcher = self._make_fetcher()
        captured_cmd = None

        def capture_subprocess_run(cmd, *args, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            return Mock(stdout="", stderr="", returncode=0)

        with patch('subprocess.run', side_effect=capture_subprocess_run):
            with tempfile.TemporaryDirectory() as td:
                try:
                    fetcher._fetch_subtitle_with_format(
                        "https://www.youtube.com/watch?v=test123",
                        "test123",
                        Path(td),
                        "en",
                        False,
                        "srt"
                    )
                except Exception:
                    pass

        assert captured_cmd is not None
        assert '--skip-download' in captured_cmd
        assert '--ignore-no-formats-error' in captured_cmd


@pytest.mark.fast
class TestIgnoreNoFormatsErrorInListSubs:
    """--ignore-no-formats-error must be in the list-subs command."""

    def _make_fetcher(self):
        return CaptionFetcher()

    def test_list_subs_cmd_includes_ignore_no_formats_error(self):
        """list_available_languages command includes --ignore-no-formats-error."""
        fetcher = self._make_fetcher()
        captured_cmd = None

        def capture_subprocess_run(cmd, *args, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            return Mock(stdout="", stderr="", returncode=0)

        with patch('subprocess.run', side_effect=capture_subprocess_run):
            fetcher.list_available_languages("dQw4w9WgXcQ")

        assert captured_cmd is not None, "subprocess.run was not called"
        assert '--ignore-no-formats-error' in captured_cmd, (
            f"--ignore-no-formats-error missing from list-subs command: {captured_cmd}"
        )

    def test_list_subs_flag_coexists_with_list_subs_and_skip_download(self):
        """--list-subs, --skip-download, and --ignore-no-formats-error all present."""
        fetcher = self._make_fetcher()
        captured_cmd = None

        def capture_subprocess_run(cmd, *args, **kwargs):
            nonlocal captured_cmd
            captured_cmd = cmd
            return Mock(stdout="", stderr="", returncode=0)

        with patch('subprocess.run', side_effect=capture_subprocess_run):
            fetcher.list_available_languages("dQw4w9WgXcQ")

        assert captured_cmd is not None
        assert '--list-subs' in captured_cmd
        assert '--skip-download' in captured_cmd
        assert '--ignore-no-formats-error' in captured_cmd


@pytest.mark.fast
class TestIgnoreNoFormatsErrorMutations:
    """Mutation tests: verify source contains the flag so removal would be caught."""

    def _read_source(self):
        src_path = Path(__file__).parent.parent / 'src' / 'caption_fetcher.py'
        return src_path.read_text(encoding='utf-8')

    def test_mutation_kill_fetch_subtitle_flag_present_in_source(self):
        """Source inspection: --ignore-no-formats-error appears in _fetch_subtitle_with_format context."""
        source = self._read_source()
        # Find the _fetch_subtitle_with_format method and check it contains the flag
        method_start = source.find('def _fetch_subtitle_with_format(')
        assert method_start != -1, "_fetch_subtitle_with_format method not found"
        # Look within a reasonable range (the cmd array is ~40 lines after method start)
        method_chunk = source[method_start:method_start + 2000]
        assert '--ignore-no-formats-error' in method_chunk, (
            "MUTATION SURVIVED: --ignore-no-formats-error removed from _fetch_subtitle_with_format"
        )

    def test_mutation_kill_list_subs_flag_present_in_source(self):
        """Source inspection: --ignore-no-formats-error appears in list_available_languages context."""
        source = self._read_source()
        # Find the list_available_languages method and check it contains the flag
        method_start = source.find('def list_available_languages(')
        assert method_start != -1, "list_available_languages method not found"
        # The cmd array is ~3000 chars in (after docstring + cache check)
        method_chunk = source[method_start:method_start + 5000]
        assert '--ignore-no-formats-error' in method_chunk, (
            "MUTATION SURVIVED: --ignore-no-formats-error removed from list_available_languages"
        )

    def test_mutation_kill_flag_removal_from_fetch_cmd(self):
        """Mutant: removing --ignore-no-formats-error from fetch cmd is detected."""
        source = self._read_source()
        # Apply mutation: remove the flag line from _fetch_subtitle_with_format
        mutant = source.replace(
            "            '--ignore-no-formats-error',  # Continue to subtitles even if video format resolution fails\n",
            "",
            1,  # Only first occurrence (in _fetch_subtitle_with_format)
        )
        assert mutant != source, "Mutation did not apply — flag line not found"
        # Verify the mutant no longer has the flag in that method
        method_start = mutant.find('def _fetch_subtitle_with_format(')
        method_chunk = mutant[method_start:method_start + 2000]
        assert '--ignore-no-formats-error' not in method_chunk, (
            "Mutation failed to remove flag from _fetch_subtitle_with_format"
        )

    def test_mutation_kill_flag_removal_from_listsubs_cmd(self):
        """Mutant: removing --ignore-no-formats-error from list-subs cmd is detected."""
        source = self._read_source()
        # Apply mutation: remove the flag line from list_available_languages
        mutant = source.replace(
            "            '--ignore-no-formats-error',  # Continue to list-subs even if video format resolution fails\n",
            "",
            1,
        )
        assert mutant != source, "Mutation did not apply — flag line not found"
        method_start = mutant.find('def list_available_languages(')
        method_chunk = mutant[method_start:method_start + 2000]
        assert '--ignore-no-formats-error' not in method_chunk, (
            "Mutation failed to remove flag from list_available_languages"
        )

    def test_mutation_kill_flag_count_exactly_two(self):
        """Source must contain exactly 2 occurrences of --ignore-no-formats-error."""
        source = self._read_source()
        count = source.count("'--ignore-no-formats-error'")
        assert count == 2, (
            f"Expected exactly 2 occurrences of --ignore-no-formats-error, found {count}. "
            f"MUTATION SURVIVED: flag added/removed from unexpected location"
        )
