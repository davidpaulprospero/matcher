"""Tests for src/cli/newproject.py - Google Doc parsing and project setup utilities."""
import re
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.cli.newproject import (
    extract_doc_id,
    extract_drive_links,
    get_date_suffix,
    combine_audio_files,
    download_drive_files,
)


# ---------------------------------------------------------------------------
# AC1: extract_doc_id() parses Google Doc URLs
# ---------------------------------------------------------------------------
class TestExtractDocId:
    """Test extract_doc_id() correctly parses Google Doc URLs."""

    @pytest.mark.fast
    def test_standard_edit_url(self):
        """Verify extraction from standard /edit URL."""
        url = "https://docs.google.com/document/d/ABC123/edit"
        assert extract_doc_id(url) == "ABC123"

    @pytest.mark.fast
    def test_edit_with_sharing_param(self):
        """Verify extraction from /edit?usp=sharing variant."""
        url = "https://docs.google.com/document/d/XYZ789_abc-def/edit?usp=sharing"
        assert extract_doc_id(url) == "XYZ789_abc-def"

    @pytest.mark.fast
    def test_returns_none_for_invalid_url(self):
        """Verify returns None for non-Google Doc URLs."""
        assert extract_doc_id("https://example.com/document") is None

    @pytest.mark.fast
    def test_returns_none_for_empty_string(self):
        """Verify returns None for empty string."""
        assert extract_doc_id("") is None

    @pytest.mark.fast
    def test_returns_none_for_drive_url(self):
        """Verify returns None for Google Drive (not Docs) URL."""
        url = "https://drive.google.com/file/d/ABC123/view"
        assert extract_doc_id(url) is None

    @pytest.mark.fast
    def test_long_doc_id(self):
        """Verify extraction with a realistic long doc ID."""
        doc_id = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
        url = f"https://docs.google.com/document/d/{doc_id}/edit"
        assert extract_doc_id(url) == doc_id

    @pytest.mark.fast
    def test_url_with_fragment(self):
        """Verify extraction when URL has fragment."""
        url = "https://docs.google.com/document/d/ABC123/edit#heading=h.abc"
        assert extract_doc_id(url) == "ABC123"


# ---------------------------------------------------------------------------
# AC2: extract_drive_links() finds Google Drive file IDs in text
# ---------------------------------------------------------------------------
class TestExtractDriveLinks:
    """Test extract_drive_links() finds Google Drive file IDs in text."""

    @pytest.mark.fast
    def test_finds_multiple_ids_in_multiline_text(self):
        """Verify finds multiple IDs in multiline text."""
        text = (
            "Download part 1: https://drive.google.com/file/d/FILE_ID_ONE/view\n"
            "Download part 2: https://drive.google.com/file/d/FILE_ID_TWO/view\n"
            "Download part 3: https://drive.google.com/file/d/FILE_ID_THREE/view"
        )
        result = extract_drive_links(text)
        assert result == ["FILE_ID_ONE", "FILE_ID_TWO", "FILE_ID_THREE"]

    @pytest.mark.fast
    def test_returns_empty_for_no_drive_links(self):
        """Verify returns empty list for text with no Drive links."""
        text = "This is a plain text document with no links at all."
        assert extract_drive_links(text) == []

    @pytest.mark.fast
    def test_returns_empty_for_empty_string(self):
        """Verify returns empty list for empty string."""
        assert extract_drive_links("") == []

    @pytest.mark.fast
    def test_finds_single_link(self):
        """Verify finds a single Drive link."""
        text = "Audio: https://drive.google.com/file/d/1a2B3c4D5e/view?usp=sharing"
        result = extract_drive_links(text)
        assert result == ["1a2B3c4D5e"]

    @pytest.mark.fast
    def test_ignores_non_drive_urls(self):
        """Verify ignores Google Docs URLs (not Drive file links)."""
        text = (
            "Doc: https://docs.google.com/document/d/DOC_ID/edit\n"
            "File: https://drive.google.com/file/d/DRIVE_ID/view"
        )
        result = extract_drive_links(text)
        assert result == ["DRIVE_ID"]

    @pytest.mark.fast
    def test_handles_ids_with_hyphens_and_underscores(self):
        """Verify IDs with hyphens and underscores are extracted."""
        text = "https://drive.google.com/file/d/abc-123_DEF/view"
        result = extract_drive_links(text)
        assert result == ["abc-123_DEF"]


# ---------------------------------------------------------------------------
# AC3: get_date_suffix() returns YYYY-MM-DD format
# ---------------------------------------------------------------------------
class TestGetDateSuffix:
    """Test get_date_suffix() returns correctly formatted date string."""

    @pytest.mark.fast
    def test_format_matches_yyyy_mm_dd(self):
        """Verify format matches YYYY-MM-DD regex pattern."""
        result = get_date_suffix()
        assert re.match(r"\d{4}-\d{2}-\d{2}", result), f"Got: {result}"

    @pytest.mark.fast
    def test_returns_string(self):
        """Verify returns a string type."""
        assert isinstance(get_date_suffix(), str)

    @pytest.mark.fast
    def test_length_is_10(self):
        """Verify date string is exactly 10 characters (YYYY-MM-DD)."""
        assert len(get_date_suffix()) == 10

    @patch("src.cli.newproject.datetime")
    @pytest.mark.fast
    def test_specific_date(self, mock_datetime):
        """Verify specific date formatting."""
        mock_datetime.now.return_value.strftime.return_value = "2026-01-28"
        result = get_date_suffix()
        assert result == "2026-01-28"


# ---------------------------------------------------------------------------
# AC4: combine_audio_files() combines audio using pydub
# ---------------------------------------------------------------------------
class TestCombineAudioFiles:
    """Test combine_audio_files() handles audio combination correctly."""

    @pytest.mark.fast
    def test_returns_false_for_empty_file_list(self):
        """Verify returns False when given empty file list."""
        result = combine_audio_files([], Path("/tmp/output.mp3"))
        assert result is False

    @patch("src.cli.newproject.AudioSegment", create=True)
    @pytest.mark.fast
    def test_combines_multiple_files(self, mock_audio_cls, tmp_path):
        """Verify combines multiple audio files in order."""
        # Create mock audio segments
        seg1 = MagicMock()
        seg2 = MagicMock()
        combined = MagicMock()
        combined.__len__ = MagicMock(return_value=60000)  # 60 seconds
        combined.__iadd__ = MagicMock(return_value=combined)

        mock_audio_cls.empty.return_value = combined
        mock_audio_cls.from_file.side_effect = [seg1, seg2]

        # Create fake input files
        file1 = tmp_path / "part1.mp3"
        file2 = tmp_path / "part2.mp3"
        file1.touch()
        file2.touch()

        output = tmp_path / "output" / "combined.mp3"

        with patch.dict("sys.modules", {"pydub": MagicMock(AudioSegment=mock_audio_cls)}):
            # Re-import to pick up patched module
            import importlib
            import src.cli.newproject as mod
            importlib.reload(mod)

            result = mod.combine_audio_files([file1, file2], output)

        # The function uses pydub internally; verify it called from_file for each input
        assert mock_audio_cls.from_file.call_count == 2

    @patch.dict("sys.modules", {"pydub": None})
    @pytest.mark.fast
    def test_handles_missing_pydub(self, tmp_path, capsys):
        """Verify handles missing pydub gracefully."""
        import importlib
        import src.cli.newproject as mod
        importlib.reload(mod)

        file1 = tmp_path / "part1.mp3"
        file1.touch()

        result = mod.combine_audio_files([file1], tmp_path / "output.mp3")
        assert result is False

        captured = capsys.readouterr()
        assert "pydub" in captured.out.lower() or "error" in captured.out.lower()

        # Restore module
        importlib.reload(mod)


# ---------------------------------------------------------------------------
# AC5: download_drive_files() handles missing gdown gracefully
# ---------------------------------------------------------------------------
class TestDownloadDriveFiles:
    """Test download_drive_files() handles edge cases and missing dependencies."""

    @patch.dict("sys.modules", {"gdown": None})
    @pytest.mark.fast
    def test_handles_missing_gdown(self, tmp_path, capsys):
        """Verify ImportError is caught and meaningful error message printed."""
        import importlib
        import src.cli.newproject as mod
        importlib.reload(mod)

        result = mod.download_drive_files(["file_id_1"], tmp_path)
        assert result == []

        captured = capsys.readouterr()
        assert "gdown" in captured.out.lower()

        # Restore module
        importlib.reload(mod)

    @patch("src.cli.newproject.gdown", create=True)
    @pytest.mark.fast
    def test_downloads_files_in_order(self, mock_gdown, tmp_path):
        """Verify downloads files with correct naming pattern."""
        # Mock gdown.download to create the output files
        def fake_download(url, output, quiet=False, fuzzy=True):
            Path(output).write_text("fake audio data")

        mock_gdown.download.side_effect = fake_download

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            import importlib
            import src.cli.newproject as mod
            importlib.reload(mod)

            result = mod.download_drive_files(
                ["id_A", "id_B"],
                tmp_path / "downloads"
            )

        assert len(result) == 2
        assert result[0].name == "Pt1_download.mp3"
        assert result[1].name == "Pt2_download.mp3"

    @patch("src.cli.newproject.gdown", create=True)
    @pytest.mark.fast
    def test_skips_existing_files(self, mock_gdown, tmp_path):
        """Verify skips download when file already exists and is non-empty."""
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        existing = output_dir / "Pt1_download.mp3"
        existing.write_text("existing audio")

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            import importlib
            import src.cli.newproject as mod
            importlib.reload(mod)

            result = mod.download_drive_files(["id_A"], output_dir)

        assert len(result) == 1
        # gdown.download should NOT have been called
        mock_gdown.download.assert_not_called()

    @patch("src.cli.newproject.gdown", create=True)
    @pytest.mark.fast
    def test_handles_download_failure(self, mock_gdown, tmp_path):
        """Verify handles individual download failure gracefully."""
        mock_gdown.download.side_effect = Exception("Network error")

        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            import importlib
            import src.cli.newproject as mod
            importlib.reload(mod)

            result = mod.download_drive_files(["id_fail"], tmp_path / "dl")

        assert result == []

    @pytest.mark.fast
    def test_empty_file_ids_list(self, tmp_path):
        """Verify empty file_ids returns empty list without error."""
        # download_drive_files tries to import gdown first, mock it
        mock_gdown = MagicMock()
        with patch.dict("sys.modules", {"gdown": mock_gdown}):
            import importlib
            import src.cli.newproject as mod
            importlib.reload(mod)

            result = mod.download_drive_files([], tmp_path)
        assert result == []
