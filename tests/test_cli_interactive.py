"""
Tests for CLI interactive prompts.

Tests the find_voiceover_interactive function for:
- Finding voiceover files in project directory
- Auto-selection with single file
- User selection with multiple files
- Cancel behavior
- Invalid input handling
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from unittest.mock import mock_open

from src.cli.interactive import find_voiceover_interactive


class TestFindVoiceoverInteractive:
    """Tests for find_voiceover_interactive function"""

    @pytest.mark.fast
    def test_no_voiceover_files_found(self, tmp_path, capsys):
        """Test when no voiceover files exist in project"""
        # Create empty voiceover directory
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()

        result = find_voiceover_interactive(tmp_path)

        assert result is None
        captured = capsys.readouterr()
        assert "NO VOICEOVER FILES FOUND" in captured.out

    @pytest.mark.fast
    def test_single_voiceover_file_auto_select(self, tmp_path, capsys):
        """Test auto-selection when only one voiceover file exists"""
        # Create voiceover file
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        vo_file = voiceover_dir / " narration.srt"
        vo_file.write_text("test content")

        result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "narration.srt" in result
        captured = capsys.readouterr()
        assert "Auto-detected voiceover" in captured.out

    @pytest.mark.fast
    def test_multiple_voiceover_files_prompt(self, tmp_path, capsys):
        """Test user prompt when multiple voiceover files exist"""
        # Create multiple voiceover files
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "intro.srt").write_text("intro")
        (voiceover_dir / "main.srt").write_text("main")
        (voiceover_dir / "outro.srt").write_text("outro")

        # Mock input to select option 2
        with patch('builtins.input', return_value='2'):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "main.srt" in result
        captured = capsys.readouterr()
        assert "SELECT VOICEOVER FILE" in captured.out

    @pytest.mark.fast
    def test_user_selects_first_file(self, tmp_path, capsys):
        """Test selecting first file from multiple options"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "first.mp3").write_text("first")
        (voiceover_dir / "second.mp3").write_text("second")

        with patch('builtins.input', return_value='1'):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "first.mp3" in result

    @pytest.mark.fast
    def test_user_selects_last_file(self, tmp_path, capsys):
        """Test selecting last file from multiple options"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "a.mp3").write_text("a")
        (voiceover_dir / "b.mp3").write_text("b")
        (voiceover_dir / "c.mp3").write_text("c")

        with patch('builtins.input', return_value='3'):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "c.mp3" in result

    @pytest.mark.fast
    def test_cancel_with_zero(self, tmp_path, capsys):
        """Test cancel by entering 0"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "file1.srt").write_text("test")
        (voiceover_dir / "file2.srt").write_text("test")

        with patch('builtins.input', return_value='0'):
            result = find_voiceover_interactive(tmp_path)

        assert result is None
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out

    @pytest.mark.fast
    def test_cancel_with_keyboard_interrupt(self, tmp_path, capsys):
        """Test cancel with KeyboardInterrupt"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "file1.srt").write_text("test")
        (voiceover_dir / "file2.srt").write_text("test")

        with patch('builtins.input', side_effect=KeyboardInterrupt()):
            result = find_voiceover_interactive(tmp_path)

        assert result is None
        captured = capsys.readouterr()
        assert "Cancelled" in captured.out

    @pytest.mark.fast
    def test_cancel_with_eof_error(self, tmp_path, capsys):
        """Test cancel with EOFError"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "file1.srt").write_text("test")
        (voiceover_dir / "file2.srt").write_text("test")

        with patch('builtins.input', side_effect=EOFError()):
            result = find_voiceover_interactive(tmp_path)

        assert result is None

    @pytest.mark.fast
    def test_invalid_choice_shows_error(self, tmp_path, capsys):
        """Test error message for invalid numeric choice"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "file1.srt").write_text("test")
        (voiceover_dir / "file2.srt").write_text("test")

        # First input is invalid, second is valid
        with patch('builtins.input', side_effect=['99', '1']):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "file1.srt" in result
        captured = capsys.readouterr()
        assert "Invalid choice" in captured.out

    @pytest.mark.fast
    def test_empty_input_retries(self, tmp_path, capsys):
        """Test that empty input is ignored and retries"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "file1.srt").write_text("test")
        (voiceover_dir / "file2.srt").write_text("test")

        # First input empty, second is valid
        with patch('builtins.input', side_effect=['', '1']):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "file1.srt" in result

    @pytest.mark.fast
    def test_type_filename_for_selection(self, tmp_path, capsys):
        """Test selecting file by typing filename instead of number"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "intro.srt").write_text("test")
        (voiceover_dir / "main.srt").write_text("test")

        # Type part of filename
        with patch('builtins.input', return_value='intro'):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "intro.srt" in result

    @pytest.mark.fast
    def test_searches_project_root_too(self, tmp_path, capsys):
        """Test that files in project root are also found"""
        # Create file in project root, not voiceover subdir
        (tmp_path / "voiceover").mkdir()
        (tmp_path / "root_file.mp3").write_text("test")

        result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "root_file.mp3" in result

    @pytest.mark.fast
    def test_no_duplicates_from_both_locations(self, tmp_path, capsys):
        """Test that same-named file in voiceover dir and root are separate candidates"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()

        # Same file name in both locations - these are DIFFERENT files
        (voiceover_dir / "dup.srt").write_text("test1")
        (tmp_path / "dup.srt").write_text("test2")

        # Both files should appear as separate options
        with patch('builtins.input', return_value='1'):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        captured = capsys.readouterr()
        # Should show both files as separate options
        assert "voiceover/dup.srt" in captured.out
        assert "dup.srt" in captured.out
        assert "SELECT VOICEOVER FILE" in captured.out

    @pytest.mark.fast
    def test_files_sorted_alphabetically(self, tmp_path, capsys):
        """Test that files are presented in alphabetical order"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "zebra.mp3").write_text("test")
        (voiceover_dir / "alpha.mp3").write_text("test")
        (voiceover_dir / "beta.mp3").write_text("test")

        with patch('builtins.input', return_value='1'):
            result = find_voiceover_interactive(tmp_path)

        # First should be alpha (sorted)
        assert "alpha.mp3" in result
        captured = capsys.readouterr()
        # Check order in displayed list
        assert captured.out.index("alpha") < captured.out.index("beta") < captured.out.index("zebra")

    @pytest.mark.fast
    def test_shows_file_size_and_type(self, tmp_path, capsys):
        """Test that file size and type hint are displayed"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "test1.srt").write_text("x" * 1000)  # ~1KB
        (voiceover_dir / "test2.srt").write_text("test")

        with patch('builtins.input', return_value='1'):
            result = find_voiceover_interactive(tmp_path)

        captured = capsys.readouterr()
        # Should show file type hint (subtitles for .srt)
        assert "subtitles" in captured.out
        # Should show file size
        assert "MB" in captured.out

    @pytest.mark.fast
    def test_case_insensitive_extension_match(self, tmp_path, capsys):
        """Test that extension matching is case insensitive"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "test.SRT").write_text("test")  # uppercase
        (voiceover_dir / "test.MP3").write_text("test")  # uppercase

        # Need to mock input since multiple files trigger prompt
        with patch('builtins.input', return_value='1'):
            result = find_voiceover_interactive(tmp_path)

        # Should find both files despite uppercase extensions
        assert result is not None

    @pytest.mark.fast
    def test_invalid_input_shows_error(self, tmp_path, capsys):
        """Test error message for non-numeric input"""
        voiceover_dir = tmp_path / "voiceover"
        voiceover_dir.mkdir()
        (voiceover_dir / "file1.srt").write_text("test")
        (voiceover_dir / "file2.srt").write_text("test")

        # Non-numeric input that's not a filename
        with patch('builtins.input', side_effect=['abc', '1']):
            result = find_voiceover_interactive(tmp_path)

        assert result is not None
        captured = capsys.readouterr()
        assert "Invalid input" in captured.out
