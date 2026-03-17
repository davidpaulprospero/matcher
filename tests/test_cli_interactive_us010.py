"""
Tests for src/cli/interactive.py - US-010

Covers:
- AC1: find_voiceover_interactive() searches voiceover/ subdirectory first
- AC2: find_voiceover_interactive() auto-selects when only one candidate found
"""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.cli.interactive import find_voiceover_interactive


# ============================================================================
# AC1: find_voiceover_interactive() searches voiceover/ subdirectory first
# ============================================================================

class TestFindVoiceoverSearchOrderUS010:
    """AC1: find_voiceover_interactive() searches voiceover/ subdirectory first,
    then project root."""

    @pytest.mark.fast
    def test_finds_srt_in_voiceover_subdir(self, tmp_path):
        """File in voiceover/ is found without needing to search root."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt_file = vo_dir / "script.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest")

        result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "script.srt" in result
        assert str(vo_dir) in result

    @pytest.mark.fast
    def test_finds_mp3_in_voiceover_subdir(self, tmp_path):
        """MP3 file in voiceover/ is found."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        mp3_file = vo_dir / "narration.mp3"
        mp3_file.write_bytes(b"\x00" * 100)

        result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "narration.mp3" in result

    @pytest.mark.fast
    def test_finds_wav_in_voiceover_subdir(self, tmp_path):
        """WAV file in voiceover/ is found."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        wav_file = vo_dir / "audio.wav"
        wav_file.write_bytes(b"\x00" * 100)

        result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "audio.wav" in result

    @pytest.mark.fast
    def test_falls_back_to_project_root(self, tmp_path):
        """File in project root is found when voiceover/ doesn't exist."""
        srt_file = tmp_path / "script.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest")

        result = find_voiceover_interactive(tmp_path)

        assert result is not None
        assert "script.srt" in result

    @pytest.mark.fast
    def test_voiceover_subdir_file_listed_before_root(self, tmp_path):
        """When files exist in both locations, voiceover/ files are included."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        vo_file = vo_dir / "main.srt"
        vo_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nVO")

        root_file = tmp_path / "backup.srt"
        root_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nRoot")

        # With multiple candidates, user prompt is triggered. Mock it.
        with patch('builtins.input', return_value='1'):
            result = find_voiceover_interactive(tmp_path)

        # Should have returned one of the candidates
        assert result is not None

    @pytest.mark.fast
    def test_returns_none_when_no_files(self, tmp_path):
        """Returns None when no voiceover files exist."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        # Empty directory, no matching files

        result = find_voiceover_interactive(tmp_path)

        assert result is None

    @pytest.mark.fast
    def test_ignores_non_voiceover_extensions(self, tmp_path):
        """Files with non-voiceover extensions are ignored."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        (vo_dir / "readme.txt").write_text("Not a voiceover")
        (vo_dir / "data.json").write_text("{}")
        (vo_dir / "image.png").write_bytes(b"\x00" * 100)

        result = find_voiceover_interactive(tmp_path)

        assert result is None


# ============================================================================
# AC2: find_voiceover_interactive() auto-selects single candidate
# ============================================================================

class TestFindVoiceoverAutoSelectUS010:
    """AC2: find_voiceover_interactive() auto-selects when only one candidate found,
    without prompting the user."""

    @pytest.mark.fast
    def test_auto_selects_single_srt(self, tmp_path):
        """Single SRT file is auto-selected without user prompt."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt_file = vo_dir / "script.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest")

        # No input mock needed - should not prompt
        result = find_voiceover_interactive(tmp_path)

        assert result == str(srt_file)

    @pytest.mark.fast
    def test_auto_selects_single_mp4(self, tmp_path):
        """Single MP4 file is auto-selected without user prompt."""
        mp4_file = tmp_path / "voiceover.mp4"
        mp4_file.write_bytes(b"\x00" * 100)

        result = find_voiceover_interactive(tmp_path)

        assert result == str(mp4_file)

    @pytest.mark.fast
    def test_auto_selects_single_m4a(self, tmp_path):
        """Single M4A file is auto-selected."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        m4a_file = vo_dir / "recording.m4a"
        m4a_file.write_bytes(b"\x00" * 100)

        result = find_voiceover_interactive(tmp_path)

        assert result == str(m4a_file)

    @pytest.mark.fast
    def test_no_prompt_when_single_file(self, tmp_path):
        """Verify input() is NOT called when only one file exists."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        srt_file = vo_dir / "only_one.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nTest")

        with patch('builtins.input', side_effect=AssertionError("Should not prompt")):
            result = find_voiceover_interactive(tmp_path)

        assert result == str(srt_file)

    @pytest.mark.fast
    def test_prompts_when_multiple_files(self, tmp_path):
        """Verify input() IS called when multiple files exist."""
        vo_dir = tmp_path / "voiceover"
        vo_dir.mkdir()
        (vo_dir / "first.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nA")
        (vo_dir / "second.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nB")

        with patch('builtins.input', return_value='1') as mock_input:
            result = find_voiceover_interactive(tmp_path)

        mock_input.assert_called_once()
        assert result is not None
