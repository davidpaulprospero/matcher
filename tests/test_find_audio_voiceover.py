"""Tests for _find_audio_for_voiceover size-based selection (US-56-009)."""

import os
import tempfile
from pathlib import Path

import pytest

from src.otio.timeline import _find_audio_for_voiceover


class TestFindAudioForVoiceover:
    """Tests for _find_audio_for_voiceover function."""

    def test_single_candidate_returned_without_size_comparison(self, tmp_path):
        """Single audio file is returned directly (current behavior preserved)."""
        srt_file = tmp_path / "script.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        audio_file = tmp_path / "script.mp3"
        audio_file.write_bytes(b"\x00" * 1000)

        result = _find_audio_for_voiceover(str(srt_file))
        assert result == str(audio_file)

    def test_multiple_candidates_returns_largest(self, tmp_path):
        """When 3 audio files exist, the largest by size is returned."""
        srt_file = tmp_path / "script.srt"
        srt_file.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

        # Create 3 audio files of different sizes
        small = tmp_path / "audio.mp3"
        small.write_bytes(b"\x00" * 1_000)        # 1 KB - click track

        medium = tmp_path / "vo.wav"
        medium.write_bytes(b"\x00" * 100_000)      # 100 KB - scratch mix

        large = tmp_path / "voiceover.mp3"
        large.write_bytes(b"\x00" * 10_000_000)    # 10 MB - main voiceover

        result = _find_audio_for_voiceover(str(srt_file))
        assert result == str(large), (
            f"Expected largest file {large}, got {result}"
        )

    def test_same_stem_largest_wins(self, tmp_path):
        """Same-stem candidates with different extensions: largest wins."""
        srt_file = tmp_path / "voiceover.srt"
        srt_file.write_text("subtitle content")

        # Same stem, different extensions and sizes
        mp3 = tmp_path / "voiceover.mp3"
        mp3.write_bytes(b"\x00" * 5_000)

        wav = tmp_path / "voiceover.wav"
        wav.write_bytes(b"\x00" * 50_000)  # WAV is larger (uncompressed)

        result = _find_audio_for_voiceover(str(srt_file))
        assert result == str(wav)

    def test_no_candidates_returns_none(self, tmp_path):
        """No audio files in directory returns None."""
        srt_file = tmp_path / "script.srt"
        srt_file.write_text("subtitle content")

        result = _find_audio_for_voiceover(str(srt_file))
        assert result is None

    def test_pattern_match_largest_selected(self, tmp_path):
        """Pattern-matched files: largest wins regardless of pattern order."""
        srt_file = tmp_path / "script.srt"
        srt_file.write_text("subtitle content")

        # 'combined_output' pattern comes first in search order but is smaller
        combined = tmp_path / "combined_output.mp3"
        combined.write_bytes(b"\x00" * 500)

        # 'voiceover' pattern is second but is larger
        voiceover = tmp_path / "voiceover.mp3"
        voiceover.write_bytes(b"\x00" * 50_000)

        # 'audio' pattern is third
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00" * 5_000)

        result = _find_audio_for_voiceover(str(srt_file))
        assert result == str(voiceover), (
            f"Expected largest pattern match {voiceover}, got {result}"
        )

    def test_logs_selection_when_multiple_candidates(self, tmp_path, caplog):
        """Log message includes candidate count and selected file info."""
        import logging

        srt_file = tmp_path / "script.srt"
        srt_file.write_text("subtitle content")

        small = tmp_path / "audio.mp3"
        small.write_bytes(b"\x00" * 1_000)

        large = tmp_path / "voiceover.mp3"
        large.write_bytes(b"\x00" * 2_000_000)

        with caplog.at_level(logging.INFO):
            result = _find_audio_for_voiceover(str(srt_file))

        assert "Found 2 audio candidates" in caplog.text
        assert "voiceover.mp3" in caplog.text
        assert result == str(large)

    def test_no_duplicate_candidates(self, tmp_path):
        """A file matching both stem and pattern isn't counted twice."""
        # 'voiceover.srt' with stem 'voiceover' - voiceover.mp3 matches
        # both the stem check AND the 'voiceover' pattern check
        srt_file = tmp_path / "voiceover.srt"
        srt_file.write_text("subtitle content")

        audio = tmp_path / "voiceover.mp3"
        audio.write_bytes(b"\x00" * 5_000)

        result = _find_audio_for_voiceover(str(srt_file))
        # Should return the file (single candidate, no size log)
        assert result == str(audio)
