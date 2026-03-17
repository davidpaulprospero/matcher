"""
Tests for src/transcription/silence_removal.py

Tests SilenceRemovalResult dataclass and silence removal return paths.
"""

import json
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src/ is importable and fix the utils shadow issue
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.transcription.silence_removal import (
    SilenceRemovalResult,
    remove_voiceover_silence,
)


class TestSilenceRemovalResult:
    """Test SilenceRemovalResult dataclass"""

    @pytest.mark.fast
    def test_result_with_trimming(self):
        result = SilenceRemovalResult(
            trimmed_path="/tmp/audio_trimmed.mp3",
            speech_regions_ms=[(0, 5000), (10000, 15000)],
            crossfade_ms=50,
            was_trimmed=True,
        )
        assert result.was_trimmed is True
        assert len(result.speech_regions_ms) == 2
        assert result.crossfade_ms == 50

    @pytest.mark.fast
    def test_result_without_trimming(self):
        result = SilenceRemovalResult(
            trimmed_path="/tmp/audio.mp3",
            speech_regions_ms=[],
            crossfade_ms=0,
            was_trimmed=False,
        )
        assert result.was_trimmed is False
        assert result.speech_regions_ms == []


class TestRemoveVoiceoverSilenceReturnPaths:
    """Test that all return paths produce correct SilenceRemovalResult."""

    @pytest.mark.fast
    def test_file_not_found_returns_no_op(self, tmp_path):
        result = remove_voiceover_silence(str(tmp_path / "nonexistent.mp3"))
        assert isinstance(result, SilenceRemovalResult)
        assert result.was_trimmed is False

    @pytest.mark.fast
    def test_already_trimmed_file_returns_no_op(self, tmp_path):
        audio = tmp_path / "test_trimmed.mp3"
        audio.write_text("fake")
        result = remove_voiceover_silence(str(audio))
        assert isinstance(result, SilenceRemovalResult)
        assert result.was_trimmed is False
        assert result.trimmed_path == str(audio)

    @pytest.mark.fast
    def test_cached_trimmed_with_sidecar(self, tmp_path):
        """Cached trimmed file with sidecar JSON should return regions."""
        audio = tmp_path / "test.mp3"
        audio.write_text("fake")

        trimmed = tmp_path / "test_trimmed.mp3"
        trimmed.write_text("trimmed fake")

        # Make trimmed newer than original
        os.utime(str(trimmed), (trimmed.stat().st_mtime + 10, trimmed.stat().st_mtime + 10))

        sidecar = tmp_path / "test_trimmed_regions.json"
        sidecar.write_text(json.dumps({
            "speech_regions_ms": [[0, 5000], [10000, 15000]],
            "crossfade_ms": 50,
        }))

        result = remove_voiceover_silence(str(audio))
        assert isinstance(result, SilenceRemovalResult)
        assert result.was_trimmed is True
        assert len(result.speech_regions_ms) == 2
        assert result.crossfade_ms == 50
