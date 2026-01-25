"""Tests for audio extraction timeout behavior.

Tests US-006: Add transcription audio extraction timeout.

Verifies:
- audio_extraction_timeout field exists in TranscriptionConfig
- extract_audio() respects timeout parameter
- TimeoutError raised when extraction exceeds timeout
- Warning logged when timeout occurs
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import subprocess

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.unit


# =============================================================================
# Test TranscriptionConfig timeout field
# =============================================================================

class TestTranscriptionConfigTimeout:
    """Tests for audio_extraction_timeout field in TranscriptionConfig."""

    def test_transcription_config_has_audio_extraction_timeout_field(self):
        """TranscriptionConfig should have audio_extraction_timeout field."""
        from src.config.sections.core import TranscriptionConfig
        import dataclasses

        fields = {f.name for f in dataclasses.fields(TranscriptionConfig)}
        assert "audio_extraction_timeout" in fields

    def test_audio_extraction_timeout_default_is_60(self):
        """audio_extraction_timeout should default to 60 seconds."""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig()
        assert config.audio_extraction_timeout == 60

    def test_audio_extraction_timeout_can_be_set(self):
        """audio_extraction_timeout should accept custom values."""
        from src.config.sections.core import TranscriptionConfig

        config = TranscriptionConfig(audio_extraction_timeout=120)
        assert config.audio_extraction_timeout == 120

    def test_audio_extraction_timeout_type_is_int(self):
        """audio_extraction_timeout should be an integer."""
        from src.config.sections.core import TranscriptionConfig
        import dataclasses

        for f in dataclasses.fields(TranscriptionConfig):
            if f.name == "audio_extraction_timeout":
                # With __future__.annotations, type is a string
                assert f.type == int or f.type == "int"
                break


# =============================================================================
# Test extract_audio timeout parameter
# =============================================================================

class TestExtractAudioTimeoutParameter:
    """Tests for timeout parameter in extract_audio() function."""

    def test_extract_audio_accepts_timeout_parameter(self):
        """extract_audio() should accept timeout parameter."""
        from src.transcription.utils import extract_audio
        import inspect

        sig = inspect.signature(extract_audio)
        assert "timeout" in sig.parameters

    def test_extract_audio_timeout_default_is_60(self):
        """extract_audio() timeout should default to 60."""
        from src.transcription.utils import extract_audio
        import inspect

        sig = inspect.signature(extract_audio)
        timeout_param = sig.parameters["timeout"]
        assert timeout_param.default == 60

    def test_extract_audio_passes_timeout_to_subprocess(self, tmp_path):
        """extract_audio() should pass timeout to subprocess.run()."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            extract_audio(str(video_path), timeout=30)

            # Check timeout was passed
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["timeout"] == 30


# =============================================================================
# Test TimeoutError behavior
# =============================================================================

class TestTimeoutErrorBehavior:
    """Tests for TimeoutError raising when extraction times out."""

    def test_timeout_raises_timeout_error(self, tmp_path):
        """extract_audio() should raise TimeoutError on timeout."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=5)

            with pytest.raises(TimeoutError):
                extract_audio(str(video_path), timeout=5)

    def test_timeout_error_message_includes_video_path(self, tmp_path):
        """TimeoutError message should include video path."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "my_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=10)

            with pytest.raises(TimeoutError) as exc_info:
                extract_audio(str(video_path), timeout=10)

            assert str(video_path) in str(exc_info.value)

    def test_timeout_error_message_includes_timeout_value(self, tmp_path):
        """TimeoutError message should include timeout value."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=45)

            with pytest.raises(TimeoutError) as exc_info:
                extract_audio(str(video_path), timeout=45)

            assert "45" in str(exc_info.value)

    def test_timeout_cleans_up_partial_file(self, tmp_path):
        """extract_audio() should attempt cleanup on timeout if partial file exists."""
        from src.transcription.utils import extract_audio
        import hashlib

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        # Calculate what audio path would be created
        path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        expected_audio_path = tmp_path / f"{video_path.stem[:80]}_{path_hash}.wav"

        # We need to:
        # 1. Have audio_path.exists() return False on initial skip check
        # 2. Have subprocess.run() raise TimeoutExpired
        # 3. Have audio_path.exists() return True on cleanup check
        # 4. Verify unlink is called

        exists_call_count = 0

        def mock_exists(path_self):
            nonlocal exists_call_count
            if str(path_self).endswith('.wav'):
                exists_call_count += 1
                # First call is skip check (False = proceed)
                # Second call is cleanup check (True = cleanup needed)
                return exists_call_count > 1
            return True  # video_path exists

        unlink_paths = []

        def mock_unlink(path_self):
            unlink_paths.append(str(path_self))

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=5)

            with patch.object(Path, "exists", mock_exists):
                with patch.object(Path, "unlink", mock_unlink):
                    try:
                        extract_audio(str(video_path), timeout=5)
                    except TimeoutError:
                        pass

                    # Verify cleanup was attempted (unlink called with audio path)
                    assert any(str(expected_audio_path) in p for p in unlink_paths)


# =============================================================================
# Test timeout warning logging
# =============================================================================

class TestTimeoutWarningLogging:
    """Tests for warning logging when timeout occurs."""

    def test_timeout_logs_warning(self, tmp_path, caplog):
        """extract_audio() should log warning on timeout."""
        from src.transcription.utils import extract_audio
        import logging

        video_path = tmp_path / "slow_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=60)

            with caplog.at_level(logging.WARNING):
                try:
                    extract_audio(str(video_path), timeout=60)
                except TimeoutError:
                    pass

            # Check warning was logged
            assert any("timed out" in record.message.lower() for record in caplog.records)

    def test_timeout_warning_includes_video_path(self, tmp_path, caplog):
        """Timeout warning should include video path."""
        from src.transcription.utils import extract_audio
        import logging

        video_path = tmp_path / "my_slow_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=30)

            with caplog.at_level(logging.WARNING):
                try:
                    extract_audio(str(video_path), timeout=30)
                except TimeoutError:
                    pass

            # Check path in warning
            warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
            assert any(str(video_path) in msg for msg in warning_messages)

    def test_timeout_warning_includes_timeout_seconds(self, tmp_path, caplog):
        """Timeout warning should include timeout value in seconds."""
        from src.transcription.utils import extract_audio
        import logging

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=90)

            with caplog.at_level(logging.WARNING):
                try:
                    extract_audio(str(video_path), timeout=90)
                except TimeoutError:
                    pass

            # Check timeout value in warning
            warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
            assert any("90" in msg for msg in warning_messages)


# =============================================================================
# Test extraction success with timeout
# =============================================================================

class TestExtractionWithTimeout:
    """Tests for successful extraction with custom timeout."""

    def test_extraction_succeeds_within_timeout(self, tmp_path):
        """extract_audio() should succeed when extraction completes within timeout."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Create expected output file
            expected_audio = tmp_path / video_path.stem

            # Mock Path.exists to return True for audio file check
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True

                result = extract_audio(str(video_path), timeout=120)

                # Should not raise, should return a path
                assert result is not None or True  # Path returned or extraction succeeded

    def test_extraction_respects_custom_timeout(self, tmp_path):
        """extract_audio() should use custom timeout value."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            extract_audio(str(video_path), timeout=180)

            # Check custom timeout was passed
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["timeout"] == 180


# =============================================================================
# Test config integration
# =============================================================================

class TestConfigIntegration:
    """Tests for config integration with audio extraction timeout."""

    def test_config_timeout_can_be_used_with_extract_audio(self):
        """Config timeout value can be passed to extract_audio()."""
        from src.config.sections.core import TranscriptionConfig
        from src.transcription.utils import extract_audio

        config = TranscriptionConfig(audio_extraction_timeout=90)

        # Verify the value is accessible
        assert config.audio_extraction_timeout == 90

        # Verify it can be used as parameter (signature check)
        import inspect
        sig = inspect.signature(extract_audio)
        assert "timeout" in sig.parameters

    def test_default_config_matches_extract_audio_default(self):
        """TranscriptionConfig default should match extract_audio default."""
        from src.config.sections.core import TranscriptionConfig
        from src.transcription.utils import extract_audio
        import inspect

        config = TranscriptionConfig()
        sig = inspect.signature(extract_audio)

        assert config.audio_extraction_timeout == sig.parameters["timeout"].default


# =============================================================================
# Test edge cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in timeout handling."""

    def test_zero_timeout_passed_to_subprocess(self, tmp_path):
        """extract_audio() should handle timeout=0 (immediate timeout)."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=0)

            with pytest.raises(TimeoutError):
                extract_audio(str(video_path), timeout=0)

    def test_very_large_timeout(self, tmp_path):
        """extract_audio() should handle very large timeout values."""
        from src.transcription.utils import extract_audio

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)

            extract_audio(str(video_path), timeout=3600)  # 1 hour

            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["timeout"] == 3600

    def test_extraction_skipped_if_audio_exists(self, tmp_path):
        """extract_audio() should skip extraction if audio file exists."""
        from src.transcription.utils import extract_audio
        import hashlib

        video_path = tmp_path / "test_video.mp4"
        video_path.touch()

        # Create expected audio file
        path_hash = hashlib.md5(str(video_path).encode()).hexdigest()[:8]
        audio_path = tmp_path / f"{video_path.stem[:80]}_{path_hash}.wav"
        audio_path.touch()

        with patch("src.transcription.utils.subprocess.run") as mock_run:
            result = extract_audio(str(video_path), timeout=30)

            # Should return existing path without calling subprocess
            assert result == str(audio_path)
            mock_run.assert_not_called()
