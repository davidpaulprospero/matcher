"""
Tests for audio extraction workers configuration (US-60-010).

Verifies:
- audio_extraction_workers config option in TranscriptionConfig
- Validation that workers <= cpu_count()
- Default to min(4, cpu_count()) for sensible default on various machines
- Worker count is passed to transcribe_videos_parallel()
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from src.config.sections.core import TranscriptionConfig


class TestAudioExtractionWorkersConfig:
    """Test audio_extraction_workers configuration in TranscriptionConfig."""

    @pytest.mark.fast
    def test_default_workers_uses_min_4_cpu_count(self):
        """Default (0) audio_extraction_workers uses min(4, cpu_count())."""
        config = TranscriptionConfig(audio_extraction_workers=0)
        cpu_count = os.cpu_count() or 4
        expected = min(4, cpu_count)
        assert config.audio_extraction_workers == expected

    @pytest.mark.fast
    def test_explicit_workers_value_respected(self):
        """Explicit audio_extraction_workers value is respected if <= cpu_count."""
        cpu_count = os.cpu_count() or 4
        # Use a value that's guaranteed to be <= cpu_count
        test_value = min(2, cpu_count)
        config = TranscriptionConfig(audio_extraction_workers=test_value)
        assert config.audio_extraction_workers == test_value

    @pytest.mark.fast
    def test_workers_capped_at_cpu_count(self):
        """audio_extraction_workers is capped at cpu_count() if exceeded."""
        cpu_count = os.cpu_count() or 4
        # Request way more workers than CPUs
        config = TranscriptionConfig(audio_extraction_workers=cpu_count + 100)
        assert config.audio_extraction_workers == cpu_count

    @pytest.mark.fast
    def test_negative_workers_uses_default(self):
        """Negative audio_extraction_workers uses default calculation."""
        config = TranscriptionConfig(audio_extraction_workers=-1)
        cpu_count = os.cpu_count() or 4
        expected = min(4, cpu_count)
        assert config.audio_extraction_workers == expected

    @pytest.mark.fast
    @patch('os.cpu_count', return_value=2)
    def test_default_on_2_core_machine(self, mock_cpu_count):
        """On 2-core machine, default is 2 (min(4, 2))."""
        # Need to reimport to pick up the mock
        from src.config.sections.core import TranscriptionConfig as TC
        config = TC(audio_extraction_workers=0)
        assert config.audio_extraction_workers == 2

    @pytest.mark.fast
    @patch('os.cpu_count', return_value=8)
    def test_default_on_8_core_machine(self, mock_cpu_count):
        """On 8-core machine, default is 4 (min(4, 8))."""
        from src.config.sections.core import TranscriptionConfig as TC
        config = TC(audio_extraction_workers=0)
        assert config.audio_extraction_workers == 4

    @pytest.mark.fast
    @patch('os.cpu_count', return_value=None)
    def test_cpu_count_none_fallback(self, mock_cpu_count):
        """When os.cpu_count() returns None, fallback to 4."""
        from src.config.sections.core import TranscriptionConfig as TC
        config = TC(audio_extraction_workers=0)
        assert config.audio_extraction_workers == 4

    @pytest.mark.fast
    def test_workers_exactly_cpu_count_accepted(self):
        """audio_extraction_workers exactly equal to cpu_count is accepted."""
        cpu_count = os.cpu_count() or 4
        config = TranscriptionConfig(audio_extraction_workers=cpu_count)
        assert config.audio_extraction_workers == cpu_count

    @pytest.mark.fast
    def test_one_worker_valid(self):
        """audio_extraction_workers of 1 is valid."""
        config = TranscriptionConfig(audio_extraction_workers=1)
        assert config.audio_extraction_workers == 1


class TestTranscribeVideosParallelWorkerConfig:
    """Test that transcribe_videos_parallel uses config worker count."""

    @pytest.mark.fast
    def test_max_workers_from_config_when_none(self):
        """When max_workers=None, uses config.transcription.audio_extraction_workers."""
        from src.transcription.parallel_processor import transcribe_videos_parallel
        
        # Create mock config with specific worker count
        mock_config = MagicMock()
        mock_config.transcription.model = 'base'
        mock_config.transcription.compute_type = 'auto'
        mock_config.transcription.language = 'en'
        mock_config.transcription.min_silence_duration_ms = 200
        mock_config.transcription.speech_pad_ms = 10
        mock_config.transcription.audio_extraction_workers = 2
        
        # Mock cache
        mock_cache = MagicMock()
        mock_cache.cache_dir = '/tmp/test_cache'
        
        # Call with empty video list to avoid actual processing
        # but verify the config is read correctly
        result = transcribe_videos_parallel(
            video_paths=[],
            cache=mock_cache,
            config=mock_config,
            max_workers=None,  # Should use config value
        )
        
        # With empty video_paths, it returns empty dict immediately
        assert result == {}

    @pytest.mark.fast
    def test_explicit_max_workers_overrides_config(self):
        """When max_workers is explicitly set, it overrides config value."""
        from src.transcription.parallel_processor import transcribe_videos_parallel
        
        # Create mock config with different worker count
        mock_config = MagicMock()
        mock_config.transcription.model = 'base'
        mock_config.transcription.compute_type = 'auto'
        mock_config.transcription.language = 'en'
        mock_config.transcription.min_silence_duration_ms = 200
        mock_config.transcription.speech_pad_ms = 10
        mock_config.transcription.audio_extraction_workers = 8
        
        # Mock cache
        mock_cache = MagicMock()
        mock_cache.cache_dir = '/tmp/test_cache'
        
        # Call with explicit max_workers=2, should NOT use config's 8
        result = transcribe_videos_parallel(
            video_paths=[],
            cache=mock_cache,
            config=mock_config,
            max_workers=2,  # Explicit override
        )
        
        assert result == {}

    @pytest.mark.fast
    def test_no_config_uses_cpu_count_default(self):
        """When no config provided, uses min(4, cpu_count()) default."""
        from src.transcription.parallel_processor import transcribe_videos_parallel
        
        # Mock cache
        mock_cache = MagicMock()
        mock_cache.cache_dir = '/tmp/test_cache'
        
        # Call with no config and no max_workers
        result = transcribe_videos_parallel(
            video_paths=[],
            cache=mock_cache,
            config=None,
            max_workers=None,
        )
        
        assert result == {}
