"""
Tests for __post_init__ dict-to-dataclass conversions in config sections.

Targets:
- src/config/sections/llm.py: Lines 96, 100, 102, 104, 106, 108 (dict conversions)
- src/config/sections/download.py: Lines 281, 283, 285, 287 (dict conversions)
"""

import pytest
import os
from unittest.mock import patch

from src.config.sections.llm import (
    LLMConfig,
    LLMRetryConfig,
    LLMCacheConfig,
    LLMProviderConfig,
)

from src.config.sections.download import (
    DownloadConfig,
    LLMTitleFilterConfig,
    CaptionFirstConfig,
    AudioFirstConfig,
    ZeroDownloadRemixConfig,
    SpeechScreeningConfig,
)


class TestLLMConfigPostInit:
    """Test LLMConfig __post_init__ dict-to-dataclass conversions."""

    def test_llm_config_api_key_from_env_google(self):
        """Test API key loaded from env for Google provider (line 93-94)."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_gemini_key'}):
            config = LLMConfig(provider='google', api_key='')
            assert config.api_key == 'test_gemini_key'

    def test_llm_config_api_key_from_env_anthropic(self):
        """Test API key loaded from env for Anthropic provider (line 95-96)."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_anthropic_key'}):
            config = LLMConfig(provider='anthropic', api_key='')
            assert config.api_key == 'test_anthropic_key'

    def test_llm_config_retry_as_dict(self):
        """Test retry config converted from dict (lines 99-100)."""
        config = LLMConfig(
            retry={'max_retries': 5, 'retry_delay_seconds': 3.0}
        )
        assert isinstance(config.retry, LLMRetryConfig)
        assert config.retry.max_retries == 5
        assert config.retry.retry_delay_seconds == 3.0

    def test_llm_config_cache_as_dict(self):
        """Test cache config converted from dict (lines 101-102)."""
        config = LLMConfig(
            cache={'enabled': False, 'ttl_hours': 48}
        )
        assert isinstance(config.cache, LLMCacheConfig)
        assert config.cache.enabled == False
        assert config.cache.ttl_hours == 48

    def test_llm_config_gemini_as_dict(self):
        """Test gemini provider config converted from dict (lines 103-104)."""
        config = LLMConfig(
            gemini={'model': 'gemini-pro', 'max_tokens': 4000}
        )
        assert isinstance(config.gemini, LLMProviderConfig)
        assert config.gemini.model == 'gemini-pro'
        assert config.gemini.max_tokens == 4000

    def test_llm_config_anthropic_as_dict(self):
        """Test anthropic provider config converted from dict (lines 105-106)."""
        config = LLMConfig(
            anthropic={'model': 'claude-3-sonnet', 'temperature': 0.5}
        )
        assert isinstance(config.anthropic, LLMProviderConfig)
        assert config.anthropic.model == 'claude-3-sonnet'
        assert config.anthropic.temperature == 0.5

    def test_llm_config_ollama_as_dict(self):
        """Test ollama provider config converted from dict (lines 107-108)."""
        config = LLMConfig(
            ollama={'model': 'codellama', 'max_tokens': 1000}
        )
        assert isinstance(config.ollama, LLMProviderConfig)
        assert config.ollama.model == 'codellama'
        assert config.ollama.max_tokens == 1000

    def test_llm_config_all_nested_as_dicts(self):
        """Test all nested configs converted from dicts at once."""
        config = LLMConfig(
            retry={'max_retries': 10},
            cache={'enabled': True, 'ttl_hours': 12},
            gemini={'model': 'gemini-flash'},
            anthropic={'model': 'claude-3-opus'},
            ollama={'model': 'mistral'}
        )
        assert isinstance(config.retry, LLMRetryConfig)
        assert isinstance(config.cache, LLMCacheConfig)
        assert isinstance(config.gemini, LLMProviderConfig)
        assert isinstance(config.anthropic, LLMProviderConfig)
        assert isinstance(config.ollama, LLMProviderConfig)

    def test_llm_config_already_dataclass(self):
        """Test nested configs already as dataclass instances."""
        retry = LLMRetryConfig(max_retries=7)
        cache = LLMCacheConfig(enabled=False)
        gemini = LLMProviderConfig(model='gemini-pro')

        config = LLMConfig(retry=retry, cache=cache, gemini=gemini)

        assert config.retry is retry
        assert config.cache is cache
        assert config.gemini is gemini


class TestDownloadConfigPostInit:
    """Test DownloadConfig __post_init__ dict-to-dataclass conversions."""

    def test_download_config_llm_title_filter_as_dict(self):
        """Test llm_title_filter converted from dict (lines 280-281)."""
        config = DownloadConfig(
            llm_title_filter={'enabled': False, 'min_relevance': 0.8}
        )
        assert isinstance(config.llm_title_filter, LLMTitleFilterConfig)
        assert config.llm_title_filter.enabled == False
        assert config.llm_title_filter.min_relevance == 0.8

    def test_download_config_audio_first_as_dict(self):
        """Test audio_first converted from dict (lines 282-283)."""
        config = DownloadConfig(
            audio_first={'enabled': True, 'buffer_seconds': 45.0}
        )
        assert isinstance(config.audio_first, AudioFirstConfig)
        assert config.audio_first.enabled == True
        assert config.audio_first.buffer_seconds == 45.0

    def test_download_config_caption_first_as_dict(self):
        """Test caption_first converted from dict."""
        config = DownloadConfig(
            caption_first={
                'enabled': True,
                'fallback_to_transcription': False,
                'preferred_language': 'es',
                'timeout': 60
            }
        )
        assert isinstance(config.caption_first, CaptionFirstConfig)
        assert config.caption_first.enabled == True
        assert config.caption_first.fallback_to_transcription == False
        assert config.caption_first.preferred_language == 'es'
        assert config.caption_first.timeout == 60

    def test_download_config_zero_download_remix_as_dict(self):
        """Test zero_download_remix converted from dict (lines 284-285)."""
        config = DownloadConfig(
            zero_download_remix={'enabled': False, 'max_retries': 5}
        )
        assert isinstance(config.zero_download_remix, ZeroDownloadRemixConfig)
        assert config.zero_download_remix.enabled == False
        assert config.zero_download_remix.max_retries == 5

    def test_download_config_speech_screening_as_dict(self):
        """Test speech_screening converted from dict (lines 286-287)."""
        config = DownloadConfig(
            speech_screening={'enabled': True, 'screening_duration': 10.0}
        )
        assert isinstance(config.speech_screening, SpeechScreeningConfig)
        assert config.speech_screening.enabled == True
        assert config.speech_screening.screening_duration == 10.0

    def test_download_config_all_nested_as_dicts(self):
        """Test all nested configs converted from dicts at once."""
        config = DownloadConfig(
            llm_title_filter={'enabled': True},
            audio_first={'enabled': True},
            caption_first={'enabled': True},
            zero_download_remix={'enabled': True},
            speech_screening={'enabled': True}
        )
        assert isinstance(config.llm_title_filter, LLMTitleFilterConfig)
        assert isinstance(config.audio_first, AudioFirstConfig)
        assert isinstance(config.caption_first, CaptionFirstConfig)
        assert isinstance(config.zero_download_remix, ZeroDownloadRemixConfig)
        assert isinstance(config.speech_screening, SpeechScreeningConfig)

    def test_download_config_already_dataclass(self):
        """Test nested configs already as dataclass instances."""
        llm_filter = LLMTitleFilterConfig(enabled=False)
        audio_first = AudioFirstConfig(enabled=True)
        caption_first = CaptionFirstConfig(enabled=True)
        zero_remix = ZeroDownloadRemixConfig(enabled=False)
        speech = SpeechScreeningConfig(enabled=True)

        config = DownloadConfig(
            llm_title_filter=llm_filter,
            audio_first=audio_first,
            caption_first=caption_first,
            zero_download_remix=zero_remix,
            speech_screening=speech
        )

        assert config.llm_title_filter is llm_filter
        assert config.audio_first is audio_first
        assert config.caption_first is caption_first
        assert config.zero_download_remix is zero_remix
        assert config.speech_screening is speech


class TestLLMProviderConfigDefaults:
    """Test LLMProviderConfig default values."""

    def test_provider_config_defaults(self):
        """Test default values for provider config."""
        config = LLMProviderConfig(model='test-model')
        assert config.model == 'test-model'
        assert config.max_tokens == 2000
        assert config.temperature == 0.7

    def test_provider_config_custom_values(self):
        """Test custom values for provider config."""
        config = LLMProviderConfig(
            model='custom-model',
            max_tokens=5000,
            temperature=0.3
        )
        assert config.model == 'custom-model'
        assert config.max_tokens == 5000
        assert config.temperature == 0.3


class TestLLMRetryConfigDefaults:
    """Test LLMRetryConfig default values."""

    def test_retry_config_defaults(self):
        """Test default values for retry config."""
        config = LLMRetryConfig()
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 2.0
        assert config.timeout_seconds == 120
        assert config.exponential_backoff == True


class TestLLMCacheConfigDefaults:
    """Test LLMCacheConfig default values."""

    def test_cache_config_defaults(self):
        """Test default values for cache config."""
        config = LLMCacheConfig()
        assert config.enabled == True
        assert config.ttl_hours == 24
        assert config.cache_dir == ".cache/llm_responses"


class TestAudioFirstConfigDefaults:
    """Test AudioFirstConfig default values."""

    def test_audio_first_config_defaults(self):
        """Test default values for audio first config."""
        config = AudioFirstConfig()
        assert config.enabled == False
        assert config.buffer_seconds == 30.0
        assert config.merge_gap_seconds == 15.0
        assert config.audio_quality == 5
        assert config.fallback_full_video == True

    def test_audio_first_config_custom(self):
        """Test custom values for audio first config."""
        config = AudioFirstConfig(
            enabled=True,
            buffer_seconds=60.0,
            audio_quality=3
        )
        assert config.enabled == True
        assert config.buffer_seconds == 60.0
        assert config.audio_quality == 3


class TestSpeechScreeningConfigDefaults:
    """Test SpeechScreeningConfig default values."""

    def test_speech_screening_config_defaults(self):
        """Test default values for speech screening config."""
        config = SpeechScreeningConfig()
        assert config.enabled == False
        assert config.screening_duration == 5.0
        assert config.min_speech_duration == 0.5
        assert config.reject_with_speech == True
        assert config.whisper_model == "base"
        assert "long" in config.tiers
        assert "longer" in config.tiers


class TestZeroDownloadRemixConfigDefaults:
    """Test ZeroDownloadRemixConfig default values."""

    def test_zero_download_remix_defaults(self):
        """Test default values for zero download remix config."""
        config = ZeroDownloadRemixConfig()
        assert config.enabled == True
        assert config.max_retries == 2
        assert config.use_llm == True
        assert config.use_fallback == True
        assert config.cache_results == True


class TestLLMTitleFilterConfigDefaults:
    """Test LLMTitleFilterConfig default values."""

    def test_title_filter_defaults(self):
        """Test default values for LLM title filter config."""
        config = LLMTitleFilterConfig()
        assert config.enabled == True
        assert config.provider == "gemini"
        assert config.model == "gemini-2.0-flash"
        assert config.batch_size == 20
        assert config.min_relevance == 0.7


class TestCaptionFirstConfigDefaults:
    """Test CaptionFirstConfig default values."""

    def test_caption_first_defaults(self):
        """Test default values for caption first config."""
        config = CaptionFirstConfig()
        assert config.enabled == False
        assert config.fallback_to_transcription == True
        assert config.preferred_language == "en"
        assert config.timeout == 30
        assert config.prefer_human_captions == True
        assert config.cache_captions == True

    def test_caption_first_custom_values(self):
        """Test custom values for caption first config."""
        config = CaptionFirstConfig(
            enabled=True,
            fallback_to_transcription=False,
            preferred_language="es",
            timeout=60,
            prefer_human_captions=False,
            cache_captions=False
        )
        assert config.enabled == True
        assert config.fallback_to_transcription == False
        assert config.preferred_language == "es"
        assert config.timeout == 60
        assert config.prefer_human_captions == False
        assert config.cache_captions == False
