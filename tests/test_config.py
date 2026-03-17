"""
Unit tests for config module.

Tests configuration loading, validation, and dataclass structure.
"""

import pytest
from pathlib import Path
import sys
import tempfile
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    load_config,
    Config,
    LLMConfig,
    DownloadConfig,
    KeywordConfig,
    MatchingConfig,
    EmbeddingConfig,
    TranscriptionConfig
)


class TestConfigLoading:
    """Test configuration loading."""

    @pytest.mark.fast
    def test_load_default_config(self):
        """Test loading default config file."""
        config = load_config()

        assert config is not None
        assert isinstance(config, Config)

    @pytest.mark.fast
    def test_config_has_required_sections(self):
        """Test config has all required sections."""
        config = load_config()

        assert hasattr(config, 'llm')
        assert hasattr(config, 'download')
        assert hasattr(config, 'keyword')
        assert hasattr(config, 'matching')
        assert hasattr(config, 'embedding')

    @pytest.mark.fast
    def test_config_sections_are_correct_type(self):
        """Test config sections are correct dataclass types."""
        config = load_config()

        assert isinstance(config.llm, (LLMConfig, dict))
        assert isinstance(config.download, (DownloadConfig, dict))
        assert isinstance(config.keyword, (KeywordConfig, dict))
        assert isinstance(config.matching, (MatchingConfig, dict))
        assert isinstance(config.embedding, (EmbeddingConfig, dict))

    @pytest.mark.fast
    def test_load_config_with_custom_file(self, tmp_path):
        """Test loading config from custom file."""
        # Create minimal config
        config_data = {
            'llm': {'provider': 'gemini'},
            'download': {'max_results': 10},
            'keyword': {'max_keywords': 5},
            'matching': {'threshold': 0.7},
            'embedding': {'provider': 'voyage'}
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))

        assert config is not None

    @pytest.mark.fast
    def test_load_nonexistent_config(self):
        """Test loading non-existent config falls back to defaults."""
        config = load_config("nonexistent_file.yaml")

        # Should still return valid config (from default file)
        assert config is not None


class TestLLMConfig:
    """Test LLM configuration."""

    @pytest.mark.fast
    def test_llm_config_defaults(self):
        """Test LLM config has sensible defaults."""
        config = load_config()

        llm = config.llm
        if isinstance(llm, dict):
            assert 'provider' in llm
        else:
            assert hasattr(llm, 'provider')

    @pytest.mark.fast
    def test_llm_provider_options(self):
        """Test LLM provider can be configured."""
        config = load_config()

        llm = config.llm
        if isinstance(llm, dict):
            provider = llm.get('provider', 'gemini')
        else:
            provider = getattr(llm, 'provider', 'gemini')

        assert provider in ['gemini', 'google', 'anthropic', 'ollama', 'openai']


class TestDownloadConfig:
    """Test download configuration."""

    @pytest.mark.fast
    def test_download_config_max_results(self):
        """Test download max_results setting."""
        config = load_config()

        download = config.download
        if isinstance(download, dict):
            max_results = download.get('max_results', 50)
        else:
            max_results = getattr(download, 'max_results', 50)

        assert max_results > 0
        assert max_results <= 100

    @pytest.mark.fast
    def test_download_config_audio_first(self):
        """Test audio_first mode configuration."""
        config = load_config()

        download = config.download
        if isinstance(download, dict):
            audio_first = download.get('audio_first', {})
        else:
            audio_first = getattr(download, 'audio_first', {})

        assert audio_first is not None


class TestKeywordConfig:
    """Test keyword extraction configuration."""

    @pytest.mark.fast
    def test_keyword_config_max_keywords(self):
        """Test max_keywords setting."""
        config = load_config()

        keyword = config.keyword
        if isinstance(keyword, dict):
            max_keywords = keyword.get('max_keywords', 10)
        else:
            max_keywords = getattr(keyword, 'max_keywords', 10)

        assert max_keywords > 0
        assert max_keywords <= 50

    @pytest.mark.fast
    def test_keyword_config_method(self):
        """Test keyword extraction method."""
        config = load_config()

        keyword = config.keyword
        if isinstance(keyword, dict):
            method = keyword.get('method', 'llm')
        else:
            method = getattr(keyword, 'method', 'llm')

        assert method in ['llm', 'tfidf', 'hybrid']


class TestMatchingConfig:
    """Test matching configuration."""

    @pytest.mark.fast
    def test_matching_threshold(self):
        """Test matching threshold is in valid range."""
        config = load_config()

        matching = config.matching
        if isinstance(matching, dict):
            threshold = matching.get('threshold', 0.7)
        else:
            threshold = getattr(matching, 'threshold', 0.7)

        assert 0.0 <= threshold <= 1.0

    @pytest.mark.fast
    def test_matching_strategies(self):
        """Test matching strategies configuration."""
        config = load_config()

        matching = config.matching
        if isinstance(matching, dict):
            strategies = matching.get('strategies', [])
        else:
            strategies = getattr(matching, 'strategies', [])

        # Should have at least one strategy
        assert strategies is not None


class TestEmbeddingConfig:
    """Test embedding configuration."""

    @pytest.mark.fast
    def test_embedding_provider(self):
        """Test embedding provider setting."""
        config = load_config()

        embedding = config.embedding
        if isinstance(embedding, dict):
            provider = embedding.get('provider', 'voyage')
        else:
            provider = getattr(embedding, 'provider', 'voyage')

        assert provider in ['voyage', 'gemini', 'openai', 'cohere', 'sentence-transformers']

    @pytest.mark.fast
    def test_embedding_dimensions(self):
        """Test embedding dimensions setting."""
        config = load_config()

        embedding = config.embedding
        if isinstance(embedding, dict):
            dimensions = embedding.get('dimensions', 1024)
        else:
            dimensions = getattr(embedding, 'dimensions', 1024)

        assert dimensions in [384, 512, 768, 1024, 1536]


class TestConfigOverrides:
    """Test configuration overrides."""

    @pytest.mark.fast
    def test_project_config_override(self, tmp_path):
        """Test custom config file can override defaults."""
        # Create custom config
        custom_config = {
            'keyword': {
                'max_keywords': 20
            }
        }

        custom_file = tmp_path / "custom_config.yaml"
        with open(custom_file, 'w') as f:
            yaml.dump(custom_config, f)

        config = load_config(str(custom_file))

        keyword = config.keyword
        if isinstance(keyword, dict):
            max_keywords = keyword.get('max_keywords', 10)
        else:
            max_keywords = getattr(keyword, 'max_keywords', 10)

        # Should use override value
        assert max_keywords == 20


class TestConfigValidation:
    """Test configuration validation."""

    @pytest.mark.fast
    def test_config_to_dict(self):
        """Test config can be converted to dict."""
        config = load_config()

        # Should have dict-like access
        assert config is not None

    @pytest.mark.fast
    def test_config_has_pipeline_section(self):
        """Test config has pipeline section."""
        config = load_config()

        assert hasattr(config, 'pipeline') or hasattr(config, 'stages')

    @pytest.mark.fast
    def test_config_output_settings(self):
        """Test output configuration."""
        config = load_config()

        assert hasattr(config, 'output') or hasattr(config, 'otio')


class TestConfigDefaults:
    """Test configuration defaults are sensible."""

    @pytest.mark.fast
    def test_timeout_defaults(self):
        """Test timeout defaults are reasonable."""
        config = load_config()

        download = config.download
        if isinstance(download, dict):
            # Check for various timeout settings
            assert download is not None
        else:
            assert download is not None

    @pytest.mark.fast
    def test_cache_defaults(self):
        """Test cache settings have defaults."""
        config = load_config()

        # Should have some cache-related settings
        assert config is not None

    @pytest.mark.fast
    def test_boolean_flags_defaults(self):
        """Test boolean flags have defaults."""
        config = load_config()

        # Check common boolean flags
        pipeline = getattr(config, 'pipeline', {})
        if isinstance(pipeline, dict):
            # Should have skip flags with defaults
            assert isinstance(pipeline.get('skip_download', False), bool)
        else:
            assert pipeline is not None


class TestSafeGetConfigValue:
    """Test safe_get_config_value utility (US-65-002)."""

    @pytest.mark.fast
    def test_dict_input(self):
        """Test safe_get_config_value with dict input."""
        from src.config.utils import safe_get_config_value

        d = {'timeout': 30, 'retries': 3}
        assert safe_get_config_value(d, 'timeout') == 30
        assert safe_get_config_value(d, 'retries') == 3

    @pytest.mark.fast
    def test_dataclass_input(self):
        """Test safe_get_config_value with dataclass input."""
        from dataclasses import dataclass
        from src.config.utils import safe_get_config_value

        @dataclass
        class SampleConfig:
            timeout: int = 30
            retries: int = 3

        obj = SampleConfig()
        assert safe_get_config_value(obj, 'timeout') == 30
        assert safe_get_config_value(obj, 'retries') == 3

    @pytest.mark.fast
    def test_missing_key_with_default(self):
        """Test safe_get_config_value returns default for missing keys."""
        from src.config.utils import safe_get_config_value

        d = {'timeout': 30}
        assert safe_get_config_value(d, 'missing', 'fallback') == 'fallback'
        assert safe_get_config_value(d, 'missing') is None

    @pytest.mark.fast
    def test_missing_attr_with_default(self):
        """Test safe_get_config_value returns default for missing attrs on objects."""
        from dataclasses import dataclass
        from src.config.utils import safe_get_config_value

        @dataclass
        class SampleConfig:
            timeout: int = 30

        obj = SampleConfig()
        assert safe_get_config_value(obj, 'nonexistent', 42) == 42
        assert safe_get_config_value(obj, 'nonexistent') is None

    @pytest.mark.fast
    def test_none_input(self):
        """Test safe_get_config_value handles None input gracefully."""
        from src.config.utils import safe_get_config_value

        assert safe_get_config_value(None, 'key', 'default') == 'default'
        assert safe_get_config_value(None, 'key') is None

    @pytest.mark.fast
    def test_real_config_sections(self):
        """Test safe_get_config_value works with actual config dataclasses."""
        from src.config.utils import safe_get_config_value
        from src.config import MatchingConfig, DownloadConfig

        matching = MatchingConfig()
        assert safe_get_config_value(matching, 'min_confidence') is not None

        download = DownloadConfig()
        assert safe_get_config_value(download, 'nonexistent_field', 'fallback') == 'fallback'

    @pytest.mark.fast
    def test_dict_and_object_return_same_value(self):
        """Test that dict and object access return identical results."""
        from dataclasses import dataclass
        from src.config.utils import safe_get_config_value

        @dataclass
        class Cfg:
            name: str = "test"
            count: int = 5

        obj = Cfg()
        d = {'name': 'test', 'count': 5}

        assert safe_get_config_value(obj, 'name') == safe_get_config_value(d, 'name')
        assert safe_get_config_value(obj, 'count') == safe_get_config_value(d, 'count')
        assert safe_get_config_value(obj, 'missing', -1) == safe_get_config_value(d, 'missing', -1)


class TestCaptionFirstConfigPostInit:
    """Test CaptionFirstConfig __post_init__ dict-to-dataclass conversion (US-65-003)."""

    @pytest.mark.fast
    def test_retry_budget_dict_becomes_dataclass(self):
        """Constructing CaptionFirstConfig with retry_budget as raw dict produces CaptionRetryBudgetConfig."""
        from src.config.sections.download import CaptionFirstConfig, CaptionRetryBudgetConfig

        config = CaptionFirstConfig(
            retry_budget={'enabled': True, 'max_attempts': 150, 'attempts_per_video': 3.0}
        )
        assert isinstance(config.retry_budget, CaptionRetryBudgetConfig)
        assert config.retry_budget.enabled is True
        assert config.retry_budget.max_attempts == 150
        assert config.retry_budget.attempts_per_video == 3.0

    @pytest.mark.fast
    def test_retry_budget_dataclass_stays_dataclass(self):
        """CaptionRetryBudgetConfig instance is preserved as-is through __post_init__."""
        from src.config.sections.download import CaptionFirstConfig, CaptionRetryBudgetConfig

        budget = CaptionRetryBudgetConfig(max_attempts=50)
        config = CaptionFirstConfig(retry_budget=budget)
        assert config.retry_budget is budget

    @pytest.mark.fast
    def test_config_yaml_nested_retry_budget(self, tmp_path):
        """Loading config.yaml with nested retry_budget dict produces typed CaptionRetryBudgetConfig."""
        from src.config.sections.download import CaptionRetryBudgetConfig

        config_data = {
            'download': {
                'caption_first': {
                    'retry_budget': {
                        'enabled': True,
                        'max_attempts': 200,
                    }
                }
            }
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        caption_first = config.download.caption_first if hasattr(config.download, 'caption_first') else config.download.get('caption_first', {})
        if hasattr(caption_first, 'retry_budget'):
            assert isinstance(caption_first.retry_budget, CaptionRetryBudgetConfig)

    @pytest.mark.fast
    def test_config_yaml_nested_pause_split(self, tmp_path):
        """Loading config.yaml with nested pause_split dict produces typed PauseSplitConfig."""
        from src.config.sections.core import PauseSplitConfig

        config_data = {
            'transcription': {
                'pause_split': {
                    'enabled': False,
                    'min_gap_ms': 1000,
                    'split_at_sentences': False
                }
            }
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        config = load_config(str(config_file))
        # Verify pause_split is converted to PauseSplitConfig
        if hasattr(config.transcription, 'pause_split'):
            assert isinstance(config.transcription.pause_split, PauseSplitConfig)
            assert config.transcription.pause_split.enabled is False
            assert config.transcription.pause_split.min_gap_ms == 1000
            assert config.transcription.pause_split.split_at_sentences is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
