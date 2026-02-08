"""
Tests for config section validation.

US-009 Sprint 23: Config section validation tests

Covers:
- AC1: InfrastructureConfig __post_init__ nested dict conversion
- AC2: MatchingConfig validates strategy enum values
- AC3: MediaConfig validates search provider list
- AC4: LLMConfig validates provider selection with available keys
- AC5: All config sections handle missing optional fields with defaults
"""

import pytest
import os
from unittest.mock import patch

from src.config.base import Config, ConfigError, FrozenConfigError

# Infrastructure config imports
from src.config.sections.infrastructure import (
    HealingConfig,
    HealingLoggingConfig,
    WatcherConfig,
    LLMHealerConfig,
    LoggingConfig,
    CacheConfig,
    GlobalCacheConfig,
    PipelineConfig,
    APIKeysConfig,
)

# Matching config imports
from src.config.sections.matching import (
    MatchingConfig,
    LocationMatchingConfig,
    ChapterDetectionConfig,
    NegativeMatchingConfig,
)

# Media config imports
from src.config.sections.media import (
    VisionConfig,
    SceneDetectionConfig,
    AudioAnalysisConfig,
)

# LLM config imports
from src.config.sections.llm import (
    LLMConfig,
    LLMRetryConfig,
    LLMCacheConfig,
    LLMProviderConfig,
)

# Download config imports
from src.config.sections.download import (
    DownloadConfig,
    EnhancedFeaturesConfig,
    LLMTitleFilterConfig,
    RemixConfig,
    ZeroDownloadRemixConfig,
)

# Base config for validation
from src.config.base import Config


# =============================================================================
# AC1: Test InfrastructureConfig __post_init__ converts nested dicts to dataclasses
# =============================================================================

@pytest.mark.fast
class TestHealingConfigPostInit:
    """Test HealingConfig __post_init__ dict-to-dataclass conversions."""

    def test_logging_dict_converted(self):
        """Test logging dict is converted to HealingLoggingConfig."""
        config = HealingConfig(
            logging={'enabled': False, 'log_dir': '/custom/logs', 'json_log': False}
        )

        assert isinstance(config.logging, HealingLoggingConfig)
        assert config.logging.enabled is False
        assert config.logging.log_dir == '/custom/logs'
        assert config.logging.json_log is False

    @pytest.mark.fast
    def test_watcher_dict_converted(self):
        """Test watcher dict is converted to WatcherConfig."""
        config = HealingConfig(
            watcher={
                'enabled': True,
                'provider': 'anthropic',
                'model': 'claude-3-haiku',
                'timeout': 60.0
            }
        )

        assert isinstance(config.watcher, WatcherConfig)
        assert config.watcher.enabled is True
        assert config.watcher.provider == 'anthropic'
        assert config.watcher.model == 'claude-3-haiku'
        assert config.watcher.timeout == 60.0

    @pytest.mark.fast
    def test_llm_healer_dict_converted(self):
        """Test llm_healer dict is converted to LLMHealerConfig."""
        config = HealingConfig(
            llm_healer={
                'enabled': True,
                'provider': 'gemini',
                'model': 'gemini-pro',
                'max_tokens': 8192,
                'timeout': 120.0
            }
        )

        assert isinstance(config.llm_healer, LLMHealerConfig)
        assert config.llm_healer.enabled is True
        assert config.llm_healer.provider == 'gemini'
        assert config.llm_healer.model == 'gemini-pro'
        assert config.llm_healer.max_tokens == 8192
        assert config.llm_healer.timeout == 120.0

    @pytest.mark.fast
    def test_all_nested_dicts_converted(self):
        """Test all nested configs converted from dicts at once."""
        config = HealingConfig(
            enabled=True,
            strategy='aggressive',
            logging={'enabled': True, 'console_format': 'simple'},
            watcher={'enabled': True, 'escalate_threshold': 0.5},
            llm_healer={'enabled': True, 'max_retries': 5}
        )

        assert isinstance(config.logging, HealingLoggingConfig)
        assert isinstance(config.watcher, WatcherConfig)
        assert isinstance(config.llm_healer, LLMHealerConfig)
        assert config.logging.console_format == 'simple'
        assert config.watcher.escalate_threshold == 0.5
        assert config.llm_healer.max_retries == 5

    @pytest.mark.fast
    def test_dataclass_objects_unchanged(self):
        """Test that dataclass instances are not modified."""
        logging_cfg = HealingLoggingConfig(enabled=False)
        watcher_cfg = WatcherConfig(model='llama3.1')
        healer_cfg = LLMHealerConfig(max_tokens=2048)

        config = HealingConfig(
            logging=logging_cfg,
            watcher=watcher_cfg,
            llm_healer=healer_cfg
        )

        assert config.logging is logging_cfg
        assert config.watcher is watcher_cfg
        assert config.llm_healer is healer_cfg

    @pytest.mark.fast
    def test_partial_dict_conversion(self):
        """Test conversion with mix of dict and dataclass instances."""
        logging_cfg = HealingLoggingConfig(json_log=False)

        config = HealingConfig(
            logging=logging_cfg,
            watcher={'provider': 'ollama'},
            llm_healer={'provider': 'anthropic'}
        )

        assert config.logging is logging_cfg
        assert isinstance(config.watcher, WatcherConfig)
        assert isinstance(config.llm_healer, LLMHealerConfig)


# =============================================================================
# AC2: Test MatchingConfig validates strategy enum values
# =============================================================================

@pytest.mark.fast
class TestMatchingConfigStrategyValidation:
    """Test MatchingConfig validates strategy/enum values."""

    def test_valid_primary_provider(self):
        """Test valid primary_provider values accepted."""
        for provider in ['gemini', 'anthropic', 'ollama']:
            config = MatchingConfig(primary_provider=provider)
            assert config.primary_provider == provider

    @pytest.mark.fast
    def test_valid_secondary_provider(self):
        """Test valid secondary_provider values accepted."""
        for provider in ['gemini', 'anthropic', 'ollama']:
            config = MatchingConfig(secondary_provider=provider)
            assert config.secondary_provider == provider

    @pytest.mark.fast
    def test_valid_local_provider(self):
        """Test valid local_provider values accepted."""
        config = MatchingConfig(local_provider='ollama')
        assert config.local_provider == 'ollama'

    @pytest.mark.fast
    def test_location_matching_hard_filter_level_valid(self):
        """Test valid hard_filter_level values in LocationMatchingConfig."""
        for level in ['city', 'state', 'country', 'continent']:
            config = LocationMatchingConfig(hard_filter_level=level)
            assert config.hard_filter_level == level

    @pytest.mark.fast
    def test_invalid_hard_filter_level_detected(self, tmp_path):
        """Test invalid hard_filter_level is detected during validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
matching:
  location_matching:
    enabled: true
    hard_filter_level: invalid_level
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_enums()

        assert any("hard_filter_level" in e for e in errors)

    @pytest.mark.fast
    def test_chapter_detection_strategy_valid(self):
        """Test valid default_strategy values in ChapterDetectionConfig."""
        for strategy in ['topic', 'location']:
            config = ChapterDetectionConfig(default_strategy=strategy)
            assert config.default_strategy == strategy

    @pytest.mark.fast
    def test_matching_config_strategy_field(self):
        """Test MatchingConfig healing strategy-related fields."""
        # Note: MatchingConfig doesn't have a 'strategy' field, but HealingConfig does
        healing_config = HealingConfig(strategy='conservative')
        assert healing_config.strategy == 'conservative'

        for strategy in ['aggressive', 'conservative', 'interactive', 'minimal']:
            config = HealingConfig(strategy=strategy)
            assert config.strategy == strategy


@pytest.mark.fast
class TestMatchingConfigLocationMatchingPostInit:
    """Test MatchingConfig __post_init__ for location_matching."""

    def test_location_matching_none_creates_default(self):
        """Test location_matching=None creates default LocationMatchingConfig."""
        config = MatchingConfig(location_matching=None)

        assert isinstance(config.location_matching, LocationMatchingConfig)
        assert config.location_matching.enabled is True

    @pytest.mark.fast
    def test_location_matching_dict_converted(self):
        """Test location_matching dict is converted to dataclass."""
        config = MatchingConfig(
            location_matching={
                'enabled': True,
                'hard_filter_level': 'country',
                'geographic_penalty': 0.3
            }
        )

        assert isinstance(config.location_matching, LocationMatchingConfig)
        assert config.location_matching.hard_filter_level == 'country'
        assert config.location_matching.geographic_penalty == 0.3

    @pytest.mark.fast
    def test_chapter_detection_none_creates_default(self):
        """Test chapter_detection=None creates default ChapterDetectionConfig."""
        config = MatchingConfig(chapter_detection=None)

        assert isinstance(config.chapter_detection, ChapterDetectionConfig)
        assert config.chapter_detection.enabled is True

    @pytest.mark.fast
    def test_chapter_detection_dict_converted(self):
        """Test chapter_detection dict is converted to dataclass."""
        config = MatchingConfig(
            chapter_detection={
                'enabled': True,
                'max_chapters': 30,
                'min_chapter_confidence': 0.6
            }
        )

        assert isinstance(config.chapter_detection, ChapterDetectionConfig)
        assert config.chapter_detection.max_chapters == 30
        assert config.chapter_detection.min_chapter_confidence == 0.6


# =============================================================================
# AC3: Test MediaConfig validates search provider list
# =============================================================================

@pytest.mark.fast
class TestMediaConfigProviderValidation:
    """Test Media config validates provider lists and values."""

    def test_vision_provider_valid(self):
        """Test valid vision provider values."""
        for provider in ['gemini', 'openai']:
            config = VisionConfig(provider=provider)
            assert config.provider == provider

    @pytest.mark.fast
    def test_scene_detection_preset_valid(self):
        """Test valid scene detection preset values."""
        for preset in ['fast', 'balanced', 'accurate']:
            config = SceneDetectionConfig(preset=preset)
            assert config.preset == preset

    @pytest.mark.fast
    def test_invalid_embedding_provider_detected(self, tmp_path):
        """Test invalid embedding provider is detected during validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
embedding:
  provider: invalid_provider
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_enums()

        assert any("embedding.provider" in e for e in errors)

    @pytest.mark.fast
    def test_valid_embedding_providers(self, tmp_path):
        """Test valid embedding provider values."""
        # Valid providers as defined in src/config/base.py line 639
        for provider in ['gemini', 'openai', 'local', 'sentence_transformers']:
            config_file = tmp_path / f"config_{provider}.yaml"
            config_file.write_text(f"""
embedding:
  provider: {provider}
""")
            config = Config.from_yaml(str(config_file))
            errors = config._validate_enums()

            provider_errors = [e for e in errors if "embedding.provider" in e]
            assert len(provider_errors) == 0, f"Provider {provider} should be valid"


@pytest.mark.fast
class TestSceneDetectionConfigValidation:
    """Test SceneDetectionConfig validation."""

    def test_gpu_settings(self):
        """Test GPU acceleration settings."""
        config = SceneDetectionConfig(use_gpu=True, force_gpu=False)
        assert config.use_gpu is True
        assert config.force_gpu is False

    @pytest.mark.fast
    def test_threshold_range(self):
        """Test scene detection threshold is in valid range."""
        config = SceneDetectionConfig(threshold=27.0)
        assert 0 < config.threshold < 100

    @pytest.mark.fast
    def test_min_scene_len_positive(self):
        """Test min_scene_len must be positive."""
        config = SceneDetectionConfig(min_scene_len=15)
        assert config.min_scene_len > 0


# =============================================================================
# AC4: Test LLMConfig validates provider selection with available keys
# =============================================================================

@pytest.mark.fast
class TestLLMConfigProviderValidation:
    """Test LLMConfig validates provider selection with available keys."""

    @pytest.fixture(autouse=True)
    def clear_env_keys(self, monkeypatch):
        """Clear API key environment variables."""
        for key in ['GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'VOYAGE_API_KEY']:
            monkeypatch.delenv(key, raising=False)

    @pytest.mark.requires_api
    def test_google_provider_loads_gemini_key(self):
        """Test google provider loads GEMINI_API_KEY from env."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_gemini_key'}):
            config = LLMConfig(provider='google', api_key='')
            assert config.api_key == 'test_gemini_key'

    @pytest.mark.requires_api
    def test_anthropic_provider_loads_anthropic_key(self):
        """Test anthropic provider loads ANTHROPIC_API_KEY from env."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_anthropic_key'}):
            config = LLMConfig(provider='anthropic', api_key='')
            assert config.api_key == 'test_anthropic_key'

    @pytest.mark.requires_api
    def test_explicit_key_not_overwritten(self):
        """Test explicitly set api_key is not overwritten."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'env_key'}):
            config = LLMConfig(provider='google', api_key='explicit_key')
            assert config.api_key == 'explicit_key'

    @pytest.mark.fast
    def test_invalid_matching_provider_detected(self, tmp_path):
        """Test invalid matching provider is detected during validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
matching:
  primary_provider: invalid_provider
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_enums()

        assert any("matching.primary_provider" in e for e in errors)

    @pytest.mark.requires_api
    def test_gemini_key_required_when_gemini_provider(self, tmp_path):
        """Test GEMINI_API_KEY required when using gemini provider."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
matching:
  primary_provider: gemini
api_keys:
  gemini_api_key: ""
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("GEMINI_API_KEY required" in e for e in errors)

    @pytest.mark.requires_api
    def test_anthropic_key_required_when_anthropic_secondary(self, tmp_path):
        """Test ANTHROPIC_API_KEY required when using anthropic as secondary."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
matching:
  secondary_provider: anthropic
api_keys:
  anthropic_api_key: ""
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        assert any("ANTHROPIC_API_KEY required" in e for e in errors)

    @pytest.mark.requires_api
    def test_no_key_error_when_key_present(self, tmp_path):
        """Test no key error when API key is present."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
matching:
  primary_provider: gemini
api_keys:
  gemini_api_key: "test_key_123"
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()

        gemini_errors = [e for e in errors if "GEMINI_API_KEY required" in e]
        assert len(gemini_errors) == 0


@pytest.mark.fast
class TestLLMConfigPostInit:
    """Test LLMConfig __post_init__ conversions."""

    def test_retry_dict_converted(self):
        """Test retry dict is converted to LLMRetryConfig."""
        config = LLMConfig(
            retry={'max_retries': 5, 'retry_delay_seconds': 5.0}
        )

        assert isinstance(config.retry, LLMRetryConfig)
        assert config.retry.max_retries == 5
        assert config.retry.retry_delay_seconds == 5.0

    @pytest.mark.fast
    def test_cache_dict_converted(self):
        """Test cache dict is converted to LLMCacheConfig."""
        config = LLMConfig(
            cache={'enabled': False, 'ttl_hours': 48}
        )

        assert isinstance(config.cache, LLMCacheConfig)
        assert config.cache.enabled is False
        assert config.cache.ttl_hours == 48

    @pytest.mark.fast
    def test_provider_configs_dict_converted(self):
        """Test provider configs (gemini, anthropic, ollama) dicts are converted."""
        config = LLMConfig(
            gemini={'model': 'gemini-pro', 'max_tokens': 4096},
            anthropic={'model': 'claude-3-opus', 'temperature': 0.3},
            ollama={'model': 'mistral'}
        )

        assert isinstance(config.gemini, LLMProviderConfig)
        assert isinstance(config.anthropic, LLMProviderConfig)
        assert isinstance(config.ollama, LLMProviderConfig)

        assert config.gemini.model == 'gemini-pro'
        assert config.gemini.max_tokens == 4096
        assert config.anthropic.model == 'claude-3-opus'
        assert config.anthropic.temperature == 0.3
        assert config.ollama.model == 'mistral'


# =============================================================================
# AC5: Test all config sections handle missing optional fields with defaults
# =============================================================================

@pytest.mark.fast
class TestLoggingConfigDefaults:
    """Test LoggingConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test LoggingConfig default values."""
        config = LoggingConfig()

        assert config.enabled is True
        assert config.log_dir == "logs"
        assert config.log_level == "INFO"
        assert config.log_to_file is True
        assert config.log_to_console is True
        assert config.generate_json_log is True
        assert config.log_api_calls is True
        assert config.track_api_costs is True
        assert config.log_config_access is False
        assert config.warn_on_hardcoded is True

    @pytest.mark.fast
    def test_partial_override(self):
        """Test partial override preserves other defaults."""
        config = LoggingConfig(log_level="DEBUG", log_to_console=False)

        assert config.log_level == "DEBUG"
        assert config.log_to_console is False
        assert config.enabled is True  # Default preserved
        assert config.log_to_file is True  # Default preserved


@pytest.mark.fast
class TestCacheConfigDefaults:
    """Test CacheConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test CacheConfig default values."""
        config = CacheConfig()

        assert config.cache_dir == ".cache"
        assert config.cache_transcriptions is True
        assert config.cache_embeddings is True
        assert config.cache_scenes is True
        assert config.cache_llm_responses is True
        assert config.cache_vision is True
        assert config.cross_project_cache is False
        assert config.validate_cache_on_load is True


@pytest.mark.fast
class TestGlobalCacheConfigDefaults:
    """Test GlobalCacheConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test GlobalCacheConfig default values."""
        config = GlobalCacheConfig()

        assert config.enabled is True
        assert config.cache_dir == "~/.matcher_global_cache"
        assert config.check_before_download is True
        assert config.min_keyword_similarity == 0.8
        assert config.min_topic_overlap == 0.3
        assert config.max_reuse_videos == 50
        assert config.share_transcripts is True


@pytest.mark.fast
class TestPipelineConfigDefaults:
    """Test PipelineConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test PipelineConfig default values."""
        config = PipelineConfig()

        assert config.skip_download is False
        assert config.skip_image_search is False
        assert config.skip_transcription is False
        assert config.skip_scene_detection is False
        assert config.skip_matching is False
        assert config.resume_enabled is True
        assert config.max_retries == 3
        assert config.parallel_transcription is True


@pytest.mark.fast
class TestAPIKeysConfigDefaults:
    """Test APIKeysConfig handles missing optional fields with defaults."""

    @pytest.mark.requires_api
    def test_default_empty_keys(self, monkeypatch):
        """Test APIKeysConfig starts with empty keys (not from env)."""
        # Clear all env variables
        for key in ['GEMINI_API_KEY', 'ANTHROPIC_API_KEY', 'VOYAGE_API_KEY',
                    'PEXELS_API_KEY', 'PIXABAY_API_KEY', 'UNSPLASH_API_KEY']:
            monkeypatch.delenv(key, raising=False)

        config = APIKeysConfig()

        # All should be empty since env vars are cleared
        assert config.gemini_api_key == ""
        assert config.anthropic_api_key == ""
        assert config.voyage_api_key == ""
        assert config.pexels_api_key == ""
        assert config.pixabay_api_key == ""
        assert config.unsplash_api_key == ""

    @pytest.mark.requires_api
    def test_loads_from_environment(self):
        """Test APIKeysConfig loads keys from environment."""
        with patch.dict(os.environ, {
            'GEMINI_API_KEY': 'gemini_test',
            'ANTHROPIC_API_KEY': 'anthropic_test'
        }):
            config = APIKeysConfig()

            assert config.gemini_api_key == 'gemini_test'
            assert config.anthropic_api_key == 'anthropic_test'


@pytest.mark.fast
class TestHealingConfigDefaults:
    """Test HealingConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test HealingConfig default values."""
        config = HealingConfig()

        assert config.enabled is True
        assert config.strategy == "conservative"
        assert config.max_attempts_per_stage == 3
        assert config.max_total_heals == 20
        assert config.heal_delay == 2.0
        assert config.run_preflight is True
        assert config.auto_fix_preflight is True
        assert config.enable_rollback is True
        assert config.print_report is True

    @pytest.mark.fast
    def test_nested_configs_have_defaults(self):
        """Test nested configs are created with defaults."""
        config = HealingConfig()

        assert isinstance(config.logging, HealingLoggingConfig)
        assert isinstance(config.watcher, WatcherConfig)
        assert isinstance(config.llm_healer, LLMHealerConfig)


@pytest.mark.fast
class TestWatcherConfigDefaults:
    """Test WatcherConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test WatcherConfig default values."""
        config = WatcherConfig()

        assert config.enabled is True
        assert config.provider == "ollama"
        assert config.model == "llama3.2"
        assert config.fallback_model == "llama3.1"
        assert config.host == "http://localhost:11434"
        assert config.timeout == 30.0
        assert config.escalate_threshold == 0.7
        assert config.max_failures == 3


@pytest.mark.fast
class TestLLMHealerConfigDefaults:
    """Test LLMHealerConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test LLMHealerConfig default values."""
        config = LLMHealerConfig()

        assert config.enabled is True
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.max_tokens == 4096
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.include_stack_trace is True
        assert config.include_config_context is True


@pytest.mark.fast
class TestMatchingConfigDefaults:
    """Test MatchingConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test MatchingConfig default values."""
        config = MatchingConfig()

        assert config.min_confidence == 0.7
        assert config.high_confidence_threshold == 0.85
        assert config.max_clip_reuse == 1
        assert config.embedding_candidates == 50
        assert config.llm_rerank_candidates == 5
        assert config.primary_provider == "gemini"
        assert config.secondary_provider == "anthropic"
        assert config.cache_llm_responses is True
        assert config.delta_matching_enabled is True

    @pytest.mark.fast
    def test_nested_configs_created(self):
        """Test nested configs are created by default."""
        config = MatchingConfig()

        assert isinstance(config.location_matching, LocationMatchingConfig)
        assert isinstance(config.chapter_detection, ChapterDetectionConfig)


@pytest.mark.fast
class TestVisionConfigDefaults:
    """Test VisionConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test VisionConfig default values."""
        config = VisionConfig()

        assert config.provider == "gemini"
        assert config.model == "gemini-2.0-flash"
        assert config.enabled is True
        assert config.min_words_per_scene == 5
        assert config.coverage_threshold == 0.3
        assert config.max_scenes_per_video == 50
        assert config.frame_format == "jpg"
        assert config.frame_quality == 85


@pytest.mark.fast
class TestSceneDetectionConfigDefaults:
    """Test SceneDetectionConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test SceneDetectionConfig default values."""
        config = SceneDetectionConfig()

        assert config.enabled is True
        assert config.preset == "balanced"
        assert config.threshold == 27.0
        assert config.min_scene_len == 15
        assert config.downscale_factor == 4
        assert config.use_gpu is True
        assert config.detect_faces_per_scene is True


@pytest.mark.fast
class TestAudioAnalysisConfigDefaults:
    """Test AudioAnalysisConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test AudioAnalysisConfig default values."""
        config = AudioAnalysisConfig()

        assert config.enabled is True
        assert config.sample_rate == 22050
        assert config.silence_threshold_db == -40.0
        assert config.min_silence_duration == 0.3
        assert config.speech_threshold == 0.5


@pytest.mark.fast
class TestLLMConfigDefaults:
    """Test LLMConfig handles missing optional fields with defaults."""

    @pytest.mark.requires_api
    def test_default_values(self, monkeypatch):
        """Test LLMConfig default values."""
        # Clear env to avoid loading API keys
        monkeypatch.delenv('GEMINI_API_KEY', raising=False)
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)

        config = LLMConfig()

        assert config.provider == "google"
        assert config.model == "gemini-2.0-flash"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.ollama_host == "http://localhost:11434"

    @pytest.mark.fast
    def test_nested_configs_created(self):
        """Test nested configs are created with defaults."""
        config = LLMConfig()

        assert isinstance(config.retry, LLMRetryConfig)
        assert isinstance(config.cache, LLMCacheConfig)
        assert isinstance(config.gemini, LLMProviderConfig)
        assert isinstance(config.anthropic, LLMProviderConfig)
        assert isinstance(config.ollama, LLMProviderConfig)


@pytest.mark.fast
class TestNegativeMatchingConfigDefaults:
    """Test NegativeMatchingConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test NegativeMatchingConfig default values."""
        config = NegativeMatchingConfig()

        assert config.enabled is True
        assert isinstance(config.rules, list)
        assert len(config.rules) == 3


@pytest.mark.fast
class TestChapterDetectionConfigDefaults:
    """Test ChapterDetectionConfig handles missing optional fields with defaults."""

    def test_default_values(self):
        """Test ChapterDetectionConfig default values."""
        config = ChapterDetectionConfig()

        assert config.enabled is True
        assert config.use_validation_pass is True
        assert config.use_boundary_refinement is True
        assert config.default_strategy == 'topic'
        assert config.auto_detect_content_type is True
        assert config.max_chunk_chars == 6000
        assert config.min_chapter_confidence == 0.5
        assert config.max_chapters == 20


@pytest.mark.fast
class TestLocationMatchingConfigDefaults:
    """Test LocationMatchingConfig handles missing optional fields with defaults."""

    def test_default_values(self, monkeypatch):
        """Test LocationMatchingConfig default values."""
        monkeypatch.delenv('GEONAMES_USERNAME', raising=False)

        config = LocationMatchingConfig()

        assert config.enabled is True
        assert config.hard_filter_level == "city"
        assert config.geographic_penalty == 0.4
        assert config.hierarchy_bonus == 0.15
        assert config.landmark_bonus == 0.2
        assert config.use_llm_disambiguation is True

    @pytest.mark.fast
    def test_loads_geonames_from_env(self):
        """Test geonames_username loaded from environment."""
        with patch.dict(os.environ, {'GEONAMES_USERNAME': 'test_user'}):
            config = LocationMatchingConfig()
            assert config.geonames_username == 'test_user'


# =============================================================================
# Parametrized Tests for Enum Values (US-006 Sprint 24)
# =============================================================================

@pytest.mark.fast
class TestMatchingConfigProviderParametrized:
    """Parametrized tests for MatchingConfig provider enum values."""

    @pytest.mark.parametrize("provider", [
        "gemini",
        "anthropic",
        "local",
        "embedding_only",
    ])
    @pytest.mark.fast
    def test_valid_primary_providers(self, provider):
        """Test all valid primary_provider values are accepted."""
        config = MatchingConfig(primary_provider=provider)
        assert config.primary_provider == provider

    @pytest.mark.parametrize("provider", [
        "gemini",
        "anthropic",
        "ollama",
    ])
    @pytest.mark.fast
    def test_valid_secondary_providers(self, provider):
        """Test all valid secondary_provider values are accepted."""
        config = MatchingConfig(secondary_provider=provider)
        assert config.secondary_provider == provider

    @pytest.mark.parametrize("invalid_provider", [
        "invalid",
        "openai",
        "gpt4",
        "",
        "GEMINI",  # Case-sensitive
    ])
    @pytest.mark.fast
    def test_invalid_primary_provider_detected(self, invalid_provider, tmp_path):
        """Test invalid primary_provider values are detected during validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
matching:
  primary_provider: {invalid_provider if invalid_provider else '""'}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_enums()
        assert any("primary_provider" in e for e in errors)


@pytest.mark.fast
class TestLLMConfigProviderParametrized:
    """Parametrized tests for LLMConfig provider enum values."""

    @pytest.mark.parametrize("provider,expected_env_key", [
        ("google", "GEMINI_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
    ])
    @pytest.mark.fast
    def test_provider_loads_correct_env_key(self, provider, expected_env_key):
        """Test each provider loads the correct environment variable."""
        test_key = f"test_{provider}_key_123"
        with patch.dict(os.environ, {expected_env_key: test_key}):
            config = LLMConfig(provider=provider, api_key='')
            assert config.api_key == test_key

    @pytest.mark.parametrize("provider", [
        "google",
        "anthropic",
        "ollama",
    ])
    @pytest.mark.fast
    def test_valid_llm_providers(self, provider):
        """Test all valid LLMConfig.provider values are accepted."""
        config = LLMConfig(provider=provider)
        assert config.provider == provider

    @pytest.mark.parametrize("provider,model_attr,expected_default", [
        ("google", "gemini", "gemini-2.0-flash"),
        ("anthropic", "anthropic", "claude-3-haiku-20240307"),
        ("ollama", "ollama", "llama3.2"),
    ])
    @pytest.mark.fast
    def test_provider_default_models(self, provider, model_attr, expected_default):
        """Test each provider has correct default model configuration."""
        config = LLMConfig(provider=provider)
        provider_config = getattr(config, model_attr)
        assert provider_config.model == expected_default


@pytest.mark.fast
class TestEmbeddingProviderParametrized:
    """Parametrized tests for embedding provider enum values."""

    @pytest.mark.parametrize("provider", [
        "gemini",
        "openai",
        "local",
        "sentence_transformers",
        "ollama",
    ])
    @pytest.mark.fast
    def test_valid_embedding_providers(self, provider, tmp_path):
        """Test all valid embedding.provider values are accepted."""
        config_file = tmp_path / f"config_{provider}.yaml"
        config_file.write_text(f"""
embedding:
  provider: {provider}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_enums()
        provider_errors = [e for e in errors if "embedding.provider" in e]
        assert len(provider_errors) == 0, f"Provider {provider} should be valid"

    @pytest.mark.parametrize("invalid_provider", [
        "cohere",
        "huggingface",
        "invalid",
        "",
    ])
    @pytest.mark.fast
    def test_invalid_embedding_provider_detected(self, invalid_provider, tmp_path):
        """Test invalid embedding.provider values are detected."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
embedding:
  provider: {invalid_provider if invalid_provider else '""'}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_enums()
        assert any("embedding.provider" in e for e in errors)


@pytest.mark.fast
class TestHealingStrategyParametrized:
    """Parametrized tests for HealingConfig strategy enum values."""

    @pytest.mark.parametrize("strategy", [
        "aggressive",
        "conservative",
        "interactive",
        "minimal",
    ])
    @pytest.mark.fast
    def test_valid_healing_strategies(self, strategy):
        """Test all valid HealingConfig.strategy values are accepted."""
        config = HealingConfig(strategy=strategy)
        assert config.strategy == strategy


@pytest.mark.fast
class TestBoundaryValuesParametrized:
    """Parametrized tests for config boundary values (min/max thresholds)."""

    @pytest.mark.parametrize("min_conf,high_conf,should_pass", [
        (0.5, 0.85, True),   # min < high - valid
        (0.7, 0.85, True),   # min < high - valid (default values)
        (0.85, 0.85, True),  # min == high - edge case, valid
        (0.9, 0.85, False),  # min > high - invalid
        (1.0, 0.85, False),  # min way higher - invalid
        (0.0, 0.1, True),    # very low thresholds - valid
    ])
    @pytest.mark.fast
    def test_confidence_threshold_constraints(self, min_conf, high_conf, should_pass, tmp_path):
        """Test min_confidence <= high_confidence_threshold constraint."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
matching:
  min_confidence: {min_conf}
  high_confidence_threshold: {high_conf}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_constraints()
        constraint_errors = [e for e in errors if "min_confidence" in e and "high_confidence_threshold" in e]

        if should_pass:
            assert len(constraint_errors) == 0
        else:
            assert len(constraint_errors) > 0

    @pytest.mark.parametrize("embedding_candidates,num_alternatives,should_pass", [
        (50, 2, True),   # 50 >= 2*3=6 - valid (default)
        (10, 2, True),   # 10 >= 6 - valid
        (6, 2, True),    # exactly 6 = 2*3 - valid edge case
        (5, 2, False),   # 5 < 6 - invalid
        (100, 5, True),  # 100 >= 15 - valid
        (14, 5, False),  # 14 < 15 - invalid
        (3, 1, True),    # 3 >= 3 - valid edge case
        (2, 1, False),   # 2 < 3 - invalid
    ])
    @pytest.mark.fast
    def test_embedding_candidates_constraint(self, embedding_candidates, num_alternatives, should_pass, tmp_path):
        """Test embedding_candidates >= num_alternatives * 3 constraint."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
matching:
  embedding_candidates: {embedding_candidates}
output:
  num_alternatives: {num_alternatives}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_constraints()
        candidate_errors = [e for e in errors if "embedding_candidates" in e]

        if should_pass:
            assert len(candidate_errors) == 0
        else:
            assert len(candidate_errors) > 0

    @pytest.mark.parametrize("threshold,expected_valid", [
        (0.5, True),     # Normal value
        (1.0, True),     # Max threshold
        (27.0, True),    # Default value
    ])
    @pytest.mark.fast
    def test_scene_detection_threshold_bounds(self, threshold, expected_valid):
        """Test SceneDetectionConfig threshold boundary values."""
        config = SceneDetectionConfig(threshold=threshold)
        # Threshold should be > 0
        is_valid = config.threshold > 0
        assert is_valid == expected_valid

    @pytest.mark.parametrize("temperature", [
        0.0,    # Minimum - deterministic
        0.5,    # Normal
        0.7,    # Default
        1.0,    # Maximum standard
        2.0,    # Extended range (some providers support)
    ])
    @pytest.mark.fast
    def test_llm_temperature_bounds(self, temperature):
        """Test LLMConfig temperature accepts valid range values."""
        config = LLMConfig(temperature=temperature)
        assert config.temperature == temperature

    @pytest.mark.parametrize("min_scene_len", [1, 15, 60])
    @pytest.mark.fast
    def test_min_scene_len_bounds_valid(self, min_scene_len):
        """Test SceneDetectionConfig min_scene_len accepts valid values."""
        config = SceneDetectionConfig(min_scene_len=min_scene_len)
        assert config.min_scene_len > 0

    @pytest.mark.parametrize("min_scene_len", [0, -1])
    @pytest.mark.fast
    def test_min_scene_len_bounds_invalid(self, min_scene_len):
        """Test SceneDetectionConfig min_scene_len rejects invalid values."""
        with pytest.raises(ValueError, match="min_scene_len"):
            SceneDetectionConfig(min_scene_len=min_scene_len)


@pytest.mark.fast
class TestLocationMatchingLevelParametrized:
    """Parametrized tests for LocationMatchingConfig hard_filter_level enum values."""

    @pytest.mark.parametrize("level", [
        "city",
        "state",
        "country",
        "continent",
    ])
    @pytest.mark.fast
    def test_valid_hard_filter_levels(self, level):
        """Test all valid hard_filter_level values are accepted."""
        config = LocationMatchingConfig(hard_filter_level=level)
        assert config.hard_filter_level == level

    @pytest.mark.parametrize("invalid_level", [
        "region",
        "neighborhood",
        "invalid",
        "",
    ])
    @pytest.mark.fast
    def test_invalid_hard_filter_level_detected(self, invalid_level, tmp_path):
        """Test invalid hard_filter_level values are detected during validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
matching:
  location_matching:
    enabled: true
    hard_filter_level: {invalid_level if invalid_level else '""'}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_enums()
        assert any("hard_filter_level" in e for e in errors)


@pytest.mark.fast
class TestChapterDetectionStrategyParametrized:
    """Parametrized tests for ChapterDetectionConfig strategy enum values."""

    @pytest.mark.parametrize("strategy", [
        "topic",
        "location",
    ])
    @pytest.mark.fast
    def test_valid_chapter_strategies(self, strategy):
        """Test all valid default_strategy values are accepted."""
        config = ChapterDetectionConfig(default_strategy=strategy)
        assert config.default_strategy == strategy


@pytest.mark.fast
class TestSceneDetectionPresetParametrized:
    """Parametrized tests for SceneDetectionConfig preset enum values."""

    @pytest.mark.parametrize("preset", [
        "fast",
        "balanced",
        "accurate",
    ])
    @pytest.mark.fast
    def test_valid_scene_detection_presets(self, preset):
        """Test all valid preset values are accepted."""
        config = SceneDetectionConfig(preset=preset)
        assert config.preset == preset


@pytest.mark.fast
class TestVisionProviderParametrized:
    """Parametrized tests for VisionConfig provider enum values."""

    @pytest.mark.parametrize("provider", [
        "gemini",
        "openai",
    ])
    @pytest.mark.fast
    def test_valid_vision_providers(self, provider):
        """Test all valid VisionConfig.provider values are accepted."""
        config = VisionConfig(provider=provider)
        assert config.provider == provider


# =============================================================================
# US-50-002: DownloadConfig cookies_from_browser validation
# =============================================================================

@pytest.mark.fast
class TestDownloadConfigCookiesFromBrowserValidation:
    """Test DownloadConfig validates cookies_from_browser browser names."""

    @pytest.mark.parametrize("browser", [
        "firefox",
        "chrome",
        "edge",
        "safari",
        "opera",
        "brave",
        "",  # Empty string = disabled
    ])
    def test_valid_browser_names_accepted(self, browser):
        """Test all valid cookies_from_browser values are accepted."""
        config = DownloadConfig(cookies_from_browser=browser)
        assert config.cookies_from_browser == browser

    @pytest.mark.parametrize("invalid_browser", [
        "netscape",
        "vivaldi",
        "ie",
        "Internet Explorer",
        "FIREFOX",  # Case-sensitive
        "Chrome",   # Case-sensitive
        "invalid",
    ])
    def test_invalid_browser_names_raise_valueerror(self, invalid_browser):
        """Test invalid cookies_from_browser values raise ValueError."""
        with pytest.raises(ValueError, match="cookies_from_browser"):
            DownloadConfig(cookies_from_browser=invalid_browser)

    def test_cookies_from_browser_loads_from_dict(self):
        """Test cookies_from_browser loads correctly when DownloadConfig is created from dict values."""
        config = DownloadConfig(
            cookies_from_browser="chrome",
            cookies_path="/path/to/cookies.txt",
        )
        assert config.cookies_from_browser == "chrome"
        assert config.cookies_path == "/path/to/cookies.txt"

    def test_default_cookies_from_browser_is_empty(self):
        """Test default cookies_from_browser is empty string (disabled)."""
        config = DownloadConfig()
        assert config.cookies_from_browser == ""

    def test_default_cookies_path_is_empty(self):
        """Test default cookies_path is empty string."""
        config = DownloadConfig()
        assert config.cookies_path == ""


@pytest.mark.fast
class TestDownloadConfigCookieYamlRoundTrip:
    """Test cookie fields survive YAML load/dump round-trip."""

    def test_cookie_fields_roundtrip_through_yaml(self, tmp_path):
        """Test cookies_from_browser and cookies_path survive YAML load/dump."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
download:
  cookies_from_browser: firefox
  cookies_path: "/path/to/cookies.txt"
""")
        config = Config.from_yaml(str(config_file))
        assert config.download.cookies_from_browser == "firefox"
        assert config.download.cookies_path == "/path/to/cookies.txt"

    def test_cookie_fields_default_when_not_in_yaml(self, tmp_path):
        """Test cookie fields use defaults when not specified in YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
download:
  quality: "720p"
""")
        config = Config.from_yaml(str(config_file))
        assert config.download.cookies_from_browser == ""
        assert config.download.cookies_path == ""

    def test_empty_browser_in_yaml_accepted(self, tmp_path):
        """Test empty cookies_from_browser in YAML is accepted (disabled)."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
download:
  cookies_from_browser: ""
  cookies_path: ""
""")
        config = Config.from_yaml(str(config_file))
        assert config.download.cookies_from_browser == ""
        assert config.download.cookies_path == ""


# =============================================================================
# US-65-004: Expanded constraint validation tests
# =============================================================================

@pytest.mark.fast
class TestThresholdOrderingValidation:
    """Test matching confidence threshold ordering validation."""

    @pytest.mark.parametrize("low,ambig,min_c,high,expected_errors", [
        (0.5, 0.6, 0.7, 0.85, 0),   # Default ordering - valid
        (0.3, 0.4, 0.5, 0.9, 0),     # All properly ordered - valid
        (0.5, 0.5, 0.5, 0.5, 0),     # All equal - valid (edge case)
        (0.7, 0.6, 0.7, 0.85, 1),    # low > ambig - 1 error
        (0.5, 0.8, 0.7, 0.85, 1),    # ambig > min - 1 error
        (0.5, 0.6, 0.9, 0.85, 1),    # min > high - 1 error
        (0.9, 0.3, 0.5, 0.85, 1),    # low > ambig - 1 error
        (0.9, 0.8, 0.7, 0.6, 3),     # Fully reversed - 3 errors
    ])
    def test_threshold_ordering(self, low, ambig, min_c, high, expected_errors, tmp_path):
        """Test threshold ordering validation catches misordered thresholds."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
matching:
  low_confidence_threshold: {low}
  ambiguous_threshold: {ambig}
  min_confidence: {min_c}
  high_confidence_threshold: {high}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_constraints()
        threshold_errors = [e for e in errors if "threshold" in e.lower() or "ambiguous" in e.lower() or "min_confidence" in e.lower() and "should be <=" in e]
        assert len(threshold_errors) == expected_errors, f"Expected {expected_errors} errors, got {len(threshold_errors)}: {threshold_errors}"

    def test_threshold_errors_in_validate(self, tmp_path):
        """Test threshold ordering errors appear in full validate() output."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
matching:
  low_confidence_threshold: 0.9
  ambiguous_threshold: 0.3
  min_confidence: 0.7
  high_confidence_threshold: 0.85
""")
        config = Config.from_yaml(str(config_file))
        errors = config.validate()
        assert any("low_confidence_threshold" in e and "ambiguous_threshold" in e for e in errors)


@pytest.mark.fast
class TestOutputPositiveValueValidation:
    """Test output.frame_rate and output.time_scale_factor positive value validation."""

    @pytest.mark.parametrize("frame_rate,should_error", [
        (30.0, False),   # Default - valid
        (24.0, False),   # 24fps - valid
        (0.001, False),  # Very small positive - valid
        (0, True),       # Zero - invalid
        (-1, True),      # Negative - invalid
        (-30.0, True),   # Negative - invalid
    ])
    def test_frame_rate_validation(self, frame_rate, should_error, tmp_path):
        """Test frame_rate > 0 validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
output:
  frame_rate: {frame_rate}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_constraints()
        frame_errors = [e for e in errors if "frame_rate" in e]
        if should_error:
            assert len(frame_errors) > 0, f"Expected error for frame_rate={frame_rate}"
        else:
            assert len(frame_errors) == 0, f"Unexpected error for frame_rate={frame_rate}: {frame_errors}"

    @pytest.mark.parametrize("time_scale,should_error", [
        (1.0, False),    # Default - valid
        (1.065, False),  # Scaled up - valid
        (0.5, False),    # Scaled down - valid
        (0.001, False),  # Very small positive - valid
        (0, True),       # Zero - invalid
        (-1, True),      # Negative - invalid
    ])
    def test_time_scale_factor_validation(self, time_scale, should_error, tmp_path):
        """Test time_scale_factor > 0 validation."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
output:
  time_scale_factor: {time_scale}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_constraints()
        scale_errors = [e for e in errors if "time_scale_factor" in e]
        if should_error:
            assert len(scale_errors) > 0, f"Expected error for time_scale_factor={time_scale}"
        else:
            assert len(scale_errors) == 0, f"Unexpected error for time_scale_factor={time_scale}: {scale_errors}"


@pytest.mark.fast
class TestDurationTierMinMaxValidation:
    """Test duration tier min_seconds <= max_seconds validation."""

    @pytest.mark.parametrize("tier_name,min_s,max_s,should_error", [
        ("short", 20, 120, False),    # Default - valid
        ("medium", 120, 600, False),  # Default - valid
        ("short", 100, 100, False),   # Equal - valid (edge case)
        ("short", 200, 100, True),    # Reversed - invalid
        ("long", 1500, 600, True),    # Reversed - invalid
    ])
    def test_tier_min_max(self, tier_name, min_s, max_s, should_error, tmp_path):
        """Test duration tier min <= max validation."""
        config_file = tmp_path / "config.yaml"
        # _build_duration_tiers uses 'min'/'max' keys (short format)
        config_file.write_text(f"""
duration_tiers:
  {tier_name}:
    min: {min_s}
    max: {max_s}
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_constraints()
        tier_errors = [e for e in errors if f"duration_tiers.{tier_name}" in e]
        if should_error:
            assert len(tier_errors) > 0, f"Expected error for {tier_name} min={min_s} max={max_s}"
        else:
            assert len(tier_errors) == 0, f"Unexpected error for {tier_name}: {tier_errors}"

    def test_multiple_invalid_tiers(self, tmp_path):
        """Test multiple invalid tiers generate separate errors."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
duration_tiers:
  short:
    min: 200
    max: 100
  medium:
    min: 700
    max: 500
""")
        config = Config.from_yaml(str(config_file))
        errors = config._validate_constraints()
        tier_errors = [e for e in errors if "duration_tiers" in e]
        assert len(tier_errors) >= 2, f"Expected at least 2 tier errors, got {len(tier_errors)}: {tier_errors}"


# =============================================================================
# US-69-011: MultiStyleConfig and StockFootageConfig validation
# =============================================================================

from src.config.sections.output import MultiStyleConfig
from src.config.sections.duration import StockFootageConfig


@pytest.mark.fast
class TestMultiStyleConfigValueErrorValidation:
    """Test MultiStyleConfig raises ValueError for invalid values."""

    def test_enabled_with_empty_styles_raises_valueerror(self):
        """Test MultiStyleConfig(enabled=True, styles=[]) raises ValueError."""
        with pytest.raises(ValueError, match="styles.*non-empty.*enabled"):
            MultiStyleConfig(enabled=True, styles=[])

    def test_enabled_with_only_default_warns(self):
        """Test MultiStyleConfig(enabled=True, styles=['default']) warns."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = MultiStyleConfig(enabled=True, styles=["default"])
            assert len(w) == 1
            assert "only 'default' style" in str(w[0].message)
            assert config.enabled is True

    def test_disabled_with_empty_styles_accepted(self):
        """Test MultiStyleConfig(enabled=False, styles=[]) is valid."""
        config = MultiStyleConfig(enabled=False, styles=[])
        assert config.enabled is False
        assert config.styles == []

    def test_enabled_with_multiple_styles_accepted(self):
        """Test MultiStyleConfig(enabled=True, styles=['default', 'strict']) is valid."""
        config = MultiStyleConfig(enabled=True, styles=["default", "strict"])
        assert config.enabled is True
        assert config.styles == ["default", "strict"]

    def test_disabled_with_only_default_no_warning(self):
        """Test disabled config with only 'default' does NOT warn."""
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = MultiStyleConfig(enabled=False, styles=["default"])
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0
            assert config.styles == ["default"]

    def test_defaults_accepted(self):
        """Test default MultiStyleConfig values pass validation."""
        config = MultiStyleConfig()
        assert config.enabled is False
        assert config.styles == ["default", "strict"]


@pytest.mark.fast
class TestStockFootageConfigValueErrorValidation:
    """Test StockFootageConfig raises ValueError for invalid values."""

    def test_min_duration_zero_raises_valueerror(self):
        """Test min_duration=0 raises ValueError."""
        with pytest.raises(ValueError, match="min_duration.*positive"):
            StockFootageConfig(min_duration=0)

    def test_min_duration_negative_raises_valueerror(self):
        """Test min_duration=-5 raises ValueError."""
        with pytest.raises(ValueError, match="min_duration.*positive"):
            StockFootageConfig(min_duration=-5)

    def test_max_duration_zero_raises_valueerror(self):
        """Test max_duration=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_duration.*positive"):
            StockFootageConfig(max_duration=0)

    def test_max_duration_negative_raises_valueerror(self):
        """Test max_duration=-10 raises ValueError."""
        with pytest.raises(ValueError, match="max_duration.*positive"):
            StockFootageConfig(max_duration=-10)

    def test_min_greater_than_max_raises_valueerror(self):
        """Test min_duration > max_duration raises ValueError."""
        with pytest.raises(ValueError, match="min_duration.*<=.*max_duration"):
            StockFootageConfig(min_duration=60, max_duration=5)

    def test_min_equals_max_accepted(self):
        """Test min_duration == max_duration is valid (exact duration)."""
        config = StockFootageConfig(min_duration=30, max_duration=30)
        assert config.min_duration == 30
        assert config.max_duration == 30

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = StockFootageConfig(min_duration=1, max_duration=1)
        assert config.min_duration == 1
        assert config.max_duration == 1

        config = StockFootageConfig(min_duration=5, max_duration=60)
        assert config.min_duration == 5
        assert config.max_duration == 60

    def test_defaults_accepted(self):
        """Test default StockFootageConfig values pass validation."""
        config = StockFootageConfig()
        assert config.min_duration == 5
        assert config.max_duration == 60


# =============================================================================
# US-69-004: ValueError for impossible config values
# =============================================================================

from src.config.sections.core import EmbeddingConfig
from src.config.sections.output import DeduplicationConfig, VarietyConfig


@pytest.mark.fast
class TestMatchingConfigValueErrorValidation:
    """Test MatchingConfig raises ValueError for clearly invalid (impossible) values."""

    @pytest.mark.parametrize("field_name,value", [
        ("min_confidence", -0.1),
        ("high_confidence_threshold", -1.0),
        ("low_confidence_threshold", -0.5),
        ("ambiguous_threshold", -0.01),
        ("skip_llm_threshold", -99.0),
        ("confidence_threshold", -0.001),
    ])
    def test_negative_confidence_raises_valueerror(self, field_name, value):
        """Test negative confidence thresholds raise ValueError."""
        with pytest.raises(ValueError, match=f"MatchingConfig.{field_name}.*negative"):
            MatchingConfig(**{field_name: value})

    @pytest.mark.parametrize("field_name", [
        "embedding_candidates",
        "llm_rerank_candidates",
        "top_k_candidates",
        "context_window",
    ])
    def test_zero_or_negative_int_raises_valueerror(self, field_name):
        """Test zero or negative positive-int fields raise ValueError."""
        with pytest.raises(ValueError, match=f"MatchingConfig.{field_name}"):
            MatchingConfig(**{field_name: 0})
        with pytest.raises(ValueError, match=f"MatchingConfig.{field_name}"):
            MatchingConfig(**{field_name: -5})

    def test_confidence_above_one_clamped_not_error(self):
        """Test confidence > 1.0 is warn+clamped, not ValueError."""
        config = MatchingConfig(min_confidence=1.5)
        assert config.min_confidence == 1.0

    def test_valid_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = MatchingConfig(
            min_confidence=0.0,
            high_confidence_threshold=1.0,
            embedding_candidates=1,
        )
        assert config.min_confidence == 0.0
        assert config.high_confidence_threshold == 1.0
        assert config.embedding_candidates == 1


@pytest.mark.fast
class TestEmbeddingConfigValueErrorValidation:
    """Test EmbeddingConfig raises ValueError for clearly invalid (impossible) values."""

    def test_negative_batch_size_raises_valueerror(self):
        """Test negative batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingConfig(batch_size=-1)

    def test_zero_batch_size_raises_valueerror(self):
        """Test zero batch_size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            EmbeddingConfig(batch_size=0)

    def test_negative_max_retries_raises_valueerror(self):
        """Test negative max_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            EmbeddingConfig(max_retries=-1)

    def test_negative_retry_delay_raises_valueerror(self):
        """Test negative retry_delay raises ValueError."""
        with pytest.raises(ValueError, match="retry_delay"):
            EmbeddingConfig(retry_delay=-0.5)

    def test_negative_max_workers_raises_valueerror(self):
        """Test negative max_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers"):
            EmbeddingConfig(max_workers=-1)

    def test_zero_max_workers_raises_valueerror(self):
        """Test zero max_workers raises ValueError."""
        with pytest.raises(ValueError, match="max_workers"):
            EmbeddingConfig(max_workers=0)

    def test_valid_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = EmbeddingConfig(
            batch_size=1,
            max_retries=0,
            retry_delay=0.0,
            max_workers=1,
        )
        assert config.batch_size == 1
        assert config.max_retries == 0
        assert config.retry_delay == 0.0
        assert config.max_workers == 1


@pytest.mark.fast
class TestDeduplicationConfigValueErrorValidation:
    """Test DeduplicationConfig raises ValueError for invalid values."""

    def test_negative_hash_threshold_raises_valueerror(self):
        """Test hash_threshold=-1 raises ValueError."""
        with pytest.raises(ValueError, match="hash_threshold"):
            DeduplicationConfig(hash_threshold=-1)

    def test_hash_threshold_above_64_raises_valueerror(self):
        """Test hash_threshold=65 raises ValueError."""
        with pytest.raises(ValueError, match="hash_threshold"):
            DeduplicationConfig(hash_threshold=65)

    def test_zero_frame_timeout_raises_valueerror(self):
        """Test frame_timeout=0 raises ValueError."""
        with pytest.raises(ValueError, match="frame_timeout"):
            DeduplicationConfig(frame_timeout=0)

    def test_negative_frame_timeout_raises_valueerror(self):
        """Test frame_timeout=-5 raises ValueError."""
        with pytest.raises(ValueError, match="frame_timeout"):
            DeduplicationConfig(frame_timeout=-5)

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = DeduplicationConfig(hash_threshold=0, frame_timeout=1)
        assert config.hash_threshold == 0
        assert config.frame_timeout == 1

        config = DeduplicationConfig(hash_threshold=64, frame_timeout=30)
        assert config.hash_threshold == 64
        assert config.frame_timeout == 30


@pytest.mark.fast
class TestVisionConfigValueErrorValidation:
    """Test VisionConfig raises ValueError for invalid values."""

    def test_coverage_threshold_above_one_raises_valueerror(self):
        """Test coverage_threshold=1.5 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_threshold"):
            VisionConfig(coverage_threshold=1.5)

    def test_coverage_threshold_negative_raises_valueerror(self):
        """Test coverage_threshold=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_threshold"):
            VisionConfig(coverage_threshold=-0.1)

    def test_max_api_calls_zero_raises_valueerror(self):
        """Test max_api_calls_per_run=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_api_calls_per_run"):
            VisionConfig(max_api_calls_per_run=0)

    def test_max_api_calls_negative_raises_valueerror(self):
        """Test max_api_calls_per_run=-5 raises ValueError."""
        with pytest.raises(ValueError, match="max_api_calls_per_run"):
            VisionConfig(max_api_calls_per_run=-5)

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = VisionConfig(coverage_threshold=0.0, max_api_calls_per_run=1)
        assert config.coverage_threshold == 0.0
        assert config.max_api_calls_per_run == 1

        config = VisionConfig(coverage_threshold=1.0, max_api_calls_per_run=100)
        assert config.coverage_threshold == 1.0
        assert config.max_api_calls_per_run == 100


@pytest.mark.fast
class TestSceneDetectionConfigValueErrorValidation:
    """Test SceneDetectionConfig raises ValueError for invalid values."""

    def test_negative_threshold_raises_valueerror(self):
        """Test threshold=-1 raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            SceneDetectionConfig(threshold=-1)

    def test_zero_threshold_raises_valueerror(self):
        """Test threshold=0 raises ValueError."""
        with pytest.raises(ValueError, match="threshold"):
            SceneDetectionConfig(threshold=0)

    def test_negative_min_scene_len_raises_valueerror(self):
        """Test min_scene_len=-5 raises ValueError."""
        with pytest.raises(ValueError, match="min_scene_len"):
            SceneDetectionConfig(min_scene_len=-5)

    def test_zero_min_scene_len_raises_valueerror(self):
        """Test min_scene_len=0 raises ValueError."""
        with pytest.raises(ValueError, match="min_scene_len"):
            SceneDetectionConfig(min_scene_len=0)

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = SceneDetectionConfig(threshold=0.1, min_scene_len=1)
        assert config.threshold == 0.1
        assert config.min_scene_len == 1

        config = SceneDetectionConfig(threshold=27.0, min_scene_len=15)
        assert config.threshold == 27.0
        assert config.min_scene_len == 15


# =============================================================================
# US-69-007: EnhancedFeaturesConfig and LLMTitleFilterConfig validation
# =============================================================================


@pytest.mark.fast
class TestEnhancedFeaturesConfigValueErrorValidation:
    """Test EnhancedFeaturesConfig raises ValueError for invalid values."""

    def test_min_confidence_above_one_raises_valueerror(self):
        """Test min_confidence=1.5 raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence"):
            EnhancedFeaturesConfig(min_confidence=1.5)

    def test_min_confidence_negative_raises_valueerror(self):
        """Test min_confidence=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="min_confidence"):
            EnhancedFeaturesConfig(min_confidence=-0.1)

    def test_max_retries_zero_raises_valueerror(self):
        """Test max_retries=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            EnhancedFeaturesConfig(max_retries=0)

    def test_max_retries_negative_raises_valueerror(self):
        """Test max_retries=-3 raises ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            EnhancedFeaturesConfig(max_retries=-3)

    def test_face_preference_invalid_raises_valueerror(self):
        """Test face_preference='invalid' raises ValueError."""
        with pytest.raises(ValueError, match="face_preference"):
            EnhancedFeaturesConfig(face_preference="invalid")

    def test_face_preference_case_sensitive(self):
        """Test face_preference='Neutral' (wrong case) raises ValueError."""
        with pytest.raises(ValueError, match="face_preference"):
            EnhancedFeaturesConfig(face_preference="Neutral")

    def test_valid_face_preferences_accepted(self):
        """Test all valid face_preference values are accepted."""
        for pref in ("neutral", "more", "none"):
            config = EnhancedFeaturesConfig(face_preference=pref)
            assert config.face_preference == pref

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = EnhancedFeaturesConfig(min_confidence=0.0, max_retries=1)
        assert config.min_confidence == 0.0
        assert config.max_retries == 1

        config = EnhancedFeaturesConfig(min_confidence=1.0, max_retries=100)
        assert config.min_confidence == 1.0
        assert config.max_retries == 100


@pytest.mark.fast
class TestLLMTitleFilterConfigValueErrorValidation:
    """Test LLMTitleFilterConfig raises ValueError for invalid values."""

    def test_min_relevance_above_one_raises_valueerror(self):
        """Test min_relevance=1.5 raises ValueError."""
        with pytest.raises(ValueError, match="min_relevance"):
            LLMTitleFilterConfig(min_relevance=1.5)

    def test_min_relevance_negative_raises_valueerror(self):
        """Test min_relevance=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="min_relevance"):
            LLMTitleFilterConfig(min_relevance=-0.1)

    def test_batch_size_zero_raises_valueerror(self):
        """Test batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            LLMTitleFilterConfig(batch_size=0)

    def test_batch_size_negative_raises_valueerror(self):
        """Test batch_size=-5 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            LLMTitleFilterConfig(batch_size=-5)

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = LLMTitleFilterConfig(min_relevance=0.0, batch_size=1)
        assert config.min_relevance == 0.0
        assert config.batch_size == 1

        config = LLMTitleFilterConfig(min_relevance=1.0, batch_size=100)
        assert config.min_relevance == 1.0
        assert config.batch_size == 100


@pytest.mark.fast
class TestRemixConfigValueErrorValidation:
    """Test RemixConfig raises ValueError for invalid values."""

    def test_min_relevance_score_above_one_raises_valueerror(self):
        """Test min_relevance_score=1.5 raises ValueError."""
        with pytest.raises(ValueError, match="min_relevance_score"):
            RemixConfig(min_relevance_score=1.5)

    def test_min_relevance_score_negative_raises_valueerror(self):
        """Test min_relevance_score=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="min_relevance_score"):
            RemixConfig(min_relevance_score=-0.1)

    def test_max_workers_zero_raises_valueerror(self):
        """Test max_workers=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_workers"):
            RemixConfig(max_workers=0)

    def test_max_workers_negative_raises_valueerror(self):
        """Test max_workers=-2 raises ValueError."""
        with pytest.raises(ValueError, match="max_workers"):
            RemixConfig(max_workers=-2)

    def test_auto_accept_filter_invalid_raises_valueerror(self):
        """Test auto_accept_filter='invalid' raises ValueError."""
        with pytest.raises(ValueError, match="auto_accept_filter"):
            RemixConfig(auto_accept_filter="invalid")

    def test_auto_accept_filter_case_sensitive(self):
        """Test auto_accept_filter='Filtered' (wrong case) raises ValueError."""
        with pytest.raises(ValueError, match="auto_accept_filter"):
            RemixConfig(auto_accept_filter="Filtered")

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = RemixConfig(min_relevance_score=0.0, max_workers=1, auto_accept_filter="filtered")
        assert config.min_relevance_score == 0.0
        assert config.max_workers == 1
        assert config.auto_accept_filter == "filtered"

        config = RemixConfig(min_relevance_score=1.0, max_workers=100, auto_accept_filter="all")
        assert config.min_relevance_score == 1.0
        assert config.max_workers == 100
        assert config.auto_accept_filter == "all"

    def test_all_valid_auto_accept_values(self):
        """Test all valid auto_accept_filter values are accepted."""
        for value in ("filtered", "all", "prompt"):
            config = RemixConfig(auto_accept_filter=value)
            assert config.auto_accept_filter == value


@pytest.mark.fast
class TestZeroDownloadRemixConfigValueErrorValidation:
    """Test ZeroDownloadRemixConfig raises ValueError for invalid values."""

    def test_max_retries_zero_raises_valueerror(self):
        """Test max_retries=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            ZeroDownloadRemixConfig(max_retries=0)

    def test_max_retries_negative_raises_valueerror(self):
        """Test max_retries=-1 raises ValueError."""
        with pytest.raises(ValueError, match="max_retries"):
            ZeroDownloadRemixConfig(max_retries=-1)

    def test_max_keywords_per_batch_zero_raises_valueerror(self):
        """Test max_keywords_per_batch=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_keywords_per_batch"):
            ZeroDownloadRemixConfig(max_keywords_per_batch=0)

    def test_max_keywords_per_batch_negative_raises_valueerror(self):
        """Test max_keywords_per_batch=-5 raises ValueError."""
        with pytest.raises(ValueError, match="max_keywords_per_batch"):
            ZeroDownloadRemixConfig(max_keywords_per_batch=-5)

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = ZeroDownloadRemixConfig(max_retries=1, max_keywords_per_batch=1)
        assert config.max_retries == 1
        assert config.max_keywords_per_batch == 1

        config = ZeroDownloadRemixConfig(max_retries=50, max_keywords_per_batch=100)
        assert config.max_retries == 50
        assert config.max_keywords_per_batch == 100


@pytest.mark.fast
class TestVarietyConfigValueErrorValidation:
    """Test VarietyConfig raises ValueError for invalid values."""

    def test_min_time_distance_negative_raises_valueerror(self):
        """Test min_time_distance=-1 raises ValueError."""
        with pytest.raises(ValueError, match="min_time_distance"):
            VarietyConfig(min_time_distance=-1)

    def test_min_time_distance_negative_float_raises_valueerror(self):
        """Test min_time_distance=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="min_time_distance"):
            VarietyConfig(min_time_distance=-0.1)

    def test_min_time_distance_zero_accepted(self):
        """Test min_time_distance=0 is valid (no minimum distance)."""
        config = VarietyConfig(min_time_distance=0)
        assert config.min_time_distance == 0

    def test_min_embedding_distance_negative_raises_valueerror(self):
        """Test min_embedding_distance=-0.1 raises ValueError."""
        with pytest.raises(ValueError, match="min_embedding_distance"):
            VarietyConfig(min_embedding_distance=-0.1)

    def test_min_embedding_distance_above_max_raises_valueerror(self):
        """Test min_embedding_distance=2.1 raises ValueError (cosine max is 2.0)."""
        with pytest.raises(ValueError, match="min_embedding_distance"):
            VarietyConfig(min_embedding_distance=2.1)

    def test_min_embedding_distance_boundary_values_accepted(self):
        """Test min_embedding_distance boundary values 0.0 and 2.0 are valid."""
        config = VarietyConfig(min_embedding_distance=0.0)
        assert config.min_embedding_distance == 0.0
        config = VarietyConfig(min_embedding_distance=2.0)
        assert config.min_embedding_distance == 2.0

    def test_timeline_variety_window_zero_raises_valueerror(self):
        """Test timeline_variety_window=0 raises ValueError."""
        with pytest.raises(ValueError, match="timeline_variety_window"):
            VarietyConfig(timeline_variety_window=0)

    def test_timeline_variety_window_negative_raises_valueerror(self):
        """Test timeline_variety_window=-100 raises ValueError."""
        with pytest.raises(ValueError, match="timeline_variety_window"):
            VarietyConfig(timeline_variety_window=-100)

    def test_max_source_repeats_in_window_zero_raises_valueerror(self):
        """Test max_source_repeats_in_window=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_source_repeats_in_window"):
            VarietyConfig(max_source_repeats_in_window=0)

    def test_max_source_repeats_in_window_negative_raises_valueerror(self):
        """Test max_source_repeats_in_window=-1 raises ValueError."""
        with pytest.raises(ValueError, match="max_source_repeats_in_window"):
            VarietyConfig(max_source_repeats_in_window=-1)

    def test_valid_boundary_values_accepted(self):
        """Test valid boundary values are accepted without error."""
        config = VarietyConfig(
            min_time_distance=0.0,
            min_embedding_distance=0.0,
            timeline_variety_window=0.1,
            max_source_repeats_in_window=1,
        )
        assert config.min_time_distance == 0.0
        assert config.min_embedding_distance == 0.0
        assert config.timeline_variety_window == 0.1
        assert config.max_source_repeats_in_window == 1

    def test_defaults_accepted(self):
        """Test default VarietyConfig values pass validation."""
        config = VarietyConfig()
        assert config.min_time_distance == 10.0
        assert config.min_embedding_distance == 0.3
        assert config.timeline_variety_window == 600.0
        assert config.max_source_repeats_in_window == 1


# =============================================================================
# US-80-003: _convert_nested_configs handles all section dataclasses
# =============================================================================


class TestConvertNestedConfigsAllSections:
    """Test that _convert_nested_configs converts all section fields, not just healing."""

    def test_matching_dict_converted_to_dataclass(self):
        """After merge_config replaces matching with a dict, _convert_nested_configs restores it."""
        config = Config()
        # Simulate merge_config replacing matching section with a raw dict
        config.matching = {"min_confidence": 0.5, "max_clip_reuse": 3}
        config._convert_nested_configs()
        assert isinstance(config.matching, MatchingConfig)
        assert config.matching.min_confidence == 0.5
        assert config.matching.max_clip_reuse == 3

    def test_download_dict_converted_to_dataclass(self):
        """After merge_config replaces download with a dict, _convert_nested_configs restores it."""
        config = Config()
        config.download = {"quality": "720p", "davinci_mode": False}
        config._convert_nested_configs()
        assert isinstance(config.download, DownloadConfig)
        assert config.download.quality == "720p"
        assert config.download.davinci_mode is False

    def test_healing_dict_still_converted(self):
        """Healing dict conversion still works with the generic approach."""
        config = Config()
        config.healing = {"enabled": False}
        config._convert_nested_configs()
        assert isinstance(config.healing, HealingConfig)
        assert config.healing.enabled is False

    def test_already_converted_sections_unchanged(self):
        """Dataclass instances that are already correct type are left unchanged (idempotent)."""
        config = Config()
        original_matching = config.matching
        original_download = config.download
        original_healing = config.healing
        # Running _convert_nested_configs on already-correct instances should not break them
        config._convert_nested_configs()
        assert isinstance(config.matching, MatchingConfig)
        assert isinstance(config.download, DownloadConfig)
        assert isinstance(config.healing, HealingConfig)
        # Values should be preserved
        assert config.matching.min_confidence == original_matching.min_confidence
        assert config.download.quality == original_download.quality

    def test_idempotent_double_call(self):
        """Calling _convert_nested_configs twice produces same result."""
        config = Config()
        config.matching = {"min_confidence": 0.7}
        config._convert_nested_configs()
        first_result = config.matching.min_confidence
        config._convert_nested_configs()
        assert config.matching.min_confidence == first_result
        assert isinstance(config.matching, MatchingConfig)


# =============================================================================
# US-80-011: Comprehensive config section coverage for all 14 section files
# =============================================================================

# --- BrollConfig (broll.py) ---

from src.config.sections.broll import BrollConfig, BrollSourceBoostConfig


@pytest.mark.fast
class TestBrollConfigDefaults:
    """Test BrollConfig default construction and dict conversion."""

    def test_default_construction(self):
        """Test BrollConfig constructs with defaults."""
        config = BrollConfig()
        assert config.enabled is True
        assert config.downloads_per_term == 3
        assert config.max_total_downloads == 30
        assert config.min_match_score == 0.3
        assert isinstance(config.source_boost, BrollSourceBoostConfig)

    def test_dict_construction(self):
        """Test BrollConfig from **kwargs."""
        config = BrollConfig(
            enabled=False, downloads_per_term=5, min_match_score=0.5
        )
        assert config.enabled is False
        assert config.downloads_per_term == 5
        assert config.min_match_score == 0.5

    def test_source_boost_dict_converted(self):
        """Test source_boost dict is converted to BrollSourceBoostConfig."""
        config = BrollConfig(
            source_boost={'youtube': 0.2, 'pexels': 0.1, 'pixabay': 0.0}
        )
        assert isinstance(config.source_boost, BrollSourceBoostConfig)
        assert config.source_boost.youtube == 0.2

    def test_source_boost_dataclass_unchanged(self):
        """Test source_boost dataclass instance is left unchanged."""
        boost = BrollSourceBoostConfig(youtube=0.3)
        config = BrollConfig(source_boost=boost)
        assert config.source_boost is boost


# --- ImageSearchConfig (entity.py) ---

from src.config.sections.entity import (
    ImageSearchConfig, EntityCacheConfig, StockVideoConfig, SilentVideoConfig,
)


@pytest.mark.fast
class TestImageSearchConfigDefaults:
    """Test ImageSearchConfig default construction and dict conversion."""

    def test_default_construction(self):
        """Test ImageSearchConfig constructs with defaults."""
        config = ImageSearchConfig()
        assert config.enabled is True
        assert config.images_per_entity == 5
        assert isinstance(config.stock_video, StockVideoConfig)
        assert isinstance(config.entity_cache, EntityCacheConfig)

    def test_dict_construction(self):
        """Test ImageSearchConfig from **kwargs."""
        config = ImageSearchConfig(enabled=False, images_per_entity=10)
        assert config.enabled is False
        assert config.images_per_entity == 10

    def test_stock_video_dict_converted(self):
        """Test stock_video dict is converted to StockVideoConfig."""
        config = ImageSearchConfig(
            stock_video={'min_duration': 5.0, 'max_duration': 60.0}
        )
        assert isinstance(config.stock_video, StockVideoConfig)
        assert config.stock_video.min_duration == 5.0

    def test_entity_cache_dict_converted(self):
        """Test entity_cache dict is converted to EntityCacheConfig."""
        config = ImageSearchConfig(
            entity_cache={'enabled': True, 'fuzzy_threshold': 0.9}
        )
        assert isinstance(config.entity_cache, EntityCacheConfig)
        assert config.entity_cache.fuzzy_threshold == 0.9


@pytest.mark.fast
class TestEntityCacheConfigDefaults:
    """Test EntityCacheConfig default values."""

    def test_default_values(self):
        config = EntityCacheConfig()
        assert config.enabled is False
        assert config.fuzzy_threshold == 0.85
        assert config.cache_strategy == "copy"


@pytest.mark.fast
class TestSilentVideoConfigDefaults:
    """Test SilentVideoConfig default values."""

    def test_default_values(self):
        config = SilentVideoConfig()
        assert config.enabled is True
        assert config.min_words_threshold == 10
        assert config.use_vision_api is True


# --- RateLimitConfig (rate_limit.py) ---

from src.config.sections.rate_limit import RateLimitConfig as RLConfig


@pytest.mark.fast
class TestRateLimitConfigDefaults:
    """Test RateLimitConfig default construction."""

    def test_default_construction(self):
        """Test RateLimitConfig constructs with defaults."""
        config = RLConfig()
        assert config.slots_per_second == 0.5
        assert config.burst_size == 3
        assert config.jitter_factor == 0.2
        assert config.max_backoff_seconds == 60.0

    def test_dict_construction(self):
        """Test RateLimitConfig from **kwargs."""
        config = RLConfig(slots_per_second=1.0, burst_size=5)
        assert config.slots_per_second == 1.0
        assert config.burst_size == 5


# --- KeywordConfig (keywords.py) ---

from src.config.sections.keywords import KeywordConfig, ListDetectionConfig


@pytest.mark.fast
class TestKeywordConfigDefaults:
    """Test KeywordConfig default construction and dict conversion."""

    def test_default_construction(self):
        """Test KeywordConfig constructs with defaults."""
        config = KeywordConfig()
        assert config.provider == "gemini"
        assert config.max_keywords == 30
        assert isinstance(config.list_detection, ListDetectionConfig)

    def test_dict_construction(self):
        """Test KeywordConfig from **kwargs."""
        config = KeywordConfig(provider="anthropic", max_keywords=50)
        assert config.provider == "anthropic"
        assert config.max_keywords == 50

    def test_list_detection_none_creates_default(self):
        """Test list_detection=None creates default ListDetectionConfig."""
        config = KeywordConfig(list_detection=None)
        assert isinstance(config.list_detection, ListDetectionConfig)
        assert config.list_detection.enabled is True

    def test_list_detection_dict_converted(self):
        """Test list_detection dict is converted to ListDetectionConfig."""
        config = KeywordConfig(
            list_detection={'enabled': False, 'keyword_suffix': 'video'}
        )
        assert isinstance(config.list_detection, ListDetectionConfig)
        assert config.list_detection.enabled is False
        assert config.list_detection.keyword_suffix == 'video'


# --- IterativeMatchingConfig (iterative_matching.py) ---

from src.config.sections.iterative_matching import IterativeMatchingConfig


@pytest.mark.fast
class TestIterativeMatchingConfigDefaults:
    """Test IterativeMatchingConfig default construction and clamping."""

    def test_default_construction(self):
        """Test IterativeMatchingConfig constructs with defaults."""
        config = IterativeMatchingConfig()
        assert config.enabled is True
        assert config.target_confidence == 0.90
        assert config.source_spacing_seconds == 300.0
        assert config.max_iterations == 5

    def test_dict_construction(self):
        """Test IterativeMatchingConfig from **kwargs."""
        config = IterativeMatchingConfig(
            target_confidence=0.85, max_iterations=3
        )
        assert config.target_confidence == 0.85
        assert config.max_iterations == 3

    def test_target_confidence_clamped_above_one(self):
        """Test target_confidence > 1.0 is clamped to 1.0."""
        config = IterativeMatchingConfig(target_confidence=1.5)
        assert config.target_confidence == 1.0

    def test_target_confidence_clamped_below_zero(self):
        """Test target_confidence < 0.0 is clamped to 0.0."""
        config = IterativeMatchingConfig(target_confidence=-0.5)
        assert config.target_confidence == 0.0

    def test_max_iterations_clamped_to_one(self):
        """Test max_iterations < 1 is clamped to 1."""
        config = IterativeMatchingConfig(max_iterations=0)
        assert config.max_iterations == 1

    def test_negative_spacing_clamped_to_zero(self):
        """Test source_spacing_seconds < 0 is clamped to 0."""
        config = IterativeMatchingConfig(source_spacing_seconds=-100)
        assert config.source_spacing_seconds == 0.0


# --- TranscriptionConfig (core.py) ---

from src.config.sections.core import (
    TranscriptionConfig, PauseSplitConfig, ProjectConfig, IndexingConfig,
)


@pytest.mark.fast
class TestTranscriptionConfigDefaults:
    """Test TranscriptionConfig default construction and validation."""

    def test_default_construction(self):
        """Test TranscriptionConfig constructs with defaults."""
        config = TranscriptionConfig()
        assert config.model == "base"
        assert config.language == "en"
        assert config.use_gpu is True
        assert isinstance(config.pause_split, PauseSplitConfig)

    def test_dict_construction(self):
        """Test TranscriptionConfig from **kwargs."""
        config = TranscriptionConfig(model="small", language="fr")
        assert config.model == "small"
        assert config.language == "fr"

    def test_invalid_whisper_model_raises_valueerror(self):
        """AC5: TranscriptionConfig with invalid whisper_model raises ValueError."""
        with pytest.raises(ValueError, match="not a known Whisper model"):
            TranscriptionConfig(model="super-large")

    def test_invalid_compute_type_raises_valueerror(self):
        """Test invalid compute_type raises ValueError."""
        with pytest.raises(ValueError, match="compute_type"):
            TranscriptionConfig(compute_type="bfloat16")

    def test_low_gpu_memory_raises_valueerror(self):
        """Test minimum_gpu_memory_mb < 100 raises ValueError."""
        with pytest.raises(ValueError, match="minimum_gpu_memory_mb"):
            TranscriptionConfig(minimum_gpu_memory_mb=50)

    def test_zero_max_workers_raises_valueerror(self):
        """Test max_workers=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_workers"):
            TranscriptionConfig(max_workers=0)

    def test_zero_batch_size_raises_valueerror(self):
        """Test batch_size=0 raises ValueError."""
        with pytest.raises(ValueError, match="batch_size"):
            TranscriptionConfig(batch_size=0)

    def test_gpu_timeout_too_low_raises_valueerror(self):
        """Test gpu_transcription_timeout < 30 raises ValueError."""
        with pytest.raises(ValueError, match="gpu_transcription_timeout"):
            TranscriptionConfig(gpu_transcription_timeout=10)

    def test_valid_models_accepted(self):
        """Test all known Whisper models are accepted."""
        for model in ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']:
            config = TranscriptionConfig(model=model)
            assert config.model == model

    def test_pause_split_none_creates_default(self):
        """Test pause_split=None creates default PauseSplitConfig."""
        config = TranscriptionConfig(pause_split=None)
        assert isinstance(config.pause_split, PauseSplitConfig)


@pytest.mark.fast
class TestProjectConfigDefaults:
    """Test ProjectConfig default values."""

    def test_default_values(self):
        config = ProjectConfig()
        assert config.name == "matcher-alt"
        assert config.version == "3.0.0"

    def test_dict_construction(self):
        config = ProjectConfig(name="my-project", description="test")
        assert config.name == "my-project"
        assert config.description == "test"


@pytest.mark.fast
class TestIndexingConfigDefaults:
    """Test IndexingConfig default values."""

    def test_default_values(self):
        config = IndexingConfig()
        assert config.index_type == "flat"
        assert config.use_faiss is True
        assert config.similarity_metric == "cosine"


# --- DownloadConfig parallel_workers validation ---


@pytest.mark.fast
class TestDownloadConfigParallelWorkersValidation:
    """AC3: Test DownloadConfig with negative parallel_workers raises ValueError."""

    def test_negative_parallel_workers_raises_valueerror(self):
        """Test parallel_workers=-1 raises ValueError."""
        with pytest.raises(ValueError, match="parallel_workers"):
            DownloadConfig(parallel_workers=-1)

    def test_zero_parallel_workers_raises_valueerror(self):
        """Test parallel_workers=0 raises ValueError."""
        with pytest.raises(ValueError, match="parallel_workers"):
            DownloadConfig(parallel_workers=0)

    def test_valid_parallel_workers_accepted(self):
        """Test parallel_workers=1 (minimum) is accepted."""
        config = DownloadConfig(parallel_workers=1)
        assert config.parallel_workers == 1

    def test_default_parallel_workers_valid(self):
        """Test default parallel_workers passes validation."""
        config = DownloadConfig()
        assert config.parallel_workers >= 1


# --- MatchingConfig confidence threshold ordering (warns + clamps) ---


@pytest.mark.fast
class TestMatchingConfigConfidenceClampingAndWarning:
    """AC4: Test MatchingConfig with confidence thresholds out of order warns and clamps."""

    def test_confidence_above_one_clamped(self):
        """Test confidence > 1.0 is warn+clamped to 1.0."""
        config = MatchingConfig(min_confidence=1.5)
        assert config.min_confidence == 1.0

    def test_high_confidence_threshold_clamped(self):
        """Test high_confidence_threshold > 1.0 is clamped."""
        config = MatchingConfig(high_confidence_threshold=2.0)
        assert config.high_confidence_threshold == 1.0

    def test_low_confidence_threshold_clamped(self):
        """Test low_confidence_threshold > 1.0 is clamped."""
        config = MatchingConfig(low_confidence_threshold=1.1)
        assert config.low_confidence_threshold == 1.0

    def test_ambiguous_threshold_clamped(self):
        """Test ambiguous_threshold > 1.0 is clamped."""
        config = MatchingConfig(ambiguous_threshold=1.3)
        assert config.ambiguous_threshold == 1.0

    def test_negative_confidence_raises_valueerror(self):
        """Test negative confidence raises ValueError."""
        with pytest.raises(ValueError, match="negative"):
            MatchingConfig(min_confidence=-0.1)

    def test_all_thresholds_at_boundary_accepted(self):
        """Test all thresholds at 0.0 and 1.0 boundaries are accepted."""
        config = MatchingConfig(
            min_confidence=0.0,
            high_confidence_threshold=1.0,
            low_confidence_threshold=0.0,
            ambiguous_threshold=0.0,
        )
        assert config.min_confidence == 0.0
        assert config.high_confidence_threshold == 1.0


# --- ContextEnrichmentConfig and ChapterGroupingConfig (matching.py) ---

from src.config.sections.matching import ContextEnrichmentConfig, ChapterGroupingConfig


@pytest.mark.fast
class TestContextEnrichmentConfigCoverage:
    """Test ContextEnrichmentConfig validation coverage."""

    def test_default_construction(self):
        config = ContextEnrichmentConfig()
        assert config.max_description_length == 500
        assert config.extract_video_description is True

    def test_negative_description_length_raises_valueerror(self):
        with pytest.raises(ValueError, match="max_description_length"):
            ContextEnrichmentConfig(max_description_length=-1)

    def test_large_description_length_clamped(self):
        config = ContextEnrichmentConfig(max_description_length=20000)
        assert config.max_description_length == 10000


@pytest.mark.fast
class TestChapterGroupingConfigCoverage:
    """Test ChapterGroupingConfig validation coverage."""

    def test_default_construction(self):
        config = ChapterGroupingConfig()
        assert config.enabled is True
        assert config.coherence_penalty_threshold == 5

    def test_zero_coherence_threshold_raises_valueerror(self):
        with pytest.raises(ValueError, match="coherence_penalty_threshold"):
            ChapterGroupingConfig(coherence_penalty_threshold=0)

    def test_positive_mismatch_penalty_raises_valueerror(self):
        with pytest.raises(ValueError, match="chapter_topic_mismatch_penalty"):
            ChapterGroupingConfig(chapter_topic_mismatch_penalty=0.1)

    def test_negative_relevance_boost_raises_valueerror(self):
        with pytest.raises(ValueError, match="relevance_boost_weight"):
            ChapterGroupingConfig(relevance_boost_weight=-0.5)

    def test_boost_min_greater_than_max_raises_valueerror(self):
        with pytest.raises(ValueError, match="chapter_topic_match_boost"):
            ChapterGroupingConfig(chapter_topic_match_boost=[0.2, 0.1])


# --- MatchingScoringConfig (matching.py) ---

from src.config.sections.matching import MatchingScoringConfig


@pytest.mark.fast
class TestMatchingScoringConfigCoverage:
    """Test MatchingScoringConfig default construction and dict conversion."""

    def test_default_construction(self):
        config = MatchingScoringConfig()
        assert config.confidence_floor == 0.05
        assert isinstance(config.entity_match_boosts, dict)

    def test_dict_keys_converted_to_strings(self):
        """Test numeric dict keys are converted to strings in __post_init__."""
        config = MatchingScoringConfig(
            entity_match_boosts={1: 0.05, 2: 0.08}
        )
        assert '1' in config.entity_match_boosts
        assert '2' in config.entity_match_boosts


class TestConfigFreezeMechanism:
    """US-80-012: Config freeze mechanism prevents mutation after pipeline start."""

    def test_config_before_freeze_allows_normal_setting(self):
        """Before freeze(), normal attribute setting works."""
        config = Config()
        config.matching.min_confidence = 0.99
        assert config.matching.min_confidence == 0.99

    def test_config_freeze_blocks_top_level_setattr(self):
        """After freeze(), setting attributes on Config raises FrozenConfigError."""
        config = Config()
        config.freeze()
        try:
            with pytest.raises(FrozenConfigError):
                config.project_dir = "/new/path"
        finally:
            config.unfreeze()

    def test_config_freeze_blocks_section_setattr(self):
        """After freeze(), setting config.matching.min_confidence raises FrozenConfigError."""
        config = Config()
        config.freeze()
        try:
            with pytest.raises(FrozenConfigError):
                config.matching.min_confidence = 0.99
        finally:
            config.unfreeze()

    def test_config_unfreeze_restores_mutability(self):
        """After unfreeze(), attributes can be set again."""
        config = Config()
        config.freeze()
        config.unfreeze()
        config.matching.min_confidence = 0.42
        assert config.matching.min_confidence == 0.42

    def test_freeze_sets_frozen_flag(self):
        """freeze() sets internal _frozen flag."""
        config = Config()
        assert not getattr(config, '_frozen', False)
        config.freeze()
        try:
            assert config._frozen is True
        finally:
            config.unfreeze()

    def test_unfreeze_clears_frozen_flag(self):
        """unfreeze() clears internal _frozen flag."""
        config = Config()
        config.freeze()
        config.unfreeze()
        assert config._frozen is False

    def test_freeze_blocks_nested_section_setattr(self):
        """After freeze(), setting deeply nested config raises FrozenConfigError."""
        config = Config()
        config.freeze()
        try:
            with pytest.raises(FrozenConfigError):
                config.download.parallel_workers = 8
        finally:
            config.unfreeze()

    def test_frozen_config_error_is_config_error_subclass(self):
        """FrozenConfigError is a subclass of ConfigError."""
        assert issubclass(FrozenConfigError, ConfigError)

    def test_freeze_allows_private_attr_setting(self):
        """Frozen config still allows setting private/internal attributes."""
        config = Config()
        config.freeze()
        try:
            # Private attrs should still work (e.g., for internal bookkeeping)
            config._config_hash = "test_hash"
            assert config._config_hash == "test_hash"
        finally:
            config.unfreeze()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
