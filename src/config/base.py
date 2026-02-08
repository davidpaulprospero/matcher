"""
Configuration Management System v3.0

Chain-of-thought: Centralized configuration loader for matcher-alt pipeline
Reasoning: Single source of truth reduces maintenance, improves consistency, enables hot-reload
Decision: CSafeLoader for performance, schema validation, file change detection

Features:
- CSafeLoader (C-based YAML parser) for 40-60% faster parsing
- Schema validation with clear error messages
- Hot-reload with file change detection (hash-based)
- Caching mechanism for repeated config access
- Comprehensive dataclasses mirroring config.yaml structure
- Logging of config access patterns

Performance:
- Config loading: <100ms target
- Component init: <50ms target
- Cached access: <1ms

Usage:
    from src.config import load_config, get_config

    # At startup
    config = load_config("config.yaml")

    # Anywhere in codebase
    config = get_config()
    threshold = config.matching.min_confidence

Config Validation Convention:
    All config dataclass __post_init__ methods follow this two-tier pattern:

    1. ValueError for invalid types and impossible values:
       - Negative counts (batch_size, max_workers, max_retries)
       - Negative thresholds (confidence < 0.0, retry_delay < 0)
       - Invalid enum values (unknown model names, unknown compute types)
       These represent configuration errors that cannot produce correct behavior.

    2. warn + clamp for soft limits (values slightly outside recommended range):
       - Confidence threshold > 1.0 (clamped to 1.0 with warning)
       - Workers exceeding CPU count (capped with warning)
       These represent likely typos or misunderstandings but are recoverable.

    See TranscriptionConfig, MatchingConfig, and EmbeddingConfig for examples.
"""

from __future__ import annotations

import os
import sys
import difflib
import hashlib
import time
import logging
import threading
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields, is_dataclass, MISSING
from typing import Optional, List, Dict, Any, Tuple, Union, TypeVar, Type
from datetime import datetime

# Use CSafeLoader for performance (40-60% faster than pure Python)
try:
    import yaml
    from yaml import CSafeLoader as SafeLoader
    YAML_FAST = True
except ImportError:
    import yaml
    from yaml import SafeLoader
    YAML_FAST = False

# Import all section configs
from .utils import safe_get_config_value

from .sections import (
    # Infrastructure
    LoggingConfig,
    CacheConfig,
    GlobalCacheConfig,
    PipelineConfig,
    APIKeysConfig,
    HealingConfig,
    # Core
    ProjectConfig,
    PauseSplitConfig,
    TranscriptionConfig,
    EmbeddingConfig,
    IndexingConfig,
    # Matching
    LocationMatchingConfig,
    NegativeMatchingConfig,
    MatchingConfig,
    # LLM
    LLMRetryConfig,
    LLMCacheConfig,
    LLMProviderConfig,
    LLMConfig,
    # Download
    RemixConfig,
    ZeroDownloadRemixConfig,
    EnhancedFeaturesConfig,
    LLMTitleFilterConfig,
    AudioFirstConfig,
    SpeechScreeningConfig,
    DownloadConfig,
    DownloadingConfig,
    # Keywords
    ListDetectionConfig,
    KeywordConfig,
    # Entity
    StockVideoConfig,
    SilentVideoConfig,
    EntityCacheConfig,
    ImageSearchConfig,
    # Duration
    DurationTierConfig,
    DurationTiersConfig,
    StockFootageConfig,
    # Output
    DeduplicationConfig,
    VarietyConfig,
    OutputConfig,
    MultiStyleConfig,
    # Media
    VisionConfig,
    SceneDetectionConfig,
    AudioAnalysisConfig,
    # B-roll
    BrollSourceBoostConfig,
    BrollConfig,
    # Iterative matching
    IterativeMatchingConfig,
    # Rate limiting
    RateLimitConfig,
)

logger = logging.getLogger(__name__)

# Type variable for dataclass building
T = TypeVar('T')


class ConfigError(Exception):
    """Raised when a critical config section fails to build from YAML data."""
    pass


class FrozenConfigError(ConfigError):
    """Raised when attempting to modify a frozen config after pipeline start."""
    pass


# Sections whose construction failure should raise ConfigError instead of
# silently returning empty defaults.  These are the sections without which
# the pipeline cannot produce meaningful output.
CRITICAL_SECTIONS = frozenset({
    'download', 'downloading', 'matching', 'output',
    'transcription', 'embedding', 'pipeline', 'llm',
    'healing', 'logging', 'cache', 'api_keys', 'project',
})

# =============================================================================
# PERFORMANCE TRACKING
# =============================================================================

_config_metrics = {
    'load_count': 0,
    'load_time_total_ms': 0.0,
    'cache_hits': 0,
    'cache_misses': 0,
    'reload_count': 0,
    'validation_errors': 0,
}


def get_config_metrics() -> Dict[str, Any]:
    """Get configuration performance metrics"""
    return _config_metrics.copy()


# =============================================================================
# MAIN CONFIG CLASS
# =============================================================================

def _frozen_setattr(self: Any, name: str, value: Any) -> None:
    """Shared __setattr__ for section dataclasses that support freezing."""
    if name.startswith('_') or not getattr(self, '_frozen', False):
        object.__setattr__(self, name, value)
    else:
        raise FrozenConfigError(
            f"Cannot set '{name}' on frozen {type(self).__name__}. "
            f"Config is immutable after pipeline start."
        )


def _freeze_dataclass(obj: Any) -> None:
    """Recursively set _frozen=True on a dataclass and its nested dataclass fields.

    Also installs _frozen_setattr as __setattr__ on the class so that
    attribute writes are blocked for all instances of that class while frozen.
    """
    cls = type(obj)
    # Install frozen __setattr__ if not already installed
    if getattr(cls, '_original_setattr', None) is None:
        cls._original_setattr = cls.__dict__.get('__setattr__', None)
        cls.__setattr__ = _frozen_setattr
    object.__setattr__(obj, '_frozen', True)
    for f in fields(obj):
        child = getattr(obj, f.name)
        if is_dataclass(child) and not isinstance(child, type):
            _freeze_dataclass(child)


def _unfreeze_dataclass(obj: Any) -> None:
    """Recursively set _frozen=False on a dataclass and its nested dataclass fields.

    Restores the original __setattr__ on the class.
    """
    cls = type(obj)
    object.__setattr__(obj, '_frozen', False)
    # Restore original __setattr__
    original = getattr(cls, '_original_setattr', None)
    if original is not None:
        cls.__setattr__ = original
        del cls._original_setattr
    elif hasattr(cls, '_original_setattr'):
        # _original_setattr was None (no custom __setattr__ existed),
        # remove our override to restore default behavior
        if '__setattr__' in cls.__dict__:
            delattr(cls, '__setattr__')
        if '_original_setattr' in cls.__dict__:
            delattr(cls, '_original_setattr')


@dataclass
class Config:
    """
    Master configuration class - Single Source of Truth

    Chain-of-thought: All components access settings through this class
    Reasoning: Centralizes configuration, enables validation and hot-reload
    Decision: Load from config.yaml using Config.from_yaml(path)

    Usage:
        config = Config.from_yaml("config.yaml")
        threshold = config.matching.min_confidence
    """
    # Project settings
    project: ProjectConfig = field(default_factory=ProjectConfig)

    # Section configs
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    negative_matching: NegativeMatchingConfig = field(default_factory=NegativeMatchingConfig)
    remix: RemixConfig = field(default_factory=RemixConfig)
    zero_download_remix: ZeroDownloadRemixConfig = field(default_factory=ZeroDownloadRemixConfig)
    image_search: ImageSearchConfig = field(default_factory=ImageSearchConfig)
    keyword: KeywordConfig = field(default_factory=KeywordConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    enhanced: EnhancedFeaturesConfig = field(default_factory=EnhancedFeaturesConfig)
    downloading: DownloadingConfig = field(default_factory=DownloadingConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    duration_tiers: DurationTiersConfig = field(default_factory=DurationTiersConfig)
    stock_footage: StockFootageConfig = field(default_factory=StockFootageConfig)
    silent_video: SilentVideoConfig = field(default_factory=SilentVideoConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    multi_style: MultiStyleConfig = field(default_factory=MultiStyleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    global_cache: GlobalCacheConfig = field(default_factory=GlobalCacheConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    broll: BrollConfig = field(default_factory=BrollConfig)
    healing: HealingConfig = field(default_factory=HealingConfig)
    iterative_matching: IterativeMatchingConfig = field(default_factory=IterativeMatchingConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Convenience paths (resolved at load time)
    project_dir: str = "."
    downloaded_videos_dir: str = "downloaded_videos"
    otio_output_dir: str = "output"
    cache_dir: str = ".cache"

    # API key aliases (for matching.py compatibility)
    # These are populated from api_keys in __post_init__
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    # Metadata (private)
    _config_path: str = ""
    _config_hash: str = ""
    _loaded_at: str = ""
    _load_time_ms: float = 0.0

    # Internal frozen state (not a dataclass field to avoid __init__ issues)
    _frozen: bool = False

    def __setattr__(self, name: str, value: Any):
        """Prevent attribute mutation when config is frozen."""
        # Allow setting private/internal attrs and the _frozen flag itself
        if name.startswith('_') or not getattr(self, '_frozen', False):
            object.__setattr__(self, name, value)
        else:
            raise FrozenConfigError(
                f"Cannot set '{name}' on frozen Config. "
                f"Call config.unfreeze() first (test scenarios only)."
            )

    def freeze(self):
        """Mark config as immutable. Called by Pipeline.run() before first stage."""
        object.__setattr__(self, '_frozen', True)
        # Also freeze all section dataclasses
        for f in fields(self):
            section = getattr(self, f.name)
            if is_dataclass(section) and not isinstance(section, type):
                _freeze_dataclass(section)

    def unfreeze(self):
        """Unmark config as immutable. For test scenarios that need mutation."""
        object.__setattr__(self, '_frozen', False)
        # Also unfreeze all section dataclasses
        for f in fields(self):
            section = getattr(self, f.name)
            if is_dataclass(section) and not isinstance(section, type):
                _unfreeze_dataclass(section)

    def __post_init__(self):
        """Initialize after dataclass creation"""
        self._loaded_at = datetime.now().isoformat()
        self._convert_nested_configs()
        self._resolve_paths()
        self._populate_api_keys()

    def _convert_nested_configs(self):
        """Convert dict configs to dataclass instances (per Rule 2).

        When loading from YAML, nested dataclass fields come in as dicts.
        This method converts them to their proper dataclass types.

        Also handles the case where a dataclass instance has nested dict
        fields that need conversion (e.g., after merge_config updates).

        Iterates over ALL dataclass fields on Config, not just healing,
        so that merge_config replacing entire sections with dicts is handled.
        """
        for f in fields(self):
            # Use default_factory to get the actual type (annotations are
            # strings due to `from __future__ import annotations`)
            if f.default_factory is MISSING:
                continue
            factory = f.default_factory
            if not is_dataclass(factory):
                continue
            value = getattr(self, f.name)
            if isinstance(value, dict):
                setattr(self, f.name, factory(**value))
            elif hasattr(value, '__post_init__'):
                # Re-run __post_init__ to convert any nested dicts
                # (e.g., watcher dict -> WatcherConfig after merge)
                value.__post_init__()

    def _populate_api_keys(self):
        """Populate top-level API key aliases from api_keys config"""
        self.gemini_api_key = self.api_keys.gemini_api_key
        self.anthropic_api_key = self.api_keys.anthropic_api_key
        self.voyage_api_key = self.api_keys.voyage_api_key

    def _resolve_paths(self):
        """Resolve relative paths to absolute based on project_dir.

        This is the single source of truth for all config path resolution.
        Handles: convenience paths (downloaded_videos_dir, otio_output_dir),
        section paths (output.output_dir, download.download_dir,
        transcription.cache_dir, cache.cache_dir, logging.log_dir).

        Paths that are already absolute are left unchanged, ensuring
        idempotent behavior when called multiple times.
        Mixed forward/backslash separators are normalized via pathlib.
        """
        base = Path(self.project_dir).resolve()

        # Determine video directory with priority:
        # 1. pipeline.video_source_dir (explicit override)
        # 2. download.root_dir (short path mode like E:/v)
        # 3. downloading.output_dir (legacy fallback)
        video_source = getattr(self.pipeline, 'video_source_dir', '') or ''
        if video_source:
            video_dir = video_source
        elif getattr(self.download, 'root_dir', ''):
            root_dir = Path(self.download.root_dir)
            # Use project name, truncated to max_name_display_length for short paths
            project_name = getattr(self.project, 'name', base.name) or base.name
            max_len = getattr(self.project, 'max_name_display_length', 15)
            project_name = project_name[:max_len]
            video_dir = str(root_dir / project_name)
        else:
            video_dir = self.downloading.output_dir

        # Resolve convenience path attributes on the Config object itself
        path_attrs = [
            ('downloaded_videos_dir', video_dir),
            ('otio_output_dir', self.output.output_dir),
        ]

        for attr_name, config_value in path_attrs:
            if config_value and not Path(config_value).is_absolute():
                setattr(self, attr_name, str(base / config_value))
            else:
                setattr(self, attr_name, config_value)

        # Resolve section-level paths (relative -> project-relative).
        # Each tuple: (section_obj, field_name)
        section_paths = [
            (self.output, 'output_dir'),
            (self.download, 'download_dir'),
            (self.transcription, 'cache_dir'),
            (self.cache, 'cache_dir'),
            (self.logging, 'log_dir'),
        ]

        for section_obj, field_name in section_paths:
            value = getattr(section_obj, field_name, None)
            if value and not Path(value).is_absolute():
                setattr(section_obj, field_name, str(base / value))

    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file with CSafeLoader for performance.

        Performance: ~50ms for typical config file
        """
        global _config_metrics
        start_time = time.perf_counter()

        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = cls()
            config._config_path = str(config_path)
            _config_metrics['cache_misses'] += 1
            return config

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                # Use CSafeLoader for 40-60% faster parsing
                data = yaml.load(f, Loader=SafeLoader) or {}
        except yaml.YAMLError as e:
            # Extract line/column from YAML error marks
            location = ""
            if hasattr(e, 'problem_mark') and e.problem_mark is not None:
                mark = e.problem_mark
                location = f" at line {mark.line + 1}, column {mark.column + 1}"
            problem = getattr(e, 'problem', str(e))
            _config_metrics['validation_errors'] += 1
            raise ConfigError(
                f"YAML syntax error in '{config_path}'{location}: {problem}"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            _config_metrics['validation_errors'] += 1
            return cls()

        # Compute hash for change detection
        config_hash = hashlib.md5(
            yaml.dump(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Build config object
        config = cls._from_dict(data)
        config._config_path = str(config_path)
        config._config_hash = config_hash

        # Track metrics
        load_time = (time.perf_counter() - start_time) * 1000
        config._load_time_ms = load_time
        _config_metrics['load_count'] += 1
        _config_metrics['load_time_total_ms'] += load_time

        logger.info(f"Loaded config from {config_path} in {load_time:.1f}ms (hash: {config_hash})")
        if YAML_FAST:
            logger.debug("Using CSafeLoader (C-based) for optimized parsing")

        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build Config from dictionary with nested dataclass handling"""
        config = cls()

        # Project settings
        if 'project' in data:
            config.project = cls._build_dataclass(ProjectConfig, data['project'], section_name='project')

        config.project_dir = data.get('project_dir', config.project_dir)

        # Section mapping: yaml_key -> (dataclass_type, attr_name)
        section_mapping = {
            'transcription': (TranscriptionConfig, 'transcription'),
            'embedding': (EmbeddingConfig, 'embedding'),
            'indexing': (IndexingConfig, 'indexing'),
            'vision': (VisionConfig, 'vision'),
            'scene_detection': (SceneDetectionConfig, 'scene_detection'),
            'audio_analysis': (AudioAnalysisConfig, 'audio_analysis'),
            'matching': (MatchingConfig, 'matching'),
            'negative_matching': (NegativeMatchingConfig, 'negative_matching'),
            'remix': (RemixConfig, 'remix'),
            'zero_download_remix': (ZeroDownloadRemixConfig, 'zero_download_remix'),
            'image_search': (ImageSearchConfig, 'image_search'),
            'keyword': (KeywordConfig, 'keyword'),
            'llm': (LLMConfig, 'llm'),
            'enhanced': (EnhancedFeaturesConfig, 'enhanced'),
            'downloading': (DownloadingConfig, 'downloading'),
            'download': (DownloadConfig, 'download'),
            'stock_footage': (StockFootageConfig, 'stock_footage'),
            'deduplication': (DeduplicationConfig, 'deduplication'),
            'output': (OutputConfig, 'output'),
            'multi_style': (MultiStyleConfig, 'multi_style'),
            'logging': (LoggingConfig, 'logging'),
            'cache': (CacheConfig, 'cache'),
            'pipeline': (PipelineConfig, 'pipeline'),
            'api_keys': (APIKeysConfig, 'api_keys'),
            'healing': (HealingConfig, 'healing'),
            'iterative_matching': (IterativeMatchingConfig, 'iterative_matching'),
            'rate_limit': (RateLimitConfig, 'rate_limit'),
            'broll': (BrollConfig, 'broll'),
            'global_cache': (GlobalCacheConfig, 'global_cache'),
            'silent_video': (SilentVideoConfig, 'silent_video'),
        }

        for yaml_key, (dataclass_type, attr_name) in section_mapping.items():
            section_data = data.get(yaml_key, {})
            if section_data:
                section_config = cls._build_dataclass(dataclass_type, section_data, section_name=yaml_key)
                setattr(config, attr_name, section_config)

        # Handle duration_tiers specially (nested structure)
        if 'duration_tiers' in data:
            dt_data = data['duration_tiers']
            if not isinstance(dt_data, dict):
                raise ConfigError(
                    f"Critical config section 'duration_tiers' (DurationTiersConfig) "
                    f"failed to build: expected dict, got {type(dt_data).__name__}"
                )
            try:
                config.duration_tiers = cls._build_duration_tiers(dt_data)
            except ConfigError:
                raise
            except (TypeError, KeyError, AttributeError, ValueError) as e:
                raise ConfigError(
                    f"Critical config section 'duration_tiers' (DurationTiersConfig) "
                    f"failed to build: {e}. Data provided: {list(dt_data.keys())}"
                ) from e

        return config

    @staticmethod
    def _build_dataclass(dataclass_type: Type[T], data: Dict, section_name: Optional[str] = None) -> T:
        """Build a dataclass from dict, handling missing/extra fields and nested dataclasses.

        Args:
            dataclass_type: The dataclass type to construct.
            data: Dictionary of field values from YAML.
            section_name: Optional config section name (e.g. 'download').
                When provided, critical sections raise ConfigError on failure
                while optional sections fall back to empty defaults with a
                debug-level log message.
        """
        if not data:
            return dataclass_type()

        # Get type hints to resolve string annotations (from __future__ import annotations)
        try:
            from typing import get_type_hints
            type_hints = get_type_hints(dataclass_type)
        except Exception:
            type_hints = {}

        valid_fields = {f.name: f for f in fields(dataclass_type)}
        filtered_data = {}

        for key, value in data.items():
            if key in valid_fields:
                field_info = valid_fields[key]

                # Use resolved type hints if available, otherwise fall back to field.type
                field_type = type_hints.get(key, field_info.type)

                # Handle string type annotations that couldn't be resolved
                if isinstance(field_type, str):
                    # Try to find the type in the module's namespace
                    module = sys.modules.get(__name__, None)
                    if module and hasattr(module, field_type):
                        field_type = getattr(module, field_type)

                # Check if this field is a nested dataclass
                # Handle Optional types and get the actual type
                origin = getattr(field_type, '__origin__', None)
                if origin is not None:
                    # For Optional[X], Union[X, None], etc.
                    args = getattr(field_type, '__args__', ())
                    if args:
                        field_type = args[0]

                # Check if the field type is a dataclass
                if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
                    # Recursively build nested dataclass
                    filtered_data[key] = Config._build_dataclass(field_type, value)
                else:
                    filtered_data[key] = value
            else:
                is_critical = section_name is not None and section_name in CRITICAL_SECTIONS
                valid_names = list(valid_fields.keys())
                close = difflib.get_close_matches(key, valid_names, n=1, cutoff=0.6)
                suggestion = f" (did you mean '{close[0]}'?)" if close else ""
                if is_critical:
                    logger.warning(
                        f"Unknown config key '{key}' in critical section "
                        f"'{section_name}'{suggestion}"
                    )
                else:
                    logger.debug(
                        f"Ignoring unknown config field in "
                        f"{dataclass_type.__name__}: {key}{suggestion}"
                    )

        # Track unexpected keys with suggestions for error context
        unexpected_keys = [k for k in data.keys() if k not in valid_fields]
        unexpected_with_suggestions = []
        for uk in unexpected_keys:
            close = difflib.get_close_matches(uk, list(valid_fields.keys()), n=1, cutoff=0.6)
            if close:
                unexpected_with_suggestions.append(f"'{uk}' (did you mean '{close[0]}'?)")
            else:
                unexpected_with_suggestions.append(f"'{uk}'")

        try:
            return dataclass_type(**filtered_data)
        except TypeError as e:
            failed_fields = list(data.keys())
            is_critical = section_name is not None and section_name in CRITICAL_SECTIONS

            unexpected_ctx = ""
            if unexpected_with_suggestions:
                unexpected_ctx = f" Unexpected keys: {', '.join(unexpected_with_suggestions)}."

            if is_critical:
                raise ConfigError(
                    f"Critical config section '{section_name}' ({dataclass_type.__name__}) "
                    f"failed to build: {e}.{unexpected_ctx} "
                    f"Fields provided: {failed_fields}"
                ) from e
            else:
                # Optional section — fall back to empty defaults
                logger.debug(
                    f"Optional config section '{section_name or dataclass_type.__name__}' "
                    f"falling back to defaults: {e}.{unexpected_ctx} "
                    f"Fields that failed: {failed_fields}"
                )
                return dataclass_type()

    @staticmethod
    def _build_duration_tiers(data: Dict) -> DurationTiersConfig:
        """Build duration tiers from nested config"""
        tiers = DurationTiersConfig()

        for tier_name in ['short', 'medium', 'long', 'longer']:
            if tier_name in data:
                tier_data = data[tier_name]
                tier_config = DurationTierConfig(
                    min_seconds=tier_data.get('min', 0),
                    max_seconds=tier_data.get('max', 120),
                    videos_per_keyword=tier_data.get('count', 5),
                    max_total=tier_data.get('max_total', 0)
                )
                setattr(tiers, tier_name, tier_config)

        return tiers

    def to_dict(self, redact_sensitive: bool = True) -> Dict[str, Any]:
        """Convert config to nested dict suitable for YAML serialization.

        Args:
            redact_sensitive: If True (default), redact API key values
                with '***REDACTED***'. Set False to include actual values.

        Returns:
            Nested dictionary of all config sections.
        """
        result: Dict[str, Any] = {
            'project': asdict(self.project),
            'project_dir': self.project_dir,
        }

        # All sections from _from_dict's section_mapping
        sections = [
            'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'audio_analysis', 'matching',
            'negative_matching', 'remix', 'zero_download_remix',
            'image_search', 'keyword', 'llm', 'enhanced',
            'downloading', 'download', 'stock_footage', 'deduplication',
            'output', 'multi_style', 'logging', 'cache', 'pipeline',
            'api_keys', 'healing', 'iterative_matching', 'rate_limit',
            'broll', 'global_cache', 'silent_video',
        ]

        for section in sections:
            section_config = getattr(self, section, None)
            if section_config and is_dataclass(section_config):
                result[section] = asdict(section_config)

        # Duration tiers (nested structure with non-standard field names)
        if self.duration_tiers and is_dataclass(self.duration_tiers):
            dt = self.duration_tiers
            dt_dict = {}
            for tier_name in ['short', 'medium', 'long', 'longer']:
                tier = getattr(dt, tier_name, None)
                if tier and is_dataclass(tier):
                    dt_dict[tier_name] = {
                        'min': tier.min_seconds,
                        'max': tier.max_seconds,
                        'count': tier.videos_per_keyword,
                        'max_total': tier.max_total,
                    }
            if dt_dict:
                result['duration_tiers'] = dt_dict

        # Redact sensitive fields
        if redact_sensitive and 'api_keys' in result:
            redacted = {}
            for key, value in result['api_keys'].items():
                redacted[key] = '***REDACTED***' if value else ''
            result['api_keys'] = redacted

        return result

    def to_yaml(self, output_path: str = None, *, redact_sensitive: bool = True) -> str:
        """Serialize effective config to YAML.

        Args:
            output_path: If provided, write YAML to this file path.
            redact_sensitive: If True (default), redact API keys.

        Returns:
            YAML string of the effective config.
        """
        data = self.to_dict(redact_sensitive=redact_sensitive)
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            logger.info(f"Saved config to {output_path}")

        return yaml_str

    def reload(self) -> bool:
        """
        Reload config from file if changed.

        Returns:
            True if config was reloaded
        """
        global _config_metrics

        if not self._config_path:
            return False

        config_path = Path(self._config_path)
        if not config_path.exists():
            return False

        # Check hash for changes
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.load(f, Loader=SafeLoader) or {}

        new_hash = hashlib.md5(
            yaml.dump(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        if new_hash == self._config_hash:
            return False

        # Reload
        logger.info(f"Config changed, reloading (old: {self._config_hash}, new: {new_hash})")
        new_config = Config.from_yaml(self._config_path)

        # Copy all attributes
        for attr in dir(new_config):
            if not attr.startswith('_') and not callable(getattr(new_config, attr)):
                setattr(self, attr, getattr(new_config, attr))

        self._config_hash = new_hash
        self._loaded_at = datetime.now().isoformat()
        _config_metrics['reload_count'] += 1

        return True

    def validate(self) -> List[str]:
        """
        Validate configuration with clear error messages.

        Checks:
        - Required API keys for enabled features
        - Value ranges (min/max, 0-1 percentages)
        - Enum values (valid options)
        - Constraint relationships (min <= max)

        Returns:
            List of validation error messages (empty if valid)
        """
        global _config_metrics
        errors = []

        # API key checks
        api_checks = [
            (self.matching.primary_provider == "gemini",
             self.api_keys.gemini_api_key,
             "GEMINI_API_KEY required for gemini matching"),
            (self.matching.secondary_provider == "anthropic",
             self.api_keys.anthropic_api_key,
             "ANTHROPIC_API_KEY required for anthropic matching (secondary)"),
            (self.embedding.provider == "gemini",
             self.api_keys.gemini_api_key,
             "GEMINI_API_KEY required for gemini embeddings"),
            (self.stock_footage.pexels_enabled,
             self.api_keys.pexels_api_key,
             "PEXELS_API_KEY required for Pexels stock footage"),
            (self.stock_footage.pixabay_enabled,
             self.api_keys.pixabay_api_key,
             "PIXABAY_API_KEY required for Pixabay stock footage"),
        ]

        for condition, key, message in api_checks:
            if condition and not key:
                errors.append(message)

        # Value range checks
        range_checks = [
            (0 <= self.matching.min_confidence <= 1,
             f"matching.min_confidence must be 0-1, got {self.matching.min_confidence}"),
            (self.matching.max_clip_reuse >= 0,
             f"matching.max_clip_reuse must be >= 0, got {self.matching.max_clip_reuse}"),
            (self.transcription.max_workers >= 1,
             f"transcription.max_workers must be >= 1, got {self.transcription.max_workers}"),
            (self.embedding.batch_size >= 1,
             f"embedding.batch_size must be >= 1, got {self.embedding.batch_size}"),
            (self.keyword.max_keywords >= 1,
             f"keyword.max_keywords must be >= 1, got {self.keyword.max_keywords}"),
        ]

        for valid, message in range_checks:
            if not valid:
                errors.append(message)

        # Enum value checks
        errors.extend(self._validate_enums())

        # Constraint relationship checks
        errors.extend(self._validate_constraints())

        _config_metrics['validation_errors'] += len(errors)

        return errors

    def _validate_enums(self) -> List[str]:
        """Validate enum-like fields have valid values"""
        errors = []

        # Location matching filter level
        location_config = safe_get_config_value(self.matching, 'location_matching')
        if location_config:
            filter_level = safe_get_config_value(location_config, 'hard_filter_level', 'city')

            valid_levels = {'city', 'state', 'country', 'continent'}
            if filter_level not in valid_levels:
                errors.append(
                    f"matching.location_matching.hard_filter_level must be one of "
                    f"{valid_levels}, got '{filter_level}'"
                )

        # Face preference
        face_pref = safe_get_config_value(self.enhanced, 'face_preference', 'neutral')
        valid_face_prefs = {'prefer_faces', 'avoid_faces', 'neutral'}
        if face_pref not in valid_face_prefs:
            errors.append(
                f"enhanced.face_preference must be one of {valid_face_prefs}, "
                f"got '{face_pref}'"
            )

        # Audio quality (0-9)
        audio_first = safe_get_config_value(self.download, 'audio_first')
        if audio_first:
            audio_quality = safe_get_config_value(audio_first, 'audio_quality', 5)

            if not (0 <= audio_quality <= 9):
                errors.append(
                    f"download.audio_first.audio_quality must be 0-9, "
                    f"got {audio_quality}"
                )

        # Embedding provider
        valid_providers = EmbeddingConfig.KNOWN_PROVIDERS
        if self.embedding.provider not in valid_providers:
            errors.append(
                f"embedding.provider must be one of {valid_providers}, "
                f"got '{self.embedding.provider}'"
            )

        # Matching provider
        valid_matching = {'gemini', 'anthropic', 'local', 'embedding_only'}
        if self.matching.primary_provider not in valid_matching:
            errors.append(
                f"matching.primary_provider must be one of {valid_matching}, "
                f"got '{self.matching.primary_provider}'"
            )

        return errors

    def _validate_constraints(self) -> List[str]:
        """Validate constraint relationships between config values"""
        errors = []

        # embedding_candidates should be >= num_alternatives * 3 for diversity
        num_alts = safe_get_config_value(self.output, 'num_alternatives', 2)
        embed_candidates = safe_get_config_value(self.matching, 'embedding_candidates', 50)
        min_required = num_alts * 3
        if embed_candidates < min_required:
            errors.append(
                f"matching.embedding_candidates ({embed_candidates}) should be >= "
                f"num_alternatives * 3 ({min_required}) for proper diversity"
            )

        # split_otio requires generate_otio
        if safe_get_config_value(self.output, 'split_otio', False) and not safe_get_config_value(self.output, 'generate_otio', True):
            errors.append(
                "output.split_otio=true requires output.generate_otio=true"
            )

        # Pause split thresholds
        pause_split = safe_get_config_value(self.transcription, 'pause_split')
        if pause_split:
            min_gap = safe_get_config_value(pause_split, 'min_gap_ms', 300)
            min_seg = safe_get_config_value(pause_split, 'min_segment_duration', 0.5)

            # min_gap_ms (in ms) should be greater than min_segment_duration (in s) * 1000
            if min_gap < min_seg * 1000:
                errors.append(
                    f"transcription.pause_split.min_gap_ms ({min_gap}ms) should be >= "
                    f"min_segment_duration ({min_seg}s = {min_seg * 1000}ms)"
                )

        # Matching confidence threshold ordering:
        # low_confidence_threshold <= ambiguous_threshold <= min_confidence <= high_confidence_threshold
        low_conf = safe_get_config_value(self.matching, 'low_confidence_threshold', 0.5)
        ambig = safe_get_config_value(self.matching, 'ambiguous_threshold', 0.6)
        min_conf = safe_get_config_value(self.matching, 'min_confidence', 0.7)
        high_conf = safe_get_config_value(self.matching, 'high_confidence_threshold', 0.85)

        if low_conf > ambig:
            errors.append(
                f"matching.low_confidence_threshold ({low_conf}) should be <= "
                f"ambiguous_threshold ({ambig})"
            )
        if ambig > min_conf:
            errors.append(
                f"matching.ambiguous_threshold ({ambig}) should be <= "
                f"min_confidence ({min_conf})"
            )
        if min_conf > high_conf:
            errors.append(
                f"matching.min_confidence ({min_conf}) should be <= "
                f"high_confidence_threshold ({high_conf})"
            )

        # Output positive value checks
        frame_rate = safe_get_config_value(self.output, 'frame_rate', 30.0)
        if frame_rate is not None and frame_rate <= 0:
            errors.append(
                f"output.frame_rate must be > 0, got {frame_rate}"
            )
        time_scale = safe_get_config_value(self.output, 'time_scale_factor', 1.0)
        if time_scale is not None and time_scale <= 0:
            errors.append(
                f"output.time_scale_factor must be > 0, got {time_scale}"
            )

        # Chapter grouping constraints
        chapter_grouping = safe_get_config_value(self.matching, 'chapter_grouping')
        if chapter_grouping:
            # chapter_topic_match_boost: list of exactly 2 floats [min, max] with min <= max
            boost = safe_get_config_value(chapter_grouping, 'chapter_topic_match_boost')
            if boost is not None:
                if not isinstance(boost, list) or len(boost) != 2:
                    errors.append(
                        f"matching.chapter_grouping.chapter_topic_match_boost must be a "
                        f"list of exactly 2 floats [min, max], got {boost}"
                    )
                else:
                    try:
                        boost_min, boost_max = float(boost[0]), float(boost[1])
                        if boost_min > boost_max:
                            errors.append(
                                f"matching.chapter_grouping.chapter_topic_match_boost: "
                                f"min ({boost_min}) must be <= max ({boost_max})"
                            )
                    except (TypeError, ValueError):
                        errors.append(
                            f"matching.chapter_grouping.chapter_topic_match_boost values "
                            f"must be numeric, got {boost}"
                        )

            # chapter_topic_mismatch_penalty: must be negative (it's a penalty)
            mismatch_penalty = safe_get_config_value(chapter_grouping, 'chapter_topic_mismatch_penalty')
            if mismatch_penalty is not None and mismatch_penalty >= 0:
                errors.append(
                    f"matching.chapter_grouping.chapter_topic_mismatch_penalty "
                    f"({mismatch_penalty}) must be negative (it's a penalty)"
                )

        # Context enrichment constraints
        context_enrichment = safe_get_config_value(self.matching, 'context_enrichment')
        if context_enrichment:
            # max_description_length must be positive
            max_desc_len = safe_get_config_value(context_enrichment, 'max_description_length')
            if max_desc_len is not None and max_desc_len <= 0:
                errors.append(
                    f"matching.context_enrichment.max_description_length "
                    f"({max_desc_len}) must be positive"
                )

            # Cross-section: title_enriched_embeddings requires embedding provider
            title_enriched = safe_get_config_value(context_enrichment, 'title_enriched_embeddings', False)
            if title_enriched:
                embedding_provider = safe_get_config_value(self.embedding, 'provider', '')
                if not embedding_provider:
                    errors.append(
                        "matching.context_enrichment.title_enriched_embeddings=true "
                        "requires embedding.provider to be configured"
                    )

        # Duration tier min <= max for all configured tiers
        duration_tiers = safe_get_config_value(self, 'duration_tiers')
        if duration_tiers:
            for tier_name in ['short', 'medium', 'long', 'longer']:
                tier = safe_get_config_value(duration_tiers, tier_name)
                if tier:
                    tier_min = safe_get_config_value(tier, 'min_seconds', 0)
                    tier_max = safe_get_config_value(tier, 'max_seconds', 0)
                    if tier_min > 0 and tier_max > 0 and tier_min > tier_max:
                        errors.append(
                            f"duration_tiers.{tier_name}: min_seconds ({tier_min}) "
                            f"should be <= max_seconds ({tier_max})"
                        )

        return errors

    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get a nested config value by dot-notation path.

        Example:
            config.get_nested("matching.min_confidence")
            config.get_nested("duration_tiers.short.min_seconds")
        """
        parts = path.split(".")
        obj = self

        _sentinel = object()
        for part in parts:
            result = safe_get_config_value(obj, part, _sentinel)
            if result is _sentinel:
                if self.logging.log_config_access:
                    logger.debug(f"Config path not found: {path}")
                return default
            obj = result

        return obj


# =============================================================================
# GLOBAL CONFIG INSTANCE (Thread-Safe Singleton)
# =============================================================================

_global_config: Optional[Config] = None
_config_lock = threading.Lock()


def get_config() -> Config:
    """Get the global config instance (creates default if not loaded)"""
    global _global_config, _config_metrics

    with _config_lock:
        if _global_config is None:
            _global_config = Config()
            _config_metrics['cache_misses'] += 1
            logger.warning("Using default config - call load_config() to load from file")
        else:
            _config_metrics['cache_hits'] += 1

        return _global_config


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load config from file and set as global instance.

    Args:
        config_path: Path to config.yaml

    Returns:
        Loaded Config object
    """
    global _global_config

    with _config_lock:
        _global_config = Config.from_yaml(config_path)

        # Validate and warn
        errors = _global_config.validate()
        for error in errors:
            logger.warning(f"Config validation: {error}")

        return _global_config


def set_config(config: Config):
    """Set the global config instance"""
    global _global_config

    with _config_lock:
        _global_config = config


def reload_config() -> bool:
    """Reload global config if file changed"""
    global _global_config

    with _config_lock:
        if _global_config:
            return _global_config.reload()
        return False


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider from config"""
    config = get_config()
    key_map = {
        "gemini": config.api_keys.gemini_api_key,
        "anthropic": config.api_keys.anthropic_api_key,
        "voyage": config.api_keys.voyage_api_key,
        "pexels": config.api_keys.pexels_api_key,
        "pixabay": config.api_keys.pixabay_api_key,
        "unsplash": config.api_keys.unsplash_api_key,
    }
    return key_map.get(provider.lower())


def ensure_dirs(config: Config = None):
    """Create necessary directories from config"""
    config = config or get_config()

    dirs = [
        config.cache.cache_dir,
        config.output.output_dir,
        config.downloading.output_dir,
        config.logging.log_dir,
        config.downloaded_videos_dir,
        config.otio_output_dir,
    ]

    for dir_path in dirs:
        if dir_path:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def log_hardcoded_warning(component: str, value_name: str, value: Any):
    """Log warning when a component uses hardcoded value"""
    config = get_config()
    if config.logging.warn_on_hardcoded:
        logger.warning(
            f"HARDCODED VALUE in {component}: {value_name}={value} - "
            f"Consider adding to config.yaml"
        )


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Export section configs for backward compatibility
__all__ = [
    # Main config and functions
    'Config', 'ConfigError', 'FrozenConfigError',
    'load_config', 'get_config', 'set_config', 'reload_config',
    'ensure_dirs', 'get_api_key', 'get_config_metrics', 'log_hardcoded_warning',
    # Section configs (for backward compatibility)
    'TranscriptionConfig', 'EmbeddingConfig', 'MatchingConfig', 'OutputConfig',
    'KeywordConfig', 'DownloadingConfig', 'LoggingConfig', 'CacheConfig',
]
