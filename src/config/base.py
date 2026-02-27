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
from typing import Optional, List, Dict, Any, Tuple, Union, TypeVar, Type, Callable
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
from .utils import safe_get_config_value, ConfigMigration, CURRENT_CONFIG_VERSION

# Import logging templates for structured logging
from ..logging_templates import log_error_with_context

from .sections import (
    # Infrastructure
    LoggingConfig,
    CacheConfig,
    GlobalCacheConfig,
    PipelineConfig,
    APIKeysConfig,
    HealingConfig,
    UnifiedErrorAggregationConfig,
    CrossKeywordRetryLearningConfig,
    ValidationWebhookConfig,
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
    # Search
    SearchBudgetConfig,
    VideoSearchConfig,
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
    # Test mode
    TestModeConfig,
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
    unified_error_aggregation: UnifiedErrorAggregationConfig = field(default_factory=UnifiedErrorAggregationConfig)
    cross_keyword_learning: CrossKeywordRetryLearningConfig = field(default_factory=CrossKeywordRetryLearningConfig)
    validation_webhook: ValidationWebhookConfig = field(default_factory=ValidationWebhookConfig)
    iterative_matching: IterativeMatchingConfig = field(default_factory=IterativeMatchingConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    test_mode: TestModeConfig = field(default_factory=TestModeConfig)
    # Search
    search_budget: SearchBudgetConfig = field(default_factory=SearchBudgetConfig)
    video_search: VideoSearchConfig = field(default_factory=VideoSearchConfig)

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
    # Per-phase loading timings: {phase_name: duration_ms}
    _load_timings: Dict[str, float] = field(default_factory=dict, repr=False)
    # Config drift tracking: field-level hashes at startup for runtime drift detection
    _field_hashes: Dict[str, str] = field(default_factory=dict, repr=False)
    _drift_warnings_logged: bool = False
    # Version history: list of {version, date, notes} entries tracking config changes
    _version_history: List[Dict[str, str]] = field(default_factory=list, repr=False)

    # Section-level schema versions: {section_name: version} for granular migrations
    _section_versions: Dict[str, str] = field(default_factory=dict, repr=False)

    # Internal frozen state (not a dataclass field to avoid __init__ issues)
    _frozen: bool = False

    # Callback registry for hot-reload notifications
    # Structure: {section_name: [callbacks]} or {'*': [global_callbacks]}
    _change_callbacks: Dict[str, List[Callable]] = field(default_factory=dict, repr=False)

    # Usage tracking: {field_name: access_count} for analytics
    _usage_tracker: Dict[str, int] = field(default_factory=dict, repr=False)

    # Raw config data for profile merging
    _config_data: Dict[str, Any] = field(default_factory=dict, repr=False)

    # Profile applied (if any)
    _profile_applied: str = ""

    # Value sources: {field_path: 'default|yaml|profile|env'} for tracing config value origins
    # Precedence order: default < yaml < profile < env
    _value_sources: Dict[str, str] = field(default_factory=dict, repr=False)

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

    def __getattribute__(self, name: str) -> Any:
        """Track config field access for analytics."""
        # Call parent __getattribute__ to get the actual value
        result = object.__getattribute__(self, name)
        # Track non-private attributes (skip internal/dunder methods)
        if not name.startswith('_') and not hasattr(result, '__call__'):
            tracker = object.__getattribute__(self, '_usage_tracker')
            tracker[name] = tracker.get(name, 0) + 1
        return result

    def track_usage(self, field_name: str) -> None:
        """
        Manually track usage of a config field.

        Args:
            field_name: The name of the config field being accessed

        Example:
            config.track_usage('matching.min_confidence')
        """
        if not field_name.startswith('_'):
            self._usage_tracker[field_name] = self._usage_tracker.get(field_name, 0) + 1

    def get_usage_stats(self) -> Dict[str, int]:
        """
        Get usage statistics for config fields.

        Returns:
            Dict mapping field names to their access counts

        Example:
            stats = config.get_usage_stats()
            # {'matching': 10, 'download': 5, 'project': 2}
        """
        return self._usage_tracker.copy()

    def reset_usage_stats(self) -> None:
        """Reset usage statistics to empty state."""
        self._usage_tracker.clear()

    def get_value_source(self, field_path: str) -> Optional[Dict[str, Any]]:
        """
        Get the source of a config value by field path.

        Args:
            field_path: Dot-separated path to config field (e.g., 'matching.min_confidence')

        Returns:
            Dict with keys:
                - 'source': 'default' | 'yaml' | 'profile' | 'env'
                - 'value': The current value
                - 'section': Section name (if applicable)
            Returns None if field not found

        Example:
            source = config.get_value_source('matching.min_confidence')
            # {'source': 'env', 'value': 0.85, 'section': 'matching'}

            source = config.get_value_source('project.version')
            # {'source': 'yaml', 'value': '4.0.0', 'section': 'project'}
        """
        # Normalize field_path
        field_path = field_path.lower().strip()

        # Check if we have this field tracked
        if field_path in self._value_sources:
            section = field_path.split('.')[0] if '.' in field_path else field_path
            value = self._get_nested_value(field_path)
            return {
                'source': self._value_sources.get(field_path, 'default'),
                'value': value,
                'section': section
            }

        # Try to find it dynamically
        parts = field_path.split('.')
        if not parts:
            return None

        # Try to get the value
        try:
            value = self._get_nested_value(field_path)
            # Determine source based on what's been set
            source = self._infer_source(field_path, value)
            return {
                'source': source,
                'value': value,
                'section': parts[0]
            }
        except (AttributeError, KeyError, TypeError):
            return None

    def _get_nested_value(self, field_path: str) -> Any:
        """Get a nested value from config using dot notation."""
        parts = field_path.split('.')
        obj = self

        for part in parts:
            if obj is None:
                return None
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                obj = getattr(obj, part, None)

        return obj

    def _infer_source(self, field_path: str, value: Any) -> str:
        """Infer the source of a config value."""
        # Check if it's a known env var override
        import os
        env_prefix = 'MATCHER_'
        for env_name in os.environ:
            if env_name.startswith(env_prefix):
                suffix = env_name[len(env_prefix):].lower().replace('_', '.')
                if field_path.endswith(suffix) or field_path.replace('.', '_') == suffix.replace('.', '_'):
                    return 'env'

        # Check if it's from yaml/profile
        if field_path in self._value_sources:
            return self._value_sources[field_path]

        # Default to default if we can't determine
        return 'default'

    def get_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Get sources for all tracked config values.

        Returns:
            Dict mapping field paths to their source info

        Example:
            sources = config.get_all_sources()
            # {'matching.min_confidence': {'source': 'env', 'value': 0.85, 'section': 'matching'}, ...}
        """
        result = {}
        for field_path in self._value_sources:
            value = self._get_nested_value(field_path)
            result[field_path] = {
                'source': self._value_sources[field_path],
                'value': value,
                'section': field_path.split('.')[0] if '.' in field_path else field_path
            }
        return result

    def get_load_metrics(self) -> Dict[str, float]:
        """
        Get timing metrics for config loading phases.

        Returns:
            Dict mapping phase names to duration in milliseconds

        Phases recorded:
            - migrate: Auto-migration if needed
            - parse: YAML file parsing
            - line_tracking: YAML parsing with line number tracking
            - interpolation: ${section.field} and ${ENV_VAR} resolution
            - validation: Schema validation
            - convert: Building config object from dict
            - env_overrides: Environment variable overrides
            - final_validation: Final validation after env overrides
            - total: Total loading time

        Example:
            metrics = config.get_load_metrics()
            # {'parse': 5.2, 'validation': 12.3, 'convert': 8.1, 'total': 30.5, ...}
        """
        result = dict(self._load_timings)
        # Add total if not present
        if 'total' not in result:
            result['total'] = self._load_time_ms
        return result

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

    def set_version_notes(self, notes: str) -> None:
        """
        Record a note about why the config was changed.

        Adds an entry to the version history with the current version and timestamp.

        Args:
            notes: Description of why the config was changed

        Example:
            config.set_version_notes("Increased min_confidence for better match quality")
        """
        version = getattr(self.project, 'version', '4.0.0')
        entry = {
            'version': version,
            'date': datetime.now().isoformat(),
            'notes': notes,
        }
        self._version_history.append(entry)

    def get_version_history(self) -> List[Dict[str, str]]:
        """
        Get the version history.

        Returns:
            List of {version, date, notes} entries
        """
        return self._version_history.copy()

    def get_section_version(self, section_name: str) -> Optional[str]:
        """
        Get the version of a specific config section.

        Args:
            section_name: Name of the config section (e.g., 'matching', 'video_search')

        Returns:
            Version string if section has a version, None otherwise

        Example:
            version = config.get_section_version('matching')
            # Returns '1.0.0' or None
        """
        return self._section_versions.get(section_name)

    def set_section_version(self, section_name: str, version: str) -> None:
        """
        Set the version of a specific config section.

        Args:
            section_name: Name of the config section
            version: Version string for the section
        """
        self._section_versions[section_name] = version

    def get_all_section_versions(self) -> Dict[str, str]:
        """
        Get all section versions.

        Returns:
            Dict mapping section names to their versions

        Example:
            versions = config.get_all_section_versions()
            # {'matching': '1.0.0', 'video_search': '1.0.0', ...}
        """
        return self._section_versions.copy()

    def migrate_section(self, section_name: str, target_version: str, migration_func: callable) -> bool:
        """
        Migrate a specific config section to a target version.

        Args:
            section_name: Name of the section to migrate
            target_version: Version to migrate to
            migration_func: Function that takes (section_data, current_version) and returns migrated data

        Returns:
            True if migration was performed, False if section already at target version

        Example:
            def migrate_matching_v1_to_v2(section_data):
                # Add new field with default
                section_data['new_field'] = 'default_value'
                return section_data

            config.migrate_section('matching', '2.0.0', migrate_matching_v1_to_v2)
        """
        current_version = self._section_versions.get(section_name)

        if current_version == target_version:
            logger.info(f"Section '{section_name}' already at version {target_version}")
            return False

        # Get the section data
        section_data = getattr(self, section_name, None)
        if section_data is None:
            logger.warning(f"Section '{section_name}' not found, skipping migration")
            return False

        # Apply migration
        try:
            if hasattr(section_data, '__dict__'):
                migrated = migration_func(section_data.__dict__, current_version)
                # Update the section with migrated data
                for key, value in migrated.items():
                    if not key.startswith('_'):
                        setattr(section_data, key, value)
            else:
                logger.warning(f"Section '{section_name}' is not a mutable object")
                return False
        except Exception as e:
            logger.error(f"Migration failed for section '{section_name}': {e}")
            return False

        # Update section version
        self._section_versions[section_name] = target_version
        logger.info(f"Section '{section_name}' migrated to version {target_version}")
        return True

    def register_change_callback(self, callback: Callable, section: Optional[str] = None) -> None:
        """
        Register a callback to be invoked when config is reloaded.

        Args:
            callback: Function to call on config change. Signature: callback(config, changed_sections: List[str])
            section: Optional section name to limit callbacks to specific sections.
                     Use '*' for global callbacks that fire on any change.
                     If None, registers as global callback (equivalent to '*').

        Example:
            def on_config_change(config, changed_sections):
                print(f"Config changed: {changed_sections}")

            config.register_change_callback(on_config_change)
            config.register_change_callback(on_config_change, section='matching')
        """
        if section is None:
            section = '*'

        if section not in self._change_callbacks:
            self._change_callbacks[section] = []

        self._change_callbacks[section].append(callback)

    def unregister_change_callback(self, callback: Callable, section: Optional[str] = None) -> bool:
        """
        Unregister a previously registered callback.

        Args:
            callback: The callback function to remove.
            section: The section it was registered under. If None, checks '*'.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        if section is None:
            section = '*'

        if section in self._change_callbacks:
            try:
                self._change_callbacks[section].remove(callback)
                return True
            except ValueError:
                pass
        return False

    def _invoke_change_callbacks(self, changed_sections: List[str]) -> None:
        """Invoke all registered callbacks for the changed sections."""
        # Collect all callbacks to invoke (avoid modifying during iteration)
        callbacks_to_invoke: List[Tuple[str, Callable]] = []

        # First invoke section-specific callbacks
        for section in changed_sections:
            if section in self._change_callbacks:
                for cb in self._change_callbacks[section]:
                    callbacks_to_invoke.append((section, cb))

        # Then invoke global callbacks
        if '*' in self._change_callbacks:
            for cb in self._change_callbacks['*']:
                callbacks_to_invoke.append(('*', cb))

        logger.info(f"Config callbacks to invoke: {len(callbacks_to_invoke)} for sections: {changed_sections}")

        # Invoke all collected callbacks
        for section, cb in callbacks_to_invoke:
            cb_name = getattr(cb, '__name__', repr(cb))
            logger.info(f"Executing config callback '{cb_name}' for section '{section}'")
            try:
                cb(self, changed_sections)
                logger.info(f"Config callback '{cb_name}' completed successfully")
            except Exception as e:
                logger.error(f"Config callback '{cb_name}' failed: {e}")

    def __post_init__(self):
        """Initialize after dataclass creation"""
        self._loaded_at = datetime.now().isoformat()
        self._convert_nested_configs()
        self._resolve_paths()
        self._populate_api_keys()
        # Compute field-level hashes for drift detection
        self._compute_field_hashes()

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
    def from_yaml(cls, config_path: str, skip_final_validation: bool = False) -> "Config":
        """
        Load configuration from YAML file with CSafeLoader for performance.

        Performance: ~50ms for typical config file
        """
        global _config_metrics
        start_time = time.perf_counter()

        # Initialize timings dict for this load
        timings: Dict[str, float] = {}

        def record_timing(phase: str):
            """Record timing for a phase and return elapsed since last recording."""
            now = time.perf_counter()
            elapsed = (now - start_time) * 1000
            timings[phase] = elapsed
            return elapsed

        # Track the overall start for first phase
        _ = record_timing('start')

        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = cls()
            config._config_path = str(config_path)
            config._load_timings = timings
            _config_metrics['cache_misses'] += 1
            return config

        logger.info(f"Loading config from: {config_path}")

        # Auto-migrate config if needed (v3 -> v4)
        phase_start = time.perf_counter()
        migrator = ConfigMigration()
        if migrator.needs_migration(str(config_path)):
            logger.info(f"Config migration triggered: migrating to version {CURRENT_CONFIG_VERSION}")
            try:
                migrator.backup_config(str(config_path))
                migrator.migrate_config(str(config_path), CURRENT_CONFIG_VERSION)
                logger.info("Config migration completed successfully")
            except Exception as e:
                logger.error(f"Config migration failed: {e}")
                _config_metrics['validation_errors'] += 1
                raise ConfigError(f"Config migration failed: {e}") from e
        timings['migrate'] = (time.perf_counter() - phase_start) * 1000

        # Phase: parse - Parse YAML file
        phase_start = time.perf_counter()
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
        timings['parse'] = (time.perf_counter() - phase_start) * 1000

        # Phase: line_tracking - Parse YAML with line numbers
        from .schema_validation import (
            validate_config_schema,
            ConfigValidationError,
            _parse_yaml_with_lines,
        )

        phase_start = time.perf_counter()
        try:
            # Re-parse with line tracking for error messages
            yaml_content = config_path.read_text(encoding='utf-8')
            data, line_map = _parse_yaml_with_lines(yaml_content)
        except Exception:
            # If line tracking fails, continue without line numbers
            line_map = {}
        timings['line_tracking'] = (time.perf_counter() - phase_start) * 1000

        # Phase: interpolation - Resolve value interpolations
        phase_start = time.perf_counter()
        try:
            cls._resolve_value_interpolations(data)
        except ConfigError as e:
            logger.error(f"Config interpolation error: {e}")
            _config_metrics['validation_errors'] += 1
            raise
        timings['interpolation'] = (time.perf_counter() - phase_start) * 1000

        # Phase: validation - Schema validation
        phase_start = time.perf_counter()
        try:
            validate_config_schema(data, raise_on_error=True, line_map=line_map)
        except ConfigValidationError as e:
            _config_metrics['validation_errors'] += 1
            # Log validation errors with specific field paths
            error_details = str(e).split('\n')
            for detail in error_details:
                if detail.strip() and '- ' in detail:
                    field_path = detail.strip().lstrip('- ')
                    logger.error(f"[CFG-001] Validation error: {field_path}")
            raise
        timings['validation'] = (time.perf_counter() - phase_start) * 1000

        # Phase: convert - Build config object
        phase_start = time.perf_counter()
        # Compute hash for change detection
        config_hash = hashlib.md5(
            yaml.dump(data, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Build config object
        config = cls._from_dict(data)
        config._config_path = str(config_path)
        config._config_hash = config_hash
        config._config_data = data  # Store raw data for profile merging

        # Track value sources from YAML (US-142-010)
        config._track_yaml_sources(data)

        # Load version history from config file (if present)
        if '_version_history' in data:
            config._version_history = data['_version_history']

        # Load section versions from config file (if present)
        if '_section_versions' in data:
            config._section_versions = data['_section_versions']
        timings['convert'] = (time.perf_counter() - phase_start) * 1000

        # Track metrics
        load_time = (time.perf_counter() - start_time) * 1000
        config._load_time_ms = load_time
        config._load_timings = timings
        _config_metrics['load_count'] += 1
        _config_metrics['load_time_total_ms'] += load_time

        logger.info(f"Loaded config from {config_path} in {load_time:.1f}ms (hash: {config_hash})")
        if YAML_FAST:
            logger.debug("Using CSafeLoader (C-based) for optimized parsing")

        # Phase: env_overrides - Apply environment variable overrides
        phase_start = time.perf_counter()
        config._apply_env_overrides()
        timings['env_overrides'] = (time.perf_counter() - phase_start) * 1000

        # Phase: final_validation - Run validation after environment overrides
        phase_start = time.perf_counter()
        # Skip if skip_final_validation=True (used by --dry-run-config to show validation errors)
        if not skip_final_validation:
            validation_errors = config.validate()
            if validation_errors:
                error_msg = "Config validation failed after environment overrides:\n" + "\n".join(f"  - {e}" for e in validation_errors)
                logger.error(error_msg)
                _config_metrics['validation_errors'] += 1
                raise ConfigError(error_msg)
        timings['final_validation'] = (time.perf_counter() - phase_start) * 1000

        # Update timings with final values
        config._load_timings = timings

        # DEBUG: Log effective config values after loading
        logger.debug(f"Config loaded - effective values: matching.min_confidence={config.matching.min_confidence}, "
                     f"video_search.max_total_results={config.video_search.max_total_results}, "
                     f"download.max_concurrent={config.download.max_concurrent}")

        return config

    @classmethod
    def load_with_profile(cls, config_path: str, profile_name: str,
                         skip_final_validation: bool = False) -> "Config":
        """
        Load configuration with a profile that overrides base config values.

        Profile is merged with base config.yaml, where profile values take precedence.

        Args:
            config_path: Path to base config file (usually config.yaml)
            profile_name: Name of profile to load (e.g., 'dev', 'staging', 'prod')
            skip_final_validation: Skip API key and constraint validation

        Returns:
            Config object with profile overrides applied

        Raises:
            FileNotFoundError: If profile file doesn't exist
            ConfigError: If profile validation fails
        """
        # Find profile directory
        config_dir = Path(config_path).parent.resolve()
        profile_dir = config_dir / "profiles"

        # Also check in project root if not found
        if not profile_dir.exists():
            # Try relative to current working directory
            profile_dir = Path("config/profiles")

        profile_file = profile_dir / f"{profile_name}.yaml"

        if not profile_file.exists():
            available = []
            if profile_dir.exists():
                available = [f.stem for f in profile_dir.glob("*.yaml")]
            raise FileNotFoundError(
                f"Profile '{profile_name}' not found at {profile_file}. "
                f"Available profiles: {available}"
            )

        logger.info(f"Loading profile '{profile_name}' from {profile_file}")

        # Load base config first
        base_config = cls.from_yaml(config_path, skip_final_validation=True)

        # Load profile data
        try:
            with open(profile_file, 'r', encoding='utf-8') as f:
                profile_data = yaml.load(f, Loader=SafeLoader) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"YAML syntax error in profile '{profile_name}': {e}") from e

        # Validate profile has required metadata
        if 'name' not in profile_data:
            raise ConfigError(f"Profile '{profile_name}' missing required 'name' field")
        if profile_data['name'] != profile_name:
            logger.warning(f"Profile name mismatch: '{profile_data['name']}' != '{profile_name}'")

        # Merge profile overrides into base config data
        # Profile values override base config values
        merged_data = cls._merge_profile_with_config(base_config._config_data, profile_data)

        # Rebuild config with merged data
        config = cls._from_dict(merged_data)
        config._config_path = str(config_path)
        config._profile_applied = profile_name

        # Track profile sources (US-142-010)
        config._track_profile_sources(profile_data)

        # Apply environment variable overrides (US-128-002)
        config._apply_env_overrides()

        # Run validation after profile and env overrides are applied
        if not skip_final_validation:
            validation_errors = config.validate()
            if validation_errors:
                error_msg = f"Config validation failed for profile '{profile_name}':\n" + "\n".join(f"  - {e}" for e in validation_errors)
                logger.error(error_msg)
                raise ConfigError(error_msg)

        logger.info(f"Applied profile '{profile_name}' to base config")
        return config

    @classmethod
    def _merge_profile_with_config(cls, base_data: Dict[str, Any],
                                   profile_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge profile overrides with base config data.

        Profile values override base values. Nested dicts are merged recursively.

        Args:
            base_data: Base configuration dictionary
            profile_data: Profile override dictionary

        Returns:
            Merged configuration dictionary
        """
        import copy
        result = copy.deepcopy(base_data)

        # Remove metadata fields from profile before merging
        metadata_fields = {'name', 'description', 'inherits_from'}
        merge_data = {k: v for k, v in profile_data.items() if k not in metadata_fields}

        # Recursively merge
        for key, value in merge_data.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Merge nested dicts
                result[key] = cls._deep_merge(result[key], value)
            else:
                # Override value
                result[key] = copy.deepcopy(value)

        # Add profile metadata to result
        if 'name' in profile_data:
            result['_profile_applied'] = profile_data['name']

        return result

    @classmethod
    def _deep_merge(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries, with override taking precedence."""
        import copy
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    ENV_VAR_PREFIX = "MATCHER_"

    def _track_yaml_sources(self, data: Dict[str, Any], prefix: str = "") -> None:
        """Track which config values came from YAML file.

        Args:
            data: Raw config data dictionary
            prefix: Field path prefix for nested values
        """
        # Section mapping: yaml_key -> attr_name
        section_mapping = {
            'project': 'project', 'transcription': 'transcription', 'embedding': 'embedding',
            'indexing': 'indexing', 'vision': 'vision', 'scene_detection': 'scene_detection',
            'audio_analysis': 'audio_analysis', 'matching': 'matching',
            'negative_matching': 'negative_matching', 'remix': 'remix',
            'zero_download_remix': 'zero_download_remix', 'image_search': 'image_search',
            'keyword': 'keyword', 'llm': 'llm', 'enhanced': 'enhanced',
            'downloading': 'downloading', 'download': 'download',
            'duration_tiers': 'duration_tiers', 'stock_footage': 'stock_footage',
            'deduplication': 'deduplication', 'output': 'output', 'multi_style': 'multi_style',
            'logging': 'logging', 'cache': 'cache', 'global_cache': 'global_cache',
            'pipeline': 'pipeline', 'api_keys': 'api_keys', 'broll': 'broll',
            'healing': 'healing', 'unified_error_aggregation': 'unified_error_aggregation',
            'cross_keyword_learning': 'cross_keyword_learning',
            'validation_webhook': 'validation_webhook', 'iterative_matching': 'iterative_matching',
            'rate_limit': 'rate_limit', 'search_budget': 'search_budget',
            'video_search': 'video_search', 'silent_video': 'silent_video',
        }

        for yaml_key, attr_name in section_mapping.items():
            if yaml_key in data and data[yaml_key]:
                section_path = prefix + attr_name if prefix else attr_name
                self._track_section_sources(data[yaml_key], section_path)

    def _track_section_sources(self, section_data: Any, section_path: str) -> None:
        """Track sources for a single config section."""
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                field_path = f"{section_path}.{key}"
                # Only track if it's not a metadata field
                if not key.startswith('_'):
                    self._value_sources[field_path] = 'yaml'
                    # Recurse for nested dicts
                    if isinstance(value, dict):
                        self._track_section_sources(value, field_path)

    def _track_profile_sources(self, profile_data: Dict[str, Any], prefix: str = "") -> None:
        """Track which config values came from profile overrides.

        Args:
            profile_data: Profile override dictionary
            prefix: Field path prefix for nested values
        """
        # Skip metadata fields
        metadata_fields = {'name', 'description', 'version', 'extends', 'includes'}

        for key, value in profile_data.items():
            if key in metadata_fields:
                continue
            field_path = prefix + key if prefix else key
            if isinstance(value, dict):
                self._track_profile_sources(value, field_path + ".")
            else:
                self._value_sources[field_path] = 'profile'

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config values.

        Supports:
        - MATCHER_<SECTION>_<FIELD>=value for top-level section fields
        - MATCHER_<SECTION>_<NESTED>_<FIELD>=value for nested fields

        Examples:
        - MATCHER_MATCHING_MIN_CONFIDENCE=0.8
        - MATCHER_DOWNLOAD_MULLVAD_ENABLED=true
        - MATCHER_VIDEO_SEARCH_MAX_TOTAL_RESULTS=100
        """
        import os
        from typing import get_type_hints

        # Build a mapping of section names to their config objects and nested field info
        # section_name -> { 'obj': section_obj, 'nested': { nested_name: nested_obj } }
        section_info = {}

        for field in fields(self):
            section_name = field.name
            section_obj = getattr(self, section_name)
            if is_dataclass(section_obj) and not isinstance(section_obj, type):
                # Find nested dataclasses within this section
                nested_info = {}
                for sub_field in fields(section_obj):
                    sub_obj = getattr(section_obj, sub_field.name)
                    if is_dataclass(sub_obj) and not isinstance(sub_obj, type):
                        nested_info[sub_field.name.upper()] = sub_obj
                section_info[section_name.upper()] = {'obj': section_obj, 'nested': nested_info}

        # Process each MATCHER_* environment variable
        overrides_applied = []

        # Special case: YouTube API shorthand env vars (US-148-005)
        # Allow MATCHER_YOUTUBE_API_KEY and MATCHER_YOUTUBE_API_ENABLED as shorthand
        if 'MATCHER_YOUTUBE_API_KEY' in os.environ:
            env_value = os.environ['MATCHER_YOUTUBE_API_KEY']
            # Find download.youtube_api nested config
            if 'DOWNLOAD' in section_info:
                nested = section_info['DOWNLOAD'].get('nested', {})
                if 'YOUTUBE_API' in nested:
                    nested['YOUTUBE_API'].api_key = env_value
                    self._value_sources['download.youtube_api.api_key'] = 'env'
                    overrides_applied.append('download.youtube_api.api_key')
                    logger.info(f"Environment override applied: MATCHER_YOUTUBE_API_KEY")

        if 'MATCHER_YOUTUBE_API_ENABLED' in os.environ:
            env_value = os.environ['MATCHER_YOUTUBE_API_ENABLED']
            if 'DOWNLOAD' in section_info:
                nested = section_info['DOWNLOAD'].get('nested', {})
                if 'YOUTUBE_API' in nested:
                    # Coerce to bool
                    bool_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    nested['YOUTUBE_API'].enabled = bool_value
                    self._value_sources['download.youtube_api.enabled'] = 'env'
                    overrides_applied.append('download.youtube_api.enabled')
                    logger.info(f"Environment override applied: MATCHER_YOUTUBE_API_ENABLED")

        # Skip already-processed shorthand env vars in the generic loop below
        handled_env_vars = {'MATCHER_YOUTUBE_API_KEY', 'MATCHER_YOUTUBE_API_ENABLED'}

        for env_name, env_value in os.environ.items():
            if env_name in handled_env_vars:
                continue
            if not env_name.startswith(self.ENV_VAR_PREFIX):
                continue

            # Parse: MATCHER_SECTION_FIELD or MATCHER_SECTION_NESTED_FIELD
            suffix = env_name[len(self.ENV_VAR_PREFIX):]
            parts = suffix.split('_')

            if len(parts) < 2:
                logger.warning(f"Invalid environment variable format: {env_name} (need at least SECTION_FIELD)")
                continue

            # Try to find the section name - it might have underscores (e.g., VIDEO_SEARCH)
            # Try from longest to shortest to match the longest section name first
            section_upper = None
            section_parts_used = 0
            for i in range(len(parts) - 1, 0, -1):
                potential_section = '_'.join(parts[:i]).upper()
                if potential_section in section_info:
                    section_upper = potential_section
                    section_parts_used = i
                    break

            if section_upper is None:
                logger.warning(f"Unknown config section in {env_name}")
                continue

            section_data = section_info[section_upper]
            section_obj = section_data['obj']

            # Get type hints for this section
            try:
                type_hints = get_type_hints(section_obj.__class__)
            except Exception:
                type_hints = {}
            field_types = {f.name: type_hints.get(f.name, f.type) for f in fields(section_obj)}

            # Remaining parts after section name
            remaining_parts = parts[section_parts_used:]

            # First, try to match as nested field (SECTION_NESTED_FIELD)
            if len(remaining_parts) >= 2:
                nested_name = remaining_parts[0].upper()
                if nested_name in section_data['nested']:
                    nested_obj = section_data['nested'][nested_name]
                    # Get the actual field name (remaining parts after nested name)
                    field_name = '_'.join(remaining_parts[1:]).lower()
                    try:
                        nested_type_hints = get_type_hints(nested_obj.__class__)
                    except Exception:
                        nested_type_hints = {}
                    nested_field_types = {f.name: nested_type_hints.get(f.name, f.type) for f in fields(nested_obj)}
                    if field_name in nested_field_types:
                        coerced_value = self._coerce_value(env_value, nested_field_types[field_name], field_name)
                        setattr(nested_obj, field_name, coerced_value)
                        # Track source as env (US-142-010)
                        field_path = f"{section_upper.lower()}.{nested_name.lower()}.{field_name}"
                        self._value_sources[field_path] = 'env'
                        overrides_applied.append(f"{section_upper.lower()}.{nested_name.lower()}.{field_name}")
                        logger.debug(f"Environment override applied: {env_name}={env_value} -> {field_path}")
                        logger.info(f"Environment override applied: {env_name}")
                        continue

            # Try to match as direct field - try all possible field name combinations
            # (since field names like min_confidence have underscore)
            for field_name_len in range(1, len(remaining_parts) + 1):
                potential_field_name = '_'.join(remaining_parts[:field_name_len]).lower()
                if potential_field_name in field_types:
                    coerced_value = self._coerce_value(env_value, field_types[potential_field_name], potential_field_name)
                    setattr(section_obj, potential_field_name, coerced_value)
                    # Track source as env (US-142-010)
                    field_path = f"{section_upper.lower()}.{potential_field_name}"
                    self._value_sources[field_path] = 'env'
                    overrides_applied.append(f"{section_upper.lower()}.{potential_field_name}")
                    logger.debug(f"Environment override applied: {env_name}={env_value} -> {field_path}")
                    logger.info(f"Environment override applied: {env_name}")
                    break

        if overrides_applied:
            logger.info(f"Applied {len(overrides_applied)} environment variable overrides")

    @classmethod
    def _resolve_value_interpolations(cls, data: Dict[str, Any]) -> None:
        """
        Resolve ${section.field} and ${ENV_VAR} interpolations in config values.

        Supports:
        - ${section.field} - Reference to another config value
        - ${ENV_VAR} - Reference to environment variable

        Examples:
        - ${video_search.results_per_keyword}
        - ${matching.max_results}
        - ${HOME}
        - ${PATH}
        """
        import re

        # Pattern to match ${...} interpolations
        interpolation_pattern = re.compile(r'\$\{([^}]+)\}')

        # Track which keys we're currently resolving (for circular detection)
        resolving_keys: set = set()

        def _flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Tuple[Any, str]]:
            """Flatten nested dict with full path keys, returning (value, full_path)."""
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(_flatten_dict(v, new_key))
                else:
                    items[new_key] = (v, new_key)
            return items

        def _resolve_value(value: Any, depth: int = 0, current_path: str = '') -> Any:
            """Recursively resolve interpolations in a value."""
            # Prevent infinite recursion
            max_depth = 10
            if depth > max_depth:
                raise ConfigError(f"Circular reference detected in config interpolation (depth > {max_depth})")

            if isinstance(value, str):
                # Find all interpolations in the string
                matches = interpolation_pattern.findall(value)

                if not matches:
                    return value

                result = value
                for match in matches:
                    resolved = None

                    # Check if it's an environment variable (all caps or contains known env var pattern)
                    if match.isupper() or match in os.environ:
                        # Try to resolve as environment variable
                        env_val = os.environ.get(match)
                        if env_val is not None:
                            resolved = env_val

                    if resolved is None:
                        # Try to resolve as config reference (section.field)
                        parts = match.split('.')
                        if len(parts) >= 2:
                            section = parts[0]
                            field = '.'.join(parts[1:])
                            lookup_key = f"{section}.{field}"

                            # Check circular reference
                            if lookup_key in resolving_keys:
                                raise ConfigError(
                                    f"Circular reference detected: ${{{match}}} creates a cycle (already resolving: {resolving_keys})"
                                )

                            # Look up in flattened data
                            if lookup_key in flat_data:
                                resolving_keys.add(lookup_key)
                                try:
                                    resolved_val, _ = flat_data[lookup_key]
                                    # Recursively resolve if it contains more interpolations
                                    resolved = str(_resolve_value(resolved_val, depth + 1, lookup_key))
                                finally:
                                    resolving_keys.discard(lookup_key)

                    if resolved is not None:
                        result = result.replace(f'${{{match}}}', str(resolved))

                return result

            elif isinstance(value, dict):
                return {k: _resolve_value(v, depth, current_path) for k, v in value.items()}
            elif isinstance(value, list):
                return [_resolve_value(item, depth, current_path) for item in value]
            else:
                return value

        # First pass: flatten the dict to get all values (before interpolation)
        flat_data = _flatten_dict(data)

        # Helper to coerce string to appropriate type based on content
        def _coerce_type(resolved_value: Any) -> Any:
            """Coerce resolved string value to appropriate type based on content."""
            if not isinstance(resolved_value, str):
                return resolved_value

            # Try to convert to int first (most common case)
            try:
                return int(resolved_value)
            except ValueError:
                pass

            # Try to convert to float
            try:
                return float(resolved_value)
            except ValueError:
                pass

            # Handle booleans
            if resolved_value.lower() in ('true', 'yes'):
                return True
            elif resolved_value.lower() in ('false', 'no'):
                return False

            return resolved_value

        # Second pass: resolve interpolations in-place
        def _resolve_in_place(d: Dict[str, Any], parent_key: str = '') -> None:
            """Recursively resolve interpolations in-place in the dict."""
            for key in list(d.keys()):
                value = d[key]
                full_key = f"{parent_key}.{key}" if parent_key else key

                if isinstance(value, dict):
                    _resolve_in_place(value, full_key)
                elif isinstance(value, str):
                    # Check if this string contains interpolation
                    if '$' in value:
                        resolved = _resolve_value(value, 0, full_key)
                        # Try to coerce type if the result is purely numeric
                        d[key] = _coerce_type(resolved)
                    # Else leave as-is (no interpolation to resolve)
                elif isinstance(value, list):
                    d[key] = _resolve_value(value, 0, full_key)

        _resolve_in_place(data)

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
            'search_budget': (SearchBudgetConfig, 'search_budget'),
            'video_search': (VideoSearchConfig, 'video_search'),
            'validation_webhook': (ValidationWebhookConfig, 'validation_webhook'),
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
    def _coerce_value(value: Any, field_type: Type, field_name: str) -> Any:
        """Coerce string values to their expected type (int, float, bool).

        Args:
            value: The value to coerce (may be a string).
            field_type: The expected type (int, float, bool, or their Union/Optional variants).
            field_name: Name of the field (for error messages).

        Returns:
            The coerced value if conversion was needed and successful, otherwise the original value.

        Raises:
            ConfigError: If the string cannot be converted to the expected type.
        """
        # Skip if not a string or already the correct type
        if not isinstance(value, str):
            return value

        # Get the unwrapped type for Union/Optional types
        actual_type = field_type
        origin = getattr(field_type, '__origin__', None)
        if origin is not None:
            args = getattr(field_type, '__args__', ())
            if args:
                actual_type = args[0]

        # Coerce string to int
        if actual_type is int:
            try:
                return int(value)
            except ValueError:
                raise ConfigError(
                    f"Invalid value '{value}' for field '{field_name}': expected int, got '{value}'"
                )

        # Coerce string to float
        if actual_type is float:
            try:
                return float(value)
            except ValueError:
                raise ConfigError(
                    f"Invalid value '{value}' for field '{field_name}': expected float, got '{value}'"
                )

        # Coerce string to bool (case-insensitive)
        if actual_type is bool:
            lower_val = value.lower()
            if lower_val in ('true', 'yes', '1', 'on'):
                return True
            elif lower_val in ('false', 'no', '0', 'off'):
                return False
            else:
                raise ConfigError(
                    f"Invalid value '{value}' for field '{field_name}': expected bool "
                    f"(true/false, yes/no, 1/0, on/off), got '{value}'"
                )

        return value

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
                    # Coerce string values to their expected types (int, float, bool)
                    filtered_data[key] = Config._coerce_value(value, field_type, key)
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

    def to_dict(self, redact_sensitive: bool = True, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """Convert config to nested dict suitable for YAML serialization.

        Args:
            redact_sensitive: If True (default), redact API key values
                with '***REDACTED***'. Set False to include actual values.
            sections: If provided, only export these sections. If None,
                export all sections. Valid sections: project, project_dir,
                transcription, embedding, indexing, vision, scene_detection,
                audio_analysis, matching, negative_matching, remix,
                zero_download_remix, image_search, keyword, llm, enhanced,
                downloading, download, stock_footage, deduplication, output,
                multi_style, logging, cache, pipeline, api_keys, healing,
                iterative_matching, rate_limit, broll, global_cache,
                silent_video, duration_tiers.

        Returns:
            Nested dictionary of config sections (all or selected).
        """
        result: Dict[str, Any] = {}

        # Handle project and project_dir
        if sections is None or 'project' in sections:
            result['project'] = asdict(self.project)
        if sections is None or 'project_dir' in sections:
            result['project_dir'] = self.project_dir

        # All sections from _from_dict's section_mapping
        all_sections = [
            'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'audio_analysis', 'matching',
            'negative_matching', 'remix', 'zero_download_remix',
            'image_search', 'keyword', 'llm', 'enhanced',
            'downloading', 'download', 'stock_footage', 'deduplication',
            'output', 'multi_style', 'logging', 'cache', 'pipeline',
            'api_keys', 'healing', 'iterative_matching', 'rate_limit',
            'broll', 'global_cache', 'silent_video',
        ]

        # Filter to requested sections if specified
        if sections is not None:
            section_set = set(sections)
            all_sections = [s for s in all_sections if s in section_set]

        for section in all_sections:
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

    def to_yaml(self, output_path: str = None, *, redact_sensitive: bool = True,
                sections: Optional[List[str]] = None) -> str:
        """Serialize effective config to YAML.

        Args:
            output_path: If provided, write YAML to this file path.
            redact_sensitive: If True (default), redact API keys.
            sections: If provided, only export these sections.

        Returns:
            YAML string of the effective config.
        """
        data = self.to_dict(redact_sensitive=redact_sensitive, sections=sections)
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            logger.info(f"Saved config to {output_path}")

        return yaml_str

    def to_json(self, output_path: str = None, *, redact_sensitive: bool = True,
                sections: Optional[List[str]] = None) -> str:
        """Serialize effective config to JSON with metadata.

        Args:
            output_path: If provided, write JSON to this file path.
            redact_sensitive: If True (default), redact API keys.
            sections: If provided, only export these sections.

        Returns:
            JSON string of the effective config with metadata.
        """
        import json

        # Get the base config data
        data = self.to_dict(redact_sensitive=redact_sensitive, sections=sections)

        # Build metadata
        metadata = {
            'version': self.project.version if self.project else '4.0.0',
            'loaded_at': self._loaded_at,
            'config_hash': self._config_hash,
            'exported_at': datetime.now().isoformat(),
            'redacted': redact_sensitive,
            'sections_included': list(data.keys()) if sections else None,
            'version_history': self._version_history,
            'section_versions': self._section_versions,
            'usage_stats': self.get_usage_stats(),
            'load_metrics': self.get_load_metrics(),
        }

        # Wrap in metadata structure
        output = {
            'metadata': metadata,
            'config': data
        }

        json_str = json.dumps(output, indent=2, default=str)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            logger.info(f"Saved config to {output_path}")

        return json_str

    def to_diff(self, output_path: str = None, *, redact_sensitive: bool = True,
                sections: Optional[List[str]] = None) -> str:
        """Serialize config as diff: only non-default values with their defaults.

        This shows what values have been changed from the default configuration.
        Each entry shows: current_value (default: default_value)

        Args:
            output_path: If provided, write diff to this file path.
            redact_sensitive: If True (default), redact API keys.
            sections: If provided, only export these sections.

        Returns:
            Diff string showing non-default values with their defaults.
        """
        # Get current config as dict
        current_data = self.to_dict(redact_sensitive=redact_sensitive, sections=sections)

        # Get default config for comparison
        default_config = Config()
        default_data = default_config.to_dict(redact_sensitive=redact_sensitive, sections=sections)

        # Build diff output
        diff_lines = ["# Config Diff - Non-default values", "# Format: current_value (default: default_value)", ""]

        def format_value(val):
            """Format a value for display."""
            if val is None:
                return "null"
            elif isinstance(val, bool):
                return "true" if val else "false"
            elif isinstance(val, str):
                return f'"{val}"'
            elif isinstance(val, (int, float)):
                return str(val)
            elif isinstance(val, dict):
                # For nested dicts, show as summary
                return f"{{{len(val)} keys}}"
            elif isinstance(val, list):
                return f"[{len(val)} items]"
            else:
                return str(val)

        def compare_dicts(current: Dict, default: Dict, path: str = "") -> List[str]:
            """Recursively compare dicts and return diff lines."""
            lines = []

            # Get all keys from both
            all_keys = set(current.keys()) | set(default.keys())

            for key in sorted(all_keys):
                current_val = current.get(key)
                default_val = default.get(key)
                current_path = f"{path}.{key}" if path else key

                # Skip if both are None or both are empty
                if current_val == default_val:
                    continue

                # Check if this is a nested dict (dataclass)
                if isinstance(current_val, dict) and isinstance(default_val, dict):
                    # Check if both have the same structure (both are nested configs)
                    # Include type(None) for None values
                    simple_types = (dict, str, int, float, bool, list, type(None))
                    if current_val and default_val and all(
                        isinstance(v, simple_types) for v in current_val.values()
                    ) and all(
                        isinstance(v, simple_types) for v in default_val.values()
                    ):
                        # Recurse into nested dict
                        lines.extend(compare_dicts(current_val, default_val, current_path))
                    else:
                        # Simple comparison
                        if current_val != default_val:
                            lines.append(f"{current_path}: {format_value(current_val)} (default: {format_value(default_val)})")
                else:
                    # Simple value comparison
                    if current_val != default_val:
                        lines.append(f"{current_path}: {format_value(current_val)} (default: {format_value(default_val)})")

            return lines

        diff_lines.extend(compare_dicts(current_data, default_data))

        diff_str = "\n".join(diff_lines)

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(diff_str)
            logger.info(f"Saved config diff to {output_path}")

        return diff_str

    def reload(self) -> bool:
        """
        Reload config from file if changed.

        Returns:
            True if config was reloaded, False if unchanged or no config path

        Raises:
            FrozenConfigError: If config is frozen and reload is attempted
        """
        global _config_metrics

        # Check if frozen - raise error on reload attempt
        if self._frozen:
            raise FrozenConfigError(
                "Cannot reload frozen Config. Call config.unfreeze() first "
                "(test scenarios only - not allowed during pipeline execution)."
            )

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

        # Detect which sections changed by comparing old vs new
        old_config = self
        changed_sections = self._detect_changed_sections(old_config, data)

        # Log changed sections with structured logging
        logger.warning(f"[CFG-005] Config changes detected in sections: {changed_sections}")
        for section in changed_sections:
            old_section = getattr(old_config, section, None)
            if old_section is not None and is_dataclass(old_section):
                old_dict = asdict(old_section)
                new_dict = data.get(section, {})
                logger.warning(f"[CFG-005] Config section '{section}' changed: {len(old_dict)} -> {len(new_dict)} keys")

        # Reload with error handling
        try:
            logger.info(f"[CFG-005] Reloading config (old hash: {self._config_hash}, new hash: {new_hash})")
            new_config = Config.from_yaml(self._config_path)
        except Exception as e:
            log_error_with_context(
                logger,
                "CFG-005",
                f"Config reload failed: {e}",
                config_path=self._config_path,
                old_hash=self._config_hash,
                new_hash=new_hash
            )
            return False

        # Copy all attributes except callbacks (preserve them across reloads)
        callbacks_backup = self._change_callbacks
        for attr in dir(new_config):
            if not attr.startswith('_') and not callable(getattr(new_config, attr)):
                setattr(self, attr, getattr(new_config, attr))

        # Restore callback registry
        self._change_callbacks = callbacks_backup

        self._config_hash = new_hash
        self._loaded_at = datetime.now().isoformat()
        _config_metrics['reload_count'] += 1

        # Log successful reload
        logger.info(f"[CFG-005] Config reloaded successfully, changed sections: {changed_sections}")

        # Invoke change callbacks
        self._invoke_change_callbacks(changed_sections)

        return True

    def reload_section(self, section_name: str) -> bool:
        """
        Reload only a specific config section from file.

        This is useful for long-running pipelines where only some settings
        need updating without reloading the entire config.

        Args:
            section_name: The name of the section to reload (e.g., 'matching', 'download')

        Returns:
            True if section was reloaded (content changed), False if unchanged

        Raises:
            ValueError: If section_name is not valid
            FrozenConfigError: If config is frozen (unless section unfreeze works)
        """
        global _config_metrics

        # Section mapping for validation and building
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
            'search_budget': (SearchBudgetConfig, 'search_budget'),
            'video_search': (VideoSearchConfig, 'video_search'),
            'duration_tiers': (None, 'duration_tiers'),  # Special handling
            'project': (ProjectConfig, 'project'),  # Special handling
            'project_dir': (None, 'project_dir'),  # Non-dataclass
        }

        # Validate section_name
        if section_name not in section_mapping:
            valid_sections = ', '.join(sorted(section_mapping.keys()))
            raise ValueError(
                f"Invalid section '{section_name}'. Valid sections: {valid_sections}"
            )

        dataclass_type, attr_name = section_mapping[section_name]

        # Handle frozen state
        was_frozen = self._frozen
        if was_frozen:
            # Temporarily unfreeze for section reload
            self.unfreeze()

        try:
            if not self._config_path:
                return False

            config_path = Path(self._config_path)
            if not config_path.exists():
                return False

            # Load YAML data
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.load(f, Loader=SafeLoader) or {}

            # Get section data from YAML
            section_data = data.get(section_name, {})

            # Get current section value
            current_section = getattr(self, attr_name, None)

            # Compare old vs new
            if dataclass_type and is_dataclass(current_section):
                old_dict = asdict(current_section)
                # Compare only keys present in both - YAML data may be partial (defaults fill rest)
                if section_data and old_dict.get('min_confidence') == section_data.get('min_confidence'):
                    # For simple case: check if the key we care about matches
                    # More robust: check if all YAML-specified keys match their old values
                    unchanged = True
                    for key in section_data:
                        if old_dict.get(key) != section_data.get(key):
                            unchanged = False
                            break
                    if unchanged:
                        return False
            elif current_section == section_data:
                # Non-dataclass attribute (e.g., project_dir)
                return False

            # Rebuild the section
            logger.info(f"Reloading config section: {section_name}")

            if section_name == 'duration_tiers':
                # Special handling for nested duration_tiers structure
                self.duration_tiers = self._build_duration_tiers(section_data)
            elif section_name == 'project':
                # Special handling for project config
                self.project = self._build_dataclass(
                    ProjectConfig, section_data, section_name='project'
                )
            elif section_name == 'project_dir':
                # Non-dataclass simple value
                self.project_dir = section_data
            elif dataclass_type:
                # Regular section - rebuild from data
                new_section = self._build_dataclass(
                    dataclass_type, section_data, section_name=section_name
                )
                setattr(self, attr_name, new_section)

            _config_metrics['reload_count'] += 1

            # Invoke section-specific callbacks
            self._invoke_change_callbacks([section_name])

            return True

        finally:
            # Refreeze if we unfroze
            if was_frozen:
                self.freeze()

    def reset_section(self, section_name: str) -> bool:
        """
        Reset a specific config section to its default values.

        This is useful for testing scenarios or reverting changes made to
        a specific section without affecting the rest of the config.

        Args:
            section_name: The name of the section to reset (e.g., 'matching', 'download')

        Returns:
            True if section was reset successfully, False if section not found

        Raises:
            FrozenConfigError: If config is frozen (call unfreeze() first)

        Example:
            # Reset matching section to defaults
            config.reset_section('matching')

            # Reset download section to defaults
            config.reset_section('download')
        """
        # Section mapping (same as reload_section)
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
            'search_budget': (SearchBudgetConfig, 'search_budget'),
            'video_search': (VideoSearchConfig, 'video_search'),
            'duration_tiers': (DurationTiersConfig, 'duration_tiers'),
            'project': (ProjectConfig, 'project'),
        }

        # Check if section exists
        if section_name not in section_mapping:
            return False

        dataclass_type, attr_name = section_mapping[section_name]

        # Handle frozen state
        was_frozen = self._frozen
        if was_frozen:
            self.unfreeze()

        try:
            # Create new default instance and set it
            default_section = dataclass_type()
            setattr(self, attr_name, default_section)

            # Invoke change callbacks
            self._invoke_change_callbacks([section_name])

            return True

        finally:
            # Refreeze if we unfroze
            if was_frozen:
                self.freeze()

    def update_section(self, section: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific config values at runtime without reloading from file.

        This method allows external tools and debugging workflows to modify
        config values in-memory without requiring a full config reload.

        Args:
            section: The section name to update (e.g., 'matching', 'download')
            updates: Dictionary of field names and their new values

        Returns:
            True if updates were applied

        Raises:
            ValueError: If section is invalid or field doesn't exist
            FrozenConfigError: If config is frozen (call unfreeze() first)
            TypeError: If a value cannot be coerced to the expected type

        Example:
            # Increase matching threshold at runtime
            config.update_section('matching', {'min_confidence': 0.75})

            # Update multiple fields
            config.update_section('download', {
                'max_concurrent': 3,
                'retry_delay': 10
            })
        """
        # Section mapping (same as reload_section)
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
            'search_budget': (SearchBudgetConfig, 'search_budget'),
            'video_search': (VideoSearchConfig, 'video_search'),
            'duration_tiers': (None, 'duration_tiers'),  # Special handling
            'project': (ProjectConfig, 'project'),  # Special handling
            'project_dir': (None, 'project_dir'),  # Non-dataclass
        }

        # Validate section_name
        if section not in section_mapping:
            valid_sections = ', '.join(sorted(section_mapping.keys()))
            raise ValueError(
                f"Invalid section '{section}'. Valid sections: {valid_sections}"
            )

        dataclass_type, attr_name = section_mapping[section]

        # Handle frozen state
        was_frozen = self._frozen
        if was_frozen:
            # Temporarily unfreeze for update
            self.unfreeze()

        try:
            # Get current section value
            current_section = getattr(self, attr_name, None)

            if current_section is None:
                raise ValueError(f"Section '{section}' is not initialized")

            # Handle non-dataclass sections
            if dataclass_type is None:
                if section == 'project_dir':
                    # Special case: project_dir is just a string
                    if not updates:
                        return False
                    key = list(updates.keys())[0]
                    if key != 'project_dir':
                        raise ValueError(f"Invalid field '{key}' for section 'project_dir'")
                    setattr(self, attr_name, updates[key])
                    self._invoke_change_callbacks([section])
                    return True
                elif section == 'duration_tiers':
                    # duration_tiers is a dict, update it directly
                    for key, value in updates.items():
                        if key in self.duration_tiers:
                            self.duration_tiers[key] = value
                        else:
                            raise ValueError(f"Invalid field '{key}' for section 'duration_tiers'")
                    self._invoke_change_callbacks([section])
                    return True
                else:
                    raise ValueError(f"Section '{section}' cannot be updated")

            # For dataclass sections, we need to validate and apply updates
            if not is_dataclass(current_section):
                raise ValueError(f"Section '{section}' is not a dataclass")

            # Get valid fields for this dataclass
            valid_fields = {f.name: f for f in fields(dataclass_type)}

            # Get type hints to resolve string annotations (from __future__ import annotations)
            try:
                from typing import get_type_hints
                type_hints = get_type_hints(dataclass_type)
            except Exception:
                type_hints = {}

            # Validate all updates first (fail fast)
            for key in updates:
                if key not in valid_fields:
                    close = difflib.get_close_matches(key, list(valid_fields.keys()), n=1, cutoff=0.6)
                    suggestion = f" (did you mean '{close[0]}')?" if close else ""
                    raise ValueError(
                        f"Invalid field '{key}' for section '{section}'{suggestion}. "
                        f"Valid fields: {', '.join(sorted(valid_fields.keys()))}"
                    )

            # Apply updates with type coercion
            for key, value in updates.items():
                field_info = valid_fields[key]

                # Get the expected type - use resolved type hints if available
                field_type = type_hints.get(key, field_info.type)

                # Handle string type annotations that couldn't be resolved
                if isinstance(field_type, str):
                    module = sys.modules.get(__name__, None)
                    if module and hasattr(module, field_type):
                        field_type = getattr(module, field_type)

                # Handle Optional types
                origin = getattr(field_type, '__origin__', None)
                if origin is not None:
                    args = getattr(field_type, '__args__', ())
                    if args:
                        field_type = args[0]

                # Check if field is a nested dataclass (dict update)
                if hasattr(field_type, '__dataclass_fields__') and isinstance(value, dict):
                    # Get current nested dataclass
                    current_nested = getattr(current_section, key)
                    if current_nested is None:
                        # Create new nested dataclass
                        current_nested = field_type(**value)
                    else:
                        # Update existing nested dataclass fields
                        for nested_key, nested_value in value.items():
                            if hasattr(current_nested, nested_key):
                                setattr(current_nested, nested_key, nested_value)
                            else:
                                raise ValueError(
                                    f"Invalid field '{nested_key}' for nested dataclass '{field_type.__name__}'"
                                )
                    setattr(current_section, key, current_nested)
                else:
                    # Coerce and set simple value
                    coerced = Config._coerce_value(value, field_type, key)
                    setattr(current_section, key, coerced)

            # Update the section attribute
            setattr(self, attr_name, current_section)

            # Invoke section-specific callbacks
            self._invoke_change_callbacks([section])

            return True

        finally:
            # Refreeze if we unfroze
            if was_frozen:
                self.freeze()

    def _detect_changed_sections(self, old_config: "Config", new_data: Dict[str, Any]) -> List[str]:
        """Detect which config sections changed between old config and new data.

        Args:
            old_config: The current Config object
            new_data: The newly loaded YAML data

        Returns:
            List of section names that changed
        """
        changed = []

        # Known section names from section_mapping
        sections = [
            'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'audio_analysis', 'matching',
            'negative_matching', 'remix', 'zero_download_remix',
            'image_search', 'keyword', 'llm', 'enhanced',
            'downloading', 'download', 'stock_footage', 'deduplication',
            'output', 'multi_style', 'logging', 'cache', 'pipeline',
            'api_keys', 'healing', 'iterative_matching', 'rate_limit',
            'broll', 'global_cache', 'silent_video', 'search_budget',
            'video_search', 'duration_tiers', 'project'
        ]

        for section in sections:
            if section in new_data:
                old_section = getattr(old_config, section, None)
                if old_section is not None:
                    # Compare section values
                    if is_dataclass(old_section):
                        old_dict = asdict(old_section)
                        if old_dict != new_data.get(section, {}):
                            changed.append(section)
                    else:
                        # Non-dataclass attribute changed
                        changed.append(section)
                elif new_data.get(section):
                    # New section added
                    changed.append(section)

        return changed

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
            # US-146-010: YouTube API validation
            (getattr(self.download.youtube_api, 'enabled', False),
             getattr(self.download.youtube_api, 'api_key', ''),
             "YouTube Data API enabled but no API key configured. "
             "Get an API key at https://console.cloud.google.com/apis/credentials"),
        ]

        for condition, key, message in api_checks:
            if condition and not key:
                errors.append(message)

        # US-146-010: YouTube API key format validation (must start with AIza)
        yt_api_key = getattr(self.download.youtube_api, 'api_key', '')
        if yt_api_key and not yt_api_key.startswith('AIza'):
            errors.append(
                f"Invalid YouTube Data API key format: '{yt_api_key[:10]}...'. "
                "API key must start with 'AIza'. Get a valid key at https://console.cloud.google.com/apis/credentials"
            )

        # US-149-011: YouTubeAPIConfig field validation
        yt_config = self.download.youtube_api
        yt_api_keys = getattr(yt_config, 'api_keys', [])
        yt_enabled = getattr(yt_config, 'enabled', False)

        # Only validate if YouTube API is enabled
        if yt_enabled:
            # Validate api_key is non-empty string when enabled (single key)
            yt_single_key = getattr(yt_config, 'api_key', '')
            if not yt_single_key and not yt_api_keys:
                errors.append(
                    "YouTube Data API enabled but no API key configured. "
                    "Set either 'api_key' (single key) or 'api_keys' (list of keys)."
                )

            # Validate quota_limit is positive and within allowed range (1-100000)
            yt_quota_limit = getattr(yt_config, 'quota_limit', 10000)
            if yt_quota_limit < 1 or yt_quota_limit > 100000:
                errors.append(
                    f"download.youtube_api.quota_limit must be 1-100000, got {yt_quota_limit}"
                )

            # Validate warn_at_percent is 0-100
            yt_warn_at = getattr(yt_config, 'warn_at_percent', 80)
            if yt_warn_at < 0 or yt_warn_at > 100:
                errors.append(
                    f"download.youtube_api.warn_at_percent must be 0-100, got {yt_warn_at}"
                )

            # Validate timeout_seconds is reasonable (5-120)
            yt_timeout = getattr(yt_config, 'timeout_seconds', 30)
            if yt_timeout < 5 or yt_timeout > 120:
                errors.append(
                    f"download.youtube_api.timeout_seconds must be 5-120, got {yt_timeout}"
                )

            # Validate cache_ttl_seconds is reasonable (60-86400)
            yt_cache_ttl = getattr(yt_config, 'cache_ttl_seconds', 3600)
            if yt_cache_ttl < 60 or yt_cache_ttl > 86400:
                errors.append(
                    f"download.youtube_api.cache_ttl_seconds must be 60-86400, got {yt_cache_ttl}"
                )

            # Validate min_subscriber_count is non-negative
            yt_min_subs = getattr(yt_config, 'min_subscriber_count', 1000)
            if yt_min_subs < 0:
                errors.append(
                    f"download.youtube_api.min_subscriber_count must be >= 0, got {yt_min_subs}"
                )

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
            (self.stock_footage.segment_interval >= 1,
             f"stock_footage.segment_interval must be >= 1, got {self.stock_footage.segment_interval}"),
            (self.stock_footage.max_clips_per_selected_segment >= 1,
             "stock_footage.max_clips_per_selected_segment must be >= 1, "
             f"got {self.stock_footage.max_clips_per_selected_segment}"),
            # Video search validation
            (self.video_search.results_per_keyword > 0,
             f"video_search.results_per_keyword must be > 0, got {self.video_search.results_per_keyword}"),
            (self.video_search.max_total_results > 0,
             f"video_search.max_total_results must be > 0, got {self.video_search.max_total_results}"),

            # US-112-008: Region backoff multiplier validation (must be >= 1.0)
            (self.download.region_backoff.us_multiplier >= 1.0,
             f"download.region_backoff.us_multiplier must be >= 1.0, got {self.download.region_backoff.us_multiplier}"),
            (self.download.region_backoff.eu_multiplier >= 1.0,
             f"download.region_backoff.eu_multiplier must be >= 1.0, got {self.download.region_backoff.eu_multiplier}"),
            (self.download.region_backoff.asia_multiplier >= 1.0,
             f"download.region_backoff.asia_multiplier must be >= 1.0, got {self.download.region_backoff.asia_multiplier}"),
            (self.download.region_backoff.other_multiplier >= 1.0,
             f"download.region_backoff.other_multiplier must be >= 1.0, got {self.download.region_backoff.other_multiplier}"),

            # US-112-008: region_backoff enabled requires VPN configuration
            (not self.download.region_backoff.enabled or self.download.mullvad.enabled,
             "download.region_backoff.enabled=true requires download.mullvad.enabled=true"),
        ]

        for valid, message in range_checks:
            if not valid:
                errors.append(message)

        # US-111-011: Context richness signal weights validation (must sum to 1.0)
        mc = self.matching
        context_weights = (
            getattr(mc, 'context_richness_title_weight', 0.25) +
            getattr(mc, 'context_richness_description_weight', 0.25) +
            getattr(mc, 'context_richness_tags_weight', 0.25) +
            getattr(mc, 'context_richness_chapters_weight', 0.25)
        )
        if abs(context_weights - 1.0) > 0.001:
            errors.append(
                f"context richness signal weights must sum to 1.0, got {context_weights:.3f} "
                f"(title={getattr(mc, 'context_richness_title_weight', 0.25)}, "
                f"description={getattr(mc, 'context_richness_description_weight', 0.25)}, "
                f"tags={getattr(mc, 'context_richness_tags_weight', 0.25)}, "
                f"chapters={getattr(mc, 'context_richness_chapters_weight', 0.25)})"
            )

        # Deprecated field warnings
        self._validate_deprecations()

        # Enum value checks
        errors.extend(self._validate_enums())

        # Constraint relationship checks
        errors.extend(self._validate_constraints())

        # Dependency validation between config sections (US-142-012)
        errors.extend(self._validate_dependencies())

        # External webhook validation (US-142-007)
        # Runs after other validation; logs results but doesn't block by default
        errors.extend(self._validate_webhook())

        _config_metrics['validation_errors'] += len(errors)

        return errors

    def _validate_deprecations(self) -> None:
        """Check for deprecated config fields and log warnings.

        This method validates deprecated configuration fields and provides
        migration guidance. Deprecated fields are handled gracefully to
        ensure backward compatibility without runtime errors.

        Deprecated fields documented here:
        - image_search.output_dir: Use image_search.root_dir instead
        - download.caption_first.enabled: Caption-first is now always enabled
        - download.caption_first.negative_cache_ttl_hours: Use negative_cache_ttl_seconds
        - download.caption_first.prefer_human_captions: Now the default behavior
        - download.audio_first (legacy): Moved to nested config section
        """
        logger = logging.getLogger(__name__)

        # Known deprecated fields and their replacements (for documentation)
        deprecated_fields = {
            'image_search': ['output_dir'],
            'download': {
                'caption_first': ['enabled', 'negative_cache_ttl_hours', 'prefer_human_captions'],
                'audio_first': ['legacy_field_note'],
            },
        }

        # Check image_search.output_dir (deprecated, use root_dir instead)
        if hasattr(self.image_search, 'output_dir') and self.image_search.output_dir != "images":
            logger.warning(
                "config 'image_search.output_dir' is deprecated, use 'image_search.root_dir' instead"
            )

        # Check download.caption_first.enabled (deprecated - always enabled)
        # Only warn if explicitly set to False (old behavior) - default True doesn't warrant warning
        download_config = safe_get_config_value(self.download, 'caption_first')
        if download_config and getattr(download_config, 'enabled', None) is False:
            logger.warning(
                "config 'download.caption_first.enabled' is deprecated - "
                "caption-first mode is now always enabled"
            )

        # Check download.caption_first.negative_cache_ttl_hours vs negative_cache_ttl_seconds
        caption_first = safe_get_config_value(self.download, 'caption_first')
        if caption_first:
            # Warn about negative_cache_ttl_hours deprecation
            ttl_hours = getattr(caption_first, 'negative_cache_ttl_hours', None)
            if ttl_hours is not None and ttl_hours != 1.0:
                logger.warning(
                    "config 'download.caption_first.negative_cache_ttl_hours' is deprecated, "
                    "use 'download.caption_first.negative_cache_ttl_seconds' instead"
                )

            # Check for caption_source preference handling (prefer_human_captions)
            # This field controls whether to prefer human vs auto-generated captions
            prefer_human = getattr(caption_first, 'prefer_human_captions', None)
            if prefer_human is not None and prefer_human is not True:
                # Only warn if explicitly set to False (non-default)
                logger.warning(
                    "config 'download.caption_first.prefer_human_captions' is set to false - "
                    "auto-generated captions may have lower quality. Recommended: remove this "
                    "setting to use human captions when available (default behavior)"
                )

        # Check for legacy audio_first config at top-level download
        # The audio_first config is now in a nested section, but check for any
        # legacy top-level fields that might indicate old config format
        audio_first_config = safe_get_config_value(self.download, 'audio_first')
        if audio_first_config:
            # Check if any legacy fields are present that should have been migrated
            legacy_fields = ['audio_quality', 'buffer_seconds', 'merge_gap_seconds']
            for field in legacy_fields:
                # If we detect a legacy pattern where audio_first fields exist at
                # download level (not nested), log migration warning
                if hasattr(self.download, field):
                    logger.warning(
                        f"config 'download.{field}' appears to be at legacy location - "
                        "audio_first configuration should be nested under 'download.audio_first'"
                    )
                    break

    def _validate_webhook(self) -> List[str]:
        """Validate config via external webhook (US-142-007).

        Calls the configured validation webhook with the config dict as JSON POST body.
        The webhook should return: {valid: bool, errors: [], warnings: []}

        Webhook failures are logged but don't block pipeline execution by default.
        Only adds validation errors if fail_on_error is True.

        Returns:
            List of validation errors (empty if valid or fail_on_error=False)
        """
        import json
        import urllib.request
        import urllib.error
        from dataclasses import asdict, is_dataclass

        logger = logging.getLogger(__name__)
        webhook_config = self.validation_webhook

        # Skip if not enabled
        if not webhook_config.enabled:
            return []

        # Skip if no URL configured
        if not webhook_config.url:
            logger.warning("Validation webhook enabled but no URL configured")
            return []

        def convert_to_json_serializable(obj):
            """Recursively convert dataclasses to dicts for JSON serialization."""
            if is_dataclass(obj) and not isinstance(obj, type):
                return {k: convert_to_json_serializable(v) for k, v in asdict(obj).items()}
            elif hasattr(obj, '__dict__') and not isinstance(obj, (str, int, float, bool, list, dict, tuple, type(None))):
                # Handle non-dataclass objects with __dict__ (like CrossKeywordRetryLearningConfig)
                return {k: convert_to_json_serializable(v) for k, v in vars(obj).items()}
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        # Convert config to dict and make JSON serializable
        config_dict = convert_to_json_serializable(self)

        # Remove private/metadata fields
        for key in list(config_dict.keys()):
            if key.startswith('_'):
                del config_dict[key]

        # Remove alias fields and paths
        for field in ('project_dir', 'downloaded_videos_dir', 'otio_output_dir', 'cache_dir',
                      'gemini_api_key', 'anthropic_api_key', 'voyage_api_key'):
            if field in config_dict:
                del config_dict[field]

        # Make HTTP POST request to webhook
        try:
            # Prepare request
            data = json.dumps(config_dict).encode('utf-8')
            req = urllib.request.Request(
                webhook_config.url,
                data=data,
                headers={'Content-Type': 'application/json', **webhook_config.headers}
            )

            # Make request with timeout
            with urllib.request.urlopen(req, timeout=webhook_config.timeout_seconds) as response:
                response_body = response.read().decode('utf-8')
                result = json.loads(response_body)

            # Parse response
            valid = result.get('valid', True)
            errors = result.get('errors', [])
            warnings = result.get('warnings', [])

            # Log warnings
            for warning in warnings:
                logger.warning(f"Validation webhook warning: {warning}")

            # Log errors
            for error in errors:
                logger.error(f"Validation webhook error: {error}")

            # Handle invalid response - add to errors list if fail_on_error is True
            returned_errors = []
            if not valid:
                if webhook_config.fail_on_error:
                    logger.error(f"Validation webhook reported invalid config: {errors}")
                    returned_errors = [f"Validation webhook: {err}" for err in errors]
                else:
                    logger.warning(
                        f"Validation webhook reported invalid config but fail_on_error=False: {errors}"
                    )

            logger.info(f"Validation webhook completed: valid={valid}, errors={len(errors)}, warnings={len(warnings)}")

            return returned_errors

        except urllib.error.URLError as e:
            logger.warning(f"Validation webhook request failed: {e}")
            if webhook_config.fail_on_error:
                logger.error(f"Validation webhook failed with fail_on_error=True: {e}")
                return [f"Validation webhook failed: {e}"]
            return []
        except Exception as e:
            logger.warning(f"Validation webhook unexpected error: {e}")
            if webhook_config.fail_on_error:
                logger.error(f"Validation webhook failed with fail_on_error=True: {e}")
                return [f"Validation webhook failed: {e}"]
            return []

        return []

    def validate_file_references(self) -> List[Dict[str, Any]]:
        """
        Validate that file paths referenced in config actually exist or can be created.

        Checks:
        - output_dir: Output directory for generated files
        - cache_dir: Cache directory for temporary files
        - entity_cache_dir: Entity image cache directory
        - Custom config files referenced in config (via 'config' field)
        - Download directory parent exists

        For non-existent directories, suggests creation if parent exists.
        For config file references, checks that files exist.

        Returns:
            List of dicts with keys: 'type', 'field', 'path', 'suggestion'
            Empty list if all references are valid.
        """
        import os
        from pathlib import Path

        issues = []

        # 1. Check output_dir (from output section)
        output_dir = getattr(self.output, 'output_dir', None)
        if output_dir:
            output_path = Path(output_dir)
            if not output_path.exists():
                parent = output_path.parent
                if parent.exists():
                    issues.append({
                        'type': 'directory_missing',
                        'field': 'output.output_dir',
                        'path': str(output_path),
                        'suggestion': f"Directory will be created: mkdir -p '{output_dir}'"
                    })
                else:
                    issues.append({
                        'type': 'directory_parent_missing',
                        'field': 'output.output_dir',
                        'path': str(output_path),
                        'suggestion': f"Parent directory does not exist: {parent}"
                    })

        # 2. Check cache_dir (from cache section)
        cache_dir = getattr(self.cache, 'cache_dir', None)
        if cache_dir:
            # Expand ~ to home directory
            cache_dir_expanded = os.path.expanduser(cache_dir)
            cache_path = Path(cache_dir_expanded)
            if not cache_path.exists():
                parent = cache_path.parent
                if parent.exists():
                    issues.append({
                        'type': 'directory_missing',
                        'field': 'cache.cache_dir',
                        'path': str(cache_path),
                        'suggestion': f"Directory will be created: mkdir -p '{cache_dir_expanded}'"
                    })
                else:
                    issues.append({
                        'type': 'directory_parent_missing',
                        'field': 'cache.cache_dir',
                        'path': str(cache_path),
                        'suggestion': f"Parent directory does not exist: {parent}"
                    })

        # 3. Check entity_cache_dir (from image_search.entity_cache section)
        image_search = getattr(self, 'image_search', None)
        if image_search:
            entity_cache_config = getattr(image_search, 'entity_cache', None)
            if entity_cache_config:
                entity_cache_dir = getattr(entity_cache_config, 'cache_dir', None)
                if entity_cache_dir:
                    # Expand ~ to home directory
                    entity_cache_dir_expanded = os.path.expanduser(entity_cache_dir)
                    entity_cache_path = Path(entity_cache_dir_expanded)
                    if not entity_cache_path.exists():
                        parent = entity_cache_path.parent
                        if parent.exists():
                            issues.append({
                                'type': 'directory_missing',
                                'field': 'image_search.entity_cache.cache_dir',
                                'path': str(entity_cache_path),
                                'suggestion': f"Directory will be created: mkdir -p '{entity_cache_dir_expanded}'"
                            })
                        else:
                            issues.append({
                                'type': 'directory_parent_missing',
                                'field': 'image_search.entity_cache.cache_dir',
                                'path': str(entity_cache_path),
                                'suggestion': f"Parent directory does not exist: {parent}"
                            })

        # 4. Check download directory parent exists
        if hasattr(self.download, 'download_dir'):
            download_dir = self.download.download_dir
            if download_dir:
                download_path = Path(download_dir)
                if download_path.is_absolute():
                    parent = download_path.parent
                    if not parent.exists():
                        issues.append({
                            'type': 'directory_parent_missing',
                            'field': 'download.download_dir',
                            'path': str(download_path),
                            'suggestion': f"Parent directory does not exist: {parent}"
                        })

        # 5. Check for custom config file references
        # Check if there's a 'config' field in any section that references external config files
        # This is typically used in profiles or includes
        # Look for patterns like 'config_file', 'include', 'extends', etc.
        custom_config_fields = ['config_file', 'include', 'extends', 'profile']
        for section_name in ['project', 'pipeline', 'download']:
            section = getattr(self, section_name, None)
            if section:
                for field_name in custom_config_fields:
                    if hasattr(section, field_name):
                        config_file = getattr(section, field_name, None)
                        if config_file and isinstance(config_file, str):
                            config_path = Path(config_file)
                            # Skip if relative path (will be resolved from project_dir)
                            if config_path.is_absolute() and not config_path.exists():
                                issues.append({
                                    'type': 'config_file_missing',
                                    'field': f'{section_name}.{field_name}',
                                    'path': str(config_path),
                                    'suggestion': f"Config file not found: {config_file}"
                                })

        return issues

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Convert dataclass to dict recursively."""
        result = {}
        for field_name in dir(obj):
            if field_name.startswith('_'):
                continue
            try:
                value = getattr(obj, field_name)
                if hasattr(value, '__dataclass_fields__'):
                    result[field_name] = self._dataclass_to_dict(value)
                elif isinstance(value, (list, tuple)):
                    result[field_name] = list(value)
                elif not callable(value):
                    result[field_name] = value
            except Exception:
                continue
        return result

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

        # Stock footage segment selection mode
        valid_stock_selection_modes = {'best_in_block', 'first_in_block', 'rotate_in_block'}
        if self.stock_footage.selection_mode not in valid_stock_selection_modes:
            errors.append(
                f"stock_footage.selection_mode must be one of {valid_stock_selection_modes}, "
                f"got '{self.stock_footage.selection_mode}'"
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

            # coherence_penalty_threshold must be >= min_source_diversity
            coherence_threshold = safe_get_config_value(chapter_grouping, 'coherence_penalty_threshold')
            min_diversity = safe_get_config_value(chapter_grouping, 'min_source_diversity')
            if coherence_threshold is not None and min_diversity is not None:
                if coherence_threshold < min_diversity:
                    errors.append(
                        f"matching.chapter_grouping.coherence_penalty_threshold "
                        f"({coherence_threshold}) must be >= min_source_diversity ({min_diversity})"
                    )

            # multi_chapter_assignment_strategy must be one of valid options
            strategy = safe_get_config_value(chapter_grouping, 'multi_chapter_assignment_strategy')
            valid_strategies = ['first', 'split', 'best_match']
            if strategy is not None and strategy not in valid_strategies:
                errors.append(
                    f"matching.chapter_grouping.multi_chapter_assignment_strategy "
                    f"('{strategy}') must be one of {valid_strategies}"
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

        # US-112-002: Cross-validation between iterative_matching and matching configs
        iterative = safe_get_config_value(self, 'iterative_matching')
        if iterative:
            # Check if iterative matching is enabled
            iterative_enabled = safe_get_config_value(iterative, 'enabled', True)

            # 1. iterative_matching.enabled=False when matching.min_confidence > 0.9
            min_conf = safe_get_config_value(self.matching, 'min_confidence', 0.7)
            if iterative_enabled and min_conf > 0.9:
                errors.append(
                    f"iterative_matching.enabled=true requires matching.min_confidence <= 0.9, "
                    f"got {min_conf}. Set min_confidence <= 0.9 or disable iterative_matching"
                )

            # 2. iterative_matching.max_iterations >= matching.max_retries
            max_iterations = safe_get_config_value(iterative, 'max_iterations', 5)
            max_retries = safe_get_config_value(self.matching, 'max_retries', 3)
            if max_iterations < max_retries:
                errors.append(
                    f"iterative_matching.max_iterations ({max_iterations}) should be >= "
                    f"matching.max_retries ({max_retries})"
                )

            # 3. iterative_matching.target_confidence >= matching.low_confidence_threshold
            target_confidence = safe_get_config_value(iterative, 'target_confidence', 0.90)
            low_conf_threshold = safe_get_config_value(self.matching, 'low_confidence_threshold', 0.5)
            if target_confidence < low_conf_threshold:
                errors.append(
                    f"iterative_matching.target_confidence ({target_confidence}) should be >= "
                    f"matching.low_confidence_threshold ({low_conf_threshold})"
                )

            # 4. Test for iterative chapter boost when chapter_matching_enabled is False
            # This is a warning, not an error - iterative_chapter_boost has no effect without chapter matching
            chapter_matching_enabled = safe_get_config_value(self.matching, 'chapter_matching_enabled', False)
            iterative_chapter_boost = safe_get_config_value(iterative, 'iterative_chapter_boost', 0.1)
            if not chapter_matching_enabled and iterative_chapter_boost > 0:
                # Just log a debug message - this is informational, not blocking
                logger.debug(
                    f"iterative_matching.iterative_chapter_boost ({iterative_chapter_boost}) has no effect "
                    f"when matching.chapter_matching_enabled is False"
                )

        # US-112-006: Video search budget and distribution constraints
        # 1. results_per_keyword must be <= max_total_results
        vs_results_per_kw = safe_get_config_value(self.video_search, 'results_per_keyword', 20)
        vs_max_total = safe_get_config_value(self.video_search, 'max_total_results', 200)
        if vs_results_per_kw > vs_max_total:
            errors.append(
                f"video_search.results_per_keyword ({vs_results_per_kw}) must be <= "
                f"video_search.max_total_results ({vs_max_total})"
            )

        # 2. max_videos_per_channel must be < results_per_keyword
        max_videos_per_channel = safe_get_config_value(self.video_search, 'max_videos_per_channel', 3)
        if max_videos_per_channel >= vs_results_per_kw:
            errors.append(
                f"video_search.max_videos_per_channel ({max_videos_per_channel}) must be < "
                f"video_search.results_per_keyword ({vs_results_per_kw})"
            )

        # 3. search_timeout must be > 0
        search_timeout = safe_get_config_value(self.video_search, 'search_timeout', 30)
        if search_timeout <= 0:
            errors.append(
                f"video_search.search_timeout must be > 0, got {search_timeout}"
            )

        # 4. search_budget vs video_search consistency
        # Both sections should have compatible results_per_keyword and max_total_results
        sb_results_per_kw = safe_get_config_value(self.search_budget, 'results_per_keyword', 20)
        sb_max_total = safe_get_config_value(self.search_budget, 'max_total_results', 200)

        # Warn if search_budget and video_search have different values for results_per_keyword
        if sb_results_per_kw != vs_results_per_kw:
            logger.warning(
                f"search_budget.results_per_keyword ({sb_results_per_kw}) differs from "
                f"video_search.results_per_keyword ({vs_results_per_kw}). "
                f"video_search values take precedence during search."
            )

        # Warn if search_budget and video_search have different values for max_total_results
        if sb_max_total != vs_max_total:
            logger.warning(
                f"search_budget.max_total_results ({sb_max_total}) differs from "
                f"video_search.max_total_results ({vs_max_total}). "
                f"video_search values take precedence during search."
            )

        return errors

    def _validate_dependencies(self) -> List[str]:
        """Validate dependency relationships between config sections (US-142-012).

        Validates that enabled features have their dependencies met:
        - chapter_matching_enabled requires extract_video_chapters enabled
        - iterative_chapter_boost requires iterative_matching enabled
        - region_backoff requires mullvad enabled
        - search_budget_aware requires search_budget section configured

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # 1. chapter_matching_enabled requires extract_video_chapters
        chapter_matching = safe_get_config_value(self.matching, 'chapter_matching_enabled', True)
        # extract_video_chapters is in matching.context_enrichment
        context_enrichment = safe_get_config_value(self.matching, 'context_enrichment')
        if context_enrichment:
            extract_chapters = safe_get_config_value(context_enrichment, 'extract_video_chapters', True)
        else:
            extract_chapters = True  # default if not configured
        if chapter_matching and not extract_chapters:
            errors.append(
                "matching.chapter_matching_enabled=true requires "
                "matching.context_enrichment.extract_video_chapters=true"
            )

        # 2. iterative_chapter_boost requires iterative_matching enabled
        iterative_chapter_boost = safe_get_config_value(
            self.iterative_matching, 'iterative_chapter_boost', 0.1
        )
        iterative_enabled = safe_get_config_value(self.iterative_matching, 'enabled', True)
        if iterative_chapter_boost > 0 and not iterative_enabled:
            errors.append(
                "iterative_matching.iterative_chapter_boost > 0 requires "
                "iterative_matching.enabled=true"
            )

        # 3. region_backoff requires mullvad enabled
        region_backoff_enabled = safe_get_config_value(self.download.region_backoff, 'enabled', False)
        mullvad_enabled = safe_get_config_value(self.download.mullvad, 'enabled', False)
        if region_backoff_enabled and not mullvad_enabled:
            errors.append(
                "download.region_backoff.enabled=true requires "
                "download.mullvad.enabled=true"
            )

        # 4. search_budget_aware requires valid search_budget values
        search_budget_aware = safe_get_config_value(self.video_search, 'search_budget_aware', True)
        if search_budget_aware:
            sb_max_total = safe_get_config_value(self.search_budget, 'max_total_results', 200)
            sb_results_per_kw = safe_get_config_value(self.search_budget, 'results_per_keyword', 20)
            if sb_max_total <= 0:
                errors.append(
                    "video_search.search_budget_aware=true requires "
                    "search_budget.max_total_results > 0"
                )
            if sb_results_per_kw <= 0:
                errors.append(
                    "video_search.search_budget_aware=true requires "
                    "search_budget.results_per_keyword > 0"
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

    def _compute_field_hashes(self) -> None:
        """Compute and store hashes for all config fields at startup.

        This enables runtime drift detection - we can detect if config values
        change during pipeline execution and warn about unexpected mutations.
        """
        self._field_hashes.clear()

        # Known section names
        sections = [
            'transcription', 'embedding', 'indexing', 'vision',
            'scene_detection', 'audio_analysis', 'matching',
            'negative_matching', 'remix', 'zero_download_remix',
            'image_search', 'keyword', 'llm', 'enhanced',
            'downloading', 'download', 'stock_footage', 'deduplication',
            'output', 'multi_style', 'logging', 'cache', 'pipeline',
            'api_keys', 'healing', 'iterative_matching', 'rate_limit',
            'broll', 'global_cache', 'silent_video', 'search_budget',
            'video_search', 'duration_tiers', 'project'
        ]

        for section in sections:
            section_config = getattr(self, section, None)
            if section_config is not None and is_dataclass(section_config):
                section_dict = asdict(section_config)
                section_hash = hashlib.md5(
                    yaml.dump(section_dict, sort_keys=True).encode()
                ).hexdigest()[:16]
                self._field_hashes[section] = section_hash

        logger.debug(f"Computed field hashes for {len(self._field_hashes)} config sections")

    def check_drift(self, force_warn: bool = False) -> List[Dict[str, Any]]:
        """Check for config drift since startup and log warnings.

        Detects when config values have changed during pipeline execution,
        which can indicate bugs or unintended mutations.

        Args:
            force_warn: If True, always log warnings even if already logged once.

        Returns:
            List of drift info dicts with keys: section, old_hash, new_hash, changed_fields
        """
        drift_detected = []

        # Skip if already logged warnings (unless forced)
        if self._drift_warnings_logged and not force_warn:
            return drift_detected

        for section, old_hash in self._field_hashes.items():
            section_config = getattr(self, section, None)
            if section_config is None or not is_dataclass(section_config):
                continue

            current_dict = asdict(section_config)
            new_hash = hashlib.md5(
                yaml.dump(current_dict, sort_keys=True).encode()
            ).hexdigest()[:16]

            if new_hash != old_hash:
                # Find which specific fields changed
                old_data = yaml.safe_load(yaml.dump({
                    k: v for k, v in self._field_hashes.items()
                    if k == section
                }, sort_keys=True))

                # Get old section data from initial config if stored
                changed_fields = self._detect_changed_fields(section, current_dict)

                drift_info = {
                    'section': section,
                    'old_hash': old_hash,
                    'new_hash': new_hash,
                    'changed_fields': changed_fields
                }
                drift_detected.append(drift_info)

                # Log warning
                logger.warning(
                    f"CONFIG DRIFT DETECTED in '{section}': "
                    f"values changed during pipeline execution. "
                    f"Changed fields: {', '.join(changed_fields) if changed_fields else 'unknown'}"
                )

        if drift_detected and not self._drift_warnings_logged:
            self._drift_warnings_logged = True

        return drift_detected

    def _detect_changed_fields(self, section: str, current_dict: Dict[str, Any]) -> List[str]:
        """Detect which specific fields changed in a section.

        Args:
            section: Section name
            current_dict: Current section values

        Returns:
            List of field names that changed
        """
        # This requires storing the original section data
        # For now, we'll return a simplified list - the section hash changed
        # In a more complete implementation, we'd store initial values
        return [f"{section} (hash changed)"]


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


def load_config(config_path: str = "config.yaml", skip_final_validation: bool = False) -> Config:
    """
    Load config from file and set as global instance.

    Args:
        config_path: Path to config.yaml
        skip_final_validation: If True, skip API key and constraint validation (for --dry-run-config)

    Returns:
        Loaded Config object
    """
    global _global_config

    with _config_lock:
        _global_config = Config.from_yaml(config_path, skip_final_validation=skip_final_validation)

        # Validate and warn (only if not skipped in from_yaml)
        if not skip_final_validation:
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
    # Schema validation
    'validate_config_schema', 'ConfigValidationError',
    # Section configs (for backward compatibility)
    'TranscriptionConfig', 'EmbeddingConfig', 'MatchingConfig', 'OutputConfig',
    'KeywordConfig', 'DownloadingConfig', 'LoggingConfig', 'CacheConfig',
]
