"""
Configuration Management System v3.0

Centralized configuration loader that:
- Loads all settings from config.yaml as single source of truth
- Provides type-safe dataclasses for all sections
- Validates configuration on load
- Supports hot-reload for development
- Logs configuration access patterns
- Falls back to sensible defaults when values missing

All components should import Config from this module and access settings
through the config object rather than hardcoding values.
"""

import os
import sys
import json
import logging
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List, Dict, Any, Tuple, Union
from datetime import datetime

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION DATACLASSES - Mirrors config.yaml structure
# =============================================================================

@dataclass
class TranscriptionConfig:
    """Transcription settings (faster-whisper)"""
    provider: str = "faster-whisper"
    model: str = "base"  # tiny, base, small, medium, large, large-v2, large-v3
    language: str = "en"
    use_gpu: bool = True
    compute_type: str = "auto"  # auto, float16, int8, float32
    
    # Parallel processing
    max_workers: int = 4
    batch_size: int = 10
    
    # VAD settings
    vad_filter: bool = True
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200
    
    # Cache settings
    cache_transcriptions: bool = True
    cache_dir: str = "transcriptions"


@dataclass
class EmbeddingConfig:
    """Embedding generation settings"""
    provider: str = "gemini"  # gemini, voyage, local
    
    # Model settings per provider
    gemini_model: str = "models/text-embedding-004"
    voyage_model: str = "voyage-2"
    local_model: str = "all-MiniLM-L6-v2"
    
    # Batch processing
    batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Caching
    cache_embeddings: bool = True
    cache_batch_results: bool = True


@dataclass
class IndexingConfig:
    """FAISS indexing settings"""
    index_type: str = "flat"  # flat, ivf
    use_faiss: bool = True
    
    # IVF settings (for large datasets)
    ivf_nlist: int = 100  # Number of clusters
    ivf_nprobe: int = 10  # Clusters to search
    
    # Similarity settings
    normalize_embeddings: bool = True
    similarity_metric: str = "cosine"  # cosine, dot, euclidean


@dataclass
class VisionConfig:
    """Vision processing settings"""
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    
    # Selective processing
    enabled: bool = True
    min_words_per_scene: int = 5
    coverage_threshold: float = 0.3
    max_scenes_per_video: int = 50
    
    # Frame extraction
    frame_format: str = "jpg"
    frame_quality: int = 85
    
    # Cost management
    max_api_calls_per_run: int = 100
    estimated_cost_per_call: float = 0.001


@dataclass
class SceneDetectionConfig:
    """Scene detection settings (PySceneDetect)"""
    enabled: bool = True
    preset: str = "balanced"  # fast, balanced, accurate
    
    # Preset overrides (if not using preset)
    threshold: float = 27.0
    min_scene_len: int = 15  # frames
    downscale_factor: int = 4
    frame_skip: int = 2
    
    # Presets definition
    presets: Dict[str, Dict] = field(default_factory=lambda: {
        "fast": {"downscale": 6, "frame_skip": 3, "threshold": 30.0, "min_scene": 20},
        "balanced": {"downscale": 4, "frame_skip": 2, "threshold": 27.0, "min_scene": 15},
        "accurate": {"downscale": 2, "frame_skip": 0, "threshold": 25.0, "min_scene": 10}
    })
    
    # Audio analysis integration
    audio_analysis: bool = True
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.3
    
    # Hardware acceleration
    use_gpu: bool = True
    force_gpu: bool = False


@dataclass
class AudioAnalysisConfig:
    """Audio analysis settings (librosa)"""
    enabled: bool = True
    sample_rate: int = 22050
    
    # Silence detection
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.3
    
    # Speech detection
    speech_threshold: float = 0.5
    min_speech_duration: float = 0.2
    
    # Spectral features
    spectral_centroid_range: Tuple[int, int] = (300, 4000)
    spectral_flatness_threshold: float = 0.3


@dataclass
class MatchingConfig:
    """Matching engine settings"""
    # Confidence thresholds
    min_confidence: float = 0.7
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.5
    
    # Clip reuse prevention
    max_clip_reuse: int = 1
    reuse_penalty: float = 0.5
    
    # Two-stage matching
    embedding_candidates: int = 20  # Retrieve from FAISS
    llm_rerank_candidates: int = 5  # Send to LLM
    
    # Duration scoring
    ideal_speed_range: Tuple[float, float] = (0.85, 1.15)
    soft_speed_range: Tuple[float, float] = (0.70, 1.50)
    duration_penalty_factor: float = 0.1
    
    # Keyword matching
    keyword_boost: float = 0.05
    entity_boost: float = 0.08
    
    # LLM providers (tiered)
    primary_provider: str = "gemini"
    secondary_provider: str = "anthropic"
    local_provider: str = "ollama"
    
    # Model names
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.2"
    
    # Caching
    cache_llm_responses: bool = True
    cache_ttl_hours: int = 24


@dataclass 
class KeywordConfig:
    """Keyword extraction settings"""
    provider: str = "gemini"  # gemini, anthropic, tfidf
    
    # Extraction settings
    max_keywords: int = 30
    min_keyword_length: int = 3
    
    # Entity extraction
    extract_entities: bool = True
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "GPE", "ORG", "DATE", "EVENT"
    ])
    
    # Keyword weights (TF-IDF fallback)
    use_tfidf_weights: bool = True
    tfidf_max_features: int = 100
    
    # Footage suffixes
    add_footage_suffixes: bool = True
    footage_suffixes: List[str] = field(default_factory=lambda: [
        "4K footage", "news footage", "drone footage", 
        "aerial footage", "stock footage", "documentary footage"
    ])
    
    # Batch processing
    batch_size: int = 50


@dataclass
class KeywordRemixConfig:
    """Keyword remix settings (when confidence < threshold)"""
    enabled: bool = True
    confidence_threshold: float = 0.9
    max_remix_attempts: int = 2
    remix_keyword_count: int = 10


@dataclass
class DownloadingConfig:
    """Video downloading settings (yt-dlp)"""
    # Output settings
    output_dir: str = "downloaded_videos"
    organize_by_keyword: bool = True
    
    # Duration tiers
    duration_tiers: Dict[str, Dict] = field(default_factory=lambda: {
        "short": {"min": 20, "max": 120, "count": 8},
        "medium": {"min": 120, "max": 600, "count": 8},
        "long": {"min": 600, "max": 1500, "count": 5},
        "longer": {"min": 1500, "max": 3000, "count": 5}
    })
    
    # Quality settings
    preferred_quality: str = "1080"
    max_quality: str = "2160"
    prefer_h264: bool = True  # Avoid VP9/AV1 transcoding
    
    # Codec handling
    davinci_compatible_codecs: List[str] = field(default_factory=lambda: [
        "h264", "hevc", "prores", "dnxhd", "dnxhr"
    ])
    transcode_codecs: List[str] = field(default_factory=lambda: [
        "vp9", "av1", "vp8"
    ])
    
    # Transcoding
    transcode_codec: str = "h264"
    transcode_crf: int = 18
    use_hw_accel: bool = True
    hw_accel_type: str = "auto"  # auto, cuda, amf, qsv, videotoolbox
    
    # Rate limiting
    max_downloads_per_keyword: int = 10
    download_delay: float = 1.0
    
    # Checkpointing
    use_checkpoints: bool = True
    checkpoint_file: str = "download_checkpoint.json"


@dataclass
class StockFootageConfig:
    """Stock footage API settings (Pexels, Pixabay)"""
    enabled: bool = True
    
    # Per-source settings
    pexels_enabled: bool = True
    pixabay_enabled: bool = True
    
    # Download limits
    per_keyword: int = 3
    min_duration: int = 5
    max_duration: int = 60
    min_height: int = 720
    
    # Preferences
    prefer_landscape: bool = True
    
    # Rate limiting
    request_interval: float = 0.5


@dataclass
class ImageDownloadConfig:
    """Image download settings"""
    enabled: bool = False
    
    # Sources
    pexels_enabled: bool = True
    pixabay_enabled: bool = True
    unsplash_enabled: bool = True
    
    # Quality filtering
    min_size_mb: float = 1.0
    min_width: int = 1920
    prefer_landscape: bool = True
    
    # Limits
    per_keyword: int = 3


@dataclass
class DeduplicationConfig:
    """Video deduplication settings"""
    enabled: bool = True
    
    # Hash settings
    hash_threshold: int = 10  # Hamming distance (0-64)
    use_first_frame: bool = True
    
    # Actions
    auto_delete: bool = True
    generate_report: bool = True


@dataclass
class OutputConfig:
    """Output generation settings"""
    # Output directory
    output_dir: str = "output"
    
    # File formats
    generate_otio: bool = True
    generate_edl: bool = True
    generate_xml: bool = True
    generate_report: bool = True
    
    # Track structure
    num_alternatives: int = 2  # V2-V3
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = field(default_factory=lambda: [
        "visual_first", "different_source", "keyword_only",
        "embedding_diversity", "source_rotation"
    ])
    
    # Track naming
    track_names: Dict[str, str] = field(default_factory=lambda: {
        "V1": "Primary",
        "V2": "Alternative 1",
        "V3": "Alternative 2",
        "V4": "Visual-First",
        "V5": "Different-Source",
        "V6": "Keyword-Only",
        "V7": "Embedding-Diversity",
        "V8": "Source-Rotation",
        "A8": "Voiceover"
    })
    
    # Confidence colors (for markers)
    confidence_colors: Dict[str, str] = field(default_factory=lambda: {
        "high": "GREEN",      # >= 0.80
        "good": "CYAN",       # >= 0.60
        "medium": "YELLOW",   # >= 0.40
        "low": "ORANGE",      # >= 0.20
        "gap": "RED"          # < 0.20
    })
    
    # Speed recommendations
    include_speed_markers: bool = True
    max_recommended_speed: float = 1.5


@dataclass
class MultiStyleConfig:
    """Multi-style OTIO generation settings"""
    enabled: bool = False
    styles: List[str] = field(default_factory=lambda: ["default", "strict"])
    
    # Style presets
    presets: Dict[str, Dict] = field(default_factory=lambda: {
        "default": {"confidence_threshold": 0.5, "num_alternatives": 2},
        "strict": {"confidence_threshold": 0.7, "num_alternatives": 1},
        "stock_heavy": {"prefer_stock_footage": True, "confidence_threshold": 0.5},
        "fast_paced": {"prefer_shorter_clips": True, "ideal_speed_range": [0.7, 1.0]},
        "cinematic": {"prefer_longer_clips": True, "ideal_speed_range": [1.0, 1.5]}
    })


@dataclass
class LoggingConfig:
    """Logging settings"""
    log_dir: str = "logs"
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # Dual output
    log_to_file: bool = True
    log_to_console: bool = True
    generate_json_log: bool = True
    
    # Log content
    log_api_calls: bool = True
    log_match_decisions: bool = True
    log_performance: bool = True
    
    # Cost tracking
    track_api_costs: bool = True
    
    # Config logging
    log_config_access: bool = False  # Enable to debug config usage
    warn_on_hardcoded: bool = True   # Warn when components use hardcoded values


@dataclass
class CacheConfig:
    """Cache settings"""
    cache_dir: str = ".cache"
    
    # Cache types
    cache_transcriptions: bool = True
    cache_embeddings: bool = True
    cache_scenes: bool = True
    cache_llm_responses: bool = True
    cache_vision: bool = True
    
    # Cross-project cache
    cross_project_cache: bool = False
    cross_project_cache_dir: str = ""
    
    # Cache validation
    validate_cache_on_load: bool = True
    use_file_hash: bool = True  # Fast hash (path+size+mtime)


@dataclass
class WatcherConfig:
    """File watcher settings"""
    enabled: bool = False
    watch_interval: int = 30  # seconds
    incremental: bool = True  # Only process new files


@dataclass
class PipelineConfig:
    """Pipeline automation settings"""
    # Stage control
    skip_download: bool = False
    skip_transcription: bool = False
    skip_scene_detection: bool = False
    skip_matching: bool = False
    
    # Resume/retry
    resume_enabled: bool = True
    max_retries: int = 3
    
    # Parallel processing
    parallel_transcription: bool = True
    parallel_embedding: bool = True


@dataclass
class APIKeysConfig:
    """API keys (loaded from environment)"""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    pexels_api_key: str = ""
    pixabay_api_key: str = ""
    unsplash_api_key: str = ""
    
    def __post_init__(self):
        """Load from environment if not set"""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.voyage_api_key = self.voyage_api_key or os.getenv("VOYAGE_API_KEY", "")
        self.pexels_api_key = self.pexels_api_key or os.getenv("PEXELS_API_KEY", "")
        self.pixabay_api_key = self.pixabay_api_key or os.getenv("PIXABAY_API_KEY", "")
        self.unsplash_api_key = self.unsplash_api_key or os.getenv("UNSPLASH_API_KEY", "")


# =============================================================================
# MAIN CONFIG CLASS
# =============================================================================

@dataclass
class Config:
    """
    Master configuration class.
    
    All components should access settings through this class.
    Load from config.yaml using Config.from_yaml(path).
    """
    # Project settings
    project_name: str = "voiceover_match"
    project_dir: str = "."
    
    # Section configs
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    keyword: KeywordConfig = field(default_factory=KeywordConfig)
    keyword_remix: KeywordRemixConfig = field(default_factory=KeywordRemixConfig)
    downloading: DownloadingConfig = field(default_factory=DownloadingConfig)
    stock_footage: StockFootageConfig = field(default_factory=StockFootageConfig)
    image_download: ImageDownloadConfig = field(default_factory=ImageDownloadConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    multi_style: MultiStyleConfig = field(default_factory=MultiStyleConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    
    # Metadata
    _config_path: str = ""
    _config_hash: str = ""
    _loaded_at: str = ""
    
    def __post_init__(self):
        """Validate and resolve paths"""
        self._loaded_at = datetime.now().isoformat()
        self._resolve_paths()
    
    def _resolve_paths(self):
        """Resolve relative paths to absolute"""
        base = Path(self.project_dir).resolve()
        
        # Resolve cache dir
        if not Path(self.cache.cache_dir).is_absolute():
            self.cache.cache_dir = str(base / self.cache.cache_dir)
        
        # Resolve output dir
        if not Path(self.output.output_dir).is_absolute():
            self.output.output_dir = str(base / self.output.output_dir)
        
        # Resolve download dir
        if not Path(self.downloading.output_dir).is_absolute():
            self.downloading.output_dir = str(base / self.downloading.output_dir)
        
        # Resolve log dir
        if not Path(self.logging.log_dir).is_absolute():
            self.logging.log_dir = str(base / self.logging.log_dir)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml
            
        Returns:
            Config object with all settings loaded
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            config = cls()
            config._config_path = str(config_path)
            return config
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls()
        
        # Compute hash for change detection
        config_hash = hashlib.md5(
            yaml.dump(data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        # Build config object
        config = cls._from_dict(data)
        config._config_path = str(config_path)
        config._config_hash = config_hash
        
        logger.info(f"Loaded config from {config_path} (hash: {config_hash})")
        
        return config
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Build Config from dictionary"""
        config = cls()
        
        # Top-level settings
        config.project_name = data.get("project_name", config.project_name)
        config.project_dir = data.get("project_dir", config.project_dir)
        
        # Section configs
        section_mapping = {
            "transcription": (TranscriptionConfig, "transcription"),
            "embedding": (EmbeddingConfig, "embedding"),
            "indexing": (IndexingConfig, "indexing"),
            "vision": (VisionConfig, "vision"),
            "scene_detection": (SceneDetectionConfig, "scene_detection"),
            "audio_analysis": (AudioAnalysisConfig, "audio_analysis"),
            "matching": (MatchingConfig, "matching"),
            "keyword": (KeywordConfig, "keyword"),
            "keyword_remix": (KeywordRemixConfig, "keyword_remix"),
            "downloading": (DownloadingConfig, "downloading"),
            "stock_footage": (StockFootageConfig, "stock_footage"),
            "image_download": (ImageDownloadConfig, "image_download"),
            "deduplication": (DeduplicationConfig, "deduplication"),
            "output": (OutputConfig, "output"),
            "multi_style": (MultiStyleConfig, "multi_style"),
            "logging": (LoggingConfig, "logging"),
            "cache": (CacheConfig, "cache"),
            "watcher": (WatcherConfig, "watcher"),
            "pipeline": (PipelineConfig, "pipeline"),
            "api_keys": (APIKeysConfig, "api_keys"),
        }
        
        for yaml_key, (dataclass_type, attr_name) in section_mapping.items():
            section_data = data.get(yaml_key, {})
            if section_data:
                section_config = cls._build_dataclass(dataclass_type, section_data)
                setattr(config, attr_name, section_config)
        
        return config
    
    @staticmethod
    def _build_dataclass(dataclass_type, data: Dict) -> Any:
        """Build a dataclass from dict, handling missing/extra fields"""
        valid_fields = {f.name for f in fields(dataclass_type)}
        filtered_data = {}
        
        for key, value in data.items():
            if key in valid_fields:
                filtered_data[key] = value
            else:
                logger.debug(f"Ignoring unknown config field: {key}")
        
        try:
            return dataclass_type(**filtered_data)
        except TypeError as e:
            logger.warning(f"Error building {dataclass_type.__name__}: {e}")
            return dataclass_type()
    
    def to_yaml(self, output_path: str = None) -> str:
        """
        Serialize config to YAML.
        
        Args:
            output_path: If provided, write to file
            
        Returns:
            YAML string
        """
        data = self._to_dict()
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(yaml_str)
            logger.info(f"Saved config to {output_path}")
        
        return yaml_str
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excluding private fields)"""
        result = {
            "project_name": self.project_name,
            "project_dir": self.project_dir,
        }
        
        # Add all section configs
        sections = [
            "transcription", "embedding", "indexing", "vision",
            "scene_detection", "audio_analysis", "matching", "keyword",
            "keyword_remix", "downloading", "stock_footage", "image_download",
            "deduplication", "output", "multi_style", "logging",
            "cache", "watcher", "pipeline"
        ]
        
        for section in sections:
            section_config = getattr(self, section, None)
            if section_config:
                result[section] = asdict(section_config)
        
        # Don't include API keys in serialization
        return result
    
    def reload(self) -> bool:
        """
        Reload config from file if changed.
        
        Returns:
            True if config was reloaded
        """
        if not self._config_path:
            return False
        
        config_path = Path(self._config_path)
        if not config_path.exists():
            return False
        
        # Check hash
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
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
        
        return True
    
    def validate(self) -> List[str]:
        """
        Validate configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check API keys for enabled features
        if self.matching.primary_provider == "gemini" and not self.api_keys.gemini_api_key:
            errors.append("GEMINI_API_KEY required for gemini matching")
        
        if self.matching.secondary_provider == "anthropic" and not self.api_keys.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY required for anthropic matching")
        
        if self.embedding.provider == "gemini" and not self.api_keys.gemini_api_key:
            errors.append("GEMINI_API_KEY required for gemini embeddings")
        
        if self.embedding.provider == "voyage" and not self.api_keys.voyage_api_key:
            errors.append("VOYAGE_API_KEY required for voyage embeddings")
        
        if self.stock_footage.pexels_enabled and not self.api_keys.pexels_api_key:
            errors.append("PEXELS_API_KEY required for Pexels stock footage")
        
        if self.stock_footage.pixabay_enabled and not self.api_keys.pixabay_api_key:
            errors.append("PIXABAY_API_KEY required for Pixabay stock footage")
        
        # Check paths
        project_path = Path(self.project_dir)
        if not project_path.exists():
            errors.append(f"Project directory does not exist: {self.project_dir}")
        
        # Check value ranges
        if not 0 <= self.matching.min_confidence <= 1:
            errors.append("matching.min_confidence must be between 0 and 1")
        
        if self.matching.max_clip_reuse < 0:
            errors.append("matching.max_clip_reuse must be >= 0")
        
        if self.transcription.max_workers < 1:
            errors.append("transcription.max_workers must be >= 1")
        
        return errors
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """
        Get a nested config value by dot-notation path.
        
        Example:
            config.get_nested("matching.min_confidence")
            config.get_nested("output.track_names.V1")
        """
        parts = path.split(".")
        obj = self
        
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                if self.logging.log_config_access:
                    logger.debug(f"Config path not found: {path}")
                return default
        
        return obj


# =============================================================================
# GLOBAL CONFIG INSTANCE
# =============================================================================

_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
        logger.warning("Using default config - call load_config() to load from file")
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
    _global_config = Config.from_yaml(config_path)
    
    # Validate
    errors = _global_config.validate()
    for error in errors:
        logger.warning(f"Config validation: {error}")
    
    return _global_config


def set_config(config: Config):
    """Set the global config instance"""
    global _global_config
    _global_config = config


# =============================================================================
# CONFIG ACCESS HELPERS
# =============================================================================

def get_api_key(provider: str) -> Optional[str]:
    """Get API key for a provider"""
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


def ensure_dirs(config: Config):
    """Create necessary directories from config"""
    dirs = [
        config.cache.cache_dir,
        config.output.output_dir,
        config.downloading.output_dir,
        config.logging.log_dir,
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# =============================================================================
# BACKWARDS COMPATIBILITY
# =============================================================================

# Aliases for old code that imports specific classes
TranscriptionConfig = TranscriptionConfig
EmbeddingConfig = EmbeddingConfig
MatchingConfig = MatchingConfig
OutputConfig = OutputConfig
