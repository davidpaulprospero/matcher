"""
Configuration management with YAML support
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict


@dataclass
class TranscriptionConfig:
    """Whisper transcription settings"""
    provider: str = "faster-whisper"  # faster-whisper (4x faster), openai-whisper (original)
    model: str = "base"  # tiny, base, small, medium, large, large-v2, large-v3
    language: str = "en"
    use_gpu: bool = True  # GPU is much faster (10-50x)
    compute_type: str = "auto"  # auto, float16 (GPU), int8 (CPU), float32 (fallback)


@dataclass
class EmbeddingConfig:
    """Embedding settings"""
    provider: str = "gemini"  # gemini, voyage, local
    local_model: str = "all-MiniLM-L6-v2"  # For local embeddings
    gemini_model: str = "models/text-embedding-004"
    voyage_model: str = "voyage-2"
    
    # Hybrid embeddings (text + visual combined)
    hybrid_mode: bool = True  # Combine text and visual embeddings
    text_weight: float = 0.7  # Weight for text embedding in hybrid
    visual_weight: float = 0.3  # Weight for visual embedding in hybrid


@dataclass
class IndexingConfig:
    """FAISS/Annoy indexing settings for faster retrieval"""
    use_faiss: bool = True  # Use FAISS for similarity search (falls back to numpy if not installed)
    index_type: str = "flat"  # "flat" (exact) or "ivf" (faster, approximate)


@dataclass
class VisionConfig:
    """Multi-modal vision settings"""
    enabled: bool = True
    provider: str = "gemini"  # gemini, openai
    frames_per_scene: int = 3  # Keyframes to extract per scene
    describe_scenes: bool = True
    max_scenes_per_video: int = 10  # Max scenes to describe per video (cost control)


@dataclass
class VideoFilterConfig:
    """Video relevance filtering settings (pre-embedding step)"""
    enabled: bool = True
    relevance_threshold: float = 0.3  # Remove videos scoring below this (0-1)
    use_llm: bool = True  # Use LLM for smart filtering
    
    # Filtering approach
    check_transcript: bool = True  # Analyze video transcript for relevance
    check_filename: bool = True  # Check filename for relevance hints
    
    # Optional explicit filters
    exclude_keywords: List[str] = field(default_factory=list)  # Videos containing these are excluded
    include_keywords: List[str] = field(default_factory=list)  # Videos must contain at least one of these
    
    # Auto-detect main topic from voiceover
    auto_detect_topic: bool = True
    
    # Output
    save_filter_report: bool = True  # Save report of filtered videos
    filter_report_filename: str = "filtered_videos_report.txt"


@dataclass 
class MatchingConfig:
    """LLM matching settings"""
    # Tiered matching
    primary_provider: str = "gemini"  # gemini, anthropic
    secondary_provider: str = "anthropic"  # For ambiguous matches
    local_llm_provider: str = "ollama"  # For finishing touches
    local_llm_model: str = "llama3.2"
    use_local_for_review: bool = True
    
    # Thresholds
    top_k_candidates: int = 10
    confidence_threshold: float = 0.5
    skip_llm_threshold: float = 0.85  # Skip LLM if embedding similarity > this
    ambiguous_threshold: float = 0.6  # Use secondary LLM if confidence < this
    
    # Two-stage matching optimization
    embedding_candidates: int = 20  # Retrieve more candidates from embedding search
    llm_rerank_candidates: int = 5  # Only send top N to LLM for reranking
    
    # Batching (disabled when smart_reuse is true for accurate tracking)
    batch_size: int = 10
    
    # Contextual matching
    context_window: int = 2  # Consider N segments before/after
    
    # Smart reuse prevention
    smart_reuse: bool = True
    max_clip_reuse: int = 1  # Max times a clip can be reused (1 = no reuse)
    reuse_penalty: float = 0.5  # Heavy penalty for reused clips (50% confidence drop)
    sequential_when_reuse: bool = True  # Process sequentially to track reuse accurately
    
    # Duration-aware scoring
    duration_scoring_enabled: bool = True
    ideal_speed_range: List[float] = field(default_factory=lambda: [0.85, 1.15])  # 85%-115% = no penalty
    soft_penalty_range: List[float] = field(default_factory=lambda: [0.7, 1.5])  # 70%-150% = small penalty
    duration_penalty_factor: float = 0.1  # Confidence reduction per penalty tier
    
    # Caching
    cache_llm_responses: bool = True


@dataclass
class KeywordConfig:
    """Keyword extraction settings"""
    enabled: bool = True
    extract_entities: bool = True
    extract_topics: bool = True
    boost_keyword_matches: float = 0.1  # Boost confidence for keyword matches


@dataclass
class KeywordWeightsConfig:
    """Auto-detected keyword weighting (topic-agnostic)"""
    enabled: bool = True  # Enable keyword-based confidence boosting
    auto_detect: bool = True  # Auto-extract top terms from voiceover using TF-IDF
    auto_detect_count: int = 20  # Number of top terms to auto-boost
    boost_factor: float = 0.15  # Confidence boost per matching keyword
    
    # Optional manual overrides (per-project customization)
    custom_boost_terms: List[str] = field(default_factory=list)  # Terms to always boost
    custom_penalty_terms: List[str] = field(default_factory=list)  # Terms to penalize


@dataclass
class KeywordSuggesterConfig:
    """Keyword suggester settings for --suggest-keywords"""
    # Output counts
    num_keywords: int = 30  # Top keywords to show
    num_entities: int = 20  # Top entities to show
    num_search_queries: int = 50  # Total search queries to output (more is better for variety)
    
    # Extraction categories (all enabled by default)
    extract_names: bool = True  # Scientists, people, characters
    extract_places: bool = True  # Locations, countries, cities
    extract_numbers: bool = True  # Statistics, dates, quantities, measurements
    extract_events: bool = True  # Historical events, incidents
    extract_technical_terms: bool = True  # Scientific/technical vocabulary
    extract_talking_points: bool = True  # Main themes per segment
    
    # Search query types to generate
    query_types: List[str] = field(default_factory=lambda: [
        "scenic/landscape footage",
        "documentary footage",
        "news footage",
        "drone/aerial footage",
        "stock footage",
        "historical footage",
        "scientific visualization"
    ])
    
    # Custom keywords to always include (user can add their own)
    custom_keywords: List[str] = field(default_factory=list)
    
    # Output format
    include_yt_dlp_commands: bool = True
    output_filename: str = "suggested_footage_keywords.txt"


@dataclass
class NegativeMatchingConfig:
    """Negative matching rules"""
    enabled: bool = True
    rules: List[str] = field(default_factory=lambda: [
        "Don't match talking head shots to action narration",
        "Don't match static images to dynamic narration", 
        "Avoid matching unrelated b-roll to specific statements"
    ])


@dataclass
class StrategyTrackConfig:
    """Configuration for a single strategy track"""
    strategy: str = "visual_first"  # visual_first, different_source, keyword_only, embedding_diversity
    weight_visual: float = 0.7  # For visual_first
    weight_text: float = 0.3   # For visual_first
    min_embedding_distance: float = 0.3  # For embedding_diversity


@dataclass
class VarietyEnforcementConfig:
    """Settings to ensure variety across tracks"""
    require_different_source: bool = True  # Each track must use different source video
    exclude_same_clip: bool = True  # Never use exact same clip on multiple tracks
    min_time_distance: float = 10.0  # Clips must be N seconds apart (same source)
    min_embedding_distance: float = 0.3  # V4+ must have distance > this from V1-V3


@dataclass
class OutputConfig:
    """Output settings"""
    include_alternatives: bool = True
    num_alternatives: int = 2  # Number of alternative matches per segment (V2-V3)
    
    # Strategy tracks (V4+)
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = field(default_factory=lambda: [
        "visual_first",       # V4: Prioritize scene descriptions
        "different_source",   # V5: Force different source video
        "keyword_only",       # V6: Match on keywords only
        "embedding_diversity", # V7: Maximally different from V1
        "source_rotation"     # V8: Cycle through all sources systematically
    ])
    
    # Variety enforcement
    variety: VarietyEnforcementConfig = field(default_factory=VarietyEnforcementConfig)
    
    markers_for_low_confidence: bool = True
    markers_for_gaps: bool = True
    markers_for_speed: bool = True
    
    # Timeline start timecode (for EDL marker export)
    timeline_start_tc: str = "01:00:00:00"  # Standard broadcast start


@dataclass
class LoggingConfig:
    """Comprehensive logging settings"""
    enabled: bool = True
    log_dir: str = "./logs"
    
    # What to log
    log_match_decisions: bool = True  # Every match decision
    log_api_calls: bool = True  # API calls and estimated cost
    log_performance: bool = True  # Timing for each stage
    log_config_snapshot: bool = True  # Config used for this run
    log_warnings_errors: bool = True  # Warnings and errors
    
    # Output formats
    output_log_file: bool = True  # Human-readable .log file
    output_json_file: bool = True  # Machine-parseable .json file


@dataclass
class TopicConfig:
    """Topic segmentation settings"""
    enabled: bool = True
    min_topic_segments: int = 3  # Minimum segments to form a topic
    similarity_threshold: float = 0.7  # Threshold for grouping segments


@dataclass
class SceneDetectionConfig:
    """Scene detection settings (PySceneDetect integration)"""
    enabled: bool = True
    
    # Speed preset: fast, balanced, accurate
    preset: str = "fast"
    
    # Force CUDA GPU acceleration
    force_cuda: bool = True
    
    # Manual overrides (None = use preset values)
    downscale: Optional[int] = None  # 1=full, 2=half, 4=quarter, 6=faster
    frame_skip: Optional[int] = None  # 0=none, 2=every 3rd frame
    threshold: Optional[float] = None  # Higher = fewer cuts detected
    min_scene_length: Optional[int] = None  # Minimum frames per scene
    
    # Video filter
    min_video_duration: float = 0  # Only process videos longer than this (seconds)
    
    # DaVinci Resolve timecode fix
    start_timecode: str = "01:00:00:00"  # Match your DaVinci timeline start
    
    # Output
    output_otio: bool = True  # Generate OTIO files per video
    output_scene_index: bool = True  # Generate merged scene index JSON
    
    # Audio analysis (silence/speech detection)
    audio_analysis: bool = True  # Enable librosa audio analysis


@dataclass
class LLMConfig:
    """LLM provider settings for keyword extraction and other tasks"""
    provider: str = "anthropic"  # anthropic, google
    model: str = "claude-3-haiku-20240307"  # or gemini-1.5-flash
    api_key: Optional[str] = None  # Will use env var if not set


@dataclass
class DownloadTierConfig:
    """Configuration for a single download duration tier"""
    min_duration: int = 20  # Minimum video duration in seconds
    max_duration: int = 120  # Maximum video duration in seconds
    per_keyword: int = 3  # Videos to download per keyword


@dataclass
class DownloadConfig:
    """Video download settings (integrated yt-dlp downloader)"""
    enabled: bool = True
    
    # Duration tiers (all three run by default)
    tiers: Dict[str, Dict] = field(default_factory=lambda: {
        'short': {'min': 20, 'max': 120, 'per_keyword': 3},
        'medium': {'min': 120, 'max': 600, 'per_keyword': 3},
        'long': {'min': 600, 'max': 1500, 'per_keyword': 1}
    })
    
    # Download settings
    quality: str = "1080p"  # 1080p, 720p, best
    format: str = "mp4"
    min_views: int = 0
    max_concurrent: int = 3  # Parallel downloads
    delay_between_keywords: int = 2  # Seconds between keyword batches
    
    # Sites to search
    sites: List[str] = field(default_factory=lambda: ["youtube"])
    
    # DaVinci Resolve transcoding
    davinci_mode: bool = True
    davinci_codec: str = "h264"  # h264, h265, prores, dnxhd
    davinci_prores_profile: str = "proxy"  # proxy, lt, standard, hq
    delete_original: bool = True  # Delete original after transcode
    
    # Hardware acceleration
    hw_accel: str = "auto"  # auto, nvidia, amd, intel, mac, none
    hw_quality: str = "high"  # low, medium, high
    
    # Source tracking
    track_sources: bool = True
    sources_file: str = "sources.json"


@dataclass
class PipelineConfig:
    """End-to-end pipeline settings"""
    # Keyword extraction
    max_keywords: int = 50  # Maximum keywords to extract
    keyword_extraction_method: str = "llm"  # llm, tfidf
    
    # Pre-run confirmation
    confirm_before_download: bool = True
    
    # Stage control
    skip_download: bool = False
    skip_transcribe: bool = False
    skip_match: bool = False
    download_only: bool = False
    match_only: bool = False
    
    # Resume support
    enable_checkpoints: bool = True


@dataclass
class Config:
    """Main configuration"""
    # Paths
    downloaded_videos_dir: str = "./downloaded_videos"
    otio_output_dir: str = "./otio_output"
    voiceover_path: str = "./voiceover/voiceover.srt"
    output_path: str = "./matched_timeline.otio"
    cache_dir: str = "./.cache"
    
    # API Keys (from environment)
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # Sub-configs
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    video_filter: VideoFilterConfig = field(default_factory=VideoFilterConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    keywords: KeywordConfig = field(default_factory=KeywordConfig)
    keyword_weights: KeywordWeightsConfig = field(default_factory=KeywordWeightsConfig)
    keyword_suggester: KeywordSuggesterConfig = field(default_factory=KeywordSuggesterConfig)
    negative_matching: NegativeMatchingConfig = field(default_factory=NegativeMatchingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)
    scene_detection: SceneDetectionConfig = field(default_factory=SceneDetectionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    
    # Incremental indexing
    incremental: bool = True
    
    # Watch mode
    watch_mode: bool = False
    watch_interval: int = 30  # seconds
    
    def __post_init__(self):
        """Load API keys from environment"""
        self.gemini_api_key = self.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.anthropic_api_key = self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.voyage_api_key = self.voyage_api_key or os.getenv("VOYAGE_API_KEY")
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Set LLM API key based on provider
        if self.llm.api_key is None:
            if self.llm.provider == 'anthropic':
                self.llm.api_key = self.anthropic_api_key
            elif self.llm.provider == 'google':
                self.llm.api_key = self.gemini_api_key
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle nested configs
        if 'transcription' in data:
            data['transcription'] = TranscriptionConfig(**data['transcription'])
        if 'embedding' in data:
            data['embedding'] = EmbeddingConfig(**data['embedding'])
        if 'indexing' in data:
            data['indexing'] = IndexingConfig(**data['indexing'])
        if 'vision' in data:
            data['vision'] = VisionConfig(**data['vision'])
        if 'video_filter' in data:
            data['video_filter'] = VideoFilterConfig(**data['video_filter'])
        if 'matching' in data:
            data['matching'] = MatchingConfig(**data['matching'])
        if 'keywords' in data:
            data['keywords'] = KeywordConfig(**data['keywords'])
        if 'keyword_weights' in data:
            data['keyword_weights'] = KeywordWeightsConfig(**data['keyword_weights'])
        if 'keyword_suggester' in data:
            data['keyword_suggester'] = KeywordSuggesterConfig(**data['keyword_suggester'])
        if 'negative_matching' in data:
            data['negative_matching'] = NegativeMatchingConfig(**data['negative_matching'])
        if 'output' in data:
            output_data = data['output']
            # Handle nested variety config
            if 'variety' in output_data:
                output_data['variety'] = VarietyEnforcementConfig(**output_data['variety'])
            data['output'] = OutputConfig(**output_data)
        if 'logging' in data:
            data['logging'] = LoggingConfig(**data['logging'])
        if 'topics' in data:
            data['topics'] = TopicConfig(**data['topics'])
        if 'scene_detection' in data:
            data['scene_detection'] = SceneDetectionConfig(**data['scene_detection'])
        if 'llm' in data:
            data['llm'] = LLMConfig(**data['llm'])
        if 'download' in data:
            data['download'] = DownloadConfig(**data['download'])
        if 'pipeline' in data:
            data['pipeline'] = PipelineConfig(**data['pipeline'])
        
        config = cls(**data)
        return config
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file"""
        data = self._to_dict()
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def _to_dict(self) -> dict:
        """Convert config to dictionary"""
        result = {}
        for key, value in asdict(self).items():
            if key.endswith('_api_key') and value:
                # Don't save API keys to file
                continue
            result[key] = value
        return result


def create_default_config(path: str = "config.yaml"):
    """Create a default configuration file"""
    config = Config()
    config.to_yaml(path)
    return config


def load_config(path: str = "config.yaml") -> Config:
    """Load config from file or create default"""
    if Path(path).exists():
        return Config.from_yaml(path)
    else:
        return create_default_config(path)
