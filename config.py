"""
Configuration management with YAML support
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List


@dataclass
class TranscriptionConfig:
    """Whisper transcription settings"""
    model: str = "base"  # tiny, base, small, medium, large
    language: str = "en"
    use_gpu: bool = True  # GPU is much faster (10-50x)


@dataclass
class EmbeddingConfig:
    """Embedding settings"""
    provider: str = "gemini"  # gemini, voyage, local
    local_model: str = "all-MiniLM-L6-v2"  # For local embeddings
    gemini_model: str = "models/text-embedding-004"
    voyage_model: str = "voyage-2"


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
    
    # Batching (disabled when smart_reuse is true for accurate tracking)
    batch_size: int = 10
    
    # Contextual matching
    context_window: int = 2  # Consider N segments before/after
    
    # Smart reuse prevention
    smart_reuse: bool = True
    max_clip_reuse: int = 1  # Max times a clip can be reused (1 = no reuse)
    reuse_penalty: float = 0.5  # Heavy penalty for reused clips (50% confidence drop)
    sequential_when_reuse: bool = True  # Process sequentially to track reuse accurately
    
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
        "embedding_diversity" # V7: Maximally different from V1
    ])
    
    # Variety enforcement
    variety: VarietyEnforcementConfig = field(default_factory=VarietyEnforcementConfig)
    
    markers_for_low_confidence: bool = True
    markers_for_gaps: bool = True
    markers_for_speed: bool = True
    
    # Timeline start timecode (for EDL marker export)
    timeline_start_tc: str = "01:00:00:00"  # Standard broadcast start


@dataclass
class TopicConfig:
    """Topic segmentation settings"""
    enabled: bool = True
    min_topic_segments: int = 3  # Minimum segments to form a topic
    similarity_threshold: float = 0.7  # Threshold for grouping segments


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
    vision: VisionConfig = field(default_factory=VisionConfig)
    video_filter: VideoFilterConfig = field(default_factory=VideoFilterConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    keywords: KeywordConfig = field(default_factory=KeywordConfig)
    keyword_suggester: KeywordSuggesterConfig = field(default_factory=KeywordSuggesterConfig)
    negative_matching: NegativeMatchingConfig = field(default_factory=NegativeMatchingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    topics: TopicConfig = field(default_factory=TopicConfig)
    
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
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Handle nested configs
        if 'transcription' in data:
            data['transcription'] = TranscriptionConfig(**data['transcription'])
        if 'embedding' in data:
            data['embedding'] = EmbeddingConfig(**data['embedding'])
        if 'vision' in data:
            data['vision'] = VisionConfig(**data['vision'])
        if 'video_filter' in data:
            data['video_filter'] = VideoFilterConfig(**data['video_filter'])
        if 'matching' in data:
            data['matching'] = MatchingConfig(**data['matching'])
        if 'keywords' in data:
            data['keywords'] = KeywordConfig(**data['keywords'])
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
        if 'topics' in data:
            data['topics'] = TopicConfig(**data['topics'])
        
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
