"""
Pipeline State Management

Consolidates all pipeline state into a single dataclass,
replacing the 30+ instance attributes scattered across the Pipeline class.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import numpy for type hints
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@dataclass
class VoiceoverSegment:
    """A segment of the voiceover with timing and text"""
    index: int
    start: float
    end: float
    text: str
    duration: float = 0.0

    def __post_init__(self):
        if self.duration == 0.0:
            self.duration = self.end - self.start


@dataclass
class KeywordSegment:
    """A segment generated from a keyword (for keyword mode without voiceover).

    Used in montage mode where each keyword becomes a fixed-duration segment,
    or as intermediate representation before SRT generation in script mode.
    """
    index: int
    keyword: str
    start_time: float
    end_time: float
    description: str = ""  # Optional expanded description from LLM
    source: str = "keyword_montage"  # keyword_montage | keyword_script

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_voiceover_segment(self) -> VoiceoverSegment:
        """Convert to VoiceoverSegment for pipeline compatibility."""
        return VoiceoverSegment(
            index=self.index,
            start=self.start_time,
            end=self.end_time,
            text=self.description or self.keyword,
            duration=self.duration,
        )


@dataclass
class DetectedChapter:
    """A detected chapter/list item from voiceover analysis.

    Used to scope keyword extraction and video matching to chapter boundaries.
    For listicle content, each chapter represents one list item.
    """
    name: str  # Original name from ASR
    corrected_name: str  # ASR-corrected name (e.g., "Keatsahut" -> "Pizza Hut")
    rank: Optional[int]  # Position in list (15, 14, 13...) or None for non-listicle
    start_segment: int  # First segment index
    end_segment: int  # Last segment index (inclusive)
    keywords: List[str] = field(default_factory=list)  # Per-chapter search keywords
    description: str = ""  # Brief description of chapter content

    def contains_segment(self, segment_idx: int) -> bool:
        """Check if segment index is within this chapter."""
        return self.start_segment <= segment_idx <= self.end_segment

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TranscriptSegment:
    """A single transcript segment from video transcription"""
    index: int
    start_time: float
    end_time: float
    text: str
    source_file: str = ""
    # B-roll/silent video attributes
    is_broll: bool = False  # True if this is a silent/B-roll video segment
    description_source: str = ""  # How description was generated: 'vision', 'llm', 'keyword', or ''
    # Transcript source tracking for caption-first mode
    transcript_source: str = ""  # 'whisper', 'manual_caption', 'auto_caption', 'metadata'

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DownloadedVideo:
    """Metadata for a downloaded video file"""
    file: str
    url: str = ""
    title: str = ""
    channel: str = ""
    upload_date: str = ""
    duration: float = 0.0
    duration_tier: str = ""
    keyword: str = ""
    download_date: str = ""
    license: str = "Unknown"
    source: str = ""  # 'download', 'global_cache', etc.
    video_hash: str = ""
    face_score: float = 0.5


@dataclass
class AudioDownload:
    """Metadata for downloaded audio (audio-first mode)"""
    file: str
    video_id: str
    url: str = ""
    title: str = ""
    duration: float = 0.0
    keyword: str = ""
    from_cache: bool = False  # True if file already existed (skip rate-limit delay)


@dataclass
class CaptionDownload:
    """Metadata for downloaded caption (caption-first mode)"""
    file: str  # Path to downloaded .srt file
    video_id: str
    url: str = ""
    title: str = ""
    duration: float = 0.0
    keyword: str = ""
    language: str = "en"
    is_auto_generated: bool = True  # False for manual captions


@dataclass
class VideoCandidate:
    """
    Video metadata before download decision (caption-first mode).

    In caption-first mode, we fetch video metadata and captions BEFORE
    downloading any media. This allows us to:
    1. Fetch captions for all candidate videos
    2. Only download audio for videos without captions (Whisper fallback)
    3. Match against caption transcripts
    4. Download only matched video segments

    This dramatically reduces bandwidth and processing time since most
    YouTube videos have captions available.
    """
    video_id: str
    url: str
    title: str = ""
    channel: str = ""
    duration: float = 0.0
    duration_tier: str = ""
    keyword: str = ""
    upload_date: str = ""
    # Caption state (populated by CAPTION stage)
    has_captions: bool = False
    caption_language: str = ""
    is_auto_caption: bool = True
    # Transcript source after processing
    transcript_source: str = ""  # 'manual_caption', 'auto_caption', 'whisper', ''

    def to_dict(self) -> Dict[str, Any]:
        return {
            'video_id': self.video_id,
            'url': self.url,
            'title': self.title,
            'channel': self.channel,
            'duration': self.duration,
            'duration_tier': self.duration_tier,
            'keyword': self.keyword,
            'upload_date': self.upload_date,
            'has_captions': self.has_captions,
            'caption_language': self.caption_language,
            'is_auto_caption': self.is_auto_caption,
            'transcript_source': self.transcript_source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoCandidate':
        return cls(
            video_id=data.get('video_id', ''),
            url=data.get('url', ''),
            title=data.get('title', ''),
            channel=data.get('channel', ''),
            duration=data.get('duration', 0.0),
            duration_tier=data.get('duration_tier', ''),
            keyword=data.get('keyword', ''),
            upload_date=data.get('upload_date', ''),
            has_captions=data.get('has_captions', False),
            caption_language=data.get('caption_language', ''),
            is_auto_caption=data.get('is_auto_caption', True),
            transcript_source=data.get('transcript_source', ''),
        )


@dataclass
class Match:
    """A match between a voiceover segment and video clip"""
    segment_index: int
    video_file: str
    video_start: float
    video_end: float
    confidence: float
    strategy: str = ""
    reason: str = ""
    face_score: float = 0.5


@dataclass
class IterativeMatchState:
    """State tracking for high matches mode iterative matching.

    Tracks progress through multiple download → match cycles when
    trying to achieve a target confidence coverage.
    """
    iteration_count: int = 0
    coverage_history: List[float] = field(default_factory=list)
    videos_added_per_iteration: List[int] = field(default_factory=list)
    final_coverage: float = 0.0
    target_achieved: bool = False
    weak_segment_count: int = 0


@dataclass
class EntityImage:
    """Downloaded image for an entity"""
    entity: str
    file: str
    source_url: str = ""
    width: int = 0
    height: int = 0


@dataclass
class EntityVideo:
    """Downloaded stock video for an entity"""
    entity: str
    file: str
    source: str = ""  # 'pexels', 'pixabay'
    duration: float = 0.0


@dataclass
class PipelineState:
    """
    Central state object for the video matching pipeline.

    All pipeline stages read from and write to this object.
    This replaces the scattered instance attributes in the Pipeline class.
    """

    # === INPUT STATE ===
    project_dir: str = ""  # Project directory path
    voiceover_path: str = ""
    num_keywords: int = None  # Number of keywords to extract (CLI override)
    voiceover_segments: List[VoiceoverSegment] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topic_context: str = ""
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)

    # === CHAPTER DETECTION STATE ===
    chapters: List[DetectedChapter] = field(default_factory=list)  # Detected chapters/list items
    is_listicle: bool = False  # True if listicle/ranking content detected
    chapter_detection_enabled: bool = True  # Whether chapter detection was attempted

    # === VIDEO METADATA STATE (caption-first mode) ===
    # Video candidates discovered before download decision
    video_candidates: List[VideoCandidate] = field(default_factory=list)

    # === DOWNLOAD STATE ===
    downloaded_videos: List[DownloadedVideo] = field(default_factory=list)
    downloaded_audio: List[AudioDownload] = field(default_factory=list)
    failed_keywords: List[str] = field(default_factory=list)
    global_cache_videos: List[Dict[str, Any]] = field(default_factory=list)
    remix_files: List[str] = field(default_factory=list)  # Video paths from REMIX stage

    # === CAPTION STATE (caption-first mode) ===
    caption_downloads: List[CaptionDownload] = field(default_factory=list)
    videos_need_audio: List[str] = field(default_factory=list)  # Video IDs needing Whisper fallback

    # === ENTITY MEDIA STATE ===
    entity_images: Dict[str, EntityImage] = field(default_factory=dict)
    entity_videos: Dict[str, EntityVideo] = field(default_factory=dict)

    # === TRANSCRIPTION STATE ===
    transcripts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    embeddings: List[Any] = field(default_factory=list)  # numpy arrays
    text_metadata: List[Dict[str, Any]] = field(default_factory=list)
    embedding_index: Any = None  # FAISS index

    # === PREMISE STATE ===
    # Video premises: brief topic/theme summaries for theme-based matching
    # Format: {video_id: "Documentary about restaurant closures"}
    video_premises: Dict[str, str] = field(default_factory=dict)

    # === SCENE DETECTION STATE ===
    scene_data: Dict[str, Any] = field(default_factory=dict)  # video_name -> VideoSceneData

    # === MATCHING STATE ===
    matches: List[Match] = field(default_factory=list)
    alternatives: Dict[int, List[Match]] = field(default_factory=dict)  # segment_idx -> alt matches

    # === B-ROLL STATE ===
    broll_downloads: List[Dict[str, Any]] = field(default_factory=list)  # B-roll specific downloads
    broll_matches: List[Dict[str, Any]] = field(default_factory=list)  # Silent scene matches

    # === OUTPUT STATE ===
    output_files: List[Path] = field(default_factory=list)
    otio_files: List[Path] = field(default_factory=list)

    # === LOCATION STATE ===
    location_chapters: List[Dict[str, Any]] = field(default_factory=list)

    # === RUNTIME STATE ===
    face_preference: str = "neutral"
    stage_timings: Dict[str, float] = field(default_factory=dict)

    # === ITERATIVE MATCHING STATE ===
    iterative_match_state: IterativeMatchState = None

    # === METADATA ===
    # General purpose metadata storage for stages
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_video_count(self) -> int:
        """Get total number of downloaded videos"""
        return len(self.downloaded_videos)

    def get_match_count(self) -> int:
        """Get number of matched segments"""
        return len(self.matches)

    def get_segment_count(self) -> int:
        """Get number of voiceover segments"""
        return len(self.voiceover_segments)

    def clear_downloads(self):
        """Clear download state for fresh start"""
        self.downloaded_videos = []
        self.downloaded_audio = []
        self.failed_keywords = []

    def clear_matches(self):
        """Clear matching state for re-matching"""
        self.matches = []
        self.alternatives = {}

    def to_checkpoint_dict(self) -> Dict[str, Any]:
        """Convert state to dict for checkpointing"""
        return {
            'voiceover_path': self.voiceover_path,
            'keywords': self.keywords,
            'topic_context': self.topic_context,
            'segment_count': len(self.voiceover_segments),
            'video_count': len(self.downloaded_videos),
            'match_count': len(self.matches),
            'stage_timings': self.stage_timings,
        }

    @classmethod
    def from_legacy_pipeline(cls, pipeline: Any) -> 'PipelineState':
        """
        Create PipelineState from legacy Pipeline object.

        Used for gradual migration - allows existing code to work
        while we transition to the new architecture.
        """
        state = cls()

        # Copy basic state
        state.keywords = getattr(pipeline, 'keywords', [])
        state.topic_context = getattr(pipeline, 'topic_context', '')
        state.extracted_entities = getattr(pipeline, 'extracted_entities', [])
        state.failed_keywords = getattr(pipeline, 'failed_keywords', [])
        state.face_preference = getattr(pipeline, 'face_preference', 'neutral')
        state.stage_timings = getattr(pipeline, 'stage_timings', {})

        # Copy voiceover segments (convert dicts to VoiceoverSegment)
        for i, seg in enumerate(getattr(pipeline, 'voiceover_segments', [])):
            if isinstance(seg, dict):
                state.voiceover_segments.append(VoiceoverSegment(
                    index=seg.get('index', i),
                    start=seg.get('start', 0.0),
                    end=seg.get('end', 0.0),
                    text=seg.get('text', ''),
                ))
            else:
                state.voiceover_segments.append(seg)

        # Copy downloaded videos (convert dicts to DownloadedVideo)
        for vid in getattr(pipeline, 'downloaded_videos', []):
            if isinstance(vid, dict):
                state.downloaded_videos.append(DownloadedVideo(
                    file=vid.get('file', vid.get('path', '')),
                    url=vid.get('url', ''),
                    title=vid.get('title', ''),
                    channel=vid.get('channel', ''),
                    duration=vid.get('duration', 0.0),
                    duration_tier=vid.get('duration_tier', vid.get('tier', '')),
                    keyword=vid.get('keyword', ''),
                    source=vid.get('source', 'download'),
                ))
            else:
                state.downloaded_videos.append(vid)

        # Copy transcription state
        state.transcripts = getattr(pipeline, 'transcripts', {})
        state.embeddings = getattr(pipeline, 'embeddings', [])
        state.text_metadata = getattr(pipeline, 'text_metadata', [])
        state.embedding_index = getattr(pipeline, 'embedding_index', None)

        # Copy matches (convert dicts to Match)
        # Handle multiple checkpoint formats:
        # - Old: video_file, video_start, video_end
        # - New: source_file, start_time (from scene detection)
        for m in getattr(pipeline, 'matches', []):
            if isinstance(m, dict):
                state.matches.append(Match(
                    segment_index=m.get('segment_index', m.get('vo_index', 0)),
                    video_file=m.get('video_file', m.get('source_file', m.get('file', ''))),
                    video_start=m.get('video_start', m.get('start_time', m.get('start', 0.0))),
                    video_end=m.get('video_end', m.get('end_time', m.get('end', 0.0))),
                    confidence=m.get('confidence', 0.0),
                    strategy=m.get('strategy', ''),
                    reason=m.get('reason', ''),
                ))
            else:
                state.matches.append(m)

        # Copy entity media
        state.entity_images = getattr(pipeline, 'entity_images', {})
        state.entity_videos = getattr(pipeline, 'entity_videos', {})

        return state
