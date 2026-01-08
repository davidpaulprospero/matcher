"""
Pipeline State Management

Consolidates all pipeline state into a single dataclass,
replacing the 30+ instance attributes scattered across the Pipeline class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    voiceover_path: str = ""
    voiceover_segments: List[VoiceoverSegment] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topic_context: str = ""
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)

    # === DOWNLOAD STATE ===
    downloaded_videos: List[DownloadedVideo] = field(default_factory=list)
    downloaded_audio: List[AudioDownload] = field(default_factory=list)
    failed_keywords: List[str] = field(default_factory=list)
    global_cache_videos: List[Dict[str, Any]] = field(default_factory=list)

    # === ENTITY MEDIA STATE ===
    entity_images: Dict[str, EntityImage] = field(default_factory=dict)
    entity_videos: Dict[str, EntityVideo] = field(default_factory=dict)

    # === TRANSCRIPTION STATE ===
    transcripts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    embeddings: List[Any] = field(default_factory=list)  # numpy arrays
    text_metadata: List[Dict[str, Any]] = field(default_factory=list)
    embedding_index: Any = None  # FAISS index

    # === SCENE DETECTION STATE ===
    scene_data: Dict[str, Any] = field(default_factory=dict)  # video_name -> VideoSceneData

    # === MATCHING STATE ===
    matches: List[Match] = field(default_factory=list)
    alternatives: Dict[int, List[Match]] = field(default_factory=dict)  # segment_idx -> alt matches

    # === OUTPUT STATE ===
    output_files: List[Path] = field(default_factory=list)
    otio_files: List[Path] = field(default_factory=list)

    # === LOCATION STATE ===
    location_chapters: List[Dict[str, Any]] = field(default_factory=list)

    # === RUNTIME STATE ===
    face_preference: str = "neutral"
    stage_timings: Dict[str, float] = field(default_factory=dict)

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
        for m in getattr(pipeline, 'matches', []):
            if isinstance(m, dict):
                state.matches.append(Match(
                    segment_index=m.get('segment_index', m.get('vo_index', 0)),
                    video_file=m.get('video_file', m.get('file', '')),
                    video_start=m.get('video_start', m.get('start', 0.0)),
                    video_end=m.get('video_end', m.get('end', 0.0)),
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
