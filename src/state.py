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
import logging

# Try to import numpy for type hints
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

logger = logging.getLogger(__name__)


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
class VideoSearchResult:
    """Search result for a video (before download)"""
    video_id: str
    url: str = ""
    title: str = ""
    channel: str = ""
    duration: float = 0.0
    duration_tier: str = ""
    keyword: str = ""


@dataclass
class PipelineState:
    """
    Central state object for the video matching pipeline.

    All pipeline stages read from and write to this object.
    Simplified 7-stage pipeline: ANALYZE → VIDEO_SEARCH → CAPTION → MATCH →
    ITERATIVE_MATCH → DOWNLOAD_SEGMENTS → OUTPUT
    """

    # === INPUT STATE ===
    voiceover_path: str = ""
    voiceover_segments: List[VoiceoverSegment] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    topic_context: str = ""
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)

    # === VIDEO SEARCH STATE (replaces DOWNLOAD) ===
    video_ids: List[str] = field(default_factory=list)  # YouTube video IDs from search
    video_search_results: List[VideoSearchResult] = field(default_factory=list)  # Full search metadata
    search_failed_keywords: List[str] = field(default_factory=list)  # Keywords with no results

    # === CAPTION STATE ===
    caption_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # video_id -> caption data
    text_metadata: List[Dict[str, Any]] = field(default_factory=list)  # Populated by _populate_text_metadata

    # === MATCHING STATE ===
    matches: List[Match] = field(default_factory=list)
    alternatives: Dict[int, List[Match]] = field(default_factory=dict)  # segment_idx -> alt matches

    # === DOWNLOAD STATE (for segment downloads only) ===
    downloaded_segments: List[DownloadedVideo] = field(default_factory=list)  # Only matched segments

    # === OUTPUT STATE ===
    output_files: List[Path] = field(default_factory=list)
    otio_files: List[Path] = field(default_factory=list)

    # === RUNTIME STATE ===
    face_preference: str = "neutral"
    stage_timings: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Defensive initialization for fields that must never be None."""
        # Belt-and-suspenders fix for US-38-008: ensure text_metadata is never None
        # This guards against edge cases like deserialization or manual construction
        if self.text_metadata is None:
            self.text_metadata = []

    def validate_state_attributes(self) -> List[str]:
        """
        Validate and initialize required state attributes after checkpoint restoration.

        Ensures all required fields exist with proper default values. This is called
        after checkpoint restoration to handle incomplete state data from older
        checkpoints or corrupted files.

        Returns:
            List of field names that were initialized (empty list if all were valid)
        """
        initialized_fields = []

        # Required fields with their default values
        required_fields = {
            'text_metadata': [],
            'caption_results': {},
            'video_ids': [],
        }

        for field_name, default_value in required_fields.items():
            # Check if field is missing or None
            current_value = getattr(self, field_name, None)
            if current_value is None:
                setattr(self, field_name, default_value)
                logger.warning(f"Restored missing {field_name} after checkpoint load")
                initialized_fields.append(field_name)

        return initialized_fields

    def get_video_count(self) -> int:
        """Get total number of video IDs from search"""
        return len(self.video_ids)

    def get_match_count(self) -> int:
        """Get number of matched segments"""
        return len(self.matches)

    def get_segment_count(self) -> int:
        """Get number of voiceover segments"""
        return len(self.voiceover_segments)

    def clear_search(self):
        """Clear video search state for fresh start"""
        self.video_ids = []
        self.video_search_results = []
        self.search_failed_keywords = []

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
            'video_count': len(self.video_ids),
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
        import re
        state = cls()

        # Copy basic state
        state.keywords = getattr(pipeline, 'keywords', [])
        state.topic_context = getattr(pipeline, 'topic_context', '')
        state.extracted_entities = getattr(pipeline, 'extracted_entities', [])
        state.search_failed_keywords = getattr(pipeline, 'failed_keywords', [])
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

        # Copy video_ids directly if present (e.g., from test mocks or newer pipelines)
        for vid_id in getattr(pipeline, 'video_ids', []):
            if vid_id and vid_id not in state.video_ids:
                state.video_ids.append(vid_id)

        # Migrate downloaded_videos to video_ids (extract video IDs from URLs)
        # Only if video_ids wasn't copied directly above
        if not state.video_ids:
            for vid in getattr(pipeline, 'downloaded_videos', []):
                url = vid.get('url', '') if isinstance(vid, dict) else getattr(vid, 'url', '')
                if url and ('youtube.com' in url or 'youtu.be' in url):
                    match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', url)
                    if match:
                        video_id = match.group(1)
                        if video_id not in state.video_ids:
                            state.video_ids.append(video_id)

        # Copy video_search_results if present
        state.video_search_results = list(getattr(pipeline, 'video_search_results', []))

        # Copy caption results
        state.caption_results = getattr(pipeline, 'caption_results', {})

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

        return state
