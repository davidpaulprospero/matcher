"""
Utility classes and functions
"""

import os
import sys
import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field, fields
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import threading
import time

logger = logging.getLogger(__name__)


# =============================================================================
# EMBEDDINGS UTILITIES
# =============================================================================

def is_embeddings_empty(embeddings: Any) -> bool:
    """
    Check if embeddings are None or empty.

    Handles numpy arrays, lists, and None safely without triggering
    "truth value of array is ambiguous" errors.

    Args:
        embeddings: Embeddings to check (numpy array, list, or None)

    Returns:
        True if embeddings are None or have length 0, False otherwise
    """
    return embeddings is None or len(embeddings) == 0


# =============================================================================
# PATH UTILITIES
# =============================================================================

def sanitize_path(path: Union[str, Path]) -> str:
    r"""
    Sanitize a file path by removing Windows extended-length path prefixes
    and converting to forward slashes.
    
    This handles paths like \\?\C:\... that can cause issues with some applications.
    
    Args:
        path: Path string or Path object
        
    Returns:
        Clean path string with forward slashes
    """
    path_str = str(path)
    
    # Remove Windows extended-length path prefix in various forms
    prefixes = [
        '\\\\?\\',    # Standard form: \\?\
        '\\\\.\\',    # Device form: \\.\
        '//?/',       # Forward slash form
        '//.//',      # Device forward slash
    ]
    
    for prefix in prefixes:
        if path_str.startswith(prefix):
            path_str = path_str[len(prefix):]
            break
    
    # Also check if it starts with ?\ or ?/ (edge case)
    if path_str.startswith('?\\') or path_str.startswith('?/'):
        path_str = path_str[2:]
    
    # Convert backslashes to forward slashes
    path_str = path_str.replace('\\', '/')
    
    # Remove any double slashes (except preserving drive letter format)
    while '//' in path_str:
        path_str = path_str.replace('//', '/')
    
    return path_str


def normalize_path(path: str) -> str:
    """
    Normalize path for consistent cache matching across platforms.

    Converts to forward slashes and lowercase for consistent key matching
    regardless of OS or path format.

    Args:
        path: File path string

    Returns:
        Normalized path (forward slashes, lowercase)
    """
    if not path:
        return ""
    return str(path).replace('\\', '/').lower()


def resolve_path(path: Union[str, Path], base_dir: Union[str, Path] = None) -> str:
    """
    Resolve a path to absolute and sanitize it.
    
    Args:
        path: Path to resolve
        base_dir: Optional base directory for relative paths
        
    Returns:
        Absolute, sanitized path string
    """
    path = Path(path)
    
    if not path.is_absolute() and base_dir:
        path = Path(base_dir) / path
    
    # Resolve to absolute (this may add \\?\ on Windows for long paths)
    try:
        path = path.resolve()
    except OSError:
        # If resolve fails, try to make it absolute without resolving symlinks
        if not path.is_absolute():
            path = Path.cwd() / path
    
    # Sanitize to remove any extended-length prefix
    return sanitize_path(path)


# =============================================================================
# FFMPEG DEBUG LOGGING
# =============================================================================

# Global FFmpeg debug log path (set by setup_ffmpeg_debug_log)
_ffmpeg_debug_log_path: Optional[Path] = None
_ffmpeg_debug_lock = threading.Lock()


def setup_ffmpeg_debug_log(log_dir: Union[str, Path]) -> Path:
    """
    Set up FFmpeg debug log file in the specified directory.

    Args:
        log_dir: Directory to create the debug log in

    Returns:
        Path to the debug log file
    """
    global _ffmpeg_debug_log_path

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    _ffmpeg_debug_log_path = log_dir / "ffmpeg_debug.log"

    # Write header
    with open(_ffmpeg_debug_log_path, 'w', encoding='utf-8') as f:
        f.write(f"# FFmpeg Debug Log\n")
        f.write(f"# Started: {datetime.now().isoformat()}\n")
        f.write(f"# This file captures low-level FFmpeg/OpenCV decoder messages\n")
        f.write(f"# (e.g., H.264 'mmco: unref short failure' warnings)\n")
        f.write("=" * 60 + "\n\n")

    return _ffmpeg_debug_log_path


def log_ffmpeg_debug(message: str, source: str = "ffmpeg"):
    """
    Log a message to the FFmpeg debug log file.

    Args:
        message: Message to log
        source: Source identifier (e.g., 'opencv', 'ffmpeg', 'scene_detection')
    """
    global _ffmpeg_debug_log_path

    if _ffmpeg_debug_log_path is None:
        return

    with _ffmpeg_debug_lock:
        try:
            with open(_ffmpeg_debug_log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().strftime("%H:%M:%S")
                f.write(f"[{timestamp}] [{source}] {message}\n")
        except Exception:
            pass  # Don't let debug logging break the main process


class FFmpegStderrCapture:
    """
    Context manager to capture stderr (FFmpeg/OpenCV decoder messages)
    and redirect them to the debug log file.

    Usage:
        with FFmpegStderrCapture("scene_detection"):
            cap = cv2.VideoCapture(video_path)
            # ... process video ...
    """

    def __init__(self, source: str = "opencv"):
        self.source = source
        self.old_stderr = None
        self.stderr_capture = None

    def __enter__(self):
        global _ffmpeg_debug_log_path

        # Only capture if debug log is set up
        if _ffmpeg_debug_log_path is None:
            return self

        try:
            import io
            # Save original stderr
            self.old_stderr = sys.stderr
            # Create a string buffer to capture stderr
            self.stderr_capture = io.StringIO()
            sys.stderr = self.stderr_capture
        except Exception:
            pass

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_stderr is not None:
            # Restore original stderr
            sys.stderr = self.old_stderr

            # Get captured content and log it
            if self.stderr_capture:
                captured = self.stderr_capture.getvalue()
                if captured.strip():
                    log_ffmpeg_debug(captured.strip(), self.source)
                self.stderr_capture.close()

        return False  # Don't suppress exceptions


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Chapter:
    """Represents a chapter/topic section in voiceover"""
    chapter_id: int
    start_segment_idx: int
    end_segment_idx: int
    title: str = ""
    topics: List[str] = field(default_factory=list)

    def contains_segment(self, segment_idx: int) -> bool:
        return self.start_segment_idx <= segment_idx <= self.end_segment_idx

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SRTSegment:
    """Represents a single SRT segment"""
    index: int
    start_time: float  # seconds
    end_time: float    # seconds
    text: str
    source_file: str = ""  # Source video file path (empty for voiceover)

    # Extended attributes
    keywords: List[str] = field(default_factory=list)
    entities: List = field(default_factory=list)  # List of entity dicts with text, type, context
    topic_id: Optional[int] = None
    topics: List[str] = field(default_factory=list)  # Topic keywords for this segment/video
    is_broll: bool = False  # True if silent/B-roll video (no speech, face_score < threshold)
    transcript_source: str = ""  # Source: 'whisper', 'manual_caption', 'auto_caption', 'metadata'

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (handles numpy types)"""
        return {
            'index': int(self.index),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'text': self.text,
            'source_file': self.source_file,
            'keywords': list(self.keywords) if self.keywords else [],
            'entities': list(self.entities) if self.entities else [],
            'topic_id': int(self.topic_id) if self.topic_id is not None else None,
            'topics': list(self.topics) if self.topics else [],
            'is_broll': bool(self.is_broll),
            'transcript_source': self.transcript_source or ''
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SRTSegment":
        """Create from dict, handling extra/missing fields gracefully"""
        # Get only the fields that SRTSegment expects
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        # Ensure required fields have defaults
        filtered_data.setdefault('index', 0)
        filtered_data.setdefault('start_time', 0.0)
        filtered_data.setdefault('end_time', 0.0)
        filtered_data.setdefault('text', '')
        filtered_data.setdefault('source_file', '')
        filtered_data.setdefault('keywords', [])
        filtered_data.setdefault('entities', [])
        filtered_data.setdefault('topic_id', None)
        filtered_data.setdefault('topics', [])
        filtered_data.setdefault('is_broll', False)
        filtered_data.setdefault('transcript_source', '')

        return cls(**filtered_data)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class SceneInfo:
    """Information about a video scene"""
    video_path: str
    scene_index: int
    start_time: float
    end_time: float
    
    # Scene description from vision model
    description: str = ""
    visual_keywords: List[str] = field(default_factory=list)
    
    # Keyframe paths
    keyframes: List[str] = field(default_factory=list)
    
    # Associated transcript segment
    transcript_segment: Optional[SRTSegment] = None
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (handles numpy types)"""
        return {
            'video_path': self.video_path,
            'scene_index': int(self.scene_index),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'description': self.description,
            'visual_keywords': list(self.visual_keywords) if self.visual_keywords else [],
            'keyframes': list(self.keyframes) if self.keyframes else [],
            'transcript_segment': self.transcript_segment.to_dict() if self.transcript_segment else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneInfo":
        """Create from dict, handling extra/missing fields gracefully"""
        # Handle nested transcript_segment
        transcript_seg = None
        if data.get('transcript_segment'):
            transcript_seg = SRTSegment.from_dict(data['transcript_segment'])
        
        # Get only valid fields
        valid_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        # Set defaults
        filtered_data.setdefault('video_path', '')
        filtered_data.setdefault('scene_index', 0)
        filtered_data.setdefault('start_time', 0.0)
        filtered_data.setdefault('end_time', 0.0)
        filtered_data.setdefault('description', '')
        filtered_data.setdefault('visual_keywords', [])
        filtered_data.setdefault('keyframes', [])
        filtered_data['transcript_segment'] = transcript_seg
        
        return cls(**filtered_data)


@dataclass
class VideoIndex:
    """Index entry for a video"""
    video_path: str
    video_hash: str
    duration: float
    
    # Transcription
    transcript_segments: List[SRTSegment] = field(default_factory=list)
    
    # Scenes
    scenes: List[SceneInfo] = field(default_factory=list)
    
    # Embeddings (stored separately, just keep flag)
    has_embeddings: bool = False
    
    # Metadata
    indexed_at: str = ""
    
    def to_dict(self) -> dict:
        return {
            'video_path': self.video_path,
            'video_hash': self.video_hash,
            'duration': self.duration,
            'transcript_segments': [s.to_dict() for s in self.transcript_segments],
            'scenes': [s.to_dict() for s in self.scenes],
            'has_embeddings': self.has_embeddings,
            'indexed_at': self.indexed_at
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VideoIndex":
        return cls(
            video_path=data['video_path'],
            video_hash=data['video_hash'],
            duration=data['duration'],
            transcript_segments=[SRTSegment.from_dict(s) for s in data.get('transcript_segments', [])],
            scenes=[SceneInfo.from_dict(s) for s in data.get('scenes', [])],
            has_embeddings=data.get('has_embeddings', False),
            indexed_at=data.get('indexed_at', '')
        )


@dataclass
class Match:
    """Represents a matched segment"""
    voiceover_segment: SRTSegment
    video_segment: SRTSegment
    video_scene: Optional[SceneInfo]
    confidence: float
    reasoning: str

    # Match quality indicators
    is_keyword_match: bool = False
    is_visual_match: bool = False
    embedding_similarity: float = 0.0

    # Reuse tracking
    clip_reuse_count: int = 0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict (handles numpy types)"""
        return {
            'voiceover_segment': self.voiceover_segment.to_dict(),
            'video_segment': self.video_segment.to_dict(),
            'video_scene': self.video_scene.to_dict() if self.video_scene else None,
            'confidence': float(self.confidence),  # Convert numpy float to Python float
            'reasoning': self.reasoning,
            'is_keyword_match': self.is_keyword_match,
            'is_visual_match': self.is_visual_match,
            'embedding_similarity': float(self.embedding_similarity),  # Convert numpy float
            'clip_reuse_count': int(self.clip_reuse_count)
        }


@dataclass
class AlternativeMatch:
    """Alternative match option"""
    video_segment: SRTSegment
    video_scene: Optional[SceneInfo]
    confidence: float
    reasoning: str


@dataclass
class StrategyMatch:
    """Match from an alternative strategy (V4-V7)"""
    video_segment: SRTSegment
    video_scene: Optional[SceneInfo]
    confidence: float
    reasoning: str
    strategy: str  # visual_first, different_source, keyword_only, embedding_diversity
    
    def to_dict(self) -> dict:
        return {
            'video_segment': self.video_segment.to_dict(),
            'video_scene': self.video_scene.to_dict() if self.video_scene else None,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'strategy': self.strategy
        }


@dataclass
class MatchResult:
    """Complete match result with alternatives and strategy matches"""
    primary_match: Match
    alternatives: List[AlternativeMatch] = field(default_factory=list)  # V2-V3
    secondary_matches: List[AlternativeMatch] = field(default_factory=list)  # V4-V6 (different video files from V1-V3)
    strategy_matches: List[StrategyMatch] = field(default_factory=list)  # V7+
    has_gap: bool = False  # True if no good match found
    gap_reason: str = ""


@dataclass
class Topic:
    """A topic cluster of voiceover segments"""
    topic_id: int
    name: str
    segments: List[SRTSegment]
    keywords: List[str]
    start_time: float
    end_time: float


# =============================================================================
# PROGRESS BAR
# =============================================================================

class ProgressBar:
    """Thread-safe progress bar with ETA"""
    
    def __init__(self, total: int, description: str = "", bar_length: int = 40):
        self.total = total
        self.current = 0
        self.description = description
        self.bar_length = bar_length
        self.start_time = time.time()
        self.lock = threading.Lock()
        self._last_print_time = 0
    
    def update(self, amount: int = 1, status: str = ""):
        """Update progress"""
        with self.lock:
            self.current += amount
            self._print_bar(status)
    
    def set(self, value: int, status: str = ""):
        """Set absolute progress"""
        with self.lock:
            self.current = value
            self._print_bar(status)
    
    def _print_bar(self, status: str = ""):
        """Print the progress bar"""
        # Throttle updates to avoid flickering
        current_time = time.time()
        if current_time - self._last_print_time < 0.1 and self.current < self.total:
            return
        self._last_print_time = current_time
        
        # Calculate progress
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.bar_length * progress)
        bar = "█" * filled + "░" * (self.bar_length - filled)
        
        # Calculate ETA
        elapsed = current_time - self.start_time
        if progress > 0:
            eta = elapsed / progress - elapsed
            eta_str = self._format_time(eta)
        else:
            eta_str = "--:--"
        
        # Format output
        percent = progress * 100
        desc = f"{self.description}: " if self.description else ""
        status_str = f" | {status}" if status else ""
        
        line = f"\r{desc}|{bar}| {percent:5.1f}% [{self.current}/{self.total}] ETA: {eta_str}{status_str}"

        # Print with encoding error handling for Windows consoles
        try:
            sys.stdout.write(line + " " * 10)  # Extra spaces to clear previous longer lines
            sys.stdout.flush()
        except UnicodeEncodeError:
            # Fallback: replace non-ASCII characters
            safe_line = line.encode('ascii', 'replace').decode('ascii')
            sys.stdout.write(safe_line + " " * 10)
            sys.stdout.flush()
        
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS"""
        if seconds < 3600:
            return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def close(self):
        """Close the progress bar"""
        if self.current < self.total:
            self.set(self.total)


# =============================================================================
# CACHING
# =============================================================================

class CacheManager:
    """Manages caching for various data types"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Sub-directories
        self.transcription_dir = self.cache_dir / "transcriptions"
        self.embedding_dir = self.cache_dir / "embeddings"
        self.scene_dir = self.cache_dir / "scenes"
        self.llm_dir = self.cache_dir / "llm_responses"
        self.index_dir = self.cache_dir / "index"
        self.audio_dir = self.cache_dir / "audio"
        self.keyframes_dir = self.cache_dir / "keyframes"
        
        for d in [self.transcription_dir, self.embedding_dir, self.scene_dir,
                  self.llm_dir, self.index_dir, self.audio_dir, self.keyframes_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Get a fast hash based on file path, size, and modification time.
        This is much faster than MD5 of entire file content for large videos.
        """
        p = Path(file_path)
        stat = p.stat()
        # Use absolute path, size, and mtime for uniqueness
        hash_input = f"{p.resolve()}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def get_file_hash_slow(self, file_path: str) -> str:
        """Get MD5 hash of file content (slow, but precise)"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_text_hash(self, text: str) -> str:
        """Get hash of text content"""
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    # Transcription cache
    def get_transcription(self, video_hash: str) -> Optional[List[SRTSegment]]:
        """Get cached transcription"""
        cache_path = self.transcription_dir / f"{video_hash}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return [SRTSegment.from_dict(s) for s in data]
        return None
    
    def save_transcription(self, video_hash: str, segments: List[SRTSegment]):
        """Save transcription to cache"""
        cache_path = self.transcription_dir / f"{video_hash}.json"
        with open(cache_path, 'w') as f:
            json.dump([s.to_dict() for s in segments], f)
    
    # Embedding cache
    def get_embeddings(self, cache_key: str) -> Optional[List[List[float]]]:
        """Get cached embeddings"""
        cache_path = self.embedding_dir / f"{cache_key}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_embeddings(self, cache_key: str, embeddings: List[List[float]]):
        """Save embeddings to cache"""
        cache_path = self.embedding_dir / f"{cache_key}.json"
        with open(cache_path, 'w') as f:
            json.dump(embeddings, f)
    
    # Scene cache
    def get_scenes(self, video_hash: str) -> Optional[List[SceneInfo]]:
        """Get cached scenes"""
        cache_path = self.scene_dir / f"{video_hash}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return [SceneInfo.from_dict(s) for s in data]
        return None
    
    def save_scenes(self, video_hash: str, scenes: List[SceneInfo]):
        """Save scenes to cache"""
        cache_path = self.scene_dir / f"{video_hash}.json"
        with open(cache_path, 'w') as f:
            json.dump([s.to_dict() for s in scenes], f)
    
    # LLM response cache
    def get_llm_response(self, prompt_hash: str) -> Optional[dict]:
        """Get cached LLM response"""
        cache_path = self.llm_dir / f"{prompt_hash}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_llm_response(self, prompt_hash: str, response: dict):
        """Save LLM response to cache"""
        cache_path = self.llm_dir / f"{prompt_hash}.json"
        with open(cache_path, 'w') as f:
            json.dump(response, f)
    
    # Video index cache
    def get_video_index(self, video_hash: str) -> Optional[VideoIndex]:
        """Get cached video index"""
        cache_path = self.index_dir / f"{video_hash}.json"
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
                return VideoIndex.from_dict(data)
        return None
    
    def save_video_index(self, video_index: VideoIndex):
        """Save video index to cache"""
        cache_path = self.index_dir / f"{video_index.video_hash}.json"
        with open(cache_path, 'w') as f:
            json.dump(video_index.to_dict(), f)
    
    def get_all_video_indices(self) -> List[VideoIndex]:
        """Get all cached video indices"""
        indices = []
        for cache_path in self.index_dir.glob("*.json"):
            with open(cache_path, 'r') as f:
                data = json.load(f)
                indices.append(VideoIndex.from_dict(data))
        return indices
    
    # Master index (tracks which videos have been processed)
    def get_master_index(self) -> Dict[str, str]:
        """Get master index mapping video paths to hashes"""
        index_path = self.cache_dir / "master_index.json"
        if index_path.exists():
            with open(index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_master_index(self, index: Dict[str, str]):
        """Save master index"""
        index_path = self.cache_dir / "master_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f)


# =============================================================================
# SMART REUSE TRACKER
# =============================================================================

class ReuseTracker:
    """
    Tracks clip usage to prevent over-reuse.

    Tracks both:
    - Clip-level reuse (same time range from same video)
    - Source-file-level reuse (any segment from same video file)
    """

    def __init__(
        self,
        max_reuse: int = 2,
        reuse_penalty: float = 0.2,
        max_source_file_reuse: int = 0,  # 0 = unlimited
        source_file_penalty: float = 0.1
    ):
        self.max_reuse = max_reuse
        self.reuse_penalty = reuse_penalty
        self.max_source_file_reuse = max_source_file_reuse
        self.source_file_penalty = source_file_penalty
        self.usage_count: Dict[str, int] = {}  # clip_id -> count
        self.source_file_count: Dict[str, int] = {}  # source_file -> count
        self.lock = threading.Lock()

    def get_clip_id(self, segment: SRTSegment) -> str:
        """Generate unique ID for a clip"""
        return f"{segment.source_file}:{segment.start_time:.2f}-{segment.end_time:.2f}"

    def get_source_file(self, segment: SRTSegment) -> str:
        """Get normalized source file path"""
        return str(segment.source_file).replace('\\', '/').lower()

    def get_usage_count(self, segment: SRTSegment) -> int:
        """Get how many times a clip has been used"""
        clip_id = self.get_clip_id(segment)
        with self.lock:
            return self.usage_count.get(clip_id, 0)

    def get_source_file_usage(self, segment: SRTSegment) -> int:
        """Get how many times any segment from this source file has been used"""
        source = self.get_source_file(segment)
        with self.lock:
            return self.source_file_count.get(source, 0)

    def record_usage(self, segment: SRTSegment):
        """Record that a clip was used"""
        clip_id = self.get_clip_id(segment)
        source = self.get_source_file(segment)
        with self.lock:
            self.usage_count[clip_id] = self.usage_count.get(clip_id, 0) + 1
            self.source_file_count[source] = self.source_file_count.get(source, 0) + 1

    def can_use(self, segment: SRTSegment) -> bool:
        """Check if clip can be used (hasn't exceeded max reuse)"""
        # Check clip-level limit
        if self.get_usage_count(segment) >= self.max_reuse:
            return False
        # Check source-file-level limit (if enabled)
        if self.max_source_file_reuse > 0:
            if self.get_source_file_usage(segment) >= self.max_source_file_reuse:
                return False
        return True

    def get_penalty(self, segment: SRTSegment) -> float:
        """Get confidence penalty based on reuse count (clip + source file)"""
        clip_count = self.get_usage_count(segment)
        clip_penalty = clip_count * self.reuse_penalty

        # Add source file penalty if over threshold
        source_penalty = 0.0
        if self.max_source_file_reuse > 0:
            source_count = self.get_source_file_usage(segment)
            # Apply escalating penalty after half the max
            threshold = self.max_source_file_reuse // 2
            if source_count > threshold:
                excess = source_count - threshold
                source_penalty = excess * self.source_file_penalty

        return clip_penalty + source_penalty

    def adjust_confidence(self, segment: SRTSegment, confidence: float) -> float:
        """Adjust confidence based on reuse"""
        penalty = self.get_penalty(segment)
        return max(0.0, confidence - penalty)

    def reset(self):
        """Reset usage tracking"""
        with self.lock:
            self.usage_count.clear()
            self.source_file_count.clear()

    def get_top_sources(self, limit: int = 10) -> List[tuple]:
        """Get the most-used source files for debugging"""
        with self.lock:
            sorted_sources = sorted(
                self.source_file_count.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_sources[:limit]


# =============================================================================
# SRT PARSING
# =============================================================================

def parse_srt_timestamp(timestamp: str) -> float:
    """Convert SRT timestamp to seconds"""
    timestamp = timestamp.strip().replace(',', '.')
    parts = timestamp.split(':')
    hours = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def format_srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')


def parse_srt_file(srt_path: str) -> List[SRTSegment]:
    """Parse an SRT file into segments"""
    segments = []

    if not Path(srt_path).exists():
        logger.error(f"SRT file not found: {srt_path}")
        return segments

    # Try multiple encodings - SRT files often have different encodings
    encodings_to_try = ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1', 'cp1252']
    content = None

    for encoding in encodings_to_try:
        try:
            with open(srt_path, 'r', encoding=encoding) as f:
                content = f.read()
            # If we got here without error, check if content looks valid
            if content and ('-->' in content or '\n' in content):
                logger.debug(f"Successfully read SRT with encoding: {encoding}")
                break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            logger.debug(f"Failed to read SRT with {encoding}: {e}")
            continue

    if content is None:
        # Last resort: read as binary and decode with errors ignored
        try:
            with open(srt_path, 'rb') as f:
                raw = f.read()
            # Remove BOM if present
            if raw.startswith(b'\xff\xfe') or raw.startswith(b'\xfe\xff'):
                content = raw.decode('utf-16', errors='ignore')
            else:
                content = raw.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Could not read SRT file: {e}")
            return segments
    
    # Check if content looks like binary/audio data
    if '\x00' in content[:1000] or 'ID3' in content[:10]:
        logger.error(f"File appears to be binary/audio, not SRT text: {srt_path}")
        return segments
    
    # Split by double newline (segment separator)
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0])
                times = lines[1].split(' --> ')
                if len(times) != 2:
                    continue
                start_time = parse_srt_timestamp(times[0])
                end_time = parse_srt_timestamp(times[1])
                text = ' '.join(lines[2:]).strip()
                
                if text:
                    segments.append(SRTSegment(
                        index=index,
                        start_time=start_time,
                        end_time=end_time,
                        text=text,
                        source_file=srt_path
                    ))
            except (ValueError, IndexError):
                continue
    
    return segments


def write_srt_file(segments: List[SRTSegment], output_path: str):
    """Write segments to an SRT file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(seg.start_time)} --> {format_srt_timestamp(seg.end_time)}\n")
            f.write(f"{seg.text}\n\n")
