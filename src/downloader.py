"""
Unified Video Downloader Module

Integrates with voiceover-matcher pipeline:
- Three duration tiers (short/medium/long)
- Parallel downloads with rate limiting
- DaVinci Resolve transcoding
- Source attribution tracking
- Checkpoint/resume support
"""

import os
import re
import json
import subprocess
import time
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class DownloadedVideo:
    """Metadata for a downloaded video"""
    file: str
    url: str
    title: str
    channel: str
    upload_date: str
    duration: float
    duration_tier: str
    keyword: str
    download_date: str
    license: str = "Unknown"
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DownloadCheckpoint:
    """Checkpoint for resuming downloads"""
    completed_keywords: List[str]
    completed_videos: List[str]
    failed_keywords: List[str]
    current_keyword: Optional[str]
    current_tier: Optional[str]
    timestamp: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "DownloadCheckpoint":
        return cls(**data)


# =============================================================================
# AUDIO-FIRST PIPELINE DATA STRUCTURES
# =============================================================================

@dataclass
class AudioDownload:
    """Audio-only download for fast transcription in audio-first pipeline.

    Downloads only the audio track (MP3) for transcription and matching,
    before downloading actual video segments.
    """
    audio_file: str       # Path to downloaded MP3
    video_id: str         # YouTube video ID
    video_url: str        # Full YouTube URL for later video download
    title: str
    channel: str
    duration: float       # Full video duration (for clamping)
    keyword: str
    duration_tier: str
    upload_date: str = ""
    license: str = "Unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchedSegment:
    """A segment from matching that needs video download.

    Created during matching phase, before video segments are downloaded.
    """
    video_id: str
    video_url: str
    start_time: float     # Original timestamp in source video
    end_time: float       # Original end timestamp
    track: str            # "V1", "V4", etc.
    voiceover_segment_idx: int
    keyword: str = ""     # Source keyword (for organizing downloads)


@dataclass
class MergedSegment:
    """Merged segments ready for download.

    Multiple close matches are merged with buffer applied to reduce
    download requests.
    """
    video_id: str
    video_url: str
    start_time: float     # After buffer applied
    end_time: float       # After buffer applied
    original_matches: List[MatchedSegment]  # Which matches this covers
    keyword: str = ""


@dataclass
class DownloadedSegment:
    """Downloaded video segment with timing info for OTIO mapping.

    The file contains a portion of the original video, from original_start
    to original_end. To find a match within this file, calculate:
        offset = match.start_time - original_start
    """
    file: str             # Path to downloaded segment file
    video_id: str
    original_start: float # Start time in source video
    original_end: float   # End time in source video
    file_duration: float  # Actual file duration
    matches: List[MatchedSegment]  # Matches contained in this segment
    keyword: str = ""

    def get_offset(self, match_time: float) -> float:
        """Get offset within this file for a match timestamp."""
        return match_time - self.original_start


# Characters that cause issues in DaVinci Resolve
# Note: Spaces are OK! Only these specific chars cause crashes.
_UNSAFE_FILENAME_CHARS = re.compile(r'[%&$#]')

def sanitize_filename_for_nle(filepath: Path) -> Path:
    """
    Sanitize filename to remove characters that cause issues in DaVinci Resolve.
    
    Based on testing: Spaces are OK, but % & $ # cause crashes.
    
    Args:
        filepath: Path to file
        
    Returns:
        New path (renamed if necessary), or original path if already safe
    """
    filename = filepath.name
    stem = filepath.stem
    suffix = filepath.suffix
    
    # Check if sanitization is needed
    if not _UNSAFE_FILENAME_CHARS.search(stem):
        return filepath
    
    # Create safe filename
    safe_stem = _UNSAFE_FILENAME_CHARS.sub('_', stem)
    safe_stem = re.sub(r'_+', '_', safe_stem)  # Remove multiple underscores
    
    new_path = filepath.parent / f"{safe_stem}{suffix}"
    
    # Handle collision
    counter = 1
    while new_path.exists() and new_path != filepath:
        new_path = filepath.parent / f"{safe_stem}_{counter}{suffix}"
        counter += 1
    
    # Rename the file
    try:
        filepath.rename(new_path)
        logger.info(f"Sanitized filename: {filename} → {new_path.name}")
        return new_path
    except Exception as e:
        logger.warning(f"Could not sanitize filename {filename}: {e}")
        return filepath


class VideoDownloader:
    """
    Advanced video downloader with:
    - Three duration tiers
    - Parallel downloads
    - DaVinci transcoding
    - Source tracking
    - Checkpoint/resume
    """
    
    # Duration tier presets
    DURATION_TIERS = {
        'short': {'min': 20, 'max': 120, 'per_keyword': 8},
        'medium': {'min': 120, 'max': 600, 'per_keyword': 8},
        'long': {'min': 600, 'max': 1500, 'per_keyword': 5},
        'longer': {'min': 1500, 'max': 3000, 'per_keyword': 5}
    }
    
    def __init__(self, config):
        """Initialize downloader with config."""
        self.config = config
        self.download_config = config.download
        
        # Override tiers from config if present
        if hasattr(self.download_config, 'tiers') and self.download_config.tiers:
            self.DURATION_TIERS = self.download_config.tiers
        
        # Source tracking
        self.sources: List[DownloadedVideo] = []
        self.sources_file = Path(config.downloaded_videos_dir) / "sources.json"
        
        # Track downloads per tier (for max_total limits)
        self.tier_download_counts: Dict[str, int] = {
            'short': 0, 'medium': 0, 'long': 0, 'longer': 0
        }
        
        # Checkpoint
        self.checkpoint_file = Path(config.cache_dir) / "download_checkpoint.json"
        self.checkpoint: Optional[DownloadCheckpoint] = None
        
        # Thread safety
        self._lock = threading.RLock()  # RLock allows reentrant locking
        
        # Load existing sources
        self._load_sources()
        
        # Cookie authentication for YouTube
        # Priority: cookies_from_browser > cookies_path > auto-detect cookies.txt
        self._cookies_from_browser = getattr(self.download_config, 'cookies_from_browser', '')
        self._cookies_path = None
        self._last_download_timed_out = False  # Track timeouts for retry logic
        
        if self._cookies_from_browser:
            logger.info(f"Using cookies from browser: {self._cookies_from_browser}")
        else:
            self._cookies_path = self._find_cookies_file()
            if self._cookies_path:
                logger.info(f"Found cookies file: {self._cookies_path}")
            else:
                logger.warning("No cookies configured - YouTube downloads may fail!")
                logger.warning("Set cookies_from_browser: firefox in config.yaml")
                logger.warning("Or export cookies from browser and save as cookies.txt")
    
    def _add_cookies_to_cmd(self, cmd: list) -> None:
        """Add cookie authentication to yt-dlp command"""
        if self._cookies_from_browser:
            cmd.extend(['--cookies-from-browser', self._cookies_from_browser])
        elif self._cookies_path:
            cmd.extend(['--cookies', str(self._cookies_path)])
    
    def _get_tier_value(self, tier: str, key: str, default: int = 0) -> int:
        """Get tier config value, handling both dict and dataclass formats"""
        tier_config = self.DURATION_TIERS.get(tier, {})
        
        if isinstance(tier_config, dict):
            return tier_config.get(key, default)
        else:
            # Dataclass format - try different attribute names
            if key == 'min':
                return getattr(tier_config, 'min_seconds', getattr(tier_config, 'min', default))
            elif key == 'max':
                return getattr(tier_config, 'max_seconds', getattr(tier_config, 'max', default))
            elif key == 'per_keyword':
                return getattr(tier_config, 'videos_per_keyword', getattr(tier_config, 'per_keyword', default))
            else:
                return getattr(tier_config, key, default)
    
    def _find_cookies_file(self) -> Optional[Path]:
        """
        Find cookies.txt file for YouTube authentication.
        Searches in order:
        1. Explicit path from config (download.cookies_path)
        2. Install directory (same as config.yaml)
        3. Project directory
        4. Current working directory
        """
        search_locations = []
        
        # 0. Check if explicit path is set in config
        if hasattr(self.download_config, 'cookies_path') and self.download_config.cookies_path:
            explicit_path = Path(self.download_config.cookies_path)
            if explicit_path.exists():
                return explicit_path
            else:
                logger.warning(f"Configured cookies_path does not exist: {explicit_path}")
        
        # 1. Install directory (where config.yaml is)
        if hasattr(self.config, '_config_path') and self.config._config_path:
            install_dir = Path(self.config._config_path).parent
            search_locations.append(install_dir / 'cookies.txt')
        
        # 2. Project directory
        if hasattr(self.config, 'project_dir') and self.config.project_dir:
            search_locations.append(Path(self.config.project_dir) / 'cookies.txt')
        
        # 3. Current working directory
        search_locations.append(Path.cwd() / 'cookies.txt')
        
        # 4. User home directory
        search_locations.append(Path.home() / 'cookies.txt')
        
        for path in search_locations:
            if path.exists():
                return path
        
        return None
    
    def _load_sources(self):
        """Load existing sources.json"""
        if self.sources_file.exists():
            try:
                with open(self.sources_file, 'r') as f:
                    data = json.load(f)
                    self.sources = [DownloadedVideo(**v) for v in data]
                logger.info(f"Loaded {len(self.sources)} existing source records")
            except Exception as e:
                logger.warning(f"Could not load sources.json: {e}")
                self.sources = []
    
    def _save_sources(self):
        """Save sources.json"""
        with self._lock:
            self.sources_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sources_file, 'w') as f:
                json.dump([s.to_dict() for s in self.sources], f, indent=2)
    
    def _load_checkpoint(self) -> Optional[DownloadCheckpoint]:
        """Load checkpoint for resume"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    return DownloadCheckpoint.from_dict(data)
            except Exception as e:
                logger.warning(f"Could not load checkpoint: {e}")
        return None
    
    def _save_checkpoint(self):
        """Save checkpoint"""
        if self.checkpoint:
            with self._lock:
                self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
                self.checkpoint.timestamp = datetime.now().isoformat()
                with open(self.checkpoint_file, 'w') as f:
                    json.dump(self.checkpoint.to_dict(), f, indent=2)
    
    def _clear_checkpoint(self):
        """Clear checkpoint after successful completion"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self.checkpoint = None
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if yt-dlp and ffmpeg are installed"""
        messages = []
        
        # Check yt-dlp
        try:
            result = subprocess.run(['yt-dlp', '--version'], capture_output=True, text=True)
            messages.append(f"✓ yt-dlp {result.stdout.strip()}")
        except FileNotFoundError:
            return False, "✗ yt-dlp not found! Install with: pip install yt-dlp"
        
        # Check ffmpeg
        if self.download_config.davinci_mode:
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                hw_accel = self._detect_hw_accel()
                hw_names = {
                    'nvidia': 'NVIDIA NVENC',
                    'amd': 'AMD AMF', 
                    'intel': 'Intel QuickSync',
                    'mac': 'Apple VideoToolbox',
                    'none': 'CPU'
                }
                messages.append(f"✓ FFmpeg ({hw_names.get(hw_accel, 'Unknown')})")
            except FileNotFoundError:
                return False, "✗ FFmpeg not found! Required for DaVinci mode."
        
        return True, "\n".join(messages)
    
    def _get_video_codec(self, video_path: str) -> tuple:
        """
        Get video codec info using ffprobe.
        Returns (codec_name, container_format) or (None, None) if failed.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            codec = result.stdout.strip().lower()
            
            # Get container format
            container = Path(video_path).suffix.lower().lstrip('.')
            
            return codec, container
        except Exception as e:
            logger.debug(f"Could not get codec info: {e}")
            return None, None
    
    def _needs_transcoding(self, video_path: str) -> tuple:
        """
        Check if video needs transcoding for DaVinci Resolve.
        
        Returns (needs_transcode: bool, reason: str)
        
        DaVinci-compatible codecs (no transcode needed):
        - H.264/AVC in MP4/MOV container
        - H.265/HEVC in MP4/MOV container  
        - ProRes in MOV container
        - DNxHD/DNxHR in MOV/MXF container
        
        Needs transcoding:
        - VP9 (WebM) - common from YouTube
        - AV1 - newer YouTube format
        - VP8 - older WebM
        """
        codec, container = self._get_video_codec(video_path)
        
        if not codec:
            return True, "Could not determine codec"
        
        # DaVinci-friendly codec+container combinations
        compatible_combinations = {
            # H.264 in standard containers
            ('h264', 'mp4'), ('h264', 'mov'), ('h264', 'mkv'),
            ('avc', 'mp4'), ('avc', 'mov'),
            # H.265/HEVC
            ('hevc', 'mp4'), ('hevc', 'mov'), ('hevc', 'mkv'),
            ('h265', 'mp4'), ('h265', 'mov'),
            # ProRes
            ('prores', 'mov'),
            # DNxHD/DNxHR
            ('dnxhd', 'mov'), ('dnxhd', 'mxf'),
            ('dnxhr', 'mov'), ('dnxhr', 'mxf'),
        }
        
        # Codecs that definitely need transcoding
        needs_transcode_codecs = {'vp9', 'vp8', 'av1', 'theora'}
        
        # Check if codec needs transcoding
        if codec in needs_transcode_codecs:
            return True, f"Codec {codec} not DaVinci-compatible"
        
        # Check if combination is compatible
        if (codec, container) in compatible_combinations:
            return False, f"Already compatible ({codec}/{container})"
        
        # For edge cases, check if it's a common compatible codec
        if codec in ('h264', 'avc', 'hevc', 'h265'):
            # H.264/H.265 in any container is usually fine
            return False, f"Compatible codec ({codec})"
        
        # Unknown - safer to transcode
        return True, f"Unknown codec combination ({codec}/{container})"
    
    def _detect_hw_accel(self) -> str:
        """Auto-detect available hardware acceleration"""
        hw_accel = self.download_config.hw_accel
        
        if hw_accel != 'auto':
            return hw_accel
        
        try:
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True)
            encoders = result.stdout + result.stderr
            
            if 'h264_nvenc' in encoders:
                return 'nvidia'
            elif 'h264_amf' in encoders:
                return 'amd'
            elif 'h264_qsv' in encoders:
                return 'intel'
            elif 'h264_videotoolbox' in encoders:
                return 'mac'
        except:
            pass
        
        return 'none'
    
    def _build_format_string(self) -> str:
        """
        Build yt-dlp format selection string.
        In DaVinci mode, prefers h264 to avoid transcoding vp9/av1.
        """
        quality = self.download_config.quality
        fmt = self.download_config.format
        davinci_mode = self.download_config.davinci_mode
        
        if quality == 'best':
            if davinci_mode:
                # Prefer h264 (avc1) over vp9/av1 to avoid transcoding
                # Simplified format with good fallbacks
                return 'bestvideo[vcodec^=avc1]+bestaudio/best[vcodec^=avc1]/bestvideo+bestaudio/best'
            return 'bestvideo+bestaudio/best'
        elif quality == 'audio':
            return 'bestaudio'
        else:
            height = quality.rstrip('p')
            if davinci_mode:
                # Prefer h264 at specified quality, with fallbacks
                return f'bestvideo[height<={height}][vcodec^=avc1]+bestaudio/bestvideo[height<={height}]+bestaudio/best[height<={height}]/best'
            return f'bestvideo[height<={height}]+bestaudio/best[height<={height}]/best'
    
    def _build_filter_string(self, tier: str) -> str:
        """Build filter string for duration and title blacklist"""
        tier_config = self.DURATION_TIERS[tier]
        
        # Handle both dict and dataclass formats
        if isinstance(tier_config, dict):
            min_dur = tier_config.get('min', 0)
            max_dur = tier_config.get('max', 120)
        else:
            # Dataclass format (DurationTierConfig)
            min_dur = getattr(tier_config, 'min_seconds', getattr(tier_config, 'min', 0))
            max_dur = getattr(tier_config, 'max_seconds', getattr(tier_config, 'max', 120))
        
        filters = [
            f"duration>{min_dur}",
            f"duration<{max_dur}",
            "!is_live"  # Skip live streams (they never end)
        ]
        
        if self.download_config.min_views > 0:
            filters.append(f"view_count>{self.download_config.min_views}")
        
        # Add title blacklist filters
        # yt-dlp syntax: title!*=term means "title does not contain term"
        title_blacklist = getattr(self.download_config, 'title_blacklist', [])
        for term in title_blacklist:
            # Escape special characters and add filter
            safe_term = term.replace("'", "\\'")
            filters.append(f"title!*='{safe_term}'")
        
        return ' & '.join(filters)
    
    def _search_video_metadata(
        self,
        keyword: str,
        tier: str,
        max_results: int = 50
    ) -> List[Dict]:
        """
        Search YouTube and get video metadata WITHOUT downloading.
        Used for LLM title filtering.
        
        Returns list of dicts with: id, title, duration, channel, url
        """
        min_dur = self._get_tier_value(tier, 'min', 0)
        max_dur = self._get_tier_value(tier, 'max', 120)
        
        cmd = [
            'yt-dlp',
            f'ytsearch{max_results}:{keyword}',
            '--dump-json',  # Get metadata only, no download
            '--flat-playlist',  # Faster - don't extract full info
            '--no-download',
            '--match-filter', f"duration>{min_dur} & duration<{max_dur} & !is_live",
        ]
        
        self._add_cookies_to_cmd(cmd)
        
        try:
            # Use config timeout or default
            search_timeout = getattr(self.download_config, 'search_timeout', 60)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=search_timeout
            )
            
            videos = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    try:
                        data = json.loads(line)
                        videos.append({
                            'id': data.get('id', ''),
                            'title': data.get('title', ''),
                            'duration': data.get('duration', 0),
                            'channel': data.get('channel', data.get('uploader', '')),
                            'url': f"https://www.youtube.com/watch?v={data.get('id', '')}"
                        })
                    except json.JSONDecodeError:
                        continue
            
            return videos
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout searching metadata for '{keyword}'")
            return []
        except Exception as e:
            logger.warning(f"Error searching metadata: {e}")
            return []
    
    def _filter_titles_with_llm(
        self,
        videos: List[Dict],
        keyword: str,
        topic: str = ""
    ) -> List[Dict]:
        """
        Filter video titles using LLM to check relevance.
        
        Args:
            videos: List of video metadata dicts
            keyword: The search keyword
            topic: Optional topic context
            
        Returns:
            List of approved videos
        """
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        if not llm_config or not getattr(llm_config, 'enabled', False):
            return videos  # Return all if LLM filter disabled
        
        if not videos:
            return []
        
        provider = getattr(llm_config, 'provider', 'gemini')
        model = getattr(llm_config, 'model', 'gemini-2.0-flash')
        min_relevance = getattr(llm_config, 'min_relevance', 0.7)
        batch_size = getattr(llm_config, 'batch_size', 20)
        
        approved = []
        
        # Process in batches
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]
            
            # Build prompt
            titles_list = "\n".join([f"{j+1}. {v['title']}" for j, v in enumerate(batch)])
            
            prompt = f"""You are filtering YouTube video titles for a video editing project.

SEARCH KEYWORD: "{keyword}"
{f'TOPIC CONTEXT: {topic}' if topic else ''}

VIDEO TITLES:
{titles_list}

For each title, determine if it would provide relevant B-roll footage for the keyword/topic.

REJECT videos that are:
- Live streams, webcams, 24/7 streams, live cams
- Sports highlights, game recaps, match footage
- Music videos, lyric videos, karaoke
- Gaming content, Let's Play, walkthroughs
- Personal vlogs unrelated to the topic
- News commentary/opinion pieces (unless specifically needed)
- Reaction videos
- Compilations of memes/fails

APPROVE videos that are:
- Documentary or educational content
- Stock footage, travel footage, city views
- Nature, landscapes, aerial shots
- Professional productions about the topic
- News reports with actual footage
- Explainer videos with relevant visuals

Respond with a JSON array of objects, one per video:
[
  {{"index": 1, "approve": true, "reason": "Documentary about topic"}},
  {{"index": 2, "approve": false, "reason": "Sports highlights"}}
]

Only output the JSON array, no other text."""

            try:
                if provider == 'gemini':
                    response = self._call_gemini(prompt, model)
                else:
                    response = self._call_anthropic(prompt, model)

                # Parse response - extract JSON array
                import re
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    json_str = json_match.group()
                    try:
                        results = json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to fix truncated JSON by closing brackets
                        json_str = json_str.rstrip()
                        if not json_str.endswith(']'):
                            # Find last complete object
                            last_brace = json_str.rfind('}')
                            if last_brace > 0:
                                json_str = json_str[:last_brace + 1] + ']'
                                try:
                                    results = json.loads(json_str)
                                    logger.debug("Fixed truncated JSON response")
                                except json.JSONDecodeError:
                                    raise
                            else:
                                raise
                        else:
                            raise

                    for result in results:
                        idx = result.get('index', 0) - 1
                        if 0 <= idx < len(batch) and result.get('approve', False):
                            video = batch[idx]
                            video['llm_reason'] = result.get('reason', 'Approved')
                            approved.append(video)
                            logger.debug(f"    ✓ Approved: {video['title'][:50]}...")
                        elif 0 <= idx < len(batch):
                            logger.debug(f"    ✗ Rejected: {batch[idx]['title'][:50]}... ({result.get('reason', 'No reason')})")
                            
            except Exception as e:
                logger.warning(f"LLM title filter error: {e}")
                # On error, approve all in batch (fail open)
                approved.extend(batch)
        
        logger.info(f"    LLM filter: {len(approved)}/{len(videos)} videos approved")
        return approved
    
    def _call_gemini(self, prompt: str, model: str) -> str:
        """Call Gemini API for title filtering."""
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY')
            if not api_key:
                logger.warning("No Gemini API key found")
                return "[]"
            
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(model)
            response = client.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.warning(f"Gemini API error: {e}")
            return "[]"
    
    def _call_anthropic(self, prompt: str, model: str) -> str:
        """Call Anthropic API for title filtering."""
        try:
            import anthropic
            
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("No Anthropic API key found")
                return "[]"
            
            # Get max_tokens from config if available
            max_tokens = 2000
            if hasattr(self.config, 'llm'):
                max_tokens = getattr(self.config.llm, 'max_tokens', 2000)
            
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
            
        except Exception as e:
            logger.warning(f"Anthropic API error: {e}")
            return "[]"
    
    def _get_ffmpeg_transcode_cmd(self, input_path: str, output_path: str) -> List[str]:
        """Build FFmpeg transcode command for DaVinci with GPU acceleration"""
        codec = self.download_config.davinci_codec
        hw_accel = self._detect_hw_accel()
        quality = self.download_config.hw_quality
        
        # Quality presets - can be overridden by transcode_crf config
        quality_map = {
            'low': {'cq': 28, 'bitrate': '8M'},
            'medium': {'cq': 23, 'bitrate': '15M'},
            'high': {'cq': 18, 'bitrate': '25M'}
        }
        q = quality_map.get(quality, quality_map['high'])
        
        # Allow direct CRF override from config
        if hasattr(self.config, 'downloading'):
            transcode_crf = getattr(self.config.downloading, 'transcode_crf', None)
            if transcode_crf is not None:
                q['cq'] = transcode_crf
        
        # -nostdin prevents FFmpeg from waiting for keyboard input
        # -hide_banner reduces log noise
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-y']
        
        # Add hardware-accelerated DECODING (input side) for NVIDIA
        # Note: For ProRes/DNxHD we don't use CUDA since they require CPU encoding
        use_gpu_decode = hw_accel == 'nvidia' and codec not in ('prores', 'dnxhd')
        
        if use_gpu_decode:
            # Use CUDA for decoding - much faster than CPU
            cmd.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
        elif hw_accel == 'intel' and codec not in ('prores', 'dnxhd'):
            cmd.extend(['-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv'])
        elif hw_accel == 'nvidia':
            # For ProRes/DNxHD: GPU decode to system memory (still faster than CPU decode)
            cmd.extend(['-hwaccel', 'cuda'])
        
        cmd.extend(['-i', input_path])
        
        # Codec-specific settings (ENCODING with GPU)
        if codec == 'prores':
            # ProRes requires CPU encoding (no GPU encoder available)
            profile_map = {'proxy': '0', 'lt': '1', 'standard': '2', 'hq': '3'}
            profile = profile_map.get(self.download_config.davinci_prores_profile, '0')
            cmd.extend(['-c:v', 'prores_ks', '-profile:v', profile, '-c:a', 'pcm_s16le'])
            output_path = output_path.rsplit('.', 1)[0] + '.mov'
        elif codec == 'dnxhd':
            cmd.extend(['-c:v', 'dnxhd', '-profile:v', 'dnxhr_sq', '-c:a', 'pcm_s16le'])
            output_path = output_path.rsplit('.', 1)[0] + '.mxf'
        elif codec == 'h265':
            if hw_accel == 'nvidia':
                # NVENC encoding - stays on GPU (decode → encode without CPU)
                cmd.extend(['-c:v', 'hevc_nvenc', '-preset', 'p4', '-cq', str(q['cq']), '-rc', 'vbr'])
            elif hw_accel == 'mac':
                cmd.extend(['-c:v', 'hevc_videotoolbox', '-q:v', '65'])
            else:
                cmd.extend(['-c:v', 'libx265', '-crf', str(q['cq'])])
            cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        else:  # h264 default
            if hw_accel == 'nvidia':
                # NVENC encoding - stays on GPU (decode → encode without CPU)
                cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', str(q['cq']), '-rc', 'vbr'])
            elif hw_accel == 'amd':
                cmd.extend(['-c:v', 'h264_amf', '-quality', 'quality'])
            elif hw_accel == 'intel':
                cmd.extend(['-c:v', 'h264_qsv', '-global_quality', str(q['cq'])])
            elif hw_accel == 'mac':
                cmd.extend(['-c:v', 'h264_videotoolbox', '-q:v', '65'])
            else:
                cmd.extend(['-c:v', 'libx264', '-crf', str(q['cq']), '-preset', 'medium'])
            cmd.extend(['-c:a', 'aac', '-b:a', '192k'])
        
        cmd.append(output_path)
        return cmd, output_path
    
    def _download_single(
        self,
        keyword: str,
        tier: str,
        output_dir: Path,
        topic: str = ""
    ) -> List[DownloadedVideo]:
        """Download videos for a single keyword and tier with optional LLM filtering"""
        max_downloads = self._get_tier_value(tier, 'per_keyword', 5)
        
        # Search a larger pool to find videos that match duration filters
        multiplier = getattr(self.download_config, 'search_pool_multiplier', 5)
        min_pool = getattr(self.download_config, 'min_search_pool', 30)
        search_pool = max(max_downloads * multiplier, min_pool)
        
        # Get path length settings from config
        max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)
        
        # Create keyword subdirectory (keep folder names short for NLE compatibility)
        safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
        safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
        tier_short = tier[0]  # s/m/l instead of short/medium/long
        keyword_dir = output_dir / f"{safe_keyword}_{tier_short}"
        keyword_dir.mkdir(parents=True, exist_ok=True)
        
        # Get existing files
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
        
        # Check if LLM filtering is enabled
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        use_llm_filter = llm_config and getattr(llm_config, 'enabled', False)
        
        if use_llm_filter:
            # NEW FLOW: Search metadata first, filter with LLM, then download specific videos
            logger.debug(f"    Searching {search_pool} videos for LLM filtering...")

            videos = self._search_video_metadata(keyword, tier, search_pool)

            if not videos:
                logger.debug(f"    No videos found for '{keyword}'")
                return []

            logger.debug(f"    Found {len(videos)} candidate videos")

            # Apply blacklist filter first (fast, no API cost)
            title_blacklist = getattr(self.download_config, 'title_blacklist', [])
            if title_blacklist:
                before_count = len(videos)
                videos = [v for v in videos if not any(
                    term.lower() in v['title'].lower() for term in title_blacklist
                )]
                if before_count > len(videos):
                    logger.debug(f"    Blacklist filter: {len(videos)}/{before_count} passed")

            # Apply LLM filter
            approved_videos = self._filter_titles_with_llm(videos[:search_pool], keyword, topic)

            if not approved_videos:
                logger.debug(f"    No videos passed LLM filter for '{keyword}'")
                return []

            # Download only approved videos (by ID)
            video_ids = [v['id'] for v in approved_videos[:max_downloads]]
            logger.debug(f"    Downloading {len(video_ids)} approved videos...")
            
            return self._download_by_ids(video_ids, keyword_dir, output_dir, keyword, tier)
        
        else:
            # ORIGINAL FLOW: Direct search and download with yt-dlp filters
            cmd = [
                'yt-dlp',
                f'ytsearch{search_pool}:{keyword}',
                '-f', self._build_format_string(),
                '--match-filter', self._build_filter_string(tier),
                '--max-downloads', str(max_downloads),
                '--merge-output-format', 'mp4',
                '--no-playlist',
                '--write-info-json',
                '--restrict-filenames',
                '--no-overwrites',
                '--no-continue',  # Don't resume partial downloads (prevents hangs after crash)
                # Truncate title for short paths (title + _ + 11 ID + .mp4)
                '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
                '--progress',
                '--newline',
                '--quiet',  # Suppress progress output
                '--no-warnings',
            ]
            
            self._add_cookies_to_cmd(cmd)

            logger.debug(f"    Search: ytsearch{search_pool}, Max: {max_downloads}")
            
            return self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_before)
    
    def _download_by_ids(
        self,
        video_ids: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str
    ) -> List[DownloadedVideo]:
        """Download specific videos by their YouTube IDs
        
        Automatically skips videos that already exist on disk (crash-resilient).
        """
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
        
        # Check which video IDs are already downloaded (file-based, not checkpoint-based)
        # This makes resumption work even without checkpoints
        already_downloaded = []
        missing_ids = []
        
        for vid_id in video_ids:
            # Check if any file contains this video ID
            found = False
            for existing_file in existing_before:
                if vid_id in existing_file and existing_file.endswith(('.mp4', '.mkv', '.webm')):
                    found = True
                    logger.debug(f"    Skipping {vid_id} - already exists: {existing_file}")
                    # Create DownloadedVideo for existing file
                    video_path = keyword_dir / existing_file
                    already_downloaded.append(DownloadedVideo(
                        file=str(video_path),
                        url=f"https://www.youtube.com/watch?v={vid_id}",
                        title=existing_file,
                        channel="",
                        upload_date="",
                        duration=0,  # Will be updated if info.json exists
                        duration_tier=tier,
                        keyword=keyword,
                        download_date="",
                        license="Unknown"
                    ))
                    break
            if not found:
                missing_ids.append(vid_id)
        
        if already_downloaded:
            logger.debug(f"    {len(already_downloaded)} already downloaded, {len(missing_ids)} to fetch")
        
        if not missing_ids:
            # All videos already exist
            return already_downloaded
        
        # Get filename length from config
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)
        
        # Build URLs from IDs
        urls = [f"https://www.youtube.com/watch?v={vid}" for vid in missing_ids]
        
        cmd = [
            'yt-dlp',
            '-f', self._build_format_string(),
            '--merge-output-format', 'mp4',
            '--no-playlist',
            '--write-info-json',
            '--restrict-filenames',
            '--no-overwrites',
            '--no-continue',  # Don't resume partial downloads (prevents hangs after crash)
            # Truncate title for short paths
            '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
            '--quiet',  # Suppress progress spam
            '--no-warnings',
            '--progress',
        ] + urls
        
        self._add_cookies_to_cmd(cmd)
        
        newly_downloaded = self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_before)
        
        # Combine already downloaded + newly downloaded
        return already_downloaded + newly_downloaded
    
    def _run_download_cmd(
        self,
        cmd: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str,
        existing_before: set,
        timeout_override: int = None
    ) -> List[DownloadedVideo]:
        """Execute download command and process results
        
        Returns:
            List of downloaded videos, or empty list on failure/timeout.
            Sets self._last_download_timed_out = True if timeout occurred.
        """
        # Clean up any partial downloads from previous crashes
        # These can cause yt-dlp to hang even with --no-continue
        if keyword_dir.exists():
            for part_file in keyword_dir.glob('*.part'):
                try:
                    part_file.unlink()
                    logger.debug(f"Cleaned up partial download: {part_file.name}")
                except Exception:
                    pass
            # Also clean up .ytdl files (download state)
            for ytdl_file in keyword_dir.glob('*.ytdl'):
                try:
                    ytdl_file.unlink()
                except Exception:
                    pass
        
        # Use override, tier-specific timeout, config default, or fallback (2 min)
        # Tier-specific timeouts: longer videos need more download time
        if timeout_override:
            download_timeout = timeout_override
        else:
            tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
            if tier and tier in tier_timeouts:
                download_timeout = tier_timeouts[tier]
            else:
                download_timeout = getattr(self.download_config, 'download_timeout', 120)
        
        # Track timeout for retry logic
        self._last_download_timed_out = False
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=download_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                logger.warning(f"Timeout downloading '{keyword}' ({tier}) after {download_timeout}s")
                self._last_download_timed_out = True
                return []
            
            # Only log actual errors
            if stderr:
                for line in stderr.strip().split('\n'):
                    if line and 'WARNING' not in line and 'ERROR' in line:
                        logger.warning(f"    yt-dlp: {line}")
            
            # Find new files
            existing_after = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
            new_files = existing_after - existing_before
            
            # Filter to video files
            video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
            new_videos = [f for f in new_files if Path(f).suffix.lower() in video_extensions]
            
            if new_videos:
                logger.debug(f"    Downloaded {len(new_videos)} video(s)")
            
            downloaded = []
            
            for idx, video_file in enumerate(new_videos, 1):
                video_path = keyword_dir / video_file
                
                # Try to get metadata from info.json
                info_file = video_path.with_suffix('.info.json')
                metadata = {}
                if info_file.exists():
                    try:
                        with open(info_file, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                # Transcode for DaVinci if enabled AND necessary
                final_path = video_path
                if self.download_config.davinci_mode:
                    needs_transcode, reason = self._needs_transcoding(str(video_path))
                    
                    if not needs_transcode:
                        logger.debug(f"    No transcode needed: {reason}")
                        final_path = video_path
                    else:
                        logger.debug(f"    Transcoding {video_file[:40]}...")
                        transcode_cmd, output_path = self._get_ffmpeg_transcode_cmd(
                            str(video_path), str(video_path)
                        )
                        temp_output = Path(output_path).with_stem(Path(output_path).stem + '_davinci')
                        transcode_cmd[-1] = str(temp_output)
                        
                        try:
                            # Use Popen to avoid hanging on large output
                            # stdin=DEVNULL prevents waiting for input
                            process = subprocess.Popen(
                                transcode_cmd,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            
                            # Wait with timeout
                            try:
                                _, stderr = process.communicate(timeout=download_timeout)
                                if process.returncode != 0:
                                    logger.warning(f"FFmpeg error: {stderr[-500:] if stderr else 'unknown'}")
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.communicate()
                                logger.warning(f"Transcode timeout for {video_file}")
                            
                            if temp_output.exists() and temp_output.stat().st_size > 0:
                                if self.download_config.delete_original:
                                    video_path.unlink()
                                final_path = temp_output.rename(temp_output.with_stem(
                                    temp_output.stem.replace('_davinci', '')
                                ))
                                logger.info(f"    ↳ ✓ Transcode complete")
                            else:
                                logger.warning(f"    ↳ Transcode produced no output, using original")
                                final_path = video_path
                        except Exception as e:
                            logger.warning(f"Transcode failed for {video_file}: {e}")
                            final_path = video_path
                
                # Sanitize filename for NLE compatibility (removes %, &, etc.)
                final_path = sanitize_filename_for_nle(final_path)
                
                # Create source record
                source = DownloadedVideo(
                    file=str(final_path.relative_to(output_dir)),
                    url=metadata.get('webpage_url', metadata.get('url', 'Unknown')),
                    title=metadata.get('title', video_file),
                    channel=metadata.get('uploader', metadata.get('channel', 'Unknown')),
                    upload_date=metadata.get('upload_date', 'Unknown'),
                    duration=metadata.get('duration', 0),
                    duration_tier=tier,
                    keyword=keyword,
                    download_date=datetime.now().strftime('%Y-%m-%d'),
                    license=metadata.get('license', 'Unknown')
                )
                
                downloaded.append(source)
                
                # Clean up info.json
                if info_file.exists():
                    info_file.unlink()
            
            return downloaded
            
        except Exception as e:
            logger.error(f"Error downloading '{keyword}' ({tier}): {e}")
            return []
    
    def download_for_keyword(
        self,
        keyword: str,
        output_dir: Path,
        tiers: List[str] = None,
        topic: str = ""
    ) -> List[DownloadedVideo]:
        """Download videos for a keyword across specified tiers
        
        If download times out, will retry with modified keyword (up to 2 retries).
        """
        if tiers is None:
            tiers = list(self.DURATION_TIERS.keys())
        
        all_downloaded = []
        
        for tier in tiers:
            # Skip tiers with per_keyword=0
            per_kw = self._get_tier_value(tier, 'per_keyword', 5)
            if per_kw <= 0:
                logger.debug(f"  [{tier}] Skipped (0/kw)")
                continue

            # Check max_total limit for this tier (e.g., only 1 LONGER video total)
            max_total = self._get_tier_value(tier, 'max_total', 0)  # 0 = no limit
            if max_total > 0 and self.tier_download_counts.get(tier, 0) >= max_total:
                logger.debug(f"  [{tier}] Skipped (max_total={max_total} reached)")
                continue
            
            # FILE-BASED SKIP: Check if videos already exist for this keyword/tier
            # This works even without checkpoints (crash-resilient)
            max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
            safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
            safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
            tier_short = tier[0]
            keyword_dir = output_dir / f"{safe_keyword}_{tier_short}"
            
            if keyword_dir.exists():
                existing_videos = [f for f in os.listdir(keyword_dir)
                                   if f.endswith(('.mp4', '.mkv', '.webm'))]
                per_kw = self._get_tier_value(tier, 'per_keyword', 5)
                if len(existing_videos) >= per_kw:
                    logger.debug(f"  [{tier}] Already have {len(existing_videos)} videos (skipping)")
                    # Count existing toward tier total
                    with self._lock:
                        self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(existing_videos)
                    continue
                elif existing_videos:
                    logger.debug(f"  [{tier}] Found {len(existing_videos)} existing, need {per_kw - len(existing_videos)} more")
            
            # Checkpoint-based skip (secondary check)
            if self.checkpoint:
                video_key = f"{keyword}|{tier}"
                if video_key in self.checkpoint.completed_videos:
                    logger.debug(f"Skipping {keyword} ({tier}) - checkpoint says completed")
                    continue
            
            logger.debug(f"  [{tier}] Downloading...")
            downloaded = self._download_single(keyword, tier, output_dir, topic)

            # Check for timeout and retry with modified keywords
            retry_attempt = 0
            while not downloaded and getattr(self, '_last_download_timed_out', False) and retry_attempt < 2:
                alt_keyword = self._get_retry_keyword(keyword, retry_attempt)
                if alt_keyword and alt_keyword != keyword:
                    logger.debug(f"  [{tier}] Timeout - retrying with: '{alt_keyword}'")
                    downloaded = self._download_single(alt_keyword, tier, output_dir, topic)
                    retry_attempt += 1
                else:
                    break

            if downloaded:
                logger.debug(f"  [{tier}] ✓ {len(downloaded)} video(s)")
                all_downloaded.extend(downloaded)

                # Update tier download count
                with self._lock:
                    self.tier_download_counts[tier] = self.tier_download_counts.get(tier, 0) + len(downloaded)

                # Update sources
                with self._lock:
                    self.sources.extend(downloaded)
                    self._save_sources()
            else:
                logger.debug(f"  [{tier}] No results")
            
            # Update checkpoint
            if self.checkpoint:
                logger.debug(f"  [{tier}] Saving checkpoint...")
                self.checkpoint.completed_videos.append(f"{keyword}|{tier}")
                self._save_checkpoint()
                logger.debug(f"  [{tier}] Checkpoint saved")
        
        return all_downloaded
    
    def _get_retry_keyword(self, keyword: str, retry_count: int) -> str:
        """Generate alternative keyword for retry after timeout
        
        Args:
            keyword: Original keyword that timed out
            retry_count: Which retry this is (0 = first retry, 1 = second retry)
            
        Returns:
            Modified keyword, or original if no modification possible
        """
        words = keyword.split()
        
        if retry_count == 0:
            # First retry: simplify by taking first 3 words + "footage"
            if len(words) > 3:
                return ' '.join(words[:3]) + " footage"
            elif "footage" not in keyword.lower():
                return keyword + " footage"
        
        elif retry_count == 1:
            # Second retry: just the core concept (first 2 words)
            if len(words) >= 2:
                return ' '.join(words[:2])
            
        return keyword
    
    def download_all(
        self,
        keywords: List[str],
        output_dir: Path,
        max_concurrent: int = 3,
        resume: bool = False,
        topic: str = ""
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """
        Download videos for all keywords.
        
        Args:
            keywords: List of search keywords
            output_dir: Output directory
            max_concurrent: Max concurrent downloads
            resume: Whether to resume from checkpoint
            topic: Topic context for LLM title filtering
        
        Returns:
            Tuple of (downloaded_videos, failed_keywords)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Scan existing downloads (file-based resume)
        existing_count = 0
        if output_dir.exists():
            for subdir in output_dir.iterdir():
                if subdir.is_dir():
                    videos = [f for f in os.listdir(subdir) if f.endswith(('.mp4', '.mkv', '.webm'))]
                    existing_count += len(videos)
        if existing_count > 0:
            logger.info(f"Found {existing_count} existing videos on disk (will skip)")
        
        # Load or create checkpoint
        if resume:
            self.checkpoint = self._load_checkpoint()
            if self.checkpoint:
                logger.info(f"Resuming from checkpoint ({len(self.checkpoint.completed_keywords)} keywords done)")
                # Filter out completed keywords
                keywords = [k for k in keywords if k not in self.checkpoint.completed_keywords]
        
        if not self.checkpoint:
            self.checkpoint = DownloadCheckpoint(
                completed_keywords=[],
                completed_videos=[],
                failed_keywords=[],
                current_keyword=None,
                current_tier=None,
                timestamp=datetime.now().isoformat()
            )
        
        all_downloaded = []
        failed_keywords = list(self.checkpoint.failed_keywords)
        
        # Log title blacklist if enabled
        title_blacklist = getattr(self.download_config, 'title_blacklist', [])
        if title_blacklist:
            logger.info(f"  Title blacklist: {len(title_blacklist)} terms (e.g., {', '.join(title_blacklist[:5])}...)")
        
        # Log LLM filter status
        llm_config = getattr(self.download_config, 'llm_title_filter', None)
        if llm_config and getattr(llm_config, 'enabled', False):
            provider = getattr(llm_config, 'provider', 'gemini')
            logger.info(f"  LLM title filter: enabled ({provider})")
        
        # Process keywords with thread pool
        # Note: We process keywords sequentially but tiers can overlap slightly
        # to avoid overwhelming the API

        total_videos_downloaded = 0
        print(f"\n  Downloading videos for {len(keywords)} keywords...")

        for i, keyword in enumerate(keywords, 1):
            # Compact progress line (updates in place)
            progress_pct = (i - 1) / len(keywords) * 100
            print(f"\r  [{i}/{len(keywords)}] {progress_pct:5.1f}% | {keyword[:40]:<40} | Videos: {total_videos_downloaded}", end='', flush=True)

            logger.info(f"[{i}/{len(keywords)}] Processing: {keyword}")

            self.checkpoint.current_keyword = keyword
            self._save_checkpoint()

            downloaded = self.download_for_keyword(keyword, output_dir, topic=topic)

            if downloaded:
                all_downloaded.extend(downloaded)
                total_videos_downloaded += len(downloaded)
                self.checkpoint.completed_keywords.append(keyword)
            else:
                failed_keywords.append(keyword)
                self.checkpoint.failed_keywords.append(keyword)

            self._save_checkpoint()

            # Delay between keywords to avoid rate limiting
            if i < len(keywords):
                time.sleep(self.download_config.delay_between_keywords)

        # Final progress line
        print(f"\r  [{len(keywords)}/{len(keywords)}] 100.0% | Done{' ' * 50}")
        print(f"  ✓ Downloaded {total_videos_downloaded} videos from {len(keywords)} keywords")

        # Clear checkpoint on success
        self._clear_checkpoint()
        
        return all_downloaded, failed_keywords
    
    def get_download_estimate(self, num_keywords: int) -> Dict:
        """Estimate download stats"""
        videos_per_keyword = sum(t['per_keyword'] for t in self.DURATION_TIERS.values())
        total_videos = num_keywords * videos_per_keyword
        
        # Rough estimates
        avg_size_mb = 150  # Average video size
        avg_download_time = 30  # Seconds per video
        
        return {
            'keywords': num_keywords,
            'videos_per_keyword': videos_per_keyword,
            'total_videos': total_videos,
            'est_storage_gb': round(total_videos * avg_size_mb / 1024, 1),
            'est_time_minutes': round(total_videos * avg_download_time / 60, 0),
            'tiers': {
                name: f"{self._get_tier_value(name, 'min', 0)}s-{self._get_tier_value(name, 'max', 120)}s ({self._get_tier_value(name, 'per_keyword', 5)}/kw)"
                for name in self.DURATION_TIERS.keys()
            }
        }
    
    def get_inventory_report(self, output_dir: Path) -> Dict:
        """Report on existing footage"""
        output_dir = Path(output_dir)
        
        if not output_dir.exists():
            return {'total_videos': 0, 'total_duration': 0, 'keywords_covered': []}
        
        video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov', '.mxf'}
        videos = []
        
        for ext in video_extensions:
            videos.extend(output_dir.rglob(f'*{ext}'))
        
        # Get unique keywords from directory names
        keywords_covered = set()
        for video in videos:
            parent = video.parent.name
            # Extract keyword from dir name (remove tier suffix)
            for tier in self.DURATION_TIERS.keys():
                if parent.endswith(f'_{tier}'):
                    keyword = parent[:-len(f'_{tier}')]
                    keywords_covered.add(keyword)
                    break
        
        # Calculate total duration from sources.json
        total_duration = sum(s.duration for s in self.sources) if self.sources else 0
        
        return {
            'total_videos': len(videos),
            'total_duration_hours': round(total_duration / 3600, 2),
            'keywords_covered': list(keywords_covered),
            'num_keywords_covered': len(keywords_covered)
        }

    # =========================================================================
    # AUDIO-FIRST PIPELINE METHODS
    # =========================================================================

    def download_audio_for_keyword(
        self,
        keyword: str,
        output_dir: Path,
        tier: str,
        topic: str = ""
    ) -> List[AudioDownload]:
        """Download audio only (MP3) for videos matching keyword.

        Phase 1 of audio-first pipeline. Downloads lightweight MP3 files
        for transcription and matching, before video segments.

        Args:
            keyword: Search keyword
            output_dir: Base output directory
            tier: Duration tier ('short', 'medium', 'long', 'longer')
            topic: Optional topic for LLM filter context

        Returns:
            List of AudioDownload records
        """
        audio_config = getattr(self.download_config, 'audio_first', None)
        if not audio_config:
            logger.error("Audio-first config not found")
            return []

        # Get tier settings
        tier_min = self._get_tier_value(tier, 'min', 20)
        tier_max = self._get_tier_value(tier, 'max', 120)
        per_keyword = self._get_tier_value(tier, 'per_keyword', 5)

        # Create audio output directory
        max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
        safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
        safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
        tier_short = tier[0]
        audio_dir = output_dir / f"{safe_keyword}_{tier_short}_audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        # Search for videos
        search_count = max(per_keyword * 5, 40)
        try:
            search_results = self._search_video_metadata(keyword, tier, max_results=search_count)
        except Exception as e:
            logger.error(f"Search failed for '{keyword}': {e}")
            return []

        if not search_results:
            logger.warning(f"No search results for '{keyword}'")
            return []

        # Filter by duration
        filtered = [
            v for v in search_results
            if tier_min <= v.get('duration', 0) <= tier_max
            and not v.get('is_live', False)  # Skip live videos
        ]

        if not filtered:
            logger.warning(f"No videos in duration range for '{keyword}'")
            return []

        # LLM title filter if enabled
        if getattr(self.download_config, 'llm_title_filter', None):
            filter_config = self.download_config.llm_title_filter
            if getattr(filter_config, 'enabled', False):
                filtered = self._filter_titles_with_llm(filtered, keyword, topic)

        # Take top N
        to_download = filtered[:per_keyword]

        # Download audio for each
        audio_downloads = []
        audio_quality = getattr(audio_config, 'audio_quality', 5)

        for video_info in to_download:
            video_id = video_info.get('id', '')
            video_url = video_info.get('webpage_url', f"https://www.youtube.com/watch?v={video_id}")

            # Skip if already downloaded (check multiple audio formats)
            existing_file = None
            for ext in ['.mp3', '.m4a', '.mp4', '.opus', '.webm', '.ogg', '.wav']:
                candidate = audio_dir / f"{video_id}{ext}"
                if candidate.exists():
                    existing_file = candidate
                    break

            if existing_file:
                logger.debug(f"Audio already exists: {existing_file.name}")
                audio_downloads.append(AudioDownload(
                    audio_file=str(existing_file),
                    video_id=video_id,
                    video_url=video_url,
                    title=video_info.get('title', ''),
                    channel=video_info.get('channel', video_info.get('uploader', '')),
                    duration=video_info.get('duration', 0),
                    keyword=keyword,
                    duration_tier=tier,
                    upload_date=video_info.get('upload_date', ''),
                    license=video_info.get('license', 'Unknown')
                ))
                continue

            # Build yt-dlp command for audio only
            # Use %(ext)s and let yt-dlp determine the final extension
            cmd = [
                'yt-dlp',
                video_url,
                '-x',  # Extract audio
                '--audio-format', 'mp3',
                '--audio-quality', str(audio_quality),
                '-o', str(audio_dir / '%(id)s.%(ext)s'),
                '--no-playlist',
                '--no-warnings',
            ]

            # Add cookies (browser or file)
            cmd.extend(self._get_cookies_args())

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120  # Audio should be fast
                )

                # Find the actual downloaded file (could be .mp3, .m4a, .opus, etc.)
                actual_file = None
                if result.returncode == 0:
                    # Look for any audio file matching this video_id
                    for ext in ['.mp3', '.m4a', '.mp4', '.opus', '.webm', '.ogg', '.wav']:
                        candidate = audio_dir / f"{video_id}{ext}"
                        if candidate.exists():
                            actual_file = candidate
                            break

                if actual_file:
                    audio_downloads.append(AudioDownload(
                        audio_file=str(actual_file),
                        video_id=video_id,
                        video_url=video_url,
                        title=video_info.get('title', ''),
                        channel=video_info.get('channel', video_info.get('uploader', '')),
                        duration=video_info.get('duration', 0),
                        keyword=keyword,
                        duration_tier=tier,
                        upload_date=video_info.get('upload_date', ''),
                        license=video_info.get('license', 'Unknown')
                    ))
                    logger.debug(f"Downloaded audio: {actual_file.name}")
                else:
                    # Show last 500 chars of stderr (actual error, not ffmpeg header)
                    err_msg = result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                    logger.warning(f"Audio download failed for {video_id} (rc={result.returncode}): {err_msg}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Audio download timeout for {video_id}")
            except Exception as e:
                logger.warning(f"Audio download error for {video_id}: {e}")

        logger.info(f"  Downloaded {len(audio_downloads)} audio files for '{keyword}' ({tier})")
        return audio_downloads

    def download_video_segments(
        self,
        merged_segments: List[MergedSegment],
        output_dir: Path
    ) -> List[DownloadedSegment]:
        """Download video segments using --download-sections.

        Phase 3 of audio-first pipeline. Downloads only the matched portions
        of videos, not the full files.

        Args:
            merged_segments: List of merged segments with buffer applied
            output_dir: Base output directory

        Returns:
            List of DownloadedSegment records with timing info
        """
        audio_config = getattr(self.download_config, 'audio_first', None)
        fallback_full = getattr(audio_config, 'fallback_full_video', True) if audio_config else True

        downloaded_segments = []

        # Group by video_id for efficient downloading
        by_video: Dict[str, List[MergedSegment]] = {}
        for seg in merged_segments:
            if seg.video_id not in by_video:
                by_video[seg.video_id] = []
            by_video[seg.video_id].append(seg)

        for video_id, segments in by_video.items():
            if not segments:
                continue

            # Use first segment for URL and keyword
            first_seg = segments[0]
            video_url = first_seg.video_url
            keyword = first_seg.keyword

            # Create output directory
            max_kw_len = getattr(self.download_config, 'max_keyword_len', 8)
            safe_keyword = "".join(c if c.isalnum() or c in '-_' else '_' for c in keyword)
            safe_keyword = safe_keyword.replace(' ', '_')[:max_kw_len].rstrip('_')
            video_dir = output_dir / f"{safe_keyword}_segments"
            video_dir.mkdir(parents=True, exist_ok=True)

            # Build --download-sections arguments
            section_args = []
            for seg in segments:
                # Format: *START-END (HH:MM:SS or seconds)
                start_str = self._format_time(seg.start_time)
                end_str = self._format_time(seg.end_time)
                section_args.extend(['--download-sections', f'*{start_str}-{end_str}'])

            # Build yt-dlp command
            cmd = [
                'yt-dlp',
                video_url,
                *section_args,
                '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                '--merge-output-format', 'mp4',
                '-o', str(video_dir / f'{video_id}_%(autonumber)s.%(ext)s'),
                '--no-playlist',
                '--no-warnings',
            ]

            # Add cookies (browser or file)
            cmd.extend(self._get_cookies_args())

            # Get tier-specific timeout
            tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
            timeout = tier_timeouts.get('long', 600)  # Use long tier timeout for segments

            segment_success = False
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode != 0:
                    logger.warning(f"Segment download failed for {video_id}: {result.stderr[:200]}")
                else:
                    segment_success = True
                    # Rename files from autonumber to timestamp-based names
                    downloaded = rename_segments_with_timing(video_dir, video_id, segments)

                    for seg, file_path in zip(segments, downloaded):
                        if file_path and Path(file_path).exists():
                            # Get actual file duration
                            file_duration = seg.end_time - seg.start_time  # Approximate

                            downloaded_segments.append(DownloadedSegment(
                                file=str(file_path),
                                video_id=video_id,
                                original_start=seg.start_time,
                                original_end=seg.end_time,
                                file_duration=file_duration,
                                matches=seg.original_matches,
                                keyword=keyword
                            ))

            except subprocess.TimeoutExpired:
                logger.warning(f"Segment download timeout for {video_id}")
            except Exception as e:
                logger.error(f"Segment download error for {video_id}: {e}")

            # Fallback to full video if segment download failed
            if not segment_success and fallback_full:
                fallback_segments = self._download_full_video_fallback(
                    video_id=video_id,
                    video_url=video_url,
                    video_dir=video_dir,
                    segments=segments,
                    keyword=keyword,
                    timeout=timeout
                )
                downloaded_segments.extend(fallback_segments)

        logger.info(f"Downloaded {len(downloaded_segments)} video segments")
        return downloaded_segments

    def _format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS for yt-dlp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_cookies_args(self) -> List[str]:
        """Get yt-dlp cookie arguments (browser or file).

        Prefers cookies_from_browser, falls back to cookies_path.
        Returns list of arguments to extend yt-dlp command.
        """
        # Prefer browser cookies
        cookies_browser = getattr(self.download_config, 'cookies_from_browser', '')
        if cookies_browser:
            logger.debug(f"Using cookies from browser: {cookies_browser}")
            return ['--cookies-from-browser', cookies_browser]

        # Fall back to cookies file
        cookies_path = getattr(self.download_config, 'cookies_path', '')
        if cookies_path and Path(cookies_path).exists():
            logger.debug(f"Using cookies file: {cookies_path}")
            return ['--cookies', cookies_path]

        return []

    def _download_full_video_fallback(
        self,
        video_id: str,
        video_url: str,
        video_dir: Path,
        segments: List[MergedSegment],
        keyword: str,
        timeout: int = 600
    ) -> List[DownloadedSegment]:
        """Download full video as fallback when segment download fails.

        Args:
            video_id: YouTube video ID
            video_url: Full YouTube URL
            video_dir: Output directory
            segments: Original segments (for match info)
            keyword: Source keyword
            timeout: Download timeout in seconds

        Returns:
            List with single DownloadedSegment covering full video
        """
        logger.info(f"  Downloading full video fallback: {video_id}")

        output_file = video_dir / f"{video_id}_0000.mp4"

        cmd = [
            'yt-dlp',
            video_url,
            '-f', 'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
            '--merge-output-format', 'mp4',
            '-o', str(output_file),
            '--no-playlist',
            '--no-warnings',
        ]
        cmd.extend(self._get_cookies_args())

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode == 0 and output_file.exists():
                # Get video duration using ffprobe if available
                video_duration = self._get_video_duration(output_file)
                if video_duration is None:
                    # Estimate from segments
                    video_duration = max(seg.end_time for seg in segments) + 60

                # Collect all original matches from all segments
                all_matches = []
                for seg in segments:
                    all_matches.extend(seg.original_matches)

                logger.info(f"  ✓ Full video fallback success: {video_id}")
                return [DownloadedSegment(
                    file=str(output_file),
                    video_id=video_id,
                    original_start=0,
                    original_end=video_duration,
                    file_duration=video_duration,
                    matches=all_matches,
                    keyword=keyword
                )]
            else:
                logger.error(f"Full video fallback failed for {video_id}: {result.stderr[:200]}")
                return []

        except subprocess.TimeoutExpired:
            logger.error(f"Full video fallback timeout for {video_id}")
            return []
        except Exception as e:
            logger.error(f"Full video fallback error for {video_id}: {e}")
            return []

    def _get_video_duration(self, video_path: Path) -> Optional[float]:
        """Get video duration using ffprobe."""
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return None


# =============================================================================
# AUDIO-FIRST HELPER FUNCTIONS (Standalone)
# =============================================================================

def get_segment_filename(video_id: str, start_seconds: float) -> str:
    """Generate filename with start time encoded.

    Format: {video_id}_{start_seconds:04d}.mp4
    Example: abc123_0330.mp4 = video abc123, starts at 330 seconds (5:30)

    This naming allows:
    - Easy sorting by time
    - Extracting start time from filename
    - Direct mapping to original timestamps
    """
    start_int = int(start_seconds)
    return f"{video_id}_{start_int:04d}.mp4"


def rename_segments_with_timing(
    download_dir: Path,
    video_id: str,
    merged_segments: List[MergedSegment]
) -> List[Optional[str]]:
    """Rename autonumber files to timestamp-based names.

    yt-dlp with --download-sections creates files like:
        abc123_1.mp4, abc123_2.mp4, ...

    This renames them to:
        abc123_0000.mp4 (starts at 0s)
        abc123_0330.mp4 (starts at 330s)

    Args:
        download_dir: Directory containing downloaded files
        video_id: YouTube video ID
        merged_segments: Segments in order they were downloaded

    Returns:
        List of renamed file paths (None if file not found)
    """
    renamed_files = []

    for idx, segment in enumerate(merged_segments, start=1):
        old_name = download_dir / f"{video_id}_{idx}.mp4"
        new_name = download_dir / get_segment_filename(video_id, segment.start_time)

        try:
            if old_name.exists():
                old_name.rename(new_name)
                renamed_files.append(str(new_name))
                logger.debug(f"Renamed {old_name.name} -> {new_name.name}")
            else:
                # Try with different extensions
                for ext in ['.mkv', '.webm']:
                    alt_old = old_name.with_suffix(ext)
                    if alt_old.exists():
                        alt_new = new_name.with_suffix(ext)
                        alt_old.rename(alt_new)
                        renamed_files.append(str(alt_new))
                        break
                else:
                    logger.warning(f"Expected file not found: {old_name}")
                    renamed_files.append(None)
        except OSError as e:
            logger.error(f"Failed to rename {old_name}: {e}")
            renamed_files.append(None)

    return renamed_files


def merge_segments_with_buffer(
    segments: List[Tuple[float, float]],
    buffer_seconds: float = 30.0,
    merge_gap_seconds: float = 15.0,
    video_duration: float = None
) -> List[Tuple[float, float]]:
    """Merge overlapping segments after adding buffer.

    Args:
        segments: List of (start, end) tuples
        buffer_seconds: Add this before/after each segment
        merge_gap_seconds: Merge if gap is less than this
        video_duration: Clamp end to video duration if provided

    Returns:
        List of merged (start, end) tuples
    """
    if not segments:
        return []

    # Step 0: Validate and filter segments
    validated = []
    for start, end in segments:
        # Type check
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            logger.warning(f"Invalid segment timestamps ({start}, {end}), skipping")
            continue

        # Fix negative start times
        if start < 0:
            logger.debug(f"Negative start time {start}, clamping to 0")
            start = 0

        # Skip invalid segments where end <= start
        if end <= start:
            logger.warning(f"Invalid segment [{start}, {end}] (end <= start), skipping")
            continue

        validated.append((start, end))

    if not validated:
        logger.warning("No valid segments after validation")
        return []

    # Step 1: Add buffer and clamp to valid range
    buffered = []
    for start, end in validated:
        new_start = max(0, start - buffer_seconds)
        new_end = end + buffer_seconds
        if video_duration:
            new_end = min(new_end, video_duration)
        buffered.append((new_start, new_end))

    # Step 2: Sort by start time
    buffered.sort(key=lambda x: x[0])

    # Step 3: Merge overlapping or close segments
    merged = [buffered[0]]
    for start, end in buffered[1:]:
        last_start, last_end = merged[-1]

        # Merge if overlapping OR gap is small
        if start <= last_end + merge_gap_seconds:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def collect_matched_segments(
    match_results: List,  # List[MatchResult]
    audio_downloads: Dict[str, AudioDownload]
) -> Dict[str, List[MatchedSegment]]:
    """Collect all matched segments from matching results.

    Groups matches by video_id for efficient downloading.

    Args:
        match_results: Results from matching phase
        audio_downloads: Map of video_id -> AudioDownload

    Returns:
        Dict of video_id -> List[MatchedSegment]
    """
    segments_by_video: Dict[str, List[MatchedSegment]] = {}

    for idx, result in enumerate(match_results):
        # Process primary match (V1)
        if result.primary_match:
            seg = result.primary_match.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.video_url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track="V1",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

        # Process alternatives (V2-V3)
        for alt_idx, alt in enumerate(result.alternatives or [], start=2):
            seg = alt.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.video_url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track=f"V{alt_idx}",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

        # Process secondary matches (V4-V6)
        for sec_idx, sec in enumerate(result.secondary_matches or [], start=4):
            seg = sec.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.video_url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track=f"V{sec_idx}",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

        # Process strategy matches (V7)
        for strat in result.strategy_matches or []:
            seg = strat.video_segment
            video_id = _extract_video_id(seg.source_file)

            if video_id and video_id in audio_downloads:
                audio = audio_downloads[video_id]
                matched = MatchedSegment(
                    video_id=video_id,
                    video_url=audio.video_url,
                    start_time=seg.start_time,
                    end_time=seg.end_time,
                    track="V7",
                    voiceover_segment_idx=idx,
                    keyword=audio.keyword
                )
                if video_id not in segments_by_video:
                    segments_by_video[video_id] = []
                segments_by_video[video_id].append(matched)

    return segments_by_video


def _extract_video_id(file_path: str) -> Optional[str]:
    """Extract YouTube video ID from file path.

    Assumes filename format: {video_id}.mp3 or {video_id}_0000.mp4
    """
    if not file_path:
        return None

    filename = Path(file_path).stem

    # Handle segment format: abc123_0000
    if '_' in filename and filename.split('_')[-1].isdigit():
        return filename.rsplit('_', 1)[0]

    # Simple format: abc123
    return filename


def prepare_merged_segments(
    segments_by_video: Dict[str, List[MatchedSegment]],
    audio_downloads: Dict[str, AudioDownload],
    buffer_seconds: float = 30.0,
    merge_gap_seconds: float = 15.0
) -> List[MergedSegment]:
    """Prepare merged segments for download.

    Applies buffer, merges overlapping/close segments, and creates
    MergedSegment records ready for download.

    Args:
        segments_by_video: Dict of video_id -> List[MatchedSegment]
        audio_downloads: Map of video_id -> AudioDownload
        buffer_seconds: Padding around each match
        merge_gap_seconds: Merge if gap is smaller

    Returns:
        List of MergedSegment ready for download
    """
    all_merged = []

    for video_id, matches in segments_by_video.items():
        if not matches:
            continue

        # Get video duration for clamping
        audio = audio_downloads.get(video_id)
        video_duration = audio.duration if audio else None

        # Extract time ranges
        time_ranges = [(m.start_time, m.end_time) for m in matches]

        # Merge with buffer
        merged_ranges = merge_segments_with_buffer(
            time_ranges,
            buffer_seconds=buffer_seconds,
            merge_gap_seconds=merge_gap_seconds,
            video_duration=video_duration
        )

        # Create MergedSegment for each merged range
        for start, end in merged_ranges:
            # Find which original matches fall within this range
            contained_matches = [
                m for m in matches
                if start <= m.start_time and m.end_time <= end
            ]

            all_merged.append(MergedSegment(
                video_id=video_id,
                video_url=matches[0].video_url,
                start_time=start,
                end_time=end,
                original_matches=contained_matches,
                keyword=matches[0].keyword
            ))

    return all_merged