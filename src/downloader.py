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
        
        # Checkpoint
        self.checkpoint_file = Path(config.cache_dir) / "download_checkpoint.json"
        self.checkpoint: Optional[DownloadCheckpoint] = None
        
        # Thread safety
        self._lock = threading.RLock()  # RLock allows reentrant locking
        
        # Load existing sources
        self._load_sources()
        
        # Find cookies file for YouTube authentication
        self._cookies_path = self._find_cookies_file()
        if self._cookies_path:
            logger.info(f"Found cookies file: {self._cookies_path}")
        else:
            logger.warning("No cookies.txt found - YouTube downloads may fail!")
            logger.warning("Export cookies from browser and save as cookies.txt in install directory")
    
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
            f"duration<{max_dur}"
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
            '--match-filter', f"duration>{min_dur} & duration<{max_dur}",
        ]
        
        if self._cookies_path:
            cmd.extend(['--cookies', str(self._cookies_path)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
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
                
                # Parse response
                import re
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    results = json.loads(json_match.group())
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
            
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=2000,
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
        
        # Quality presets
        quality_map = {
            'low': {'cq': 28, 'bitrate': '8M'},
            'medium': {'cq': 23, 'bitrate': '15M'},
            'high': {'cq': 18, 'bitrate': '25M'}
        }
        q = quality_map.get(quality, quality_map['high'])
        
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
            logger.info(f"    Searching {search_pool} videos for LLM filtering...")
            
            videos = self._search_video_metadata(keyword, tier, search_pool)
            
            if not videos:
                logger.warning(f"    No videos found for '{keyword}'")
                return []
            
            logger.info(f"    Found {len(videos)} candidate videos")
            
            # Apply blacklist filter first (fast, no API cost)
            title_blacklist = getattr(self.download_config, 'title_blacklist', [])
            if title_blacklist:
                before_count = len(videos)
                videos = [v for v in videos if not any(
                    term.lower() in v['title'].lower() for term in title_blacklist
                )]
                if before_count > len(videos):
                    logger.info(f"    Blacklist filter: {len(videos)}/{before_count} passed")
            
            # Apply LLM filter
            approved_videos = self._filter_titles_with_llm(videos[:search_pool], keyword, topic)
            
            if not approved_videos:
                logger.warning(f"    No videos passed LLM filter for '{keyword}'")
                return []
            
            # Download only approved videos (by ID)
            video_ids = [v['id'] for v in approved_videos[:max_downloads]]
            logger.info(f"    Downloading {len(video_ids)} approved videos...")
            
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
                # Truncate title for short paths (title + _ + 11 ID + .mp4)
                '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
                '--progress',
                '--newline',
                '--quiet',  # Suppress progress output
                '--no-warnings',
            ]
            
            if self._cookies_path:
                cmd.extend(['--cookies', str(self._cookies_path)])
            
            logger.info(f"    Search: ytsearch{search_pool}, Max: {max_downloads}")
            
            return self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_before)
    
    def _download_by_ids(
        self,
        video_ids: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str
    ) -> List[DownloadedVideo]:
        """Download specific videos by their YouTube IDs"""
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
        
        # Get filename length from config
        max_fn_len = getattr(self.download_config, 'max_filename_len', 10)
        
        # Build URLs from IDs
        urls = [f"https://www.youtube.com/watch?v={vid}" for vid in video_ids]
        
        cmd = [
            'yt-dlp',
            '-f', self._build_format_string(),
            '--merge-output-format', 'mp4',
            '--no-playlist',
            '--write-info-json',
            '--restrict-filenames',
            '--no-overwrites',
            # Truncate title for short paths
            '-o', str(keyword_dir / f'%(title).{max_fn_len}s_%(id)s.%(ext)s'),
            '--quiet',  # Suppress progress spam
            '--no-warnings',
            '--progress',
        ] + urls
        
        if self._cookies_path:
            cmd.extend(['--cookies', str(self._cookies_path)])
        
        return self._run_download_cmd(cmd, keyword_dir, output_dir, keyword, tier, existing_before)
    
    def _run_download_cmd(
        self,
        cmd: List[str],
        keyword_dir: Path,
        output_dir: Path,
        keyword: str,
        tier: str,
        existing_before: set
    ) -> List[DownloadedVideo]:
        """Execute download command and process results"""
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=600)  # 10 min timeout
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                logger.warning(f"Timeout downloading '{keyword}' ({tier})")
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
                logger.info(f"    ✓ Downloaded {len(new_videos)} video(s)")
            
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
                        logger.debug(f"    ↳ No transcode needed: {reason}")
                        final_path = video_path
                    else:
                        logger.info(f"    ↳ Transcoding {video_file[:40]}...")
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
                                _, stderr = process.communicate(timeout=600)
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
        """Download videos for a keyword across specified tiers"""
        if tiers is None:
            tiers = list(self.DURATION_TIERS.keys())
        
        all_downloaded = []
        
        for tier in tiers:
            # Skip tiers with per_keyword=0
            per_kw = self._get_tier_value(tier, 'per_keyword', 5)
            if per_kw <= 0:
                logger.info(f"  [{tier}] Skipped (0/kw)")
                continue
            
            # Check checkpoint - skip if already done
            if self.checkpoint:
                video_key = f"{keyword}|{tier}"
                if video_key in self.checkpoint.completed_videos:
                    logger.debug(f"Skipping {keyword} ({tier}) - already completed")
                    continue
            
            logger.info(f"  [{tier}] Downloading...")
            downloaded = self._download_single(keyword, tier, output_dir, topic)
            
            if downloaded:
                logger.info(f"  [{tier}] ✓ {len(downloaded)} video(s)")
                all_downloaded.extend(downloaded)
                
                # Update sources
                logger.debug(f"  [{tier}] Saving sources...")
                with self._lock:
                    self.sources.extend(downloaded)
                    self._save_sources()
                logger.debug(f"  [{tier}] Sources saved")
            else:
                logger.info(f"  [{tier}] ⚠ No results")
            
            # Update checkpoint
            if self.checkpoint:
                logger.debug(f"  [{tier}] Saving checkpoint...")
                self.checkpoint.completed_videos.append(f"{keyword}|{tier}")
                self._save_checkpoint()
                logger.debug(f"  [{tier}] Checkpoint saved")
        
        return all_downloaded
    
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
        
        for i, keyword in enumerate(keywords, 1):
            logger.info(f"\n[{i}/{len(keywords)}] Processing: {keyword}")
            
            self.checkpoint.current_keyword = keyword
            self._save_checkpoint()
            
            downloaded = self.download_for_keyword(keyword, output_dir, topic=topic)
            
            if downloaded:
                all_downloaded.extend(downloaded)
                self.checkpoint.completed_keywords.append(keyword)
            else:
                failed_keywords.append(keyword)
                self.checkpoint.failed_keywords.append(keyword)
            
            self._save_checkpoint()
            
            # Delay between keywords to avoid rate limiting
            if i < len(keywords):
                time.sleep(self.download_config.delay_between_keywords)
        
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