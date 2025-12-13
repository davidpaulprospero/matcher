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
        'short': {'min': 20, 'max': 120, 'per_keyword': 3},
        'medium': {'min': 120, 'max': 600, 'per_keyword': 3},
        'long': {'min': 600, 'max': 1500, 'per_keyword': 1}
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
        """Build yt-dlp format selection string"""
        quality = self.download_config.quality
        fmt = self.download_config.format
        
        if quality == 'best':
            return 'best'
        elif quality == 'audio':
            return 'bestaudio'
        else:
            height = quality.rstrip('p')
            return f'best[height<={height}][ext={fmt}]/best[height<={height}]/best'
    
    def _build_filter_string(self, tier: str) -> str:
        """Build filter string for duration"""
        tier_config = self.DURATION_TIERS[tier]
        filters = [
            f"duration>{tier_config['min']}",
            f"duration<{tier_config['max']}"
        ]
        
        if self.download_config.min_views > 0:
            filters.append(f"view_count>{self.download_config.min_views}")
        
        return ' & '.join(filters)
    
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
        output_dir: Path
    ) -> List[DownloadedVideo]:
        """Download videos for a single keyword and tier"""
        tier_config = self.DURATION_TIERS[tier]
        max_downloads = tier_config['per_keyword']
        
        # Create keyword subdirectory
        safe_keyword = "".join(c if c.isalnum() or c in ' -_' else '_' for c in keyword)[:50]
        keyword_dir = output_dir / f"{safe_keyword}_{tier}"
        keyword_dir.mkdir(parents=True, exist_ok=True)
        
        # Get existing files
        existing_before = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
        
        # Build yt-dlp command
        cmd = [
            'yt-dlp',
            f'ytsearch{max_downloads}:{keyword}',
            '-f', self._build_format_string(),
            '--match-filter', self._build_filter_string(tier),
            '--no-playlist',
            '--write-info-json',  # Need this for metadata
            '--restrict-filenames',
            '--no-overwrites',
            '-o', str(keyword_dir / '%(title)s-%(id)s.%(ext)s'),
            '--quiet',
            '--no-warnings',
            '--progress'
        ]
        
        logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            # Use Popen with stdin=DEVNULL to prevent hanging
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=300)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                logger.warning(f"Timeout downloading '{keyword}' ({tier})")
                return []
            
            # Find new files
            existing_after = set(os.listdir(keyword_dir)) if keyword_dir.exists() else set()
            new_files = existing_after - existing_before
            
            # Filter to video files
            video_extensions = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
            new_videos = [f for f in new_files if Path(f).suffix.lower() in video_extensions]
            
            logger.debug(f"  Found {len(new_videos)} new video(s) in {keyword_dir}")
            
            downloaded = []
            
            for idx, video_file in enumerate(new_videos, 1):
                logger.debug(f"  Processing video {idx}/{len(new_videos)}: {video_file}")
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
                
                # Transcode for DaVinci if enabled
                final_path = video_path
                if self.download_config.davinci_mode:
                    logger.info(f"    ↳ Transcoding {video_file} for DaVinci...")
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
        tiers: List[str] = None
    ) -> List[DownloadedVideo]:
        """Download videos for a keyword across specified tiers"""
        if tiers is None:
            tiers = list(self.DURATION_TIERS.keys())
        
        all_downloaded = []
        
        for tier in tiers:
            # Check checkpoint - skip if already done
            if self.checkpoint:
                video_key = f"{keyword}|{tier}"
                if video_key in self.checkpoint.completed_videos:
                    logger.debug(f"Skipping {keyword} ({tier}) - already completed")
                    continue
            
            logger.info(f"  [{tier}] Downloading...")
            downloaded = self._download_single(keyword, tier, output_dir)
            
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
        resume: bool = False
    ) -> Tuple[List[DownloadedVideo], List[str]]:
        """
        Download videos for all keywords.
        
        Args:
            keywords: List of search keywords
            output_dir: Output directory
            max_concurrent: Max concurrent downloads
            resume: Whether to resume from checkpoint
        
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
        
        # Process keywords with thread pool
        # Note: We process keywords sequentially but tiers can overlap slightly
        # to avoid overwhelming the API
        
        for i, keyword in enumerate(keywords, 1):
            logger.info(f"\n[{i}/{len(keywords)}] Processing: {keyword}")
            
            self.checkpoint.current_keyword = keyword
            self._save_checkpoint()
            
            downloaded = self.download_for_keyword(keyword, output_dir)
            
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
                name: f"{config['min']}s-{config['max']}s ({config['per_keyword']}/kw)"
                for name, config in self.DURATION_TIERS.items()
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
