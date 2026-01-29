"""
FFmpeg transcoding and codec detection.

Migrated from VideoDownloader transcoding methods (lines 416-548, 1032-1105).
Handles video codec detection, hardware acceleration, and FFmpeg command building.
"""

from __future__ import annotations

import subprocess
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


class TranscodingManager:
    """Handles video transcoding and codec detection for DaVinci Resolve compatibility.

    Migrated from VideoDownloader transcoding methods.
    """

    def __init__(self, config: 'Config'):
        """
        Initialize TranscodingManager.

        Args:
            config: Config object with download settings
        """
        self.config = config
        self.download_config = config.download
        self.hw_accel = self._detect_hw_accel()

    def get_video_codec(self, video_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get video codec info using ffprobe.

        Migrated from downloader.py lines 416-439.

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (codec_name, container_format) or (None, None) if failed
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, encoding='utf-8', errors='replace')
            codec = result.stdout.strip().lower()

            # Get container format
            container = Path(video_path).suffix.lower().lstrip('.')

            return codec, container
        except Exception as e:
            logger.debug(f"Could not get codec info: {e}")
            return None, None

    def needs_transcoding(self, video_path: str) -> Tuple[bool, str]:
        """
        Check if video needs transcoding for DaVinci Resolve.

        Migrated from downloader.py lines 441-500.

        DaVinci-compatible codecs (no transcode needed):
        - H.264/AVC in MP4/MOV container
        - H.265/HEVC in MP4/MOV container
        - ProRes in MOV container
        - DNxHD/DNxHR in MOV/MXF container

        Needs transcoding:
        - VP9 (WebM) - common from YouTube
        - AV1 - newer YouTube format
        - VP8 - older WebM

        Args:
            video_path: Path to video file

        Returns:
            Tuple of (needs_transcode: bool, reason: str)
        """
        # Skip audio-only files (audio-first pipeline downloads mp3/m4a)
        audio_extensions = {'.mp3', '.m4a', '.opus', '.ogg', '.wav', '.flac', '.aac'}
        if Path(video_path).suffix.lower() in audio_extensions:
            return False, "Audio file (no transcode)"

        codec, container = self.get_video_codec(video_path)

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
        """
        Auto-detect available hardware acceleration.

        Migrated from downloader.py lines 502-524.

        Returns:
            Hardware acceleration type: 'nvidia', 'amd', 'intel', 'mac', or 'none'
        """
        hw_accel = self.download_config.hw_accel

        if hw_accel != 'auto':
            return hw_accel

        try:
            result = subprocess.run(['ffmpeg', '-encoders'], capture_output=True, text=True, encoding='utf-8', errors='replace')
            encoders = result.stdout + result.stderr

            if 'h264_nvenc' in encoders:
                return 'nvidia'
            elif 'h264_amf' in encoders:
                return 'amd'
            elif 'h264_qsv' in encoders:
                return 'intel'
            elif 'h264_videotoolbox' in encoders:
                return 'mac'
        except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
            # FFmpeg not installed or failed to run - fall back to no hardware acceleration
            logger.debug(f"Could not detect GPU encoder (ffmpeg unavailable): {e}")

        return 'none'

    def build_format_string(self) -> str:
        """
        Build yt-dlp format selection string.

        Migrated from downloader.py lines 526-548.

        In DaVinci mode, prefers h264 to avoid transcoding vp9/av1.

        Returns:
            Format string for yt-dlp -f argument
        """
        quality = self.download_config.quality
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

    def build_filter_string(self, tier: str, duration_tiers: dict) -> str:
        """
        Build filter string for duration and title blacklist.

        Migrated from downloader.py lines 550-580.

        Args:
            tier: Duration tier name (short, medium, long, longer)
            duration_tiers: Duration tier configuration dict

        Returns:
            Filter string for yt-dlp --match-filter argument
        """
        tier_config = duration_tiers[tier]

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

    def get_ffmpeg_transcode_cmd(
        self,
        input_path: str,
        output_path: str
    ) -> Tuple[List[str], str]:
        """
        Build FFmpeg transcode command for DaVinci with GPU acceleration.

        Migrated from downloader.py lines 1032-1105.

        Args:
            input_path: Source video file path
            output_path: Destination video file path

        Returns:
            Tuple of (ffmpeg_command_list, actual_output_path)
            Output path may change based on codec (e.g., .mov for ProRes)
        """
        codec = self.download_config.davinci_codec
        hw_accel = self.hw_accel
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
        # -hide_banner and -loglevel error suppress decoder warnings (mmco, etc)
        cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y']

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
