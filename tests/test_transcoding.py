"""
Comprehensive tests for transcoding module.

Covers:
- TranscodingManager initialization
- Video codec detection via ffprobe
- Transcode necessity checking (DaVinci compatibility)
- Hardware acceleration detection
- yt-dlp format string building (DaVinci mode)
- Filter string building (duration, title blacklist)
- FFmpeg transcode command building (GPU acceleration)

Created: 2026-01-09 (Phase 7.3)
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
from pathlib import Path

from src.downloader.transcoding import TranscodingManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with transcoding settings"""
    config = Mock()
    config.download = Mock()
    config.download.hw_accel = 'auto'
    config.download.quality = 'best'
    config.download.davinci_mode = True
    config.download.davinci_codec = 'h264'
    config.download.davinci_prores_profile = 'proxy'
    config.download.hw_quality = 'high'
    config.download.min_views = 0
    config.download.title_blacklist = []
    return config


@pytest.fixture
def transcoder(mock_config):
    """Create TranscodingManager instance"""
    with patch.object(TranscodingManager, '_detect_hw_accel', return_value='nvidia'):
        return TranscodingManager(config=mock_config)


@pytest.fixture
def mock_duration_tiers():
    """Create mock duration tiers"""
    return {
        'short': {'min': 0, 'max': 30},
        'medium': {'min': 30, 'max': 90},
        'long': {'min': 90, 'max': 300},
        'longer': {'min': 300, 'max': 900}
    }


# ============================================================================
# Test Initialization
# ============================================================================

class TestTranscodingManagerInit:
    """Test TranscodingManager initialization"""

    @patch('subprocess.run')
    def test_init_with_config(self, mock_run, mock_config):
        """Test initialization with config"""
        mock_run.return_value = Mock(stdout='h264_nvenc', stderr='', returncode=0)

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.config == mock_config
        assert transcoder.download_config == mock_config.download
        assert transcoder.hw_accel in ['nvidia', 'amd', 'intel', 'mac', 'none']

    @patch('subprocess.run')
    def test_init_auto_detects_hw_accel(self, mock_run, mock_config):
        """Test auto-detection of hardware acceleration"""
        mock_run.return_value = Mock(stdout='h264_nvenc available', stderr='', returncode=0)

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.hw_accel == 'nvidia'


# ============================================================================
# Test Codec Detection
# ============================================================================

class TestCodecDetection:
    """Test video codec detection"""

    @patch('subprocess.run')
    def test_get_video_codec_h264_mp4(self, mock_run, transcoder):
        """Test detecting H.264 codec in MP4 container"""
        mock_run.return_value = Mock(
            stdout='h264\n',
            stderr='',
            returncode=0
        )

        codec, container = transcoder.get_video_codec('video.mp4')

        assert codec == 'h264'
        assert container == 'mp4'

    @patch('subprocess.run')
    def test_get_video_codec_vp9_webm(self, mock_run, transcoder):
        """Test detecting VP9 codec in WebM container"""
        mock_run.return_value = Mock(
            stdout='vp9\n',
            stderr='',
            returncode=0
        )

        codec, container = transcoder.get_video_codec('video.webm')

        assert codec == 'vp9'
        assert container == 'webm'

    @patch('subprocess.run')
    def test_get_video_codec_hevc_mov(self, mock_run, transcoder):
        """Test detecting HEVC codec in MOV container"""
        mock_run.return_value = Mock(
            stdout='hevc\n',
            stderr='',
            returncode=0
        )

        codec, container = transcoder.get_video_codec('video.mov')

        assert codec == 'hevc'
        assert container == 'mov'

    @patch('subprocess.run')
    def test_get_video_codec_timeout(self, mock_run, transcoder):
        """Test timeout handling in codec detection"""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd='ffprobe', timeout=30)

        codec, container = transcoder.get_video_codec('video.mp4')

        assert codec is None
        assert container is None

    @patch('subprocess.run')
    def test_get_video_codec_error(self, mock_run, transcoder):
        """Test error handling in codec detection"""
        mock_run.side_effect = Exception("ffprobe error")

        codec, container = transcoder.get_video_codec('video.mp4')

        assert codec is None
        assert container is None


# ============================================================================
# Test Transcode Necessity
# ============================================================================

class TestTranscodeNecessity:
    """Test checking if videos need transcoding"""

    def test_needs_transcoding_audio_file(self, transcoder):
        """Test that audio files don't need transcoding"""
        needs, reason = transcoder.needs_transcoding('audio.mp3')

        assert needs is False
        assert "Audio file" in reason

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_h264_mp4(self, mock_codec, transcoder):
        """Test H.264 MP4 doesn't need transcoding"""
        mock_codec.return_value = ('h264', 'mp4')

        needs, reason = transcoder.needs_transcoding('video.mp4')

        assert needs is False
        assert "compatible" in reason.lower()

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_vp9_webm(self, mock_codec, transcoder):
        """Test VP9 WebM needs transcoding"""
        mock_codec.return_value = ('vp9', 'webm')

        needs, reason = transcoder.needs_transcoding('video.webm')

        assert needs is True
        assert "vp9" in reason.lower()

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_av1(self, mock_codec, transcoder):
        """Test AV1 needs transcoding"""
        mock_codec.return_value = ('av1', 'mp4')

        needs, reason = transcoder.needs_transcoding('video.mp4')

        assert needs is True
        assert "av1" in reason.lower()

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_hevc_mov(self, mock_codec, transcoder):
        """Test HEVC MOV doesn't need transcoding"""
        mock_codec.return_value = ('hevc', 'mov')

        needs, reason = transcoder.needs_transcoding('video.mov')

        assert needs is False
        assert "compatible" in reason.lower()

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_prores(self, mock_codec, transcoder):
        """Test ProRes doesn't need transcoding"""
        mock_codec.return_value = ('prores', 'mov')

        needs, reason = transcoder.needs_transcoding('video.mov')

        assert needs is False

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_dnxhd(self, mock_codec, transcoder):
        """Test DNxHD doesn't need transcoding"""
        mock_codec.return_value = ('dnxhd', 'mxf')

        needs, reason = transcoder.needs_transcoding('video.mxf')

        assert needs is False

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_unknown_codec(self, mock_codec, transcoder):
        """Test unknown codec needs transcoding (safe fallback)"""
        mock_codec.return_value = ('unknown_codec', 'mp4')

        needs, reason = transcoder.needs_transcoding('video.mp4')

        assert needs is True
        assert "unknown" in reason.lower()

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_codec_detection_failed(self, mock_codec, transcoder):
        """Test when codec detection fails"""
        mock_codec.return_value = (None, None)

        needs, reason = transcoder.needs_transcoding('video.mp4')

        assert needs is True
        assert "could not determine" in reason.lower()

    @patch.object(TranscodingManager, 'get_video_codec')
    def test_needs_transcoding_h264_any_container(self, mock_codec, transcoder):
        """Test H.264 in any container is compatible"""
        mock_codec.return_value = ('h264', 'mkv')

        needs, reason = transcoder.needs_transcoding('video.mkv')

        # H.264 is compatible even in mkv
        assert needs is False


# ============================================================================
# Test Hardware Acceleration Detection
# ============================================================================

class TestHardwareAcceleration:
    """Test hardware acceleration detection"""

    @patch('subprocess.run')
    def test_detect_hw_accel_nvidia(self, mock_run, mock_config):
        """Test NVIDIA GPU detection"""
        mock_run.return_value = Mock(
            stdout='h264_nvenc is available',
            stderr='',
            returncode=0
        )

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.hw_accel == 'nvidia'

    @patch('subprocess.run')
    def test_detect_hw_accel_amd(self, mock_run, mock_config):
        """Test AMD GPU detection"""
        mock_run.return_value = Mock(
            stdout='h264_amf is available',
            stderr='',
            returncode=0
        )

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.hw_accel == 'amd'

    @patch('subprocess.run')
    def test_detect_hw_accel_intel(self, mock_run, mock_config):
        """Test Intel QSV detection"""
        mock_run.return_value = Mock(
            stdout='h264_qsv is available',
            stderr='',
            returncode=0
        )

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.hw_accel == 'intel'

    @patch('subprocess.run')
    def test_detect_hw_accel_mac(self, mock_run, mock_config):
        """Test macOS VideoToolbox detection"""
        mock_run.return_value = Mock(
            stdout='h264_videotoolbox is available',
            stderr='',
            returncode=0
        )

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.hw_accel == 'mac'

    @patch('subprocess.run')
    def test_detect_hw_accel_none(self, mock_run, mock_config):
        """Test fallback to CPU when no GPU found"""
        mock_run.return_value = Mock(
            stdout='libx264 is available',
            stderr='',
            returncode=0
        )

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.hw_accel == 'none'

    @patch('subprocess.run')
    def test_detect_hw_accel_manual_override(self, mock_run, mock_config):
        """Test manual hardware acceleration override"""
        mock_config.download.hw_accel = 'nvidia'

        transcoder = TranscodingManager(config=mock_config)

        # Should use manual setting, not call ffmpeg
        assert transcoder.hw_accel == 'nvidia'
        mock_run.assert_not_called()

    @patch('subprocess.run')
    def test_detect_hw_accel_error_handling(self, mock_run, mock_config):
        """Test error handling in hardware detection"""
        mock_run.side_effect = Exception("ffmpeg error")

        transcoder = TranscodingManager(config=mock_config)

        assert transcoder.hw_accel == 'none'


# ============================================================================
# Test Format String Building
# ============================================================================

class TestFormatStringBuilding:
    """Test yt-dlp format string building"""

    def test_build_format_string_best_davinci(self, transcoder):
        """Test best quality in DaVinci mode (prefers H.264)"""
        transcoder.download_config.quality = 'best'
        transcoder.download_config.davinci_mode = True

        format_str = transcoder.build_format_string()

        assert 'avc1' in format_str
        assert 'bestvideo' in format_str

    def test_build_format_string_best_normal(self, transcoder):
        """Test best quality in normal mode"""
        transcoder.download_config.quality = 'best'
        transcoder.download_config.davinci_mode = False

        format_str = transcoder.build_format_string()

        assert format_str == 'bestvideo+bestaudio/best'

    def test_build_format_string_audio_only(self, transcoder):
        """Test audio-only format"""
        transcoder.download_config.quality = 'audio'

        format_str = transcoder.build_format_string()

        assert format_str == 'bestaudio'

    def test_build_format_string_1080p_davinci(self, transcoder):
        """Test 1080p quality in DaVinci mode"""
        transcoder.download_config.quality = '1080p'
        transcoder.download_config.davinci_mode = True

        format_str = transcoder.build_format_string()

        assert 'height<=1080' in format_str
        assert 'avc1' in format_str

    def test_build_format_string_720p_normal(self, transcoder):
        """Test 720p quality in normal mode"""
        transcoder.download_config.quality = '720p'
        transcoder.download_config.davinci_mode = False

        format_str = transcoder.build_format_string()

        assert 'height<=720' in format_str
        assert 'avc1' not in format_str

    def test_build_format_string_480p(self, transcoder):
        """Test 480p quality"""
        transcoder.download_config.quality = '480p'

        format_str = transcoder.build_format_string()

        assert 'height<=480' in format_str


# ============================================================================
# Test Filter String Building
# ============================================================================

class TestFilterStringBuilding:
    """Test yt-dlp filter string building"""

    def test_build_filter_string_basic(self, transcoder, mock_duration_tiers):
        """Test basic filter string with duration only"""
        filter_str = transcoder.build_filter_string('medium', mock_duration_tiers)

        assert 'duration>30' in filter_str
        assert 'duration<90' in filter_str
        assert '!is_live' in filter_str
        assert '!was_live' in filter_str  # Also filter completed livestreams

    def test_build_filter_string_with_min_views(self, transcoder, mock_duration_tiers):
        """Test filter string with minimum views"""
        transcoder.download_config.min_views = 1000

        filter_str = transcoder.build_filter_string('medium', mock_duration_tiers)

        assert 'view_count>1000' in filter_str

    def test_build_filter_string_with_title_blacklist(self, transcoder, mock_duration_tiers):
        """Test filter string with title blacklist"""
        transcoder.download_config.title_blacklist = ['music video', 'vlog']

        filter_str = transcoder.build_filter_string('medium', mock_duration_tiers)

        assert "title!*='music video'" in filter_str
        assert "title!*='vlog'" in filter_str

    def test_build_filter_string_all_tiers(self, transcoder, mock_duration_tiers):
        """Test filter strings for all duration tiers"""
        for tier in ['short', 'medium', 'long', 'longer']:
            filter_str = transcoder.build_filter_string(tier, mock_duration_tiers)

            tier_config = mock_duration_tiers[tier]
            assert f"duration>{tier_config['min']}" in filter_str
            assert f"duration<{tier_config['max']}" in filter_str

    def test_build_filter_string_escapes_quotes(self, transcoder, mock_duration_tiers):
        """Test filter string escapes quotes in title blacklist"""
        transcoder.download_config.title_blacklist = ["it's raining"]

        filter_str = transcoder.build_filter_string('medium', mock_duration_tiers)

        # Should escape the single quote
        assert "it\\'s raining" in filter_str

    def test_build_filter_string_dataclass_format(self, transcoder):
        """Test filter string with dataclass duration tiers"""
        tier_config = Mock()
        tier_config.min_seconds = 30
        tier_config.max_seconds = 90

        duration_tiers = {'medium': tier_config}

        filter_str = transcoder.build_filter_string('medium', duration_tiers)

        assert 'duration>30' in filter_str
        assert 'duration<90' in filter_str


# ============================================================================
# Test FFmpeg Command Building
# ============================================================================

class TestFFmpegCommandBuilding:
    """Test FFmpeg transcode command building"""

    def test_get_ffmpeg_cmd_h264_nvidia(self, transcoder):
        """Test H.264 encoding with NVIDIA GPU"""
        transcoder.download_config.davinci_codec = 'h264'
        transcoder.hw_accel = 'nvidia'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert 'ffmpeg' in cmd
        assert '-hwaccel' in cmd
        assert 'cuda' in cmd
        assert 'h264_nvenc' in cmd
        assert '-cq' in cmd
        assert 'output.mp4' in cmd

    def test_get_ffmpeg_cmd_h264_amd(self, transcoder):
        """Test H.264 encoding with AMD GPU"""
        transcoder.download_config.davinci_codec = 'h264'
        transcoder.hw_accel = 'amd'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert 'h264_amf' in cmd

    def test_get_ffmpeg_cmd_h264_intel(self, transcoder):
        """Test H.264 encoding with Intel QSV"""
        transcoder.download_config.davinci_codec = 'h264'
        transcoder.hw_accel = 'intel'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert 'h264_qsv' in cmd
        assert 'qsv' in cmd

    def test_get_ffmpeg_cmd_h264_mac(self, transcoder):
        """Test H.264 encoding with macOS VideoToolbox"""
        transcoder.download_config.davinci_codec = 'h264'
        transcoder.hw_accel = 'mac'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert 'h264_videotoolbox' in cmd

    def test_get_ffmpeg_cmd_h264_cpu(self, transcoder):
        """Test H.264 encoding with CPU fallback"""
        transcoder.download_config.davinci_codec = 'h264'
        transcoder.hw_accel = 'none'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert 'libx264' in cmd
        assert '-crf' in cmd
        assert '-preset' in cmd

    def test_get_ffmpeg_cmd_h265_nvidia(self, transcoder):
        """Test H.265 encoding with NVIDIA GPU"""
        transcoder.download_config.davinci_codec = 'h265'
        transcoder.hw_accel = 'nvidia'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert 'hevc_nvenc' in cmd

    def test_get_ffmpeg_cmd_prores(self, transcoder):
        """Test ProRes encoding (CPU only)"""
        transcoder.download_config.davinci_codec = 'prores'
        transcoder.hw_accel = 'nvidia'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.mp4', 'output.mp4')

        assert 'prores_ks' in cmd
        assert '-profile:v' in cmd
        assert 'pcm_s16le' in cmd  # Uncompressed audio
        assert output_path.endswith('.mov')  # Output changed to MOV

    def test_get_ffmpeg_cmd_dnxhd(self, transcoder):
        """Test DNxHD encoding"""
        transcoder.download_config.davinci_codec = 'dnxhd'

        cmd, output_path = transcoder.get_ffmpeg_transcode_cmd('input.mp4', 'output.mp4')

        assert 'dnxhd' in cmd
        assert 'dnxhr_sq' in cmd
        assert output_path.endswith('.mxf')  # Output changed to MXF

    def test_get_ffmpeg_cmd_quality_presets(self, transcoder):
        """Test different quality presets"""
        for quality in ['low', 'medium', 'high']:
            transcoder.download_config.hw_quality = quality
            transcoder.hw_accel = 'none'

            cmd, _ = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

            assert '-crf' in cmd

    def test_get_ffmpeg_cmd_custom_crf(self, transcoder):
        """Test custom CRF override from config"""
        transcoder.config.downloading = Mock()
        transcoder.config.downloading.transcode_crf = 20

        cmd, _ = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        # Should use custom CRF of 20
        cq_index = cmd.index('-cq')
        assert cmd[cq_index + 1] == '20'

    def test_get_ffmpeg_cmd_prores_profiles(self, transcoder):
        """Test different ProRes profiles"""
        transcoder.download_config.davinci_codec = 'prores'

        for profile in ['proxy', 'lt', 'standard', 'hq']:
            transcoder.download_config.davinci_prores_profile = profile

            cmd, _ = transcoder.get_ffmpeg_transcode_cmd('input.mp4', 'output.mp4')

            assert '-profile:v' in cmd
            # Profile should be in command
            assert 'prores_ks' in cmd

    def test_get_ffmpeg_cmd_audio_settings(self, transcoder):
        """Test audio codec settings"""
        cmd, _ = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert '-c:a' in cmd
        assert 'aac' in cmd or 'pcm_s16le' in cmd


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    def test_needs_transcoding_all_audio_formats(self, transcoder):
        """Test all audio formats are skipped"""
        audio_files = ['audio.mp3', 'audio.m4a', 'audio.opus', 'audio.ogg',
                      'audio.wav', 'audio.flac', 'audio.aac']

        for audio_file in audio_files:
            needs, reason = transcoder.needs_transcoding(audio_file)
            assert needs is False
            assert "Audio file" in reason

    def test_build_format_string_uppercase_quality(self, transcoder):
        """Test format string with uppercase quality (edge case)"""
        transcoder.download_config.quality = '1080P'

        format_str = transcoder.build_format_string()

        # Should handle uppercase 'P'
        assert 'height<=1080' in format_str

    def test_build_filter_string_empty_blacklist(self, transcoder, mock_duration_tiers):
        """Test filter string with empty blacklist"""
        transcoder.download_config.title_blacklist = []

        filter_str = transcoder.build_filter_string('medium', mock_duration_tiers)

        # Should still have duration and live stream filters
        assert 'duration>' in filter_str
        assert '!is_live' in filter_str

    @patch('subprocess.run')
    def test_get_video_codec_case_insensitive(self, mock_run, transcoder):
        """Test codec detection is case-insensitive"""
        mock_run.return_value = Mock(stdout='H264\n', stderr='', returncode=0)

        codec, container = transcoder.get_video_codec('video.MP4')

        assert codec == 'h264'  # Lowercased
        assert container == 'mp4'  # Lowercased

    def test_get_ffmpeg_cmd_nostdin_flag(self, transcoder):
        """Test that -nostdin flag is always included"""
        cmd, _ = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert '-nostdin' in cmd

    def test_get_ffmpeg_cmd_hide_banner(self, transcoder):
        """Test that -hide_banner flag is included"""
        cmd, _ = transcoder.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')

        assert '-hide_banner' in cmd
        assert '-loglevel' in cmd
        assert 'error' in cmd
