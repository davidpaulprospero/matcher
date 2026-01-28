"""
Tests for src/downloader/transcoding.py exception and edge case paths.

Targets:
- Line 132: Compatible codec not in combination list (h264/hevc edge case)
- Lines 309-312: h265 codec with mac/else branches
- Hardware acceleration detection
"""

import pytest
from unittest.mock import patch, MagicMock, Mock
from dataclasses import dataclass

from src.downloader.transcoding import TranscodingManager


@dataclass
class MockDownloadConfig:
    """Mock download config for testing."""
    hw_accel: str = "auto"
    quality: str = "1080p"
    davinci_mode: bool = True
    davinci_codec: str = "h264"
    davinci_prores_profile: str = "proxy"
    hw_quality: str = "high"
    min_views: int = 0
    title_blacklist: list = None

    def __post_init__(self):
        if self.title_blacklist is None:
            self.title_blacklist = []


@dataclass
class MockDownloadingConfig:
    """Mock downloading config."""
    transcode_crf: int = 18


@dataclass
class MockConfig:
    """Mock config for TranscodingManager."""
    download: MockDownloadConfig = None
    downloading: MockDownloadingConfig = None

    def __post_init__(self):
        if self.download is None:
            self.download = MockDownloadConfig()
        if self.downloading is None:
            self.downloading = MockDownloadingConfig()


class TestNeedsTranscoding:
    """Test needs_transcoding edge cases."""

    @pytest.mark.fast
    def test_compatible_codec_not_in_combination_list(self, tmp_path):
        """Test h264/hevc codec in unusual container (line 132)."""
        # Create a mock video file
        video_file = tmp_path / "test.avi"
        video_file.touch()

        config = MockConfig()
        manager = TranscodingManager(config)

        # Mock get_video_codec to return h264 in avi container
        with patch.object(manager, 'get_video_codec', return_value=('h264', 'avi')):
            needs, reason = manager.needs_transcoding(str(video_file))

            # h264 in avi is not in compatible_combinations, but h264 is compatible
            assert needs == False
            assert "Compatible codec (h264)" in reason

    @pytest.mark.fast
    def test_hevc_codec_not_in_combination(self, tmp_path):
        """Test hevc codec in unusual container (line 132)."""
        video_file = tmp_path / "test.ts"
        video_file.touch()

        config = MockConfig()
        manager = TranscodingManager(config)

        with patch.object(manager, 'get_video_codec', return_value=('hevc', 'ts')):
            needs, reason = manager.needs_transcoding(str(video_file))

            assert needs == False
            assert "Compatible codec (hevc)" in reason

    @pytest.mark.fast
    def test_avc_codec_not_in_combination(self, tmp_path):
        """Test avc codec in unusual container."""
        video_file = tmp_path / "test.mts"
        video_file.touch()

        config = MockConfig()
        manager = TranscodingManager(config)

        with patch.object(manager, 'get_video_codec', return_value=('avc', 'mts')):
            needs, reason = manager.needs_transcoding(str(video_file))

            assert needs == False
            assert "Compatible codec" in reason

    @pytest.mark.fast
    def test_h265_codec_compatible(self, tmp_path):
        """Test h265 codec in unusual container."""
        video_file = tmp_path / "test.ts"
        video_file.touch()

        config = MockConfig()
        manager = TranscodingManager(config)

        with patch.object(manager, 'get_video_codec', return_value=('h265', 'ts')):
            needs, reason = manager.needs_transcoding(str(video_file))

            assert needs == False
            assert "Compatible codec" in reason

    @pytest.mark.fast
    def test_unknown_codec_needs_transcode(self, tmp_path):
        """Test unknown codec combination requires transcoding."""
        video_file = tmp_path / "test.weird"
        video_file.touch()

        config = MockConfig()
        manager = TranscodingManager(config)

        with patch.object(manager, 'get_video_codec', return_value=('unknown_codec', 'weird')):
            needs, reason = manager.needs_transcoding(str(video_file))

            assert needs == True
            assert "Unknown codec combination" in reason


class TestFFmpegTranscodeCmd:
    """Test FFmpeg transcode command building."""

    @pytest.mark.fast
    def test_h265_with_mac_acceleration(self, tmp_path):
        """Test h265 codec with mac hw_accel (line 309-310)."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "h265"
        config.download.hw_accel = "mac"

        manager = TranscodingManager(config)
        manager.hw_accel = "mac"  # Override detection

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "-c:v" in cmd
        assert "hevc_videotoolbox" in cmd
        assert "-q:v" in cmd
        assert "65" in cmd

    @pytest.mark.fast
    def test_h265_with_cpu_encoding(self, tmp_path):
        """Test h265 codec with CPU encoding (lines 311-312)."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "h265"
        config.download.hw_accel = "none"

        manager = TranscodingManager(config)
        manager.hw_accel = "none"

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "-c:v" in cmd
        assert "libx265" in cmd
        assert "-crf" in cmd

    @pytest.mark.fast
    def test_h264_with_nvidia_encoding(self, tmp_path):
        """Test h264 codec with NVIDIA hw_accel."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "h264"
        config.download.hw_accel = "nvidia"

        manager = TranscodingManager(config)
        manager.hw_accel = "nvidia"

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "h264_nvenc" in cmd
        assert "-hwaccel" in cmd
        assert "cuda" in cmd

    @pytest.mark.fast
    def test_h264_with_amd_encoding(self, tmp_path):
        """Test h264 codec with AMD hw_accel."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "h264"

        manager = TranscodingManager(config)
        manager.hw_accel = "amd"

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "h264_amf" in cmd
        assert "-quality" in cmd

    @pytest.mark.fast
    def test_h264_with_intel_encoding(self, tmp_path):
        """Test h264 codec with Intel QSV hw_accel."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "h264"

        manager = TranscodingManager(config)
        manager.hw_accel = "intel"

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "h264_qsv" in cmd
        assert "-global_quality" in cmd

    @pytest.mark.fast
    def test_h264_with_mac_videotoolbox(self, tmp_path):
        """Test h264 codec with Mac VideoToolbox."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "h264"

        manager = TranscodingManager(config)
        manager.hw_accel = "mac"

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "h264_videotoolbox" in cmd

    @pytest.mark.fast
    def test_h264_with_cpu_fallback(self, tmp_path):
        """Test h264 codec with CPU encoding."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "h264"
        config.download.hw_accel = "none"

        manager = TranscodingManager(config)
        manager.hw_accel = "none"

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "libx264" in cmd
        assert "-crf" in cmd
        assert "-preset" in cmd
        assert "medium" in cmd

    @pytest.mark.fast
    def test_prores_codec(self, tmp_path):
        """Test ProRes codec output (changes extension to .mov)."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "prores"
        config.download.davinci_prores_profile = "hq"

        manager = TranscodingManager(config)
        manager.hw_accel = "nvidia"  # GPU but ProRes uses CPU

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "prores_ks" in cmd
        assert "-profile:v" in cmd
        assert "3" in cmd  # hq profile
        assert output_path.endswith('.mov')

    @pytest.mark.fast
    def test_dnxhd_codec(self, tmp_path):
        """Test DNxHD codec output (changes extension to .mxf)."""
        input_file = tmp_path / "input.mp4"
        output_file = tmp_path / "output.mp4"
        input_file.touch()

        config = MockConfig()
        config.download.davinci_codec = "dnxhd"

        manager = TranscodingManager(config)
        manager.hw_accel = "nvidia"

        cmd, output_path = manager.get_ffmpeg_transcode_cmd(
            str(input_file),
            str(output_file)
        )

        assert "dnxhd" in cmd
        assert "dnxhr_sq" in cmd
        assert output_path.endswith('.mxf')


class TestHardwareAccelDetection:
    """Test hardware acceleration detection."""

    @pytest.mark.integration
    def test_detect_nvidia(self):
        """Test NVIDIA detection."""
        config = MockConfig()
        config.download.hw_accel = "auto"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="h264_nvenc - NVIDIA encoder",
                stderr=""
            )
            manager = TranscodingManager(config)
            assert manager.hw_accel == "nvidia"

    @pytest.mark.integration
    def test_detect_amd(self):
        """Test AMD detection."""
        config = MockConfig()
        config.download.hw_accel = "auto"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="h264_amf - AMD encoder",
                stderr=""
            )
            manager = TranscodingManager(config)
            assert manager.hw_accel == "amd"

    @pytest.mark.integration
    def test_detect_intel(self):
        """Test Intel QSV detection."""
        config = MockConfig()
        config.download.hw_accel = "auto"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="h264_qsv - Intel QuickSync",
                stderr=""
            )
            manager = TranscodingManager(config)
            assert manager.hw_accel == "intel"

    @pytest.mark.integration
    def test_detect_mac(self):
        """Test Mac VideoToolbox detection."""
        config = MockConfig()
        config.download.hw_accel = "auto"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="h264_videotoolbox - Apple encoder",
                stderr=""
            )
            manager = TranscodingManager(config)
            assert manager.hw_accel == "mac"

    @pytest.mark.integration
    def test_detect_none(self):
        """Test fallback to none when no GPU detected."""
        config = MockConfig()
        config.download.hw_accel = "auto"

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="libx264 - software encoder",
                stderr=""
            )
            manager = TranscodingManager(config)
            assert manager.hw_accel == "none"

    @pytest.mark.integration
    def test_detect_exception_fallback(self):
        """Test fallback to none on subprocess exception."""
        config = MockConfig()
        config.download.hw_accel = "auto"

        with patch('subprocess.run', side_effect=FileNotFoundError("ffmpeg not found")):
            manager = TranscodingManager(config)
            assert manager.hw_accel == "none"

    @pytest.mark.integration
    def test_manual_override(self):
        """Test manual hw_accel setting bypasses detection."""
        config = MockConfig()
        config.download.hw_accel = "cuda"

        # Should not call subprocess.run
        with patch('subprocess.run') as mock_run:
            manager = TranscodingManager(config)
            assert manager.hw_accel == "cuda"
            mock_run.assert_not_called()


class TestBuildFormatString:
    """Test format string building."""

    @pytest.mark.integration
    def test_best_quality_davinci_mode(self):
        """Test best quality with DaVinci mode."""
        config = MockConfig()
        config.download.quality = "best"
        config.download.davinci_mode = True

        with patch('subprocess.run'):  # Skip hw detection
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        format_str = manager.build_format_string()
        assert "avc1" in format_str

    @pytest.mark.integration
    def test_best_quality_no_davinci(self):
        """Test best quality without DaVinci mode."""
        config = MockConfig()
        config.download.quality = "best"
        config.download.davinci_mode = False

        with patch('subprocess.run'):
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        format_str = manager.build_format_string()
        assert "bestvideo+bestaudio" in format_str

    @pytest.mark.integration
    def test_audio_only(self):
        """Test audio-only format."""
        config = MockConfig()
        config.download.quality = "audio"

        with patch('subprocess.run'):
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        format_str = manager.build_format_string()
        assert format_str == "bestaudio"

    @pytest.mark.integration
    def test_specific_quality_davinci(self):
        """Test specific quality (720p) with DaVinci mode."""
        config = MockConfig()
        config.download.quality = "720p"
        config.download.davinci_mode = True

        with patch('subprocess.run'):
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        format_str = manager.build_format_string()
        assert "height<=720" in format_str
        assert "avc1" in format_str


class TestBuildFilterString:
    """Test filter string building."""

    @pytest.mark.integration
    def test_filter_with_min_views(self, tmp_path):
        """Test filter includes min views check."""
        config = MockConfig()
        config.download.min_views = 1000

        with patch('subprocess.run'):
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        duration_tiers = {
            'short': {'min': 0, 'max': 60}
        }

        filter_str = manager.build_filter_string('short', duration_tiers)
        assert "view_count>1000" in filter_str

    @pytest.mark.integration
    def test_filter_with_title_blacklist(self, tmp_path):
        """Test filter includes title blacklist."""
        config = MockConfig()
        config.download.title_blacklist = ["highlights", "full game"]

        with patch('subprocess.run'):
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        duration_tiers = {
            'short': {'min': 0, 'max': 60}
        }

        filter_str = manager.build_filter_string('short', duration_tiers)
        assert "title!*='highlights'" in filter_str
        assert "title!*='full game'" in filter_str

    @pytest.mark.integration
    def test_filter_escapes_quotes(self, tmp_path):
        """Test filter escapes single quotes in blacklist."""
        config = MockConfig()
        config.download.title_blacklist = ["it's a game"]

        with patch('subprocess.run'):
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        duration_tiers = {
            'short': {'min': 0, 'max': 60}
        }

        filter_str = manager.build_filter_string('short', duration_tiers)
        # Quote should be escaped
        assert "\\'s" in filter_str

    @pytest.mark.integration
    def test_filter_includes_is_live_check(self, tmp_path):
        """Test filter always excludes live streams."""
        config = MockConfig()

        with patch('subprocess.run'):
            manager = TranscodingManager(config)
            manager.hw_accel = "none"

        duration_tiers = {
            'medium': {'min': 60, 'max': 300}
        }

        filter_str = manager.build_filter_string('medium', duration_tiers)
        assert "!is_live" in filter_str


class TestGetVideoCodec:
    """Test video codec detection."""

    @pytest.mark.integration
    def test_get_video_codec_success(self, tmp_path):
        """Test successful codec detection."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        config = MockConfig()

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="h264\n",
                stderr=""
            )

            # Create manager with mocked detection
            with patch.object(TranscodingManager, '_detect_hw_accel', return_value='none'):
                manager = TranscodingManager(config)

            codec, container = manager.get_video_codec(str(video_file))

            assert codec == "h264"
            assert container == "mp4"

    @pytest.mark.integration
    def test_get_video_codec_failure(self, tmp_path):
        """Test codec detection failure."""
        video_file = tmp_path / "test.mp4"
        video_file.touch()

        config = MockConfig()

        with patch('subprocess.run', side_effect=Exception("ffprobe not found")):
            with patch.object(TranscodingManager, '_detect_hw_accel', return_value='none'):
                manager = TranscodingManager(config)

            codec, container = manager.get_video_codec(str(video_file))

            assert codec is None
            assert container is None
