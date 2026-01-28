"""
US-007: TranscodingManager FFmpeg command building tests.

Sprint 17 - Testing focus area.
Verifies get_ffmpeg_transcode_cmd() builds correct FFmpeg commands for:
- NVIDIA GPU (h264_nvenc, -hwaccel cuda)
- AMD GPU (h264_amf, decode flags)
- CPU software fallback (libx264, crf, no -hwaccel)
- ProRes (.mov) and DNxHD (.mxf) output containers
- -y overwrite flag and correct input/output path positions
"""

import pytest
from unittest.mock import Mock, patch
from src.downloader.transcoding import TranscodingManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def _make_transcoder():
    """Factory fixture for creating TranscodingManager with specific hw_accel."""
    def _factory(hw_accel='none', codec='h264', quality='high', prores_profile='proxy'):
        config = Mock()
        config.download = Mock()
        config.download.hw_accel = hw_accel
        config.download.quality = 'best'
        config.download.davinci_mode = True
        config.download.davinci_codec = codec
        config.download.davinci_prores_profile = prores_profile
        config.download.hw_quality = quality
        config.download.min_views = 0
        config.download.title_blacklist = []
        # Ensure no 'downloading' attr so transcode_crf path is skipped
        config.downloading = None
        del config.downloading

        with patch.object(TranscodingManager, '_detect_hw_accel', return_value=hw_accel):
            return TranscodingManager(config=config)
    return _factory


# ============================================================================
# AC1: NVIDIA GPU - h264_nvenc encoder, -hwaccel cuda flags
# ============================================================================

class TestFFmpegNvidiaGpuUS007:
    """AC1: Test get_ffmpeg_transcode_cmd() builds command with NVIDIA GPU acceleration."""

    @pytest.mark.fast
    def test_nvidia_uses_h264_nvenc_encoder(self, _make_transcoder):
        """Verify NVIDIA GPU uses h264_nvenc hardware encoder."""
        t = _make_transcoder(hw_accel='nvidia', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert 'h264_nvenc' in cmd

    @pytest.mark.fast
    def test_nvidia_includes_hwaccel_cuda_flag(self, _make_transcoder):
        """Verify -hwaccel cuda is present for NVIDIA h264 encoding."""
        t = _make_transcoder(hw_accel='nvidia', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        hwaccel_idx = cmd.index('-hwaccel')
        assert cmd[hwaccel_idx + 1] == 'cuda'

    @pytest.mark.fast
    def test_nvidia_includes_hwaccel_output_format_cuda(self, _make_transcoder):
        """Verify -hwaccel_output_format cuda for GPU memory transfer."""
        t = _make_transcoder(hw_accel='nvidia', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        fmt_idx = cmd.index('-hwaccel_output_format')
        assert cmd[fmt_idx + 1] == 'cuda'

    @pytest.mark.fast
    def test_nvidia_h264_includes_nvenc_preset_and_quality(self, _make_transcoder):
        """Verify NVIDIA command includes -preset p4 and -cq with quality value."""
        t = _make_transcoder(hw_accel='nvidia', codec='h264', quality='high')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert '-preset' in cmd
        assert 'p4' in cmd
        assert '-cq' in cmd
        assert '-rc' in cmd
        assert 'vbr' in cmd

    @pytest.mark.fast
    def test_nvidia_h265_uses_hevc_nvenc(self, _make_transcoder):
        """Verify NVIDIA + h265 uses hevc_nvenc encoder."""
        t = _make_transcoder(hw_accel='nvidia', codec='h265')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert 'hevc_nvenc' in cmd

    @pytest.mark.fast
    def test_nvidia_prores_no_cuda_output_format(self, _make_transcoder):
        """ProRes on NVIDIA: uses -hwaccel cuda but NOT -hwaccel_output_format cuda."""
        t = _make_transcoder(hw_accel='nvidia', codec='prores')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        # ProRes uses CPU encoding, so GPU decode goes to system memory only
        assert '-hwaccel' in cmd
        assert 'cuda' in cmd
        assert '-hwaccel_output_format' not in cmd


# ============================================================================
# AC2: AMD GPU - h264_amf encoder and decode flags
# ============================================================================

class TestFFmpegAmdGpuUS007:
    """AC2: Test get_ffmpeg_transcode_cmd() builds command with AMD GPU (h264_amf)."""

    @pytest.mark.fast
    def test_amd_uses_h264_amf_encoder(self, _make_transcoder):
        """Verify AMD GPU uses h264_amf hardware encoder."""
        t = _make_transcoder(hw_accel='amd', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert 'h264_amf' in cmd

    @pytest.mark.fast
    def test_amd_includes_quality_flag(self, _make_transcoder):
        """Verify AMD command includes -quality quality flag."""
        t = _make_transcoder(hw_accel='amd', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        quality_idx = cmd.index('-quality')
        assert cmd[quality_idx + 1] == 'quality'

    @pytest.mark.fast
    def test_amd_no_hwaccel_decode_flags(self, _make_transcoder):
        """AMD h264 encoding should NOT have -hwaccel flags (no GPU decode path)."""
        t = _make_transcoder(hw_accel='amd', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        # AMD path does not add -hwaccel cuda or -hwaccel qsv
        assert '-hwaccel' not in cmd

    @pytest.mark.fast
    def test_amd_includes_video_codec_flag(self, _make_transcoder):
        """Verify -c:v flag is present for AMD encoder."""
        t = _make_transcoder(hw_accel='amd', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        cv_idx = cmd.index('-c:v')
        assert cmd[cv_idx + 1] == 'h264_amf'

    @pytest.mark.fast
    def test_amd_includes_aac_audio(self, _make_transcoder):
        """Verify AMD h264 encoding includes AAC audio codec."""
        t = _make_transcoder(hw_accel='amd', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        ca_idx = cmd.index('-c:a')
        assert cmd[ca_idx + 1] == 'aac'


# ============================================================================
# AC3: CPU fallback - libx264, crf, no -hwaccel flags
# ============================================================================

class TestFFmpegCpuFallbackUS007:
    """AC3: Test get_ffmpeg_transcode_cmd() falls back to libx264 with no GPU."""

    @pytest.mark.fast
    def test_cpu_uses_libx264_encoder(self, _make_transcoder):
        """Verify CPU fallback uses libx264 software encoder."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        cv_idx = cmd.index('-c:v')
        assert cmd[cv_idx + 1] == 'libx264'

    @pytest.mark.fast
    def test_cpu_includes_crf_quality_setting(self, _make_transcoder):
        """Verify CPU fallback uses -crf with quality-derived value."""
        t = _make_transcoder(hw_accel='none', codec='h264', quality='high')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        crf_idx = cmd.index('-crf')
        # 'high' quality maps to cq=18
        assert cmd[crf_idx + 1] == '18'

    @pytest.mark.fast
    def test_cpu_includes_preset_medium(self, _make_transcoder):
        """Verify CPU fallback uses -preset medium."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        preset_idx = cmd.index('-preset')
        assert cmd[preset_idx + 1] == 'medium'

    @pytest.mark.fast
    def test_cpu_has_no_hwaccel_flags(self, _make_transcoder):
        """Verify no -hwaccel flags when using CPU encoding."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert '-hwaccel' not in cmd
        assert '-hwaccel_output_format' not in cmd

    @pytest.mark.fast
    def test_cpu_h265_uses_libx265(self, _make_transcoder):
        """Verify CPU h265 uses libx265 software encoder."""
        t = _make_transcoder(hw_accel='none', codec='h265')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        cv_idx = cmd.index('-c:v')
        assert cmd[cv_idx + 1] == 'libx265'
        assert '-crf' in cmd

    @pytest.mark.fast
    def test_cpu_quality_presets_map_correctly(self, _make_transcoder):
        """Verify quality presets low/medium/high map to expected CRF values."""
        expected = {'low': '28', 'medium': '23', 'high': '18'}
        for quality, crf_val in expected.items():
            t = _make_transcoder(hw_accel='none', codec='h264', quality=quality)
            cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
            crf_idx = cmd.index('-crf')
            assert cmd[crf_idx + 1] == crf_val, f"Quality '{quality}' expected CRF {crf_val}"


# ============================================================================
# AC4: ProRes outputs .mov, DNxHD outputs .mxf
# ============================================================================

class TestFFmpegOutputContainerUS007:
    """AC4: Test get_ffmpeg_transcode_cmd() output extension matches codec."""

    @pytest.mark.fast
    def test_prores_outputs_mov_container(self, _make_transcoder):
        """Verify ProRes codec changes output extension to .mov."""
        t = _make_transcoder(hw_accel='none', codec='prores')
        cmd, output_path = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert output_path.endswith('.mov')

    @pytest.mark.fast
    def test_dnxhd_outputs_mxf_container(self, _make_transcoder):
        """Verify DNxHD codec changes output extension to .mxf."""
        t = _make_transcoder(hw_accel='none', codec='dnxhd')
        cmd, output_path = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert output_path.endswith('.mxf')

    @pytest.mark.fast
    def test_h264_preserves_mp4_container(self, _make_transcoder):
        """Verify h264 codec preserves original .mp4 extension."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, output_path = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert output_path == 'output.mp4'

    @pytest.mark.fast
    def test_prores_uses_prores_ks_encoder(self, _make_transcoder):
        """Verify ProRes uses prores_ks encoder with profile."""
        t = _make_transcoder(hw_accel='none', codec='prores', prores_profile='hq')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        cv_idx = cmd.index('-c:v')
        assert cmd[cv_idx + 1] == 'prores_ks'
        profile_idx = cmd.index('-profile:v')
        assert cmd[profile_idx + 1] == '3'  # 'hq' maps to '3'

    @pytest.mark.fast
    def test_prores_uses_pcm_audio(self, _make_transcoder):
        """Verify ProRes codec uses uncompressed PCM audio."""
        t = _make_transcoder(hw_accel='none', codec='prores')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        ca_idx = cmd.index('-c:a')
        assert cmd[ca_idx + 1] == 'pcm_s16le'

    @pytest.mark.fast
    def test_dnxhd_uses_dnxhr_sq_profile(self, _make_transcoder):
        """Verify DNxHD uses dnxhr_sq profile and PCM audio."""
        t = _make_transcoder(hw_accel='none', codec='dnxhd')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert 'dnxhd' in cmd
        assert 'dnxhr_sq' in cmd
        ca_idx = cmd.index('-c:a')
        assert cmd[ca_idx + 1] == 'pcm_s16le'

    @pytest.mark.fast
    def test_prores_all_profiles_map_correctly(self, _make_transcoder):
        """Verify all ProRes profiles map to correct numeric values."""
        profile_map = {'proxy': '0', 'lt': '1', 'standard': '2', 'hq': '3'}
        for profile_name, profile_num in profile_map.items():
            t = _make_transcoder(hw_accel='none', codec='prores', prores_profile=profile_name)
            cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
            profile_idx = cmd.index('-profile:v')
            assert cmd[profile_idx + 1] == profile_num, (
                f"Profile '{profile_name}' should map to '{profile_num}'"
            )


# ============================================================================
# AC5: -y overwrite flag and correct input/output paths
# ============================================================================

class TestFFmpegCommandStructureUS007:
    """AC5: Test get_ffmpeg_transcode_cmd() includes -y flag and correct paths."""

    @pytest.mark.fast
    def test_command_starts_with_ffmpeg(self, _make_transcoder):
        """Verify command starts with 'ffmpeg'."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert cmd[0] == 'ffmpeg'

    @pytest.mark.fast
    def test_command_includes_y_overwrite_flag(self, _make_transcoder):
        """Verify -y overwrite flag is present in command."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert '-y' in cmd

    @pytest.mark.fast
    def test_command_includes_input_path_after_i_flag(self, _make_transcoder):
        """Verify input path follows -i flag correctly."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('/path/to/input.webm', 'output.mp4')
        i_idx = cmd.index('-i')
        assert cmd[i_idx + 1] == '/path/to/input.webm'

    @pytest.mark.fast
    def test_command_ends_with_output_path(self, _make_transcoder):
        """Verify output path is the last element in the command."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, output_path = t.get_ffmpeg_transcode_cmd('input.webm', '/path/to/output.mp4')
        assert cmd[-1] == output_path
        assert cmd[-1] == '/path/to/output.mp4'

    @pytest.mark.fast
    def test_returned_output_path_matches_command(self, _make_transcoder):
        """Verify returned output_path matches the path in the command."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, output_path = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert cmd[-1] == output_path

    @pytest.mark.fast
    def test_command_is_valid_list_for_subprocess(self, _make_transcoder):
        """Verify command is a list of strings suitable for subprocess execution."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert isinstance(cmd, list)
        assert all(isinstance(arg, str) for arg in cmd)
        assert len(cmd) > 5  # ffmpeg + flags + input + codec + output

    @pytest.mark.fast
    def test_command_has_nostdin_flag(self, _make_transcoder):
        """Verify -nostdin prevents FFmpeg from waiting for keyboard input."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        assert '-nostdin' in cmd

    @pytest.mark.fast
    def test_y_flag_appears_before_input(self, _make_transcoder):
        """Verify -y flag appears before -i input flag in command order."""
        t = _make_transcoder(hw_accel='none', codec='h264')
        cmd, _ = t.get_ffmpeg_transcode_cmd('input.webm', 'output.mp4')
        y_idx = cmd.index('-y')
        i_idx = cmd.index('-i')
        assert y_idx < i_idx, "-y must appear before -i in FFmpeg command"
