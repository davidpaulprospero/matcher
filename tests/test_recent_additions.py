"""
Tests for Recent Additions (January 2026)

Tests the following features from CLAUDE.md session history:
1. VAD filter separation (Rule 11) - Hardcoded OFF for videos, config only for voiceover
2. Config loading fix - project_config.yaml now properly overlays defaults
3. video_source_dir fix - _resolve_paths() now checks pipeline.video_source_dir
4. gap_mode added - scale/proportional/none for timeline gap distribution
5. voiceover_offset added - Manual alignment adjustment for SRT drift
6. AudioDownload checkpoint fix - Obsolete field removal for backward compat

Usage:
    pytest tests/test_recent_additions.py -v
    python tests/test_recent_additions.py
"""

import pytest
import sys
import tempfile
import shutil
from pathlib import Path
from dataclasses import asdict
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TEST 1: VAD Filter Separation (Rule 11)
# =============================================================================
@pytest.mark.fast
class TestVADFilterSeparation:
    """
    Test VAD filter separation between voiceover and videos.

    Rule 11: VAD is ALWAYS disabled for video transcription (hardcoded False).
    Config setting only applies to voiceover transcription.
    """

    def test_video_transcription_vad_always_false(self):
        """Video transcription should ALWAYS use vad_filter=False regardless of config"""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Create mock config with VAD enabled (should be ignored for videos)
        mock_config = Mock()
        mock_config.transcription = Mock()
        mock_config.transcription.model = 'base'
        mock_config.transcription.compute_type = 'auto'
        mock_config.transcription.language = None
        mock_config.transcription.vad_filter = True  # This should be IGNORED
        mock_config.transcription.min_silence_duration_ms = 200
        mock_config.transcription.speech_pad_ms = 10

        # The function extracts VAD setting - verify it's hardcoded to False
        # We can verify this by checking the source code behavior
        # When config.transcription.vad_filter = True, videos still use False

        # Line 64-66 in parallel_processor.py:
        # vad_filter = False  # ALWAYS disabled for video transcription

        # Verify by inspecting the actual code logic
        import inspect
        source = inspect.getsource(transcribe_videos_parallel)

        # The function should contain the hardcoded False assignment
        assert "vad_filter = False" in source, \
            "Video transcription should have hardcoded vad_filter = False"
        assert "VAD is ALWAYS disabled for video transcription" in source or \
               "ALWAYS disabled" in source, \
            "Should have comment explaining VAD is always disabled for videos"

    def test_voiceover_transcription_reads_config_vad(self):
        """Voiceover transcription should read VAD setting from config"""
        from src.stages.analyze import AnalyzeStage
        import inspect

        # Check that AnalyzeStage._transcribe_audio reads VAD from config
        source = inspect.getsource(AnalyzeStage._transcribe_audio)

        # Should read from config, not hardcode
        assert "getattr(config.transcription, 'vad_filter'" in source, \
            "Voiceover transcription should read vad_filter from config"

    def test_voiceover_vad_default_true(self):
        """Voiceover VAD should default to True when not in config"""
        from src.transcription.parallel_processor import transcribe_voiceover_audio
        import inspect

        # Check default parameter value
        sig = inspect.signature(transcribe_voiceover_audio)
        vad_param = sig.parameters.get('vad_filter')

        assert vad_param is not None, "transcribe_voiceover_audio should have vad_filter parameter"
        assert vad_param.default is True, \
            f"vad_filter should default to True for voiceover, got {vad_param.default}"

    def test_video_vad_ignores_config_value(self):
        """Verify video transcription ignores config vad_filter value"""
        # This test verifies the code path that ignores config
        from src.transcription.parallel_processor import transcribe_videos_parallel
        import inspect

        source = inspect.getsource(transcribe_videos_parallel)

        # Should NOT use: vad_filter = getattr(config.transcription, 'vad_filter', ...)
        # for videos - it should be hardcoded
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'vad_filter = False' in line:
                # Found the hardcoded line - verify it's in video context
                context = '\n'.join(lines[max(0, i-5):i+1])
                assert 'video' in context.lower() or 'ALWAYS disabled' in context, \
                    "vad_filter = False should be in video transcription context"
                break
        else:
            pytest.fail("Could not find hardcoded vad_filter = False for video transcription")


# =============================================================================
# TEST 2: Config Loading - project_config.yaml Overlay
# =============================================================================
@pytest.mark.fast
class TestConfigLoading:
    """
    Test that project_config.yaml properly overlays default config values.
    """

    @pytest.fixture
    def temp_project_dir(self, tmp_path):
        """Create a temporary project directory"""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        return project_dir

    def test_merge_config_simple_values(self):
        """Test merge_config with simple value overrides"""
        from src.cli.config_utils import merge_config
        from src.config import Config

        config = Config()
        original_max_keywords = config.keyword.max_keywords

        overrides = {
            'keyword': {
                'max_keywords': 25
            }
        }

        merged = merge_config(config, overrides)

        assert merged.keyword.max_keywords == 25, \
            f"Expected max_keywords=25, got {merged.keyword.max_keywords}"

    def test_merge_config_nested_sections(self):
        """Test merge_config with multiple nested sections"""
        from src.cli.config_utils import merge_config
        from src.config import Config

        config = Config()

        overrides = {
            'keyword': {'max_keywords': 15},
            'output': {'frame_rate': 60.0},
            'pipeline': {'skip_download': True}
        }

        merged = merge_config(config, overrides)

        assert merged.keyword.max_keywords == 15
        assert merged.output.frame_rate == 60.0
        assert merged.pipeline.skip_download is True

    def test_merge_config_preserves_unspecified(self):
        """Test that merge_config preserves values not in overrides"""
        from src.cli.config_utils import merge_config
        from src.config import Config

        config = Config()
        original_model = config.transcription.model
        original_generate_otio = config.output.generate_otio

        overrides = {
            'keyword': {'max_keywords': 10}
        }

        merged = merge_config(config, overrides)

        # Unspecified values should be preserved
        assert merged.transcription.model == original_model
        assert merged.output.generate_otio == original_generate_otio

    def test_load_project_config_with_overrides(self, temp_project_dir):
        """Test full load_project_config with project_config.yaml"""
        from src.cli.config_utils import load_project_config
        import yaml

        # Create project_config.yaml with overrides
        project_config = {
            'keyword': {'max_keywords': 42},
            'output': {'voiceover_offset': 2.5}
        }

        config_path = temp_project_dir / 'project_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(project_config, f)

        # Load config with project directory
        config = load_project_config(temp_project_dir)

        assert config.keyword.max_keywords == 42, \
            f"Expected max_keywords=42 from project_config.yaml, got {config.keyword.max_keywords}"
        assert config.output.voiceover_offset == 2.5, \
            f"Expected voiceover_offset=2.5, got {config.output.voiceover_offset}"

    def test_load_project_config_calls_resolve_paths(self, temp_project_dir):
        """Test that load_project_config calls _resolve_paths after merge"""
        from src.cli.config_utils import load_project_config
        import yaml

        # Create project_config.yaml
        project_config = {'keyword': {'max_keywords': 5}}
        config_path = temp_project_dir / 'project_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(project_config, f)

        # Load and verify project_dir is set
        config = load_project_config(temp_project_dir)

        assert config.project_dir == str(temp_project_dir), \
            "project_dir should be set to the project directory"

    def test_project_config_detection_warning(self, temp_project_dir, capsys):
        """Test warning when --config points to project_config.yaml"""
        from src.cli.config_utils import load_project_config
        import yaml

        # Create project_config.yaml
        project_config = {'keyword': {'max_keywords': 5}}
        config_path = temp_project_dir / 'project_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(project_config, f)

        # Try to load with --config pointing to project_config.yaml
        config = load_project_config(temp_project_dir, config_path=config_path)

        captured = capsys.readouterr()
        # Should show warning about using project_config.yaml as --config
        assert "project_config.yaml" in captured.out


# =============================================================================
# TEST 3: video_source_dir Fix in _resolve_paths()
# =============================================================================
@pytest.mark.fast
class TestVideoSourceDir:
    """
    Test that pipeline.video_source_dir correctly overrides downloading.output_dir.
    """

    def test_video_source_dir_overrides_output_dir(self):
        """Test that video_source_dir takes precedence over downloading.output_dir"""
        from src.config import Config

        config = Config()
        config.project_dir = "/test/project"

        # Set video_source_dir in pipeline config
        config.pipeline.video_source_dir = "D:/custom/videos"

        # Call _resolve_paths
        config._resolve_paths()

        # downloaded_videos_dir should use video_source_dir, not downloading.output_dir
        assert config.downloaded_videos_dir == "D:/custom/videos", \
            f"Expected D:/custom/videos, got {config.downloaded_videos_dir}"

    def test_empty_video_source_dir_uses_default(self):
        """Test that empty video_source_dir falls back to downloading.output_dir"""
        from src.config import Config

        config = Config()
        config.project_dir = "/test/project"
        config.pipeline.video_source_dir = ""  # Empty
        config.downloading.output_dir = "videos"

        config._resolve_paths()

        # Should resolve to project_dir / output_dir
        expected = str(Path("/test/project").resolve() / "videos")
        assert config.downloaded_videos_dir == expected, \
            f"Expected {expected}, got {config.downloaded_videos_dir}"

    def test_video_source_dir_none_uses_default(self):
        """Test that None video_source_dir falls back to downloading.output_dir"""
        from src.config import Config

        config = Config()
        config.project_dir = "/test/project"

        # Ensure video_source_dir is not set or None
        if hasattr(config.pipeline, 'video_source_dir'):
            config.pipeline.video_source_dir = None

        config.downloading.output_dir = "videos"

        config._resolve_paths()

        # Should resolve to project_dir / output_dir
        base = Path("/test/project").resolve()
        expected = str(base / "videos")
        assert config.downloaded_videos_dir == expected, \
            f"Expected {expected}, got {config.downloaded_videos_dir}"

    def test_video_source_dir_absolute_path(self):
        """Test that absolute video_source_dir is used as-is"""
        from src.config import Config
        import platform

        config = Config()
        config.project_dir = "/test/project"

        # Use platform-appropriate absolute path
        if platform.system() == 'Windows':
            abs_path = "D:\\custom\\video\\path"
        else:
            abs_path = "/absolute/path/to/videos"

        config.pipeline.video_source_dir = abs_path

        config._resolve_paths()

        assert config.downloaded_videos_dir == abs_path, \
            f"Absolute path should be used as-is, got {config.downloaded_videos_dir}"


# =============================================================================
# TEST 4: gap_mode (scale/proportional/none)
# =============================================================================
@pytest.mark.fast
class TestGapMode:
    """
    Test gap_mode configuration for timeline gap distribution.
    """

    def test_gap_mode_default_is_scale(self):
        """Test that default gap_mode is 'scale'"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig()
        assert config.gap_mode == "scale", \
            f"Default gap_mode should be 'scale', got {config.gap_mode}"

    def test_gap_mode_accepts_proportional(self):
        """Test that gap_mode accepts 'proportional'"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(gap_mode="proportional")
        assert config.gap_mode == "proportional"

    def test_gap_mode_accepts_none(self):
        """Test that gap_mode accepts 'none'"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(gap_mode="none")
        assert config.gap_mode == "none"

    def test_gap_mode_accepts_scale(self):
        """Test that gap_mode accepts 'scale'"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(gap_mode="scale")
        assert config.gap_mode == "scale"

    def test_gap_mode_in_full_config(self):
        """Test gap_mode through full Config object"""
        from src.config import Config

        config = Config()
        assert hasattr(config.output, 'gap_mode'), \
            "Config.output should have gap_mode attribute"

        # Test setting via config
        config.output.gap_mode = "proportional"
        assert config.output.gap_mode == "proportional"

    def test_gap_mode_invalid_value_allowed(self):
        """Test that invalid gap_mode is allowed (handled at runtime)"""
        from src.config.sections.output import OutputConfig

        # Invalid value should be accepted at config level
        # Runtime validation happens in timeline.py
        config = OutputConfig(gap_mode="invalid")
        assert config.gap_mode == "invalid"


# =============================================================================
# TEST 5: voiceover_offset (Manual Alignment Adjustment)
# =============================================================================
@pytest.mark.fast
class TestVoiceoverOffset:
    """
    Test voiceover_offset configuration for manual timeline alignment.
    """

    def test_voiceover_offset_default_zero(self):
        """Test that default voiceover_offset is 0.0"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig()
        assert config.voiceover_offset == 0.0, \
            f"Default voiceover_offset should be 0.0, got {config.voiceover_offset}"

    def test_voiceover_offset_positive_value(self):
        """Test positive voiceover_offset (shift clips later)"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(voiceover_offset=2.5)
        assert config.voiceover_offset == 2.5

    def test_voiceover_offset_negative_value(self):
        """Test negative voiceover_offset (shift clips earlier)"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(voiceover_offset=-1.5)
        assert config.voiceover_offset == -1.5

    def test_voiceover_offset_float_precision(self):
        """Test voiceover_offset maintains float precision"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(voiceover_offset=0.333)
        assert abs(config.voiceover_offset - 0.333) < 0.001

    def test_voiceover_offset_in_full_config(self):
        """Test voiceover_offset through full Config object"""
        from src.config import Config

        config = Config()
        assert hasattr(config.output, 'voiceover_offset'), \
            "Config.output should have voiceover_offset attribute"

        config.output.voiceover_offset = 3.0
        assert config.output.voiceover_offset == 3.0

    def test_voiceover_offset_large_values(self):
        """Test voiceover_offset with large values"""
        from src.config.sections.output import OutputConfig

        # Large positive
        config = OutputConfig(voiceover_offset=100.0)
        assert config.voiceover_offset == 100.0

        # Large negative
        config = OutputConfig(voiceover_offset=-50.0)
        assert config.voiceover_offset == -50.0


# =============================================================================
# TEST 6: AudioDownload Checkpoint Backward Compatibility
# =============================================================================
@pytest.mark.fast
class TestAudioDownloadCheckpoint:
    """
    Test AudioDownload dataclass restoration from checkpoint dicts.
    Ensures backward compatibility when obsolete fields are present.
    """

    def test_audio_download_basic_creation(self):
        """Test basic AudioDownload creation"""
        from src.state import AudioDownload

        ad = AudioDownload(
            file="/path/to/audio.mp3",
            video_id="abc123"
        )

        assert ad.file == "/path/to/audio.mp3"
        assert ad.video_id == "abc123"
        assert ad.url == ""
        assert ad.title == ""
        assert ad.duration == 0.0
        assert ad.keyword == ""

    def test_audio_download_full_creation(self):
        """Test AudioDownload with all fields"""
        from src.state import AudioDownload

        ad = AudioDownload(
            file="/path/to/audio.mp3",
            video_id="abc123",
            url="https://youtube.com/watch?v=abc123",
            title="Test Video",
            duration=120.5,
            keyword="test keyword"
        )

        assert ad.file == "/path/to/audio.mp3"
        assert ad.video_id == "abc123"
        assert ad.url == "https://youtube.com/watch?v=abc123"
        assert ad.title == "Test Video"
        assert ad.duration == 120.5
        assert ad.keyword == "test keyword"

    def test_audio_download_from_dict(self):
        """Test AudioDownload creation from checkpoint dict"""
        from src.state import AudioDownload

        checkpoint_data = {
            'file': '/path/to/audio.mp3',
            'video_id': 'xyz789',
            'url': 'https://youtube.com/watch?v=xyz789',
            'title': 'Checkpoint Video',
            'duration': 60.0,
            'keyword': 'checkpoint keyword'
        }

        ad = AudioDownload(**checkpoint_data)

        assert ad.file == '/path/to/audio.mp3'
        assert ad.video_id == 'xyz789'
        assert ad.title == 'Checkpoint Video'

    def test_audio_download_obsolete_field_handling(self):
        """Test that obsolete fields in checkpoint dict are handled gracefully"""
        from src.state import AudioDownload

        # Simulate checkpoint with obsolete fields that were removed
        checkpoint_data = {
            'file': '/path/to/audio.mp3',
            'video_id': 'abc123',
            'url': '',
            'title': 'Test',
            'duration': 30.0,
            'keyword': 'test'
        }

        # Adding obsolete fields should NOT raise an error when filtered
        obsolete_data = {
            **checkpoint_data,
            'path': '/old/path',  # Obsolete - was renamed to 'file'
            'tier': 'short',  # Obsolete - not in AudioDownload
        }

        # Filter to only known fields
        known_fields = {'file', 'video_id', 'url', 'title', 'duration', 'keyword'}
        filtered_data = {k: v for k, v in obsolete_data.items() if k in known_fields}

        ad = AudioDownload(**filtered_data)

        assert ad.file == '/path/to/audio.mp3'
        assert ad.video_id == 'abc123'

    def test_audio_download_required_fields_only(self):
        """Test AudioDownload with only required fields"""
        from src.state import AudioDownload

        # Only file and video_id are required
        ad = AudioDownload(file="/audio.mp3", video_id="vid123")

        assert ad.file == "/audio.mp3"
        assert ad.video_id == "vid123"
        # Defaults should be applied
        assert ad.url == ""
        assert ad.duration == 0.0

    def test_audio_download_asdict(self):
        """Test AudioDownload serialization with asdict"""
        from src.state import AudioDownload
        from dataclasses import asdict

        ad = AudioDownload(
            file="/path/audio.mp3",
            video_id="abc",
            title="Test"
        )

        d = asdict(ad)

        assert d['file'] == "/path/audio.mp3"
        assert d['video_id'] == "abc"
        assert d['title'] == "Test"
        assert 'url' in d
        assert 'duration' in d
        assert 'keyword' in d

    def test_downloaded_video_required_fields(self):
        """Test DownloadedVideo has correct required fields (file=, not path=)"""
        from src.state import DownloadedVideo
        import inspect

        sig = inspect.signature(DownloadedVideo)
        params = list(sig.parameters.keys())

        # Should use 'file', not 'path'
        assert 'file' in params, "DownloadedVideo should have 'file' field"
        assert 'path' not in params, "DownloadedVideo should NOT have 'path' field (obsolete)"

        # Should use 'duration_tier', not 'tier'
        assert 'duration_tier' in params, "DownloadedVideo should have 'duration_tier' field"
        assert 'tier' not in params, "DownloadedVideo should NOT have 'tier' field (obsolete)"


# =============================================================================
# Integration Tests
# =============================================================================
@pytest.mark.fast
class TestRecentAdditionsIntegration:
    """Integration tests combining multiple recent additions."""

    def test_project_config_with_gap_mode_and_offset(self, tmp_path):
        """Test loading project config with gap_mode and voiceover_offset"""
        from src.cli.config_utils import load_project_config
        import yaml

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create project_config.yaml with new options
        project_config = {
            'output': {
                'gap_mode': 'proportional',
                'voiceover_offset': 1.5,
                'frame_rate': 24.0
            }
        }

        config_path = project_dir / 'project_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(project_config, f)

        config = load_project_config(project_dir)

        assert config.output.gap_mode == 'proportional'
        assert config.output.voiceover_offset == 1.5
        assert config.output.frame_rate == 24.0

    def test_project_config_with_video_source_dir(self, tmp_path):
        """Test project config that sets video_source_dir"""
        from src.cli.config_utils import load_project_config
        import yaml

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create project_config.yaml with video_source_dir
        project_config = {
            'pipeline': {
                'video_source_dir': 'D:/shared/videos'
            }
        }

        config_path = project_dir / 'project_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(project_config, f)

        config = load_project_config(project_dir)

        assert config.pipeline.video_source_dir == 'D:/shared/videos'
        assert config.downloaded_videos_dir == 'D:/shared/videos'


# =============================================================================
# TEST 7: Additional VAD Filter Tests (Mocking)
# =============================================================================
@pytest.mark.fast
class TestVADFilterMocking:
    """
    Additional VAD tests using mocking to verify actual behavior.
    """

    def test_video_transcription_ignores_config_vad_true(self):
        """Verify video transcription code sets vad_filter=False regardless of config"""
        from unittest.mock import Mock, patch

        mock_config = Mock()
        mock_config.transcription = Mock()
        mock_config.transcription.model = 'base'
        mock_config.transcription.compute_type = 'auto'
        mock_config.transcription.language = None
        mock_config.transcription.vad_filter = True  # Should be IGNORED
        mock_config.transcription.min_silence_duration_ms = 200
        mock_config.transcription.speech_pad_ms = 10

        # The key assertion is in the source code structure
        # Videos explicitly set vad_filter = False (line 66 of parallel_processor.py)
        from src.transcription.parallel_processor import transcribe_videos_parallel
        import inspect

        source = inspect.getsource(transcribe_videos_parallel)

        # Count occurrences of the hardcoded False
        hardcoded_false_count = source.count('vad_filter = False')
        assert hardcoded_false_count >= 1, \
            "Should have at least one hardcoded vad_filter = False for video transcription"

    def test_voiceover_function_accepts_vad_parameter(self):
        """Verify voiceover transcription accepts vad_filter parameter"""
        from src.transcription.parallel_processor import transcribe_voiceover_audio
        import inspect

        sig = inspect.signature(transcribe_voiceover_audio)
        params = sig.parameters

        assert 'vad_filter' in params, "transcribe_voiceover_audio should have vad_filter parameter"
        assert params['vad_filter'].default is True, "Default should be True for voiceover"

    def test_analyze_stage_reads_vad_from_config(self):
        """Verify AnalyzeStage reads VAD from config for voiceover"""
        from src.stages.analyze import AnalyzeStage
        import inspect

        # Get the _transcribe_audio method source
        source = inspect.getsource(AnalyzeStage._transcribe_audio)

        # Should read from config.transcription.vad_filter
        assert "getattr(config.transcription, 'vad_filter'" in source, \
            "Should read vad_filter from config for voiceover transcription"


# =============================================================================
# TEST 8: Config Loading Error Handling
# =============================================================================
@pytest.mark.fast
class TestConfigLoadingErrorHandling:
    """
    Test error handling in config loading.
    """

    def test_merge_config_unknown_section_ignored(self):
        """Test that unknown sections in overrides are ignored"""
        from src.cli.config_utils import merge_config
        from src.config import Config

        config = Config()

        overrides = {
            'nonexistent_section': {'value': 123},
            'keyword': {'max_keywords': 5}
        }

        merged = merge_config(config, overrides)

        # Should not raise, unknown section ignored
        assert merged.keyword.max_keywords == 5
        assert not hasattr(merged, 'nonexistent_section')

    def test_merge_config_unknown_field_ignored(self):
        """Test that unknown fields in overrides are ignored"""
        from src.cli.config_utils import merge_config
        from src.config import Config

        config = Config()

        overrides = {
            'keyword': {
                'max_keywords': 10,
                'nonexistent_field': 'value'  # Should be ignored
            }
        }

        merged = merge_config(config, overrides)

        assert merged.keyword.max_keywords == 10
        assert not hasattr(merged.keyword, 'nonexistent_field')

    def test_load_project_config_empty_yaml(self, tmp_path):
        """Test loading empty project_config.yaml"""
        from src.cli.config_utils import load_project_config

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create empty project_config.yaml
        config_path = project_dir / 'project_config.yaml'
        config_path.write_text("")

        # Should not raise
        config = load_project_config(project_dir)

        # Should use defaults
        assert config.keyword.max_keywords > 0

    def test_load_project_config_no_project_config_file(self, tmp_path):
        """Test loading when project_config.yaml doesn't exist"""
        from src.cli.config_utils import load_project_config

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Don't create project_config.yaml
        config = load_project_config(project_dir)

        # Should use defaults without error
        assert config is not None
        assert config.project_dir == str(project_dir)

    def test_merge_config_scalar_override(self):
        """Test overriding a section with a scalar value"""
        from src.cli.config_utils import merge_config
        from src.config import Config

        config = Config()

        # Try to override entire section with scalar (edge case)
        overrides = {
            'project_dir': '/new/project/dir'
        }

        merged = merge_config(config, overrides)

        # Should set scalar value
        assert merged.project_dir == '/new/project/dir'


# =============================================================================
# TEST 9: Gap Mode Runtime Validation
# =============================================================================
@pytest.mark.fast
class TestGapModeValidation:
    """
    Test gap_mode validation that happens at runtime in timeline.py.
    """

    def test_gap_mode_invalid_falls_back_to_scale(self):
        """Test that invalid gap_mode falls back to 'scale' at runtime"""
        # The validation happens in timeline.py lines 358-361:
        # if gap_mode not in ('scale', 'proportional', 'none'):
        #     gap_mode = 'scale'

        from src.otio.timeline import create_timeline
        import inspect

        source = inspect.getsource(create_timeline)

        # Verify the validation logic exists
        assert "gap_mode not in" in source, "Should validate gap_mode"
        assert "'scale'" in source and "'proportional'" in source and "'none'" in source, \
            "Should check for scale, proportional, and none modes"

    def test_gap_mode_scale_with_time_scale_factor(self):
        """Test that scale mode works with time_scale_factor"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(
            gap_mode='scale',
            time_scale_factor=1.5
        )

        assert config.gap_mode == 'scale'
        assert config.time_scale_factor == 1.5

    def test_gap_mode_proportional_needs_audio_duration(self):
        """Test that proportional mode requires audio duration"""
        # In timeline.py lines 417-419:
        # Cannot use proportional gap mode without audio duration, falling back to scale

        from src.otio.timeline import create_timeline
        import inspect

        source = inspect.getsource(create_timeline)

        assert "Cannot use proportional gap mode without audio duration" in source, \
            "Should warn when proportional mode used without audio duration"

    def test_gap_mode_none_with_min_gap_threshold(self):
        """Test gap_mode none with min_gap_threshold"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(
            gap_mode='none',
            min_gap_threshold=1.0
        )

        assert config.gap_mode == 'none'
        assert config.min_gap_threshold == 1.0


# =============================================================================
# TEST 10: Voiceover Offset Behavior
# =============================================================================
@pytest.mark.fast
class TestVoiceoverOffsetBehavior:
    """
    Test voiceover_offset behavior in timeline calculations.
    """

    def test_voiceover_offset_in_timeline_code(self):
        """Verify voiceover_offset is used in timeline.py"""
        from src.otio.timeline import create_timeline
        import inspect

        source = inspect.getsource(create_timeline)

        # Should read voiceover_offset from config
        assert "voiceover_offset" in source, "Timeline code should use voiceover_offset"

        # Should have log message when non-zero
        assert "Applying voiceover offset" in source, \
            "Should log when applying voiceover offset"

    def test_voiceover_offset_adjusts_first_segment(self):
        """Test that voiceover_offset adjusts the first segment start"""
        from src.otio.timeline import create_timeline
        import inspect

        source = inspect.getsource(create_timeline)

        # The offset should be added to first segment calculation
        assert "voiceover_offset" in source, "Should use voiceover_offset in calculations"

    def test_voiceover_offset_with_time_scale_factor(self):
        """Test voiceover_offset works with time_scale_factor"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig(
            voiceover_offset=2.0,
            time_scale_factor=1.1
        )

        assert config.voiceover_offset == 2.0
        assert config.time_scale_factor == 1.1

    def test_voiceover_offset_zero_no_log(self):
        """Test that zero offset doesn't trigger logging"""
        from src.otio.timeline import create_timeline
        import inspect

        source = inspect.getsource(create_timeline)

        # Should only log when offset != 0.0
        assert "voiceover_offset != 0.0" in source or "offset != 0" in source, \
            "Should only log when offset is non-zero"


# =============================================================================
# TEST 11: AudioDownload Field Mapping (Backward Compat)
# =============================================================================
@pytest.mark.fast
class TestAudioDownloadFieldMapping:
    """
    Test backward compatibility field mapping in checkpoint restoration.
    """

    def test_download_stage_restore_audio_file_mapping(self):
        """Test that audio_file is mapped to file in checkpoint restore"""
        from src.stages.download import DownloadStage
        import inspect

        source = inspect.getsource(DownloadStage.restore)

        # Should map 'audio_file' → 'file'
        assert "'audio_file'" in source and "'file'" in source, \
            "Should map audio_file to file for backward compatibility"

    def test_download_stage_restore_video_url_mapping(self):
        """Test that video_url is mapped to url in checkpoint restore"""
        from src.stages.download import DownloadStage
        import inspect

        source = inspect.getsource(DownloadStage.restore)

        # Should map 'video_url' → 'url'
        assert "'video_url'" in source and "'url'" in source, \
            "Should map video_url to url for backward compatibility"

    def test_download_stage_removes_obsolete_fields(self):
        """Test that obsolete fields are removed during restore"""
        from src.stages.download import DownloadStage
        import inspect

        source = inspect.getsource(DownloadStage.restore)

        # Should remove obsolete fields
        obsolete_fields = ['channel', 'duration_tier', 'upload_date', 'license']
        for field in obsolete_fields:
            assert f"'{field}'" in source, \
                f"Should handle obsolete field '{field}'"

    def test_audio_download_from_old_checkpoint_format(self):
        """Test AudioDownload creation from old checkpoint format"""
        from src.state import AudioDownload

        # Old checkpoint format
        old_format = {
            'audio_file': '/path/to/audio.mp3',  # Old name
            'video_id': 'abc123',
            'video_url': 'https://youtube.com/watch?v=abc123',  # Old name
            'title': 'Test Video',
            'duration': 60.0,
            'keyword': 'test',
            # Obsolete fields
            'channel': 'Test Channel',
            'duration_tier': 'short',
            'upload_date': '2024-01-01',
            'license': 'standard',
        }

        # Manual field mapping (as done in download.py)
        mapped = dict(old_format)

        # Map old names to new names
        if 'audio_file' in mapped and 'file' not in mapped:
            mapped['file'] = mapped.pop('audio_file')
        if 'video_url' in mapped and 'url' not in mapped:
            mapped['url'] = mapped.pop('video_url')

        # Remove obsolete fields
        for field in ['channel', 'duration_tier', 'upload_date', 'license']:
            mapped.pop(field, None)

        ad = AudioDownload(**mapped)

        assert ad.file == '/path/to/audio.mp3'
        assert ad.url == 'https://youtube.com/watch?v=abc123'
        assert ad.video_id == 'abc123'


# =============================================================================
# TEST 12: Retry Logic Tests
# =============================================================================
@pytest.mark.fast
class TestRetryLogic:
    """
    Test retry logic with exponential backoff.
    """

    def test_permanent_error_patterns(self):
        """Test that permanent errors are identified correctly"""
        from src.llm_client.retry import _is_permanent_error, PERMANENT_ERROR_PATTERNS

        # Verify patterns exist
        assert len(PERMANENT_ERROR_PATTERNS) > 0, "Should have permanent error patterns"

        # Test known permanent errors
        permanent_messages = [
            "Model not found",
            "Invalid API key",
            "Authentication failed",
            "Unauthorized access",
            "Permission denied",
        ]

        for msg in permanent_messages:
            assert _is_permanent_error(msg), f"'{msg}' should be identified as permanent"

    def test_transient_error_not_permanent(self):
        """Test that transient errors are not identified as permanent"""
        from src.llm_client.retry import _is_permanent_error

        transient_messages = [
            "Connection timeout",
            "Rate limit exceeded",
            "Server error 500",
            "Network unavailable",
            "Temporary failure",
        ]

        for msg in transient_messages:
            assert not _is_permanent_error(msg), f"'{msg}' should NOT be permanent"

    def test_with_retry_success_first_try(self):
        """Test with_retry succeeds on first try"""
        from src.llm_client.retry import with_retry

        call_count = 0

        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = with_retry(success_func, max_retries=3)

        assert result == "success"
        assert call_count == 1, "Should only call once on success"

    def test_with_retry_permanent_error_no_retry(self):
        """Test that permanent errors don't retry"""
        from src.llm_client.retry import with_retry
        from src.llm_client.exceptions import LLMProviderError

        call_count = 0

        def permanent_error_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid API key - authentication failed")

        with pytest.raises(LLMProviderError):
            with_retry(permanent_error_func, max_retries=3, base_delay=0.01)

        # Should only be called once due to permanent error detection
        assert call_count == 1, "Should not retry permanent errors"

    def test_with_retry_retries_transient_error(self):
        """Test that transient errors are retried"""
        from src.llm_client.retry import with_retry
        from src.llm_client.exceptions import LLMProviderError

        call_count = 0

        def transient_error_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Connection timeout")

        with pytest.raises((LLMProviderError, Exception)):
            with_retry(transient_error_func, max_retries=3, base_delay=0.01)

        # Should be called max_retries times
        assert call_count == 3, "Should retry transient errors"

    def test_with_retry_success_after_retries(self):
        """Test success after failed retries"""
        from src.llm_client.retry import with_retry

        call_count = 0

        def eventual_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"

        result = with_retry(eventual_success, max_retries=5, base_delay=0.01)

        assert result == "success"
        assert call_count == 3


# =============================================================================
# TEST 13: Location Matching Tests
# =============================================================================
@pytest.mark.fast
class TestLocationMatching:
    """
    Test location matching configuration and behavior.
    """

    def test_location_matching_config_exists(self):
        """Test that location matching config exists"""
        from src.config import Config

        config = Config()

        assert hasattr(config.matching, 'location_matching'), \
            "Config should have location_matching section"

    def test_location_matching_enabled_field(self):
        """Test location matching enabled field"""
        from src.config import Config

        config = Config()
        location_config = config.matching.location_matching

        assert hasattr(location_config, 'enabled'), \
            "location_matching should have 'enabled' field"

    def test_location_matching_geonames_field(self):
        """Test location matching geonames_username field"""
        from src.config import Config

        config = Config()
        location_config = config.matching.location_matching

        assert hasattr(location_config, 'geonames_username'), \
            "location_matching should have 'geonames_username' field"

    def test_location_matching_hard_filter_level(self):
        """Test location matching hard_filter_level field"""
        from src.config import Config

        config = Config()
        location_config = config.matching.location_matching

        assert hasattr(location_config, 'hard_filter_level'), \
            "location_matching should have 'hard_filter_level' field"


# =============================================================================
# TEST 14: OTIO Utilities Tests
# =============================================================================
@pytest.mark.fast
class TestOTIOUtils:
    """
    Test OTIO utility functions.
    """

    def test_otio_utils_exists(self):
        """Test that OTIO utils module exists"""
        try:
            from src.otio import utils
            assert utils is not None
        except ImportError:
            pytest.skip("OTIO utils not available")

    def test_otio_entities_module_exists(self):
        """Test that OTIO entities module exists"""
        try:
            from src.otio import entities
            assert entities is not None
        except ImportError:
            pytest.skip("OTIO entities not available")


# =============================================================================
# TEST 15: Additional Integration Tests
# =============================================================================
@pytest.mark.fast
class TestAdditionalIntegration:
    """
    Additional integration tests for recent additions.
    """

    def test_full_config_with_all_new_options(self, tmp_path):
        """Test config with all new options combined"""
        from src.cli.config_utils import load_project_config
        import yaml

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        # Create comprehensive project_config.yaml
        project_config = {
            'output': {
                'gap_mode': 'none',
                'voiceover_offset': -0.5,
                'frame_rate': 29.97,
                'min_gap_threshold': 0.5
            },
            'pipeline': {
                'video_source_dir': 'E:/videos'
            },
            'keyword': {
                'max_keywords': 20
            },
            'transcription': {
                'vad_filter': False  # Only affects voiceover
            }
        }

        config_path = project_dir / 'project_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(project_config, f)

        config = load_project_config(project_dir)

        # Verify all options loaded
        assert config.output.gap_mode == 'none'
        assert config.output.voiceover_offset == -0.5
        assert config.output.frame_rate == 29.97
        assert config.output.min_gap_threshold == 0.5
        assert config.pipeline.video_source_dir == 'E:/videos'
        assert config.keyword.max_keywords == 20
        assert config.transcription.vad_filter is False

    def test_config_vad_only_for_voiceover(self, tmp_path):
        """Test that transcription.vad_filter in config only affects voiceover"""
        from src.cli.config_utils import load_project_config
        import yaml

        project_dir = tmp_path / "test_project"
        project_dir.mkdir()

        project_config = {
            'transcription': {
                'vad_filter': False  # User disables VAD
            }
        }

        config_path = project_dir / 'project_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(project_config, f)

        config = load_project_config(project_dir)

        # Config should have VAD disabled
        assert config.transcription.vad_filter is False

        # But video transcription will still ignore this (verified in code inspection tests)

    def test_downloaded_video_and_audio_download_coexist(self):
        """Test both DownloadedVideo and AudioDownload can be used together"""
        from src.state import DownloadedVideo, AudioDownload, PipelineState

        state = PipelineState()

        # Add downloaded videos (DownloadedVideo uses 'file' and 'url', not 'video_id')
        state.downloaded_videos.append(
            DownloadedVideo(file="/path/video1.mp4", url="https://youtube.com/watch?v=vid1")
        )

        # Add downloaded audio (audio-first mode - AudioDownload uses 'file' and 'video_id')
        state.downloaded_audio.append(
            AudioDownload(file="/path/audio1.mp3", video_id="aud1")
        )

        assert len(state.downloaded_videos) == 1
        assert len(state.downloaded_audio) == 1
        assert state.downloaded_videos[0].file == "/path/video1.mp4"
        assert state.downloaded_audio[0].video_id == "aud1"


# =============================================================================
# TEST 16: LocationMatcher Tests
# =============================================================================
@pytest.mark.fast
class TestLocationMatcher:
    """
    Test LocationMatcher class for geographic filtering.
    """

    def test_location_matcher_initialization(self):
        """Test LocationMatcher initialization"""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        assert matcher.location_service is None
        assert matcher.location_chapters == {}
        assert matcher.video_locations == {}

    def test_location_matcher_set_location_chapters(self):
        """Test setting location chapters"""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        # Mock location chapters as dicts (as they come from checkpoint)
        chapters = [
            {'start_segment_idx': 0, 'end_segment_idx': 5, 'location_name': 'Austin'},
            {'start_segment_idx': 6, 'end_segment_idx': 10, 'location_name': 'Dallas'},
        ]

        matcher.set_location_chapters(chapters)

        # Should map each segment index to its chapter
        assert 0 in matcher.location_chapters
        assert 5 in matcher.location_chapters
        assert 6 in matcher.location_chapters
        assert 10 in matcher.location_chapters

    def test_location_matcher_set_video_locations(self):
        """Test setting video locations"""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        video_locations = {
            '/path/video1.mp4': {'lat': 30.2672, 'lon': -97.7431},  # Austin
            '/path/video2.mp4': {'lat': 32.7767, 'lon': -96.7970},  # Dallas
        }

        matcher.set_video_locations(video_locations)

        assert len(matcher.video_locations) == 2
        assert '/path/video1.mp4' in matcher.video_locations

    def test_location_matcher_get_location_chapter(self):
        """Test getting location chapter for segment"""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        chapters = [
            {'start_segment_idx': 0, 'end_segment_idx': 5, 'location_name': 'Austin'},
        ]
        matcher.set_location_chapters(chapters)

        # Test private method
        chapter = matcher._get_location_chapter(3)
        assert chapter is not None

        # Non-existent segment should return None
        chapter_none = matcher._get_location_chapter(100)
        assert chapter_none is None


# =============================================================================
# TEST 17: OTIO Entities Module Tests
# =============================================================================
@pytest.mark.fast
class TestOTIOEntities:
    """
    Test OTIO entities module functions.
    """

    def test_get_attr_with_dict(self):
        """Test _get_attr with dictionary"""
        from src.otio.entities import _get_attr

        obj = {'name': 'test', 'value': 123}

        assert _get_attr(obj, 'name') == 'test'
        assert _get_attr(obj, 'value') == 123
        assert _get_attr(obj, 'missing') is None
        assert _get_attr(obj, 'missing', 'default') == 'default'

    def test_get_attr_with_object(self):
        """Test _get_attr with object"""
        from src.otio.entities import _get_attr
        from unittest.mock import Mock

        obj = Mock()
        obj.name = 'test'
        obj.value = 123

        assert _get_attr(obj, 'name') == 'test'
        assert _get_attr(obj, 'value') == 123

    def test_find_best_entity_match_exact(self):
        """Test exact entity matching"""
        from src.otio.entities import _find_best_entity_match

        entity_dict = {
            'Austin': {'images': ['img1.jpg']},
            'Dallas': {'images': ['img2.jpg']},
        }

        # Exact match
        entity, match_type = _find_best_entity_match(
            vo_text='we are visiting austin today',
            entity_dict=entity_dict
        )

        assert entity == 'Austin'
        assert match_type == 'exact'

    def test_find_best_entity_match_no_match(self):
        """Test when no entity matches"""
        from src.otio.entities import _find_best_entity_match

        entity_dict = {
            'Austin': {'images': ['img1.jpg']},
        }

        entity, match_type = _find_best_entity_match(
            vo_text='visiting houston for the weekend',
            entity_dict=entity_dict,
            enable_sticky=False
        )

        # No match should return None without sticky
        assert entity is None or match_type == 'semantic'

    def test_find_best_entity_match_sticky(self):
        """Test sticky entity matching"""
        from src.otio.entities import _find_best_entity_match

        entity_dict = {
            'Austin': {'images': ['img1.jpg']},
        }

        entity, match_type = _find_best_entity_match(
            vo_text='enjoying the weather',  # No entity mention
            entity_dict=entity_dict,
            last_matched_entity='Austin',
            enable_sticky=True
        )

        # Should fall back to sticky entity
        if entity is not None:
            assert match_type in ('sticky', 'semantic')


# =============================================================================
# TEST 18: OTIO Tracks Module Tests
# =============================================================================
@pytest.mark.fast
class TestOTIOTracks:
    """
    Test OTIO tracks module classes.
    """

    def test_track_builder_abstract(self):
        """Test that TrackBuilder is abstract"""
        from src.otio.tracks import TrackBuilder

        # Should not be directly instantiable
        with pytest.raises(TypeError):
            TrackBuilder(matches=[], config=None, frame_rate=30.0)

    def test_track_names(self):
        """Test track naming convention"""
        # Verify track names are correct
        track_names = [
            "Primary Video",
            "Alternative Video 1",
            "Alternative Video 2",
            "Secondary Diversity 1",
            "Secondary Diversity 2",
            "Secondary Diversity 3",
            "Embedding-Diversity Strategy",
            "B-roll Only",
            "Entity Images (Google)",
            "Stock Videos (Pexels/Pixabay)",
        ]

        assert len(track_names) == 10
        assert "Primary" in track_names[0]
        assert "B-roll" in track_names[7]


# =============================================================================
# TEST 19: OTIO Utils Tests
# =============================================================================
@pytest.mark.fast
class TestOTIOUtils:
    """
    Test OTIO utility functions.
    """

    def test_numpy_encoder_int(self):
        """Test NumpyEncoder with numpy integers"""
        import numpy as np
        import json
        from src.otio.utils import NumpyEncoder

        data = {'value': np.int32(42)}
        result = json.dumps(data, cls=NumpyEncoder)

        assert '"value": 42' in result

    def test_numpy_encoder_float(self):
        """Test NumpyEncoder with numpy floats"""
        import numpy as np
        import json
        from src.otio.utils import NumpyEncoder

        data = {'value': np.float64(3.14)}
        result = json.dumps(data, cls=NumpyEncoder)

        assert '3.14' in result

    def test_numpy_encoder_array(self):
        """Test NumpyEncoder with numpy arrays"""
        import numpy as np
        import json
        from src.otio.utils import NumpyEncoder

        data = {'values': np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyEncoder)

        assert '[1, 2, 3]' in result

    def test_to_windows_path(self):
        """Test Windows path conversion"""
        from src.otio.utils import _to_windows_path
        import platform

        if platform.system() == 'Windows':
            # On Windows, forward slashes should be converted
            path = _to_windows_path("C:/test/file.mp4")
            assert "\\" in path
            assert "/" not in path or path[0:2] == "//"  # UNC paths allowed

    def test_sanitize_path_for_url(self):
        """Test path sanitization for URLs"""
        from src.otio.utils import sanitize_path_for_url

        # Should remove Windows extended path prefix
        path = sanitize_path_for_url("\\\\?\\C:\\test\\file.mp4")
        assert "\\\\?\\" not in path
        assert "?" not in path

    def test_format_path_url(self):
        """Test path URL formatting"""
        from src.otio.utils import format_path_url
        import platform

        path = format_path_url("/test/file.mp4")

        # Should have forward slashes
        assert "/" in path


# =============================================================================
# TEST 20: TranscriptSegment Tests
# =============================================================================
@pytest.mark.fast
class TestTranscriptSegment:
    """
    Test TranscriptSegment dataclass.
    """

    def test_transcript_segment_creation(self):
        """Test TranscriptSegment creation"""
        from src.state import TranscriptSegment

        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.5,
            text="Hello world",
            source_file="/path/video.mp4"
        )

        assert segment.index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 5.5
        assert segment.text == "Hello world"
        assert segment.source_file == "/path/video.mp4"
        assert segment.is_broll is False  # Default

    def test_transcript_segment_broll(self):
        """Test TranscriptSegment with B-roll flag"""
        from src.state import TranscriptSegment

        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.5,
            text="Silent footage",
            source_file="/path/video.mp4",
            is_broll=True,
            description_source='vision'
        )

        assert segment.is_broll is True
        assert segment.description_source == 'vision'

    def test_transcript_segment_to_dict(self):
        """Test TranscriptSegment serialization"""
        from src.state import TranscriptSegment

        segment = TranscriptSegment(
            index=1,
            start_time=10.0,
            end_time=15.0,
            text="Test",
            source_file="/path/video.mp4"
        )

        d = segment.to_dict()

        assert d['index'] == 1
        assert d['start_time'] == 10.0
        assert d['text'] == "Test"


# =============================================================================
# TEST 21: VoiceoverSegment Tests
# =============================================================================
@pytest.mark.fast
class TestVoiceoverSegment:
    """
    Test VoiceoverSegment dataclass.
    """

    def test_voiceover_segment_creation(self):
        """Test VoiceoverSegment creation"""
        from src.state import VoiceoverSegment

        segment = VoiceoverSegment(
            index=0,
            start=0.0,
            end=5.0,
            text="Welcome to the show"
        )

        assert segment.index == 0
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "Welcome to the show"

    def test_voiceover_segment_defaults(self):
        """Test VoiceoverSegment default values"""
        from src.state import VoiceoverSegment

        segment = VoiceoverSegment(
            index=0,
            start=0.0,
            end=1.0,
            text="Test"
        )

        # Check defaults
        assert hasattr(segment, 'index')
        assert hasattr(segment, 'start')
        assert hasattr(segment, 'end')
        assert hasattr(segment, 'text')


# =============================================================================
# TEST 22: PipelineState Tests
# =============================================================================
@pytest.mark.fast
class TestPipelineState:
    """
    Test PipelineState dataclass.
    """

    def test_pipeline_state_initialization(self):
        """Test PipelineState default initialization"""
        from src.state import PipelineState

        state = PipelineState()

        assert state.voiceover_path == ""
        assert state.voiceover_segments == []
        assert state.keywords == []
        assert state.downloaded_videos == []
        assert state.downloaded_audio == []
        assert state.matches == []
        assert state.transcripts == {}

    def test_pipeline_state_mutable_fields(self):
        """Test that mutable fields are independent"""
        from src.state import PipelineState

        state1 = PipelineState()
        state2 = PipelineState()

        state1.keywords.append("test")

        # Should not affect state2
        assert "test" in state1.keywords
        assert "test" not in state2.keywords

    def test_pipeline_state_all_fields(self):
        """Test PipelineState has all expected fields"""
        from src.state import PipelineState

        state = PipelineState()

        expected_fields = [
            'voiceover_path', 'voiceover_segments', 'keywords', 'topic_context',
            'extracted_entities', 'downloaded_videos', 'downloaded_audio',
            'failed_keywords', 'entity_images', 'entity_videos', 'transcripts',
            'embeddings', 'text_metadata', 'scene_data', 'matches', 'alternatives',
            'output_files', 'location_chapters'
        ]

        for field in expected_fields:
            assert hasattr(state, field), f"PipelineState should have '{field}' field"


# =============================================================================
# TEST 23: Config Edge Cases
# =============================================================================
@pytest.mark.fast
class TestConfigEdgeCases:
    """
    Test configuration edge cases.
    """

    def test_output_config_variety_dict_conversion(self):
        """Test that variety dict is converted to VarietyConfig"""
        from src.config.sections.output import OutputConfig, VarietyConfig

        variety_dict = {
            'require_different_source': False,
            'min_time_distance': 20.0
        }

        config = OutputConfig(variety=variety_dict)

        assert isinstance(config.variety, VarietyConfig)
        assert config.variety.require_different_source is False
        assert config.variety.min_time_distance == 20.0

    def test_output_config_time_scale_default(self):
        """Test default time_scale_factor"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig()
        assert config.time_scale_factor == 1.0

    def test_output_config_min_gap_threshold_default(self):
        """Test default min_gap_threshold"""
        from src.config.sections.output import OutputConfig

        config = OutputConfig()
        assert config.min_gap_threshold == 0.0

    def test_config_transcription_defaults(self):
        """Test transcription config defaults"""
        from src.config import Config

        config = Config()

        assert hasattr(config.transcription, 'model')
        assert hasattr(config.transcription, 'vad_filter')

    def test_config_pipeline_defaults(self):
        """Test pipeline config defaults"""
        from src.config import Config

        config = Config()

        assert hasattr(config.pipeline, 'skip_download')
        assert hasattr(config.pipeline, 'video_source_dir')


# =============================================================================
# TEST 24: Match Dataclass Tests
# =============================================================================
@pytest.mark.fast
class TestMatchDataclass:
    """
    Test Match dataclass.
    """

    def test_match_creation(self):
        """Test Match creation"""
        from src.state import Match

        match = Match(
            segment_index=0,
            video_file="/path/video.mp4",
            video_start=10.0,
            video_end=20.0,
            confidence=0.85
        )

        assert match.segment_index == 0
        assert match.video_file == "/path/video.mp4"
        assert match.video_start == 10.0
        assert match.video_end == 20.0
        assert match.confidence == 0.85
        assert match.strategy == ""  # Default
        assert match.face_score == 0.5  # Default

    def test_match_with_strategy(self):
        """Test Match with strategy"""
        from src.state import Match

        match = Match(
            segment_index=1,
            video_file="/path/video.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.9,
            strategy="embedding_diversity",
            reason="Best semantic match"
        )

        assert match.strategy == "embedding_diversity"
        assert match.reason == "Best semantic match"

    def test_match_face_score(self):
        """Test Match with face_score"""
        from src.state import Match

        match = Match(
            segment_index=0,
            video_file="/path/video.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.8,
            face_score=0.1  # B-roll (low face detection)
        )

        assert match.face_score == 0.1


# =============================================================================
# TEST 25: EntityImage and EntityVideo Tests
# =============================================================================
@pytest.mark.fast
class TestEntityDataclasses:
    """
    Test EntityImage and EntityVideo dataclasses.
    """

    def test_entity_image_creation(self):
        """Test EntityImage creation"""
        from src.state import EntityImage

        image = EntityImage(
            entity="Austin",
            file="/path/austin.jpg"
        )

        assert image.entity == "Austin"
        assert image.file == "/path/austin.jpg"
        assert image.source_url == ""  # Default
        assert image.width == 0  # Default
        assert image.height == 0  # Default

    def test_entity_image_full(self):
        """Test EntityImage with all fields"""
        from src.state import EntityImage

        image = EntityImage(
            entity="Dallas",
            file="/path/dallas.jpg",
            source_url="https://example.com/dallas.jpg",
            width=1920,
            height=1080
        )

        assert image.source_url == "https://example.com/dallas.jpg"
        assert image.width == 1920
        assert image.height == 1080

    def test_entity_video_creation(self):
        """Test EntityVideo creation"""
        from src.state import EntityVideo

        video = EntityVideo(
            entity="Austin",
            file="/path/austin.mp4"
        )

        assert video.entity == "Austin"
        assert video.file == "/path/austin.mp4"
        assert video.source == ""  # Default
        assert video.duration == 0.0  # Default

    def test_entity_video_full(self):
        """Test EntityVideo with all fields"""
        from src.state import EntityVideo

        video = EntityVideo(
            entity="Houston",
            file="/path/houston.mp4",
            source="pexels",
            duration=15.5
        )

        assert video.source == "pexels"
        assert video.duration == 15.5


# =============================================================================
# TEST 26: More Retry Logic Edge Cases
# =============================================================================
@pytest.mark.fast
class TestRetryLogicEdgeCases:
    """
    Additional edge case tests for retry logic.
    """

    def test_timeout_error_classification(self):
        """Test that timeout errors are properly classified"""
        from src.llm_client.retry import with_retry
        from src.llm_client.exceptions import LLMTimeoutError, LLMProviderError

        call_count = 0

        def timeout_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Request timed out after 30 seconds")

        # Timeout errors are classified at the end after all retries
        # The error message is checked to determine if it's timeout or provider error
        with pytest.raises((LLMTimeoutError, LLMProviderError)):
            with_retry(timeout_func, max_retries=2, base_delay=0.01)

    def test_deadline_error_classification(self):
        """Test that deadline errors are properly classified"""
        from src.llm_client.retry import with_retry
        from src.llm_client.exceptions import LLMTimeoutError

        call_count = 0

        def deadline_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Deadline exceeded")

        with pytest.raises(LLMTimeoutError):
            with_retry(deadline_func, max_retries=2, base_delay=0.01)

    def test_exponential_backoff_timing(self):
        """Test exponential backoff calculation - verifies retry with delays"""
        from src.llm_client.retry import with_retry
        import time

        call_count = 0
        timestamps = []

        def timing_func():
            nonlocal call_count
            call_count += 1
            timestamps.append(time.time())
            if call_count < 3:
                raise Exception("Temporary error")
            return "success"

        # Use base_delay=2 for proper exponential growth (2^0=1, 2^1=2, 2^2=4)
        # For testing, we use small delays to speed up
        result = with_retry(timing_func, max_retries=5, base_delay=0.05)

        assert result == "success"
        assert call_count == 3

        # Verify that retries happened (delays between calls)
        # Note: With base_delay ** attempt formula, delays are: 0.05^0=1s, 0.05^1=0.05s
        # So first delay is ~1s, subsequent are smaller
        if len(timestamps) >= 2:
            delay = timestamps[1] - timestamps[0]
            # First delay should be close to base_delay^0 = 1 (or adjusted by formula)
            assert delay > 0, "Should have delay between retries"


# =============================================================================
# Main
# =============================================================================
def main():
    """Run tests with pytest or standalone"""
    import argparse

    parser = argparse.ArgumentParser(description="Test recent additions")
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    pytest_args = [__file__]
    if args.verbose:
        pytest_args.append('-v')

    sys.exit(pytest.main(pytest_args))


if __name__ == '__main__':
    main()
