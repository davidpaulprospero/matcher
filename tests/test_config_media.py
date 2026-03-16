"""
Unit tests for src/config/sections/media.py

US-119 Sprint 119: Unit tests for config/sections/media module

Tests:
- VisionConfig initialization and validation
- SceneDetectionConfig initialization and validation
- AudioAnalysisConfig initialization and validation
- Dataclass serialization/deserialization
- Boundary value tests
"""

import pytest
from dataclasses import asdict, is_dataclass

from src.config.sections.media import (
    VisionConfig,
    SceneDetectionConfig,
    AudioAnalysisConfig,
)


# =============================================================================
# VisionConfig Tests
# =============================================================================

@pytest.mark.fast
class TestVisionConfigSerialization:
    """Test VisionConfig dataclass serialization."""

    def test_is_dataclass(self):
        """Test VisionConfig is a dataclass."""
        assert is_dataclass(VisionConfig)

    def test_asdict_includes_all_fields(self):
        """Test dataclasses.asdict includes all fields."""
        config = VisionConfig()
        result = asdict(config)

        assert result['provider'] == "gemini"
        assert result['model'] == "gemini-2.0-flash"
        assert result['enabled'] is True
        assert result['min_words_per_scene'] == 5
        assert result['coverage_threshold'] == 0.3
        assert result['max_scenes_per_video'] == 50
        assert result['frame_format'] == "jpg"
        assert result['frame_quality'] == 85
        assert result['max_api_calls_per_run'] == 100
        assert result['estimated_cost_per_call'] == 0.001

    def test_asdict_custom_values(self):
        """Test serialization with custom values."""
        config = VisionConfig(
            provider="openai",
            model="gpt-4o",
            enabled=False,
            min_words_per_scene=10,
            coverage_threshold=0.5,
            max_scenes_per_video=25,
            frame_format="png",
            frame_quality=95,
            max_api_calls_per_run=50,
            estimated_cost_per_call=0.002,
        )
        result = asdict(config)

        assert result['provider'] == "openai"
        assert result['model'] == "gpt-4o"
        assert result['enabled'] is False
        assert result['min_words_per_scene'] == 10
        assert result['coverage_threshold'] == 0.5
        assert result['max_scenes_per_video'] == 25
        assert result['frame_format'] == "png"
        assert result['frame_quality'] == 95
        assert result['max_api_calls_per_run'] == 50
        assert result['estimated_cost_per_call'] == 0.002

    def test_roundtrip_via_dict(self):
        """Test serialization preserves values through dict conversion."""
        original = VisionConfig(
            provider="anthropic",
            model="claude-3-5-sonnet",
            coverage_threshold=0.4,
        )
        result = asdict(original)

        # Verify values are preserved
        assert result['provider'] == original.provider
        assert result['model'] == original.model
        assert result['coverage_threshold'] == original.coverage_threshold


@pytest.mark.fast
class TestVisionConfigBoundaryValues:
    """Test VisionConfig boundary values."""

    def test_coverage_threshold_zero(self):
        """Test coverage_threshold at minimum boundary (0.0)."""
        config = VisionConfig(coverage_threshold=0.0)
        assert config.coverage_threshold == 0.0

    def test_coverage_threshold_one(self):
        """Test coverage_threshold at maximum boundary (1.0)."""
        config = VisionConfig(coverage_threshold=1.0)
        assert config.coverage_threshold == 1.0

    def test_frame_quality_boundary(self):
        """Test frame_quality at JPEG boundaries."""
        config_min = VisionConfig(frame_quality=1)
        config_max = VisionConfig(frame_quality=100)
        assert config_min.frame_quality == 1
        assert config_max.frame_quality == 100

    def test_max_api_calls_one(self):
        """Test max_api_calls_per_run at minimum (1)."""
        config = VisionConfig(max_api_calls_per_run=1)
        assert config.max_api_calls_per_run == 1


# =============================================================================
# SceneDetectionConfig Tests
# =============================================================================

@pytest.mark.fast
class TestSceneDetectionConfigSerialization:
    """Test SceneDetectionConfig dataclass serialization."""

    def test_is_dataclass(self):
        """Test SceneDetectionConfig is a dataclass."""
        assert is_dataclass(SceneDetectionConfig)

    def test_asdict_includes_all_fields(self):
        """Test dataclasses.asdict includes all fields."""
        config = SceneDetectionConfig()
        result = asdict(config)

        assert result['enabled'] is True
        assert result['preset'] == "balanced"
        assert result['threshold'] == 27.0
        assert result['min_scene_len'] == 15
        assert result['downscale_factor'] == 4
        assert result['frame_skip'] == 2
        assert result['min_video_duration'] == 0
        assert result['audio_analysis'] is True
        assert result['silence_threshold_db'] == -40.0
        assert result['min_silence_duration'] == 0.3
        assert result['use_gpu'] is True
        assert result['force_gpu'] is False
        assert result['detect_faces_per_scene'] is True
        assert result['face_sample_frames'] == 3

    def test_asdict_custom_values(self):
        """Test serialization with custom values."""
        config = SceneDetectionConfig(
            enabled=False,
            preset="fast",
            threshold=30.0,
            min_scene_len=20,
            downscale_factor=2,
            frame_skip=4,
            min_video_duration=10.0,
            audio_analysis=False,
            silence_threshold_db=-50.0,
            min_silence_duration=0.5,
            use_gpu=False,
            force_gpu=True,
            detect_faces_per_scene=False,
            face_sample_frames=5,
        )
        result = asdict(config)

        assert result['enabled'] is False
        assert result['preset'] == "fast"
        assert result['threshold'] == 30.0
        assert result['min_scene_len'] == 20
        assert result['downscale_factor'] == 2
        assert result['frame_skip'] == 4
        assert result['min_video_duration'] == 10.0
        assert result['audio_analysis'] is False
        assert result['silence_threshold_db'] == -50.0
        assert result['min_silence_duration'] == 0.5
        assert result['use_gpu'] is False
        assert result['force_gpu'] is True
        assert result['detect_faces_per_scene'] is False
        assert result['face_sample_frames'] == 5


@pytest.mark.fast
class TestSceneDetectionConfigBoundaryValues:
    """Test SceneDetectionConfig boundary values."""

    def test_threshold_min_boundary(self):
        """Test threshold at minimum positive boundary."""
        config = SceneDetectionConfig(threshold=0.1)
        assert config.threshold == 0.1

    def test_min_scene_len_one(self):
        """Test min_scene_len at minimum (1)."""
        config = SceneDetectionConfig(min_scene_len=1)
        assert config.min_scene_len == 1

    def test_downscale_factor_min(self):
        """Test downscale_factor minimum (1)."""
        config = SceneDetectionConfig(downscale_factor=1)
        assert config.downscale_factor == 1

    def test_face_sample_frames_min(self):
        """Test face_sample_frames minimum (1)."""
        config = SceneDetectionConfig(face_sample_frames=1)
        assert config.face_sample_frames == 1


# =============================================================================
# AudioAnalysisConfig Tests
# =============================================================================

@pytest.mark.fast
class TestAudioAnalysisConfigSerialization:
    """Test AudioAnalysisConfig dataclass serialization."""

    def test_is_dataclass(self):
        """Test AudioAnalysisConfig is a dataclass."""
        assert is_dataclass(AudioAnalysisConfig)

    def test_asdict_includes_all_fields(self):
        """Test dataclasses.asdict includes all fields."""
        config = AudioAnalysisConfig()
        result = asdict(config)

        assert result['enabled'] is True
        assert result['sample_rate'] == 22050
        assert result['silence_threshold_db'] == -40.0
        assert result['min_silence_duration'] == 0.3
        assert result['speech_threshold'] == 0.5
        assert result['min_speech_duration'] == 0.2

    def test_asdict_custom_values(self):
        """Test serialization with custom values."""
        config = AudioAnalysisConfig(
            enabled=False,
            sample_rate=44100,
            silence_threshold_db=-50.0,
            min_silence_duration=0.5,
            speech_threshold=0.7,
            min_speech_duration=0.3,
        )
        result = asdict(config)

        assert result['enabled'] is False
        assert result['sample_rate'] == 44100
        assert result['silence_threshold_db'] == -50.0
        assert result['min_silence_duration'] == 0.5
        assert result['speech_threshold'] == 0.7
        assert result['min_speech_duration'] == 0.3


@pytest.mark.fast
class TestAudioAnalysisConfigBoundaryValues:
    """Test AudioAnalysisConfig boundary values."""

    def test_sample_rate_standard(self):
        """Test standard sample rates."""
        config_22050 = AudioAnalysisConfig(sample_rate=22050)
        config_44100 = AudioAnalysisConfig(sample_rate=44100)
        config_48000 = AudioAnalysisConfig(sample_rate=48000)
        assert config_22050.sample_rate == 22050
        assert config_44100.sample_rate == 44100
        assert config_48000.sample_rate == 48000

    def test_silence_threshold_boundary(self):
        """Test silence_threshold_db at boundaries."""
        config = AudioAnalysisConfig(silence_threshold_db=-60.0)
        assert config.silence_threshold_db == -60.0

    def test_duration_boundaries(self):
        """Test duration boundaries."""
        config = AudioAnalysisConfig(
            min_silence_duration=0.1,
            min_speech_duration=0.1,
        )
        assert config.min_silence_duration == 0.1
        assert config.min_speech_duration == 0.1


# =============================================================================
# VisionConfig __post_init__ Validation Tests
# =============================================================================

@pytest.mark.fast
class TestVisionConfigPostInitValidation:
    """Test VisionConfig __post_init__ raises ValueError on invalid values."""

    def test_coverage_threshold_below_zero_raises(self):
        """coverage_threshold < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_threshold must be between"):
            VisionConfig(coverage_threshold=-0.1)

    def test_coverage_threshold_above_one_raises(self):
        """coverage_threshold > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_threshold must be between"):
            VisionConfig(coverage_threshold=1.5)

    def test_coverage_threshold_negative_one_raises(self):
        """coverage_threshold = -1.0 raises ValueError."""
        with pytest.raises(ValueError, match="coverage_threshold must be between"):
            VisionConfig(coverage_threshold=-1.0)

    def test_max_api_calls_zero_raises(self):
        """max_api_calls_per_run = 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_api_calls_per_run must be positive"):
            VisionConfig(max_api_calls_per_run=0)

    def test_max_api_calls_negative_raises(self):
        """max_api_calls_per_run negative raises ValueError."""
        with pytest.raises(ValueError, match="max_api_calls_per_run must be positive"):
            VisionConfig(max_api_calls_per_run=-10)

    def test_valid_boundaries_accept(self):
        """Valid boundary values are accepted."""
        config = VisionConfig(coverage_threshold=0.0, max_api_calls_per_run=1)
        assert config.coverage_threshold == 0.0
        assert config.max_api_calls_per_run == 1


# =============================================================================
# SceneDetectionConfig __post_init__ Validation Tests
# =============================================================================

@pytest.mark.fast
class TestSceneDetectionConfigPostInitValidation:
    """Test SceneDetectionConfig __post_init__ raises ValueError on invalid values."""

    def test_threshold_zero_raises(self):
        """threshold = 0 raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            SceneDetectionConfig(threshold=0)

    def test_threshold_negative_raises(self):
        """threshold negative raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be positive"):
            SceneDetectionConfig(threshold=-5.0)

    def test_min_scene_len_zero_raises(self):
        """min_scene_len = 0 raises ValueError."""
        with pytest.raises(ValueError, match="min_scene_len must be positive"):
            SceneDetectionConfig(min_scene_len=0)

    def test_min_scene_len_negative_raises(self):
        """min_scene_len negative raises ValueError."""
        with pytest.raises(ValueError, match="min_scene_len must be positive"):
            SceneDetectionConfig(min_scene_len=-10)

    def test_valid_boundaries_accept(self):
        """Valid boundary values are accepted."""
        config = SceneDetectionConfig(threshold=0.1, min_scene_len=1)
        assert config.threshold == 0.1
        assert config.min_scene_len == 1


# =============================================================================
# AudioAnalysisConfig - No __post_init__ validation
# =============================================================================

@pytest.mark.fast
class TestAudioAnalysisConfigNoValidation:
    """Test AudioAnalysisConfig has no __post_init__ validation."""

    def test_all_values_accepted(self):
        """AudioAnalysisConfig accepts any values (no validation)."""
        # Negative values are allowed (no validation in __post_init__)
        config = AudioAnalysisConfig(
            sample_rate=0,
            silence_threshold_db=0.0,
            min_silence_duration=0.0,
            speech_threshold=0.0,
            min_speech_duration=0.0,
        )
        assert config.sample_rate == 0
        assert config.silence_threshold_db == 0.0
        assert config.min_silence_duration == 0.0
        assert config.speech_threshold == 0.0
        assert config.min_speech_duration == 0.0


# =============================================================================
# Media Source URLs and Credentials - Not Applicable
# =============================================================================

@pytest.mark.fast
class TestMediaSourceNotApplicable:
    """Verify media.py does not have media source URLs/credentials.

    Note: The acceptance criterion "Test media source URLs and credentials"
    is not applicable to this module. The media.py config defines VisionConfig,
    SceneDetectionConfig, and AudioAnalysisConfig - none of which contain
    media source URLs or credentials. These would be in a different module
    (e.g., media_sources package).
    """

    def test_no_media_source_credentials_fields(self):
        """Verify no credential fields exist in media configs."""
        vision = VisionConfig()
        scene = SceneDetectionConfig()
        audio = AudioAnalysisConfig()

        # These configs don't have URL/credential fields
        vision_dict = asdict(vision)
        scene_dict = asdict(scene)
        audio_dict = asdict(audio)

        # Check no fields contain 'url', 'credential', 'api_key', 'secret'
        for field_name in vision_dict.keys():
            assert 'url' not in field_name.lower()
            assert 'credential' not in field_name.lower()
            assert 'api_key' not in field_name.lower()
            assert 'secret' not in field_name.lower()

        for field_name in scene_dict.keys():
            assert 'url' not in field_name.lower()
            assert 'credential' not in field_name.lower()
            assert 'api_key' not in field_name.lower()
            assert 'secret' not in field_name.lower()

        for field_name in audio_dict.keys():
            assert 'url' not in field_name.lower()
            assert 'credential' not in field_name.lower()
            assert 'api_key' not in field_name.lower()
            assert 'secret' not in field_name.lower()


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.fast
class TestMediaConfigIntegration:
    """Integration tests for media config classes."""

    def test_all_configs_instantiate(self):
        """Test all config classes can be instantiated."""
        vision = VisionConfig()
        scene = SceneDetectionConfig()
        audio = AudioAnalysisConfig()

        assert vision is not None
        assert scene is not None
        assert audio is not None

    def test_all_configs_serialize(self):
        """Test all configs serialize correctly."""
        vision = VisionConfig(provider="gemini")
        scene = SceneDetectionConfig(preset="fast")
        audio = AudioAnalysisConfig(enabled=False)

        vision_dict = asdict(vision)
        scene_dict = asdict(scene)
        audio_dict = asdict(audio)

        assert isinstance(vision_dict, dict)
        assert isinstance(scene_dict, dict)
        assert isinstance(audio_dict, dict)

    def test_presets_work(self):
        """Test scene detection presets."""
        for preset in ["fast", "balanced", "accurate"]:
            config = SceneDetectionConfig(preset=preset)
            assert config.preset == preset
