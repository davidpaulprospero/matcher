"""Media processing configuration: Vision API, scene detection, audio analysis.

Extracted from monolithic config.py during refactoring (Jan 7, 2026).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    'VisionConfig',
    'SceneDetectionConfig',
    'AudioAnalysisConfig',
]


@dataclass
class VisionConfig:
    """Vision processing settings

    Chain-of-thought: Vision API expensive, use selectively
    Reasoning: Only process scenes where transcript is sparse
    Decision: coverage_threshold=0.3 means skip if >30% transcribed
    """
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    enabled: bool = True

    # Selective processing
    min_words_per_scene: int = 5
    coverage_threshold: float = 0.3
    max_scenes_per_video: int = 50

    # Frame extraction
    frame_format: str = "jpg"
    frame_quality: int = 85

    # Cost management
    max_api_calls_per_run: int = 100
    estimated_cost_per_call: float = 0.001

    def __post_init__(self):
        if not (0.0 <= self.coverage_threshold <= 1.0):
            raise ValueError(
                f"VisionConfig.coverage_threshold must be between 0.0 and 1.0, "
                f"got {self.coverage_threshold}"
            )
        if self.max_api_calls_per_run <= 0:
            raise ValueError(
                f"VisionConfig.max_api_calls_per_run must be positive, "
                f"got {self.max_api_calls_per_run}"
            )


@dataclass
class SceneDetectionConfig:
    """Scene detection settings (PySceneDetect)

    Chain-of-thought: Scene boundaries enable granular clip selection
    Reasoning: Presets balance speed vs accuracy for different use cases
    Decision: "balanced" default, "fast" for preview, "accurate" for final
    """
    enabled: bool = True
    preset: str = "balanced"  # fast, balanced, accurate

    # Individual settings (can override preset)
    threshold: float = 27.0
    min_scene_len: int = 15  # frames
    downscale_factor: int = 4
    frame_skip: int = 2

    # Minimum video duration to process (seconds, 0 = no minimum)
    min_video_duration: float = 0

    # Audio analysis integration
    audio_analysis: bool = True
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.3

    # Hardware acceleration
    use_gpu: bool = True
    force_gpu: bool = False

    # Face detection per scene (B-roll identification)
    detect_faces_per_scene: bool = True  # Enable scene-level face detection
    face_sample_frames: int = 3  # Frames to sample per scene for face detection

    def __post_init__(self):
        if self.threshold <= 0:
            raise ValueError(
                f"SceneDetectionConfig.threshold must be positive, "
                f"got {self.threshold}"
            )
        if self.min_scene_len <= 0:
            raise ValueError(
                f"SceneDetectionConfig.min_scene_len must be positive, "
                f"got {self.min_scene_len}"
            )


@dataclass
class AudioAnalysisConfig:
    """Audio analysis settings (librosa)"""
    enabled: bool = True
    sample_rate: int = 22050

    # Silence detection
    silence_threshold_db: float = -40.0
    min_silence_duration: float = 0.3

    # Speech detection
    speech_threshold: float = 0.5
    min_speech_duration: float = 0.2
