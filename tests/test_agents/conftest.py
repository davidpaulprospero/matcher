"""
Pytest fixtures for agent tests.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@pytest.fixture
def project_dir(tmp_path) -> Path:
    """Create a temporary project directory."""
    project = tmp_path / "test_project"
    project.mkdir()
    (project / ".cache").mkdir()
    (project / "videos").mkdir()
    (project / "output").mkdir()
    return project


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = Mock()

    # Output config
    config.output = Mock()
    config.output.gap_mode = "scale"
    config.output.frame_rate = 30.0
    config.output.include_alternatives = True
    config.output.include_strategy_tracks = True
    config.output.include_entity_images = True
    config.output.include_entity_videos = True
    config.output.export_edl = True
    config.output.export_xml = True
    config.output.num_alternatives = 2
    config.output.align_to_voiceover = True

    # Download config
    config.download = Mock()
    config.download.format = "bestvideo+bestaudio/best"
    config.download.socket_timeout = 30
    config.download.continue_dl = False
    config.download.root_dir = None

    # Cookie rotation config - disabled by default to prevent CookieRotator initialization
    config.download.cookie_rotation = Mock()
    config.download.cookie_rotation.enabled = False
    config.download.cookie_rotation.cookie_files = []
    config.download.cookie_rotation.rotate_on_errors = ["429", "rate limit"]
    config.download.cookie_rotation.rotation_strategy = "on_error"
    config.download.cookie_rotation.cooldown_seconds = 300
    config.download.cookie_rotation.max_rotations_per_session = 0

    # VPN config - disabled by default
    config.download.vpn = Mock()
    config.download.vpn.enabled = False

    # LLM config
    config.llm = Mock()
    config.llm.provider = "gemini"
    config.llm.default_provider = "gemini"
    config.llm.timeout = 30

    # Matching config
    config.matching = Mock()

    return config


@dataclass
class MockSegment:
    """Mock voiceover segment."""
    start: float = 0.0
    end: float = 5.0
    text: str = "Test segment"
    duration: float = 5.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class MockMatch:
    """Mock match result."""
    segment: MockSegment = field(default_factory=MockSegment)
    video_path: str = ""
    video_id: str = "test_video_123"
    source_id: str = "test_video_123"
    start_time: float = 0.0
    end_time: float = 5.0
    confidence: float = 0.8
    time_scalar: float = 1.0
    metadata: Dict = field(default_factory=dict)


@pytest.fixture
def mock_segment():
    """Create a mock segment."""
    return MockSegment()


@pytest.fixture
def mock_match(tmp_path):
    """Create a mock match with existing video file."""
    video_file = tmp_path / "videos" / "test_video_123.mp4"
    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(b"fake video content")

    return MockMatch(
        video_path=str(video_file),
        segment=MockSegment(start=0.0, end=5.0)
    )


@pytest.fixture
def mock_matches(tmp_path):
    """Create multiple mock matches."""
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    matches = []
    for i in range(5):
        video_file = videos_dir / f"test_video_{i}.mp4"
        video_file.write_bytes(b"fake video content")

        match = MockMatch(
            video_path=str(video_file),
            video_id=f"test_video_{i}",
            source_id=f"test_video_{i}",
            segment=MockSegment(
                start=i * 10.0,
                end=(i + 1) * 10.0 - 1.0,
                duration=9.0
            ),
            start_time=0.0,
            end_time=9.0
        )
        matches.append(match)

    return matches


@pytest.fixture
def mock_state(mock_matches):
    """Create a mock pipeline state."""
    state = Mock()
    state.matches = mock_matches
    state.voiceover_path = "/path/to/voiceover.srt"
    state.entity_images = {}
    state.entity_videos = {}
    state.downloads = []
    state.stage_timings = {}
    return state


@pytest.fixture
def mock_checkpoint():
    """Create a mock checkpoint manager."""
    checkpoint = Mock()
    checkpoint.exists.return_value = False
    checkpoint.load.return_value = None
    checkpoint.save.return_value = True
    return checkpoint


@pytest.fixture
def mock_stage():
    """Create a mock pipeline stage."""
    from src.stages import StageResult

    stage = Mock()
    stage.name = "TEST_STAGE"
    stage.can_skip.return_value = False
    stage.validate_inputs.return_value = None
    stage.run.return_value = StageResult.ok({})
    stage.restore.return_value = True
    return stage
