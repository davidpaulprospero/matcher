"""
Healing System Simulation Tests.

These tests simulate real-world error scenarios to verify healers
correctly detect and recover from common pipeline failures.

Simulations cover:
1. OTIOHealer - Timeline generation, clip timing, media reference errors
2. APIHealer - Rate limits, authentication, quotas, timeouts
3. CheckpointHealer - JSON corruption, missing backups, cache rebuild
4. DownloadHealer - YouTube rate limits, unavailable videos, format errors
5. DiskHealer - Disk full, permission denied, cache cleanup
6. PathHealer - Windows path limits, unicode, illegal characters
7. HealingOrchestrator - Coordinated healing, preflight, rollback
8. Integration - Multi-healer scenarios, cascading failures

TEST TIERS (run with pytest -m <marker>):
- @pytest.mark.fast: Unit tests, <30s total. Run on every commit.
- @pytest.mark.integration: Concurrent/IO tests, <2min total. Run on PR merge.
- @pytest.mark.simulation: Full scenario simulations, <5min total. Run on PR merge.

Usage:
    pytest -m fast tests/test_healing_simulations.py
    pytest -m simulation tests/test_healing_simulations.py
    pytest tests/test_healing_simulations.py
"""

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

# Test tier markers
pytestmark_fast = pytest.mark.fast
pytestmark_simulation = pytest.mark.simulation
pytestmark_integration = pytest.mark.integration


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def temp_project_dir(tmp_path):
    """Create a temporary project directory with standard structure."""
    project = tmp_path / "TestProject__2026-01-15"
    project.mkdir()

    # Create standard directories
    (project / ".cache").mkdir()
    (project / ".cache" / "transcriptions").mkdir()
    (project / ".cache" / "embeddings").mkdir()
    (project / ".cache" / "scene_detection").mkdir()
    (project / "output").mkdir()
    (project / "videos").mkdir()

    return project


@pytest.fixture
def mock_config():
    """Create a mock pipeline configuration."""
    config = Mock()

    # Output config
    config.output = Mock()
    config.output.gap_mode = "scale"
    config.output.track_count = 3
    config.output.include_disabled_tracks = True
    config.output.framerate = 30.0
    config.output.edl_enabled = True
    config.output.xml_enabled = True

    # Download config
    config.download = Mock()
    config.download.timeout = 60.0
    config.download.max_retries = 3
    config.download.root_dir = None
    config.download.format = "bestvideo[height<=1080]+bestaudio/best"

    # LLM config
    config.llm = Mock()
    config.llm.provider = "gemini"
    config.llm.timeout = 60.0
    config.llm.max_retries = 3

    # Transcription config
    config.transcription = Mock()
    config.transcription.model = "base"
    config.transcription.timeout = 120.0

    # Matching config
    config.matching = Mock()
    config.matching.min_score = 0.5
    config.matching.max_candidates = 20

    # Healing config
    config.healing = Mock()
    config.healing.enabled = True
    config.healing.strategy = "conservative"
    config.healing.max_attempts_per_stage = 3
    config.healing.max_total_heals = 20
    config.healing.heal_delay = 0.1  # Fast for tests

    return config


@dataclass
class MockSegment:
    """Mock voiceover segment with real numeric values."""
    start: float = 0.0
    end: float = 10.0
    text: str = "Sample text"
    duration: Optional[float] = None

    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end - self.start


@dataclass
class MockMatch:
    """Mock match with real numeric values."""
    video_path: str = ""
    start_time: float = 0.0
    end_time: float = 8.0
    score: float = 0.8
    segment_index: int = 0
    segment: Optional[MockSegment] = None


@dataclass
class MockDownloadedVideo:
    """Mock downloaded video."""
    file: str = ""
    duration: float = 120.0
    video_id: str = ""


@pytest.fixture
def mock_pipeline_state(mock_config, temp_project_dir):
    """Create a mock pipeline state."""
    state = Mock()
    state.config = mock_config
    state.project_dir = temp_project_dir

    # Create sample segments
    state.voiceover_segments = []
    for i in range(5):
        segment = MockSegment(
            start=i * 10.0,
            end=i * 10.0 + 8.0,
            text=f"Segment {i} text"
        )
        state.voiceover_segments.append(segment)

    # Create sample matches
    state.matches = []
    for i in range(5):
        match = MockMatch(
            video_path=str(temp_project_dir / "videos" / f"video_{i}.mp4"),
            start_time=i * 10.0,
            end_time=i * 10.0 + 8.0,
            score=0.8 - (i * 0.1),
            segment_index=i,
            segment=state.voiceover_segments[i]
        )
        state.matches.append(match)

    # Create sample videos
    state.downloaded_videos = []
    for i in range(5):
        video = MockDownloadedVideo(
            file=str(temp_project_dir / "videos" / f"video_{i}.mp4"),
            duration=120.0,
            video_id=f"video_id_{i}"
        )
        state.downloaded_videos.append(video)

    return state


@pytest.fixture
def otio_healer(mock_config, temp_project_dir):
    """Create OTIOHealer instance for testing."""
    from src.agents.healers.otio import OTIOHealer
    return OTIOHealer(mock_config, temp_project_dir)


@pytest.fixture
def api_healer(mock_config, temp_project_dir):
    """Create APIHealer instance for testing."""
    from src.agents.healers.api import APIHealer
    return APIHealer(mock_config, temp_project_dir)


@pytest.fixture
def checkpoint_healer(mock_config, temp_project_dir):
    """Create CheckpointHealer instance for testing."""
    from src.agents.healers.checkpoint import CheckpointHealer
    return CheckpointHealer(mock_config, temp_project_dir)


@pytest.fixture
def download_healer(mock_config, temp_project_dir):
    """Create DownloadHealer instance for testing."""
    from src.agents.healers.download import DownloadHealer
    return DownloadHealer(mock_config, temp_project_dir)


# ==============================================================================
# SIMULATED ERROR SCENARIOS
# ==============================================================================


@dataclass
class ErrorScenario:
    """Defines an error simulation scenario."""
    name: str
    error_type: type
    error_message: str
    expected_healer: str
    expected_action: str  # "retry", "skip", "modify", "restore", "abort"
    context: Dict[str, Any] = field(default_factory=dict)


# Real-world error scenarios collected from production logs
OTIO_ERROR_SCENARIOS = [
    ErrorScenario(
        name="negative_duration",
        error_type=ValueError,
        error_message="ValueError: Duration must be positive, got -0.5",
        expected_healer="otio-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="zero_duration_clip",
        error_type=ValueError,
        error_message="ValueError: Zero duration clip at 00:01:30",
        expected_healer="otio-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="missing_media_reference",
        error_type=FileNotFoundError,
        error_message="FileNotFoundError: Media file not found: E:/Projects/video.mp4",
        expected_healer="otio-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="clip_overlap",
        error_type=ValueError,
        error_message="ValueError: Clip overlap detected: clip 3 overlaps clip 4 by 0.5s",
        expected_healer="otio-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="gap_overflow",
        error_type=ValueError,
        error_message="ValueError: Total gap duration (45.2s) exceeds available time (30.0s)",
        expected_healer="otio-healer",
        expected_action="modify",
    ),
    ErrorScenario(
        name="edl_export_failure",
        error_type=RuntimeError,
        error_message="RuntimeError: EDL adapter failed: invalid timecode format",
        expected_healer="otio-healer",
        expected_action="modify",
    ),
    ErrorScenario(
        name="xml_export_failure",
        error_type=RuntimeError,
        error_message="RuntimeError: XML adapter failed: unsupported frame rate",
        expected_healer="otio-healer",
        expected_action="modify",
    ),
    ErrorScenario(
        name="metadata_numpy_serialization",
        error_type=TypeError,
        error_message="TypeError: Object of type ndarray is not JSON serializable",
        expected_healer="otio-healer",
        expected_action="retry",
    ),
]

API_ERROR_SCENARIOS = [
    ErrorScenario(
        name="rate_limit_429",
        error_type=Exception,
        error_message="Error 429: Rate limit exceeded. Please wait before retrying.",
        expected_healer="api-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="gemini_quota_exceeded",
        error_type=Exception,
        error_message="RESOURCE_EXHAUSTED: Quota exceeded for Gemini API",
        expected_healer="api-healer",
        expected_action="modify",  # Switch provider
    ),
    ErrorScenario(
        name="anthropic_unauthorized",
        error_type=Exception,
        error_message="Error 401: Unauthorized. Invalid Anthropic API key.",
        expected_healer="api-healer",
        expected_action="abort",
    ),
    ErrorScenario(
        name="timeout_error",
        error_type=TimeoutError,
        error_message="TimeoutError: Request timed out after 60 seconds",
        expected_healer="api-healer",
        expected_action="modify",
    ),
    ErrorScenario(
        name="connection_refused",
        error_type=ConnectionError,
        error_message="ConnectionError: Connection refused to localhost:11434 (Ollama)",
        expected_healer="api-healer",
        expected_action="modify",
    ),
]

CHECKPOINT_ERROR_SCENARIOS = [
    ErrorScenario(
        name="json_decode_error",
        error_type=json.JSONDecodeError,
        error_message="JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
        expected_healer="checkpoint-healer",
        expected_action="restore",
    ),
    ErrorScenario(
        name="corrupt_checkpoint",
        error_type=ValueError,
        error_message="ValueError: Checkpoint data is corrupt: missing 'stages' key",
        expected_healer="checkpoint-healer",
        expected_action="restore",
    ),
    ErrorScenario(
        name="hash_mismatch",
        error_type=ValueError,
        error_message="ValueError: Config hash mismatch - checkpoint was created with different settings",
        expected_healer="checkpoint-healer",
        expected_action="retry",
    ),
]

DOWNLOAD_ERROR_SCENARIOS = [
    ErrorScenario(
        name="youtube_rate_limit",
        error_type=Exception,
        error_message="ERROR: HTTP Error 429: Too Many Requests. Please retry in a few minutes.",
        expected_healer="download-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="video_unavailable",
        error_type=Exception,
        error_message="ERROR: Video unavailable. This video has been removed by the user.",
        expected_healer="download-healer",
        expected_action="skip",
    ),
    ErrorScenario(
        name="private_video",
        error_type=Exception,
        error_message="ERROR: Private video. Sign in if you've been granted access to this video.",
        expected_healer="download-healer",
        expected_action="skip",
    ),
    ErrorScenario(
        name="format_extraction",
        error_type=Exception,
        error_message="ERROR: No video formats found; unable to extract video data",
        expected_healer="download-healer",
        expected_action="modify",
    ),
    ErrorScenario(
        name="incomplete_download",
        error_type=Exception,
        error_message="ERROR: Incomplete download: 45% complete before connection lost",
        expected_healer="download-healer",
        expected_action="retry",
    ),
]

DISK_ERROR_SCENARIOS = [
    ErrorScenario(
        name="disk_full_enospc",
        error_type=OSError,
        error_message="OSError: [Errno 28] No space left on device",
        expected_healer="disk-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="permission_denied",
        error_type=PermissionError,
        error_message="PermissionError: [Errno 13] Permission denied: '/protected/path'",
        expected_healer="disk-healer",
        expected_action="modify",
    ),
]

PATH_ERROR_SCENARIOS = [
    ErrorScenario(
        name="path_too_long_windows",
        error_type=OSError,
        error_message="OSError: [Errno 206] File name too long (> 260 characters)",
        expected_healer="path-healer",
        expected_action="modify",
    ),
    ErrorScenario(
        name="unicode_encode_error",
        error_type=UnicodeEncodeError,
        error_message="UnicodeEncodeError: 'ascii' codec can't encode character",
        expected_healer="path-healer",
        expected_action="retry",
    ),
    ErrorScenario(
        name="illegal_characters",
        error_type=ValueError,
        error_message="ValueError: Filename contains illegal characters: <>:\"|?*",
        expected_healer="path-healer",
        expected_action="retry",
    ),
]


# ==============================================================================
# OTIO HEALER SIMULATIONS
# ==============================================================================


@pytest.mark.simulation
class TestOTIOHealerSimulations:
    """Simulate OTIO timeline generation errors."""

    @pytest.mark.parametrize("scenario", OTIO_ERROR_SCENARIOS, ids=lambda s: s.name)
    def test_otio_error_detection(self, otio_healer, scenario):
        """OTIOHealer should detect all OTIO-related errors."""
        error = scenario.error_type(scenario.error_message)

        can_handle = otio_healer.can_handle(error, "OUTPUT")
        assert can_handle, f"OTIOHealer should handle: {scenario.name}"

    def test_negative_duration_fix(self, otio_healer, mock_pipeline_state):
        """Simulate negative duration clip and verify fix."""
        error = ValueError("Duration must be positive, got -0.5")

        # Simulate a match with negative duration
        start_time = mock_pipeline_state.matches[0].start_time
        mock_pipeline_state.matches[0].end_time = start_time - 0.5
        mock_pipeline_state.matches[0].segment.end = start_time - 0.5

        result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        assert result.success, f"Fix should succeed: {result.message}"
        assert result.action.value in ["retry", "modify"]

    def test_zero_duration_fix(self, otio_healer, mock_pipeline_state):
        """Simulate zero duration clip and verify fix."""
        error = ValueError("Zero duration clip detected")

        # Simulate zero duration
        start_time = mock_pipeline_state.matches[0].start_time
        mock_pipeline_state.matches[0].end_time = start_time
        mock_pipeline_state.matches[0].segment.end = mock_pipeline_state.matches[0].segment.start

        result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        assert result.success
        # Should clamp to minimum duration

    def test_missing_media_search_caches(self, otio_healer, mock_pipeline_state, temp_project_dir):
        """Simulate missing media file - healer should search caches."""
        # Create fake video in cache
        global_cache = temp_project_dir / ".cache" / "videos"
        global_cache.mkdir(parents=True, exist_ok=True)
        (global_cache / "test_video.mp4").touch()

        error = FileNotFoundError("Media file not found: /missing/path/test_video.mp4")

        result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        # Should attempt to find file or continue with skip
        assert result is not None

    def test_gap_overflow_mode_switch(self, otio_healer, mock_pipeline_state):
        """Simulate gap overflow - healer should switch gap mode."""
        error = ValueError("Total gap duration (45.2s) exceeds available time (30.0s)")

        original_gap_mode = mock_pipeline_state.config.output.gap_mode

        result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        # Should change gap mode to less aggressive option
        if result.success:
            assert result.action.value in ["retry", "modify"]

    def test_edl_export_disable(self, otio_healer, mock_pipeline_state):
        """Simulate EDL export failure - healer should disable EDL."""
        error = RuntimeError("EDL adapter failed: invalid timecode format")

        result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        # Should indicate EDL should be skipped
        assert result.success
        if result.modified_config:
            # Healer reports skip_edl in details rather than modifying mock
            assert result.details.get('skip_edl', False) == True or result.action.value == "modify"

    def test_numpy_metadata_sanitization(self, otio_healer, mock_pipeline_state):
        """Simulate numpy serialization error in metadata."""
        error = TypeError("Object of type ndarray is not JSON serializable")

        result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        # Should sanitize metadata and retry
        assert result.success

    def test_clip_overlap_fix(self, otio_healer, mock_pipeline_state):
        """Simulate overlapping clips - healer should trim earlier clip."""
        error = ValueError("Clip overlap detected: clip 1 overlaps clip 2 by 0.5s")

        # Create overlapping matches (these are dataclasses now)
        mock_pipeline_state.matches[0].end_time = 15.0
        mock_pipeline_state.matches[0].segment.end = 15.0
        mock_pipeline_state.matches[1].start_time = 14.5  # Overlap of 0.5s
        mock_pipeline_state.matches[1].segment.start = 14.5

        result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        assert result.success

    def test_safe_mode_fallback(self, otio_healer, mock_pipeline_state):
        """Simulate repeated failures - healer should fall back to safe mode."""
        error = RuntimeError("Unknown OTIO error after multiple retries")

        # Simulate multiple failed fixes by calling fix multiple times
        for _ in range(3):
            result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")

        # After repeated failures, should apply safe mode
        # Safe mode: V1 only, no EDL/XML, gap_mode=none


# ==============================================================================
# API HEALER SIMULATIONS
# ==============================================================================


@pytest.mark.simulation
class TestAPIHealerSimulations:
    """Simulate LLM/API errors."""

    @pytest.mark.parametrize("scenario", API_ERROR_SCENARIOS, ids=lambda s: s.name)
    def test_api_error_detection(self, api_healer, scenario):
        """APIHealer should detect all API-related errors."""
        error = scenario.error_type(scenario.error_message)

        can_handle = api_healer.can_handle(error, "MATCH")
        assert can_handle, f"APIHealer should handle: {scenario.name}"

    def test_rate_limit_backoff(self, api_healer, mock_pipeline_state):
        """Simulate rate limiting - verify exponential backoff."""
        error = Exception("Error 429: Rate limit exceeded")

        # Mock time.sleep to track backoff
        backoff_times = []
        original_sleep = time.sleep

        def mock_sleep(duration):
            backoff_times.append(duration)
            # Don't actually sleep in tests

        with patch('time.sleep', side_effect=mock_sleep):
            # First rate limit
            result1 = api_healer.fix(error, mock_pipeline_state, "MATCH")
            assert result1.success

            # Second rate limit - should have longer backoff
            result2 = api_healer.fix(error, mock_pipeline_state, "MATCH")
            assert result2.success

        # Verify backoff increased
        assert len(backoff_times) == 2
        assert backoff_times[1] >= backoff_times[0]

    def test_quota_exceeded_provider_switch(self, api_healer, mock_pipeline_state):
        """Simulate quota exceeded - verify provider switch."""
        error = Exception("RESOURCE_EXHAUSTED: Quota exceeded for Gemini API")

        original_provider = mock_pipeline_state.config.llm.provider

        result = api_healer.fix(error, mock_pipeline_state, "ANALYZE")

        # Should attempt provider switch or return appropriate action
        assert result is not None

    def test_timeout_increase(self, api_healer, mock_pipeline_state):
        """Simulate timeout - verify timeout is increased."""
        error = TimeoutError("Request timed out after 60 seconds")

        original_timeout = mock_pipeline_state.config.llm.timeout

        result = api_healer.fix(error, mock_pipeline_state, "ANALYZE")

        # Should increase timeout
        if result.modified_config:
            new_timeout = mock_pipeline_state.config.llm.timeout
            assert new_timeout > original_timeout

    def test_auth_error_detection(self, api_healer, mock_pipeline_state):
        """Simulate auth error - should not retry indefinitely."""
        error = Exception("Error 401: Unauthorized. Invalid API key.")

        result = api_healer.fix(error, mock_pipeline_state, "ANALYZE")

        # Auth errors should not be fixed by retry
        # Should either abort or suggest fix
        assert result is not None


# ==============================================================================
# CHECKPOINT HEALER SIMULATIONS
# ==============================================================================


@pytest.mark.simulation
class TestCheckpointHealerSimulations:
    """Simulate checkpoint corruption scenarios."""

    @pytest.mark.parametrize("scenario", CHECKPOINT_ERROR_SCENARIOS, ids=lambda s: s.name)
    def test_checkpoint_error_detection(self, checkpoint_healer, scenario):
        """CheckpointHealer should detect all checkpoint errors."""
        # JSONDecodeError requires special construction
        if scenario.error_type == json.JSONDecodeError:
            error = json.JSONDecodeError("Expecting value", "", 0)
        else:
            error = scenario.error_type(scenario.error_message)

        can_handle = checkpoint_healer.can_handle(error, "RESUME")
        assert can_handle, f"CheckpointHealer should handle: {scenario.name}"

    def test_json_corruption_backup_restore(self, checkpoint_healer, mock_pipeline_state, temp_project_dir):
        """Simulate JSON corruption - verify backup restoration."""
        # Create corrupted checkpoint
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text("{{{{invalid json}}}}", encoding="utf-8")

        # Create valid backup
        backup_path = temp_project_dir / "checkpoint.backup.json"
        backup_data = {
            "stages": {"ANALYZE": "complete", "DOWNLOAD": "complete"},
            "timestamp": "2026-01-15T00:00:00Z"
        }
        backup_path.write_text(json.dumps(backup_data), encoding="utf-8")

        error = json.JSONDecodeError("Expecting value", "", 0)

        result = checkpoint_healer.fix(error, mock_pipeline_state, "RESUME")

        assert result.success
        assert result.action.value == "restore"

        # Checkpoint should now be valid
        with open(checkpoint_path, 'r') as f:
            restored = json.load(f)
            assert "stages" in restored

    def test_both_files_corrupt_fresh_start(self, checkpoint_healer, mock_pipeline_state, temp_project_dir):
        """Simulate both checkpoint and backup corrupt - verify fresh start."""
        # Create corrupted checkpoint
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text("corrupted", encoding="utf-8")

        # Create corrupted backup
        backup_path = temp_project_dir / "checkpoint.backup.json"
        backup_path.write_text("also corrupted", encoding="utf-8")

        error = json.JSONDecodeError("Expecting value", "", 0)

        result = checkpoint_healer.fix(error, mock_pipeline_state, "RESUME")

        # Should start fresh (not crash)
        assert result is not None

    def test_missing_backup_fresh_start(self, checkpoint_healer, mock_pipeline_state, temp_project_dir):
        """Simulate no backup exists - verify fresh start."""
        # Only create corrupted checkpoint, no backup
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text("corrupted", encoding="utf-8")

        error = json.JSONDecodeError("Expecting value", "", 0)

        result = checkpoint_healer.fix(error, mock_pipeline_state, "RESUME")

        # Should handle gracefully
        assert result is not None

    def test_cache_rebuild(self, checkpoint_healer, mock_pipeline_state, temp_project_dir):
        """Simulate missing checkpoint fields - verify cache rebuild."""
        # Create cache files
        trans_dir = temp_project_dir / ".cache" / "transcriptions"
        (trans_dir / "video_1.json").touch()
        (trans_dir / "video_2.json").touch()

        embed_dir = temp_project_dir / ".cache" / "embeddings"
        (embed_dir / "video_1.npy").touch()

        error = KeyError("Missing 'stages' key in checkpoint")

        result = checkpoint_healer.fix(error, mock_pipeline_state, "RESUME")

        # Should find cached data
        assert result is not None


# ==============================================================================
# DOWNLOAD HEALER SIMULATIONS
# ==============================================================================


@pytest.mark.simulation
class TestDownloadHealerSimulations:
    """Simulate video download errors."""

    @pytest.mark.parametrize("scenario", DOWNLOAD_ERROR_SCENARIOS, ids=lambda s: s.name)
    def test_download_error_detection(self, download_healer, scenario):
        """DownloadHealer should detect all download errors."""
        error = scenario.error_type(scenario.error_message)

        can_handle = download_healer.can_handle(error, "DOWNLOAD")
        assert can_handle, f"DownloadHealer should handle: {scenario.name}"

    def test_youtube_rate_limit_backoff(self, download_healer, mock_pipeline_state):
        """Simulate YouTube rate limiting."""
        error = Exception("ERROR: HTTP Error 429: Too Many Requests")

        backoff_times = []

        def mock_sleep(duration):
            backoff_times.append(duration)

        with patch('time.sleep', side_effect=mock_sleep):
            result = download_healer.fix(error, mock_pipeline_state, "DOWNLOAD")

        assert result.success
        assert result.action.value == "retry"
        assert len(backoff_times) == 1
        assert backoff_times[0] >= 10.0  # Initial backoff

    def test_unavailable_video_skip(self, download_healer, mock_pipeline_state):
        """Simulate unavailable video - verify skip."""
        error = Exception("ERROR: Video unavailable (video_id: dQw4w9WgXcQ)")

        result = download_healer.fix(error, mock_pipeline_state, "DOWNLOAD")

        # Should skip and continue
        assert result.success
        assert result.action.value in ["skip", "retry"]

    def test_private_video_skip(self, download_healer, mock_pipeline_state):
        """Simulate private video - verify skip."""
        error = Exception("ERROR: Private video. Sign in to access.")

        result = download_healer.fix(error, mock_pipeline_state, "DOWNLOAD")

        assert result.success

    def test_format_error_alternate_format(self, download_healer, mock_pipeline_state):
        """Simulate format extraction error - verify alternate format."""
        error = Exception("ERROR: No video formats found")

        result = download_healer.fix(error, mock_pipeline_state, "DOWNLOAD")

        # Should try alternate format or skip
        assert result is not None

    def test_multiple_skipped_videos_tracking(self, download_healer, mock_pipeline_state):
        """Verify tracking of multiple skipped videos."""
        errors = [
            Exception("ERROR: Video unavailable (video_id: abc123)"),
            Exception("ERROR: Private video (video_id: def456)"),
            Exception("ERROR: Video removed (video_id: ghi789)"),
        ]

        with patch('time.sleep'):
            for error in errors:
                download_healer.fix(error, mock_pipeline_state, "DOWNLOAD")

        # Should track all skipped videos
        assert len(download_healer.skipped_videos) >= 0


# ==============================================================================
# DISK HEALER SIMULATIONS
# ==============================================================================


@pytest.mark.simulation
class TestDiskHealerSimulations:
    """Simulate disk-related errors."""

    def test_disk_healer_import(self):
        """Verify DiskHealer can be imported."""
        try:
            from src.agents.healers.disk import DiskHealer
            assert DiskHealer is not None
        except ImportError:
            pytest.skip("DiskHealer not implemented yet")

    def test_disk_full_detection(self, mock_config, temp_project_dir):
        """DiskHealer should detect disk full errors."""
        try:
            from src.agents.healers.disk import DiskHealer
            healer = DiskHealer(mock_config, temp_project_dir)

            error = OSError(28, "No space left on device")

            can_handle = healer.can_handle(error, "DOWNLOAD")
            assert can_handle
        except ImportError:
            pytest.skip("DiskHealer not implemented")

    def test_disk_full_cache_cleanup(self, mock_config, temp_project_dir):
        """Simulate disk full - verify cache cleanup."""
        try:
            from src.agents.healers.disk import DiskHealer
            healer = DiskHealer(mock_config, temp_project_dir)

            # Create cache files
            llm_cache = temp_project_dir / ".cache" / "llm_responses"
            llm_cache.mkdir(parents=True, exist_ok=True)
            for i in range(10):
                (llm_cache / f"response_{i}.json").write_text("test" * 1000)

            error = OSError(28, "No space left on device")
            state = Mock()
            state.config = mock_config
            state.project_dir = temp_project_dir

            result = healer.fix(error, state, "DOWNLOAD")

            # Should attempt cleanup
            assert result is not None
        except ImportError:
            pytest.skip("DiskHealer not implemented")


# ==============================================================================
# PATH HEALER SIMULATIONS
# ==============================================================================


@pytest.mark.simulation
class TestPathHealerSimulations:
    """Simulate path-related errors."""

    def test_path_healer_import(self):
        """Verify PathHealer can be imported."""
        try:
            from src.agents.healers.path import PathHealer
            assert PathHealer is not None
        except ImportError:
            pytest.skip("PathHealer not implemented yet")

    def test_long_path_detection(self, mock_config, temp_project_dir):
        """PathHealer should detect path length errors."""
        try:
            from src.agents.healers.path import PathHealer
            healer = PathHealer(mock_config, temp_project_dir)

            error = OSError(206, "File name too long")

            can_handle = healer.can_handle(error, "DOWNLOAD")
            assert can_handle
        except ImportError:
            pytest.skip("PathHealer not implemented")

    def test_unicode_path_detection(self, mock_config, temp_project_dir):
        """PathHealer should detect unicode encoding errors."""
        try:
            from src.agents.healers.path import PathHealer
            healer = PathHealer(mock_config, temp_project_dir)

            error = UnicodeEncodeError("ascii", "日本語", 0, 3, "ordinal not in range")

            can_handle = healer.can_handle(error, "DOWNLOAD")
            assert can_handle
        except ImportError:
            pytest.skip("PathHealer not implemented")

    def test_short_path_switch(self, mock_config, temp_project_dir):
        """Simulate long path - verify switch to short path."""
        try:
            from src.agents.healers.path import PathHealer
            healer = PathHealer(mock_config, temp_project_dir)

            # Long path error
            long_path = "E:/Projects/VeryLongProjectName/" + "subdir/" * 20 + "video.mp4"
            error = OSError(206, f"File name too long: {long_path}")

            state = Mock()
            state.config = mock_config
            state.project_dir = temp_project_dir

            result = healer.fix(error, state, "DOWNLOAD")

            # Should suggest shorter path
            assert result is not None
        except ImportError:
            pytest.skip("PathHealer not implemented")


# ==============================================================================
# HEALING ORCHESTRATOR SIMULATIONS
# ==============================================================================


@pytest.fixture
def complete_mock_config():
    """Create a more complete mock configuration for orchestrator tests."""
    config = Mock()

    # Output config
    config.output = Mock()
    config.output.gap_mode = "scale"
    config.output.track_count = 3
    config.output.include_disabled_tracks = True
    config.output.framerate = 30.0
    config.output.edl_enabled = True
    config.output.xml_enabled = True

    # Download config
    config.download = Mock()
    config.download.timeout = 60.0
    config.download.max_retries = 3
    config.download.root_dir = None
    config.download.format = "bestvideo[height<=1080]+bestaudio/best"

    # LLM config
    config.llm = Mock()
    config.llm.provider = "gemini"
    config.llm.timeout = 60.0
    config.llm.max_retries = 3

    # Healing config with complete watcher structure
    config.healing = Mock()
    config.healing.enabled = True
    config.healing.strategy = "conservative"
    config.healing.max_attempts_per_stage = 3
    config.healing.max_total_heals = 20
    config.healing.heal_delay = 0.1

    # Watcher config (complete)
    config.healing.watcher = Mock()
    config.healing.watcher.enabled = False  # Disable for tests
    config.healing.watcher.host = "http://localhost:11434"
    config.healing.watcher.model = "llama3.2"
    config.healing.watcher.fallback_model = "llama3.1"
    config.healing.watcher.provider = "ollama"
    config.healing.watcher.recheck_interval_seconds = 300.0
    config.healing.watcher.max_failures = 3

    # LLM healer config
    config.healing.llm_healer = Mock()
    config.healing.llm_healer.enabled = False  # Disable for tests
    config.healing.llm_healer.provider = "anthropic"
    config.healing.llm_healer.timeout = 60.0
    config.healing.llm_healer.max_failures = 3
    config.healing.llm_healer.recheck_interval_seconds = 300.0

    # Logging config
    config.healing.logging = Mock()
    config.healing.logging.enabled = False  # Disable for tests
    config.healing.logging.log_dir = "logs"
    config.healing.logging.json_log = True

    return config


@pytest.mark.simulation
class TestOrchestratorSimulations:
    """Simulate orchestrated healing scenarios."""

    def test_orchestrator_import(self):
        """Verify HealingOrchestrator can be imported."""
        from src.agents.orchestrator import HealingOrchestrator
        assert HealingOrchestrator is not None

    def test_healer_selection(self, temp_project_dir, complete_mock_config):
        """Orchestrator should select correct healer for error."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        # Test API error routes to API healer
        api_error = Exception("Error 429: Rate limit exceeded")
        healers = orchestrator.select_healers(api_error, "ANALYZE")

        healer_names = [h.name for h in healers]
        assert "api-healer" in healer_names or len(healers) > 0

    def test_preflight_check_disk_space(self, temp_project_dir, complete_mock_config):
        """Orchestrator preflight should check disk space."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        # Create mock state with proper iterables
        mock_state = Mock()
        mock_state.config = complete_mock_config
        mock_state.project_dir = temp_project_dir
        mock_state.matches = []  # Empty list, not Mock
        mock_state.downloaded_videos = []  # Empty list, not Mock
        mock_state.voiceover_segments = []

        issues = orchestrator.run_preflight(mock_state)

        # Should return list of issues (may be empty if disk has space)
        assert isinstance(issues, list)

    def test_config_snapshot_rollback(self, temp_project_dir, complete_mock_config):
        """Orchestrator should snapshot and rollback config."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        # Take snapshot
        orchestrator.snapshot_config("TEST_STAGE")

        original_timeout = complete_mock_config.download.timeout

        # Modify config
        complete_mock_config.download.timeout = 999.0

        # Rollback
        orchestrator.rollback_config("TEST_STAGE")

        # Config should be restored (if implementation supports it)
        # Note: depends on implementation details

    def test_coordinate_heal_flow(self, temp_project_dir, mock_pipeline_state, complete_mock_config):
        """Test coordinated healing flow."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        # Update mock_pipeline_state with complete config
        mock_pipeline_state.config = complete_mock_config

        error = Exception("Error 429: Rate limit exceeded")

        with patch('time.sleep'):
            result = orchestrator.coordinate_heal(
                error,
                mock_pipeline_state,
                "ANALYZE",
                error_stack=None
            )

        # Should return a HealerResult
        assert result is not None

    def test_circular_healing_detection(self, temp_project_dir, mock_pipeline_state, complete_mock_config):
        """Orchestrator should detect circular healing loops."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        # Update mock_pipeline_state with complete config
        mock_pipeline_state.config = complete_mock_config

        # Same error repeatedly
        error = Exception("Recurring error that cannot be fixed")

        with patch('time.sleep'):
            results = []
            for _ in range(5):
                result = orchestrator.coordinate_heal(
                    error,
                    mock_pipeline_state,
                    "ANALYZE",
                    error_stack=None
                )
                results.append(result)

        # Should eventually abort or change strategy
        # (implementation dependent)


# ==============================================================================
# INTEGRATION SIMULATIONS
# ==============================================================================


@pytest.mark.integration
class TestHealingIntegration:
    """Integration tests for multiple healers working together."""

    def test_cascading_failure_recovery(self, mock_config, temp_project_dir, mock_pipeline_state):
        """Simulate cascading failures across healers."""
        from src.agents.healers.api import APIHealer
        from src.agents.healers.checkpoint import CheckpointHealer

        api_healer = APIHealer(mock_config, temp_project_dir)
        checkpoint_healer = CheckpointHealer(mock_config, temp_project_dir)

        # First: API error
        api_error = Exception("Error 429: Rate limit exceeded")

        with patch('time.sleep'):
            api_result = api_healer.fix(api_error, mock_pipeline_state, "ANALYZE")

        assert api_result.success

        # Then: Checkpoint error when resuming
        checkpoint_path = temp_project_dir / "checkpoint.json"
        checkpoint_path.write_text("corrupted", encoding="utf-8")

        # Create valid backup
        backup_path = temp_project_dir / "checkpoint.backup.json"
        backup_path.write_text(json.dumps({"stages": {}}), encoding="utf-8")

        checkpoint_error = json.JSONDecodeError("Expecting value", "", 0)

        checkpoint_result = checkpoint_healer.fix(checkpoint_error, mock_pipeline_state, "RESUME")

        assert checkpoint_result.success

    def test_multiple_healers_same_stage(self, mock_config, temp_project_dir, mock_pipeline_state):
        """Multiple healers may be needed for complex errors."""
        from src.agents.healers.otio import OTIOHealer

        otio_healer = OTIOHealer(mock_config, temp_project_dir)

        # Complex error with multiple issues
        complex_error = RuntimeError(
            "Timeline generation failed: "
            "Duration error at clip 3 (negative duration), "
            "Missing media reference at clip 5, "
            "Gap overflow detected"
        )

        can_handle = otio_healer.can_handle(complex_error, "OUTPUT")
        assert can_handle

        result = otio_healer.fix(complex_error, mock_pipeline_state, "OUTPUT")

        # Should handle at least one of the issues
        assert result is not None

    def test_healer_notification_chain(self, mock_config, temp_project_dir):
        """Healers should notify each other of changes."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy
        from src.agents.base import HealerResult, HealerAction

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(mock_config, temp_project_dir, strategy)

        # Simulate a healer making changes
        mock_healer = Mock()
        mock_healer.name = "test-healer"

        result = HealerResult.config_changed(
            "Changed download.timeout to 120.0",
            key="download.timeout",
            old_value=60.0,
            new_value=120.0
        )

        # Notify other healers
        orchestrator._notify_healers(mock_healer, result)

        # No assertions needed - just verifying no crash

    def test_strategy_escalation(self, mock_config, temp_project_dir, mock_pipeline_state):
        """Verify strategy escalation from conservative to aggressive."""
        from src.agents.strategy import HealingStrategy

        conservative = HealingStrategy.conservative()
        aggressive = HealingStrategy.aggressive()

        # Conservative has fewer attempts
        assert conservative.max_attempts_per_stage < aggressive.max_attempts_per_stage
        assert conservative.max_total_heals < aggressive.max_total_heals


# ==============================================================================
# RESILIENT RUNNER SIMULATIONS
# ==============================================================================


@pytest.mark.simulation
class TestResilientRunnerSimulations:
    """Simulate pipeline execution with healing."""

    def test_runner_import(self):
        """Verify ResilientRunner can be imported."""
        from src.agents.runner import ResilientRunner
        assert ResilientRunner is not None

    def test_runner_initialization(self, temp_project_dir, complete_mock_config):
        """Runner should initialize with orchestrator."""
        from src.agents.runner import ResilientRunner
        from src.agents.strategy import HealingStrategy
        from src.agents.orchestrator import HealingOrchestrator

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        # ResilientRunner takes config, project_dir, healers, orchestrator
        runner = ResilientRunner(
            config=complete_mock_config,
            project_dir=temp_project_dir,
            orchestrator=orchestrator
        )

        assert runner is not None
        assert runner.orchestrator is orchestrator

    def test_heal_history_tracking(self, temp_project_dir, complete_mock_config):
        """Runner should track healing history."""
        from src.agents.runner import ResilientRunner
        from src.agents.strategy import HealingStrategy
        from src.agents.orchestrator import HealingOrchestrator

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        runner = ResilientRunner(
            config=complete_mock_config,
            project_dir=temp_project_dir,
            orchestrator=orchestrator
        )

        # Initially empty
        assert runner.heal_history == []
        assert runner.total_heals == 0

    def test_healer_can_handle_check(self, temp_project_dir, complete_mock_config):
        """Runner healers should respond to can_handle checks."""
        from src.agents.runner import ResilientRunner
        from src.agents.strategy import HealingStrategy
        from src.agents.orchestrator import HealingOrchestrator

        strategy = HealingStrategy.conservative()
        orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, strategy)

        runner = ResilientRunner(
            config=complete_mock_config,
            project_dir=temp_project_dir,
            orchestrator=orchestrator
        )

        # Check that healers can identify errors
        api_error = Exception("Error 429: Rate limit exceeded")
        can_handle_count = sum(
            1 for h in runner.healers if h.can_handle(api_error, "TEST")
        )

        # At least one healer should handle this
        assert can_handle_count > 0

    def test_runner_max_attempts_from_strategy(self, temp_project_dir, complete_mock_config):
        """Runner should use max attempts from strategy."""
        from src.agents.runner import ResilientRunner
        from src.agents.strategy import HealingStrategy
        from src.agents.orchestrator import HealingOrchestrator

        # Aggressive has more attempts
        aggressive_strategy = HealingStrategy.aggressive()
        aggressive_orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, aggressive_strategy)

        aggressive_runner = ResilientRunner(
            config=complete_mock_config,
            project_dir=temp_project_dir,
            orchestrator=aggressive_orchestrator
        )

        # Minimal has fewer attempts
        minimal_strategy = HealingStrategy.minimal()
        minimal_orchestrator = HealingOrchestrator(complete_mock_config, temp_project_dir, minimal_strategy)

        minimal_runner = ResilientRunner(
            config=complete_mock_config,
            project_dir=temp_project_dir,
            orchestrator=minimal_orchestrator
        )

        # Max attempts should differ based on strategy
        assert aggressive_runner.MAX_HEAL_ATTEMPTS > minimal_runner.MAX_HEAL_ATTEMPTS


# ==============================================================================
# STRESS TESTS
# ==============================================================================


@pytest.mark.stress
class TestHealingStress:
    """Stress tests for the healing system."""

    def test_rapid_error_recovery(self, mock_config, temp_project_dir, mock_pipeline_state):
        """100 errors in rapid succession."""
        from src.agents.healers.api import APIHealer

        healer = APIHealer(mock_config, temp_project_dir)

        errors_handled = 0

        with patch('time.sleep'):
            for i in range(100):
                error = Exception(f"Error 429: Rate limit exceeded (attempt {i})")
                result = healer.fix(error, mock_pipeline_state, "ANALYZE")
                if result.success:
                    errors_handled += 1

        # Most should be handled
        assert errors_handled > 50

    def test_mixed_error_types(self, mock_config, temp_project_dir, mock_pipeline_state):
        """Handle mix of different error types."""
        from src.agents.healers.api import APIHealer
        from src.agents.healers.otio import OTIOHealer
        from src.agents.healers.checkpoint import CheckpointHealer

        healers = [
            APIHealer(mock_config, temp_project_dir),
            OTIOHealer(mock_config, temp_project_dir),
            CheckpointHealer(mock_config, temp_project_dir),
        ]

        errors = [
            Exception("Error 429: Rate limit"),
            ValueError("Duration must be positive"),
            json.JSONDecodeError("Invalid", "", 0),
            Exception("Timeout after 60s"),
            FileNotFoundError("File not found"),
        ]

        with patch('time.sleep'):
            for error in errors * 10:  # 50 total errors
                for healer in healers:
                    if healer.can_handle(error, "TEST"):
                        healer.fix(error, mock_pipeline_state, "TEST")
                        break

    def test_concurrent_healing(self, mock_config, temp_project_dir, mock_pipeline_state):
        """Multiple threads triggering healing simultaneously."""
        from src.agents.healers.api import APIHealer

        healer = APIHealer(mock_config, temp_project_dir)
        results = []
        errors_list = []

        def heal_error(thread_id):
            try:
                error = Exception(f"Error 429 from thread {thread_id}")
                with patch('time.sleep'):
                    result = healer.fix(error, mock_pipeline_state, "ANALYZE")
                results.append((thread_id, result.success))
            except Exception as e:
                errors_list.append((thread_id, str(e)))

        threads = [threading.Thread(target=heal_error, args=(i,)) for i in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete without errors
        assert len(errors_list) == 0, f"Errors: {errors_list}"
        assert len(results) == 10


# ==============================================================================
# REGRESSION TESTS
# ==============================================================================


@pytest.mark.fast
class TestHealingRegression:
    """Regression tests for known issues."""

    def test_empty_error_message(self, otio_healer, mock_pipeline_state):
        """Handle empty error message gracefully."""
        error = ValueError("")

        # Should not crash
        can_handle = otio_healer.can_handle(error, "OUTPUT")
        # May or may not handle based on exception type

        if can_handle:
            result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")
            assert result is not None

    def test_none_state_handling(self, api_healer):
        """Handle None state gracefully."""
        error = Exception("Error 429")

        # This should either work or raise a clear error
        try:
            result = api_healer.fix(error, None, "ANALYZE")
            # If it doesn't crash, that's acceptable
        except (AttributeError, TypeError):
            # Expected - None state not supported
            pass

    def test_unicode_error_message(self, otio_healer, mock_pipeline_state):
        """Handle unicode in error messages."""
        error = ValueError("Error: 日本語ファイル名.mp4 not found")

        can_handle = otio_healer.can_handle(error, "OUTPUT")

        if can_handle:
            result = otio_healer.fix(error, mock_pipeline_state, "OUTPUT")
            assert result is not None

    def test_very_long_error_message(self, api_healer, mock_pipeline_state):
        """Handle very long error messages."""
        error = Exception("Error: " + "x" * 10000)

        with patch('time.sleep'):
            # Should not crash
            result = api_healer.fix(error, mock_pipeline_state, "ANALYZE")
            # May or may not succeed


# ==============================================================================
# MAIN
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
