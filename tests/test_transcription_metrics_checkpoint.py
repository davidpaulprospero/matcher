"""
Tests for transcription metrics persistence to checkpoint (US-79-012).

Covers:
- AC4: Transcription metrics are persisted in checkpoint after batch processing
- AC5: Resumed pipeline logs previous transcription metrics from checkpoint
- AC1-3: Summary includes required fields (total_videos, cached_videos,
         transcribed_videos, failed_videos, total_duration_seconds, average_speed_ratio)
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData
from src.transcription.metrics import TranscriptionMetrics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def checkpoint_manager(tmp_path):
    """Create a checkpoint manager with a temp directory."""
    return CheckpointManager(tmp_path, config_hash="test_hash")


@pytest.fixture
def sample_transcription_metrics():
    """Create a sample TranscriptionMetrics with realistic data."""
    metrics = TranscriptionMetrics(total_videos=10)
    metrics.record_cache_hit("cached1.mp4")
    metrics.record_cache_hit("cached2.mp4")
    metrics.record_cache_hit("cached3.mp4")
    metrics.record_transcription("video1.mp4", 120.0, 40.0)
    metrics.record_transcription("video2.mp4", 60.0, 20.0)
    metrics.record_transcription("video3.mp4", 90.0, 30.0)
    metrics.record_failure("failed1.mp4")
    metrics.set_phase_times(10.0, 90.0)
    return metrics


@pytest.fixture
def sample_metrics_summary(sample_transcription_metrics):
    """Get a summary dict from sample metrics, mapped to checkpoint field names."""
    summary = sample_transcription_metrics.get_summary_dict()
    return {
        'total_videos': summary['total_videos'],
        'cached_videos': summary['cached_hits'],
        'transcribed_videos': summary['transcribed_count'],
        'failed_videos': summary['failed_count'],
        'total_duration_seconds': summary['total_duration_s'],
        'average_speed_ratio': summary['avg_speed_ratio'],
        'cache_hit_rate': summary['cache_hit_rate'],
        'success_rate': summary['success_rate'],
        'phase1_time_s': summary['phase1_time_s'],
        'phase2_time_s': summary['phase2_time_s'],
    }


# ============================================================================
# AC4: Transcription metrics are persisted in checkpoint after batch processing
# ============================================================================

class TestTranscriptionMetricsCheckpointPersistence:
    """Test that transcription metrics are saved to and loaded from checkpoint."""

    @pytest.mark.fast
    def test_save_transcription_metrics_persists_to_disk(
        self, checkpoint_manager, sample_metrics_summary
    ):
        """Verify save_transcription_metrics writes to checkpoint.json on disk."""
        # Save metrics
        checkpoint_manager.save_transcription_metrics(sample_metrics_summary)

        # Verify file exists
        assert checkpoint_manager.checkpoint_path.exists()

        # Load raw JSON and verify transcription_metrics key
        with open(checkpoint_manager.checkpoint_path, 'r') as f:
            data = json.load(f)

        assert 'transcription_metrics' in data
        tm = data['transcription_metrics']
        assert tm['total_videos'] == 10
        assert tm['cached_videos'] == 3
        assert tm['transcribed_videos'] == 3
        assert tm['failed_videos'] == 1
        assert tm['total_duration_seconds'] == 270.0
        assert tm['average_speed_ratio'] == pytest.approx(3.0)

    @pytest.mark.fast
    def test_transcription_metrics_survive_checkpoint_roundtrip(
        self, checkpoint_manager, sample_metrics_summary, tmp_path
    ):
        """Verify metrics survive save -> load cycle."""
        # First save some stage data so checkpoint is valid
        checkpoint_manager.save("ANALYZE", {"keywords": ["test"]})

        # Then save transcription metrics
        checkpoint_manager.save_transcription_metrics(sample_metrics_summary)

        # Create new manager pointing to same dir and load
        new_manager = CheckpointManager(tmp_path, config_hash="test_hash")
        loaded = new_manager.load()

        assert loaded is not None
        assert loaded.transcription_metrics is not None
        assert loaded.transcription_metrics['total_videos'] == 10
        assert loaded.transcription_metrics['cached_videos'] == 3
        assert loaded.transcription_metrics['transcribed_videos'] == 3
        assert loaded.transcription_metrics['failed_videos'] == 1
        assert loaded.transcription_metrics['total_duration_seconds'] == 270.0
        assert loaded.transcription_metrics['average_speed_ratio'] == pytest.approx(3.0)

    @pytest.mark.fast
    def test_summary_includes_required_fields(self, sample_metrics_summary):
        """AC2: Summary must include all required fields from acceptance criteria."""
        required_fields = [
            'total_videos',
            'cached_videos',
            'transcribed_videos',
            'failed_videos',
            'total_duration_seconds',
            'average_speed_ratio',
        ]
        for field in required_fields:
            assert field in sample_metrics_summary, f"Missing required field: {field}"

    @pytest.mark.fast
    def test_get_transcription_metrics_empty_when_not_set(self, checkpoint_manager):
        """get_transcription_metrics returns empty dict when no metrics saved."""
        assert checkpoint_manager.get_transcription_metrics() == {}

    @pytest.mark.fast
    def test_get_transcription_metrics_after_save(
        self, checkpoint_manager, sample_metrics_summary
    ):
        """get_transcription_metrics returns saved data after save."""
        checkpoint_manager.save_transcription_metrics(sample_metrics_summary)
        result = checkpoint_manager.get_transcription_metrics()
        assert result['total_videos'] == 10
        assert result['cached_videos'] == 3

    @pytest.mark.fast
    def test_checkpoint_summary_includes_transcription_metrics(
        self, checkpoint_manager, sample_metrics_summary
    ):
        """Verify get_summary() includes transcription metrics line."""
        checkpoint_manager.save("ANALYZE", {"keywords": ["test"]})
        checkpoint_manager.save_transcription_metrics(sample_metrics_summary)

        summary = checkpoint_manager.get_summary()
        assert "Transcription:" in summary
        assert "3 transcribed" in summary
        assert "3 cached" in summary
        assert "1 failed" in summary
        assert "realtime" in summary


# ============================================================================
# AC5: Resumed pipeline logs previous transcription metrics from checkpoint
# ============================================================================

class TestTranscriptionMetricsResumeLogging:
    """Test that loading a checkpoint with transcription metrics logs them."""

    @pytest.mark.fast
    def test_load_logs_previous_transcription_metrics(
        self, checkpoint_manager, sample_metrics_summary, tmp_path, caplog
    ):
        """When resuming from checkpoint, previous transcription metrics are logged at INFO."""
        # Save checkpoint with transcription metrics
        checkpoint_manager.save("ANALYZE", {"keywords": ["test"]})
        checkpoint_manager.save_transcription_metrics(sample_metrics_summary)

        # Create new manager and load (simulating resume)
        new_manager = CheckpointManager(tmp_path, config_hash="test_hash")
        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            new_manager.load()

        # Verify the INFO log message about previous metrics
        log_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        metric_logs = [m for m in log_messages if "Previous transcription metrics" in m]

        assert len(metric_logs) == 1, f"Expected 1 'Previous transcription metrics' log, got {len(metric_logs)}"
        log_msg = metric_logs[0]
        assert "10 videos" in log_msg
        assert "3 cached" in log_msg
        assert "3 transcribed" in log_msg
        assert "1 failed" in log_msg
        assert "3.0x realtime" in log_msg

    @pytest.mark.fast
    def test_load_without_transcription_metrics_no_log(
        self, checkpoint_manager, tmp_path, caplog
    ):
        """When checkpoint has no transcription metrics, no 'Previous' log emitted."""
        # Save checkpoint WITHOUT transcription metrics
        checkpoint_manager.save("ANALYZE", {"keywords": ["test"]})

        # Create new manager and load
        new_manager = CheckpointManager(tmp_path, config_hash="test_hash")
        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            new_manager.load()

        log_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        metric_logs = [m for m in log_messages if "Previous transcription metrics" in m]
        assert len(metric_logs) == 0, "Should not log previous metrics when none exist"

    @pytest.mark.fast
    def test_log_previous_transcription_metrics_directly(
        self, checkpoint_manager, sample_metrics_summary, caplog
    ):
        """Test log_previous_transcription_metrics method directly."""
        checkpoint_manager.save_transcription_metrics(sample_metrics_summary)

        with caplog.at_level(logging.INFO, logger="src.checkpoint"):
            checkpoint_manager.log_previous_transcription_metrics()

        log_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        metric_logs = [m for m in log_messages if "Previous transcription metrics" in m]
        assert len(metric_logs) >= 1


# ============================================================================
# Integration: transcribe_videos_parallel persists metrics via checkpoint param
# ============================================================================

class TestTranscribeVideosParallelCheckpointIntegration:
    """Test that transcribe_videos_parallel saves metrics when checkpoint is provided."""

    @pytest.mark.fast
    def test_parallel_processor_saves_metrics_to_checkpoint(self, tmp_path):
        """When checkpoint param is provided, metrics are persisted after batch."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        # Create a real checkpoint manager
        cm = CheckpointManager(tmp_path, config_hash="test")

        mock_cache = MagicMock()
        mock_cache.cache_dir = str(tmp_path / "cache")

        # Mock TranscriptCache to return cache hits for all videos
        mock_transcript_cache = MagicMock()
        mock_transcript_cache_instance = MagicMock()
        mock_transcript_cache_instance._source_map = {}
        mock_transcript_cache_instance.get.return_value = [
            {'start': 0.0, 'end': 1.0, 'text': 'test'}
        ]
        mock_transcript_cache.return_value = mock_transcript_cache_instance

        with patch("src.transcription.parallel_processor.TranscriptCache", mock_transcript_cache):
            with patch("src.transcription.parallel_processor.WhisperClient"):
                with patch("src.transcription.parallel_processor.DeltaAwareIndex") as mock_delta:
                    mock_delta.return_value.check_staleness.return_value = (False, 0, 0)
                    transcribe_videos_parallel(
                        video_paths=["video1.mp4", "video2.mp4"],
                        cache=mock_cache,
                        config=None,
                        checkpoint=cm,
                    )

        # Verify metrics were persisted
        metrics = cm.get_transcription_metrics()
        assert metrics != {}
        assert metrics['total_videos'] == 2
        assert metrics['cached_videos'] == 2  # Both were cache hits
        assert metrics['transcribed_videos'] == 0
        assert metrics['failed_videos'] == 0

    @pytest.mark.fast
    def test_parallel_processor_no_checkpoint_no_error(self, tmp_path):
        """When checkpoint=None, no error raised (backward compatible)."""
        from src.transcription.parallel_processor import transcribe_videos_parallel

        mock_cache = MagicMock()
        mock_cache.cache_dir = str(tmp_path / "cache")

        mock_transcript_cache = MagicMock()
        mock_transcript_cache_instance = MagicMock()
        mock_transcript_cache_instance._source_map = {}
        mock_transcript_cache_instance.get.return_value = [
            {'start': 0.0, 'end': 1.0, 'text': 'test'}
        ]
        mock_transcript_cache.return_value = mock_transcript_cache_instance

        with patch("src.transcription.parallel_processor.TranscriptCache", mock_transcript_cache):
            with patch("src.transcription.parallel_processor.WhisperClient"):
                with patch("src.transcription.parallel_processor.DeltaAwareIndex") as mock_delta:
                    mock_delta.return_value.check_staleness.return_value = (False, 0, 0)
                    # Should not raise - checkpoint=None is the default
                    result = transcribe_videos_parallel(
                        video_paths=["video1.mp4"],
                        cache=mock_cache,
                        config=None,
                        checkpoint=None,
                    )

        assert isinstance(result, dict)
