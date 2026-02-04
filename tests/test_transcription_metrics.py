"""
Tests for TranscriptionMetrics dataclass.

Tests US-60-009: Add batch transcription progress metrics.
"""

import pytest
from src.transcription.metrics import TranscriptionMetrics


class TestTranscriptionMetrics:
    """Tests for TranscriptionMetrics dataclass."""

    def test_empty_metrics(self):
        """Test that empty metrics have sensible defaults."""
        metrics = TranscriptionMetrics()

        assert metrics.total_videos == 0
        assert metrics.cached_hits == 0
        assert metrics.transcribed_count == 0
        assert metrics.failed_count == 0
        assert metrics.total_audio_duration_s == 0.0
        assert metrics.total_transcription_time_s == 0.0
        assert metrics.avg_speed_ratio == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.success_rate == 0.0

    def test_record_cache_hit(self):
        """Test recording cache hits."""
        metrics = TranscriptionMetrics(total_videos=3)

        metrics.record_cache_hit("video1.mp4")
        metrics.record_cache_hit("video2.mp4")

        assert metrics.cached_hits == 2
        assert metrics.cache_hit_rate == pytest.approx(100.0)  # 2/2 = 100%

    def test_record_transcription(self):
        """Test recording transcription with timing."""
        metrics = TranscriptionMetrics(total_videos=2)

        # Transcribe 60s of audio in 30s (2x realtime)
        metrics.record_transcription("video1.mp4", audio_duration_s=60.0, transcription_time_s=30.0)

        assert metrics.transcribed_count == 1
        assert metrics.total_audio_duration_s == 60.0
        assert metrics.total_transcription_time_s == 30.0
        assert metrics.avg_speed_ratio == pytest.approx(2.0)

    def test_speed_ratio_calculation(self):
        """Test real-time speed ratio calculation accuracy."""
        metrics = TranscriptionMetrics(total_videos=3)

        # First video: 120s audio in 40s (3x)
        metrics.record_transcription("video1.mp4", 120.0, 40.0)

        # Second video: 60s audio in 30s (2x)
        metrics.record_transcription("video2.mp4", 60.0, 30.0)

        # Third video: 30s audio in 15s (2x)
        metrics.record_transcription("video3.mp4", 30.0, 15.0)

        # Total: 210s audio in 85s = 2.47x
        assert metrics.total_audio_duration_s == 210.0
        assert metrics.total_transcription_time_s == 85.0
        assert metrics.avg_speed_ratio == pytest.approx(210.0 / 85.0, rel=0.01)

    def test_record_failure(self):
        """Test recording failed transcriptions."""
        metrics = TranscriptionMetrics(total_videos=3)

        metrics.record_transcription("video1.mp4", 60.0, 30.0)
        metrics.record_failure("video2.mp4")

        assert metrics.transcribed_count == 1
        assert metrics.failed_count == 1
        assert metrics.success_rate == pytest.approx(50.0)  # 1/2 = 50%

    def test_success_rate_all_cached(self):
        """Test success rate when all videos are cached."""
        metrics = TranscriptionMetrics(total_videos=3)

        metrics.record_cache_hit("video1.mp4")
        metrics.record_cache_hit("video2.mp4")
        metrics.record_cache_hit("video3.mp4")

        # When all cached with no transcription attempts, success rate is 100%
        assert metrics.success_rate == pytest.approx(100.0)

    def test_set_phase_times(self):
        """Test setting phase timing."""
        metrics = TranscriptionMetrics(total_videos=5)

        metrics.set_phase_times(phase1_time=10.5, phase2_time=45.3)

        assert metrics.phase1_extraction_time_s == 10.5
        assert metrics.phase2_transcription_time_s == 45.3

    def test_get_summary_dict(self):
        """Test get_summary_dict returns correct structure."""
        metrics = TranscriptionMetrics(total_videos=5)

        metrics.record_cache_hit("cached1.mp4")
        metrics.record_transcription("video1.mp4", 60.0, 20.0)
        metrics.record_transcription("video2.mp4", 120.0, 40.0)
        metrics.record_failure("failed1.mp4")
        metrics.set_phase_times(5.0, 60.0)

        summary = metrics.get_summary_dict()

        assert summary['total_videos'] == 5
        assert summary['cached_hits'] == 1
        assert summary['transcribed_count'] == 2
        assert summary['failed_count'] == 1
        assert summary['total_duration_s'] == 180.0
        assert summary['total_time_s'] == 60.0
        assert summary['avg_speed_ratio'] == pytest.approx(3.0)
        assert summary['cache_hit_rate'] == pytest.approx(25.0)  # 1/4 = 25%
        assert summary['success_rate'] == pytest.approx(66.7, rel=0.1)  # 2/3
        assert summary['phase1_time_s'] == 5.0
        assert summary['phase2_time_s'] == 60.0

    def test_get_slowest_videos(self):
        """Test getting slowest videos by transcription time."""
        metrics = TranscriptionMetrics(total_videos=4)

        metrics.record_transcription("fast.mp4", 30.0, 10.0)
        metrics.record_transcription("medium.mp4", 60.0, 25.0)
        metrics.record_transcription("slow.mp4", 120.0, 60.0)
        metrics.record_transcription("slowest.mp4", 90.0, 90.0)

        slowest = metrics.get_slowest_videos(2)

        assert len(slowest) == 2
        assert slowest[0][0] == "slowest.mp4"
        assert slowest[0][1] == 90.0
        assert slowest[1][0] == "slow.mp4"
        assert slowest[1][1] == 60.0

    def test_summary_string(self):
        """Test human-readable summary generation."""
        metrics = TranscriptionMetrics(total_videos=3)

        metrics.record_cache_hit("cached.mp4")
        metrics.record_transcription("video1.mp4", 60.0, 20.0)
        metrics.record_failure("failed.mp4")
        metrics.set_phase_times(5.0, 20.0)

        summary = metrics.summary()

        assert "3 videos" in summary
        assert "1 cached" in summary
        assert "1 transcribed" in summary
        assert "1 failed" in summary
        assert "realtime" in summary
        assert "Phase 1" in summary
        assert "Phase 2" in summary

    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        metrics = TranscriptionMetrics(total_videos=4)

        metrics.record_cache_hit("cached.mp4")
        metrics.record_transcription("video1.mp4", 60.0, 20.0)
        metrics.record_transcription("video2.mp4", 90.0, 30.0)
        metrics.record_failure("failed.mp4")
        metrics.set_phase_times(10.0, 50.0)

        # Serialize to dict
        data = metrics.to_dict()

        # Deserialize back
        restored = TranscriptionMetrics.from_dict(data)

        assert restored.total_videos == metrics.total_videos
        assert restored.cached_hits == metrics.cached_hits
        assert restored.transcribed_count == metrics.transcribed_count
        assert restored.failed_count == metrics.failed_count
        assert restored.total_audio_duration_s == metrics.total_audio_duration_s
        assert restored.total_transcription_time_s == metrics.total_transcription_time_s
        assert restored.phase1_extraction_time_s == metrics.phase1_extraction_time_s
        assert restored.phase2_transcription_time_s == metrics.phase2_transcription_time_s
        assert restored.video_durations == metrics.video_durations
        assert restored.video_transcription_times == metrics.video_transcription_times

    def test_from_dict_empty(self):
        """Test from_dict with empty data returns empty metrics."""
        metrics = TranscriptionMetrics.from_dict({})

        assert metrics.total_videos == 0
        assert metrics.cached_hits == 0

        metrics2 = TranscriptionMetrics.from_dict(None)
        assert metrics2.total_videos == 0

    def test_total_processed_property(self):
        """Test total_processed calculation."""
        metrics = TranscriptionMetrics(total_videos=5)

        metrics.record_cache_hit("cached.mp4")
        metrics.record_transcription("video.mp4", 60.0, 20.0)
        metrics.record_failure("failed.mp4")

        # total_processed = cached + transcribed + failed
        assert metrics.total_processed == 3
