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

    # US-110-006: GPU memory tracking tests

    def test_record_gpu_memory(self):
        """Test recording GPU memory usage for videos (US-110-006)."""
        metrics = TranscriptionMetrics(total_videos=3)

        metrics.record_gpu_memory("video1.mp4", 512.0)
        metrics.record_gpu_memory("video2.mp4", 1024.0)

        assert metrics.video_gpu_memory_usage["video1.mp4"] == 512.0
        assert metrics.video_gpu_memory_usage["video2.mp4"] == 1024.0
        assert metrics.gpu_memory_peak_mb == 1024.0

    def test_gpu_memory_peak_tracking(self):
        """Test that peak GPU memory is correctly tracked (US-110-006)."""
        metrics = TranscriptionMetrics(total_videos=4)

        # Record memory in non-sequential order
        metrics.record_gpu_memory("video1.mp4", 256.0)
        metrics.record_gpu_memory("video2.mp4", 1024.0)
        metrics.record_gpu_memory("video3.mp4", 512.0)
        metrics.record_gpu_memory("video4.mp4", 768.0)

        # Peak should be the highest value
        assert metrics.gpu_memory_peak_mb == 1024.0

    def test_get_highest_memory_videos(self):
        """Test getting videos with highest GPU memory usage (US-110-006)."""
        metrics = TranscriptionMetrics(total_videos=4)

        metrics.record_gpu_memory("low_mem.mp4", 256.0)
        metrics.record_gpu_memory("high_mem.mp4", 2048.0)
        metrics.record_gpu_memory("medium_mem.mp4", 512.0)
        metrics.record_gpu_memory("top_mem.mp4", 1024.0)

        highest = metrics.get_highest_memory_videos(2)

        assert len(highest) == 2
        assert highest[0][0] == "high_mem.mp4"
        assert highest[0][1] == 2048.0
        assert highest[1][0] == "top_mem.mp4"
        assert highest[1][1] == 1024.0

    def test_summary_includes_gpu_memory(self):
        """Test that summary includes GPU memory info when present (US-110-006)."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_transcription("video1.mp4", 60.0, 20.0)
        metrics.record_gpu_memory("video1.mp4", 512.0)
        metrics.record_gpu_memory("video2.mp4", 1024.0)

        summary = metrics.summary()

        assert "GPU memory peak" in summary
        assert "1024.0 MB" in summary
        assert "Highest memory" in summary

    def test_summary_excludes_gpu_memory_when_zero(self):
        """Test that summary excludes GPU memory when not recorded (US-110-006)."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_transcription("video1.mp4", 60.0, 20.0)
        # No GPU memory recorded

        summary = metrics.summary()

        assert "GPU memory" not in summary

    def test_to_dict_includes_gpu_memory(self):
        """Test that to_dict includes GPU memory fields (US-110-006)."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_gpu_memory("video1.mp4", 512.0)
        metrics.record_gpu_memory("video2.mp4", 1024.0)

        data = metrics.to_dict()

        assert "gpu_memory_peak_mb" in data
        assert "video_gpu_memory_usage" in data
        assert data["gpu_memory_peak_mb"] == 1024.0
        assert data["video_gpu_memory_usage"]["video1.mp4"] == 512.0

    def test_from_dict_includes_gpu_memory(self):
        """Test that from_dict restores GPU memory fields (US-110-006)."""
        data = {
            "total_videos": 2,
            "gpu_memory_peak_mb": 1024.0,
            "video_gpu_memory_usage": {
                "video1.mp4": 512.0,
                "video2.mp4": 1024.0
            }
        }

        metrics = TranscriptionMetrics.from_dict(data)

        assert metrics.gpu_memory_peak_mb == 1024.0
        assert metrics.video_gpu_memory_usage["video1.mp4"] == 512.0
        assert metrics.video_gpu_memory_usage["video2.mp4"] == 1024.0

    def test_get_summary_dict_includes_gpu_memory(self):
        """Test that get_summary_dict includes GPU memory (US-110-006)."""
        metrics = TranscriptionMetrics(total_videos=3)

        metrics.record_gpu_memory("video1.mp4", 512.0)
        metrics.record_gpu_memory("video2.mp4", 1024.0)

        summary = metrics.get_summary_dict()

        assert "gpu_memory_peak_mb" in summary
        assert summary["gpu_memory_peak_mb"] == 1024.0


class TestTranscriptionQualityMetrics:
    """Tests for transcription quality metrics (US-110-007)."""

    def test_record_confidence_basic(self):
        """Test basic confidence recording."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_confidence(
            "video1.mp4",
            avg_word_confidence=0.85,
            min_segment_confidence=0.72
        )

        assert metrics.avg_word_confidence == 0.85
        assert metrics.min_segment_confidence == 0.72
        assert "video1.mp4" in metrics.video_confidences

    def test_record_confidence_updates_min(self):
        """Test that min_segment_confidence tracks lowest value."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_confidence("video1.mp4", 0.85, 0.80)
        metrics.record_confidence("video2.mp4", 0.90, 0.65)

        assert metrics.min_segment_confidence == 0.65

    def test_record_confidence_multiple_videos_weighted(self):
        """Test weighted average for multiple videos."""
        metrics = TranscriptionMetrics(total_videos=3)

        # Video durations affect the weighted average
        metrics.record_transcription("video1.mp4", 100.0, 10.0)  # 10s transcription
        metrics.record_transcription("video2.mp4", 200.0, 20.0)  # 20s transcription
        metrics.record_confidence("video1.mp4", 0.80, 0.70)
        metrics.record_confidence("video2.mp4", 0.90, 0.80)

        # Weighted avg: (0.80 * 100 + 0.90 * 200) / (100 + 200) = 260 / 300 = 0.867
        assert metrics.avg_word_confidence == pytest.approx(0.867, rel=0.01)

    def test_to_dict_includes_quality_metrics(self):
        """Test that to_dict includes quality metrics (US-110-007)."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_confidence("video1.mp4", 0.85, 0.72)
        data = metrics.to_dict()

        assert "avg_word_confidence" in data
        assert "min_segment_confidence" in data
        assert "video_confidences" in data
        assert data["avg_word_confidence"] == 0.85
        assert data["min_segment_confidence"] == 0.72

    def test_from_dict_restores_quality_metrics(self):
        """Test that from_dict restores quality metrics (US-110-007)."""
        data = {
            "total_videos": 2,
            "avg_word_confidence": 0.85,
            "min_segment_confidence": 0.72,
            "video_confidences": {
                "video1.mp4": {"avg_word_confidence": 0.85, "min_segment_confidence": 0.72}
            }
        }

        metrics = TranscriptionMetrics.from_dict(data)

        assert metrics.avg_word_confidence == 0.85
        assert metrics.min_segment_confidence == 0.72
        assert "video1.mp4" in metrics.video_confidences

    def test_get_summary_dict_includes_quality(self):
        """Test that get_summary_dict includes quality metrics (US-110-007)."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_confidence("video1.mp4", 0.85, 0.72)
        summary = metrics.get_summary_dict()

        assert "avg_word_confidence" in summary
        assert "min_segment_confidence" in summary
        assert summary["avg_word_confidence"] == 0.85
        assert summary["min_segment_confidence"] == 0.72

    def test_summary_output_includes_quality(self):
        """Test that summary() includes quality metrics (US-110-007)."""
        metrics = TranscriptionMetrics(total_videos=2)

        metrics.record_transcription("video1.mp4", 60.0, 30.0)
        metrics.record_confidence("video1.mp4", 0.85, 0.72)

        summary_text = metrics.summary()

        assert "Avg word confidence" in summary_text
        assert "85" in summary_text or "0.85" in summary_text
        assert "Min segment confidence" in summary_text

    def test_quality_warning_on_low_confidence(self, caplog):
        """Test that warning is logged when confidence is below 0.7 (US-110-007)."""
        import logging
        caplog.set_level(logging.WARNING)

        metrics = TranscriptionMetrics(total_videos=1)

        # Record low confidence (< 0.7 threshold)
        metrics.record_confidence("video1.mp4", avg_word_confidence=0.65, min_segment_confidence=0.50)

        # Check warning was logged
        assert any("Low transcription quality" in record.message for record in caplog.records)

    def test_no_warning_on_good_confidence(self, caplog):
        """Test that no warning is logged when confidence is above 0.7."""
        import logging
        caplog.set_level(logging.WARNING)

        metrics = TranscriptionMetrics(total_videos=1)

        # Record good confidence (>= 0.7 threshold)
        metrics.record_confidence("video1.mp4", avg_word_confidence=0.85, min_segment_confidence=0.72)

        # Check no warning was logged
        assert not any("Low transcription quality" in record.message for record in caplog.records)

    def test_quality_metrics_default_values(self):
        """Test default values for quality metrics."""
        metrics = TranscriptionMetrics()

        assert metrics.avg_word_confidence == 0.0
        assert metrics.min_segment_confidence == 1.0
        assert metrics.video_confidences == {}


class TestTranscriptionMetricsExport:
    """Tests for transcription metrics export (US-137-003)."""

    def test_export_json_returns_dict(self):
        """Test that export_json returns dict when no path provided."""
        import json

        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)
        metrics.record_cache_hit("video2.mp4")

        result = metrics.export_json()

        assert isinstance(result, dict)
        assert "format" in result
        assert result["format"] == "transcription_metrics_json"
        assert "summary" in result
        assert "details" in result

    def test_export_json_to_file(self, tmp_path):
        """Test export_json writes to file."""
        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)

        output_path = tmp_path / "metrics.json"
        result = metrics.export_json(output_path)

        assert output_path.exists()
        assert result == str(output_path)

        # Verify contents
        import json
        with open(output_path) as f:
            data = json.load(f)
        assert data["format"] == "transcription_metrics_json"
        assert data["summary"]["transcribed_count"] == 1

    def test_export_prometheus_returns_string(self):
        """Test that export_prometheus returns string when no path provided."""
        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)

        result = metrics.export_prometheus()

        assert isinstance(result, str)
        assert "transcription_total_videos" in result
        assert "transcription_transcribed_count" in result
        assert "# HELP" in result
        assert "# TYPE" in result

    def test_export_prometheus_to_file(self, tmp_path):
        """Test export_prometheus writes to file."""
        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)

        output_path = tmp_path / "metrics.prom"
        result = metrics.export_prometheus(output_path)

        assert output_path.exists()
        assert result == str(output_path)

        # Verify contents
        with open(output_path) as f:
            content = f.read()
        assert "transcription_total_videos" in content
        assert "transcription_transcribed_count 1" in content

    def test_export_csv_returns_string(self):
        """Test that export_csv returns string when no path provided."""
        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)

        result = metrics.export_csv()

        assert isinstance(result, str)
        assert "metric,value" in result
        assert "total_videos,3" in result

    def test_export_csv_to_file(self, tmp_path):
        """Test export_csv writes to file."""
        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)
        metrics.record_transcription("video2.mp4", 120.0, 40.0)

        output_path = tmp_path / "metrics.csv"
        result = metrics.export_csv(output_path)

        assert output_path.exists()
        assert result == str(output_path)

        # Verify main CSV contents
        with open(output_path) as f:
            content = f.read()
        assert "metric,value" in content
        assert "total_videos,3" in content
        assert "transcribed_count,2" in content

    def test_export_csv_creates_per_video_files(self, tmp_path):
        """Test that export_csv creates per-video CSV files."""
        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)
        metrics.record_transcription("video2.mp4", 120.0, 40.0)

        output_path = tmp_path / "metrics.csv"
        metrics.export_csv(output_path)

        # Check for per-video CSV
        video_csv = tmp_path / "metrics_videos.csv"
        assert video_csv.exists()

        # Check GPU memory CSV
        gpu_csv = tmp_path / "metrics_gpu_memory.csv"
        # Should not exist since no GPU memory was recorded
        assert not gpu_csv.exists()

    def test_export_csv_with_gpu_memory(self, tmp_path):
        """Test that export_csv creates GPU memory CSV when memory recorded."""
        metrics = TranscriptionMetrics(total_videos=2)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)
        metrics.record_gpu_memory("video1.mp4", 512.0)

        output_path = tmp_path / "metrics.csv"
        metrics.export_csv(output_path)

        # Check GPU memory CSV exists
        gpu_csv = tmp_path / "metrics_gpu_memory.csv"
        assert gpu_csv.exists()

        with open(gpu_csv) as f:
            content = f.read()
        assert "video_path" in content
        assert "gpu_memory_mb" in content

    def test_export_json_includes_all_fields(self, tmp_path):
        """Test that export_json includes all relevant fields."""
        metrics = TranscriptionMetrics(total_videos=3)
        metrics.record_transcription("video1.mp4", 60.0, 30.0)
        metrics.record_cache_hit("video2.mp4")
        metrics.record_failure("video3.mp4")
        metrics.set_phase_times(10.0, 20.0)
        metrics.record_gpu_memory("video1.mp4", 512.0)
        metrics.record_confidence("video1.mp4", 0.85, 0.72)

        result = metrics.export_json()

        assert "summary" in result
        summary = result["summary"]
        assert summary["total_videos"] == 3
        assert summary["cached_hits"] == 1
        assert summary["transcribed_count"] == 1
        assert summary["failed_count"] == 1
        assert summary["phase1_time_s"] == 10.0
        assert summary["phase2_time_s"] == 20.0
        assert summary["gpu_memory_peak_mb"] == 512.0

    def test_export_prometheus_includes_budget_metrics(self):
        """Test that export_prometheus includes budget metrics."""
        metrics = TranscriptionMetrics(total_videos=5)
        metrics.set_retry_budget_summary({
            "total_attempts": 10,
            "failed_attempts": 2,
            "videos_skipped": 1
        })

        result = metrics.export_prometheus()

        assert "transcription_budget_total_attempts" in result
        assert "transcription_budget_failed_attempts" in result
        assert "transcription_budget_exhausted_count" in result
