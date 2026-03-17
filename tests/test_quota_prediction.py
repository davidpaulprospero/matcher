"""Tests for QuotaPredictor historical prediction features (US-156-009).

Tests the load_history, predict_daily_usage, time_of_day factor,
get_prediction_confidence, and save_usage_record features.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime, timedelta
from src.downloader.youtube_retry_budget import QuotaPredictor


class TestQuotaPredictorLoadHistory:
    """Tests for load_history method."""

    def test_load_history_empty_file(self, tmp_path):
        """Test loading history from non-existent file returns empty list."""
        predictor = QuotaPredictor()
        predictor._history_file = str(tmp_path / "nonexistent.json")

        history = predictor.load_history()

        assert history == []

    def test_load_history_valid_file(self, tmp_path):
        """Test loading history from valid JSON file."""
        history_file = tmp_path / "quota_history.json"
        test_data = [
            {"date": "2026-02-20", "total_usage": 5000, "search_used": 3000, "caption_used": 1500, "metadata_used": 500},
            {"date": "2026-02-19", "total_usage": 4500, "search_used": 2500, "caption_used": 1500, "metadata_used": 500},
        ]
        history_file.write_text(json.dumps(test_data))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        history = predictor.load_history()

        assert len(history) == 2
        assert history[0]["total_usage"] == 5000

    def test_load_history_with_wrapper_format(self, tmp_path):
        """Test loading history with wrapped format (dict with 'history' key)."""
        history_file = tmp_path / "quota_history.json"
        test_data = {
            "history": [
                {"date": "2026-02-20", "total_usage": 5000},
                {"date": "2026-02-19", "total_usage": 4500},
            ]
        }
        history_file.write_text(json.dumps(test_data))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        history = predictor.load_history()

        assert len(history) == 2


class TestQuotaPredictorTimeOfDay:
    """Tests for time_of_day factor in predictions."""

    def test_get_time_period_morning(self):
        """Test _get_time_period returns 'morning' for hours 6-12."""
        predictor = QuotaPredictor()

        assert predictor._get_time_period(6) == "morning"
        assert predictor._get_time_period(9) == "morning"
        assert predictor._get_time_period(11) == "morning"

    def test_get_time_period_afternoon(self):
        """Test _get_time_period returns 'afternoon' for hours 12-18."""
        predictor = QuotaPredictor()

        assert predictor._get_time_period(12) == "afternoon"
        assert predictor._get_time_period(15) == "afternoon"
        assert predictor._get_time_period(17) == "afternoon"

    def test_get_time_period_evening(self):
        """Test _get_time_period returns 'evening' for hours 18-22."""
        predictor = QuotaPredictor()

        assert predictor._get_time_period(18) == "evening"
        assert predictor._get_time_period(20) == "evening"
        assert predictor._get_time_period(21) == "evening"

    def test_get_time_period_overnight(self):
        """Test _get_time_period returns 'overnight' for hours 22-6."""
        predictor = QuotaPredictor()

        assert predictor._get_time_period(22) == "overnight"
        assert predictor._get_time_period(23) == "overnight"
        assert predictor._get_time_period(0) == "overnight"
        assert predictor._get_time_period(5) == "overnight"

    def test_get_time_multiplier_evening(self):
        """Test evening has highest multiplier."""
        predictor = QuotaPredictor()

        # Evening should have multiplier > 1.0 (highest)
        evening_multiplier = predictor._get_time_multiplier(20)
        morning_multiplier = predictor._get_time_multiplier(9)

        assert evening_multiplier > 1.0
        assert evening_multiplier > morning_multiplier

    def test_get_time_multiplier_overnight(self):
        """Test overnight has lowest multiplier."""
        predictor = QuotaPredictor()

        overnight_multiplier = predictor._get_time_multiplier(2)
        afternoon_multiplier = predictor._get_time_multiplier(14)

        assert overnight_multiplier < 1.0
        assert overnight_multiplier < afternoon_multiplier


class TestQuotaPredictorPredictDailyUsage:
    """Tests for predict_daily_usage method."""

    def test_predict_no_history(self, tmp_path):
        """Test prediction with no historical data returns estimate."""
        predictor = QuotaPredictor()
        predictor._history_file = str(tmp_path / "nonexistent.json")
        predictor.segment_count = 10
        predictor.keyword_count = 5
        predictor.estimated_videos_per_keyword = 50

        prediction = predictor.predict_daily_usage()

        assert prediction["based_on_days"] == 0
        assert "message" in prediction
        assert prediction["message"] == "No historical data - using estimate"

    def test_predict_with_history_calculates_weighted_avg(self, tmp_path):
        """Test prediction uses weighted moving average with history."""
        history_file = tmp_path / "quota_history.json"
        # Create 7 days of history with increasing usage
        test_data = [
            {"date": f"2026-02-{20-i:02d}", "total_usage": 3000 + i * 200, "search_used": 2000, "caption_used": 800, "metadata_used": 200}
            for i in range(7)
        ]
        history_file.write_text(json.dumps(test_data))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        prediction = predictor.predict_daily_usage()

        # Most recent (3000) should have highest weight, oldest (4200) lowest
        # Weighted average should be closer to recent values
        assert prediction["based_on_days"] == 7
        # Base prediction should be less than simple average (which would be 3600)
        assert prediction["base_prediction"] < 3600
        assert prediction["base_prediction"] > 3000

    def test_predict_time_factor_applied(self, tmp_path):
        """Test time factor is applied to prediction."""
        history_file = tmp_path / "quota_history.json"
        test_data = [{"date": "2026-02-20", "total_usage": 5000, "search_used": 3000, "caption_used": 1500, "metadata_used": 500}]
        history_file.write_text(json.dumps(test_data))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        prediction = predictor.predict_daily_usage()

        # predicted_daily_usage should be base_prediction * time_factor
        assert prediction["predicted_daily_usage"] == int(prediction["base_prediction"] * prediction["time_factor"])
        assert prediction["time_factor"] != 1.0


class TestQuotaPredictorConfidence:
    """Tests for get_prediction_confidence method."""

    def test_confidence_low_no_history(self, tmp_path):
        """Test confidence is low with no historical data."""
        predictor = QuotaPredictor()
        predictor._history_file = str(tmp_path / "nonexistent.json")

        confidence = predictor.get_prediction_confidence()

        assert confidence == "low"

    def test_confidence_low_insufficient_data(self, tmp_path):
        """Test confidence is low with less than 3 days of data."""
        history_file = tmp_path / "quota_history.json"
        test_data = [
            {"date": "2026-02-20", "total_usage": 5000, "search_used": 3000, "caption_used": 1500, "metadata_used": 500},
            {"date": "2026-02-19", "total_usage": 5000, "search_used": 3000, "caption_used": 1500, "metadata_used": 500},  # Same values
        ]
        history_file.write_text(json.dumps(test_data))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        confidence = predictor.get_prediction_confidence()

        assert confidence == "low"

    def test_confidence_medium_sufficient_data(self, tmp_path):
        """Test confidence is medium with 3+ days of data."""
        history_file = tmp_path / "quota_history.json"
        test_data = [
            {"date": f"2026-02-{20-i:02d}", "total_usage": 4000 + i * 200, "search_used": 2500, "caption_used": 1200, "metadata_used": 300}
            for i in range(4)
        ]
        history_file.write_text(json.dumps(test_data))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        confidence = predictor.get_prediction_confidence()

        assert confidence == "medium"

    def test_confidence_high_sufficient_varied_data(self, tmp_path):
        """Test confidence is high with 7+ days of varied data."""
        history_file = tmp_path / "quota_history.json"
        test_data = [
            {"date": f"2026-02-{20-i:02d}", "total_usage": 3000 + (i % 3) * 500, "search_used": 2000, "caption_used": 800, "metadata_used": 200}
            for i in range(7)
        ]
        history_file.write_text(json.dumps(test_data))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        confidence = predictor.get_prediction_confidence()

        assert confidence == "high"


class TestQuotaPredictorSaveRecord:
    """Tests for save_usage_record method."""

    def test_save_record_creates_file(self, tmp_path):
        """Test save_usage_record creates history file."""
        history_file = tmp_path / "quota_history.json"
        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)

        predictor._search_used = 3000
        predictor._caption_used = 1500
        predictor._metadata_used = 500
        predictor.save_usage_record(5000)

        assert history_file.exists()

        with open(history_file) as f:
            data = json.load(f)

        assert "history" in data
        assert len(data["history"]) == 1
        assert data["history"][0]["total_usage"] == 5000

    def test_save_record_appends_to_existing(self, tmp_path):
        """Test save_usage_record appends to existing history."""
        history_file = tmp_path / "quota_history.json"
        existing_data = [
            {"date": "2026-02-19", "total_usage": 4000, "timestamp": "2026-02-19T10:00:00", "search_used": 2500, "caption_used": 1200, "metadata_used": 300, "hour": 10, "time_period": "morning"},
        ]
        history_file.write_text(json.dumps({"history": existing_data}))

        predictor = QuotaPredictor()
        predictor._history_file = str(history_file)
        predictor._search_used = 3000
        predictor._caption_used = 1500
        predictor._metadata_used = 500
        predictor.save_usage_record(5000)

        with open(history_file) as f:
            data = json.load(f)

        assert len(data["history"]) == 2


class TestQuotaPredictorPredictionMetrics:
    """Tests for get_prediction_metrics method."""

    def test_get_prediction_metrics_returns_all_fields(self, tmp_path):
        """Test get_prediction_metrics returns all expected fields."""
        predictor = QuotaPredictor()
        predictor._history_file = str(tmp_path / "nonexistent.json")

        metrics = predictor.get_prediction_metrics()

        assert "prediction" in metrics
        assert "confidence" in metrics
        assert "history_count" in metrics
        assert "history_file" in metrics
        assert "weighted_avg_window" in metrics
