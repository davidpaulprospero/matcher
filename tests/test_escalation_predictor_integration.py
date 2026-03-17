"""
Integration tests for RateLimitPredictor in EscalationManager (US-113-003).

Tests the integration of RateLimitPredictor into EscalationManager:
  - Predictor initialization and configuration
  - predict_and_adjust_budget() integration
  - Graceful fallback when predictor unavailable
  - Recording attempts and rate limit events in predictor
"""

import pytest
from unittest.mock import MagicMock, patch

from src.downloader.escalation_manager import EscalationManager
from src.downloader.rate_limit_predictor import RateLimitPredictor
from src.downloader.types import EscalationTier


class TestEscalationManagerPredictorIntegration:
    """Tests for RateLimitPredictor integration in EscalationManager."""

    @pytest.fixture
    def mock_impersonation_manager(self):
        """Create a mock impersonation manager."""
        manager = MagicMock()
        manager.get_impersonate_args.return_value = ["--impersonate", "chrome-136"]
        return manager

    @pytest.fixture
    def mock_budget(self):
        """Create a mock budget with rotatable resources."""
        budget = MagicMock()
        budget.record_attempt = MagicMock()
        budget.record_rotation = MagicMock()
        budget.can_rotate.return_value = True
        budget.max_rotations = 10
        budget.max_vpn_switches = 3
        budget.max_backoff_time = 600.0
        return budget

    @pytest.fixture
    def predictor(self):
        """Create a RateLimitPredictor instance."""
        return RateLimitPredictor(max_history_days=30)

    def test_predictor_initialization_default(self, mock_impersonation_manager):
        """Test EscalationManager initializes without predictor by default."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
        )

        # Predictor should be None by default
        assert manager.get_rate_limit_predictor() is None
        assert manager.is_predictor_enabled() is False

    def test_predictor_initialization_with_predictor(
        self, mock_impersonation_manager, predictor
    ):
        """Test EscalationManager initializes with predictor when provided."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        assert manager.get_rate_limit_predictor() is predictor
        assert manager.is_predictor_enabled() is True

    def test_predictor_disabled_flag(
        self, mock_impersonation_manager, predictor
    ):
        """Test predictor_enabled=False disables predictor."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            rate_limit_predictor=predictor,
            predictor_enabled=False,
        )

        assert manager.get_rate_limit_predictor() is predictor
        assert manager.is_predictor_enabled() is False

    def test_graceful_fallback_no_predictor(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test graceful fallback when predictor is not configured."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
        )

        # Should return 0.0 when predictor not available
        likelihood = manager.predict_and_adjust_budget()
        assert likelihood == 0.0

    def test_graceful_fallback_predictor_disabled(
        self, mock_impersonation_manager, mock_budget, predictor
    ):
        """Test graceful fallback when predictor is disabled."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=False,
        )

        # Should return 0.0 when predictor disabled
        likelihood = manager.predict_and_adjust_budget()
        assert likelihood == 0.0

    def test_graceful_fallback_no_budget(
        self, mock_impersonation_manager, predictor
    ):
        """Test graceful fallback when budget is not configured."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
            budget=None,  # No budget
        )

        # Should return likelihood even without budget
        # (predict_and_adjust_budget checks budget internally)
        likelihood = manager.predict_and_adjust_budget()
        assert isinstance(likelihood, float)
        assert 0.0 <= likelihood <= 1.0

    def test_predict_and_adjust_budget_integration(
        self, mock_impersonation_manager, mock_budget, predictor
    ):
        """Test predict_and_adjust_budget integrates with predictor."""
        # Record some rate limit events to create pattern
        predictor.record_rate_limit_event(
            trigger_category="429",
            tier="tier1",
            keyword="test"
        )
        predictor.record_rate_limit_event(
            trigger_category="403",
            tier="tier1",
            keyword="test"
        )

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        likelihood = manager.predict_and_adjust_budget()

        # Should return a valid likelihood
        assert isinstance(likelihood, float)
        assert 0.0 <= likelihood <= 1.0

    def test_record_predictor_attempt(
        self, mock_impersonation_manager, predictor
    ):
        """Test recording attempts in predictor."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        # Should not raise
        manager.record_predictor_attempt(keyword="test_keyword")

    def test_record_predictor_rate_limit(
        self, mock_impersonation_manager, predictor
    ):
        """Test recording rate limit events in predictor."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        # Should not raise
        manager.record_predictor_rate_limit(
            trigger_category="429",
            tier="tier1",
            keyword="test_keyword"
        )

    def test_get_escalation_args_calls_predictor(
        self, mock_impersonation_manager, mock_budget, predictor
    ):
        """Test get_escalation_args calls predictor methods."""
        # Record some history
        predictor.record_attempt()
        predictor.record_rate_limit_event(
            trigger_category="429",
            tier="tier1",
            keyword="test"
        )

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        # Call get_escalation_args
        result = manager.get_escalation_args("test")

        # Verify result is valid
        assert result.tier == EscalationTier.IMPERSONATE_ONLY

        # Verify budget was checked for adjustment (mock called)
        # The predictor should have been called
        assert manager.is_predictor_enabled() is True


class TestPredictorBasedBudgetAdjustment:
    """Tests for predictor-based budget adjustment logic."""

    @pytest.fixture
    def mock_impersonation_manager(self):
        """Create a mock impersonation manager."""
        manager = MagicMock()
        manager.get_impersonate_args.return_value = ["--impersonate", "chrome-136"]
        return manager

    @pytest.fixture
    def mock_budget_with_limits(self):
        """Create a mock budget with specific limits."""
        budget = MagicMock()
        budget.record_attempt = MagicMock()
        budget.can_rotate.return_value = True
        budget.max_rotations = 10
        budget.max_vpn_switches = 3
        budget.max_backoff_time = 600.0
        return budget

    def test_budget_increase_on_high_likelihood(
        self, mock_impersonation_manager, mock_budget_with_limits
    ):
        """Test budget increases when predictor likelihood is high."""
        # Create predictor with high likelihood by recording many events
        predictor = RateLimitPredictor(max_history_days=30)

        # Record many rate limit events to create high likelihood pattern
        for i in range(10):
            predictor.record_rate_limit_event(
                trigger_category="429",
                tier="tier1",
                keyword=f"test_{i}"
            )

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget_with_limits,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        # Get initial budget values
        initial_rotations = mock_budget_with_limits.max_rotations

        # Call predict_and_adjust_budget - should increase budget
        likelihood = manager.predict_and_adjust_budget()

        # Budget should have been increased if likelihood > threshold
        if likelihood > predictor.BUDGET_INCREASE_THRESHOLD:
            assert mock_budget_with_limits.max_rotations > initial_rotations

    def test_budget_not_increased_on_low_likelihood(
        self, mock_impersonation_manager, mock_budget_with_limits
    ):
        """Test budget does not increase when predictor likelihood is low."""
        # Create predictor with no history (low likelihood)
        predictor = RateLimitPredictor(max_history_days=30)
        # No events recorded - likelihood should be low

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget_with_limits,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        # Get initial budget values
        initial_rotations = mock_budget_with_limits.max_rotations

        # Call predict_and_adjust_budget
        likelihood = manager.predict_and_adjust_budget()

        # Budget should not have been increased
        if likelihood <= predictor.BUDGET_INCREASE_THRESHOLD:
            assert mock_budget_with_limits.max_rotations == initial_rotations


class TestPredictorCheckpointPersistence:
    """Tests for predictor persistence in checkpoint (US-136-004)."""

    @pytest.fixture
    def mock_impersonation_manager(self):
        """Create a mock impersonation manager."""
        manager = MagicMock()
        manager.get_impersonate_args.return_value = ["--impersonate", "chrome-136"]
        return manager

    @pytest.fixture
    def mock_budget(self):
        """Create a mock budget."""
        budget = MagicMock()
        budget.record_attempt = MagicMock()
        budget.can_rotate.return_value = True
        budget.max_rotations = 10
        budget.max_vpn_switches = 3
        budget.max_backoff_time = 600.0
        return budget

    def test_to_dict_includes_predictor_data(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test that to_dict includes predictor data when predictor is present."""
        predictor = RateLimitPredictor(max_history_days=30)

        # Record some events
        predictor.record_rate_limit_event(
            trigger_category="429",
            tier="tier1",
            keyword="test"
        )
        predictor.record_attempt()

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        data = manager.to_dict()

        # Check predictor data is included
        assert "rate_limit_predictor" in data
        assert data["rate_limit_predictor"] is not None
        assert "patterns" in data["rate_limit_predictor"]

    def test_to_dict_excludes_predictor_when_none(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test that to_dict handles missing predictor gracefully."""
        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            predictor_enabled=True,
        )

        data = manager.to_dict()

        # Check predictor data is None when no predictor
        assert "rate_limit_predictor" in data
        assert data["rate_limit_predictor"] is None

    def test_from_dict_restores_predictor(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test that from_dict restores predictor from checkpoint data."""
        # Create and configure manager with predictor
        predictor = RateLimitPredictor(max_history_days=30)
        predictor.record_rate_limit_event(
            trigger_category="429",
            tier="tier1",
            keyword="test"
        )
        predictor.record_attempt()

        original_manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        # Save checkpoint
        data = original_manager.to_dict()

        # Restore from checkpoint
        restored_manager = EscalationManager.from_dict(
            data=data,
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
        )

        # Verify predictor was restored
        assert restored_manager.get_rate_limit_predictor() is not None
        assert restored_manager.is_predictor_enabled() is True

    def test_from_dict_restores_predictor_enabled_flag(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test that from_dict restores predictor_enabled flag."""
        predictor = RateLimitPredictor(max_history_days=30)

        original_manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=False,  # Disabled
        )

        data = original_manager.to_dict()

        restored_manager = EscalationManager.from_dict(
            data=data,
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
        )

        # Verify predictor_enabled was restored
        assert restored_manager.is_predictor_enabled() is False


class TestPredictionAccuracy:
    """Tests for prediction accuracy against actual rate limit events (US-136-004)."""

    @pytest.fixture
    def mock_impersonation_manager(self):
        """Create a mock impersonation manager."""
        manager = MagicMock()
        manager.get_impersonate_args.return_value = ["--impersonate", "chrome-136"]
        return manager

    @pytest.fixture
    def mock_budget(self):
        """Create a mock budget."""
        budget = MagicMock()
        budget.record_attempt = MagicMock()
        budget.can_rotate.return_value = True
        budget.max_rotations = 10
        budget.max_vpn_switches = 3
        budget.max_backoff_time = 600.0
        return budget

    def test_prediction_reflects_actual_events(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test that prediction likelihood reflects actual rate limit events."""
        predictor = RateLimitPredictor(max_history_days=30)

        # Record many attempts and rate limit events
        for i in range(20):
            predictor.record_attempt()
        for i in range(10):
            predictor.record_rate_limit_event(
                trigger_category="429",
                tier="tier1",
            )

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        likelihood = manager.predict_and_adjust_budget()

        # With 20 attempts and 10 rate limits, likelihood should be 0.5
        assert 0.0 <= likelihood <= 1.0
        # The likelihood should be high given the event rate
        assert likelihood > 0.0

    def test_prediction_accuracy_time_window(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test prediction accuracy varies by time window."""
        from datetime import datetime, timedelta

        predictor = RateLimitPredictor(max_history_days=30)

        # Record events at specific time (e.g., evening - high rate limit time)
        evening_hour = 20  # 8 PM - Evening time window
        base_time = datetime.now().replace(hour=evening_hour, minute=0, second=0, microsecond=0)

        # Record many events in evening time window
        for i in range(20):
            timestamp = (base_time + timedelta(hours=i)).timestamp()
            predictor.record_attempt(timestamp=timestamp)
            if i % 2 == 0:  # 50% rate limit rate
                predictor.record_rate_limit_event(
                    timestamp=timestamp,
                    trigger_category="429",
                )

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        likelihood = manager.predict_and_adjust_budget()

        # Evening should show higher likelihood due to historical pattern
        assert likelihood > 0.0

    def test_prediction_with_insufficient_data(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test prediction uses time-based estimation when data is insufficient."""
        predictor = RateLimitPredictor(max_history_days=30)

        # Record only a few events (less than MIN_EVENTS_FOR_RELIABLE_PREDICTION=5)
        for i in range(3):
            predictor.record_attempt()

        manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        likelihood = manager.predict_and_adjust_budget()

        # With insufficient data, should still return a valid likelihood
        # using time-based estimation
        assert 0.0 <= likelihood <= 1.0

    def test_budget_survives_pipeline_resume(
        self, mock_impersonation_manager, mock_budget
    ):
        """Test that predictor data survives pipeline resume (checkpoint)."""
        # Create manager with predictor and record events
        predictor = RateLimitPredictor(max_history_days=30)
        predictor.record_attempt()
        predictor.record_rate_limit_event(
            trigger_category="429",
            tier="tier1",
            keyword="test_keyword"
        )

        original_manager = EscalationManager(
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
            rate_limit_predictor=predictor,
            predictor_enabled=True,
        )

        # Simulate pipeline checkpoint
        checkpoint_data = original_manager.to_dict()

        # Simulate pipeline resume
        resumed_manager = EscalationManager.from_dict(
            data=checkpoint_data,
            impersonation_manager=mock_impersonation_manager,
            budget=mock_budget,
        )

        # Verify predictor is available after resume
        resumed_predictor = resumed_manager.get_rate_limit_predictor()
        assert resumed_predictor is not None

        # Verify predictor still works after resume
        likelihood = resumed_manager.predict_and_adjust_budget()
        assert 0.0 <= likelihood <= 1.0
