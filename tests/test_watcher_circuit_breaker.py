"""Tests for WatcherAgent low-confidence circuit breaker.

US-68-005: Circuit breaker trips after repeated low-confidence classifications,
disabling the watcher and falling through to pattern routing.
"""

import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.agents.watcher import ErrorClassification, WatcherAgent


@dataclass
class FakeWatcherConfig:
    """Minimal config for WatcherAgent."""
    provider: str = "ollama"
    model: str = "llama3.2"
    fallback_model: str = "llama3.1"
    host: str = "http://localhost:11434"
    timeout: float = 5.0
    escalate_threshold: float = 0.7


def _make_watcher(**kwargs) -> WatcherAgent:
    """Create a WatcherAgent with fake config."""
    config = FakeWatcherConfig(**kwargs)
    return WatcherAgent(config=config)


def _make_classification(confidence: float = 0.8) -> ErrorClassification:
    """Create a classification with given confidence."""
    return ErrorClassification(
        category="api",
        severity="transient",
        suggested_healer="api-healer",
        confidence=confidence,
        needs_llm_healer=False,
        reasoning="test classification",
    )


class TestWatcherCircuitBreakerTrips:
    """Test that circuit breaker trips after 5 low-confidence results."""

    def test_trips_after_5_low_confidence(self):
        """Circuit breaker trips after exactly 5 low-confidence classifications."""
        watcher = _make_watcher()

        # 4 low-confidence results should NOT trip
        for _ in range(4):
            watcher._record_low_confidence(0.1)
        assert not watcher._cb_tripped
        assert not watcher.is_circuit_breaker_tripped()

        # 5th should trip
        watcher._record_low_confidence(0.1)
        assert watcher._cb_tripped
        assert watcher.is_circuit_breaker_tripped()

    def test_classify_returns_none_when_tripped(self):
        """classify_error returns None when circuit breaker is tripped."""
        watcher = _make_watcher()

        # Trip the breaker
        for _ in range(5):
            watcher._record_low_confidence(0.2)
        assert watcher.is_circuit_breaker_tripped()

        # classify_error should short-circuit to None
        result = watcher.classify_error(
            ValueError("test error"),
            {"stage": "DOWNLOAD_SEGMENTS"},
        )
        assert result is None

    def test_good_confidence_resets_counter(self):
        """A good-confidence result resets the consecutive low counter."""
        watcher = _make_watcher()

        # 4 low-confidence
        for _ in range(4):
            watcher._record_low_confidence(0.1)
        assert watcher._cb_consecutive_low == 4

        # 1 good-confidence resets
        watcher._record_good_confidence()
        assert watcher._cb_consecutive_low == 0

        # Need 5 more to trip
        for _ in range(4):
            watcher._record_low_confidence(0.15)
        assert not watcher._cb_tripped

    def test_confidence_threshold_boundary(self):
        """Confidence exactly at 0.3 is NOT low (< 0.3 is low)."""
        watcher = _make_watcher()

        # Record at exactly the threshold - should count as good
        assert 0.3 >= watcher._low_confidence_threshold
        # 0.3 is not < 0.3, so it's good
        watcher._record_good_confidence()
        assert watcher._cb_consecutive_low == 0

        # 0.29 IS low
        watcher._record_low_confidence(0.29)
        assert watcher._cb_consecutive_low == 1

    def test_end_to_end_classify_tracks_low_confidence(self):
        """Full classify_error flow tracks low-confidence and trips breaker."""
        watcher = _make_watcher()
        watcher._initialized = True

        low_classification = _make_classification(confidence=0.1)

        # Mock the client to return a low-confidence classification
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed_data = {
            "category": "api",
            "severity": "transient",
            "suggested_healer": "api-healer",
            "confidence": 0.1,
            "needs_llm_healer": False,
            "reasoning": "garbage",
        }
        mock_client.generate.return_value = mock_response
        watcher.client = mock_client

        error = ValueError("test error")
        context = {"stage": "MATCH"}

        # 5 low-confidence classifications should trip the breaker
        for i in range(5):
            result = watcher.classify_error(error, context)
            assert result is not None  # Still returns classification
            assert result.confidence == 0.1

        # Now circuit breaker is tripped
        assert watcher.is_circuit_breaker_tripped()

        # 6th call returns None without calling the LLM
        mock_client.generate.reset_mock()
        result = watcher.classify_error(error, context)
        assert result is None
        mock_client.generate.assert_not_called()


class TestWatcherCircuitBreakerAutoReset:
    """Test that circuit breaker auto-resets after cooldown period."""

    def test_auto_resets_after_cooldown(self):
        """Circuit breaker auto-resets when cooldown elapses."""
        watcher = _make_watcher()

        # Trip the breaker
        for _ in range(5):
            watcher._record_low_confidence(0.1)
        assert watcher.is_circuit_breaker_tripped()

        # Simulate cooldown elapsed by backdating the trip time
        watcher._cb_tripped_at = time.time() - 601  # 601s > 600s cooldown

        # Should auto-reset
        assert not watcher.is_circuit_breaker_tripped()
        assert watcher._cb_consecutive_low == 0
        assert not watcher._cb_tripped

    def test_still_tripped_during_cooldown(self):
        """Circuit breaker stays tripped during the cooldown window."""
        watcher = _make_watcher()

        for _ in range(5):
            watcher._record_low_confidence(0.1)
        assert watcher.is_circuit_breaker_tripped()

        # Only 5 minutes elapsed (< 10 minute cooldown)
        watcher._cb_tripped_at = time.time() - 300

        # Still tripped
        assert watcher.is_circuit_breaker_tripped()

    def test_classify_works_after_auto_reset(self):
        """classify_error works again after cooldown auto-reset."""
        watcher = _make_watcher()
        watcher._initialized = True

        # Trip the breaker
        for _ in range(5):
            watcher._record_low_confidence(0.1)
        assert watcher.is_circuit_breaker_tripped()

        # Backdate to simulate cooldown
        watcher._cb_tripped_at = time.time() - 700

        # Mock client for a good classification this time
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed_data = {
            "category": "download",
            "severity": "recoverable",
            "suggested_healer": "download-healer",
            "confidence": 0.9,
            "needs_llm_healer": False,
            "reasoning": "good classification",
        }
        mock_client.generate.return_value = mock_response
        watcher.client = mock_client

        result = watcher.classify_error(ValueError("test"), {"stage": "DOWNLOAD"})
        assert result is not None
        assert result.confidence == 0.9
        mock_client.generate.assert_called_once()


class TestWatcherCircuitBreakerReset:
    """Test manual reset functionality."""

    def test_manual_reset_clears_tripped(self):
        """reset_circuit_breaker() clears the tripped state."""
        watcher = _make_watcher()

        for _ in range(5):
            watcher._record_low_confidence(0.1)
        assert watcher.is_circuit_breaker_tripped()

        watcher.reset_circuit_breaker()

        assert not watcher._cb_tripped
        assert watcher._cb_tripped_at is None
        assert watcher._cb_consecutive_low == 0
        assert not watcher.is_circuit_breaker_tripped()

    def test_manual_reset_when_not_tripped(self):
        """reset_circuit_breaker() is safe to call when not tripped."""
        watcher = _make_watcher()
        watcher._record_low_confidence(0.2)
        assert watcher._cb_consecutive_low == 1

        watcher.reset_circuit_breaker()
        assert watcher._cb_consecutive_low == 0
        assert not watcher._cb_tripped
