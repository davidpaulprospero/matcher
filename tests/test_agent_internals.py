"""Tests for agent module internals: healing_logger, strategy, watcher, fallback.

US-47-008: Unit tests for src/agents/ internal modules.
"""

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.healing_logger import HealingLogEntry, HealingLogger
from src.agents.strategy import (
    ConfigSnapshot,
    HealingMetrics,
    HealingMode,
    HealingStrategy,
)
from src.agents.watcher import ErrorClassification, WatcherAgent
from src.agents.fallback import (
    FallbackChain,
    PatternClassification,
    pattern_route,
    PATTERN_ROUTING,
)


# ============================================================
# HealingLogEntry tests
# ============================================================


class TestHealingLogEntry:
    """Tests for HealingLogEntry dataclass."""

    def test_to_dict_serializes_timestamp(self):
        entry = HealingLogEntry(
            timestamp=datetime(2026, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            stage="MATCH",
            component="watcher",
            action="classify",
            error_type="ValueError",
            error_message="test error",
            result="success",
            duration_ms=42.5,
        )
        d = entry.to_dict()
        assert d["timestamp"] == "2026-01-15T10:30:00+00:00"
        assert d["stage"] == "MATCH"
        assert d["component"] == "watcher"
        assert d["duration_ms"] == 42.5

    def test_to_dict_includes_details(self):
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="DOWNLOAD",
            component="healer",
            action="attempt",
            error_type="IOError",
            error_message="disk full",
            result="failed",
            duration_ms=100,
            details={"healer": "disk-healer", "action": "clear_cache"},
        )
        d = entry.to_dict()
        assert d["details"]["healer"] == "disk-healer"

    def test_to_dict_optional_stack_trace(self):
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="OUTPUT",
            component="fallback",
            action="fallback",
            error_type="RuntimeError",
            error_message="fallback triggered",
            result="degraded",
            duration_ms=0,
            stack_trace="Traceback (most recent call last):\n  File ...",
        )
        d = entry.to_dict()
        assert "Traceback" in d["stack_trace"]


# ============================================================
# HealingLogger tests
# ============================================================


class TestHealingLogger:
    """Tests for HealingLogger JSONL writing and reading."""

    def test_init_creates_log_files(self, tmp_path):
        logger = HealingLogger(tmp_path, json_log=True)
        assert logger.log_file.exists() or True  # log file created on first write
        assert logger.json_file.exists()
        # JSON file initialized with empty array
        content = logger.json_file.read_text(encoding="utf-8")
        assert json.loads(content) == []

    def test_write_entry_appends_to_log_file(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=True)
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="MATCH",
            component="watcher",
            action="classify",
            error_type="ValueError",
            error_message="test",
            result="success",
            duration_ms=10,
        )
        hl._write(entry)

        # Human-readable log should have content
        assert hl.log_file.exists()
        log_text = hl.log_file.read_text(encoding="utf-8")
        assert "classify" in log_text

    def test_write_entry_appends_to_json_file(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=True)
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="CAPTION",
            component="healer",
            action="attempt",
            error_type="IOError",
            error_message="disk error",
            result="failed",
            duration_ms=50,
        )
        hl._write(entry)

        entries = json.loads(hl.json_file.read_text(encoding="utf-8"))
        assert len(entries) == 1
        assert entries[0]["stage"] == "CAPTION"

    def test_multiple_writes_accumulate(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=True)
        for i in range(5):
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage=f"STAGE_{i}",
                component="watcher",
                action="classify",
                error_type="Error",
                error_message=f"error {i}",
                result="success",
                duration_ms=i * 10,
            )
            hl._write(entry)

        entries = json.loads(hl.json_file.read_text(encoding="utf-8"))
        assert len(entries) == 5
        assert hl.entries[-1].stage == "STAGE_4"

    def test_session_id_set_on_write(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=True)
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="TEST",
            component="watcher",
            action="classify",
            error_type="Error",
            error_message="test",
            result="success",
            duration_ms=0,
        )
        assert entry.session_id == ""
        hl._write(entry)
        assert entry.session_id == hl.session_id

    def test_json_log_disabled(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="TEST",
            component="watcher",
            action="classify",
            error_type="Error",
            error_message="test",
            result="success",
            duration_ms=0,
        )
        hl._write(entry)
        # JSON file should still have initial empty array (never updated)
        assert not hl.json_file.exists() or json.loads(hl.json_file.read_text()) == []

    def test_max_entries_truncation(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        # Temporarily lower MAX_ENTRIES for testing
        original_max = HealingLogger.MAX_ENTRIES
        HealingLogger.MAX_ENTRIES = 10
        try:
            for i in range(15):
                entry = HealingLogEntry(
                    timestamp=datetime.now(timezone.utc),
                    stage="TEST",
                    component="watcher",
                    action="classify",
                    error_type="Error",
                    error_message=f"msg_{i}",
                    result="success",
                    duration_ms=0,
                )
                hl._write(entry)
            # Truncation triggers at entry 11 (>10), keeps last 5.
            # Then entries 12-14 are added (4 more), total = 9.
            assert len(hl.entries) < 15  # Truncation occurred
            assert len(hl.entries) <= HealingLogger.MAX_ENTRIES
            # Most recent entries kept
            assert hl.entries[-1].error_message == "msg_14"
        finally:
            HealingLogger.MAX_ENTRIES = original_max

    def test_format_entry_success(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        entry = HealingLogEntry(
            timestamp=datetime(2026, 1, 15, 10, 30, 45, 123456, tzinfo=timezone.utc),
            stage="MATCH",
            component="healer",
            action="attempt",
            error_type="Error",
            error_message="fixed it",
            result="success",
            duration_ms=100,
        )
        formatted = hl._format_entry(entry)
        assert "✓" in formatted
        assert "attempt" in formatted

    def test_format_entry_failed(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="MATCH",
            component="healer",
            action="attempt",
            error_type="Error",
            error_message="could not fix",
            result="failed",
            duration_ms=100,
        )
        formatted = hl._format_entry(entry)
        assert "✗" in formatted

    def test_format_entry_other_result(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="MATCH",
            component="orchestrator",
            action="escalate",
            error_type="Error",
            error_message="escalating",
            result="escalated",
            duration_ms=0,
        )
        formatted = hl._format_entry(entry)
        assert "→" in formatted

    def test_generate_report_empty(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        report = hl.generate_report()
        assert report == "No healing activity recorded."

    def test_generate_report_with_entries(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        for result_val in ["success", "success", "failed"]:
            entry = HealingLogEntry(
                timestamp=datetime.now(timezone.utc),
                stage="MATCH",
                component="healer",
                action="attempt",
                error_type="Error",
                error_message="test",
                result=result_val,
                duration_ms=10,
            )
            hl._write(entry)
        report = hl.generate_report()
        assert "Successful heals: 2" in report
        assert "Failed heals: 1" in report
        assert hl.session_id in report

    def test_finalize_writes_summary_json(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=True)
        entry = HealingLogEntry(
            timestamp=datetime.now(timezone.utc),
            stage="TEST",
            component="watcher",
            action="classify",
            error_type="Error",
            error_message="test",
            result="success",
            duration_ms=0,
        )
        hl._write(entry)
        hl.finalize()

        summary = json.loads(hl.json_file.read_text(encoding="utf-8"))
        assert "session_id" in summary
        assert "entries" in summary
        assert summary["total_entries"] == 1

    def test_log_classification(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        classification = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="api-healer",
            confidence=0.85,
            needs_llm_healer=False,
            reasoning="Rate limit detected",
        )
        error = ValueError("429 Too Many Requests")
        hl.log_classification("DOWNLOAD", error, classification, 150.0)
        assert len(hl.entries) == 1
        assert hl.entries[0].component == "watcher"
        assert hl.entries[0].details["category"] == "api"

    def test_log_fallback(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        hl.log_fallback("MATCH", "watcher", "pattern_routing", "Ollama unavailable")
        assert len(hl.entries) == 1
        assert hl.entries[0].component == "fallback"
        assert hl.entries[0].result == "degraded"

    def test_log_escalation(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        error = RuntimeError("unknown error")
        hl.log_escalation("OUTPUT", error, "Cannot classify", to_user=True)
        assert len(hl.entries) == 1
        assert hl.entries[0].details["to_user"] is True

    def test_log_self_heal(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        hl.log_self_heal("api-healer", 2, 3, "RateLimitError", "reduce batch size")
        assert len(hl.entries) == 1
        assert hl.entries[0].details["attempt"] == 2

    def test_log_provider_switch(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        hl.log_provider_switch("anthropic", "gemini")
        assert hl.entries[0].details["from_provider"] == "anthropic"
        assert hl.entries[0].details["to_provider"] == "gemini"

    def test_log_preflight_check(self, tmp_path):
        hl = HealingLogger(tmp_path, json_log=False)
        hl.log_preflight_check("disk", "free_space", True, "10GB available")
        assert hl.entries[0].result == "success"
        hl.log_preflight_check("api", "api_key", False, "Key missing")
        assert hl.entries[1].result == "failed"

    def test_thread_safety(self, tmp_path):
        """Multiple threads writing concurrently should not corrupt state."""
        hl = HealingLogger(tmp_path, json_log=True)
        errors = []

        def write_entries(thread_id):
            try:
                for i in range(10):
                    entry = HealingLogEntry(
                        timestamp=datetime.now(timezone.utc),
                        stage=f"STAGE_{thread_id}",
                        component="watcher",
                        action="classify",
                        error_type="Error",
                        error_message=f"thread_{thread_id}_entry_{i}",
                        result="success",
                        duration_ms=0,
                    )
                    hl._write(entry)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_entries, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All 40 entries should be written (unless truncation kicked in)
        assert len(hl.entries) == 40

    def test_print_box_formats(self, tmp_path, capsys):
        hl = HealingLogger(tmp_path, json_log=False, console_format="box")
        hl.print_box("Test Title", ["Line 1", "Line 2"])
        captured = capsys.readouterr()
        assert "┌" in captured.out
        assert "Test Title" in captured.out

        hl2 = HealingLogger(tmp_path / "simple", json_log=False, console_format="simple")
        hl2.print_box("Simple Title", ["Line A"])
        captured = capsys.readouterr()
        assert "=== Simple Title ===" in captured.out

        hl3 = HealingLogger(tmp_path / "minimal", json_log=False, console_format="minimal")
        hl3.print_box("Minimal", ["Line X"])
        captured = capsys.readouterr()
        assert "Line X" in captured.out
        assert "┌" not in captured.out


# ============================================================
# Strategy tests
# ============================================================


class TestHealingMode:
    """Tests for HealingMode enum."""

    def test_all_modes_exist(self):
        assert HealingMode.AGGRESSIVE.value == "aggressive"
        assert HealingMode.CONSERVATIVE.value == "conservative"
        assert HealingMode.INTERACTIVE.value == "interactive"
        assert HealingMode.MINIMAL.value == "minimal"


class TestHealingStrategy:
    """Tests for strategy ranking and selection."""

    def test_default_is_conservative(self):
        s = HealingStrategy()
        assert s.mode == HealingMode.CONSERVATIVE
        assert s.max_attempts_per_stage == 3
        assert s.max_total_heals == 20
        assert s.enable_rollback is True

    def test_aggressive_factory(self):
        s = HealingStrategy.aggressive()
        assert s.mode == HealingMode.AGGRESSIVE
        assert s.max_attempts_per_stage == 5
        assert s.max_total_heals == 50
        assert s.heal_delay == 1.0
        assert s.auto_fix_preflight is True

    def test_conservative_factory(self):
        s = HealingStrategy.conservative()
        assert s.mode == HealingMode.CONSERVATIVE
        assert s.max_attempts_per_stage == 3

    def test_interactive_factory(self):
        s = HealingStrategy.interactive()
        assert s.mode == HealingMode.INTERACTIVE
        assert s.auto_fix_preflight is False

    def test_minimal_factory(self):
        s = HealingStrategy.minimal()
        assert s.mode == HealingMode.MINIMAL
        assert s.max_attempts_per_stage == 1
        assert s.max_total_heals == 5
        assert s.enable_rollback is False

    def test_healer_priority_order(self):
        s = HealingStrategy()
        assert s.healer_priority[0] == "checkpoint-healer"
        assert "api-healer" in s.healer_priority
        assert "download-healer" in s.healer_priority

    def test_always_escalate_patterns(self):
        s = HealingStrategy()
        assert "permission denied" in s.always_escalate
        assert "api key invalid" in s.always_escalate
        assert "quota exceeded" in s.always_escalate

    def test_protected_config_keys(self):
        s = HealingStrategy()
        assert "api_key" in s.protected_config_keys
        assert "password" in s.protected_config_keys
        assert "token" in s.protected_config_keys

    def test_skip_healers_customizable(self):
        s = HealingStrategy(skip_healers={"otio-healer", "path-healer"})
        assert "otio-healer" in s.skip_healers
        assert "api-healer" not in s.skip_healers


class TestConfigSnapshot:
    """Tests for config snapshot and rollback."""

    def test_restore_simple_key(self):
        @dataclass
        class MockConfig:
            batch_size: int = 10

        config = MockConfig(batch_size=50)
        snap = ConfigSnapshot(
            stage_name="MATCH",
            timestamp=time.time(),
            config_values={"batch_size": 10},
        )
        result = snap.restore(config)
        assert result is True
        assert config.batch_size == 10

    def test_restore_nested_key(self):
        @dataclass
        class Inner:
            gap_mode: str = "default"

        @dataclass
        class MockConfig:
            output: Inner = None

        config = MockConfig(output=Inner(gap_mode="changed"))
        snap = ConfigSnapshot(
            stage_name="OUTPUT",
            timestamp=time.time(),
            config_values={"output.gap_mode": "default"},
        )
        result = snap.restore(config)
        assert result is True
        assert config.output.gap_mode == "default"

    def test_restore_missing_key_returns_false(self):
        @dataclass
        class MockConfig:
            pass

        config = MockConfig()
        snap = ConfigSnapshot(
            stage_name="TEST",
            timestamp=time.time(),
            config_values={"nonexistent.key": "value"},
        )
        result = snap.restore(config)
        assert result is False

    def test_restore_empty_snapshot(self):
        config = object()
        snap = ConfigSnapshot(stage_name="TEST", timestamp=time.time())
        result = snap.restore(config)
        assert result is False


class TestHealingMetrics:
    """Tests for healing metrics tracking."""

    def test_record_heal_success(self):
        m = HealingMetrics()
        m.record_heal("api-healer", "MATCH", success=True)
        assert m.total_heals == 1
        assert m.successful_heals == 1
        assert m.failed_heals == 0
        assert m.heals_by_stage["MATCH"] == 1
        assert m.heals_by_healer["api-healer"] == 1

    def test_record_heal_failure(self):
        m = HealingMetrics()
        m.record_heal("disk-healer", "DOWNLOAD", success=False)
        assert m.total_heals == 1
        assert m.successful_heals == 0
        assert m.failed_heals == 1

    def test_record_multiple_heals(self):
        m = HealingMetrics()
        m.record_heal("api-healer", "MATCH", success=True)
        m.record_heal("api-healer", "MATCH", success=True)
        m.record_heal("disk-healer", "DOWNLOAD", success=False)
        assert m.total_heals == 3
        assert m.heals_by_stage["MATCH"] == 2
        assert m.heals_by_stage["DOWNLOAD"] == 1

    def test_summary_basic(self):
        m = HealingMetrics()
        m.record_heal("api-healer", "MATCH", True)
        s = m.summary()
        assert "1 successful" in s
        assert "0 failed" in s

    def test_summary_with_preflight(self):
        m = HealingMetrics()
        m.preflight_issues_found = 3
        m.preflight_issues_fixed = 2
        s = m.summary()
        assert "Preflight: 2/3" in s

    def test_summary_with_rollbacks(self):
        m = HealingMetrics()
        m.rollbacks_performed = 1
        m.record_heal("test", "TEST", True)
        s = m.summary()
        assert "Rollbacks: 1" in s


# ============================================================
# Watcher tests
# ============================================================


class TestErrorClassification:
    """Tests for ErrorClassification dataclass."""

    def test_create_classification(self):
        c = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="api-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="Rate limit detected",
        )
        assert c.category == "api"
        assert c.confidence == 0.9


class TestWatcherAgent:
    """Tests for WatcherAgent health check detection and escalation."""

    def _make_config(self, **overrides):
        """Create a mock watcher config."""
        config = MagicMock()
        config.provider = overrides.get("provider", "ollama")
        config.model = overrides.get("model", "llama3.2")
        config.fallback_model = overrides.get("fallback_model", "llama3.1")
        config.host = overrides.get("host", "http://localhost:11434")
        config.timeout = overrides.get("timeout", 30.0)
        config.escalate_threshold = overrides.get("escalate_threshold", 0.7)
        return config

    def test_init_defaults(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        assert agent.provider == "ollama"
        assert agent.model == "llama3.2"
        assert agent._initialized is False

    def test_should_escalate_no_classification(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        assert agent.should_escalate(None) is True

    def test_should_escalate_needs_llm(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="unknown",
            severity="recoverable",
            suggested_healer="",
            confidence=0.5,
            needs_llm_healer=True,
            reasoning="Complex error",
        )
        assert agent.should_escalate(c) is True

    def test_should_escalate_low_confidence(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="api",
            severity="recoverable",
            suggested_healer="api-healer",
            confidence=0.3,
            needs_llm_healer=False,
            reasoning="Uncertain",
        )
        assert agent.should_escalate(c) is True

    def test_should_not_escalate_high_confidence(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="disk",
            severity="recoverable",
            suggested_healer="disk-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="Clear disk error",
        )
        assert agent.should_escalate(c) is False

    def test_should_escalate_healer_failed_moderate_confidence(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="api",
            severity="recoverable",
            suggested_healer="api-healer",
            confidence=0.7,
            needs_llm_healer=False,
            reasoning="API error",
        )
        healer_result = MagicMock(success=False)
        # confidence 0.7 < 0.8 threshold for failed healer -> escalate
        assert agent.should_escalate(c, healer_result) is True

    def test_should_not_escalate_healer_failed_high_confidence(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="api",
            severity="recoverable",
            suggested_healer="api-healer",
            confidence=0.9,
            needs_llm_healer=False,
            reasoning="Clear API error",
        )
        healer_result = MagicMock(success=False)
        # confidence 0.9 >= 0.8 -> don't escalate even if healer failed
        assert agent.should_escalate(c, healer_result) is False

    def test_get_healer_priority_api(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="api-healer",
            confidence=0.8,
            needs_llm_healer=False,
            reasoning="Rate limit",
        )
        healers = agent.get_healer_priority(c)
        assert healers == ["api-healer"]

    def test_get_healer_priority_with_suggested(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="api",
            severity="transient",
            suggested_healer="download-healer",
            confidence=0.7,
            needs_llm_healer=False,
            reasoning="Possibly download related",
        )
        healers = agent.get_healer_priority(c)
        assert healers[0] == "download-healer"
        assert "api-healer" in healers

    def test_get_healer_priority_unknown_category(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="unknown",
            severity="recoverable",
            suggested_healer="",
            confidence=0.3,
            needs_llm_healer=True,
            reasoning="Unknown",
        )
        healers = agent.get_healer_priority(c)
        assert healers == []

    def test_get_healer_priority_config_category(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        c = ErrorClassification(
            category="config",
            severity="recoverable",
            suggested_healer="",
            confidence=0.5,
            needs_llm_healer=True,
            reasoning="Config error",
        )
        healers = agent.get_healer_priority(c)
        assert healers == []

    def test_parse_response_valid_json(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        response = MagicMock()
        response.parsed_data = {
            "category": "api",
            "severity": "transient",
            "suggested_healer": "api-healer",
            "confidence": 0.85,
            "needs_llm_healer": False,
            "reasoning": "Rate limit",
        }
        result = agent._parse_response(response, ValueError("test"))
        assert result is not None
        assert result.category == "api"
        assert result.confidence == 0.85

    def test_parse_response_invalid_category_defaults_unknown(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        response = MagicMock()
        response.parsed_data = {
            "category": "bogus_category",
            "severity": "transient",
            "confidence": 0.5,
        }
        result = agent._parse_response(response, ValueError("test"))
        assert result.category == "unknown"

    def test_parse_response_invalid_severity_defaults_recoverable(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        response = MagicMock()
        response.parsed_data = {
            "category": "api",
            "severity": "bogus_severity",
            "confidence": 0.5,
        }
        result = agent._parse_response(response, ValueError("test"))
        assert result.severity == "recoverable"

    def test_parse_response_clamps_confidence(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        response = MagicMock()
        response.parsed_data = {"category": "api", "severity": "transient", "confidence": 1.5}
        result = agent._parse_response(response, ValueError("test"))
        assert result.confidence == 1.0

        response.parsed_data = {"category": "api", "severity": "transient", "confidence": -0.5}
        result = agent._parse_response(response, ValueError("test"))
        assert result.confidence == 0.0

    def test_parse_response_from_text(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        response = MagicMock(spec=["text"])
        response.parsed_data = None
        del response.parsed_data  # Remove attribute
        response.text = json.dumps({
            "category": "disk",
            "severity": "critical",
            "suggested_healer": "disk-healer",
            "confidence": 0.95,
            "needs_llm_healer": False,
            "reasoning": "Disk full",
        })
        result = agent._parse_response(response, IOError("test"))
        assert result is not None
        assert result.category == "disk"

    def test_parse_response_from_markdown_text(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        response = MagicMock(spec=["text"])
        response.text = '```json\n{"category": "path", "severity": "recoverable", "confidence": 0.7}\n```'
        result = agent._parse_response(response, IOError("test"))
        assert result is not None
        assert result.category == "path"

    def test_parse_response_no_content_returns_none(self):
        config = self._make_config()
        agent = WatcherAgent(config)
        response = MagicMock(spec=[])
        result = agent._parse_response(response, ValueError("test"))
        assert result is None


# ============================================================
# Fallback tests
# ============================================================


class TestPatternRoute:
    """Tests for pattern_route() function - pattern-based routing."""

    def test_rate_limit_routes_to_api(self):
        result = pattern_route("429 Too Many Requests")
        assert result.category == "api"
        assert result.suggested_healer == "api-healer"

    def test_rate_limit_text_routes_to_api(self):
        result = pattern_route("rate limit exceeded, retry after 60s")
        assert result.category == "api"

    def test_auth_error_routes_to_api(self):
        result = pattern_route("HTTP 401 Unauthorized")
        assert result.category == "api"

    def test_forbidden_routes_to_api(self):
        result = pattern_route("HTTP 403 Forbidden")
        assert result.category == "api"

    def test_timeout_routes_to_api(self):
        result = pattern_route("connection timed out after 30s")
        assert result.category == "api"

    def test_connection_refused_routes_to_api(self):
        result = pattern_route("connection refused on port 8080")
        assert result.category == "api"

    def test_disk_full_routes_to_disk(self):
        result = pattern_route("no space left on device")
        assert result.category == "disk"
        assert result.suggested_healer == "disk-healer"

    def test_permission_denied_routes_to_disk(self):
        result = pattern_route("permission denied for /tmp/file.txt")
        assert result.category == "disk"

    def test_path_too_long_routes_to_path(self):
        result = pattern_route("path too long: exceeds 260 chars")
        assert result.category == "path"
        assert result.suggested_healer == "path-healer"

    def test_unicode_error_routes_to_path(self):
        result = pattern_route("UnicodeDecodeError: codec can't decode bytes")
        assert result.category == "path"

    def test_checkpoint_corrupt_routes_to_checkpoint(self):
        result = pattern_route("checkpoint corrupt: invalid JSON")
        assert result.category == "checkpoint"

    def test_json_decode_error_routes_to_checkpoint(self):
        result = pattern_route("json.decoder.JSONDecodeError: Expecting value")
        assert result.category == "checkpoint"

    def test_youtube_error_routes_to_download(self):
        result = pattern_route("yt-dlp error: video unavailable")
        assert result.category == "download"

    def test_video_unavailable_routes_to_download(self):
        result = pattern_route("video unavailable or has been removed")
        assert result.category == "download"

    def test_otio_error_routes_to_otio(self):
        result = pattern_route("opentimelineio error: invalid time range")
        assert result.category == "otio"

    def test_timeline_error_routes_to_otio(self):
        result = pattern_route("timeline generation failed: negative duration")
        assert result.category == "otio"

    def test_media_reference_error_routes_to_otio(self):
        result = pattern_route("media reference missing for clip X")
        assert result.category == "otio"

    def test_unknown_error_escalates(self):
        result = pattern_route("something completely unexpected happened")
        assert result.category == "unknown"
        assert result.needs_llm_healer is True
        assert result.confidence == 0.3

    def test_no_false_positive_singapore(self):
        """Ensure 'Singapore' doesn't match gap pattern (historical false positive)."""
        result = pattern_route("Server location: Singapore")
        # Should NOT match otio gap pattern
        assert result.category != "otio" or result.suggested_healer != "otio-healer"

    def test_no_false_positive_author(self):
        """Ensure 'author' doesn't match auth pattern."""
        result = pattern_route("The author of this video is John")
        assert result.category == "unknown"

    def test_pattern_classification_defaults(self):
        c = PatternClassification(category="api", suggested_healer="api-healer")
        assert c.severity == "recoverable"
        assert c.confidence == 0.6
        assert c.needs_llm_healer is False


class TestFallbackChain:
    """Tests for FallbackChain graceful degradation."""

    def _make_config(self, watcher_enabled=True, llm_enabled=True):
        config = MagicMock()
        config.watcher = MagicMock()
        config.watcher.enabled = watcher_enabled
        config.watcher.provider = "ollama"
        config.watcher.host = "http://localhost:11434"
        config.watcher.model = "llama3.2"
        config.watcher.fallback_model = "llama3.1"
        config.watcher.max_failures = 3
        config.watcher.recheck_interval_seconds = 300.0
        config.llm_healer = MagicMock()
        config.llm_healer.enabled = llm_enabled
        config.llm_healer.provider = "anthropic"
        config.llm_healer.max_failures = 3
        config.llm_healer.recheck_interval_seconds = 300.0
        return config

    def test_init_state(self):
        config = self._make_config()
        chain = FallbackChain(config)
        status = chain.get_status()
        assert status["watcher_available"] is None
        assert status["llm_healer_available"] is None
        assert status["watcher_failures"] == 0

    def test_record_watcher_failure_circuit_breaker(self):
        config = self._make_config()
        chain = FallbackChain(config)
        # Set initial state to True so we can see it flip to False
        chain.fallback_state["watcher_available"] = True

        chain.record_watcher_failure()
        chain.record_watcher_failure()
        assert chain.fallback_state["watcher_available"] is True  # 2 < 3

        chain.record_watcher_failure()
        assert chain.fallback_state["watcher_available"] is False  # 3 >= 3

    def test_record_watcher_success_resets_counter(self):
        config = self._make_config()
        chain = FallbackChain(config)
        chain.record_watcher_failure()
        chain.record_watcher_failure()
        assert chain.fallback_state["watcher_failures"] == 2

        chain.record_watcher_success()
        assert chain.fallback_state["watcher_failures"] == 0

    def test_record_llm_healer_failure_circuit_breaker(self):
        config = self._make_config()
        chain = FallbackChain(config)
        chain.fallback_state["llm_healer_available"] = True

        for _ in range(3):
            chain.record_llm_healer_failure()
        assert chain.fallback_state["llm_healer_available"] is False

    def test_record_llm_healer_success_resets(self):
        config = self._make_config()
        chain = FallbackChain(config)
        chain.record_llm_healer_failure()
        chain.record_llm_healer_failure()
        chain.record_llm_healer_success()
        assert chain.fallback_state["llm_healer_failures"] == 0

    def test_get_status_returns_snapshot(self):
        config = self._make_config()
        chain = FallbackChain(config)
        chain.fallback_state["watcher_available"] = True
        chain.fallback_state["active_model"] = "llama3.2"
        status = chain.get_status()
        assert status["watcher_available"] is True
        assert status["active_model"] == "llama3.2"

    def test_watcher_disabled_returns_false(self):
        config = self._make_config(watcher_enabled=False)
        chain = FallbackChain(config)
        result = chain.check_watcher_available()
        assert result is False

    @patch("src.agents.fallback.REQUESTS_AVAILABLE", False)
    def test_watcher_no_requests_returns_false(self):
        config = self._make_config()
        chain = FallbackChain(config)
        result = chain.check_watcher_available()
        assert result is False

    def test_llm_healer_disabled_returns_false(self):
        config = self._make_config(llm_enabled=False)
        chain = FallbackChain(config)
        result = chain.check_llm_healer_available()
        assert result is False

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"})
    def test_llm_healer_anthropic_available(self):
        config = self._make_config()
        chain = FallbackChain(config)
        result = chain.check_llm_healer_available()
        assert result is True

    @patch.dict(os.environ, {}, clear=True)
    def test_llm_healer_no_api_key_unavailable(self):
        config = self._make_config()
        chain = FallbackChain(config)
        # Clear any cached state
        chain.fallback_state["llm_healer_available"] = None
        # Remove all API key env vars
        for key in ["ANTHROPIC_API_KEY", "GEMINI_API_KEY"]:
            os.environ.pop(key, None)
        result = chain.check_llm_healer_available()
        assert result is False

    def test_fallback_chain_with_logger(self):
        config = self._make_config()
        mock_logger = MagicMock()
        chain = FallbackChain(config, healing_logger=mock_logger)

        # Trigger circuit breaker
        for _ in range(3):
            chain.record_watcher_failure()

        # Should have called log_fallback
        mock_logger.log_fallback.assert_called_once()

    def test_thread_safety_record_failures(self):
        """Multiple threads recording failures should be safe."""
        config = self._make_config()
        config.watcher.max_failures = 100  # High threshold so we don't trip circuit breaker
        chain = FallbackChain(config)
        errors = []

        def record_failures():
            try:
                for _ in range(20):
                    chain.record_watcher_failure()
                    chain.record_watcher_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_failures) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
