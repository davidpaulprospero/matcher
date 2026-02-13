"""Tests for escalation state persistence via CheckpointManager (US-89-003).

Tests cover:
  - CheckpointData has escalation_state field
  - save_escalation_state() serializes EscalationManager to checkpoint
  - load_escalation_state() restores EscalationManager from checkpoint
  - Edge case: keywords in saved state that no longer exist in current run are filtered
  - Round-trip: save then load preserves escalation tiers
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from src.checkpoint import CheckpointManager, CheckpointData
from src.downloader.escalation_manager import EscalationManager
from src.downloader.types import EscalationTier


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@dataclass
class FakeExtractorArgsConfig:
    """Minimal stand-in for ExtractorArgsConfig."""
    enabled: bool = True
    player_clients: List[str] = field(
        default_factory=lambda: ["web_safari", "tv_downgraded", "web"]
    )
    escalation_threshold: int = 2
    cooldown_seconds: float = 0.0  # No cooldown for test speed
    max_tier: int = 3


def _make_impersonation_manager():
    """Create a mock ImpersonationManager."""
    mgr = MagicMock()
    mgr.get_impersonate_args.return_value = [
        "--impersonate", "Chrome-136:Macos-15"
    ]
    return mgr


def _make_manager(cooldown: float = 0.0) -> EscalationManager:
    """Create an EscalationManager with zero cooldown for fast tests."""
    config = FakeExtractorArgsConfig(cooldown_seconds=cooldown)
    return EscalationManager(
        impersonation_manager=_make_impersonation_manager(),
        extractor_args_config=config,
    )


def _force_escalation(mgr: EscalationManager, keyword: str,
                       error_output: str = "HTTP Error 403") -> None:
    """Force an escalation by recording enough failures to cross threshold."""
    mgr.record_failure(keyword, error_output)
    mgr.record_failure(keyword, error_output)


# ---------------------------------------------------------------------------
# AC1: CheckpointData has escalation_state field
# ---------------------------------------------------------------------------

class TestCheckpointDataEscalationField:
    """Verify CheckpointData has escalation_state field."""

    @pytest.mark.fast
    def test_checkpoint_data_has_escalation_field(self):
        """CheckpointData should have escalation_state field."""
        data = CheckpointData()
        assert hasattr(data, 'escalation_state')

    @pytest.mark.fast
    def test_escalation_state_defaults_to_empty_dict(self):
        """escalation_state should default to empty dict."""
        data = CheckpointData()
        assert data.escalation_state == {}

    @pytest.mark.fast
    def test_escalation_state_serializes_to_dict(self):
        """escalation_state should serialize to dict."""
        data = CheckpointData()
        data.escalation_state = {'keyword_states': {'test': {'tier': 1}}}
        d = data.to_dict()
        assert 'escalation_state' in d
        assert d['escalation_state'] == {'keyword_states': {'test': {'tier': 1}}}

    @pytest.mark.fast
    def test_escalation_state_deserializes_from_dict(self):
        """escalation_state should deserialize from dict."""
        raw = {
            'version': '2.0',
            'created_at': '',
            'updated_at': '',
            'last_completed_stage': '',
            'config_hash': '',
            'voiceover_path': '',
            'voiceover_hash': '',
            'analyze': {},
            'video_search': {},
            'caption': {},
            'match': {},
            'iterative_match': {},
            'download_segments': {},
            'chapter_data': {},
            'stage_metrics': {},
            'transcription_metrics': {},
            'validation_cache': {},
            'escalation_state': {'keyword_states': {'test': {'tier': 2}}},
        }
        data = CheckpointData.from_dict(raw)
        assert data.escalation_state == {'keyword_states': {'test': {'tier': 2}}}


# ---------------------------------------------------------------------------
# AC2: save_escalation_state() serializes EscalationManager to checkpoint
# ---------------------------------------------------------------------------

class TestSaveEscalationState:
    """Verify save_escalation_state() correctly serializes escalation state."""

    @pytest.mark.fast
    def test_save_escalation_state_basic(self, tmp_path):
        """save_escalation_state should save escalation state to checkpoint."""
        # Setup checkpoint manager
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            config_hash="test123"
        )

        # Setup escalation manager with some state
        esc_mgr = _make_manager()
        _force_escalation(esc_mgr, "beach", "HTTP Error 403")
        _force_escalation(esc_mgr, "mountain", "HTTP Error 429")

        # Save
        cp_manager.save_escalation_state(esc_mgr)

        # Verify saved
        assert cp_manager.data.escalation_state
        assert 'keyword_states' in cp_manager.data.escalation_state
        assert 'beach' in cp_manager.data.escalation_state['keyword_states']
        assert 'mountain' in cp_manager.data.escalation_state['keyword_states']

    @pytest.mark.fast
    def test_save_escalation_state_creates_checkpoint_if_missing(self, tmp_path):
        """save_escalation_state should create checkpoint data if missing."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        # data is None initially

        esc_mgr = _make_manager()
        cp_manager.save_escalation_state(esc_mgr)

        # Should have created
        assert cp_manager.data is not None
        assert cp_manager.data.escalation_state

    @pytest.mark.fast
    def test_save_escalation_state_empty_manager(self, tmp_path):
        """save_escalation_state should handle empty escalation manager."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            config_hash="test123"
        )

        esc_mgr = _make_manager()  # No state
        cp_manager.save_escalation_state(esc_mgr)

        assert cp_manager.data.escalation_state
        assert cp_manager.data.escalation_state.get('keyword_states', {}) == {}


# ---------------------------------------------------------------------------
# AC3: load_escalation_state() restores EscalationManager from checkpoint
# ---------------------------------------------------------------------------

class TestLoadEscalationState:
    """Verify load_escalation_state() correctly restores escalation state."""

    @pytest.mark.fast
    def test_load_escalation_state_basic(self, tmp_path):
        """load_escalation_state should restore escalation state from checkpoint."""
        # Setup with saved state
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            config_hash="test123",
            escalation_state={
                'keyword_states': {
                    'beach': {'tier': EscalationTier.EXTRACTOR_ARGS.value, 'consecutive_403s': 0},
                    'mountain': {'tier': EscalationTier.IMPERSONATE_ONLY.value, 'consecutive_403s': 1},
                },
                'saved_at': time.time(),
            }
        )

        # Create fresh escalation manager
        esc_mgr = _make_manager()

        # Load
        cp_manager.load_escalation_state(esc_mgr)

        # Verify restored
        assert 'beach' in esc_mgr.keyword_states
        assert 'mountain' in esc_mgr.keyword_states
        assert esc_mgr.keyword_states['beach'].current_tier == EscalationTier.EXTRACTOR_ARGS
        assert esc_mgr.keyword_states['mountain'].current_tier == EscalationTier.IMPERSONATE_ONLY

    @pytest.mark.fast
    def test_load_escalation_state_no_checkpoint(self, tmp_path):
        """load_escalation_state should handle missing checkpoint gracefully."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        # No data

        esc_mgr = _make_manager()
        cp_manager.load_escalation_state(esc_mgr)

        # Should not crash, state should be empty
        assert esc_mgr.keyword_states == {}

    @pytest.mark.fast
    def test_load_escalation_state_empty_state(self, tmp_path):
        """load_escalation_state should handle empty escalation_state."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            config_hash="test123",
            escalation_state={}
        )

        esc_mgr = _make_manager()
        cp_manager.load_escalation_state(esc_mgr)

        assert esc_mgr.keyword_states == {}


# ---------------------------------------------------------------------------
# AC4: Edge case - keywords in saved state that no longer exist in current run
# ---------------------------------------------------------------------------

class TestStaleKeywordsFiltering:
    """Verify keywords that no longer exist are filtered during restoration."""

    @pytest.mark.fast
    def test_filter_stale_keywords(self, tmp_path):
        """Keywords not in current_keywords should be excluded."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            config_hash="test123",
            escalation_state={
                'keyword_states': {
                    'beach': {'tier': 2, 'consecutive_403s': 0},
                    'mountain': {'tier': 1, 'consecutive_403s': 1},
                    'river': {'tier': 1, 'consecutive_403s': 0},  # Will be filtered
                },
                'saved_at': time.time(),
            }
        )

        esc_mgr = _make_manager()

        # Only current keywords are beach and mountain
        cp_manager.load_escalation_state(esc_mgr, current_keywords=['beach', 'mountain'])

        # river should be filtered out
        assert 'river' not in esc_mgr.keyword_states
        assert 'beach' in esc_mgr.keyword_states
        assert 'mountain' in esc_mgr.keyword_states

    @pytest.mark.fast
    def test_filter_stale_keywords_logs_exclusion(self, tmp_path, caplog):
        """Filtering stale keywords should log the exclusion."""
        import logging
        caplog.set_level(logging.INFO)

        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            config_hash="test123",
            escalation_state={
                'keyword_states': {
                    'stale_kw': {'tier': 1, 'consecutive_403s': 0},
                },
                'saved_at': time.time(),
            }
        )

        esc_mgr = _make_manager()
        cp_manager.load_escalation_state(esc_mgr, current_keywords=['active_kw'])

        assert 'stale_kw' not in esc_mgr.keyword_states
        assert 'Excluding 1 keywords' in caplog.text

    @pytest.mark.fast
    def test_all_keywords_stale_returns_empty(self, tmp_path):
        """If all saved keywords are stale, restore should return empty state."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            created_at="2026-01-01T00:00:00",
            config_hash="test123",
            escalation_state={
                'keyword_states': {
                    'old1': {'tier': 1, 'consecutive_403s': 0},
                    'old2': {'tier': 2, 'consecutive_403s': 0},
                },
                'saved_at': time.time(),
            }
        )

        esc_mgr = _make_manager()
        cp_manager.load_escalation_state(esc_mgr, current_keywords=['new1', 'new2'])

        assert esc_mgr.keyword_states == {}


# ---------------------------------------------------------------------------
# AC5: Round-trip - save then load preserves escalation tiers
# ---------------------------------------------------------------------------

class TestEscalationStateRoundTrip:
    """Verify save then load preserves escalation state completely."""

    @pytest.mark.fast
    def test_roundtrip_preserves_tiers(self, tmp_path):
        """Save and load should preserve exact tier for each keyword."""
        # Setup checkpoint
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")

        # Create escalation manager with specific state
        esc_mgr = _make_manager()
        _force_escalation(esc_mgr, "beach", "HTTP Error 403")
        _force_escalation(esc_mgr, "beach", "HTTP Error 403")  # Tier 2 -> 3

        # Save
        cp_manager.save_escalation_state(esc_mgr)

        # Create new manager and load
        esc_mgr2 = _make_manager()
        cp_manager.load_escalation_state(esc_mgr2)

        # Verify tiers preserved
        assert esc_mgr2.keyword_states['beach'].current_tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_roundtrip_preserves_multiple_keywords(self, tmp_path):
        """Save and load should preserve state for multiple keywords."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")

        esc_mgr = _make_manager()
        _force_escalation(esc_mgr, "kw1", "HTTP Error 403")
        _force_escalation(esc_mgr, "kw2", "HTTP Error 429")
        _force_escalation(esc_mgr, "kw2", "HTTP Error 429")  # Tier 2 -> 3

        cp_manager.save_escalation_state(esc_mgr)

        esc_mgr2 = _make_manager()
        cp_manager.load_escalation_state(esc_mgr2)

        assert esc_mgr2.keyword_states['kw1'].current_tier == EscalationTier.EXTRACTOR_ARGS
        assert esc_mgr2.keyword_states['kw2'].current_tier == EscalationTier.FULL_BYPASS

    @pytest.mark.fast
    def test_roundtrip_preserves_metrics(self, tmp_path):
        """Save and load should preserve escalation metrics."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")

        esc_mgr = _make_manager()
        esc_mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=True)
        esc_mgr.record_outcome('403', EscalationTier.IMPERSONATE_ONLY, success=False)

        cp_manager.save_escalation_state(esc_mgr)

        esc_mgr2 = _make_manager()
        cp_manager.load_escalation_state(esc_mgr2)

        effectiveness = esc_mgr2.get_tier_effectiveness()
        assert '403' in effectiveness
        assert effectiveness['403']['tier_1'] == 0.5


# ---------------------------------------------------------------------------
# AC6: has_escalation_state() check
# ---------------------------------------------------------------------------

class TestHasEscalationState:
    """Verify has_escalation_state() correctly detects saved state."""

    @pytest.mark.fast
    def test_has_escalation_state_true(self, tmp_path):
        """has_escalation_state should return True when state exists."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(
            escalation_state={'keyword_states': {}}
        )

        assert cp_manager.has_escalation_state() is True

    @pytest.mark.fast
    def test_has_escalation_state_false_no_data(self, tmp_path):
        """has_escalation_state should return False when no data."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")

        assert cp_manager.has_escalation_state() is False

    @pytest.mark.fast
    def test_has_escalation_state_false_empty_state(self, tmp_path):
        """has_escalation_state should return False when state is empty dict."""
        cp_manager = CheckpointManager(tmp_path, config_hash="test123")
        cp_manager.data = CheckpointData(escalation_state={})

        assert cp_manager.has_escalation_state() is False
