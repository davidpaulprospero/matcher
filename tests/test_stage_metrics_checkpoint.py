"""
Tests for stage metrics persistence to checkpoint (US-49-012).

Covers:
- AC1: StageMetrics persisted under 'stage_metrics.DOWNLOAD_SEGMENTS' key at stage completion
- AC2: error_categories breakdown in persisted metrics
- AC3: escalation_summary (videos per tier, total escalations, tier effectiveness) in persisted metrics
- AC4: Stage metrics written to checkpoint after stage completion
- AC5: Metrics survive checkpoint restore and are accessible

Uses the existing checkpoint save/load infrastructure.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.checkpoint import CheckpointManager, CheckpointData, STAGE_ORDER
from src.stages import StageMetrics


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def checkpoint_manager(tmp_path):
    """Create a checkpoint manager with a temp directory."""
    return CheckpointManager(tmp_path, config_hash="test_hash")


@pytest.fixture
def sample_metrics_dict():
    """Sample serialized StageMetrics dict with error_categories and escalation_summary."""
    return {
        'items_processed': 15,
        'items_failed': 3,
        'duration_seconds': 42.5,
        'failed': False,
        'error_categories': {
            'network': 1,
            'bot_detection': 1,
            'video_specific': 1,
            'timeout': 0,
        },
        'escalation_summary': {
            'total_escalations': 4,
            'videos_per_tier': {
                'IMPERSONATION': 10,
                'EXTRACTOR_ARGS': 3,
                'COOKIE_ROTATION': 2,
            },
            'escalations_per_tier': {
                'EXTRACTOR_ARGS': 2,
                'COOKIE_ROTATION': 2,
            },
            'total_403s': 5,
            'total_successes': 15,
            'average_tier': 1.33,
            'tier_effectiveness': {
                'bot_detection': {
                    'tier_1': 0.8,
                    'tier_2': 0.6,
                    'tier_3': 0.9,
                },
            },
        },
    }


# ============================================================================
# AC1 + AC4: Stage metrics persisted to checkpoint at stage completion
# ============================================================================

class TestStageMetricsWrittenToCheckpoint:
    """AC1 + AC4: Verify stage metrics are written to checkpoint after stage completion."""

    @pytest.mark.fast
    def test_save_with_stage_metrics(self, checkpoint_manager, sample_metrics_dict):
        """Stage metrics are persisted under stage_metrics.DOWNLOAD_SEGMENTS key."""
        stage_data = {
            'segment_count': 15,
            'total_matches': 20,
            'retry_count': 2,
        }

        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            stage_data,
            stage_metrics=sample_metrics_dict,
        )

        # Verify in-memory
        assert 'DOWNLOAD_SEGMENTS' in checkpoint_manager.data.stage_metrics
        assert checkpoint_manager.data.stage_metrics['DOWNLOAD_SEGMENTS'] == sample_metrics_dict

        # Verify on disk
        cp_path = checkpoint_manager.checkpoint_path
        with open(cp_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        assert 'stage_metrics' in raw
        assert 'DOWNLOAD_SEGMENTS' in raw['stage_metrics']
        persisted = raw['stage_metrics']['DOWNLOAD_SEGMENTS']
        assert persisted['items_processed'] == 15
        assert persisted['items_failed'] == 3
        assert persisted['duration_seconds'] == 42.5

    @pytest.mark.fast
    def test_save_without_metrics_preserves_existing(self, checkpoint_manager, sample_metrics_dict):
        """Saving a different stage without metrics doesn't remove existing stage metrics."""
        # Save DOWNLOAD_SEGMENTS with metrics
        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 10},
            stage_metrics=sample_metrics_dict,
        )

        # Save OUTPUT without metrics
        checkpoint_manager.save('OUTPUT', {'otio_path': '/output/timeline.otio'})

        # DOWNLOAD_SEGMENTS metrics should still be there
        assert 'DOWNLOAD_SEGMENTS' in checkpoint_manager.data.stage_metrics
        assert checkpoint_manager.data.stage_metrics['DOWNLOAD_SEGMENTS']['items_processed'] == 15

    @pytest.mark.fast
    def test_get_stage_metrics_accessor(self, checkpoint_manager, sample_metrics_dict):
        """get_stage_metrics() returns persisted metrics for a specific stage."""
        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 10},
            stage_metrics=sample_metrics_dict,
        )

        result = checkpoint_manager.get_stage_metrics('DOWNLOAD_SEGMENTS')
        assert result == sample_metrics_dict

    @pytest.mark.fast
    def test_get_stage_metrics_missing_stage(self, checkpoint_manager):
        """get_stage_metrics() returns empty dict for stages with no metrics."""
        checkpoint_manager.save('ANALYZE', {'keywords': ['test']})

        assert checkpoint_manager.get_stage_metrics('ANALYZE') == {}
        assert checkpoint_manager.get_stage_metrics('DOWNLOAD_SEGMENTS') == {}

    @pytest.mark.fast
    def test_get_stage_metrics_no_data(self, checkpoint_manager):
        """get_stage_metrics() returns empty dict when no checkpoint data exists."""
        assert checkpoint_manager.get_stage_metrics('DOWNLOAD_SEGMENTS') == {}


# ============================================================================
# AC2: error_categories breakdown in persisted metrics
# ============================================================================

class TestErrorCategoriesInMetrics:
    """AC2: Verify error_categories breakdown is persisted."""

    @pytest.mark.fast
    def test_error_categories_persisted(self, checkpoint_manager):
        """error_categories dict with network/bot_detection/video_specific/timeout counts."""
        error_cats = {
            'network': 2,
            'bot_detection': 5,
            'video_specific': 1,
            'timeout': 3,
        }
        metrics_dict = StageMetrics(
            items_processed=10,
            items_failed=11,
            duration_seconds=30.0,
            error_categories=error_cats,
        ).to_dict()

        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 10},
            stage_metrics=metrics_dict,
        )

        # Reload from disk
        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        manager2.load()

        restored_metrics = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')
        assert restored_metrics['error_categories'] == error_cats
        assert restored_metrics['error_categories']['network'] == 2
        assert restored_metrics['error_categories']['bot_detection'] == 5
        assert restored_metrics['error_categories']['timeout'] == 3

    @pytest.mark.fast
    def test_empty_error_categories(self, checkpoint_manager):
        """Empty error_categories dict is handled gracefully."""
        metrics_dict = StageMetrics(
            items_processed=5,
            items_failed=0,
            duration_seconds=10.0,
        ).to_dict()

        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 5},
            stage_metrics=metrics_dict,
        )

        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        manager2.load()

        restored = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')
        # error_categories should be absent or empty
        assert restored.get('error_categories', {}) == {}


# ============================================================================
# AC3: escalation_summary in persisted metrics
# ============================================================================

class TestEscalationSummaryInMetrics:
    """AC3: Verify escalation_summary with tier breakdown is persisted."""

    @pytest.mark.fast
    def test_escalation_summary_persisted(self, checkpoint_manager, sample_metrics_dict):
        """escalation_summary with videos_per_tier, total_escalations, tier_effectiveness."""
        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 15},
            stage_metrics=sample_metrics_dict,
        )

        # Reload from disk
        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        manager2.load()

        restored = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')
        esc = restored['escalation_summary']

        # Videos per tier
        assert esc['videos_per_tier']['IMPERSONATION'] == 10
        assert esc['videos_per_tier']['EXTRACTOR_ARGS'] == 3
        assert esc['videos_per_tier']['COOKIE_ROTATION'] == 2

        # Total escalations
        assert esc['total_escalations'] == 4
        assert esc['escalations_per_tier']['EXTRACTOR_ARGS'] == 2
        assert esc['escalations_per_tier']['COOKIE_ROTATION'] == 2

        # Tier effectiveness
        assert esc['tier_effectiveness']['bot_detection']['tier_1'] == 0.8
        assert esc['tier_effectiveness']['bot_detection']['tier_3'] == 0.9

    @pytest.mark.fast
    def test_escalation_summary_totals(self, checkpoint_manager, sample_metrics_dict):
        """escalation_summary includes total_403s, total_successes, average_tier."""
        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 15},
            stage_metrics=sample_metrics_dict,
        )

        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        manager2.load()

        esc = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')['escalation_summary']
        assert esc['total_403s'] == 5
        assert esc['total_successes'] == 15
        assert esc['average_tier'] == 1.33


# ============================================================================
# AC5: Metrics survive checkpoint restore and are accessible
# ============================================================================

class TestMetricsSurviveRestore:
    """AC5: Metrics survive checkpoint save → disk → load cycle."""

    @pytest.mark.fast
    def test_full_roundtrip_metrics(self, checkpoint_manager, sample_metrics_dict):
        """Complete save → disk → load → access cycle preserves all metric fields."""
        # Save checkpoint with metrics
        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 15, 'total_matches': 20, 'retry_count': 2},
            stage_metrics=sample_metrics_dict,
        )

        # Create fresh manager and load from disk
        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        loaded = manager2.load()

        assert loaded is not None

        # Access via get_stage_metrics
        metrics = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')
        assert metrics['items_processed'] == 15
        assert metrics['items_failed'] == 3
        assert metrics['duration_seconds'] == 42.5
        assert metrics['failed'] is False
        assert metrics['error_categories']['bot_detection'] == 1
        assert metrics['escalation_summary']['total_escalations'] == 4

    @pytest.mark.fast
    def test_roundtrip_via_checkpoint_data(self, checkpoint_manager, sample_metrics_dict):
        """Metrics accessible via loaded CheckpointData.stage_metrics directly."""
        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 10},
            stage_metrics=sample_metrics_dict,
        )

        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        loaded = manager2.load()

        assert loaded.stage_metrics is not None
        assert 'DOWNLOAD_SEGMENTS' in loaded.stage_metrics
        dl_metrics = loaded.stage_metrics['DOWNLOAD_SEGMENTS']
        assert dl_metrics['items_processed'] == 15

    @pytest.mark.fast
    def test_from_dict_preserves_stage_metrics(self):
        """CheckpointData.from_dict() preserves stage_metrics field."""
        raw = {
            'version': '2.0',
            'created_at': '2026-02-04T12:00:00',
            'updated_at': '2026-02-04T12:00:00',
            'last_completed_stage': 'DOWNLOAD_SEGMENTS',
            'config_hash': 'abc',
            'voiceover_path': '',
            'voiceover_hash': '',
            'analyze': {},
            'video_search': {},
            'caption': {},
            'match': {},
            'iterative_match': {},
            'download_segments': {'segment_count': 5},
            'stage_metrics': {
                'DOWNLOAD_SEGMENTS': {
                    'items_processed': 5,
                    'items_failed': 0,
                    'duration_seconds': 10.0,
                    'failed': False,
                    'error_categories': {'network': 1},
                    'escalation_summary': {'total_escalations': 2},
                }
            },
        }

        cp = CheckpointData.from_dict(raw)
        assert cp.stage_metrics is not None
        assert 'DOWNLOAD_SEGMENTS' in cp.stage_metrics
        assert cp.stage_metrics['DOWNLOAD_SEGMENTS']['items_processed'] == 5
        assert cp.stage_metrics['DOWNLOAD_SEGMENTS']['escalation_summary']['total_escalations'] == 2

    @pytest.mark.fast
    def test_multiple_stages_metrics(self, checkpoint_manager):
        """Metrics from multiple stages coexist in checkpoint."""
        dl_metrics = StageMetrics(items_processed=10, items_failed=2, duration_seconds=30.0).to_dict()
        match_metrics = StageMetrics(items_processed=20, items_failed=0, duration_seconds=5.0).to_dict()

        checkpoint_manager.save('MATCH', {'match_count': 20}, stage_metrics=match_metrics)
        checkpoint_manager.save('DOWNLOAD_SEGMENTS', {'segment_count': 10}, stage_metrics=dl_metrics)

        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        manager2.load()

        match_m = manager2.get_stage_metrics('MATCH')
        dl_m = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')

        assert match_m['items_processed'] == 20
        assert dl_m['items_processed'] == 10
        assert dl_m['items_failed'] == 2


# ============================================================================
# StageMetrics serialization round-trip
# ============================================================================

class TestStageMetricsSerialization:
    """Verify StageMetrics to_dict/from_dict handles new fields."""

    @pytest.mark.fast
    def test_to_dict_includes_escalation_summary(self):
        """StageMetrics.to_dict() includes escalation_summary when set."""
        m = StageMetrics(
            items_processed=10,
            items_failed=2,
            duration_seconds=20.0,
            error_categories={'network': 1},
            escalation_summary={'total_escalations': 3, 'videos_per_tier': {'IMPERSONATION': 8}},
        )
        d = m.to_dict()
        assert 'escalation_summary' in d
        assert d['escalation_summary']['total_escalations'] == 3

    @pytest.mark.fast
    def test_to_dict_omits_empty_escalation_summary(self):
        """StageMetrics.to_dict() omits escalation_summary when empty."""
        m = StageMetrics(items_processed=5, items_failed=0, duration_seconds=10.0)
        d = m.to_dict()
        assert 'escalation_summary' not in d

    @pytest.mark.fast
    def test_from_dict_restores_escalation_summary(self):
        """StageMetrics.from_dict() restores escalation_summary."""
        d = {
            'items_processed': 10,
            'items_failed': 2,
            'duration_seconds': 20.0,
            'failed': False,
            'error_categories': {'bot_detection': 3},
            'escalation_summary': {
                'total_escalations': 5,
                'videos_per_tier': {'IMPERSONATION': 7, 'EXTRACTOR_ARGS': 3},
            },
        }
        m = StageMetrics.from_dict(d)
        assert m.escalation_summary['total_escalations'] == 5
        assert m.escalation_summary['videos_per_tier']['IMPERSONATION'] == 7
        assert m.error_categories['bot_detection'] == 3

    @pytest.mark.fast
    def test_from_dict_missing_escalation_summary(self):
        """StageMetrics.from_dict() defaults escalation_summary to empty dict."""
        d = {'items_processed': 5, 'items_failed': 0, 'duration_seconds': 10.0}
        m = StageMetrics.from_dict(d)
        assert m.escalation_summary == {}
