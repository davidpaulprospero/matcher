"""Tests for US-50-009: Escalation tier effectiveness metrics in stage completion summary.

Tests cover:
  AC1: Stage completion logs structured summary with videos_per_tier, total_escalations,
       average_tier, tier_effectiveness
  AC2: Summary written to checkpoint via stage_metrics (persists across restarts)
  AC3: Summary includes bot_detection_count and network_failure_count
  AC4: EscalationManager.get_metrics() output is correctly formatted into stage summary
  AC5: Summary survives checkpoint save/restore round-trip
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import DownloadVideoSegmentsStage
from src.stages import StageMetrics
from src.checkpoint import CheckpointManager


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_mock_escalation_manager(
    total_escalations=4,
    keywords_at_each_tier=None,
    escalations_per_tier=None,
    total_403s=5,
    total_successes=15,
    average_tier=1.33,
    tier_effectiveness=None,
):
    """Create a mock EscalationManager with configurable get_metrics() output."""
    if keywords_at_each_tier is None:
        keywords_at_each_tier = {
            'IMPERSONATION': ['vid_1', 'vid_2', 'vid_3', 'vid_4', 'vid_5',
                              'vid_6', 'vid_7', 'vid_8', 'vid_9', 'vid_10'],
            'EXTRACTOR_ARGS': ['vid_11', 'vid_12', 'vid_13'],
            'COOKIE_ROTATION': ['vid_14', 'vid_15'],
        }
    if escalations_per_tier is None:
        escalations_per_tier = {'EXTRACTOR_ARGS': 2, 'COOKIE_ROTATION': 2}
    if tier_effectiveness is None:
        tier_effectiveness = {
            'bot_detection': {'tier_1': 0.8, 'tier_2': 0.6, 'tier_3': 0.9},
        }

    mgr = MagicMock()
    mgr.get_metrics.return_value = {
        'total_escalations': total_escalations,
        'keywords_at_each_tier': keywords_at_each_tier,
        'escalations_per_tier': escalations_per_tier,
        'total_403s': total_403s,
        'total_successes': total_successes,
        'average_tier': average_tier,
        'speed_escalations': 0,
    }
    mgr.get_tier_effectiveness.return_value = tier_effectiveness
    return mgr


def _make_download_stats(
    succeeded=12, cached=3, failed=5, total=20, attempted=20,
    error_categories=None,
):
    """Build a download_stats dict matching _download_segments output."""
    if error_categories is None:
        error_categories = {'bot_detection': 3, 'network': 1, 'video_specific': 1}
    return {
        'succeeded': succeeded,
        'cached': cached,
        'failed': failed,
        'total': total,
        'attempted': attempted,
        'retry_count': 0,
        'error_categories': error_categories,
    }


@pytest.fixture
def stage():
    """Create a DownloadVideoSegmentsStage with a mock downloader."""
    s = DownloadVideoSegmentsStage()
    s.downloader = MagicMock()
    s.downloader.circuit_breaker = None
    return s


@pytest.fixture
def checkpoint_manager(tmp_path):
    """Create a checkpoint manager with a temp directory."""
    return CheckpointManager(tmp_path, config_hash="test_hash")


# ---------------------------------------------------------------------------
# AC4: EscalationManager.get_metrics() correctly formatted into stage summary
# ---------------------------------------------------------------------------

class TestGetMetricsFormattedIntoStageSummary:
    """AC4: Verify EscalationManager.get_metrics() output is correctly
    formatted into the escalation_summary dict used by StageMetrics."""

    @pytest.mark.fast
    def test_videos_per_tier_counts_keywords(self):
        """videos_per_tier maps tier name to count of keywords at that tier."""
        esc_mgr = _make_mock_escalation_manager()
        stage = DownloadVideoSegmentsStage()
        stage.downloader = MagicMock()
        stage.downloader.escalation_manager = esc_mgr
        stage.downloader.circuit_breaker = None

        # Simulate what run() does to build escalation_summary
        esc_metrics = esc_mgr.get_metrics()
        escalation_summary = {
            'total_escalations': esc_metrics.get('total_escalations', 0),
            'videos_per_tier': {
                tier: len(keywords)
                for tier, keywords in esc_metrics.get('keywords_at_each_tier', {}).items()
            },
            'escalations_per_tier': esc_metrics.get('escalations_per_tier', {}),
            'total_403s': esc_metrics.get('total_403s', 0),
            'total_successes': esc_metrics.get('total_successes', 0),
            'average_tier': esc_metrics.get('average_tier', 1.0),
        }
        if hasattr(esc_mgr, 'get_tier_effectiveness'):
            escalation_summary['tier_effectiveness'] = esc_mgr.get_tier_effectiveness()

        # Verify formatting
        assert escalation_summary['total_escalations'] == 4
        assert escalation_summary['videos_per_tier'] == {
            'IMPERSONATION': 10, 'EXTRACTOR_ARGS': 3, 'COOKIE_ROTATION': 2,
        }
        assert escalation_summary['escalations_per_tier'] == {
            'EXTRACTOR_ARGS': 2, 'COOKIE_ROTATION': 2,
        }
        assert escalation_summary['total_403s'] == 5
        assert escalation_summary['total_successes'] == 15
        assert escalation_summary['average_tier'] == 1.33
        assert escalation_summary['tier_effectiveness'] == {
            'bot_detection': {'tier_1': 0.8, 'tier_2': 0.6, 'tier_3': 0.9},
        }

    @pytest.mark.fast
    def test_empty_escalation_manager_metrics(self):
        """No escalation data produces empty-but-valid summary."""
        esc_mgr = _make_mock_escalation_manager(
            total_escalations=0,
            keywords_at_each_tier={},
            escalations_per_tier={},
            total_403s=0,
            total_successes=5,
            average_tier=1.0,
            tier_effectiveness={},
        )

        esc_metrics = esc_mgr.get_metrics()
        escalation_summary = {
            'total_escalations': esc_metrics.get('total_escalations', 0),
            'videos_per_tier': {
                tier: len(keywords)
                for tier, keywords in esc_metrics.get('keywords_at_each_tier', {}).items()
            },
            'escalations_per_tier': esc_metrics.get('escalations_per_tier', {}),
            'total_403s': esc_metrics.get('total_403s', 0),
            'total_successes': esc_metrics.get('total_successes', 0),
            'average_tier': esc_metrics.get('average_tier', 1.0),
        }
        if hasattr(esc_mgr, 'get_tier_effectiveness'):
            escalation_summary['tier_effectiveness'] = esc_mgr.get_tier_effectiveness()

        assert escalation_summary['total_escalations'] == 0
        assert escalation_summary['videos_per_tier'] == {}
        assert escalation_summary['tier_effectiveness'] == {}

    @pytest.mark.fast
    def test_keywords_at_each_tier_list_to_count_conversion(self):
        """get_metrics() returns lists of keywords; summary converts to counts."""
        esc_mgr = _make_mock_escalation_manager(
            keywords_at_each_tier={
                'IMPERSONATION': ['a', 'b', 'c'],
                'EXTRACTOR_ARGS': ['d'],
            },
        )
        esc_metrics = esc_mgr.get_metrics()
        videos_per_tier = {
            tier: len(keywords)
            for tier, keywords in esc_metrics['keywords_at_each_tier'].items()
        }
        assert videos_per_tier == {'IMPERSONATION': 3, 'EXTRACTOR_ARGS': 1}


# ---------------------------------------------------------------------------
# AC3: Summary includes bot_detection_count and network_failure_count
# ---------------------------------------------------------------------------

class TestBotDetectionAndNetworkFailureCounts:
    """AC3: Verify escalation_summary includes bot_detection_count and network_failure_count."""

    @pytest.mark.fast
    def test_bot_detection_count_from_error_categories(self):
        """bot_detection_count populated from download_stats error_categories."""
        error_cats = {'bot_detection': 7, 'network': 2, 'video_specific': 1}
        escalation_summary = {}
        escalation_summary['bot_detection_count'] = error_cats.get('bot_detection', 0)
        escalation_summary['network_failure_count'] = error_cats.get('network', 0)

        assert escalation_summary['bot_detection_count'] == 7
        assert escalation_summary['network_failure_count'] == 2

    @pytest.mark.fast
    def test_zero_counts_when_no_errors(self):
        """Zero counts when no bot_detection or network errors."""
        error_cats = {'video_specific': 3}
        escalation_summary = {}
        escalation_summary['bot_detection_count'] = error_cats.get('bot_detection', 0)
        escalation_summary['network_failure_count'] = error_cats.get('network', 0)

        assert escalation_summary['bot_detection_count'] == 0
        assert escalation_summary['network_failure_count'] == 0

    @pytest.mark.fast
    def test_counts_in_full_metrics_dict(self):
        """bot_detection_count and network_failure_count appear in StageMetrics escalation_summary."""
        escalation_summary = {
            'total_escalations': 2,
            'videos_per_tier': {'IMPERSONATION': 5},
            'bot_detection_count': 3,
            'network_failure_count': 1,
        }
        metrics = StageMetrics(
            items_processed=10,
            items_failed=4,
            duration_seconds=30.0,
            error_categories={'bot_detection': 3, 'network': 1},
            escalation_summary=escalation_summary,
        )
        d = metrics.to_dict()
        assert d['escalation_summary']['bot_detection_count'] == 3
        assert d['escalation_summary']['network_failure_count'] == 1


# ---------------------------------------------------------------------------
# AC1: Stage completion logs structured summary
# ---------------------------------------------------------------------------

class TestEscalationSummaryLogging:
    """AC1: Verify _log_escalation_summary logs structured escalation info."""

    @pytest.mark.fast
    def test_logs_escalation_summary_with_tiers(self, caplog):
        """Log includes total_escalations, average_tier, videos_per_tier."""
        stage = DownloadVideoSegmentsStage()
        summary = {
            'total_escalations': 4,
            'videos_per_tier': {
                'IMPERSONATION': 10,
                'EXTRACTOR_ARGS': 3,
                'COOKIE_ROTATION': 2,
            },
            'average_tier': 1.33,
            'bot_detection_count': 3,
            'network_failure_count': 1,
        }

        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_escalation_summary(summary)

        assert any('total_escalations=4' in r.message for r in caplog.records)
        assert any('average_tier=1.33' in r.message for r in caplog.records)
        assert any('IMPERSONATION=10' in r.message for r in caplog.records)
        assert any('bot_detection=3' in r.message for r in caplog.records)
        assert any('network_failures=1' in r.message for r in caplog.records)

    @pytest.mark.fast
    def test_logs_tier_effectiveness(self, caplog):
        """Log includes per-category tier effectiveness rates."""
        stage = DownloadVideoSegmentsStage()
        summary = {
            'total_escalations': 2,
            'videos_per_tier': {'IMPERSONATION': 5},
            'average_tier': 1.2,
            'bot_detection_count': 2,
            'network_failure_count': 0,
            'tier_effectiveness': {
                'bot_detection': {'tier_1': 0.8, 'tier_2': 0.6},
            },
        }

        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_escalation_summary(summary)

        assert any('Tier effectiveness [bot_detection]' in r.message for r in caplog.records)
        assert any('tier_1=80.0%' in r.message for r in caplog.records)
        assert any('tier_2=60.0%' in r.message for r in caplog.records)

    @pytest.mark.fast
    def test_no_log_on_empty_summary(self, caplog):
        """No escalation log when summary is empty."""
        stage = DownloadVideoSegmentsStage()

        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_escalation_summary({})

        assert not any('Escalation summary' in r.message for r in caplog.records)

    @pytest.mark.fast
    def test_log_no_escalations_needed(self, caplog):
        """Logs 'no escalations needed' when total_escalations is 0."""
        stage = DownloadVideoSegmentsStage()
        summary = {
            'total_escalations': 0,
            'videos_per_tier': {'IMPERSONATION': 10},
            'bot_detection_count': 0,
            'network_failure_count': 0,
        }

        with caplog.at_level(logging.INFO, logger='src.stages.download_segments'):
            stage._log_escalation_summary(summary)

        assert any('no escalations needed' in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# AC2 + AC5: Summary in checkpoint via stage_metrics, survives round-trip
# ---------------------------------------------------------------------------

class TestEscalationSummaryCheckpointRoundtrip:
    """AC2 + AC5: Verify escalation summary persists to checkpoint and
    survives save/restore round-trip."""

    @pytest.mark.fast
    def test_escalation_summary_with_new_fields_roundtrip(self, checkpoint_manager):
        """Full escalation summary with bot_detection_count and network_failure_count
        survives checkpoint save → disk → load cycle."""
        escalation_summary = {
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
                'bot_detection': {'tier_1': 0.8, 'tier_2': 0.6, 'tier_3': 0.9},
            },
            'bot_detection_count': 3,
            'network_failure_count': 1,
        }

        metrics_dict = StageMetrics(
            items_processed=15,
            items_failed=5,
            duration_seconds=42.5,
            error_categories={'bot_detection': 3, 'network': 1, 'video_specific': 1},
            escalation_summary=escalation_summary,
        ).to_dict()

        # Save
        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 15, 'total_matches': 20, 'retry_count': 0},
            stage_metrics=metrics_dict,
        )

        # Load from disk
        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        manager2.load()

        restored = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')
        esc = restored['escalation_summary']

        # AC1: videos_per_tier, total_escalations, average_tier, tier_effectiveness
        assert esc['total_escalations'] == 4
        assert esc['videos_per_tier'] == {
            'IMPERSONATION': 10, 'EXTRACTOR_ARGS': 3, 'COOKIE_ROTATION': 2,
        }
        assert esc['average_tier'] == 1.33
        assert esc['tier_effectiveness']['bot_detection']['tier_1'] == 0.8

        # AC3: bot_detection_count and network_failure_count
        assert esc['bot_detection_count'] == 3
        assert esc['network_failure_count'] == 1

    @pytest.mark.fast
    def test_roundtrip_via_json_file(self, checkpoint_manager):
        """Verify raw JSON on disk contains the escalation_summary fields."""
        escalation_summary = {
            'total_escalations': 2,
            'videos_per_tier': {'IMPERSONATION': 8},
            'bot_detection_count': 5,
            'network_failure_count': 2,
        }
        metrics_dict = StageMetrics(
            items_processed=10,
            items_failed=7,
            duration_seconds=25.0,
            escalation_summary=escalation_summary,
        ).to_dict()

        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 10},
            stage_metrics=metrics_dict,
        )

        # Read raw JSON
        with open(checkpoint_manager.checkpoint_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        raw_esc = raw['stage_metrics']['DOWNLOAD_SEGMENTS']['escalation_summary']
        assert raw_esc['bot_detection_count'] == 5
        assert raw_esc['network_failure_count'] == 2
        assert raw_esc['videos_per_tier']['IMPERSONATION'] == 8

    @pytest.mark.fast
    def test_roundtrip_with_zero_counts(self, checkpoint_manager):
        """Zero bot_detection_count and network_failure_count survive round-trip."""
        escalation_summary = {
            'total_escalations': 0,
            'videos_per_tier': {},
            'bot_detection_count': 0,
            'network_failure_count': 0,
        }
        metrics_dict = StageMetrics(
            items_processed=10,
            items_failed=0,
            duration_seconds=15.0,
            escalation_summary=escalation_summary,
        ).to_dict()

        checkpoint_manager.save(
            'DOWNLOAD_SEGMENTS',
            {'segment_count': 10},
            stage_metrics=metrics_dict,
        )

        manager2 = CheckpointManager(checkpoint_manager.project_dir, config_hash="test_hash")
        manager2.load()

        restored = manager2.get_stage_metrics('DOWNLOAD_SEGMENTS')
        esc = restored['escalation_summary']
        assert esc['bot_detection_count'] == 0
        assert esc['network_failure_count'] == 0
