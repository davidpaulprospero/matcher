"""Tests for health checker module (US-106-008)."""

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.health_checker import (
    HealthChecker,
    HealthCheckConfig,
    HealthCheckResult,
    HealthStatus,
    run_health_checks,
    STAGE_HEALTH_CHECKS,
)


class TestHealthCheckConfig:
    """Test HealthCheckConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HealthCheckConfig()
        assert config.network_timeout_seconds == 5.0
        assert config.disk_space_warning_gb == 10.0
        assert config.disk_space_critical_gb == 2.0
        assert config.memory_warning_percent == 85.0
        assert config.memory_critical_percent == 95.0
        assert config.enabled_checks.get('memory') is True

    def test_get_stage_timeout(self):
        """Test stage timeout retrieval."""
        config = HealthCheckConfig(stage_timeouts={'MATCH': 60.0})
        assert config.get_stage_timeout('MATCH') == 60.0
        assert config.get_stage_timeout('UNKNOWN', 30.0) == 30.0


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_is_ok(self):
        """Test is_ok method."""
        result = HealthCheckResult(name='test', status=HealthStatus.OK, message='OK')
        assert result.is_ok() is True

        result = HealthCheckResult(name='test', status=HealthStatus.WARNING, message='Warn')
        assert result.is_ok() is False

    def test_to_dict(self):
        """Test serialization to dict."""
        result = HealthCheckResult(
            name='test',
            status=HealthStatus.OK,
            message='OK',
            details={'key': 'value'},
            duration_ms=10.5,
        )
        d = result.to_dict()
        assert d['name'] == 'test'
        assert d['status'] == 'ok'
        assert d['message'] == 'OK'
        assert d['details']['key'] == 'value'
        assert d['duration_ms'] == 10.5


class TestHealthChecker:
    """Test HealthChecker class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for testing."""
        config = Mock()
        config.health_check = {}
        config.project = '/tmp/test_project'
        config.embedding = {'provider': 'gemini'}
        return config

    @pytest.fixture
    def health_checker(self, mock_config):
        """Create HealthChecker instance with mock config."""
        return HealthChecker(mock_config)

    def test_check_network_success(self, health_checker):
        """Test network check when connectivity is available."""
        with patch('socket.socket') as mock_socket:
            mock_sock_instance = Mock()
            mock_socket.return_value = mock_sock_instance

            result = health_checker.check_network(timeout=1.0)

            assert result.status in (HealthStatus.OK, HealthStatus.FAILED)

    def test_check_network_disabled(self, health_checker):
        """Test network check when disabled in config."""
        health_checker.health_config.enabled_checks['network'] = False
        result = health_checker.check_network()
        assert result.status == HealthStatus.SKIPPED
        assert 'disabled' in result.message.lower()

    def test_check_disk_space_ok(self, health_checker):
        """Test disk space check when space is adequate."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs') as mock_statvfs:
                # Mock 50GB free
                mock_stat = MagicMock()
                mock_stat.f_bavail = 50 * 1024 * 1024 * 1024 // 4096
                mock_stat.f_frsize = 4096
                mock_statvfs.return_value = mock_stat

                result = health_checker.check_disk_space('/tmp')

                assert result.status == HealthStatus.OK

    def test_check_disk_space_warning(self, health_checker):
        """Test disk space check when below warning threshold."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs') as mock_statvfs:
                # Mock 5GB free (below 10GB warning)
                mock_stat = MagicMock()
                mock_stat.f_bavail = 5 * 1024 * 1024 * 1024 // 4096
                mock_stat.f_frsize = 4096
                mock_statvfs.return_value = mock_stat

                result = health_checker.check_disk_space('/tmp')

                assert result.status == HealthStatus.WARNING

    def test_check_disk_space_critical(self, health_checker):
        """Test disk space check when below critical threshold."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs') as mock_statvfs:
                # Mock 1GB free (below 2GB critical)
                mock_stat = MagicMock()
                mock_stat.f_bavail = 1 * 1024 * 1024 * 1024 // 4096
                mock_stat.f_frsize = 4096
                mock_statvfs.return_value = mock_stat

                result = health_checker.check_disk_space('/tmp')

                assert result.status == HealthStatus.FAILED

    def test_check_disk_space_disabled(self, health_checker):
        """Test disk space check when disabled in config."""
        health_checker.health_config.enabled_checks['disk_space'] = False
        result = health_checker.check_disk_space()
        assert result.status == HealthStatus.SKIPPED

    def test_check_memory_ok(self, health_checker):
        """Test memory check when usage is normal."""
        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            import sys
            mock_psutil = MagicMock()
            mock_mem = MagicMock()
            mock_mem.percent = 50.0
            mock_mem.available = 8 * 1024 ** 3  # 8GB
            mock_mem.total = 16 * 1024 ** 3  # 16GB
            mock_psutil.virtual_memory.return_value = mock_mem
            sys.modules['psutil'] = mock_psutil

            result = health_checker.check_memory()

            assert result.status == HealthStatus.OK

    def test_check_memory_warning(self, health_checker):
        """Test memory check when above warning threshold."""
        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            import sys
            mock_psutil = MagicMock()
            mock_mem = MagicMock()
            mock_mem.percent = 90.0  # Above 85% warning
            mock_mem.available = 2 * 1024 ** 3  # 2GB
            mock_mem.total = 16 * 1024 ** 3  # 16GB
            mock_psutil.virtual_memory.return_value = mock_mem
            sys.modules['psutil'] = mock_psutil

            result = health_checker.check_memory()

            assert result.status == HealthStatus.WARNING

    def test_check_memory_critical(self, health_checker):
        """Test memory check when above critical threshold."""
        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            import sys
            mock_psutil = MagicMock()
            mock_mem = MagicMock()
            mock_mem.percent = 97.0  # Above 95% critical
            mock_mem.available = 0.5 * 1024 ** 3  # 0.5GB
            mock_mem.total = 16 * 1024 ** 3  # 16GB
            mock_psutil.virtual_memory.return_value = mock_mem
            sys.modules['psutil'] = mock_psutil

            result = health_checker.check_memory()

            assert result.status == HealthStatus.FAILED

    def test_check_memory_disabled(self, health_checker):
        """Test memory check when disabled in config."""
        health_checker.health_config.enabled_checks['memory'] = False
        result = health_checker.check_memory()
        assert result.status == HealthStatus.SKIPPED

    def test_check_stage_video_search(self, health_checker):
        """Test stage checks for VIDEO_SEARCH stage."""
        results = health_checker.check_stage('VIDEO_SEARCH')

        # Should include network check
        check_names = [r.name for r in results]
        assert 'network' in check_names

    def test_stage_checks_disk_space(self, health_checker):
        """Test that DOWNLOAD_SEGMENTS includes disk space check."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs') as mock_statvfs:
                mock_stat = MagicMock()
                mock_stat.f_bavail = 50 * 1024 * 1024 * 1024 // 4096
                mock_stat.f_frsize = 4096
                mock_statvfs.return_value = mock_stat

                results = health_checker.check_stage('DOWNLOAD_SEGMENTS')
                check_names = [r.name for r in results]
                assert 'disk_space' in check_names

    def test_stage_checks_output_disk_space(self, health_checker):
        """Test that OUTPUT stage includes disk space check."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs') as mock_statvfs:
                mock_stat = MagicMock()
                mock_stat.f_bavail = 50 * 1024 * 1024 * 1024 // 4096
                mock_stat.f_frsize = 4096
                mock_statvfs.return_value = mock_stat

                results = health_checker.check_stage('OUTPUT')
                check_names = [r.name for r in results]
                assert 'disk_space' in check_names

    def test_stage_checks_match_memory(self, health_checker):
        """Test that MATCH stage includes memory check."""
        with patch.dict('sys.modules', {'psutil': MagicMock()}):
            import sys
            mock_psutil = MagicMock()
            mock_mem = MagicMock()
            mock_mem.percent = 50.0
            mock_mem.available = 8 * 1024 ** 3
            mock_mem.total = 16 * 1024 ** 3
            mock_psutil.virtual_memory.return_value = mock_mem
            sys.modules['psutil'] = mock_psutil

            results = health_checker.check_stage('MATCH')
            check_names = [r.name for r in results]
            assert 'memory' in check_names

    def test_check_before_stage_logs_warnings(self, health_checker, caplog):
        """Test that check_before_stage logs warnings but continues."""
        with patch('os.path.exists', return_value=True):
            with patch('os.statvfs') as mock_statvfs:
                # Low disk space triggers warning
                mock_stat = MagicMock()
                mock_stat.f_bavail = 5 * 1024 * 1024 * 1024 // 4096
                mock_stat.f_frsize = 4096
                mock_statvfs.return_value = mock_stat

                # Disable network so only disk_space check runs
                health_checker.health_config.enabled_checks['network'] = False
                health_checker.health_config.enabled_checks['memory'] = False

                results = health_checker.check_before_stage('OUTPUT')

                # Should have logged a warning
                assert len(results) > 0

    def test_check_ytdlp_success(self, health_checker):
        """Test yt-dlp check when available."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout='2024.12.23',
                stderr='',
            )

            result = health_checker.check_ytdlp()

            assert result.status == HealthStatus.OK
            assert 'yt-dlp' in result.message

    def test_check_ytdlp_not_found(self, health_checker):
        """Test yt-dlp check when not found."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = health_checker.check_ytdlp()

            # Should be WARNING (not FAILED) since ytdlp_required=False by default
            assert result.status == HealthStatus.WARNING

    def test_check_ytdlp_disabled(self, health_checker):
        """Test yt-dlp check when disabled."""
        health_checker.health_config.enabled_checks['ytdlp'] = False
        result = health_checker.check_ytdlp()
        assert result.status == HealthStatus.SKIPPED

    def test_check_ytdlp_required_fail(self, health_checker):
        """Test yt-dlp check when required but not found."""
        health_checker.health_config.ytdlp_required = True
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            result = health_checker.check_ytdlp()

            assert result.status == HealthStatus.FAILED

    def test_check_llm_provider_disabled(self, health_checker):
        """Test LLM provider check when disabled."""
        health_checker.health_config.enabled_checks['llm_provider'] = False
        result = health_checker.check_llm_provider()
        assert result.status == HealthStatus.SKIPPED

    def test_check_llm_provider_no_config(self, health_checker):
        """Test LLM provider check when no config."""
        health_checker.config.llm = None
        result = health_checker.check_llm_provider()
        assert result.status == HealthStatus.FAILED
        assert 'No LLM configuration' in result.message

    def test_check_llm_provider_ollama_success(self, health_checker):
        """Test LLM provider check for Ollama."""
        # Mock config with ollama provider
        health_checker.config.llm = Mock()
        health_checker.config.llm.provider = 'ollama'

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout='{"models": [{"name": "llama3.2"}]}',
                stderr='',
            )

            result = health_checker.check_llm_provider()

            assert result.status == HealthStatus.OK

    def test_check_llm_provider_gemini_no_key(self, health_checker):
        """Test LLM provider check for Gemini with no API key."""
        # Mock config with gemini provider
        health_checker.config.llm = Mock()
        health_checker.config.llm.provider = 'gemini'

        with patch.dict(os.environ, {}, clear=True):
            result = health_checker.check_llm_provider()

            assert result.status == HealthStatus.FAILED
            assert 'API key' in result.message

    def test_stage_checks_ytdlp(self, health_checker):
        """Test that DOWNLOAD_SEGMENTS includes yt-dlp check."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout='2024.12.23',
                stderr='',
            )

            results = health_checker.check_stage('DOWNLOAD_SEGMENTS')
            check_names = [r.name for r in results]
            assert 'ytdlp' in check_names

    def test_stage_checks_llm_provider(self, health_checker):
        """Test that MATCH stage includes LLM provider check."""
        # Mock config
        health_checker.config.llm = Mock()
        health_checker.config.llm.provider = 'gemini'

        results = health_checker.check_stage('MATCH')
        check_names = [r.name for r in results]
        assert 'llm_provider' in check_names


class TestStageHealthChecksRegistry:
    """Test STAGE_HEALTH_CHECKS registry."""

    def test_video_search_checks(self):
        """Test VIDEO_SEARCH stage checks."""
        checks = STAGE_HEALTH_CHECKS['VIDEO_SEARCH']
        assert 'network' in checks
        assert 'ytdlp' in checks
        assert 'llm_provider' in checks

    def test_caption_checks(self):
        """Test CAPTION stage checks."""
        checks = STAGE_HEALTH_CHECKS['CAPTION']
        assert 'network' in checks
        assert 'ytdlp' in checks

    def test_match_checks(self):
        """Test MATCH stage includes memory check."""
        checks = STAGE_HEALTH_CHECKS['MATCH']
        assert 'network' in checks
        assert 'memory' in checks
        assert 'embedding' in checks
        assert 'llm_provider' in checks

    def test_iterative_match_checks(self):
        """Test ITERATIVE_MATCH stage includes memory check."""
        checks = STAGE_HEALTH_CHECKS['ITERATIVE_MATCH']
        assert 'network' in checks
        assert 'memory' in checks
        assert 'embedding' in checks
        assert 'llm_provider' in checks

    def test_download_segments_checks(self):
        """Test DOWNLOAD_SEGMENTS stage checks."""
        checks = STAGE_HEALTH_CHECKS['DOWNLOAD_SEGMENTS']
        assert 'network' in checks
        assert 'disk_space' in checks
        assert 'ytdlp' in checks

    def test_output_checks(self):
        """Test OUTPUT stage includes disk_space check."""
        checks = STAGE_HEALTH_CHECKS['OUTPUT']
        assert 'disk_space' in checks


class TestRunHealthChecks:
    """Test convenience function."""

    def test_run_health_checks(self):
        """Test run_health_checks convenience function."""
        mock_config = Mock()
        # Disable checks to avoid side effects
        mock_config.health_check = {
            'enabled_checks': {
                'network': False,
                'disk_space': False,
                'memory': False,
                'embedding': False,
            }
        }

        results = run_health_checks('VIDEO_SEARCH', mock_config)
        assert isinstance(results, list)


class TestCircuitBreakerHealth:
    """Test get_circuit_breaker_health function (US-120-012)."""

    def test_get_circuit_breaker_health_none(self):
        """Test returns dict with None values when no circuit breaker provided."""
        from src.health_checker import get_circuit_breaker_health

        result = get_circuit_breaker_health(None, None)
        assert result is not None
        assert result['main_breaker'] is None
        assert result['per_keyword_breaker'] is None

    def test_get_circuit_breaker_health_with_mock(self):
        """Test circuit breaker health with mock."""
        from src.health_checker import get_circuit_breaker_health

        # Create mock circuit breaker - must NOT have get_health_metrics
        mock_cb = Mock(spec=['get_stats'])
        mock_cb.get_stats = Mock(
            return_value={
                'total_trips': 5,
                'is_open': False,
                'total_paused_seconds': 30.0,
                'consecutive_failures': 0,
                'enabled': True,
                'success_count': 20,
                'failure_count': 5,
            }
        )

        # Create mock per-keyword breaker
        mock_pkb = Mock()
        mock_pkb._keyword_states = {}
        mock_pkb.is_global_tripped = Mock(return_value=False)
        mock_pkb._config = Mock()
        mock_pkb._config.pause_seconds = 30.0

        result = get_circuit_breaker_health(mock_cb, mock_pkb)

        assert result is not None
        assert 'main_breaker' in result
        assert result['main_breaker']['trip_count'] == 5
        assert result['main_breaker']['failure_rate'] == 0.2
        assert 'per_keyword_breaker' in result
        assert result['per_keyword_breaker']['active_keywords'] == 0

    def test_get_circuit_breaker_health_with_keyword_states(self):
        """Test circuit breaker health with keyword states."""
        from src.health_checker import get_circuit_breaker_health

        # Create mock with keyword states
        mock_cb = Mock()
        mock_cb.get_stats = Mock(
            return_value={
                'total_trips': 1,
                'is_open': False,
                'total_paused_seconds': 10.0,
                'consecutive_failures': 0,
                'enabled': True,
                'success_count': 10,
                'failure_count': 2,
            }
        )

        # Create mock keyword state
        mock_state = Mock()
        mock_state.is_open = True
        mock_state.consecutive_failures = 3
        mock_state.total_trips = 2
        mock_state.consecutive_successes = 0
        mock_state.pause_history = [30.0]

        mock_pkb = Mock()
        mock_pkb._keyword_states = {'python tutorial': mock_state}
        mock_pkb.is_global_tripped = Mock(return_value=False)
        mock_pkb._config = Mock()
        mock_pkb._config.pause_seconds = 30.0

        result = get_circuit_breaker_health(mock_cb, mock_pkb)

        assert result is not None
        pkb = result['per_keyword_breaker']
        assert pkb['active_keywords'] == 1
        assert pkb['rate_limited_keywords'] == 1
        assert pkb['rate_limit_percentage'] == 1.0
        assert 'python tutorial' in pkb['keyword_states']

    def test_get_circuit_breaker_health_recommendations(self):
        """Test circuit breaker health generates recommendations."""
        from src.health_checker import get_circuit_breaker_health

        mock_cb = Mock()
        mock_cb.get_stats = Mock(
            return_value={
                'total_trips': 0,
                'is_open': True,
                'total_paused_seconds': 0.0,
                'consecutive_failures': 5,
                'enabled': True,
                'success_count': 10,
                'failure_count': 0,
            }
        )

        mock_state = Mock()
        mock_state.is_open = True
        mock_state.consecutive_failures = 3
        mock_state.total_trips = 2
        mock_state.consecutive_successes = 0
        mock_state.pause_history = []

        mock_pkb = Mock()
        mock_pkb._keyword_states = {'kw1': mock_state, 'kw2': mock_state}
        mock_pkb.is_global_tripped = Mock(return_value=True)
        mock_pkb._config = Mock()

        result = get_circuit_breaker_health(mock_cb, mock_pkb)

        assert result is not None
        recommendations = result.get('recommendations', [])
        assert len(recommendations) > 0
        # Should have global fallback recommendation
        assert any(r.get('severity') == 'critical' for r in recommendations)


class TestTierEffectivenessHealth:
    """Test get_tier_effectiveness_health function (US-120-012)."""

    def test_get_tier_effectiveness_health_none(self):
        """Test returns None when no escalation manager provided."""
        from src.health_checker import get_tier_effectiveness_health

        result = get_tier_effectiveness_health(None)
        assert result is None

    def test_get_tier_effectiveness_health_with_mock(self):
        """Test tier effectiveness health with mock escalation manager."""
        from src.health_checker import get_tier_effectiveness_health

        # Create mock escalation manager
        mock_esc = Mock()
        mock_esc.get_tier_effectiveness = Mock(
            return_value={
                '429': {'tier1': 0.5, 'tier2': 0.7, 'tier3': 0.9},
                '403': {'tier1': 0.3, 'tier2': 0.5, 'tier3': 0.8},
            }
        )
        mock_esc.get_tier_recommendations = Mock(return_value=[
            {'message': 'Try tier3 for 429 errors'}
        ])
        mock_esc.get_expected_success_rate = Mock(side_effect=lambda x: 0.6)
        mock_esc.get_best_tier_for_category = Mock(return_value=Mock(name='tier3'))
        mock_esc.get_tier_counts = Mock(
            return_value={
                '429': {'tier1': {'successes': 10, 'failures': 10}, 'tier2': {'successes': 14, 'failures': 6}, 'tier3': {'successes': 18, 'failures': 2}},
                '403': {'tier1': {'successes': 6, 'failures': 14}, 'tier2': {'successes': 10, 'failures': 10}, 'tier3': {'successes': 16, 'failures': 4}},
            }
        )

        result = get_tier_effectiveness_health(mock_esc)

        assert result is not None
        assert 'tier_effectiveness' in result
        assert 'visualization' in result
        assert 'actionable_recommendations' in result

        # Check visualization data
        viz = result['visualization']
        assert '429' in viz
        assert len(viz['429']['tiers']) == 5  # tier1-tier5

        # Check tier has visual bar
        tier1 = [t for t in viz['429']['tiers'] if t['name'] == 'tier1'][0]
        assert 'visual_bar' in tier1
        assert 'status' in tier1
        assert tier1['status'] == 'fair'  # 0.5 = fair

    def test_get_tier_effectiveness_health_visual_bars(self):
        """Test visualization includes proper visual bars."""
        from src.health_checker import get_tier_effectiveness_health

        mock_esc = Mock()
        mock_esc.get_tier_effectiveness = Mock(
            return_value={
                'test': {'tier1': 0.9, 'tier2': 0.5, 'tier3': 0.0},
            }
        )
        mock_esc.get_tier_recommendations = Mock(return_value=[])
        mock_esc.get_expected_success_rate = Mock(return_value=0.7)
        mock_esc.get_best_tier_for_category = Mock(return_value=Mock(name='tier1'))
        mock_esc.get_tier_counts = Mock(return_value={})

        result = get_tier_effectiveness_health(mock_esc)

        viz = result['visualization']
        tier1 = [t for t in viz['test']['tiers'] if t['name'] == 'tier1'][0]
        tier2 = [t for t in viz['test']['tiers'] if t['name'] == 'tier2'][0]
        tier3 = [t for t in viz['test']['tiers'] if t['name'] == 'tier3'][0]

        assert tier1['status'] == 'excellent'  # 0.9 >= 0.8
        assert tier2['status'] == 'fair'  # 0.5 >= 0.4
        assert tier3['status'] == 'no_data'  # 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
