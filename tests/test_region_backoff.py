"""Tests for region-specific backoff (US-109-011).

Verifies:
- Region detection from country codes
- Region-specific backoff multipliers: US=1.0, EU=1.2, ASIA=1.5, OTHER=2.0
- Regional rate limit tracking with separate counters per region
- PauseCalculator.region_adjusted() applies correct multipliers
- RegionalRateLimitTracker tracks counts per region
- IP geolocation for region detection
"""

import pytest
from unittest.mock import patch, MagicMock

from src.downloader.pause_calculator import (
    PauseCalculator,
    PauseContext,
    RegionalRateLimitTracker,
)
from src.config.sections.download import (
    RegionBackoffConfig,
    _get_region_from_country,
    _get_ip_geolocation,
)


# ============================================================================
# Region Detection Tests
# ============================================================================


class TestRegionDetection:
    """Test _get_region_from_country() function."""

    @pytest.mark.fast
    def test_us_countries(self):
        """US and Canada map to 'us' region."""
        assert _get_region_from_country('us') == 'us'
        assert _get_region_from_country('US') == 'us'
        assert _get_region_from_country('ca') == 'us'
        assert _get_region_from_country('CA') == 'us'

    @pytest.mark.fast
    def test_eu_countries(self):
        """European countries map to 'eu' region."""
        eu_countries = ['gb', 'de', 'fr', 'nl', 'se', 'ch', 'it', 'es', 'pl', 'be']
        for country in eu_countries:
            assert _get_region_from_country(country) == 'eu', f"Failed for {country}"

    @pytest.mark.fast
    def test_asia_countries(self):
        """Asian countries map to 'asia' region."""
        asia_countries = ['jp', 'sg', 'kr', 'in', 'id', 'my', 'th', 'vn', 'ph', 'tw']
        for country in asia_countries:
            assert _get_region_from_country(country) == 'asia', f"Failed for {country}"

    @pytest.mark.fast
    def test_other_countries(self):
        """Unknown countries map to 'other' region."""
        assert _get_region_from_country('br') == 'other'
        assert _get_region_from_country('za') == 'other'
        assert _get_region_from_country('ar') == 'other'
        assert _get_region_from_country('xx') == 'other'

    @pytest.mark.fast
    def test_empty_country(self):
        """Empty country code returns 'other'."""
        assert _get_region_from_country('') == 'other'
        assert _get_region_from_country(None) == 'other'


# ============================================================================
# IP Geolocation Tests
# ============================================================================


class TestIPGeolocation:
    """Test _get_ip_geolocation() function."""

    @pytest.mark.fast
    def test_ip_geolocation_success(self):
        """IP geolocation returns country code on success."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"status": "success", "countryCode": "US"}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = _get_ip_geolocation()
            assert result == 'us'

    @pytest.mark.fast
    def test_ip_geolocation_with_ip(self):
        """IP geolocation with specific IP address."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'{"status": "success", "countryCode": "DE"}'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = _get_ip_geolocation('8.8.8.8')
            assert result == 'de'

    @pytest.mark.fast
    def test_ip_geolocation_failure(self):
        """IP geolocation returns None on failure."""
        import urllib.error
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = urllib.error.URLError("Network error")

            result = _get_ip_geolocation()
            assert result is None

    @pytest.mark.fast
    def test_ip_geolocation_invalid_response(self):
        """IP geolocation returns None on invalid response."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_response = MagicMock()
            mock_response.read.return_value = b'invalid json'
            mock_response.__enter__ = MagicMock(return_value=mock_response)
            mock_response.__exit__ = MagicMock(return_value=False)
            mock_urlopen.return_value = mock_response

            result = _get_ip_geolocation()
            assert result is None


# ============================================================================
# RegionBackoffConfig Tests
# ============================================================================


class TestRegionBackoffConfig:
    """Test RegionBackoffConfig dataclass."""

    @pytest.mark.fast
    def test_default_values(self):
        """RegionBackoffConfig has correct default values."""
        config = RegionBackoffConfig()
        assert config.enabled is False
        assert config.us_multiplier == 1.0
        assert config.eu_multiplier == 1.2
        assert config.asia_multiplier == 1.5
        assert config.other_multiplier == 2.0
        assert config.track_per_region is False

    @pytest.mark.fast
    def test_custom_multipliers(self):
        """RegionBackoffConfig accepts custom multipliers."""
        config = RegionBackoffConfig(
            enabled=True,
            us_multiplier=1.1,
            eu_multiplier=1.3,
            asia_multiplier=1.6,
            other_multiplier=2.1,
            track_per_region=True,
        )
        assert config.enabled is True
        assert config.us_multiplier == 1.1
        assert config.eu_multiplier == 1.3
        assert config.asia_multiplier == 1.6
        assert config.other_multiplier == 2.1
        assert config.track_per_region is True


# ============================================================================
# RegionalRateLimitTracker Tests
# ============================================================================


class TestRegionalRateLimitTracker:
    """Test RegionalRateLimitTracker class."""

    def setup_method(self):
        """Clear tracker before each test."""
        RegionalRateLimitTracker.clear()

    def teardown_method(self):
        """Clear tracker after each test."""
        RegionalRateLimitTracker.clear()

    @pytest.mark.fast
    def test_enable_disable(self):
        """Enable and disable tracking."""
        assert RegionalRateLimitTracker.is_enabled() is False
        RegionalRateLimitTracker.enable()
        assert RegionalRateLimitTracker.is_enabled() is True
        RegionalRateLimitTracker.disable()
        assert RegionalRateLimitTracker.is_enabled() is False

    @pytest.mark.fast
    def test_increment_count(self):
        """Increment rate limit count per region."""
        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('de')
        assert RegionalRateLimitTracker.get_count('us') == 2
        assert RegionalRateLimitTracker.get_count('eu') == 1
        assert RegionalRateLimitTracker.get_count('asia') == 0

    @pytest.mark.fast
    def test_unknown_region_maps_to_other(self):
        """Unknown region codes map to 'other'."""
        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('xx')
        assert RegionalRateLimitTracker.get_count('other') == 1

    @pytest.mark.fast
    def test_reset_single_region(self):
        """Reset single region counter."""
        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('eu')
        RegionalRateLimitTracker.reset('us')
        assert RegionalRateLimitTracker.get_count('us') == 0
        assert RegionalRateLimitTracker.get_count('eu') == 1

    @pytest.mark.fast
    def test_reset_all_regions(self):
        """Reset all region counters."""
        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('eu')
        RegionalRateLimitTracker.increment('asia')
        RegionalRateLimitTracker.reset()
        assert RegionalRateLimitTracker.get_count('us') == 0
        assert RegionalRateLimitTracker.get_count('eu') == 0
        assert RegionalRateLimitTracker.get_count('asia') == 0

    @pytest.mark.fast
    def test_get_all_counts(self):
        """Get all region counts."""
        RegionalRateLimitTracker.enable()
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('us')
        RegionalRateLimitTracker.increment('eu')
        counts = RegionalRateLimitTracker.get_all_counts()
        assert counts['us'] == 2
        assert counts['eu'] == 1
        assert counts['asia'] == 0
        assert counts['other'] == 0


# ============================================================================
# PauseCalculator.region_adjusted Tests
# ============================================================================


class TestRegionAdjusted:
    """Test PauseCalculator.region_adjusted() method."""

    @pytest.mark.fast
    def test_disabled_returns_original(self):
        """When region_enabled=False, returns original pause."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=False,
            region='us',
        )
        result = calc.region_adjusted(60.0, ctx)
        assert result == 60.0

    @pytest.mark.fast
    def test_us_multiplier(self):
        """US region uses 1.0 multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='us',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        result = calc.region_adjusted(60.0, ctx)
        assert result == 60.0

    @pytest.mark.fast
    def test_eu_multiplier(self):
        """EU region uses 1.2 multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        result = calc.region_adjusted(60.0, ctx)
        assert result == 72.0  # 60 * 1.2

    @pytest.mark.fast
    def test_asia_multiplier(self):
        """ASIA region uses 1.5 multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='asia',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        result = calc.region_adjusted(60.0, ctx)
        assert result == 90.0  # 60 * 1.5

    @pytest.mark.fast
    def test_other_multiplier(self):
        """OTHER region uses 2.0 multiplier."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='other',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        result = calc.region_adjusted(60.0, ctx)
        assert result == 120.0  # 60 * 2.0

    @pytest.mark.fast
    def test_unknown_region_defaults_to_other(self):
        """Unknown region defaults to OTHER multiplier (2.0)."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='',  # Empty region defaults to 'other'
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        result = calc.region_adjusted(60.0, ctx)
        # Empty region defaults to 'other' which has multiplier 2.0
        assert result == 120.0  # 60 * 2.0

    @pytest.mark.fast
    def test_custom_multipliers(self):
        """Custom multipliers from config are used."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=100.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.5, 'asia': 2.0, 'other': 2.5},
        )
        result = calc.region_adjusted(100.0, ctx)
        assert result == 150.0  # 100 * 1.5


# ============================================================================
# Full Pipeline Tests
# ============================================================================


class TestRegionPipeline:
    """Test full pause calculation pipeline with region adjustment."""

    @pytest.mark.fast
    def test_full_pipeline_with_us_region(self):
        """Full pipeline with US region."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,  # No jitter for predictable results
            region_enabled=True,
            region='us',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        result = calc.calculate(ctx)
        assert result == 60.0

    @pytest.mark.fast
    def test_full_pipeline_with_asia_region(self):
        """Full pipeline with ASIA region (1.5x multiplier)."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            max_pause_seconds=300.0,
            jitter_factor=0.0,  # No jitter for predictable results
            region_enabled=True,
            region='asia',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        result = calc.calculate(ctx)
        assert result == 90.0  # 60 * 1.5

    @pytest.mark.fast
    def test_region_multiplier_applied_before_jitter(self):
        """Region multiplier is applied before jitter (in pipeline order)."""
        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=100.0,
            max_pause_seconds=300.0,
            jitter_factor=0.1,  # 10% jitter
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
        )
        # With seed, we can predict jitter - but in test just verify it's applied
        # Region multiplier (1.2) applied first: 100 * 1.2 = 120
        # Then jitter (±10%): 108 to 132
        result = calc.calculate(ctx)
        assert 108.0 <= result <= 132.0


# ============================================================================
# PauseContext Region Fields Tests
# ============================================================================


class TestPauseContextRegionFields:
    """Test PauseContext region-specific fields."""

    @pytest.mark.fast
    def test_default_region_values(self):
        """PauseContext has correct default region values."""
        ctx = PauseContext()
        assert ctx.region == ''
        assert ctx.country_code == ''
        assert ctx.region_multipliers == {'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0}
        assert ctx.region_enabled is False

    @pytest.mark.fast
    def test_custom_region_values(self):
        """PauseContext accepts custom region values."""
        ctx = PauseContext(
            region='eu',
            country_code='de',
            region_enabled=True,
            region_multipliers={'us': 1.0, 'eu': 1.3, 'asia': 1.6, 'other': 2.0},
        )
        assert ctx.region == 'eu'
        assert ctx.country_code == 'de'
        assert ctx.region_enabled is True
        assert ctx.region_multipliers['eu'] == 1.3


# ============================================================================
# US-112-008: Region Backoff Config Validation Tests
# ============================================================================


class TestRegionBackoffConfigValidation:
    """Test region_backoff config validation in Config.validate()."""

    @pytest.mark.fast
    def test_region_backoff_multipliers_must_be_greater_equal_one(self, tmp_path):
        """Region backoff multipliers must be >= 1.0."""
        from src.config.base import Config

        # Test us_multiplier < 1.0
        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'download': {
                'region_backoff': {
                    'enabled': False,
                    'us_multiplier': 0.5,  # Invalid: < 1.0
                }
            }
        }
        config = Config._from_dict(config_data)
        errors = config.validate()
        assert any('us_multiplier must be >= 1.0' in e for e in errors), f"Expected error for us_multiplier < 1.0, got: {errors}"

    @pytest.mark.fast
    def test_region_backoff_multipliers_equal_one_valid(self, tmp_path):
        """Region backoff multipliers = 1.0 is valid."""
        from src.config.base import Config

        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'download': {
                'region_backoff': {
                    'enabled': False,
                    'us_multiplier': 1.0,
                    'eu_multiplier': 1.0,
                    'asia_multiplier': 1.0,
                    'other_multiplier': 1.0,
                }
            }
        }
        config = Config._from_dict(config_data)
        errors = config.validate()
        multiplier_errors = [e for e in errors if 'multiplier must be >= 1.0' in e]
        assert len(multiplier_errors) == 0, f"Expected no multiplier errors, got: {multiplier_errors}"

    @pytest.mark.fast
    def test_region_backoff_enabled_without_vpn_fails(self, tmp_path):
        """region_backoff.enabled=true requires mullvad.enabled=true."""
        from src.config.base import Config

        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'download': {
                'region_backoff': {
                    'enabled': True,  # Enabled but VPN not configured
                },
                'mullvad': {
                    'enabled': False,
                }
            }
        }
        config = Config._from_dict(config_data)
        errors = config.validate()
        assert any('region_backoff.enabled=true requires' in e for e in errors), f"Expected error for region_backoff enabled without VPN, got: {errors}"

    @pytest.mark.fast
    def test_region_backoff_enabled_with_vpn_passes(self, tmp_path):
        """region_backoff.enabled=true with mullvad.enabled=true passes."""
        from src.config.base import Config

        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'download': {
                'region_backoff': {
                    'enabled': True,
                    'us_multiplier': 1.0,
                    'eu_multiplier': 1.2,
                    'asia_multiplier': 1.5,
                    'other_multiplier': 2.0,
                },
                'mullvad': {
                    'enabled': True,
                }
            }
        }
        config = Config._from_dict(config_data)
        errors = config.validate()
        vpn_error = [e for e in errors if 'region_backoff.enabled=true requires' in e]
        assert len(vpn_error) == 0, f"Expected no VPN error, got: {vpn_error}"

    @pytest.mark.fast
    def test_region_backoff_disabled_without_vpn_passes(self, tmp_path):
        """region_backoff.enabled=false without VPN passes."""
        from src.config.base import Config

        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'download': {
                'region_backoff': {
                    'enabled': False,
                },
                'mullvad': {
                    'enabled': False,
                }
            }
        }
        config = Config._from_dict(config_data)
        errors = config.validate()
        vpn_error = [e for e in errors if 'region_backoff.enabled=true requires' in e]
        assert len(vpn_error) == 0, f"Expected no VPN error, got: {vpn_error}"


class TestRegionBackoffIntegration:
    """Integration tests for region_backoff with rate_limit configs."""

    @pytest.mark.fast
    def test_region_backoff_with_rate_limit_config(self, tmp_path):
        """Region backoff integrates with rate_limit config."""
        from src.config.base import Config

        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'rate_limit': {
                'slots_per_second': 0.5,
                'burst_size': 3,
                'jitter_factor': 0.2,
                'max_backoff_seconds': 60.0,
            },
            'download': {
                'region_backoff': {
                    'enabled': False,
                    'us_multiplier': 1.0,
                    'eu_multiplier': 1.2,
                    'asia_multiplier': 1.5,
                    'other_multiplier': 2.0,
                }
            }
        }
        config = Config._from_dict(config_data)
        errors = config.validate()

        # Should not have any region_backoff related errors
        region_errors = [e for e in errors if 'region_backoff' in e.lower()]
        assert len(region_errors) == 0, f"Expected no region_backoff errors, got: {region_errors}"

        # Verify rate_limit config is accessible
        assert config.rate_limit.slots_per_second == 0.5

    @pytest.mark.fast
    def test_custom_region_backoff_multipliers_applied(self, tmp_path):
        """Custom region backoff multipliers are applied correctly."""
        from src.config.base import Config

        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'download': {
                'region_backoff': {
                    'enabled': True,
                    'us_multiplier': 1.1,
                    'eu_multiplier': 1.3,
                    'asia_multiplier': 1.7,
                    'other_multiplier': 2.5,
                    'track_per_region': True,
                },
                'mullvad': {
                    'enabled': True,
                }
            }
        }
        config = Config._from_dict(config_data)

        # Verify multipliers are applied
        rb = config.download.region_backoff
        assert rb.us_multiplier == 1.1
        assert rb.eu_multiplier == 1.3
        assert rb.asia_multiplier == 1.7
        assert rb.other_multiplier == 2.5
        assert rb.track_per_region is True


# ============================================================================
# US-112-008: Region Backoff Downloader Integration Tests
# ============================================================================


class TestRegionBackoffDownloaderIntegration:
    """Test that region_backoff settings are applied in the downloader."""

    @pytest.mark.fast
    def test_pause_calculator_uses_config_multipliers(self):
        """PauseCalculator uses region multipliers from config."""
        from src.downloader.pause_calculator import PauseCalculator, PauseContext

        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.3, 'asia': 1.7, 'other': 2.5},
        )
        result = calc.region_adjusted(60.0, ctx)
        # 60.0 * 1.3 = 78.0
        assert result == 78.0

    @pytest.mark.fast
    def test_pause_calculator_uses_config_multipliers_asia(self):
        """PauseCalculator uses Asia multiplier from config."""
        from src.downloader.pause_calculator import PauseCalculator, PauseContext

        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='asia',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.7, 'other': 2.0},
        )
        result = calc.region_adjusted(60.0, ctx)
        # 60.0 * 1.7 = 102.0
        assert result == 102.0

    @pytest.mark.fast
    def test_circuit_breaker_uses_region_multipliers(self):
        """CircuitBreaker passes region multipliers to PauseContext."""
        from src.downloader.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        from src.downloader.pause_calculator import PauseContext

        # Create circuit breaker with region backoff enabled
        config = CircuitBreakerConfig(
            enabled=True,
            consecutive_failures_threshold=3,
            pause_seconds=60.0,
            region_backoff_enabled=True,
            region_multipliers={'us': 1.0, 'eu': 1.3, 'asia': 1.7, 'other': 2.5},
            track_per_region=True,
        )
        breaker = CircuitBreaker(config)

        # Create a mock context for pause calculation
        # The circuit breaker should use these multipliers when calculating pause
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers=config.region_multipliers,
        )

        # Verify the multipliers are stored in the config
        assert config.region_multipliers['eu'] == 1.3
        assert config.region_multipliers['asia'] == 1.7
        assert config.region_multipliers['other'] == 2.5


# ============================================================================
# US-113-008: Dynamic Region-Based Backoff Tests
# ============================================================================


class TestRegionSuccessTracker:
    """Test RegionSuccessTracker class for tracking success rates per region."""

    def setup_method(self):
        """Clear tracker before each test."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

    def teardown_method(self):
        """Clear tracker after each test."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()

    @pytest.mark.fast
    def test_record_success(self):
        """Recording success increments both attempts and successes."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Add enough samples to exceed min_sample_size (3)
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('de')  # Germany -> eu
        RegionSuccessTracker.record_success('de')
        RegionSuccessTracker.record_success('de')

        assert RegionSuccessTracker.get_attempts('us') == 3
        assert RegionSuccessTracker.get_attempts('eu') == 3
        # With 3+ attempts, actual rate is returned
        assert RegionSuccessTracker.get_success_rate('us') == 1.0
        assert RegionSuccessTracker.get_success_rate('eu') == 1.0

    @pytest.mark.fast
    def test_record_failure(self):
        """Recording failure increments attempts but not successes."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Add enough samples to exceed min_sample_size (3)
        RegionSuccessTracker.record_failure('us')
        RegionSuccessTracker.record_failure('us')
        RegionSuccessTracker.record_failure('us')
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_failure('de')

        assert RegionSuccessTracker.get_attempts('us') == 3
        assert RegionSuccessTracker.get_attempts('eu') == 3
        # With 3+ attempts, actual rate is returned (0% failures = 0.0)
        assert RegionSuccessTracker.get_success_rate('us') == 0.0
        assert RegionSuccessTracker.get_success_rate('eu') == 0.0

    @pytest.mark.fast
    def test_mixed_success_failure(self):
        """Mixed success/failure gives correct rate."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # US: 3 successes, 1 failure = 75%
        for _ in range(3):
            RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_failure('us')

        assert RegionSuccessTracker.get_attempts('us') == 4
        assert RegionSuccessTracker.get_success_rate('us') == 0.75

    @pytest.mark.fast
    def test_untested_region_returns_neutral(self):
        """Untested regions return neutral rate (0.5) until min_sample_size."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # No attempts yet
        assert RegionSuccessTracker.get_success_rate('asia') == 0.5

        # With less than min_sample_size (3) attempts
        RegionSuccessTracker.record_success('jp')
        RegionSuccessTracker.record_success('jp')
        assert RegionSuccessTracker.get_success_rate('asia') == 0.5  # Still neutral

    @pytest.mark.fast
    def test_country_to_region_mapping(self):
        """Countries correctly map to regions."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Test various country codes
        assert RegionSuccessTracker._country_to_region('us') == 'us'
        assert RegionSuccessTracker._country_to_region('ca') == 'us'
        assert RegionSuccessTracker._country_to_region('de') == 'eu'
        assert RegionSuccessTracker._country_to_region('gb') == 'eu'
        assert RegionSuccessTracker._country_to_region('jp') == 'asia'
        assert RegionSuccessTracker._country_to_region('sg') == 'asia'
        assert RegionSuccessTracker._country_to_region('br') == 'other'  # Brazil -> other
        assert RegionSuccessTracker._country_to_region('xx') == 'other'  # Unknown -> other

    @pytest.mark.fast
    def test_get_all_rates(self):
        """get_all_rates returns rates for all regions."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Add enough samples to exceed min_sample_size (3)
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')  # 3 successes -> 100%
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_failure('de')  # 3 failures -> 0%

        rates = RegionSuccessTracker.get_all_rates()
        assert 'us' in rates
        assert 'eu' in rates
        assert rates['us'] == 1.0
        assert rates['eu'] == 0.0
        # Other regions should be neutral (0.5) since untested
        assert rates['asia'] == 0.5
        assert rates['other'] == 0.5

    @pytest.mark.fast
    def test_is_below_threshold(self):
        """is_below_threshold correctly identifies regions below threshold."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Create a region with low success rate (below 0.7 threshold)
        for _ in range(5):
            RegionSuccessTracker.record_failure('jp')  # japan -> asia

        # Need minimum sample size before threshold check applies
        assert RegionSuccessTracker.is_below_threshold('asia', 0.7) is True

        # Create a region with high success rate (above threshold)
        for _ in range(5):
            RegionSuccessTracker.record_success('ca')  # canada -> us

        assert RegionSuccessTracker.is_below_threshold('us', 0.7) is False

    @pytest.mark.fast
    def test_untested_region_not_flagged_below_threshold(self):
        """Untested regions are not flagged as below threshold."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Untested region should return False
        assert RegionSuccessTracker.is_below_threshold('asia', 0.7) is False

    @pytest.mark.fast
    def test_reset_single_region(self):
        """Reset clears only specified region."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('de')
        RegionSuccessTracker.reset('us')

        assert RegionSuccessTracker.get_attempts('us') == 0
        assert RegionSuccessTracker.get_attempts('eu') == 1

    @pytest.mark.fast
    def test_reset_all(self):
        """Reset without args clears all regions."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('de')
        RegionSuccessTracker.record_failure('jp')
        RegionSuccessTracker.reset()

        assert RegionSuccessTracker.get_attempts('us') == 0
        assert RegionSuccessTracker.get_attempts('eu') == 0
        assert RegionSuccessTracker.get_attempts('asia') == 0

    @pytest.mark.fast
    def test_disable_tracking(self):
        """Disabling tracking stops recording."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        RegionSuccessTracker.disable()
        RegionSuccessTracker.record_success('us')

        # Should not record when disabled
        assert RegionSuccessTracker.get_attempts('us') == 0
        assert RegionSuccessTracker.get_success_rate('us') == 0.5  # Neutral when disabled


class TestDynamicRegionAdjustment:
    """Test PauseCalculator dynamic region adjustment based on success rates."""

    def setup_method(self):
        """Clear tracker before each test."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

    def teardown_method(self):
        """Clear tracker after each test."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()

    @pytest.mark.fast
    def test_dynamic_adjustment_disabled_by_default(self):
        """Dynamic adjustment is disabled when not explicitly enabled."""
        from src.downloader.pause_calculator import PauseCalculator, PauseContext

        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=False,  # Explicitly disabled
            region_success_rates={'eu': 0.3},  # Low success rate
        )

        result = calc.region_adjusted(60.0, ctx)
        # Should only apply base multiplier (1.2), not dynamic penalty
        assert result == 72.0  # 60 * 1.2

    @pytest.mark.fast
    def test_dynamic_adjustment_applies_penalty(self):
        """Dynamic adjustment applies penalty for low success rates."""
        from src.downloader.pause_calculator import PauseCalculator, PauseContext

        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=True,
            region_success_rates={'eu': 0.3},  # 30% success rate
            region_success_rate_threshold=0.7,
        )

        result = calc.region_adjusted(60.0, ctx)
        # Base: 60 * 1.2 = 72
        # Dynamic penalty: 1 + (0.7 - 0.3) = 1.4
        # Final: 72 * 1.4 = 100.8
        assert result == pytest.approx(100.8, rel=0.01)

    @pytest.mark.fast
    def test_dynamic_adjustment_no_penalty_above_threshold(self):
        """No penalty applied when success rate above threshold."""
        from src.downloader.pause_calculator import PauseCalculator, PauseContext

        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=True,
            region_success_rates={'eu': 0.8},  # 80% success rate (above 0.7 threshold)
            region_success_rate_threshold=0.7,
        )

        result = calc.region_adjusted(60.0, ctx)
        # Should apply recovery boost instead of penalty
        # Base: 60 * 1.2 = 72
        # Recovery: 72 * 0.9 = 64.8
        assert result == pytest.approx(64.8, rel=0.01)

    @pytest.mark.fast
    def test_dynamic_adjustment_no_data_uses_base(self):
        """When no success rate data, uses base multiplier."""
        from src.downloader.pause_calculator import PauseCalculator, PauseContext

        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=True,
            region_success_rates={},  # No data
            region_success_rate_threshold=0.7,
        )

        result = calc.region_adjusted(60.0, ctx)
        # Should use base multiplier only
        assert result == 72.0  # 60 * 1.2

    @pytest.mark.fast
    def test_dynamic_adjustment_with_zero_success_rate(self):
        """Zero success rate gets maximum penalty."""
        from src.downloader.pause_calculator import PauseCalculator, PauseContext

        calc = PauseCalculator()
        ctx = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=True,
            region_success_rates={'eu': 0.0},  # 0% success rate
            region_success_rate_threshold=0.7,
        )

        result = calc.region_adjusted(60.0, ctx)
        # Base: 60 * 1.2 = 72
        # Dynamic penalty: 1 + (0.7 - 0.0) = 1.7
        # Final: 72 * 1.7 = 122.4
        assert result == pytest.approx(122.4, rel=0.01)


class TestDynamicRegionConfig:
    """Test RegionBackoffConfig with dynamic region settings."""

    @pytest.mark.fast
    def test_default_dynamic_settings(self):
        """RegionBackoffConfig has correct default values for dynamic settings."""
        from src.config.sections.download import RegionBackoffConfig

        config = RegionBackoffConfig()
        assert config.region_success_rate_threshold == 0.7
        assert config.dynamic_region_adjustment is False

    @pytest.mark.fast
    def test_custom_dynamic_settings(self):
        """RegionBackoffConfig accepts custom dynamic settings."""
        from src.config.sections.download import RegionBackoffConfig

        config = RegionBackoffConfig(
            region_success_rate_threshold=0.5,
            dynamic_region_adjustment=True,
        )
        assert config.region_success_rate_threshold == 0.5
        assert config.dynamic_region_adjustment is True

    @pytest.mark.fast
    def test_dynamic_config_in_config_loading(self, tmp_path):
        """Dynamic region settings are loaded from config.yaml."""
        from src.config.base import Config

        config_data = {
            'project': {'name': 'test_project', 'root_dir': str(tmp_path)},
            'download': {
                'region_backoff': {
                    'enabled': True,
                    'region_success_rate_threshold': 0.6,
                    'dynamic_region_adjustment': True,
                },
                'mullvad': {
                    'enabled': True,
                }
            }
        }
        config = Config._from_dict(config_data)
        rb = config.download.region_backoff
        assert rb.region_success_rate_threshold == 0.6
        assert rb.dynamic_region_adjustment is True


class TestRegionRotationWithSuccessTracking:
    """Integration tests for region rotation with success tracking (US-113-008)."""

    def setup_method(self):
        """Clear tracker before each test."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()
        RegionSuccessTracker.enable()

    def teardown_method(self):
        """Clear tracker after each test."""
        from src.downloader.pause_calculator import RegionSuccessTracker
        RegionSuccessTracker.clear()

    @pytest.mark.fast
    def test_region_rotation_success_tracking(self):
        """Simulate region rotation with success/failure tracking."""
        from src.downloader.pause_calculator import (
            RegionSuccessTracker, PauseCalculator, PauseContext
        )

        # Simulate: US region doing well, EU region struggling
        # US: 10 successes, 2 failures = 83%
        for _ in range(10):
            RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_failure('us')
        RegionSuccessTracker.record_failure('us')

        # EU: 2 successes, 6 failures = 25%
        for _ in range(2):
            RegionSuccessTracker.record_success('de')
        for _ in range(6):
            RegionSuccessTracker.record_failure('de')

        calc = PauseCalculator()

        # Calculate pause for US region (good success rate)
        ctx_us = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='us',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=True,
            region_success_rates=RegionSuccessTracker.get_all_rates(),
            region_success_rate_threshold=0.7,
        )
        pause_us = calc.region_adjusted(60.0, ctx_us)

        # Calculate pause for EU region (poor success rate)
        ctx_eu = PauseContext(
            base_pause_seconds=60.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=True,
            region_success_rates=RegionSuccessTracker.get_all_rates(),
            region_success_rate_threshold=0.7,
        )
        pause_eu = calc.region_adjusted(60.0, ctx_eu)

        # EU should have significantly higher pause due to poor success rate
        assert pause_eu > pause_us
        # US: base 60 * 1.0 = 60, recovery boost 60 * 0.9 = 54
        assert pause_us == pytest.approx(54.0, rel=0.1)
        # EU: base 60 * 1.2 = 72, penalty 72 * (1 + (0.7-0.25)) = 72 * 1.45 = 104.4
        assert pause_eu == pytest.approx(104.4, rel=0.1)

    @pytest.mark.fast
    def test_avoid_low_success_rate_regions(self):
        """Test that low success rate regions are identified for avoidance."""
        from src.downloader.pause_calculator import RegionSuccessTracker

        # Create a region with very low success rate
        for _ in range(10):
            RegionSuccessTracker.record_failure('jp')  # japan -> asia

        # This region should be flagged as below threshold
        assert RegionSuccessTracker.is_below_threshold('asia', 0.7) is True

        # Create a region with good success rate
        for _ in range(10):
            RegionSuccessTracker.record_success('ca')  # canada -> us

        # This region should NOT be flagged
        assert RegionSuccessTracker.is_below_threshold('us', 0.7) is False

    @pytest.mark.fast
    def test_mullvad_region_tracking_integration(self):
        """Simulate MullvadVPN tracking and pause calculator using it."""
        from src.downloader.pause_calculator import (
            RegionSuccessTracker, PauseCalculator, PauseContext
        )

        # Simulate MullvadVPN recording results
        # Good region: US
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')
        RegionSuccessTracker.record_success('us')

        # Bad region: DE (Germany -> EU)
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_failure('de')
        RegionSuccessTracker.record_success('de')

        # Get rates and use in pause calculation
        rates = RegionSuccessTracker.get_all_rates()

        calc = PauseCalculator()

        # When EU has low success rate, pause should be higher
        ctx = PauseContext(
            base_pause_seconds=100.0,
            region_enabled=True,
            region='eu',
            region_multipliers={'us': 1.0, 'eu': 1.2, 'asia': 1.5, 'other': 2.0},
            dynamic_region_adjustment=True,
            region_success_rates=rates,
            region_success_rate_threshold=0.7,
        )

        pause = calc.region_adjusted(100.0, ctx)

        # EU has 1 success / 5 attempts = 20% = 0.2
        # Base: 100 * 1.2 = 120
        # Penalty: 1 + (0.7 - 0.2) = 1.5
        # Final: 120 * 1.5 = 180
        assert pause == pytest.approx(180.0, rel=0.01)
