"""Tests for LLM cache quality tier tracking (US-009).

Tests that the LLM cache tracks quality_tier based on confidence scores:
- quality_tier field in cache entries
- compute_quality_tier() function
- extract_confidence_from_response() function
- skip_low_quality behavior in LLMCache
- logging of cache quality tier on hit
- expire_low_quality config option
"""

import json
import sys
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.unit


# =============================================================================
# Test Quality Tier Computation
# =============================================================================

class TestComputeQualityTier:
    """Tests for compute_quality_tier() function."""

    @pytest.mark.fast
    def test_high_confidence_returns_high_tier(self):
        """Confidence >= 0.8 should return 'high' tier."""
        from src.llm_client.cache import compute_quality_tier
        assert compute_quality_tier(0.8) == 'high'
        assert compute_quality_tier(0.9) == 'high'
        assert compute_quality_tier(1.0) == 'high'
        assert compute_quality_tier(0.85) == 'high'

    @pytest.mark.fast
    def test_medium_confidence_returns_medium_tier(self):
        """Confidence 0.5 <= x < 0.8 should return 'medium' tier."""
        from src.llm_client.cache import compute_quality_tier
        assert compute_quality_tier(0.5) == 'medium'
        assert compute_quality_tier(0.6) == 'medium'
        assert compute_quality_tier(0.7) == 'medium'
        assert compute_quality_tier(0.79) == 'medium'

    @pytest.mark.fast
    def test_low_confidence_returns_low_tier(self):
        """Confidence < 0.5 should return 'low' tier."""
        from src.llm_client.cache import compute_quality_tier
        assert compute_quality_tier(0.0) == 'low'
        assert compute_quality_tier(0.1) == 'low'
        assert compute_quality_tier(0.3) == 'low'
        assert compute_quality_tier(0.49) == 'low'

    @pytest.mark.fast
    def test_boundary_values(self):
        """Test exact boundary values."""
        from src.llm_client.cache import compute_quality_tier
        # 0.5 is medium (>= MEDIUM threshold)
        assert compute_quality_tier(0.5) == 'medium'
        # 0.8 is high (>= HIGH threshold)
        assert compute_quality_tier(0.8) == 'high'


class TestExtractConfidenceFromResponse:
    """Tests for extract_confidence_from_response() function."""

    @pytest.mark.fast
    def test_extract_from_dict_parsed_data(self):
        """Extract confidence from dict parsed_data."""
        from src.llm_client.cache import extract_confidence_from_response
        from src.llm_client.base import LLMResponse

        response = LLMResponse(
            text='{"confidence": 0.85}',
            parsed_data={'confidence': 0.85, 'selected': 1}
        )
        assert extract_confidence_from_response(response) == 0.85

    @pytest.mark.fast
    def test_extract_from_list_parsed_data(self):
        """Extract confidence from list parsed_data (batch responses)."""
        from src.llm_client.cache import extract_confidence_from_response
        from src.llm_client.base import LLMResponse

        response = LLMResponse(
            text='[{"confidence": 0.75}]',
            parsed_data=[{'confidence': 0.75, 'selected': 1}]
        )
        assert extract_confidence_from_response(response) == 0.75

    @pytest.mark.fast
    def test_returns_none_when_no_parsed_data(self):
        """Return None when parsed_data is None."""
        from src.llm_client.cache import extract_confidence_from_response
        from src.llm_client.base import LLMResponse

        response = LLMResponse(text='some text')
        assert extract_confidence_from_response(response) is None

    @pytest.mark.fast
    def test_returns_none_when_no_confidence_in_dict(self):
        """Return None when confidence not in dict."""
        from src.llm_client.cache import extract_confidence_from_response
        from src.llm_client.base import LLMResponse

        response = LLMResponse(
            text='{"selected": 1}',
            parsed_data={'selected': 1}
        )
        assert extract_confidence_from_response(response) is None

    @pytest.mark.fast
    def test_returns_none_when_empty_list(self):
        """Return None when parsed_data is empty list."""
        from src.llm_client.cache import extract_confidence_from_response
        from src.llm_client.base import LLMResponse

        response = LLMResponse(text='[]', parsed_data=[])
        assert extract_confidence_from_response(response) is None


# =============================================================================
# Test Quality Tier Constants
# =============================================================================

class TestQualityTierConstants:
    """Tests for quality tier threshold constants."""

    @pytest.mark.fast
    def test_high_threshold_is_0_8(self):
        """High tier threshold should be 0.8."""
        from src.llm_client.cache import QUALITY_TIER_HIGH_THRESHOLD
        assert QUALITY_TIER_HIGH_THRESHOLD == 0.8

    @pytest.mark.fast
    def test_medium_threshold_is_0_5(self):
        """Medium tier threshold should be 0.5."""
        from src.llm_client.cache import QUALITY_TIER_MEDIUM_THRESHOLD
        assert QUALITY_TIER_MEDIUM_THRESHOLD == 0.5


# =============================================================================
# Test Cache Entry Quality Tier Storage
# =============================================================================

class TestCacheEntryQualityTier:
    """Tests for quality_tier field in cache entries."""

    def setup_method(self):
        """Create a temporary directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.fast
    def test_cache_entry_has_quality_tier_field(self):
        """Cache entry should have quality_tier field after set()."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider')
        request = LLMRequest(prompt="test prompt")
        response = LLMResponse(
            text='{"confidence": 0.85}',
            parsed_data={'confidence': 0.85},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        # Read the cache file directly
        cache_files = list(Path(self.temp_dir).rglob('*.json'))
        assert len(cache_files) == 1

        with open(cache_files[0], 'r') as f:
            data = json.load(f)

        assert 'quality_tier' in data
        assert data['quality_tier'] == 'high'

    @pytest.mark.fast
    def test_cache_entry_stores_high_tier_for_high_confidence(self):
        """Cache entry should store 'high' tier for confidence >= 0.8."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider')
        request = LLMRequest(prompt="high confidence test")
        response = LLMResponse(
            text='{"confidence": 0.92}',
            parsed_data={'confidence': 0.92},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        cached = cache.get(request)
        assert cached['quality_tier'] == 'high'

    @pytest.mark.fast
    def test_cache_entry_stores_medium_tier_for_medium_confidence(self):
        """Cache entry should store 'medium' tier for 0.5 <= confidence < 0.8."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider')
        request = LLMRequest(prompt="medium confidence test")
        response = LLMResponse(
            text='{"confidence": 0.65}',
            parsed_data={'confidence': 0.65},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        cached = cache.get(request)
        assert cached['quality_tier'] == 'medium'

    @pytest.mark.fast
    def test_cache_entry_stores_low_tier_for_low_confidence(self):
        """Cache entry should store 'low' tier for confidence < 0.5."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider')
        request = LLMRequest(prompt="low confidence test")
        response = LLMResponse(
            text='{"confidence": 0.35}',
            parsed_data={'confidence': 0.35},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        cached = cache.get(request)
        assert cached['quality_tier'] == 'low'

    @pytest.mark.fast
    def test_cache_entry_stores_unknown_tier_when_no_confidence(self):
        """Cache entry should store 'unknown' tier when confidence not in response."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider')
        request = LLMRequest(prompt="no confidence test")
        response = LLMResponse(
            text='just text response',
            parsed_data=None,
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        cached = cache.get(request)
        assert cached['quality_tier'] == 'unknown'


# =============================================================================
# Test Skip Low Quality Cache Entries
# =============================================================================

class TestSkipLowQualityCacheEntries:
    """Tests for skipping low-quality cache entries."""

    def setup_method(self):
        """Create a temporary directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.fast
    def test_skip_low_quality_returns_none_for_low_tier(self):
        """When skip_low_quality=True, should return None for low-quality entries."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider', skip_low_quality=True)
        request = LLMRequest(prompt="low quality test")
        response = LLMResponse(
            text='{"confidence": 0.35}',
            parsed_data={'confidence': 0.35},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        # Should return None because skip_low_quality is True
        cached = cache.get(request)
        assert cached is None

    @pytest.mark.fast
    def test_skip_low_quality_returns_data_for_high_tier(self):
        """When skip_low_quality=True, should return data for high-quality entries."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider', skip_low_quality=True)
        request = LLMRequest(prompt="high quality test")
        response = LLMResponse(
            text='{"confidence": 0.9}',
            parsed_data={'confidence': 0.9},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        cached = cache.get(request)
        assert cached is not None
        assert cached['quality_tier'] == 'high'

    @pytest.mark.fast
    def test_skip_low_quality_returns_data_for_medium_tier(self):
        """When skip_low_quality=True, should return data for medium-quality entries."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider', skip_low_quality=True)
        request = LLMRequest(prompt="medium quality test")
        response = LLMResponse(
            text='{"confidence": 0.65}',
            parsed_data={'confidence': 0.65},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        cached = cache.get(request)
        assert cached is not None
        assert cached['quality_tier'] == 'medium'

    @pytest.mark.fast
    def test_skip_low_quality_false_returns_low_tier_data(self):
        """When skip_low_quality=False (default), should return low-quality entries."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider', skip_low_quality=False)
        request = LLMRequest(prompt="low quality test default")
        response = LLMResponse(
            text='{"confidence": 0.35}',
            parsed_data={'confidence': 0.35},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        cached = cache.get(request)
        assert cached is not None
        assert cached['quality_tier'] == 'low'

    @pytest.mark.fast
    def test_skip_low_quality_logs_on_skip(self):
        """When skipping low-quality entry, should log info message."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider', skip_low_quality=True)
        request = LLMRequest(prompt="log test")
        response = LLMResponse(
            text='{"confidence": 0.35}',
            parsed_data={'confidence': 0.35},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        with patch('src.llm_client.cache.logger') as mock_logger:
            cached = cache.get(request)
            assert cached is None
            mock_logger.info.assert_called()
            # Check that log message mentions low-quality and skip
            log_message = mock_logger.info.call_args[0][0]
            assert 'low-quality' in log_message or 'skip' in log_message.lower()


# =============================================================================
# Test Cache Quality Tier Logging
# =============================================================================

class TestCacheQualityTierLogging:
    """Tests for logging cache quality tier on hit."""

    def setup_method(self):
        """Create a temporary directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.fast
    def test_cache_hit_logs_quality_tier(self):
        """Cache hit should log quality tier info."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider')
        request = LLMRequest(prompt="log tier test")
        response = LLMResponse(
            text='{"confidence": 0.85}',
            parsed_data={'confidence': 0.85},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        with patch('src.llm_client.cache.logger') as mock_logger:
            cached = cache.get(request)
            assert cached is not None
            mock_logger.debug.assert_called()
            # Check that log message includes quality_tier
            log_message = mock_logger.debug.call_args[0][0]
            assert 'quality_tier=high' in log_message or 'high' in log_message

    @pytest.mark.fast
    def test_cache_hit_logs_confidence_range(self):
        """Cache hit should log confidence range for tier."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest, LLMResponse

        cache = LLMCache(self.temp_dir, 'test_provider')
        request = LLMRequest(prompt="log range test")
        response = LLMResponse(
            text='{"confidence": 0.65}',
            parsed_data={'confidence': 0.65},
            provider='test',
            model='test-model'
        )

        cache.set(request, response)

        with patch('src.llm_client.cache.logger') as mock_logger:
            cached = cache.get(request)
            assert cached is not None
            mock_logger.debug.assert_called()


# =============================================================================
# Test Config Option
# =============================================================================

class TestConfigOption:
    """Tests for expire_low_quality config option."""

    @pytest.mark.fast
    def test_llm_cache_config_has_expire_low_quality_field(self):
        """LLMCacheConfig should have expire_low_quality field."""
        from src.config.sections.llm import LLMCacheConfig
        config = LLMCacheConfig()
        assert hasattr(config, 'expire_low_quality')

    @pytest.mark.fast
    def test_llm_cache_config_expire_low_quality_default_false(self):
        """expire_low_quality should default to False."""
        from src.config.sections.llm import LLMCacheConfig
        config = LLMCacheConfig()
        assert config.expire_low_quality is False

    @pytest.mark.fast
    def test_llm_cache_config_expire_low_quality_can_be_true(self):
        """expire_low_quality can be set to True."""
        from src.config.sections.llm import LLMCacheConfig
        config = LLMCacheConfig(expire_low_quality=True)
        assert config.expire_low_quality is True


# =============================================================================
# Test LLMCache Constructor
# =============================================================================

class TestLLMCacheConstructor:
    """Tests for LLMCache constructor with skip_low_quality parameter."""

    def setup_method(self):
        """Create a temporary directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.fast
    def test_llm_cache_accepts_skip_low_quality_param(self):
        """LLMCache should accept skip_low_quality parameter."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(self.temp_dir, 'test', skip_low_quality=True)
        assert cache.skip_low_quality is True

    @pytest.mark.fast
    def test_llm_cache_skip_low_quality_defaults_to_false(self):
        """LLMCache skip_low_quality should default to False."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(self.temp_dir, 'test')
        assert cache.skip_low_quality is False


# =============================================================================
# Test Factory Integration
# =============================================================================

class TestFactoryIntegration:
    """Tests for factory passing cache_skip_low_quality to clients."""

    @pytest.mark.fast
    def test_create_client_accepts_cache_skip_low_quality(self):
        """create_client should accept cache_skip_low_quality parameter."""
        from src.llm_client.factory import create_client
        # Should not raise an error
        import inspect
        sig = inspect.signature(create_client)
        assert 'cache_skip_low_quality' in sig.parameters

    @pytest.mark.fast
    def test_create_client_from_config_reads_expire_low_quality(self):
        """create_client_from_config should read expire_low_quality from config."""
        from src.llm_client.factory import create_client_from_config
        from src.config.sections.llm import LLMConfig, LLMCacheConfig

        # Create mock config with expire_low_quality=True
        mock_config = Mock()
        mock_config.llm = LLMConfig()
        mock_config.llm.cache = LLMCacheConfig(expire_low_quality=True)
        mock_config.llm.provider = 'ollama'  # Use ollama to avoid API key requirement
        mock_config.cache_dir = '.cache'

        with patch('src.llm_client.factory.create_client') as mock_create:
            mock_create.return_value = Mock()
            create_client_from_config(mock_config)
            mock_create.assert_called_once()
            # Check that cache_skip_low_quality=True was passed
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs.get('cache_skip_low_quality') is True


# =============================================================================
# Test LLMClient Base Class Integration
# =============================================================================

class TestLLMClientBaseIntegration:
    """Tests for LLMClient passing skip_low_quality to cache."""

    @pytest.mark.fast
    def test_llm_client_accepts_cache_skip_low_quality(self):
        """LLMClient.__init__ should accept cache_skip_low_quality parameter."""
        from src.llm_client.base import LLMClient
        import inspect
        sig = inspect.signature(LLMClient.__init__)
        assert 'cache_skip_low_quality' in sig.parameters

    @pytest.mark.integration
    def test_llm_client_passes_skip_low_quality_to_cache(self):
        """LLMClient should pass cache_skip_low_quality to LLMCache."""
        # We can't instantiate abstract LLMClient directly, so test via providers
        from src.llm_client.providers import OllamaClient
        import tempfile
        import shutil

        temp_dir = tempfile.mkdtemp()
        try:
            client = OllamaClient(
                model='test',
                cache_dir=temp_dir,
                cache_skip_low_quality=True
            )
            # Access the cache property to trigger lazy initialization
            cache = client.cache
            assert cache.skip_low_quality is True
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# Test Provider Classes Integration
# =============================================================================

class TestProviderClassesIntegration:
    """Tests for provider classes accepting cache_skip_low_quality."""

    @pytest.mark.fast
    def test_gemini_client_accepts_cache_skip_low_quality(self):
        """GeminiClient should accept cache_skip_low_quality parameter."""
        from src.llm_client.providers import GeminiClient
        import inspect
        sig = inspect.signature(GeminiClient.__init__)
        assert 'cache_skip_low_quality' in sig.parameters

    @pytest.mark.fast
    def test_anthropic_client_accepts_cache_skip_low_quality(self):
        """AnthropicClient should accept cache_skip_low_quality parameter."""
        from src.llm_client.providers import AnthropicClient
        import inspect
        sig = inspect.signature(AnthropicClient.__init__)
        assert 'cache_skip_low_quality' in sig.parameters

    @pytest.mark.fast
    def test_ollama_client_accepts_cache_skip_low_quality(self):
        """OllamaClient should accept cache_skip_low_quality parameter."""
        from src.llm_client.providers import OllamaClient
        import inspect
        sig = inspect.signature(OllamaClient.__init__)
        assert 'cache_skip_low_quality' in sig.parameters


# =============================================================================
# Test Legacy Cache Entry Compatibility
# =============================================================================

class TestLegacyCacheEntryCompatibility:
    """Tests for handling cache entries without quality_tier (migration)."""

    def setup_method(self):
        """Create a temporary directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.fast
    def test_legacy_cache_entry_without_quality_tier_returns_unknown(self):
        """Legacy cache entries without quality_tier should be treated as 'unknown'."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest

        cache = LLMCache(self.temp_dir, 'test_provider')

        # Create a legacy cache entry manually (without quality_tier)
        legacy_data = {
            "text": "legacy response",
            "parsed_data": {"confidence": 0.75},
            "cached_at": time.time(),
            "provider": "test",
            "model": "test-model"
            # No quality_tier field
        }

        cache_dir = Path(self.temp_dir) / 'test_provider'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / 'default_abc123def456.json'
        with open(cache_file, 'w') as f:
            json.dump(legacy_data, f)

        # Create request that would match this cache key
        request = LLMRequest(prompt="legacy test")

        # Manually create the correct cache file name based on request
        cache._cache_key(request)  # To ensure cache_key is computed
        # Re-save with correct filename
        correct_file = cache_dir / f"default_{cache._cache_key(request)}.json"
        with open(correct_file, 'w') as f:
            json.dump(legacy_data, f)

        cached = cache.get(request)
        assert cached is not None
        assert cached.get('quality_tier', 'unknown') == 'unknown'

    @pytest.mark.fast
    def test_legacy_cache_entry_not_skipped_when_skip_low_quality_true(self):
        """Legacy entries (unknown tier) should NOT be skipped even with skip_low_quality=True."""
        from src.llm_client.cache import LLMCache
        from src.llm_client.base import LLMRequest

        cache = LLMCache(self.temp_dir, 'test_provider', skip_low_quality=True)

        # Create a legacy cache entry manually (without quality_tier)
        legacy_data = {
            "text": "legacy response",
            "parsed_data": {"confidence": 0.35},  # Low confidence but no tier
            "cached_at": time.time(),
            "provider": "test",
            "model": "test-model"
            # No quality_tier field
        }

        request = LLMRequest(prompt="legacy skip test")
        cache_dir = Path(self.temp_dir) / 'test_provider'
        cache_dir.mkdir(parents=True, exist_ok=True)
        correct_file = cache_dir / f"default_{cache._cache_key(request)}.json"
        with open(correct_file, 'w') as f:
            json.dump(legacy_data, f)

        # Should NOT be skipped because quality_tier is 'unknown', not 'low'
        cached = cache.get(request)
        assert cached is not None


# =============================================================================
# Test _get_confidence_range Helper
# =============================================================================

class TestGetConfidenceRangeHelper:
    """Tests for _get_confidence_range helper method."""

    def setup_method(self):
        """Create a temporary directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.fast
    def test_get_confidence_range_high(self):
        """High tier should show >= 0.8."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(self.temp_dir, 'test')
        result = cache._get_confidence_range('high')
        assert '0.8' in result

    @pytest.mark.fast
    def test_get_confidence_range_medium(self):
        """Medium tier should show range."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(self.temp_dir, 'test')
        result = cache._get_confidence_range('medium')
        assert '0.5' in result
        assert '0.8' in result

    @pytest.mark.fast
    def test_get_confidence_range_low(self):
        """Low tier should show < 0.5."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(self.temp_dir, 'test')
        result = cache._get_confidence_range('low')
        assert '0.5' in result

    @pytest.mark.fast
    def test_get_confidence_range_unknown(self):
        """Unknown tier should return empty string."""
        from src.llm_client.cache import LLMCache
        cache = LLMCache(self.temp_dir, 'test')
        result = cache._get_confidence_range('unknown')
        assert result == ""
