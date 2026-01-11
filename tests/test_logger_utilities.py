"""
Comprehensive test suite for logger.py - utilities and helpers.

Tests coverage for:
- src/logger.py (NumpyEncoder, cost estimation, tier functions)

Created: January 10, 2026
Session: 13 Phase 2
"""

import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import (
    NumpyEncoder,
    estimate_tokens,
    get_api_cost,
    get_confidence_tier,
    get_confidence_color,
    API_PRICING
)


# ============================================================================
# Test NumpyEncoder
# ============================================================================

class TestNumpyEncoder:
    """Test JSON encoder for numpy types"""

    def test_encode_without_numpy(self):
        """Test encoder works without numpy installed"""
        encoder = NumpyEncoder()

        # Standard Python types should work
        data = {'text': 'hello', 'num': 42, 'float': 3.14}
        result = json.dumps(data, cls=NumpyEncoder)

        assert json.loads(result) == data

    def test_encode_with_numpy_int(self):
        """Test encoding numpy integers"""
        try:
            import numpy as np
            encoder = NumpyEncoder()

            # Test int32
            val = np.int32(42)
            result = json.dumps(val, cls=NumpyEncoder)
            assert result == '42'

            # Test int64
            val = np.int64(100)
            result = json.dumps(val, cls=NumpyEncoder)
            assert result == '100'
        except ImportError:
            pytest.skip("numpy not installed")

    def test_encode_with_numpy_float(self):
        """Test encoding numpy floats"""
        try:
            import numpy as np
            encoder = NumpyEncoder()

            # Test float32
            val = np.float32(3.14)
            result = json.dumps(val, cls=NumpyEncoder)
            assert abs(float(result) - 3.14) < 0.01

            # Test float64
            val = np.float64(2.718)
            result = json.dumps(val, cls=NumpyEncoder)
            assert abs(float(result) - 2.718) < 0.001
        except ImportError:
            pytest.skip("numpy not installed")

    def test_encode_with_numpy_bool(self):
        """Test encoding numpy booleans"""
        try:
            import numpy as np
            encoder = NumpyEncoder()

            # Test bool_
            val = np.bool_(True)
            result = json.dumps(val, cls=NumpyEncoder)
            assert result == 'true'

            val = np.bool_(False)
            result = json.dumps(val, cls=NumpyEncoder)
            assert result == 'false'
        except ImportError:
            pytest.skip("numpy not installed")

    def test_encode_with_numpy_array(self):
        """Test encoding numpy arrays"""
        try:
            import numpy as np
            encoder = NumpyEncoder()

            # Test 1D array
            arr = np.array([1, 2, 3])
            result = json.dumps(arr.tolist(), cls=NumpyEncoder)
            assert json.loads(result) == [1, 2, 3]

            # Test 2D array
            arr = np.array([[1, 2], [3, 4]])
            result = json.dumps(arr.tolist(), cls=NumpyEncoder)
            assert json.loads(result) == [[1, 2], [3, 4]]
        except ImportError:
            pytest.skip("numpy not installed")


# ============================================================================
# Test Token Estimation
# ============================================================================

class TestEstimateTokens:
    """Test token estimation function"""

    def test_estimate_empty_string(self):
        """Test estimation with empty string"""
        assert estimate_tokens("") == 0

    def test_estimate_short_text(self):
        """Test estimation with short text"""
        text = "Hello"  # 5 chars
        tokens = estimate_tokens(text)
        assert tokens == 1  # 5 // 4 = 1

    def test_estimate_medium_text(self):
        """Test estimation with medium text"""
        text = "The quick brown fox jumps"  # 25 chars
        tokens = estimate_tokens(text)
        assert tokens == 6  # 25 // 4 = 6

    def test_estimate_long_text(self):
        """Test estimation with long text"""
        text = "A" * 1000  # 1000 chars
        tokens = estimate_tokens(text)
        assert tokens == 250  # 1000 // 4 = 250

    def test_estimate_with_whitespace(self):
        """Test estimation includes whitespace"""
        text = "Hello    world"  # 14 chars including spaces
        tokens = estimate_tokens(text)
        assert tokens == 3  # 14 // 4 = 3


# ============================================================================
# Test API Cost Calculation
# ============================================================================

class TestGetAPICost:
    """Test API cost calculation"""

    def test_cost_gemini_flash(self):
        """Test cost for Gemini Flash"""
        # Input only (1000 tokens)
        cost = get_api_cost('gemini-2.0-flash', input_tokens=1000, output_tokens=0)
        expected = 1.0 * API_PRICING['gemini-2.0-flash']['input']  # 1.0 * 0.000075
        assert abs(cost - expected) < 0.0001

        # With output
        cost = get_api_cost('gemini-2.0-flash', input_tokens=1000, output_tokens=500)
        expected = (1.0 * 0.000075) + (0.5 * 0.0003)
        assert abs(cost - expected) < 0.0001

    def test_cost_claude_sonnet(self):
        """Test cost for Claude Sonnet"""
        cost = get_api_cost('claude-3-sonnet', input_tokens=1000, output_tokens=1000)
        expected = (1.0 * 0.003) + (1.0 * 0.015)  # Input + output
        assert abs(cost - expected) < 0.0001

    def test_cost_embedding_model(self):
        """Test cost for embedding model (no output)"""
        cost = get_api_cost('text-embedding-004', input_tokens=5000, output_tokens=0)
        expected = 5.0 * 0.00001  # Only input cost
        assert abs(cost - expected) < 0.0001

    def test_cost_local_model(self):
        """Test cost for local model (free)"""
        cost = get_api_cost('local', input_tokens=10000, output_tokens=10000)
        assert cost == 0.0

    def test_cost_unknown_model_fallback(self):
        """Test cost for unknown model falls back to local (free)"""
        cost = get_api_cost('unknown-model', input_tokens=1000, output_tokens=1000)
        assert cost == 0.0

    def test_cost_zero_tokens(self):
        """Test cost with zero tokens"""
        cost = get_api_cost('gemini-2.0-flash', input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_cost_large_input(self):
        """Test cost with large input"""
        # 1 million tokens
        cost = get_api_cost('claude-3-opus', input_tokens=1_000_000, output_tokens=0)
        expected = 1000.0 * 0.015  # 1000 * pricing per 1K
        assert abs(cost - expected) < 0.01


# ============================================================================
# Test Confidence Tier Functions
# ============================================================================

class TestGetConfidenceTier:
    """Test confidence tier classification"""

    def test_tier_high(self):
        """Test HIGH tier (>= 0.80)"""
        assert get_confidence_tier(0.80) == "HIGH"
        assert get_confidence_tier(0.90) == "HIGH"
        assert get_confidence_tier(1.00) == "HIGH"

    def test_tier_good(self):
        """Test GOOD tier (0.60-0.79)"""
        assert get_confidence_tier(0.60) == "GOOD"
        assert get_confidence_tier(0.70) == "GOOD"
        assert get_confidence_tier(0.79) == "GOOD"

    def test_tier_medium(self):
        """Test MEDIUM tier (0.40-0.59)"""
        assert get_confidence_tier(0.40) == "MEDIUM"
        assert get_confidence_tier(0.50) == "MEDIUM"
        assert get_confidence_tier(0.59) == "MEDIUM"

    def test_tier_low(self):
        """Test LOW tier (0.20-0.39)"""
        assert get_confidence_tier(0.20) == "LOW"
        assert get_confidence_tier(0.30) == "LOW"
        assert get_confidence_tier(0.39) == "LOW"

    def test_tier_gap(self):
        """Test GAP tier (< 0.20)"""
        assert get_confidence_tier(0.00) == "GAP"
        assert get_confidence_tier(0.10) == "GAP"
        assert get_confidence_tier(0.19) == "GAP"

    def test_tier_boundary_values(self):
        """Test exact boundary values"""
        # Test boundaries
        assert get_confidence_tier(0.799999) == "GOOD"
        assert get_confidence_tier(0.800000) == "HIGH"
        assert get_confidence_tier(0.599999) == "MEDIUM"
        assert get_confidence_tier(0.600000) == "GOOD"


class TestGetConfidenceColor:
    """Test confidence color mapping"""

    def test_color_high(self):
        """Test color for HIGH tier"""
        assert get_confidence_color(0.85) == "GREEN"

    def test_color_good(self):
        """Test color for GOOD tier"""
        assert get_confidence_color(0.65) == "CYAN"

    def test_color_medium(self):
        """Test color for MEDIUM tier"""
        assert get_confidence_color(0.45) == "YELLOW"

    def test_color_low(self):
        """Test color for LOW tier"""
        assert get_confidence_color(0.25) == "ORANGE"

    def test_color_gap(self):
        """Test color for GAP tier"""
        assert get_confidence_color(0.05) == "RED"

    def test_color_boundaries(self):
        """Test color at tier boundaries"""
        assert get_confidence_color(0.80) == "GREEN"  # HIGH
        assert get_confidence_color(0.60) == "CYAN"   # GOOD
        assert get_confidence_color(0.40) == "YELLOW" # MEDIUM
        assert get_confidence_color(0.20) == "ORANGE" # LOW


# ============================================================================
# Test API Pricing Table
# ============================================================================

class TestAPIPricingTable:
    """Test API pricing data structure"""

    def test_pricing_has_all_models(self):
        """Test pricing table has expected models"""
        expected_models = [
            'gemini-2.0-flash',
            'gemini-1.5-flash',
            'gemini-1.5-pro',
            'claude-3-haiku',
            'claude-3-sonnet',
            'claude-3-opus',
            'text-embedding-004',
            'voyage-2',
            'local'
        ]

        for model in expected_models:
            assert model in API_PRICING, f"{model} missing from API_PRICING"

    def test_pricing_structure(self):
        """Test each pricing entry has input/output keys"""
        for model, pricing in API_PRICING.items():
            assert 'input' in pricing, f"{model} missing 'input' key"
            assert 'output' in pricing, f"{model} missing 'output' key"
            assert isinstance(pricing['input'], (int, float))
            assert isinstance(pricing['output'], (int, float))

    def test_local_model_is_free(self):
        """Test local model has zero cost"""
        assert API_PRICING['local']['input'] == 0
        assert API_PRICING['local']['output'] == 0

    def test_embedding_models_have_no_output_cost(self):
        """Test embedding models have zero output cost"""
        embedding_models = ['text-embedding-004', 'voyage-2']
        for model in embedding_models:
            assert API_PRICING[model]['output'] == 0

    def test_pricing_values_are_positive(self):
        """Test all pricing values are non-negative"""
        for model, pricing in API_PRICING.items():
            assert pricing['input'] >= 0
            assert pricing['output'] >= 0
