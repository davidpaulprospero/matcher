"""
Tests for logger utility functions in src/logger.py

Tests utility functions for:
- estimate_tokens() - String to token estimation
- get_api_cost() - API cost calculation
- NumpyEncoder - JSON encoding for numpy types
"""

import pytest
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import estimate_tokens, get_api_cost, NumpyEncoder, API_PRICING

# Optional numpy import for testing encoder
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False


class TestEstimateTokens:
    """Test estimate_tokens() function"""

    def test_estimate_tokens_empty_string(self):
        """Test with empty string"""
        result = estimate_tokens("")
        assert result == 0

    def test_estimate_tokens_short_string(self):
        """Test with short string"""
        text = "Hello"  # 5 chars
        result = estimate_tokens(text)
        assert result == 1  # 5 // 4 = 1

    def test_estimate_tokens_exact_multiple(self):
        """Test with exact multiple of 4"""
        text = "1234567890123456"  # 16 chars
        result = estimate_tokens(text)
        assert result == 4  # 16 // 4 = 4

    def test_estimate_tokens_long_string(self):
        """Test with longer string"""
        text = "This is a longer test string with multiple words and spaces." * 10
        # 61 chars * 10 = 610 chars (actual length without trailing newline)
        result = estimate_tokens(text)
        expected_length = len(text)
        assert result == expected_length // 4

    def test_estimate_tokens_unicode_characters(self):
        """Test with Unicode characters"""
        text = "Hello 世界 🌍"  # Mixed ASCII and Unicode
        result = estimate_tokens(text)
        assert isinstance(result, int)
        assert result >= 0

    def test_estimate_tokens_single_character(self):
        """Test with single character"""
        result = estimate_tokens("A")
        assert result == 0  # 1 // 4 = 0

    def test_estimate_tokens_three_characters(self):
        """Test with three characters"""
        result = estimate_tokens("ABC")
        assert result == 0  # 3 // 4 = 0

    def test_estimate_tokens_four_characters(self):
        """Test with four characters"""
        result = estimate_tokens("ABCD")
        assert result == 1  # 4 // 4 = 1

    def test_estimate_tokens_whitespace(self):
        """Test with whitespace only"""
        text = "    "  # 4 spaces
        result = estimate_tokens(text)
        assert result == 1  # 4 // 4 = 1

    def test_estimate_tokens_newlines(self):
        """Test with newlines"""
        text = "Line 1\nLine 2\nLine 3"  # 20 chars
        result = estimate_tokens(text)
        assert result == 5  # 20 // 4 = 5


class TestGetApiCost:
    """Test get_api_cost() function"""

    def test_get_api_cost_gemini_flash(self):
        """Test cost calculation for Gemini Flash"""
        # Gemini 2.0 Flash: input=0.000075, output=0.0003 per 1K tokens
        cost = get_api_cost('gemini-2.0-flash', input_tokens=1000, output_tokens=1000)

        expected = (1000 / 1000) * 0.000075 + (1000 / 1000) * 0.0003
        assert abs(cost - expected) < 0.000001

    def test_get_api_cost_claude_opus(self):
        """Test cost calculation for Claude Opus"""
        # Claude 3 Opus: input=0.015, output=0.075 per 1K tokens
        cost = get_api_cost('claude-3-opus', input_tokens=2000, output_tokens=500)

        expected = (2000 / 1000) * 0.015 + (500 / 1000) * 0.075
        assert abs(cost - expected) < 0.000001

    def test_get_api_cost_zero_tokens(self):
        """Test with zero tokens"""
        cost = get_api_cost('gemini-2.0-flash', input_tokens=0, output_tokens=0)
        assert cost == 0.0

    def test_get_api_cost_input_only(self):
        """Test with input tokens only"""
        cost = get_api_cost('gemini-2.0-flash', input_tokens=1000, output_tokens=0)

        expected = (1000 / 1000) * 0.000075
        assert abs(cost - expected) < 0.000001

    def test_get_api_cost_output_only(self):
        """Test with output tokens only"""
        cost = get_api_cost('gemini-2.0-flash', input_tokens=0, output_tokens=1000)

        expected = (1000 / 1000) * 0.0003
        assert abs(cost - expected) < 0.000001

    def test_get_api_cost_unknown_model_fallback(self):
        """Test with unknown model (should fall back to 'local')"""
        cost = get_api_cost('unknown-model-xyz', input_tokens=1000, output_tokens=1000)

        # Should fall back to 'local' pricing (0, 0)
        assert cost == 0.0

    def test_get_api_cost_local_model(self):
        """Test with local model"""
        cost = get_api_cost('local', input_tokens=1000, output_tokens=1000)
        assert cost == 0.0

    def test_get_api_cost_embedding_model(self):
        """Test with embedding model (no output cost)"""
        # text-embedding-004: input=0.00001, output=0
        cost = get_api_cost('text-embedding-004', input_tokens=5000, output_tokens=0)

        expected = (5000 / 1000) * 0.00001
        assert abs(cost - expected) < 0.000001

    def test_get_api_cost_voyage_embedding(self):
        """Test with Voyage embedding model"""
        # voyage-2: input=0.0001, output=0
        cost = get_api_cost('voyage-2', input_tokens=10000, output_tokens=0)

        expected = (10000 / 1000) * 0.0001
        assert abs(cost - expected) < 0.000001

    def test_get_api_cost_large_token_counts(self):
        """Test with very large token counts"""
        cost = get_api_cost('claude-3-opus', input_tokens=100000, output_tokens=50000)

        expected = (100000 / 1000) * 0.015 + (50000 / 1000) * 0.075
        assert abs(cost - expected) < 0.001

    def test_get_api_cost_fractional_tokens(self):
        """Test with fractional token counts"""
        # Even though estimate_tokens returns int, cost calculation should handle fractions
        cost = get_api_cost('gemini-2.0-flash', input_tokens=1500, output_tokens=750)

        expected = (1500 / 1000) * 0.000075 + (750 / 1000) * 0.0003
        assert abs(cost - expected) < 0.000001


@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestNumpyEncoder:
    """Test NumpyEncoder class for JSON serialization"""

    def test_numpy_encoder_int32(self):
        """Test encoding numpy int32"""
        obj = {'value': np.int32(42)}
        result = json.dumps(obj, cls=NumpyEncoder)

        assert result == '{"value": 42}'
        parsed = json.loads(result)
        assert parsed['value'] == 42

    def test_numpy_encoder_int64(self):
        """Test encoding numpy int64"""
        obj = {'value': np.int64(12345)}
        result = json.dumps(obj, cls=NumpyEncoder)

        assert result == '{"value": 12345}'
        parsed = json.loads(result)
        assert parsed['value'] == 12345

    def test_numpy_encoder_float32(self):
        """Test encoding numpy float32"""
        obj = {'value': np.float32(3.14)}
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert abs(parsed['value'] - 3.14) < 0.001

    def test_numpy_encoder_float64(self):
        """Test encoding numpy float64"""
        obj = {'value': np.float64(2.71828)}
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert abs(parsed['value'] - 2.71828) < 0.00001

    def test_numpy_encoder_bool(self):
        """Test encoding numpy bool"""
        obj = {'true_val': np.bool_(True), 'false_val': np.bool_(False)}
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert parsed['true_val'] is True
        assert parsed['false_val'] is False

    def test_numpy_encoder_array_1d(self):
        """Test encoding 1D numpy array"""
        obj = {'array': np.array([1, 2, 3, 4, 5])}
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert parsed['array'] == [1, 2, 3, 4, 5]

    def test_numpy_encoder_array_2d(self):
        """Test encoding 2D numpy array"""
        obj = {'matrix': np.array([[1, 2], [3, 4]])}
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert parsed['matrix'] == [[1, 2], [3, 4]]

    def test_numpy_encoder_array_float(self):
        """Test encoding float numpy array"""
        obj = {'floats': np.array([1.1, 2.2, 3.3])}
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert len(parsed['floats']) == 3
        assert abs(parsed['floats'][0] - 1.1) < 0.001

    def test_numpy_encoder_empty_array(self):
        """Test encoding empty numpy array"""
        obj = {'empty': np.array([])}
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert parsed['empty'] == []

    def test_numpy_encoder_mixed_types(self):
        """Test encoding dict with mixed numpy types"""
        obj = {
            'int': np.int32(100),
            'float': np.float64(3.14159),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3])
        }
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert parsed['int'] == 100
        assert abs(parsed['float'] - 3.14159) < 0.00001
        assert parsed['bool'] is True
        assert parsed['array'] == [1, 2, 3]

    def test_numpy_encoder_regular_types(self):
        """Test that regular Python types still work"""
        obj = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'}
        }
        result = json.dumps(obj, cls=NumpyEncoder)

        parsed = json.loads(result)
        assert parsed == obj


class TestApiPricing:
    """Test API_PRICING table integrity"""

    def test_api_pricing_has_required_models(self):
        """Test that API_PRICING has all expected models"""
        required_models = [
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

        for model in required_models:
            assert model in API_PRICING, f"Missing pricing for {model}"

    def test_api_pricing_structure(self):
        """Test that each pricing entry has input and output keys"""
        for model, pricing in API_PRICING.items():
            assert 'input' in pricing, f"{model} missing input pricing"
            assert 'output' in pricing, f"{model} missing output pricing"
            assert isinstance(pricing['input'], (int, float))
            assert isinstance(pricing['output'], (int, float))

    def test_api_pricing_non_negative(self):
        """Test that all prices are non-negative"""
        for model, pricing in API_PRICING.items():
            assert pricing['input'] >= 0, f"{model} has negative input price"
            assert pricing['output'] >= 0, f"{model} has negative output price"

    def test_api_pricing_local_is_free(self):
        """Test that local model has zero cost"""
        assert API_PRICING['local']['input'] == 0
        assert API_PRICING['local']['output'] == 0

    def test_api_pricing_embedding_models_no_output(self):
        """Test that embedding models have zero output cost"""
        embedding_models = ['text-embedding-004', 'voyage-2']

        for model in embedding_models:
            assert API_PRICING[model]['output'] == 0, f"{model} should have zero output cost"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_estimate_tokens_very_long_string(self):
        """Test with very long string"""
        text = "A" * 100000  # 100K characters
        result = estimate_tokens(text)
        assert result == 25000  # 100000 // 4

    def test_get_api_cost_consistency(self):
        """Test that cost calculation is consistent"""
        # Same inputs should always give same output
        cost1 = get_api_cost('gemini-2.0-flash', 1000, 500)
        cost2 = get_api_cost('gemini-2.0-flash', 1000, 500)
        assert cost1 == cost2

    def test_get_api_cost_all_models(self):
        """Test cost calculation for all models in API_PRICING"""
        for model in API_PRICING.keys():
            cost = get_api_cost(model, input_tokens=1000, output_tokens=500)
            assert isinstance(cost, float)
            assert cost >= 0

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
    def test_numpy_encoder_with_nested_arrays(self):
        """Test NumpyEncoder with deeply nested numpy arrays"""
        obj = {
            'level1': {
                'level2': {
                    'array': np.array([1, 2, 3])
                }
            }
        }
        result = json.dumps(obj, cls=NumpyEncoder)
        parsed = json.loads(result)
        assert parsed['level1']['level2']['array'] == [1, 2, 3]

    def test_estimate_tokens_special_characters(self):
        """Test estimate_tokens with special characters"""
        text = "!@#$%^&*()_+-=[]{}|;:',.<>?/"  # 31 chars
        result = estimate_tokens(text)
        assert result == 7  # 31 // 4 = 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
