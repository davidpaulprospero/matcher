"""
Tests for embedding validation.

US-004: Add embedding batch size validation
- validate_batch_size() function checks provider limits
- Returns warning if batch_size exceeds provider maximum
- Provider limits: Gemini: 100, Voyage: 128, OpenAI: 2048, Local: 256

US-53-004: Add embedding integrity validation
- validate_embedding_integrity() checks None values, dimensions, norms
"""

import pytest
import logging
import sys
import numpy as np
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.embeddings import (
    validate_batch_size,
    validate_embedding_integrity,
    EmbeddingValidationResult,
    PROVIDER_MAX_BATCH_SIZES,
    BATCH_SIZES,
)

pytestmark = pytest.mark.unit


class TestValidateBatchSizeFunction:
    """Tests for validate_batch_size() function existence and signature"""

    @pytest.mark.fast
    def test_function_exists(self):
        """validate_batch_size function exists in embeddings module"""
        from src import embeddings
        assert hasattr(embeddings, 'validate_batch_size')
        assert callable(embeddings.validate_batch_size)

    @pytest.mark.fast
    def test_function_accepts_batch_size_and_provider(self):
        """Function accepts batch_size and provider parameters"""
        # Should not raise any errors
        result = validate_batch_size(100, 'gemini')
        assert result is None or isinstance(result, str)

    @pytest.mark.fast
    def test_function_returns_string_or_none(self):
        """Function returns either string warning or None"""
        # Within limit - returns None
        result = validate_batch_size(50, 'gemini')
        assert result is None

        # Exceeds limit - returns string
        result = validate_batch_size(200, 'gemini')
        assert isinstance(result, str)


class TestProviderMaxBatchSizes:
    """Tests for PROVIDER_MAX_BATCH_SIZES constant"""

    @pytest.mark.fast
    def test_constant_exists(self):
        """PROVIDER_MAX_BATCH_SIZES constant exists"""
        from src import embeddings
        assert hasattr(embeddings, 'PROVIDER_MAX_BATCH_SIZES')

    @pytest.mark.fast
    def test_gemini_max_is_100(self):
        """Gemini provider max batch size is 100"""
        assert PROVIDER_MAX_BATCH_SIZES['gemini'] == 100

    @pytest.mark.fast
    def test_voyage_max_is_128(self):
        """Voyage provider max batch size is 128"""
        assert PROVIDER_MAX_BATCH_SIZES['voyage'] == 128

    @pytest.mark.fast
    def test_openai_max_is_2048(self):
        """OpenAI provider max batch size is 2048"""
        assert PROVIDER_MAX_BATCH_SIZES['openai'] == 2048

    @pytest.mark.fast
    def test_local_max_is_256(self):
        """Local provider max batch size is 256"""
        assert PROVIDER_MAX_BATCH_SIZES['local'] == 256

    @pytest.mark.fast
    def test_all_providers_have_limits(self):
        """All providers in BATCH_SIZES have corresponding max limits"""
        for provider in BATCH_SIZES.keys():
            assert provider in PROVIDER_MAX_BATCH_SIZES, f"Missing max limit for {provider}"


class TestGeminiValidation:
    """Tests for Gemini provider batch size validation"""

    @pytest.mark.fast
    def test_batch_size_100_gemini_no_warning(self):
        """batch_size=100 with gemini provider returns no warning"""
        result = validate_batch_size(100, 'gemini')
        assert result is None

    @pytest.mark.fast
    def test_batch_size_150_gemini_returns_warning(self):
        """batch_size=150 with gemini provider returns warning"""
        result = validate_batch_size(150, 'gemini')
        assert result is not None
        assert isinstance(result, str)
        assert '150' in result
        assert 'gemini' in result.lower()
        assert '100' in result

    @pytest.mark.fast
    def test_batch_size_50_gemini_no_warning(self):
        """batch_size=50 (under limit) with gemini returns no warning"""
        result = validate_batch_size(50, 'gemini')
        assert result is None

    @pytest.mark.fast
    def test_batch_size_99_gemini_no_warning(self):
        """batch_size=99 (just under limit) with gemini returns no warning"""
        result = validate_batch_size(99, 'gemini')
        assert result is None

    @pytest.mark.fast
    def test_batch_size_101_gemini_returns_warning(self):
        """batch_size=101 (just over limit) with gemini returns warning"""
        result = validate_batch_size(101, 'gemini')
        assert result is not None
        assert '101' in result


class TestVoyageValidation:
    """Tests for Voyage provider batch size validation"""

    @pytest.mark.fast
    def test_batch_size_128_voyage_no_warning(self):
        """batch_size=128 with voyage provider returns no warning"""
        result = validate_batch_size(128, 'voyage')
        assert result is None

    @pytest.mark.fast
    def test_batch_size_150_voyage_returns_warning(self):
        """batch_size=150 with voyage provider returns warning"""
        result = validate_batch_size(150, 'voyage')
        assert result is not None
        assert '150' in result
        assert 'voyage' in result.lower()
        assert '128' in result

    @pytest.mark.fast
    def test_batch_size_100_voyage_no_warning(self):
        """batch_size=100 (under limit) with voyage returns no warning"""
        result = validate_batch_size(100, 'voyage')
        assert result is None


class TestOpenAIValidation:
    """Tests for OpenAI provider batch size validation"""

    @pytest.mark.fast
    def test_batch_size_2048_openai_no_warning(self):
        """batch_size=2048 with openai provider returns no warning"""
        result = validate_batch_size(2048, 'openai')
        assert result is None

    @pytest.mark.fast
    def test_batch_size_3000_openai_returns_warning(self):
        """batch_size=3000 with openai provider returns warning"""
        result = validate_batch_size(3000, 'openai')
        assert result is not None
        assert '3000' in result
        assert '2048' in result


class TestLocalValidation:
    """Tests for local provider batch size validation"""

    @pytest.mark.fast
    def test_batch_size_256_local_no_warning(self):
        """batch_size=256 with local provider returns no warning"""
        result = validate_batch_size(256, 'local')
        assert result is None

    @pytest.mark.fast
    def test_batch_size_500_local_returns_warning(self):
        """batch_size=500 with local provider returns warning"""
        result = validate_batch_size(500, 'local')
        assert result is not None
        assert '500' in result
        assert '256' in result


class TestCaseInsensitiveProvider:
    """Tests for case-insensitive provider name handling"""

    @pytest.mark.fast
    def test_uppercase_provider_name(self):
        """Provider name is case-insensitive (GEMINI)"""
        result = validate_batch_size(150, 'GEMINI')
        assert result is not None
        assert 'gemini' in result.lower()

    @pytest.mark.fast
    def test_mixed_case_provider_name(self):
        """Provider name is case-insensitive (Gemini)"""
        result = validate_batch_size(150, 'Gemini')
        assert result is not None
        assert 'gemini' in result.lower()

    @pytest.mark.fast
    def test_voyage_mixed_case(self):
        """Provider name is case-insensitive (Voyage)"""
        result = validate_batch_size(200, 'Voyage')
        assert result is not None
        assert 'voyage' in result.lower()


class TestUnknownProvider:
    """Tests for unknown provider handling"""

    @pytest.mark.fast
    def test_unknown_provider_uses_default_100(self):
        """Unknown provider defaults to max of 100"""
        result = validate_batch_size(150, 'unknown_provider')
        assert result is not None
        assert '150' in result
        assert '100' in result

    @pytest.mark.fast
    def test_unknown_provider_under_100_no_warning(self):
        """Unknown provider with batch_size <= 100 returns no warning"""
        result = validate_batch_size(100, 'unknown_provider')
        assert result is None


class TestWarningLogging:
    """Tests for warning logging behavior"""

    @pytest.mark.fast
    def test_warning_logged_when_exceeds_limit(self, caplog):
        """Warning is logged when batch size exceeds provider limit"""
        with caplog.at_level(logging.WARNING):
            validate_batch_size(200, 'gemini')

        assert len(caplog.records) == 1
        assert 'Batch size 200 exceeds gemini maximum of 100' in caplog.text

    @pytest.mark.fast
    def test_no_warning_logged_when_within_limit(self, caplog):
        """No warning is logged when batch size is within limit"""
        with caplog.at_level(logging.WARNING):
            validate_batch_size(50, 'gemini')

        assert len(caplog.records) == 0

    @pytest.mark.fast
    def test_warning_message_includes_suggestion(self, caplog):
        """Warning message includes suggestion to reduce batch_size"""
        with caplog.at_level(logging.WARNING):
            result = validate_batch_size(200, 'gemini')

        assert 'Consider reducing batch_size' in result


class TestEdgeCases:
    """Tests for edge cases"""

    @pytest.mark.fast
    def test_batch_size_zero(self):
        """batch_size=0 returns no warning (under all limits)"""
        result = validate_batch_size(0, 'gemini')
        assert result is None

    @pytest.mark.fast
    def test_batch_size_one(self):
        """batch_size=1 returns no warning"""
        result = validate_batch_size(1, 'gemini')
        assert result is None

    @pytest.mark.fast
    def test_very_large_batch_size(self):
        """Very large batch_size returns warning for all providers"""
        result = validate_batch_size(10000, 'openai')
        assert result is not None
        assert '10000' in result

    @pytest.mark.fast
    def test_negative_batch_size(self):
        """Negative batch_size returns no warning (technically under limit)"""
        result = validate_batch_size(-1, 'gemini')
        assert result is None


# =====================================================================
# US-53-004: validate_embedding_integrity() tests
# =====================================================================

class TestValidateEmbeddingIntegrityBasic:
    """Basic tests for validate_embedding_integrity()."""

    @pytest.mark.fast
    def test_function_exists(self):
        """validate_embedding_integrity function exists in embeddings module."""
        from src import embeddings
        assert hasattr(embeddings, 'validate_embedding_integrity')
        assert callable(embeddings.validate_embedding_integrity)

    @pytest.mark.fast
    def test_returns_validation_result(self):
        """Function returns EmbeddingValidationResult."""
        embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        result = validate_embedding_integrity(embeddings)
        assert isinstance(result, EmbeddingValidationResult)

    @pytest.mark.fast
    def test_all_valid_embeddings(self):
        """All valid unit vectors should pass validation."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        result = validate_embedding_integrity(embeddings)
        assert result.is_valid
        assert result.total_count == 3
        assert result.valid_count == 3
        assert result.none_count == 0
        assert result.wrong_dimension_count == 0
        assert result.invalid_indices == []

    @pytest.mark.fast
    def test_empty_embeddings(self):
        """Empty list returns zero counts."""
        result = validate_embedding_integrity([])
        assert result.total_count == 0
        assert result.valid_count == 0
        assert result.all_none  # 0 == 0 (vacuous truth — no embeddings at all)


class TestValidateEmbeddingIntegrityNone:
    """Tests for None embedding detection."""

    @pytest.mark.fast
    def test_detects_none_embeddings(self):
        """None embeddings should be detected and counted."""
        embeddings = [[1.0, 0.0, 0.0], None, [0.0, 1.0, 0.0], None]
        result = validate_embedding_integrity(embeddings)
        assert result.none_count == 2
        assert result.valid_count == 2
        assert result.total_count == 4
        assert 1 in result.invalid_indices
        assert 3 in result.invalid_indices

    @pytest.mark.fast
    def test_all_none_embeddings(self):
        """All-None embeddings should be flagged."""
        embeddings = [None, None, None]
        result = validate_embedding_integrity(embeddings)
        assert result.all_none
        assert result.none_count == 3
        assert result.valid_count == 0
        assert not result.is_valid


class TestValidateEmbeddingIntegrityDimensions:
    """Tests for dimension consistency checks."""

    @pytest.mark.fast
    def test_detects_wrong_dimensions(self):
        """Embeddings with inconsistent dimensions should be detected."""
        embeddings = [
            [1.0, 0.0, 0.0],     # 3D - sets expected
            [0.0, 1.0],           # 2D - wrong
            [0.0, 0.0, 1.0],     # 3D - OK
        ]
        result = validate_embedding_integrity(embeddings)
        assert result.wrong_dimension_count == 1
        assert result.expected_dimension == 3
        assert 1 in result.invalid_indices

    @pytest.mark.fast
    def test_consistent_dimensions_pass(self):
        """Embeddings with consistent dimensions should all pass."""
        dim = 768
        embeddings = [np.random.randn(dim).tolist() for _ in range(5)]
        # Normalize them
        for i in range(len(embeddings)):
            vec = np.array(embeddings[i])
            embeddings[i] = (vec / np.linalg.norm(vec)).tolist()
        result = validate_embedding_integrity(embeddings)
        assert result.wrong_dimension_count == 0
        assert result.expected_dimension == dim


class TestValidateEmbeddingIntegrityNorms:
    """Tests for norm (unit vector) checks."""

    @pytest.mark.fast
    def test_zero_vectors_flagged(self):
        """Zero vectors should be flagged as invalid."""
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Zero vector
            [0.0, 1.0, 0.0],
        ]
        result = validate_embedding_integrity(embeddings)
        assert result.unnormalized_count == 1
        assert 1 in result.invalid_indices

    @pytest.mark.fast
    def test_unnormalized_vectors_counted_but_valid(self):
        """Unnormalized (non-zero, non-unit) vectors are counted but stay valid."""
        embeddings = [
            [1.0, 0.0, 0.0],      # Unit vector
            [5.0, 0.0, 0.0],      # Unnormalized (norm=5)
            [0.0, 1.0, 0.0],      # Unit vector
        ]
        result = validate_embedding_integrity(embeddings)
        assert result.unnormalized_count == 1
        # Unnormalized but non-zero vectors are still in valid_indices
        assert 1 in result.valid_indices


class TestValidateEmbeddingIntegrityMixed:
    """Tests with mixed valid/None/wrong-dimension inputs (acceptance criterion)."""

    @pytest.mark.fast
    def test_mixed_valid_none_wrong_dimension(self):
        """Mixed inputs: valid, None, wrong dimension."""
        embeddings = [
            [1.0, 0.0, 0.0],      # Valid (3D unit)
            None,                   # None
            [0.0, 1.0],            # Wrong dimension (2D)
            [0.0, 0.0, 1.0],      # Valid (3D unit)
            None,                   # None
            [0.0, 0.0, 0.0],      # Zero vector
        ]
        result = validate_embedding_integrity(embeddings)
        assert result.total_count == 6
        assert result.none_count == 2
        assert result.wrong_dimension_count == 1
        assert result.unnormalized_count == 1  # zero vector
        assert result.valid_count == 2
        assert result.expected_dimension == 3
        assert not result.is_valid
        assert not result.all_none
        # Invalid indices: 1 (None), 2 (wrong dim), 4 (None), 5 (zero vec)
        assert set(result.invalid_indices) == {1, 2, 4, 5}
        assert set(result.valid_indices) == {0, 3}

    @pytest.mark.fast
    def test_numpy_array_input(self):
        """Works with numpy arrays as input."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype='float32')
        result = validate_embedding_integrity(embeddings)
        assert result.is_valid
        assert result.total_count == 3
        assert result.valid_count == 3

    @pytest.mark.fast
    def test_none_input(self):
        """None input returns empty result."""
        result = validate_embedding_integrity(None)
        assert result.total_count == 0
        assert result.valid_count == 0
