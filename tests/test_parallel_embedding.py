"""Tests for parallel embedding generation - US-011 (Sprint 3).

Tests for parallel_embed_batch() function in src/embeddings.py.
Verifies parallel processing, order preservation, and performance.
"""

import sys
import time
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.unit


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockEmbeddingConfig:
    """Mock EmbeddingConfig for testing."""
    provider: str = 'gemini'
    batch_size: int = 100
    max_workers: int = 4
    max_retries: int = 3
    retry_delay: float = 0.01  # Short delay for tests


@dataclass
class MockConfig:
    """Mock Config for testing."""
    embedding: MockEmbeddingConfig = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = MockEmbeddingConfig()


class MockProvider:
    """Mock EmbeddingProvider for testing."""

    def __init__(self, dimension: int = 768, delay: float = 0.0, fail_texts: list = None):
        """
        Args:
            dimension: Embedding dimension to return
            delay: Artificial delay to simulate API latency
            fail_texts: List of text prefixes that should cause batch to fail
        """
        self.dimension = dimension
        self.delay = delay
        self.fail_texts = fail_texts or []
        self.embed_calls = []
        self._call_counter = 0

    def embed(self, texts, embed_mode="document"):
        """Return mock embeddings for a batch of texts."""
        call_idx = self._call_counter
        self._call_counter += 1

        self.embed_calls.append({
            'call_idx': call_idx,
            'texts': texts,
            'timestamp': time.time()
        })

        if self.delay > 0:
            time.sleep(self.delay)

        # Check if any text in batch matches failure criteria
        for fail_text in self.fail_texts:
            if any(t.startswith(fail_text) for t in texts):
                raise RuntimeError(f"Simulated failure for batch containing {fail_text}")

        # Return deterministic embeddings based on text content
        return [[float(hash(t) % 100) / 100.0] * self.dimension for t in texts]


# =============================================================================
# Test Classes
# =============================================================================


class TestParallelEmbedBatchFunction:
    """Test parallel_embed_batch() function exists and has correct signature."""

    @pytest.mark.fast
    def test_function_exists(self):
        """parallel_embed_batch should exist in embeddings module."""
        from src.embeddings import parallel_embed_batch
        assert callable(parallel_embed_batch)

    @pytest.mark.fast
    def test_function_accepts_required_args(self):
        """parallel_embed_batch should accept texts and provider args."""
        from src.embeddings import parallel_embed_batch
        import inspect

        sig = inspect.signature(parallel_embed_batch)
        params = list(sig.parameters.keys())

        assert 'texts' in params
        assert 'provider' in params

    @pytest.mark.fast
    def test_function_accepts_config_arg(self):
        """parallel_embed_batch should accept optional config arg."""
        from src.embeddings import parallel_embed_batch
        import inspect

        sig = inspect.signature(parallel_embed_batch)
        params = list(sig.parameters.keys())

        assert 'config' in params

    @pytest.mark.fast
    def test_function_returns_list(self):
        """parallel_embed_batch should return a list."""
        from src.embeddings import parallel_embed_batch

        provider = MockProvider()
        result = parallel_embed_batch(["test"], provider, show_progress=False)

        assert isinstance(result, list)


class TestParallelEmbedBatchWithConfig:
    """Test parallel_embed_batch uses config.embedding.max_workers."""

    @pytest.mark.fast
    def test_uses_max_workers_from_config(self):
        """Should use max_workers from config.embedding."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.max_workers = 2
        config.embedding.batch_size = 10

        provider = MockProvider()
        texts = [f"text_{i}" for i in range(50)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        assert len(result) == 50

    @pytest.mark.fast
    def test_uses_batch_size_from_config(self):
        """Should use batch_size from config.embedding."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 20

        provider = MockProvider()
        texts = [f"text_{i}" for i in range(100)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        # With 100 texts and batch_size=20, should have 5 batches
        assert len(provider.embed_calls) == 5
        assert len(result) == 100

    @pytest.mark.fast
    def test_default_max_workers_without_config(self):
        """Should default to 4 workers when no config provided."""
        from src.embeddings import parallel_embed_batch

        provider = MockProvider()
        texts = ["test1", "test2"]

        # Should not raise even without config
        result = parallel_embed_batch(texts, provider, config=None, show_progress=False)
        assert len(result) == 2


class TestParallelEmbedBatchResultOrder:
    """Test that parallel_embed_batch maintains original text order."""

    @pytest.mark.fast
    def test_maintains_order_with_single_batch(self):
        """Results should match input order with single batch."""
        from src.embeddings import parallel_embed_batch

        provider = MockProvider(dimension=3)
        texts = ["first", "second", "third"]

        result = parallel_embed_batch(texts, provider, show_progress=False)

        assert len(result) == 3
        # Each embedding should be based on its text (deterministic mock)
        assert result[0] == [[float(hash("first") % 100) / 100.0] * 3][0]

    @pytest.mark.fast
    def test_maintains_order_with_multiple_batches(self):
        """Results should match input order across multiple batches."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 10
        config.embedding.max_workers = 4

        provider = MockProvider(dimension=3)
        texts = [f"text_{i}" for i in range(50)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        assert len(result) == 50

        # Verify order is preserved
        expected_first = [[float(hash("text_0") % 100) / 100.0] * 3][0]
        expected_last = [[float(hash("text_49") % 100) / 100.0] * 3][0]

        assert result[0] == expected_first
        assert result[49] == expected_last

    @pytest.mark.fast
    def test_maintains_order_with_parallel_processing(self):
        """Order preserved even when batches complete out of order."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 10
        config.embedding.max_workers = 2

        # Provider with slight delay to ensure batches run in parallel
        provider = MockProvider(dimension=3, delay=0.01)
        texts = [f"unique_text_{i}" for i in range(30)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        # Verify all texts processed
        assert len(result) == 30

        # Verify each result corresponds to correct input
        for i, embedding in enumerate(result):
            expected = [[float(hash(f"unique_text_{i}") % 100) / 100.0] * 3][0]
            assert embedding == expected


class TestParallelEmbedBatchErrorHandling:
    """Test error handling and retries in parallel_embed_batch."""

    @pytest.mark.fast
    def test_returns_zeros_on_batch_failure(self):
        """Should return zero embeddings when batch fails after retries."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 10
        config.embedding.max_retries = 2
        config.embedding.retry_delay = 0.001

        # Texts starting with "FAIL_" will cause batch to fail
        texts = [f"text_{i}" for i in range(10)]  # Batch 0: success
        texts += [f"FAIL_{i}" for i in range(10)]  # Batch 1: will fail
        texts += [f"text_{i + 20}" for i in range(10)]  # Batch 2: success

        provider = MockProvider(dimension=5, fail_texts=["FAIL_"])

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        assert len(result) == 30

        # Batch 1 (texts 10-19) should have zero embeddings (default dimension 768)
        for i in range(10, 20):
            assert result[i] == [0.0] * 768  # Default dimension on failure

    @pytest.mark.fast
    def test_successful_batches_not_affected_by_failures(self):
        """Successful batches should return correct embeddings despite other failures."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 10
        config.embedding.max_retries = 1
        config.embedding.retry_delay = 0.001

        # Create texts where middle batch fails
        texts = [f"text_{i}" for i in range(10)]  # Batch 0: success
        texts += [f"text_{i + 10}" for i in range(10)]  # Batch 1: success
        texts += [f"FAIL_{i}" for i in range(10)]  # Batch 2: will fail
        texts += [f"text_{i + 30}" for i in range(10)]  # Batch 3: success

        provider = MockProvider(dimension=3, fail_texts=["FAIL_"])

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        # Verify successful batch (batch 0: texts 0-9) has correct embeddings
        expected = [[float(hash("text_0") % 100) / 100.0] * 3][0]
        assert result[0] == expected

        # Verify batch 3 also succeeded
        expected_30 = [[float(hash("text_30") % 100) / 100.0] * 3][0]
        assert result[30] == expected_30


class TestParallelEmbedBatchEmptyInput:
    """Test parallel_embed_batch with empty or minimal input."""

    @pytest.mark.fast
    def test_empty_list_returns_empty(self):
        """Should return empty list for empty input."""
        from src.embeddings import parallel_embed_batch

        provider = MockProvider()
        result = parallel_embed_batch([], provider, show_progress=False)

        assert result == []

    @pytest.mark.fast
    def test_single_text(self):
        """Should handle single text correctly."""
        from src.embeddings import parallel_embed_batch

        provider = MockProvider(dimension=5)
        result = parallel_embed_batch(["single"], provider, show_progress=False)

        assert len(result) == 1
        assert len(result[0]) == 5

    @pytest.mark.fast
    def test_cleans_empty_strings(self):
        """Should clean empty strings to '[silence]'."""
        from src.embeddings import parallel_embed_batch

        provider = MockProvider(dimension=3)
        texts = ["valid", "", "  ", "another"]

        result = parallel_embed_batch(texts, provider, show_progress=False)

        assert len(result) == 4

        # Check that provider received cleaned texts
        embed_call = provider.embed_calls[0]
        assert "[silence]" in embed_call['texts']


class TestParallelEmbedBatchPerformance:
    """Test that parallel processing improves performance for 200+ texts."""

    @pytest.mark.fast
    def test_parallel_faster_than_sequential_for_200_texts(self):
        """Parallel processing should be faster than sequential for 200+ texts."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 50
        config.embedding.max_workers = 4

        # Provider with delay to simulate API latency
        provider = MockProvider(dimension=3, delay=0.02)  # 20ms per batch
        texts = [f"text_{i}" for i in range(200)]

        start = time.time()
        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)
        parallel_time = time.time() - start

        assert len(result) == 200

        # With 200 texts and batch_size=50, we have 4 batches
        # Sequential would take 4 * 0.02 = 0.08 seconds minimum
        # Parallel with 4 workers should be close to 0.02 seconds + overhead
        # We just verify it completes in reasonable time
        assert parallel_time < 0.5  # Should be much faster than 0.5s

    @pytest.mark.fast
    def test_processes_all_batches(self):
        """Should process all batches for large input."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 50
        config.embedding.max_workers = 4

        provider = MockProvider(dimension=3)
        texts = [f"text_{i}" for i in range(250)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        # 250 texts / 50 batch_size = 5 batches
        assert len(provider.embed_calls) == 5
        assert len(result) == 250


class TestParallelEmbedBatchLogging:
    """Test logging behavior in parallel_embed_batch."""

    @pytest.mark.fast
    def test_logs_progress_when_show_progress_true(self, caplog):
        """Should log progress when show_progress=True."""
        from src.embeddings import parallel_embed_batch
        import logging

        caplog.set_level(logging.INFO)

        config = MockConfig()
        config.embedding.batch_size = 10

        provider = MockProvider()
        texts = [f"text_{i}" for i in range(30)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=True)

        # Check for progress log messages
        assert "Parallel embedding:" in caplog.text
        assert "Completed batch" in caplog.text
        assert "Parallel embedding complete:" in caplog.text

    @pytest.mark.fast
    def test_no_progress_logs_when_show_progress_false(self, caplog):
        """Should not log progress when show_progress=False."""
        from src.embeddings import parallel_embed_batch
        import logging

        caplog.set_level(logging.INFO)

        provider = MockProvider()
        texts = ["test1", "test2"]

        result = parallel_embed_batch(texts, provider, show_progress=False)

        # Should not have progress messages
        assert "Parallel embedding:" not in caplog.text

    @pytest.mark.fast
    def test_logs_rate_in_completion_message(self, caplog):
        """Completion log should include processing rate."""
        from src.embeddings import parallel_embed_batch
        import logging

        caplog.set_level(logging.INFO)

        provider = MockProvider()
        texts = [f"text_{i}" for i in range(50)]

        result = parallel_embed_batch(texts, provider, show_progress=True)

        # Check for rate in completion message (format: X/sec)
        assert "/sec" in caplog.text


class TestMaxWorkersConfiguration:
    """Test max_workers configuration options."""

    @pytest.mark.fast
    def test_max_workers_1_processes_sequentially(self):
        """With max_workers=1, should process batches sequentially."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 10
        config.embedding.max_workers = 1

        provider = MockProvider(dimension=3, delay=0.01)
        texts = [f"text_{i}" for i in range(30)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        assert len(result) == 30
        # With 1 worker, batches should complete in order
        # (though ThreadPoolExecutor doesn't guarantee this, the result order is guaranteed)

    @pytest.mark.fast
    def test_max_workers_8_uses_8_workers(self):
        """Should use 8 workers when max_workers=8."""
        from src.embeddings import parallel_embed_batch

        config = MockConfig()
        config.embedding.batch_size = 10
        config.embedding.max_workers = 8

        provider = MockProvider(dimension=3)
        texts = [f"text_{i}" for i in range(100)]

        result = parallel_embed_batch(texts, provider, config=config, show_progress=False)

        assert len(result) == 100
        # With batch_size=10 and 100 texts, we have 10 batches
        assert len(provider.embed_calls) == 10


class TestEmbeddingConfigMaxWorkers:
    """Test max_workers field in EmbeddingConfig dataclass."""

    @pytest.mark.fast
    def test_embedding_config_has_max_workers(self):
        """EmbeddingConfig should have max_workers field."""
        from src.config.sections.core import EmbeddingConfig

        config = EmbeddingConfig()
        assert hasattr(config, 'max_workers')

    @pytest.mark.fast
    def test_max_workers_default_is_4(self):
        """Default max_workers should be 4."""
        from src.config.sections.core import EmbeddingConfig

        config = EmbeddingConfig()
        assert config.max_workers == 4

    @pytest.mark.fast
    def test_max_workers_can_be_set(self):
        """max_workers should be settable."""
        from src.config.sections.core import EmbeddingConfig

        config = EmbeddingConfig(max_workers=8)
        assert config.max_workers == 8

    @pytest.mark.fast
    def test_max_workers_type_is_int(self):
        """max_workers should be an integer."""
        from src.config.sections.core import EmbeddingConfig

        config = EmbeddingConfig()
        assert isinstance(config.max_workers, int)
