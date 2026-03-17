"""
Simplified keyword extraction performance benchmarks.

Tests data structure operations and text processing (not LLM calls).
"""

import pytest


class TestKeywordDataStructures:
    """Benchmark keyword data structure operations."""

    @pytest.mark.fast
    def test_segment_text_extraction(self, sample_segments, benchmark):
        """Benchmark extracting text from segments."""
        def extract_texts():
            return [seg.text for seg in sample_segments]

        result = benchmark(extract_texts)
        assert len(result) == len(sample_segments)
        print(f"\nProcessed {len(result)} segment texts")

    @pytest.mark.fast
    def test_keyword_deduplication(self, benchmark):
        """Benchmark keyword deduplication."""
        keywords = ["keyword" + str(i % 50) for i in range(1000)]  # Many duplicates

        def deduplicate():
            return list(set(keywords))

        result = benchmark(deduplicate)
        assert len(result) <= 50
        print(f"\nDedup'd {len(keywords)} keywords to {len(result)} unique")

    @pytest.mark.fast
    def test_keyword_sorting(self, sample_keywords, benchmark):
        """Benchmark keyword sorting by length."""
        extended_keywords = sample_keywords * 100  # 1000 keywords

        def sort_by_length():
            return sorted(extended_keywords, key=len, reverse=True)

        result = benchmark(sort_by_length)
        assert len(result) == len(extended_keywords)
        print(f"\nSorted {len(result)} keywords by length")

    @pytest.mark.fast
    def test_text_tokenization(self, sample_text_corpus, benchmark):
        """Benchmark simple text tokenization."""
        def tokenize():
            return sample_text_corpus.lower().split()

        result = benchmark(tokenize)
        assert len(result) > 0
        print(f"\nTokenized into {len(result)} words")


class TestKeywordMemoryUsage:
    """Memory profiling for keyword operations."""

    @pytest.mark.fast
    def test_entity_dict_memory(self):
        """Profile memory usage of entity dictionary."""
        import tracemalloc

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Create entity dictionary (simulate extraction result)
        entities = {}
        for i in range(100):
            entities[f"Entity {i}"] = {
                'type': 'ORG',
                'query': f'entity {i} query',
                'images': [f'/path/image_{j}.jpg' for j in range(5)]
            }

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        print(f"\nEntity dict memory: {memory_used_mb:.2f} MB for 100 entities")

        # Should use less than 10MB
        assert memory_used_mb < 10

    @pytest.mark.fast
    def test_keyword_list_memory(self):
        """Profile memory usage of large keyword list."""
        import tracemalloc

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Create large keyword list
        keywords = [f"keyword {i} with some descriptive text" for i in range(10000)]

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        print(f"\nKeyword list memory: {memory_used_mb:.2f} MB for 10,000 keywords")

        # Should use less than 20MB
        assert memory_used_mb < 20


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
