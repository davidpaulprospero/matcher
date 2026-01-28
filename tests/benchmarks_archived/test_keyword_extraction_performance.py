"""
Performance benchmarks for keyword extraction.

Run with: pytest tests/benchmarks/test_keyword_extraction_performance.py -v
"""

import pytest
import time
import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestKeywordExtractionPerformance:
    """Benchmark keyword extraction operations."""

    @pytest.mark.fast
    def test_tfidf_extraction_speed(self, sample_segments, benchmark):
        """Benchmark TF-IDF keyword extraction speed."""
        from src.keyword_extractor import TFIDFKeywordExtractor

        extractor = TFIDFKeywordExtractor()

        def extract_keywords():
            return extractor.extract_keywords(
                segments=sample_segments,
                max_keywords=10
            )

        result = benchmark(extract_keywords)
        assert len(result) > 0
        print(f"\nTF-IDF extracted {len(result)} keywords from {len(sample_segments)} segments")

    @pytest.mark.fast
    def test_entity_extraction_speed(self, sample_text_corpus, benchmark):
        """Benchmark entity extraction speed."""
        from src.keyword_extractor import EntityExtractor

        extractor = EntityExtractor()

        def extract_entities():
            return extractor.extract_entities(sample_text_corpus)

        result = benchmark(extract_entities)
        assert isinstance(result, list)
        print(f"\nEntity extraction found {len(result)} entities")

    @pytest.mark.fast
    def test_topic_detection_speed(self, sample_text_corpus, benchmark):
        """Benchmark topic detection speed."""
        from src.topic_extraction import TopicDetector

        detector = TopicDetector()

        def detect_topics():
            return detector.detect_topics(sample_text_corpus)

        result = benchmark(detect_topics)
        assert isinstance(result, str)
        print(f"\nTopic detection generated {len(result)} characters")

    @pytest.mark.slow
    def test_llm_keyword_extraction_latency(self, sample_segments):
        """Measure LLM keyword extraction latency (not a benchmark, actual timing)."""
        pytest.skip("Requires API key and network - run manually")

        from src.keyword_extractor import LLMKeywordExtractor
        from src.config import load_config

        config = load_config()
        extractor = LLMKeywordExtractor(config.llm)

        start = time.time()
        result = extractor.extract_keywords(
            segments=sample_segments[:10],  # Only 10 segments
            max_keywords=10
        )
        elapsed = time.time() - start

        print(f"\nLLM extraction: {elapsed:.2f}s for 10 segments ({elapsed/10:.2f}s per segment)")
        assert len(result) > 0

    @pytest.mark.fast
    def test_keyword_remixing_speed(self, sample_keywords, benchmark):
        """Benchmark keyword remixing speed."""
        from src.keyword_remix import KeywordRemixer

        remixer = KeywordRemixer()

        def remix_keywords():
            return remixer.generate_alternatives(sample_keywords)

        result = benchmark(remix_keywords)
        assert len(result) > 0
        print(f"\nKeyword remixer generated {len(result)} alternatives from {len(sample_keywords)} keywords")

    @pytest.mark.fast
    def test_segment_processing_throughput(self, sample_segments, benchmark):
        """Benchmark segment processing throughput."""
        from src.keyword_extractor.segment_processor import SegmentProcessor

        processor = SegmentProcessor()

        def process_segments():
            return processor.process_segments(sample_segments)

        result = benchmark(process_segments)
        assert result is not None

        # Calculate throughput
        segments_per_second = len(sample_segments) / benchmark.stats['mean']
        print(f"\nProcessed {segments_per_second:.0f} segments/second")


class TestKeywordMemoryUsage:
    """Memory profiling for keyword extraction."""

    @pytest.mark.fast
    def test_tfidf_memory_usage(self, sample_segments):
        """Profile memory usage of TF-IDF extraction."""
        import tracemalloc
        from src.keyword_extractor import TFIDFKeywordExtractor

        extractor = TFIDFKeywordExtractor()

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        result = extractor.extract_keywords(
            segments=sample_segments,
            max_keywords=10
        )

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        print(f"\nTF-IDF memory usage: {memory_used_mb:.2f} MB for {len(sample_segments)} segments")

        # Should use less than 50MB for 100 segments
        assert memory_used_mb < 50

    @pytest.mark.fast
    def test_entity_extraction_memory_usage(self, sample_text_corpus):
        """Profile memory usage of entity extraction."""
        import tracemalloc
        from src.keyword_extractor import EntityExtractor

        extractor = EntityExtractor()

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        result = extractor.extract_entities(sample_text_corpus)

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        print(f"\nEntity extraction memory usage: {memory_used_mb:.2f} MB")

        # Should use less than 100MB
        assert memory_used_mb < 100


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
