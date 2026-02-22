"""
Performance benchmarks for matching scoring algorithms.

Run with: pytest tests/benchmarks/test_matching_scoring_performance.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.matching.scoring import calculate_adaptive_threshold
from src.utils import SRTSegment


class TestAdaptiveThresholdPerformance:
    """Benchmark adaptive threshold calculation."""

    @pytest.mark.fast
    def test_adaptive_threshold_short_vo(self, benchmark):
        """Benchmark adaptive threshold with short voiceover."""
        base_threshold = 0.7
        voiceover_text = "Hello world"  # Short text
        candidates = [
            (Mock(), 0.8),
            (Mock(), 0.75),
            (Mock(), 0.7),
            (Mock(), 0.65),
            (Mock(), 0.6),
        ]

        def compute_threshold():
            return calculate_adaptive_threshold(base_threshold, voiceover_text, candidates)

        result = benchmark(compute_threshold)
        print(f"\nAdaptive threshold: {result[0]:.2f}, reason: {result[1]}")

    @pytest.mark.fast
    def test_adaptive_threshold_long_vo(self, benchmark):
        """Benchmark adaptive threshold with long voiceover."""
        base_threshold = 0.7
        voiceover_text = "This is a much longer voiceover segment with detailed information about the topic at hand"
        candidates = [
            (Mock(), 0.8),
            (Mock(), 0.75),
            (Mock(), 0.7),
            (Mock(), 0.65),
            (Mock(), 0.6),
        ]

        def compute_threshold():
            return calculate_adaptive_threshold(base_threshold, voiceover_text, candidates)

        result = benchmark(compute_threshold)
        print(f"\nAdaptive threshold: {result[0]:.2f}, reason: {result[1]}")

    @pytest.mark.fast
    def test_adaptive_threshold_large_pool(self, benchmark):
        """Benchmark adaptive threshold with large candidate pool."""
        base_threshold = 0.7
        voiceover_text = "This is a test segment with some content"
        candidates = [
            (Mock(), 0.9 - i * 0.02) for i in range(20)
        ]

        def compute_threshold():
            return calculate_adaptive_threshold(base_threshold, voiceover_text, candidates)

        result = benchmark(compute_threshold)
        print(f"\nAdaptive threshold: {result[0]:.2f}, reason: {result[1]}")

    @pytest.mark.fast
    def test_adaptive_threshold_low_variance(self, benchmark):
        """Benchmark adaptive threshold with low variance candidates."""
        base_threshold = 0.7
        voiceover_text = "This is a test segment"
        # Low variance: candidates are very similar
        candidates = [
            (Mock(), 0.75 - i * 0.001) for i in range(10)
        ]

        def compute_threshold():
            return calculate_adaptive_threshold(base_threshold, voiceover_text, candidates)

        result = benchmark(compute_threshold)
        print(f"\nAdaptive threshold: {result[0]:.2f}, reason: {result[1]}")


class TestDurationPenaltyPerformance:
    """Benchmark duration penalty calculations."""

    @pytest.mark.fast
    def test_duration_penalty_computation(self, benchmark):
        """Benchmark duration penalty for mismatched segments."""
        from src.matching.scoring import apply_duration_penalty

        # Create mock config with proper numeric values
        mock_config = Mock()
        mock_config.matching = Mock()
        mock_config.matching.duration_penalty_factor = 0.2  # Must be a numeric value
        mock_config.matching.scoring = Mock()
        mock_config.matching.scoring.duration_ratio_reward_threshold = 0.1
        mock_config.matching.scoring.duration_ratio_reward_boost = 0.02

        confidence = 0.8
        speed_ratio = 2.5  # Video is 2.5x longer than voiceover

        def compute_penalty():
            return apply_duration_penalty(confidence, speed_ratio, mock_config)

        result = benchmark(compute_penalty)
        print(f"\nAfter duration penalty: {result:.3f}")


class TestConfidenceCalculationPerformance:
    """Benchmark overall confidence calculation."""

    @pytest.mark.fast
    def test_confidence_with_multiple_factors(self, benchmark):
        """Benchmark confidence calculation with multiple adjustment factors."""
        from src.matching.scoring import calculate_adaptive_threshold

        base_threshold = 0.7
        voiceover_text = "The new artificial intelligence system uses deep neural networks"
        candidates = [
            (Mock(), 0.85),
            (Mock(), 0.82),
            (Mock(), 0.80),
            (Mock(), 0.78),
            (Mock(), 0.75),
            (Mock(), 0.72),
            (Mock(), 0.70),
            (Mock(), 0.68),
            (Mock(), 0.65),
            (Mock(), 0.60),
        ]

        def compute():
            return calculate_adaptive_threshold(base_threshold, voiceover_text, candidates)

        result = benchmark(compute)
        print(f"\nFinal threshold: {result[0]:.3f}")


class TestCandidateRankingPerformance:
    """Benchmark candidate ranking operations."""

    @pytest.mark.fast
    def test_rank_candidates_by_score(self, benchmark):
        """Benchmark ranking candidates by similarity score."""
        # Create 1000 candidates with random scores
        candidates = [
            (Mock(index=i), np.random.random()) for i in range(1000)
        ]

        def rank_candidates():
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            return sorted_candidates[:50]  # Top 50

        result = benchmark(rank_candidates)
        assert len(result) == 50
        print(f"\nRanked 1000 candidates, selected top 50")

    @pytest.mark.fast
    def test_filter_candidates_by_threshold(self, benchmark):
        """Benchmark filtering candidates by confidence threshold."""
        candidates = [
            (Mock(index=i), np.random.random() * 0.5 + 0.3) for i in range(1000)
        ]
        threshold = 0.7

        def filter_candidates():
            return [(seg, score) for seg, score in candidates if score >= threshold]

        result = benchmark(filter_candidates)
        print(f"\nFiltered {len(result)} candidates above threshold {threshold}")

    @pytest.mark.fast
    def test_deduplicate_by_source(self, benchmark):
        """Benchmark deduplication by video source."""
        candidates = []
        for i in range(500):
            video_id = f"video_{i // 5}"  # 5 clips per video
            segment = Mock(index=i, video_id=video_id)
            candidates.append((segment, 0.9 - i * 0.001))

        def deduplicate():
            source_counts = {}
            result = []
            for seg, score in candidates:
                vid = seg.video_id
                count = source_counts.get(vid, 0)
                if count < 3:  # Max 3 per source
                    result.append((seg, score))
                    source_counts[vid] = count + 1
                if len(result) >= 100:
                    break
            return result

        result = benchmark(deduplicate)
        print(f"\nDeduplicated to {len(result)} candidates")


class TestScoringIntegrationPerformance:
    """End-to-end scoring performance tests."""

    @pytest.mark.fast
    def test_full_scoring_pipeline(self, benchmark):
        """Benchmark full scoring pipeline for 100 segments."""
        from src.matching.scoring import calculate_adaptive_threshold

        # Simulate 100 voiceover segments
        voiceover_segments = [
            SRTSegment(i, i*5.0, (i+1)*5.0, f"Voiceover segment {i}", "vo.srt")
            for i in range(100)
        ]

        def full_pipeline():
            results = []
            for vo_seg in voiceover_segments:
                # Create random candidates
                candidates = [
                    (Mock(), np.random.random() * 0.3 + 0.6) for _ in range(20)
                ]
                threshold, _ = calculate_adaptive_threshold(0.7, vo_seg.text, candidates)
                results.append(threshold)
            return results

        result = benchmark(full_pipeline)
        assert len(result) == 100
        print(f"\nProcessed 100 segments, average threshold: {sum(result)/len(result):.3f}")

    @pytest.mark.fast
    def test_batch_threshold_calculation(self, benchmark):
        """Benchmark batch threshold calculation."""
        from src.matching.scoring import calculate_adaptive_threshold

        base_threshold = 0.7
        # Create 500 candidate pools
        candidate_pools = [
            [
                (Mock(), 0.9 - i * 0.02) for i in range(20)
            ] for _ in range(500)
        ]

        def batch_thresholds():
            return [
                calculate_adaptive_threshold(base_threshold, f"Segment {i}", pool)
                for i, pool in enumerate(candidate_pools)
            ]

        result = benchmark(batch_thresholds)
        assert len(result) == 500
        print(f"\nCalculated {len(result)} thresholds")


class TestScoringMemoryUsage:
    """Memory profiling for scoring operations."""

    @pytest.mark.fast
    def test_large_candidate_pool_memory(self):
        """Profile memory usage with large candidate pools."""
        import tracemalloc

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Create 10000 candidates
        candidates = [
            (Mock(index=i, text=f"Segment {i}"), 0.9 - i * 0.0001)
            for i in range(10000)
        ]

        # Sort and filter
        sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)
        filtered = [(s, sc) for s, sc in sorted_cands if sc > 0.5]

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        print(f"\nMemory for 10k candidates: {memory_used_mb:.2f} MB")

        # Should be reasonable
        assert memory_used_mb < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
