"""
Performance benchmarks for matching algorithms.

Run with: pytest tests/benchmarks/test_matching_performance.py -v
"""

import pytest
import time
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class MockSRTSegment:
    """Mock SRT segment for matching benchmarks."""
    index: int
    start_time: float
    end_time: float
    text: str
    embedding: np.ndarray = None


@dataclass
class MockTranscript:
    """Mock transcript for matching benchmarks."""
    text: str
    video_id: str
    start_time: float
    end_time: float
    embedding: np.ndarray = None


class TestMatchingPerformance:
    """Benchmark matching algorithm operations."""

    @pytest.fixture
    def mock_voiceover_segments(self) -> List[MockSRTSegment]:
        """Create mock voiceover segments with embeddings."""
        segments = []
        for i in range(100):
            segments.append(MockSRTSegment(
                index=i,
                start_time=i * 5.0,
                end_time=(i + 1) * 5.0,
                text=f"Voiceover segment {i} about technology",
                embedding=np.random.randn(1024).astype(np.float32)
            ))
        return segments

    @pytest.fixture
    def mock_video_transcripts(self) -> List[MockTranscript]:
        """Create mock video transcripts with embeddings."""
        transcripts = []
        for i in range(1000):
            transcripts.append(MockTranscript(
                text=f"Video transcript {i} discussing innovation",
                video_id=f"video_{i:04d}",
                start_time=0.0,
                end_time=60.0,
                embedding=np.random.randn(1024).astype(np.float32)
            ))
        return transcripts

    @pytest.mark.fast
    def test_embedding_matching_speed(self, mock_voiceover_segments, mock_video_transcripts, benchmark):
        """Benchmark embedding-based matching speed."""
        from src.matching import EmbeddingMatcher

        matcher = EmbeddingMatcher()

        def match_segments():
            return matcher.match(
                voiceover_segments=mock_voiceover_segments[:10],  # 10 segments
                video_transcripts=mock_video_transcripts
            )

        result = benchmark(match_segments)

        assert len(result) > 0
        print(f"\nMatched {len(result)} segments against {len(mock_video_transcripts)} transcripts")

        # Calculate throughput
        matches_per_second = len(mock_voiceover_segments[:10]) / benchmark.stats['mean']
        print(f"Matching throughput: {matches_per_second:.1f} segments/second")

    @pytest.mark.fast
    def test_scoring_algorithm_speed(self, mock_voiceover_segments, mock_video_transcripts, benchmark):
        """Benchmark scoring algorithm speed."""
        from src.matching.scoring import calculate_match_score

        voiceover = mock_voiceover_segments[0]
        candidates = mock_video_transcripts[:100]

        def score_candidates():
            scores = []
            for candidate in candidates:
                score = calculate_match_score(
                    voiceover_embedding=voiceover.embedding,
                    candidate_embedding=candidate.embedding,
                    text_similarity=0.8,
                    duration_match=1.0
                )
                scores.append(score)
            return scores

        result = benchmark(score_candidates)

        assert len(result) == 100
        print(f"\nScored {len(result)} candidates")

        # Calculate throughput
        scores_per_second = len(result) / benchmark.stats['mean']
        print(f"Scoring throughput: {scores_per_second:.0f} candidates/second")

    @pytest.mark.fast
    def test_diversity_filtering_speed(self, mock_video_transcripts, benchmark):
        """Benchmark diversity filtering speed."""
        from src.matching.strategies import DiversityStrategy

        strategy = DiversityStrategy()

        # Mock match results with source diversity
        matches = []
        for i, transcript in enumerate(mock_video_transcripts[:100]):
            matches.append({
                'transcript': transcript,
                'score': 0.9 - (i * 0.001),
                'source': f"channel_{i % 10}"  # 10 different sources
            })

        def apply_diversity():
            return strategy.filter_diverse_matches(matches, max_results=20)

        result = benchmark(apply_diversity)

        assert len(result) <= 20
        print(f"\nFiltered {len(matches)} matches to {len(result)} diverse results")

    @pytest.mark.fast
    def test_location_filtering_speed(self, mock_video_transcripts, benchmark):
        """Benchmark location-based filtering speed."""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        # Add location metadata to transcripts
        for i, transcript in enumerate(mock_video_transcripts):
            transcript.location = {"city": "Paris", "country": "France"} if i % 2 == 0 else {"city": "London", "country": "UK"}

        target_location = {"city": "Paris", "country": "France"}

        def filter_by_location():
            return matcher.filter_by_location(
                candidates=mock_video_transcripts,
                target_location=target_location,
                threshold="city"
            )

        result = benchmark(filter_by_location)

        assert len(result) > 0
        print(f"\nFiltered {len(mock_video_transcripts)} candidates to {len(result)} location matches")

    @pytest.mark.slow
    def test_full_matching_pipeline_latency(self, mock_voiceover_segments, mock_video_transcripts):
        """Measure end-to-end matching pipeline latency."""
        from src.matching import TieredMatcher
        from src.config import load_config

        config = load_config()
        matcher = TieredMatcher(config)

        start = time.time()
        results = matcher.match_all(
            voiceover_segments=mock_voiceover_segments[:10],
            video_transcripts=mock_video_transcripts[:100]  # Smaller set for speed
        )
        elapsed = time.time() - start

        print(f"\nFull pipeline: {elapsed:.2f}s for 10 segments against 100 videos")
        print(f"Average: {elapsed/10:.2f}s per segment")
        print(f"Matched {len(results)} segments")

        assert len(results) > 0


class TestMatchingMemoryUsage:
    """Memory profiling for matching operations."""

    @pytest.mark.fast
    def test_embedding_matching_memory(self):
        """Profile memory usage of embedding-based matching."""
        import tracemalloc

        # Create embeddings for 100 segments and 1000 candidates
        voiceover_embeddings = np.random.randn(100, 1024).astype(np.float32)
        candidate_embeddings = np.random.randn(1000, 1024).astype(np.float32)

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Compute all similarities
        from src.embeddings import compute_similarity
        for vo_emb in voiceover_embeddings:
            similarities = compute_similarity(vo_emb, candidate_embeddings)

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        print(f"\nMatching memory usage: {memory_used_mb:.2f} MB")
        print(f"For 100 segments × 1000 candidates = 100,000 comparisons")

        # Should use less than 200MB
        assert memory_used_mb < 200

    @pytest.mark.fast
    def test_match_result_storage_memory(self):
        """Profile memory usage of match result storage."""
        import tracemalloc
        from src.state import Match

        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]

        # Create 1000 match results
        matches = []
        for i in range(1000):
            matches.append(Match(
                voiceover_index=i % 100,
                video_id=f"video_{i:04d}",
                start_time=float(i * 5),
                end_time=float((i + 1) * 5),
                score=0.85,
                text="Sample matched text segment",
                confidence="high"
            ))

        peak_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        memory_used_mb = (peak_memory - initial_memory) / 1024 / 1024
        memory_per_match_kb = (peak_memory - initial_memory) / 1024 / len(matches)

        print(f"\nMatch storage memory: {memory_used_mb:.2f} MB for 1000 matches")
        print(f"Memory per match: {memory_per_match_kb:.2f} KB")

        # Should use less than 50MB for 1000 matches
        assert memory_used_mb < 50


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])
