"""
Comprehensive unit tests for the matching module.

Tests matching algorithms, scoring, strategies, and filters in isolation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockSRTSegment:
    """Mock voiceover segment."""
    index: int
    start_time: float
    end_time: float
    text: str
    embedding: np.ndarray = None


@dataclass
class MockTranscript:
    """Mock video transcript."""
    text: str
    video_id: str
    start_time: float
    end_time: float
    embedding: np.ndarray = None
    source: str = "youtube"
    location: dict = None
    is_broll: bool = False


@pytest.fixture
def sample_voiceover():
    """Create sample voiceover segment."""
    return MockSRTSegment(
        index=0,
        start_time=0.0,
        end_time=5.0,
        text="This is a test voiceover about artificial intelligence",
        embedding=np.random.randn(1024).astype(np.float32)
    )


@pytest.fixture
def sample_candidates():
    """Create sample candidate transcripts."""
    candidates = []
    for i in range(20):
        candidates.append(MockTranscript(
            text=f"Video {i} discusses technology and innovation",
            video_id=f"video_{i:03d}",
            start_time=0.0,
            end_time=10.0,
            embedding=np.random.randn(1024).astype(np.float32),
            source=f"channel_{i % 5}",  # 5 different sources
            location={"city": "Paris" if i % 2 == 0 else "London", "country": "France" if i % 2 == 0 else "UK"}
        ))
    return candidates


# =============================================================================
# Scoring Algorithm Tests
# =============================================================================

class TestScoringAlgorithms:
    """Test scoring algorithms."""

    def test_calculate_match_score_basic(self):
        """Test basic match score calculation."""
        from src.matching.scoring import calculate_match_score

        score = calculate_match_score(
            voiceover_embedding=np.array([1.0, 0.0, 0.0]),
            candidate_embedding=np.array([0.9, 0.1, 0.0]),
            text_similarity=0.8,
            duration_match=1.0
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be reasonably high for similar embeddings

    def test_calculate_match_score_perfect_match(self):
        """Test score for perfect match."""
        from src.matching.scoring import calculate_match_score

        embedding = np.random.randn(1024).astype(np.float32)

        score = calculate_match_score(
            voiceover_embedding=embedding,
            candidate_embedding=embedding,  # Identical
            text_similarity=1.0,
            duration_match=1.0
        )

        assert score >= 0.95  # Near-perfect score

    def test_calculate_match_score_poor_match(self):
        """Test score for poor match."""
        from src.matching.scoring import calculate_match_score

        # Orthogonal embeddings (very different)
        voiceover_emb = np.zeros(1024)
        voiceover_emb[0] = 1.0

        candidate_emb = np.zeros(1024)
        candidate_emb[512] = 1.0

        score = calculate_match_score(
            voiceover_embedding=voiceover_emb,
            candidate_embedding=candidate_emb,
            text_similarity=0.2,
            duration_match=0.5
        )

        assert score < 0.5  # Low score for poor match

    def test_score_weighting(self):
        """Test that different components are weighted correctly."""
        from src.matching.scoring import calculate_match_score

        embedding = np.random.randn(1024).astype(np.float32)

        # High embedding similarity, low text similarity
        score1 = calculate_match_score(
            voiceover_embedding=embedding,
            candidate_embedding=embedding * 0.95,
            text_similarity=0.3,
            duration_match=1.0
        )

        # Low embedding similarity, high text similarity
        score2 = calculate_match_score(
            voiceover_embedding=embedding,
            candidate_embedding=-embedding * 0.5,  # Opposite direction
            text_similarity=0.9,
            duration_match=1.0
        )

        # Embedding similarity should have more weight
        assert score1 > score2

    def test_duration_penalty(self):
        """Test duration mismatch penalty."""
        from src.matching.scoring import calculate_match_score

        embedding = np.random.randn(1024).astype(np.float32)

        # Perfect duration match
        score1 = calculate_match_score(
            voiceover_embedding=embedding,
            candidate_embedding=embedding,
            text_similarity=0.8,
            duration_match=1.0
        )

        # Poor duration match
        score2 = calculate_match_score(
            voiceover_embedding=embedding,
            candidate_embedding=embedding,
            text_similarity=0.8,
            duration_match=0.2  # Very different duration
        )

        assert score1 > score2


# =============================================================================
# LLM Matcher Tests
# =============================================================================

class TestLLMMatchers:
    """Test LLM-based matching providers."""

    @patch('src.matching.llm_providers.create_client')
    def test_gemini_matcher_initialization(self, mock_create_client):
        """Test Gemini matcher initialization."""
        from src.matching.llm_providers import GeminiMatcher

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key", model="gemini-2.0-flash")

        mock_create_client.assert_called_once()
        assert matcher.client == mock_client

    @patch('src.matching.llm_providers.create_client')
    def test_anthropic_matcher_initialization(self, mock_create_client):
        """Test Anthropic matcher initialization."""
        from src.matching.llm_providers import AnthropicMatcher

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        matcher = AnthropicMatcher(api_key="test_key", model="claude-3-5-sonnet-20241022")

        mock_create_client.assert_called_once()
        assert matcher.client == mock_client

    @patch('src.matching.llm_providers.create_client')
    def test_ollama_matcher_initialization(self, mock_create_client):
        """Test Ollama matcher initialization."""
        from src.matching.llm_providers import OllamaMatcher

        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        matcher = OllamaMatcher(model="llama2")

        mock_create_client.assert_called_once()
        assert matcher.client == mock_client

    @patch('src.matching.llm_providers.create_client')
    def test_llm_matcher_reranking(self, mock_create_client, sample_voiceover, sample_candidates):
        """Test LLM-based reranking."""
        from src.matching.llm_providers import GeminiMatcher

        # Mock LLM response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed_data = {
            "rankings": [
                {"video_id": "video_000", "score": 0.95, "reasoning": "Perfect match"},
                {"video_id": "video_001", "score": 0.85, "reasoning": "Good match"},
                {"video_id": "video_002", "score": 0.75, "reasoning": "Decent match"}
            ]
        }
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")

        # Rerank top 3 candidates
        reranked = matcher.rerank_candidates(
            voiceover=sample_voiceover,
            candidates=sample_candidates[:3]
        )

        assert len(reranked) == 3
        assert reranked[0][0].video_id == "video_000"
        assert reranked[0][1] == 0.95

    @patch('src.matching.llm_providers.create_client')
    def test_llm_matcher_error_handling(self, mock_create_client, sample_voiceover, sample_candidates):
        """Test LLM matcher error handling."""
        from src.matching.llm_providers import GeminiMatcher

        # Mock API failure
        mock_client = MagicMock()
        mock_client.generate.side_effect = Exception("API Error")
        mock_create_client.return_value = mock_client

        matcher = GeminiMatcher(api_key="test_key")

        # Should not crash, return original candidates
        reranked = matcher.rerank_candidates(
            voiceover=sample_voiceover,
            candidates=sample_candidates[:3]
        )

        # Should return same candidates (fallback behavior)
        assert len(reranked) == 3


# =============================================================================
# Location Filtering Tests
# =============================================================================

class TestLocationFiltering:
    """Test location-based filtering."""

    def test_location_filter_exact_city_match(self, sample_candidates):
        """Test filtering by exact city match."""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        filtered = matcher.filter_by_location(
            candidates=sample_candidates,
            target_location={"city": "Paris", "country": "France"},
            threshold="city"
        )

        # Should only return Paris candidates
        assert all(c.location["city"] == "Paris" for c in filtered)
        assert len(filtered) == 10  # Half the candidates

    def test_location_filter_country_match(self, sample_candidates):
        """Test filtering by country match."""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        filtered = matcher.filter_by_location(
            candidates=sample_candidates,
            target_location={"city": "Paris", "country": "France"},
            threshold="country"
        )

        # Should return all French candidates (including Paris)
        assert all(c.location["country"] == "France" for c in filtered)

    def test_location_filter_no_match(self, sample_candidates):
        """Test location filtering with no matches."""
        from src.matching.location_matching import LocationMatcher

        matcher = LocationMatcher()

        filtered = matcher.filter_by_location(
            candidates=sample_candidates,
            target_location={"city": "Tokyo", "country": "Japan"},
            threshold="city"
        )

        # Should return empty if strict, or all if lenient
        assert isinstance(filtered, list)

    def test_location_filter_missing_data(self, sample_candidates):
        """Test location filtering with missing location data."""
        from src.matching.location_matching import LocationMatcher

        # Remove location from some candidates
        for i in range(5):
            sample_candidates[i].location = None

        matcher = LocationMatcher()

        filtered = matcher.filter_by_location(
            candidates=sample_candidates,
            target_location={"city": "Paris", "country": "France"},
            threshold="city"
        )

        # Should handle missing data gracefully
        assert isinstance(filtered, list)


# =============================================================================
# Diversity Strategy Tests
# =============================================================================

class TestDiversityStrategies:
    """Test diversity filtering strategies."""

    def test_source_diversity_filtering(self, sample_candidates):
        """Test filtering for source diversity."""
        from src.matching.strategies import DiversityStrategy

        strategy = DiversityStrategy()

        # Create matches with scores
        matches = [(c, 0.9 - i * 0.01) for i, c in enumerate(sample_candidates)]

        diverse_matches = strategy.filter_diverse_matches(
            matches=matches,
            max_results=10,
            require_different_source=True
        )

        # Should have at most one match per source
        sources = [m[0].source for m in diverse_matches]
        assert len(sources) == len(set(sources))  # All unique

    def test_diversity_respects_score_order(self, sample_candidates):
        """Test that diversity filtering respects score ordering."""
        from src.matching.strategies import DiversityStrategy

        strategy = DiversityStrategy()

        # Create matches with scores
        matches = [(c, 0.9 - i * 0.01) for i, c in enumerate(sample_candidates)]

        diverse_matches = strategy.filter_diverse_matches(
            matches=matches,
            max_results=5,
            require_different_source=True
        )

        # Should select highest-scoring match from each source
        scores = [m[1] for m in diverse_matches]
        assert scores == sorted(scores, reverse=True)  # Descending order

    def test_embedding_diversity_strategy(self, sample_candidates):
        """Test embedding-based diversity strategy."""
        from src.matching.strategies import EmbeddingDiversityStrategy

        strategy = EmbeddingDiversityStrategy()

        # Create matches with embeddings
        matches = [(c, 0.85) for c in sample_candidates[:10]]

        diverse_matches = strategy.select_diverse_by_embedding(
            matches=matches,
            max_results=5,
            diversity_threshold=0.7
        )

        assert len(diverse_matches) <= 5
        assert len(diverse_matches) > 0

    def test_broll_only_strategy(self):
        """Test B-roll only filtering strategy."""
        from src.matching.strategies import BRollOnlyStrategy

        strategy = BRollOnlyStrategy()

        # Create mix of B-roll and non-B-roll candidates
        candidates = []
        for i in range(10):
            candidates.append(MockTranscript(
                text=f"Video {i}",
                video_id=f"video_{i}",
                start_time=0.0,
                end_time=10.0,
                is_broll=(i % 2 == 0)  # Every other one is B-roll
            ))

        matches = [(c, 0.85) for c in candidates]

        broll_matches = strategy.filter_broll_only(matches)

        # Should only return B-roll matches
        assert all(m[0].is_broll for m in broll_matches)
        assert len(broll_matches) == 5


# =============================================================================
# Integration Tests
# =============================================================================

class TestMatchingIntegration:
    """Integration tests for complete matching workflow."""

    def test_end_to_end_matching(self, sample_voiceover, sample_candidates):
        """Test complete matching workflow."""
        from src.matching import EmbeddingMatcher

        matcher = EmbeddingMatcher()

        matches = matcher.match(
            voiceover_segments=[sample_voiceover],
            video_transcripts=sample_candidates
        )

        assert len(matches) > 0
        assert matches[0].score > 0.0

    def test_matching_with_strategies(self, sample_voiceover, sample_candidates):
        """Test matching with different strategies applied."""
        from src.matching import EmbeddingMatcher
        from src.matching.strategies import DiversityStrategy

        matcher = EmbeddingMatcher()
        diversity_strategy = DiversityStrategy()

        # Get initial matches
        matches = matcher.match(
            voiceover_segments=[sample_voiceover],
            video_transcripts=sample_candidates
        )

        # Apply diversity filtering
        diverse_matches = diversity_strategy.filter_diverse_matches(
            matches=[(m, m.score) for m in matches],
            max_results=5,
            require_different_source=True
        )

        assert len(diverse_matches) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
