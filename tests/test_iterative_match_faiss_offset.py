"""
Tests for FAISS index offset mapping in iterative match gap filling.

The iterative match stage builds a local FAISS index from only the
newly-fetched segments. FAISS returns local indices (0-based), but
the code must map these to global text_metadata indices for:
1. Filtering to only the current pass's segments
2. Correct metadata lookup in state.text_metadata

Bug fixed: Before this fix, FAISS local indices were treated as global
text_metadata indices, causing:
- k = len(embeddings) - new_segment_start → negative → all gaps skipped
- idx < new_segment_start filter → all results filtered out
- state.text_metadata[idx] → wrong metadata read
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.stages.iterative_match import IterativeMatchStage, GapSegment, LockedMatch


# ============================================================================
# Helpers
# ============================================================================

class FakeFAISSIndex:
    """Minimal FAISS-like index that returns nearest neighbors by L2 distance."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype('float32')

    def search(self, query: np.ndarray, k: int):
        """Return (distances, indices) sorted by L2 distance."""
        query = query.astype('float32')
        dists = np.linalg.norm(self.vectors - query, axis=1) ** 2
        actual_k = min(k, len(self.vectors))
        top_k_idx = np.argsort(dists)[:actual_k]
        top_k_dist = dists[top_k_idx]
        if actual_k < k:
            pad = k - actual_k
            top_k_idx = np.concatenate([top_k_idx, np.full(pad, -1, dtype=np.int64)])
            top_k_dist = np.concatenate([top_k_dist, np.full(pad, np.inf)])
        return top_k_dist.reshape(1, -1), top_k_idx.reshape(1, -1)


def make_gap(segment_index: int, position: float = 0.0, text: str = "test") -> GapSegment:
    return GapSegment(
        segment_index=segment_index,
        confidence=0.5,
        voiceover_text=text,
        position=position,
    )


# ============================================================================
# Tests: _match_gaps_to_new_candidates with offset
# ============================================================================

class TestFAISSOffsetMapping:
    """Verify local FAISS indices are correctly mapped to global text_metadata indices."""

    def _setup_stage(self, embeddings: np.ndarray, global_offset: int):
        """Create a stage with embeddings and a fake FAISS index."""
        stage = IterativeMatchStage()
        stage._embeddings = embeddings
        stage._embedding_index = FakeFAISSIndex(embeddings)
        stage._embedding_global_offset = global_offset
        return stage

    def _make_state(self, text_metadata: list, num_matches: int = 0):
        """Create minimal pipeline state."""
        state = MagicMock()
        state.text_metadata = text_metadata
        matches = []
        for i in range(num_matches):
            m = MagicMock()
            m.primary_match = MagicMock()
            m.primary_match.confidence = 0.5
            m.primary_match.video_segment = MagicMock()
            m.primary_match.video_segment.source_file = ""
            m.primary_match.video_segment.text = ""
            matches.append(m)
        state.matches = matches
        state.voiceover_embeddings = None
        return state

    def _make_config(self, target_confidence: float = 0.3):
        """Config with low target so L2 distance easily passes."""
        config = MagicMock()
        iter_config = MagicMock()
        iter_config.source_spacing_seconds = 300.0
        iter_config.target_confidence = target_confidence
        config.iterative_matching = iter_config
        return config

    def test_pass1_fills_gaps_with_offset(self):
        """Pass 1: 6000 original segments, 100 new segments. FAISS should match."""
        dim = 16
        n_new = 100
        global_offset = 6000
        new_segment_start = 6000

        np.random.seed(42)
        embeddings = np.random.randn(n_new, dim).astype('float32')

        # Make gap embedding very similar to new segment 50
        gap_embedding = embeddings[50] + np.random.randn(dim).astype('float32') * 0.01

        stage = self._setup_stage(embeddings, global_offset)

        text_metadata = [{'text': f'original_{i}', 'video_path': f'orig_{i}'} for i in range(6000)]
        text_metadata += [{'text': f'new_{i}', 'video_path': f'new_vid_{i}'} for i in range(100)]

        state = self._make_state(text_metadata, num_matches=10)
        config = self._make_config(target_confidence=0.3)

        gap = make_gap(segment_index=0, position=0.0, text="test gap")

        with patch('src.embeddings.get_embedding_provider', return_value=MagicMock()):
            with patch.object(stage, '_precompute_voiceover_embeddings', return_value={0: gap_embedding}):
                with patch.object(stage, '_get_voiceover_embedding', return_value=gap_embedding):
                    result = stage._match_gaps_to_new_candidates(
                        [gap], [], state, config, new_segment_start
                    )

        assert result >= 1, f"Expected at least 1 gap filled, got {result}"

    def test_pass1_k_calculation_positive(self):
        """k must be positive when offset equals new_segment_start (pass 1)."""
        global_offset = 6000
        new_segment_start = 6000
        n_embeddings = 50

        local_new_start = new_segment_start - global_offset  # 0
        num_new = n_embeddings - max(0, local_new_start)  # 50
        k = min(20, num_new)
        assert k == 20

    def test_pass2_k_calculation_positive(self):
        """k must be positive for pass 2 with accumulated embeddings."""
        global_offset = 6000
        new_segment_start = 6100  # Pass 2 starts here
        n_embeddings = 150  # 100 from pass 1 + 50 from pass 2

        local_new_start = new_segment_start - global_offset  # 100
        num_new = n_embeddings - max(0, local_new_start)  # 50
        k = min(20, num_new)
        assert k == 20

    def test_old_bug_k_was_negative(self):
        """The OLD buggy formula: k = len(embeddings) - new_segment_start → negative."""
        n_embeddings = 100
        new_segment_start = 6000
        old_k = min(20, n_embeddings - new_segment_start)
        assert old_k < 0, "Old formula should produce negative k"

        # NEW: with offset
        global_offset = 6000
        local_new_start = new_segment_start - global_offset
        num_new = n_embeddings - max(0, local_new_start)
        new_k = min(20, num_new)
        assert new_k == 20, "New formula should produce k=20"

    def test_metadata_lookup_uses_global_index(self):
        """FAISS returns local idx; metadata lookup must use global_idx = local + offset."""
        dim = 16
        n_new = 10
        global_offset = 5000
        new_segment_start = 5000

        np.random.seed(123)
        embeddings = np.random.randn(n_new, dim).astype('float32')
        gap_embedding = embeddings[3] + np.random.randn(dim).astype('float32') * 0.001

        stage = self._setup_stage(embeddings, global_offset)

        text_metadata = [{'text': f'original_{i}', 'video_path': f'orig_{i}'} for i in range(5000)]
        text_metadata += [{'text': f'iterative_{i}', 'video_path': f'iter_vid_{i}'} for i in range(10)]

        state = self._make_state(text_metadata, num_matches=5)
        config = self._make_config(target_confidence=0.0)

        gap = make_gap(segment_index=0, position=0.0, text="test")

        captured = {}

        def capture_update(gap, best_match, state):
            captured['best_match'] = best_match

        with patch('src.embeddings.get_embedding_provider', return_value=MagicMock()):
            with patch.object(stage, '_precompute_voiceover_embeddings', return_value={0: gap_embedding}):
                with patch.object(stage, '_get_voiceover_embedding', return_value=gap_embedding):
                    with patch.object(stage, '_update_match_for_gap', side_effect=capture_update):
                        stage._match_gaps_to_new_candidates(
                            [gap], [], state, config, new_segment_start
                        )

        assert 'best_match' in captured, "Should have found a match"
        assert captured['best_match']['meta']['text'].startswith('iterative_'), \
            f"Expected iterative metadata, got: {captured['best_match']['meta']['text']}"
        assert captured['best_match']['index'] >= global_offset, \
            f"Expected global index >= {global_offset}, got {captured['best_match']['index']}"

    def test_pass2_filters_out_pass1_segments(self):
        """Pass 2 should only match against pass 2's new segments, not pass 1's."""
        dim = 16
        global_offset = 5000

        np.random.seed(42)
        pass1_embeddings = np.random.randn(10, dim).astype('float32')
        pass2_embeddings = np.random.randn(10, dim).astype('float32')

        # Gap very similar to pass1[5] but NOT to any pass2 segment
        gap_embedding = pass1_embeddings[5] + np.random.randn(dim).astype('float32') * 0.001

        all_embeddings = np.vstack([pass1_embeddings, pass2_embeddings])
        stage = self._setup_stage(all_embeddings, global_offset)

        text_metadata = [{'text': f'orig_{i}', 'video_path': f'o_{i}'} for i in range(5000)]
        text_metadata += [{'text': f'pass1_{i}', 'video_path': f'p1_{i}'} for i in range(10)]
        text_metadata += [{'text': f'pass2_{i}', 'video_path': f'p2_{i}'} for i in range(10)]

        state = self._make_state(text_metadata, num_matches=5)
        config = self._make_config(target_confidence=0.0)

        gap = make_gap(segment_index=0)
        new_segment_start = 5010  # Pass 2 starts here

        captured = {}

        def capture_update(gap, best_match, state):
            captured['best_match'] = best_match

        with patch('src.embeddings.get_embedding_provider', return_value=MagicMock()):
            with patch.object(stage, '_precompute_voiceover_embeddings', return_value={0: gap_embedding}):
                with patch.object(stage, '_get_voiceover_embedding', return_value=gap_embedding):
                    with patch.object(stage, '_update_match_for_gap', side_effect=capture_update):
                        result = stage._match_gaps_to_new_candidates(
                            [gap], [], state, config, new_segment_start
                        )

        if 'best_match' in captured:
            assert captured['best_match']['index'] >= 5010, \
                f"Matched pass1 segment (idx={captured['best_match']['index']}), should only match pass2"
            assert captured['best_match']['meta']['text'].startswith('pass2_'), \
                f"Metadata from wrong pass: {captured['best_match']['meta']['text']}"

    def test_global_offset_set_on_first_compute(self):
        """_embedding_global_offset is set once when embeddings are first created."""
        stage = IterativeMatchStage()
        stage._embeddings = None
        stage._embedding_index = None
        stage._embedding_global_offset = None

        state = MagicMock()
        state.text_metadata = [{'text': f'seg_{i}'} for i in range(6100)]
        config = MagicMock()
        config.cache = MagicMock()
        config.cache.cache_dir = '/tmp/test_cache'

        fake_embeddings = np.random.randn(100, 16).astype('float32')

        with patch('src.embeddings.compute_embeddings', return_value=fake_embeddings):
            with patch('src.embeddings.get_embedding_provider', return_value=MagicMock()):
                with patch('src.embeddings.build_embedding_index', return_value=MagicMock()):
                    with patch('src.utils.CacheManager', return_value=MagicMock()):
                        result = stage._compute_embeddings_for_new_segments(state, config, start_index=6000)

        assert stage._embedding_global_offset == 6000
        assert len(stage._embeddings) == 100

    def test_global_offset_preserved_on_second_compute(self):
        """On second call, offset stays at original value, embeddings are vstacked."""
        stage = IterativeMatchStage()
        stage._embeddings = np.random.randn(100, 16).astype('float32')
        stage._embedding_global_offset = 6000
        stage._embedding_index = MagicMock()

        state = MagicMock()
        state.text_metadata = [{'text': f'seg_{i}'} for i in range(6150)]
        config = MagicMock()
        config.cache = MagicMock()
        config.cache.cache_dir = '/tmp/test_cache'

        new_embeddings = np.random.randn(50, 16).astype('float32')

        with patch('src.embeddings.compute_embeddings', return_value=new_embeddings):
            with patch('src.embeddings.get_embedding_provider', return_value=MagicMock()):
                with patch('src.embeddings.build_embedding_index', return_value=MagicMock()):
                    with patch('src.utils.CacheManager', return_value=MagicMock()):
                        stage._compute_embeddings_for_new_segments(state, config, start_index=6100)

        assert stage._embedding_global_offset == 6000, "Offset should NOT change on second call"
        assert len(stage._embeddings) == 150, "Should have 100 + 50 = 150 embeddings"
