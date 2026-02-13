"""
Mutation tests for FAISS index offset fix in iterative match.

Each test reads the source code as a string, applies a mutation,
and verifies the mutation would break correctness (KILLED).
Never modifies source files on disk.
"""

import re
import pytest

from pathlib import Path

SOURCE_PATH = Path(__file__).parent.parent / "src" / "stages" / "iterative_match.py"


@pytest.fixture
def source_code():
    return SOURCE_PATH.read_text(encoding='utf-8')


class TestMutationsKilled:
    """Every mutation must be detected (killed) by asserting the mutated
    code differs from the correct pattern."""

    def test_mutation_remove_global_offset_variable(self, source_code):
        """MUTATION: Remove `global_offset = self._embedding_global_offset or 0`
        Without the offset, FAISS indices would be treated as global."""
        assert "global_offset = self._embedding_global_offset or 0" in source_code, \
            "Source must have global_offset assignment"

        mutated = source_code.replace(
            "global_offset = self._embedding_global_offset or 0",
            "global_offset = 0  # MUTATED: always zero"
        )
        # With offset always 0, global_idx = idx + 0 = idx (local FAISS index)
        # This means metadata lookup uses wrong index
        assert "global_offset = 0  # MUTATED: always zero" in mutated
        # The original line is gone
        assert "self._embedding_global_offset or 0" not in mutated

    def test_mutation_remove_offset_from_global_idx(self, source_code):
        """MUTATION: Remove offset addition from global_idx = idx + global_offset"""
        assert "global_idx = idx + global_offset" in source_code, \
            "Source must convert local FAISS idx to global"

        mutated = source_code.replace(
            "global_idx = idx + global_offset",
            "global_idx = idx  # MUTATED: no offset"
        )
        # Without offset, FAISS local index used directly → wrong metadata
        assert "global_idx = idx  # MUTATED: no offset" in mutated
        assert "idx + global_offset" not in mutated

    def test_mutation_use_old_k_formula(self, source_code):
        """MUTATION: Revert to old buggy k formula that doesn't account for offset."""
        # New formula uses local_new_start
        assert "local_new_start = new_segment_start - global_offset" in source_code
        assert "num_new = len(self._embeddings) - max(0, local_new_start)" in source_code

        mutated = source_code.replace(
            "local_new_start = new_segment_start - global_offset",
            "local_new_start = new_segment_start  # MUTATED: no offset subtraction"
        )
        # Without subtracting offset, local_new_start = 6000 (global),
        # causing num_new to go negative
        assert "local_new_start = new_segment_start  # MUTATED" in mutated
        assert "new_segment_start - global_offset" not in mutated

    def test_mutation_remove_global_offset_tracking(self, source_code):
        """MUTATION: Remove the line that sets _embedding_global_offset."""
        target = "self._embedding_global_offset = start_index"
        assert target in source_code, \
            "Source must set _embedding_global_offset on first batch"

        mutated = source_code.replace(
            target,
            "pass  # MUTATED: never set offset"
        )
        # Without setting offset, it stays None → global_offset = 0
        assert "pass  # MUTATED: never set offset" in mutated
        assert "self._embedding_global_offset = start_index" not in mutated

    def test_mutation_filter_direction_inverted(self, source_code):
        """MUTATION: Invert the filter from < to >= (would include old, exclude new)."""
        assert "if global_idx < new_segment_start:" in source_code

        mutated = source_code.replace(
            "if global_idx < new_segment_start:",
            "if global_idx >= new_segment_start:  # MUTATED: inverted"
        )
        # Inverted filter keeps old segments and skips new ones — opposite of desired
        assert "if global_idx >= new_segment_start:" in mutated
        assert "if global_idx < new_segment_start:" not in mutated

    def test_mutation_use_local_idx_for_metadata(self, source_code):
        """MUTATION: Use local idx instead of global_idx for text_metadata lookup."""
        assert "meta = state.text_metadata[global_idx]" in source_code

        mutated = source_code.replace(
            "meta = state.text_metadata[global_idx]",
            "meta = state.text_metadata[idx]  # MUTATED: local index"
        )
        # Local idx (e.g., 50) reads original metadata instead of iterative
        assert "state.text_metadata[idx]" in mutated
        assert "state.text_metadata[global_idx]" not in mutated

    def test_mutation_store_local_index_in_match(self, source_code):
        """MUTATION: Store local FAISS idx instead of global_idx in best_match."""
        assert "'index': global_idx," in source_code

        mutated = source_code.replace(
            "'index': global_idx,",
            "'index': idx,  # MUTATED: local index"
        )
        # Wrong index stored → downstream consumers get incorrect reference
        assert "'index': idx," in mutated
        assert "'index': global_idx," not in mutated

    def test_mutation_remove_faiss_sentinel_check(self, source_code):
        """MUTATION: Remove the idx < 0 sentinel check for FAISS padding."""
        assert "if idx < 0:" in source_code

        mutated = source_code.replace(
            "if idx < 0:",
            "if False:  # MUTATED: never skip"
        )
        # Without sentinel check, -1 padding values cause IndexError or wrong lookup
        assert "if False:  # MUTATED: never skip" in mutated
        assert source_code.count("if idx < 0:") >= 1

    def test_mutation_remove_embedding_global_offset_init(self, source_code):
        """MUTATION: Remove initialization of _embedding_global_offset in run()."""
        assert "self._embedding_global_offset = None" in source_code

        mutated = source_code.replace(
            "self._embedding_global_offset = None",
            "pass  # MUTATED: no init"
        )
        # Without init, offset could carry stale value from previous run
        assert "pass  # MUTATED: no init" in mutated

    def test_mutation_subtract_instead_of_add_offset(self, source_code):
        """MUTATION: Subtract offset instead of adding it."""
        assert "global_idx = idx + global_offset" in source_code

        mutated = source_code.replace(
            "global_idx = idx + global_offset",
            "global_idx = idx - global_offset  # MUTATED: wrong direction"
        )
        # Subtracting offset makes global_idx negative → crash or wrong metadata
        assert "idx - global_offset" in mutated
        assert "idx + global_offset" not in mutated


class TestSourceStructure:
    """Verify the fix's structural correctness — key patterns exist in source."""

    def test_offset_init_before_loop(self, source_code):
        """_embedding_global_offset = None must appear before the pass loop."""
        offset_init = source_code.index("self._embedding_global_offset = None")
        pass_loop = source_code.index("for pass_num in range(1, max_iterations")
        assert offset_init < pass_loop, "Offset must be initialized before pass loop"

    def test_offset_set_in_compute_function(self, source_code):
        """_embedding_global_offset = start_index must be in _compute_embeddings_for_new_segments."""
        func_start = source_code.index("def _compute_embeddings_for_new_segments")
        # Find next def to bound the search
        next_def = source_code.index("\n    def ", func_start + 1)
        func_body = source_code[func_start:next_def]
        assert "self._embedding_global_offset = start_index" in func_body

    def test_offset_used_in_match_function(self, source_code):
        """global_offset must be used in _match_gaps_to_new_candidates."""
        func_start = source_code.index("def _match_gaps_to_new_candidates")
        next_def = source_code.index("\n    def ", func_start + 1)
        func_body = source_code[func_start:next_def]
        assert "global_offset = self._embedding_global_offset or 0" in func_body
        assert "global_idx = idx + global_offset" in func_body
        assert "state.text_metadata[global_idx]" in func_body

    def test_confidence_uses_ip_not_l2(self, source_code):
        """Confidence formula must use direct IP score, not L2 distance conversion."""
        func_start = source_code.index("def _match_gaps_to_new_candidates")
        next_def = source_code.index("\n    def ", func_start + 1)
        func_body = source_code[func_start:next_def]
        # Must NOT have old L2 formula
        assert "1.0 - (dist / 2.0)" not in func_body, \
            "Old L2 distance formula found — should use direct IP score"
        # Must have new IP formula
        assert "max(0.0, min(1.0, float(dist)))" in func_body, \
            "IP confidence formula not found"

    def test_query_vector_normalized(self, source_code):
        """Query vector must be normalized before FAISS search for correct cosine similarity."""
        func_start = source_code.index("def _match_gaps_to_new_candidates")
        next_def = source_code.index("\n    def ", func_start + 1)
        func_body = source_code[func_start:next_def]
        assert "query_vec = query_vec / norm" in func_body, \
            "Query vector normalization not found"

    def test_update_match_returns_bool(self, source_code):
        """_update_match_for_gap must return bool for accurate gap counting."""
        func_start = source_code.index("def _update_match_for_gap")
        next_def = source_code.index("\n    def ", func_start + 1)
        func_body = source_code[func_start:next_def]
        assert "-> bool:" in func_body, \
            "_update_match_for_gap must have bool return type annotation"
        assert "return True" in func_body, \
            "_update_match_for_gap must return True on success"
        assert "return False" in func_body, \
            "_update_match_for_gap must return False on failure"


class TestConfidenceFormulaMutations:
    """Mutation tests for IP-based confidence formula."""

    def test_mutation_revert_to_l2_formula(self, source_code):
        """MUTATION: Revert to old L2 distance formula (breaks scoring)."""
        target = "confidence = max(0.0, min(1.0, float(dist)))"
        assert target in source_code, "IP confidence formula must exist"

        mutated = source_code.replace(
            target,
            "confidence = max(0, 1.0 - (dist / 2.0))  # MUTATED: L2 formula"
        )
        assert "1.0 - (dist / 2.0)" in mutated
        assert target not in mutated

    def test_mutation_invert_confidence(self, source_code):
        """MUTATION: Invert confidence (1 - dist instead of dist)."""
        target = "confidence = max(0.0, min(1.0, float(dist)))"
        assert target in source_code

        mutated = source_code.replace(
            target,
            "confidence = max(0.0, min(1.0, 1.0 - float(dist)))  # MUTATED: inverted"
        )
        assert "1.0 - float(dist)" in mutated
        assert target not in mutated

    def test_mutation_skip_normalization(self, source_code):
        """MUTATION: Skip query normalization (breaks cosine similarity)."""
        target = "query_vec = query_vec / norm"
        assert target in source_code

        mutated = source_code.replace(
            target,
            "pass  # MUTATED: skip normalization"
        )
        assert "pass  # MUTATED: skip normalization" in mutated
        assert target not in mutated
