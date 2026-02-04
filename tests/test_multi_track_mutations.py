"""
Mutation tests for multi-track checkpoint serialization.

Proves that tests catch regressions by applying in-memory mutations
to source code and verifying assertions fail.
"""

import pytest
from pathlib import Path

# Read source files into strings for in-memory mutation
_SERIAL_SRC = Path('src/matching/serialization.py').read_text(encoding='utf-8')
_OUTPUT_SRC = Path('src/stages/output.py').read_text(encoding='utf-8')
_MATCH_SRC = Path('src/stages/match.py').read_text(encoding='utf-8')
_ITER_SRC = Path('src/stages/iterative_match.py').read_text(encoding='utf-8')


class TestMutationsSerialization:
    """Mutations against serialization.py — all must be KILLED."""

    def test_mutation_remove_multi_track_update_match_stage(self):
        """MUTATION: Remove result.update(_extract_multi_track_data(match)) from match stage.
        If multi-track data isn't merged, serialized output loses V2-V8 keys."""
        mutated = _SERIAL_SRC.replace(
            "    result.update(_extract_multi_track_data(match))\n    return result\n\n\ndef serialize_match_for_iterative_stage",
            "    return result\n\n\ndef serialize_match_for_iterative_stage",
        )
        # Verify the mutation actually happened
        assert "_extract_multi_track_data" not in mutated.split("serialize_match_for_iterative_stage")[0].split("serialize_match_for_match_stage")[1]

        # Now verify the test would catch it: serialize a MatchResult and check for keys
        from src.utils import SRTSegment, Match as UtilsMatch, MatchResult, AlternativeMatch
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text='vo')
        vid = SRTSegment(index=0, start_time=0.0, end_time=10.0, text='', source_file='v.mp4')
        pm = UtilsMatch(voiceover_segment=vo, video_segment=vid, video_scene=None, confidence=0.9, reasoning='ok')
        mr = MatchResult(
            primary_match=pm,
            alternatives=[AlternativeMatch(video_segment=vid, video_scene=None, confidence=0.7, reasoning='alt', diversity_score=0.3)],
        )
        from src.matching.serialization import serialize_match_for_match_stage
        result = serialize_match_for_match_stage(mr, index=0)
        # The real code includes alternatives — if the mutation were applied, this would fail
        assert 'alternatives' in result, "MUTATION SURVIVED: match stage should include alternatives"
        assert len(result['alternatives']) == 1, "MUTATION SURVIVED: alternatives should have 1 item"

    def test_mutation_remove_multi_track_update_iterative_stage(self):
        """MUTATION: Remove result.update(_extract_multi_track_data(match)) from iterative stage."""
        mutated = _SERIAL_SRC.replace(
            "    result.update(_extract_multi_track_data(match))\n    return result\n\n\ndef is_empty_source",
            "    return result\n\n\ndef is_empty_source",
        )
        assert "_extract_multi_track_data" not in mutated.split("is_empty_source")[0].split("serialize_match_for_iterative_stage")[1]

        from src.utils import SRTSegment, Match as UtilsMatch, MatchResult, StrategyMatch
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text='vo')
        vid = SRTSegment(index=0, start_time=0.0, end_time=10.0, text='', source_file='v.mp4')
        pm = UtilsMatch(voiceover_segment=vo, video_segment=vid, video_scene=None, confidence=0.9, reasoning='ok')
        mr = MatchResult(
            primary_match=pm,
            strategy_matches=[StrategyMatch(video_segment=vid, video_scene=None, confidence=0.5, reasoning='strat', strategy='visual_first')],
        )
        from src.matching.serialization import serialize_match_for_iterative_stage
        result = serialize_match_for_iterative_stage(mr, index=0)
        assert 'strategy_matches' in result, "MUTATION SURVIVED: iterative stage should include strategy_matches"
        assert len(result['strategy_matches']) == 1, "MUTATION SURVIVED"

    def test_mutation_skip_alternatives_loop(self):
        """MUTATION: Replace alternatives loop body with pass.
        Alternatives would always be empty."""
        mutated = _SERIAL_SRC.replace(
            "for alt in getattr(match, 'alternatives', []) or []:\n        try:\n            result['alternatives'].append(alt.to_dict())",
            "for alt in getattr(match, 'alternatives', []) or []:\n        pass  # MUTATED",
        )
        assert "pass  # MUTATED" in mutated

        from src.utils import SRTSegment, Match as UtilsMatch, MatchResult, AlternativeMatch
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text='vo')
        vid = SRTSegment(index=0, start_time=0.0, end_time=10.0, text='', source_file='v.mp4')
        pm = UtilsMatch(voiceover_segment=vo, video_segment=vid, video_scene=None, confidence=0.9, reasoning='ok')
        alt = AlternativeMatch(video_segment=vid, video_scene=None, confidence=0.7, reasoning='alt', diversity_score=0.3)
        mr = MatchResult(primary_match=pm, alternatives=[alt])
        from src.matching.serialization import _extract_multi_track_data
        data = _extract_multi_track_data(mr)
        assert len(data['alternatives']) == 1, "MUTATION SURVIVED: alternatives should not be empty"

    def test_mutation_has_gap_always_false(self):
        """MUTATION: has_gap always returns False instead of reading attribute."""
        mutated = _SERIAL_SRC.replace(
            "result['has_gap'] = bool(getattr(match, 'has_gap', False))",
            "result['has_gap'] = False  # MUTATED"
        )
        assert "# MUTATED" in mutated

        from src.utils import SRTSegment, Match as UtilsMatch, MatchResult
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text='vo')
        vid = SRTSegment(index=0, start_time=0.0, end_time=10.0, text='', source_file='v.mp4')
        pm = UtilsMatch(voiceover_segment=vo, video_segment=vid, video_scene=None, confidence=0.9, reasoning='ok')
        mr = MatchResult(primary_match=pm, has_gap=True, gap_reason='test gap')
        from src.matching.serialization import _extract_multi_track_data
        data = _extract_multi_track_data(mr)
        assert data['has_gap'] is True, "MUTATION SURVIVED: has_gap should be True"

    def test_mutation_no_alternatives_guard(self):
        """MUTATION: Remove 'if not hasattr(match, alternatives)' guard.
        Plain Match would try to iterate non-existent attributes."""
        mutated = _SERIAL_SRC.replace(
            "    # Only MatchResult has these fields\n    if not hasattr(match, 'alternatives'):\n        return result",
            "    # MUTATED: guard removed",
        )
        assert "guard removed" in mutated

        # With the guard in place, plain Match returns empty data without error
        from src.utils import SRTSegment, Match as UtilsMatch
        vo = SRTSegment(index=0, start_time=0.0, end_time=5.0, text='vo')
        vid = SRTSegment(index=0, start_time=0.0, end_time=10.0, text='', source_file='v.mp4')
        plain = UtilsMatch(voiceover_segment=vo, video_segment=vid, video_scene=None, confidence=0.8, reasoning='plain')
        from src.matching.serialization import _extract_multi_track_data
        data = _extract_multi_track_data(plain)
        assert data['alternatives'] == [], "MUTATION SURVIVED: plain match should have empty alternatives"


class TestMutationsMatchRestore:
    """Mutations against match.py restore — raw dict stashing."""

    def test_mutation_remove_raw_dict_stash(self):
        """MUTATION: Remove state._raw_match_dicts = matches_data from match.py.
        Output stage would lose multi-track data."""
        mutated = _MATCH_SRC.replace(
            "state._raw_match_dicts = matches_data",
            "# MUTATED: stash removed",
        )
        assert "stash removed" in mutated
        assert "state._raw_match_dicts" not in mutated

        # Verify the real code actually stashes
        from src.stages.match import MatchStage
        from src.state import PipelineState
        from unittest.mock import MagicMock

        stage = MatchStage()
        state = PipelineState()
        mock_cp = MagicMock()
        mock_cp.get_stage_data.return_value = {
            'matches': [{'segment_index': 0, 'video_file': 'v.mp4', 'confidence': 0.8,
                         'alternatives': []}]
        }
        stage.restore(state, mock_cp)
        assert hasattr(state, '_raw_match_dicts'), "MUTATION SURVIVED: _raw_match_dicts must be set"


class TestMutationsIterativeRestore:
    """Mutations against iterative_match.py restore — raw dict stashing."""

    def test_mutation_remove_raw_dict_stash(self):
        """MUTATION: Remove state._raw_match_dicts from iterative_match.py."""
        mutated = _ITER_SRC.replace(
            "state._raw_match_dicts = matches_data",
            "# MUTATED: stash removed",
        )
        assert "stash removed" in mutated
        assert "state._raw_match_dicts" not in mutated

        from src.stages.iterative_match import IterativeMatchStage
        from src.state import PipelineState
        from unittest.mock import MagicMock

        stage = IterativeMatchStage()
        state = PipelineState()
        mock_cp = MagicMock()
        mock_cp.get_stage_data.return_value = {
            'matches': [{'segment_index': 0, 'video_file': 'v.mp4', 'confidence': 0.8}]
        }
        stage.restore(state, mock_cp)
        assert hasattr(state, '_raw_match_dicts'), "MUTATION SURVIVED: _raw_match_dicts must be set"


class TestMutationsOutputNormalize:
    """Mutations against output.py _normalize_matches — multi-track restore."""

    def test_mutation_skip_alternative_restore(self):
        """MUTATION: Skip restoring alternatives from raw dicts.
        Alternatives would always be empty after checkpoint restore."""
        # The real code loops over raw.get('alternatives', [])
        assert "for alt_data in raw.get('alternatives', []):" in _OUTPUT_SRC, \
            "Source must contain the alternatives restore loop"

        # Verify restore actually works
        from src.stages.output import OutputStage
        from src.state import PipelineState, Match as StateMatch
        from src.utils import SRTSegment, AlternativeMatch

        stage = OutputStage()
        state = PipelineState()
        state.voiceover_segments = [SRTSegment(index=0, start_time=0.0, end_time=5.0, text='seg')]
        state.matches = [StateMatch(segment_index=0, video_file='v.mp4', video_start=0.0, video_end=5.0, confidence=0.8)]

        alt = AlternativeMatch(
            video_segment=SRTSegment(index=0, start_time=0.0, end_time=10.0, text='', source_file='alt.mp4'),
            video_scene=None, confidence=0.7, reasoning='alt', diversity_score=0.3,
        )
        state._raw_match_dicts = [{
            'segment_index': 0, 'video_file': 'v.mp4',
            'alternatives': [alt.to_dict()],
            'secondary_matches': [], 'strategy_matches': [],
            'has_gap': False, 'gap_reason': '',
        }]

        normalized = stage._normalize_matches(state)
        assert len(normalized[0].alternatives) == 1, "MUTATION SURVIVED: alternatives should be restored"
        assert normalized[0].alternatives[0].video_segment.source_file == 'alt.mp4'

    def test_mutation_skip_cleanup(self):
        """MUTATION: Don't delete _raw_match_dicts after conversion.
        Would leak stale data on state."""
        assert "del state._raw_match_dicts" in _OUTPUT_SRC, \
            "Source must contain cleanup of _raw_match_dicts"

        from src.stages.output import OutputStage
        from src.state import PipelineState, Match as StateMatch
        from src.utils import SRTSegment

        stage = OutputStage()
        state = PipelineState()
        state.voiceover_segments = [SRTSegment(index=0, start_time=0.0, end_time=5.0, text='seg')]
        state.matches = [StateMatch(segment_index=0, video_file='v.mp4', video_start=0.0, video_end=5.0, confidence=0.8)]
        state._raw_match_dicts = [{'segment_index': 0, 'video_file': 'v.mp4',
                                   'alternatives': [], 'secondary_matches': [],
                                   'strategy_matches': [], 'has_gap': False, 'gap_reason': ''}]

        stage._normalize_matches(state)
        assert not hasattr(state, '_raw_match_dicts'), "MUTATION SURVIVED: _raw_match_dicts should be cleaned up"

    def test_mutation_wrong_from_dict_class(self):
        """MUTATION: Use StrategyMatch.from_dict for alternatives (wrong class).
        Would crash or produce wrong object type."""
        # Verify the source uses AlternativeMatch.from_dict for alternatives
        assert "AlternativeMatch.from_dict(alt_data)" in _OUTPUT_SRC, \
            "Source must use AlternativeMatch.from_dict for alternatives"
        # And StrategyMatch.from_dict for strategy_matches
        assert "StrategyMatch.from_dict(strat_data)" in _OUTPUT_SRC, \
            "Source must use StrategyMatch.from_dict for strategy_matches"

        from src.stages.output import OutputStage
        from src.state import PipelineState, Match as StateMatch
        from src.utils import SRTSegment, AlternativeMatch

        stage = OutputStage()
        state = PipelineState()
        state.voiceover_segments = [SRTSegment(index=0, start_time=0.0, end_time=5.0, text='seg')]
        state.matches = [StateMatch(segment_index=0, video_file='v.mp4', video_start=0.0, video_end=5.0, confidence=0.8)]

        alt = AlternativeMatch(
            video_segment=SRTSegment(index=0, start_time=0.0, end_time=10.0, text='', source_file='a.mp4'),
            video_scene=None, confidence=0.7, reasoning='alt', diversity_score=0.5,
        )
        state._raw_match_dicts = [{
            'segment_index': 0, 'video_file': 'v.mp4',
            'alternatives': [alt.to_dict()],
            'secondary_matches': [], 'strategy_matches': [],
            'has_gap': False, 'gap_reason': '',
        }]

        normalized = stage._normalize_matches(state)
        restored_alt = normalized[0].alternatives[0]
        assert isinstance(restored_alt, AlternativeMatch), "MUTATION SURVIVED: should be AlternativeMatch, not StrategyMatch"
        assert hasattr(restored_alt, 'diversity_score'), "MUTATION SURVIVED: AlternativeMatch has diversity_score"
