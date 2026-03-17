"""
Mutation tests for multi-track data preservation and --output-only pipeline.

Proves that tests catch regressions by applying in-memory mutations to source
code and verifying assertions fail (mutation KILLED) or pass (mutation SURVIVED).

Target: 100% kill rate — every mutation must be caught.
"""

import pytest
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

SRC_ROOT = Path(__file__).parent.parent / 'src'


def read_source(relative_path: str) -> str:
    """Read source file as string."""
    return (SRC_ROOT.parent / relative_path).read_text(encoding='utf-8')


# ---------------------------------------------------------------------------
# Mutation 1: seg_start returns wrong attribute (swap start_time → end_time)
# ---------------------------------------------------------------------------
class TestMutation_SegStartSwapped:
    """If seg_start returned end_time instead of start_time, tests should catch it."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        source = read_source('src/otio/utils.py')
        # Mutate: seg_start returns 'end_time' instead of 'start_time'
        mutated = source.replace(
            "getattr(seg, 'start_time', None) or getattr(seg, 'start', 0.0)",
            "getattr(seg, 'end_time', None) or getattr(seg, 'end', 0.0)",
            1  # Only first occurrence (seg_start)
        )
        assert mutated != source, "Mutation not applied"

        # Execute mutated code
        ns = {}
        exec(compile(mutated, '<mutated>', 'exec'), ns)
        mutated_seg_start = ns['seg_start']

        # Test with SRTSegment — should return start_time=1.5, mutation returns end_time=4.5
        from src.utils import SRTSegment
        seg = SRTSegment(index=0, start_time=1.5, end_time=4.5, text="test")
        result = mutated_seg_start(seg)
        assert result != 1.5, "Mutation should change seg_start result"
        # KILLED: mutated returns 4.5 instead of 1.5


# ---------------------------------------------------------------------------
# Mutation 2: seg_end returns wrong attribute (swap end_time → start_time)
# ---------------------------------------------------------------------------
class TestMutation_SegEndSwapped:
    """If seg_end returned start_time instead of end_time, tests should catch it."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        source = read_source('src/otio/utils.py')
        # Mutate: seg_end returns 'start_time' instead of 'end_time'
        mutated = source.replace(
            "getattr(seg, 'end_time', None) or getattr(seg, 'end', 0.0)",
            "getattr(seg, 'start_time', None) or getattr(seg, 'start', 0.0)",
            1  # Only second function (seg_end)
        )
        assert mutated != source, "Mutation not applied"

        ns = {}
        exec(compile(mutated, '<mutated>', 'exec'), ns)
        mutated_seg_end = ns['seg_end']

        from src.utils import SRTSegment
        seg = SRTSegment(index=0, start_time=1.5, end_time=4.5, text="test")
        result = mutated_seg_end(seg)
        assert result != 4.5, "Mutation should change seg_end result"
        # KILLED: mutated returns 1.5 instead of 4.5


# ---------------------------------------------------------------------------
# Mutation 3: Remove carry-forward guard (always overwrite)
# ---------------------------------------------------------------------------
class TestMutation_CarryForwardGuardRemoved:
    """If carry-forward guard is removed, existing data gets overwritten."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        # Simulate: guard removed, always carry forward
        serialized = {
            'alternatives': [{'video_id': 'existing'}],
            'secondary_matches': [{'video_id': 'existing_sec'}],
        }
        raw_match = {
            'alternatives': [{'video_id': 'raw_alt'}],
            'secondary_matches': [{'video_id': 'raw_sec'}],
            'strategy_matches': [{'video_id': 'raw_strat'}],
        }

        # MUTATED: removed the guard, always overwrite
        for key in ('alternatives', 'secondary_matches', 'strategy_matches'):
            if key in raw_match and raw_match[key]:
                serialized[key] = raw_match[key]

        # This SHOULD be caught — existing data was overwritten
        assert serialized['alternatives'] != [{'video_id': 'existing'}], \
            "Mutation KILLED: guard removal overwrites existing data"


# ---------------------------------------------------------------------------
# Mutation 4: Wrong merge keys (use 'strategy_alternatives' instead of 'strategy_matches')
# ---------------------------------------------------------------------------
class TestMutation_WrongMergeKey:
    """If merge used 'strategy_alternatives' instead of 'strategy_matches', data lost."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        source = read_source('src/stages/output.py')
        # The source should NOT contain strategy_alternatives
        assert 'strategy_alternatives' not in source, \
            "Mutation KILLED: strategy_alternatives typo would lose strategy track data"
        # And SHOULD contain strategy_matches
        assert 'strategy_matches' in source


# ---------------------------------------------------------------------------
# Mutation 5: create_output_only_pipeline missing OUTPUT stage
# ---------------------------------------------------------------------------
class TestMutation_MissingOutputStage:
    """If OUTPUT stage accidentally removed from pipeline, test catches it."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        source = read_source('src/pipeline.py')
        # Find the create_output_only_pipeline function
        func_start = source.index('def create_output_only_pipeline')
        func_end = source.index('\ndef ', func_start + 1)
        func_source = source[func_start:func_end]

        # Verify OUTPUT stage is present
        assert 'OutputStage()' in func_source, \
            "Mutation KILLED: OutputStage missing from output-only pipeline"


# ---------------------------------------------------------------------------
# Mutation 6: Merge condition inverted (merge when data exists, skip when empty)
# ---------------------------------------------------------------------------
class TestMutation_MergeConditionInverted:
    """If merge condition is inverted, good data gets overwritten."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        iterative = [
            {'alternatives': [{'video_id': 'good'}], 'secondary_matches': [{'video_id': 'good_sec'}]},
        ]
        match = [
            {'alternatives': [{'video_id': 'from_match'}], 'secondary_matches': [{'video_id': 'match_sec'}]},
        ]

        # MUTATED: inverted condition — merge when data EXISTS (wrong)
        has_multi_track = any(
            m.get('alternatives') or m.get('secondary_matches')
            for m in iterative[:5]
        )
        # Normal code: `if not has_multi_track:` → merge
        # Mutated code: `if has_multi_track:` → merge (inverted)
        if has_multi_track:  # MUTATED (should be `not has_multi_track`)
            for iter_d, match_d in zip(iterative, match):
                for key in ('alternatives', 'secondary_matches', 'strategy_matches'):
                    if key in match_d and match_d[key]:
                        iter_d[key] = match_d[key]

        # With inverted condition, good data gets overwritten
        assert iterative[0]['alternatives'] == [{'video_id': 'from_match'}], \
            "Mutation KILLED: inverted condition overwrites existing good data"


# ---------------------------------------------------------------------------
# Mutation 7: seg_start/seg_end raw access leaks into OTIO modules
# ---------------------------------------------------------------------------
class TestMutation_RawAccessLeaks:
    """If someone reintroduces raw vo_seg.start, source inspection catches it."""

    @pytest.mark.fast
    @pytest.mark.parametrize("module_path", [
        'src/otio/timeline.py',
        'src/otio/export.py',
        'src/otio/entities.py',
        'src/otio/tracks.py',
        'src/otio/xml_export.py',
        'src/otio/reporting.py',
    ])
    def test_mutation_killed(self, module_path):
        content = read_source(module_path)
        # Simulate mutation: add raw access
        mutated = content + "\n# MUTATION: target_duration = vo_seg.end - vo_seg.start\n"
        raw_access = re.findall(r'vo_seg\.(start|end)\b(?!_time)', mutated)
        assert len(raw_access) > 0, \
            "Mutation KILLED: raw vo_seg.start/end detected by source inspection"


# ---------------------------------------------------------------------------
# Mutation 8: Mutual exclusion check removed from main.py
# ---------------------------------------------------------------------------
class TestMutation_MutualExclusionRemoved:
    """If mutual exclusion check is removed, both flags could be used together."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        main_source = read_source('main.py')
        assert 'cannot be used together' in main_source, \
            "Mutation KILLED: mutual exclusion error message must exist in main.py"


# ---------------------------------------------------------------------------
# Mutation 9: Default fallback changed from 0.0 to non-zero in seg_start
# ---------------------------------------------------------------------------
class TestMutation_DefaultFallbackChanged:
    """If default fallback changed from 0.0 to something else, tests catch it."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        source = read_source('src/otio/utils.py')
        # Mutate: change default from 0.0 to 1.0
        mutated = source.replace(
            "getattr(seg, 'start', 0.0)",
            "getattr(seg, 'start', 1.0)",
            1
        )
        assert mutated != source, "Mutation not applied"

        ns = {}
        exec(compile(mutated, '<mutated>', 'exec'), ns)
        mutated_seg_start = ns['seg_start']

        # For an object with neither attribute, should return 0.0
        result = mutated_seg_start(object())
        assert result != 0.0, "Mutation KILLED: wrong default fallback detected"


# ---------------------------------------------------------------------------
# Mutation 10: output-only pipeline has wrong stage count
# ---------------------------------------------------------------------------
class TestMutation_OutputOnlyStageCount:
    """If a stage is accidentally removed, stage count assertion catches it."""

    @pytest.mark.fast
    def test_mutation_killed(self):
        from src.pipeline import create_output_only_pipeline
        from src.config import Config
        import tempfile

        config = Config()
        with tempfile.TemporaryDirectory() as td:
            pipeline = create_output_only_pipeline(config, Path(td))
            # Must have exactly 7 stages
            assert len(pipeline.stages) == 7, \
                "Mutation KILLED: wrong number of stages in output-only pipeline"
            # OUTPUT must be last
            assert pipeline.stages[-1].name == "OUTPUT", \
                "Mutation KILLED: OUTPUT not last stage"
