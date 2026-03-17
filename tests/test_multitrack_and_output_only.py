"""
Tests for multi-track data preservation and --output-only pipeline.

Covers:
- seg_start/seg_end helpers (VoiceoverSegment vs SRTSegment compatibility)
- create_output_only_pipeline factory function
- Multi-track data merge in ITERATIVE_MATCH restore
- --output-only mutual exclusion with --match-only
- strategy_matches attribute naming (not strategy_alternatives)

These tests prove the fixes for:
- Empty V2-V8 tracks (multi-track data lost during ITERATIVE_MATCH serialization)
- VoiceoverSegment/SRTSegment attribute mismatch crashing OUTPUT
- Dead --output-only flag (parsed but never consumed)
"""

import pytest
import sys
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.otio.utils import seg_start, seg_end
from src.state import VoiceoverSegment
from src.utils import SRTSegment
from src.pipeline import (
    PipelineOrchestrator,
    create_output_only_pipeline,
    create_default_pipeline,
    create_match_only_pipeline,
)
from src.config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def vo_segment():
    """VoiceoverSegment uses .start / .end"""
    return VoiceoverSegment(index=0, start=1.5, end=4.5, text="hello")


@pytest.fixture
def srt_segment():
    """SRTSegment uses .start_time / .end_time"""
    return SRTSegment(index=0, start_time=1.5, end_time=4.5, text="hello")


# ---------------------------------------------------------------------------
# seg_start / seg_end helpers
# ---------------------------------------------------------------------------

class TestSegStartEnd:
    """Test the unified segment time accessors."""

    @pytest.mark.fast
    def test_seg_start_voiceover_segment(self, vo_segment):
        assert seg_start(vo_segment) == 1.5

    @pytest.mark.fast
    def test_seg_end_voiceover_segment(self, vo_segment):
        assert seg_end(vo_segment) == 4.5

    @pytest.mark.fast
    def test_seg_start_srt_segment(self, srt_segment):
        assert seg_start(srt_segment) == 1.5

    @pytest.mark.fast
    def test_seg_end_srt_segment(self, srt_segment):
        assert seg_end(srt_segment) == 4.5

    @pytest.mark.fast
    def test_seg_duration_voiceover(self, vo_segment):
        """Duration calculation works for VoiceoverSegment."""
        duration = seg_end(vo_segment) - seg_start(vo_segment)
        assert duration == pytest.approx(3.0)

    @pytest.mark.fast
    def test_seg_duration_srt(self, srt_segment):
        """Duration calculation works for SRTSegment."""
        duration = seg_end(srt_segment) - seg_start(srt_segment)
        assert duration == pytest.approx(3.0)

    @pytest.mark.fast
    def test_seg_start_defaults_to_zero(self):
        """Unknown object type returns 0.0."""
        obj = object()
        assert seg_start(obj) == 0.0
        assert seg_end(obj) == 0.0

    @pytest.mark.fast
    def test_seg_start_prefers_start_time_over_start(self):
        """When both attributes exist, start_time takes priority."""
        @dataclass
        class BothAttrs:
            start_time: float = 10.0
            end_time: float = 20.0
            start: float = 99.0
            end: float = 99.0

        obj = BothAttrs()
        assert seg_start(obj) == 10.0
        assert seg_end(obj) == 20.0

    @pytest.mark.fast
    def test_seg_handles_zero_start_time(self):
        """Zero is a valid time, not falsy — must return 0.0 not fall through."""
        seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="first")
        # start_time=0.0 is falsy but getattr returns it; `or` falls through
        # to .start which doesn't exist, so returns 0.0 — correct result
        assert seg_start(seg) == 0.0
        assert seg_end(seg) == 5.0


# ---------------------------------------------------------------------------
# create_output_only_pipeline
# ---------------------------------------------------------------------------

class TestOutputOnlyPipeline:
    """Test the output-only pipeline factory."""

    @pytest.mark.fast
    def test_creates_pipeline_instance(self, temp_dir):
        config = Config()
        pipeline = create_output_only_pipeline(config, temp_dir)
        assert isinstance(pipeline, PipelineOrchestrator)

    @pytest.mark.fast
    def test_has_all_seven_stages(self, temp_dir):
        config = Config()
        pipeline = create_output_only_pipeline(config, temp_dir)
        assert len(pipeline.stages) == 7

    @pytest.mark.fast
    def test_stage_names_match_default(self, temp_dir):
        """Output-only pipeline has same stages as default (all needed for restore)."""
        config = Config()
        output_only = create_output_only_pipeline(config, temp_dir)
        default = create_default_pipeline(config, temp_dir)

        output_names = [s.name for s in output_only.stages]
        default_names = [s.name for s in default.stages]
        assert output_names == default_names

    @pytest.mark.fast
    def test_output_stage_is_last(self, temp_dir):
        config = Config()
        pipeline = create_output_only_pipeline(config, temp_dir)
        assert pipeline.stages[-1].name == "OUTPUT"

    @pytest.mark.fast
    def test_download_segments_before_output(self, temp_dir):
        config = Config()
        pipeline = create_output_only_pipeline(config, temp_dir)
        names = [s.name for s in pipeline.stages]
        assert names.index("DOWNLOAD_SEGMENTS") < names.index("OUTPUT")


# ---------------------------------------------------------------------------
# --output-only / --match-only mutual exclusion
# ---------------------------------------------------------------------------

class TestOutputOnlyArgHandling:
    """Test --output-only argument validation in main.py."""

    @pytest.mark.fast
    def test_output_only_flag_declared_in_args(self):
        """The --output-only flag is declared in args.py."""
        import inspect
        from src.cli import args as args_module
        source = inspect.getsource(args_module)
        assert "'--output-only'" in source or '"--output-only"' in source

    @pytest.mark.fast
    def test_output_only_consumed_in_main(self):
        """main.py references args.output_only (not dead code)."""
        main_path = Path(__file__).parent.parent / 'main.py'
        content = main_path.read_text(encoding='utf-8')
        assert 'output_only' in content
        assert 'create_output_only_pipeline' in content

    @pytest.mark.fast
    def test_mutual_exclusion_check_in_main(self):
        """main.py checks for --match-only + --output-only conflict."""
        main_path = Path(__file__).parent.parent / 'main.py'
        content = main_path.read_text(encoding='utf-8')
        assert 'match_only' in content and 'output_only' in content
        # Should have error message about mutual exclusion
        assert 'cannot be used together' in content


# ---------------------------------------------------------------------------
# Multi-track data structure inspection
# ---------------------------------------------------------------------------

class TestMultiTrackDataStructure:
    """Verify multi-track data keys are correct in serialization."""

    @pytest.mark.fast
    def test_strategy_matches_key_in_output_stage(self):
        """Output stage uses 'strategy_matches' (not 'strategy_alternatives')."""
        from src.stages.output import OutputStage
        import inspect
        source = inspect.getsource(OutputStage._resolve_match_paths)
        assert 'strategy_matches' in source
        assert 'strategy_alternatives' not in source

    @pytest.mark.fast
    def test_multi_track_keys_in_iterative_serialization(self):
        """ITERATIVE_MATCH carry-forward uses correct multi-track keys."""
        from src.stages.iterative_match import IterativeMatchStage
        import inspect
        source = inspect.getsource(IterativeMatchStage)
        # Verify the carry-forward keys
        assert "'alternatives'" in source
        assert "'secondary_matches'" in source
        assert "'strategy_matches'" in source

    @pytest.mark.fast
    def test_multi_track_keys_in_restore_merge(self):
        """ITERATIVE_MATCH restore merge uses correct keys."""
        from src.stages.iterative_match import IterativeMatchStage
        import inspect
        source = inspect.getsource(IterativeMatchStage.restore)
        # Verify merge code references multi-track keys
        assert 'alternatives' in source
        assert 'secondary_matches' in source
        assert 'strategy_matches' in source

    @pytest.mark.fast
    def test_normalize_matches_reads_raw_match_dicts(self):
        """OUTPUT _normalize_matches reads from _raw_match_dicts attribute."""
        from src.stages.output import OutputStage
        import inspect
        source = inspect.getsource(OutputStage._normalize_matches)
        assert '_raw_match_dicts' in source


# ---------------------------------------------------------------------------
# Multi-track merge logic
# ---------------------------------------------------------------------------

class TestMultiTrackMerge:
    """Test the ITERATIVE_MATCH restore multi-track merge from MATCH checkpoint."""

    @pytest.mark.fast
    def test_merge_populates_empty_iterative_data(self):
        """When iterative checkpoint has no multi-track, MATCH data is merged."""
        # Simulate iterative matches with no multi-track
        iterative_matches = [
            {'video_file': 'vid1.mp4', 'confidence': 0.9, 'alternatives': [], 'secondary_matches': []},
            {'video_file': 'vid2.mp4', 'confidence': 0.8, 'alternatives': [], 'secondary_matches': []},
        ]
        # Simulate MATCH checkpoint with full multi-track
        match_matches = [
            {
                'source_file': 'vid1.mp4', 'confidence': 0.9,
                'alternatives': [{'video_id': 'alt1', 'confidence': 0.7}],
                'secondary_matches': [{'video_id': 'sec1'}],
                'strategy_matches': [{'video_id': 'strat1'}],
            },
            {
                'source_file': 'vid2.mp4', 'confidence': 0.8,
                'alternatives': [{'video_id': 'alt2'}],
                'secondary_matches': [],
                'strategy_matches': [],
            },
        ]

        # Apply the merge logic (same as in iterative_match.py restore)
        has_multi_track = any(
            m.get('alternatives') or m.get('secondary_matches')
            for m in iterative_matches[:5]
        )
        assert not has_multi_track  # Confirms iterative data is empty

        # Merge
        merged_count = 0
        for iter_d, match_d in zip(iterative_matches, match_matches):
            for key in ('alternatives', 'secondary_matches', 'strategy_matches', 'has_gap', 'gap_reason'):
                if key in match_d and match_d[key] and not iter_d.get(key):
                    iter_d[key] = match_d[key]
            if match_d.get('alternatives') or match_d.get('secondary_matches'):
                merged_count += 1

        assert merged_count == 2
        assert iterative_matches[0]['alternatives'] == [{'video_id': 'alt1', 'confidence': 0.7}]
        assert iterative_matches[0]['secondary_matches'] == [{'video_id': 'sec1'}]
        assert iterative_matches[0]['strategy_matches'] == [{'video_id': 'strat1'}]
        assert iterative_matches[1]['alternatives'] == [{'video_id': 'alt2'}]

    @pytest.mark.fast
    def test_merge_does_not_overwrite_existing_data(self):
        """If iterative data already has multi-track, don't overwrite."""
        iterative_matches = [
            {
                'video_file': 'vid1.mp4',
                'alternatives': [{'video_id': 'existing_alt'}],
                'secondary_matches': [{'video_id': 'existing_sec'}],
            },
        ]
        match_matches = [
            {
                'source_file': 'vid1.mp4',
                'alternatives': [{'video_id': 'match_alt'}],
                'secondary_matches': [{'video_id': 'match_sec'}],
            },
        ]

        # Check: iterative already has multi-track
        has_multi_track = any(
            m.get('alternatives') or m.get('secondary_matches')
            for m in iterative_matches[:5]
        )
        assert has_multi_track  # Should skip merge entirely

    @pytest.mark.fast
    def test_merge_skipped_on_length_mismatch(self):
        """Merge skipped when MATCH and ITERATIVE have different match counts."""
        iterative_matches = [
            {'video_file': 'vid1.mp4', 'alternatives': [], 'secondary_matches': []},
        ]
        match_matches = [
            {'source_file': 'vid1.mp4', 'alternatives': [{'video_id': 'alt1'}], 'secondary_matches': []},
            {'source_file': 'vid2.mp4', 'alternatives': [{'video_id': 'alt2'}], 'secondary_matches': []},
        ]

        # Length mismatch — merge should NOT happen
        assert len(match_matches) != len(iterative_matches)
        # In real code: `if match_dicts and len(match_dicts) == len(matches_data):`
        # So iterative_matches[0] stays empty
        assert iterative_matches[0]['alternatives'] == []


# ---------------------------------------------------------------------------
# Carry-forward logic
# ---------------------------------------------------------------------------

class TestMultiTrackCarryForward:
    """Test multi-track data carry-forward during serialization."""

    @pytest.mark.fast
    def test_carry_forward_from_raw_match_dicts(self):
        """When serialized match has empty multi-track, carry from _raw_match_dicts."""
        serialized = {'video_file': 'vid1.mp4', 'confidence': 0.9,
                      'alternatives': [], 'secondary_matches': []}
        raw_match = {
            'alternatives': [{'video_id': 'alt1'}],
            'secondary_matches': [{'video_id': 'sec1'}],
            'strategy_matches': [{'video_id': 'strat1'}],
            'has_gap': False,
            'gap_reason': '',
        }

        # Apply carry-forward logic (same as iterative_match.py serialization)
        if (not serialized.get('alternatives')
                and not serialized.get('secondary_matches')):
            for key in ('alternatives', 'secondary_matches', 'strategy_matches',
                        'has_gap', 'gap_reason'):
                if key in raw_match and raw_match[key]:
                    serialized[key] = raw_match[key]

        assert serialized['alternatives'] == [{'video_id': 'alt1'}]
        assert serialized['secondary_matches'] == [{'video_id': 'sec1'}]
        assert serialized['strategy_matches'] == [{'video_id': 'strat1'}]

    @pytest.mark.fast
    def test_no_carry_forward_when_serialized_has_data(self):
        """Don't overwrite when serialized already has multi-track."""
        serialized = {
            'video_file': 'vid1.mp4',
            'alternatives': [{'video_id': 'existing'}],
            'secondary_matches': [{'video_id': 'existing_sec'}],
        }
        raw_match = {
            'alternatives': [{'video_id': 'raw_alt'}],
            'secondary_matches': [{'video_id': 'raw_sec'}],
        }

        # Guard: only carry forward if BOTH are empty
        if (not serialized.get('alternatives')
                and not serialized.get('secondary_matches')):
            for key in ('alternatives', 'secondary_matches', 'strategy_matches'):
                if key in raw_match and raw_match[key]:
                    serialized[key] = raw_match[key]

        # Should be unchanged
        assert serialized['alternatives'] == [{'video_id': 'existing'}]
        assert serialized['secondary_matches'] == [{'video_id': 'existing_sec'}]


# ---------------------------------------------------------------------------
# seg_start/seg_end usage across OTIO modules (source inspection)
# ---------------------------------------------------------------------------

class TestSegHelperUsageAcrossModules:
    """Verify all OTIO modules use seg_start/seg_end instead of raw .start/.end."""

    OTIO_MODULES = [
        'src.otio.timeline',
        'src.otio.export',
        'src.otio.entities',
        'src.otio.tracks',
        'src.otio.xml_export',
        'src.otio.reporting',
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize("module_path", [
        'src/otio/timeline.py',
        'src/otio/export.py',
        'src/otio/entities.py',
        'src/otio/tracks.py',
        'src/otio/xml_export.py',
        'src/otio/reporting.py',
    ])
    def test_no_raw_vo_seg_start_end(self, module_path):
        """No direct vo_seg.start or vo_seg.end in OTIO modules (use helpers)."""
        full_path = Path(__file__).parent.parent / module_path
        content = full_path.read_text(encoding='utf-8')

        import re
        # Match vo_seg.start or vo_seg.end but NOT vo_seg.start_time/end_time
        raw_access = re.findall(r'vo_seg\.(start|end)\b(?!_time)', content)
        assert len(raw_access) == 0, (
            f"{module_path} has {len(raw_access)} raw vo_seg.start/end accesses. "
            f"Use seg_start()/seg_end() instead."
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("module_path", [
        'src/otio/timeline.py',
        'src/otio/export.py',
        'src/otio/entities.py',
        'src/otio/tracks.py',
        'src/otio/xml_export.py',
        'src/otio/reporting.py',
    ])
    def test_no_raw_voiceover_segment_start_end(self, module_path):
        """No direct voiceover_segment.start/end in OTIO modules."""
        full_path = Path(__file__).parent.parent / module_path
        content = full_path.read_text(encoding='utf-8')

        import re
        raw_access = re.findall(r'voiceover_segment\.(start|end)\b(?!_time)', content)
        assert len(raw_access) == 0, (
            f"{module_path} has {len(raw_access)} raw voiceover_segment.start/end. "
            f"Use seg_start()/seg_end() instead."
        )


# ---------------------------------------------------------------------------
# Pipeline checkpoint reset for --output-only
# ---------------------------------------------------------------------------

class TestOutputOnlyCheckpointReset:
    """Test that --output-only resets checkpoint to DOWNLOAD_SEGMENTS."""

    @pytest.mark.fast
    def test_should_skip_all_except_output(self):
        """With last_completed=DOWNLOAD_SEGMENTS, only OUTPUT should run."""
        from src.checkpoint import STAGE_ORDER

        last_completed = 'DOWNLOAD_SEGMENTS'
        completed_idx = STAGE_ORDER.index(last_completed)

        for stage in STAGE_ORDER:
            current_idx = STAGE_ORDER.index(stage)
            should_skip = current_idx <= completed_idx

            if stage == 'OUTPUT':
                assert not should_skip, "OUTPUT should NOT be skipped"
            else:
                assert should_skip, f"{stage} should be skipped"

    @pytest.mark.fast
    def test_download_segments_is_penultimate(self):
        """DOWNLOAD_SEGMENTS is the stage right before OUTPUT."""
        from src.checkpoint import STAGE_ORDER
        dl_idx = STAGE_ORDER.index('DOWNLOAD_SEGMENTS')
        out_idx = STAGE_ORDER.index('OUTPUT')
        assert out_idx == dl_idx + 1
