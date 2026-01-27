"""
Unit tests for compilation modules:
- src/compilation/stages/download.py (DownloadStage)
- src/compilation/stages/gap_check.py (GapAnalyzer, TopicBroadener, GapCheckStage)
- src/compilation/timeline/builder.py (CompilationTimelineBuilder)

Since src/compilation/state.py and orchestrator.py don't exist yet,
we inject fake modules into sys.modules BEFORE importing the real code.
"""

from __future__ import annotations

import copy
import json
import logging
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import opentimelineio as otio


# ---------------------------------------------------------------------------
# Stand-in dataclasses for the not-yet-created src/compilation/state.py
# ---------------------------------------------------------------------------

@dataclass
class _DownloadedClip:
    file: str = ""
    video_id: str = ""
    actual_duration: float = 30.0
    keyword: str = ""
    title: str = ""


@dataclass
class _CompilationCandidate:
    video_id: str = ""
    keyword: str = ""
    title: str = ""
    duration: float = 30.0


@dataclass
class _GapReport:
    complete: bool = False
    total_shortfall: float = 0.0
    track_shortfalls: list = field(default_factory=list)
    new_keywords: list = field(default_factory=list)


@dataclass
class _CompilationState:
    filtered_candidates: list = field(default_factory=list)
    downloaded_clips: list = field(default_factory=list)
    arranged_tracks: list = field(default_factory=list)
    target_duration: float = 300.0
    topic: str = "cats"
    all_used_keywords: list = field(default_factory=lambda: ["cats"])
    retry_count: int = 0
    _project_path: Path = field(default_factory=lambda: Path("/tmp/test_project"))

    def get_project_path(self) -> Path:
        return self._project_path


# ---------------------------------------------------------------------------
# Inject fake state and orchestrator modules BEFORE importing real code.
# This allows `from ..state import X` inside download.py / gap_check.py /
# builder.py to resolve successfully.
# ---------------------------------------------------------------------------

def _bootstrap_compilation_modules():
    """Pre-seed sys.modules so the compilation package can be imported."""
    import importlib.util
    import os

    # Resolve real filesystem paths for the compilation sub-packages
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _comp_dir = os.path.join(_repo, "src", "compilation")
    _stages_dir = os.path.join(_comp_dir, "stages")
    _timeline_dir = os.path.join(_comp_dir, "timeline")

    # Create the fake state module
    state_mod = types.ModuleType("src.compilation.state")
    state_mod.CompilationState = _CompilationState
    state_mod.CompilationCandidate = _CompilationCandidate
    state_mod.DownloadedClip = _DownloadedClip
    state_mod.GapReport = _GapReport

    # Create a fake orchestrator module (also missing)
    orch_mod = types.ModuleType("src.compilation.orchestrator")
    orch_mod.CompilationOrchestrator = MagicMock
    orch_mod.load_compilation_config = MagicMock(return_value={})
    orch_mod.get_default_config = MagicMock(return_value={})

    # Remove any cached broken imports of compilation
    for key in list(sys.modules.keys()):
        if key.startswith("src.compilation"):
            del sys.modules[key]

    # Make sure the real src package exists in sys.modules
    if "src" in sys.modules:
        src_pkg = sys.modules["src"]
    else:
        src_pkg = types.ModuleType("src")
        sys.modules["src"] = src_pkg

    # Create compilation package with real __path__
    compilation_pkg = types.ModuleType("src.compilation")
    compilation_pkg.__path__ = [_comp_dir]
    compilation_pkg.__package__ = "src.compilation"

    # Create sub-packages with real __path__
    stages_pkg = types.ModuleType("src.compilation.stages")
    stages_pkg.__path__ = [_stages_dir]
    stages_pkg.__package__ = "src.compilation.stages"

    timeline_pkg = types.ModuleType("src.compilation.timeline")
    timeline_pkg.__path__ = [_timeline_dir]
    timeline_pkg.__package__ = "src.compilation.timeline"

    # Register all in sys.modules (state & orchestrator MUST be registered
    # before we import the real modules, because they do relative imports).
    sys.modules["src.compilation"] = compilation_pkg
    sys.modules["src.compilation.state"] = state_mod
    sys.modules["src.compilation.orchestrator"] = orch_mod
    sys.modules["src.compilation.stages"] = stages_pkg
    sys.modules["src.compilation.timeline"] = timeline_pkg

    # Load real modules via importlib to get proper module objects
    def _load(name, filepath):
        spec = importlib.util.spec_from_file_location(name, filepath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    dl_mod = _load(
        "src.compilation.stages.download",
        os.path.join(_stages_dir, "download.py"),
    )
    gc_mod = _load(
        "src.compilation.stages.gap_check",
        os.path.join(_stages_dir, "gap_check.py"),
    )
    bl_mod = _load(
        "src.compilation.timeline.builder",
        os.path.join(_timeline_dir, "builder.py"),
    )

    # Attach sub-packages to parent
    compilation_pkg.state = state_mod
    compilation_pkg.orchestrator = orch_mod
    compilation_pkg.stages = stages_pkg
    compilation_pkg.timeline = timeline_pkg
    stages_pkg.download = dl_mod
    stages_pkg.gap_check = gc_mod
    timeline_pkg.builder = bl_mod

    # Attach compilation to src so patch("src.compilation...") works
    src_pkg.compilation = compilation_pkg

    return dl_mod, gc_mod, bl_mod


_dl_mod, _gc_mod, _bl_mod = _bootstrap_compilation_modules()

# Now we can import the classes directly
DownloadStage = _dl_mod.DownloadStage
GapAnalyzer = _gc_mod.GapAnalyzer
TopicBroadener = _gc_mod.TopicBroadener
GapCheckStage = _gc_mod.GapCheckStage
CompilationTimelineBuilder = _bl_mod.CompilationTimelineBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clip_factory():
    """Factory for creating DownloadedClip instances."""
    def _make(
        file: str = "E:/v/test/video.mp4",
        video_id: str = "abc12345678",
        actual_duration: float = 30.0,
        keyword: str = "cats",
        title: str = "Funny cats",
    ) -> _DownloadedClip:
        return _DownloadedClip(
            file=file, video_id=video_id,
            actual_duration=actual_duration,
            keyword=keyword, title=title,
        )
    return _make


@pytest.fixture
def candidate_factory():
    """Factory for creating CompilationCandidate instances."""
    def _make(
        video_id: str = "abc12345678",
        keyword: str = "cats",
        title: str = "Funny cats",
        duration: float = 30.0,
    ) -> _CompilationCandidate:
        return _CompilationCandidate(
            video_id=video_id, keyword=keyword,
            title=title, duration=duration,
        )
    return _make


# ===================================================================
# GAP CHECK MODULE TESTS
# ===================================================================

class TestGapAnalyzer:
    """Tests for GapAnalyzer.analyze()."""

    @pytest.mark.fast
    def test_empty_tracks_returns_incomplete(self):
        """Empty tracks list should report incomplete with estimated shortfall."""
        analyzer = GapAnalyzer()
        report = analyzer.analyze(tracks=[], target_duration=300.0)

        assert not report.complete, "Empty tracks should be incomplete"
        assert report.total_shortfall > 0, "Should have positive shortfall for empty tracks"
        assert len(report.track_shortfalls) == 4, "Should assume 4 tracks when empty"

    @pytest.mark.fast
    def test_complete_tracks_returns_complete(self, clip_factory):
        """Tracks meeting target duration should report complete."""
        analyzer = GapAnalyzer()
        clip_300s = clip_factory(actual_duration=300.0)
        tracks = [[clip_300s], [clip_300s]]

        report = analyzer.analyze(tracks=tracks, target_duration=300.0)

        assert report.complete, "Tracks meeting target should be complete"
        assert report.total_shortfall == 0, "No shortfall when tracks are full"

    @pytest.mark.fast
    def test_partial_tracks_reports_shortfall(self, clip_factory):
        """Tracks below target should report per-track shortfalls."""
        analyzer = GapAnalyzer()
        clip_100s = clip_factory(actual_duration=100.0)
        clip_300s = clip_factory(actual_duration=300.0)
        tracks = [[clip_100s], [clip_300s]]

        report = analyzer.analyze(tracks=tracks, target_duration=300.0)

        assert not report.complete, "Partial tracks should not be complete"
        assert report.track_shortfalls[0] == 200.0, "First track shortfall should be 200s"
        assert report.track_shortfalls[1] == 0, "Second track has no shortfall"
        assert report.total_shortfall == 200.0

    @pytest.mark.fast
    def test_empty_track_in_list_marks_incomplete(self, clip_factory):
        """A single empty track in an otherwise sufficient list is incomplete."""
        analyzer = GapAnalyzer()
        clip_300s = clip_factory(actual_duration=300.0)
        tracks = [[clip_300s], []]

        report = analyzer.analyze(tracks=tracks, target_duration=300.0)

        assert not report.complete, "Should be incomplete if any track is empty"

    @pytest.mark.fast
    def test_multiple_clips_sum_duration(self, clip_factory):
        """Multiple clips in a track should have durations summed."""
        analyzer = GapAnalyzer()
        clip_a = clip_factory(actual_duration=150.0)
        clip_b = clip_factory(actual_duration=150.0)
        tracks = [[clip_a, clip_b]]

        report = analyzer.analyze(tracks=tracks, target_duration=300.0)

        assert report.complete, "Two clips summing to target should be complete"
        assert report.track_shortfalls[0] == 0


class TestTopicBroadener:
    """Tests for TopicBroadener keyword generation."""

    @pytest.mark.fast
    def test_parse_keywords_json_array(self):
        """Should parse a clean JSON array response."""
        broadener = TopicBroadener(MagicMock())
        response = '["funny dogs", "cat fails", "animal compilation"]'
        result = broadener._parse_keywords(response)

        assert result == ["funny dogs", "cat fails", "animal compilation"]

    @pytest.mark.fast
    def test_parse_keywords_json_embedded_in_text(self):
        """Should extract JSON array from surrounding text."""
        broadener = TopicBroadener(MagicMock())
        response = 'Here are the keywords:\n["alpha", "beta", "gamma"]\nEnjoy!'
        result = broadener._parse_keywords(response)

        assert result == ["alpha", "beta", "gamma"]

    @pytest.mark.fast
    def test_parse_keywords_fallback_newlines(self):
        """Should fall back to newline parsing when no JSON found."""
        broadener = TopicBroadener(MagicMock())
        response = "- funny dogs\n- cat fails\n- animal videos"
        result = broadener._parse_keywords(response)

        assert len(result) >= 3
        # The parser strips dashes but may leave surrounding whitespace
        stripped = [kw.strip() for kw in result]
        assert "funny dogs" in stripped

    @pytest.mark.fast
    def test_parse_keywords_fallback_limits_to_five(self):
        """Fallback (non-JSON) path should limit results to 5 keywords max."""
        broadener = TopicBroadener(MagicMock())
        # Newline-delimited input (no JSON) triggers the fallback limiter
        response = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\neta\ntheta"
        result = broadener._parse_keywords(response)

        assert len(result) <= 5, "Fallback parser should limit to 5 keywords"

    @pytest.mark.fast
    def test_generate_fallback_keywords(self):
        """Fallback should produce keywords from topic words + suffixes."""
        broadener = TopicBroadener(MagicMock())
        result = broadener._generate_fallback_keywords("cats dogs", used_keywords=[])

        assert len(result) > 0
        for kw in result:
            assert any(w in kw for w in ["cats", "dogs"]) or \
                any(p in kw for p in ["viral", "epic", "amazing", "top"])

    @pytest.mark.fast
    def test_generate_fallback_excludes_used(self):
        """Fallback should not return keywords already used."""
        broadener = TopicBroadener(MagicMock())
        used = ["cats compilation", "cats funny"]
        result = broadener._generate_fallback_keywords("cats", used_keywords=used)

        for kw in result:
            assert kw not in used, f"Fallback returned already-used keyword: {kw}"

    @pytest.mark.fast
    def test_broaden_filters_used_keywords(self):
        """broaden() should filter out keywords that have already been tried."""
        broadener = TopicBroadener(MagicMock())
        with patch.object(
            type(broadener), 'llm_client',
            new_callable=PropertyMock,
        ) as mock_llm_prop:
            mock_llm = MagicMock()
            mock_llm.generate.return_value = '["cats", "funny dogs", "bird videos"]'
            mock_llm_prop.return_value = mock_llm

            result = broadener.broaden(
                topic="cats",
                used_keywords=["cats"],
                shortfall_seconds=120.0,
                compilation_config={'llm': {'provider': 'gemini'}},
            )

            assert "cats" not in result, "Should filter already-used keyword"
            assert "funny dogs" in result

    @pytest.mark.fast
    def test_broaden_llm_failure_uses_fallback(self):
        """broaden() should use fallback when LLM call fails."""
        broadener = TopicBroadener(MagicMock())
        with patch.object(
            type(broadener), 'llm_client',
            new_callable=PropertyMock,
        ) as mock_llm_prop:
            mock_llm = MagicMock()
            mock_llm.generate.side_effect = RuntimeError("LLM unavailable")
            mock_llm_prop.return_value = mock_llm

            result = broadener.broaden(
                topic="cats",
                used_keywords=["cats"],
                shortfall_seconds=120.0,
                compilation_config={'llm': {'provider': 'gemini'}},
            )

            assert isinstance(result, list)
            assert len(result) > 0, "Fallback should produce at least one keyword"


class TestGapCheckStage:
    """Tests for GapCheckStage.run()."""

    @pytest.mark.fast
    def test_stage_name_and_description(self):
        """Stage should have correct name and description."""
        stage = GapCheckStage(MagicMock())
        assert stage.name == "GAP_CHECK"
        assert "track" in stage.description.lower() or "gap" in stage.description.lower()

    @pytest.mark.fast
    def test_complete_state_returns_complete_report(self, clip_factory):
        """When tracks are complete, should return complete report without broadening."""
        stage = GapCheckStage(MagicMock())
        state = _CompilationState(
            arranged_tracks=[[clip_factory(actual_duration=300.0)]],
            target_duration=300.0,
        )
        comp_config = {'max_retries': 3}

        report = stage.run(state, comp_config)

        assert report.complete
        assert report.new_keywords == []

    @pytest.mark.fast
    def test_max_retries_prevents_broadening(self, clip_factory):
        """When retry_count >= max_retries, should not call broadener."""
        stage = GapCheckStage(MagicMock())
        state = _CompilationState(
            arranged_tracks=[[clip_factory(actual_duration=50.0)]],
            target_duration=300.0,
            retry_count=5,
        )
        comp_config = {'max_retries': 3}

        with patch.object(stage.topic_broadener, 'broaden') as mock_broaden:
            report = stage.run(state, comp_config)
            mock_broaden.assert_not_called()

        assert not report.complete


# ===================================================================
# DOWNLOAD STAGE TESTS
# ===================================================================

class TestDownloadStage:
    """Tests for compilation DownloadStage."""

    @pytest.mark.fast
    def test_stage_name(self):
        """Stage should have DOWNLOAD name."""
        stage = DownloadStage(MagicMock())
        assert stage.name == "DOWNLOAD"

    @pytest.mark.fast
    def test_sanitize_dirname_removes_special_chars(self):
        """_sanitize_dirname should replace illegal directory characters."""
        stage = DownloadStage(MagicMock())

        assert stage._sanitize_dirname('cats/dogs') == 'cats_dogs'
        assert stage._sanitize_dirname('a:b') == 'a_b'
        assert stage._sanitize_dirname('test<>video') == 'test__video'
        assert stage._sanitize_dirname('hello?world*') == 'hello_world_'

    @pytest.mark.fast
    def test_sanitize_dirname_limits_length(self):
        """_sanitize_dirname should truncate to 50 characters."""
        stage = DownloadStage(MagicMock())
        long_name = "a" * 100
        result = stage._sanitize_dirname(long_name)
        assert len(result) <= 50

    @pytest.mark.fast
    def test_sanitize_dirname_empty_returns_general(self):
        """_sanitize_dirname should return 'general' for empty strings."""
        stage = DownloadStage(MagicMock())
        assert stage._sanitize_dirname("") == "general"
        assert stage._sanitize_dirname("   ") == "general"

    @pytest.mark.fast
    def test_extract_video_id_standard_pattern(self):
        """_extract_video_id should find 11-char ID before extension."""
        stage = DownloadStage(MagicMock())
        result = stage._extract_video_id("some_title_aBcD1234567.mp4")
        assert result == "aBcD1234567"

    @pytest.mark.fast
    def test_extract_video_id_no_match(self):
        """_extract_video_id should return empty string when no ID found."""
        stage = DownloadStage(MagicMock())
        result = stage._extract_video_id("short.mp4")
        assert result == ""

    @pytest.mark.fast
    def test_get_video_duration_success(self):
        """_get_video_duration should parse ffprobe output on success."""
        stage = DownloadStage(MagicMock())
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "45.123\n"

        with patch("subprocess.run", return_value=mock_result):
            duration = stage._get_video_duration(Path("fake.mp4"))

        assert duration == pytest.approx(45.123)

    @pytest.mark.fast
    def test_get_video_duration_failure_returns_zero(self):
        """_get_video_duration should return 0.0 on ffprobe failure."""
        stage = DownloadStage(MagicMock())
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch("subprocess.run", return_value=mock_result):
            duration = stage._get_video_duration(Path("fake.mp4"))

        assert duration == 0.0

    @pytest.mark.fast
    def test_get_video_duration_exception_returns_zero(self):
        """_get_video_duration should return 0.0 on exception."""
        stage = DownloadStage(MagicMock())

        with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found")):
            duration = stage._get_video_duration(Path("fake.mp4"))

        assert duration == 0.0

    @pytest.mark.fast
    def test_run_no_candidates_returns_false(self, tmp_path):
        """run() with no candidates should return False."""
        stage = DownloadStage(MagicMock())
        state = _CompilationState(
            filtered_candidates=[],
            _project_path=tmp_path,
        )
        comp_config = {'download': {'output_subdir': 'videos'}}

        result = stage.run(state, comp_config)

        assert result is False

    @pytest.mark.fast
    def test_run_groups_by_keyword(self, tmp_path, candidate_factory):
        """run() should group candidates by keyword for organized download."""
        stage = DownloadStage(MagicMock())

        cand_a = candidate_factory(video_id="vid_a_123456", keyword="cats")
        cand_b = candidate_factory(video_id="vid_b_123456", keyword="dogs")

        state = _CompilationState(
            filtered_candidates=[cand_a, cand_b],
            _project_path=tmp_path,
        )
        comp_config = {'download': {'output_subdir': 'videos'}}

        mock_downloader = MagicMock()
        mock_downloader._download_by_ids.return_value = []

        with patch.object(
            type(stage), 'downloader',
            new_callable=PropertyMock, return_value=mock_downloader,
        ):
            stage.run(state, comp_config)

        assert mock_downloader._download_by_ids.call_count == 2

    @pytest.mark.fast
    def test_run_download_error_counts_failures(self, tmp_path, candidate_factory):
        """run() should count failed downloads but not crash."""
        stage = DownloadStage(MagicMock())
        cand = candidate_factory(video_id="fail_video_id", keyword="cats")

        state = _CompilationState(
            filtered_candidates=[cand],
            _project_path=tmp_path,
        )
        comp_config = {'download': {'output_subdir': 'videos'}}

        mock_downloader = MagicMock()
        mock_downloader._download_by_ids.side_effect = RuntimeError("Download failed")

        with patch.object(
            type(stage), 'downloader',
            new_callable=PropertyMock, return_value=mock_downloader,
        ):
            result = stage.run(state, comp_config)

        assert result is False


# ===================================================================
# TIMELINE BUILDER TESTS
# ===================================================================

class TestCompilationTimelineBuilder:
    """Tests for CompilationTimelineBuilder."""

    @pytest.mark.fast
    def test_create_empty_timeline(self):
        """Should create a valid empty timeline with no tracks."""
        builder = CompilationTimelineBuilder()
        tl = builder.create_timeline(tracks=[], name="Empty")

        assert tl.name == "Empty"
        assert len(tl.tracks) == 0

    @pytest.mark.fast
    def test_create_timeline_with_one_track(self, clip_factory):
        """Should create paired video + audio tracks for one arrangement."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory(file="E:/v/test/cats_aBcD1234567.mp4", actual_duration=30.0)
        tl = builder.create_timeline(tracks=[[clip]], name="Test")

        assert len(tl.tracks) == 2, "Should have 1 video + 1 audio track"
        assert tl.tracks[0].kind == otio.schema.TrackKind.Video
        assert tl.tracks[1].kind == otio.schema.TrackKind.Audio

    @pytest.mark.fast
    def test_create_timeline_tracks_enabled(self, clip_factory):
        """All tracks should be enabled (unlike main pipeline)."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory()
        tl = builder.create_timeline(tracks=[[clip], [clip]], name="Test")

        for track in tl.tracks:
            assert track.enabled is True, f"Track {track.name} should be enabled"

    @pytest.mark.fast
    def test_create_timeline_davinci_metadata(self, clip_factory):
        """Timeline should include DaVinci Resolve metadata."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory()
        tl = builder.create_timeline(tracks=[[clip]])

        assert 'Resolve_OTIO' in tl.metadata
        assert tl.metadata['Resolve_OTIO']['Resolve OTIO Meta Version'] == '1.0'

    @pytest.mark.fast
    def test_create_timeline_global_start_time(self, clip_factory):
        """Timeline should have global_start_time set."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory()
        tl = builder.create_timeline(tracks=[[clip]])

        assert tl.global_start_time is not None
        assert tl.global_start_time.rate == 30.0

    @pytest.mark.fast
    def test_clip_metadata_includes_video_id(self, clip_factory):
        """Created clips should have video_id in metadata."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory(video_id="XyZ98765432")
        tl = builder.create_timeline(tracks=[[clip]])

        video_track = tl.tracks[0]
        otio_clip = list(video_track)[0]
        assert otio_clip.metadata['video_id'] == "XyZ98765432"

    @pytest.mark.fast
    def test_clip_metadata_includes_keyword(self, clip_factory):
        """Created clips should have keyword in metadata."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory(keyword="funny cats")
        tl = builder.create_timeline(tracks=[[clip]])

        video_track = tl.tracks[0]
        otio_clip = list(video_track)[0]
        assert otio_clip.metadata['keyword'] == "funny cats"

    @pytest.mark.fast
    def test_format_path_adds_file_prefix(self):
        """_format_path should add file:/// prefix."""
        builder = CompilationTimelineBuilder()
        result = builder._format_path("E:\\v\\test\\video.mp4")

        assert result.startswith("file:///")
        assert "\\" not in result, "Should use forward slashes"

    @pytest.mark.fast
    def test_format_path_already_prefixed(self):
        """_format_path should not double-prefix file:// paths."""
        builder = CompilationTimelineBuilder()
        result = builder._format_path("file:///E:/v/test/video.mp4")

        assert result.count("file:///") == 1

    @pytest.mark.fast
    def test_custom_frame_rate(self, clip_factory):
        """Should respect custom frame rate for clips."""
        builder = CompilationTimelineBuilder(frame_rate=24.0)
        clip = clip_factory(actual_duration=10.0)
        tl = builder.create_timeline(tracks=[[clip]])

        assert tl.global_start_time.rate == 24.0
        video_track = tl.tracks[0]
        otio_clip = list(video_track)[0]
        assert otio_clip.source_range.duration.rate == 24.0

    @pytest.mark.fast
    def test_get_timeline_stats(self, clip_factory):
        """get_timeline_stats should return accurate statistics."""
        builder = CompilationTimelineBuilder()
        clip_a = clip_factory(actual_duration=30.0)
        clip_b = clip_factory(actual_duration=20.0)
        tl = builder.create_timeline(tracks=[[clip_a, clip_b], [clip_a]])

        stats = builder.get_timeline_stats(tl)

        assert stats['name'] == "Compilation"
        assert stats['num_tracks'] == 2, "Should count only video tracks"
        assert stats['total_clips'] == 3
        assert stats['tracks'][0]['clips'] == 2
        assert stats['tracks'][1]['clips'] == 1

    @pytest.mark.fast
    def test_clip_title_truncated_to_50(self, clip_factory):
        """Clip name should be truncated to 50 chars for long titles."""
        builder = CompilationTimelineBuilder()
        long_title = "A" * 100
        clip = clip_factory(title=long_title)
        tl = builder.create_timeline(tracks=[[clip]])

        video_track = tl.tracks[0]
        otio_clip = list(video_track)[0]
        assert len(otio_clip.name) <= 50

    @pytest.mark.fast
    def test_clip_no_title_uses_video_id(self, clip_factory):
        """Clip with empty title should fall back to video_id as name."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory(title="", video_id="fallbackVidId")
        tl = builder.create_timeline(tracks=[[clip]])

        video_track = tl.tracks[0]
        otio_clip = list(video_track)[0]
        assert otio_clip.name == "fallbackVidId"

    @pytest.mark.fast
    def test_audio_track_is_deep_copy(self, clip_factory):
        """Audio clips should be deep copies, not shared references."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory()
        tl = builder.create_timeline(tracks=[[clip]])

        video_clip = list(tl.tracks[0])[0]
        audio_clip = list(tl.tracks[1])[0]

        assert video_clip is not audio_clip

    @pytest.mark.fast
    def test_multiple_tracks_naming(self, clip_factory):
        """Tracks should be named V1, V2, ... and A1, A2, ..."""
        builder = CompilationTimelineBuilder()
        clip = clip_factory()
        tl = builder.create_timeline(tracks=[[clip], [clip], [clip]])

        assert tl.tracks[0].name.startswith("V1")
        assert tl.tracks[1].name.startswith("A1")
        assert tl.tracks[2].name.startswith("V2")
        assert tl.tracks[3].name.startswith("A2")
        assert tl.tracks[4].name.startswith("V3")
        assert tl.tracks[5].name.startswith("A3")
