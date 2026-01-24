"""
Sprint 6 Verification Tests - Compilation Module

Tests to verify compilation module improvements are implemented correctly.
US-001: Create sprint 6 compilation verification test
"""

import pytest


class TestCompilationModuleImports:
    """Test that compilation module can be imported."""

    def test_compilation_state_import(self):
        """Test CompilationState can be imported from src/compilation."""
        from src.compilation import CompilationState
        assert CompilationState is not None

    def test_downloaded_clip_import(self):
        """Test DownloadedClip can be imported from src/compilation."""
        from src.compilation import DownloadedClip
        assert DownloadedClip is not None

    def test_gap_report_import(self):
        """Test GapReport can be imported from src/compilation."""
        from src.compilation import GapReport
        assert GapReport is not None

    def test_compilation_candidate_import(self):
        """Test CompilationCandidate can be imported from src/compilation."""
        from src.compilation import CompilationCandidate
        assert CompilationCandidate is not None


class TestCompilationOrchestratorImport:
    """Test orchestrator can be imported."""

    def test_orchestrator_import(self):
        """Test CompilationOrchestrator can be imported."""
        from src.compilation import CompilationOrchestrator
        assert CompilationOrchestrator is not None

    def test_load_config_import(self):
        """Test load_compilation_config can be imported."""
        from src.compilation import load_compilation_config
        assert load_compilation_config is not None

    def test_get_default_config_import(self):
        """Test get_default_config can be imported."""
        from src.compilation import get_default_config
        assert get_default_config is not None


class TestCompilationStagesImport:
    """Test that stage classes can be imported."""

    def test_arrange_stage_import(self):
        """Test ArrangeStage can be imported from src/compilation.stages."""
        from src.compilation.stages import ArrangeStage
        assert ArrangeStage is not None

    def test_clip_distributor_import(self):
        """Test ClipDistributor can be imported from src/compilation.stages."""
        from src.compilation.stages import ClipDistributor
        assert ClipDistributor is not None

    def test_gap_check_stage_import(self):
        """Test GapCheckStage can be imported from src/compilation.stages."""
        from src.compilation.stages import GapCheckStage
        assert GapCheckStage is not None

    def test_download_stage_import(self):
        """Test DownloadStage can be imported from src/compilation.stages."""
        from src.compilation.stages import DownloadStage
        assert DownloadStage is not None

    def test_output_stage_import(self):
        """Test OutputStage can be imported from src/compilation.stages."""
        from src.compilation.stages import OutputStage
        assert OutputStage is not None

    def test_search_stage_import(self):
        """Test SearchStage can be imported from src/compilation.stages."""
        from src.compilation.stages import SearchStage
        assert SearchStage is not None

    def test_filter_stage_import(self):
        """Test FilterStage can be imported from src/compilation.stages."""
        from src.compilation.stages import FilterStage
        assert FilterStage is not None


class TestCompilationStateStructure:
    """Test CompilationState dataclass has required fields."""

    def test_state_has_topic_field(self):
        """Test CompilationState has topic field."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test topic",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert state.topic == "test topic"

    def test_state_has_keywords_field(self):
        """Test CompilationState has keywords field."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1", "kw2"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert state.keywords == ["kw1", "kw2"]

    def test_state_has_target_duration_field(self):
        """Test CompilationState has target_duration field."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=120,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert state.target_duration == 120

    def test_state_has_num_tracks_field(self):
        """Test CompilationState has num_tracks field."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=4,
            project_dir="/tmp/test"
        )
        assert state.num_tracks == 4

    def test_state_has_project_dir_field(self):
        """Test CompilationState has project_dir field."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/myproject"
        )
        assert state.project_dir == "/tmp/myproject"

    def test_state_has_retry_count_field(self):
        """Test CompilationState has retry_count field with default 0."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert state.retry_count == 0

    def test_state_has_filtered_candidates_list(self):
        """Test CompilationState has filtered_candidates list."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert isinstance(state.filtered_candidates, list)

    def test_state_has_downloaded_clips_list(self):
        """Test CompilationState has downloaded_clips list."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert isinstance(state.downloaded_clips, list)

    def test_state_has_arranged_tracks_list(self):
        """Test CompilationState has arranged_tracks list."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert isinstance(state.arranged_tracks, list)

    def test_state_has_output_files_list(self):
        """Test CompilationState has output_files list."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        assert isinstance(state.output_files, list)

    def test_state_all_used_keywords_property(self):
        """Test CompilationState has all_used_keywords property."""
        from src.compilation import CompilationState
        state = CompilationState(
            topic="test",
            keywords=["kw1", "kw2", "kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        # Should return unique keywords
        used = state.all_used_keywords
        assert "kw1" in used
        assert "kw2" in used

    def test_state_get_project_path_method(self):
        """Test CompilationState has get_project_path() method."""
        from src.compilation import CompilationState
        from pathlib import Path
        state = CompilationState(
            topic="test",
            keywords=["kw1"],
            target_duration=60,
            num_tracks=2,
            project_dir="/tmp/test"
        )
        path = state.get_project_path()
        assert isinstance(path, Path)


class TestDownloadedClipStructure:
    """Test DownloadedClip dataclass has required fields."""

    def test_clip_has_file_field(self):
        """Test DownloadedClip has file field."""
        from src.compilation import DownloadedClip
        clip = DownloadedClip(
            file="/path/to/video.mp4",
            video_id="abc123",
            actual_duration=30.0,
            keyword="test"
        )
        assert clip.file == "/path/to/video.mp4"

    def test_clip_has_video_id_field(self):
        """Test DownloadedClip has video_id field."""
        from src.compilation import DownloadedClip
        clip = DownloadedClip(
            file="/path/to/video.mp4",
            video_id="xyz789",
            actual_duration=30.0,
            keyword="test"
        )
        assert clip.video_id == "xyz789"

    def test_clip_has_actual_duration_field(self):
        """Test DownloadedClip has actual_duration field."""
        from src.compilation import DownloadedClip
        clip = DownloadedClip(
            file="/path/to/video.mp4",
            video_id="abc123",
            actual_duration=45.5,
            keyword="test"
        )
        assert clip.actual_duration == 45.5

    def test_clip_has_keyword_field(self):
        """Test DownloadedClip has keyword field."""
        from src.compilation import DownloadedClip
        clip = DownloadedClip(
            file="/path/to/video.mp4",
            video_id="abc123",
            actual_duration=30.0,
            keyword="funny cats"
        )
        assert clip.keyword == "funny cats"

    def test_clip_has_title_field(self):
        """Test DownloadedClip has optional title field."""
        from src.compilation import DownloadedClip
        clip = DownloadedClip(
            file="/path/to/video.mp4",
            video_id="abc123",
            actual_duration=30.0,
            keyword="test",
            title="My Video Title"
        )
        assert clip.title == "My Video Title"


class TestGapReportStructure:
    """Test GapReport dataclass has required fields."""

    def test_gap_report_has_complete_field(self):
        """Test GapReport has complete field."""
        from src.compilation import GapReport
        report = GapReport(
            complete=True,
            total_shortfall=0,
            track_shortfalls=[0, 0],
            new_keywords=[]
        )
        assert report.complete is True

    def test_gap_report_has_total_shortfall_field(self):
        """Test GapReport has total_shortfall field."""
        from src.compilation import GapReport
        report = GapReport(
            complete=False,
            total_shortfall=120.5,
            track_shortfalls=[60.0, 60.5],
            new_keywords=[]
        )
        assert report.total_shortfall == 120.5

    def test_gap_report_has_track_shortfalls_field(self):
        """Test GapReport has track_shortfalls field."""
        from src.compilation import GapReport
        report = GapReport(
            complete=False,
            total_shortfall=100,
            track_shortfalls=[50, 30, 20],
            new_keywords=[]
        )
        assert report.track_shortfalls == [50, 30, 20]

    def test_gap_report_has_new_keywords_field(self):
        """Test GapReport has new_keywords field."""
        from src.compilation import GapReport
        report = GapReport(
            complete=False,
            total_shortfall=100,
            track_shortfalls=[50, 50],
            new_keywords=["broader", "wider"]
        )
        assert report.new_keywords == ["broader", "wider"]


class TestCompilationCandidateStructure:
    """Test CompilationCandidate dataclass has required fields."""

    def test_candidate_has_video_id_field(self):
        """Test CompilationCandidate has video_id field."""
        from src.compilation import CompilationCandidate
        candidate = CompilationCandidate(
            video_id="abc123",
            title="Test Video",
            duration=30.0,
            keyword="test"
        )
        assert candidate.video_id == "abc123"

    def test_candidate_has_title_field(self):
        """Test CompilationCandidate has title field."""
        from src.compilation import CompilationCandidate
        candidate = CompilationCandidate(
            video_id="abc123",
            title="Amazing Video",
            duration=30.0,
            keyword="test"
        )
        assert candidate.title == "Amazing Video"

    def test_candidate_has_duration_field(self):
        """Test CompilationCandidate has duration field."""
        from src.compilation import CompilationCandidate
        candidate = CompilationCandidate(
            video_id="abc123",
            title="Test",
            duration=45.5,
            keyword="test"
        )
        assert candidate.duration == 45.5

    def test_candidate_has_keyword_field(self):
        """Test CompilationCandidate has keyword field."""
        from src.compilation import CompilationCandidate
        candidate = CompilationCandidate(
            video_id="abc123",
            title="Test",
            duration=30.0,
            keyword="funny cats"
        )
        assert candidate.keyword == "funny cats"


class TestClipDistributorFunctionality:
    """Test ClipDistributor distribution logic."""

    def test_distributor_creates_correct_number_of_tracks(self):
        """Test ClipDistributor creates the requested number of tracks."""
        from src.compilation.stages import ClipDistributor
        from src.compilation import DownloadedClip

        clips = [
            DownloadedClip(file=f"/path/{i}.mp4", video_id=f"vid{i}", actual_duration=30.0, keyword="test")
            for i in range(10)
        ]

        distributor = ClipDistributor(clips=clips, target_duration=60, num_tracks=3)
        tracks = distributor.distribute()

        assert len(tracks) == 3

    def test_distributor_no_clip_in_multiple_tracks(self):
        """Test no clip appears in more than one track."""
        from src.compilation.stages import ClipDistributor
        from src.compilation import DownloadedClip

        clips = [
            DownloadedClip(file=f"/path/{i}.mp4", video_id=f"vid{i}", actual_duration=20.0, keyword="test")
            for i in range(12)
        ]

        distributor = ClipDistributor(clips=clips, target_duration=60, num_tracks=4)
        tracks = distributor.distribute()

        # Collect all video_ids across tracks
        all_ids = []
        for track in tracks:
            all_ids.extend([clip.video_id for clip in track])

        # Check no duplicates
        assert len(all_ids) == len(set(all_ids)), "Found duplicate video_id across tracks"

    def test_distributor_stats_include_all_fields(self):
        """Test get_distribution_stats returns expected fields."""
        from src.compilation.stages import ClipDistributor
        from src.compilation import DownloadedClip

        clips = [
            DownloadedClip(file=f"/path/{i}.mp4", video_id=f"vid{i}", actual_duration=30.0, keyword="test")
            for i in range(6)
        ]

        distributor = ClipDistributor(clips=clips, target_duration=60, num_tracks=2)
        tracks = distributor.distribute()
        stats = distributor.get_distribution_stats(tracks)

        assert 'num_tracks' in stats
        assert 'complete_tracks' in stats
        assert 'total_clips' in stats
        assert 'total_duration' in stats
        assert 'track_durations' in stats


class TestArrangeStageStructure:
    """Test ArrangeStage has required attributes."""

    def test_arrange_stage_has_name(self):
        """Test ArrangeStage has name attribute."""
        from src.compilation.stages import ArrangeStage
        assert hasattr(ArrangeStage, 'name')
        assert ArrangeStage.name == "ARRANGE"

    def test_arrange_stage_has_description(self):
        """Test ArrangeStage has description attribute."""
        from src.compilation.stages import ArrangeStage
        assert hasattr(ArrangeStage, 'description')

    def test_arrange_stage_has_run_method(self):
        """Test ArrangeStage has run method."""
        from src.compilation.stages import ArrangeStage
        assert hasattr(ArrangeStage, 'run')
        assert callable(getattr(ArrangeStage, 'run', None))
