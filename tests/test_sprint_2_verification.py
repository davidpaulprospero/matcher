"""
Sprint 2 Verification Tests

US-001: Create sprint 2 OTIO verification test
Validates that OTIO module components can be imported and instantiated.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestOTIOModuleImports:
    """Test that OTIO modules can be imported."""

    def test_import_otio_timeline(self):
        """Test src.otio.timeline can be imported."""
        from src.otio import timeline
        assert timeline is not None

    def test_import_otio_tracks(self):
        """Test src.otio.tracks can be imported."""
        from src.otio import tracks
        assert tracks is not None

    def test_import_otio_utils(self):
        """Test src.otio.utils can be imported."""
        from src.otio import utils
        assert utils is not None

    def test_import_otio_export(self):
        """Test src.otio.export can be imported."""
        from src.otio import export
        assert export is not None

    def test_import_otio_package(self):
        """Test src.otio package can be imported."""
        from src import otio
        assert otio is not None
        assert hasattr(otio, '__version__')


class TestCreateTimelineFunction:
    """Test that create_timeline function exists and is callable."""

    def test_create_timeline_exists(self):
        """Test create_timeline function exists in the package."""
        from src.otio import create_timeline
        assert create_timeline is not None
        assert callable(create_timeline)

    def test_create_timeline_importable_from_timeline_module(self):
        """Test create_timeline is importable from timeline module."""
        from src.otio.timeline import create_timeline
        assert create_timeline is not None
        assert callable(create_timeline)

    def test_create_timeline_in_public_api(self):
        """Test create_timeline is in __all__."""
        from src import otio
        assert 'create_timeline' in otio.__all__


class TestTrackBuilderBaseClass:
    """Test that TrackBuilder base class can be imported."""

    def test_trackbuilder_exists(self):
        """Test TrackBuilder class exists."""
        from src.otio.tracks import TrackBuilder
        assert TrackBuilder is not None

    def test_trackbuilder_is_abstract(self):
        """Test TrackBuilder is an abstract base class."""
        from src.otio.tracks import TrackBuilder
        from abc import ABC
        assert issubclass(TrackBuilder, ABC)

    def test_trackbuilder_has_build_method(self):
        """Test TrackBuilder has abstract build method."""
        from src.otio.tracks import TrackBuilder
        assert hasattr(TrackBuilder, 'build')


class TestTrackBuilderInstantiation:
    """Test that all 7 track builders can be instantiated."""

    @pytest.fixture
    def mock_matches(self):
        """Create mock matches for track builder instantiation."""
        match = MagicMock()
        match.primary_match = MagicMock()
        match.primary_match.voiceover_segment = MagicMock()
        match.primary_match.voiceover_segment.start_time = 0.0
        match.primary_match.voiceover_segment.end_time = 5.0
        match.primary_match.video_segment = MagicMock()
        match.primary_match.video_segment.source_file = "test.mp4"
        match.primary_match.video_segment.start_time = 0.0
        match.primary_match.video_segment.end_time = 5.0
        match.alternatives = []
        match.secondary_matches = []
        match.strategy_matches = []
        return [match]

    @pytest.fixture
    def mock_config(self):
        """Create mock config for track builder instantiation."""
        config = MagicMock()
        config.output = MagicMock()
        config.output.include_alternatives = True
        config.output.num_alternatives = 2
        return config

    def test_primary_track_builder_instantiation(self, mock_matches, mock_config):
        """Test PrimaryTrackBuilder can be instantiated."""
        from src.otio.tracks import PrimaryTrackBuilder

        builder = PrimaryTrackBuilder(
            matches=mock_matches,
            config=mock_config,
            frame_rate=30.0
        )
        assert builder is not None
        assert builder.matches == mock_matches
        assert builder.config == mock_config
        assert builder.frame_rate == 30.0

    def test_alternative_track_builder_instantiation(self, mock_matches, mock_config):
        """Test AlternativeTrackBuilder can be instantiated."""
        from src.otio.tracks import AlternativeTrackBuilder

        builder = AlternativeTrackBuilder(
            matches=mock_matches,
            config=mock_config,
            frame_rate=30.0
        )
        assert builder is not None
        assert builder.matches == mock_matches

    def test_diversity_track_builder_instantiation(self, mock_matches, mock_config):
        """Test DiversityTrackBuilder can be instantiated."""
        from src.otio.tracks import DiversityTrackBuilder

        builder = DiversityTrackBuilder(
            matches=mock_matches,
            config=mock_config,
            frame_rate=30.0
        )
        assert builder is not None
        assert builder.matches == mock_matches

    def test_embedding_diversity_track_builder_instantiation(self, mock_matches, mock_config):
        """Test EmbeddingDiversityTrackBuilder can be instantiated."""
        from src.otio.tracks import EmbeddingDiversityTrackBuilder

        builder = EmbeddingDiversityTrackBuilder(
            matches=mock_matches,
            config=mock_config,
            frame_rate=30.0
        )
        assert builder is not None
        assert builder.matches == mock_matches

    def test_broll_track_builder_instantiation(self, mock_matches, mock_config):
        """Test BRollTrackBuilder can be instantiated."""
        from src.otio.tracks import BRollTrackBuilder

        builder = BRollTrackBuilder(
            matches=mock_matches,
            config=mock_config,
            frame_rate=30.0
        )
        assert builder is not None
        assert builder.matches == mock_matches

    def test_entity_image_track_builder_instantiation(self, mock_matches, mock_config):
        """Test EntityImageTrackBuilder can be instantiated."""
        from src.otio.tracks import EntityImageTrackBuilder

        builder = EntityImageTrackBuilder(
            matches=mock_matches,
            config=mock_config,
            frame_rate=30.0
        )
        assert builder is not None
        assert builder.matches == mock_matches

    def test_entity_video_track_builder_instantiation(self, mock_matches, mock_config):
        """Test EntityVideoTrackBuilder can be instantiated."""
        from src.otio.tracks import EntityVideoTrackBuilder

        builder = EntityVideoTrackBuilder(
            matches=mock_matches,
            config=mock_config,
            frame_rate=30.0
        )
        assert builder is not None
        assert builder.matches == mock_matches

    def test_all_track_builders_count(self):
        """Verify all 7 track builders exist."""
        from src.otio.tracks import (
            PrimaryTrackBuilder,
            AlternativeTrackBuilder,
            DiversityTrackBuilder,
            EmbeddingDiversityTrackBuilder,
            BRollTrackBuilder,
            EntityImageTrackBuilder,
            EntityVideoTrackBuilder,
        )

        builders = [
            PrimaryTrackBuilder,
            AlternativeTrackBuilder,
            DiversityTrackBuilder,
            EmbeddingDiversityTrackBuilder,
            BRollTrackBuilder,
            EntityImageTrackBuilder,
            EntityVideoTrackBuilder,
        ]
        assert len(builders) == 7, "Expected exactly 7 track builders"


class TestGetTrackBuilderFactory:
    """Test the get_track_builder factory function."""

    @pytest.fixture
    def mock_matches(self):
        """Create mock matches."""
        return [MagicMock()]

    @pytest.fixture
    def mock_config(self):
        """Create mock config."""
        config = MagicMock()
        config.output = MagicMock()
        config.output.include_alternatives = True
        config.output.num_alternatives = 2
        return config

    def test_get_track_builder_v1(self, mock_matches, mock_config):
        """Test get_track_builder returns PrimaryTrackBuilder for V1."""
        from src.otio.tracks import get_track_builder, PrimaryTrackBuilder

        builder = get_track_builder(0, mock_matches, mock_config, 30.0)
        assert isinstance(builder, PrimaryTrackBuilder)

    def test_get_track_builder_v2(self, mock_matches, mock_config):
        """Test get_track_builder returns AlternativeTrackBuilder for V2."""
        from src.otio.tracks import get_track_builder, AlternativeTrackBuilder

        builder = get_track_builder(1, mock_matches, mock_config, 30.0)
        assert isinstance(builder, AlternativeTrackBuilder)

    def test_get_track_builder_v8(self, mock_matches, mock_config):
        """Test get_track_builder returns BRollTrackBuilder for V8."""
        from src.otio.tracks import get_track_builder, BRollTrackBuilder

        builder = get_track_builder(7, mock_matches, mock_config, 30.0)
        assert isinstance(builder, BRollTrackBuilder)

    def test_get_track_builder_v10(self, mock_matches, mock_config):
        """Test get_track_builder returns EntityVideoTrackBuilder for V10."""
        from src.otio.tracks import get_track_builder, EntityVideoTrackBuilder

        builder = get_track_builder(9, mock_matches, mock_config, 30.0)
        assert isinstance(builder, EntityVideoTrackBuilder)

    def test_get_track_builder_invalid_index(self, mock_matches, mock_config):
        """Test get_track_builder raises ValueError for invalid index."""
        from src.otio.tracks import get_track_builder

        with pytest.raises(ValueError):
            get_track_builder(99, mock_matches, mock_config, 30.0)


class TestOTIOExportFunctions:
    """Test export functions are importable."""

    def test_save_timeline_exists(self):
        """Test save_timeline function exists."""
        from src.otio.export import save_timeline
        assert save_timeline is not None
        assert callable(save_timeline)

    def test_save_timeline_split_exists(self):
        """Test save_timeline_split function exists."""
        from src.otio.export import save_timeline_split
        assert save_timeline_split is not None
        assert callable(save_timeline_split)

    def test_save_timeline_as_edl_exists(self):
        """Test save_timeline_as_edl function exists."""
        from src.otio.export import save_timeline_as_edl
        assert save_timeline_as_edl is not None
        assert callable(save_timeline_as_edl)

    def test_export_functions_in_public_api(self):
        """Test export functions are in __all__."""
        from src import otio
        assert 'save_timeline' in otio.__all__
        assert 'save_timeline_split' in otio.__all__
        assert 'save_timeline_as_edl' in otio.__all__
