"""
Unit tests for OTIO package modules.

Tests the refactored OTIO package to ensure functionality is preserved.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestOTIOImports:
    """Test that all OTIO modules can be imported."""

    @pytest.mark.fast
    def test_import_public_api(self):
        """Test importing from main package."""
        from src.otio import (
            create_timeline,
            save_timeline,
            save_timeline_split,
            save_timeline_as_edl,
            generate_segment_map,
            generate_resolve_xml_with_bins
        )
        assert callable(create_timeline)
        assert callable(save_timeline)
        assert callable(save_timeline_split)
        assert callable(save_timeline_as_edl)
        assert callable(generate_segment_map)
        assert callable(generate_resolve_xml_with_bins)

    @pytest.mark.fast
    def test_import_utils(self):
        """Test importing utility functions."""
        from src.otio.utils import (
            _to_windows_path,
            get_confidence_color,
            create_clip_with_timewarp,
            frames_to_tc
        )
        assert callable(_to_windows_path)
        assert callable(get_confidence_color)
        assert callable(create_clip_with_timewarp)
        assert callable(frames_to_tc)

    @pytest.mark.fast
    def test_import_types(self):
        """Test importing type definitions."""
        from src.otio.types import TRACK_NAMES, DEFAULT_FRAME_RATE
        assert isinstance(TRACK_NAMES, list)
        assert len(TRACK_NAMES) == 10
        assert DEFAULT_FRAME_RATE == 30.0


class TestUtilityFunctions:
    """Test utility functions work correctly."""

    @pytest.mark.fast
    def test_get_confidence_color(self):
        """Test confidence color mapping."""
        from src.otio.utils import get_confidence_color

        assert get_confidence_color(0.9) == "GREEN"
        assert get_confidence_color(0.7) == "CYAN"
        assert get_confidence_color(0.5) == "YELLOW"
        assert get_confidence_color(0.3) == "ORANGE"
        assert get_confidence_color(0.1) == "RED"

    @pytest.mark.fast
    def test_frames_to_tc(self):
        """Test timecode conversion."""
        from src.otio.utils import frames_to_tc

        # 0 frames = 00:00:00:00
        assert frames_to_tc(0, 30.0) == "00:00:00:00"

        # 30 frames (1 second at 30fps) = 00:00:01:00
        assert frames_to_tc(30, 30.0) == "00:00:01:00"

        # 1800 frames (1 minute at 30fps) = 00:01:00:00
        assert frames_to_tc(1800, 30.0) == "00:01:00:00"

    @pytest.mark.fast
    def test_to_windows_path(self):
        """Test Windows path conversion."""
        from src.otio.utils import _to_windows_path

        # Test path conversion (basic check)
        result = _to_windows_path("test/path/file.mp4")
        assert isinstance(result, str)
        assert "\\" in result or "/" not in result  # Should use backslashes

    @pytest.mark.fast
    def test_escape_xml(self):
        """Test XML escaping."""
        from src.otio.utils import escape_xml

        assert escape_xml("test & data") == "test &amp; data"
        assert escape_xml("test < data") == "test &lt; data"
        assert escape_xml("test > data") == "test &gt; data"
        assert escape_xml('test "data"') == "test &quot;data&quot;"


class TestTrackNames:
    """Test track naming constants."""

    @pytest.mark.fast
    def test_track_names_count(self):
        """Verify we have names for all 10 tracks."""
        from src.otio.types import TRACK_NAMES

        assert len(TRACK_NAMES) == 10

    @pytest.mark.fast
    def test_track_names_content(self):
        """Verify track names are correct."""
        from src.otio.types import TRACK_NAMES

        assert TRACK_NAMES[0] == "Primary Video"
        assert TRACK_NAMES[1] == "Alternative Video 1"
        assert TRACK_NAMES[8] == "Entity Images (Google)"
        assert TRACK_NAMES[9] == "Stock Videos (Pexels/Pixabay)"


class TestModuleCompilation:
    """Test that all modules compile without errors."""

    @pytest.mark.fast
    def test_timeline_module_loads(self):
        """Test timeline module can be imported."""
        from src.otio import timeline
        assert hasattr(timeline, 'create_timeline')

    @pytest.mark.fast
    def test_export_module_loads(self):
        """Test export module can be imported."""
        from src.otio import export
        assert hasattr(export, 'save_timeline')
        assert hasattr(export, 'save_timeline_split')
        assert hasattr(export, 'save_timeline_as_edl')

    @pytest.mark.fast
    def test_reporting_module_loads(self):
        """Test reporting module can be imported."""
        from src.otio import reporting
        assert hasattr(reporting, 'generate_segment_map')
        assert hasattr(reporting, 'print_timeline_statistics')

    @pytest.mark.fast
    def test_xml_export_module_loads(self):
        """Test XML export module can be imported."""
        from src.otio import xml_export
        assert hasattr(xml_export, 'generate_resolve_xml_with_bins')

    @pytest.mark.fast
    def test_entities_module_loads(self):
        """Test entities module can be imported."""
        from src.otio import entities
        assert hasattr(entities, '_add_entity_images_to_track')
        assert hasattr(entities, '_add_entity_videos_to_track')


class TestBackwardCompatibility:
    """Test that the refactored code maintains backward compatibility."""

    @pytest.mark.fast
    def test_all_functions_accessible(self):
        """Test all public API functions are accessible."""
        import src.otio as otio

        # These were the original 6 functions in otio_builder.py
        assert hasattr(otio, 'create_timeline')
        assert hasattr(otio, 'save_timeline')
        assert hasattr(otio, 'save_timeline_split')
        assert hasattr(otio, 'save_timeline_as_edl')
        assert hasattr(otio, 'generate_segment_map')
        assert hasattr(otio, 'generate_resolve_xml_with_bins')

    @pytest.mark.fast
    def test_module_version(self):
        """Test module has version info."""
        from src.otio import __version__, __author__

        assert isinstance(__version__, str)
        assert isinstance(__author__, str)
        assert "2.0" in __version__


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
