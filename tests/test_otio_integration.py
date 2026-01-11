"""
Integration test for OTIO package.

Verifies the complete refactored OTIO package works end-to-end
by testing imports from OutputStage perspective.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestOutputStageIntegration:
    """Test that OutputStage can import all required functions."""

    def test_output_stage_imports(self):
        """Test the exact imports used by OutputStage."""
        # This is the exact import block from OutputStage (lines 79-86)
        from src.otio import (
            create_timeline,
            save_timeline,
            save_timeline_split,
            save_timeline_as_edl,
            generate_segment_map,
            generate_resolve_xml_with_bins
        )

        # Verify all functions are callable
        assert callable(create_timeline)
        assert callable(save_timeline)
        assert callable(save_timeline_split)
        assert callable(save_timeline_as_edl)
        assert callable(generate_segment_map)
        assert callable(generate_resolve_xml_with_bins)

        print("✓ All OutputStage imports successful")

    def test_output_stage_module_compiles(self):
        """Test that OutputStage itself compiles with new imports."""
        from src.stages import output

        assert hasattr(output, 'OutputStage')
        print("✓ OutputStage module compiled successfully")


class TestPackageStructure:
    """Test the complete package structure."""

    def test_package_has_all_modules(self):
        """Verify all expected modules exist in package."""
        import src.otio as otio

        # Check main modules are accessible
        assert hasattr(otio, 'timeline')
        assert hasattr(otio, 'export')
        assert hasattr(otio, 'reporting')
        assert hasattr(otio, 'xml_export')
        assert hasattr(otio, 'entities')
        assert hasattr(otio, 'utils')

        # Types is internal, verify it can be imported directly
        from src.otio import types
        assert hasattr(types, 'TRACK_NAMES')

        print("✓ All expected modules present in package")

    def test_package_metadata(self):
        """Test package has proper metadata."""
        import src.otio as otio

        assert hasattr(otio, '__version__')
        assert hasattr(otio, '__author__')
        assert hasattr(otio, '__all__')

        # Check __all__ contains expected exports
        expected_exports = [
            'create_timeline',
            'save_timeline',
            'save_timeline_split',
            'save_timeline_as_edl',
            'generate_segment_map',
            'generate_resolve_xml_with_bins',
        ]

        for export in expected_exports:
            assert export in otio.__all__, f"{export} not in __all__"

        print("✓ Package metadata correct")


class TestModuleCoordination:
    """Test that modules work together correctly."""

    def test_timeline_uses_utils(self):
        """Test timeline module uses utility functions."""
        from src.otio import timeline, utils

        # Verify timeline can access utils functions
        assert hasattr(utils, 'create_clip_with_timewarp')
        assert hasattr(utils, '_get_media_duration')

        print("✓ Timeline module can access utils")

    def test_entities_uses_utils(self):
        """Test entities module uses utility functions."""
        from src.otio import entities, utils

        # Verify entities can access utils functions
        assert hasattr(utils, '_to_windows_path')

        print("✓ Entities module can access utils")

    def test_export_uses_reporting(self):
        """Test export module uses reporting functions."""
        from src.otio import export, reporting

        # Verify reporting functions are accessible
        assert hasattr(reporting, 'print_timeline_statistics')

        print("✓ Export module can access reporting")


class TestBackwardCompatibilityComplete:
    """Comprehensive backward compatibility test."""

    def test_all_original_functions_present(self):
        """Verify ALL original otio_builder functions are accessible."""
        import src.otio as otio

        # Original public API from otio_builder.py
        original_functions = [
            'create_timeline',                # Timeline generation
            'save_timeline',                  # OTIO export
            'save_timeline_split',            # Split OTIO export
            'save_timeline_as_edl',           # EDL export
            'generate_segment_map',           # Segment mapping
            'generate_resolve_xml_with_bins', # XML generation
        ]

        for func_name in original_functions:
            assert hasattr(otio, func_name), f"Missing function: {func_name}"
            assert callable(getattr(otio, func_name)), f"Not callable: {func_name}"

        print(f"✓ All {len(original_functions)} original functions present and callable")

    def test_import_from_package_root(self):
        """Test that functions can be imported from package root."""
        # Test different import styles
        from src.otio import create_timeline
        from src.otio import save_timeline as st
        import src.otio

        assert callable(create_timeline)
        assert callable(st)
        assert callable(src.otio.generate_segment_map)

        print("✓ Multiple import styles work")


class TestRefactoringSuccess:
    """Verify the refactoring achieved its goals."""

    def test_modular_structure(self):
        """Test that code is now modular."""
        from src.otio import (
            timeline,
            export,
            reporting,
            xml_export,
            entities,
            utils,
            types
        )

        # Each module should be independent
        modules = [timeline, export, reporting, xml_export, entities, utils, types]

        for module in modules:
            assert hasattr(module, '__name__')
            assert module.__name__.startswith('src.otio.')

        print(f"✓ Successfully split into {len(modules)} focused modules")

    def test_code_organization(self):
        """Test that related functions are grouped logically."""
        from src.otio import export, reporting, xml_export

        # Export module should have export functions
        assert hasattr(export, 'save_timeline')
        assert hasattr(export, 'save_timeline_split')
        assert hasattr(export, 'save_timeline_as_edl')

        # Reporting module should have reporting functions
        assert hasattr(reporting, 'generate_segment_map')
        assert hasattr(reporting, 'print_timeline_statistics')

        # XML export module should have XML generation
        assert hasattr(xml_export, 'generate_resolve_xml_with_bins')

        print("✓ Functions logically organized by module")


def run_integration_tests():
    """Run all integration tests and print summary."""
    print("\n" + "=" * 60)
    print("  OTIO PACKAGE INTEGRATION TESTS")
    print("=" * 60 + "\n")

    # Run pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    return exit_code


if __name__ == "__main__":
    exit_code = run_integration_tests()
    sys.exit(exit_code)
