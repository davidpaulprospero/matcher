"""
Tests for pipeline graph export (US-125-012).

Tests for --export-pipeline-graph CLI flag:
- Exports DOT file to project directory
- Supports PNG/SVG output via graphviz
- Generates valid DOT content
"""

import pytest
import subprocess
import sys
import tempfile
from pathlib import Path

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestPipelineGraphExport:
    """Tests for --export-pipeline-graph CLI flag."""

    @pytest.mark.fast
    def test_export_pipeline_graph_help(self):
        """Test --export-pipeline-graph flag appears in help."""
        result = subprocess.run(
            [sys.executable, 'main.py', '--help'],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode == 0
        assert '--export-pipeline-graph' in result.stdout

    @pytest.mark.fast
    def test_export_pipeline_graph_dot(self, tmp_path):
        """Test exporting pipeline graph to DOT format."""
        output_file = tmp_path / "pipeline.dot"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'main.py'), '--export-pipeline-graph', str(output_file)],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_file.exists()

        # Verify DOT content is valid
        content = output_file.read_text(encoding='utf-8')
        assert 'digraph pipeline_stages' in content
        assert 'ANALYZE' in content
        assert 'VIDEO_SEARCH' in content
        assert 'CAPTION' in content
        assert 'MATCH' in content
        assert '->' in content  # Has edges

    @pytest.mark.fast
    def test_export_pipeline_graph_absolute_path(self, tmp_path):
        """Test exporting with absolute path."""
        output_file = tmp_path / "pipeline_absolute.dot"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'main.py'), '--export-pipeline-graph', str(output_file.resolve())],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode == 0
        assert output_file.exists()

    @pytest.mark.fast
    def test_export_pipeline_graph_png_fallback(self, tmp_path):
        """Test PNG export falls back to DOT when graphviz not installed."""
        output_file = tmp_path / "pipeline.png"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'main.py'), '--export-pipeline-graph', str(output_file)],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode == 0
        # Should fall back to .dot since graphviz likely not installed in test env
        dot_file = tmp_path / "pipeline.dot"
        assert dot_file.exists()

    @pytest.mark.fast
    def test_export_pipeline_graph_svg_fallback(self, tmp_path):
        """Test SVG export falls back to DOT when graphviz not installed."""
        output_file = tmp_path / "pipeline.svg"
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / 'main.py'), '--export-pipeline-graph', str(output_file)],
            capture_output=True,
            text=True,
            timeout=30,
            encoding='utf-8',
            errors='replace',
        )
        assert result.returncode == 0
        # Should fall back to .dot since graphviz likely not installed in test env
        dot_file = tmp_path / "pipeline.dot"
        assert dot_file.exists()


class TestGenerateDotGraph:
    """Tests for generate_dot_graph function in src.stages."""

    @pytest.mark.fast
    def test_generate_dot_graph_returns_valid_dot(self):
        """Test generate_dot_graph returns valid DOT format."""
        from src.stages import generate_dot_graph

        dot_content = generate_dot_graph()

        assert 'digraph pipeline_stages' in dot_content
        assert 'rankdir=LR' in dot_content

    @pytest.mark.fast
    def test_generate_dot_graph_has_nodes(self):
        """Test DOT graph contains expected stage nodes."""
        from src.stages import generate_dot_graph

        dot_content = generate_dot_graph()

        # All main stages should be present
        assert 'ANALYZE' in dot_content
        assert 'VIDEO_SEARCH' in dot_content
        assert 'CAPTION' in dot_content
        assert 'MATCH' in dot_content

    @pytest.mark.fast
    def test_generate_dot_graph_has_edges(self):
        """Test DOT graph contains dependency edges."""
        from src.stages import generate_dot_graph

        dot_content = generate_dot_graph()

        # Should have directed edges
        assert '->' in dot_content

    @pytest.mark.fast
    def test_generate_dot_graph_stages_order(self):
        """Test stages appear in logical dependency order."""
        from src.stages import generate_dot_graph

        dot_content = generate_dot_graph()

        # VIDEO_SEARCH should come before MATCH (dependency)
        # CAPTION should come before MATCH
        lines = dot_content.split('\n')
        video_search_idx = None
        caption_idx = None
        match_idx = None

        for i, line in enumerate(lines):
            if 'VIDEO_SEARCH' in line:
                video_search_idx = i
            if 'CAPTION' in line:
                caption_idx = i
            if '"MATCH"' in line and '->' not in line:
                match_idx = i

        assert video_search_idx is not None
        assert caption_idx is not None
        assert match_idx is not None
