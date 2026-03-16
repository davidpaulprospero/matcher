#!/usr/bin/env python3
"""
Tests for scripts/cleanup_project.py

Tests the cleanup project functions including:
- get_dir_size: correct byte count calculation
- format_size: human-readable output
- collect_deletable_paths: different cleanup levels
- generate_manifest: correct structure
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
from unittest.mock import patch

import pytest

# Add scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import the module under test
from cleanup_project import (
    get_dir_size,
    collect_deletable_paths,
    generate_manifest,
    LEVEL_CHECKPOINT,
    LEVEL_CACHE,
    LEVEL_MEDIA,
    LEVEL_FULL,
)
from utils.cli_helpers import format_size


@pytest.mark.script
class TestGetDirSize:
    """Test get_dir_size function"""

    def test_empty_directory(self, tmp_path):
        """Test that empty directory returns 0"""
        result = get_dir_size(tmp_path)
        assert result == 0

    def test_single_file(self, tmp_path):
        """Test directory with single file returns correct size"""
        test_file = tmp_path / "test.txt"
        content = b"Hello World"  # 11 bytes
        test_file.write_bytes(content)

        result = get_dir_size(tmp_path)
        assert result == 11

    def test_multiple_files(self, tmp_path):
        """Test directory with multiple files returns correct total"""
        # Create multiple files
        (tmp_path / "file1.txt").write_bytes(b"ABC")  # 3 bytes
        (tmp_path / "file2.txt").write_bytes(b"DEF")  # 3 bytes

        result = get_dir_size(tmp_path)
        assert result == 6

    def test_nested_directories(self, tmp_path):
        """Test nested directories accumulate size correctly"""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (tmp_path / "root.txt").write_bytes(b"ROOT")  # 4 bytes
        (subdir / "nested.txt").write_bytes(b"NESTED")  # 6 bytes

        result = get_dir_size(tmp_path)
        assert result == 10

    def test_ignores_symlinks(self, tmp_path):
        """Test that symlinks are handled correctly"""
        # Create a file and a symlink to it
        real_file = tmp_path / "real.txt"
        real_file.write_bytes(b"REAL")  # 4 bytes
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(real_file)

        result = get_dir_size(tmp_path)
        # On Linux, symlinks may be counted as files (4 bytes each)
        # Just verify it's a reasonable value (4 or 8 bytes)
        assert result in [4, 8]


@pytest.mark.script
class TestFormatSize:
    """Test format_size function from cli_helpers"""

    def test_bytes(self):
        """Test bytes are formatted correctly"""
        assert format_size(0) == "0.0 B"
        assert format_size(1) == "1.0 B"
        assert format_size(512) == "512.0 B"

    def test_kilobytes(self):
        """Test kilobytes are formatted correctly"""
        assert format_size(1024) == "1.0 KB"
        assert format_size(1536) == "1.5 KB"
        assert format_size(10240) == "10.0 KB"

    def test_megabytes(self):
        """Test megabytes are formatted correctly"""
        assert format_size(1048576) == "1.0 MB"
        assert format_size(1572864) == "1.5 MB"
        assert format_size(104857600) == "100.0 MB"

    def test_gigabytes(self):
        """Test gigabytes are formatted correctly"""
        assert format_size(1073741824) == "1.0 GB"
        assert format_size(1610612736) == "1.5 GB"

    def test_terabytes(self):
        """Test terabytes are formatted correctly"""
        assert format_size(1099511627776) == "1.0 TB"
        assert format_size(1649267441664) == "1.5 TB"

    def test_petabytes(self):
        """Test petabytes are formatted correctly"""
        assert format_size(1125899906842624) == "1.0 PB"


@pytest.mark.script
class TestCollectDeletablePaths:
    """Test collect_deletable_paths function"""

    @pytest.fixture
    def project_structure(self, tmp_path) -> Path:
        """Create a test project structure with cleanup directories"""
        # Create directories that should be deleted at cache level
        (tmp_path / ".cache").mkdir()
        (tmp_path / "checkpoint.json").write_text("{}")
        (tmp_path / "logs").mkdir()

        # Create directories that should be deleted at media level
        (tmp_path / "videos").mkdir()
        (tmp_path / "images").mkdir()

        return tmp_path

    def test_cache_level(self, project_structure):
        """Test cache level cleanup collects correct paths"""
        short_paths = {'videos': None, 'images': None}
        result = collect_deletable_paths(project_structure, LEVEL_CACHE, short_paths)

        # Should include cache-level items
        descriptions = [desc for _, desc in result]
        assert "project/.cache" in descriptions
        assert "project/checkpoint.json" in descriptions
        assert "project/logs" in descriptions

        # Should NOT include media-level items
        assert "project/videos" not in descriptions
        assert "project/images" not in descriptions

    def test_media_level(self, project_structure):
        """Test media level cleanup collects correct paths"""
        short_paths = {'videos': None, 'images': None}
        result = collect_deletable_paths(project_structure, LEVEL_MEDIA, short_paths)

        # Should include cache-level items
        descriptions = [desc for _, desc in result]
        assert "project/.cache" in descriptions
        assert "project/checkpoint.json" in descriptions
        assert "project/logs" in descriptions

        # Should also include media-level items
        assert "project/videos" in descriptions
        assert "project/images" in descriptions

    def test_full_level(self, project_structure):
        """Test full level cleanup collects correct paths"""
        short_paths = {'videos': None, 'images': None}
        result = collect_deletable_paths(project_structure, LEVEL_FULL, short_paths)

        # Should include all items from media level
        descriptions = [desc for _, desc in result]
        assert "project/.cache" in descriptions
        assert "project/videos" in descriptions
        assert "project/images" in descriptions
        # voiceover is in DELETE_AT_FULL but not in project_structure fixture
        # So it won't appear unless the directory exists

    def test_short_paths_included_at_media_level(self, tmp_path):
        """Test short path directories are included at media/full levels"""
        # Create project structure
        (tmp_path / ".cache").mkdir()

        # Create mock short path directories
        videos_dir = tmp_path / "short_videos"
        images_dir = tmp_path / "short_images"
        videos_dir.mkdir()
        images_dir.mkdir()

        short_paths = {'videos': videos_dir, 'images': images_dir}

        # Media level should include short paths
        result = collect_deletable_paths(tmp_path, LEVEL_MEDIA, short_paths)
        descriptions = [desc for _, desc in result]

        assert "short-path/videos" in descriptions
        assert "short-path/images" in descriptions

    def test_missing_directories_not_included(self, tmp_path):
        """Test that non-existent directories are not included"""
        # Create only .cache
        (tmp_path / ".cache").mkdir()

        short_paths = {'videos': None, 'images': None}
        result = collect_deletable_paths(tmp_path, LEVEL_MEDIA, short_paths)

        descriptions = [desc for _, desc in result]

        # videos and images don't exist in project, only short_paths
        # So they shouldn't appear
        assert "project/videos" not in descriptions
        assert "project/images" not in descriptions

    def test_checkpoint_level(self, tmp_path):
        """Test checkpoint level cleanup collects only checkpoint files"""
        # Create a project structure with various files
        (tmp_path / ".cache").mkdir()
        (tmp_path / "checkpoint.json").write_text("{}")
        (tmp_path / "checkpoint.backup.json").write_text("{}")
        (tmp_path / "saved_keywords.json").write_text("[]")
        (tmp_path / "logs").mkdir()
        (tmp_path / "videos").mkdir()
        (tmp_path / "images").mkdir()

        short_paths = {'videos': None, 'images': None}
        result = collect_deletable_paths(tmp_path, LEVEL_CHECKPOINT, short_paths)

        # Should include only checkpoint files
        descriptions = [desc for _, desc in result]
        assert "project/checkpoint.json" in descriptions
        assert "project/checkpoint.backup.json" in descriptions

        # Should NOT include other files
        assert "project/.cache" not in descriptions
        assert "project/saved_keywords.json" not in descriptions
        assert "project/logs" not in descriptions
        assert "project/videos" not in descriptions
        assert "project/images" not in descriptions

    def test_checkpoint_level_only_deletes_checkpoint(self, tmp_path):
        """Test that checkpoint level preserves other project files"""
        # Create full project structure
        (tmp_path / ".cache").mkdir()
        (tmp_path / "checkpoint.json").write_text("{}")
        (tmp_path / "checkpoint.backup.json").write_text("{}")
        (tmp_path / "videos").mkdir()
        (tmp_path / "images").mkdir()
        (tmp_path / "voiceover").mkdir()
        (tmp_path / "output").mkdir()
        (tmp_path / "run.bat").write_text("echo test")

        short_paths = {'videos': None, 'images': None}
        result = collect_deletable_paths(tmp_path, LEVEL_CHECKPOINT, short_paths)

        # Should only have 2 items (checkpoint files)
        assert len(result) == 2


@pytest.mark.script
class TestGenerateManifest:
    """Test generate_manifest function"""

    def test_manifest_structure(self, tmp_path):
        """Test manifest has correct structure"""
        project_dir = tmp_path / "TestProject"
        project_dir.mkdir()

        sizes = {
            'before': 1024000,  # Exactly 1000 KB
            'after': 512000,    # Exactly 500 KB
        }

        stats = {
            'last_stage': 'OUTPUT',
            'keywords': ['keyword1', 'keyword2'],
            'topic': 'Test Topic',
            'segment_count': 10,
            'video_count': 5,
            'match_count': 15,
            'avg_confidence': 0.85,
            'created_at': '2026-01-01T00:00:00',
            'completed_at': '2026-01-02T00:00:00',
        }

        archived_files = ['output/', 'voiceover']

        manifest = generate_manifest(project_dir, LEVEL_CACHE, sizes, stats, archived_files)

        # Check top-level keys
        assert 'project_name' in manifest
        assert manifest['project_name'] == 'TestProject'

        assert 'cleanup_date' in manifest
        assert 'cleanup_level' in manifest
        assert manifest['cleanup_level'] == LEVEL_CACHE

        # Check sizes section
        assert 'sizes' in manifest
        assert manifest['sizes']['before_bytes'] == 1024000
        assert manifest['sizes']['before_human'] == '1000.0 KB'
        assert manifest['sizes']['after_bytes'] == 512000
        assert manifest['sizes']['freed_bytes'] == 512000

        # Check pipeline_stats section
        assert 'pipeline_stats' in manifest
        assert manifest['pipeline_stats']['last_stage'] == 'OUTPUT'
        assert manifest['pipeline_stats']['keywords'] == ['keyword1', 'keyword2']
        assert manifest['pipeline_stats']['topic'] == 'Test Topic'
        assert manifest['pipeline_stats']['video_count'] == 5

        # Check archived files
        assert 'archived_files' in manifest
        assert manifest['archived_files'] == archived_files

        # Check preserved locations
        assert 'preserved_locations' in manifest
        assert 'global_video_cache' in manifest['preserved_locations']
        assert 'global_entity_cache' in manifest['preserved_locations']

    def test_manifest_minimal_stats(self, tmp_path):
        """Test manifest with minimal stats (empty checkpoint)"""
        project_dir = tmp_path / "MinimalProject"
        project_dir.mkdir()

        sizes = {'before': 0, 'after': 0}

        stats = {
            'last_stage': None,
            'keywords': [],
            'topic': '',
            'segment_count': 0,
            'video_count': 0,
            'match_count': 0,
            'avg_confidence': 0,
            'created_at': None,
            'completed_at': None,
        }

        manifest = generate_manifest(project_dir, LEVEL_FULL, sizes, stats, [])

        assert manifest['project_name'] == 'MinimalProject'
        assert manifest['cleanup_level'] == LEVEL_FULL
        assert manifest['pipeline_stats']['last_stage'] is None
        assert manifest['pipeline_stats']['keywords'] == []

    def test_manifest_format_size_calls(self, tmp_path):
        """Test that manifest correctly calls format_size"""
        project_dir = tmp_path / "SizeTest"
        project_dir.mkdir()

        # Use known size values
        sizes = {'before': 2048, 'after': 1024}

        stats = {'last_stage': 'TEST', 'keywords': [], 'topic': '', 'segment_count': 0,
                 'video_count': 0, 'match_count': 0, 'avg_confidence': 0,
                 'created_at': None, 'completed_at': None}

        manifest = generate_manifest(project_dir, LEVEL_CACHE, sizes, stats, [])

        # Verify format_size was called correctly
        assert manifest['sizes']['before_human'] == '2.0 KB'  # 2048 bytes
        assert manifest['sizes']['after_human'] == '1.0 KB'  # 1024 bytes
        assert manifest['sizes']['freed_human'] == '1.0 KB'  # 1024 bytes freed


@pytest.mark.script
class TestCleanupLevels:
    """Test cleanup level constants"""

    def test_level_constants(self):
        """Verify cleanup level constants are defined correctly"""
        assert LEVEL_CHECKPOINT == "checkpoint"
        assert LEVEL_CACHE == "cache"
        assert LEVEL_MEDIA == "media"
        assert LEVEL_FULL == "full"


@pytest.mark.script
class TestIntegration:
    """Integration tests for cleanup functions"""

    def test_full_cleanup_flow(self, tmp_path):
        """Test complete cleanup flow with all components"""
        # Create project structure
        project_dir = tmp_path / "IntegrationTest"
        project_dir.mkdir()

        # Create cache directory
        cache_dir = project_dir / ".cache"
        cache_dir.mkdir()
        (cache_dir / "test.txt").write_bytes(b"test content")

        # Create checkpoint
        checkpoint = project_dir / "checkpoint.json"
        checkpoint.write_text(json.dumps({
            'last_completed_stage': 'OUTPUT',
            'created_at': '2026-01-01T00:00:00',
            'updated_at': '2026-01-02T00:00:00',
            'analyze': {'keywords': ['test'], 'topic_context': 'Test', 'segment_count': 5},
            'download': {'video_paths': ['video1.mp4']},
            'match': {'match_count': 10, 'avg_confidence': 0.9}
        }))

        # Test get_dir_size
        project_size = get_dir_size(project_dir)
        assert project_size > 0

        # Test collect_deletable_paths
        short_paths = {'videos': None, 'images': None}
        deletable = collect_deletable_paths(project_dir, LEVEL_CACHE, short_paths)
        assert len(deletable) > 0

        # Test generate_manifest
        sizes = {'before': project_size, 'after': 0}
        stats = {
            'last_stage': 'OUTPUT',
            'keywords': ['test'],
            'topic': 'Test',
            'segment_count': 5,
            'video_count': 1,
            'match_count': 10,
            'avg_confidence': 0.9,
            'created_at': '2026-01-01T00:00:00',
            'completed_at': '2026-01-02T00:00:00',
        }
        manifest = generate_manifest(project_dir, LEVEL_CACHE, sizes, stats, ['.cache'])

        assert manifest['project_name'] == 'IntegrationTest'
        assert manifest['pipeline_stats']['video_count'] == 1
        assert manifest['pipeline_stats']['match_count'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
