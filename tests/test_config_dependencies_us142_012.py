"""
Tests for US-142-012: Config dependency validation between sections.

Tests that verify Config._validate_dependencies() catches:
- chapter_matching_enabled requires extract_video_chapters enabled
- iterative_chapter_boost requires iterative_matching enabled
- region_backoff requires mullvad enabled
- search_budget_aware requires search_budget section configured
"""

import pytest
import logging
import os
import sys
import tempfile
import yaml

# Ensure src is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.base import Config, ConfigError


# Helper function to create a minimal valid config with API keys
def create_minimal_valid_config_text() -> str:
    """Create a minimal valid config that passes API key and constraint validation."""
    return """
# Minimal config with valid API keys (test values)
api_keys:
  gemini_api_key: "test_gemini_key_12345"
  anthropic_api_key: "test_anthropic_key_12345"
  pexels_api_key: "test_pexels_key_12345"
  pixabay_api_key: "test_pixabay_key_12345"

# Required sections with valid values
matching:
  min_confidence: 0.5
  max_clip_reuse: 3
  primary_provider: "gemini"
  secondary_provider: "gemini"
  chapter_matching_enabled: true
  low_confidence_threshold: 0.3
  ambiguous_threshold: 0.4
  high_confidence_threshold: 0.7
  context_enrichment:
    extract_video_chapters: true

transcription:
  max_workers: 4

embedding:
  provider: "gemini"

iterative_matching:
  enabled: true
  iterative_chapter_boost: 0.1

video_search:
  search_budget_aware: true

search_budget:
  max_total_results: 200
  results_per_keyword: 20

download:
  mullvad:
    enabled: false
  region_backoff:
    enabled: false

output:
  generate_otio: true
  num_alternatives: 2
"""


def test_chapter_matching_requires_extract_chapters():
    """Test that chapter_matching_enabled=true requires extract_video_chapters=true."""
    config_text = create_minimal_valid_config_text()
    # Override to set chapter_matching_enabled=true but extract_video_chapters=false
    # This replaces the context_enrichment section
    config_text += """
matching:
  chapter_matching_enabled: true
  context_enrichment:
    extract_video_chapters: false
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        # This should fail during loading due to validation error
        with pytest.raises(ConfigError) as exc_info:
            Config.from_yaml(config_path)

        # Check that the error message mentions chapter_matching
        assert 'chapter_matching_enabled' in str(exc_info.value)
        assert 'extract_video_chapters' in str(exc_info.value)
    finally:
        os.unlink(config_path)


def test_chapter_matching_valid():
    """Test that chapter_matching_enabled=true with extract_video_chapters=true passes."""
    config_text = create_minimal_valid_config_text()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        config = Config.from_yaml(config_path)
        # Verify config loads successfully
        assert config is not None
    finally:
        os.unlink(config_path)


def test_iterative_chapter_boost_requires_enabled():
    """Test that iterative_chapter_boost > 0 requires iterative_matching.enabled=true."""
    config_text = create_minimal_valid_config_text()
    # Override to set iterative_chapter_boost > 0 but iterative_matching.enabled=false
    config_text += """
iterative_matching:
  enabled: false
  iterative_chapter_boost: 0.1
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        # This should fail during loading due to validation error
        with pytest.raises(ConfigError) as exc_info:
            Config.from_yaml(config_path)

        # Check that the error message mentions iterative_chapter_boost
        assert 'iterative_chapter_boost' in str(exc_info.value)
        assert 'enabled' in str(exc_info.value)
    finally:
        os.unlink(config_path)


def test_iterative_chapter_boost_zero_valid():
    """Test that iterative_chapter_boost=0 works even when iterative_matching disabled."""
    config_text = create_minimal_valid_config_text()
    config_text += """
iterative_matching:
  enabled: false
  iterative_chapter_boost: 0
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        config = Config.from_yaml(config_path)
        # Verify config loads successfully
        assert config is not None
    finally:
        os.unlink(config_path)


def test_region_backoff_requires_mullvad():
    """Test that region_backoff.enabled=true requires mullvad.enabled=true."""
    config_text = create_minimal_valid_config_text()
    # Override to set region_backoff.enabled=true but mullvad.enabled=false
    config_text += """
download:
  mullvad:
    enabled: false
  region_backoff:
    enabled: true
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        # This should fail during loading due to validation error
        with pytest.raises(ConfigError) as exc_info:
            Config.from_yaml(config_path)

        # Check that the error message mentions region_backoff
        assert 'region_backoff' in str(exc_info.value)
        assert 'mullvad' in str(exc_info.value)
    finally:
        os.unlink(config_path)


def test_region_backoff_valid():
    """Test that region_backoff.enabled=true with mullvad.enabled=true passes."""
    config_text = create_minimal_valid_config_text()
    config_text += """
download:
  mullvad:
    enabled: true
  region_backoff:
    enabled: true
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        config = Config.from_yaml(config_path)
        # Verify config loads successfully
        assert config is not None
    finally:
        os.unlink(config_path)


def test_search_budget_aware_requires_valid_budget():
    """Test that search_budget_aware=true requires search_budget values > 0."""
    config_text = create_minimal_valid_config_text()
    # Override to set search_budget_aware=true but max_total_results=0
    config_text += """
video_search:
  search_budget_aware: true

search_budget:
  max_total_results: 0
  results_per_keyword: 20
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        # This should fail during loading due to validation error
        with pytest.raises(ConfigError) as exc_info:
            Config.from_yaml(config_path)

        # Check that the error message mentions search_budget_aware
        assert 'search_budget_aware' in str(exc_info.value)
        assert 'max_total_results' in str(exc_info.value)
    finally:
        os.unlink(config_path)


def test_search_budget_aware_requires_results_per_keyword():
    """Test that search_budget_aware=true requires results_per_keyword > 0."""
    config_text = create_minimal_valid_config_text()
    # Override to set search_budget_aware=true but results_per_keyword=0
    config_text += """
video_search:
  search_budget_aware: true

search_budget:
  max_total_results: 200
  results_per_keyword: 0
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        # This should fail during loading due to validation error
        with pytest.raises(ConfigError) as exc_info:
            Config.from_yaml(config_path)

        # Check that the error message mentions search_budget_aware
        assert 'search_budget_aware' in str(exc_info.value)
        assert 'results_per_keyword' in str(exc_info.value)
    finally:
        os.unlink(config_path)


def test_search_budget_aware_disabled_valid():
    """Test that search_budget_aware=false passes even with invalid budget values."""
    config_text = create_minimal_valid_config_text()
    config_text += """
video_search:
  search_budget_aware: false

search_budget:
  max_total_results: 0
  results_per_keyword: 0
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        config = Config.from_yaml(config_path)
        # Verify config loads successfully despite invalid budget values
        assert config is not None
    finally:
        os.unlink(config_path)


def test_all_dependencies_valid():
    """Test that a fully valid config passes all dependency checks."""
    config_text = create_minimal_valid_config_text()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_text)
        config_path = f.name

    try:
        config = Config.from_yaml(config_path)
        # Verify config loads successfully
        assert config is not None

        # Also verify via validate() directly
        errors = config.validate()
        # Should have no dependency-related errors
        dependency_errors = [
            e for e in errors if any(x in e for x in [
                'chapter_matching_enabled',
                'iterative_chapter_boost',
                'region_backoff',
                'search_budget_aware'
            ])
        ]
        assert len(dependency_errors) == 0, f"Unexpected dependency errors: {dependency_errors}"
    finally:
        os.unlink(config_path)
