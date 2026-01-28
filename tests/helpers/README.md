# Test Helpers Module

Reusable assertion helpers for the voiceover-matcher-subtitle test suite.

## Overview

This module provides standardized assertion helpers that encapsulate complex validation logic and provide clear, actionable error messages. Use these helpers instead of writing custom validation logic in individual tests.

**Main helpers:**

| Helper | Purpose |
|--------|---------|
| `assert_valid_otio_timeline()` | Validate OTIO timeline structure and clips |
| `assert_valid_match_result()` | Validate match result dicts with confidence thresholds |
| `assert_checkpoint_consistent()` | Validate checkpoint structure (Rule 25 compliance) |
| `assert_config_valid()` | Validate Config object constraints |
| `assert_file_exists()` | Verify file existence with descriptive errors |
| `assert_dir_exists()` | Verify directory existence with descriptive errors |
| `assert_json_structure()` | Validate JSON data structure and types |

## Quick Import

```python
from tests.helpers import (
    assert_valid_otio_timeline,
    assert_valid_match_result,
    assert_checkpoint_consistent,
    assert_config_valid,
    assert_file_exists,
    assert_dir_exists,
    assert_json_structure,
    VALID_STAGES,
)
```

---

## assert_valid_otio_timeline()

Validates that an OTIO timeline has valid structure, tracks, and clips.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeline` | `otio.Timeline` | required | OTIO Timeline object to validate |
| `min_tracks` | `int` | `1` | Minimum number of tracks required |
| `require_video` | `bool` | `True` | Whether video tracks are required |
| `require_audio` | `bool` | `False` | Whether audio tracks are required |
| `expected_duration` | `float` | `None` | Expected timeline duration in seconds |
| `duration_tolerance` | `float` | `0.1` | Tolerance for duration comparison |
| `max_clips` | `int` | `3000` | Maximum allowed clips (Rule 17) |

### Validations Performed

- Timeline is an `opentimelineio.schema.Timeline` object
- Has expected tracks (video and/or audio)
- All clips have valid `media_reference` (non-empty `target_url` for ExternalReference)
- All clips have valid `source_range` with positive duration
- Duration matches expected (if provided)
- Total clip count is within limits (default 3000, per Rule 17 auto-split)

### Usage Examples

```python
import opentimelineio as otio
from tests.helpers import assert_valid_otio_timeline

def test_timeline_generation():
    """Test that timeline is generated correctly."""
    timeline = create_timeline(matches, config)

    # Basic validation
    assert_valid_otio_timeline(timeline)

    # Validate with specific requirements
    assert_valid_otio_timeline(
        timeline,
        min_tracks=2,
        require_video=True,
        require_audio=True,
        expected_duration=120.0,
    )

def test_large_timeline_within_limits():
    """Test that large timelines don't exceed Rule 17 limits."""
    timeline = create_large_timeline(2500)
    assert_valid_otio_timeline(timeline, max_clips=3000)
```

---

## assert_valid_match_result()

Validates that a match result dict has valid structure and values.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `result` | `Dict` | required | Match result dict to validate |
| `min_confidence` | `float` | `0.0` | Minimum acceptable confidence |
| `require_video_file` | `bool` | `True` | Whether `video_file` field is required |
| `require_timing` | `bool` | `True` | Whether timing fields are required |
| `require_strategy` | `bool` | `False` | Whether `strategy` field is required |
| `valid_strategies` | `List[str]` | `None` | List of valid strategy names |

### Field Validation

**Required fields:**
- `segment_index`: Non-negative integer
- `confidence`: Float in range [0, 1]

**Optional fields (when required):**
- `video_file`: Non-empty string path
- `start`/`end` or `video_start`/`video_end`: Non-negative numbers with start <= end
- `strategy`: Non-empty string from valid_strategies list

### Usage Examples

```python
from tests.helpers import assert_valid_match_result

def test_matching_basic():
    """Test that matcher returns valid results."""
    result = matcher.match(segment)
    assert_valid_match_result(result)

def test_matching_with_confidence_threshold():
    """Test that matches meet minimum confidence."""
    result = matcher.match(segment)
    assert_valid_match_result(result, min_confidence=0.5)

def test_matching_with_strategy():
    """Test that strategy is correctly recorded."""
    result = matcher.match(segment)
    assert_valid_match_result(
        result,
        require_strategy=True,
        valid_strategies=['embedding', 'llm', 'hybrid'],
    )

def test_batch_matching():
    """Test multiple match results."""
    results = matcher.match_batch(segments)
    for result in results:
        assert_valid_match_result(
            result,
            min_confidence=0.3,
            require_timing=True,
        )
```

---

## assert_checkpoint_consistent()

Validates that a checkpoint dict has consistent structure and data. Essential for testing `--output-only` mode (Rule 25 compliance).

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `checkpoint` | `Dict` | required | Checkpoint dict to validate |
| `require_version` | `bool` | `True` | Whether `version` field is required |
| `require_stages` | `bool` | `False` | Whether `stages` dict must be populated |
| `require_voiceover` | `bool` | `False` | Whether voiceover fields are required |
| `expected_stage` | `str` | `None` | Expected `last_completed_stage` value |
| `min_matches` | `int` | `None` | Minimum number of matches in match stage |

### Rule 25 Compliance

`--output-only` mode requires that the checkpoint has a populated `stages` dict, not just `last_completed_stage`. Use `require_stages=True` when testing this mode:

```python
# This will fail if stages dict is empty (Rule 25 violation)
assert_checkpoint_consistent(
    checkpoint,
    require_stages=True,
    expected_stage="OUTPUT",
)
```

### Valid Stage Names

The helper uses `VALID_STAGES` constant for validation:
```python
VALID_STAGES = [
    "ANALYZE", "ENTITY_IMAGES", "ENTITY_VIDEOS", "VIDEO_METADATA",
    "CAPTION", "DOWNLOAD", "STOCK", "BROLL_DOWNLOAD", "REMIX",
    "TRANSCRIBE", "PREMISE", "SCENE_DETECTION", "MATCH",
    "BROLL_MATCH", "ITERATIVE_MATCH", "DOWNLOAD_SEGMENTS", "OUTPUT",
]
```

### Usage Examples

```python
from tests.helpers import assert_checkpoint_consistent, VALID_STAGES

def test_checkpoint_after_match():
    """Test checkpoint is valid after MATCH stage."""
    checkpoint = load_checkpoint(project_dir)
    assert_checkpoint_consistent(
        checkpoint,
        expected_stage="MATCH",
        min_matches=10,
    )

def test_checkpoint_for_output_only_mode():
    """Test checkpoint is valid for --output-only mode (Rule 25)."""
    checkpoint = load_checkpoint(project_dir)

    # CRITICAL: This catches the Rule 25 issue where stages dict is empty
    assert_checkpoint_consistent(
        checkpoint,
        require_stages=True,  # Ensures stages dict is populated
        require_voiceover=True,
    )

def test_fresh_checkpoint():
    """Test that a fresh checkpoint has expected structure."""
    checkpoint = create_fresh_checkpoint()
    assert_checkpoint_consistent(
        checkpoint,
        require_version=True,
        require_stages=False,  # Fresh checkpoint has empty stages
    )

def test_checkpoint_stage_progression():
    """Test that checkpoint stage is valid."""
    checkpoint = load_checkpoint(project_dir)
    stage = checkpoint.get("last_completed_stage")
    assert stage in VALID_STAGES, f"Invalid stage: {stage}"
```

---

## assert_config_valid()

Validates that a Config object is valid and internally consistent.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `Config` | required | Config object to validate |
| `require_api_keys` | `bool` | `False` | Whether API keys must be set |
| `require_matching` | `bool` | `True` | Whether matching section is required |
| `require_download` | `bool` | `False` | Whether download section is required |
| `check_constraints` | `bool` | `True` | Whether to check constraint relationships |
| `allowed_errors` | `List[str]` | `None` | Error message substrings to ignore |

### Returns

`List[str]`: List of validation errors (empty if valid)

### Constraint Validation

When `check_constraints=True`, validates:
- `min_confidence <= high_confidence_threshold`
- `max_workers >= 1` for transcription
- `max_keywords >= 1` for keyword extraction
- Provider values are in valid set: `{'gemini', 'anthropic', 'local', 'embedding_only', 'ollama'}`
- Confidence values are in range [0, 1]

### Usage Examples

```python
from tests.helpers import assert_config_valid

def test_config_validation():
    """Test that config is valid."""
    config = load_config("config.yaml")
    errors = assert_config_valid(config)
    assert not errors, f"Config has errors: {errors}"

def test_config_with_api_keys():
    """Test that production config has API keys."""
    config = load_config("production.yaml")
    errors = assert_config_valid(config, require_api_keys=True)
    assert not errors

def test_config_ignore_known_issues():
    """Test config with allowed errors for test environment."""
    config = load_config("test.yaml")
    errors = assert_config_valid(
        config,
        require_api_keys=False,
        allowed_errors=["GEMINI_API_KEY not set"],
    )
    assert not errors

def test_config_with_custom_constraints():
    """Test config constraint validation."""
    config = load_config("config.yaml")

    # Verify min_confidence constraint
    if hasattr(config.matching, 'min_confidence'):
        assert 0 <= config.matching.min_confidence <= 1

    errors = assert_config_valid(config, check_constraints=True)
    assert not errors, f"Constraint violations: {errors}"
```

---

## Utility Assertions

### assert_file_exists()

```python
from tests.helpers import assert_file_exists

def test_output_files_created():
    assert_file_exists(output_dir / "timeline.otio", "OTIO file")
    assert_file_exists(output_dir / "timeline.edl", "EDL file")
```

### assert_dir_exists()

```python
from tests.helpers import assert_dir_exists

def test_project_structure():
    assert_dir_exists(project_dir / "output", "Output directory")
    assert_dir_exists(project_dir / ".cache", "Cache directory")
```

### assert_json_structure()

```python
from tests.helpers import assert_json_structure

def test_checkpoint_json():
    data = json.load(open("checkpoint.json"))
    assert_json_structure(
        data,
        required_keys=["version", "last_completed_stage"],
        type_checks={"version": str, "stages": dict},
    )
```

---

## Best Practices

### 1. Use Specific Assertions

```python
# Good: Specific validation with meaningful thresholds
assert_valid_match_result(result, min_confidence=0.5)

# Bad: Generic assertion
assert result["confidence"] > 0
```

### 2. Combine Helpers for Comprehensive Testing

```python
def test_full_pipeline_output():
    """Test complete pipeline output."""
    # Validate checkpoint
    checkpoint = load_checkpoint(project_dir)
    assert_checkpoint_consistent(checkpoint, require_stages=True)

    # Validate matches
    for match in checkpoint["stages"]["match"]["matches"]:
        assert_valid_match_result(match, min_confidence=0.3)

    # Validate timeline
    timeline = load_timeline(project_dir / "output" / "timeline.otio")
    assert_valid_otio_timeline(timeline, min_tracks=2)
```

### 3. Use allowed_errors for Test Environments

```python
def test_config_in_ci():
    """CI may not have all API keys."""
    errors = assert_config_valid(
        config,
        allowed_errors=["GEMINI_API_KEY", "ANTHROPIC_API_KEY"],
    )
    # Only fail on unexpected errors
    assert not errors
```

### 4. Import VALID_STAGES for Stage Validation

```python
from tests.helpers import VALID_STAGES

def test_custom_stage_logic():
    stage = get_current_stage()
    assert stage in VALID_STAGES
```

---

## Adding New Helpers

When adding new assertion helpers:

1. Follow the existing pattern with keyword-only arguments
2. Provide clear docstrings with Args, Returns, Raises, and Example sections
3. Use descriptive error messages that explain what failed and why
4. Add the helper to `__all__` in `__init__.py`
5. Document the helper in this README with parameters and examples
6. Add unit tests in `tests/test_helpers.py`
