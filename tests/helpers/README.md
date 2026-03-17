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

**Agent healer helpers:**

| Helper | Purpose |
|--------|---------|
| `assert_healer_attempt_logged()` | Verify healer logged an attempt during healing |
| `assert_healing_strategy_applied()` | Verify correct healing strategy was used |
| `assert_recovery_metrics_valid()` | Validate healing metrics match expectations |
| `assert_healer_chain_executed()` | Verify multiple healers ran in expected order |

## Quick Import

```python
from tests.helpers import (
    # Core assertion helpers
    assert_valid_otio_timeline,
    assert_valid_match_result,
    assert_checkpoint_consistent,
    assert_config_valid,
    # Utility assertions
    assert_file_exists,
    assert_dir_exists,
    assert_json_structure,
    # Agent healer assertions
    assert_healer_attempt_logged,
    assert_healing_strategy_applied,
    assert_recovery_metrics_valid,
    assert_healer_chain_executed,
    # Constants
    VALID_STAGES,
    VALID_HEALERS,
    VALID_STRATEGIES,
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

## Agent Healer Assertions

These helpers validate the self-healing pipeline system in `src/agents/`.

### assert_healer_attempt_logged()

Verifies that a specific healer logged an attempt during the healing process.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `Any` | required | HealingOrchestrator, mock logger, or dict with logged attempts |
| `healer_name` | `str` | required | Name of the healer (e.g., "checkpoint-healer", "api-healer") |
| `attempt_num` | `int` | `None` | Specific attempt number to check (1-indexed) |
| `expected_message` | `str` | `None` | Substring expected in the attempt message |

#### Valid Healer Names

```python
VALID_HEALERS = [
    "checkpoint-healer",
    "api-healer",
    "download-healer",
    "disk-healer",
    "path-healer",
    "otio-healer",
    "llm-healer",
]
```

#### Usage Examples

```python
from tests.helpers import assert_healer_attempt_logged

def test_checkpoint_healer_triggered():
    """Test that checkpoint healer is triggered on JSON error."""
    orchestrator = HealingOrchestrator(config, strategy)
    error = json.JSONDecodeError("Expecting value", "", 0)

    orchestrator.coordinate_heal(error, state, "MATCH", [])

    # Verify healer was triggered
    assert_healer_attempt_logged(orchestrator, "checkpoint-healer")

    # Verify specific attempt number
    assert_healer_attempt_logged(orchestrator, "checkpoint-healer", 1)

    # Verify attempt message
    assert_healer_attempt_logged(
        orchestrator,
        "checkpoint-healer",
        expected_message="Attempting to restore",
    )

def test_healer_with_mock_context():
    """Test using mock context dict."""
    context = {
        "attempts": [
            {"healer": "api-healer", "message": "Rate limit detected"},
            {"healer": "api-healer", "message": "Increasing delay to 30s"},
        ]
    }
    assert_healer_attempt_logged(context, "api-healer", attempt_num=2)
```

---

### assert_healing_strategy_applied()

Verifies that the correct healing strategy was applied.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `Any` | required | HealingOrchestrator, HealingStrategy, or context dict |
| `strategy_name` | `str` | required | Expected strategy ("aggressive", "conservative", "interactive", "minimal") |
| `check_max_attempts` | `int` | `None` | Verify max_attempts_per_stage value |
| `check_heal_delay` | `float` | `None` | Verify heal_delay value |

#### Valid Strategy Names

```python
VALID_STRATEGIES = ["aggressive", "conservative", "interactive", "minimal"]
```

#### Strategy Defaults

| Strategy | max_attempts_per_stage | heal_delay |
|----------|------------------------|------------|
| aggressive | 5 | 1.0 |
| conservative | 3 | 2.0 |
| interactive | 3 | 2.0 |
| minimal | 1 | 0.5 |

#### Usage Examples

```python
from tests.helpers import assert_healing_strategy_applied

def test_conservative_strategy():
    """Test that conservative strategy is applied by default."""
    orchestrator = HealingOrchestrator(config)
    assert_healing_strategy_applied(orchestrator, "conservative")

def test_aggressive_strategy_params():
    """Test aggressive strategy has expected parameters."""
    strategy = HealingStrategy.aggressive()
    orchestrator = HealingOrchestrator(config, strategy)

    assert_healing_strategy_applied(
        orchestrator,
        "aggressive",
        check_max_attempts=5,
        check_heal_delay=1.0,
    )

def test_minimal_strategy():
    """Test minimal strategy for fast-fail testing."""
    strategy = HealingStrategy.minimal()
    assert_healing_strategy_applied(strategy, "minimal", check_max_attempts=1)
```

---

### assert_recovery_metrics_valid()

Validates that healing metrics match expectations.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metrics` | `Any` | required | HealingMetrics object or dict |
| `expected_attempts` | `int` | `None` | Total expected heal attempts |
| `expected_success` | `bool` | `None` | Expected overall success |
| `min_successful_heals` | `int` | `None` | Minimum successful heals required |
| `max_failed_heals` | `int` | `None` | Maximum failed heals allowed |
| `expected_healers_used` | `List[str]` | `None` | List of healer names that should have been used |
| `expected_stages_healed` | `List[str]` | `None` | List of stage names that should have been healed |

#### Metric Fields

```python
@dataclass
class HealingMetrics:
    total_heals: int = 0
    successful_heals: int = 0
    failed_heals: int = 0
    heals_by_stage: Dict[str, int] = ...
    heals_by_healer: Dict[str, int] = ...
    time_spent_healing: float = 0.0
    preflight_issues_found: int = 0
    preflight_issues_fixed: int = 0
    rollbacks_performed: int = 0
    user_escalations: int = 0
```

#### Usage Examples

```python
from tests.helpers import assert_recovery_metrics_valid

def test_healing_metrics_after_recovery():
    """Test metrics after successful recovery."""
    runner = ResilientRunner(config, orchestrator)
    runner.run_pipeline(pipeline)

    metrics = orchestrator.get_metrics()
    assert_recovery_metrics_valid(
        metrics,
        expected_success=True,
        min_successful_heals=1,
    )

def test_healing_attempts_counted():
    """Test that all heal attempts are recorded."""
    # Simulate 3 heal attempts with 2 successes
    metrics = {
        "total_heals": 3,
        "successful_heals": 2,
        "failed_heals": 1,
        "heals_by_healer": {"api-healer": 2, "download-healer": 1},
    }

    assert_recovery_metrics_valid(
        metrics,
        expected_attempts=3,
        max_failed_heals=1,
        expected_healers_used=["api-healer", "download-healer"],
    )

def test_stage_specific_healing():
    """Test that correct stages were healed."""
    metrics = orchestrator.get_metrics()
    assert_recovery_metrics_valid(
        metrics,
        expected_stages_healed=["DOWNLOAD", "MATCH"],
    )
```

---

### assert_healer_chain_executed()

Verifies that a chain of healers was executed during healing.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | `Any` | required | HealingOrchestrator, mock logger, or context dict |
| `expected_healers` | `List[str]` | required | List of healer names expected to be invoked |
| `in_order` | `bool` | `False` | Whether healers must have been invoked in specified order |
| `all_required` | `bool` | `True` | Whether all specified healers must have been invoked |

#### Usage Examples

```python
from tests.helpers import assert_healer_chain_executed

def test_fallback_chain_triggered():
    """Test that fallback chain runs when first healer fails."""
    orchestrator = HealingOrchestrator(config, strategy)
    error = Exception("Complex error")

    orchestrator.coordinate_heal(error, state, "DOWNLOAD", [])

    # Verify multiple healers were tried
    assert_healer_chain_executed(
        orchestrator,
        ["checkpoint-healer", "api-healer", "download-healer"],
    )

def test_healer_order_respected():
    """Test that healers run in priority order."""
    orchestrator = HealingOrchestrator(config, strategy)

    # Run healing
    orchestrator.coordinate_heal(error, state, "MATCH", [])

    # Verify order: checkpoint first, then others
    assert_healer_chain_executed(
        orchestrator,
        ["checkpoint-healer", "api-healer"],
        in_order=True,
    )

def test_partial_chain_execution():
    """Test that at least some healers from chain run."""
    context = {
        "healers": {
            "checkpoint-healer": [{"message": "Restoring"}],
            "api-healer": [{"message": "Rate limit"}],
        }
    }

    # Only require api-healer, don't require all
    assert_healer_chain_executed(
        context,
        ["api-healer", "download-healer"],
        all_required=False,  # Only api-healer needs to run
    )
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
