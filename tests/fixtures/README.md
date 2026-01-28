# Test Fixtures Module

This module provides reusable factory functions and fixtures for creating test objects.

## Quick Reference

| Factory | Location | Purpose |
|---------|----------|---------|
| `srt_segment_factory` | conftest.py | Create SRTSegment instances |
| `config_factory` | conftest.py | Create mock Config objects |
| `pipeline_state_factory` | conftest.py | Create PipelineState instances |
| `match_result_factory` | conftest.py | Create MatchResult with alternatives |
| `match_factory` | conftest.py | Create individual Match objects |
| `create_mock_config` | fixtures/ | Non-pytest mock Config |
| `create_mock_state` | fixtures/ | Non-pytest mock PipelineState |
| `create_test_checkpoint` | fixtures/ | Create checkpoint dict |
| `create_checkpoint_with_populated_stages` | fixtures/ | Rule 25 compliant checkpoint |

## Quick Import

```python
# Pytest fixtures (use in test function signature)
def test_something(srt_segment_factory, config_factory, match_result_factory):
    segment = srt_segment_factory(text="Custom text")
    config = config_factory(matching={'min_confidence': 0.9})
    result = match_result_factory(confidence=0.95, num_alternatives=2)

# Direct imports (non-pytest usage)
from tests.fixtures import (
    create_mock_config,
    create_mock_state,
    create_test_checkpoint,
    create_checkpoint_with_populated_stages,
)
```

---

## Pytest Fixtures (conftest.py)

### srt_segment_factory

Creates SRTSegment instances with customizable fields.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `index` | int | 0 | Segment index |
| `start_time` | float | 0.0 | Start time in seconds |
| `end_time` | float | 5.0 | End time in seconds |
| `text` | str | "Sample voiceover text" | Segment text |
| `source_file` | str | "" | Source file path |
| `keywords` | list | [] | Associated keywords |
| `entities` | list | [] | Extracted entities |
| `topic_id` | int | None | Topic identifier |
| `topics` | list | [] | Topic list |
| `is_broll` | bool | False | B-roll flag (Rule 8) |

**Examples:**

```python
def test_basic_segment(srt_segment_factory):
    # Default segment
    segment = srt_segment_factory()
    assert segment.text == "Sample voiceover text"
    assert segment.start_time == 0.0
    assert segment.end_time == 5.0

def test_custom_segment(srt_segment_factory):
    # Custom segment with B-roll flag
    segment = srt_segment_factory(
        index=5,
        text="Scene shows landscape",
        start_time=10.0,
        end_time=15.0,
        is_broll=True,
        keywords=["landscape", "nature"]
    )
    assert segment.is_broll is True
    assert "landscape" in segment.keywords

def test_multiple_segments(srt_segment_factory):
    # Create multiple independent segments
    segments = [
        srt_segment_factory(index=i, text=f"Segment {i}")
        for i in range(3)
    ]
    assert len(segments) == 3
    assert segments[0].text != segments[1].text
```

---

### config_factory

Creates mock Config objects with section override patterns.

**Parameters (all are dicts):**

| Section | Key Fields | Defaults |
|---------|------------|----------|
| `cache` | cache_dir | tmp_path/cache |
| `output` | output_dir | tmp_path/output |
| `transcription` | model, language | "base", "en" |
| `matching` | min_confidence, location_matching | 0.5, disabled |
| `keyword` | max_keywords, min_keyword_length | 10, 3 |
| `download` | root_dir | tmp_path/downloads |

**Examples:**

```python
def test_default_config(config_factory):
    config = config_factory()
    assert config.matching.min_confidence == 0.5
    assert config.transcription.model == "base"

def test_override_matching(config_factory):
    # Override matching section
    config = config_factory(matching={
        'min_confidence': 0.85,
        'embedding_candidates': 100
    })
    assert config.matching.min_confidence == 0.85

def test_multiple_overrides(config_factory):
    # Override multiple sections
    config = config_factory(
        transcription={'model': 'large', 'language': 'es'},
        download={'root_dir': '/custom/videos'},
        keyword={'max_keywords': 20}
    )
    assert config.transcription.model == 'large'
    assert config.download.root_dir == '/custom/videos'
```

**Section Override Pattern:**

```python
# To override nested config values, use dict notation
config = config_factory(
    matching={
        'min_confidence': 0.9,
        'high_confidence_threshold': 0.95,
        'location_matching': Mock(enabled=True, geonames_username="test")
    }
)
```

---

### pipeline_state_factory

Creates PipelineState instances with pre-populated stage data.

**Common Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `voiceover_path` | str | Path to voiceover file |
| `voiceover_segments` | List | Parsed voiceover segments |
| `keywords` | List[str] | Extracted keywords |
| `downloaded_videos` | List | Downloaded video objects |
| `matches` | List | Match results |
| `text_metadata` | List[Dict] | Video transcripts |
| `entities` | List | Extracted entities |
| `entity_images` | Dict | Entity image paths |

**Examples:**

```python
def test_empty_state(pipeline_state_factory):
    state = pipeline_state_factory()
    assert state.voiceover_segments == []

def test_with_segments(pipeline_state_factory, srt_segment_factory):
    segments = [
        srt_segment_factory(index=i, text=f"Segment {i}")
        for i in range(3)
    ]
    state = pipeline_state_factory(voiceover_segments=segments)
    assert len(state.voiceover_segments) == 3

def test_with_keywords_and_matches(pipeline_state_factory):
    state = pipeline_state_factory(
        keywords=['travel', 'nature', 'adventure'],
        matches=[
            {'segment_id': 0, 'video_file': 'vid1.mp4', 'confidence': 0.9},
            {'segment_id': 1, 'video_file': 'vid2.mp4', 'confidence': 0.85}
        ]
    )
    assert 'travel' in state.keywords
    assert len(state.matches) == 2
```

**Pre-populated Data Example (simulating post-TRANSCRIBE state):**

```python
def test_post_transcribe_state(pipeline_state_factory, srt_segment_factory):
    state = pipeline_state_factory(
        voiceover_path="/project/voiceover.srt",
        voiceover_segments=[
            srt_segment_factory(index=0, text="Welcome"),
            srt_segment_factory(index=1, text="Let's begin"),
        ],
        keywords=['tutorial', 'python', 'basics'],
        text_metadata=[
            {'video_path': 'vid1.mp4', 'text': 'Python intro', 'word_count': 50},
            {'video_path': 'vid2.mp4', 'text': 'Tutorial content', 'word_count': 75},
        ]
    )
```

---

### match_result_factory

Creates MatchResult instances with configurable confidence and alternatives.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confidence` | float | 0.85 | Primary match confidence |
| `video_source_file` | str | "video.mp4" | Video source file |
| `vo_text` | str | "Voiceover text" | Voiceover text |
| `video_text` | str | "Video transcript" | Video transcript |
| `has_gap` | bool | False | Gap result flag |
| `gap_reason` | str | "" | Reason for gap |
| `num_alternatives` | int | 0 | Alternatives count |
| `matched_keywords` | list | [] | Matched keywords |

**Examples:**

```python
def test_simple_match(match_result_factory):
    result = match_result_factory()
    assert result.primary_match.confidence == 0.85
    assert not result.has_gap

def test_high_confidence(match_result_factory):
    result = match_result_factory(confidence=0.95)
    assert result.primary_match.confidence == 0.95

def test_with_alternatives(match_result_factory):
    # Create result with 3 alternatives
    result = match_result_factory(
        confidence=0.90,
        num_alternatives=3
    )
    assert len(result.alternatives) == 3
    # Alternatives have decreasing confidence
    assert result.alternatives[0].confidence == 0.85  # 0.90 - 0.05
    assert result.alternatives[1].confidence == 0.80  # 0.90 - 0.10
    assert result.alternatives[2].confidence == 0.75  # 0.90 - 0.15

def test_gap_result(match_result_factory):
    # Create gap (no suitable match)
    result = match_result_factory(
        has_gap=True,
        gap_reason="No suitable match found"
    )
    assert result.has_gap is True
    assert result.gap_reason == "No suitable match found"
```

**Alternatives Generation:**

The factory automatically generates alternatives with:
- Decreasing confidence (primary - 0.05 * n)
- Unique video files (`alt_video_1.mp4`, `alt_video_2.mp4`, etc.)
- Distinct reasoning for each alternative

---

### match_factory

Creates individual Match objects (lighter weight than match_result_factory).

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vo_text` | str | "Voiceover text" | Voiceover text |
| `video_text` | str | "Video transcript" | Video transcript |
| `video_source_file` | str | "video.mp4" | Video source file |
| `confidence` | float | 0.85 | Match confidence |
| `reasoning` | str | "Test match" | Match reasoning |

**Example:**

```python
def test_individual_match(match_factory):
    match = match_factory(
        vo_text="Welcome to the tutorial",
        video_text="Python programming basics",
        confidence=0.92
    )
    assert match.confidence == 0.92
    assert match.voiceover_segment.text == "Welcome to the tutorial"
```

---

## Direct Import Fixtures (tests/fixtures/__init__.py)

### create_mock_config

Creates a fully mocked Config object for non-pytest usage.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tmp_path` | Path | None | Base path for directories |
| `cache_dir` | str | None | Override cache directory |
| `output_dir` | str | None | Override output directory |
| `download_root` | str | None | Override download root |
| `**overrides` | dict | {} | Section-specific overrides |

**Example:**

```python
from tests.fixtures import create_mock_config

config = create_mock_config(
    tmp_path=Path("/tmp/test"),
    matching={'min_confidence': 0.9},
    healing={'enabled': False}
)
```

---

### create_mock_state

Creates a mock PipelineState with test data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `voiceover_segments` | List[Dict] | 3 default segments | Voiceover segments |
| `text_metadata` | List[Dict] | 3 default entries | Video metadata |
| `video_candidates` | List | 3 mock videos | Video candidates |
| `embeddings` | np.ndarray | Random 3x384 | Embeddings |
| `matches` | List[Dict] | 3 default matches | Match results |
| `**overrides` | dict | {} | Additional attributes |

**Example:**

```python
from tests.fixtures import create_mock_state

state = create_mock_state(
    voiceover_segments=[
        {"index": 0, "text": "Custom segment", "start": 0.0, "end": 3.0}
    ],
    keywords=["custom", "test", "keywords"]
)
```

---

### create_test_checkpoint

Creates a basic valid checkpoint dict for testing.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `last_completed_stage` | str | "MATCH" | Last completed stage |
| `stages` | Dict | 4 default stages | Stage data |
| `**overrides` | dict | {} | Additional fields |

**Example:**

```python
from tests.fixtures import create_test_checkpoint

checkpoint = create_test_checkpoint(
    last_completed_stage="TRANSCRIBE",
    stages={
        "ANALYZE": {"keywords": ["test"], "completed": True},
        "TRANSCRIBE": {"videos_transcribed": 5, "completed": True}
    }
)
```

---

### create_checkpoint_with_populated_stages

Creates a checkpoint with fully populated stage data for `--output-only` mode testing.

**Rule 25 Compliance:** This fixture addresses the requirement that `--output-only` mode needs a checkpoint with populated `stages` dict, not just `last_completed_stage`.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_dir` | Path | None | Project directory |
| `last_completed_stage` | str | "MATCH" | Last completed stage |
| `include_all_stages` | bool | True | Include all stage data |
| `voiceover_path` | str | None | Voiceover file path |
| `match_count` | int | 10 | Number of matches |
| `avg_confidence` | float | 0.85 | Average confidence |
| `video_count` | int | 5 | Number of videos |
| `**stage_overrides` | dict | {} | Stage-specific overrides |

**Examples:**

```python
from tests.fixtures import create_checkpoint_with_populated_stages

# Basic usage
checkpoint = create_checkpoint_with_populated_stages(
    project_dir=tmp_path,
    match_count=20,
    avg_confidence=0.92
)
assert checkpoint["match"]["match_count"] == 20
assert len(checkpoint["match"]["matches"]) == 20

# Override specific stage data
checkpoint = create_checkpoint_with_populated_stages(
    last_completed_stage="DOWNLOAD_SEGMENTS",
    match={"strategy": "embedding_only"},
    download={"count": 10, "total_size_mb": 500.0}
)

# Test --output-only mode
def test_output_only_with_populated_checkpoint(tmp_path):
    checkpoint = create_checkpoint_with_populated_stages(
        project_dir=tmp_path,
        last_completed_stage="DOWNLOAD_SEGMENTS",
        match_count=50,
        video_count=15
    )
    # Verify all required stage data present
    assert "analyze" in checkpoint
    assert "download" in checkpoint
    assert "match" in checkpoint
    assert checkpoint["match"]["matches"]  # Non-empty
```

**Stage Data Included (when include_all_stages=True):**

| Stage Key | Data Fields |
|-----------|-------------|
| `analyze` | keywords, segment_count, topic, entities |
| `entity_images` | images, sources, count |
| `entity_videos` | videos, api_calls, sources |
| `video_metadata` | metadata_count, caption_languages |
| `caption` | caption_count, languages, source |
| `download` | video_paths, count, total_size_mb |
| `stock` | stock_videos, source |
| `broll_download` | broll_paths, keyword_suffixes |
| `remix` | filtered_count, removed |
| `transcribe` | transcribed_count, embedding_count, model |
| `premise` | premise_detected, topic_context |
| `scene_detection` | scenes, broll_flagged, face_detection_enabled |
| `match` | match_count, avg_confidence, matches, strategy |
| `broll_match` | broll_matches, strategies, v8_entries |
| `iterative_match` | iterations, improved_count |
| `download_segments` | segments_downloaded, total_size_mb |

---

## Specialized Fixtures

### create_concurrent_escalation_fixture

Creates fixture for testing concurrent download escalation (EscalationManager).

```python
from tests.fixtures import create_concurrent_escalation_fixture

fixture = create_concurrent_escalation_fixture(
    num_keywords=5,
    escalation_threshold=2,
    cooldown_seconds=0.0
)
assert len(fixture["keywords"]) == 5
assert fixture["ext_config"]["escalation_threshold"] == 2
```

### create_otio_timeline_fixture

Creates fixture for OTIO timeline auto-split testing (Rule 17).

```python
from tests.fixtures import create_otio_timeline_fixture

# At threshold
fixture = create_otio_timeline_fixture(3000)
assert fixture["at_threshold"] is True
assert fixture["expected_parts"] == 1

# Above threshold (triggers split)
fixture = create_otio_timeline_fixture(6000)
assert fixture["above_threshold"] is True
assert fixture["expected_parts"] == 2
```

### create_broll_propagation_chain_state

Creates fixture for B-roll propagation chain validation (Rule 8).

```python
from tests.fixtures import create_broll_propagation_chain_state

state = create_broll_propagation_chain_state(
    num_scenes=10,
    broll_ratio=0.4
)
assert state["expected_broll_count"] == 4
assert state["face_detected_broll"] == 2
assert state["silent_detected_broll"] == 2
```

---

## Best Practices

1. **Prefer pytest fixtures** for test functions - they handle setup/teardown
2. **Use factories over static fixtures** for customizable test objects
3. **Import from conftest.py** via function signature (pytest auto-injects)
4. **Use direct imports** for non-pytest code or setup helpers
5. **Override only what's needed** - factories provide sensible defaults
6. **Check Rule 25** when testing `--output-only` mode - use `create_checkpoint_with_populated_stages`

## Related Documentation

- [tests/helpers/README.md](../helpers/README.md) - Assertion helpers
- [tests/README.md](../README.md) - Test suite overview
- [CLAUDE.md](../../CLAUDE.md) - Project conventions
