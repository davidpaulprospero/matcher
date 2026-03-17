# test_chapter_detection - Chapter Detection Tests

This directory contains tests for the multi-pass chapter detection system, which divides voiceover transcripts into logical chapters based on topic changes, locations, and semantic boundaries.

## Directory Overview

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures for chapter detection tests |
| `test_chunking.py` | Transcript chunking for long voiceovers (splitting, overlap, merging) |
| `test_coverage.py` | Coverage resolution pass (overlaps, gaps, tiny chapter merging) |
| `test_detector.py` | EnhancedChapterDetector orchestrator tests |
| `test_initial.py` | Initial detection pass (LLM-based, content type detection) |
| `test_models.py` | ChapterCandidate, ChapterConfidence, DetectionResult dataclasses |
| `test_refinement.py` | Boundary refinement pass (embeddings, transition phrases, cosine similarity) |
| `test_validation.py` | Validation pass (LLM-based boundary/title correction, merge suggestions) |

## Key Fixtures

### mock_config

The `mock_config` fixture provides a complete mock configuration for chapter detection testing:

```python
@pytest.fixture
def mock_config():
    """Create mock config for chapter detection tests."""
    config = Mock()
    config.matching = Mock()
    config.matching.chapter_detection = Mock(
        enabled=True,
        use_validation_pass=True,
        use_boundary_refinement=True,
        default_strategy='topic',
        auto_detect_content_type=True,
        max_chunk_chars=6000,
        chunk_overlap_segments=5,
        min_chapter_confidence=0.5,
        min_chapter_segments=3,
        max_chapters=20,
    )
    config.matching.location_matching = Mock(enabled=True)
    config.gemini_api_key = "test_key"
    config.cache = Mock(cache_dir="/tmp/test_cache")
    return config
```

**Usage Example:**

```python
def test_detector_with_config(mock_config):
    with patch.object(EnhancedChapterDetector, '_init_llm_client'):
        detector = EnhancedChapterDetector(mock_config)
        assert detector._get_config_value('min_chapter_segments', 3) == 3
```

### sample_segments

Provides sample transcript segments for testing chapter boundary detection:

```python
@pytest.fixture
def sample_segments():
    """Sample segments with two clear topics (Paris, Tokyo)."""
    return [
        {"index": 0, "text": "Welcome to Paris, the city of lights.", "start": 0.0, "end": 3.0},
        {"index": 1, "text": "The Eiffel Tower is the most iconic landmark.", "start": 3.0, "end": 6.0},
        {"index": 2, "text": "Let's explore the Louvre museum.", "start": 6.0, "end": 9.0},
        {"index": 3, "text": "Now we travel to Tokyo, Japan.", "start": 9.0, "end": 12.0},
        {"index": 4, "text": "Shibuya crossing is incredibly busy.", "start": 12.0, "end": 15.0},
        {"index": 5, "text": "The temples are beautiful and peaceful.", "start": 15.0, "end": 18.0},
    ]
```

**Usage Example:**

```python
def test_detects_topic_transition(mock_config, sample_segments):
    # sample_segments has clear Paris->Tokyo transition at index 3
    result = run_initial_detection(sample_segments, mock_llm_client, mock_config)
    assert len(result) >= 2  # Should detect 2 chapters
```

### single_topic_segments

Provides segments that represent a single coherent topic (no chapter split expected):

```python
@pytest.fixture
def single_topic_segments():
    """Segments that should be one chapter."""
    return [
        {"index": 0, "text": "Introduction to Python programming.", ...},
        {"index": 1, "text": "Python is a versatile language.", ...},
        {"index": 2, "text": "You can use it for web development.", ...},
        {"index": 3, "text": "Data science is another popular use.", ...},
    ]
```

### mock_llm_client

Creates a mock LLM client for testing without real API calls:

```python
@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock()
    return client
```

### mock_llm_response_two_chapters

Provides a pre-configured LLM response with two chapters:

```python
@pytest.fixture
def mock_llm_response_two_chapters():
    """Mock LLM response with two chapters."""
    response = Mock()
    response.parsed_data = [
        {
            "start_segment_idx": 0,
            "end_segment_idx": 2,
            "title": "Paris Tour",
            "topics": ["paris", "france", "landmarks"],
            "location_name": "Paris",
            "confidence": "high"
        },
        {
            "start_segment_idx": 3,
            "end_segment_idx": 5,
            "title": "Tokyo Adventure",
            "topics": ["tokyo", "japan", "temples"],
            "location_name": "Tokyo",
            "confidence": "high"
        }
    ]
    return response
```

**Usage Example:**

```python
def test_initial_detection_with_llm(mock_config, mock_llm_client, sample_segments, mock_llm_response_two_chapters):
    mock_llm_client.generate.return_value = mock_llm_response_two_chapters

    with patch('src.chapter_detection.passes.initial.create_chunks') as mock_chunks:
        # Setup chunk mock...
        result = run_initial_detection(sample_segments, mock_llm_client, mock_config)

    assert len(result) == 2
    assert result[0].title == "Paris Tour"
    assert result[1].title == "Tokyo Adventure"
```

## Testing Chapter Boundary Detection

The chapter detection system uses a multi-pass pipeline:

### Pipeline Passes

| Pass | Module | Purpose |
|------|--------|---------|
| Initial | `passes/initial.py` | LLM-based chapter detection with strategy selection |
| Refinement | `passes/refinement.py` | Boundary adjustment using embeddings and transition phrases |
| Validation | `passes/validation.py` | LLM-based verification and correction of chapters |
| Coverage | `passes/coverage.py` | Resolve overlaps, fill gaps, merge tiny chapters |

### Testing Pass Interactions

```python
@patch('src.chapter_detection.detector.run_initial_detection')
@patch('src.chapter_detection.detector.run_boundary_refinement')
@patch('src.chapter_detection.detector.run_validation')
@patch('src.chapter_detection.detector.run_coverage_resolution')
def test_full_pipeline(mock_coverage, mock_validation, mock_refinement, mock_initial, mock_config, sample_segments):
    """Test detector runs all passes in sequence."""
    chapters = [ChapterCandidate(chapter_id=0, ...)]

    mock_initial.return_value = chapters
    mock_refinement.return_value = chapters
    mock_validation.return_value = chapters
    mock_coverage.return_value = chapters

    with patch.object(EnhancedChapterDetector, '_init_llm_client'):
        detector = EnhancedChapterDetector(mock_config)
        detector.llm_client = Mock()
        result = detector.detect_chapters_full(sample_segments)

    assert 'initial' in result.detection_passes_run
    assert 'refinement' in result.detection_passes_run
    assert 'validation' in result.detection_passes_run
    assert 'coverage' in result.detection_passes_run
```

## Testing Chunking Strategies

The chunking module splits long transcripts into processable chunks for LLM calls.

### Test Classes

| Class | Tests |
|-------|-------|
| `TestCreateIndexedText` | Text formatting with segment markers |
| `TestCreateChunks` | Chunk creation, overlap, boundary handling |
| `TestMergeChunkResults` | Combining results from multiple chunks |
| `TestChunkingSentenceBoundaries` | Ensuring chunks split on segment boundaries |
| `TestChunkingMaxSizeLimit` | Respecting max_chars configuration |
| `TestChunkingShortTranscripts` | Handling transcripts smaller than chunk size |
| `TestChunkingTimestampPreservation` | Maintaining segment index alignment |
| `TestChunkingUnicode` | Unicode character handling |

### Testing Chunk Boundaries

```python
def test_chunks_split_on_segment_boundaries():
    """Test that chunks split on segment boundaries, not mid-text."""
    segments = [
        {"index": 0, "text": "This is the first sentence."},
        {"index": 1, "text": "This is the second sentence."},
        {"index": 2, "text": "This is the third sentence."},
    ]
    chunks = create_chunks(segments, max_chars=80, overlap_segments=1)

    # Each chunk should contain complete segments
    for chunk in chunks:
        assert chunk.text.startswith("[")  # Starts with segment marker
        for line in chunk.text.split("\n"):
            if line.strip():
                assert "]" in line  # Complete segment marker
```

### Testing Chunk Overlap

```python
def test_overlap_between_chunks():
    """Test that adjacent chunks have overlapping segments."""
    segments = [{"index": i, "text": f"Segment {i} " * 50} for i in range(20)]
    chunks = create_chunks(segments, max_chars=500, overlap_segments=3)

    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_end = chunks[i].end_segment_idx
            next_start = chunks[i + 1].start_segment_idx
            assert next_start <= current_end + 1  # Overlap exists
```

## Testing Transcript-to-Chapter Validation

The validation pass uses LLM to verify and correct detected chapters.

### Testing Boundary Validation

```python
def test_boundary_correction():
    """Test that validation can correct chapter boundaries."""
    chapters = [ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=10)]

    validation_result = {
        "validations": [{
            "chapter_id": 0,
            "boundary_correct": False,
            "suggested_start": 2,
            "suggested_end": 8,
            "title_accurate": True,
            "content_coherent": True,
        }],
    }

    result = _apply_validation_results(chapters, validation_result, segments)

    assert result[0].start_segment_idx == 2
    assert result[0].end_segment_idx == 8
    assert "[Corrected by validation]" in result[0].boundary_reasoning
```

### Testing Title Validation

```python
def test_title_correction():
    """Test that validation can suggest better titles."""
    chapters = [ChapterCandidate(chapter_id=0, title="Wrong Title")]

    validation_result = {
        "validations": [{
            "chapter_id": 0,
            "title_accurate": False,
            "suggested_title": "Better Title",
        }],
    }

    result = _apply_validation_results(chapters, validation_result, [])
    assert result[0].title == "Better Title"
```

### Testing Merge Suggestions

```python
def test_merges_chapters():
    """Test that validation can suggest chapter merges."""
    chapters = [
        ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=3, title="Part 1"),
        ChapterCandidate(chapter_id=1, start_segment_idx=4, end_segment_idx=7, title="Part 2"),
        ChapterCandidate(chapter_id=2, start_segment_idx=8, end_segment_idx=10, title="Separate"),
    ]
    merge_suggestions = [[0, 1]]  # Merge chapters 0 and 1

    result = _apply_merge_suggestions(chapters, merge_suggestions)

    assert len(result) == 2  # Two chapters: merged + separate
    merged = [c for c in result if c.start_segment_idx == 0][0]
    assert merged.end_segment_idx == 7
```

### Testing Missed Chapter Detection

```python
def test_adds_missed_chapters():
    """Test that validation can add chapters it detected but initial pass missed."""
    chapters = [
        ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=4),
        ChapterCandidate(chapter_id=1, start_segment_idx=10, end_segment_idx=15),
    ]
    missed = [{
        "start_segment_idx": 5,
        "end_segment_idx": 9,
        "suggested_title": "Missed Section",
    }]

    result = _add_missed_chapters(chapters, missed, segments)

    assert len(result) == 3
    new_ch = [c for c in result if c.title == "Missed Section"][0]
    assert new_ch.detection_strategy == "validation_added"
```

## Testing Boundary Refinement

The refinement pass adjusts boundaries using semantic embeddings and linguistic cues.

### Testing Embedding-Based Refinement

```python
def test_with_embeddings(mock_config):
    """Test refinement with embedding-based boundary scores."""
    chapters = [
        ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=4),
        ChapterCandidate(chapter_id=1, start_segment_idx=5, end_segment_idx=9),
    ]

    # Create embeddings with clear semantic gap at segment 5
    embeddings = np.array([
        [1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.9, 0.1, 0.0],  # 0-2: similar
        [0.85, 0.15, 0.0], [0.8, 0.2, 0.0],                   # 3-4: similar
        [0.0, 1.0, 0.0], [0.05, 0.95, 0.0],                   # 5-6: different direction
        [0.1, 0.9, 0.0], [0.15, 0.85, 0.0], [0.2, 0.8, 0.0],  # 7-9: similar
    ])

    result = run_boundary_refinement(chapters, segments, mock_config, embeddings)
    assert len(result) == 2  # Boundaries preserved or refined near semantic gap
```

### Testing Transition Phrase Detection

```python
def test_transition_phrase_detection():
    """Test detection of linguistic transition markers."""
    assert _has_transition_phrase("Now let's move to the next topic")
    assert _has_transition_phrase("Moving on to the city center")
    assert _has_transition_phrase("Finally, we reach our destination")
    assert not _has_transition_phrase("The weather is nice today")
```

### Testing Boundary Score Computation

```python
def test_computes_gap_scores():
    """Test semantic gap computation between adjacent segments."""
    # Orthogonal vectors should have high gap score
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],  # Different direction
    ])
    result = compute_boundary_scores(embeddings)
    assert result[0] == pytest.approx(1.0, abs=0.01)  # Max gap
```

## Testing Coverage Resolution

The coverage pass ensures complete transcript coverage without overlaps.

### Testing Overlap Resolution

```python
def test_overlap_resolution():
    """Test overlapping chapters are resolved."""
    chapters = [
        ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=7, confidence=0.8),
        ChapterCandidate(chapter_id=1, start_segment_idx=5, end_segment_idx=12, confidence=0.7),
    ]
    result = _resolve_overlaps(chapters)

    assert len(result) == 2
    assert result[0].end_segment_idx <= result[1].start_segment_idx  # No overlap
```

### Testing Gap Filling

```python
def test_fills_gap_at_start():
    """Test fills gap at transcript start."""
    chapters = [ChapterCandidate(chapter_id=0, start_segment_idx=5, end_segment_idx=9)]

    result = _fill_gaps(chapters, total_segments=10)

    assert len(result) >= 2
    intro = [c for c in result if c.start_segment_idx == 0][0]
    assert intro.title == "Introduction"
```

### Testing Tiny Chapter Merging

```python
def test_merges_small_chapter():
    """Test small chapters are merged with adjacent ones."""
    chapters = [
        ChapterCandidate(chapter_id=0, start_segment_idx=0, end_segment_idx=5, title="Big"),
        ChapterCandidate(chapter_id=1, start_segment_idx=6, end_segment_idx=7, title="Tiny"),  # 2 segments
    ]
    result = _merge_tiny_chapters(chapters, min_segments=3)

    assert len(result) == 1
    assert result[0].end_segment_idx == 7  # Merged
```

## Running Tests

```bash
# Run all chapter detection tests
pytest tests/test_chapter_detection/ -v

# Run specific test file
pytest tests/test_chapter_detection/test_chunking.py -v

# Run specific test class
pytest tests/test_chapter_detection/test_coverage.py::TestResolveOverlaps -v

# Run with markers
pytest tests/test_chapter_detection/ -m "fast" -v

# Run single test
pytest tests/test_chapter_detection/test_detector.py::TestEnhancedChapterDetector::test_detect_chapters_empty -v
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.fast` | Quick unit tests |
| `@pytest.mark.integration` | Tests requiring real LLM calls |

## Related Documentation

- `src/chapter_detection/README.md` - Implementation details
- `CLAUDE.md` - Chapter detection configuration section
- `docs/chapter-detection.md` - Architecture overview
