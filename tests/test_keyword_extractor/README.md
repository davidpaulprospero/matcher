# test_keyword_extractor - Keyword Extraction Tests

This directory contains tests for the keyword extraction system, which extracts and prioritizes keywords from voiceover scripts for YouTube video search.

## Directory Overview

| File | Purpose |
|------|---------|
| `__init__.py` | Package initialization with module summary |
| `test_core.py` | LLMKeywordExtractor integration, initialization, LLM calls |
| `test_entity_extractor.py` | Entity extraction from text, JSON parsing |
| `test_models.py` | KeywordResult and PrioritizedKeyword dataclasses |
| `test_prioritizer.py` | Priority scoring algorithm, entity/topic/general weighting |
| `test_prompts.py` | LLM prompt templates (6 prompts), placeholder validation |
| `test_segment_processor.py` | Per-segment keyword extraction, batch processing |
| `test_topic_detector.py` | Topic detection with LLM mocking |
| `test_utils.py` | SRT parsing, keyword matching utilities |
| `test_validator.py` | Visual keyword validation, abstract pattern filtering |

## Key Testing Patterns

### Mocking LLM Calls

The keyword extractor relies heavily on LLM calls. Tests mock these at different levels:

#### Mocking the LLM Client Creation

```python
from unittest.mock import Mock, patch

def test_extractor_with_mock_client(mock_config):
    with patch('src.llm_client.create_client') as mock_create:
        mock_client = Mock()
        mock_create.return_value = mock_client

        extractor = LLMKeywordExtractor(mock_config)

        assert extractor.llm_client == mock_client
```

#### Mocking the `_call_llm` Method

```python
def test_extract_keywords_basic(mock_config):
    with patch('src.llm_client.create_client'):
        extractor = LLMKeywordExtractor(mock_config)

        # Mock LLM responses in order: entities, keywords, expansion
        extractor._call_llm = Mock(side_effect=[
            '[]',  # Empty entities
            '["mountain climbing", "alpine peaks"]',  # Keywords JSON
            '["mountain climbing", "alpine peaks", "snow"]'  # Expanded keywords
        ])

        segments = [{'text': 'Documentary about mountains.'}]
        result = extractor.extract_keywords(segments, max_keywords=30)

        assert isinstance(result, KeywordResult)
        assert len(result.keywords) > 0
```

#### Mocking LLM Generate Response

```python
def test_detect_topic_basic():
    mock_llm_client = Mock()
    mock_response = Mock()
    mock_response.text = '"Wildlife Conservation"'
    mock_llm_client.generate.return_value = mock_response

    topic = detect_topic_llm("Text about animals.", mock_llm_client)

    assert topic == "Wildlife Conservation"
    mock_llm_client.generate.assert_called_once()
```

### Testing Entity Extraction

Entity extraction uses a callback pattern for LLM calls:

```python
def test_extract_entities_basic():
    mock_llm_response = '''{
        "places": [{"name": "Mount Everest", "search_keyword": "mount everest climbing"}],
        "people": [{"name": "Edmund Hillary", "search_keyword": "edmund hillary mountaineer"}]
    }'''
    mock_llm_call = Mock(return_value=mock_llm_response)

    keywords, raw_entities = extract_entities(
        text="Edmund Hillary climbed Mount Everest.",
        topic="Mountaineering",
        llm_call=mock_llm_call
    )

    assert len(raw_entities) == 2
    assert len(keywords) == 2
    mock_llm_call.assert_called_once()
```

### Testing Prioritizer Scoring

The prioritizer uses a scoring algorithm with multiple factors:

```python
def test_build_prioritized_mixed_sources():
    entity_keywords = ["Paris"]
    raw_entities = [{"name": "Paris", "type": "LOCATION"}]

    result = build_prioritized_keywords(
        keywords=["Paris", "travel adventures", "scenery"],
        entity_keywords=entity_keywords,
        raw_entities=raw_entities,
        full_text="Travel adventures in Paris.",
        topic="Travel Adventures"
    )

    # Entity keywords get highest priority (0.9+ base)
    paris = next(pk for pk in result if pk.keyword == "Paris")
    assert paris.source == "entity"
    assert paris.priority >= 0.9

    # Topic-matching keywords get medium priority (0.8+ base)
    travel = next(pk for pk in result if pk.keyword == "travel adventures")
    assert travel.source == "topic"
    assert travel.priority >= 0.8

    # General keywords get lower priority (0.5-0.7)
    scenery = next(pk for pk in result if pk.keyword == "scenery")
    assert scenery.source == "general"
    assert scenery.priority < 0.8
```

### Priority Scoring Factors

| Factor | Base Score | Boost |
|--------|------------|-------|
| Entity keyword | 0.9 | - |
| Topic-matching keyword | 0.8 | - |
| General keyword | 0.5 | - |
| Mention frequency | - | +0.1 max |
| Visual specificity ("4K", "drone", etc.) | - | +0.05 |
| Priority cap | 1.0 max | - |

### Testing Validator Rules

The validator filters abstract/non-visual keywords:

```python
# Valid visual keywords (return True)
@pytest.mark.parametrize("keyword", [
    "mountain climbing",
    "city skyline",
    "drone footage",
    "4K timelapse",
    "Paris Eiffel Tower",
])
def test_valid_visual_keywords(keyword):
    assert is_visual_keyword(keyword) is True

# Abstract patterns that get filtered (return False)
@pytest.mark.parametrize("keyword", [
    "the quiet confession",
    "the death of democracy",
    "what happened to society",
    "the end of civilization",
])
def test_abstract_keywords(keyword):
    assert is_visual_keyword(keyword) is False
```

#### Validator Constants

```python
from src.keyword_extractor.validator import ABSTRACT_PATTERNS, VISUAL_INDICATORS

# ABSTRACT_PATTERNS - phrases that indicate non-visual content
# Examples: "the end of", "the truth about", "what happened to"

# VISUAL_INDICATORS - words that indicate visual footage
# Examples: "4k", "drone", "aerial", "footage", "timelapse", "hotel"
```

### Testing Segment Processor

The segment processor handles batch extraction:

```python
def test_extract_keyword_per_segment_no_llm():
    """Test fallback when no LLM client available"""
    segments = [
        {"text": "Mountain climbing adventures"},
        {"text": "Ocean diving expeditions"}
    ]

    keywords = extract_keyword_per_segment(
        segments=segments,
        llm_client=None,
        llm_call_function=None,
        topic=""
    )

    # Uses TF-IDF-like fallback
    assert len(keywords) == 2
    assert all(isinstance(kw, str) for kw in keywords)
```

### Mock Config Fixture

Standard mock config for keyword extractor tests:

```python
@pytest.fixture
def mock_config():
    config = Mock()

    # LLM config
    config.llm = Mock()
    config.llm.provider = 'anthropic'
    config.llm.api_key = 'test-key'
    config.llm.model = 'claude-3-haiku-20240307'
    config.llm.max_tokens = 2000

    # Keyword config
    config.keyword = Mock()
    config.keyword.max_keywords = 30

    # Keyword weights config (for TF-IDF fallback)
    config.keyword_weights = Mock()
    config.keyword_weights.auto_detect_count = 30

    return config
```

## Testing SRT File Processing

Tests use temporary files for SRT parsing:

```python
import tempfile
import os

def test_extract_keywords_from_srt_basic():
    srt_content = """1
00:00:00,000 --> 00:00:05,000
This is about mountain climbing.

2
00:00:05,000 --> 00:00:10,000
We scaled the highest peaks.
"""
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.srt', delete=False, encoding='utf-8'
    ) as f:
        f.write(srt_content)
        srt_path = f.name

    try:
        with patch('src.keyword_extractor.utils.LLMKeywordExtractor') as MockExtractor:
            mock_instance = Mock()
            mock_instance.extract_keywords = Mock(return_value=Mock(
                keywords=["mountain climbing", "peaks"],
                segments_analyzed=2,
                extraction_method="llm"
            ))
            MockExtractor.return_value = mock_instance

            result = extract_keywords_from_srt(srt_path, mock_config)

            assert result.keywords == ["mountain climbing", "peaks"]
    finally:
        os.unlink(srt_path)
```

## Testing Prompt Templates

All 6 prompts have tests for existence, placeholders, and formatting:

```python
def test_all_prompts_exist():
    required_prompts = [
        'SEGMENT_KEYWORD_PROMPT',
        'BATCH_SEGMENT_KEYWORDS_PROMPT',
        'KEYWORD_EXTRACTION_PROMPT',
        'ENTITY_EXTRACTION_PROMPT',
        'KEYWORD_EXPANSION_PROMPT',
        'TOPIC_DETECTION_PROMPT'
    ]
    for prompt_name in required_prompts:
        assert hasattr(prompts, prompt_name)

def test_prompt_formatting():
    prompt = prompts.KEYWORD_EXTRACTION_PROMPT.format(
        voiceover_text="Documentary about mountains.",
        max_keywords=30
    )
    assert "Documentary about mountains." in prompt
    assert "30" in prompt
    assert '{' not in prompt  # No unfilled placeholders
```

## Common Edge Cases

### Empty Inputs

```python
def test_extract_keywords_empty_segments():
    result = extractor.extract_keywords([])
    assert result.keywords == []
    assert result.extraction_method == "none"
```

### LLM Error Handling

```python
def test_call_llm_error():
    mock_client.generate = Mock(side_effect=Exception("API Error"))
    result = extractor._call_llm("Test prompt")
    assert result == "[]"  # Graceful fallback
```

### Malformed JSON Responses

```python
def test_parse_keywords_json_fallback():
    response = """
    - keyword1
    - keyword2
    """
    result = extractor._parse_keywords_json(response)
    assert len(result) > 0  # Falls back to line-by-line parsing
```

### Special Characters

```python
def test_special_characters_in_keywords():
    result = build_prioritized_keywords(
        keywords=["Sao Paulo", "cafe culture"],
        entity_keywords=[],
        raw_entities=[],
        full_text="Sao Paulo's cafe culture is vibrant.",
        topic=""
    )
    assert len(result) == 2
```

## Running Tests

```bash
# Run all keyword extractor tests
pytest tests/test_keyword_extractor/ -v

# Run specific test file
pytest tests/test_keyword_extractor/test_validator.py -v

# Run parameterized validator tests
pytest tests/test_keyword_extractor/test_validator.py::TestIsVisualKeyword -v

# Run with coverage
pytest tests/test_keyword_extractor/ --cov=src/keyword_extractor

# Run single test
pytest tests/test_keyword_extractor/test_prioritizer.py::TestBuildPrioritizedKeywords::test_build_prioritized_entity_keywords -v
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.parametrize` | Used extensively for validator pattern testing |
| `@pytest.mark.fast` | Quick unit tests |

## Related Documentation

- `src/keyword_extractor/README.md` - Implementation details
- `CLAUDE.md` - Pipeline configuration
- `docs/keyword-extraction.md` - Keyword extraction architecture
