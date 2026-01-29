# Property-Based Testing Guide

Property-based testing uses [hypothesis](https://hypothesis.readthedocs.io/) to automatically generate test inputs and verify that mathematical properties hold across all possible inputs.

## Overview

Unlike example-based tests that check specific inputs/outputs, property tests verify **invariants** that must hold for ALL inputs:
- Confidence scores always in `[0, 1]`
- Cosine similarity is symmetric: `sim(a, b) == sim(b, a)`
- Penalties never increase confidence

## Quick Start

```bash
# Install hypothesis
pip install hypothesis

# Run property tests
pytest tests/test_matching_properties.py -v

# Run with more examples (slower, more thorough)
pytest tests/test_matching_properties.py -v --hypothesis-seed=0

# Show hypothesis statistics
pytest tests/test_matching_properties.py -v --hypothesis-show-statistics
```

## Test File Location

All property tests are in `tests/test_matching_properties.py`.

## Properties Tested

### Confidence Score Properties

| Property | Function | Invariant |
|----------|----------|-----------|
| Adaptive threshold bounded | `calculate_adaptive_threshold` | Result in `[0.5, 0.99]` |
| B-roll boost bounded | `apply_broll_boost` | Result in `[0, 1]`, never decreases |
| Caption quality bounded | `apply_caption_quality_adjustment` | Result in `[0, 1]` |
| Timing penalty monotonic | `apply_timing_penalty` | Never increases confidence |
| Pool normalization bounded | `normalize_confidence_by_pool` | Result in `[0, 1]` |

### Embedding Similarity Properties

| Property | Function | Invariant |
|----------|----------|-----------|
| Bounded | `_compute_cosine_similarity` | Result in `[-1, 1]` |
| Self-similarity | `_compute_cosine_similarity` | `sim(v, v) == 1.0` |
| Symmetric | `_compute_cosine_similarity` | `sim(a, b) == sim(b, a)` |

### Match Ranking Properties

| Property | Function | Invariant |
|----------|----------|-----------|
| Keyword overlap bounded | `calculate_keyword_overlap_score` | Result in `[0, 1]` |
| Keyword overlap symmetric | `calculate_keyword_overlap_score` | Same matches both directions |
| Entity match bounded | `calculate_entity_match_score` | Result in `[0, 1]` |
| Multimodal bounded | `compute_multimodal_score` | Result in `[0, 1]` |

### Transcript Quality Properties

| Property | Function | Invariant |
|----------|----------|-----------|
| Score bounded | `calculate_transcript_quality` | Result in `[0, 1]` |
| Tier consistent | `calculate_transcript_quality` | Tier matches score thresholds |

## Writing New Property Tests

### Basic Pattern

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    boost=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
)
@settings(max_examples=100)
def test_boost_bounded(confidence: float, boost: float):
    """Result should never exceed 1.0."""
    result = apply_some_boost(confidence, boost)
    assert 0.0 <= result <= 1.0
```

### Custom Strategies

```python
# Confidence scores
confidence_scores = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# Embedding vectors
def embedding_vector(dim: int = 384):
    return st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        min_size=dim, max_size=dim
    )

# Keywords
keyword_lists = st.lists(
    st.text(min_size=1, max_size=50),
    min_size=0, max_size=20
)
```

### Adding Explicit Edge Cases

```python
from hypothesis import example

@given(confidence_scores)
@example(0.0)   # Test minimum
@example(1.0)   # Test maximum
@example(0.5)   # Test middle
def test_with_edge_cases(confidence: float):
    result = process_confidence(confidence)
    assert 0.0 <= result <= 1.0
```

## Configuration

### Settings Decorator

```python
@settings(
    max_examples=100,           # Number of test cases
    deadline=None,              # Disable timing checks
    suppress_health_check=[HealthCheck.too_slow],
)
```

### pytest.ini Settings

```ini
[tool:pytest]
# Hypothesis settings
addopts = --hypothesis-seed=0  # Reproducible randomness
```

## When to Use Property Testing

**Good candidates:**
- Mathematical functions (scores, similarities, distances)
- Functions with clear invariants (bounded outputs, monotonic)
- Serialization/deserialization roundtrips
- Functions that should be symmetric or commutative

**Not ideal for:**
- Complex I/O operations
- Tests requiring specific external state
- Tests where behavior depends on complex business logic

## Common Patterns

### Testing Idempotence

```python
@given(st.text())
def test_normalize_idempotent(text: str):
    """Normalizing twice should equal normalizing once."""
    once = normalize(text)
    twice = normalize(normalize(text))
    assert once == twice
```

### Testing Inverse Operations

```python
@given(st.binary())
def test_encode_decode_roundtrip(data: bytes):
    """Encoding then decoding should return original."""
    assert decode(encode(data)) == data
```

### Testing Monotonicity

```python
@given(
    base=st.floats(0, 1),
    penalty=st.floats(0, 0.5),
)
def test_penalty_monotonic(base: float, penalty: float):
    """Higher penalty should always reduce result."""
    result_low = apply_penalty(base, penalty * 0.5)
    result_high = apply_penalty(base, penalty)
    assert result_high <= result_low
```

## Debugging Failures

When hypothesis finds a failing case:

1. The error shows the **minimal** failing input
2. Reproduce with the exact values shown
3. Add the failing case as an `@example()` for regression

```bash
# Run with verbose output
pytest tests/test_matching_properties.py -v --tb=long

# Show the test database (cached failing examples)
pytest --hypothesis-show-statistics
```

## References

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Hypothesis Strategies](https://hypothesis.readthedocs.io/en/latest/data.html)
- [Property-Based Testing Intro](https://hypothesis.works/articles/what-is-property-based-testing/)
