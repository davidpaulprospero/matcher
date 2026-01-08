# Testing Guide

Comprehensive guide for running and writing tests in the matcher-pipeline-stages project.

## Test Suite Overview

The project has **464 total tests** with a **100% pass rate**:
- **443 passing** - Unit and integration tests
- **21 skipped** - Integration tests requiring external resources

### Test Categories

| Category | Count | Pass Rate | Description |
|----------|-------|-----------|-------------|
| **Unit Tests** | 352 (76%) | 100% | Fast, isolated tests with mocked dependencies |
| **Integration Tests** | 91 (19%) | 100% | Tests with real API calls and file I/O |
| **End-to-End Tests** | 21 (5%) | Skipped* | Full pipeline tests requiring external resources |

*Skipped in standard pytest runs, can be run as standalone scripts

## Running Tests

### Quick Commands

```bash
# Run all tests (unit + integration, ~25 seconds)
python -m pytest tests/ -v

# Run fast unit tests only (~10 seconds)
python -m pytest tests/test_keyword_extractor/ tests/test_llm_client/ tests/test_otio/ -v

# Run with coverage report
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
python -m pytest tests/test_otio_timeline.py -v

# Run specific test
python -m pytest tests/test_otio_timeline.py::test_creates_basic_timeline -v

# Run tests matching pattern
python -m pytest tests/ -k "llm" -v

# Skip integration tests (already default behavior)
python -m pytest tests/ -m "not integration" -v
```

### Test Output Options

```bash
# Verbose output with test names
pytest tests/ -v

# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l

# Quiet mode (less output)
pytest tests/ -q

# Show slowest 10 tests
pytest tests/ --durations=10
```

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                    # Pytest configuration and fixtures
├── test_llm_client/              # LLM client tests (71 tests, 100%)
│   ├── conftest.py
│   ├── test_base.py
│   ├── test_cache.py
│   ├── test_factory.py
│   ├── test_parsers.py
│   └── test_retry.py
├── test_keyword_extractor/        # Keyword extraction tests (180 tests, 100%)
│   ├── test_core.py
│   ├── test_entity_extractor.py
│   ├── test_llm_extractor.py
│   └── ... (10 test files)
├── test_otio_timeline.py          # OTIO timeline tests (15 tests)
├── test_otio_tracks.py            # OTIO track builder tests (23 tests)
├── test_otio_integration.py       # OTIO integration tests (27 tests)
├── test_vision.py                 # Vision API tests (19 tests)
├── test_media_sources.py          # Media sources unit tests (9 tests)
├── test_media_sources_integration.py  # Media sources integration (18 tests)
├── test_cache.py                  # Cache system tests
├── test_core.py                   # Core component tests (12 tests, 10 integration)
├── test_download.py               # Download tests (5 tests, 2 integration)
├── test_features.py               # Feature tests (9 tests, 5 integration)
├── test_matching.py               # Matching tests (4 integration tests)
└── test_recent_features.py        # Recent feature validation
```

### Test Markers

Tests are marked with pytest markers for categorization:

```python
@pytest.mark.integration  # Requires external resources (API keys, videos, etc.)
```

## Integration Tests

### What Are Integration Tests?

Integration tests require external resources that aren't available in standard pytest runs:
- Real video files
- API keys (Gemini, Anthropic, etc.)
- Downloaded content
- Live network calls

### Running Integration Tests

Integration tests are marked with `@pytest.mark.integration` and are automatically skipped in standard pytest runs.

#### Option 1: Run as Standalone Scripts

Most integration test files can be run directly as Python scripts with their own test runners:

```bash
# Core component tests (keyword extraction, download, transcription, matching, OTIO)
python tests/test_core.py

# Download tests (segment merging, audio-first pipeline)
python tests/test_download.py

# Feature tests (face detection, logger stats)
python tests/test_features.py

# Matching tests (confidence thresholds, reuse prevention)
python tests/test_matching.py
```

These scripts have command-line options:

```bash
# Show available options
python tests/test_core.py --help

# Run with verbose output
python tests/test_core.py --verbose

# Keep test files after run
python tests/test_core.py --keep

# Skip download phase (use existing files)
python tests/test_core.py --skip-download
```

#### Option 2: Provide Real Fixtures

Modify `tests/conftest.py` to provide actual resources instead of skipping:

```python
# Example: Provide real video path
@pytest.fixture
def video_path(tmp_path) -> str:
    # Download or copy a real test video
    video_file = tmp_path / "test_video.mp4"
    # ... download or copy logic ...
    return str(video_file)

@pytest.fixture
def config():
    from src.config import load_config
    return load_config()
```

Then run with:

```bash
# Run integration tests with real fixtures
python -m pytest tests/ -m integration -v
```

## Writing Tests

### Unit Test Template

```python
"""
Unit tests for src/module_name.py
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.module_name import function_to_test


class TestFunctionName:
    """Test function_to_test() function."""

    def test_basic_functionality(self):
        """Test basic happy path."""
        result = function_to_test(input_data)
        assert result == expected_output

    def test_edge_case(self):
        """Test edge case handling."""
        result = function_to_test(edge_case_input)
        assert result is not None

    def test_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError, match="Expected error message"):
            function_to_test(invalid_input)

    @patch('src.module_name.external_dependency')
    def test_with_mock(self, mock_dependency):
        """Test with mocked dependency."""
        mock_dependency.return_value = mock_data
        result = function_to_test(input_data)
        mock_dependency.assert_called_once_with(expected_args)
        assert result == expected_output
```

### Integration Test Template

```python
"""
Integration tests for src/module_name.py
"""

import pytest
from pathlib import Path


@pytest.mark.integration
def test_integration_with_real_api(api_key: str, output_dir: Path):
    """Test with real API calls."""
    from src.module_name import APIClient

    client = APIClient(api_key)
    result = client.process(test_data)

    assert result is not None
    assert output_dir.exists()
```

### Mock Best Practices

#### 1. Use Dataclasses for Complex Mocks

```python
from dataclasses import dataclass

@dataclass
class MockMatch:
    """Mock Match object with all required attributes."""
    file: str
    start: float
    end: float
    confidence: float = 0.8
    speed: float = 1.0
    reasoning: str = "Test reasoning"
    # ... add all attributes used by code under test
```

#### 2. Use Fixtures for Reusable Mocks

```python
# In conftest.py or test file
@pytest.fixture
def mock_config():
    """Mock configuration object."""
    config = Mock()
    config.llm.provider = "gemini"
    config.llm.model = "gemini-2.0-flash"
    return config

# In test
def test_with_config(mock_config):
    result = function_that_needs_config(mock_config)
    assert result is not None
```

#### 3. Patch at the Right Level

```python
# BAD: Patching too high
@patch('google.generativeai.GenerativeModel')
def test_bad(mock_model):
    ...

# GOOD: Patch where it's imported
@patch('src.llm_client.providers.gemini.genai.GenerativeModel')
def test_good(mock_model):
    ...
```

## Test Coverage

### Viewing Coverage

```bash
# Generate HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Open report in browser
# Windows
start htmlcov/index.html

# Linux/Mac
open htmlcov/index.html
```

### Coverage by Module

| Module | Coverage | Tests | Status |
|--------|----------|-------|--------|
| `src/llm_client/` | 78% | 71 | ✅ Excellent |
| `src/keyword_extractor/` | 83% | 180 | ✅ Excellent |
| `src/otio/` | 40% | 65 | ⚠️ Good |
| `src/otio/timeline.py` | 74% | 15 | ✅ Very Good |
| `src/otio/tracks.py` | 97% | 23 | ✅ Excellent |
| `src/media_sources/` | 65% | 27 | ✅ Good |
| `src/vision.py` | 65% | 19 | ✅ Good |
| `src/matching/` | 45% | 12 | ⚠️ Needs Work |
| `src/transcription/` | 35% | Integration | ⚠️ Needs Work |

### Coverage Goals

- **Critical modules** (OTIO, LLM client, keyword extraction): ≥70%
- **Core modules** (matching, transcription): ≥50%
- **Utility modules**: ≥40%

## Continuous Integration

### GitHub Actions Workflows

The project has two CI workflows:

#### 1. Full Test Suite (`.github/workflows/tests.yml`)

Runs on: Push to `main`, Pull requests

Matrix testing:
- Python versions: 3.9, 3.10, 3.11
- Operating systems: Ubuntu, Windows

Steps:
1. Install system dependencies (ffmpeg)
2. Install Python dependencies
3. Run pytest with coverage
4. Upload coverage to Codecov
5. Run linters (separate job)

#### 2. Quick Check (`.github/workflows/quick-check.yml`)

Runs on: Push to any branch

Quick validation (< 10 minutes):
1. Syntax validation
2. Core unit tests only (fast tests)
3. Import validation

### Running CI Locally

```bash
# Install dependencies like CI
pip install -r requirements.txt

# Run tests like CI
python -m pytest tests/ -v --tb=short

# Run linting
flake8 src/ tests/ --max-line-length=120
```

## Common Testing Patterns

### 1. Testing with Temporary Files

```python
def test_with_temp_file(tmp_path):
    """Test file operations with pytest's tmp_path fixture."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test data")

    result = process_file(str(test_file))

    assert test_file.exists()
    assert result is not None
```

### 2. Testing LLM Calls

```python
@patch('src.llm_client.create_client')
def test_llm_integration(mock_create_client):
    """Test LLM integration with mocked client."""
    mock_client = Mock()
    mock_client.generate.return_value = Mock(
        text='{"keywords": ["test", "example"]}',
        parsed_data={"keywords": ["test", "example"]}
    )
    mock_create_client.return_value = mock_client

    result = extract_keywords(text="test content")

    assert len(result.keywords) == 2
    mock_client.generate.assert_called_once()
```

### 3. Testing OTIO Timeline Generation

```python
def test_otio_timeline_creation():
    """Test OTIO timeline with minimal mocks."""
    from src.otio.timeline import create_timeline

    matches = [MockMatchResult(
        primary=MockMatch(file="test.mp4", start=0.0, end=5.0),
        alternatives=[],
        secondaries=[],
        strategies={}
    )]
    config = MockConfig(output=MockOutputConfig())

    timeline = create_timeline(matches, config)

    assert isinstance(timeline, otio.schema.Timeline)
    assert len(timeline.tracks) > 0
```

## Troubleshooting

### Common Test Failures

#### 1. Import Errors

```
ImportError: cannot import name 'X' from 'src.module'
```

**Fix**: Check `__all__` exports in `__init__.py`:

```python
# src/module/__init__.py
__all__ = ['X', 'Y', 'Z']  # Make sure X is listed
```

#### 2. Fixture Not Found

```
fixture 'video_path' not found
```

**Fix**: Add fixture to `conftest.py` or mark test as integration:

```python
@pytest.mark.integration
def test_needs_video(video_path):
    ...
```

#### 3. Mock Attribute Errors

```
AttributeError: Mock object has no attribute 'field_name'
```

**Fix**: Add all required attributes to mock:

```python
@dataclass
class MockObject:
    required_field: str
    optional_field: str = "default"
```

#### 4. Numpy Array Truth Value

```
ValueError: The truth value of an array is ambiguous
```

**Fix**: Use `is_embeddings_empty()` helper (Rule 7):

```python
from src.utils import is_embeddings_empty

# BAD
if embeddings:
    ...

# GOOD
if not is_embeddings_empty(embeddings):
    ...
```

### Getting Help

- Check [CLAUDE.md](CLAUDE.md) for project conventions
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Review existing tests in `tests/` for patterns
- Run tests with `-vv` for very verbose output

## Test Maintenance

### When to Update Tests

1. **After refactoring**: Update mocks to match new interfaces
2. **After adding features**: Add new test cases
3. **After fixing bugs**: Add regression tests
4. **After changing config**: Update config mocks

### Test Quality Checklist

- [ ] Test name describes what is being tested
- [ ] Test has clear arrange/act/assert structure
- [ ] Test is independent (no shared state)
- [ ] Test uses minimal mocking (only external dependencies)
- [ ] Test covers happy path and edge cases
- [ ] Test has meaningful assertions (not just "doesn't crash")
- [ ] Integration tests are marked with `@pytest.mark.integration`

## Performance

### Test Suite Performance

- **Full suite**: ~25 seconds (443 tests)
- **Fast unit tests**: ~10 seconds (352 tests)
- **Integration tests**: Varies (API-dependent)

### Speeding Up Tests

```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/ -n auto

# Run only failed tests from last run
pytest tests/ --lf

# Run tests that failed last time first
pytest tests/ --ff
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock documentation](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Project CLAUDE.md](CLAUDE.md) - Development conventions
- [Project TEST_COVERAGE.md](TEST_COVERAGE.md) - Detailed coverage analysis
