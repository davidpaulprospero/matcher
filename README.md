# Matcher Pipeline Stages

![Tests](https://github.com/davidpaulprospero/matcher-pipeline-stages/workflows/Tests/badge.svg)
![Quick Check](https://github.com/davidpaulprospero/matcher-pipeline-stages/workflows/Quick%20Check/badge.svg)
[![codecov](https://codecov.io/gh/davidpaulprospero/matcher-pipeline-stages/branch/main/graph/badge.svg)](https://codecov.io/gh/davidpaulprospero/matcher-pipeline-stages)

An intelligent video matching pipeline that automatically synchronizes stock footage with voiceover scripts using AI-powered semantic matching.

## Features

- **AI-Powered Matching**: Uses LLM and embedding-based semantic matching to find the best video clips for each voiceover segment
- **Audio-First Mode**: Downloads only audio first, then fetches matched video segments (~95% bandwidth savings)
- **Multi-Track Output**: Generates OTIO timelines with 10 video tracks (primary, alternatives, diversity tracks, B-roll, entity images/videos)
- **Location-Aware Matching**: Filters candidates by geographic proximity for travel content
- **Entity Detection**: Automatically finds and downloads relevant images/videos for detected entities
- **Vision API**: Generates semantic descriptions of silent footage for better matching
- **Modular Architecture**: Clean separation of concerns with 10 pipeline stages

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic pipeline
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"

# Resume interrupted run
python main.py --resume

# Re-run matching only (after config tweaks)
python main.py --match-only
```

## Pipeline Stages

| Stage | Purpose |
|-------|---------|
| ANALYZE | Extract keywords, topics, entities, and location chapters |
| ENTITY_IMAGES | Download entity images (Google, Bing, Pexels) |
| ENTITY_VIDEOS | Download stock videos for entities |
| DOWNLOAD | YouTube video/audio download |
| STOCK | Download generic stock footage (B-roll) |
| REMIX | Filter videos by keyword relevance |
| TRANSCRIBE | Whisper transcription + embeddings |
| SCENE_DETECTION | Scene boundaries + face detection for B-roll |
| MATCH | Embedding + LLM matching |
| OUTPUT | OTIO, EDL, XML generation |

## OTIO Track Layout

| Track | Purpose | Default State |
|-------|---------|---------------|
| V1 | Primary video | Enabled |
| V2-V3 | Alternatives 1-2 | Disabled |
| V4-V6 | Secondary (diversity-scored) | Disabled |
| V7 | Embedding-Diversity strategy | Disabled |
| V8 | B-roll Only (silent footage) | Disabled |
| V9 | Entity Images (Google stills) | Disabled |
| V10 | Stock Videos (Pexels/Pixabay) | Disabled |
| A1-A8 | Corresponding audio | Matches video |
| A9 | Voiceover | Enabled |

## Configuration

Configuration is managed via `config.yaml`. See [config.yaml](config.yaml) for all available options.

Key settings:
- **LLM Provider**: Gemini, Anthropic, or Ollama
- **Matching Strategy**: Embedding, LLM, or hybrid
- **Download Options**: Audio-first mode, tier timeouts, live stream filtering
- **Output Options**: Number of alternatives, strategy tracks, format

## Development

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Fast unit tests only
python -m pytest tests/test_keyword_extractor/ tests/test_llm_client/ -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Suite

- **464 total tests** with **91% pass rate** (423 passing)
- Unit tests (76%), integration tests (19%), e2e tests (5%)
- Comprehensive coverage of refactored modules (keyword extractor: 83%, LLM client: 78%, vision: 65%)

### CI/CD

- **Full Tests**: Matrix testing across Python 3.9-3.11 on Ubuntu/Windows
- **Quick Check**: Fast syntax validation and core unit tests (< 10 minutes)

## Architecture

### Refactored Modules

The codebase has undergone extensive refactoring into modular packages:

- `src/llm_client/` - Unified LLM abstraction (Gemini, Anthropic, Ollama)
- `src/keyword_extractor/` - Keyword extraction with LLM fallback
- `src/matching/` - Tiered matching with multiple strategies
- `src/otio/` - OTIO timeline generation and export
- `src/media_sources/` - Image/video search across multiple providers
- `src/transcription/` - Whisper transcription with parallel processing
- `src/downloader/` - YouTube download with audio-first pipeline
- `src/config/` - Modular configuration management

See [REFACTORING.md](REFACTORING.md) for the full refactoring roadmap.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project guide and conventions
- [REFACTORING.md](REFACTORING.md) - Refactoring roadmap and architecture
- [CHANGELOG.md](CHANGELOG.md) - Version history and migration guides
- [TEST_COVERAGE.md](TEST_COVERAGE.md) - Test suite organization
- [COVERAGE_SUMMARY.md](COVERAGE_SUMMARY.md) - Code coverage analysis
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
- [EXAMPLES.md](EXAMPLES.md) - Usage examples
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

## Requirements

- Python 3.9+
- FFmpeg (for video processing)
- API keys for LLM providers (Gemini, Anthropic, or Ollama)
- Optional: GeoNames account for location matching

## License

[Add your license here]

## Contributing

See [CLAUDE.md](CLAUDE.md) for development conventions and testing guidelines.
