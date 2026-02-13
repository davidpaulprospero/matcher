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
- **Modular Architecture**: Clean separation of concerns with 7 pipeline stages

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
| ANALYZE | Extract keywords, topics, entities from voiceover |
| VIDEO_SEARCH | Search and queue videos from YouTube |
| CAPTION | Fetch YouTube captions (always on) |
| MATCH | Embedding + LLM semantic matching |
| ITERATIVE_MATCH | Gap analysis and query refinement |
| DOWNLOAD_SEGMENTS | Download matched video segments |
| OUTPUT | OTIO, EDL, XML timeline generation |

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

## Download Resilience

The downloader uses a 4-tier escalation system to handle YouTube's anti-bot protections:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Download Escalation Flow                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐    403     ┌─────────┐    403     ┌─────────┐        │
│   │ Tier 1  │ ────────►  │ Tier 2  │ ────────►  │ Tier 3  │        │
│   │ Imperson│            │ Extractor│           │ Cookie  │        │
│   └─────────┘            └─────────┘            └─────────┘        │
│       │                                              │              │
│       │ Success                             exhausted│              │
│       ▼                                              ▼              │
│   ┌─────────┐                               ┌─────────┐            │
│   │ Download│                               │ Tier 4  │            │
│   │ Complete│                               │ VPN     │            │
│   └─────────┘                               └─────────┘            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

| Tier | Strategy | Trigger | Implementation |
|------|----------|---------|----------------|
| **1** | TLS Impersonation | Always on | `--impersonate Chrome-136:Macos-15` via curl_cffi |
| **2** | Extractor Args | On 403 | `player_client=web_safari,tv_downgraded,web` |
| **3** | Cookie Rotation | Continued 403s | Rotates through configured cookie files |
| **4** | VPN Rotation | Cookies exhausted | Mullvad VPN IP rotation |

### Key Features

- **Automatic escalation**: Each tier triggers on failure of the previous
- **Budget tracking**: Per-tier rate limits prevent excessive retries
- **Circuit breaker**: Pauses after repeated failures to avoid bans
- **Checkpoint persistence**: Escalation state survives pipeline restarts

### Configuration

```yaml
download:
  # Tier 4: Mullvad VPN (optional)
  mullvad:
    enabled: true
    preferred_countries: ['us', 'gb', 'de', 'nl']
    rotation_strategy: 'random'  # 'random', 'sequential', 'nearest'
    max_rotations_per_session: 5
```

See [CLAUDE.md](CLAUDE.md#bypass--escalation-yt-dlp) for detailed configuration options.

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
- **Coverage Check**: Enforces 85% minimum coverage via `.coveragerc` (`fail_under = 85`)
- **Coverage Report**: Run `pytest --cov=src --cov-report=html` to generate HTML report in `htmlcov/`

#### Coverage Checklist

```bash
# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing

# Generate HTML report
python -m pytest tests/ --cov=src --cov-report=html

# Check specific module coverage
python -m pytest tests/ --cov=src --cov-report=term --cov-report=xml

# View coverage summary (requires coverage installed)
coverage report
coverage html
```

**Coverage Goals:**
- Overall: 85% minimum (enforced by CI)
- Core modules (matching, LLM client): 80%+
- New code: 90%+

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
