# Changelog

All notable changes to the matcher-pipeline-stages project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD workflows for automated testing
  - `.github/workflows/tests.yml`: Full matrix testing across Python 3.9-3.11 on Ubuntu/Windows
  - `.github/workflows/quick-check.yml`: Fast checks for all branches (< 10 minutes)
- Comprehensive code coverage reporting with pytest-cov
  - `COVERAGE_SUMMARY.md`: Detailed coverage analysis by module
  - HTML coverage reports in `htmlcov/` directory
- Test suite organization and documentation
  - 426 total tests with 92% pass rate
  - Unit tests (76%), integration tests (19%), e2e tests (5%)
- Test for audio downloads checkpoint restore
  - `test_restore_audio_downloads`: Validates proper restoration of `downloaded_audio` list from checkpoint
  - `test_restore_audio_downloads_legacy_field_names`: Tests backward compatibility with old field names
  - Ensures AudioDownload objects have correct field names (56/56 download stage tests passing)
- **Obsolete Field Removal Tests** (`tests/test_obsolete_fields_fix.py`)
  - `test_restore_audio_downloads_with_obsolete_fields`: Validates removal of obsolete fields during checkpoint restore
  - `test_audio_download_dataclass_fields`: Verifies AudioDownload only has expected fields
  - Comprehensive backward compatibility testing for legacy checkpoints (2 new tests, 100% passing)

### Fixed
- **AudioDownload Checkpoint Restore Bug**: Obsolete fields causing restore failures
  - Root cause: Old checkpoints contained obsolete fields (`channel`, `duration_tier`, `upload_date`, `license`) that are no longer in the AudioDownload dataclass
  - Symptoms: `AudioDownload.__init__() got an unexpected keyword argument 'channel'` error during checkpoint resume
  - Fixed: Added obsolete field removal in [src/stages/download.py:118-121](src/stages/download.py#L118-L121)
  - Impact: Checkpoints from pre-refactoring versions now restore successfully without errors
  - Tests: Added comprehensive backward compatibility tests in `tests/test_obsolete_fields_fix.py` (96 total tests passing)
- **Empty Transcriptions Bug**: VAD filter too aggressive for YouTube videos
  - Root cause: `vad_filter: true` default was filtering out all speech from YouTube audio downloads
  - Symptoms: "No text segments to embed" error, empty transcription cache files (`[]`)
  - Fixed: Changed dataclass default `vad_filter: bool = False` in [src/config/sections/core.py:64](src/config/sections/core.py#L64)
  - Fixed: Changed function defaults to `False` in parallel_processor.py (lines 64, 71, 230) and whisper_client.py (line 135)
  - Fixed: Changed config.yaml default to `false` (line 82)
  - Fixed: Added explicit `vad_filter=False` in sequential transcription (src/stages/transcribe.py:239)
  - Impact: YouTube videos now transcribe correctly with all speech detected
  - **ACTION REQUIRED**: Clear transcription cache to re-transcribe: `rm -rf .cache/transcriptions`
  - Note: VAD can be re-enabled in config.yaml if needed for high-quality voiceover audio
- **Audio-First Mode Checkpoint Restore Bug**: REMIX stage `'AudioDownload' object has no attribute 'file'` error
  - Root cause 1: `DownloadStage.restore()` was only restoring `downloaded_videos`, not `downloaded_audio`
  - Root cause 2: Old checkpoints had incorrect field names (`audio_file`, `video_url` instead of `file`, `url`)
  - Fixed: Added audio downloads restoration in checkpoint restore logic (src/stages/download.py:102-121)
  - Fixed: Added backward compatibility mapping for old checkpoint field names (audio_file→file, video_url→url)
  - Impact: Audio-first mode now properly resumes from checkpoint without re-downloading audio
  - Tests: Added `test_restore_audio_downloads` and `test_restore_audio_downloads_legacy_field_names` (56/56 tests passing)
- **Unicode Encoding Errors**: Windows console compatibility fixes
  - Replaced emoji characters (⚠️, ✓, →, ─) with ASCII equivalents (!, +, ->, -)
  - Fixed in `src/stages/download.py` (line 420) and `src/stages/remix.py` (lines 85, 136, 147, 164, 169)
  - Prevents `'charmap' codec can't encode characters` errors on Windows
- **Checkpoint Stage Order**: Added missing SCENE_DETECTION to STAGE_ORDER
  - Fixed: Added "SCENE_DETECTION" to checkpoint.py STAGE_ORDER (line 32)
  - Impact: Checkpoint resume now recognizes SCENE_DETECTION as a valid completed stage
  - Prevents "Unknown stage 'SCENE_DETECTION' in checkpoint" warning
- pytest collection warning in `tests/test_recent_features.py` (renamed class to avoid "Test" prefix)

## [0.3.0] - 2026-01-07

### Added
- **Vision API Integration** for silent video processing
  - Gemini Vision API integration for semantic descriptions of silent footage
  - Automatic detection of silent videos in SceneDetectionStage
  - Vision response caching in `.cache/vision_cache/`
  - Cost tracking and API call monitoring
  - 19 comprehensive unit tests with 100% pass rate
  - See `tests/test_vision.py` for full test coverage
- **Config Modularization**: Split monolithic `config.py` into organized package
  - 13 focused modules in `src/config/` (2,421 total lines)
  - Sections: infrastructure, core, matching, llm, download, keywords, entity, duration, output, media
  - 100% backward compatible with import preservation
  - Average module size reduced from 1,929 to ~186 lines

### Fixed
- **Audio-First Mode OTIO Bug**: DOWNLOAD_SEGMENTS stage now correctly remaps MatchResult objects
  - Fixed .mp3 → .mp4 video segment references for V1-V10 tracks
  - Properly handles primary + alternatives + strategy matches
- **YouTube Search Timeout**: Enhanced yt-dlp diagnostics with stderr logging
  - Root cause: Firefox locking cookies database (close browser before pipeline runs)
  - Added return code checking and output line counting in title_filter.py
- TranscriptCache import typo in transcribe.py (TranscriptionCache → TranscriptCache)

### Changed
- **Transcription Module Refactoring** (Phase 2 completed)
  - Extracted `parallel_processor.py` with 5 orchestration functions (435 lines)
  - Updated `__init__.py` to import directly from parallel_processor
  - Moved `TranscriptSegment` to `src/state.py` (canonical location)
  - Deprecated original `transcription.py` → `_transcription_legacy_backup.py`
  - Complete 6-module package (1,299 lines), 100% backward compatible

## [0.2.0] - 2026-01-06

### Added
- **LLM Client Abstraction Package** (`src/llm_client/`)
  - Unified interface for Gemini, Anthropic, and Ollama providers
  - Automatic retry with exponential backoff
  - File-based response caching with TTL support
  - JSON parsing with fallback strategies
  - Consistent error handling across providers
  - Eliminated ~290 lines of duplicated LLM integration code
  - Migrated 7 files to use unified client (matching, keyword extraction, topic detection, etc.)
- **Development Rule 9**: LLM Client Usage
  - Always use `src/llm_client/` for all LLM operations
  - Never directly initialize provider SDKs
- **Development Rule 10**: Avoid Dataclass Duplication
  - Import from canonical locations (`src/state.py` for pipeline data)
  - Prevent field name mismatches and runtime errors

### Fixed
- **AudioDownload Duplicate Class Bug**: Consolidated duplicate definitions
  - Removed duplicate in `src/downloader.py` (had wrong field names: `audio_file` vs `file`)
  - Canonical definition remains in `src/state.py`
  - Fixes "unexpected keyword argument 'file'" runtime error
- **V8 B-roll Track**: Fixed 0 clips → 238 clips
  - SceneDetectionStage now propagates `is_broll` to `text_metadata`
  - MatchStage restores `is_broll` from metadata to SRTSegment objects
  - Face detection data flow: face_score → is_broll → metadata → matching

### Refactored
- **OTIO Builder Package** (`src/otio/`)
  - Created 9 focused modules from 3,175-line monolith
  - Migrated ALL 6 public functions including `generate_resolve_xml_with_bins()`
  - Created `xml_export.py` for FCP7 XML generation
  - 100% backward compatible
  - 27 tests passing (16 unit + 11 integration)
  - Average module size ~300 lines
- **Matching Package** (`src/matching/`)
  - Created 7 focused modules from 2,872-line file
  - Modules: tracking, llm_providers, strategies, scoring, location_matching, main
  - 21% code reduction (2,872 → 2,269 lines)
  - All LLM providers now use unified client (Rule 9)
  - 6 matching strategies modularized
  - 100% backward compatible
  - Average module size ~324 lines
- **Media Sources Package** (`src/media_sources/`)
  - Split `entity_images.py` (1,818 lines) into 14 focused modules
  - Created `BaseMediaClient` abstract class
  - Eliminated ~30 LOC duplication
  - 25/27 unit tests passing
  - Updated entity_images/entity_videos stages
  - 100% backward compatible

### Documentation
- Added `REFACTORING.md` with comprehensive refactoring roadmap
- Updated `CLAUDE.md` with Vision API integration section
- Added testing checklist and dataclass import conventions

## [0.1.0] - 2026-01-05

### Added
- **Pipeline Architecture Refactor**: Modular stage classes in `src/stages/`
  - 10 pipeline stages: ANALYZE, ENTITY_IMAGES, ENTITY_VIDEOS, DOWNLOAD, STOCK, REMIX, TRANSCRIBE, SCENE_DETECTION, MATCH, OUTPUT
  - `PipelineState` for shared data management
  - Checkpoint/resume support with automatic backups
- **Entity Image Caching**: Cross-project entity cache
  - Global cache in `~/.matcher_entity_cache/`
  - Eliminates redundant image searches across projects
  - Configurable cache expiration
- **Keyword Extractor Module Testing**: 180 tests with 100% pass rate
  - 10 test modules covering all keyword extraction components
  - Fixed validator, entity_extractor, segment_processor tests
  - Fixed import errors in stages and OTIO builder

### Fixed
- Import errors in `src/stages/entity_videos.py` and `src/otio_builder.py`
  - Fixed old `entity_images` → `media_sources.models` imports
- Keyword saving bug in `main.py` (save_preset → save_keywords)

## [0.0.9] - 2026-01-04

### Added
- **Match-Only Mode** (`--match-only` flag)
  - Skip download/transcribe stages, use cached data
  - Ideal for tweaking matching config without re-downloading
- **Force Rematch** (`--force-rematch` flag)
  - Re-run matching even if cached results exist

## [0.0.8] - 2026-01-03

### Added
- **Location-Aware Matching**: Geographic proximity filtering
  - GeoNames API integration for geocoding
  - Hard filter levels: city, state, country, continent
  - Configurable in `matching.location_matching` section
  - Requires GeoNames username with web services enabled

### Fixed
- Project setup script (`setup_project.py`)
  - Corrected project folder structure creation
  - Fixed `run.bat` launcher script paths

## [0.0.7] - 2026-01-01

### Added
- **Audio-First Mode**: Download audio only, transcribe, then fetch matched video segments
  - ~95% bandwidth savings
  - Configurable buffer and merge gap settings
  - DOWNLOAD_SEGMENTS stage for segment downloading
- **V4-V6 Diversity Tracks**: Secondary matches with strict source diversity
  - Embedding-diversity strategy on V7
  - All diversity tracks use `require_different_source: true`
- **Tier-Based Timeouts**: Duration-specific download timeouts
  - Short (<10m): 180s, Medium (10-30m): 300s, Long (>30m): 600s
  - Prevents hanging on slow downloads

## [0.0.6] - 2025-12-31

### Documentation
- Major expansion of `CLAUDE.md` project guide
  - CLI flags reference table
  - Architecture and file structure documentation
  - Development rules and testing checklist
  - Session history tracking

## [0.0.5] - 2025-12-30

### Added
- **Pause-Split Segments**: Intelligent transcript segmentation
  - Split at sentences with configurable pause detection
  - List marker detection (numbered/bulleted lists)
  - Location-based splitting for travel content
- **Enhanced List Detection**: Improved parsing of enumerated content
  - Detects numbered lists (1., 2., 3., etc.)
  - Detects bulleted lists (-, *, •)
  - Preserves list structure in timeline

### Fixed
- Multiple config loading and validation issues
- Nested dataclass initialization bugs
- yt-dlp live stream filter issues

## Migration Guide

### Upgrading from 0.2.x to 0.3.x

**Vision API Integration:**
- Set `GEMINI_API_KEY` environment variable for Vision API features
- Enable in config: `vision.enabled: true`
- Vision API automatically processes silent videos (no action required)
- Cost estimate: ~$0.001 per scene (configurable)

**Config Modularization:**
- All config imports remain backward compatible
- No code changes required for existing projects
- Optional: Update imports to use specific config modules for better clarity:
  ```python
  # Old (still works)
  from src.config import Config, LLMConfig

  # New (recommended)
  from src.config.core import Config
  from src.config.llm import LLMConfig
  ```

### Upgrading from 0.1.x to 0.2.x

**LLM Client Abstraction:**
- All LLM operations now use `src/llm_client/` package
- Old direct provider initialization code will continue to work but is deprecated
- Recommended: Update custom LLM integrations to use unified client:
  ```python
  # Old (deprecated)
  import google.generativeai as genai
  genai.configure(api_key=api_key)
  model = genai.GenerativeModel("gemini-2.0-flash")
  response = model.generate_content(prompt)

  # New (recommended)
  from src.llm_client import create_client, LLMRequest
  client = create_client("gemini", api_key=api_key, model="gemini-2.0-flash")
  response = client.generate(LLMRequest(prompt=prompt))
  ```

**Dataclass Imports:**
- Always import from canonical locations (see Rule 10 in CLAUDE.md)
- `AudioDownload`, `DownloadedVideo`, etc. must be imported from `src/state.py`
- Check your code for duplicate dataclass definitions and consolidate

**B-roll Track Fix:**
- Delete caches to regenerate B-roll detection: `rm -rf .cache/transcriptions .cache/scene_detection`
- V8 track should now populate with B-roll clips (face_score < 0.3)

### Upgrading from 0.0.x to 0.1.x

**Pipeline Architecture:**
- Old monolithic pipeline replaced with modular stages
- `PipelineState` now holds all shared data (no more scattered instance vars)
- Checkpoint format changed (incompatible with 0.0.x checkpoints)
- Use `--fresh` flag to start from scratch if upgrading

**Entity Caching:**
- First run will be slower as global entity cache populates
- Subsequent runs across projects will be much faster
- Cache location: `~/.matcher_entity_cache/`
- To clear cache: `rm -rf ~/.matcher_entity_cache/`

## Known Issues

### Test Suite (as of 0.3.0)
- 12 tests failing (fixture-dependent)
  - Audio-first mode tests require actual YouTube video fixtures
  - Face detection tests require MediaPipe installation
  - Location matching tests require GeoNames API key
- 21 test errors (mostly import/fixture issues)
- Overall pass rate: 92% (393 passing / 426 total)

### Coverage (as of 0.3.0)
- Overall coverage: 18% (misleading - see COVERAGE_SUMMARY.md)
- Legacy files (3,000+ lines) not yet deleted drag down metrics
- Effective coverage of active code: ~50%
- Refactored modules have excellent coverage (65-83%)

### Platform-Specific
- **Windows**: yt-dlp cookie extraction may fail if Firefox is running
  - Solution: Close Firefox before running pipeline
- **Linux**: MediaPipe may require additional system dependencies
  - Solution: `sudo apt-get install libgl1-mesa-glx`

## Development

### Running Tests
```bash
# All tests
python -m pytest tests/ -v

# Fast unit tests only
python -m pytest tests/test_keyword_extractor/ tests/test_llm_client/ tests/test_vision.py -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### CI/CD
- **Full Tests**: Runs on push to main/develop/feature/* branches
  - Matrix: Python 3.9-3.11 on Ubuntu/Windows
  - Generates coverage reports
  - Uploads to Codecov
- **Quick Check**: Runs on all branch pushes
  - Syntax validation
  - Fast unit tests only
  - Completes in < 10 minutes

### Contributing
See [REFACTORING.md](REFACTORING.md) for ongoing refactoring efforts and coding standards.

## Links

- [Project Documentation](./CLAUDE.md)
- [Test Coverage Report](./TEST_COVERAGE.md)
- [Coverage Summary](./COVERAGE_SUMMARY.md)
- [Refactoring Roadmap](./REFACTORING.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)
- [API Reference](./API_REFERENCE.md)
- [Examples](./EXAMPLES.md)
