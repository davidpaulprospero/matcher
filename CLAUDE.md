# Claude Code Project Guide

This file helps Claude Code sessions understand project conventions and avoid common mistakes.

## Critical Rules

### 1. Config Synchronization (MOST IMPORTANT!)
When adding/modifying configuration options:
- **Always update BOTH files:**
  - `src/config.py` - Python dataclass with type hints and defaults
  - `config.yaml` - User-facing YAML config with comments
- They must stay in sync or users won't see new options
- Run `python -m py_compile src/config.py` to verify syntax

### 2. Future Annotations & Type Hints
Both `main.py` and `src/config.py` use:
```python
from __future__ import annotations
```
This makes type hints strings, which can cause issues:
- **Problem:** Nested dataclass fields may load as `dict` instead of the dataclass
- **Solution:** Always add `__post_init__` to convert dicts:
```python
def __post_init__(self):
    if isinstance(self.nested_field, dict):
        self.nested_field = NestedConfig(**self.nested_field)
```

### 3. Config Access Pattern
```python
# Safe config access with fallback:
value = getattr(self.download_config, 'field_name', default_value)

# For nested configs that might be dicts:
if isinstance(config_section, dict):
    value = config_section.get('field_name', default)
else:
    value = getattr(config_section, 'field_name', default)

# For nested configs (e.g., dicts):
tier_timeouts = getattr(self.download_config, 'download_timeouts', {})
timeout = tier_timeouts.get('medium', 120)
```

### 4. Dataclass Field Names
`DownloadedVideo` in `src/downloader.py` has these EXACT field names:
```python
@dataclass
class DownloadedVideo:
    file: str           # NOT 'path'
    url: str
    title: str
    channel: str
    upload_date: str
    duration: float
    duration_tier: str  # NOT 'tier'
    keyword: str
    download_date: str
    license: str = "Unknown"
```
**Never use:** `path=`, `tier=`, `video_id=` - these will cause errors.

### 5. Regex Lookbehind Limitations
Python regex requires **fixed-width lookbehinds**. This will ERROR:
```python
# BAD - variable width lookbehind
r'(?<=[A-Z][a-z]+,\s[A-Z][a-z]+,)\s+'
```
Use capture groups instead:
```python
# GOOD - capture and replace
re.sub(r'([A-Z][a-z]+,\s+[A-Z][a-z]+,)\s+', r'\1|||SPLIT|||', text)
```

## File Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point, pipeline orchestration |
| `config.yaml` | User-editable settings (YAML format) |
| `src/config.py` | Dataclasses defining config schema with defaults |
| `src/downloader.py` | Video download logic, uses `self.download_config` |
| `src/matching.py` | Video-to-voiceover matching engine |
| `src/transcription.py` | Audio transcription (faster-whisper) |
| `src/checkpoint.py` | Resume/checkpoint management |
| `src/keyword_extractor.py` | LLM-based keyword extraction |
| `setup_project.py` | Creates new project folders with run scripts |

## Project Directory Architecture

The system uses a **two-directory model**:

### Install Directory (INSTALL_DIR)
Where the application code lives. Contains `main.py`, `src/`, `config.yaml`, etc.
```
E:\voiceover-matcher\          # Central installation
├── main.py
├── config.yaml
├── setup_project.py
└── src/
```

### Project Directory (PROJECT_DIR)
Individual project folders created by `setup_project.py`. Each project has its own voiceover, outputs, and caches.
```
E:\Edit Job\Client\Series\MyProject__2026-01-03\
├── run.bat                    # Generated launcher script
├── project_config.yaml        # Project-specific overrides
├── voiceover/                 # User puts audio here
├── output/                    # Generated OTIO, videos
├── logs/
├── checkpoint.json            # Resume state
├── saved_keywords.json        # Cached keywords
└── .cache/                    # Transcriptions, embeddings
```

### How setup_project.py Works
Location: `setup_project.py`

**Key variables:**
```python
INSTALL_DIR = Path(__file__).parent.resolve()  # Where setup_project.py lives
project_dir = Path(project_path).resolve()      # User-chosen project location
```

**Generated run.bat structure:**
```batch
set INSTALL_DIR=E:\voiceover-matcher
set PROJECT_DIR=%~dp0
cd /d "%INSTALL_DIR%"
python main.py --project "%PROJECT_DIR%" ...
```

The `run.bat`:
1. Sets `INSTALL_DIR` to the central installation (hardcoded at creation time)
2. Sets `PROJECT_DIR` to the folder containing `run.bat` itself (`%~dp0`)
3. Changes to `INSTALL_DIR` to run `main.py`
4. Passes `--project` flag so `main.py` knows where project files are

**Regenerating run scripts:**
```bash
python setup_project.py --regenerate "E:\Path\To\Project"
```
Use this if `INSTALL_DIR` changed or `run.bat` was corrupted.

## Common Bugs & Fixes

### 1. `'dict' object has no attribute 'X'`
**Cause:** Nested config loaded as dict due to `from __future__ import annotations`
**Fix:** Add `__post_init__` to parent dataclass to convert dict to proper type.
Example in `OutputConfig`:
```python
def __post_init__(self):
    if isinstance(self.variety, dict):
        self.variety = VarietyConfig(**self.variety)
```

### 2. `DownloadedVideo.__init__() got unexpected keyword argument`
**Cause:** Using wrong field names when creating DownloadedVideo
**Fix:** Use exact field names: `file=`, `duration_tier=`, NOT `path=`, `tier=`

### 3. Live streams hanging forever
**Cause:** yt-dlp tries to download infinite live streams
**Fix:** Add `!is_live` to match-filter:
```python
'--match-filter', f"duration>{min_dur} & duration<{max_dur} & !is_live"
```
Also add to title_blacklist: `live stream`, `livestream`, `webcam`, `24/7`

### 4. Pylance "Module cannot be used as a type"
**Cause:** Type checker can't resolve dataclass types
**Fix:** Add `from __future__ import annotations` at top of file (after docstring)

### 5. Regex lookbehind error
**Cause:** Variable-width patterns in lookbehind
**Fix:** Use capture groups with replacement instead

## OTIO Timeline Structure

**Location:** `src/otio_builder.py`
**IMPORTANT:** When editing track assignments or adding new tracks, always update this section!

### Track Layout
| Track | Purpose | Default State |
|-------|---------|---------------|
| V1 | Primary video | Enabled |
| V2 | Alternative 1 | Disabled |
| V3 | Alternative 2 | Disabled |
| V4 | Secondary Primary (diversity-scored, strict different source) | Disabled |
| V5 | Secondary Alt 1 (diversity-scored, strict different source) | Disabled |
| V6 | Secondary Alt 2 (diversity-scored, strict different source) | Disabled |
| V7 | Embedding-Diversity strategy | Disabled |
| V8 | (Reserved) | - |
| V9 | Entity Images (Google stills) | Disabled |
| V10 | Stock Videos (Pexels/Pixabay) | Disabled |
| A1-A7 | Corresponding audio tracks | Matches video |

### Key Functions
- `build_otio_timeline()` - Main entry point, creates timeline with all tracks
- `_create_clip()` - Creates individual clips with speed adjustment
- `_validate_entity_images()` - Filters invalid image paths before OTIO
- `_to_windows_path()` - Converts paths for DaVinci Resolve compatibility
- `get_secondary_matches_diversity()` - V4-V6 matching with strict source enforcement

### Config Options
```yaml
output:
  generate_otio: true
  split_otio: true           # Split into multiple files
  otio_clips_per_file: 10    # Clips per OTIO file
  num_alternatives: 2        # V2-V3 count
  include_strategy_tracks: true
  strategy_tracks:
    - "embedding_diversity"  # V7
```

### Path Handling
DaVinci Resolve requires Windows-style backslash paths:
```python
# WRONG: forward slashes fail in DaVinci
path = "E:/videos/clip.mp4"

# CORRECT: use _to_windows_path()
path = _to_windows_path("E:/videos/clip.mp4")  # Returns "E:\\videos\\clip.mp4"
```

## Key Features

### Pause-Split Segments
Location: `main.py` → `_pause_split_segments()`
Config:
```yaml
transcription:
  pause_split:
    enabled: true
    split_at_sentences: true      # Every sentence = own segment
    split_at_list_markers: true   # "Number 10, City" isolation
    split_at_locations: true      # "City, State," splits
    min_phrase_words: 2
    min_segment_duration: 0.5
```

### LLM List Detection
Location: `main.py` → `_detect_list_items()`
Uses Gemini to detect "Top 10" style countdowns and extract list items.
Config:
```yaml
keyword:
  list_detection:
    enabled: true
    download_first: true
    skip_if_entity_covered: true
    keyword_suffix: "footage"
```

### Crash Resilience
Location: `src/downloader.py`
- File-based skip logic (checks existing videos by ID)
- Auto-cleanup of `.part` and `.ytdl` files
- `--no-continue` flag to prevent resume hangs

### Audio-First Mode (Download Optimization)
Location: `src/downloader.py`, `main.py`
Dramatically reduces download time by only downloading matched video segments.

**How it works:**
1. **Audio download:** Downloads audio-only (MP3, ~5% of video size)
2. **Transcribe & match:** Runs matching on audio transcripts
3. **Segment download:** Downloads only matched video portions with buffer

**Config:**
```yaml
download:
  audio_first:
    enabled: true           # Enable audio-first mode
    buffer_seconds: 30.0    # Extra footage before/after each match
    merge_gap_seconds: 15.0 # Merge segments closer than this
    audio_quality: 5        # 0 (best) to 9 (worst), 5 = ~128kbps
    fallback_full_video: true      # Download full video if segment fails
    delete_audio_after_video: false # Keep audio files
```

**Data structures** (`src/downloader.py`):
- `AudioDownload` - Downloaded audio file metadata
- `MatchedSegment` - Matched segment from a specific video
- `MergedSegment` - Merged segments with buffer applied
- `DownloadedSegment` - Downloaded video segment file

**Segment naming:** `{video_id}_{start_seconds:04d}.mp4`
- Example: `abc12345678_0045.mp4` (segment starting at 45s)
- OTIO builder uses `get_segment_file_offset()` to calculate clip offsets

**yt-dlp commands:**
```bash
# Audio download
yt-dlp -f "bestaudio" --extract-audio --audio-format mp3 --audio-quality 5

# Video segment download
yt-dlp --download-sections "*45-90" -o "%(id)s_%(autonumber)s.%(ext)s"
```

### Variety Config
Location: `src/matching.py` → `StrategyMatcher`
Must handle as both dict and object:
```python
if isinstance(vc, dict):
    value = vc.get('require_different_source', True)
else:
    value = getattr(vc, 'require_different_source', True)
```

### Location-Aware Matching
Location: `src/location_service.py`, `src/matching.py`, `main.py`
For travel/location-focused content, filters video candidates by geographic proximity.

**Filter levels** (strictest to loosest):
- `city` - Same city name OR within 25km
- `state` - Same state/province
- `country` - Same country
- `continent` - Same continent

**Key gotchas:**
1. **GeoNames web services:** Must enable at https://www.geonames.org/login (not enabled by default)
2. **Alternate paths:** Location detection must run in saved keywords AND checkpoint resume paths (fixed in main.py)
3. **401 errors:** Mean web services aren't enabled on GeoNames account

**Config:**
```yaml
matching:
  location_matching:
    enabled: true
    geonames_username: "your_username"
    hard_filter_level: "city"  # city, state, country, continent
```

## Git Conventions

- **Commit prefixes:** `feat:`, `fix:`, `docs:`, `refactor:`
- **Branch naming:** Branches from Claude Code start with `claude/`
- Always push to the assigned feature branch, not main
- **Push branch to remote:** `git push -u origin branch-name`

## Adding New Config Options

1. Add field to appropriate dataclass in `src/config.py`:
   ```python
   new_option: int = 100  # Description
   ```

2. Add `__post_init__` if it's a nested dataclass:
   ```python
   def __post_init__(self):
       if self.nested is None:
           self.nested = NestedConfig()
   ```

3. Add to `config.yaml` with comment:
   ```yaml
   new_option: 100  # Description
   ```

4. Access in code with fallback:
   ```python
   value = getattr(self.config_section, 'new_option', 100)
   ```

5. Verify syntax:
   ```bash
   python -m py_compile src/config.py
   python -m py_compile main.py
   ```

## Project-Specific Overrides

Users can create `project_config.yaml` in project folders to override settings:
```yaml
keyword:
  max_keywords: 5  # Only 5 keywords for this project

pipeline:
  skip_download: false
  skip_image_search: true
```

## Testing Checklist

Before committing changes:
- [ ] `python -m py_compile main.py`
- [ ] `python -m py_compile src/config.py`
- [ ] `python -m py_compile src/downloader.py`
- [ ] `python -m py_compile src/matching.py`
- [ ] Config options in both `config.py` AND `config.yaml`
- [ ] Nested configs have `__post_init__` for dict conversion
- [ ] DownloadedVideo uses correct field names

## Session History

### 2025-12-30: Comprehensive fixes session
- **Pause-split:** Made aggressive (sentence-level, list isolation)
- **List detection:** Replaced regex with LLM-based detection
- **Live streams:** Added `!is_live` filter and title blacklist
- **Variety config:** Fixed dict vs object handling in matching.py
- **Pylance errors:** Added `from __future__ import annotations`
- **DownloadedVideo:** Fixed field names (file, duration_tier)
- **Config loading:** Enhanced `_build_dataclass` for string annotations

### 2025-12-31: CLAUDE.md expansion
- **Expanded documentation:** Added Common Bugs & Fixes, Key Features, Testing Checklist
- **Future annotations warning:** Documented `__post_init__` requirement for nested dataclasses
- **Regex gotcha:** Added lookbehind limitation and workaround
- **Project overrides:** Documented `project_config.yaml` feature

### 2026-01-01: Tier-specific download timeouts
- **Problem:** 120s timeout too short for medium/long videos
- **Solution:** Added `download_timeouts` dict with per-tier values
- **Files changed:** `src/config.py`, `src/downloader.py`, `config.yaml`
- **Timeouts:** short=120s, medium=300s, long=600s, longer=900s

### 2026-01-01: V4-V6 diversity scoring with strict source enforcement
- **Problem:** V4-V6 could share sources, fallback logic reduced variety
- **Solution:** New `get_secondary_matches_diversity()` method in StrategyMatcher
- **Algorithm:** Same as V7 (40% relevance + 60% diversity), min 0.3 relevance
- **Strict enforcement:** V4, V5, V6 each MUST use different source videos
- **No fallbacks:** Empty track if no different source available
- **Files changed:** `src/matching.py`

### 2026-01-01: Audio-First Download Mode
- **Problem:** Downloading full videos wastes bandwidth (only ~5% of footage used)
- **Solution:** Audio-first pipeline - download audio, transcribe, match, then download only needed video segments
- **Workflow:**
  1. Stage 2A: Download audio-only (MP3, ~5% of video size)
  2. Stage 3: Transcribe audio files
  3. Stage 4: Match voiceover to video transcripts
  4. Stage 4.5: Download only matched video segments with buffer
- **Key features:**
  - 30s buffer before/after each match
  - Segments within 15s merged to reduce download requests
  - Segment files named with start time: `{video_id}_{start:04d}.mp4`
  - OTIO builder auto-adjusts clip offsets for segment files
- **Files changed:**
  - `src/config.py` - Added `AudioFirstConfig` dataclass
  - `src/downloader.py` - Added `AudioDownload`, `MatchedSegment`, `MergedSegment`, `DownloadedSegment` dataclasses and download methods
  - `main.py` - Added `stage_download_audio()`, `stage_download_video_segments()`, pipeline integration
  - `src/otio_builder.py` - Added `get_segment_file_offset()`, segment offset adjustments
  - `config.yaml` - Added `audio_first` config section

### 2026-01-03: Location-Aware Chapter Matching
- **Problem:** Travel/location videos need geographic context (Paris, France vs Paris, Texas)
- **Solution:** GeoNames API integration with location filtering at city/state/country/continent levels
- **Key files:**
  - `src/location_service.py` (NEW) - GeoNames API, geocoding, caching, disambiguation
  - `src/topic_extraction.py` - `LocationChapter`, `detect_location_chapters()`
  - `src/matching.py` - `_apply_location_filter()`, location-aware scoring
  - `main.py` - `_detect_location_chapters()`, pipeline integration
- **Config:**
  ```yaml
  matching:
    location_matching:
      enabled: true
      geonames_username: "your_username"  # Required - enable web services at geonames.org
      hard_filter_level: "city"  # city, state, country, or continent
  ```
- **IMPORTANT:** GeoNames requires enabling web services on your account (not enabled by default)
- **Bug fixed:** Location detection was skipped when using saved keywords or checkpoint resume - now runs in all paths

### 2026-01-03: Project setup and run.bat fix
- **Problem:** `run.bat` used `cmd /k` which kept window open indefinitely after each run
- **Solution:** Changed to `exit /b` so batch file exits cleanly after completion
- **Files changed:** `setup_project.py` → `create_run_bat()` function
- **Documentation:** Added "Project Directory Architecture" section explaining INSTALL_DIR vs PROJECT_DIR

### 2026-01-04: Match Only menu option
- **Problem:** Users want to re-run matching with different config without re-downloading/transcribing
- **Solution:** Added [M] Match only option to initial menu in run.bat/run.sh
- **Use cases:**
  - Re-run matching with different `min_conf`, `candidates`, `face_preference` settings
  - Test different matching strategies without waiting for transcription/embedding
  - Regenerate timelines after tweaking `config.yaml`
- **Cache validation:** Menu checks for transcription + embedding caches before enabling [M]
- **Menu shows:**
  - Transcriptions cached [X videos]
  - Embeddings cached [Y entries]
- **Command line:** `run --match-only` or `./run.sh --match-only`
- **Files changed:** `setup_project.py` → `create_run_bat()`, `create_run_sh()` functions (v3.2)