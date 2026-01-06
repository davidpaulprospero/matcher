# Claude Code Project Guide

This file helps Claude Code sessions understand project conventions and avoid common mistakes.

## Quick Reference

### CLI Flags

| Flag | Description |
|------|-------------|
| `--voiceover`, `-v` | Path to voiceover file (SRT, MP3, WAV, MP4) |
| `--project`, `-p` | Project directory path |
| `--config`, `-c` | Path to config file |
| `--keywords`, `-k` | Number of keywords to extract |
| `--match-only` | Skip download/transcribe, use cached data |
| `--resume` | Resume from checkpoint |
| `--fresh` | Force fresh start, ignore checkpoint |
| `--force-rematch` | Force rematch all videos |
| `--use-keywords [NAME]` | Use saved keywords (latest or named preset) |
| `--save-keywords [NAME]` | Save keywords as preset |
| `--list-keywords` | List saved keyword presets |
| `--validate-config` | Validate config and exit |
| `--refresh-entities` | Re-download entity images (ignore local cache) |
| `--non-interactive` | Skip prompts, use defaults |
| `--save-matching-fixtures` | Save matching inputs for testing |

### Common Commands

```bash
# Basic run
python main.py --voiceover script.srt --project "E:\Projects\MyDoc"

# Resume interrupted run
python main.py --resume

# Re-run matching only (after config tweaks)
python main.py --match-only

# Use saved keywords
python main.py --use-keywords mypreset
```

## Architecture

### File Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point, pipeline orchestration |
| `config.yaml` | User-editable settings |
| `src/config.py` | Config dataclasses with defaults |
| `src/downloader.py` | Video/audio download logic |
| `src/matching.py` | Video-to-voiceover matching |
| `src/transcription.py` | Whisper transcription |
| `src/checkpoint.py` | Resume/checkpoint management |
| `src/keyword_extractor.py` | LLM-based keyword extraction |
| `src/entity_images.py` | Entity image search (Google, Bing) |
| `src/entity_cache.py` | Cross-project entity cache |
| `src/location_service.py` | GeoNames geocoding |
| `src/otio_builder.py` | Timeline generation |
| `src/pipeline.py` | PipelineOrchestrator |
| `src/state.py` | PipelineState dataclass |
| `src/stages/` | Modular stage classes |
| `setup_project.py` | Creates project folders with run scripts |

### Pipeline Stages

| Stage | Class | Purpose |
|-------|-------|---------|
| ANALYZE | AnalyzeStage | Keywords, topics, entities, location chapters |
| ENTITY_IMAGES | EntityImagesStage | Download entity images (Google, Bing, Pexels) |
| ENTITY_VIDEOS | EntityVideosStage | Download stock videos for entities |
| DOWNLOAD | DownloadStage | YouTube video/audio download |
| STOCK | StockVideoStage | Download generic stock footage (B-roll) |
| REMIX | RemixStage | Filter videos by keyword relevance |
| TRANSCRIBE | TranscribeStage | Whisper + embeddings |
| SCENE_DETECTION | SceneDetectionStage | Scene boundaries + face detection for B-roll |
| MATCH | MatchStage | Embedding + LLM matching |
| OUTPUT | OutputStage | OTIO, EDL, XML generation |

### OTIO Track Layout

| Track | Purpose | Default State |
|-------|---------|---------------|
| V1 | Primary video | Enabled |
| V2-V3 | Alternatives 1-2 | Disabled |
| V4-V6 | Secondary (diversity-scored, strict different source) | Disabled |
| V7 | Embedding-Diversity strategy | Disabled |
| V8 | B-roll Only (silent footage) | Disabled |
| V9 | Entity Images (Google stills) | Disabled |
| V10 | Stock Videos (Pexels/Pixabay) | Disabled |
| A1-A8 | Corresponding audio | Matches video |

### V8-V10 Track Requirements

**V8 - B-roll Only:**
- Requires: SCENE_DETECTION stage (sets `is_broll=True` on segments with face_score < 0.3)
- Config: `pipeline.skip_scene_detection: false` (default), `scene_detection.detect_faces_per_scene: true` (default)
- **Detection**: Samples 3 frames/scene, MediaPipe/OpenCV face detection, face_score = (frames_with_faces / 3)
- **Threshold**: `broll_face_threshold: 0.3` - scenes with < 30% face presence = B-roll
- **Data flow**: SceneDetectionStage sets is_broll on transcripts + text_metadata → MatchStage restores flag → broll_only strategy filters by is_broll
- **Troubleshooting**: If V8 empty, check logs for "B-roll: X/Y scenes" and "Updated X text_metadata entries". If 0 B-roll, lower threshold

**V9 - Entity Images:**
- Requires: Entity extraction (ANALYZE stage) + EntityImagesStage
- Config: `pipeline.skip_image_search: false`
- Sources: Google Images, Bing Images, Pexels, Pixabay
- **Troubleshooting**: If using `--match-only`, ensure `EntityImagesStage` is in the pipeline (fixed in src/pipeline.py:275)

**V10 - Stock Videos:**
- Requires: Entity extraction + EntityVideosStage + API keys
- Config: `pipeline.skip_image_search: false`, `image_search.use_stock_apis: true`
- Sources: Pexels, Pixabay stock video APIs
- **Troubleshooting**: Same as V9. EntityVideosStage runs independently from YouTube downloads (fixed: removed incorrect `skip_download` check)

### Project Directory Structure

```
ProjectName__2026-01-03/
├── run.bat              # Launcher script
├── project_config.yaml  # Project-specific overrides
├── voiceover/           # User puts audio here
├── output/              # Generated OTIO, EDL, XML
├── checkpoint.json      # Resume state
├── saved_keywords.json  # Keyword presets
└── .cache/              # Transcriptions, embeddings
```

## Caching

### Cache Locations

| Cache | Location | Purpose |
|-------|----------|---------|
| Checkpoint | `checkpoint.json` | Resume state |
| Checkpoint backup | `checkpoint.backup.json` | Corruption recovery |
| Transcriptions | `.cache/transcriptions/` | Whisper output |
| Embeddings | `.cache/embeddings/` | Vector embeddings |
| LLM responses | `.cache/llm_responses/` | Cached LLM calls |
| Locations | `.cache/locations/` | GeoNames geocoding |
| Global cache | `~/.matcher_global_cache/` | Cross-project videos |
| Entity cache | `~/.matcher_entity_cache/` | Cross-project images |
| Saved keywords | `saved_keywords.json` | Keyword presets |

### Clear/Refresh

```bash
# Force fresh start (clears checkpoint)
python main.py --fresh

# Re-download entity images
python main.py --refresh-entities

# Delete caches manually (bash/Git Bash)
rm -rf .cache/

# Delete specific caches to force re-transcription (for V8 B-roll fix)
rm -rf .cache/transcriptions
rm -rf .cache/scene_detection
```

**Note for run.bat users**: When using `run.bat`, the working directory is the project folder (e.g., `E:\Projects\MyDoc__2026-01-03\`). Use relative paths:
```bash
# From project directory via run.bat
rm -rf .cache/transcriptions
rm -rf .cache/scene_detection
```

## Configuration

### Config Access Pattern

```python
# Safe config access with fallback:
value = getattr(self.download_config, 'field_name', default_value)

# For nested configs that might be dicts:
if isinstance(config_section, dict):
    value = config_section.get('field_name', default)
else:
    value = getattr(config_section, 'field_name', default)
```

### Adding New Config Options

1. Add field to dataclass in `src/config.py`
2. Add `__post_init__` if nested dataclass (see Rule 2 below)
3. Add to `config.yaml` with comment
4. Access in code with `getattr()` fallback
5. Verify: `python -m py_compile src/config.py`

### Project Overrides

Create `project_config.yaml` in project folder:
```yaml
keyword:
  max_keywords: 5
pipeline:
  skip_image_search: true
```

## Development Rules

### Rule 1: Config Synchronization
Always update BOTH `src/config.py` AND `config.yaml` when adding options.

### Rule 2: Future Annotations + `__post_init__`
Files use `from __future__ import annotations`. Nested dataclass fields may load as `dict`. Fix with:
```python
def __post_init__(self):
    if isinstance(self.nested_field, dict):
        self.nested_field = NestedConfig(**self.nested_field)
```

### Rule 3: DownloadedVideo Field Names
Use EXACT field names: `file=`, `duration_tier=`. NOT `path=`, `tier=`, `video_id=`.

### Rule 4: Regex Lookbehind
Python requires fixed-width lookbehinds. Use capture groups instead:
```python
# BAD: r'(?<=[A-Z][a-z]+,\s[A-Z][a-z]+,)\s+'
# GOOD: re.sub(r'([A-Z][a-z]+,\s+[A-Z][a-z]+,)\s+', r'\1|||SPLIT|||', text)
```

### Rule 5: Live Stream Filtering
Add `!is_live` to yt-dlp match-filter to prevent hanging on live streams.

### Rule 6: Dict vs Object Config
Handle variety config as both dict and object:
```python
if isinstance(vc, dict):
    value = vc.get('require_different_source', True)
else:
    value = getattr(vc, 'require_different_source', True)
```

### Rule 7: Embeddings Truthiness Checks
Never use `state.embeddings` directly in boolean contexts. Numpy arrays raise "truth value of array is ambiguous" errors.

```python
# BAD: Direct truthiness check
if state.embeddings:           # ValueError!
if not state.embeddings:       # ValueError!

# GOOD: Use helper function
from src.utils import is_embeddings_empty
if is_embeddings_empty(state.embeddings):
if not is_embeddings_empty(state.embeddings):
```

### Rule 8: B-roll Face Detection & Propagation
B-roll detection logic and data flow:
- **face_score**: Ratio of frames with faces (0.0-1.0), from 3 sampled frames per scene
- **is_broll**: Set `True` when `face_score < broll_face_threshold` (default 0.3)
- **Data flow**: SceneDetectionStage MUST update both `state.transcripts` AND `state.text_metadata` with is_broll/face_score/scene_index
- **MatchStage**: Restores is_broll from text_metadata dict to SRTSegment objects before matching
- **"no faces" in logs**: Shorthand for "below threshold", not literally zero
- **Backends**: MediaPipe (primary), OpenCV Haar (fallback)
- **Threshold tuning**: Lower (0.1) = stricter, Higher (0.5) = more lenient

### Testing Checklist

- [ ] `python -m py_compile main.py`
- [ ] `python -m py_compile src/config.py`
- [ ] `python -m py_compile src/downloader.py`
- [ ] `python -m py_compile src/matching.py`
- [ ] Config in both `config.py` AND `config.yaml`
- [ ] Nested configs have `__post_init__`

## Key Features

### Audio-First Mode
Downloads audio only, transcribes, matches, then downloads only matched video segments (~95% bandwidth savings).
```yaml
download:
  audio_first:
    enabled: true
    buffer_seconds: 30.0
    merge_gap_seconds: 15.0
```

### Location-Aware Matching
Filters video candidates by geographic proximity for travel content.
```yaml
matching:
  location_matching:
    enabled: true
    geonames_username: "your_username"  # Must enable web services at geonames.org
    hard_filter_level: "city"  # city, state, country, continent
```
**Note:** GeoNames web services must be enabled at https://www.geonames.org/login (401 errors = not enabled).

### Pause-Split Segments
```yaml
transcription:
  pause_split:
    enabled: true
    split_at_sentences: true
    split_at_list_markers: true
    split_at_locations: true
```

## Git Conventions

- **Prefixes:** `feat:`, `fix:`, `docs:`, `refactor:`
- **Branches:** Claude Code branches start with `claude/`
- Push to feature branch, not main

## Session History

| Date | Changes |
|------|---------|
| 2026-01-06 | Fixed V8 B-roll track (0→238 clips): SceneDetectionStage now propagates is_broll to text_metadata, MatchStage restores from metadata |
| 2026-01-05 | Pipeline architecture refactor, entity image caching |
| 2026-01-04 | Match-only mode |
| 2026-01-03 | Location-aware matching, project setup fix |
| 2026-01-01 | Audio-first mode, V4-V6 diversity, tier timeouts |
| 2025-12-31 | CLAUDE.md expansion |
| 2025-12-30 | Pause-split, list detection, config fixes |
