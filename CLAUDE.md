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
| DOWNLOAD | DownloadStage | YouTube video/audio download |
| TRANSCRIBE | TranscribeStage | Whisper + embeddings |
| MATCH | MatchStage | Embedding + LLM matching |
| OUTPUT | OutputStage | OTIO, EDL, XML generation |

### OTIO Track Layout

| Track | Purpose | Default State |
|-------|---------|---------------|
| V1 | Primary video | Enabled |
| V2-V3 | Alternatives 1-2 | Disabled |
| V4-V6 | Secondary (diversity-scored, strict different source) | Disabled |
| V7 | Embedding-Diversity strategy | Disabled |
| V9 | Entity Images (Google stills) | Disabled |
| V10 | Stock Videos (Pexels/Pixabay) | Disabled |
| A1-A7 | Corresponding audio | Matches video |

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

# Delete caches manually
rm -rf .cache/
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
| 2026-01-05 | Pipeline architecture refactor, entity image caching |
| 2026-01-04 | Match-only mode |
| 2026-01-03 | Location-aware matching, project setup fix |
| 2026-01-01 | Audio-first mode, V4-V6 diversity, tier timeouts |
| 2025-12-31 | CLAUDE.md expansion |
| 2025-12-30 | Pause-split, list detection, config fixes |
