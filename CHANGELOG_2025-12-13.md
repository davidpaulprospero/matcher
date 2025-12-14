# Changelog - December 13, 2025

## v2.2.2 - Video Voiceover & Smart Transcoding

### 🎬 Video File Voiceover Support
- Now accepts **video files** as voiceover source (MP4, MOV, AVI, MKV, WebM, etc.)
- Automatically extracts audio and transcribes to SRT
- GPU-accelerated audio extraction with CUDA
- Example: `python main.py -v voiceover/my_recording.mp4`

### ⚡ Smart Transcoding (Skip When Possible)
- **Codec Detection**: Checks downloaded video codec before transcoding
- **Skip H.264/H.265**: Videos already in DaVinci-compatible formats are NOT transcoded
- **Only Transcode VP9/AV1**: WebM and AV1 files get transcoded to H.264
- **Prefer H.264 Downloads**: yt-dlp now prefers H.264 over VP9 when in DaVinci mode
- **Result**: ~70% of YouTube downloads skip transcoding entirely

**DaVinci-Compatible (no transcode):**
- H.264/AVC in MP4/MOV
- H.265/HEVC in MP4/MOV
- ProRes in MOV
- DNxHD/DNxHR in MOV/MXF

**Needs Transcode:**
- VP9 (WebM) - common YouTube format
- AV1 - newer YouTube format
- VP8, Theora

### 📋 Download Output Example
```
Downloading: Japan earthquake footage
  ✓ Downloaded: japan_quake_2025.mp4
    ↳ ✓ No transcode needed: Already compatible (h264/mp4)
  ✓ Downloaded: tsunami_drone.webm  
    ↳ Transcoding (Codec vp9 not DaVinci-compatible)...
    ↳ ✓ Transcode complete
```

---

## v2.2.1 - Project Mode & GPU Acceleration

### 🆕 Interactive Project Setup
- **Default Base Path**: `E:\Edit Job\` (configurable)
- **2-Level Navigation**: Client → Series → Project Name
- **Folder Browser**: List existing folders, create new ones, or select
- **Auto-Date**: Project names automatically get `__YYYY-MM-DD` suffix
- **Remembers Last Path**: Next run defaults to your last used location
- **Example Flow**:
  ```
  Base: E:\Edit Job\
  
  STEP 1: Select Client
    1. Gold Media
    2. Silver Media
    3. [Create new client]
  Select: 1
  
  STEP 2: Select Series
    1. Earth Unleashed-
    2. Ocean Disasters
    3. [Create new series]
  Select: 1
  
  STEP 3: Name Your Project
  Project name: JAPAN Earthquake
  
  Creating: E:\Edit Job\Gold Media\Earth Unleashed-\JAPAN Earthquake__2025-12-13\
  ```

### 🆕 Project-Based Architecture
- **Central Installation**: Install once to `C:\Tools\voiceover-matcher\`, use from any project
- **Project Folders**: Each documentary gets its own folder with:
  - `run.bat` - Double-click to launch
  - `voiceover/` - Input files
  - `downloaded_videos/` - Downloaded footage
  - `otio_output/` - Generated timelines
  - `.cache/` - Transcription/embedding cache
  - `logs/` - Run logs
  - `project_config.yaml` - Per-project setting overrides
- **Global .env**: API keys shared across all projects (no more copying!)
- **Project Config Overrides**: Override global settings per-project without editing main config

### 🚀 GPU CUDA Acceleration (All FFmpeg Operations)
- **Video Decoding**: `-hwaccel cuda` for NVIDIA GPUs
- **Video Encoding**: `h264_nvenc` / `hevc_nvenc` for hardware encoding
- **Audio Extraction**: GPU-accelerated decoding with CPU fallback
- **Frame Extraction**: GPU-accelerated for deduplication
- **Transcription**: faster-whisper with CUDA (10-50x speedup)

### 🎤 MP3/Audio Voiceover Support
- Now accepts audio files directly: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.wma`, `.aac`, `.opus`
- Auto-transcribes to SRT using faster-whisper (GPU)
- Skips transcription if `.srt` already exists alongside audio
- Example: `python main.py -v voiceover/narration.mp3`

### 🔍 Entity-Aware Keyword Extraction
- **Two-pass extraction**:
  1. Named entities (people, places, organizations, dates, events)
  2. General visual keywords
- **Priority**: Entity keywords always included first
- **Smart search terms**: Combines entities with topic context
  - "President Marcos" → "President Marcos typhoon response footage"
  - "Tacloban City" → "Tacloban City storm surge 2013"
  - "Red Cross" → "Red Cross rescue operation Philippines"

### 🎬 Different Sources Per Track
- V1-V8 now **enforce different video sources** for each segment
- No more same-source clips appearing across multiple tracks
- `force_different_source=True` in all strategy methods
- Better variety for editor choices

### ⚡ `--match-only` Improvements
- Now **skips keyword extraction** (no Claude API call)
- Skips the "How many keywords?" prompt
- Goes straight to transcription → matching
- Much faster for re-runs

### 📁 New Files
- `setup_project.py` - Create new project folders interactively
- `install.bat` - Windows installation helper
- `README.md` - Comprehensive documentation

### 🛠 Command Line Updates
```
New arguments:
  --project, -p PATH    Project directory (all outputs go here)
  
Example:
  python main.py --project "E:\Edit Job\...\JAPAN Earthquake" -v voiceover/script.srt
```

### 📋 Files Modified
- `main.py` - Project mode, environment loading, path resolution
- `src/transcription.py` - GPU audio extraction, voiceover transcription function
- `src/downloader.py` - Full GPU decode→encode pipeline
- `src/audio_analysis.py` - GPU-accelerated audio extraction
- `src/deduplication.py` - GPU-accelerated frame extraction
- `src/matching.py` - force_different_source for all strategies
- `src/keyword_extractor.py` - Entity extraction pass, improved prompts
- `src/__init__.py` - Version bump to 2.2.0

---

## Previous Session Changes (Earlier Dec 13)

### Fixed Empty Strategy Tracks (V4-V8)
- **Problem**: V4 (Visual-First), V5 (Different-Source), V6 (Keyword-Only), V8 (Source-Rotation) had 0-2 clips
- **Root Cause**: `require_different_source: true` was too strict globally
- **Fix**: Set `require_different_source: false` in config, let each strategy decide

### Keyword Matching Fallback
- When `.keywords` array empty, extracts 4+ char words from `.text`
- Stop words filtered: this, that, with, from, have, been, etc.
- Enables V6 (Keyword-Only) track to work without NER

### Visual Matching Fallback  
- When no scene descriptions, checks filename for visual terms
- Visual terms: earthquake, tsunami, flood, storm, fire, volcano, disaster, etc.
- Enables V4 (Visual-First) track to work without scene detection

### Numpy JSON Serialization Fix
- `np.bool_` and `np.float64` from librosa caused JSON errors
- Added explicit `bool()`, `float()`, `int()` conversions
- Added `_json_default()` fallback encoder

### Configurable Same-Source Time Distance
- `variety.min_time_distance: 60.0` (was 10s)
- Clips from same video must be 60+ seconds apart
- Prevents clustering of clips from same source

### Auto-Logging Every Run
- Enabled by default when `logging.enabled: true`
- Logs to `./logs/voiceover_match_YYYY-MM-DD_HH-MM-SS.log`
- JSON companion file for machine parsing
- Records all match decisions with reasoning
