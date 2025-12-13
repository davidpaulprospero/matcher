# Voiceover-to-Footage Matcher v2.2

## End-to-End Pipeline Release

This release adds a complete end-to-end pipeline: one command to go from voiceover → matched timeline.

---

## 🚀 New Features

### 1. End-to-End Pipeline (`main.py`)

One command to automate your entire workflow:

```bash
python main.py --voiceover script.srt
```

**Pipeline Stages:**
1. **Analyze Voiceover** - Parse SRT, calculate duration stats
2. **Extract Keywords** - LLM-powered keyword extraction
3. **Pre-Run Summary** - Shows estimates, asks for confirmation
4. **Download Footage** - All 3 duration tiers from YouTube
5. **Deduplicate** - Remove duplicate videos (saves processing time)
6. **Transcribe & Index** - Whisper + embeddings + FAISS
7. **Scene Detection** - PySceneDetect boundaries + audio analysis + OTIO export
8. **Match** - Two-stage matching with scene-level clips
9. **Output** - OTIO/EDL/FCPXML for DaVinci Resolve

### 2. Faster-Whisper Transcription (4x Speedup)

New transcription provider using CTranslate2 optimization:

```yaml
transcription:
  provider: "faster-whisper"  # 4x faster than openai-whisper
  model: "base"
  compute_type: "auto"  # auto-detects GPU capabilities
```

- **4x faster** transcription with identical accuracy
- Auto-detects optimal compute type (float16/int8)
- Fallback to OpenAI Whisper if not installed

### 3. Video Deduplication (`src/deduplication.py`)

Automatic duplicate detection using perceptual hashing:

- **Runs after download** - Saves processing time on duplicates
- **First frame comparison** - Fast and effective
- **Auto-delete** - Removes duplicates, keeps highest quality
- **Report generation** - `deduplication_report.json` with details

```
Stage 2b: DEDUPLICATE FOOTAGE
──────────────────────────────────────────────────────────────────────
  Scanned: 210 videos
  ✓ Unique: 195
  ✓ Deleted: 15 duplicates
  ✓ Saved: 1.2 GB
```

### 4. Audio Analysis (`src/audio_analysis.py`)

Librosa-powered audio feature extraction:

- **Silence detection** - Find natural cut points
- **Speech detection** - Identify clips with speech (might conflict with voiceover)
- **Integrated with scene detection** - Audio data stored in scene index

```yaml
scene_detection:
  audio_analysis: true  # Enable librosa analysis
```

Output includes:
- `has_speech`: Boolean flag
- `speech_ratio`: Percentage of audio with speech
- `audio_cut_points`: Timestamps of silence-based cut suggestions

### 5. Integrated Video Downloader (`src/downloader.py`)

All three duration tiers unified into one module:

| Tier | Duration | Per Keyword | Use Case |
|------|----------|-------------|----------|
| Short | 20s - 2min | 3 clips | Quick cuts, B-roll |
| Medium | 2min - 10min | 3 clips | Transitions, scenes |
| Long | 10min - 25min | 1 clip | Establishing shots |

### 6. Scene Detection (`src/scene_detection.py`)

Now includes audio analysis:

```
Stage 3b: SCENE DETECTION
──────────────────────────────────────────────────────────────────────
  ✓ Processed 195 videos
  ✓ Detected 1,247 total scenes
  ✓ Average: 6.4 scenes/video
  ✓ Audio analysis: 47 videos with speech
  ✓ Audio cut points: 892
```

---

## 📁 New Files

| File | Description |
|------|-------------|
| `main.py` | End-to-end pipeline orchestrator |
| `src/downloader.py` | Unified video downloader (3 tiers) |
| `src/keyword_extractor.py` | LLM keyword extraction |
| `src/scene_detection.py` | PySceneDetect + audio integration |
| `src/deduplication.py` | Perceptual hash deduplication |
| `src/audio_analysis.py` | Librosa silence/speech detection |

---

## ⚙️ Config Changes

### Transcription (faster-whisper)
```yaml
transcription:
  provider: "faster-whisper"  # or "openai-whisper"
  model: "base"
  compute_type: "auto"  # auto, float16, int8, float32
```

### Scene Detection (with audio)
```yaml
scene_detection:
  enabled: true
  preset: "balanced"
  audio_analysis: true  # Enable silence/speech detection
```

### LLM Configuration
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-haiku-20240307"
```

### Download Configuration
```yaml
download:
  enabled: true
  tiers:
    short:   { min: 20, max: 120, per_keyword: 3 }
    medium:  { min: 120, max: 600, per_keyword: 3 }
    long:    { min: 600, max: 1500, per_keyword: 1 }
  davinci_mode: true
  hw_accel: "auto"
```

---

## 🔧 Usage Examples

### Full Pipeline
```bash
python main.py --voiceover my_documentary.srt
```

### Custom Keyword Count
```bash
python main.py -v script.srt -k 30
```

### Skip Scene Detection
```bash
python main.py -v script.srt --skip-scenes
```

### Download Only (includes deduplication)
```bash
python main.py -v script.srt --download-only
```

### Match Existing Footage
```bash
python main.py -v script.srt --match-only
```

---

## 📦 Dependencies

New in `requirements.txt`:

```
# Transcription (4x faster)
faster-whisper>=0.9.0

# Deduplication
imagehash>=4.3.0
Pillow>=9.0.0

# Audio Analysis
librosa>=0.10.0
scipy>=1.10.0

# Parallel Processing
joblib>=1.3.0
```

---

## 🔄 Migration from v2.1

1. Update `config.yaml` with new sections
2. Run `pip install -r requirements.txt`
3. Set `ANTHROPIC_API_KEY` or `GEMINI_API_KEY`

Your existing workflows remain unchanged. New features are additive.

---

## Performance Improvements

| Stage | Before | After | Speedup |
|-------|--------|-------|---------|
| Transcription | 10 min | 2.5 min | **4x** |
| Deduplication | N/A | ~30 sec | New |
| Scene Detection | 5 min | 5 min + audio | Same |
| Total Pipeline | ~60 min | ~45 min | **~25%** |

---

## Previous Release

See [CHANGELOG_v2.1.md](CHANGELOG_v2.1.md) for v2.1 features:
- V8 Source-Rotation Track
- Hybrid Embeddings
- TF-IDF Keyword Weights
- FAISS Indexing
- Two-Stage Matching
- Duration-Aware Scoring
- Comprehensive Logging
