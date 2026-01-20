# DaVinci Resolve Integration Roadmap

> **Vision:** AI handles 60-80% of editing work, DaVinci becomes review/polish tool, system learns from every project.

## Executive Summary

| Layer | Purpose | Tools | Cost |
|-------|---------|-------|------|
| **1. Analysis** | Transcription, faces, beats | Whisper.cpp, Pyannote, MediaPipe, Librosa | Free (local) |
| **2. Organization** | Bins, timelines, markers | DaVinci Resolve Python API | Free (Studio for API) |
| **3. Scaling** | Batch renders, variations | Shotstack, Creatomate, Frame.io | ~$0.10-0.30/render |
| **4. Feedback** | Pacing, continuity, diversity | Ollama, Moondream 2B, CLIP | Free (local) |

---

## Monthly Cost Estimates (2026)

> **Exchange Rate:** 1 USD = ₱59.50

| Tier | Description | USD | PHP |
|------|-------------|-----|-----|
| **Hobbyist** | Personal projects | $0-15 | ₱0-893 |
| **Freelancer** | 5-10 projects/mo | $50-150 | ₱2,975-8,925 |
| **Small Studio** | 20-30 projects/mo | $200-500 | ₱11,900-29,750 |
| **Production House** | 50+ projects | $800-2,000 | ₱47,600-119,000 |

### Service Pricing

| Service | Rate | Notes |
|---------|------|-------|
| **Local Whisper** | Free | Batch only, ~3% WER |
| **Deepgram Nova-2** | $0.0043/min | <300ms latency, 2.8% WER |
| **AssemblyAI** | $0.0058/min | Real-time + diarization |
| **Gemini 2.5 Flash** | $0.15/$0.60 per 1M tokens | Best value |
| **Gemini Vision** | ~$0.001/img | Multimodal |
| **Ollama (local)** | Free | Privacy, offline |
| **Pexels/Pixabay** | Free tier | 20k/5k requests |
| **ElevenLabs** | $5-330/mo | Voice synthesis |
| **Shotstack** | ~$0.10/min | Cloud rendering |
| **AI Video (Sora/Veo/Runway)** | ~$3/min | B-roll only, expensive |

**Cost Tips:** Local first ($0) → Gemini Flash over Pro (8x cheaper) → AI video sparingly

---

## Current State

| Component | Status | Priority |
|-----------|--------|----------|
| Pipeline core, OTIO/XML/EDL | ✅ Complete | — |
| Voiceover matching, B-roll detection | ✅ Complete | — |
| **Fusion template system** | 🔲 Not started | **Highest** |
| **graphics_cues.json** | 🔲 Not started | **Highest** |
| Batch render queue | 🔲 Not started | High |
| Feedback capture system | 🔲 Not started | High |
| Reaper audio bridge | 🔲 Not started | Medium |

---

## DaVinci API Reference

| Category | Can Do | Cannot Do |
|----------|--------|-----------|
| **Project** | Create/load/save, switch databases | — |
| **Media Pool** | Import, bins, metadata | Analyze waveforms |
| **Timeline** | Create, add/move clips, markers | — |
| **Rendering** | Queue jobs, set formats, monitor | — |
| **Fusion** | Modify via page-switching | Insert templates from library |
| **AI Features** | Scene detection, smart reframe | Custom vision |

**Key APIs:**
```python
# Timeline insertion (batch = 10-20x faster)
media_pool.AppendToTimeline([{'mediaPoolItem': clip, 'startFrame': 120, 'endFrame': 240, 'trackIndex': 1}])

# Markers with metadata
timeline.AddMarker(frameId, color, name, note, duration, customData)

# Fusion modification (workaround for no template insertion)
resolve.OpenPage("fusion")
comp = resolve.Fusion().GetCurrentComp()
comp.FindTool("TextPlus").SetInput("StyledText", "TEXT")
```

**DaVinci 19:** Clip color groups, node queries, keyframe control. **Breaking:** UI Manager now Studio-only.

### DaVinci 20 New APIs

| Object | New Method | Purpose |
|--------|-----------|---------|
| **Project** | `GetColorGroupsList()`, `AddColorGroup()` | Batch color grading |
| **Timeline** | `GetVoiceIsolationState()`, `SetVoiceIsolationState()` | Fairlight AI isolation |
| **TimelineItem** | `SetName()`, `ExportLUT()`, `AssignToColorGroup()` | Clip management |
| **Graph** | `SetNodeCacheMode()`, `GetNumNodes()`, `SetLUT()` | Node graph control |
| **Folder** | `TranscribeAudio()`, `ClearTranscription()` | Built-in transcription |
| **Resolve** | `GetFairlightPresets()` | Audio preset automation |

**New Objects:** `ColorGroup` (batch grading), `Graph` (node manipulation). **Cloud:** `CreateCloudProject()`, `LoadCloudProject()`.

**Still GUI-only:** IntelliScript, AI Multicam SmartSwitch, AI Animated Subtitles, IntelliCut.

---

## Phase 1: Foundation & Graphics

### Fusion Templates

**Limitation:** No direct template insertion. Workaround: pre-create → duplicate → modify via Fusion page.

| Aspect | Details |
|--------|---------|
| **Path** | `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Templates\Edit\` |
| **.setting Files** | Text-based JSON, editable, drag to import |
| **TextPlus** | Follower, TextScramble, TextTimer hidden features |
| **Macros** | Group tools → Macro Editor → save to Templates |

### Auto-Graphics System

Pipeline outputs `graphics_cues.json` for entity first-mentions:
```json
{"cues": [{"time": "00:01:23", "entity": "Tokyo", "type": "location", "template": "lower_third_location"}]}
```

**Templates:** lower_third_location/person, chapter_title, stat_counter, quote_callout, map_locator

### Reaper ReaScript

| Feature | Tool | Notes |
|---------|------|-------|
| LUFS Normalization | SWS Extension | -23 LUFS broadcast standard |
| Noise Reduction | ReaFIR | Subtractive mode |
| Python API | `reapy` library | 30-60 calls/sec limit from external |

### Animated Maps

| Tool | Best For | Cost |
|------|----------|------|
| **Blender + Globe** | 3D procedural globes | Free |
| **AvoMap** | Quick flight paths | Free |
| **Travel Animator** | Professional AE routes | $59/yr |
| **Mapbox Static** | Programmatic 2D maps | 50k free/mo |

---

## Phase 2: Audio Analysis

### Transcription (2026)

| Tool | Speed | Best For |
|------|-------|----------|
| **faster-whisper** | 8x | Pure transcription, 4.7GB |
| **WhisperX** | 2-3 min | Multi-speaker + diarization |
| **Pyannote community-1** | — | 8-10% DER (best open-source) |

### Audio Separation

| Tool | SDR | Speed | VRAM |
|------|-----|-------|------|
| **Demucs v4** | 9.0 dB | 1x | 4GB |
| **Spleeter** | 8.2 dB | 10x | 2GB |

**Use:** Dialogue isolation, music bed creation, ducking, beat sync from drums stem.

### Fairlight API

**Limited Python access.** Workaround: markers for ducking points, Ducker Track FX (Resolve 19+).

| Feature | Method |
|---------|--------|
| Volume/Panning | Timeline automation |
| EQ/Dynamics | Track effects |
| Ducking | Ducker Track FX |
| VST/AU plugins | Supported |

---

## Phase 3: Vision Analysis

### Tool Stack

| Tool | Purpose | Speed |
|------|---------|-------|
| **MediaPipe** | Face detection (468 landmarks) | <10ms |
| **OpenCV FFT** | Blur detection | Fast |
| **CLIP ViT-L/14** | Scene similarity | Fast |

### Smart Reframe

**Studio only.** No direct API. Fusion workaround: Transform node + MediaPipe face positions → animated keyframes.

### Local Vision-Language Models

| Model | Size | Best For |
|-------|------|----------|
| **Llama 3.2 Vision 11B** | 7GB | General video, Ollama native |
| **MiniCPM-V 2.6** | 5.5GB | OCR, 95.7% DocVQA |
| **Moondream 2B** | 2GB | Edge/Raspberry Pi |

### Video Stabilization

**DaVinci API:** `TimelineItem.Stabilize()` - no parameters, GUI-only settings. Pre-process for automation.

| Tool | Method | Cropping | Speed | Best For |
|------|--------|----------|-------|----------|
| **Gyroflow** | Gyro data | Minimal (5-10%) | Real-time | Action cams with gyro |
| **FFmpeg vidstab** | Two-pass optical flow | 10-20% | 0.3-0.5x | Automation, batch |
| **DaVinci built-in** | Perspective/Similarity | Variable | Fast | Interactive preview |
| **warp_stabilizer (AE)** | Subspace warp | 5-15% | Slow | Complex motion |

**FFmpeg workflow (automatable):**
```bash
ffmpeg -i input.mp4 -vf vidstabdetect=shakiness=8:accuracy=15 -f null -  # Pass 1: analyze
ffmpeg -i input.mp4 -vf vidstabtransform=smoothing=30:crop=black:zoom=5 out.mp4  # Pass 2: apply
```

**Gyroflow integration:** OpenFX plugin → DaVinci timeline, minimal cropping with gyro-based correction.

### Video Upscaling

| Tool | Quality | Speed (1080p→4K) | VRAM | Cost |
|------|---------|------------------|------|------|
| **Topaz Video AI** | Best (Proteus/Gaia) | 2-5 FPS | 8GB | $396/yr |
| **DaVinci Super Scale** | Very Good | 40 FPS | 6GB | Studio |
| **Real-ESRGAN** | Good | 1-3 FPS | 6GB | Free |
| **waifu2x** | Anime-focused | 5-10 FPS | 4GB | Free |

**DaVinci Super Scale:** Project Settings → Master Settings → Enable Super Scale (2x/3x/4x), 8x faster than Topaz.

**Real-ESRGAN batch:**
```bash
realesrgan-ncnn-vulkan -i input_folder -o output_folder -n realesrgan-x4plus -s 4
```

**Model selection:** Proteus (general), Gaia (CG/faces), Artemis (denoising), realesrgan-x4plus (free).

---

## Phase 4: LLM & Assembly

### Assembly Automation

| Type | Features |
|------|----------|
| **Rough Cut** | Topic assembly, duration targeting |
| **Interview** | Paper edit, question removal |
| **Multicam** | Speaker switching, reaction shots |
| **Jump Cuts** | Silence/filler removal |
| **Repurposing** | Long→short, horizontal→vertical |

### Multicam Auto-Switching

| Tool | Platform | Cost |
|------|----------|------|
| **DaVinci SmartSwitch** | Resolve 20+ | Studio |
| **AutoPod** | Premiere | $29/mo |
| **Gling AI** | Standalone | $15/mo |

**Custom:** Pyannote (who speaks) + MediaPipe (face→camera) + rules (8s same angle → reaction shot)

### Jump Cut & Silence Removal

| Method | Tool | Accuracy |
|--------|------|----------|
| **Silero VAD** | PyTorch | 95%+ |
| **Filler detection** | Whisper + NLP | 90%+ |

**Parameters:** min_silence=0.3s, max_cut=2.0s, padding=0.05s

### Vertical Video (9:16)

| Platform | Resolution | Safe Zones (top/bottom/sides) |
|----------|------------|-------------------------------|
| **TikTok** | 1080x1920 | 150px / 270px / 40px |
| **Reels** | 1080x1920 | 120px / 250px / 40px |
| **Shorts** | 1080x1920 | 100px / 200px / 40px |

Auto-crop: MediaPipe face detection → center crop → animated keyframes

### Speed Ramping & Frame Interpolation

**DaVinci API:** No speed point/retime access. Workaround: OTIO `LinearTimeWarp(time_scalar=0.5)` for 2x slow-mo.

| Tool | Quality | Speed (1080p) | VRAM | Best For |
|------|---------|---------------|------|----------|
| **RIFE v4.25** | High | 50-100 FPS | 6GB | General, real-time |
| **FILM** | Best | 5-15 FPS | 8-12GB | Complex motion |
| **Topaz Apollo** | High | 10-30 FPS | 6GB | Sports, action |
| **DaVinci Speed Warp** | Highest | Slow | GPU | Hero shots (Studio) |

**DaVinci methods:** Optical Flow Standard/Enhanced, Speed Warp (Neural Engine, Studio).

**Auto slow-mo:** librosa beat detection → high-motion frames (OpenCV optical flow) → RIFE interpolation.

### Auto Chapter Generation

**YouTube requirements:** First chapter at `0:00`, minimum 3 chapters, 10s+ each.

| Approach | Accuracy | Speed | Notes |
|----------|----------|-------|-------|
| **AssemblyAI auto_chapters** | ~85% | Fast | Commercial, best quality |
| **LLM + TF-IDF** | ~80% | Medium | Preserves timestamps via cosine similarity |
| **DeepTiling (embeddings)** | ~75% | Slow | sentence-transformers sliding window |

**Workflow:** Transcript → topic segmentation → LLM title generation → YouTube API `videos.update`.

**DaVinci integration:** Export chapters as EDL markers → import via `Timeline Markers from EDL`.

---

## Phase 5: Feedback & Learning

### ML Feedback Loops

| Data Source | Purpose |
|-------------|---------|
| Marker corrections (green/red) | Positive/negative examples |
| Timing adjustments | Timing preference model |
| Alternative selections (V2/V3 over V1) | Ranking model |

**Training progression:** Weight tuning (50+ corrections) → Preference ranking (200+) → Fine-tuned embeddings (1000+)

**Bias prevention:** 10% random exploration to discover new patterns.

---

## Phase 6: External APIs

### Cloud Rendering

| Service | Speed | Cost |
|---------|-------|------|
| **Shotstack** | 10x | ~$0.10/min |
| **Creatomate** | 8x | Marketing focus |
| **Local DaVinci** | 1x | Free |

### Voice Synthesis

| Service | Quality | Latency | Cost |
|---------|---------|---------|------|
| **ElevenLabs** | Excellent | 75ms | $5-330/mo |
| **Coqui/XTTS v2** | Very Good | 200ms | Free (local) |
| **Bark** | Excellent | 5-10s | Free (12GB VRAM) |
| **Piper** | Good | <50ms | Free (<1GB) |

### n8n Workflow Automation

| Workflow | Trigger | Actions |
|----------|---------|---------|
| Render notification | File in output folder | Slack + upload |
| Footage ingest | New file in watch folder | Transcode → organize |
| Error alerting | Log pattern match | Pushover alert |

**DaVinci integration:** Folder monitoring + webhook from Python script

### Music Licensing APIs

| Provider | API Type | Search | Stems | Content ID | Cost |
|----------|----------|--------|-------|------------|------|
| **Epidemic Sound** | Partner | BPM, mood, vocals, duration | Yes | Yes | Custom |
| **Artlist** | Enterprise | Full catalog | No | Yes | Custom |
| **Jamendo** | Public | Tags, duration | No | No | Free/Paid |
| **Pixabay** | Public | Limited | No | N/A | Free |
| **Loudly** | Enterprise | AI-generated | Yes | Yes | Custom |

**Best free option:** Jamendo API (500k+ tracks, 35k req/month free).

**Epidemic MCP Server:** Claude integration for natural language music search (`search_music`, `edit_recordings_for_custom_lengths`, `download_music_track`).

**Stem separation:** PoYo API ($0.35/track), LALAL.AI, StemRoller (free/Demucs).

### Subtitles

| Format | Styling | Use Case |
|--------|---------|----------|
| **SRT** | Basic | Universal |
| **WebVTT** | CSS | Web |
| **ASS** | Full | Anime (limited DaVinci support) |

**DaVinci:** 128 subtitle tracks, no built-in STT or translation. Workflow: Whisper → SRT → DeepL → reimport.

### Scene Detection

| Detector | F1 Score |
|----------|----------|
| **PySceneDetect Adaptive** | 91.59% |
| **DaVinci built-in** | ~86% |

### Social Media Auto-Publishing

| Platform | API | Upload Limit | Quota/Rate Limit | Notes |
|----------|-----|--------------|------------------|-------|
| **YouTube** | Data API v3 | 256GB | 1600 units/upload, 10k/day | Best documented |
| **TikTok** | Content Posting | 4GB/287MB | Audit required for public | 10 uploads/day/user |
| **Instagram** | Graph API | 1GB | Business/Creator only | Via `media` + `media_publish` |
| **Twitter/X** | Media API | 512MB | Rate limited | Chunked upload |

**YouTube workflow:**
```python
from googleapiclient.discovery import build
youtube = build('youtube', 'v3', credentials=creds)
request = youtube.videos().insert(
    part="snippet,status",
    body={"snippet": {"title": title, "categoryId": "22"}, "status": {"privacyStatus": "private"}},
    media_body=MediaFileUpload(video_path, chunksize=1024*1024, resumable=True)
)
```

**TikTok requirements:** App audit for public posting, `video.upload` scope, 10 uploads/day.

**Multi-platform:** Use n8n or custom Python with platform SDKs. Stagger uploads (YouTube first, TikTok 24h later).

### YouTube Algorithm & SEO

| Factor | Target | Weight |
|--------|--------|--------|
| **Watch Time** | Maximize minutes | #1 Primary |
| **AVD (30s mark)** | >50% retention | Critical |
| **CTR** | 5-10% (elite: 10%+) | High |
| **Engagement velocity** | First 48 hours | High |

**Retention hooks:** Bold claim (0-3s) → Pattern interrupt (8-15s) → Preview payoff (15-30s).

**Optimal posting:** Tue-Fri, 12-4 PM local. **Shorts:** 30-45s sweet spot, loop completion rate.

### YouTube Monetization

| Tier | Subscribers | Watch Hours | Shorts Views |
|------|-------------|-------------|--------------|
| **Early Access** | 500 | 3,000 (12 mo) | 3M (90 days) |
| **Full Monetization** | 1,000 | 4,000 (12 mo) | 10M (90 days) |

**Mid-rolls:** 8+ min required, place at scene changes. **Shorts RPM:** $0.01-0.06/view (45% creator share).

| Niche | RPM Range |
|-------|-----------|
| Finance/Legal | $10-25 |
| Tech/Education | $8-14 |
| Gaming | $3-7 |

### Copyright & Content ID

| Service | Use Case | Pricing |
|---------|----------|---------|
| **YouTube Checks** | Pre-publish scan | Free (upload as Private) |
| **ACRCloud** | Audio fingerprinting | Free tier available |
| **Audible Magic** | Enterprise detection | ~$0.08/min |
| **Songview (ASCAP/BMI)** | Rights lookup | Free |

**Pre-publish:** Upload Private → wait for Checks → fix claims → publish. **Fair use:** No safe duration threshold—courts decide.

### AI Content Detection & Disclosure

| Platform | Required Disclosure | Penalty |
|----------|---------------------|---------|
| **YouTube** | Cloned voices, deepfakes, fake events | Strikes, demonetization |
| **TikTok** | All AI-generated content (Jan 2025) | Immediate strikes |
| **Instagram** | Realistic altered video/audio | Reach suppression |

**Safe:** AI color correction, scriptwriting, production assistance. **Flagged:** 100% AI without human perspective.

**Watermarking:** C2PA (metadata, easily stripped), SynthID (pixel-level, survives compression).

### AI Dubbing & Localization

| Tool | Languages | Lip Sync | Pricing | API |
|------|-----------|----------|---------|-----|
| **ElevenLabs** | 29 | Yes | ~$0.80-1.20/min | Yes |
| **Rask.ai** | 130+ | Yes (2x cost) | $1.20-1.50/min | Yes |
| **HeyGen** | 175+ | Yes (Avatar) | ~$0.55/min | Yes |
| **Deepdub** | 100+ | Yes | Enterprise | Yes |

**Best for:** ElevenLabs (voice cloning), Rask.ai (high-volume), HeyGen (avatars), Deepdub (broadcast).

### Review & Collaboration Tools

| Tool | Pricing | DaVinci Panel | API |
|------|---------|---------------|-----|
| **Frame.io** | $15-25/user/mo | Yes | Yes |
| **Dropbox Replay** | $10-12/user/mo | Yes | No |
| **ftrack** | $15-30/user/mo | Community | Yes |
| **Wipster** | $12-40/user/mo | No | Yes |

**DaVinci integration:** Frame.io and Dropbox Replay have direct panels. ftrack via community plugin.

### Proxy Workflows

| Codec | Quality | File Size | Best For |
|-------|---------|-----------|----------|
| **ProRes Proxy** | Excellent | Medium | DaVinci/FCPX default |
| **DNxHR LB** | Very Good | Medium-Large | Avid/broadcast |
| **H.264** | Good | Smallest | Cross-platform |

**DaVinci:** Right-click → Generate Proxy Media → Playback → Prefer Proxies. Auto-switches to full-res on export.

**Resolution:** 4K source → 1/2 res (2K) for quality; 6K/8K → 1/4 res for speed.

### Competitor Auto-Edit Tools

| Tool | Key Feature | Pricing | API |
|------|-------------|---------|-----|
| **Descript** | Edit-by-transcript, Overdub | $15-30/mo | Enterprise |
| **Gling AI** | Jump cuts, bad take detection | $15/mo | No |
| **OpusClip** | Long-to-short, virality scoring | $15-29/mo | Business tier |
| **Runway** | Gen-3/4 AI video | $15/mo+ | Yes |
| **AutoPod** | Multicam auto-switch | $29/mo | No |

**Our differentiators:** Voiceover-first matching (unique), auto B-roll sourcing, OTIO/EDL/XML output.

### Auto-Color Matching

**DaVinci:** Color Match (source→target), Auto Color (starting point), Shot Match (manual).
**Workflow:** Grade hero shot → export 3D LUT → apply to similar shots via script.

### Thumbnail Generation

| Factor | Weight | Method |
|--------|--------|--------|
| Face presence | High | MediaPipe |
| Visual saliency | Medium | OpenCV |
| Motion blur | Negative | FFT |

**YouTube:** 1280x720 min, <2MB, faces with eye contact +38% CTR.

### Logo/Watermark Removal

| Element | Detection | Accuracy |
|---------|-----------|----------|
| Static logos | YOLO/CNN | 95%+ |
| Transparent watermarks | Texture learning | 85-90% |
| News tickers | OCR + motion | 90%+ |
| Letterbox/pillarbox | Edge detection | 98%+ |

**Process:** Mask detection (YOLO) → Inpainting (ProPainter) → Temporal consistency (ConvLSTM)

**Tools:** ProPainter (free, SOTA), Runway ($15/mo), HitPaw ($50/yr)

**Broadcast Safe Areas:** EBU 90%/80%, SMPTE 93%/90%, Web 100%/95%

---

## Phase 7: Render & Performance

### Render API

**Critical:** `SetCurrentRenderFormatAndCodec()` MUST precede `SetRenderSettings()`

```python
project.SetCurrentRenderFormatAndCodec("MP4", "H265")
project.SetRenderSettings({"TargetDir": "/exports", "CustomName": "video"})
project.AddRenderJob()
project.StartRendering(isInteractiveMode=False)  # Headless = 15% faster
```

### Platform Presets

| Platform | Resolution | Format | Limit |
|----------|------------|--------|-------|
| YouTube | 1920x1080/4K | H.264/H.265 | — |
| TikTok/Reels | 1080x1920 | H.264 | 10min/90sec |

### GPU Optimization

| Optimization | Impact |
|--------------|--------|
| Proxies | 4x playback |
| NVENC | Hardware H.264/H.265/AV1 |
| Pre-transcode | DNxHR/ProRes before import |

**VRAM:** 1080p=4GB, 4K=8GB, 8K=16GB. **AV1:** RTX 40+ only, 40% better than H.265.

### DaVinci Project Server

**PostgreSQL-based** collaboration + render distribution.
- Single job can't distribute across nodes simultaneously
- Render node needs identical media paths or shared storage
- Blackmagic Cloud: ~$5/user/month, up to 50 users

---

## Implementation Priority

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| **NOW** | Fusion templates, graphics_cues.json | 2-3d | VH |
| | Batch render queue | 1d | H |
| **SOON** | Jump cut generator | 1w | VH |
| | Speaker diarization | 1w | H |
| | Reaper audio bridge | 3d | H |
| **LATER** | Multicam auto-switch | 2w | VH |
| | Cross-project learning | 1w | H |
| | Video stabilization (vidstab) | 2d | M |
| | Social media auto-publish | 1w | H |
| **FUTURE** | ML model training | 1mo | VH |
| | Video upscaling (Super Scale) | 3d | M |

---

## Technical Requirements

### Software

| Category | Tools |
|----------|-------|
| **Core** | Python 3.10+, DaVinci Resolve Studio 18+ |
| **Audio** | Librosa, Pyannote, Demucs/Spleeter |
| **Vision** | OpenCV, MediaPipe, CLIP |
| **ML/AI** | PyTorch, sentence-transformers, Ollama |

### Hardware (2026)

| Component | 1080p | 4K | 8K/Heavy AI |
|-----------|-------|-----|-------------|
| **CPU** | 8 cores | 12+ cores | 16+ cores |
| **RAM** | 16-32 GB | 32-64 GB | 64-128 GB |
| **GPU** | RTX 4060 / Arc B580 | RTX 5080 / M4 Pro | RTX 5090 (32GB) |
| **Storage** | Gen4 NVMe | Gen4/5 NVMe | Gen5 NVMe RAID |

**GPU Benchmarks (DaVinci):**
| GPU | Score | Export Speed |
|-----|-------|--------------|
| RTX 5090 (32GB) | 13,370 | 60% faster than 4090 |
| RTX 5080 (16GB) | 11,659 | +23% vs 4080 Super |
| M4 Max | — | 2x faster than M4 Pro |
| Arc B580 (12GB) | — | Best AV1 under $300 |

**Storage:** Gen5 NVMe = 14,000+ MB/s (8K RAW, multi-cam 4K).

### API Keys

| Service | Purpose | Priority |
|---------|---------|----------|
| **Gemini** | Vision, LLM | Required |
| **Pexels/Pixabay** | Stock footage | Required |
| **GeoNames** | Locations | Required |
| **ElevenLabs** | Voice | Optional |
| **Frame.io** | Review | Optional |

---

## File Structure

**Scripts (`scripts/davinci/`):**
- `foundation/` - resolve_utils, quick_setup, track_manager ✅
- `graphics/` - fusion_templates, map_generator (NEW)
- `render/` - batch_render, platform_presets (NEW)
- `analysis/` - audio, vision, content (Planned)
- `assembly/` - rough_cut, multicam, jump_cut (Planned)

**Output:** `timeline.otio`, `timeline_PART1-4.otio`, `segments.json`, `graphics_cues.json`, `render_queue.json`

---

## Timeline Interchange

| Format | Contents | Maturity |
|--------|----------|----------|
| **OTIO** | Metadata only | Maturing (use XML for critical handoffs until 2027) |
| **OTIOZ** | Timeline + media (ZIP) | Portable but large |
| **XML (FCPXML)** | Full timeline + paths | Most compatible |
| **AAF** | Timeline + optional media | Industry standard |

**Media Offline:** Import XML bins first → populate Media Pool → then import OTIO (auto-links)

---

## Technology Maturity (2026)

| Technology | Status | Notes |
|------------|--------|-------|
| **DaVinci Fusion template insertion** | ❌ | Page-switching workaround |
| **DaVinci auto-captioning** | ❌ | Use Whisper → SRT |
| **OTIO interchange** | ⚠️ | XML safer until 2027 |
| **AI video consistency** | ⚠️ | Challenging across 30+ shots |
| **Smart Reframe API** | ⚠️ | Studio only, Fusion workaround |
| **Fairlight Python API** | ⚠️ | Limited, use markers |
| **Transparent watermark removal** | ⚠️ | 85-90% accuracy |
| **faster-whisper** | ✅ | 4x faster, production-ready |
| **WhisperX + Pyannote** | ✅ | Best open-source diarization |
| **Deepgram real-time** | ✅ | <300ms latency |
| **NVENC AV1** | ✅ | RTX 40+, 40% better than H.265 |
| **PySceneDetect** | ✅ | 91.59% F1 |
| **MediaPipe** | ✅ | Real-time face detection |
| **ProPainter inpainting** | ✅ | 2023 SOTA |
| **Demucs/Spleeter** | ✅ | Audio separation |
| **n8n automation** | ✅ | Open-source, self-hosted |
| **RIFE frame interpolation** | ✅ | 50-100 FPS @ 1080p, 6GB VRAM |
| **AssemblyAI auto chapters** | ✅ | ~85% accuracy, commercial API |
| **Jamendo music API** | ✅ | Public, 500k+ tracks free |
| **Epidemic Sound API** | ⚠️ | Partner-only, MCP integration |
| **DaVinci speed point API** | ❌ | GUI only, use OTIO LinearTimeWarp |
| **DaVinci Stabilize API** | ⚠️ | No parameters, pre-process with vidstab |
| **DaVinci Super Scale** | ✅ | 8x faster than Topaz, Studio only |
| **Gyroflow stabilization** | ✅ | OpenFX plugin, gyro-based, minimal crop |
| **FFmpeg vidstab** | ✅ | Two-pass, fully automatable |
| **Real-ESRGAN upscaling** | ✅ | Free, 4x upscale, batch support |
| **YouTube Data API v3** | ✅ | 1600 units/upload, well-documented |
| **TikTok Content Posting** | ⚠️ | Audit required for public posting |
| **Instagram Graph API** | ⚠️ | Business/Creator accounts only |
| **DaVinci 20 ColorGroup API** | ✅ | Batch grading automation |
| **DaVinci 20 Voice Isolation** | ✅ | Fairlight AI, scriptable |
| **ElevenLabs Dubbing API** | ✅ | 29 languages, lip sync |
| **Rask.ai Dubbing API** | ✅ | 130+ languages, first API-first |
| **Frame.io API v4** | ✅ | Adobe-integrated, Camera to Cloud |
| **ACRCloud fingerprinting** | ✅ | Free tier, 150M+ tracks |
| **C2PA watermarking** | ⚠️ | Easily stripped on upload |
| **SynthID watermarking** | ✅ | Survives compression |
| **YouTube Analytics API** | ✅ | Retention curves, CTR tracking |
| **RTX 5090 NVENC** | ✅ | 9th-gen, AV1 Ultra Quality |
| **ProRes Proxy workflow** | ✅ | 4-10x playback improvement |

### When to Use Cloud vs Local

| Scenario | Recommendation |
|----------|----------------|
| Privacy-sensitive | Local Whisper |
| Large batch | Deepgram/AssemblyAI |
| Real-time | Deepgram (<300ms) |
| 100+ renders | Shotstack |
| Single project | Local DaVinci |

---

## Research Sources

| Category | Resources |
|----------|-----------|
| **DaVinci** | [ResolveDevDoc](https://resolvedevdoc.readthedocs.io), [We Suck Less Forum](https://www.steakunderwater.com/wesuckless/) |
| **Transcription** | [WhisperX](https://github.com/m-bain/whisperX), [Pyannote](https://pyannote.ai), [Deepgram](https://deepgram.com) |
| **Vision** | [MediaPipe](https://developers.google.com/mediapipe), [CLIP](https://openai.com/research/clip), [Moondream](https://moondream.ai) |
| **Audio** | [Demucs](https://github.com/facebookresearch/demucs), [Silero VAD](https://github.com/snakers4/silero-vad), [ElevenLabs](https://elevenlabs.io) |
| **Video APIs** | [Shotstack](https://shotstack.io), [OTIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO) |
| **AI Video** | [Sora](https://openai.com/sora), [Veo 2](https://cloud.google.com/vertex-ai), [Runway](https://runwayml.com) |
| **Inpainting** | [ProPainter](https://github.com/sczhou/ProPainter), [EasyOCR](https://github.com/JaidedAI/EasyOCR), [YOLO](https://github.com/ultralytics/ultralytics) |
| **Automation** | [n8n](https://n8n.io), [Reaper ReaScript](https://www.reaper.fm/sdk/reascript/) |
| **Frame Interp** | [RIFE](https://github.com/hzwer/ECCV2022-RIFE), [FILM](https://film-net.github.io/), [Topaz Video AI](https://www.topazlabs.com/topaz-video-ai) |
| **Chapters** | [AssemblyAI auto_chapters](https://www.assemblyai.com/docs/speech-understanding/auto-chapters), [DeepTiling](https://github.com/Ighina/DeepTiling) |
| **Music APIs** | [Epidemic Sound](https://developers.epidemicsound.com), [Jamendo](https://developer.jamendo.com), [Artlist](https://developer.artlist.io) |
| **Stabilization** | [Gyroflow](https://gyroflow.xyz), [FFmpeg vidstab](https://github.com/georgmartius/vid.stab), [DaVinci Stabilizer](https://documents.blackmagicdesign.com) |
| **Upscaling** | [Topaz Video AI](https://www.topazlabs.com/topaz-video-ai), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [DaVinci Super Scale](https://documents.blackmagicdesign.com) |
| **Social APIs** | [YouTube Data API](https://developers.google.com/youtube/v3), [TikTok Content Posting](https://developers.tiktok.com), [Instagram Graph API](https://developers.facebook.com/docs/instagram-api) |
| **AI Dubbing** | [ElevenLabs](https://elevenlabs.io), [Rask.ai](https://rask.ai), [HeyGen](https://heygen.com), [Deepdub](https://deepdub.ai) |
| **Copyright** | [ACRCloud](https://acrcloud.com), [Audible Magic](https://audiblemagic.com), [Songview](https://ascap.com/songview) |
| **Review Tools** | [Frame.io](https://frame.io), [ftrack](https://ftrack.com), [Dropbox Replay](https://dropbox.com/replay) |
| **AI Detection** | [C2PA](https://c2pa.org), [SynthID](https://deepmind.google/models/synthid), [Reality Defender](https://realitydefender.com) |
| **Competitors** | [Descript](https://descript.com), [Gling](https://gling.ai), [OpusClip](https://opus.pro), [Runway](https://runwayml.com) |
| **Hardware** | [Puget Systems](https://pugetsystems.com), [Tom's Hardware](https://tomshardware.com) |

---

*Last updated: January 21, 2026 | Document version: 5.0*

**Changelog:**
- v5.0: Major research update - DaVinci 20 API (ColorGroup, Voice Isolation, Graph), YouTube SEO/Algorithm, YouTube Monetization (YPP, RPM by niche), Copyright Detection (Content ID, ACRCloud, pre-publish), AI Content Detection (C2PA, SynthID, disclosure requirements), AI Dubbing (ElevenLabs, Rask.ai, HeyGen), Review Tools (Frame.io, ftrack, Dropbox Replay), Proxy Workflows, Competitor Analysis (Descript, Gling, OpusClip, Runway), Hardware 2026 (RTX 50, M4, Gen5 NVMe)
- v4.1: Video Stabilization (Gyroflow, FFmpeg vidstab, DaVinci limitations), Video Upscaling (Topaz, Real-ESRGAN, DaVinci Super Scale), Social Media Auto-Publishing (YouTube, TikTok, Instagram APIs)
- v4.0: Music Licensing APIs (Epidemic, Jamendo, Artlist), Speed Ramping (RIFE, FILM, OTIO LinearTimeWarp), Auto Chapter Generation (AssemblyAI, LLM+TF-IDF, YouTube API)
- v3.9: Major compaction (1861→~550 lines, 70% reduction) - consolidated tables, removed verbose code examples, preserved all essential information
- v3.8: Logo/Watermark Removal & Content-Aware Cropping
- v3.7: Project Server, Fairlight API, Jump Cuts, Vertical Video, Color Matching, Thumbnails
- v3.6: ML Feedback Loops, Multicam Auto-Switching, n8n Automation
- v3.5: Smart Reframe, Audio Separation, Voice Synthesis, Animated Maps
- v3.4: Fusion Templates, Reaper ReaScript, Auto-Captioning
