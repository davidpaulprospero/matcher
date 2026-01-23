# DaVinci Resolve Integration Roadmap

> **Vision:** AI handles 60-80% of editing work, DaVinci becomes review/polish tool, system learns from every project.

## Executive Summary

| Layer | Purpose | Tools | Cost |
|-------|---------|-------|------|
| **1. Analysis** | Transcription, faces, beats | Whisper, Pyannote, MediaPipe, Librosa | Free (local) |
| **2. Organization** | Bins, timelines, markers | DaVinci Resolve Python API | Free (Studio for API) |
| **3. Scaling** | Batch renders, variations | Shotstack, Creatomate, Frame.io | ~$0.10-0.30/render |
| **4. Feedback** | Pacing, continuity, diversity | Ollama, Moondream, CLIP | Free (local) |
| **5. Programmatic** | Direct renders, social clips | Remotion, Lambda | $0.01-0.10/render |

---

## Service Pricing (2026)

| Service | Rate | Service | Rate |
|---------|------|---------|------|
| Local Whisper | Free | Deepgram Nova-3 | $0.0043/min |
| AssemblyAI | $0.0025/min | Rev.ai | $0.003/min |
| Gemini Flash | $0.15/1M tokens | ElevenLabs | $0.30/1K chars |
| Pexels/Pixabay | Free | Shotstack | ~$0.10/min |
| Storyblocks | $15-35/mo | AI Video (Sora/Runway) | ~$3/min |

**Tiers:** Hobbyist $0-15/mo | Freelancer $50-150/mo | Studio $200-500/mo | Production $800-2000/mo

---

## Current State

| Component | Status | Component | Status |
|-----------|--------|-----------|--------|
| Pipeline core, OTIO/XML/EDL | ✅ | **Fusion templates** | 🔲 Highest |
| Voiceover matching, B-roll | ✅ | **graphics_cues.json** | 🔲 Highest |
| Batch render queue | 🔲 High | Feedback capture | 🔲 High |

---

## DaVinci API

| Category | Can Do | Cannot Do |
|----------|--------|-----------|
| Project/Media Pool | Create, import, bins, metadata | Analyze waveforms |
| Timeline | Create, clips, markers | Speed points, retime |
| Rendering | Queue, formats, monitor | — |
| Fusion | Modify via page-switch | Insert templates directly |
| AI Features | Scene detection, reframe, stabilize | Custom vision, IntelliScript |

**Key Pattern:** `media_pool.AppendToTimeline([{...}])` for batch (10-20x faster). Markers: `timeline.AddMarker(frame, color, name, note, duration, customData)`.

### DaVinci 20 New APIs

| Method | Purpose | Method | Purpose |
|--------|---------|--------|---------|
| `GetColorGroupsList()` | Batch grading | `SetVoiceIsolationState()` | Fairlight AI |
| `SetName()` | Clip naming | `ExportLUT()` | Per-clip LUT |
| `TranscribeAudio()` | Built-in STT | `GetFairlightPresets()` | Audio presets |

**New Objects:** ColorGroup, Graph. **Still GUI-only:** IntelliScript, SmartSwitch, IntelliCut.

---

## Phase 1: Foundation

### Fusion Templates
**Path:** `%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Templates\Edit\`
**Workaround:** Pre-create → duplicate → modify via Fusion page. `.setting` files are editable JSON.

### Auto-Graphics
Pipeline outputs `graphics_cues.json`: `{"cues": [{"time": "00:01:23", "entity": "Tokyo", "type": "location"}]}`

### Animated Maps
| Tool | Cost | Tool | Cost |
|------|------|------|------|
| Blender + Globe | Free | AvoMap | Free |
| Mapbox Static | 50k free/mo | Travel Animator | $59/yr |

---

## Phase 2: Audio

### Transcription Benchmarks

| Service | WER (Real) | Speed | Price/Min | Diarization |
|---------|------------|-------|-----------|-------------|
| **faster-whisper** | 7-10% | 4x RT | Free | Via WhisperX |
| **AssemblyAI** | 14.5% | 0.3-0.6 RT | $0.0025 | Yes (50 spkrs) |
| **Deepgram Nova-3** | ~18% | <0.3 RT | $0.0043 | Yes |
| **Rev.ai** | ~15% | Real-time | $0.003 | Yes |

**Note:** Provider WER uses clean audio; real-world is 14-21%.

### Audio Separation
| Tool | SDR | Speed | | Tool | SDR | Speed |
|------|-----|-------|-|------|-----|-------|
| Demucs v4 | 9.0 dB | 1x | | Spleeter | 8.2 dB | 10x |

---

## Phase 3: Vision

### Vision API Comparison (Cost per 10K images)

| Provider | Cost | Best For | Latency |
|----------|------|----------|---------|
| **Gemini 1.5 Flash** | **$0.20-0.80** | Bulk/budget | Fast |
| **GPT-4o (low-res)** | $2.10 | Reasoning | ~300ms |
| **AWS Rekognition** | ~$10 | Object/face detection | 5-15s |
| **Google Cloud Vision** | $15 | Labels, OCR | Low |
| **Claude Sonnet** | $48 | Complex scenes | 1-3s |
| **Local (Moondream)** | $0 | Edge, batch | 184 tok/s |

### Local Vision Models

| Model | VRAM | Speed | Best For |
|-------|------|-------|----------|
| **Moondream 2B** | 2.5GB | 184 tok/s | Detection, OCR, edge |
| **Llama 3.2 11B** | 22GB | 71 tok/s | Charts, 128K context |
| **MiniCPM-V 4.5** | 16GB | Fast | Video 10fps, beats GPT-4o |
| **Qwen2.5-VL 7B** | 29GB | Fast | Near GPT-4o accuracy |

**Recommendation:** Gemini Flash for cloud bulk ($0.02/1K), Moondream for local, MiniCPM-V for video frames.

### Detection Tools

| Tool | Purpose | Speed |
|------|---------|-------|
| MediaPipe | Face detection (468 landmarks) | <10ms |
| CLIP ViT-L/14 | Scene similarity | Fast |
| OpenCV FFT | Blur detection | Fast |

### Video Processing

| Task | Best Tool | Notes |
|------|-----------|-------|
| **Stabilization** | FFmpeg vidstab | Two-pass, automatable. DaVinci `Stabilize()` has no API params |
| **Upscaling** | DaVinci Super Scale | 8x faster than Topaz. Also: Real-ESRGAN (free) |
| **Frame Interp** | RIFE v4.25 | 50-100 FPS. DaVinci Speed Warp for hero shots |

### Watermark Removal & Avoidance

| Element | Detection | Accuracy | Tool |
|---------|-----------|----------|------|
| Static logos | YOLO/CNN | 95%+ | ProPainter |
| Transparent watermarks | Texture learning | 85-90% | ProPainter |
| News tickers | OCR + motion | 90%+ | EasyOCR |
| Letterbox/pillarbox | Edge detection | 98%+ | OpenCV |

**Inpainting Process:** Mask detection (YOLO) → Inpainting (ProPainter) → Temporal consistency (ConvLSTM)

**Inpainting Tools:** ProPainter (free, SOTA), Runway ($15/mo), HitPaw ($50/yr)

### Watermark Avoidance (Smart Cropping)

| Strategy | Method | Best For |
|----------|--------|----------|
| **Aspect ratio conversion** | 16:9→9:16 crops 68% width | Corner watermarks (natural removal) |
| **Smart Reframe** | DaVinci/Premiere AI subject tracking | Keep subject, avoid edges |
| **Exclusion zones** | YOLO + saliency, 15% margin | Programmable avoidance |
| **Pre-download check** | Stock API metadata, IMATAG | Avoid watermarked previews |

**NLE Tools:**
| Tool | Feature | Notes |
|------|---------|-------|
| DaVinci Smart Reframe | AI face/subject tracking | Studio only, manual power windows |
| Premiere Auto Reframe | Motion presets | Keyframeable masks |
| CapCut Auto Reframe | Free, social presets | Quick vertical conversion |

**Python approach:** YOLO subject detection + OpenCV saliency → find crop avoiding watermark zones while keeping subjects.

**Limitations:** Invisible watermarks (forensic/neural) cannot be avoided by cropping. Quality loss from over-cropping. Legal: avoidance on unlicensed footage violates ToS.

**Broadcast Safe Areas:** EBU 90%/80%, SMPTE 93%/90%, Web 100%/95%

### Local Vision Models
| Model | Size | Best For |
|-------|------|----------|
| Llama 3.2 Vision 11B | 7GB | General, Ollama |
| MiniCPM-V 2.6 | 5.5GB | OCR |
| Moondream 2B | 2GB | Edge devices |

---

## Phase 4: Assembly

| Type | Features |
|------|----------|
| **Jump Cuts** | Silero VAD (95%+), filler detection via Whisper+NLP |
| **Multicam** | Pyannote (speaker) + MediaPipe (face→camera) |
| **Vertical** | TikTok/Reels/Shorts: 1080x1920, safe zones vary |
| **Chapters** | AssemblyAI (~85%) or LLM+TF-IDF (~80%) |

**Speed Ramping:** No API. Use OTIO `LinearTimeWarp(time_scalar=0.5)`.

---

## Phase 5: Feedback & Learning

| Data Source | Purpose |
|-------------|---------|
| Marker corrections (green/red) | Positive/negative examples |
| Timing adjustments | Preference model |
| V2/V3 selections over V1 | Ranking model |

**Progression:** Weight tuning (50+) → Preference ranking (200+) → Fine-tuned embeddings (1000+)

---

## Phase 6: External APIs

### Voice Synthesis (TTS)

| Service | Quality | Latency | Cost | Cloning | Local |
|---------|---------|---------|------|---------|-------|
| ElevenLabs | 4.5/5 | 200-500ms | $0.30/1K | Yes | No |
| OpenAI TTS | 4.0/5 | ~500ms | $0.015/min | No | No |
| Coqui XTTS-v2 | 4.0/5 | <200ms | Free | Yes | 8-24GB |
| Piper | 3.5/5 | 20-30ms | Free | No | CPU/RPi |

### Stock Footage & Image Generation

| Provider | API | Pricing | Best For |
|----------|-----|---------|----------|
| Storyblocks | Yes (HMAC) | $15-35/mo unlimited | Automation |
| Pexels/Pixabay | Yes (free) | Free | Budget |
| **Ideogram** | Yes | $7-42/mo | Text in images |
| **Flux** | Yes | $0.014-0.06/img | Photorealism |
| Stable Diffusion | Self-host | Free | Volume |

**No API:** Artgrid, Envato Elements, Midjourney (ToS risk).

### Music & Audio

| Provider | API | Cost | Content ID |
|----------|-----|------|------------|
| Jamendo | Public | Free/Paid | No |
| Epidemic Sound | Partner | Custom | Yes |
| ACRCloud | Yes | Free tier | Fingerprinting |

### Social Media Publishing

| Platform | API | Quota | Notes |
|----------|-----|-------|-------|
| YouTube | Data API v3 | 1600 units/upload | Best documented |
| TikTok | Content Posting | Audit required | 10/day |
| Instagram | Graph API | Business only | Via `media_publish` |

### YouTube Optimization

| Factor | Target | | Tier | Requirements |
|--------|--------|-|------|--------------|
| Watch Time | Maximize | | Early Access | 500 subs, 3K hrs |
| AVD (30s) | >50% | | Full Monetization | 1K subs, 4K hrs |
| CTR | 5-10% | | Shorts RPM | $0.01-0.06/view |

**RPM by niche:** Finance $10-25 | Tech $8-14 | Gaming $3-7

### AI Dubbing

| Tool | Languages | Lip Sync | Price/Min |
|------|-----------|----------|-----------|
| ElevenLabs | 29 | Yes | $0.80-1.20 |
| Rask.ai | 130+ | Yes (2x) | $1.20-1.50 |
| HeyGen | 175+ | Avatar | $0.55 |

### Review Tools

| Tool | DaVinci Panel | API | Pricing |
|------|---------------|-----|---------|
| Frame.io | Yes | Yes | $15-25/user |
| Dropbox Replay | Yes | No | $10-12/user |
| ftrack | Community | Yes | $15-30/user |

### Copyright & AI Detection

**Pre-publish:** Upload Private → YouTube Checks → fix claims → publish. **Fair use:** No safe threshold.

**AI Disclosure required:** Cloned voices, deepfakes (YouTube/TikTok/Instagram). **Safe:** Color correction, scriptwriting.

**Watermarking:** C2PA (easily stripped), SynthID (survives compression).

---

## Phase 7: Render & Performance

**Critical:** `SetCurrentRenderFormatAndCodec()` MUST precede `SetRenderSettings()`.

### Hardware (2026)

| Component | 1080p | 4K | 8K/AI |
|-----------|-------|-----|-------|
| GPU | RTX 4060 / Arc B580 | RTX 5080 / M4 Pro | RTX 5090 (32GB) |
| RAM | 16-32 GB | 32-64 GB | 64-128 GB |
| Storage | Gen4 NVMe | Gen4/5 NVMe | Gen5 RAID |

**Benchmarks:** RTX 5090 = +60% vs 4090 | M4 Max = 2x M4 Pro | Gen5 NVMe = 14,000+ MB/s

### Proxy Workflows
**Codecs:** ProRes Proxy (best), DNxHR LB (broadcast), H.264 (smallest).
**Resolution:** 4K→1/2 res, 6K/8K→1/4 res. DaVinci auto-switches on export.

---

## Phase 8: Privacy & Accessibility

### Face Blur (GDPR)

| Tool | Speed | API | GDPR Certified |
|------|-------|-----|----------------|
| MediaPipe | 30-200 FPS | Python | Dev responsibility |
| DeepPrivacy2 | 8-12 FPS | CLI | Strong (synthetic) |
| brighter AI | Real-time | REST | Yes |
| Celantur | Real-time | REST+Docker | Yes |

**DaVinci:** No batch face blur API. Use BCC+ plugin or external preprocessing.

### 3D Integration

| Software | Python API | Batch Render | DaVinci |
|----------|------------|--------------|---------|
| Blender | `bpy` (PyPI) | `bpy.ops.render` | OTIO/XML |
| Unreal | `unreal` | MRQ | XML |
| Houdini | `hou` | ROPs | USD (no DaVinci) |

**USD:** 90% VFX adoption by 2026. No DaVinci support yet.

---

## Phase 9: Remotion (Programmatic Video)

**What:** React-based framework for code-driven video generation. Outputs MP4 directly without NLE.

### Core Capabilities

| Component | Purpose | Key Props |
|-----------|---------|-----------|
| `<OffthreadVideo>` | Import existing clips | `src`, `trimBefore`, `trimAfter`, `playbackRate` |
| `<Series>` | Sequential clip arrangement | `durationInFrames`, `offset` |
| `@remotion/captions` | SRT parsing, TikTok-style animation | `parseSrt()`, `createTikTokStyleCaptions()` |
| `renderMedia()` | Node.js programmatic rendering | `codec`, `inputProps`, `outputLocation` |

### Integration Options

| Option | Input | Output | Value |
|--------|-------|--------|-------|
| **Social Clip Generator** | matches.json | 9:16 vertical + animated captions | High |
| **Preview Renderer** | matches.json | Quick MP4 (no DaVinci needed) | Medium |
| **Graphics Layer** | graphics_cues.json | React components (titles, maps) | High |

### Rendering Options

| Method | Cost | Speed | Best For |
|--------|------|-------|----------|
| Local (Node.js) | Free | ~1x realtime | Single renders |
| Lambda (AWS) | $0.01-0.10/min | Parallel | Batch, <80min videos |
| Cloud Run (GCP) | Similar | Alpha | GCP shops |

### Pipeline Integration

| Phase | Deliverable |
|-------|-------------|
| 1. Export format | `--output-format remotion` → `remotion-composition.json` |
| 2. Remotion project | `/remotion` folder with composition reading JSON |
| 3. Social clips | Vertical generator with `@remotion/captions` |
| 4. CLI integration | `python main.py --render-social` calls Remotion |

**Licensing:** Free for individuals/small companies <$1M revenue. Company license $200-5000/yr.

**Docs:** [remotion.dev/docs](https://remotion.dev/docs) | MCP: `@remotion/mcp`

---

## Phase 10: Programmatic Video Ecosystem

### Cloud Video APIs (Template-based)

| Service | Approach | Pricing | Best For |
|---------|----------|---------|----------|
| **Shotstack** | JSON timeline | ~$0.05-0.15/min | Developer-first, MCP server available |
| **Creatomate** | REST API, visual editor | ~$0.08/min | Fast (<15s renders), keyframes |
| **JSON2Video** | JSON scenes | ~$0.10/min | Simple, web-dev mindset |
| **Plainly** | After Effects templates | $69+/mo (50 renders) | AE template automation |
| **Bannerbear** | Image→video overlays | $49-149/mo | Social media assets |
| **Editframe** | Node.js SDK | Custom | Figma→video pipelines |
| **Rendi** | Raw FFmpeg access | $0.15/GB | Full control, no templates |

### Python Libraries (Local)

| Library | Purpose | Performance | Notes |
|---------|---------|-------------|-------|
| **MoviePy 2.x** | High-level editing | Moderate | Cuts, concat, effects. v2 breaking changes |
| **PyAV** | FFmpeg bindings | Fast | Direct frame access, Numpy/Pillow integration |
| **ffmpeg-python** | FFmpeg wrapper | Fast | Programmatic command building |
| **GStreamer** | Pipeline framework | Very fast | ML integration via gst-python-ml (2025) |

### Video Delivery/Processing

| Service | Strength | Pricing |
|---------|----------|---------|
| **Cloudinary** | Auto-transcoding, CDN, transformations | Free tier, then usage |
| **Mux** | Streaming, analytics, HLS/DASH | ~$0.007/min streaming |
| **AWS MediaConvert** | Batch transcoding | ~$0.015/min |
| **Coconut** | Simple transcoding API | Affordable |
| **Transloadit** | Subtitle burn-in, robots | Usage-based |

### AI Video Tools (Silence/Jump Cuts)

| Tool | Type | Platform | Notes |
|------|------|----------|-------|
| **TimeBolt** | Standalone | Win/Mac | 1hr in 13s, UMCHECK™ filler detection |
| **SavvyCut** | Standalone | Web | AI speech detection |
| **Gling AI** | Standalone | Web | Text-based editing, bad take removal |
| **AutoCut** | Plugin | Premiere/DaVinci | Silence, captions, zoom, B-roll |
| **Cutback** | Plugin | Premiere | AI silence detection |
| **VEED.io** | Cloud | Web | Magic Cut, filler word removal |
| **Visla** | Cloud | Web | Auto Cut in free tier |

### Video Repurposing (Long→Short)

| Service | Strength | API | Pricing |
|---------|----------|-----|---------|
| **OpusClip** | ClipAnything™, Virality Score | Yes (Enterprise) | Freemium |
| **Vizard.ai** | Fast, multi-speaker, UGC ads | Limited | Freemium |
| **Klap** | Highlight detection | Yes | Usage-based |

### Embeddable SDKs

| SDK | Platforms | Use Case |
|-----|-----------|----------|
| **IMG.LY CE.SDK** | Web, iOS, Android, React Native, Flutter | White-label editor (500M+ creations/mo) |
| **OpenShot Library** | Desktop (C++/Python) | MLT-based, timeline control |

### Subtitle/Caption APIs

| Service | Features | Pricing |
|---------|----------|---------|
| **AssemblyAI** | 99 languages, SRT export | $0.0025/min |
| **Whisper (local)** | Free, accurate, Whisper.cpp/faster-whisper | Free |
| **Transloadit** | Generate + burn-in via robot | Usage-based |
| **Creatomate** | Word-by-word animation | Included |

### Selection Guide

| Need | Best Option |
|------|-------------|
| Social clips with captions | Remotion + @remotion/captions |
| AE template automation | Plainly |
| Simple JSON→video | Shotstack, JSON2Video, Creatomate |
| Local Python processing | MoviePy (simple), PyAV (performance) |
| Silence removal | TimeBolt (standalone), AutoCut (plugin) |
| Long→short repurposing | OpusClip, Vizard.ai |
| Embeddable editor | IMG.LY CE.SDK |
| Raw FFmpeg control | Rendi, ffmpeg-python |

---

## Accessibility

| Standard | Requirement |
|----------|-------------|
| WCAG 2.1 A | Captions for prerecorded |
| WCAG 2.1 AA | Live captions, audio descriptions |
| FCC/CVAA | CEA-708, 99%+ accuracy |

**Formats:** SRT (universal), WebVTT (web), CEA-708 (broadcast). **Seizure:** Max 3 flashes/sec.

---

## Timeline Interchange

| Format | Best For | Maturity |
|--------|----------|----------|
| OTIO | Metadata | Use XML until 2027 |
| XML (FCPXML) | Full timeline | Most compatible |
| AAF | Industry | Standard |

---

## Technology Maturity

| ✅ Ready | ⚠️ Limited | ❌ Not Available |
|----------|-----------|------------------|
| Remotion, Shotstack, Creatomate | OTIO interchange | Fusion template insertion |
| MoviePy 2.x, PyAV, ffmpeg-python | OpusClip API (enterprise) | Sora API |
| TimeBolt, AutoCut, Gling AI | Vizard API | |
| faster-whisper, WhisperX | | |
| Deepgram, AssemblyAI | Smart Reframe API | DaVinci auto-captioning |
| MediaPipe, PySceneDetect | Fairlight Python API | Speed point API |
| RIFE, Demucs, ProPainter | TikTok (audit required) | Midjourney API |
| DaVinci Super Scale | Instagram (business only) | USD in DaVinci |
| Storyblocks, Pexels API | C2PA watermarking | |
| Ideogram, Flux, DALL-E 3 | | |
| ElevenLabs, Rask.ai | | |
| Frame.io, YouTube API | | |
| RTX 5090 NVENC, Blender bpy | | |
| ProPainter inpainting | | |
| Smart Reframe (DaVinci/Premiere) | | |
| Gemini 1.5 Flash Vision | | |
| Moondream 2B, MiniCPM-V | | |

---

## Implementation Priority

| Priority | Features |
|----------|----------|
| **NOW** | Fusion templates, graphics_cues.json, batch render |
| **SOON** | Remotion social clips, Shotstack/Creatomate integration, AutoCut silence removal |
| **LATER** | OpusClip repurposing, MoviePy local processing, Cloudinary delivery |
| **FUTURE** | ML training, upscaling pipeline, IMG.LY embeddable editor |

---

## Research Sources

| Category | Resources |
|----------|-----------|
| **DaVinci** | [ResolveDevDoc](https://resolvedevdoc.readthedocs.io), [We Suck Less](https://steakunderwater.com/wesuckless/) |
| **Remotion** | [Docs](https://remotion.dev/docs), [Lambda](https://remotion.dev/lambda), [Captions](https://remotion.dev/docs/captions), [MCP](https://npmjs.com/@remotion/mcp) |
| **Video APIs** | [Shotstack](https://shotstack.io), [Creatomate](https://creatomate.com), [JSON2Video](https://json2video.com), [Plainly](https://plainlyvideos.com), [Editframe](https://editframe.com) |
| **Python** | [MoviePy](https://zulko.github.io/moviepy), [PyAV](https://pyav.org), [GStreamer](https://gstreamer.freedesktop.org) |
| **AI Editing** | [TimeBolt](https://timebolt.io), [AutoCut](https://autocut.com), [Gling](https://gling.ai), [OpusClip](https://opus.pro), [Vizard](https://vizard.ai) |
| **Delivery** | [Cloudinary](https://cloudinary.com), [Mux](https://mux.com), [Transloadit](https://transloadit.com) |
| **Transcription** | [WhisperX](https://github.com/m-bain/whisperX), [AssemblyAI](https://assemblyai.com), [Deepgram](https://deepgram.com) |
| **Vision/AI** | [MediaPipe](https://mediapipe.dev), [CLIP](https://openai.com/research/clip), [RIFE](https://github.com/hzwer/ECCV2022-RIFE) |
| **Audio** | [Demucs](https://github.com/facebookresearch/demucs), [ElevenLabs](https://elevenlabs.io), [Silero VAD](https://github.com/snakers4/silero-vad) |
| **Stock/Images** | [Storyblocks](https://api.storyblocks.com), [Pexels](https://pexels.com/api), [Ideogram](https://ideogram.ai), [Flux](https://bfl.ai) |
| **Social** | [YouTube API](https://developers.google.com/youtube/v3), [TikTok](https://developers.tiktok.com), [Frame.io](https://frame.io) |
| **Privacy** | [brighter AI](https://brighter.ai), [Celantur](https://celantur.com), [DeepPrivacy2](https://github.com/hukkelas/deep_privacy2) |
| **3D** | [Blender Python](https://docs.blender.org/api), [USD](https://openusd.org), [Houdini](https://sidefx.com) |
| **Hardware** | [Puget Systems](https://pugetsystems.com), [Tom's Hardware](https://tomshardware.com) |

---

*Last updated: January 22, 2026 | Document version: 6.5*

**Changelog:**
- v6.5: Phase 10 Programmatic Video Ecosystem - cloud APIs (Shotstack, Creatomate, Plainly), Python libs (MoviePy, PyAV, GStreamer), AI editing (TimeBolt, AutoCut, Gling), repurposing (OpusClip, Vizard), delivery (Cloudinary, Mux)
- v6.4: Remotion integration (Phase 9) - programmatic video, social clips, Lambda rendering, captions API
- v6.3: Vision API comparison (Gemini Flash cheapest at $0.02/1K, local models: Moondream, MiniCPM-V, Qwen-VL)
- v6.2: Restored watermark removal (ProPainter, YOLO), added watermark avoidance research (smart cropping, aspect ratio conversion, exclusion zones, pre-download detection)
- v6.1: Compacted from 867→~350 lines. Merged tables, removed redundancy, preserved all data.
- v6.0: Privacy & Accessibility, Stock APIs, Transcription benchmarks, Image Gen, TTS, Face Blur, 3D, WCAG
- v5.0: DaVinci 20 API, YouTube SEO/Monetization, Copyright, AI Detection, Dubbing, Review Tools, Hardware 2026
- v4.x: Stabilization, Upscaling, Social Publishing, Music APIs, Speed Ramping, Chapters
- v3.x: Watermark removal, Multicam, Feedback loops, Fusion templates
