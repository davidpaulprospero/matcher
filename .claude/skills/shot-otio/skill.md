---
name: shot-otio
description: Transcribe a video to SRT using faster-whisper, then generate one AI still image per voiceover segment using Google Gemini Flash (gemini-2.5-flash-image), and output a V12-image-only OTIO timeline. Use when user says "make a shot OTIO", "one image per segment", "shot images from video", or wants to transcribe a video and generate per-segment AI images on V12.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Agent
---

# Shot OTIO Skill

Transcribe a video to SRT using **faster-whisper**, then generate **one AI image per voiceover segment** using **Google Gemini Flash** (`gemini-2.5-flash-image`), and produce an OTIO timeline with those images on the **V12 track**.

Unlike `/imagen-otio` which batches 7-9 segments per image, this skill generates **one image per individual SRT segment** — ideal for short videos where every line of narration needs its own visual.

**Default model: `gemini-2.5-flash-image`** — Uses the standard Gemini API `generate_content` endpoint. Fixed 1024px resolution, $0.039/image.

## When to Use

- User says "make a shot OTIO", "one image per segment", "shot images from video"
- User provides a video file (not an SRT) and wants per-segment AI images
- User wants to transcribe a video and generate AI images for each line of narration
- User has a short video and wants every segment to have its own generated image

## Invocation

```
/shot-otio <video_or_project_path> [--size WxH] [--budget USD] [--output <path>]
/shot-otio <video_or_project_path> --dry-run
/shot-otio <video_or_project_path> --enhance-prompts
/shot-otio <video_or_project_path> --skip-transcribe
```

## Arguments

- `<video_or_project_path>` (required): Path to a video file (`.mov`, `.mp4`, etc.) or a project directory containing a video. If a directory is given, the most-recently-modified video file is used.
- `--size <WxH>` (optional): Image resolution. **Default: `1408x768`** (1K 16:9).
- `--budget <USD>` (optional): Max spend in USD. Default: `2.0`
- `--output <path>` (optional): Output path for the OTIO file. Default: `<project>/shot_images.otio`
- `--dry-run` (optional): Show all segments and prompts without calling the API — **zero billable calls**. Outputs an enhanced preview including timing in HH:MM:SS:FF format, resolution aspect ratio (e.g. "16:9 1408x768"), per-batch cost, total timeline duration and frame count, and bare-prompt warnings suggesting `--enhance-prompts`. Use `--verbose` for additional detail.
- `--verbose` (optional): Show extra detail in dry-run: segment word count, prompt length rating relative to the 75-word recommended max, and estimated generation time per image (~3s base + word-proportional overhead).
- `--enhance-prompts` (optional): Use Ollama (local, free) to rewrite each segment into a vivid visual scene description before generating images. Zero Gemini API calls for enhancement — only the Imagen image generation costs apply. Also updates the dry-run preview to show enhanced vs raw text.
- `--skip-transcribe` (optional): Skip transcription step and use an existing SRT if found
- `--project-prefix` (optional): Prefix for renamed output files (e.g. `wcyzksaq`)

## Workflow

### Phase 1: Discover Video

Resolve the input to a video file path:
1. If path is a video file (`.mov`, `.mp4`, `.avi`, etc.) → use it directly
2. If path is a directory → find the most-recently-modified video in that directory
3. Fail if no video found

Project directory is the parent of the video file.

### Phase 2: Transcribe to SRT

```python
from src.transcription.whisper_client import WhisperClient
from src.transcription.utils import write_srt
from src.state import VoiceoverSegment

video_path = Path("E:/Edit Job/Stu/Shorts/1/Stu tiktok1 .mov")
client = WhisperClient(model_name='medium', model_version='v3-turbo')
segments_raw, info = client.transcribe(str(video_path), language=None, word_timestamps=False)

# Normalize to VoiceoverSegments
segments = []
for i, seg in enumerate(segments_raw):
    text = seg.get('text', '').strip() if isinstance(seg, dict) else getattr(seg, 'text', '').strip()
    start = seg.get('start', 0.0) if isinstance(seg, dict) else getattr(seg, 'start', 0.0)
    end = seg.get('end', 0.0) if isinstance(seg, dict) else getattr(seg, 'end', 0.0)
    if text:
        segments.append(VoiceoverSegment(index=i, start=float(start), end=float(end), text=text))

# Write SRT for reference
srt_path = project_dir / "voiceover.srt"
write_srt([{'id': i, 'start': s.start, 'end': s.end, 'text': s.text} for i, s in enumerate(segments)], str(srt_path))
```

**Note**: `WhisperClient.transcribe()` returns `(segments, info)` — always unpack as a tuple. Segments may be dicts or objects; normalize to dict before use.

### Phase 3: Build Per-Segment Batches

```python
from src.generated_images import build_generated_image_batches
from src.config.sections.generated_images import GeneratedImagesConfig, GeneratedImageSizeConfig

config = GeneratedImagesConfig(
    enabled=True,
    image_size=GeneratedImageSizeConfig(width=W, height=H),
    quality="standard",
    budget_usd=budget,
    min_segments_per_image=1,   # KEY: one segment per image
    max_segments_per_image=1,   # KEY: one segment per image
    batch_target_segments=1,    # KEY: one segment per image
)

batches = build_generated_image_batches(segments, config)
# len(batches) == len(segments) — one batch per segment
```

### Phase 4: Generate Images

```python
from src.generated_images import GeneratedImageService
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')
service = GeneratedImageService(config, api_key=api_key)
output_dir = str(project_dir / "shot_images")

results = service.generate_for_batches(
    batches=batches,
    topic_context="",
    output_dir=output_dir,
)
```

Each `GeneratedImageResult` has: `batch_id`, `file`, `prompt`, `width`, `height`, `start_time`, `end_time`, `segment_indices`, `cost_usd`.

### Phase 5: Build OTIO Timeline

Same V12 structure as `/imagen-otio` — one clip per image with a `FreezeFrame` effect.

```python
from scripts.shot_otio import build_v12_otio

timeline = build_v12_otio(results, prefix=args.project_prefix or "")
otio.adapters.write_to_file(timeline, str(output_otio))
```

**Important**: Use forward-slash Windows paths (`E:/path/to/file.png`) in `target_url`.

## Output Location

- OTIO: `<project_dir>/shot_images.otio`
- Images: `<project_dir>/shot_images/`
- SRT: `<project_dir>/voiceover.srt`
- Summary: `<project_dir>/shot_images/summary.json`

## Complete Example

```
/shot-otio "E:\Edit Job\Stu\Shorts\1\Stu tiktok1 .mov" --size 1408x768 --budget 3.0
```

Output:
- SRT: `E:\Edit Job\Stu\Shorts\1\voiceover.srt`
- OTIO: `E:\Edit Job\Stu\Shorts\1\shot_images.otio`
- Images: `E:\Edit Job\Stu\Shorts\1\shot_images/shot_000.png`, etc.
- Summary: `E:\Edit Job\Stu\Shorts\1\shot_images\summary.json`

## Key Differences from `/imagen-otio`

| Feature | `/imagen-otio` | `/shot-otio` |
|---|---|---|
| Input | SRT or project with SRT | Video file or project with video |
| Batching | 7-9 segments per image | 1 segment per image |
| Transcription | User must provide SRT | Built-in via faster-whisper |
| Use case | Long narrations, montage | Short videos, every line illustrated |
| Output file | `generated_images.otio` | `shot_images.otio` |
| Images dir | `generated_images/` | `shot_images/` |
| SRT output | Not created | `voiceover.srt` written to project dir |

## Key Dataclasses

**`VoiceoverSegment`** (from `src.state`):
```python
index: int
start: float   # seconds
end: float     # seconds
text: str
```

**`GeneratedImageBatch`** (from `src.state`):
```python
batch_id: str
segment_start_index: int
segment_end_index: int
segment_count: int      # will be 1 for shot-otio
start_time: float
end_time: float
text: str              # single segment text
segment_indices: List[int]
```

**`GeneratedImageResult`** (from `src.state`):
```python
batch_id: str
file: str
prompt: str
width: int
height: int
start_time: float
end_time: float
segment_indices: List[int]
cost_usd: float
```

## Error Handling

- **No video found**: Fail with clear message listing supported extensions
- **Transcription fails**: Propagate whisper exception with clear log message
- **No GEMINI_API_KEY**: Fail with clear message about needing the key in `.env`
- **Empty results**: Produce an empty-V12 timeline and report it
- **Invalid size**: Validate against `IMAGEN_SUPPORTED_SIZES` before generating

## Script Location

`scripts/shot_otio.py` — standalone script with no pipeline dependencies. Calls into `src/generated_images/`, `src/transcription/`, and `src/otio/` directly.
