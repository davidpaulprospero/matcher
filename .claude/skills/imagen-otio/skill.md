---
name: imagen-otio
description: Generate AI still images from voiceover text using Google Imagen or Gemini Flash (gemini-2.5-flash-image) and output a V12-image-only OTIO timeline. Use when user says "generate images from SRT", "make imagen OTIO", "imagen timeline", or wants to produce an OTIO with AI-generated still images placed by voiceover timing.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
  - Agent
---

# Imagen OTIO Skill

Generate AI still images from voiceover segments using **Google Imagen or Gemini Flash** (`gemini-2.5-flash-image`), then build an OTIO timeline with those images placed on the **V12 track** at their corresponding voiceover timestamps.

**Default model: `gemini-2.5-flash-image`** — Uses the standard Gemini API `generate_content` endpoint with a separate quota pool from Imagen. Does NOT hit the Imagen 70 RPD limit. Fixed 1024px resolution, $0.039/image.

## When to Use

- User says "generate images from SRT", "make imagen OTIO", "build imagen timeline"
- User wants AI-generated still images placed on V12 track from an SRT voiceover file
- User asks to "generate imagen images" or "create OTIO with generated images"
- User provides an SRT and wants an image-only OTIO timeline

## Invocation

```
/imagen-otio <voiceover_or_project_path> [--size WxH] [--quality standard|fast|ultra] [--budget USD] [--output <path>]
/imagen-otio <voiceover_or_project_path> --dry-run
```

## Arguments

- `<voiceover_or_project_path>` (required): Path to an SRT file or a project directory containing `voiceover/` subdirectory with an SRT. If a project dir is given, finds the SRT automatically.
- `--size <WxH>` (optional): Image resolution. **Default: `1408x768`** (1K 16:9). `gemini-2.5-flash-image` uses `aspect_ratio` internally — size maps to ratio but resolution is fixed 1024px.
- `--quality` (optional): Ignored for `gemini-2.5-flash-image`. For `imagen-4.0-*` models: `fast` ($0.02/img), `standard` ($0.04/img), `ultra` ($0.06/img).
- `--budget <USD>` (optional): Max spend in USD. Default: `2.0`
- `--output <path>` (optional): Output path for the OTIO file. Default: same directory as input, named `generated_images.otio`
- `--dry-run` (optional): Parse SRT, build batches, show all prompts and estimated costs — **zero billable calls**

## Workflow

### Phase 1: Locate Voiceover

Resolve the input to an SRT file path:

1. If path is a `.srt` file → use it directly
2. If path is a directory → look for `voiceover/*.srt` (or `voiceover/srt/*.srt`) and pick the first or most recent
3. Fail if no SRT found

Also resolve the project directory (for output path resolution):
- If input was a file: parent directory of the SRT
- If input was a directory: that directory

### Phase 2: Parse SRT and Build Batches

```python
import opentimelineio as otio
from src.generated_images import build_generated_image_batches, GeneratedImageService
from src.config.sections.generated_images import GeneratedImagesConfig, IMAGEN_SUPPORTED_SIZES
from src.state import VoiceoverSegment

# Parse SRT into VoiceoverSegments
def parse_srt(srt_path: str) -> List[VoiceoverSegment]:
    # Use src.transcription.srt_parser if available, or manual parse
    # Each entry: index (1-based), start, end, text
    ...

segments = parse_srt(srt_path)
```

Build batches using `build_generated_image_batches()` from `src.generated_images`:

```python
config = GeneratedImagesConfig(
    enabled=True,
    min_segments_per_image=7,
    max_segments_per_image=9,
    batch_target_segments=8,
    image_size={"width": W, "height": H},
    quality=quality,
    budget_usd=budget,
)
batches = build_generated_image_batches(segments, config)
```

### Phase 3: Generate Images with Imagen

```python
from src.generated_images import GeneratedImageService

# Load API key from .env or config
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get('GEMINI_API_KEY', '') or os.environ.get('GOOGLE_API_KEY', '')

service = GeneratedImageService(config, api_key=api_key)
output_dir = str(project_dir / "generated_images")

results = service.generate_for_batches(
    batches=batches,
    topic_context="",  # no topic context for standalone use
    output_dir=output_dir,
)
```

Each `GeneratedImageResult` has: `batch_id`, `file`, `prompt`, `width`, `height`, `start_time`, `end_time`, `segment_indices`, `cost_usd`.

### Phase 4: Build OTIO Timeline with V12 Images

```python
import opentimelineio as otio
from opentimelineio.opentime import RationalTime, TimeRange

RATE = 30.0  # frames per second

# Create timeline
timeline = otio.schema.Timeline(
    name="generated_images_timeline",
    metadata={
        "Resolve_OTIO": {"Resolve OTIO Meta Version": "1.0"},
    }
)

# Global start time (1 hour timecode = 108000 frames at 30fps)
global_start = RationalTime(value=108000.0, rate=RATE)
timeline.global_start_time = global_start

# Stack
stack = otio.schema.Stack(name="Video", metadata={"Resolve_OTIO": {"Locked": False}})
stack.enabled = True
stack.color = None

# V12: Generated Images track (AI-generated stills)
v12_track = otio.schema.Track(name="V12 - Generated Images", kind=otio.schema.TrackKind.Video)
v12_track.enabled = True  # Images ARE enabled — user wants to see them
v12_track.color = None
v12_track.metadata["Resolve_OTIO"] = {"Locked": False}

# Add a Gap from 0 to first image start time
first_start = results[0].start_time if results else 0.0
if first_start > 0:
    gap_duration = RationalTime(value=int(first_start * RATE), rate=RATE)
    v12_track.append(otio.schema.Gap(duration=gap_duration))

# Add each generated image
for result in results:
    start_frame = int(result.start_time * RATE)
    end_frame = int(result.end_time * RATE)
    duration_frames = end_frame - start_frame

    # Media reference
    media_ref = otio.schema.ExternalReference(
        target_url=result.file.replace("\\", "/"),
        available_range=TimeRange(
            duration=RationalTime(value=duration_frames, rate=RATE),
            start_time=RationalTime(value=0, rate=RATE),
        ),
    )
    media_ref.name = Path(result.file).name

    # Clip
    clip = otio.schema.Clip(name=f"IMG:{result.batch_id}", media_reference=media_ref)
    clip.source_range = TimeRange(
        start_time=RationalTime(value=0, rate=RATE),
        duration=RationalTime(value=duration_frames, rate=RATE),
    )
    clip.enabled = True
    clip.color = None
    clip.metadata["batch_id"] = result.batch_id
    clip.metadata["prompt"] = result.prompt[:200] if result.prompt else ""
    clip.metadata["width"] = result.width
    clip.metadata["height"] = result.height

    # Add FreezeFrame effect to make it a still image
    freeze = otio.schema.FreezeFrame()
    clip.effects.append(freeze)

    v12_track.append(clip)

    # Gap to next image
    # (handled in next iteration or at end)

stack.append(v12_track)
timeline.tracks.append(stack)
```

**Important**: Use `file:///E:/...` format for Windows absolute paths in `target_url`, or forward-slash Windows paths like `E:/path/to/file.png`.

### Phase 5: Write OTIO

```python
output_path = Path(output_otio_path)
otio.adapters.write_to_file(timeline, str(output_path))
```

Also write a summary JSON alongside the OTIO:
```json
{
  "images_generated": <count>,
  "total_cost_usd": <float>,
  "images": [
    {
      "batch_id": "generated_000",
      "file": "E:/path/to/generated_000.png",
      "start_time": 12.5,
      "end_time": 24.3,
      "segment_count": 8,
      "cost_usd": 0.02
    }
  ]
}
```

## Output Location

Default: `<project_dir>/generated_images.otio`
Default images directory: `<project_dir>/generated_images/`

## Complete Example

```
/imagen-otio "E:\Edit Job\Stu\my_project\voiceover\voiceover.srt" --size 1792x1024 --budget 5.0
```

Output:
- OTIO: `E:\Edit Job\Stu\my_project\generated_images.otio`
- Images: `E:\Edit Job\Stu\my_project\generated_images/generated_000.png`, etc.
- Summary: `E:\Edit Job\Stu\my_project\generated_images\summary.json`

## Key Dataclasses

**`GeneratedImageBatch`** (from `src.state`):
```python
batch_id: str
segment_start_index: int
segment_end_index: int
segment_count: int
start_time: float      # seconds
end_time: float        # seconds
text: str              # concatenated segment texts
segment_indices: List[int]
```

**`GeneratedImageResult`** (from `src.state`):
```python
batch_id: str
file: str               # absolute path to PNG
prompt: str
width: int
height: int
start_time: float      # seconds
end_time: float        # seconds
segment_start_index: int
segment_end_index: int
segment_indices: List[int]
cost_usd: float
```

**`GeneratedImagesConfig`** (from `src.config.sections.generated_images`):
```python
enabled: bool = False
output_dir: str = "generated_images"
image_size: GeneratedImageSizeConfig = field(default_factory=GeneratedImageSizeConfig)
quality: str = "standard"
budget_usd: float = 2.0
allowed_channels: List[str] = field(default_factory=lambda: ["STU"])
model: str = "gemini-2.5-flash-image"  # $0.039/image — uses Gemini API, separate quota, no 70 RPD limit
```

**`IMAGEN_SUPPORTED_SIZES`** (from `src.config.sections.generated_images`):
```python
{(1024, 576), (1408, 768), (1536, 2816), (2816, 1536), (2048, 1152)}  # 16:9 focused + 2K
```

## Error Handling

- **No GEMINI_API_KEY**: Fail with clear message about needing the key in `.env`
- **No SRT found**: Fail with path resolution details
- **Imagen API error**: Log warning, continue to next batch, report partial results
- **Empty results**: Produce an empty-V12 timeline (track exists but no clips) and report it
- **Invalid size**: Validate against `IMAGEN_SUPPORTED_SIZES` before generating

## Script Location

`scripts/imagen_otio.py` — standalone script with no pipeline dependencies. Calls into `src/generated_images/` and `src/otio/` directly.