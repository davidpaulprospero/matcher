---
name: wan-facecam
description: Generate facecam (talking-head lip-synced) videos from presenter image + voiceover audio using Alibaba WAN API
allowed-tools:
  - Bash(python:*)
  - Bash(ffmpeg:*)
  - Bash(ffprobe:*)
  - Bash(pip:*)
  - Glob
  - Read
  - Grep
---

# WAN Facecam Generation

Generate talking-head lip-synced videos from a presenter image and voiceover audio using the Alibaba WAN 2.5 API (DashScope).

Trigger phrases: "generate facecam", "wan facecam", "facecam video", "talking head video", "wan lipsync"

## Workflow

### 1. Verify Prerequisites

```bash
# Check API key is set
python -c "import os; key = os.environ.get('DASHSCOPE_API_KEY', ''); print(f'API key: {\"set (\" + key[:8] + \"...)\" if key else \"NOT SET\"}')"

# Check dashscope is installed
python -c "import dashscope; print(f'dashscope {dashscope.__version__}')"

# Check ffmpeg is available
ffmpeg -version 2>&1 | head -1
```

If `dashscope` is not installed:
```bash
pip install "dashscope>=1.25.8,<2.0"
```

### 2. Validate Inputs

Confirm the user has provided:
- **Audio file** — voiceover audio (MP3, WAV, M4A, etc.)
- **Image file** — presenter/facecam image (JPG, PNG)

Both files must exist on disk. Verify with a quick check:
```bash
python -c "from pathlib import Path; [print(f'{p}: {\"OK\" if Path(p).exists() else \"MISSING\"}') for p in ['<audio_path>', '<image_path>']]"
```

### 3. Run Generation

```bash
python scripts/wan_facecam.py \
  --audio "<audio_path>" \
  --image "<image_path>" \
  --output "<output_dir>" \
  --max-duration <seconds>
```

Default options (override as needed):
- `--model wan2.5-i2v-preview` — primary model with audio lip-sync
- `--resolution 480P` — cheapest, fastest; use `720P` or `1080P` for quality
- `--chunk-duration 10` — 10-second video chunks (wan2.5 supports 5 or 10)
- `--max-duration 120` — process up to 2 minutes of audio
- `--region international` — use `beijing` for China region

Typical generation time: ~3-5 minutes per 10-second chunk.

### 4. Verify Output

```bash
ffprobe -v error -show_entries stream=codec_name,duration,width,height -of default=noprint_wrappers=1 "<output_path>"
```

### 5. Report Results

Report to the user:
- Output video path and file size
- Total duration generated
- Number of chunks (successful/failed)
- Resolution and model used

## Model Details

| Model | Audio Support | Duration | Notes |
|-------|--------------|----------|-------|
| `wan2.5-i2v-preview` | Yes (lip-sync) | 5 or 10s | Primary — best for facecam |
| `wan2.2-i2v-plus` | No (silent) | 5s fixed | Fallback — more stable |

The script automatically falls back to wan2.2 if wan2.5 fails.

## Pricing (480P)

- wan2.5-i2v-preview 480P: ~$0.04 per 10s chunk
- 1 minute of facecam ≈ 6 chunks ≈ ~$0.24

## Troubleshooting

- **"DASHSCOPE_API_KEY not set"** — Export the key: `export DASHSCOPE_API_KEY=sk-xxx`
- **Task timeout** — WAN generation can take 3-10 minutes per chunk; the SDK polls automatically
- **"Both models failed"** — Check API key validity, quota, and region setting
- **Audio too short** — API requires minimum 3s audio; the script auto-pads short final chunks
