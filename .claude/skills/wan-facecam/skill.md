# WAN Facecam Generation

## Argument Routing

This skill has two modes selected by the first argument:

| Argument | Usage | Description |
|----------|-------|-------------|
| `intro` | `/wan-facecam intro <project_path>` | Generate full intro facecam for a single project |
| `batch` | `/wan-facecam batch` | Batch-generate facecam segments for completed Stu queue projects |

**If no argument is provided**, infer from context:
- If a project path is given → `intro`
- If trigger phrase matches batch (see below) → `batch`
- Otherwise → ask the user

---

## Mode: `intro` — Single Project Facecam

**Use `scripts/auto_wan_facecam.py` for project-path based generation.** The script auto-detects channel, voiceover, avatar, and title -- then generates a talking-head lip-synced video via the Alibaba WAN 2.5 API (DashScope).

### Prep Helper (for file preparation only)

Use `auto_wan_facecam.py` helper functions or manual steps for prep work:
- Detect channel from folder name
- Find voiceover (prefers `voiceover_trimmed.mp3`)
- Select/rotate avatar
- Trim audio if exceeding max duration

### Fallback: Direct wan_facecam.py (low-level)

```bash
python scripts/wan_facecam.py --audio vo.mp3 --image presenter.jpg --output ./output/facecam
```

### Trigger Phrases

- "generate facecam"
- "wan facecam"
- "facecam video"
- "talking head video"
- "wan lipsync"
- "auto facecam"

## Auto-Detection

### Channel Detection from Folder Name

The auto script detects channel from project path:

| Folder Pattern | Channel |
|---------------|---------|
| DeepSeaReports | DSR |
| RennReports | RRU |
| JournalOfDrunkPeople | JDRP |
| Stu | STU |
| Contains "DSR" | DSR |
| Contains "RRU" | RRU |

### Voiceover Detection

Looks in order:
1. `<project>/voiceover/` subfolder
2. `<project>/` root

File extensions: .mp3, .wav, .m4a, .mp4

**ALWAYS prefer `voiceover_trimmed.mp3` over `voiceover.mp3`** when it exists. The trimmed version has silence/dead air removed by the pipeline and produces better results. Fall back to `voiceover.mp3` only if no trimmed version is available.

### Avatar Detection

| Channel | Avatar(s) | Notes |
|---------|-----------|-------|
| DSR | `Harold_V1.jpg` | Only V1 works -- V2-V6 fail on Degold backend |
| RRU | `RennActor.jpg` | Single avatar |
| STU | `Ethan.jpeg` | Located in `Stu/avatars/` |

Avatar paths:
- **STU**: `Stu/avatars/<avatar_file>` (dedicated directory)
- **All others**: `Degold/avatars/<channel>/<avatar_file>`, then `Degold/avatars/<avatar_file>`

## Parameters

- `$PROJECT_PATH` - Project directory path (required for auto mode)
- `$VIDEO_TITLE` - Video title (optional, auto-detected from folder name)
- `$CHANNEL_CODE` - Channel code like RRU, JDRP, DSR, STU (optional, auto-detected)
- `$AVATAR_PATH` - Path to avatar/presenter image (optional, auto-selected from channel)
- `$AUDIO_PATH` - Path to voiceover audio (optional, auto-detected from project)
- `$TRELLO_URL` - Trello card URL (optional, will extract title automatically)
- `$RESOLUTION` - 480P, 720P, or 1080P (default: 480P)
- `$MAX_DURATION` - Max audio duration in seconds (default: 30 for intro, 120 for batch)

## Parsing Titles from Trello URLs

If a Trello URL is provided (e.g., `https://trello.com/c/fr9huYEa/103-1-minute-ago-underwater-drone...`), fetch the card title using the Trello API.

### Method 1: Trello API (Preferred)

Use the Trello API to get the actual card title:

1. **Extract card ID** from URL: `https://trello.com/c/fr9huYEa/...` -> card ID is `fr9huYEa`
2. **Load Trello credentials** - Try accounts in order: `stuart.env`, `david.env`
3. **Call API**:
   ```
   GET https://api.trello.com/1/cards/{card_id}?fields=name&key={api_key}&token={token}
   ```
4. **Response** contains `name` field with the exact card title

**Important:** Handle encoding - the API may return garbled em dashes. Fix with:
```python
card_name = response.json()["name"]
card_name = card_name.replace('\ufffd', '\u2014')  # Replace replacement char with em dash
```

Example using Python:
```python
import requests
import os
from dotenv import load_dotenv

# Try accounts in order
for account in ["stuart", "david"]:
    try:
        load_dotenv(f"Degold/accounts/{account}.env")
        api_key = os.getenv("TRELLO_API_KEY")
        token = os.getenv("TRELLO_TOKEN")
        if not api_key or not token:
            continue

        response = requests.get(
            f"https://api.trello.com/1/cards/{card_id}",
            params={"fields": "name", "key": api_key, "token": token}
        )
        if response.status_code == 200:
            card_name = response.json()["name"]
            card_name = card_name.replace('\ufffd', '\u2014')
            break
    except:
        continue
```

### Method 2: URL Slug Fallback

If API fails, parse from URL slug:
1. Extract slug from URL path after `/c/[card-id]/`
2. Replace hyphens with spaces
3. Capitalize appropriately
4. Convert double-hyphens to em dashes

## Workflow

### Step 0: Check for Existing Facecam (ALWAYS DO FIRST)

Before generating, check if a facecam video already exists in the project's `facecam/` subfolder. This avoids redundant generation (which costs money and time).

```python
from pathlib import Path

project = Path(project_path)
facecam_dir = project / "facecam"

if facecam_dir.exists():
    videos = list(facecam_dir.glob("*.mp4"))
    if videos:
        latest = max(videos, key=lambda p: p.stat().st_mtime)
        size_mb = latest.stat().st_size / (1024 * 1024)
        print(f"Facecam already exists: {latest.name} ({size_mb:.1f} MB)")
        # Ask user: "Facecam already exists. Regenerate? (yes/no)"
```

**If a match is found:**
- Report the existing file to the user with name, date, and size
- Ask: "Facecam already exists. Regenerate it? (yes/no)"
- If no -> done
- If yes -> continue with `--force` flag

**If no match found:** Continue to Step 1 as normal.

### Step 1: Gather Information

If not all parameters provided, auto-detect or ask user for:

1. **Video Title** - Title for the generated video
   - If user provides a Trello URL, extract the title from the API
   - Otherwise auto-detected from project folder name
2. **Channel Code** - Auto-detected from folder path (RRU, JDRP, DSR, STU)
3. **Avatar Path** - Auto-selected from channel config
   - Download from Google Drive if not available locally (see Step 2b)
4. **Audio Path** - Auto-detected voiceover from project folder

**Trello URL Detection:** If the user provides a Trello URL (contains `trello.com/c/`), automatically:
1. Extract the card ID from the URL
2. Fetch the title using the Trello API
3. Present the parsed title to the user for confirmation

### Step 2: Cross-Reference with channels.py

Read `Degold/channels.py` and verify:
- Channel code exists in the config
- Get the `avatar_folder` ID for downloading the avatar if needed

### Step 2b: Download Avatar from Google Drive (if needed)

If the avatar image doesn't exist locally, download from Google Drive:

```python
import sys, os
from pathlib import Path
sys.path.insert(0, 'scripts')
from dotenv import load_dotenv
from gws_drive import GwsDriveContext, list_drive_folder_files, download_drive_file

load_dotenv('Degold/accounts/david.env')
ctx = GwsDriveContext(token=os.getenv('GOOGLE_WORKSPACE_CLI_TOKEN', ''))

# List avatar folder contents
avatar_folder_id = channel_config.avatar_folder
files = list_drive_folder_files(avatar_folder_id, context=ctx)

# Download avatar images
avatars_dir = Path(f"Degold/avatars/{channel_code}")
avatars_dir.mkdir(parents=True, exist_ok=True)
for f in files:
    if f['mimeType'].startswith('image/'):
        download_drive_file(f['id'], avatars_dir / f['name'], context=ctx)
```

**Do NOT use gdown** - it fails on Windows with long filenames containing special characters.

### Step 3: Dry Run (MANDATORY before spending money)

Before any generation, **always** verify inputs and show a cost plan:

1. Check audio has actual speech (not silence) using ffprobe volumedetect
2. Confirm max duration is 30s (intro mode default)
3. Show the user a summary:

```
WAN Facecam Generation — DRY RUN
==================================
Video Title: [title]
Channel: [channel code]
Avatar: [path]
Audio: [path] (mean volume: -XX.X dB — has speech)
Resolution: [480P/720P/1080P]
Max Duration: [seconds]s
Chunks: [N] × [chunk_dur]s
Estimated Time: ~[N] minutes
Estimated Cost: ~$[X.XX]

Type "yes" or "y" to confirm and generate.
```

**Cost estimation:**
- 480P: $0.50 per 10s chunk ($0.05/s)
- 720P: $1.00 per 10s chunk ($0.10/s)
- 1080P: $1.50 per 10s chunk ($0.15/s)
- Formula: `ceil(duration / chunk_duration) * cost_per_chunk`

Wait for user confirmation before proceeding.

### Step 3b: Verify Prerequisites

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

### Step 4: Run Generation

```bash
python scripts/auto_wan_facecam.py "<project_path>" \
  --resolution <resolution> \
  --max-duration <seconds> \
  --chunk-duration <chunk_seconds>
```

Or with overrides:
```bash
python scripts/auto_wan_facecam.py "<project_path>" \
  --title "Custom Title" \
  --channel DSR \
  --image "path/to/avatar.jpg" \
  --audio "path/to/voiceover.mp3" \
  --resolution 720P \
  --max-duration 60
```

**Options:**
- `--title "Custom Title"` - Override auto-detected title
- `--channel DSR` - Override auto-detected channel
- `--image "path.jpg"` - Override auto-detected avatar
- `--audio "path.mp3"` - Override auto-detected voiceover
- `--resolution 480P|720P|1080P` - Video resolution (default: 480P)
- `--max-duration 30` - Max audio seconds to process (default: 30)
- `--chunk-duration 10` - Seconds per video chunk, 5 or 10 (default: 10)
- `--no-trim` - Don't trim audio to max-duration
- `--force` - Regenerate even if facecam already exists
- `--region international|beijing` - API region (default: international)
- `--verbose` - Debug logging

Typical generation time: ~3-5 minutes per 10-second chunk.

### Step 5: Report Result

The script reports:
- Title, channel, chunks generated/failed
- Total duration, resolution, models used
- Output file path and size
- Any errors encountered

Report to the user:
- Output video path and file size
- Total duration generated
- Number of chunks (successful/failed)
- Resolution and model used
- Total estimated cost

### Step 5b: Generation Tracking (Automatic)

After successful generation, `auto_wan_facecam.py` automatically:
1. Saves a generation log to `<project>/facecam/generation_log.json` (mode, cost, duration, timestamps)
2. Updates the global budget tracker at `Stu/facecam_budget.json` (same tracker used by batch mode)

This means both intro and batch modes share the same budget/spend tracking.

### Step 6: Verify Output

```bash
ffprobe -v error -show_entries stream=codec_name,duration,width,height -of default=noprint_wrappers=1 "<output_path>"
```

## Billable vs Free Calls

Per [Alibaba Cloud Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing):

| Operation | Cost | Notes |
|-----------|------|-------|
| Ollama classification | **Free** | Local LLM, cached after first run |
| Whisper alignment | **Free** | Local faster-whisper, no API |
| `--dry-run` | **Free** | Classification + planning only |
| Silence check (ffmpeg volumedetect) | **Free** | Local ffprobe |
| Task submission (`async_call`) | **Free** | No charge at submission time |
| Task polling (`wait`/`fetch`) | **Free** | Status queries are not billed |
| OSS file upload (SDK auto-upload of image/audio) | **Free** | Uses DashScope's OSS, not yours |
| Prompt rewriting (`extend_prompt`) | **Free** | Bundled into generation cost (we disable it anyway) |
| **Successful video generation** | **Billable** | Charged per second of output video duration |
| Failed generation tasks | **Free** | Explicitly: "Failed requests do not incur charges" |
| Cancelled tasks (still PENDING) | **Free** | Can cancel cleanly before processing starts |
| Cancelled tasks (already RUNNING) | **Maybe** | May be billed for partial processing |

**Only one outcome costs money:** a successfully generated video. Billing unit is **price/second x video duration (seconds)**.

### Official list prices (wan2.5-i2v-preview, international)

| Resolution | Per second | Per 10s chunk |
|------------|-----------|---------------|
| 480P | $0.05/s | $0.50 |
| 720P | $0.10/s | $1.00 |
| 1080P | $0.15/s | $1.50 |

**Note:** Code constants were corrected to match these official rates after cross-referencing the Mar 2026 invoice ($8.00 pretax for 172.3s of generated video).

## Generation Rules (Hard Constraints)

1. **Max chunk/group duration: 10 seconds** — longer segments cause first-frame freeze (avatar pauses at default face while talking). Groups >10s are automatically skipped by batch mode.
2. **Silence check before every generation** — audio with mean volume below -40 dBFS is silence. Don't generate video of a motionless avatar. Both `wan_facecam.py` (chunk level) and `facecam_queue_gen.py` (group level) enforce this.
3. **Intro mode: 30s max** — always generates from the first 30 seconds of voiceover only.
4. **Dry run first** — always show a dry run plan before spending money. No exceptions.
5. **Full sentences** — chunks should align to sentence boundaries. If a sentence spans >10s, skip it.

## Error Handling

- If channel code not found in channels.py, warn the user (avatar must be provided manually)
- If avatar or audio files don't exist, report the error with paths searched
- If DASHSCOPE_API_KEY is not set, instruct user to export it
- If dashscope package is missing, install it
- If generation fails on primary model (wan2.5), auto-fallback to wan2.2 (silent video)
- If both models fail, report the error and suggest checking API key/quota/region

## Model Details

| Model | Audio Support | Duration | Notes |
|-------|--------------|----------|-------|
| `wan2.5-i2v-preview` | Yes (lip-sync) | 5 or 10s | Primary - best for facecam |
| `wan2.2-i2v-plus` | No (silent) | 5s fixed | Fallback - more stable |

The script automatically falls back to wan2.2 if wan2.5 fails.

## Pricing

Per [Alibaba Cloud Model Studio pricing](https://www.alibabacloud.com/help/en/model-studio/model-pricing), verified against Mar 2026 invoice ($8.00 for 172.3s = $0.046/s):

| Resolution | Cost/second | Cost per 10s chunk | 1 minute | 2 minutes |
|-----------|------------|-------------------|----------|-----------|
| 480P | $0.05/s | $0.50 | $3.00 | $6.00 |
| 720P | $0.10/s | $1.00 | $6.00 | $12.00 |
| 1080P | $0.15/s | $1.50 | $9.00 | $18.00 |

## Key Learnings

### Auto-Detection from Project Path
- Channel auto-detected from folder name (DeepSeaReports -> DSR, Stu -> STU)
- Avatar automatically selected from channel config
- Voiceover auto-detected from project's `voiceover/` subfolder
- Title extracted from folder name or SRT content
- Output saved to project's `facecam/` subfolder

### DSR Avatar: Use Harold_V1 Only
- DSR lipsync/facecam jobs fail with Harold_V2-V6 avatars
- **Always use `Harold_V1.jpg`** for DSR until this is resolved
- RRU (RennActor.jpg) and STU (Ethan.jpeg) work fine

### STU Avatar Location
- STU avatars are in `Stu/avatars/` (not `Degold/avatars/STU/`)
- The auto script uses `Stu/avatars/Ethan.jpeg` directly for STU channel
- Full path: `D:\_Projects\voiceover-matcher-dev\Stu\avatars\Ethan.jpeg`

### Voiceover Preference
- **ALWAYS prefer `voiceover_trimmed.mp3`** over `voiceover.mp3` when available
- The trimmed version removes silence/dead air and produces better lip-sync results

### Existing Facecam Check
- Always check `<project>/facecam/` for existing videos before generating
- Generation costs money (~$0.25-0.75 per 5s chunk) so avoid unnecessary regeneration
- Use `--force` to explicitly regenerate

### Audio Trimming
- The script auto-trims audio to `--max-duration` (default 120s) using ffmpeg
- Short final chunks (< 3s) are padded with silence to meet API minimum
- Use `--no-trim` to skip trimming

### Output Naming
- The final concatenated video is renamed to `{sanitized_title}_facecam.mp4`
- Individual chunks are saved as `facecam_000.mp4`, `facecam_001.mp4`, etc.

### Downloading Avatars: Use gws_drive (NOT gdown)
- **gdown fails on Windows** with long filenames containing special characters
- Always use `gws_drive` from `scripts/gws_drive.py` for Drive downloads
- Load credentials from `Degold/accounts/david.env` -> `GOOGLE_WORKSPACE_CLI_TOKEN`

### Trello API Credentials
- The `stuart.env` account may not have access to all boards
- The `david.env` account has broader access - try it if stuart fails
- Card IDs are the short code in the URL: `trello.com/c/Fr9huYEa/...` -> `Fr9huYEa`

## Troubleshooting

- **"DASHSCOPE_API_KEY not set"** - Export the key: `export DASHSCOPE_API_KEY=sk-xxx`
- **"No video formats found"** - Check API key validity and quota
- **Task timeout** - WAN generation can take 3-10 minutes per chunk; the SDK polls automatically
- **"Both models failed"** - Check API key validity, quota, and region setting
- **Audio too short** - API requires minimum 3s audio; the script auto-pads short final chunks
- **Avatar not found** - Check channel avatar config; download from Drive if missing locally
- **Channel not detected** - Provide `--channel` flag or `--image` for direct avatar path

## Example Sessions

### Example 0: Facecam Already Exists

```
User: generate facecam for E:/Edit Job/Degold/DeepSeaReports/USS Charlotte...

Assistant: [Checks project facecam/ folder...]

Found existing facecam:
  Name: USS_Charlotte_TORPEDOED_facecam.mp4
  Size: 12.3 MB, Date: 2026-03-20

Facecam already exists. Regenerate it? (yes/no)

User: no

Assistant: OK, using existing facecam at:
  E:/Edit Job/.../facecam/USS_Charlotte_TORPEDOED_facecam.mp4
```

### Example 1: Fully Automated

```
User: generate facecam for E:/Edit Job/Degold/DeepSeaReports/3dWWwtJc-Project

Assistant: [Checks facecam/ - none found]

[INFO] Detected channel: DSR (from folder 'DeepSeaReports')
[INFO] Found voiceover: .../voiceover/voiceover_trimmed.mp3
[INFO] Avatar: Degold/avatars/DSR/Harold_V1.jpg
[INFO] Title: How USS Charlotte Sank an Iranian Warship

WAN Facecam Generation
==================================
Video Title: How USS Charlotte Sank an Iranian Warship
Channel: DSR
Avatar: Degold/avatars/DSR/Harold_V1.jpg
Audio: .../voiceover/voiceover_trimmed.mp3
Resolution: 480P
Max Duration: 120s
Estimated Time: ~48 minutes
Estimated Cost: ~$1.50

Type "yes" or "y" to confirm and generate.

User: yes

Assistant: [Runs auto_wan_facecam.py...]

Facecam Generation Complete
==================================
  Chunks generated: 12
  Chunks failed:    0
  Total duration:   120.0s
  Resolution:       480P
  Output:           .../facecam/How_USS_Charlotte_Sank_facecam.mp4 (15.2 MB)
```

### Example 2: With Trello URL

```
User: generate facecam for this card: https://trello.com/c/fr9huYEa/103-...
       project at E:/Edit Job/Degold/RennReports/fr9huYEa-Project

Assistant: Parsed Title: Underwater Drone Entered the Last Known Position of Flight MH370

WAN Facecam Generation
==================================
Video Title: Underwater Drone Entered the Last Known Position of Flight MH370
Channel: RRU (auto-detected)
Avatar: Degold/avatars/RRU/RennActor.jpg
Audio: .../voiceover/voiceover_trimmed.mp3
Resolution: 480P | Est. Cost: ~$1.50

User: yes

Assistant: Done! Output: .../facecam/Underwater_Drone_Entered_facecam.mp4 (14.8 MB)
```

### Example 3: Manual Override

```
User: generate 720P facecam, image is Stu/avatars/Ethan.jpeg
      project E:/Edit Job/Stu/Stu/abc123-Project

Assistant:
WAN Facecam Generation
==================================
Video Title: Project (auto-detected)
Channel: STU | Avatar: Stu/avatars/Ethan.jpeg
Resolution: 720P | Est. Cost: ~$3.00

User: yes

Assistant: [Runs auto_wan_facecam.py --image ... --resolution 720P]
Done! Output: .../facecam/Project_facecam.mp4 (28.5 MB)
```

---

## Mode: `batch` — Batch Queue Generation

Generate facecam segments for multiple completed Stu pipeline projects at once.
Uses Ollama (free, local) to classify "talking point" segments, merges adjacent
segments for cost efficiency, and enforces a hard $8 USD global budget.

### Trigger Phrases

- "facecam queue"
- "batch facecam"
- "generate facecams for completed"
- "facecam for completed projects"
- "queue facecam generation"

### Prerequisites

- Ollama running locally with `llama3.2` model (`ollama pull llama3.2`)
- `DASHSCOPE_API_KEY` set in environment (load from project root `.env`)
- Completed projects in Stu queue with voiceover SRT files

### Resolving Card IDs from Project Paths

When the user specifies a project path instead of a card ID for batch mode, look up the card ID in `Stu/pipeline_queue_state.json`:
- Search `pipelines` entries for matching `local_project_dirs`
- Or match the title text from the folder name
- Pipeline keys are **lowercase** but `card_id` values are mixed-case — use the `card_id` value for `--card-ids`

### Workflow

#### Step 1: Load DASHSCOPE_API_KEY

```python
from dotenv import load_dotenv
load_dotenv('.env')  # Project root .env — NOT Stu/accounts/david.env
```

Or check it's already set:
```bash
python -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; key = os.environ.get('DASHSCOPE_API_KEY', ''); print(f'API key: {\"set (\" + key[:8] + \"...)\" if key else \"NOT SET\"}')"
```

#### Step 2: Dry Run (Free)

Always start with a dry run to show the plan:

```bash
python scripts/facecam_queue_gen.py --dry-run
```

This displays completed projects, Ollama segment classification, merged groups,
cost estimates, and budget status. No money is spent.

Present the output to the user:

```
Batch Facecam Queue Generator
============================================================
  Resolution: 480P
  Budget: $8.00 total | $0.00 spent | $8.00 remaining
  Facecam target: 5% of project runtime
  Completed projects: 2
  Mode: DRY RUN (no generation)

  [KxQ0SoGh] THE $36 TRILLION BOMB...
    VO duration: 4m05s | Segments: 48 | Talking points: 12/48
    Merged groups: 3 | Already generated: 0 | Target: 12s (5%)
    Selected: 2 groups | Duration: 11s | Est. cost: $1.00
      - seg 5-7 (8s, conf=0.92, $0.50)
      - seg 30-31 (3s, conf=0.88, $0.50)

  [ujKR12Dg] THE ESCAPE PLAN...
    VO duration: 5m10s | Segments: 62 | Talking points: 15/62
    Merged groups: 4 | Already generated: 0 | Target: 15s (5%)
    Selected: 3 groups | Duration: 14s | Est. cost: $1.00

Plan Summary
============================================================
  Projects to generate: 2
  Total groups: 5
  Estimated cost: $2.00
  Budget remaining after: $6.00
```

Ask the user: "Generate facecams for all projects, or specify which ones?"

#### Step 3: Confirmation Table (MANDATORY before spending money)

After the dry run, and before any real API calls, **you MUST present a detailed confirmation table** to the user. This is the final gate before money is spent.

For each selected project, read the actual SRT segment text and build a table:

1. Parse the SRT file to get the text for each selected segment group
2. Check avatar exists and note its path/size
3. Check voiceover file exists
4. Present in this exact format:

```
**Facecam Generation Plan — {card_id}**

| Detail | Value |
|--------|-------|
| Project | {title} ({duration}) |
| Channel | {channel} |
| Avatar | {avatar_path} ({size} KB) |
| Audio source | {voiceover_path} |
| Resolution | 480P |
| Model | wan2.5-i2v-preview (lip-sync) |

**{N} segments to generate:**

| Group | Time | Duration | Chunks | Cost | Text |
|-------|------|----------|--------|------|------|
| seg 5-7 | 0:15-0:23 | 8s | 2 | $0.50 | *"actual text from SRT..."* |
| seg 30-31 | 1:12-1:18 | 6s | 2 | $0.50 | *"actual text from SRT..."* |

**Totals:**
- Duration: **14 seconds** of facecam
- Chunks: **4 total**
- Cost: **$1.00 USD**
- Gen time: ~12-20 minutes (3-5 min per chunk)
- Budget after: **$7.00 remaining**
```

**How to get the segment text:**
```python
import re, glob
from pathlib import Path

# Find SRT via glob (handles special chars in paths)
matches = glob.glob(f'{project_dir}/**/voiceover.srt', recursive=True)
if not matches:
    matches = glob.glob(f'{project_dir}/*.srt')
srt_path = Path(matches[0])
text = srt_path.read_text(encoding='utf-8-sig', errors='replace')
blocks = re.split(r'\n\s*\n', text.strip())

# Build index -> (timestamp, text) lookup
for block in blocks:
    lines = block.strip().split('\n')
    if len(lines) < 3: continue
    try: idx = int(lines[0].strip())
    except ValueError: continue
    ts = lines[1].strip()
    seg_text = ' '.join(l.strip() for l in lines[2:] if l.strip())
    # Print segments matching the selected group indices
```

**Wait for explicit user confirmation** ("yes", "proceed", "go ahead") before running the actual generation. Do NOT proceed without it.

#### Step 4: Generate (after confirmation)

```bash
# All completed projects
python scripts/facecam_queue_gen.py

# Specific projects only
python scripts/facecam_queue_gen.py --card-ids KxQ0SoGh

# With budget override
python scripts/facecam_queue_gen.py --budget 5.0

# Force regenerate (ignores generation log)
python scripts/facecam_queue_gen.py --force
```

**Important:** Load the API key before running:
```bash
export DASHSCOPE_API_KEY=$(python -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; print(os.environ.get('DASHSCOPE_API_KEY',''))")
```

#### Step 5: Report Results

The script reports per-project and total results. Present to user:

```
Generation Complete
============================================================
  Segments generated: 5
  Total cost: $2.00
  Budget spent: $2.00 / $8.00
  Budget remaining: $6.00
```

### Key Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dry-run` | off | Classify and plan, no generation (free) |
| `--card-ids ID...` | all | Process specific card IDs only |
| `--budget N` | 8.0 | Total budget in USD |
| `--facecam-percent N` | 0.05 | Target % of project duration |
| `--resolution` | 480P | 480P, 720P, or 1080P |
| `--force` | off | Regenerate even if in log |
| `--max-gap N` | 2.0 | Max gap (s) for merging adjacent segments |
| `--skip-whisper` | off | Skip Whisper alignment (use raw SRT timestamps) |

### Budget Enforcement

- **Hard cap**: The script stops before any generation that would exceed budget
- **Crash-safe**: Budget tracker (`Stu/facecam_budget.json`) saved after each generation
- **Re-run safe**: Generation log (`<project>/facecam/generation_log.json`) prevents duplicate spending
- **Global**: Budget shared across ALL projects, not per-project
- To reset budget: delete `Stu/facecam_budget.json`

### Already-Done Detection (Two Layers)

Re-runs never regenerate existing segments. Detection works at two levels:

1. **Planning layer** (`get_already_generated_indices`): collects indices from BOTH `generation_log.json` AND `facecam_seg*.mp4` files on disk. Groups where all indices are already done are excluded before selection.
2. **Generation layer**: immediately before each API call, checks if `facecam_{label}.mp4` exists with non-zero size. Skips with `[SKIP]` if so.

Use `--force` to bypass both layers and regenerate everything.

### How Segments Are Selected

1. SRT parsed into voiceover segments
2. Ollama classifies each as "talking point" or not (free, cached)
3. Adjacent talking points merged into groups (saves chunks = saves money)
4. Groups ranked by confidence score
5. Best groups selected until 5% duration cap OR budget cap reached (whichever is smaller)
6. **Whisper alignment**: Selected groups re-transcribed with faster-whisper (`word_timestamps=True`) to get precise speech boundaries — SRT timestamps can drift by 0.5-2s, Whisper corrects this for accurate lip-sync. Use `--skip-whisper` to bypass.

### Output Files

Per project in `<project>/facecam/`:
- `facecam_seg005-007.mp4` - Facecam video for merged segments 5-7
- `facecam_seg012.mp4` - Facecam video for single segment 12
- `generation_log.json` - Tracks what was generated, cost, timestamps

Global in `Stu/`:
- `facecam_budget.json` - Total spend tracking across all projects
- `facecam_billing.jsonl` - Append-only ledger of every DashScope API call (billable and non-billable)

**Note:** Both modes output to `<project>/facecam/` and write to `generation_log.json` there. Both modes update the global `Stu/facecam_budget.json` tracker. Intro mode entries use `"mode": "intro"`, batch mode entries use segment-level tracking.

### Example Session: Batch Queue

```
User: generate facecams for completed projects


Assistant: [Loads DASHSCOPE_API_KEY, runs dry-run first...]

Batch Facecam Queue Generator
============================================================
  Resolution: 480P
  Budget: $8.00 total | $0.00 spent | $8.00 remaining
  Completed projects: 2

  [KxQ0SoGh] THE $36 TRILLION BOMB...
    VO duration: 4m05s | Talking points: 12/48 | Est. cost: $1.00
  [ujKR12Dg] THE ESCAPE PLAN...
    VO duration: 5m10s | Talking points: 15/62 | Est. cost: $1.00

Total estimated cost: $2.00 | Budget remaining: $6.00

Generate facecams for all projects, or specify which ones?

User: all

Assistant: [Runs facecam_queue_gen.py...]

Generation Complete
============================================================
  Segments generated: 5
  Total cost: $2.00
  Budget remaining: $6.00

Generated files:
  KxQ0SoGh: facecam_seg005-007.mp4, facecam_seg030-031.mp4
  ujKR12Dg: facecam_seg003-005.mp4, facecam_seg018.mp4, facecam_seg042-044.mp4
```
