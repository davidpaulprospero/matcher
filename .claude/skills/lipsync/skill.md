# Lipsync Submit Skill

## Two Workflows Available

### 1. Fully Automated (Recommended - Minimal Intervention)

Use `Degold/auto_lipsync.py` for the simplest workflow. Just provide the project path:

```bash
python Degold/auto_lipsync.py "E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-Project..."
```

The script automatically:
- **Detects channel** from folder name (DeepSeaReports → DSR, RennReports → RRU)
- **Finds voiceover** file in project folder (voiceover/ subfolder)
- **Trims to 1 minute** automatically (Degold limit)
- **Rotates avatars** using tracker (V1-V6 for DSR)
- **Extracts title** from SRT or folder name

Options:
- `--title "Custom Title"` - Override auto-detected title
- `--channel DSR` - Override auto-detected channel
- `--no-trim` - Skip 1-minute trimming

### 2. Playwright Browser (Manual/Fallback)

Use Playwright MCP when browser needs to stay open for processing, or for custom workflows.

## Trigger Phrases

- "submit lipsync"
- "lipsync job"
- "submit to degold"
- "generate lipsync"
- "lipsync auto"
- "auto lipsync"

## Auto-Detection

### Channel Detection from Folder Name

The auto_lipsync.py script detects channel from project path:

| Folder Pattern | Channel |
|---------------|---------|
| DeepSeaReports | DSR |
| RennReports | RRU |
| JournalOfDrunkPeople | JDRP |
| Contains "DSR" | DSR |
| Contains "RRU" | RRU |

### Voiceover Detection

Looks in order:
1. `<project>/voiceover/` subfolder
2. `<project>/` root

File extensions: .mp3, .wav, .m4a, .mp4

Prefers files with "voiceover" in name, or matching project name.

## Parameters

- `$VIDEO_TITLE` - Video title (optional, will prompt if not provided)
- `$CHANNEL_CODE` - Channel code like RRU, JDRP, DSR (optional)
- `$AVATAR_PATH` - Path to avatar image (optional)
- `$AUDIO_FILES` - Audio segment files (optional)
- `$TRELLO_URL` - Trello card URL (optional, will extract title automatically)

## Parsing Titles from Trello URLs

If a Trello URL is provided (e.g., `https://trello.com/c/fr9huYEa/103-1-minute-ago-underwater-drone...`), fetch the card title using the Trello API.

### Method 1: Trello API (Preferred)

Use the Trello API to get the actual card title:

1. **Extract card ID** from URL: `https://trello.com/c/fr9huYEa/...` → card ID is `fr9huYEa`
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

# Try accounts in order (stuart usually has access)
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
            card_name = card_name.replace('\ufffd', '\u2014')  # Fix encoding
            break
    except:
        continue
```

### Method 2: URL Slug Fallback

If API fails, parse from URL slug:
1. Extract slug from URL path after `/c/[card-id]/`
2. Replace hyphens with spaces
3. Capitalize appropriately
4. Convert double-hyphens to em dashes (`—`)

## Workflow

### Step 1: Gather Information

If not all parameters provided, ask user for:

1. **Video Title** - Title for the generated video (e.g., "EP42 - Trabant Secrets")
   - If user provides a Trello URL, extract the title from the API using the rules in "Parsing Titles from Trello URLs"
2. **Channel Code** - Channel to use (RRU, JDRP, DSR, etc.)
3. **Avatar Path** - Path to avatar image file (optional - will download from Google Drive if not provided)
4. **Audio Files** - List of audio segment files in order

**Trello URL Detection:** If the user provides a Trello URL (contains `trello.com/c/`), automatically:
1. Extract the card ID from the URL
2. Fetch the title using the Trello API
3. Present the parsed title to the user for confirmation

**Auto-Download Avatar:** If no avatar path is provided:
1. Get the `avatar_folder` ID from `Degold/channels.py` for the channel
2. Download the avatar from Google Drive using gdown
3. Save to a local path (e.g., `Degold/avatars/{channel}_avatar.png`)
4. Use the local path for the form submission

### Step 2: Cross-Reference with CHANNELS.md or channels.py

Read `Degold/channels.py` and verify:
- Channel code exists in the config
- Get the drive_folder ID for the output
- Get the avatar_folder ID for downloading the avatar

### Step 2b: Download Avatar from Google Drive (if needed)

If no avatar path is provided, download the avatar from Google Drive:

```python
import gdown
from pathlib import Path

# Get avatar folder from channel config
avatar_folder_id = "16cHw8fgefC89zelexv_OSQNoqzhKekwO"  # RRU example

# Create avatars directory
avatars_dir = Path("Degold/avatars")
avatars_dir.mkdir(parents=True, exist_ok=True)

# Download from Drive folder
output_dir = avatars_dir / "temp_avatar"
output_dir.mkdir(exist_ok=True)

print(f"Downloading avatar from Google Drive folder: {avatar_folder_id}")

# Download the folder (gdown will get all files)
gdown.download_folder(
    f"https://drive.google.com/drive/folders/{avatar_folder_id}",
    output=str(output_dir),
    quiet=False
)

# Find the image file (png, jpg, jpeg)
avatar_files = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg"))
if avatar_files:
    avatar_path = avatar_files[0]  # Use first image found
    print(f"Avatar downloaded: {avatar_path}")
else:
    print("[ERROR] No avatar image found in Google Drive folder")
```

Or via command line:
```bash
gdown --folder "https://drive.google.com/drive/folders/16cHw8fgefC89zelexv_OSQNoqzhKekwO" -O Degold/avatars/temp/
```

**Note:** The Google Drive folder may contain multiple avatars.

#### Avatar Rotation Tracking

For channels with multiple avatars (like DSR with V1-V6), use the avatar tracker to cycle through them randomly while ensuring each is used once before repeating.

**Python helper:**
```python
import json
import random
from pathlib import Path

TRACKER_FILE = Path("Degold/avatars/avatar_usage.json")

def get_next_avatar(channel: str) -> str:
    """Get next avatar for channel, cycling randomly but using all once before repeating."""
    with open(TRACKER_FILE, 'r') as f:
        tracker = json.load(f)

    if channel not in tracker:
        raise ValueError(f"Channel {channel} not in tracker")

    data = tracker[channel]
    avatars = data['avatars']
    used_order = data.get('used_order', [])
    current_index = data.get('current_index', 0)

    # If we've used all avatars, shuffle and reset
    if current_index >= len(avatars):
        random.shuffle(avatars)
        used_order = []
        current_index = 0
        data['avatars'] = avatars

    # Get next avatar
    avatar = avatars[current_index]
    used_order.append(avatar)
    current_index += 1

    # Update tracker
    data['used_order'] = used_order
    data['current_index'] = current_index

    with open(TRACKER_FILE, 'w') as f:
        json.dump(tracker, f, indent=2)

    return avatar
```

**Usage:**
```python
avatar_file = get_next_avatar("DSR")
# Returns: "Harold_V5.jpg" (random, then cycles through all 6 before repeating)

avatar_path = f"Degold/avatars/DSR/{avatar_file}"
```

**Tracker file:** `Degold/avatars/avatar_usage.json`

### Step 3: Confirm with User

Show the user a summary:

```
Lipsync Job Submission (Playwright)
==================================
Video Title: [title]
Channel: [channel code]
  - Drive Folder: [folder_id from channels.py]
  - Avatar Folder: [avatar_folder from channels.py]
Avatar: [path]
Audio Files: [list]
  -> Each file will be trimmed to 1 minute before upload

Type "yes" or "y" to confirm and submit.
```

Wait for user confirmation before proceeding.

### Step 3b: Auto-Trim Audio to 1 Minute (ALWAYS REQUIRED!)

**The Degold form only accepts audio up to 1 minute!** This is NOT optional - ALL audio files must be trimmed to the first 60 seconds before upload.

For each audio file:
1. Check if a `_1min` version already exists
2. If not, run ffmpeg to trim to 60 seconds
3. Use the trimmed file for upload

```python
from pathlib import Path

def trim_audio_to_1min(audio_path: str) -> str:
    """Trim audio to first 60 seconds using ffmpeg."""
    audio_path_obj = Path(audio_path)
    suffix = audio_path_obj.suffix
    trimmed_path = audio_path_obj.parent / f"{audio_path_obj.stem}_1min{suffix}"

    # Skip if already trimmed
    if trimmed_path.exists():
        print(f"  Using existing trimmed file: {trimmed_path}")
        return str(trimmed_path)

    print(f"  Trimming {audio_path} to 1 minute...")

    result = subprocess.run(
        ["ffmpeg", "-i", audio_path, "-t", "60", "-c", "copy", str(trimmed_path), "-y"],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode == 0 and trimmed_path.exists():
        print(f"  [OK] Saved trimmed audio: {trimmed_path}")
        return str(trimmed_path)
    else:
        print(f"  [WARN] Trim failed, using original: {result.stderr[:200] if result.stderr else ''}")
        return audio_path
```

Or via command line:
```bash
ffmpeg -i "input.mp3" -t 60 -c copy "input_1min.mp3" -y
```

**Do NOT ask the user about trimming - just do it automatically.**

### Step 4: Run Playwright Automation

**IMPORTANT: Use Playwright MCP tools (not Python playwright)!** The MCP browser stays open in the Claude Code session, allowing processing to complete. Python playwright scripts close when the script ends, which aborts processing.

Use the Playwright MCP tools to automate the form:

**Form URL:** https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c

**Step 4a: Navigate to the form**
```
mcp__plugin_playwright_playwright__browser_navigate
url: "https://degoldmedia.duckdns.org/form/a82405ee-4d7f-4ff6-9142-cd97c909897c"
```

**Step 4b: Wait for form to load**
```
mcp__plugin_playwright_playwright__browser_wait_for
text: "Video Title"
time: 5
```

**Step 4c: Fill in the form fields**
```
mcp__plugin_playwright_playwright__browser_fill_form
fields: [
  {"name": "Video Title", "type": "textbox", "ref": "<field-0 ref>", "value": video_title},
  {"name": "Channel Code", "type": "combobox", "ref": "<field-1 ref>", "value": channel_code},
  {"name": "Drive Folder ID", "type": "textbox", "ref": "<field-4 ref>", "value": drive_folder_id}
]
```

**Step 4d: Upload avatar image**

Use `browser_run_code` with `setInputFiles()` (browser_file_upload has a bug):

```
mcp__plugin_playwright_playwright__browser_run_code
code: |
  async (page) => {
    await page.setInputFiles('input[name="field-2"]', 'D:/_Projects/.../avatar.jpg');
    return 'Avatar uploaded';
  }
```

**Step 4e: Upload audio files**

Same approach - use `browser_run_code`:

```
mcp__plugin_playwright_playwright__browser_run_code
code: |
  async (page) => {
    await page.setInputFiles('input[name="field-3"]', 'D:/_Projects/.../audio.mp3');
    return 'Audio uploaded';
  }
```

**Step 4f: Take screenshot to verify**
```
mcp__plugin_playwright_playwright__browser_take_screenshot
filename: "lipsync-form-filled.png"
type: "png"
```

**Step 4g: Click submit button**

Find and click the submit button.

**Step 4h: Wait for submission confirmation**

The form does NOT show a "success" message. Instead, submission is confirmed when:
- The Submit button becomes [disabled]
- Processing happens in the background

**Do NOT refresh or navigate away after clicking submit!** Just leave the browser open and inform the user the job has been submitted.

```
# After clicking submit, check if button is disabled
# If disabled, submission was successful
mcp__plugin_playwright_playwright__browser_snapshot
# Look for: button "Submit" [disabled]
```

If the button is disabled, the job has been submitted. Take a final screenshot and report success.

### Step 5: Report Result

**CRITICAL: The browser MUST remain open for processing to complete!**

The n8n backend requires the browser to remain open during processing. If you close the browser before processing completes, the job will NOT be processed.

After submission:
1. Take a screenshot showing the Submit button is disabled
2. Tell the user: "Job submitted. KEEP THE BROWSER OPEN - processing is happening in the background"
3. DO NOT close the browser - wait for the user to manually close it when they're ready
4. The user should check the Google Drive folder for results after a few minutes

## Error Handling

- If channel code not found in channels.py, warn the user
- If avatar or audio files don't exist, report the error
- If form doesn't load, try refreshing and wait longer
- If submission fails, show screenshot of error and suggest checking manually

## Key Learnings

### Auto-Detection Improvements (2025-03)
- Channel auto-detected from folder name (DeepSeaReports → DSR)
- Avatar automatically rotated through V1-V6 using `avatar_usage.json` tracker
- Audio auto-trimmed to 1 minute (required by Degold limit)
- Title extracted from SRT or folder name if not provided
- Files copied to project directory for API/MCP access

### Files Must Be in Project Directory
- **API approach**: Files can be anywhere, script copies to temp location
- **Playwright MCP**: Can only access files within project directory (`D:\_Projects\voiceover-matcher-stable`)
- Copy files from external drives (E:\) to project dir before upload

### Trello API Credentials
- The `stuart.env` account may not have access to all boards (returns "unauthorized card permission requested")
- The `david.env` account has broader access - try it if stuart fails
- Card IDs are the short code in the URL: `trello.com/c/Fr9huYEa/...` → `Fr9huYEa`

### File Upload Path Limitation
- Playwright MCP can only access files within the project directory (`D:\_Projects\voiceover-matcher-subtitle`)
- If audio/avatar files are on external drives (e.g., `E:\...`), copy them to the project directory first:
  ```bash
  powershell -Command "Copy-Item -Path 'E:\path\to\file.mp3' -Destination 'D:\_Projects\voiceover-matcher-subtitle\Degold\avatars\'"
  ```
- Use forward slashes for paths in `browser_file_upload`: `D:/_Projects/...`

### Submission Confirmation
- The form does NOT display a "success" message
- Submission is confirmed when the Submit button becomes disabled
- **CRITICAL: The browser MUST remain open for processing to complete!**
- If you close the browser after clicking Submit but before processing finishes, the job will NOT be processed
- The n8n backend streams the upload and processes it while the browser is open
- Check the Google Drive folder for results after a few minutes

### Python Playwright vs MCP Browser
- **Python Playwright script**: Browser closes when script ends (even with `input()` - doesn't work in Claude Code)
- **Playwright MCP browser**: Stays open as part of Claude Code session until you explicitly close it or session ends
- **Always use MCP browser** for lipsync submission to ensure browser stays open for processing
- The MCP browser is the recommended approach - it stays alive for the duration of the conversation

### File Upload in MCP Browser
- The `browser_file_upload` tool has a bug that rejects valid JSON arrays
- Use `browser_run_code` with `page.setInputFiles()` instead:
  ```javascript
  await page.setInputFiles('input[name="field-2"]', 'D:/_Projects/.../avatar.jpg');
  await page.setInputFiles('input[name="field-3"]', 'D:/_Projects/.../audio.mp3');
  ```

## Playwright MCP Tool Reference

| Tool | Purpose |
|------|---------|
| `browser_navigate` | Go to form URL |
| `browser_snapshot` | Get current page state |
| `browser_fill_form` | Fill text fields and dropdowns |
| `browser_run_code` | Upload files using `page.setInputFiles()` (preferred) |
| `browser_click` | Click buttons |
| `browser_take_screenshot` | Capture current view |
| `browser_wait_for` | Wait for text/element |
| `browser_evaluate` | Run JS if needed |

## Example Sessions

### Example 1: Fully Automated (Minimal Input)

```
User: submit lipsync for E:\Edit Job\Degold\DeepSeaReports\3dWWwtJc-Project

Assistant: Running auto lipsync submitter...

[INFO] Project: 3dWWwtJc-Project
[INFO] Detected channel: DSR (from folder 'DeepSeaReports')
[INFO] Found voiceover: .../voiceover/3dWWwtJc-Voiceover.mp3
[INFO] Trimming audio to 1 minute...
[OK] Created trimmed audio
[INFO] Avatar: DSR/Harold_V3.jpg (auto-rotated)
[INFO] Title: How USS Charlotte Sank an Iranian Warship (from SRT)

[INFO] Submitting job...
[OK] Job submitted successfully!

Result: Done! Check Google Drive folder for output.
```

### Example 2: With Trello URL

```
User: submit lipsync for this trello card: https://trello.com/c/fr9huYEa/103-1-minute-ago-underwater-drone-entered-the-last-known-position-of-flight-mh370-the-structure-found-on-the-ocean-floor-is-not-natu

Assistant: I'll extract the title from that Trello URL and prepare the lipsync job.

Parsed Title: 1 MINUTE AGO: Underwater Drone Entered the Last Known Position of Flight MH370 — The Structure Found on the Ocean Floor Is Not Natural

Now I need:
1. Channel code (RRU, JDRP, DSR, etc.)
2. Avatar path
3. Audio files

[User provides: RRU, Degold/avatars/face.png, audio/ep103_01.mp3, audio/ep103_02.mp3]

Lipsync Job Submission (Playwright)
==================================
Video Title: 1 MINUTE AGO: Underwater Drone Entered the Last Known Position of Flight MH370 — The Structure Found on the Ocean Floor Is Not Natural
Channel: RRU
  - Drive Folder: 1XJY8HUEWvFH0cI68tvyrPEyU7bTpeNLQ
  - Avatar Folder: 16cHw8fgefC89zelexv_OSQNoqzhKekwO
Avatar: Degold/avatars/face.png
Audio Files: audio/ep103_01.mp3, audio/ep103_02.mp3
  -> Each file will be trimmed to 1 minute before upload

Type "yes" or "y" to confirm and submit.

User: yes

[Assistant runs Playwright automation...]

Result: Job submitted successfully! KEEP THE BROWSER OPEN - processing is happening in the background.
```

### Example 2: Manual Title

```
User: submit lipsync job for EP42 with RRU channel

Assistant: I'll help you submit a lipsync job. I need a few details:

1. Avatar image path (e.g., "Degold/avatars/face.png"):
   - Provide path

2. Audio files (list in order):
   - Provide paths

[User provides: Degold/avatars/face.png, audio/01.mp3, audio/02.mp3]

Assistant: Let me verify the channel and then confirm:

Lipsync Job Submission (Playwright)
==================================
Video Title: EP42
Channel: RRU
  - Drive Folder: 1XJY8HUEWvFH0cI68tvyrPEyU7bTpeNLQ
  - Avatar Folder: 16cHw8fgefC89zelexv_OSQNoqzhKekwO
Avatar: Degold/avatars/face.png
Audio: audio/01.mp3, audio/02.mp3

Type "yes" to confirm and submit.

User: yes

[Assistant runs Playwright automation...]

Result: Job submitted successfully! Screenshot shows confirmation page.
```
