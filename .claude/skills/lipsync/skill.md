# Lipsync Submit Skill

## Description

Submit an AI Lipsync job to the Degold form. Prompts for required info, cross-references with CHANNELS.md, confirms with user, then runs the automator.

## Trigger Phrases

- "submit lipsync"
- "lipsync job"
- "submit to degold"
- "generate lipsync"

## Parameters

- `$VIDEO_TITLE` - Video title (optional, will prompt if not provided)
- `$CHANNEL_CODE` - Channel code like RRU, JDRP, DSR (optional)
- `$AVATAR_PATH` - Path to avatar image (optional)
- `$AUDIO_FILES` - Audio segment files (optional)

## Workflow

### Step 1: Gather Information

If not all parameters provided, ask user for:

1. **Video Title** - Title for the generated video (e.g., "EP42 - Trabant Secrets")
2. **Channel Code** - Channel to use (RRU, JDRP, DSR, etc.)
3. **Avatar Path** - Path to avatar image file
4. **Audio Files** - List of audio segment files in order

### Step 2: Cross-Reference with CHANNELS.md

Read `Degold/CHANNELS.md` and verify:
- Channel code exists in the document
- Display the channel's drive folder ID and avatar folder from the config

### Step 3: Confirm with User

Show the user a summary:

```
Lipsync Job Submission
=====================
Video Title: [title]
Channel: [channel code]
  - Drive Folder: [folder_id from CHANNELS.md]
  - Avatar Folder: [avatar_folder from CHANNELS.md]
Avatar: [path]
Audio Files: [list]

Type "yes" or "y" to confirm and submit.
```

Wait for user confirmation before proceeding.

### Step 4: Run the Automator

Execute:
```bash
cd Degold && python lipsync_automator.py -t "$VIDEO_TITLE" -c $CHANNEL_CODE -a "$AVATAR_PATH" $AUDIO_FILES
```

Report the result to the user.

## Error Handling

- If channel code not found in CHANNELS.md, warn the user and ask if they want to proceed anyway or add the channel
- If avatar or audio files don't exist, report the error
- If submission fails, show the error and suggest checking the form manually
