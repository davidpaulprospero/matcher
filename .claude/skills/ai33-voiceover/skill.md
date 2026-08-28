---
name: ai33-voiceover
description: Generate a voiceover MP3 via the AI33 TTS API (ElevenLabs-compatible, base URL https://api.ai33.pro/v3), then emit a matching placeholder SRT ready for `python main.py --voiceover`. Use when the user has a script (.txt/.md/.srt/inline) and wants a freshly-rendered AI33 voiceover before running the matcher pipeline. Falls back gracefully to whatever AI33_VOICE_ID / AI33_* voice settings are already in .env.
allowed-tools:
  - Read
  - Write
  - Edit
  - Glob
  - Grep
  - Bash
---

# AI33 Voiceover Skill

Render a **voiceover MP3** from a script using the **AI33 TTS API**, then write a
matching `voiceover.srt` so the user can immediately feed it to:

```
python main.py --voiceover <output_dir>/voiceover.srt
```

The pipeline re-transcribes the MP3 to recover real SRT timings — the SRT this
skill writes has placeholder `00:00:00,000 --> 00:00:00,000` markers, and that
is fine. The pipeline's whisper stage overwrites them.

## When to Use

- User has a script (text file, inline string, or existing SRT) and wants AI33 TTS
- User says "make a voiceover with AI33", "TTS this", "AI33 voiceover for this script"
- User wants to drop a freshly-rendered MP3 into a project before running the matcher
- User wants to re-render audio for an existing `voiceover.srt` (different voice / speed)

## When NOT to Use

- User wants to transcribe an existing video/audio (use `shot-otio`)
- User has no API key (the skill will fail loudly with `AI33_API_KEY missing`)
- User wants a different TTS provider (Fish Audio, OpenAI, Edge TTS, etc. — out of scope)

## Prerequisites

API key + voice settings live in the repo root `.env`. Lines added 2026-08-17,
updated 2026-08-21:

```ini
AI33_API_KEY=sk_...                              # required
AI33_VOICE_ID=elevenlabs_<id>                    # required (see "Voice ID format" below)
AI33_API_URL=https://api.ai33.pro/v3             # required (v3 only; v1 returns 400)
AI33_SPEED=1.04                                  # optional
AI33_STABILITY=0.50                              # optional
AI33_SIMILARITY=0.60                             # optional (similarity_boost)
AI33_STYLE=0.75                                  # optional
AI33_SPEAKER_BOOST=true                          # optional
AI33_WITH_TRANSCRIPT=false                       # optional (set true to also get transcript_url)
AI33_POLL_INTERVAL=2                             # optional (seconds between task status polls)
```

`.env` is already in `.gitignore` (`/^\.env$/`). Use `scripts/.env.example` as a
reference; copy the lines above into your `.env` if missing.

### Voice ID format (v3 requirement)

The `AI33_VOICE_ID` must start with one of these prefixes, followed by the
upstream provider's voice id:

| Prefix       | Provider        | Example                                |
|--------------|-----------------|----------------------------------------|
| `elevenlabs_`| ElevenLabs      | `elevenlabs_pNInz6obpgDQGcFmaJgB` (Adam)|
| `minimax_`   | MiniMax         | `minimax_209533299589198`              |
| `clone_`     | Cloned voice    | (per-account clone id)                 |
| `edge_`      | Edge TTS        | `edge_en-US-AriaNeural`                |
| `kokoro_`    | Kokoro          | `kokoro_af_heart`                      |
| `vbee_`      | Vbee            | (Vietnamese TTS)                       |
| `fishaudio_` | Fish Audio      | `fishaudio_<id>`                       |

The historical `"default"` placeholder and bare provider-less ids (e.g.
`"Brian"`, `"Adam"`) are rejected with HTTP 400
`voice_id must start with elevenlabs_, minimax_, clone_, edge_, kokoro_, vbee_, or fishaudio_`.

The `/voices` catalog endpoint does NOT list every valid id — a voice that
isn't in the listing can still be submitted successfully. Probe with a 6-char
test ("Test.") before committing to a full render.

## Invocation

```
/ai33-voiceover --input-file script.txt --output-dir "E:\Edit Job\client\project"
```

### Examples

```
# Render script.txt into MP3 + placeholder SRT, drop into project voiceover/ dir
/ai33-voiceover --input-file "E:\scripts\cuba.txt" \
                --output-dir "E:\Edit Job\cuba\Project\voiceover"

# Inline script via --text
/ai33-voiceover --text "In 1492, Columbus sailed the ocean blue..." \
                --output-dir "E:\Edit Job\cuba\Project\voiceover"

# Re-render audio for an existing SRT (text extracted from cues)
/ai33-voiceover --input-srt "E:\Edit Job\cuba\Project\voiceover\voiceover.srt" \
                --output-dir "E:\Edit Job\cuba\Project\voiceover"

# Different voice + slower pace
/ai33-voiceover --input-file script.txt --output-dir out/ \
                --voice-id some_other_voice --speed 0.95

# Preview what would be sent without actually calling the API
/ai33-voiceover --input-file script.txt --output-dir out/ --dry-run

# Submit only (returns task_id, exits immediately)
/ai33-voiceover --input-file script.txt --output-dir out/ --no-poll
```

## Arguments

- **Input (one of, required):**
  - `--text "..."` — Inline script text.
  - `--input-file <path>` — `.txt` or `.md` script. **Do NOT pass `.docx`** — see "Gotchas" below; use `python-docx` to extract prose first.
  - `--input-srt <path>` — Explicit SRT re-render path (same as `--input-file` for `.srt`). SRT timestamps are stripped automatically.
- `--output-dir <dir>` (required): Where to write `voiceover.mp3`, `voiceover.srt`, `voiceover.txt`.
- `--filename <stem>` (default `voiceover`): Output file stem.
- `--voice-id <id>` (optional): Override `AI33_VOICE_ID` for this run only.
- `--speed <float>`, `--stability <float>`, `--similarity <float>`, `--style <float>` (optional): Per-run voice setting overrides.
- `--no-speaker-boost` (optional): Disable `use_speaker_boost` for this run.
- `--format <id>` (default `mp3_44100_128`): AI33 output format string.
- `--with-transcript` (optional): Force `AI33_WITH_TRANSCRIPT=true` for this run; AI33 returns a transcript_url alongside the audio_url.
- `--no-poll` (optional): Submit only, exit immediately. Printed `task_id` can be polled later via `python scripts/ai33_client.py ...` or reused.
- `--dry-run` (optional): Resolve script text, write `.txt` and `.srt` scaffolds, print what *would* be sent. No API call.
- `--keep-srt-text` (optional): When feeding an existing SRT, pass through the original text verbatim instead of extracting just the dialogue.
- `--dotenv <path>` (optional): Path to .env. Default: `<repo-root>/.env` (resolved relative to this skill).
- `--verbose / -v` (optional): Debug logging.

## Outputs

For `--output-dir out --filename voiceover` (defaults), three files appear:

| Path                | Content                                                  |
|---------------------|----------------------------------------------------------|
| `out/voiceover.mp3` | AI33-rendered audio (always)                             |
| `out/voiceover.srt` | SRT scaffold (placeholder `00:00:00` timings)           |
| `out/voiceover.txt` | Plain-text script (always, even on `--dry-run`)          |

The `.srt` is **intentionally** a scaffold — the matcher's `whisper` stage will
re-transcribe `voiceover.mp3` and overwrite the timings during the next
`python main.py --voiceover out/voiceover.srt` run.

## Gotchas

These all came from real bugs during the 2026-08-21 batch render. Read before
you start a long render:

### 1. `.docx` files are ZIP archives — do not pass them directly

`scripts/generate_voiceover.py` does `path.read_text(encoding="utf-8")`. For a
`.docx` (which is a ZIP), that returns the binary `PK\x03\x04` header bytes
plus XML gibberish. The TTS API silently returns HTTP 500
`"Something went wrong"` on that binary garbage — no useful error.

**Always extract prose first:**

```python
import docx
d = docx.Document("script.docx")
text = "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
```

If the skill's `--input-file` is invoked on a `.docx` path, this should be
auto-detected and converted (TODO: not yet wired — see `render_5_voiceovers.py`
in repo root for the pattern used in production).

### 2. Long scripts must be chunked — per-request char limit

ElevenLabs's `eleven_multilingual_v2` model has a per-request character cap
(~10K for paid plans). Submitting 17K+ chars in one request returns HTTP 500
even though the API endpoint "accepts" the payload. AI33 does not auto-chunk.

**Required pattern for long scripts:**

1. Split text into ~900-char chunks on sentence boundaries (`[.!?]\s+`).
2. Submit each chunk separately, wait for completion, collect MP3s.
3. Concatenate the per-chunk MP3s with ffmpeg's concat demuxer:

   ```bash
   ffmpeg -y -f concat -safe 0 -i concat.txt -c copy voiceover.mp3
   ```

   The `-c copy` flag stream-copies without re-encoding, so it's instant.

4. Keep the `.srt` scaffold in chunk order so cue boundaries roughly match the
   audio joins (whisper re-transcription will produce better real timings than
   the scaffold anyway).

### 3. Concurrency cap — 10 in-flight tasks

AI33 enforces a 10-task queue cap per account:

```
HTTP 429: "You have too many tasks in queue (10/10), please wait for it to
finish. Or buy more credits to get higher rate limit."
```

Submit sequentially with `wait_for_completion=True`. Do not parallelize batch
jobs across processes — the cap is per-account, not per-process.

### 4. Voice IDs in `.env.example` were stale (now fixed)

The shipped `scripts/.env.example` previously showed
`AI33_VOICE_ID=default` (rejected) and `AI33_API_URL=.../v1` (rejected). Both
are corrected in the current file.

### 5. Windows path gotchas

- Use `python -X utf8` when invoking scripts that print paths containing em
  dashes, apostrophes, or `$` (e.g. "South Korea's $576 Billion ...").
- Bash treats `$576` as a variable expansion; quote paths with single quotes
  or use Python's `pathlib`/`glob` to locate the file instead.
- `ffmpeg` lives at `C:\ffmpeg\bin\ffmpeg.EXE` on this machine — use
  `shutil.which("ffmpeg")` for portability instead of hard-coding
  `/c/ffmpeg/bin/ffmpeg.exe` (which only works inside Git Bash).

### 6. HTTP 500 vs HTTP 429

Both can mean "queue overloaded" or "voice temporarily broken". If you see
5+ HTTP 500s in a row with `code=unknown_error`, retry after 30s — the
backend usually self-recovers. Persistent 500s across multiple voices indicate
account-level rate limiting.

## Workflow

1. Resolve `--text` / `--input-file` / `--input-srt` to raw script text.
   - **If `--input-file` ends in `.docx`, extract prose via `python-docx` first.**
   - If input is `.srt`, strip timestamps (unless `--keep-srt-text`).
2. Always write `voiceover.txt` and a `voiceover.srt` scaffold first
   (so the user has the files even if the API call fails).
3. **If text > 900 chars, chunk on sentence boundaries before submitting.**
4. Submit text to `POST {AI33_API_URL}/text-to-speech` with model
   `eleven_multilingual_v2`, `provider=elevenlabs` (or matching prefix).
5. Poll `GET {AI33_API_URL}/task/{task_id}` every `AI33_POLL_INTERVAL`s until
   `status=done` with a populated `metadata.audio_url`.
6. Stream-download `audio_url` → `voiceover.mp3`.
7. **If you chunked, concatenate the per-chunk MP3s with ffmpeg before exit.**
8. Print next-step hint: `python main.py --voiceover "<srt_path>"`.

Exit codes: `0` success, `1` API failure, `2` bad input / missing key.

## Polling Behaviour

- Default poll interval: `2s` (`AI33_POLL_INTERVAL`).
- Progress logged every `30s` (`Task <id>... still processing (1.4 min)`).
- No upper bound on total wait — long scripts may take several minutes.
- `failed` status raises immediately; transient HTTP errors retry forever.

## Implementation Notes

- Pure-stdlib + `requests` + `python-dotenv` (already a project dep).
- No EU-AUTO imports — the client is self-contained.
- `scripts/ai33_client.py` is reusable as a library: `from ai33_client import AI33TextToSpeech`.
- API key, voice settings, base URL all sourced from env first; CLI flags override per-run.
- Long-script chunking + ffmpeg concatenation is NOT yet wired into
  `scripts/generate_voiceover.py` — for `.docx` or scripts >900 chars, use the
  batch pattern below or `render_5_voiceovers.py` in repo root as a reference.

## Files

- `skill.md` — this manifest
- `scripts/ai33_client.py` — standalone AI33 TTS client + CLI
- `scripts/generate_voiceover.py` — script-to-MP3 orchestrator + SRT scaffold writer
- `scripts/.env.example` — copyable env stub

## Batch rendering (.docx, multiple projects)

When you have several `.docx` scripts to render into separate project
`voiceover/` dirs, the per-invocation `generate_voiceover.py` is too slow and
will fail on long scripts. Use the batch pattern from
`render_5_voiceovers.py` (repo root) — the high-level steps:

1. **Extract prose per docx** with `python-docx` (`Document.paragraphs`).
2. **Chunk into ≤900-char sentence-aligned pieces** (regex
   `(?<=[.!?])\s+`).
3. **Submit chunks sequentially** to AI33 with `wait_for_completion=True`
   (do NOT parallelize — see Gotcha #3).
4. **Concatenate per-chunk MP3s** with `ffmpeg -f concat -c copy`.
5. Write placeholder `.srt` (one cue per chunk) and the original `.txt`.

Budget: ~10s per ~900-char chunk on ElevenLabs via AI33. A 20K-char docx takes
~5 min end-to-end. Five docx files in serial = ~30–50 min.

A reference implementation lives at `D:\_Projects\voiceover-matcher-dev\render_5_voiceovers.py`
if you want to copy/adapt it.

## Related Skills

- `shot-otio` — Transcribe an existing video to SRT (opposite direction).
- `imagen-otio` — Generate still-image timeline from the same SRT.
- `lower-thirds` — Generate name/title graphics from the same SRT.