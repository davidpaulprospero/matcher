# Keyword Mode: Pipeline Without Voiceover

> **Status:** Planning
> **Created:** 2026-01-19
> **Goal:** Run pipeline with keywords only, no voiceover/SRT required

## Overview

Enable the pipeline to work with just keywords as input, synthesizing the segment structure that normally comes from voiceover. This allows users to generate video compilations without needing narration upfront.

## Three Modes

| Mode | Purpose | Input | Output |
|------|---------|-------|--------|
| **Montage** | Quick assembly | Keywords + duration | Equal-time segments, silent timeline |
| **Script** | Intelligent structure | Keywords + style | LLM-generated SRT + matched timeline |
| **Collection** | Research/gathering | Keywords + count | Organized folders, no timeline |

---

## Mode A: Montage (Simplest)

### Concept

Each keyword becomes a segment with equal duration.

```
Input:  keywords = ["sunset", "ocean", "beach"]
        duration = 60s

Output: 3 segments × 20s each
        - Segment 1: "sunset" (0:00-0:20)
        - Segment 2: "ocean" (0:20-0:40)
        - Segment 3: "beach" (0:40-1:00)
```

### Implementation

```python
@dataclass
class KeywordSegment:
    keyword: str
    start_time: float
    end_time: float
    embedding: Optional[np.ndarray] = None

    def to_voiceover_segment(self) -> VoiceoverSegment:
        return VoiceoverSegment(
            text=self.keyword,
            start=self.start_time,
            end=self.end_time,
            embedding=self.embedding,
            source="keyword_montage"
        )
```

### Config

```yaml
keyword_mode:
  enabled: true
  mode: "montage"

  montage:
    segment_duration: 10.0    # Seconds per keyword (if no total_duration)
    total_duration: null      # If set, divides evenly among keywords
    transition_gap: 0.0       # Gap between segments
```

---

## Mode B: Script (Primary Focus)

### Concept

LLM generates narration script from keywords → convert to timed SRT → pipeline uses SRT normally.

**The SRT is the artifact.** User can optionally do TTS later.

```
Keywords → LLM Script → Timed SRT → Normal Pipeline → Silent Timeline + SRT
```

### Why This Works

| Current Pipeline | Keyword Mode |
|------------------|--------------|
| User provides `voiceover.srt` | System generates `synthetic.srt` |
| Parse SRT → VoiceoverSegments | Parse synthetic SRT → VoiceoverSegments |
| Match videos to segments | Match videos to segments (unchanged) |
| Output timeline | Output timeline + the SRT as deliverable |

### LLM Script Generation

**Input:**
```
keywords: ["coral reef", "marine life", "ocean conservation"]
style: "documentary"
target_duration: 120s
words_per_minute: 150
```

**LLM Output:**
```
Beneath the waves lies a world of extraordinary beauty.

Coral reefs, often called the rainforests of the sea, support nearly a quarter of all marine species.

These intricate structures take thousands of years to form, yet can be destroyed in moments.

Fish of every color dart through the coral branches. Sea turtles glide past ancient formations. Sharks patrol the deeper edges.

But this underwater paradise faces an uncertain future.

Rising ocean temperatures cause coral bleaching, turning vibrant reefs into ghostly white graveyards.

Pollution and overfishing compound the damage, disrupting delicate ecosystems that took millennia to develop.

Yet there is hope.

Marine protected areas have shown remarkable results. Given time and protection, reefs can recover.

Conservation efforts worldwide are expanding. Scientists are developing heat-resistant coral strains.

The ocean's future depends on the choices we make today.
```

### SRT Timing Algorithm

```python
def script_to_srt(script: str, target_duration: float, wpm: int = 150) -> str:
    """
    Convert script text to timed SRT format.

    Strategy:
    1. Split on paragraph breaks (double newline) → segments
    2. Calculate duration per segment based on word count
    3. Add padding between segments
    4. Scale to fit target duration
    """
    paragraphs = [p.strip() for p in script.split('\n\n') if p.strip()]

    total_words = sum(len(p.split()) for p in paragraphs)
    words_per_second = wpm / 60

    natural_duration = total_words / words_per_second
    time_scale = target_duration / natural_duration if natural_duration > 0 else 1.0

    srt_entries = []
    current_time = 0.0
    gap = 0.3  # seconds between segments

    for i, paragraph in enumerate(paragraphs):
        word_count = len(paragraph.split())
        segment_duration = max((word_count / words_per_second) * time_scale, 2.0)

        if i > 0:
            current_time += gap

        srt_entries.append({
            'index': i + 1,
            'start': current_time,
            'end': current_time + segment_duration,
            'text': paragraph
        })

        current_time += segment_duration

    return format_srt(srt_entries)
```

### Generated SRT Example

```srt
1
00:00:00,000 --> 00:00:04,800
Beneath the waves lies a world of extraordinary beauty.

2
00:00:05,100 --> 00:00:12,600
Coral reefs, often called the rainforests of the sea, support nearly a quarter of all marine species.

3
00:00:12,900 --> 00:00:20,100
These intricate structures take thousands of years to form, yet can be destroyed in moments.
```

### Script Styles

| Style | Characteristics | Use Case |
|-------|-----------------|----------|
| `documentary` | Authoritative, informative, measured | Nature, history, science |
| `promotional` | Punchy, benefit-focused, CTA | Products, services |
| `narrative` | Story arc, emotional beats | Travel, personal, event |
| `listicle` | Numbered points, clear structure | Tutorials, top-10s |
| `poetic` | Evocative, metaphorical | Mood pieces, art |
| `minimal` | Short phrases, sparse | Music videos, montages |

### LLM Prompt Template

```python
SCRIPT_GENERATION_PROMPT = """
You are a documentary scriptwriter. Write narration for a {style} video.

Topic keywords: {keywords}
Target duration: {duration} seconds (approximately {word_count} words at {wpm} WPM)
Tone: {tone}

Requirements:
1. Write in a natural speaking voice suitable for narration
2. Use paragraph breaks to indicate natural pauses/scene changes
3. Each paragraph should be 1-3 sentences (one visual idea)
4. Start with a hook, end with impact
5. Weave ALL keywords naturally into the script
6. Don't use headers, bullets, or formatting - just flowing paragraphs

Style notes for {style}:
{style_notes}

Write the script only, no commentary.
"""
```

### Config

```yaml
keyword_mode:
  enabled: true
  mode: "script"

  script:
    # Script generation
    style: "documentary"      # documentary | promotional | narrative | listicle | poetic | minimal
    tone: "inspiring"         # inspiring | serious | playful | urgent | contemplative

    # Timing
    target_duration: 120      # seconds
    words_per_minute: 150     # 120=slow, 150=normal, 180=fast
    min_segment_duration: 2.0
    segment_gap: 0.3

    # Output
    save_script_txt: true     # Save raw script as .txt
    save_srt: true            # Save generated .srt

    # LLM settings
    llm:
      provider: "ollama"
      model: "mistral:7b"
      fallback_models:
        - "qwen2.5:7b"
        - "llama3.2:3b"
      temperature: 0.7
      max_tokens: 1000
      ollama_host: "http://localhost:11434"
      timeout: 60

    # Refinement
    require_all_keywords: true

    # Segment splitting
    split_mode: "paragraph"   # paragraph | sentence | hybrid
```

### Ollama Model Recommendations

| Model | Size | RAM | Speed | Quality | Notes |
|-------|------|-----|-------|---------|-------|
| **mistral:7b** | 4.1GB | 8GB | Fast | Good | **Default** - reliable |
| **qwen2.5:7b** | 4.7GB | 8GB | Fast | Good+ | Better multilingual |
| **llama3.2:3b** | 2.0GB | 4GB | Very Fast | Fair | Lightweight fallback |
| **phi4:14b** | 9.1GB | 16GB | Slower | Very Good | Quality option |

**Primary recommendation: `mistral:7b`** - fast, reliable, good instruction following.

### Model Auto-Detection

```python
PREFERRED_MODELS = [
    "mistral:7b",
    "mistral:latest",
    "qwen2.5:7b",
    "qwen2.5:latest",
    "llama3.2:3b",
    "llama3.2:latest",
    "phi4:14b",
]

def _get_ollama_model(self) -> str:
    """Find best available Ollama model"""
    available = self._list_ollama_models()

    for model in PREFERRED_MODELS:
        if model in available:
            return model

    if available:
        return available[0]  # Fallback to any installed

    raise RuntimeError("No Ollama models found. Run: ollama pull mistral:7b")
```

### Output Artifacts

```
ProjectName/
├── voiceover/
│   ├── synthetic.srt           # Generated SRT (drives matching)
│   └── synthetic_script.txt    # Raw script (for editing/TTS)
├── output/
│   ├── timeline.otio           # Silent video timeline
│   ├── timeline.xml
│   └── segments.json
```

---

## Mode C: Collection (No Timeline)

### Concept

Download and organize videos by keyword without timeline generation.

```
Input:  keywords = ["sunset", "ocean", "beach"]
        clips_per_keyword = 10

Output: Organized folders with scored clips
        output/
        ├── sunset/
        │   ├── video_001.mp4
        │   └── manifest.json
        ├── ocean/
        └── beach/
```

### Config

```yaml
keyword_mode:
  enabled: true
  mode: "collection"

  collection:
    clips_per_keyword: 10
    output_format: "folders"    # folders | flat_with_manifest
    include_metadata: true      # Save transcripts, scores
```

---

## Pipeline Integration

### Stage: ScriptSynthesisStage

```python
class ScriptSynthesisStage(Stage):
    """Generates synthetic SRT from keywords using LLM"""

    name = "SCRIPT_SYNTHESIS"

    def run(self, state: PipelineState) -> PipelineState:
        keywords = state.keywords

        # Generate script via LLM
        script = self._generate_script(keywords)

        # Convert to timed SRT
        srt_content = self._script_to_srt(script)

        # Save outputs
        srt_path = self.project_dir / "voiceover" / "synthetic.srt"
        srt_path.parent.mkdir(exist_ok=True)
        srt_path.write_text(srt_content, encoding='utf-8')

        # Parse SRT into segments (reuse existing parser)
        from src.voiceover_parser import parse_srt
        segments = parse_srt(srt_path)

        state.voiceover_segments = segments
        state.voiceover_path = srt_path
        state.metadata['script_source'] = 'synthetic'

        return state
```

### Pipeline Builder Modification

```python
def build_pipeline(config: Config, project_dir: Path) -> List[Stage]:
    stages = []

    if config.keyword_mode.enabled:
        mode = config.keyword_mode.mode

        if mode == "script":
            stages.append(ScriptSynthesisStage(config, project_dir))
        elif mode == "montage":
            stages.append(MontageSegmentStage(config, project_dir))
        elif mode == "collection":
            stages.append(CollectionDownloadStage(config, project_dir))
            stages.append(CollectionOrganizeStage(config, project_dir))
            return stages  # Skip rest of pipeline

    # Standard pipeline continues...
    stages.extend([
        AnalyzeStage(config, project_dir),
        DownloadStage(config, project_dir),
        # ...
    ])

    return stages
```

### CLI Interface

```bash
# Simple montage
python main.py --keywords "sunset,ocean,beach" --duration 60

# Script mode (default)
python main.py --keywords "coral reef,marine life,conservation" \
    --mode script \
    --style documentary \
    --duration 120

# Collection mode
python main.py --keywords "sunset,ocean,beach" \
    --mode collection \
    --clips-per-keyword 15
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLI: --keywords "a,b,c"                    │
└─────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
                ▼               ▼               ▼
         ┌──────────┐    ┌──────────┐    ┌──────────┐
         │ MONTAGE  │    │  SCRIPT  │    │COLLECTION│
         │ Equal    │    │ LLM Gen  │    │ Organize │
         │ Segments │    │ → SRT    │    │ Folders  │
         └────┬─────┘    └────┬─────┘    └────┬─────┘
              │               │               │
              ▼               ▼               │
         ┌─────────────────────────┐         │
         │   VoiceoverSegments     │         │
         │   (from SRT or virtual) │         │
         └───────────┬─────────────┘         │
                     │                       │
                     ▼                       ▼
         ┌───────────────────────────────────────────┐
         │           NORMAL PIPELINE                 │
         │   DOWNLOAD → TRANSCRIBE → MATCH → OUTPUT  │
         └───────────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Silent Timeline │
                    │ + SRT artifact  │
                    └─────────────────┘
```

---

## User Workflow

```bash
# 1. Generate script + match videos
python main.py --keywords "coral reef,marine life,conservation" \
    --mode script --style documentary --duration 120

# 2. Review/edit the generated script (optional)
notepad project/voiceover/synthetic_script.txt

# 3. If edited, re-run matching with edited SRT
python main.py --voiceover project/voiceover/edited.srt --match-only

# 4. (Optional) User does TTS externally
#    - ElevenLabs, Edge TTS, professional recording

# 5. Import timeline + final audio into NLE
```

---

## Implementation Phases

### Phase 1: Montage Mode
- [ ] Add `KeywordSegment` dataclass
- [ ] Create `MontageSegmentStage`
- [ ] Add `--keywords` CLI argument
- [ ] Add `keyword_mode.montage` config section
- [ ] Modify pipeline builder

### Phase 2: Script Mode
- [ ] Create `ScriptSynthesisStage`
- [ ] LLM prompt templates for each style
- [ ] SRT timing algorithm
- [ ] Ollama model auto-detection
- [ ] Add `keyword_mode.script` config section
- [ ] Output script.txt alongside SRT

### Phase 3: Collection Mode
- [ ] Create `CollectionDownloadStage`
- [ ] Create `CollectionOrganizeStage`
- [ ] Folder structure with manifests
- [ ] Skip timeline generation path

### Phase 4: Polish
- [ ] Validation and error messages
- [ ] Ollama setup instructions
- [ ] Documentation
- [ ] Tests

---

## Ollama Setup

```bash
# Install Ollama (Windows)
winget install Ollama.Ollama

# Or download from https://ollama.com/download

# Set custom model directory (if needed)
set OLLAMA_MODELS=D:\ollama

# Pull recommended model
ollama pull mistral:7b

# Verify
ollama list
ollama run mistral:7b "Say hello"
```

---

## Open Questions

1. **Keyword weighting**: Should some keywords get longer segments?
   - Option: `--keywords "sunset:20s,ocean:30s,beach:10s"`

2. **Segment granularity**: Split paragraphs further at sentences?
   - Config: `split_mode: hybrid`

3. **Keyword expansion**: Should LLM add related concepts?
   - "ocean" → also search "waves", "sea", "water"

4. **B-roll integration**: How do V8 B-roll tracks work without voiceover?
   - Probably skip, or use "interstitial" segments

---

## References

- [Ollama Models Library](https://ollama.com/library)
- [Best Ollama Models 2025](https://collabnix.com/best-ollama-models-in-2025-complete-performance-comparison/)
- [LLM Benchmarks](https://www.inferless.com/learn/exploring-llms-speed-benchmarks-independent-analysis---part-3)
