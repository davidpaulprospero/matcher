# Enhanced Pipeline Features

## Overview

These new modules add powerful features to voiceover-matcher:

1. **90% Confidence Enforcement** - Remix keywords until matches meet threshold
2. **Zero-Download Keyword Remix** - Auto-regenerate failed keywords with topic context
3. **Pexels/Pixabay Integration** - Stock footage on separate OTIO track
4. **Image Downloads** - High-res images (>1MB filter)
5. **Multi-Style OTIO** - Generate 2 timelines with different settings

---

## Installation

### 1. Copy modules to `src/` folder:

```bash
cp pexels.py pixabay.py imagedl.py keyword_remix.py multi_style.py enhanced_pipeline.py D:/matcher-alt/src/
```

### 2. Add API keys to `.env`:

```env
# Stock footage APIs (free tiers available)
PEXELS_API_KEY=your_pexels_key
PIXABAY_API_KEY=your_pixabay_key
UNSPLASH_API_KEY=your_unsplash_key   # Optional for images
```

### 3. Add config sections to `config.yaml`:

See `config_additions.yaml` for all new settings.

---

## Feature Details

### 1. Confidence Enforcement (90% minimum)

```yaml
confidence_enforcement:
  enabled: true
  min_confidence: 0.90        # 90% minimum
  max_retries: 3              # Max remix attempts
  remix_low_confidence: true
  remix_zero_downloads: true
```

**How it works:**
- After matching, checks all matches below 90% confidence
- Groups low-confidence matches by source keyword
- Remixes keywords using Gemini with topic context
- Re-downloads footage for remixed keywords
- Re-matches affected segments
- Repeats up to 3 times

### 2. Zero-Download Keyword Remix

When a keyword returns 0 results:

```
Original: "2018"
→ Gemini remixes with topic context
→ New: ["2018 Kilauea eruption footage", "2018 Hawaii volcanic activity"]
```

**Context-aware generation:**
- Never generates date-only keywords
- Always combines dates with topic terms
- Uses main topic context for relevance

### 3. Stock Footage Integration

```yaml
stock_footage:
  pexels:
    enabled: true
    per_keyword: 2
    min_duration: 5
    max_duration: 60
  pixabay:
    enabled: true
    per_keyword: 2
```

**OTIO Track Assignment:**
- YouTube footage → V1 (primary)
- Pexels/Pixabay → V_Stock (separate track)
- Metadata tags: `is_stock_footage: true`

### 4. Image Downloads

```yaml
image_downloads:
  enabled: true
  per_keyword: 2
  min_size_mb: 1.0  # Only download images > 1MB
```

**Sources:** Pexels, Pixabay, Unsplash (if API key available)

### 5. Multi-Style OTIO

Generate 2 timelines with different settings:

**Style 1 (default):** Standard matching  
**Style 2 (configurable):** Prompted before download

```
Available presets:
1. strict      - High confidence, fewer alternatives
2. stock_heavy - Prefer Pexels/Pixabay footage
3. fast_paced  - Shorter clips, faster cutting
4. cinematic   - Longer clips, slower pace
5. custom      - Configure manually
```

**Output files:**
```
otio_output/
├── timeline_default_20251216.otio
└── timeline_strict_20251216.otio
```

---

## Usage Examples

### Basic usage with confidence enforcement:

```python
from src.enhanced_pipeline import (
    EnhancedPipelineConfig,
    ConfidenceEnforcementLoop,
    prompt_enhanced_settings,
    prompt_topic_context
)
from src.keyword_remix import KeywordRemixer, create_confidence_enforcer

# Get topic context
topic = prompt_topic_context()
# > Main topic: Hawaii Kilauea volcano eruption

# Get enhanced settings
config = prompt_enhanced_settings()
# > Minimum confidence % [90]: 
# > Enable Pexels? [Y/n]: 
# > Generate multiple OTIO styles? [Y/n]: 

# Create remixer with topic context
enforcer, remixer = create_confidence_enforcer(
    min_confidence=0.90,
    max_retries=3,
    topic_context=topic
)

# Run with enforcement
loop = ConfidenceEnforcementLoop(config, remixer)
final_keywords, final_matches = loop.run_with_enforcement(
    keywords=initial_keywords,
    download_func=download_footage,
    match_func=match_segments
)
```

### Download from all sources:

```python
from src.pexels import download_pexels_footage
from src.pixabay import download_pixabay_footage
from src.imagedl import download_images

keywords = ["Hawaii volcano", "lava flow", "Kilauea eruption"]

# Stock footage
pexels_paths, pexels_counts = download_pexels_footage(
    keywords, output_dir="./downloaded_videos/stock", per_keyword=2
)

pixabay_paths, pixabay_counts = download_pixabay_footage(
    keywords, output_dir="./downloaded_videos/stock", per_keyword=2
)

# Images (>1MB only)
image_paths, image_counts = download_images(
    keywords, output_dir="./downloaded_images", 
    per_keyword=2, min_size_mb=1.0
)
```

### Multi-style OTIO:

```python
from src.multi_style import (
    MultiStyleOTIOGenerator,
    prompt_for_second_style,
    STYLE_DEFAULT
)

# Prompt for second style before downloading
second_style = prompt_for_second_style()

# Generate both timelines
generator = MultiStyleOTIOGenerator(output_dir="./otio_output")
generator.add_style(STYLE_DEFAULT)
generator.add_style(second_style)

# Later, generate timelines with different configs
for style in generator.styles:
    config_overrides = generator.get_style_config_for_matching(style)
    # ... run matching with overrides
    # ... generate OTIO with style name
```

---

## Integration with main.py

Add to `main.py` imports:

```python
from src.keyword_remix import KeywordRemixer, create_confidence_enforcer
from src.multi_style import prompt_for_second_style, prompt_multi_style_enabled
from src.enhanced_pipeline import (
    EnhancedPipelineConfig,
    prompt_enhanced_settings,
    prompt_topic_context
)
```

Add before keyword extraction:

```python
# Enhanced settings prompt
enhanced_config = prompt_enhanced_settings()

# Topic context for keyword remixing
topic_context = prompt_topic_context()

# Multi-style OTIO prompt
if prompt_multi_style_enabled():
    second_style = prompt_for_second_style()
```

Add after matching:

```python
# Confidence enforcement loop
if enhanced_config.min_confidence > 0:
    enforcer, remixer = create_confidence_enforcer(
        min_confidence=enhanced_config.min_confidence,
        topic_context=topic_context
    )
    # ... run enforcement loop
```

---

## OTIO Track Layout

```
V1:      Primary matches (YouTube)
V2:      Alternative 1
V3:      Alternative 2
V4-V8:   Strategy tracks (visual_first, different_source, etc.)
V_Stock: Stock footage (Pexels/Pixabay)
V_Images: Still images

A1-A8:   Corresponding audio
A_VO:    Voiceover
```

---

## API Key Setup

### Pexels (Free)
1. Go to https://www.pexels.com/api/
2. Sign up and request API key
3. Add to `.env`: `PEXELS_API_KEY=your_key`

### Pixabay (Free)
1. Go to https://pixabay.com/api/docs/
2. Sign up and get API key
3. Add to `.env`: `PIXABAY_API_KEY=your_key`

### Unsplash (Optional, Free)
1. Go to https://unsplash.com/developers
2. Create app and get Access Key
3. Add to `.env`: `UNSPLASH_API_KEY=your_key`

---

## File Summary

| File | Description |
|------|-------------|
| `pexels.py` | Pexels API integration |
| `pixabay.py` | Pixabay API integration |
| `imagedl.py` | Multi-source image downloader |
| `keyword_remix.py` | Gemini-powered keyword remixing |
| `multi_style.py` | Multi-style OTIO generation |
| `enhanced_pipeline.py` | Integration module |
| `config_additions.yaml` | Config sections to add |
| `matching.py` | Updated with robust JSON parsing |
| `logger.py` | Fixed RunLogger alias |
