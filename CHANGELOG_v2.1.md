# Voiceover-to-Footage Matcher v2.1

## New Features

### 1. V8 Source-Rotation Track
Cycles through all source videos systematically for maximum variety.
- Segment 1 → Video A (best clip from A)
- Segment 2 → Video B (best clip from B)
- Segment 3 → Video C (best clip from C)
- ...cycles back to A

**Config:**
```yaml
output:
  strategy_tracks:
    - "source_rotation"  # V8
```

### 2. Hybrid Embeddings (Text + Visual)
Combines text transcript embeddings with visual scene description embeddings.
- Weighted average: 70% text + 30% visual
- Falls back to text-only when vision disabled

**Config:**
```yaml
embedding:
  hybrid_mode: true
  text_weight: 0.7
  visual_weight: 0.3
```

### 3. Auto-Detected Keyword Weights (Topic-Agnostic)
Uses TF-IDF to automatically extract distinctive terms from your voiceover.
- Works for ANY topic (disasters, cooking, tech reviews)
- No hardcoded keywords needed

**Config:**
```yaml
keyword_weights:
  enabled: true
  auto_detect: true
  auto_detect_count: 20
  boost_factor: 0.15
  custom_boost_terms: []    # Optional manual additions
  custom_penalty_terms: []  # Optional manual penalties
```

### 4. FAISS Indexing (Optional)
10-50x faster candidate retrieval for large video libraries.
- Falls back to numpy if FAISS not installed
- Install with: `pip install faiss-cpu`

**Config:**
```yaml
indexing:
  use_faiss: true
  index_type: "flat"  # or "ivf" for larger libraries
```

### 5. Two-Stage Matching Optimization
- Stage 1: Retrieve 20 candidates from embeddings (fast)
- Stage 2: Send only 5 to LLM for reranking (saves API calls)

**Config:**
```yaml
matching:
  embedding_candidates: 20
  llm_rerank_candidates: 5
```

### 6. Duration-Aware Scoring
Penalizes clips requiring extreme speed changes.
- 85%-115% speed = no penalty (ideal)
- 70%-150% speed = small penalty
- Outside = larger penalty

**Config:**
```yaml
matching:
  duration_scoring_enabled: true
  ideal_speed_range: [0.85, 1.15]
  soft_penalty_range: [0.7, 1.5]
  duration_penalty_factor: 0.1
```

### 7. Comprehensive Logging
Every run creates timestamped logs in `./logs/`:
- `.log` file: Human-readable
- `.json` file: Machine-parseable

**Config:**
```yaml
logging:
  enabled: true
  log_dir: "./logs"
  log_match_decisions: true
  log_api_calls: true
  log_performance: true
  log_config_snapshot: true
```

## Updated Track Structure

| Track | Strategy | Description |
|-------|----------|-------------|
| V1 | Primary | Best overall match (enabled) |
| V2 | Alternative 1 | Second best (disabled) |
| V3 | Alternative 2 | Third best (disabled) |
| V4 | Visual-First | Prioritizes scene descriptions |
| V5 | Different-Source | Forces different video file |
| V6 | Keyword-Only | Matches on keywords/entities |
| V7 | Embedding-Diversity | Maximally different from V1 |
| V8 | Source-Rotation | Cycles through all sources |
| A1-A8 | Audio | Corresponding audio tracks |
| A9 | Voiceover | Your narration (enabled) |

## Installation

```bash
pip install -r requirements.txt

# Optional: Install FAISS for faster search
pip install faiss-cpu
```

## Files Modified

- `src/config.py` - Added new config dataclasses
- `src/embeddings.py` - Added FAISS indexing, hybrid embedding
- `src/keywords.py` - Added TF-IDF keyword weight extraction
- `src/matching.py` - Added source_rotation, two-stage, duration scoring
- `src/logger.py` - New comprehensive logging system
- `src/otio_builder.py` - Added V8 track support
- `config.yaml` - Updated with all new options + comments
