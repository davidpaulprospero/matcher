# Video Premise Scoring

## Overview

Replace word-for-word transcript matching with **topic/theme-based matching**. Each video gets an LLM-generated premise summary, and matching scores how well the video's premise fits the voiceover topic.

**Problem solved:**
- Music video "24 Hours (Lyrics)" matching voiceover about "24 hour sanctuary"
- Generic B-roll overused because transcript words loosely match
- Literal matching misses semantic intent

**Solution:**
- Extract video premise: "This is a love song music video"
- Compare to voiceover: "Denny's 24-hour restaurants closing"
- Premise mismatch → low score → not selected

## New Stage: PREMISE

**Pipeline order:**
```
DOWNLOAD → CAPTION → TRANSCRIBE → PREMISE → SCENE_DETECTION → MATCH
```

**Location:** `src/stages/premise.py`

### Stage Logic

```python
class PremiseStage(Stage):
    """Extract topic + context premise for each video."""

    name = "PREMISE"

    def run(self, state: PipelineState) -> StageResult:
        for video in state.downloaded_videos:
            # Skip if cached
            if self.cache.has_premise(video.id):
                video.premise = self.cache.get_premise(video.id)
                continue

            # Get inputs
            transcript = state.transcriptions.get(video.id, "")
            title = video.title or ""
            description = video.description or ""

            # Extract premise via LLM
            premise = self.extract_premise(transcript, title, description)

            # Cache and store
            self.cache.save_premise(video.id, premise)
            video.premise = premise
```

### LLM Prompt for Premise Extraction

```python
PREMISE_EXTRACTION_PROMPT = """
Analyze this video and provide a brief premise summary.

Title: {title}
Description: {description}
Transcript excerpt: {transcript_excerpt}

Respond with a single sentence describing:
1. What type of content this is (documentary, music video, tutorial, news clip, B-roll footage, etc.)
2. The main topic or subject matter

Format: "[Content type] about [topic/subject]"

Examples:
- "Documentary about fast food restaurant closures in America"
- "Music lyric video for a love song"
- "B-roll footage of urban cityscapes at night"
- "News report covering inflation impact on food prices"
- "Tutorial on restaurant kitchen operations"

Premise:
"""
```

### Cache Structure

**Location:** `.cache/premises/`

**File format:** `{video_id}.json`
```json
{
  "video_id": "cDTg1DBMx8s",
  "premise": "Music lyric video for a love song called '24 Hours'",
  "extracted_at": "2026-01-21T10:30:00Z",
  "source": {
    "title": "Cueshe - 24 Hours (Lyrics)",
    "has_transcript": true,
    "has_description": false
  }
}
```

## New Scoring Weights

### Current Weights (to be replaced)
```python
# Old matching logic
transcript_similarity: 40%
embedding_distance: 30%
keyword_overlap: 20%
boosts (broll, entity): 10%
```

### New Weights
```python
# New matching logic
premise_match: 50%      # Video theme ↔ voiceover topic
keyword_overlap: 25%    # Keywords in common
embedding_distance: 15% # Semantic similarity (reduced)
transcript_similarity: 10% # Word-for-word (heavily reduced)
```

### Config Structure

**Add to `config.yaml`:**
```yaml
matching:
  # Premise-based scoring (new)
  premise_scoring:
    enabled: true
    weight: 0.50           # 50% of total score

    # Two-stage comparison
    embedding_filter: true  # Fast filtering via embeddings
    llm_rerank: true        # LLM scoring for top candidates
    llm_rerank_top_n: 10    # Send top 10 to LLM for premise comparison

    # LLM settings
    provider: "gemini"
    model: "gemini-2.0-flash"

  # Adjusted weights
  weights:
    premise: 0.50
    keyword_overlap: 0.25
    embedding: 0.15
    transcript: 0.10
```

## Premise Matching Logic

### Two-Stage Comparison

**Stage 1: Embedding Filter (Fast)**
```python
def filter_by_premise_embedding(vo_segment, candidates, top_k=50):
    """Fast filtering using premise embeddings."""
    vo_embedding = get_embedding(vo_segment.text)

    scored = []
    for video in candidates:
        premise_embedding = get_embedding(video.premise)
        similarity = cosine_similarity(vo_embedding, premise_embedding)
        scored.append((video, similarity))

    # Return top K by premise embedding similarity
    return sorted(scored, key=lambda x: -x[1])[:top_k]
```

**Stage 2: LLM Rerank (Accurate)**
```python
def llm_score_premise_match(vo_segment, video, premise):
    """LLM scores how well video premise fits voiceover."""

    prompt = f"""
    Rate how well this video fits the voiceover on a scale of 0-100.

    Voiceover text: "{vo_segment.text}"
    Video premise: "{premise}"

    Consider:
    - Does the video topic relate to what's being discussed?
    - Would this video make sense as B-roll for this voiceover?
    - Ignore word-for-word matching, focus on thematic fit.

    Score (0-100):
    """

    response = llm.generate(prompt)
    return parse_score(response) / 100.0  # Normalize to 0-1
```

### Combined Scoring

```python
def compute_match_score(vo_segment, video, config):
    """Compute final match score with new weights."""

    weights = config.matching.weights

    # Premise match (50%)
    premise_score = get_premise_score(vo_segment, video)

    # Keyword overlap (25%)
    keyword_score = compute_keyword_overlap(vo_segment, video)

    # Embedding distance (15%)
    embedding_score = compute_embedding_similarity(vo_segment, video)

    # Transcript similarity (10%)
    transcript_score = compute_transcript_similarity(vo_segment, video)

    # Weighted sum
    final_score = (
        weights.premise * premise_score +
        weights.keyword_overlap * keyword_score +
        weights.embedding * embedding_score +
        weights.transcript * transcript_score
    )

    return final_score
```

## Implementation Tasks

### Phase 1: Premise Extraction Stage
- [ ] Create `src/stages/premise.py`
- [ ] Add `PremiseStage` class
- [ ] Implement LLM premise extraction
- [ ] Add premise caching in `.cache/premises/`
- [ ] Add `premise` field to video dataclass
- [ ] Register stage in pipeline

### Phase 2: Premise Embeddings
- [ ] Generate embeddings for premises (reuse existing embedding infra)
- [ ] Store premise embeddings in cache
- [ ] Add fast filtering by premise embedding similarity

### Phase 3: Scoring Integration
- [ ] Add `premise_scoring` config section
- [ ] Implement `get_premise_score()` function
- [ ] Add LLM reranking for premise match
- [ ] Update `compute_match_score()` with new weights
- [ ] Adjust existing weights (transcript 40% → 10%)

### Phase 4: Testing & Tuning
- [ ] Test with music video case (cDTg1DBMx8s)
- [ ] Verify premise extraction quality
- [ ] Tune weights based on results
- [ ] Add premise info to match report

## Expected Outcomes

### Before (Current)
```
Voiceover: "24 hour sanctuary for night shift workers"
Match: cDTg1DBMx8s_0090.mp4 (Cueshe - 24 Hours lyrics)
Reason: transcript contains "24 hours" - HIGH similarity
```

### After (Premise-Based)
```
Voiceover: "24 hour sanctuary for night shift workers"
Video premise: "Music lyric video for a love song"
Premise match: LOW (music ≠ restaurants)
Result: NOT SELECTED

Better match: Documentary about late-night diners
Video premise: "Documentary about 24-hour restaurant culture"
Premise match: HIGH (restaurants = restaurants)
Result: SELECTED
```

## Edge Cases

1. **Stock footage with no transcript**
   - Premise from title + visual description only
   - Example: "B-roll footage of empty restaurant interior"

2. **Videos with misleading titles**
   - LLM sees transcript content, not just title
   - Music video with food in title still identified as music

3. **Generic B-roll**
   - Premise: "Generic stock footage of nature scenery"
   - Lower premise match to specific topics
   - Still usable but won't dominate

4. **Multiple topics in one video**
   - Premise captures primary topic
   - Scene-level matching handles sub-topics (future enhancement)
