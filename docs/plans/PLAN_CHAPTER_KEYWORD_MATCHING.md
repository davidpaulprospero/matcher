# Chapter-Aware Keyword Matching

## Problem Statement

### User Feedback (markers.edl)

| Timecode | Marker | Note |
|----------|--------|------|
| 01:03:28:20 | Blue | "Find related to dennys, probably fix up the chapter integration" |
| 01:03:31:09 | Blue | "not related to the chapter, chapter fixing neaded" |
| 01:03:33:05 | Blue | "should be related to wallburgers, a search term each chapter would solve this" |

### Root Cause Analysis (2026-01-21)

Testing on `E:\Edit Job\Stu\January\22__2026-01-17` (15 Fast Food Chains Dying listicle):

**Chapter Detection Results:**
- Expected: 15 list items detected
- Actual: **2/15 detected** (Denny's, Wallburgers only)
- Detection rate: **13%**

**Why Detection Failed:**

| List Item | Problem |
|-----------|---------|
| #15 Denny's | ✓ Detected |
| #14 Wallburgers | ✓ Detected |
| #13 salad and go | Regex requires capital letter, ASR has lowercase 's' |
| #12 Boston Market | Beyond 8000 char truncation limit |
| #11 little Caesars | lowercase 'l' + extra text "for decades" |
| #10 Taco Bell | Beyond truncation |
| #9 Popeyes | Word number "Number nine" not matched |
| #8 Jack in the Box | Word number + beyond truncation |
| #7 dairy queen | lowercase 'd' + word number |
| #6 KFC | Word number |
| #5 Pizza Hut | ASR error: "Keatsahut" |
| #4 subway | lowercase 's' + word number |
| #3 Burger King | Word number |
| #2 Wendy's | Word number |
| #1 McDonald's | Word number |

**Regex Limitations:**
1. **8000 char truncation** - `text[:8000]` in `core.py:193` cuts off items 12-1
2. **Case sensitivity** - `[A-Z]` pattern misses ASR lowercase
3. **Word vs digit numbers** - Only handles "Number 10", not "Number ten"
4. **ASR transcription errors** - Can't correct "Keatsahut" → "Pizza Hut"
5. **Extra text after entity** - "little Caesars for decades" fails pattern

**Core Issue:** Clips aren't matching chapter context - when voiceover is about "Denny's", matched clips should be Denny's-related, not generic food clips.

## Proposed Solution: LLM-Based Chapter Detection

### Why LLM?

Regex is fundamentally brittle for this task. LLM-based detection handles:
- Case variations ("little Caesars" vs "Little Caesars")
- Word/digit number variants ("Number nine" vs "Number 9")
- ASR correction ("Keatsahut" → "Pizza Hut", "Wallburgers" → "Wahlburgers")
- Sentence structure variations ("Number 11, little Caesars for decades,")
- Context understanding (knows "KFC" is "Kentucky Fried Chicken")

### New Architecture: CHAPTER_DETECT Stage

Add dedicated stage before ANALYZE:

```
CHAPTER_DETECT → ANALYZE → ENTITY_IMAGES → DOWNLOAD → ...
```

### LLM Prompt for Listicle Detection

```
Analyze this voiceover script and detect if it contains a list/ranking structure.

If it's a listicle/ranking video, extract EVERY list item with:
1. Rank/position (number)
2. Entity name (CORRECTED for likely ASR errors)
3. Start segment index
4. End segment index (when next item starts)

Common ASR corrections to apply:
- "Keatsahut" → "Pizza Hut"
- "Wallburgers" → "Wahlburgers"
- "dairy queen" → "Dairy Queen"
- "little Caesars" → "Little Caesars"
- "salad and go" → "Salad and Go"

Script (with segment indices):
{indexed_segments}

Return JSON:
{
  "is_listicle": true,
  "list_type": "countdown",  // countdown, top_n, ranked, unordered
  "total_items": 15,
  "items": [
    {
      "rank": 15,
      "name": "Denny's",
      "corrected_name": "Denny's",
      "start_segment": 17,
      "end_segment": 80,
      "keywords": ["Denny's restaurant", "grand slam breakfast", "24 hour diner"]
    },
    {
      "rank": 14,
      "name": "Wallburgers",
      "corrected_name": "Wahlburgers",  // ASR correction
      "start_segment": 81,
      "end_segment": 130,
      "keywords": ["Wahlburgers restaurant", "Mark Wahlberg", "celebrity burger"]
    }
    // ... all 15 items
  ]
}
```

### Per-Chapter Keyword Extraction

Instead of global keywords, extract per-chapter:

```python
@dataclass
class ChapterKeywords:
    chapter_name: str           # "Denny's"
    corrected_name: str         # "Denny's" (same, or ASR-corrected)
    rank: int                   # 15
    start_segment: int          # 17
    end_segment: int            # 80
    entity_keywords: List[str]  # ["Denny's restaurant exterior", "Denny's sign"]
    topic_keywords: List[str]   # ["24 hour diner", "grand slam breakfast"]
    fallback_keywords: List[str]  # ["american diner", "late night restaurant"]
```

### Chapter-Scoped Video Downloads

```python
# Current (broken): Global search
keywords = ["American fast food dream", "backbone affordable meals", ...]
download_videos(keywords)  # Generic results

# New: Per-chapter search
for chapter in chapters:
    search_queries = [
        f"{chapter.corrected_name} restaurant",  # "Denny's restaurant"
        f"{chapter.corrected_name} exterior",    # "Denny's exterior"
        chapter.entity_keywords[0],              # "Denny's sign"
    ]
    download_videos(search_queries, chapter_tag=chapter.name)
```

### Chapter-Aware Matching

```python
def score_match(segment, video, chapter):
    base_score = embedding_similarity(segment, video)

    # Big boost for same-chapter video
    if video.chapter_tag == chapter.name:
        base_score *= 1.5

    # Extra boost for entity in video title/transcript
    if chapter.corrected_name.lower() in video.title.lower():
        base_score *= 2.0

    # Penalty for different-chapter video
    elif video.chapter_tag and video.chapter_tag != chapter.name:
        base_score *= 0.5

    return base_score
```

### Implementation Files

| New Files | Purpose |
|-----------|---------|
| `src/chapter_detector/core.py` | LLM-based chapter detection |
| `src/chapter_detector/prompts.py` | Detection prompts |
| `src/chapter_detector/models.py` | Chapter dataclasses |
| `src/stages/chapter_detect.py` | Pipeline stage |

| Modified Files | Changes |
|----------------|---------|
| `src/state.py` | Add `chapters: List[Chapter]` to PipelineState |
| `src/keyword_extractor/core.py` | Accept chapter context, per-chapter extraction |
| `src/keyword_extractor/entity_extractor.py` | Remove regex, use LLM chapter data |
| `src/stages/analyze.py` | Call chapter-scoped extraction |
| `src/stages/download.py` | Tag videos with source chapter |
| `src/matching/scoring.py` | Chapter-aware scoring |
| `src/matching/tiered_matcher.py` | Chapter context in matching |

### Config

```yaml
chapter_detection:
  enabled: true
  model: "gemini"  # LLM for detection

  # Detection settings
  detect_listicles: true
  detect_topic_chapters: true  # For non-listicle multi-topic videos
  asr_correction: true

  # Per-chapter settings
  keywords_per_chapter: 5
  videos_per_chapter: 10
  max_chapters: 20

  # Matching settings
  same_chapter_boost: 1.5
  entity_match_boost: 2.0
  different_chapter_penalty: 0.5
```

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| List items detected | 2/15 (13%) | 15/15 (100%) |
| Segments with chapter-relevant match | ~30% | >80% |
| Entity-specific videos downloaded | 0 | 3+ per chapter |
| Cross-chapter bleeding | High | <10% |

---

## Current Architecture (Legacy)

### Chapter Detection Flow
```
ANALYZE Stage → EnhancedChapterDetector → ChapterCandidate
                                         ├── visual_keywords: ["Denny's sign", "diner"]
                                         ├── context_keywords: ["american food", "chain restaurant"]
                                         └── topics: ["fast food", "restaurants"]
```

### Current Keyword Usage

| Usage | Where | Keywords Used |
|-------|-------|---------------|
| Location disambiguation | `chapter_detection/detector.py:280` | visual_keywords + context_keywords |
| Topic mismatch penalty | `matching/scoring.py:47` | chapter.topics only |
| Video downloads | `stages/download.py` | state.keywords (NOT chapter-specific) |
| Matching boost | ❌ NOT IMPLEMENTED | - |

### Gap Analysis

1. **Downloads use global keywords** - not chapter-specific terms like "Denny's" or "Wahlburgers"
2. **Scoring only applies penalties** - no positive boost for chapter keyword matches
3. **chapter.visual_keywords/context_keywords unused in matching** - only used for location disambiguation

## Solution Design

### Phase 1: Chapter Keyword Boost in Matching (Immediate Fix)

Add positive confidence boost when video content matches chapter keywords.

**File:** `src/matching/scoring.py`

```python
def apply_chapter_keyword_boost(
    confidence: float,
    vo_segment: SRTSegment,
    video_segment: SRTSegment,
    chapter_keywords: List[str],  # visual_keywords + context_keywords + topics
    config
) -> Tuple[float, str]:
    """
    Apply confidence boost when video matches chapter keywords.

    Checks:
    1. Video transcript contains chapter keywords
    2. Video filename contains chapter keywords
    3. Video source_keyword matches chapter
    """
    if not chapter_keywords:
        return confidence, ""

    # Get video text content
    video_text = video_segment.text.lower()
    video_file = Path(video_segment.source_file).stem.lower()
    source_keyword = getattr(video_segment, 'source_keyword', '').lower()

    # Check for keyword matches
    matches = []
    for kw in chapter_keywords:
        kw_lower = kw.lower()
        if kw_lower in video_text or kw_lower in video_file or kw_lower in source_keyword:
            matches.append(kw)

    if not matches:
        return confidence, ""

    # Boost based on number of matches (max 0.15)
    boost = min(0.15, len(matches) * 0.05)
    boosted = min(1.0, confidence + boost)

    return boosted, f"chapter keyword boost: +{boost:.2f} ({', '.join(matches[:3])})"
```

**Integration Point:** `src/matching/tiered_matcher.py:match_segment()`

```python
# After existing scoring adjustments
if segment_idx is not None and self.location_chapters:
    chapter = self._get_chapter_for_segment(segment_idx)
    if chapter:
        chapter_keywords = (
            chapter.get('visual_keywords', []) +
            chapter.get('context_keywords', []) +
            chapter.get('topics', []) +
            [chapter.get('title', '')]  # Include chapter title
        )
        for i, (seg, conf) in enumerate(valid_candidates):
            boosted_conf, reason = apply_chapter_keyword_boost(
                conf, vo_segment, seg, chapter_keywords, self.config
            )
            if boosted_conf != conf:
                valid_candidates[i] = (seg, boosted_conf)
```

### Phase 2: Chapter-Specific Downloads (Future Enhancement)

Download additional videos using chapter-specific search terms.

**Approach:** During ANALYZE stage, collect all unique chapter titles/keywords and add them to `state.keywords`.

**File:** `src/stages/analyze.py`

```python
# After chapter detection
if state.location_chapters:
    chapter_keywords = set()
    for ch in state.location_chapters:
        # Extract chapter-specific terms
        if ch.get('title'):
            chapter_keywords.add(ch['title'])
        chapter_keywords.update(ch.get('visual_keywords', []))
        chapter_keywords.update(ch.get('context_keywords', []))

    # Add to state keywords (deduped)
    existing = set(state.keywords)
    new_keywords = chapter_keywords - existing
    if new_keywords:
        state.keywords.extend(list(new_keywords)[:10])  # Max 10 new
        logger.info(f"Added {len(new_keywords)} chapter keywords to downloads")
```

## Implementation Priority

| Phase | Priority | Effort | Impact |
|-------|----------|--------|--------|
| 1: Chapter keyword boost | HIGH | 2-3 hours | Immediate fix for matching |
| 2: Chapter-specific downloads | MEDIUM | 4-6 hours | Better video pool |

## Testing

### Test Case 1: Denny's Chapter
```python
# Voiceover segment in "Denny's" chapter
vo_segment.text = "The classic American diner experience at Denny's"
chapter_keywords = ["Denny's", "diner", "American food", "breakfast"]

# Video A: Generic food (no boost)
video_a.text = "cooking tutorial for eggs and bacon"
video_a.source_file = "cooking_eggs_001.mp4"
# Expected: no boost

# Video B: Denny's related (boost)
video_b.text = "inside a Denny's restaurant"
video_b.source_file = "dennys_restaurant_tour.mp4"
# Expected: +0.10 boost (2 keyword matches)
```

### Test Case 2: Wahlburgers Chapter
```python
chapter_keywords = ["Wahlburgers", "burger", "Mark Wahlberg", "Boston"]

# Video with burger content + Wahlberg
# Expected: +0.10 boost
```

## Config

```yaml
matching:
  chapter_matching:
    enabled: true
    keyword_boost: 0.05          # Per keyword match
    max_keyword_boost: 0.15      # Cap total boost
    include_title: true          # Use chapter title as keyword
    include_visual: true         # Use visual_keywords
    include_context: true        # Use context_keywords
```

## Files to Modify

1. `src/matching/scoring.py` - Add `apply_chapter_keyword_boost()`
2. `src/matching/tiered_matcher.py` - Call boost function during scoring
3. `src/config/sections/matching.py` - Add config options
4. `config.yaml` - Document new options

## Success Criteria

1. Videos with chapter keywords in transcript/filename get confidence boost
2. Clips about "Denny's" appear higher when voiceover is about Denny's
3. No performance regression (keyword matching is fast)
4. Backwards compatible (disabled by default or seamlessly enabled)
