# Context-Enriched Matching & Chapter Detection
## Problem Statement
Matching quality suffers because video titles, descriptions, tags, and YouTube chapter markers are available from yt-dlp but completely discarded during matching. Voiceover listicle structure ("first... second... next up...") is not detected, so clips within a numbered section scatter across unrelated video sources instead of maintaining thematic coherence. This wastes the richest metadata signal we have.
**Target files:** `src/matching/scoring.py`, `src/matching/llm_reranker.py`, `src/matching/tiered_matcher.py`, `src/matching/embedding_search.py`, `src/caption/models.py`, `src/caption_fetcher.py`, `src/state.py`, `src/stages/caption_stage.py`, `src/config/sections/matching.py`, `config.yaml`, `src/chapter_detection/`, `src/iterative_match/gap_analyzer.py`
**Acceptance patterns:** `test_match*.py`, `test_scoring*.py`, `test_context*.py`, `test_caption*.py`, `test_chapter*.py`, `test_listicle*.py`
---
## Area 1 — Video Metadata Extraction (Foundation)
**Goal:** Extract and persist video title, description, chapters, and tags from yt-dlp info_dict so downstream stages can use them.
### Implementation
1. **`src/caption_fetcher.py`** — In the yt-dlp `--dump-json` / info_dict handling:
   - Extract `info_dict['description']` (truncate to configurable max length, default 500 chars)
   - Extract `info_dict['chapters']` (list of `{title, start_time, end_time}`)
   - Extract `info_dict['tags']` (list of strings)
   - These fields already exist in yt-dlp output but are currently ignored
2. **`src/caption/models.py`** — Add fields to `CaptionResult`:
   - `video_description: str = ""`
   - `video_chapters: List[dict] = field(default_factory=list)` — each dict: `{title, start_time, end_time}`
   - `video_tags: List[str] = field(default_factory=list)`
3. **`src/state.py`** — Add `description: str = ""` field to `VideoSearchResult` and `DownloadedVideo`
4. **New dataclass `VideoChapter`** (in `src/caption/models.py` or new `src/chapter_detection/models.py`):
   ```python
   @dataclass
   class VideoChapter:
       title: str
       start_time: float
       end_time: float
       topic_keywords: List[str] = field(default_factory=list)
       segment_indices: List[int] = field(default_factory=list)
   ```
5. **Description chapter parsing** — When `info_dict['chapters']` is empty, parse chapter markers from description text using regex patterns like `(\d+:\d+)\s+(.+)` (common YouTube description format)
6. **Config** in `src/config/sections/matching.py` and `config.yaml`:
   ```yaml
   matching:
     context_enrichment:
       extract_video_description: true
       extract_video_chapters: true
       extract_video_tags: true
       max_description_length: 500
       parse_description_chapters: true
   ```
### Tests
- Unit test: extract description/chapters/tags from mock info_dict
- Unit test: parse chapters from description text when chapters field is empty
- Unit test: CaptionResult serialization with new fields (backward compat with existing cache)
---
## Area 2 — Context-Enriched Matching (Biggest Quality Impact)
**Goal:** Use video titles, descriptions, and tags to improve match confidence and ranking.
### Implementation
1. **Title-enriched embeddings** — In `src/stages/caption_stage.py` `_populate_text_metadata()`:
   - Prepend `"[{video_title}] "` to caption text before computing embeddings
   - This gives the embedding model title context without changing the caption text itself
   - Config toggle: `matching.context_enrichment.title_enriched_embeddings: true`
2. **Title similarity scoring** — New adjustment in `src/matching/scoring.py` `apply_all_adjustments()`:
   - Extract keywords from voiceover segment text and video title
   - Compute keyword overlap ratio
   - Graduated boost: 1 keyword match → +0.03, 2+ matches → +0.05, 3+ matches → +0.08
   - Add to `confidence_breakdown` as `"title_relevance"` entry
3. **Description context for LLM reranker** — In `src/matching/llm_reranker.py`:
   - When building candidate descriptions for the LLM prompt, add: `"Video context: {title}. {first_sentence_of_description}"`
   - This gives the LLM more signal about what the video is actually about
   - Truncate description to first sentence or 100 chars, whichever is shorter
4. **Tag-based keyword boost** — Reuse existing `keyword_boost` mechanism in scoring:
   - If voiceover segment keywords overlap with video tags, apply existing keyword boost
   - This requires no new scoring code, just feeding tags into the existing keyword system
5. **Audit trail** — All new adjustments must appear in `confidence_breakdown` dict so they're visible in logs and debug output
### Tests
- Unit test: title-enriched embedding text construction
- Unit test: title keyword overlap scoring with graduated boost values
- Unit test: LLM reranker prompt includes video context
- Unit test: confidence_breakdown contains title_relevance entry
- Integration test: matching with vs without context enrichment on fixture data
---
## Area 3 — Video-Side Chapter Detection
**Goal:** Map caption segments to their containing YouTube chapter, then use chapter topics as a matching signal.
### Implementation
1. **Chapter-segment mapping** — In caption stage or matching stage:
   - For each `CaptionSegment`, find the `VideoChapter` whose `[start_time, end_time)` contains it
   - Store chapter index/title on the segment or in a lookup dict
   - Handle edge cases: segments spanning chapter boundaries → assign to chapter with most overlap
2. **Chapter-aware scoring** — New adjustment in `src/matching/scoring.py`:
   - When both voiceover and video have chapter structure:
     - Compute topic similarity between voiceover chapter keywords and video chapter title/keywords
     - Topic match → boost +0.05 to +0.10 (graduated by similarity)
     - Topic mismatch → penalty -0.05
   - Add to `confidence_breakdown` as `"chapter_topic_match"` entry
3. **Chapter-level source consistency** — Soft boost in scoring:
   - Within a single voiceover chapter, prefer candidates from the same video source
   - Boost +0.03 for same video source as the previous segment in the same voiceover chapter
   - This overrides `consecutive_source_penalty` within chapter boundaries (diversity is less important when building a coherent section)
### Tests
- Unit test: chapter-segment mapping with various timestamp configurations
- Unit test: chapter topic similarity computation
- Unit test: source consistency boost within chapter boundaries
- Unit test: consecutive_source_penalty correctly overridden within chapters
---
## Area 4 — Voiceover Listicle Detection
**Goal:** Detect numbered/ordered structure in voiceover narration so segments can be grouped into thematic sections.
### Implementation
1. **New module `src/chapter_detection/listicle_detector.py`**:
   - Detect ordinal markers: "first", "second", "third", "finally", "lastly"
   - Detect numbered markers: "#1", "number one", "step 1", "item 1"
   - Detect transition markers: "next up", "moving on to", "let's talk about"
   - Detect list headers: "top 10", "5 reasons", "3 ways"
   - Return list of `ListicleGroup` objects
2. **`ListicleGroup` dataclass**:
   ```python
   @dataclass
   class ListicleGroup:
       group_id: int
       item_label: str  # e.g., "#1", "first", "step 3"
       start_segment_idx: int
       end_segment_idx: int
       topic_keywords: List[str] = field(default_factory=list)
   ```
3. **Integration** — Call listicle detector as additional pass in existing `src/chapter_detection/detector.py`:
   - If YouTube chapters exist, use those as primary structure
   - If no chapters, fall back to listicle detection from voiceover text
   - Both can coexist (voiceover structure + video chapter structure)
### Tests
- Unit test: detect ordinal markers in voiceover text
- Unit test: detect numbered markers (#1, step 1, etc.)
- Unit test: detect transition markers
- Unit test: detect list headers with count extraction
- Unit test: full listicle detection on sample voiceover SRT
- Edge case: mixed numbering styles ("first... #2... third...")
---
## Area 5 — Cross-Chapter Segment Grouping (Depends on Areas 3 + 4)
**Goal:** Cross-reference video chapters with voiceover chapters/listicle groups to constrain candidate pools and improve thematic coherence.
### Implementation
1. **Relevance matrix** — `voiceover_chapter × video_chapter → score`:
   - Compute topic keyword overlap between each pair
   - Also compute embedding similarity between chapter-level text (chapter title + first segment)
   - Normalize to 0-1 scale
2. **Chapter-constrained candidate pool** — In `src/matching/embedding_search.py`:
   - When voiceover has chapter structure, boost candidates from video chapters with high relevance score
   - Not a hard filter (still consider all candidates), but a soft boost proportional to relevance score
   - Config: `matching.chapter_grouping.relevance_boost_weight: 0.1`
3. **Chapter coherence scoring** — Track diversity within voiceover chapters:
   - Count unique video chapters used across segments in a voiceover chapter
   - Penalize excessive scatter (using >5 different video chapters in a single voiceover section)
   - Add to `confidence_breakdown` as `"chapter_coherence"` entry
### Tests
- Unit test: relevance matrix computation
- Unit test: chapter-constrained candidate boosting
- Unit test: coherence penalty for scattered sources
- Integration test: end-to-end matching with chapter grouping enabled
---
## Area 6 — Enhanced Iterative Matching with Context (Polish, Depends on Area 5)
**Goal:** Use chapter and context information to improve gap-filling in the iterative match phase.
### Implementation
1. **Chapter-aware gap analysis** — In `src/iterative_match/gap_analyzer.py`:
   - When identifying gaps, note which voiceover chapter the gap falls in
   - Prioritize filling gaps within high-importance chapters (e.g., introduction, conclusion)
2. **Description-derived search queries** — For gap filling:
   - Extract key phrases from video descriptions of already-matched videos
   - Use these phrases to generate more targeted search queries for gap filling
   - Config: `iterative_match.use_description_queries: true`
3. **Chapter-type-aware query learning** — In query learning:
   - Track which query types work best for different chapter types (intro, body, conclusion, listicle items)
   - Apply learned preferences when generating queries for similar chapter types
### Tests
- Unit test: chapter-aware gap analysis prioritization
- Unit test: description-derived query generation
- Unit test: chapter-type query learning storage and retrieval
---
## Dependency Chain
```
Area 1 (Metadata Extraction)
    ├── Area 2 (Context-Enriched Matching)  ── parallel
    ├── Area 3 (Video-Side Chapters)        ── parallel
    └── Area 4 (Voiceover Listicle)         ── parallel
              ├── Area 5 (Cross-Chapter Grouping)
              └── Area 6 (Enhanced Iterative Match)
```
## Config Summary
All new config lives under `matching.context_enrichment` and `matching.chapter_grouping`:
```yaml
matching:
  context_enrichment:
    extract_video_description: true
    extract_video_chapters: true
    extract_video_tags: true
    max_description_length: 500
    parse_description_chapters: true
    title_enriched_embeddings: true
  chapter_grouping:
    enabled: true
    relevance_boost_weight: 0.1
    coherence_penalty_threshold: 5
    source_consistency_boost: 0.03
    chapter_topic_match_boost: [0.05, 0.10]  # min, max
    chapter_topic_mismatch_penalty: -0.05

iterative_match:
  use_description_queries: true
```
