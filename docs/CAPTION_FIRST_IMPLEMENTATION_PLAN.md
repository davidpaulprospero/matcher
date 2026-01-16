# Caption-First Matching: Implementation Plan

> **Status**: Feature is **~90% implemented**. This document clarifies the current state, identifies gaps, and proposes solutions for remaining issues.

## Current vs Requested Architecture

### User's Expected Pipeline (13 stages):
```
ANALYZE → ENTITY_IMAGES → ENTITY_VIDEOS → DOWNLOAD → STOCK → BROLL_DOWNLOAD
→ REMIX → TRANSCRIBE → SCENE_DETECTION → MATCH → BROLL_MATCH → DOWNLOAD_SEGMENTS → OUTPUT
```

### Actual Current Pipeline (15 stages):
```
ANALYZE → ENTITY_IMAGES → ENTITY_VIDEOS → VIDEO_METADATA → CAPTION → DOWNLOAD
→ STOCK → BROLL_DOWNLOAD → REMIX → TRANSCRIBE → SCENE_DETECTION → MATCH
→ BROLL_MATCH → DOWNLOAD_SEGMENTS → OUTPUT
```

**Key Difference**: `VIDEO_METADATA` and `CAPTION` stages are inserted **BEFORE** `DOWNLOAD`, not after.

### Why Caption-Before-Download?

The current architecture places caption fetching **before** any media download because:

1. **Maximum bandwidth savings**: If we fetch captions first, we can skip downloading audio entirely for captioned videos
2. **Faster iteration**: Caption metadata (few KB) vs audio download (5-50MB) per video
3. **Segment-level download**: Only matched segments are downloaded after MATCH stage

**Trade-off**: Videos with captions don't have frames available for SCENE_DETECTION until after MATCH.

---

## What's Already Implemented ✅

### Core Components

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| `VideoMetadataStage` | `src/stages/video_metadata.py` | ✅ Complete | Searches YouTube, populates `state.video_candidates` |
| `CaptionStage` | `src/stages/caption.py` | ✅ Complete | Fetches captions via yt-dlp, sets `transcript_source` |
| `CaptionFetcher` | `src/downloader/caption_fetcher.py` | ✅ Complete | yt-dlp wrapper, SRT/VTT parsing |
| `CaptionFirstConfig` | `src/config/sections/download.py` | ✅ Complete | Full config dataclass |
| `CaptionDownload` | `src/state.py` | ✅ Complete | Dataclass for caption metadata |
| `VideoCandidate` | `src/state.py` | ✅ Complete | Pre-download video metadata |
| Checkpoint support | `src/checkpoint.py` | ✅ Complete | `CAPTION` in `STAGE_ORDER` |
| Pipeline integration | `src/pipeline.py` | ✅ Complete | Both `create_default_pipeline` and `create_match_only_pipeline` |
| Self-healing | `src/agents/healers/caption.py` | ✅ Complete | `CaptionHealer` with fallback chains |

### Data Flow (Implemented)

```
VIDEO_METADATA Stage
    ↓ state.video_candidates (VideoCandidate objects)
CAPTION Stage
    ├─→ CaptionFetcher.fetch_captions(video_id)
    ├─→ Parse SRT/VTT → segments with transcript_source='manual_caption'|'auto_caption'
    ├─→ state.transcripts[video_id] = segments
    └─→ state.videos_need_audio = [uncaptioned video_ids]
    ↓
DOWNLOAD Stage (modified)
    ├─→ Checks _is_caption_first_enabled()
    └─→ Only downloads for videos in state.videos_need_audio
    ↓
TRANSCRIBE Stage (modified)
    ├─→ _filter_captioned_videos() skips captioned videos
    ├─→ Whisper only for state.videos_need_audio
    └─→ Merges caption + Whisper transcripts, computes embeddings for ALL
    ↓
MATCH Stage
    └─→ apply_caption_boost() gives +0.1 to manual_caption sources
```

### Configuration (Already Available)

```yaml
download:
  caption_first:
    enabled: true                    # Master switch
    prefer_manual_captions: true     # Manual > auto-generated
    languages: ["en", "en-US", "en-GB"]
    fallback_to_audio: true          # Whisper for uncaptioned
    confidence_boost_manual: 0.1     # +0.1 for manual captions
    min_caption_coverage: 0.3        # Minimum coverage threshold
    fetch_timeout: 30                # Per-video timeout
    cache_captions: true             # Cache downloaded captions
```

---

## Identified Gaps & Proposed Solutions

### Gap 1: SCENE_DETECTION for Caption-First Videos ❌

**Problem**: Scene detection requires video frames, but caption-first videos aren't downloaded until after MATCH.

**Current Behavior**:
- `SceneDetectionStage._get_video_files()` only finds `state.downloaded_videos`
- Caption-first videos aren't in `downloaded_videos` until `DOWNLOAD_SEGMENTS` stage (after MATCH)
- Result: No scene detection for caption-first videos

**Proposed Solutions**:

#### Option A: Deferred Scene Detection (Recommended)
Add a second scene detection pass after `DOWNLOAD_SEGMENTS`:

```python
# In STAGE_ORDER (checkpoint.py)
STAGE_ORDER = [
    ...
    "MATCH",
    "BROLL_MATCH",
    "DOWNLOAD_SEGMENTS",
    "SCENE_DETECTION_POST",  # NEW: Scene detection for caption-first segments
    "OUTPUT"
]
```

**Files to Modify**:
- `src/checkpoint.py`: Add `SCENE_DETECTION_POST` to `STAGE_ORDER`
- `src/stages/scene_detection.py`: Create `SceneDetectionPostStage` class that processes `state.matched_segments`
- `src/pipeline.py`: Add stage conditionally when caption-first enabled

**Complexity**: Medium (~100 lines)

#### Option B: Skip Scene Detection for Caption-First Videos
Mark caption-first videos as having "unknown" scene data:

```python
# In scene_detection.py
if video_id in caption_first_video_ids:
    # Use transcript timing as pseudo-scenes
    scene_data = self._create_transcript_based_scenes(transcript_segments)
```

**Complexity**: Low (~30 lines)

---

### Gap 2: STOCK/BROLL Videos Have No Transcripts ⚠️

**Problem**: Stock videos (Pexels/Pixabay) and B-roll downloads are silent footage with no transcripts.

**Current Behavior**:
- `StockVideoStage`: Downloads from Pexels/Pixabay → `state.downloaded_videos` with `source='stock'`
- `BrollDownloadStage`: Downloads B-roll → `state.downloaded_videos` with `source='broll'`
- Neither has transcript data in `state.transcripts`

**Question**: Should STOCK/BROLL use "pseudo-transcripts" for matching?

**Proposed Solutions**:

#### Option A: Metadata-Based Pseudo-Transcripts (Recommended)
Generate pseudo-transcripts from video metadata:

```python
# In TranscribeStage or new PseudoTranscriptStage
def _generate_pseudo_transcript(video: DownloadedVideo) -> List[dict]:
    """Generate pseudo-transcript from metadata for silent videos"""
    if video.source in ['stock', 'broll']:
        return [{
            'text': f"{video.title} {video.keyword}",  # Title + search keyword
            'start_time': 0.0,
            'end_time': video.duration,
            'transcript_source': 'metadata',  # New source type
            'is_broll': True,
        }]
```

**Files to Modify**:
- `src/stages/transcribe.py`: Add `_generate_pseudo_transcripts()` method
- `src/state.py`: Document `transcript_source='metadata'` value

**Complexity**: Low (~40 lines)

#### Option B: Vision API Descriptions (Already Partially Implemented)
Use Gemini Vision to generate descriptions:

```yaml
vision:
  enabled: true
  provider: gemini
```

**Current State**: Vision API is already implemented for silent videos in `SceneDetectionStage`.

**Files Involved**: `src/vision_service.py`, `src/stages/scene_detection.py`

---

### Gap 3: REMIX Stage Doesn't Use Caption Transcripts ⚠️

**Problem**: REMIX filters by filename/keyword matching, not by transcript content.

**Current Behavior** (from `src/stages/remix.py`):
```python
# Filters downloaded_videos or downloaded_audio
# Uses keyword matching against filenames, not transcript text
```

**Question**: Should REMIX filter using caption transcript content?

**Proposed Solutions**:

#### Option A: Add Caption-Aware Filtering (Recommended)
Extend REMIX to check caption transcripts for keyword relevance:

```python
def _score_video_relevance(self, video_id: str, keywords: List[str], state) -> float:
    """Score video relevance including caption content"""
    score = 0.0

    # Existing: filename/title matching
    score += self._filename_keyword_score(video_id, keywords)

    # NEW: Caption transcript matching
    if video_id in state.transcripts:
        transcript_text = ' '.join(seg['text'] for seg in state.transcripts[video_id])
        score += self._transcript_keyword_score(transcript_text, keywords)

    return score
```

**Files to Modify**:
- `src/stages/remix.py`: Add caption-aware scoring
- `src/keyword_remix.py`: Update `score_video_relevance()` function

**Complexity**: Medium (~50 lines)

#### Option B: Move REMIX After CAPTION Stage
Current order: `REMIX` is at position 9, after `CAPTION` (position 5).

**Status**: ✅ Already correct - REMIX runs after CAPTION, so transcripts ARE available.

**Fix Needed**: Just add transcript-aware scoring to existing REMIX logic.

---

### Gap 4: Confidence Boost Not Applied in All Paths

**Problem**: `apply_caption_boost()` exists but may not be called in all matching code paths.

**Current State**: Verified in `src/matching/tiered_matcher.py`:
- ✅ Line 381-383: Applied in high-confidence embedding match
- ✅ Line 449-451: Applied in cached LLM response
- ✅ Line 568-570: Applied in LLM matching result

**Status**: ✅ Complete - caption boost is applied in all three matching paths.

---

## Implementation Priority

### Phase 1: Verify & Test (Immediate) ✅
- [x] Run existing caption tests: `pytest tests/test_caption*.py`
- [x] Create integration tests: `tests/test_caption_first_integration.py`
- [x] Verify confidence boost is applied

### Phase 2: REMIX Caption Awareness (Low Effort)
- [ ] Add transcript-based scoring to `RemixStage`
- [ ] Test with caption-first enabled

### Phase 3: Pseudo-Transcripts for STOCK/BROLL (Medium Effort)
- [ ] Add metadata-based pseudo-transcripts for silent videos
- [ ] Ensure embeddings are computed for pseudo-transcripts
- [ ] Update matching to handle `transcript_source='metadata'`

### Phase 4: Post-Match Scene Detection (Medium Effort)
- [ ] Create `SceneDetectionPostStage` for caption-first segments
- [ ] Add to pipeline conditionally
- [ ] Update B-roll marking for post-download videos

---

## File Modification Summary

### Files Already Modified ✅
| File | Change |
|------|--------|
| `src/checkpoint.py` | Added `VIDEO_METADATA`, `CAPTION` to `STAGE_ORDER` |
| `src/pipeline.py` | Added `VideoMetadataStage`, `CaptionStage` |
| `src/stages/caption.py` | Full implementation (405 lines) |
| `src/stages/transcribe.py` | Caption-aware filtering, transcript merging |
| `src/stages/download.py` | Caption-first skip logic |
| `src/matching/scoring.py` | `apply_caption_boost()` function |
| `src/matching/tiered_matcher.py` | Caption boost integration |
| `src/state.py` | `CaptionDownload`, `VideoCandidate`, `transcript_source` field |
| `src/config/sections/download.py` | `CaptionFirstConfig` dataclass |
| `src/downloader/caption_fetcher.py` | yt-dlp integration (644 lines) |
| `src/agents/healers/caption.py` | `CaptionHealer` (366 lines) |
| `src/cache/caption.py` | Caption cache (200+ lines) |

### Files Needing Modification
| File | Change | Priority |
|------|--------|----------|
| `src/stages/remix.py` | Add caption transcript scoring | P2 |
| `src/keyword_remix.py` | Update scoring function | P2 |
| `src/stages/transcribe.py` | Add pseudo-transcript generation | P3 |
| `src/stages/scene_detection.py` | Create post-match variant | P4 |
| `src/checkpoint.py` | Add `SCENE_DETECTION_POST` (optional) | P4 |

### New Files (Optional)
| File | Purpose | Priority |
|------|---------|----------|
| `src/stages/scene_detection_post.py` | Post-download scene detection | P4 |

---

## Breaking Changes

### No Breaking Changes to Existing Interfaces ✅

The caption-first implementation is **additive**:
- Disabled by default (`caption_first.enabled: false`)
- New stages are skipped when disabled
- Existing `state.transcripts` structure unchanged
- `transcript_source` field is optional (defaults to `'whisper'`)

### Backward Compatibility

| Interface | Status | Notes |
|-----------|--------|-------|
| `state.transcripts` | ✅ Compatible | Same structure, new optional field |
| `state.downloaded_videos` | ✅ Compatible | Caption-first videos added after MATCH |
| `STAGE_ORDER` | ✅ Compatible | New stages added, existing order preserved |
| Config files | ✅ Compatible | New `caption_first` section is optional |

---

## Testing Checklist

### Unit Tests (Existing)
- [x] `tests/test_caption_fetcher.py` - Caption parsing, yt-dlp mocking
- [x] `tests/test_caption_stage.py` - Stage behavior, checkpoint handling
- [x] `tests/test_caption_healer.py` - Self-healing recovery

### Integration Tests (Created)
- [x] `tests/test_caption_first_integration.py` - End-to-end flow validation

### Manual Testing
- [ ] Run pipeline with `caption_first.enabled: true`
- [ ] Verify bandwidth savings (should see ~95% reduction)
- [ ] Check OTIO output includes caption-sourced matches
- [ ] Test fallback to Whisper for uncaptioned videos

---

## Summary

**Caption-first matching is ~90% implemented.** The core flow works:

1. ✅ `VIDEO_METADATA` searches YouTube metadata
2. ✅ `CAPTION` fetches captions via yt-dlp
3. ✅ `DOWNLOAD` skips captioned videos
4. ✅ `TRANSCRIBE` merges caption + Whisper transcripts
5. ✅ `MATCH` applies confidence boost for manual captions
6. ✅ `DOWNLOAD_SEGMENTS` downloads only matched segments

**Remaining gaps** (all low-medium effort):
1. REMIX should use caption transcript content for scoring
2. STOCK/BROLL need pseudo-transcripts from metadata
3. Optional: Post-match scene detection for caption-first videos

**No breaking changes** - feature is disabled by default and fully backward compatible.
