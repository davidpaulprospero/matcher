# Phase 10.4: SceneDetectionStage Tests - COMPLETED

**Date:** 2026-01-09
**Status:** ✅ COMPLETE
**Test File Created:** `tests/test_stage_scene_detection.py`
**Tests Added:** 38
**Pass Rate:** 100% (38/38)

---

## Overview

Phase 10.4 implements comprehensive unit tests for the SceneDetectionStage class, covering scene boundary detection, face detection per scene, B-roll classification, silent video handling, Vision API integration, and scene metadata merging. This is the fourth of 10 stage classes to be tested in Phase 10.

## Test Coverage

### Test Structure (38 tests across 10 test classes)

#### 1. TestSceneDetectionStageInit (3 tests)
- `test_stage_name` - Verifies stage name is "SCENE_DETECTION"
- `test_stage_description` - Verifies stage description
- `test_stage_registration` - Confirms stage is registered in stage registry

#### 2. TestInputValidation (3 tests)
- `test_validate_no_videos` - Error when no videos available
- `test_validate_with_downloaded_videos` - Successful validation with downloaded videos
- `test_validate_with_downloaded_audio` - Successful validation with downloaded audio

#### 3. TestGetVideoFiles (3 tests)
- `test_get_files_from_downloaded_videos` - Extract video file paths from state
- `test_get_files_empty_state` - Handle empty state gracefully
- `test_get_files_handles_missing_file_attribute` - Handle videos without file attribute

#### 4. TestSceneDetectionProcessing (5 tests)
- `test_process_single_video_success` - Successful scene detection with B-roll classification
- `test_process_multiple_videos` - Process multiple videos simultaneously
- `test_process_video_detection_failure` - Handle when SceneDetector returns None
- `test_process_video_exception_handling` - Graceful exception handling per video

#### 5. TestMergeSceneDataToTranscripts (4 tests)
- `test_merge_no_transcripts` - Handle empty transcripts
- `test_merge_with_transcripts` - Merge scene metadata into transcript segments
- `test_merge_broll_segment` - Correctly propagate B-roll flags (face_score < 0.3)
- `test_merge_updates_text_metadata` - Update text_metadata when embeddings exist

**Key Pattern**: Scene metadata merging uses midpoint matching - for each segment, finds scene where `scene.start_time <= segment_midpoint <= scene.end_time` and copies `is_broll`, `face_score`, `scene_index`.

#### 6. TestSilentVideoHandling (2 tests)
- `test_create_text_metadata_no_silent_videos` - Skip when all videos have transcripts
- `test_create_text_metadata_for_silent_video` - Create text_metadata entries for silent videos (stock footage)
- `test_create_text_metadata_with_vision_api` - Generate semantic descriptions via Vision API

**Critical Feature**: Silent videos (scene_data exists but no transcripts) need manual text_metadata creation for matching. Without this, silent stock footage would be invisible to the matching stage.

#### 7. TestSilentVideoEmbeddings (3 tests)
- `test_compute_embeddings_for_silent_videos` - Compute embeddings and rebuild FAISS index
- `test_compute_embeddings_no_existing_embeddings` - Handle when embeddings don't exist
- `test_compute_embeddings_empty_entries` - Handle empty entry list

**Technical Detail**: Silent video embeddings are computed after main transcription, so they must be:
1. Computed separately with SentenceTransformer
2. Stacked onto existing embeddings array
3. FAISS index rebuilt with all embeddings

#### 8. TestSkipSceneDetection (3 tests)
- `test_skip_when_configured` - Skip when `pipeline.skip_scene_detection: true`
- `test_can_skip_no_checkpoint` - Returns False when no checkpoint
- `test_can_skip_with_checkpoint` - Returns True when checkpoint exists

#### 9. TestSceneDetectionStageExecution (3 tests)
- `test_run_no_videos` - Handles empty video list gracefully
- `test_run_success_full_pipeline` - Full pipeline with transcripts, embeddings, text_metadata updates
- `test_run_exception_handling` - Exception handling in main run method

#### 10. TestSceneDetectionCheckpoint (4 tests)
- `test_restore_no_data` - Returns False when no checkpoint data
- `test_restore_skipped_stage` - Returns False when stage was skipped
- `test_restore_success` - Successful restore from checkpoint
- `test_restore_exception_handling` - Handles restore errors gracefully

#### 11. TestSceneDetectionEdgeCases (7 tests)
- `test_segment_without_dict_attribute` - Handle dict-based segments (no `__dict__`)
- `test_text_metadata_non_dict_entries` - Skip non-dict entries in text_metadata
- `test_video_name_mismatch` - Handle when video names don't match
- `test_segment_outside_scene_bounds` - Handle segments outside all scene boundaries
- `test_vision_api_initialization_failure` - Fallback to placeholder text when Vision API fails

---

## Key Testing Patterns

### 1. Mock SceneInfo and VideoSceneData
```python
@pytest.fixture
def mock_scene_info():
    """Create mock SceneInfo dataclass"""
    @dataclass
    class MockSceneInfo:
        scene_index: int
        start_frame: int
        end_frame: int
        start_time: float
        end_time: float
        duration: float
        face_score: float = 0.5
        is_broll: bool = False

    return MockSceneInfo

# Usage
scene1 = mock_scene_info(0, 0, 100, 0.0, 5.0, 5.0, face_score=0.8, is_broll=False)
scene2 = mock_scene_info(1, 100, 200, 5.0, 10.0, 5.0, face_score=0.1, is_broll=True)
```

### 2. SceneDetector Mocking
```python
@patch('src.scene_detection.SceneDetector')
def test_process_video(self, mock_detector_class, ...):
    # Mock the SceneDetector instance
    mock_detector = Mock()
    mock_detector.process_video.return_value = scene_data
    mock_detector_class.return_value = mock_detector

    # Run stage
    result = stage.run(state, mock_config, mock_checkpoint)
```

### 3. Vision API Mocking
```python
@patch('src.vision.VisionProcessor')
def test_with_vision_api(self, mock_vision_class, ...):
    mock_vision = Mock()
    mock_vision.is_available.return_value = True
    mock_vision.describe_scene.return_value = "A person walking on the beach"
    mock_vision.get_stats.return_value = {'estimated_cost': 0.001}
    mock_vision_class.return_value = mock_vision
```

### 4. Embeddings and FAISS Index Mocking
```python
@patch('sentence_transformers.SentenceTransformer')
@patch('faiss.IndexFlatL2')
def test_compute_embeddings(self, mock_index_class, mock_model_class):
    import numpy as np

    # Existing embeddings
    state.embeddings = np.array([[0.1, 0.2, 0.3]])

    # Mock model
    mock_model = Mock()
    mock_model.encode.return_value = np.array([[0.7, 0.8, 0.9]])
    mock_model_class.return_value = mock_model

    # Mock FAISS index
    mock_index = Mock()
    mock_index_class.return_value = mock_index
```

---

## Technical Challenges Resolved

### Challenge 1: Mock hasattr() Behavior
**Problem:** Mock objects always return True for `hasattr()` checks, making it impossible to test "attribute not set" scenarios.

**Initial Error:**
```python
segment = Mock()
segment.start_time = 1.0

# After merge with mismatched video name, segment should NOT have is_broll
assert not hasattr(segment, 'is_broll')  # ❌ FAILS - Mock has all attributes
```

**Solution:** Use `SimpleNamespace` instead of Mock for objects that need proper attribute checking:
```python
from types import SimpleNamespace
segment = SimpleNamespace(start_time=1.0, end_time=3.0, text="test")

# Now hasattr works correctly
assert not hasattr(segment, 'is_broll')  # ✅ PASSES
```

**Lesson Learned:** When testing attribute presence/absence, use real objects or SimpleNamespace, not Mock.

### Challenge 2: Silent Video Embedding Stacking
**Problem:** Silent videos need embeddings computed after main transcription, requiring numpy array stacking and FAISS index rebuilding.

**Solution:** Test verifies proper stacking and index rebuilding:
```python
# Original embeddings (2 vectors)
state.embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

# New entries (2 vectors)
new_entries = [{'text': 'Scene 1'}, {'text': 'Scene 2'}]

# After stacking, should have 4 vectors
stage._compute_embeddings_for_silent_videos(state, new_entries)
assert state.embeddings.shape[0] == 4
```

### Challenge 3: Vision API Integration Testing
**Problem:** Vision API integration involves multiple dependencies (VisionProcessor, API availability check, scene description, cost tracking).

**Solution:** Comprehensive mocking of entire Vision API workflow:
```python
mock_vision = Mock()
mock_vision.is_available.return_value = True
mock_vision.describe_scene.return_value = "Semantic description"
mock_vision.get_stats.return_value = {'estimated_cost': 0.001}
mock_vision_class.return_value = mock_vision
```

### Challenge 4: Scene Metadata Propagation
**Problem:** Scene metadata must be propagated to THREE locations:
1. Transcript segments (via `segment.is_broll = scene.is_broll`)
2. text_metadata entries (via `meta['is_broll'] = transcript.is_broll`)
3. Silent video text_metadata (directly from scene_data)

**Solution:** Tests cover all three propagation paths:
- `test_merge_with_transcripts` - Transcript segment propagation
- `test_merge_updates_text_metadata` - text_metadata update from transcripts
- `test_create_text_metadata_for_silent_video` - Direct scene_data propagation for silent videos

---

## Test Results

```bash
pytest tests/test_stage_scene_detection.py -v

============================= 38 passed in 5.85s ================================
Pass Rate: 100% ✅
```

**Total Test Count:**
- Before Phase 10.4: 1,489 tests
- After Phase 10.4: 1,527 tests (+38)

---

## Coverage Impact

**Module:** `src/stages/scene_detection.py` (~486 lines)
**Tests:** 38 comprehensive unit tests

**Coverage breakdown:**
- ✅ Stage initialization and registration
- ✅ Input validation (no videos, with videos, with audio)
- ✅ Video file extraction from state
- ✅ Scene detection processing (single video, multiple videos, failures)
- ✅ Scene metadata merging (transcripts, text_metadata updates)
- ✅ B-roll classification (face_score < 0.3)
- ✅ Silent video handling (text_metadata creation, Vision API integration)
- ✅ Embedding computation for silent videos (stacking, FAISS rebuild)
- ✅ Skip detection behavior (config flag, checkpoint)
- ✅ Stage execution (full pipeline, exceptions)
- ✅ Checkpoint operations (restore, skip, errors)
- ✅ Edge cases (dict segments, video mismatch, out-of-bounds segments, Vision API failures)

**Expected Coverage:** 80-90% (from baseline)

---

## Next Steps

### Phase 10.5: MatchStage
**File:** `tests/test_stage_match.py` (new)
**Expected tests:** ~60 tests

**Key areas to cover:**
- TieredMatcher with multiple strategies
- Embedding-based matching (FAISS search)
- LLM-based refinement (Gemini, Anthropic, Ollama)
- Location filtering (GeoNames integration)
- Diversity scoring (require_different_source)
- Match deduplication logic
- Confidence threshold filtering
- Checkpoint saving/loading

**Critical files:** `src/stages/match.py`, `src/matching/main.py`, `src/matching/strategies.py`, `src/matching/tiered_matcher.py`

---

## Remaining Phase 10 Stages

| Stage | File | Tests Needed | Status |
|-------|------|--------------|--------|
| ✅ ANALYZE | test_stage_analyze.py | 45 | **COMPLETE** |
| ✅ DOWNLOAD | test_stage_download.py | 51 | **COMPLETE** |
| ✅ TRANSCRIBE | test_stage_transcribe.py | 39 | **COMPLETE** |
| ✅ SCENE_DETECTION | test_stage_scene_detection.py | 38 | **COMPLETE** |
| ⏳ MATCH | test_stage_match.py | ~60 | Pending |
| ⏳ OUTPUT | test_stage_output.py | ~60 | Pending |
| ⏳ ENTITY_IMAGES | test_stage_entity_images.py | ~50 | Pending |
| ⏳ ENTITY_VIDEOS | test_stage_entity_videos.py | ~45 | Pending |
| ⏳ STOCK | test_stage_stock.py | ~40 | Pending |
| ⏳ REMIX | test_stage_remix.py | ~30 | Pending |

**Total Phase 10 Target:** ~555 tests
**Completed:** 173 tests (31.2%)
**Remaining:** ~382 tests

---

## Conclusion

Phase 10.4 successfully created a comprehensive test suite for the SceneDetectionStage class with 38 tests achieving 100% pass rate. All scene detection paths, B-roll classification, silent video handling, Vision API integration, and scene metadata propagation are thoroughly tested.

The key challenge was understanding the three-way scene metadata propagation (transcripts → text_metadata, scene_data → silent video text_metadata, transcripts → embeddings) and ensuring all paths are tested. The Mock hasattr() issue taught us to use SimpleNamespace for proper attribute testing.

**Status: PHASE 10.4 COMPLETE ✅**

Next: Phase 10.5 - MatchStage tests
