# Phase 10.3: TranscribeStage Tests - COMPLETED

**Date:** 2026-01-09
**Status:** ✅ COMPLETE
**Test File Created:** `tests/test_stage_transcribe.py`
**Tests Added:** 39
**Pass Rate:** 100% (39/39)

---

## Overview

Phase 10.3 implements comprehensive unit tests for the TranscribeStage class, covering transcription orchestration, embedding computation, face detection, topic extraction, and checkpoint operations. This is the third of 10 stage classes to be tested in Phase 10.

## Test Coverage

### Test Structure (39 tests across 11 test classes)

#### 1. TestTranscribeStageInit (3 tests)
- `test_stage_name` - Verifies stage name is "TRANSCRIBE"
- `test_stage_description` - Verifies stage description
- `test_stage_registration` - Confirms stage is registered in stage registry

#### 2. TestInputValidation (2 tests)
- `test_validate_no_videos` - Warns when no videos to transcribe
- `test_validate_with_videos` - Successful validation with video files

#### 3. TestGetVideoFiles (4 tests)
- `test_get_files_from_downloaded_videos` - Get files from state.downloaded_videos
- `test_get_files_from_audio_downloads` - Audio-first mode uses audio files
- `test_get_files_from_directory_scan` - Directory scan with glob
- `test_get_files_preference_order` - Prioritizes downloaded_videos over audio

#### 4. TestTranscriptionOrchestration (5 tests)
- `test_transcribe_videos_parallel` - Parallel transcription with DeltaAwareIndex
- `test_transcribe_sequential` - Sequential transcription fallback
- `test_transcribe_import_error` - Handles import errors gracefully
- `test_transcribe_with_caching` - Uses delta-aware index to skip cached videos
- `test_transcribe_exception_handling` - Handles individual video failures

#### 5. TestEmbeddingComputation (4 tests)
- `test_compute_embeddings_success` - Successful embedding computation and FAISS indexing
- `test_compute_embeddings_import_error` - Error handling with logger.error call
- `test_compute_embeddings_no_text` - Handles empty text segments
- `test_compute_embeddings_disabled` - Skips when `parallel_embedding: false`

#### 6. TestFacePreDetection (3 tests)
- `test_predetect_faces_disabled` - Skips when no face preference
- `test_predetect_faces_with_uncached` - Pre-detects faces for uncached videos
- `test_predetect_faces_error` - Error handling with logger.warning

#### 7. TestVideoTopicExtraction (3 tests)
- `test_extract_topics_empty_transcripts` - Handles empty transcripts
- `test_extract_topics_success` - Successful topic extraction
- `test_extract_topics_error` - Error handling with logger.warning

#### 8. TestSkipTranscription (2 tests)
- `test_can_skip_no_checkpoint` - Returns False when no checkpoint
- `test_can_skip_with_checkpoint` - Returns True when transcripts exist

#### 9. TestStageExecution (4 tests)
- `test_run_no_videos` - Fails when no videos to transcribe
- `test_run_success` - Full stage execution with transcription + embeddings
- `test_run_skip_embeddings` - Transcription without embeddings
- `test_run_exception_handling` - Graceful exception handling

#### 10. TestCheckpointOperations (3 tests)
- `test_restore_no_data` - Returns False when no checkpoint data
- `test_restore_success` - Restores transcripts from checkpoint
- `test_restore_exception_handling` - Handles restore errors

#### 11. TestTextMetadataRebuild (6 tests)
- `test_rebuild_not_match_only` - Skips rebuild when not in match-only mode
- `test_rebuild_no_cache_dir` - Skips when cache directory not found
- `test_rebuild_with_transcripts` - Rebuilds metadata from cached transcripts
- `test_rebuild_with_embeddings` - Rebuilds embeddings and FAISS index
- `test_rebuild_import_error` - Handles import errors gracefully
- `test_rebuild_exception_handling` - Handles general exceptions

---

## Key Testing Patterns

### 1. Mock Configuration
```python
@pytest.fixture
def mock_config():
    """Create mock config with required attributes"""
    config = MagicMock()
    config.cache.cache_dir = ".cache"
    config.transcription.model = "base"
    config.transcription.language = "auto"
    config.transcription.max_workers = 2
    config.pipeline.parallel_embedding = True
    config.embedding.batch_size = 32
    return config
```

### 2. Import Patching Strategy
- **Imports inside methods**: Patch at source module (e.g., `src.transcription.transcribe_videos_parallel`)
- **Standard library**: Patch directly (e.g., `glob.glob`, not `src.stages.transcribe.glob.glob`)

### 3. Multi-Stage Mocking
```python
@patch('src.transcription.transcribe_videos_parallel')
@patch('src.transcription.DeltaAwareIndex')
def test_transcribe_videos_parallel(self, mock_delta_class, mock_transcribe, mock_config):
    # Mock delta index to control which videos are new
    mock_delta = Mock()
    mock_delta.get_new_videos.return_value = ["video1.mp4", "video2.mp4"]
    mock_delta_class.return_value = mock_delta

    # Mock transcription results
    mock_transcribe.return_value = {
        "video1.mp4": [{"text": "Hello", "start_time": 0, "end_time": 2}]
    }
```

### 4. Embedding Mocking
```python
@patch('src.embeddings.compute_embeddings')
@patch('src.embeddings.build_embedding_index')
@patch('src.embeddings.get_embedding_provider')
@patch('src.utils.CacheManager')
def test_compute_embeddings_success(self, mock_cache_class, mock_provider,
                                   mock_build_index, mock_compute, mock_config):
    import numpy as np

    # Mock provider and cache
    mock_provider.return_value = Mock()
    mock_cache_class.return_value = Mock()

    # Mock embeddings array
    mock_compute.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_build_index.return_value = Mock()  # FAISS index
```

---

## Technical Challenges Resolved

### Challenge 1: Import Patching Locations
**Problem:** Initial tests patched imports at `src.stages.transcribe.X`, but imports happen inside methods from other modules.

**Solution:** Changed all patches to source modules:
- ✅ `@patch('src.transcription.transcribe_videos_parallel')`
- ✅ `@patch('src.transcription.DeltaAwareIndex')`
- ✅ `@patch('src.transcription.TranscriptCache')`
- ✅ `@patch('src.embeddings.compute_embeddings')`
- ✅ `@patch('src.embeddings.build_embedding_index')`
- ✅ `@patch('src.embeddings.get_embedding_provider')`
- ✅ `@patch('src.face_detection.FaceDetector')`
- ✅ `@patch('src.topic_extraction.TopicExtractor')`
- ✅ `@patch('glob.glob')` (standard library)

**Lesson Learned:** Always identify where imports occur - if inside methods, patch at source.

### Challenge 2: Embedding Error Handling
**Problem:** Test expected `logger.warning()` call, but code uses `logger.error()` for generic exceptions.

**Initial Error:**
```python
assert result is False
mock_logger.warning.assert_called()  # ❌ Not called!
```

**Root Cause:** Source code handles errors differently:
```python
except ImportError as e:
    logger.warning(f"Embedding computation not available: {e}")  # Only for ImportError
except Exception as e:
    logger.error(f"Embedding computation failed: {e}")  # For all other exceptions
```

**Solution:** Changed assertion to match actual behavior:
```python
assert result is False
mock_logger.error.assert_called()  # ✅ Correct
```

### Challenge 3: Glob Patch Location
**Problem:** Initial glob patch at `src.stages.transcribe.glob.glob` failed with `ModuleNotFoundError`.

**Solution:** glob is a standard library module, patch at top level:
```python
# ❌ Wrong
@patch('src.stages.transcribe.glob.glob')

# ✅ Correct
@patch('glob.glob')
```

### Challenge 4: Provider Initialization Failure
**Problem:** Test mocked `compute_embeddings` but `get_embedding_provider` failed before reaching compute call, causing test to fail.

**Solution:** Mock both the provider and the compute function:
```python
@patch('src.embeddings.get_embedding_provider')  # Mock provider first
@patch('src.embeddings.compute_embeddings', side_effect=Exception("error"))
def test_compute_embeddings_error(self, mock_compute, mock_provider, ...):
    mock_provider.return_value = Mock()  # Prevent provider initialization error
```

---

## Test Results

```bash
pytest tests/test_stage_transcribe.py -v

============================= 39 passed in 0.37s ===============================
Pass Rate: 100% ✅
```

**Total Test Count:**
- Before Phase 10.3: 1,450 tests
- After Phase 10.3: 1,489 tests (+39)

---

## Coverage Impact

**Module:** `src/stages/transcribe.py` (~500 lines)
**Tests:** 39 comprehensive unit tests

**Coverage breakdown:**
- ✅ Stage initialization and registration
- ✅ Input validation (no videos, with videos)
- ✅ Video file detection (downloaded_videos, audio_downloads, directory scan)
- ✅ Transcription orchestration (parallel, sequential, delta-aware caching)
- ✅ Embedding computation (success, errors, disabled, empty text)
- ✅ Face pre-detection (cached, uncached, disabled, errors)
- ✅ Topic extraction (success, empty, errors)
- ✅ Stage execution (full pipeline, skip embeddings, errors)
- ✅ Checkpoint operations (skip, restore, errors)
- ✅ Text metadata rebuild (match-only mode, transcripts, embeddings, errors)

**Expected Coverage:** 80-90% (from baseline)

---

## Next Steps

### Phase 10.4: SceneDetectionStage
**File:** `tests/test_stage_scene_detection.py` (new)
**Expected tests:** ~75 tests

**Key areas to cover:**
- Scene detection with PySceneDetect
- Face detection per scene (MediaPipe, OpenCV)
- B-roll classification (face_score < 0.3)
- Silent video detection
- Vision API integration for scene descriptions
- is_broll propagation to transcripts and text_metadata
- Checkpoint saving/loading

**Critical files:** `src/stages/scene_detection.py`, `src/face_detection.py`, `src/vision.py`

---

## Remaining Phase 10 Stages

| Stage | File | Tests Needed | Status |
|-------|------|--------------|--------|
| ✅ ANALYZE | test_stage_analyze.py | 45 | **COMPLETE** |
| ✅ DOWNLOAD | test_stage_download.py | 51 | **COMPLETE** |
| ✅ TRANSCRIBE | test_stage_transcribe.py | 39 | **COMPLETE** |
| ⏳ SCENE_DETECTION | test_stage_scene_detection.py | ~75 | Pending |
| ⏳ MATCH | test_stage_match.py | ~60 | Pending |
| ⏳ OUTPUT | test_stage_output.py | ~60 | Pending |
| ⏳ ENTITY_IMAGES | test_stage_entity_images.py | ~50 | Pending |
| ⏳ ENTITY_VIDEOS | test_stage_entity_videos.py | ~45 | Pending |
| ⏳ STOCK | test_stage_stock.py | ~40 | Pending |
| ⏳ REMIX | test_stage_remix.py | ~30 | Pending |

**Total Phase 10 Target:** ~555 tests
**Completed:** 135 tests (24.3%)
**Remaining:** ~420 tests

---

## Conclusion

Phase 10.3 successfully created a comprehensive test suite for the TranscribeStage class with 39 tests achieving 100% pass rate. All transcription orchestration paths, embedding computation, face detection, topic extraction, and checkpoint operations are thoroughly tested.

The key lesson from this phase was understanding import patching locations - imports inside methods must be patched at their source module, and standard library imports should be patched at the top level.

**Status: PHASE 10.3 COMPLETE ✅**

Next: Phase 10.4 - SceneDetectionStage tests
