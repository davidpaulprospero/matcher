# Phase 10.7: EntityImagesStage Tests - COMPLETED

**Date:** 2026-01-09
**Status:** ✅ COMPLETE
**Test File Created:** `tests/test_stage_entity_images.py`
**Tests Added:** 33
**Pass Rate:** 100% (33/33)

---

## Overview

Phase 10.7 implements comprehensive unit tests for the EntityImagesStage class, covering entity image downloads from Google, Bing, Pexels, and Pixabay, entity filtering, local and global entity cache integration, entity-to-segment mapping, output directory management, and checkpoint operations. This is the seventh of 10 stage classes to be tested in Phase 10.

## Test Coverage

### Test Structure (33 tests across 10 test classes)

#### 1. TestEntityImagesStageInit (3 tests)
- `test_stage_name` - Verifies stage name is "ENTITY_IMAGES"
- `test_stage_description` - Verifies stage description
- `test_stage_registration` - Confirms stage is registered in stage registry

#### 2. TestInputValidation (1 test)
- `test_validate_no_hard_requirements` - Optional stage has no hard validation requirements

**Key Pattern**: EntityImagesStage is an optional stage - it doesn't enforce hard requirements in validate_inputs().

#### 3. TestSkipConditions (4 tests)
- `test_skip_via_pipeline_config` - Skip when `pipeline.skip_image_search: true`
- `test_skip_when_disabled` - Skip when `image_search.enabled: false`
- `test_skip_when_no_entities` - Skip when no entities extracted
- `test_skip_when_no_matching_types` - Skip when no entities match configured types (PERSON, ORG, LOC)

**Skip Reasons**: The stage can skip for multiple reasons (pipeline config, disabled, no entities, no matching types), all returning success with skip metadata.

#### 4. TestEntityFiltering (2 tests)
- `test_filter_by_entity_type` - Filter entities by configured types (PERSON, ORG, LOC)
- `test_max_entities_limit` - Apply max_entities limit when configured

**Entity Types**: Only entities matching `config.image_search.entity_types` are searched. Common types: PERSON, ORG, LOC, GPE.

#### 5. TestImageDownload (3 tests)
- `test_successful_download` - Full download pipeline with entity results
- `test_no_images_downloaded` - Handle when no images found
- `test_download_with_api_keys` - Pass PEXELS_API_KEY and PIXABAY_API_KEY from environment

**Image Sources**: Google Images (scraping), Bing Images (API), Pexels (API), Pixabay (API). API keys loaded from environment variables.

#### 6. TestEntitySegmentMapping (1 test)
- `test_segment_indices_mapped` - Entity results updated with segment indices from voiceover

**Segment Mapping**: Entities are mapped to voiceover segments where they're mentioned, enabling timeline placement on V9 track.

#### 7. TestOutputDirectory (2 tests)
- `test_project_relative_mode` - Output to `project_dir/images` when root_dir not configured
- `test_short_path_mode` - Output to `root_dir/ProjectName` when root_dir configured (e.g., "E:/i")

**Directory Modes**:
- **Project-relative**: `checkpoint.project_dir / folder_name` (default: "images")
- **Short path**: `root_dir / project_name[:15]` (Windows path length workaround)

#### 8. TestEntityCache (3 tests)
- `test_cache_enabled` - Initialize EntityCache when enabled
- `test_cache_disabled` - No cache when disabled
- `test_cache_initialization_failure` - Gracefully handle cache init failures

**Entity Cache**: Cross-project cache sharing entity images between documentaries. Reduces re-downloads for common entities (e.g., "Albert Einstein" used in multiple physics docs).

#### 9. TestStageExecution (3 tests)
- `test_run_success` - Full execution with entity results
- `test_run_import_error` - Handle ImportError for missing media_sources module
- `test_run_exception_handling` - Exception handling in main run

#### 10. TestCheckpointOperations (8 tests)
- `test_can_skip_with_checkpoint` - Returns True when checkpoint exists
- `test_can_skip_no_checkpoint` - Returns False when no checkpoint
- `test_restore_success` - Successful restore from disk (.entity.json files)
- `test_restore_no_data` - Returns False when no checkpoint data
- `test_restore_no_config` - Returns False when no config available
- `test_restore_no_images_dir` - Returns False when images directory doesn't exist
- `test_restore_exception_handling` - Handles restore errors gracefully
- *(Counted in total: 7 tests in TestCheckpointOperations)*

**Checkpoint Restore**: Restores entity images from disk using `.entity.json` metadata files (created by download_entity_images). Checks for images directory existence and config availability.

#### 11. TestEdgeCases (3 tests)
- `test_empty_topic_context` - Handle empty topic_context string
- `test_refresh_entities_flag` - Pass skip_local_cache=True when refresh_entities flag set
- `test_project_name_truncation` - Truncate project name to 15 chars in short path mode
- `test_many_entities_display_limit` - Display first 5 entities, "... and X more" for remainder

---

## Key Testing Patterns

### 1. Mock Config with Image Search Settings
```python
@pytest.fixture
def mock_config():
    config = MagicMock()
    config.pipeline.skip_image_search = False
    config.image_search.enabled = True
    config.image_search.entity_types = ['PERSON', 'ORG', 'LOC']
    config.image_search.max_entities = 0  # No limit
    config.image_search.images_per_entity = 3
    config.image_search.min_size_mb = 0.1
    config.image_search.use_google = True
    config.image_search.use_bing = False
    config.image_search.use_stock_apis = False
    config.image_search.root_dir = ""  # Project-relative mode
    config.image_search.folder_name = "images"

    # Entity cache config
    cache_config = MagicMock()
    cache_config.enabled = False
    config.image_search.entity_cache = cache_config
    return config
```

### 2. Entity Fixtures
```python
@pytest.fixture
def mock_entities():
    """Create mock extracted entities"""
    return [
        {'text': 'Albert Einstein', 'type': 'PERSON', 'context': 'physicist'},
        {'text': 'CERN', 'type': 'ORG', 'context': 'research organization'},
        {'text': 'Geneva', 'type': 'LOC', 'context': 'Swiss city'},
    ]

@pytest.fixture
def mock_entity_results():
    """Create mock EntityImageResult objects"""
    return {
        'Albert Einstein': EntityImageResult(
            entity_name='Albert Einstein',
            entity_type='PERSON',
            context='physicist',
            query='Albert Einstein physicist',
            images=['/path/einstein1.jpg', '/path/einstein2.jpg'],
            segment_indices=[0, 2]
        )
    }
```

### 3. Mocking download_entity_images
```python
@patch('src.media_sources.download_entity_images')
@patch('src.media_sources.map_entities_to_segments')
def test_successful_download(self, mock_map, mock_download, ...):
    mock_download.return_value = mock_entity_results
    mock_map.return_value = {'Albert Einstein': [0, 2]}

    result = stage.run(state, mock_config, mock_checkpoint)

    # Verify download was called with correct parameters
    call_kwargs = mock_download.call_args[1]
    assert call_kwargs['entities'] == mock_entities
    assert call_kwargs['images_per_entity'] == 3
    assert call_kwargs['use_google'] is True
```

### 4. Testing Skip Conditions
```python
def test_skip_when_no_matching_types(self, mock_config, mock_checkpoint):
    stage = EntityImagesStage()
    state = PipelineState()
    state.extracted_entities = [
        {'text': 'Something', 'type': 'DATE', 'context': 'temporal'},
    ]
    mock_config.image_search.entity_types = ['PERSON', 'ORG']  # DATE not allowed

    result = stage.run(state, mock_config, mock_checkpoint)

    assert result.success is True
    assert result.data['skipped'] is True
    assert result.data['reason'] == 'no_matching_types'
```

---

## Technical Challenges Resolved

### Challenge 1: Stage Registration Test
**Problem**: Initial test tried to import `get_stage_class` which doesn't exist in `src.stages.__init__.py`.

**Initial Error:**
```python
from src.stages import get_stage_class  # ❌ Function doesn't exist
stage_class = get_stage_class("ENTITY_IMAGES")
```

**Solution:** Use correct function name `get_stage`:
```python
from src.stages import get_stage  # ✅ Correct function
stage_class = get_stage("ENTITY_IMAGES")
assert stage_class is EntityImagesStage
```

**Lesson Learned**: Verify API function names before writing tests - grep the source for actual exports.

### Challenge 2: Windows Path Format in Assertions
**Problem**: Windows Path objects use backslashes, causing assertions with forward slashes to fail.

**Initial Error:**
```python
output_dir = Path('E:/i/Project')
assert str(output_dir).startswith("E:/i")  # ❌ FAILS - actual: "E:\\i\\Project"
```

**Solution:** Normalize path separators before comparison:
```python
output_dir = Path(call_kwargs['output_dir'])
output_str = str(output_dir).replace('\\', '/')  # ✅ Normalize separators
assert output_str.startswith("E:/i")
```

**Lesson Learned**: Always normalize path separators when testing cross-platform code on Windows.

### Challenge 3: Patching Import Statements
**Problem**: Trying to patch `download_entity_images` at module level fails because it's imported dynamically inside the run() method.

**Initial Error:**
```python
with patch('src.stages.entity_images.download_entity_images', side_effect=ImportError(...)):
    # ❌ AttributeError: module 'src.stages.entity_images' does not have attribute 'download_entity_images'
```

**Solution:** Patch the builtin import mechanism:
```python
with patch('builtins.__import__', side_effect=ImportError("Module not found")):
    result = stage.run(state, mock_config, mock_checkpoint)  # ✅ Catches import
```

**Lesson Learned**: When testing ImportError handling for dynamically imported modules (inside functions), patch `builtins.__import__` instead of the module-level import.

### Challenge 4: Entity Cache Configuration Structure
**Problem**: Entity cache config is nested (`config.image_search.entity_cache.enabled`) and must be properly mocked with nested structure.

**Solution:** Create nested MagicMock:
```python
cache_config = MagicMock()
cache_config.enabled = False
config.image_search.entity_cache = cache_config  # ✅ Nested config
```

This allows the stage to access `getattr(config.image_search, 'entity_cache', None)` and `getattr(cache_config, 'enabled', False)` correctly.

---

## Test Results

```bash
pytest tests/test_stage_entity_images.py -v

============================= 33 passed in 0.36s =================================
Pass Rate: 100% ✅
```

**Total Test Count:**
- Before Phase 10.7: 1,590 tests
- After Phase 10.7: 1,623 tests (+33)

---

## Coverage Impact

**Module:** `src/stages/entity_images.py` (~361 lines)
**Tests:** 33 comprehensive unit tests

**Coverage breakdown:**
- ✅ Stage initialization and registration
- ✅ Input validation (optional stage, no hard requirements)
- ✅ Skip conditions (pipeline config, disabled, no entities, no matching types)
- ✅ Entity filtering (by type, max entities limit)
- ✅ Image download integration (Google, Bing, Pexels, Pixabay)
- ✅ API key passing from environment
- ✅ Entity-to-segment mapping
- ✅ Output directory management (project-relative vs short path modes)
- ✅ Entity cache integration (enabled, disabled, initialization failures)
- ✅ Stage execution (full pipeline, import errors, exceptions)
- ✅ Checkpoint operations (skip, restore from disk, errors)
- ✅ Edge cases (empty topic, refresh flag, project name truncation, display limits)

**Expected Coverage:** 85-90% (from baseline)

---

## Next Steps

### Phase 10.8: EntityVideosStage
**File:** `tests/test_stage_entity_videos.py` (new)
**Expected tests:** ~40 tests

**Key areas to cover:**
- Stock video downloads (Pexels Videos API, Pixabay Videos API)
- Entity filtering by type
- Video search query generation
- Video download and transcoding
- Entity-to-segment mapping
- Output directory management
- Video cache integration
- Rate limiting and API quotas
- Checkpoint saving/loading

**Critical files:** `src/stages/entity_videos.py`, `src/media_sources/videos/pexels.py`, `src/media_sources/videos/pixabay.py`

---

## Remaining Phase 10 Stages

| Stage | File | Tests Needed | Status |
|-------|------|--------------|--------|
| ✅ ANALYZE | test_stage_analyze.py | 45 | **COMPLETE** |
| ✅ DOWNLOAD | test_stage_download.py | 51 | **COMPLETE** |
| ✅ TRANSCRIBE | test_stage_transcribe.py | 39 | **COMPLETE** |
| ✅ SCENE_DETECTION | test_stage_scene_detection.py | 38 | **COMPLETE** |
| ✅ MATCH | test_stage_match.py | 33 | **COMPLETE** |
| ✅ OUTPUT | test_stage_output.py | 30 | **COMPLETE** |
| ✅ ENTITY_IMAGES | test_stage_entity_images.py | 33 | **COMPLETE** |
| ⏳ ENTITY_VIDEOS | test_stage_entity_videos.py | ~40 | Pending |
| ⏳ STOCK | test_stage_stock.py | ~40 | Pending |
| ⏳ REMIX | test_stage_remix.py | ~30 | Pending |

**Total Phase 10 Target:** ~555 tests
**Completed:** 269 tests (48.5%)
**Remaining:** ~286 tests

---

## Conclusion

Phase 10.7 successfully created a comprehensive test suite for the EntityImagesStage class with 33 tests achieving 100% pass rate. All entity image search paths (Google, Bing, Pexels, Pixabay), entity filtering, cache integration, output directory modes, and checkpoint operations are thoroughly tested.

The key challenges were understanding the correct API function names (`get_stage` not `get_stage_class`), handling Windows path separators in assertions, and properly patching dynamic imports for ImportError testing.

**Status: PHASE 10.7 COMPLETE ✅**

Next: Phase 10.8 - EntityVideosStage tests
