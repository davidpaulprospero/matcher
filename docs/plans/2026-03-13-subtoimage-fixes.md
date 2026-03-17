# Subtoimage Integration Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three issues in the generated_images (subtoimage) feature: legacy filename cache bypass, sticky pending batches on resume, and unused timeout enforcement.

**Architecture:** Three independent bug fixes to the generated_images service and analyze stage, each following TDD (test-first approach).

**Tech Stack:** Python, pytest, GeneratedImageService, AnalyzeStage

---

## Task 1: Fix Legacy Filename Cache Bypass

**Files:**
- Modify: `src/generated_images/service.py:198-220`
- Test: `tests/test_generated_images_service.py`

**Step 1: Write the failing test**

Add test that verifies legacy filenames are NOT reused when prompt/model/quality changed:

```python
def test_generate_does_not_reuse_legacy_filename_when_prompt_changed(self, temp_project_dir):
    """Legacy filename should not be reused when prompt changed."""
    from src.generated_images.service import GeneratedImageService
    from src.state import GeneratedImageBatch

    # Create legacy file
    out = temp_project_dir / "generated_images"
    out.mkdir(parents=True)
    img = Image.new("RGB", (128, 128), color=(0, 255, 0))
    img.save(out / "generated_0000_0007.png")

    config = SimpleNamespace(
        generated_images=GeneratedImagesConfig(
            enabled=True,
            output_dir="generated_images",
            prompt_prefix="Version two",  # Changed from default
        ),
        api_keys=SimpleNamespace(google_api_key="", gemini_api_key=""),
    )
    service = GeneratedImageService(config, temp_project_dir)

    batch = GeneratedImageBatch(
        batch_id="generated_000",
        segment_start_index=0,
        segment_end_index=7,
        segment_count=8,
        start_time=0.0,
        end_time=8.0,
        text="test",
    )

    result = service.generate([batch], topic_context="Travel")[0]

    # Should NOT reuse legacy file due to prompt change
    assert result.status != "generated" or "Version two" in str(result.image_path)
    # New fingerprinted filename should be generated
    assert "_v2_" in result.cache_fingerprint or result.status == "pending"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_generated_images_service.py::TestGeneratedImageService::test_generate_does_not_reuse_legacy_filename_when_prompt_changed -v`
Expected: FAIL - currently legacy file IS reused unconditionally

**Step 3: Write minimal implementation**

Modify `_build_reuse_candidates()` in `service.py` to NOT add legacy filename to candidates unconditionally. Instead, only add it if the batch's existing `image_path` points to that legacy file (indicating it was checkpointed from an older run):

```python
def _build_reuse_candidates(
    self,
    batch: GeneratedImageBatch,
    output_path: Path,
) -> List[Path]:
    candidates: List[Path] = [output_path]

    # Only add fingerprinted path if it exists (new cache location)
    if output_path.exists():
        candidates.append(output_path)

    # Only add legacy filename if the batch's image_path explicitly points to it
    # This prevents silent reuse of stale legacy files when prompt/model changes
    existing_path = self._resolve_existing_image_path(batch.image_path)
    legacy_name = self._build_legacy_filename(batch)
    if existing_path is not None and existing_path.name == legacy_name:
        candidates.append(output_path.with_name(legacy_name))

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normcase(str(candidate))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_generated_images_service.py::TestGeneratedImageService::test_generate_does_not_reuse_legacy_filename_when_prompt_changed -v`
Expected: PASS

**Step 5: Run all generated_images tests**

Run: `pytest tests/test_generated_images_service.py -v`
Expected: All pass (may need to update existing legacy test)

**Step 6: Commit**

```bash
git add src/generated_images/service.py tests/test_generated_images_service.py
git commit -m "fix: prevent legacy filename reuse when prompt/model changes"
```

---

## Task 2: Fix Sticky Pending Batches on Resume

**Files:**
- Modify: `src/stages/analyze.py:278-310`
- Test: `tests/test_analyze_generated_images_restore.py`

**Step 1: Write the failing test**

```python
def test_restore_retries_pending_batches_when_generation_enabled(self, temp_project_dir):
    """Resume should retry pending batches when generation is now enabled."""
    from src.checkpoint import CheckpointManager
    from src.stages.analyze import AnalyzeStage
    from src.state import PipelineState

    root = temp_project_dir
    stage = AnalyzeStage()
    state = PipelineState()
    checkpoint = CheckpointManager(root, config_hash="test")

    # Save checkpoint with pending batches
    checkpoint.save("ANALYZE", {
        "keywords": ["travel"],
        "segments": [{"index": 0, "start": 0.0, "end": 1.0, "text": "Hello world"}],
        "generated_images": [{
            "batch_id": "generated_000",
            "segment_start_index": 0,
            "segment_end_index": 5,
            "segment_count": 6,
            "segment_indices": [0, 1, 2, 3, 4, 5],
            "start_time": 0.0,
            "end_time": 10.0,
            "text": "Merged subtitle text",
            "status": "pending",  # Pending, not failed
            "error": "",
        }],
    })

    config = SimpleNamespace(
        keyword=SimpleNamespace(segments_per_query=3, tfidf_max_features=100),
        matching=SimpleNamespace(
            chapter_matching_enabled=False,
            location_matching=SimpleNamespace(enabled=False),
        ),
        transcription=SimpleNamespace(
            model="base",
            compute_type="int8",
            auto_generate_silence_removed_audio=False,
            auto_refresh_stale_srt=False,
        ),
        generated_images=GeneratedImagesConfig(enabled=True, generate_assets=True),
        project_dir=str(root),
    )

    should_rerun = stage._should_retry_generated_images_from_checkpoint(
        checkpoint.load("ANALYZE"), config
    )

    # Pending batches should trigger rerun when generation is enabled
    assert should_rerun is True
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyze_generated_images_restore.py::TestAnalyzeGeneratedImages::test_restore_retries_pending_batches_when_generation_enabled -v`
Expected: FAIL - currently only fails trigger rerun

**Step 3: Write minimal implementation**

Modify `_should_retry_generated_images_from_checkpoint()` in `analyze.py`:

```python
def _should_retry_generated_images_from_checkpoint(
    self,
    data: Dict[str, Any],
    config: 'Config',
) -> bool:
    """Check if we should regenerate images from checkpoint."""
    generated_config = getattr(config, 'generated_images', None)
    if not generated_config or not generated_config.enabled:
        return False

    # Must have generate_assets enabled to need retry
    if not getattr(generated_config, 'generate_assets', True):
        return False

    generated_images_data = data.get('generated_images', [])
    if not isinstance(generated_images_data, list):
        return False

    for batch_data in generated_images_data:
        if not isinstance(batch_data, dict):
            continue
        status = batch_data.get('status', '')
        # Retry if failed OR if pending (was generate_assets=false before)
        if status in ('failed', 'pending'):
            return True

    return False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyze_generated_images_restore.py::TestAnalyzeGeneratedImages::test_restore_retries_pending_batches_when_generation_enabled -v`
Expected: PASS

**Step 5: Run all analyze restore tests**

Run: `pytest tests/test_analyze_generated_images_restore.py -v`
Expected: All pass

**Step 6: Commit**

```bash
git add src/stages/analyze.py tests/test_analyze_generated_images_restore.py
git commit -m "fix: retry pending generated image batches on resume"
```

---

## Task 3: Either Implement or Remove timeout_seconds

**Files:**
- Modify: `src/generated_images/providers/imagen.py` OR `src/config/sections/generated_images.py`
- Test: May not need new test

**Step 1: Decide approach**

Option A: Remove the unused timeout from config (simpler, cleaner)
Option B: Implement timeout enforcement (more complete but complex)

**If Option A (Remove):**

Modify `config.yaml` to remove `timeout_seconds` and update config class to not validate it:

```yaml
# Remove timeout_seconds from config.yaml generated_images section
```

Modify `GeneratedImagesConfig` to remove timeout_seconds field.

**If Option B (Implement):**

The Google genai SDK may not expose direct timeout. Can wrap in asyncio.wait_for or use Thread with timeout.

**Recommended: Option A** - Remove unused config to keep things clean. The SDK handles timeouts internally.

**Step 2: Commit**

```bash
git add config.yaml src/config/sections/generated_images.py src/generated_images/service.py src/generated_images/providers/imagen.py
git commit -m "refactor: remove unused timeout_seconds config"
```

---

## Summary

| Task | Issue | Severity | Files |
|------|-------|----------|-------|
| 1 | Legacy filename bypasses fingerprint | Medium | service.py |
| 2 | Pending batches sticky on resume | Medium | analyze.py |
| 3 | timeout_seconds unused | Low | config files |

Execute tasks in order 1 → 2 → 3. Each task is independent and can be committed separately.
