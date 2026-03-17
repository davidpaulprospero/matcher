# DVIDS Implementation Fixes Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clarify the `_select_best_file` ranking logic in DVIDS client

**Architecture:** The current ranking uses `(hd_bonus, area)` which works but is confusing. The code returns the first candidate matching the best `(hd_bonus, area)` pair, which preserves provider order for same-resolution files. This should be refactored for clarity.

**Tech Stack:** Python, pytest

**Notes on other "issues":**
- Issue #2 (no streaming): Valid enhancement but low priority - would need API batching support
- Issue #3 (description field): Not an issue - VideoResult model doesn't have this field
- Issue #4 (hardcoded timeout): Not an issue - already configurable via `download_entity_videos(download_timeout=...)`

---

### Task 1: Clarify `_select_best_file` ranking logic

**Files:**
- Modify: `src/media_sources/videos/dvids.py:351-388`
- Test: `tests/test_dvids_video_client.py`

**Step 1: Review current implementation**

The current code at lines 375-386:
```python
candidates.append((hd_bonus, area, size, index, file_info))

if not candidates:
    return None

if not self.prefer_hd:
    return candidates[0][4]

best_rank = max((item[0], item[1]) for item in candidates)
for candidate in candidates:
    if (candidate[0], candidate[1]) == best_rank:
        return candidate[4]

return None
```

**Issue:** The ranking uses `(hd_bonus, area)` which works because:
1. `area = width * height` sorts by resolution descending
2. When `prefer_hd=True`, HD bonus takes priority
3. For same-resolution ties, provider order is preserved (first match wins)

But this is hard to read. The logic should be explicit about:
- Prioritizing HD resolution over SD
- For same resolution, preferring larger area
- For exact ties, preserving provider order

**Step 2: Add clarifying comments and simplify logic**

Refactor to be more explicit:

```python
def _select_best_file(self, files: List[dict]) -> Optional[dict]:
    """
    Select the best MP4 rendition based on HD preference and resolution.

    Selection priority:
    1. Prefer HD resolution (1280x720+) when prefer_hd=True
    2. Prefer higher resolution (area = width * height)
    3. Preserve provider order for same-resolution files
    """
    candidates = []
    for index, file_info in enumerate(files):
        file_url = str(file_info.get("src", ""))
        file_type = str(file_info.get("type", "")).lower()
        if "mp4" not in file_type and not file_url.lower().endswith(".mp4"):
            continue

        width = self._coerce_int(file_info.get("width"))
        height = self._coerce_int(file_info.get("height"))
        if self.landscape_only and width and height and width < height:
            continue

        area = width * height
        is_hd = self._is_hd_resolution(width, height)
        hd_bonus = 1 if is_hd else 0

        # Tuple: (hd_priority, resolution_priority, provider_order, file_info)
        # hd_priority: 1 for HD, 0 for SD (higher wins)
        # resolution_priority: area in pixels (higher wins)
        # provider_order: index in files array (lower wins - preserves provider order)
        candidates.append((hd_bonus, area, index, file_info))

    if not candidates:
        return None

    if not self.prefer_hd:
        # Return first acceptable file (provider order)
        return candidates[0][3]

    # Sort by (hd_priority DESC, resolution_priority DESC, provider_order ASC)
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=(True, True, False))
    return candidates[0][3]
```

**Step 3: Run existing tests to verify behavior unchanged**

Run: `python -m pytest tests/test_dvids_video_client.py -v --tb=short`
Expected: All 18 tests pass

**Step 4: Commit**

```bash
git add src/media_sources/videos/dvids.py
git commit -m "refactor: clarify _select_best_file ranking logic in DVIDS client"
```

---

### Summary

Only one real issue needed fixing: the `_select_best_file` logic was functional but confusing. The refactored code:
- Uses explicit tuple fields instead of positional indices
- Adds clear docstring explaining selection priority
- Uses `sort()` with explicit key for clarity
- Maintains exact same behavior (tests verify this)

The other "issues" from the review are either not bugs (description field doesn't exist in model) or already configurable (timeout via orchestrator parameter).
