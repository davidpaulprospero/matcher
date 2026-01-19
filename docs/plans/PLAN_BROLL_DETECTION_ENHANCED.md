# Enhanced B-roll Detection: Temporal + Spatial Coverage

## Overview

Improve B-roll detection accuracy by calculating face presence based on:
1. **Temporal coverage** - What percentage of the scene duration contains faces
2. **Spatial coverage** - What percentage of the frame area faces occupy

**Final formula:** `face_score = temporal_coverage × spatial_coverage`

Example: A face visible for 20% of scene duration occupying 15% of frame area:
- `face_score = 0.20 × 0.15 = 0.03` → Excellent B-roll (threshold 0.3)

## Files to Modify

| File | Changes |
|------|---------|
| [src/face_detection.py](../src/face_detection.py) | Core detection logic - extract bounding boxes, calculate spatial coverage |
| [src/scene_detection.py](../src/scene_detection.py) | Scene processing - pass adaptive sample count |
| [src/config/sections/media.py](../src/config/sections/media.py) | Add new config options |
| [config.yaml](../config.yaml) | Document new options |

## Implementation Plan

### Step 1: Update FaceDetector to Return Spatial Data

**File:** `src/face_detection.py`

Replace binary detection with spatial coverage calculation:

```python
# Current (binary):
if results.detections and len(results.detections) > 0:
    frames_with_faces += 1

# New (spatial coverage):
if results.detections:
    frame_face_area = 0.0
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        # Normalized coordinates (0-1)
        face_area = bbox.width * bbox.height
        frame_face_area += face_area
    # Cap at 1.0 (multiple overlapping faces)
    frame_spatial_coverage = min(frame_face_area, 1.0)
    spatial_scores.append(frame_spatial_coverage)
else:
    spatial_scores.append(0.0)
```

Similar change for OpenCV fallback:
```python
# OpenCV returns pixel coordinates - normalize to frame dimensions
for (x, y, w, h) in faces:
    face_area = (w * h) / (frame_width * frame_height)
    frame_face_area += face_area
```

### Step 2: Adaptive Frame Sampling

**File:** `src/face_detection.py`

Add adaptive sampling function:

```python
def _calculate_sample_count(self, duration: float, max_samples: int = 60) -> int:
    """
    Adaptive sampling: more samples for short scenes, capped for long ones.

    Strategy:
    - Scenes < 5s: 2 fps (10 samples max)
    - Scenes 5-30s: 1 fps
    - Scenes > 30s: 0.5 fps, capped at max_samples
    """
    if duration <= 5:
        return min(int(duration * 2), max_samples)
    elif duration <= 30:
        return min(int(duration), max_samples)
    else:
        return min(int(duration * 0.5), max_samples)
```

### Step 3: New Return Structure

**File:** `src/face_detection.py`

Create dataclass for detailed face metrics:

```python
@dataclass
class FaceMetrics:
    """Detailed face detection metrics for a scene"""
    temporal_coverage: float  # % of frames with faces (0.0-1.0)
    spatial_coverage: float   # Average % of frame area occupied (0.0-1.0)
    face_score: float         # Combined: temporal × spatial
    frames_sampled: int       # How many frames were analyzed
    frames_with_faces: int    # How many had any face detected
    max_spatial: float        # Highest single-frame coverage (for peaks)
```

Update `get_scene_face_score()` to return `FaceMetrics`:

```python
def get_scene_face_score(
    self, video_path: str, start_time: float, end_time: float,
    scene_index: int = 0, sample_frames: int = None, cache_dir: str = None
) -> FaceMetrics:
    """
    Get detailed face metrics for a scene.

    Args:
        sample_frames: If None, uses adaptive sampling based on duration
    """
    duration = end_time - start_time

    # Adaptive sampling if not specified
    if sample_frames is None:
        sample_frames = self._calculate_sample_count(duration)

    # ... detection logic ...

    temporal = frames_with_faces / frames_sampled
    spatial_avg = sum(spatial_scores) / len(spatial_scores) if spatial_scores else 0.0

    return FaceMetrics(
        temporal_coverage=temporal,
        spatial_coverage=spatial_avg,
        face_score=temporal * spatial_avg,  # Multiplicative combination
        frames_sampled=frames_sampled,
        frames_with_faces=frames_with_faces,
        max_spatial=max(spatial_scores) if spatial_scores else 0.0
    )
```

### Step 4: Update SceneInfo Dataclass

**File:** `src/scene_detection.py`

Expand SceneInfo to store detailed metrics:

```python
@dataclass
class SceneInfo:
    # ... existing fields ...

    # Enhanced face detection fields
    face_score: float = 0.5           # Combined score (temporal × spatial)
    temporal_coverage: float = 0.5    # % of scene duration with faces
    spatial_coverage: float = 0.5     # Avg % of frame area with faces
    is_broll: bool = False
    frames_sampled: int = 0           # For debugging/transparency
```

### Step 5: Update Scene Processing

**File:** `src/scene_detection.py` (in `_detect_faces_per_scene` method)

```python
for scene in scenes:
    metrics = self.face_detector.get_scene_face_score(
        str(video_path),
        scene.start_time,
        scene.end_time,
        scene.scene_index,
        sample_frames=None,  # Use adaptive sampling
        cache_dir=cache_dir
    )

    scene.face_score = metrics.face_score
    scene.temporal_coverage = metrics.temporal_coverage
    scene.spatial_coverage = metrics.spatial_coverage
    scene.frames_sampled = metrics.frames_sampled
    scene.is_broll = metrics.face_score < self.broll_threshold
```

### Step 6: Config Options

**File:** `src/config/sections/media.py`

```python
@dataclass
class SceneDetectionConfig:
    # ... existing ...

    # Face detection settings
    detect_faces_per_scene: bool = True
    face_sample_frames: int = None  # None = adaptive sampling
    max_face_samples: int = 60      # Cap for adaptive sampling
    face_score_formula: str = "multiply"  # "multiply", "average", "weighted"
```

**File:** `config.yaml`

```yaml
scene_detection:
  detect_faces_per_scene: true
  # Face sampling (null = adaptive based on scene duration)
  face_sample_frames: null
  max_face_samples: 60
  # Score formula: "multiply" (temporal × spatial), "average", "weighted"
  face_score_formula: "multiply"
```

### Step 7: Update Cache Format

**File:** `src/face_detection.py`

Extend cache to store detailed metrics:

```python
# Old cache format (backward compatible read):
{"path:0.0-10.0": 0.5}

# New cache format:
{
    "path:0.0-10.0": {
        "face_score": 0.03,
        "temporal": 0.2,
        "spatial": 0.15,
        "frames_sampled": 10,
        "version": 2
    }
}
```

Add migration logic to handle old cache entries.

### Step 8: Enhanced Logging

Update log output to show new metrics:

```
Current:  → B-roll: 4/23 scenes (no faces)
Enhanced: → B-roll: 4/23 scenes (face_score < 0.3)
            Scene 5: temporal=0.15, spatial=0.08, score=0.012 ✓ B-roll
            Scene 12: temporal=0.85, spatial=0.35, score=0.298 ✓ B-roll (borderline)
            Scene 18: temporal=0.90, spatial=0.40, score=0.360 ✗ Not B-roll
```

## Backward Compatibility

| Aspect | Handling |
|--------|----------|
| Old cache files | Read old format (single float), write new format |
| Existing checkpoints | face_score field preserved, new fields default to 0.5 |
| Config migration | `face_sample_frames: 3` continues to work (explicit count) |
| API compatibility | `get_scene_face_score()` returns object with `.face_score` property |

## Verification

1. **Unit tests** - Test spatial calculation with known bounding boxes
2. **Clear cache** - `rm -rf .cache/scene_detection/` to force recalculation
3. **Run pipeline** - Check logs for new temporal/spatial metrics
4. **Compare results** - Same video should show different B-roll classification
5. **Edge cases**:
   - Very short scenes (< 1 second)
   - Very long scenes (> 60 seconds)
   - No faces detected (should return 0.0)
   - Multiple overlapping faces (spatial capped at 1.0)

## Performance Impact

| Metric | Current | New | Impact |
|--------|---------|-----|--------|
| Frames per scene | 3 fixed | 10-60 adaptive | 3-20x more frames |
| Detection per frame | Binary check | Bbox extraction | Minimal overhead |
| Cache size | ~50 bytes/scene | ~150 bytes/scene | 3x larger |
| Processing time | ~0.5s/scene | ~2-5s/scene | 4-10x slower |

**Mitigation:** Results are cached, so only first run is slower.

---

## Risk Analysis: 20 Failure Modes & Mitigations

### Category 1: Detection Failures

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 1 | **MediaPipe bbox format varies by version** - `relative_bounding_box` vs `bounding_box` vs `location_data.relative_bounding_box` | Silent wrong values or AttributeError crash | Try multiple attribute paths with fallback chain |
| 2 | **Normalized vs pixel coordinate mixing** - MediaPipe returns 0-1, OpenCV returns pixels | Spatial scores of 0.000001 or 50000+ | Detect coordinate system by checking if values > 1, normalize accordingly |
| 3 | **Frame extraction fails silently** - `cap.read()` returns `(False, None)` on corrupt frames | Division by zero if all frames fail | Track `frames_succeeded`, require minimum 50% success rate, return neutral 0.5 on total failure |
| 4 | **Codec incompatibility with seeking** - MXF, ProRes, some H.265 don't support random access | Always returns frame 0, wrong temporal coverage | Detect seek failure (frame number doesn't change), fall back to sequential read with skip |
| 5 | **Variable frame rate (VFR) videos** - Timestamp-based vs frame-based seeking differs | Wrong frames sampled, temporal coverage inaccurate | Use timestamp-based seeking (`CAP_PROP_POS_MSEC`) instead of frame-based |

### Category 2: Spatial Calculation Errors

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 6 | **Bounding boxes outside frame bounds** - Negative coordinates or values > 1.0 for edge faces | Negative area or > 100% coverage | Clamp all bbox values to [0, 1] before calculation |
| 7 | **Multiple overlapping faces sum > 100%** - Two 60% faces = 120% | Inflated spatial coverage | Cap per-frame total at 1.0 (already in plan), but also consider: should overlapping faces count less? |
| 8 | **Tiny false positive faces** - Background patterns detected as 0.1% faces | Hundreds accumulate to significant coverage | Add minimum face size threshold (e.g., ignore faces < 1% of frame) |
| 9 | **Profile faces return smaller bbox** - `model_selection=0` optimized for frontal | Underestimated spatial coverage for side angles | Use `model_selection=1` (full-range) or accept as acceptable inaccuracy |
| 10 | **Partial faces at frame edge** - Face walking in shows 50% of actual face | Bbox smaller than actual face presence | Accept as feature (partial face = partial coverage is correct behavior) |

### Category 3: Temporal Calculation Errors

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 11 | **Very short scene edge case** - 0.5s scene → `int(0.5 * 2) = 1` sample | Binary outcome, defeats purpose | Enforce minimum 3 samples regardless of duration |
| 12 | **Scene boundary misalignment** - Face at 9.8-10.2s, scene starts at 10.0s | First sample at 10.5s misses face entirely | Add 0.5s buffer before first sample, after last sample |
| 13 | **Motion blur on sampled frames** - Fast camera pan = blurry face | Face detection fails despite visible face | Sample adjacent frames on detection failure (±2 frames) |
| 14 | **Periodic sampling aliasing** - Someone nods every 2s, sampling at 1fps | Always catches "no face" moment | Add jitter to sample positions (±10% random offset) |

### Category 4: Cache & Compatibility Issues

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 15 | **Cache format migration race condition** - Two processes read/write simultaneously | Data corruption, old format overwrites new | Atomic write with temp file + rename, file locking |
| 16 | **Checkpoint expects float, gets FaceMetrics** - Old code: `score = get_face_score()`, new returns object | TypeError or AttributeError | Make FaceMetrics behave like float: `__float__()` method returns `.face_score` |
| 17 | **Cache key collision** - `path:10.0-20.0` vs `path:10.00-20.00` | Cache miss, redundant computation | Normalize time format: always `f"{time:.2f}"` |
| 18 | **Disk cache grows unbounded** - 500 videos × 50 scenes × 150 bytes | Gigabytes over time across projects | Add cache TTL (30 days), LRU eviction, max size limit |

### Category 5: Performance & Resource Issues

| # | Risk | Impact | Mitigation |
|---|------|--------|------------|
| 19 | **Memory explosion with long videos** - 60-min video, frames held in memory | OOM crash | Process frames one at a time, release immediately after detection |
| 20 | **GPU/CPU contention** - MediaPipe GPU + parallel transcription | GPU OOM or 10x slowdown | Limit MediaPipe to CPU mode for face detection, or serialize with transcription |

---

## Defensive Code Additions

### Step 1 Addition: Bbox Safety

```python
def _safe_bbox_area(self, bbox, is_normalized: bool = True) -> float:
    """
    Safely calculate face area from bounding box.
    Handles edge cases: negative coords, out-of-bounds, wrong format.
    """
    try:
        # Handle different MediaPipe versions
        if hasattr(bbox, 'xmin'):
            x, y, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
        elif hasattr(bbox, 'x_min'):
            x, y, w, h = bbox.x_min, bbox.y_min, bbox.width, bbox.height
        else:
            # Assume tuple/list format
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        # Detect if pixel coordinates (values > 1 = not normalized)
        if not is_normalized and (w > 1 or h > 1):
            # Will be normalized by caller with frame dimensions
            pass

        # Clamp to valid range [0, 1]
        x = max(0.0, min(1.0, float(x)))
        y = max(0.0, min(1.0, float(y)))
        w = max(0.0, min(1.0 - x, float(w)))  # Can't extend past frame
        h = max(0.0, min(1.0 - y, float(h)))

        area = w * h

        # Ignore tiny false positives (< 1% of frame)
        if area < 0.01:
            return 0.0

        return area

    except (AttributeError, TypeError, IndexError) as e:
        logger.warning(f"Failed to extract bbox area: {e}")
        return 0.0
```

### Step 2 Addition: Robust Sampling

```python
def _calculate_sample_count(self, duration: float, max_samples: int = 60) -> int:
    """Adaptive sampling with minimum guarantee."""
    MIN_SAMPLES = 3  # Never go below this (Risk #11)

    if duration <= 0:
        return MIN_SAMPLES
    elif duration <= 5:
        count = int(duration * 2)
    elif duration <= 30:
        count = int(duration)
    else:
        count = int(duration * 0.5)

    return max(MIN_SAMPLES, min(count, max_samples))


def _get_sample_positions(self, start_time: float, end_time: float,
                          num_samples: int) -> List[float]:
    """
    Get sample timestamps with jitter to avoid aliasing (Risk #14).
    Adds buffer at boundaries (Risk #12).
    """
    import random

    duration = end_time - start_time
    if duration <= 0 or num_samples <= 0:
        return []

    # Add 0.5s buffer at boundaries (but not more than 10% of duration)
    buffer = min(0.5, duration * 0.1)
    effective_start = start_time + buffer
    effective_end = end_time - buffer
    effective_duration = effective_end - effective_start

    if effective_duration <= 0:
        # Scene too short for buffer, just sample middle
        return [start_time + duration / 2]

    positions = []
    for i in range(num_samples):
        # Base position (evenly spaced)
        base_pos = effective_start + (effective_duration * (i + 0.5) / num_samples)

        # Add jitter (±10% of interval)
        interval = effective_duration / num_samples
        jitter = random.uniform(-0.1, 0.1) * interval

        pos = max(start_time, min(end_time, base_pos + jitter))
        positions.append(pos)

    return positions
```

### Step 3 Addition: FaceMetrics Float Compatibility

```python
@dataclass
class FaceMetrics:
    """Detailed face detection metrics for a scene"""
    temporal_coverage: float
    spatial_coverage: float
    face_score: float
    frames_sampled: int
    frames_with_faces: int
    max_spatial: float

    # Risk #16: Make compatible with code expecting float
    def __float__(self) -> float:
        return self.face_score

    def __lt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.face_score < other
        return self.face_score < other.face_score

    def __gt__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return self.face_score > other
        return self.face_score > other.face_score

    def __eq__(self, other) -> bool:
        if isinstance(other, (int, float)):
            return abs(self.face_score - other) < 0.0001
        return self.face_score == other.face_score
```

### Step 7 Addition: Safe Cache Operations

```python
def _read_cache_entry(self, cache_data: dict, key: str) -> Optional[FaceMetrics]:
    """Read cache entry, handling old and new formats (Risk #15, #17)."""
    # Normalize key format
    key = self._normalize_cache_key(key)

    if key not in cache_data:
        return None

    entry = cache_data[key]

    # Old format: single float
    if isinstance(entry, (int, float)):
        return FaceMetrics(
            temporal_coverage=entry,  # Assume old score was temporal
            spatial_coverage=1.0,     # Unknown, assume worst case
            face_score=entry,
            frames_sampled=3,         # Old default
            frames_with_faces=int(entry * 3),
            max_spatial=entry
        )

    # New format: dict with version
    if isinstance(entry, dict):
        return FaceMetrics(
            temporal_coverage=entry.get('temporal', 0.5),
            spatial_coverage=entry.get('spatial', 0.5),
            face_score=entry.get('face_score', 0.5),
            frames_sampled=entry.get('frames_sampled', 0),
            frames_with_faces=entry.get('frames_with_faces', 0),
            max_spatial=entry.get('max_spatial', 0.5)
        )

    return None


def _write_cache_atomic(self, cache_path: Path, cache_data: dict):
    """Atomic cache write to prevent corruption (Risk #15)."""
    import tempfile

    temp_path = cache_path.with_suffix('.tmp')
    try:
        with open(temp_path, 'w') as f:
            json.dump(cache_data, f)

        # Atomic rename (works on POSIX, may fail on Windows if dest exists)
        if temp_path.exists():
            if cache_path.exists():
                cache_path.unlink()
            temp_path.rename(cache_path)
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")
        if temp_path.exists():
            temp_path.unlink()


def _normalize_cache_key(self, key: str) -> str:
    """Normalize cache key to prevent collisions (Risk #17)."""
    # Split path:start-end format
    if ':' in key:
        path, times = key.rsplit(':', 1)
        if '-' in times:
            start, end = times.split('-', 1)
            try:
                start = f"{float(start):.2f}"
                end = f"{float(end):.2f}"
                return f"{path}:{start}-{end}"
            except ValueError:
                pass
    return key
```

### Frame Reading Safety (Risk #3, #4, #5)

```python
def _read_frame_at_time(self, cap, timestamp_sec: float) -> Optional[np.ndarray]:
    """
    Safely read frame at timestamp with fallback strategies.
    Returns None if frame cannot be read.
    """
    # Try timestamp-based seeking first (Risk #5: VFR videos)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000)

    ret, frame = cap.read()
    if ret and frame is not None:
        return frame

    # Fallback: frame-based seeking
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_num = int(timestamp_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()
    if ret and frame is not None:
        return frame

    # Risk #13: Try adjacent frames on failure
    for offset in [-2, -1, 1, 2]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + offset)
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame

    return None


def _detect_faces_in_scene(self, video_path: str, start_time: float,
                           end_time: float, sample_count: int) -> FaceMetrics:
    """Main detection with all safety measures."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return FaceMetrics(0.5, 0.5, 0.25, 0, 0, 0.5)  # Neutral fallback

    try:
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        sample_times = self._get_sample_positions(start_time, end_time, sample_count)

        frames_succeeded = 0
        frames_with_faces = 0
        spatial_scores = []

        for ts in sample_times:
            frame = self._read_frame_at_time(cap, ts)
            if frame is None:
                continue  # Skip failed frames

            frames_succeeded += 1

            # Detect faces and calculate spatial coverage
            spatial = self._detect_faces_spatial(frame, frame_width, frame_height)
            spatial_scores.append(spatial)

            if spatial > 0:
                frames_with_faces += 1

        # Risk #3: Require minimum success rate
        if frames_succeeded < len(sample_times) * 0.5:
            logger.warning(f"Only {frames_succeeded}/{len(sample_times)} frames readable")
            if frames_succeeded == 0:
                return FaceMetrics(0.5, 0.5, 0.25, 0, 0, 0.5)

        temporal = frames_with_faces / frames_succeeded if frames_succeeded > 0 else 0.5
        spatial_avg = sum(spatial_scores) / len(spatial_scores) if spatial_scores else 0.5

        return FaceMetrics(
            temporal_coverage=temporal,
            spatial_coverage=spatial_avg,
            face_score=temporal * spatial_avg,
            frames_sampled=frames_succeeded,
            frames_with_faces=frames_with_faces,
            max_spatial=max(spatial_scores) if spatial_scores else 0.0
        )

    finally:
        cap.release()  # Risk #19: Always release resources
```

---

## Updated Verification Checklist

1. **Unit tests for safety code:**
   - [ ] Bbox clamping with negative/overflow values
   - [ ] Cache format migration (old float → new dict)
   - [ ] FaceMetrics float comparison operators
   - [ ] Sample position jitter stays within bounds
   - [ ] Minimum 3 samples enforced for 0.1s scene

2. **Integration tests:**
   - [ ] Corrupt video file (some frames unreadable)
   - [ ] VFR video (variable frame rate)
   - [ ] Very short scene (< 1 second)
   - [ ] Very long scene (> 5 minutes)
   - [ ] Video with no faces
   - [ ] Video with faces in every frame

3. **Cache migration test:**
   - [ ] Create old-format cache, run new code, verify reads correctly
   - [ ] Verify new cache entries have version field
   - [ ] Concurrent access simulation (if possible)

4. **Performance verification:**
   - [ ] Memory usage stays flat during long video processing
   - [ ] Processing time scales linearly with scene count
   - [ ] Cache hit rate > 95% on second run
