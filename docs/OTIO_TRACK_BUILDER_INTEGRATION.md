# OTIO Track Builder Integration Analysis

**Date**: 2026-01-09
**Status**: Track builders implemented and tested, integration deferred
**Decision**: Keep timeline.py monolithic implementation for stability

## Summary

Track builders are fully implemented with comprehensive testing (23/23 tests passing, 100%). However, after analyzing timeline.py for integration, we've decided to defer integration and keep the current monolithic implementation.

## What Was Built

### Track Builder Classes (src/otio/tracks.py)

- **PrimaryTrackBuilder** - V1/A1 primary tracks
- **AlternativeTrackBuilder** - V2-V3/A2-A3 alternative tracks
- **DiversityTrackBuilder** - V4-V6/A4-A6 secondary diversity tracks
- **EmbeddingDiversityTrackBuilder** - V7/A7 embedding-diversity strategy
- **BRollTrackBuilder** - V8/A8 B-roll only track
- **EntityImageTrackBuilder** - V9 entity images (Google/Bing)
- **EntityVideoTrackBuilder** - V10 stock videos (Pexels/Pixabay)

### Shared Helpers

- `_create_clip()` - Clip creation with audio-first resolution, legacy offsets
- `_create_gap()` - Gap clip creation
- `_add_gap_if_needed()` - Between-segment gap insertion (not currently used)

### Test Coverage

- **23 unit tests** covering factory function, track names, builder classes
- **Comprehensive mocks** - MockMatch, MockMatchResult, MockVoiceoverSegment, MockVideoSegment
- **100% test pass rate** for track builder tests

## Why Integration Was Deferred

### Gap Handling Complexity

Timeline.py has three types of gap handling that require timeline-wide coordination:

1. **Leading gaps** (lines 295-316)
   - Added if first voiceover segment doesn't start at 0
   - Applied to ALL video and audio tracks simultaneously
   - Uses `timeline_frames` position tracker

2. **Between-segment gaps** (lines 324-353)
   - Inserted when there's silence between voiceover segments
   - Requires checking `expected_start_frames` vs `timeline_frames`
   - Applied to ALL tracks to maintain sync

3. **Trailing gaps** (lines 684-710)
   - Added to match actual voiceover file duration
   - Ensures video tracks extend to cover trailing audio
   - Applied after all segments are processed

### Current Track Builder Limitations

Track builders currently handle **per-segment clip creation** but **not timeline-wide gap management**:

- ✅ Clip creation with audio-first resolution
- ✅ Confidence-based coloring
- ✅ Metadata building
- ✅ Per-segment gaps (when alternatives/secondaries missing)
- ❌ Leading gap insertion (requires timeline position)
- ❌ Between-segment gap detection (requires voiceover timing analysis)
- ❌ Trailing gap calculation (requires voiceover duration)
- ❌ Timeline position tracking (`timeline_frames`)

### Integration Challenges

To integrate track builders, we would need to:

1. **Extract gap logic into a separate coordinator**
   - Create `GapManager` or `TimelinePositionTracker` class
   - Handle leading/between/trailing gap insertion
   - Pass `timeline_frames` state to track builders

2. **Refactor track builders to accept timeline position**
   - Add `timeline_frames` parameter to `build()` method
   - Return updated position after building
   - Handle gap insertion between clips

3. **Modify timeline.py main loop**
   - Replace 363-line loop (lines 318-682) with track builder factory calls
   - Coordinate gap manager with track builders
   - Maintain backward compatibility

4. **Extensive testing required**
   - Test gap insertion with various voiceover patterns
   - Test audio-first mode resolution
   - Test all 10 track types with real match data
   - Verify DaVinci Resolve import still works

### Risk Assessment

**High risk** of introducing timing bugs:
- Gap calculation is sensitive to frame rounding
- Timeline position tracking uses integer frames to avoid drift
- DaVinci Resolve hangs on invalid timecode
- Audio-first mode adds additional complexity with segment resolution

**Low benefit** at current stage:
- Timeline.py works reliably in production
- Track builders are tested but not proven with real data
- Code reduction (~400 lines) doesn't justify risk
- No user-facing benefits, purely internal refactoring

## Current Value of Track Builders

Even without integration, track builders provide value:

### 1. Reference Implementation
- Demonstrates strategy pattern for OTIO generation
- Shows how to properly structure track creation
- Documents clip creation patterns

### 2. Test Infrastructure
- Comprehensive mock objects for OTIO testing
- Examples of testing timeline generation
- Patterns for testing with complex dataclasses

### 3. Future Refactoring Base
- When gap handling is refactored, track builders are ready
- Provides target architecture for timeline.py simplification
- Shows path forward for reducing monolithic functions

## Recommended Path Forward

### Short Term (Next 1-3 Months)

**Keep timeline.py unchanged**:
- Continue using monolithic implementation
- Focus on feature development and bug fixes
- Monitor for any timeline-related issues

### Medium Term (3-6 Months)

**If timeline.py becomes a pain point**:
- Extract gap handling into `GapManager` class
- Refactor `timeline_frames` tracking into `TimelinePosition` state object
- Add integration tests for gap handling

### Long Term (6+ Months)

**If gap handling is refactored**:
- Integrate track builders with new gap management
- Replace timeline.py loop with factory pattern
- Verify with extensive integration testing
- Deploy to production with careful monitoring

## Alternative Approaches

If we decide to integrate later, consider these approaches:

### Approach A: Hybrid Model
- Keep gap handling in timeline.py
- Use track builders ONLY for clip creation within segments
- Minimal changes to timeline.py

**Pros**: Lower risk, gradual migration
**Cons**: Still duplicates some logic, doesn't fully realize builder pattern

### Approach B: Timeline Builder Class
- Create `TimelineBuilder` class that coordinates:
  - Gap management
  - Track building
  - Position tracking
- Replace `create_timeline()` function entirely

**Pros**: Clean separation of concerns, testable components
**Cons**: Large refactoring, higher risk, more complex

### Approach C: Keep Current Architecture
- Accept timeline.py as monolithic
- Focus refactoring efforts elsewhere
- Revisit only if timeline generation becomes problematic

**Pros**: Zero risk, no wasted effort
**Cons**: Misses opportunity for cleaner architecture

## Conclusion

Track builders are **complete and ready** but **integration is not recommended** at this time due to gap handling complexity and low risk/benefit ratio. They serve as a valuable reference implementation and testbed for future refactoring.

Current timeline.py implementation is **stable, tested, and working**. The safest path forward is to leave it unchanged and focus development efforts on features and bug fixes.

## Files

- **Track builders**: [src/otio/tracks.py](src/otio/tracks.py)
- **Timeline implementation**: [src/otio/timeline.py](src/otio/timeline.py)
- **Track builder tests**: [tests/test_otio_tracks.py](tests/test_otio_tracks.py)
- **Timeline tests**: [tests/test_otio_timeline.py](tests/test_otio_timeline.py)
