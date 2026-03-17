"""
Cross-stage data flow integration tests (US-44-008).

Verifies that each stage's output format is compatible with the next stage's
expected input. Each test creates minimal mock output from stage N and passes
it through stage N+1's validate_required_state_attrs() and initial state checks.

This catches data contract violations like the stale PipelineState reference
bug (state.embeddings) that went undetected because each stage was tested in
isolation.

All tests use mock data and require no network access or API keys.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.state import (
    PipelineState,
    VoiceoverSegment,
    VideoSearchResult,
    Match,
)
from src.stages import validate_required_state_attrs


# =============================================================================
# Helpers: Minimal mock data factories
# =============================================================================


def make_voiceover_segments(count: int = 3) -> List[VoiceoverSegment]:
    """Create minimal voiceover segments as ANALYZE stage would produce."""
    return [
        VoiceoverSegment(
            index=i,
            start=i * 10.0,
            end=(i + 1) * 10.0 - 0.5,
            text=f"Segment {i} about wildlife in the savanna",
        )
        for i in range(count)
    ]


def make_keywords() -> List[str]:
    """Create keywords as ANALYZE stage would produce."""
    return ["wildlife", "savanna", "documentary", "nature"]


def make_video_ids(count: int = 5) -> List[str]:
    """Create 11-char YouTube video IDs as VIDEO_SEARCH would produce."""
    # 11-character IDs like real YouTube video IDs
    base_ids = ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0", "kJQP7kiw5Fk", "RgKAFK5djSk"]
    return base_ids[:count]


def make_video_search_results(video_ids: List[str]) -> List[VideoSearchResult]:
    """Create VideoSearchResult list as VIDEO_SEARCH would produce."""
    return [
        VideoSearchResult(
            video_id=vid,
            url=f"https://www.youtube.com/watch?v={vid}",
            title=f"Test Video {i}",
            channel=f"Channel {i}",
            duration=120.0 + i * 30,
            duration_tier="medium",
            keyword="wildlife",
        )
        for i, vid in enumerate(video_ids)
    ]


def make_caption_results(video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Create caption_results dict as CAPTION stage would produce."""
    results = {}
    for vid in video_ids:
        results[vid] = {
            "video_id": vid,
            "segments": [
                {"text": f"Caption segment 1 for {vid}", "start": 0.0, "end": 5.0},
                {"text": f"Caption segment 2 for {vid}", "start": 5.0, "end": 10.0},
                {"text": f"Caption segment 3 for {vid}", "start": 10.0, "end": 15.0},
            ],
            "language": "en",
            "is_auto_generated": False,
            "format_source": "manual",
            "segment_count": 3,
            "caption_quality": "high",
            "video_duration": 120.0,
            "coverage_ratio": 0.85,
            "timing_penalty": 0.0,
        }
    return results


def make_text_metadata(video_ids: List[str]) -> List[Dict[str, Any]]:
    """Create text_metadata list as CAPTION stage would populate.

    This is the key data structure consumed by MATCH and ITERATIVE_MATCH.
    Each entry represents one caption segment with video reference.
    """
    metadata = []
    for vid in video_ids:
        for seg_idx in range(3):
            metadata.append({
                "text": f"Caption segment {seg_idx} for {vid}",
                "video_path": vid,  # caption-first: video ID, not file path
                "source_file": vid,  # caption-first: video ID
                "start_time": seg_idx * 5.0,
                "end_time": (seg_idx + 1) * 5.0,
                "caption_source": "youtube",
                "caption_language": "en",
                "caption_auto_generated": False,
                "caption_quality": "high",
                "timing_penalty": 0.0,
            })
    return metadata


def make_matches(segment_count: int = 3, video_ids: List[str] = None) -> List[Match]:
    """Create Match objects as MATCH stage would produce."""
    if video_ids is None:
        video_ids = make_video_ids(3)
    return [
        Match(
            segment_index=i,
            video_file=video_ids[i % len(video_ids)],
            video_start=0.0,
            video_end=10.0,
            confidence=0.75 + (i * 0.05),
            strategy="caption_similarity",
            reason="Best caption match",
            face_score=0.5,
        )
        for i in range(segment_count)
    ]


def make_alternatives(matches: List[Match], video_ids: List[str] = None) -> Dict[int, List[Match]]:
    """Create alternatives dict as MATCH stage would produce."""
    if video_ids is None:
        video_ids = make_video_ids(5)
    alternatives = {}
    for m in matches:
        alt_vid = video_ids[(m.segment_index + 1) % len(video_ids)]
        alternatives[m.segment_index] = [
            Match(
                segment_index=m.segment_index,
                video_file=alt_vid,
                video_start=5.0,
                video_end=15.0,
                confidence=m.confidence - 0.1,
                strategy="caption_similarity",
                reason="Alternative match",
                face_score=0.4,
            )
        ]
    return alternatives


# =============================================================================
# Test: ANALYZE -> VIDEO_SEARCH transition
# =============================================================================


class TestAnalyzeToVideoSearch:
    """Verify ANALYZE output is valid input for VIDEO_SEARCH."""

    def test_analyze_output_has_required_fields(self):
        """ANALYZE must produce keywords and voiceover_segments for VIDEO_SEARCH."""
        state = PipelineState()
        # Simulate ANALYZE output
        state.voiceover_segments = make_voiceover_segments()
        state.keywords = make_keywords()
        state.topic_context = "African wildlife documentary"
        state.extracted_entities = [{"name": "Serengeti", "type": "location"}]

        # VIDEO_SEARCH requires 'keywords' attribute
        validate_required_state_attrs(state, ['keywords'], 'VIDEO_SEARCH')

        assert len(state.keywords) > 0, "VIDEO_SEARCH needs non-empty keywords"
        assert isinstance(state.keywords, list), "keywords must be a list"
        assert all(isinstance(k, str) for k in state.keywords), "keywords must be strings"

    def test_voiceover_segments_structure(self):
        """ANALYZE voiceover_segments must have required fields for downstream stages."""
        segments = make_voiceover_segments(5)
        for seg in segments:
            assert hasattr(seg, 'index'), "VoiceoverSegment must have index"
            assert hasattr(seg, 'start'), "VoiceoverSegment must have start"
            assert hasattr(seg, 'end'), "VoiceoverSegment must have end"
            assert hasattr(seg, 'text'), "VoiceoverSegment must have text"
            assert hasattr(seg, 'duration'), "VoiceoverSegment must have duration"
            assert seg.duration > 0, "duration must be positive"
            assert seg.end > seg.start, "end must be after start"

    def test_empty_keywords_detected(self):
        """VIDEO_SEARCH should detect empty keywords from ANALYZE."""
        state = PipelineState()
        state.keywords = []  # ANALYZE produced no keywords

        # validate_required_state_attrs only ensures attr exists, not non-empty
        validate_required_state_attrs(state, ['keywords'], 'VIDEO_SEARCH')
        assert state.keywords == [], "Empty keywords should pass attr validation but flag downstream"

    def test_topic_context_optional(self):
        """VIDEO_SEARCH should work even if topic_context is empty."""
        state = PipelineState()
        state.keywords = make_keywords()
        state.topic_context = ""  # Optional

        validate_required_state_attrs(state, ['keywords'], 'VIDEO_SEARCH')
        assert isinstance(state.topic_context, str)


# =============================================================================
# Test: VIDEO_SEARCH -> CAPTION transition
# =============================================================================


class TestVideoSearchToCaption:
    """Verify VIDEO_SEARCH output is valid input for CAPTION."""

    def test_video_search_output_has_video_ids(self):
        """VIDEO_SEARCH must produce video_ids that CAPTION can process."""
        state = PipelineState()
        video_ids = make_video_ids(5)
        state.video_ids = video_ids
        state.video_search_results = make_video_search_results(video_ids)

        # CAPTION requires 'video_ids' attribute
        validate_required_state_attrs(state, ['video_ids'], 'CAPTION')

        assert len(state.video_ids) > 0
        assert all(isinstance(vid, str) for vid in state.video_ids)
        # YouTube video IDs are 11 characters
        assert all(len(vid) == 11 for vid in state.video_ids), \
            "Video IDs must be 11-character YouTube IDs"

    def test_video_ids_are_pure_ids_not_urls(self):
        """CAPTION expects pure video IDs, not full YouTube URLs (Rule 22)."""
        state = PipelineState()
        state.video_ids = make_video_ids(3)

        for vid in state.video_ids:
            assert "youtube.com" not in vid, f"video_ids must be pure IDs, got URL: {vid}"
            assert "http" not in vid, f"video_ids must be pure IDs, got URL: {vid}"
            assert "/" not in vid, f"video_ids must not contain slashes: {vid}"

    def test_caption_preflight_check_passes(self):
        """CAPTION's _preflight_check should accept VIDEO_SEARCH output."""
        from src.stages.caption_stage import CaptionStage

        state = PipelineState()
        state.video_ids = make_video_ids(5)
        state.video_search_results = make_video_search_results(state.video_ids)

        stage = CaptionStage.__new__(CaptionStage)
        stage._fetcher = None
        stage._config_validated = False

        # _preflight_check initializes missing attrs and validates
        stage._preflight_check(state)

        # After preflight, required attrs should exist
        assert hasattr(state, 'caption_results')
        assert hasattr(state, 'text_metadata')
        assert hasattr(state, 'video_ids')

    def test_video_search_results_structure(self):
        """VideoSearchResult from VIDEO_SEARCH must have fields CAPTION may use."""
        video_ids = make_video_ids(3)
        results = make_video_search_results(video_ids)

        for r in results:
            assert hasattr(r, 'video_id'), "VideoSearchResult must have video_id"
            assert hasattr(r, 'duration'), "VideoSearchResult must have duration"
            assert hasattr(r, 'title'), "VideoSearchResult must have title"
            assert r.video_id in video_ids, "video_id must match"

    def test_empty_video_ids_handled(self):
        """CAPTION should handle empty video_ids from VIDEO_SEARCH gracefully."""
        state = PipelineState()
        state.video_ids = []

        # validate_required_state_attrs ensures attr exists
        validate_required_state_attrs(state, ['video_ids'], 'CAPTION')
        assert state.video_ids == []


# =============================================================================
# Test: CAPTION -> MATCH transition
# =============================================================================


class TestCaptionToMatch:
    """Verify CAPTION output is valid input for MATCH."""

    def test_caption_output_has_required_fields(self):
        """CAPTION must produce text_metadata, caption_results, and they must
        be compatible with MATCH's validate_required_state_attrs call."""
        state = PipelineState()
        video_ids = make_video_ids(3)
        state.voiceover_segments = make_voiceover_segments(3)
        state.caption_results = make_caption_results(video_ids)
        state.text_metadata = make_text_metadata(video_ids)

        # MATCH validates these three attributes
        validate_required_state_attrs(
            state,
            ['text_metadata', 'caption_results', 'voiceover_segments'],
            'MATCH'
        )

        assert len(state.text_metadata) > 0, "MATCH needs non-empty text_metadata"
        assert len(state.caption_results) > 0, "MATCH needs caption_results for recovery"
        assert len(state.voiceover_segments) > 0, "MATCH needs voiceover_segments"

    def test_text_metadata_has_required_keys(self):
        """Each text_metadata entry must have keys MATCH uses for segment matching."""
        video_ids = make_video_ids(3)
        metadata = make_text_metadata(video_ids)

        required_keys = {'text', 'video_path', 'source_file', 'start_time', 'end_time'}
        for entry in metadata:
            missing = required_keys - set(entry.keys())
            assert not missing, f"text_metadata entry missing keys: {missing}"
            # caption-first mode: video_path and source_file are video IDs
            assert isinstance(entry['text'], str), "text must be string"
            assert isinstance(entry['start_time'], (int, float)), "start_time must be numeric"
            assert isinstance(entry['end_time'], (int, float)), "end_time must be numeric"

    def test_caption_results_keyed_by_video_id(self):
        """caption_results must be dict keyed by video_id for MATCH recovery."""
        video_ids = make_video_ids(3)
        results = make_caption_results(video_ids)

        assert isinstance(results, dict), "caption_results must be dict"
        for vid in video_ids:
            assert vid in results, f"caption_results must contain {vid}"
            data = results[vid]
            assert 'segments' in data, "Each caption must have segments"
            assert isinstance(data['segments'], list), "segments must be list"

    def test_caption_segments_have_timing(self):
        """Caption segments must have start/end for MATCH to create text_metadata."""
        video_ids = make_video_ids(2)
        results = make_caption_results(video_ids)

        for vid, data in results.items():
            for seg in data['segments']:
                assert 'text' in seg, "Caption segment must have text"
                assert 'start' in seg, "Caption segment must have start time"
                assert 'end' in seg, "Caption segment must have end time"
                assert seg['end'] > seg['start'], "end must be after start"

    def test_text_metadata_video_path_is_id_not_filepath(self):
        """In caption-first mode, video_path must be a video ID, not a file path (Rule 22)."""
        video_ids = make_video_ids(3)
        metadata = make_text_metadata(video_ids)

        for entry in metadata:
            vid_path = entry['video_path']
            # Should be an 11-char video ID, not a file path
            assert "\\" not in vid_path, f"video_path should be ID, not path: {vid_path}"
            assert ":" not in vid_path, f"video_path should be ID, not path: {vid_path}"
            assert vid_path in video_ids, f"video_path {vid_path} not in video_ids"

    def test_match_recovery_from_caption_results(self):
        """If text_metadata is empty, MATCH should be able to recover from caption_results."""
        state = PipelineState()
        state.voiceover_segments = make_voiceover_segments(3)
        state.text_metadata = []  # Empty - simulates partial CAPTION failure
        state.caption_results = make_caption_results(make_video_ids(3))

        # MATCH's validate_required_state_attrs should pass (attrs exist)
        validate_required_state_attrs(
            state,
            ['text_metadata', 'caption_results', 'voiceover_segments'],
            'MATCH'
        )

        # MATCH checks: not text_metadata and caption_results -> recovery
        assert not state.text_metadata, "text_metadata should be empty for recovery path"
        assert state.caption_results, "caption_results should be available for recovery"


# =============================================================================
# Test: MATCH -> ITERATIVE_MATCH transition
# =============================================================================


class TestMatchToIterativeMatch:
    """Verify MATCH output is valid input for ITERATIVE_MATCH."""

    def test_match_output_has_required_fields(self):
        """MATCH must produce matches and text_metadata for ITERATIVE_MATCH."""
        state = PipelineState()
        video_ids = make_video_ids(5)
        state.matches = make_matches(3, video_ids)
        state.text_metadata = make_text_metadata(video_ids)
        state.voiceover_segments = make_voiceover_segments(3)

        # ITERATIVE_MATCH validates these attributes
        validate_required_state_attrs(
            state,
            ['matches', 'text_metadata'],
            'ITERATIVE_MATCH'
        )

        assert len(state.matches) > 0, "ITERATIVE_MATCH needs matches"
        assert len(state.text_metadata) > 0, "ITERATIVE_MATCH needs text_metadata candidates"

    def test_match_objects_have_required_fields(self):
        """Match objects from MATCH must have fields ITERATIVE_MATCH inspects."""
        video_ids = make_video_ids(3)
        matches = make_matches(3, video_ids)

        for m in matches:
            assert hasattr(m, 'segment_index'), "Match must have segment_index"
            assert hasattr(m, 'video_file'), "Match must have video_file"
            assert hasattr(m, 'video_start'), "Match must have video_start"
            assert hasattr(m, 'video_end'), "Match must have video_end"
            assert hasattr(m, 'confidence'), "Match must have confidence"
            assert hasattr(m, 'strategy'), "Match must have strategy"
            assert hasattr(m, 'face_score'), "Match must have face_score"
            assert 0.0 <= m.confidence <= 1.0, f"confidence out of range: {m.confidence}"

    def test_match_video_file_references_valid_videos(self):
        """Match.video_file should reference video IDs present in text_metadata."""
        video_ids = make_video_ids(5)
        matches = make_matches(3, video_ids)
        metadata = make_text_metadata(video_ids)

        metadata_video_ids = {entry['video_path'] for entry in metadata}
        for m in matches:
            assert m.video_file in metadata_video_ids, \
                f"Match video_file '{m.video_file}' not found in text_metadata video_paths"

    def test_iterative_match_gap_detection_compatible(self):
        """ITERATIVE_MATCH identifies gaps using confidence and voiceover_segments.
        Verify the data is compatible with gap detection logic."""
        state = PipelineState()
        video_ids = make_video_ids(5)
        segments = make_voiceover_segments(5)
        state.voiceover_segments = segments

        # Create matches with some low-confidence gaps
        matches = []
        for i in range(5):
            conf = 0.95 if i % 2 == 0 else 0.6  # Alternating high/low confidence
            matches.append(Match(
                segment_index=i,
                video_file=video_ids[i % len(video_ids)],
                video_start=0.0,
                video_end=10.0,
                confidence=conf,
                strategy="caption_similarity",
                reason="test",
                face_score=0.5,
            ))
        state.matches = matches
        state.text_metadata = make_text_metadata(video_ids)

        # ITERATIVE_MATCH uses segment_index to correlate matches to voiceover_segments
        match_indices = {m.segment_index for m in state.matches}
        segment_indices = {s.index for s in state.voiceover_segments}
        assert match_indices.issubset(segment_indices), \
            "All match segment_indices must reference valid voiceover segment indices"

    def test_alternatives_dict_structure(self):
        """MATCH alternatives dict must have segment_index keys and Match list values."""
        video_ids = make_video_ids(5)
        matches = make_matches(3, video_ids)
        alternatives = make_alternatives(matches, video_ids)

        assert isinstance(alternatives, dict)
        for seg_idx, alt_list in alternatives.items():
            assert isinstance(seg_idx, int), "alternatives key must be int"
            assert isinstance(alt_list, list), "alternatives value must be list"
            for alt in alt_list:
                assert isinstance(alt, Match), "alternatives entries must be Match objects"
                assert alt.segment_index == seg_idx

    def test_text_metadata_preserved_through_match(self):
        """text_metadata should be available for ITERATIVE_MATCH after MATCH completes.
        MATCH reads but does not destroy text_metadata."""
        state = PipelineState()
        video_ids = make_video_ids(5)
        state.text_metadata = make_text_metadata(video_ids)
        state.caption_results = make_caption_results(video_ids)
        state.voiceover_segments = make_voiceover_segments(3)
        state.matches = make_matches(3, video_ids)

        # After MATCH, text_metadata should still be available for ITERATIVE_MATCH
        validate_required_state_attrs(state, ['matches', 'text_metadata'], 'ITERATIVE_MATCH')
        assert len(state.text_metadata) > 0, "text_metadata must survive through MATCH"


# =============================================================================
# Test: Full chain validation (lightweight end-to-end)
# =============================================================================


class TestFullChainDataFlow:
    """Verify data flows correctly through the full ANALYZE -> ... -> ITERATIVE_MATCH chain."""

    def test_full_chain_state_population(self):
        """Simulate populating state through all stages and verify compatibility."""
        state = PipelineState()

        # === Stage 1: ANALYZE output ===
        state.voiceover_path = "test_voiceover.srt"
        state.voiceover_segments = make_voiceover_segments(5)
        state.keywords = make_keywords()
        state.topic_context = "African wildlife documentary"
        state.extracted_entities = [
            {"name": "Serengeti", "type": "location"},
            {"name": "elephant", "type": "animal"},
        ]

        # Validate for VIDEO_SEARCH
        validate_required_state_attrs(state, ['keywords'], 'VIDEO_SEARCH')
        assert state.keywords

        # === Stage 2: VIDEO_SEARCH output ===
        video_ids = make_video_ids(5)
        state.video_ids = video_ids
        state.video_search_results = make_video_search_results(video_ids)
        state.search_failed_keywords = []

        # Validate for CAPTION
        validate_required_state_attrs(state, ['video_ids'], 'CAPTION')
        assert state.video_ids

        # === Stage 3: CAPTION output ===
        state.caption_results = make_caption_results(video_ids)
        state.text_metadata = make_text_metadata(video_ids)

        # Validate for MATCH
        validate_required_state_attrs(
            state,
            ['text_metadata', 'caption_results', 'voiceover_segments'],
            'MATCH'
        )
        assert state.text_metadata
        assert state.caption_results
        assert state.voiceover_segments

        # === Stage 4: MATCH output ===
        state.matches = make_matches(5, video_ids)
        state.alternatives = make_alternatives(state.matches, video_ids)

        # Validate for ITERATIVE_MATCH
        validate_required_state_attrs(
            state,
            ['matches', 'text_metadata'],
            'ITERATIVE_MATCH'
        )
        assert state.matches
        assert state.text_metadata

        # Verify match segment indices align with voiceover segments
        for m in state.matches:
            assert 0 <= m.segment_index < len(state.voiceover_segments), \
                f"Match segment_index {m.segment_index} out of range"

    def test_missing_intermediate_state_detected(self):
        """Skipping a stage should be detectable via missing/empty state fields."""
        state = PipelineState()

        # ANALYZE done
        state.voiceover_segments = make_voiceover_segments(3)
        state.keywords = make_keywords()

        # Skip VIDEO_SEARCH and go directly to CAPTION
        # CAPTION's validate should still pass (attrs exist with defaults)
        validate_required_state_attrs(state, ['video_ids'], 'CAPTION')
        # But video_ids will be empty (default)
        assert state.video_ids == [], "Missing VIDEO_SEARCH should leave video_ids empty"

    def test_state_types_consistent_across_stages(self):
        """Verify PipelineState field types match what stages expect."""
        state = PipelineState()

        # Check default types match stage expectations
        assert isinstance(state.keywords, list), "keywords should default to list"
        assert isinstance(state.video_ids, list), "video_ids should default to list"
        assert isinstance(state.caption_results, dict), "caption_results should default to dict"
        assert isinstance(state.text_metadata, list), "text_metadata should default to list"
        assert isinstance(state.matches, list), "matches should default to list"
        assert isinstance(state.alternatives, dict), "alternatives should default to dict"
        assert isinstance(state.voiceover_segments, list), "voiceover_segments should default to list"

    def test_validate_required_state_attrs_initializes_missing(self):
        """validate_required_state_attrs should initialize missing attrs with defaults."""
        # Use a bare object to simulate corrupted state
        state = object.__new__(PipelineState)

        # Remove attrs that would normally exist
        # (object.__new__ skips __init__, so no fields set)

        validate_required_state_attrs(
            state,
            ['text_metadata', 'caption_results', 'matches', 'video_ids', 'keywords'],
            'TEST'
        )

        # After validation, attrs should exist with sensible defaults
        assert hasattr(state, 'text_metadata')
        assert hasattr(state, 'caption_results')
        assert hasattr(state, 'matches')
        assert hasattr(state, 'video_ids')
        assert hasattr(state, 'keywords')

        # Check default types
        assert state.text_metadata == []
        assert state.caption_results == {}
        assert state.matches == []
        assert state.video_ids == []
        assert state.keywords == []
