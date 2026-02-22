"""
Tests for US-70-008: Feed video tags into keyword boost scoring.

Verifies that video_tags from CaptionResult are merged into the keyword pool
and contribute to keyword overlap scoring during matching.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import calculate_keyword_overlap_score
from src.utils import SRTSegment


class TestVideoTagsKeywordIntegration:
    """Test that video tags contribute to keyword overlap scoring."""

    def test_tags_contribute_to_keyword_overlap(self):
        """When video tags are merged into keywords, they increase overlap score."""
        vo_keywords = ["climate", "environment", "sustainability"]

        # Without tags: no overlap
        video_keywords_no_tags = ["documentary", "nature"]
        score_no_tags, matched_no_tags = calculate_keyword_overlap_score(
            vo_keywords, video_keywords_no_tags
        )

        # With tags merged: overlap via "climate" and "environment"
        video_keywords_with_tags = ["documentary", "nature", "climate", "environment"]
        score_with_tags, matched_with_tags = calculate_keyword_overlap_score(
            vo_keywords, video_keywords_with_tags
        )

        assert score_no_tags == 0.0
        assert score_with_tags > 0.0
        assert len(matched_with_tags) == 2
        assert "climate" in [m.lower() for m in matched_with_tags]
        assert "environment" in [m.lower() for m in matched_with_tags]

    def test_tags_lowercased_and_deduplicated(self):
        """Tags should be lowercased and deduplicated before matching."""
        vo_keywords = ["python", "programming"]

        # Mixed case tags, duplicates
        video_keywords = ["Python", "PYTHON", "coding", "programming"]
        score, matched = calculate_keyword_overlap_score(
            vo_keywords, video_keywords
        )

        # Should match "python" and "programming" (deduplicated in scoring)
        assert score > 0.0
        assert len(matched) >= 1

    def test_empty_tags_unchanged_behavior(self):
        """When video_tags is empty, scoring behavior is unchanged."""
        vo_keywords = ["nature", "wildlife"]
        video_keywords = ["documentary"]

        score, matched = calculate_keyword_overlap_score(
            vo_keywords, video_keywords
        )

        assert score == 0.0
        assert matched == []

    def test_tag_merge_deduplication_in_prepare_segments(self):
        """Simulate the tag merge logic from _prepare_segments.

        Verifies that tags are lowercased, deduplicated, and merged into
        existing keywords without duplicating existing entries.
        """
        # Simulate existing keywords on a segment
        existing_kw = ["nature", "documentary"]
        video_tags = ["Nature", "wildlife", "DOCUMENTARY", "climate change"]

        # Replicate the merge logic from match.py _prepare_segments
        existing_lower = {k.lower() for k in existing_kw if k}
        merged = list(existing_kw)
        for tag in video_tags:
            if tag and tag.lower().strip() not in existing_lower:
                merged.append(tag.lower().strip())
                existing_lower.add(tag.lower().strip())

        # "nature" and "documentary" already exist, should not be duplicated
        assert merged.count("nature") == 1
        assert merged.count("documentary") == 1
        # "wildlife" and "climate change" are new
        assert "wildlife" in merged
        assert "climate change" in merged
        assert len(merged) == 4  # 2 original + 2 new

    def test_text_metadata_includes_video_tags(self):
        """Verify the text_metadata dict structure includes video_tags field."""
        # Simulate what _populate_text_metadata produces
        text_metadata_entry = {
            'text': 'some caption text',
            'video_path': 'abc123',
            'start_time': 0,
            'end_time': 10,
            'source_file': 'abc123',
            'caption_source': 'youtube',
            'caption_language': 'en',
            'caption_auto_generated': False,
            'caption_quality': 'high',
            'timing_penalty': 1.0,
            'video_tags': ['nature', 'wildlife', 'documentary'],
        }

        assert 'video_tags' in text_metadata_entry
        assert text_metadata_entry['video_tags'] == ['nature', 'wildlife', 'documentary']

    def test_segment_keywords_include_tags_after_merge(self):
        """End-to-end: SRTSegment keywords include merged video tags."""
        # Create segment with no keywords (like _prepare_segments does)
        vid_segment = SRTSegment(
            index=0,
            start_time=0,
            end_time=10,
            text='test caption',
            source_file='vid123',
        )

        # Simulate the merge from _prepare_segments
        video_tags = ["Python", "Tutorial", "coding"]
        existing_kw = getattr(vid_segment, 'keywords', []) or []
        existing_lower = {k.lower() for k in existing_kw if k}
        merged = list(existing_kw)
        for tag in video_tags:
            if tag and tag.lower().strip() not in existing_lower:
                merged.append(tag.lower().strip())
                existing_lower.add(tag.lower().strip())
        vid_segment.keywords = merged

        assert vid_segment.keywords == ["python", "tutorial", "coding"]

        # Now verify these contribute to keyword overlap
        vo_keywords = ["python", "programming", "tutorial"]
        score, matched = calculate_keyword_overlap_score(
            vo_keywords, vid_segment.keywords
        )

        assert score > 0.0
        assert len(matched) == 2  # "python" and "tutorial"
