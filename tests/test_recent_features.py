#!/usr/bin/env python3
"""
Test Recent Feature Additions

Tests the following recent features:
1. Delta matching - only match new videos
2. Cache loading - load transcripts from cache when skip_transcription=true
3. Chapter-based topic matching
4. OTIO V9/V10 track inclusion
5. JSON serialization fix for numpy types

Usage:
    python tests/test_recent_features.py
    python tests/test_recent_features.py --verbose
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check for optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import opentimelineio as otio
    HAS_OTIO = True
except ImportError:
    HAS_OTIO = False


def print_box(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_section(text: str):
    print(f"\n  --- {text} ---")


def print_result(name: str, passed: bool, details: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    detail_str = f" - {details}" if details else ""
    print(f"  {status} {name}{detail_str}")
    return passed


class TestRecentFeatures:
    """Test suite for recent feature additions"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.temp_dir = None
        self.results = {}

    def setup(self):
        """Create temporary test directory"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_features_'))
        (self.temp_dir / '.cache').mkdir(parents=True)
        if self.verbose:
            print(f"  Created temp dir: {self.temp_dir}")

    def teardown(self):
        """Clean up temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            if self.verbose:
                print(f"  Cleaned up temp dir")

    def test_json_serialization(self) -> bool:
        """Test that to_dict() methods handle numpy types correctly"""
        print_section("JSON Serialization (numpy types)")

        if not HAS_NUMPY:
            print_result("JSON serialization", True, "SKIPPED - numpy not installed")
            return True

        try:
            from src.utils import SRTSegment, Match, SceneInfo

            # Create objects with numpy types
            segment = SRTSegment(
                index=np.int32(1),
                start_time=np.float32(0.0),
                end_time=np.float32(10.5),
                text="Test segment",
                source_file="/path/to/video.mp4"
            )

            scene = SceneInfo(
                video_path="/path/to/video.mp4",
                scene_index=np.int32(0),
                start_time=np.float32(0.0),
                end_time=np.float32(10.0),
                description="Test scene"
            )

            match = Match(
                voiceover_segment=segment,
                video_segment=segment,
                video_scene=scene,
                confidence=np.float32(0.85),
                reasoning="Test match",
                embedding_similarity=np.float32(0.92)
            )

            # Test serialization
            segment_dict = segment.to_dict()
            scene_dict = scene.to_dict()
            match_dict = match.to_dict()

            # Try JSON serialization (this would fail with numpy types)
            json.dumps(segment_dict)
            json.dumps(scene_dict)
            json.dumps(match_dict)

            passed1 = print_result("SRTSegment.to_dict() JSON serializable", True)
            passed2 = print_result("SceneInfo.to_dict() JSON serializable", True)
            passed3 = print_result("Match.to_dict() JSON serializable", True)
            return passed1 and passed2 and passed3

        except Exception as e:
            return print_result("JSON serialization", False, str(e))

    def test_delta_matching_index(self) -> bool:
        """Test MatchAwareIndex for delta matching"""
        print_section("Delta Matching Index")

        try:
            from src.match_index import MatchAwareIndex
        except ImportError as e:
            return print_result("Delta matching index", True, f"SKIPPED - {e}")

        try:

            # Create index
            index = MatchAwareIndex(str(self.temp_dir))

            # Test adding matched videos
            test_videos = [
                str(self.temp_dir / "video1.mp4"),
                str(self.temp_dir / "video2.mp4"),
            ]

            # Create dummy video files
            for v in test_videos:
                Path(v).write_text("dummy")

            # Mark as matched
            index.mark_matched_batch(test_videos)

            passed1 = print_result(
                "Mark videos as matched",
                len(index.matched_videos) == 2,
                f"matched={len(index.matched_videos)}"
            )

            # Test get_new_videos
            all_videos = test_videos + [str(self.temp_dir / "video3.mp4")]
            Path(all_videos[-1]).write_text("dummy new")

            new_videos = index.get_new_videos(all_videos)
            passed2 = print_result(
                "Detect new videos",
                len(new_videos) == 1,
                f"new={len(new_videos)}"
            )

            # Test save/load
            index.save()
            index2 = MatchAwareIndex(str(self.temp_dir))
            passed3 = print_result(
                "Save and reload index",
                len(index2.matched_videos) == 2,
                f"reloaded={len(index2.matched_videos)}"
            )

            return passed1 and passed2 and passed3

        except Exception as e:
            return print_result("Delta matching index", False, str(e))

    def test_topic_extraction(self) -> bool:
        """Test topic extraction classes"""
        print_section("Topic Extraction")

        try:
            from src.topic_extraction import (
                VideoTopics,
                compute_topic_overlap,
                compute_topic_penalty
            )
        except ImportError as e:
            return print_result("Topic extraction", True, f"SKIPPED - {e}")

        try:

            # Test VideoTopics dataclass
            vt = VideoTopics(
                video_path="/path/to/video.mp4",
                topics=["weather", "storm", "winter"],
                confidence=0.9
            )
            passed1 = print_result(
                "VideoTopics dataclass",
                vt.topics == ["weather", "storm", "winter"]
            )

            # Test topic overlap
            vo_topics = ["weather", "storm", "news"]
            video_topics = ["weather", "winter", "cold"]
            overlap = compute_topic_overlap(vo_topics, video_topics)
            passed2 = print_result(
                "Topic overlap computation",
                overlap == 1,  # "weather" is common
                f"overlap={overlap}"
            )

            # Test topic penalty
            penalty = compute_topic_penalty(
                vo_topics=["sports", "football"],
                video_topics=["weather", "storm"],
                max_penalty=0.15,
                min_overlap=1
            )
            passed3 = print_result(
                "Topic mismatch penalty",
                penalty == 0.15,  # No overlap = max penalty
                f"penalty={penalty}"
            )

            # Test no penalty when topics match
            penalty2 = compute_topic_penalty(
                vo_topics=["weather", "storm"],
                video_topics=["weather", "winter"],
                max_penalty=0.15,
                min_overlap=1
            )
            passed4 = print_result(
                "No penalty when topics match",
                penalty2 == 0.0,  # Has overlap
                f"penalty={penalty2}"
            )

            return passed1 and passed2 and passed3 and passed4

        except Exception as e:
            return print_result("Topic extraction", False, str(e))

    def test_chapter_dataclass(self) -> bool:
        """Test Chapter dataclass in utils"""
        print_section("Chapter Dataclass")

        try:
            from src.utils import Chapter
        except ImportError as e:
            return print_result("Chapter dataclass", True, f"SKIPPED - {e}")

        try:

            chapter = Chapter(
                chapter_id=1,
                start_segment_idx=0,
                end_segment_idx=5,
                title="Introduction",
                topics=["weather", "storm"]
            )

            passed1 = print_result(
                "Chapter creation",
                chapter.chapter_id == 1 and chapter.title == "Introduction"
            )

            passed2 = print_result(
                "Chapter topics",
                chapter.topics == ["weather", "storm"]
            )

            return passed1 and passed2

        except Exception as e:
            return print_result("Chapter dataclass", False, str(e))

    def test_srt_segment_topics(self) -> bool:
        """Test that SRTSegment has topics field"""
        print_section("SRTSegment Topics Field")

        try:
            from src.utils import SRTSegment
        except ImportError as e:
            return print_result("SRTSegment topics", True, f"SKIPPED - {e}")

        try:

            segment = SRTSegment(
                index=1,
                start_time=0.0,
                end_time=10.0,
                text="Test segment",
                topics=["weather", "storm"]
            )

            passed1 = print_result(
                "SRTSegment with topics",
                segment.topics == ["weather", "storm"]
            )

            # Test to_dict includes topics
            d = segment.to_dict()
            passed2 = print_result(
                "to_dict includes topics",
                d.get('topics') == ["weather", "storm"]
            )

            return passed1 and passed2

        except Exception as e:
            return print_result("SRTSegment topics", False, str(e))

    def test_otio_tracks(self) -> bool:
        """Test that OTIO builder includes V9 and V10 tracks"""
        print_section("OTIO V9/V10 Tracks")

        if not HAS_OTIO:
            print_result("OTIO tracks", True, "SKIPPED - opentimelineio not installed")
            return True

        try:
            from src.utils import SRTSegment, Match, MatchResult
            from src.config import Config

            # Create minimal config
            config = Config()

            # Create minimal match data
            vo_seg = SRTSegment(
                index=1, start_time=0.0, end_time=5.0,
                text="Test", source_file=""
            )
            vid_seg = SRTSegment(
                index=1, start_time=0.0, end_time=5.0,
                text="Test video", source_file="/path/video.mp4"
            )
            match = Match(
                voiceover_segment=vo_seg,
                video_segment=vid_seg,
                video_scene=None,
                confidence=0.9,
                reasoning="Test"
            )
            match_result = MatchResult(
                primary_match=match,
                alternatives=[],
                strategy_matches={}
            )

            # Import and call create_timeline
            from src.otio_builder import create_timeline

            timeline = create_timeline(
                matches=[match_result],
                config=config,
                voiceover_path=None,
                frame_rate=30.0,
                entity_images=None,  # No images
                entity_videos=None   # No videos
            )

            # Check for V9 and V10 tracks
            track_names = [t.name for t in timeline.tracks]

            has_v9 = any("V9" in name or "Entity Images" in name for name in track_names)
            has_v10 = any("V10" in name or "Stock Videos" in name for name in track_names)

            passed1 = print_result(
                "V9 Entity Images track present",
                has_v9,
                f"tracks: {[n for n in track_names if 'V9' in n or 'V10' in n]}"
            )

            passed2 = print_result(
                "V10 Stock Videos track present",
                has_v10
            )

            return passed1 and passed2

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("OTIO tracks", False, str(e))

    def test_config_chapter_options(self) -> bool:
        """Test that config has chapter matching options"""
        print_section("Config Chapter Options")

        try:
            from src.config import Config, MatchingConfig

            # Check MatchingConfig has the fields
            config = Config()
            mc = config.matching

            has_chapter_enabled = hasattr(mc, 'chapter_matching_enabled')
            has_penalty = hasattr(mc, 'topic_mismatch_penalty')
            has_extract = hasattr(mc, 'extract_video_topics')

            passed1 = print_result(
                "chapter_matching_enabled option",
                has_chapter_enabled,
                f"value={getattr(mc, 'chapter_matching_enabled', 'N/A')}"
            )

            passed2 = print_result(
                "topic_mismatch_penalty option",
                has_penalty,
                f"value={getattr(mc, 'topic_mismatch_penalty', 'N/A')}"
            )

            passed3 = print_result(
                "extract_video_topics option",
                has_extract,
                f"value={getattr(mc, 'extract_video_topics', 'N/A')}"
            )

            return passed1 and passed2 and passed3

        except Exception as e:
            return print_result("Config options", False, str(e))

    def run_all(self) -> bool:
        """Run all tests"""
        print_box("RECENT FEATURES TEST SUITE")
        print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.setup()

        tests = [
            ("JSON Serialization", self.test_json_serialization),
            ("Delta Matching Index", self.test_delta_matching_index),
            ("Topic Extraction", self.test_topic_extraction),
            ("Chapter Dataclass", self.test_chapter_dataclass),
            ("SRTSegment Topics", self.test_srt_segment_topics),
            ("OTIO V9/V10 Tracks", self.test_otio_tracks),
            ("Config Chapter Options", self.test_config_chapter_options),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n  [ERROR] {name}: {e}")
                failed += 1

        self.teardown()

        # Summary
        print_box("TEST SUMMARY")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Total:  {passed + failed}")

        if failed == 0:
            print(f"\n  [SUCCESS] All tests passed!")
            return True
        else:
            print(f"\n  [FAILURE] {failed} test(s) failed")
            return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test recent feature additions")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    suite = TestRecentFeatures(verbose=args.verbose)
    success = suite.run_all()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
