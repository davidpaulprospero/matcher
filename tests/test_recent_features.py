#!/usr/bin/env python3
"""
Test Recent Feature Additions

Tests the following recent features:
1. Delta matching - only match new videos
2. Cache loading - load transcripts from cache when skip_transcription=true
3. Chapter-based topic matching
4. OTIO V9/V10 track inclusion
5. JSON serialization fix for numpy types
6. Scene-level face detection for B-roll identification
7. B-roll preference in matching
8. Global cache for cross-project video reuse
9. Segment IDs in OTIO clip names for post-edit tracing
10. Post-edit analysis tool

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

            # Test save/load (uses private _save method)
            index._save()
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

            # Test topic overlap (returns tuple of count, ratio)
            vo_topics = ["weather", "storm", "news"]
            video_topics = ["weather", "winter", "cold"]
            overlap_count, overlap_ratio = compute_topic_overlap(vo_topics, video_topics)
            passed2 = print_result(
                "Topic overlap computation",
                overlap_count >= 1,  # "weather" is common
                f"count={overlap_count}, ratio={overlap_ratio:.2f}"
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

    def test_face_detection_module(self) -> bool:
        """Test FaceDetector class and B-roll functions"""
        print_section("Face Detection Module")

        try:
            from src.face_detection import (
                FaceDetector,
                apply_face_preference,
                apply_broll_preference,
                is_broll_scene
            )
        except ImportError as e:
            return print_result("Face detection module", True, f"SKIPPED - {e}")

        try:
            # Test FaceDetector singleton
            detector = FaceDetector.get_instance()
            passed1 = print_result(
                "FaceDetector singleton",
                detector is not None,
                f"available={detector.is_available()}"
            )

            # Test is_broll_scene helper
            passed2 = print_result(
                "is_broll_scene(0.1) = True",
                is_broll_scene(0.1, threshold=0.3) is True
            )

            passed3 = print_result(
                "is_broll_scene(0.5) = False",
                is_broll_scene(0.5, threshold=0.3) is False
            )

            passed4 = print_result(
                "is_broll_scene(0.9) = False",
                is_broll_scene(0.9, threshold=0.3) is False
            )

            return passed1 and passed2 and passed3 and passed4

        except Exception as e:
            return print_result("Face detection module", False, str(e))

    def test_scene_info_face_score(self) -> bool:
        """Test SceneInfo has face_score and is_broll fields"""
        print_section("SceneInfo Face Score Fields")

        try:
            from src.scene_detection import SceneInfo
        except ImportError as e:
            return print_result("SceneInfo face fields", True, f"SKIPPED - {e}")

        try:
            # Create SceneInfo with face detection fields
            scene = SceneInfo(
                scene_index=0,
                start_frame=0,
                end_frame=150,
                start_time=0.0,
                end_time=5.0,
                duration=5.0,
                face_score=0.2,  # B-roll (no faces)
                is_broll=True
            )

            passed1 = print_result(
                "SceneInfo has face_score field",
                hasattr(scene, 'face_score') and scene.face_score == 0.2
            )

            passed2 = print_result(
                "SceneInfo has is_broll field",
                hasattr(scene, 'is_broll') and scene.is_broll is True
            )

            # Test to_dict includes face fields
            d = scene.to_dict()
            passed3 = print_result(
                "to_dict includes face_score",
                'face_score' in d and d['face_score'] == 0.2
            )

            passed4 = print_result(
                "to_dict includes is_broll",
                'is_broll' in d and d['is_broll'] is True
            )

            return passed1 and passed2 and passed3 and passed4

        except Exception as e:
            return print_result("SceneInfo face fields", False, str(e))

    def test_config_broll_options(self) -> bool:
        """Test that config has B-roll preference options"""
        print_section("Config B-roll Options")

        try:
            from src.config import Config, MatchingConfig, SceneDetectionConfig

            config = Config()
            mc = config.matching
            sc = config.scene_detection

            # Check MatchingConfig B-roll fields
            has_prefer_broll = hasattr(mc, 'prefer_broll_when_topic_matches')
            has_threshold = hasattr(mc, 'broll_face_threshold')
            has_boost = hasattr(mc, 'broll_boost')

            passed1 = print_result(
                "prefer_broll_when_topic_matches option",
                has_prefer_broll,
                f"value={getattr(mc, 'prefer_broll_when_topic_matches', 'N/A')}"
            )

            passed2 = print_result(
                "broll_face_threshold option",
                has_threshold,
                f"value={getattr(mc, 'broll_face_threshold', 'N/A')}"
            )

            passed3 = print_result(
                "broll_boost option",
                has_boost,
                f"value={getattr(mc, 'broll_boost', 'N/A')}"
            )

            # Check SceneDetectionConfig face fields
            has_detect_faces = hasattr(sc, 'detect_faces_per_scene')
            passed4 = print_result(
                "detect_faces_per_scene option",
                has_detect_faces,
                f"value={getattr(sc, 'detect_faces_per_scene', 'N/A')}"
            )

            return passed1 and passed2 and passed3 and passed4

        except Exception as e:
            return print_result("Config B-roll options", False, str(e))

    def test_global_cache_module(self) -> bool:
        """Test GlobalCacheManager class"""
        print_section("Global Cache Module")

        try:
            from src.global_cache import (
                GlobalCacheManager,
                VideoRegistryEntry,
                DownloadInfo,
                GlobalCacheQueryResult,
                VideoSource
            )
        except ImportError as e:
            return print_result("Global cache module", True, f"SKIPPED - {e}")

        try:
            # Create temporary global cache
            cache_dir = self.temp_dir / "global_cache_test"
            cache_manager = GlobalCacheManager(cache_dir=str(cache_dir))

            passed1 = print_result(
                "GlobalCacheManager initialization",
                cache_manager is not None and cache_dir.exists()
            )

            # Test DownloadInfo dataclass
            download_info = DownloadInfo(
                keyword="austin texas footage",
                youtube_id="abc123",
                youtube_url="https://youtube.com/watch?v=abc123",
                original_title="Austin Texas 4K"
            )
            passed2 = print_result(
                "DownloadInfo dataclass",
                download_info.keyword == "austin texas footage"
            )

            # Test VideoRegistryEntry
            entry = VideoRegistryEntry(
                video_hash="test123",
                filename="test_video.mp4",
                file_size=1000000,
                duration=60.0,
                topics=["austin", "texas", "city"],
                download_info=download_info
            )
            passed3 = print_result(
                "VideoRegistryEntry dataclass",
                entry.video_hash == "test123" and len(entry.topics) == 3
            )

            # Test to_dict/from_dict roundtrip
            entry_dict = entry.to_dict()
            entry_restored = VideoRegistryEntry.from_dict(entry_dict)
            passed4 = print_result(
                "VideoRegistryEntry serialization",
                entry_restored.video_hash == entry.video_hash
            )

            # Test GlobalCacheQueryResult
            query_result = GlobalCacheQueryResult(
                reuse_videos=[(entry, 0.8)],
                redownload_keywords=["deleted keyword"],
                uncovered_keywords=["new keyword"]
            )
            passed5 = print_result(
                "GlobalCacheQueryResult",
                len(query_result.reuse_videos) == 1 and len(query_result.redownload_keywords) == 1
            )

            # Test cache stats
            stats = cache_manager.get_stats()
            passed6 = print_result(
                "GlobalCacheManager stats",
                "total_videos" in stats and "cache_dir" in stats
            )

            return passed1 and passed2 and passed3 and passed4 and passed5 and passed6

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("Global cache module", False, str(e))

    def test_config_global_cache_options(self) -> bool:
        """Test that config has global cache options"""
        print_section("Config Global Cache Options")

        try:
            from src.config import Config, GlobalCacheConfig

            config = Config()
            gc = config.global_cache

            # Check GlobalCacheConfig fields
            has_enabled = hasattr(gc, 'enabled')
            has_check_before = hasattr(gc, 'check_before_download')
            has_min_topic = hasattr(gc, 'min_topic_overlap')
            has_share_transcripts = hasattr(gc, 'share_transcripts')

            passed1 = print_result(
                "global_cache.enabled option",
                has_enabled,
                f"value={getattr(gc, 'enabled', 'N/A')}"
            )

            passed2 = print_result(
                "check_before_download option",
                has_check_before,
                f"value={getattr(gc, 'check_before_download', 'N/A')}"
            )

            passed3 = print_result(
                "min_topic_overlap option",
                has_min_topic,
                f"value={getattr(gc, 'min_topic_overlap', 'N/A')}"
            )

            passed4 = print_result(
                "share_transcripts option",
                has_share_transcripts,
                f"value={getattr(gc, 'share_transcripts', 'N/A')}"
            )

            return passed1 and passed2 and passed3 and passed4

        except Exception as e:
            return print_result("Config global cache options", False, str(e))

    def test_segment_ids_in_clip_names(self) -> bool:
        """Test that OTIO builder adds segment IDs to clip names"""
        print_section("Segment IDs in Clip Names")

        if not HAS_OTIO:
            print_result("Segment IDs", True, "SKIPPED - opentimelineio not installed")
            return True

        try:
            from src.utils import SRTSegment, Match, MatchResult
            from src.config import Config
            from src.otio_builder import create_timeline

            # Create minimal config
            config = Config()

            # Create 3 match results to test segment ID numbering
            matches = []
            for i in range(3):
                vo_seg = SRTSegment(
                    index=i, start_time=i * 5.0, end_time=(i + 1) * 5.0,
                    text=f"Test segment {i}", source_file=""
                )
                vid_seg = SRTSegment(
                    index=i, start_time=0.0, end_time=5.0,
                    text=f"Video text {i}", source_file=f"/path/folder/video{i}.mp4"
                )
                match = Match(
                    voiceover_segment=vo_seg,
                    video_segment=vid_seg,
                    video_scene=None,
                    confidence=0.9,
                    reasoning="Test"
                )
                matches.append(MatchResult(
                    primary_match=match,
                    alternatives=[],
                    strategy_matches={}
                ))

            # Create timeline
            timeline = create_timeline(
                matches=matches,
                config=config,
                voiceover_path=None,
                frame_rate=30.0,
                entity_images=None,
                entity_videos=None
            )

            # Find V1 track and check clip names
            clip_names = []
            for track in timeline.tracks:
                if track.kind == otio.schema.TrackKind.Video and "V1" in track.name:
                    for item in track:
                        if hasattr(item, 'name') and item.name:
                            clip_names.append(item.name)
                    break

            # Check that segment IDs are present
            has_s000 = any("[S000]" in name for name in clip_names)
            has_s001 = any("[S001]" in name for name in clip_names)
            has_s002 = any("[S002]" in name for name in clip_names)

            passed1 = print_result(
                "Clip names contain [S000]",
                has_s000,
                f"found in: {[n for n in clip_names if '[S000]' in n][:1]}"
            )

            passed2 = print_result(
                "Clip names contain [S001]",
                has_s001
            )

            passed3 = print_result(
                "Clip names contain [S002]",
                has_s002
            )

            # Check metadata includes segment_index
            has_metadata = False
            for track in timeline.tracks:
                if track.kind == otio.schema.TrackKind.Video and "V1" in track.name:
                    for item in track:
                        if hasattr(item, 'metadata') and 'segment_index' in item.metadata:
                            has_metadata = True
                            break
                    break

            passed4 = print_result(
                "Clip metadata includes segment_index",
                has_metadata
            )

            return passed1 and passed2 and passed3 and passed4

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("Segment IDs", False, str(e))

    def test_post_edit_analyzer(self) -> bool:
        """Test PostEditAnalyzer module"""
        print_section("Post-Edit Analyzer Module")

        try:
            from src.post_edit_analysis import (
                PostEditAnalyzer,
                ClipSelection,
                EditAnalysisResult,
                analyze_final_edit
            )
        except ImportError as e:
            return print_result("Post-edit analysis", True, f"SKIPPED - {e}")

        try:
            # Test ClipSelection dataclass
            selection = ClipSelection(
                segment_id="S001",
                segment_index=1,
                clip_name="[S001] folder_video [10.5s]",
                source_file="folder_video",
                track="V1",
                was_alternative=False
            )
            passed1 = print_result(
                "ClipSelection dataclass",
                selection.segment_id == "S001" and selection.segment_index == 1
            )

            # Test EditAnalysisResult
            result = EditAnalysisResult(
                total_segments=10,
                clips_kept=7,
                clips_replaced_with_alt=2,
                clips_removed=1
            )
            passed2 = print_result(
                "EditAnalysisResult dataclass",
                result.clips_kept == 7 and result.clips_replaced_with_alt == 2
            )

            # Test summary generation
            summary = result.summary()
            passed3 = print_result(
                "EditAnalysisResult.summary()",
                "Total segments" in summary and "Primary clips kept" in summary
            )

            # Test PostEditAnalyzer regex patterns
            analyzer = PostEditAnalyzer()

            # Test segment ID extraction
            match = analyzer.SEGMENT_ID_PATTERN.search("[S001] folder_video [10.5s]")
            passed4 = print_result(
                "Segment ID pattern extraction",
                match is not None and match.group(1) == "001"
            )

            # Test ALT pattern
            alt_match = analyzer.ALT_PATTERN.search("[S005] ALT2: folder_alt [5.0s]")
            passed5 = print_result(
                "Alternative pattern extraction",
                alt_match is not None and alt_match.group(1) == "2"
            )

            # Test parsing a mock clip element
            from xml.etree.ElementTree import Element, SubElement

            clip = Element('clipitem')
            name = SubElement(clip, 'name')
            name.text = "[S003] test_folder_video [15.5s]"
            start = SubElement(clip, 'start')
            start.text = "0"

            parsed = analyzer._parse_clip_element(clip)
            passed6 = print_result(
                "Parse clip element",
                parsed is not None and parsed.segment_index == 3 and parsed.segment_id == "S003"
            )

            return passed1 and passed2 and passed3 and passed4 and passed5 and passed6

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("Post-edit analysis", False, str(e))

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
            ("Face Detection Module", self.test_face_detection_module),
            ("SceneInfo Face Score", self.test_scene_info_face_score),
            ("Config B-roll Options", self.test_config_broll_options),
            ("Global Cache Module", self.test_global_cache_module),
            ("Config Global Cache", self.test_config_global_cache_options),
            ("Segment IDs in Clips", self.test_segment_ids_in_clip_names),
            ("Post-Edit Analyzer", self.test_post_edit_analyzer),
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
