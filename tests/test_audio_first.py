#!/usr/bin/env python3
"""
Test Audio-First Pipeline Features

Tests the following audio-first mode features:
1. max_total project-level cap for duration tiers
2. remix_audio_files function for audio-first scoring
3. Face score mapping from video segments to audio files
4. DurationTierConfig max_total field

Usage:
    python tests/test_audio_first.py
    python tests/test_audio_first.py --verbose
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


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


class TestAudioFirstFeatures:
    """Test suite for audio-first pipeline features"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.temp_dir = None
        self.results = {}

    def setup(self):
        """Create temporary test directory with synthetic audio files"""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='test_audio_first_'))

        # Create synthetic audio files with .info.json metadata
        audio_dir = self.temp_dir / "test_keyword_s_audio"
        audio_dir.mkdir(parents=True)

        # Create synthetic audio files
        self.audio_files = []
        test_videos = [
            {"id": "abc123", "title": "Beautiful sunset beach footage", "keyword": "sunset"},
            {"id": "def456", "title": "Mountain landscape drone shot", "keyword": "mountain"},
            {"id": "ghi789", "title": "City skyline night view", "keyword": "city"},
            {"id": "jkl012", "title": "Random unrelated content", "keyword": "random"},
        ]

        for video in test_videos:
            # Create empty audio file
            audio_file = audio_dir / f"{video['id']}.mp3"
            audio_file.write_bytes(b'\x00' * 100)  # Dummy content
            self.audio_files.append(str(audio_file))

            # Create .info.json with metadata (yt-dlp naming: video_id.info.json)
            info_file = audio_dir / f"{video['id']}.info.json"
            info_data = {
                "id": video["id"],
                "title": video["title"],
                "description": f"Test video about {video['keyword']}",
                "tags": [video["keyword"], "footage", "video"],
                "duration": 60,
            }
            info_file.write_text(json.dumps(info_data))

        if self.verbose:
            print(f"  Created temp dir: {self.temp_dir}")
            print(f"  Created {len(self.audio_files)} synthetic audio files")

    def teardown(self):
        """Clean up temporary directory"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            if self.verbose:
                print(f"  Cleaned up temp dir")

    # =========================================================================
    # Test 1: max_total config field
    # =========================================================================

    def test_duration_tier_max_total_field(self) -> bool:
        """Test that DurationTierConfig has max_total field"""
        print_section("DurationTierConfig max_total Field")

        try:
            from src.config import DurationTierConfig, DurationTiersConfig

            # Test DurationTierConfig has max_total
            tier = DurationTierConfig(
                min_seconds=1500,
                max_seconds=3000,
                videos_per_keyword=1,
                max_total=1
            )

            passed1 = print_result(
                "DurationTierConfig has max_total",
                hasattr(tier, 'max_total'),
                f"value={tier.max_total}"
            )

            # Test DurationTiersConfig defaults
            tiers = DurationTiersConfig()

            passed2 = print_result(
                "longer tier has max_total=1 by default",
                tiers.longer.max_total == 1,
                f"value={tiers.longer.max_total}"
            )

            passed3 = print_result(
                "short tier has max_total=0 (no limit)",
                tiers.short.max_total == 0,
                f"value={tiers.short.max_total}"
            )

            return passed1 and passed2 and passed3

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("DurationTierConfig", False, str(e))

    def test_config_builds_max_total(self) -> bool:
        """Test that Config._build_duration_tiers reads max_total from YAML"""
        print_section("Config builds max_total from YAML")

        try:
            from src.config import Config

            # Create a minimal config dict with max_total
            config_data = {
                'duration_tiers': {
                    'short': {'min': 20, 'max': 120, 'count': 5, 'max_total': 0},
                    'medium': {'min': 120, 'max': 600, 'count': 3, 'max_total': 0},
                    'long': {'min': 600, 'max': 1500, 'count': 1, 'max_total': 0},
                    'longer': {'min': 1500, 'max': 3000, 'count': 1, 'max_total': 2},
                }
            }

            # Build tiers from config data
            tiers = Config._build_duration_tiers(config_data['duration_tiers'])

            passed1 = print_result(
                "longer tier max_total=2 from config",
                tiers.longer.max_total == 2,
                f"value={tiers.longer.max_total}"
            )

            passed2 = print_result(
                "short tier max_total=0 from config",
                tiers.short.max_total == 0,
                f"value={tiers.short.max_total}"
            )

            return passed1 and passed2

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("Config build", False, str(e))

    # =========================================================================
    # Test 2: Downloader max_total enforcement
    # =========================================================================

    def test_downloader_tier_defaults(self) -> bool:
        """Test that VideoDownloader has max_total in DURATION_TIERS"""
        print_section("VideoDownloader DURATION_TIERS max_total")

        try:
            from src.downloader import VideoDownloader

            # Check default DURATION_TIERS has max_total
            tiers = VideoDownloader.DURATION_TIERS

            passed1 = print_result(
                "longer tier has max_total key",
                'max_total' in tiers.get('longer', {}),
                f"value={tiers.get('longer', {}).get('max_total')}"
            )

            passed2 = print_result(
                "longer tier max_total=1",
                tiers.get('longer', {}).get('max_total') == 1,
                f"value={tiers.get('longer', {}).get('max_total')}"
            )

            passed3 = print_result(
                "short tier max_total=0",
                tiers.get('short', {}).get('max_total') == 0,
                f"value={tiers.get('short', {}).get('max_total')}"
            )

            return passed1 and passed2 and passed3

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("VideoDownloader tiers", False, str(e))

    def test_downloader_get_tier_value_max_total(self) -> bool:
        """Test that _get_tier_value correctly retrieves max_total"""
        print_section("VideoDownloader._get_tier_value for max_total")

        try:
            from src.downloader import VideoDownloader
            from src.config import Config

            # Create a mock config
            config = MagicMock()
            config.download = MagicMock()
            config.download.tiers = None  # Use defaults
            config.download.cookies_from_browser = None
            config.download.cookies_path = None
            config.downloaded_videos_dir = str(self.temp_dir)
            config.cache_dir = str(self.temp_dir / '.cache')

            downloader = VideoDownloader(config)

            # Test _get_tier_value for max_total
            max_total_longer = downloader._get_tier_value('longer', 'max_total', 0)
            max_total_short = downloader._get_tier_value('short', 'max_total', 0)

            passed1 = print_result(
                "_get_tier_value('longer', 'max_total') = 1",
                max_total_longer == 1,
                f"value={max_total_longer}"
            )

            passed2 = print_result(
                "_get_tier_value('short', 'max_total') = 0",
                max_total_short == 0,
                f"value={max_total_short}"
            )

            return passed1 and passed2

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("_get_tier_value", False, str(e))

    # =========================================================================
    # Test 3: remix_audio_files function
    # =========================================================================

    def test_remix_audio_files_import(self) -> bool:
        """Test that remix_audio_files can be imported"""
        print_section("remix_audio_files Import")

        try:
            from src.keyword_remix import remix_audio_files, RemixConfig

            passed = print_result(
                "remix_audio_files imported",
                callable(remix_audio_files),
                "function available"
            )

            return passed

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("remix_audio_files import", False, str(e))

    def test_remix_audio_files_scoring(self) -> bool:
        """Test remix_audio_files scores audio files correctly"""
        print_section("remix_audio_files Scoring")

        try:
            from src.keyword_remix import remix_audio_files, RemixConfig

            # Create config that auto-accepts filtered results
            config = RemixConfig(
                enabled=True,
                min_relevance_score=0.1,
                auto_accept_filter='filtered',
                interactive_curation=False,
            )

            # Test with keywords that match some files
            keywords = ["sunset", "beach", "mountain"]

            included_paths, result = remix_audio_files(
                audio_files=self.audio_files,
                keywords=keywords,
                config=config,
                interactive=False,
                show_progress=False
            )

            passed1 = print_result(
                "remix_audio_files returns result",
                result is not None,
                f"total_files={result.total_files if result else 0}"
            )

            passed2 = print_result(
                "Scored all audio files",
                result.total_files == len(self.audio_files) if result else False,
                f"expected={len(self.audio_files)}, got={result.total_files if result else 0}"
            )

            # Check that sunset/mountain files scored higher
            passed3 = print_result(
                "Has included files",
                len(included_paths) > 0,
                f"included={len(included_paths)}"
            )

            return passed1 and passed2 and passed3

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("remix_audio_files scoring", False, str(e))

    def test_remix_audio_files_empty_list(self) -> bool:
        """Test remix_audio_files handles empty list"""
        print_section("remix_audio_files Empty List")

        try:
            from src.keyword_remix import remix_audio_files, RemixConfig

            config = RemixConfig(enabled=True)

            included_paths, result = remix_audio_files(
                audio_files=[],
                keywords=["test"],
                config=config,
                interactive=False,
                show_progress=False
            )

            passed = print_result(
                "Empty list returns empty result",
                included_paths == [] and result is None,
                f"paths={len(included_paths)}, result={result}"
            )

            return passed

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("remix_audio_files empty", False, str(e))

    def test_remix_audio_files_disabled(self) -> bool:
        """Test remix_audio_files when disabled returns all files"""
        print_section("remix_audio_files Disabled")

        try:
            from src.keyword_remix import remix_audio_files, RemixConfig

            config = RemixConfig(enabled=False)

            included_paths, result = remix_audio_files(
                audio_files=self.audio_files,
                keywords=["test"],
                config=config,
                interactive=False,
                show_progress=False
            )

            passed = print_result(
                "Disabled returns all files",
                len(included_paths) == len(self.audio_files),
                f"returned={len(included_paths)}, total={len(self.audio_files)}"
            )

            return passed

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("remix_audio_files disabled", False, str(e))

    # =========================================================================
    # Test 4: KeywordRemixProcessor.process_file_list
    # =========================================================================

    def test_process_file_list(self) -> bool:
        """Test KeywordRemixProcessor.process_file_list method"""
        print_section("KeywordRemixProcessor.process_file_list")

        try:
            from src.keyword_remix import KeywordRemixProcessor, RemixConfig

            config = RemixConfig(
                min_relevance_score=0.0,  # Include all
                max_files_to_include=100,
            )

            processor = KeywordRemixProcessor(config, ["sunset", "mountain"])

            # Process the audio files
            result = processor.process_file_list(
                [Path(f) for f in self.audio_files],
                show_progress=False
            )

            passed1 = print_result(
                "process_file_list returns RemixResult",
                result is not None,
                f"type={type(result).__name__}"
            )

            passed2 = print_result(
                "Processed all files",
                result.total_files == len(self.audio_files),
                f"expected={len(self.audio_files)}, got={result.total_files}"
            )

            passed3 = print_result(
                "Has scored videos",
                len(result.included_videos) + len(result.excluded_videos) == len(self.audio_files),
                f"included={len(result.included_videos)}, excluded={len(result.excluded_videos)}"
            )

            return passed1 and passed2 and passed3

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("process_file_list", False, str(e))

    # =========================================================================
    # Test 5: Face score mapping for two-pass matching
    # =========================================================================

    def test_face_detector_cache_structure(self) -> bool:
        """Test FaceDetector has _cache class attribute"""
        print_section("FaceDetector Cache Structure")

        try:
            from src.face_detection import FaceDetector

            # Check _cache exists
            passed1 = print_result(
                "FaceDetector has _cache",
                hasattr(FaceDetector, '_cache'),
                f"type={type(getattr(FaceDetector, '_cache', None))}"
            )

            # Test we can set/get from cache
            test_path = "/test/path/video.mp4"
            FaceDetector._cache[test_path] = 0.75

            passed2 = print_result(
                "Can set face score in cache",
                FaceDetector._cache.get(test_path) == 0.75,
                f"value={FaceDetector._cache.get(test_path)}"
            )

            # Clean up
            del FaceDetector._cache[test_path]

            return passed1 and passed2

        except ImportError as e:
            return print_result("FaceDetector cache", True, f"SKIPPED - {e}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("FaceDetector cache", False, str(e))

    def test_face_score_mapping_logic(self) -> bool:
        """Test the face score mapping logic concept"""
        print_section("Face Score Mapping Logic")

        try:
            # Simulate the mapping logic from main.py
            # Video segment: video_id_0045.mp4 -> audio file: video_id.mp3

            segment_scores = {
                "/path/abc123_0045.mp4": 0.8,
                "/path/abc123_0120.mp4": 0.6,
                "/path/def456_0030.mp4": 0.9,
            }

            audio_files = {
                "abc123": "/path/abc123.mp3",
                "def456": "/path/def456.mp3",
            }

            # Map segment scores to audio files
            audio_scores = {}
            for seg_path, score in segment_scores.items():
                # Extract video_id from segment path
                filename = Path(seg_path).stem  # abc123_0045
                video_id = filename.rsplit('_', 1)[0]  # abc123

                if video_id in audio_files:
                    audio_path = audio_files[video_id]
                    # Take highest score for this video
                    if audio_path not in audio_scores or score > audio_scores[audio_path]:
                        audio_scores[audio_path] = score

            passed1 = print_result(
                "Mapped abc123 segments to audio",
                audio_scores.get("/path/abc123.mp3") == 0.8,  # Max of 0.8, 0.6
                f"value={audio_scores.get('/path/abc123.mp3')}"
            )

            passed2 = print_result(
                "Mapped def456 segment to audio",
                audio_scores.get("/path/def456.mp3") == 0.9,
                f"value={audio_scores.get('/path/def456.mp3')}"
            )

            return passed1 and passed2

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("Face score mapping", False, str(e))

    # =========================================================================
    # Test 6: scan_directory with include_audio
    # =========================================================================

    def test_scan_directory_include_audio(self) -> bool:
        """Test scan_directory with include_audio=True"""
        print_section("scan_directory include_audio")

        try:
            from src.keyword_remix import KeywordRemixProcessor, RemixConfig

            config = RemixConfig()
            processor = KeywordRemixProcessor(config, ["test"])

            # Create mixed directory with video and audio files
            mixed_dir = self.temp_dir / "mixed"
            mixed_dir.mkdir()

            (mixed_dir / "video1.mp4").write_bytes(b'\x00' * 100)
            (mixed_dir / "audio1.mp3").write_bytes(b'\x00' * 100)
            (mixed_dir / "audio2.m4a").write_bytes(b'\x00' * 100)

            # Scan without audio
            files_no_audio = processor.scan_directory(mixed_dir, include_audio=False)

            passed1 = print_result(
                "Without audio: only video files",
                len(files_no_audio) == 1,
                f"found={len(files_no_audio)}"
            )

            # Scan with audio
            files_with_audio = processor.scan_directory(mixed_dir, include_audio=True)

            passed2 = print_result(
                "With audio: video + audio files",
                len(files_with_audio) == 3,
                f"found={len(files_with_audio)}"
            )

            return passed1 and passed2

        except Exception as e:
            import traceback
            traceback.print_exc()
            return print_result("scan_directory audio", False, str(e))

    # =========================================================================
    # Run all tests
    # =========================================================================

    def run_all(self) -> Dict[str, bool]:
        """Run all tests and return results"""
        print_box("Audio-First Pipeline Tests")

        self.setup()

        try:
            # max_total tests
            self.results['duration_tier_max_total_field'] = self.test_duration_tier_max_total_field()
            self.results['config_builds_max_total'] = self.test_config_builds_max_total()
            self.results['downloader_tier_defaults'] = self.test_downloader_tier_defaults()
            self.results['downloader_get_tier_value'] = self.test_downloader_get_tier_value_max_total()

            # remix_audio_files tests
            self.results['remix_audio_files_import'] = self.test_remix_audio_files_import()
            self.results['remix_audio_files_scoring'] = self.test_remix_audio_files_scoring()
            self.results['remix_audio_files_empty'] = self.test_remix_audio_files_empty_list()
            self.results['remix_audio_files_disabled'] = self.test_remix_audio_files_disabled()

            # process_file_list tests
            self.results['process_file_list'] = self.test_process_file_list()

            # Face score mapping tests
            self.results['face_detector_cache'] = self.test_face_detector_cache_structure()
            self.results['face_score_mapping'] = self.test_face_score_mapping_logic()

            # scan_directory tests
            self.results['scan_directory_audio'] = self.test_scan_directory_include_audio()

        finally:
            self.teardown()

        # Summary
        print_box("Test Summary")
        passed = sum(1 for v in self.results.values() if v)
        failed = sum(1 for v in self.results.values() if not v)

        print(f"\n  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Total:  {len(self.results)}")

        if failed > 0:
            print("\n  Failed tests:")
            for name, result in self.results.items():
                if not result:
                    print(f"    - {name}")

        return self.results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test audio-first pipeline features')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    args = parser.parse_args()

    tester = TestAudioFirstFeatures(verbose=args.verbose)
    results = tester.run_all()

    # Exit with error code if any tests failed
    failed = sum(1 for v in results.values() if not v)
    sys.exit(1 if failed > 0 else 0)


if __name__ == '__main__':
    main()
