#!/usr/bin/env python3
"""
Feature Test Suite v1.0

Tests recent features with actual file processing:
1. Face Detection (MediaPipe/OpenCV)
2. Logger Stats Tracking
3. Config Loading (face_preference, root_dir)

Usage:
    python tests/test_features.py
    python tests/test_features.py --keep-videos  # Don't delete test videos after
    python tests/test_features.py --skip-download  # Use existing test videos
"""

import os
import sys
import json
import time
import shutil
import argparse
import tempfile
import pytest
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# TEST RESULT TRACKING
# =============================================================================

@dataclass
class FeatureTestResult:
    name: str
    passed: bool
    duration: float
    message: str = ""
    expected: str = ""
    actual: str = ""


class FeatureTestRunner:
    """Simple test runner with timing and reporting"""

    def __init__(self):
        self.results: List[FeatureTestResult] = []
        self.start_time = time.time()

    def run_test(self, name: str, test_func, *args, **kwargs) -> FeatureTestResult:
        """Run a single test and record result"""
        print(f"\n  ├─ Testing: {name}...", end=" ", flush=True)
        start = time.time()
        
        try:
            result = test_func(*args, **kwargs)
            duration = time.time() - start
            
            if isinstance(result, tuple):
                passed, message = result
            else:
                passed = bool(result)
                message = "OK" if passed else "Failed"
            
            test_result = FeatureTestResult(
                name=name,
                passed=passed,
                duration=duration,
                message=message
            )
            
            if passed:
                print(f"✓ ({duration:.2f}s)")
            else:
                print(f"✗ ({duration:.2f}s)")
                print(f"      └─ {message}")
                
        except Exception as e:
            duration = time.time() - start
            test_result = FeatureTestResult(
                name=name,
                passed=False,
                duration=duration,
                message=f"Exception: {str(e)}"
            )
            print(f"✗ ({duration:.2f}s)")
            print(f"      └─ {test_result.message}")
        
        self.results.append(test_result)
        return test_result
    
    def print_summary(self):
        """Print test summary"""
        total_duration = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        print(f"\n{'=' * 60}")
        print(f"  TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Total tests: {len(self.results)}")
        print(f"  Passed: {passed}")
        print(f"  Failed: {failed}")
        print(f"  Duration: {total_duration:.1f}s")
        
        if failed > 0:
            print(f"\n  Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"    • {r.name}: {r.message}")
        
        print(f"{'=' * 60}")
        return failed == 0


# =============================================================================
# VIDEO DOWNLOAD FOR TESTING
# =============================================================================

def download_test_videos(output_dir: Path, cookies_path: str = None) -> dict:
    """
    Download short test videos:
    - One with faces (news/interview)
    - One without faces (nature/scenery)
    
    Retries with different search queries until success.
    
    Returns dict with paths to downloaded videos.
    """
    import subprocess
    
    # Multiple search queries to try for each type
    test_videos = {
        'with_faces': {
            'searches': [
                'news anchor speaking',
                'interview person talking',
                'reporter live news',
                'person speaking camera',
                'youtuber talking vlog',
                'ted talk speaker',
                'podcast host speaking',
                'press conference speaker',
            ],
            'filename': 'test_with_faces.mp4',
            'expected_face_score': 0.4
        },
        'without_faces': {
            'searches': [
                'nature landscape drone',
                'ocean waves aerial',
                'mountain scenery timelapse',
                'forest trees nature',
                'desert landscape drone',
                'city skyline timelapse',
                'clouds sky timelapse',
                'waterfall nature footage',
            ],
            'filename': 'test_without_faces.mp4',
            'expected_face_score': 0.2
        }
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    
    for key, video_info in test_videos.items():
        output_path = output_dir / video_info['filename']
        
        # Skip if already exists
        if output_path.exists():
            print(f"    Using existing: {video_info['filename']}")
            results[key] = {
                'path': str(output_path),
                'expected_face_score': video_info['expected_face_score']
            }
            continue
        
        # Try each search query until one works
        success = False
        for search_query in video_info['searches']:
            print(f"    Trying: {search_query[:40]}...", end=" ", flush=True)
            
            # Build yt-dlp command
            cmd = [
                'yt-dlp',
                '--format', 'worst[ext=mp4]/worst',  # Smallest file for speed
                '--max-downloads', '1',
                '--match-filter', 'duration < 120',  # Max 2 minutes
                '--output', str(output_path),
                '--no-playlist',
                '--quiet',
                '--no-warnings',
                f'ytsearch1:{search_query}'
            ]
            
            if cookies_path and Path(cookies_path).exists():
                cmd.extend(['--cookies', cookies_path])
            
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                
                if output_path.exists() and output_path.stat().st_size > 10000:  # At least 10KB
                    results[key] = {
                        'path': str(output_path),
                        'expected_face_score': video_info['expected_face_score']
                    }
                    print(f"✓")
                    print(f"    Downloaded: {video_info['filename']}")
                    success = True
                    break
                else:
                    # Clean up partial download
                    if output_path.exists():
                        output_path.unlink()
                    print(f"✗")
                    
            except subprocess.TimeoutExpired:
                print(f"timeout")
                if output_path.exists():
                    output_path.unlink()
            except subprocess.CalledProcessError:
                print(f"✗")
            except FileNotFoundError:
                print(f"\n    ✗ yt-dlp not found. Install with: pip install yt-dlp")
                return results
        
        if not success:
            print(f"    ⚠ Could not download {video_info['filename']} after {len(video_info['searches'])} attempts")
    
    return results


# =============================================================================
# FEATURE TESTS
# =============================================================================

@pytest.mark.fast
def test_imports():
    """Test that all required modules can be imported"""
    required_modules = [
        ('src.config', 'Config'),
        ('src.logger', 'RunLogger'),
        ('src.face_detection', 'FaceDetector'),  # Fixed: FaceDetector is in face_detection, not matching
    ]
    
    failed = []
    for module_name, class_name in required_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            if not hasattr(module, class_name):
                failed.append(f"{module_name}.{class_name} not found")
        except ImportError as e:
            failed.append(f"{module_name}: {e}")
    
    if failed:
        assert False, f"Import failures: {', '.join(failed)}"


@pytest.mark.fast
def test_config_loading():
    """Test config loading with new fields (face_preference, root_dir)"""
    from src.config import load_config
    
    config = load_config()
    
    # Check face_preference exists
    if not hasattr(config.enhanced, 'face_preference'):
        assert False, "config.enhanced.face_preference not found"
    
    face_pref = config.enhanced.face_preference
    if face_pref not in ['neutral', 'more', 'none']:
        assert False, f"Invalid face_preference: {face_pref}"
    
    # Check root_dir fields exist
    if not hasattr(config.download, 'root_dir'):
        assert False, "config.download.root_dir not found"
    
    if not hasattr(config.image_search, 'root_dir'):
        assert False, "config.image_search.root_dir not found"


@pytest.mark.fast
def test_face_detector_init():
    """Test FaceDetector initialization and backend detection"""
    from src.face_detection import FaceDetector
    
    # Reset class state to force re-initialization
    FaceDetector._instance = None
    FaceDetector._mediapipe_available = None
    FaceDetector._opencv_available = None
    FaceDetector._mp_face_detection = None
    
    detector = FaceDetector.get_instance()
    
    # Check which backend is available
    mediapipe_ok = FaceDetector._mediapipe_available
    opencv_ok = FaceDetector._opencv_available
    
    if mediapipe_ok:
        backend = "MediaPipe"
    elif opencv_ok:
        backend = "OpenCV (fallback)"
    else:
        assert False, "No face detection backend available (install mediapipe or opencv-python)"


@pytest.mark.integration
def test_face_detection_with_faces(video_path: str):
    """Test face detection on video expected to have faces"""
    from src.face_detection import FaceDetector
    
    if not video_path or not Path(video_path).exists():
        assert False, "Test video not available"
    
    detector = FaceDetector.get_instance()
    score = detector.get_face_score(video_path)
    
    # Video with faces should score > 0.2
    assert score >= 0.2, f"Face score too low: {score:.2f} (expected >= 0.2)"


@pytest.mark.integration
def test_face_detection_without_faces(video_path: str):
    """Test face detection on video expected to have no faces"""
    from src.face_detection import FaceDetector
    
    if not video_path or not Path(video_path).exists():
        assert False, "Test video not available"
    
    detector = FaceDetector.get_instance()
    score = detector.get_face_score(video_path)
    
    # Video without faces should score < 0.5
    # (we're lenient here since search results aren't guaranteed)
    assert score <= 0.8, f"Face score too high: {score:.2f} (expected <= 0.8)"


@pytest.mark.integration
def test_face_detection_caching(video_path: str, cache_dir: Path):
    """Test that face detection results are cached"""
    from src.face_detection import FaceDetector
    
    if not video_path or not Path(video_path).exists():
        assert False, "Test video not available"
    
    # Clear cache
    FaceDetector._cache.clear()
    cache_file = cache_dir / '.face_cache.json'
    if cache_file.exists():
        cache_file.unlink()
    
    detector = FaceDetector.get_instance()
    
    # First call - should compute
    start1 = time.time()
    score1 = detector.get_face_score(video_path, str(cache_dir))
    time1 = time.time() - start1
    
    # Second call - should use memory cache (much faster)
    start2 = time.time()
    score2 = detector.get_face_score(video_path, str(cache_dir))
    time2 = time.time() - start2
    
    # Check scores match
    if score1 != score2:
        assert False, f"Cache returned different score: {score1} vs {score2}"
    
    # Check disk cache was created
    if not cache_file.exists():
        assert False, "Disk cache not created"
    
    # Second call should be much faster (at least 5x)
    assert time2 < time1 / 5, f"Cache didn't speed up: {time1:.3f}s vs {time2:.3f}s"


@pytest.mark.integration
def test_logger_stats_tracking(temp_dir: Path):
    """Test logger stats tracking and summary generation"""
    from src.logger import RunLogger
    
    log_dir = temp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = RunLogger(log_dir=str(log_dir))
    
    # Test set_stats
    logger.set_stats(
        total_segments=100,
        total_matches=95,
        videos_downloaded=50,
        videos_skipped=10,
        videos_failed=5,
        embeddings_computed=1000,
        embedding_cache_hits=200,
        entity_images_downloaded=25,
        entity_videos_downloaded=10
    )
    
    # Test log_stage_complete
    logger.log_stage_complete("TEST_STAGE", 5.5, {"items": 100})
    
    # Test log_file_generated
    logger.log_file_generated("test_output", "/path/to/test.otio")
    
    # Finalize
    logger.finalize()
    
    # Check JSON was created
    if not Path(logger.json_file).exists():
        assert False, "JSON log not created"
    
    # Check summary files were created
    base_name = Path(logger.json_file).stem
    md_path = log_dir / f"{base_name}_summary.md"
    txt_path = log_dir / f"{base_name}_summary.txt"
    
    if not md_path.exists():
        assert False, "Markdown summary not created"
    
    if not txt_path.exists():
        assert False, "Plaintext summary not created"
    
    # Verify JSON content
    with open(logger.json_file, 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    if summary.get('total_segments') != 100:
        assert False, f"total_segments wrong: {summary.get('total_segments')}"
    
    if summary.get('videos_downloaded') != 50:
        assert False, f"videos_downloaded wrong: {summary.get('videos_downloaded')}"
    
    if 'TEST_STAGE' not in data.get('stage_timings', {}):
        assert False, "Stage timing not recorded"


@pytest.mark.integration
def test_logger_stage_timing(temp_dir: Path):
    """Test logger stage timing accuracy"""
    from src.logger import RunLogger
    
    log_dir = temp_dir / 'logs2'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = RunLogger(log_dir=str(log_dir))
    
    # Log multiple stages
    logger.log_stage_complete("STAGE_A", 1.5, {"count": 10})
    logger.log_stage_complete("STAGE_B", 2.5, {"count": 20})
    logger.log_stage_complete("STAGE_C", 0.5, {"count": 5})
    
    # Check timings
    timings = logger.run_log.stage_timings
    
    if len(timings) != 3:
        assert False, f"Expected 3 stages, got {len(timings)}"
    
    if timings.get("STAGE_A") != 1.5:
        assert False, f"STAGE_A timing wrong"
    
    if timings.get("STAGE_B") != 2.5:
        assert False, f"STAGE_B timing wrong"
    
    total = sum(timings.values())
    if abs(total - 4.5) > 0.01:
        assert False, f"Total timing wrong: {total}"


@pytest.mark.fast
def test_apply_face_preference():
    """Test face preference score adjustment"""
    from src.face_detection import apply_face_preference, FaceDetector
    from src.utils import SRTSegment

    # Create mock segments with cached face scores
    segments = []
    for i, score in enumerate([0.8, 0.2, 0.5]):  # High, low, medium face scores
        seg = SRTSegment(
            index=i,
            start_time=i * 10.0,
            end_time=(i + 1) * 10.0,
            text=f"Test segment {i}",
            source_file=f"/fake/video_{i}.mp4"
        )
        segments.append(seg)
        # Pre-cache face scores in SCENE cache (segment-level detection)
        # Cache key format: normalized_path -> time_key -> score
        video_path = f"/fake/video_{i}.mp4"
        time_key = f"{seg.start_time:.1f}-{seg.end_time:.1f}"
        if video_path not in FaceDetector._scene_cache:
            FaceDetector._scene_cache[video_path] = {}
        FaceDetector._scene_cache[video_path][time_key] = score
    
    # Test "more" preference
    candidates = [(seg, 0.7) for seg in segments]
    adjusted_more = apply_face_preference(candidates, "more")
    
    # Video with most faces (0.8) should be boosted most
    scores_more = {seg.source_file: score for seg, score in adjusted_more}
    if scores_more["/fake/video_0.mp4"] <= scores_more["/fake/video_1.mp4"]:
        assert False, "MORE preference didn't boost high-face video"
    
    # Test "none" preference
    adjusted_none = apply_face_preference(candidates, "none")
    
    # Video with most faces should be penalized most
    scores_none = {seg.source_file: score for seg, score in adjusted_none}
    if scores_none["/fake/video_0.mp4"] >= scores_none["/fake/video_1.mp4"]:
        assert False, "NONE preference didn't penalize high-face video"
    
    # Test "neutral" preference (no change)
    adjusted_neutral = apply_face_preference(candidates, "neutral")
    scores_neutral = {seg.source_file: score for seg, score in adjusted_neutral}
    
    for seg, orig_score in candidates:
        if scores_neutral[seg.source_file] != orig_score:
            assert False, "NEUTRAL preference modified scores"


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test recent features')
    parser.add_argument('--keep-videos', action='store_true', 
                        help='Keep downloaded test videos after testing')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip video download, use existing test videos')
    parser.add_argument('--cookies', type=str, default=None,
                        help='Path to cookies.txt for YouTube')
    args = parser.parse_args()
    
    print(f"\n{'=' * 60}")
    print(f"  FEATURE TEST SUITE")
    print(f"{'=' * 60}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Working dir: {Path.cwd()}")
    
    runner = FeatureTestRunner()
    
    # Setup
    test_dir = Path(__file__).parent / 'test_data'
    test_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp())
    
    # Find cookies
    cookies_path = args.cookies
    if not cookies_path:
        for cp in ['cookies.txt', '../cookies.txt', '../../cookies.txt']:
            if Path(cp).exists():
                cookies_path = cp
                break
    
    # ─────────────────────────────────────────────────────────────────────
    # SECTION 1: Basic Tests
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  ─── Basic Tests ───")
    
    runner.run_test("Import modules", test_imports)
    runner.run_test("Config loading", test_config_loading)
    runner.run_test("FaceDetector init", test_face_detector_init)
    
    # ─────────────────────────────────────────────────────────────────────
    # SECTION 2: Download Test Videos
    # ─────────────────────────────────────────────────────────────────────
    test_videos = {}
    
    if not args.skip_download:
        print(f"\n  ─── Downloading Test Videos ───")
        test_videos = download_test_videos(test_dir, cookies_path)
        
        if not test_videos:
            print(f"\n  ⚠ No test videos available, skipping video-based tests")
    else:
        # Try to find existing test videos
        for key, filename in [('with_faces', 'test_with_faces.mp4'), 
                              ('without_faces', 'test_without_faces.mp4')]:
            path = test_dir / filename
            if path.exists():
                test_videos[key] = {'path': str(path), 'expected_face_score': 0.5}
        
        if test_videos:
            print(f"\n  Using {len(test_videos)} existing test videos")
    
    # ─────────────────────────────────────────────────────────────────────
    # SECTION 3: Face Detection Tests
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  ─── Face Detection Tests ───")
    
    if 'with_faces' in test_videos:
        runner.run_test(
            "Face detection (with faces)", 
            test_face_detection_with_faces,
            test_videos['with_faces']['path']
        )
    
    if 'without_faces' in test_videos:
        runner.run_test(
            "Face detection (without faces)",
            test_face_detection_without_faces,
            test_videos['without_faces']['path']
        )
    
    if test_videos:
        first_video = list(test_videos.values())[0]['path']
        runner.run_test(
            "Face detection caching",
            test_face_detection_caching,
            first_video,
            temp_dir
        )
    
    runner.run_test("Face preference adjustment", test_apply_face_preference)
    
    # ─────────────────────────────────────────────────────────────────────
    # SECTION 4: Logger Tests
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n  ─── Logger Tests ───")
    
    runner.run_test("Logger stats tracking", test_logger_stats_tracking, temp_dir)
    runner.run_test("Logger stage timing", test_logger_stage_timing, temp_dir)
    
    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────
    # Clean up temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Clean up test videos if requested
    if not args.keep_videos and test_videos:
        print(f"\n  Cleaning up test videos...")
        for video_info in test_videos.values():
            try:
                Path(video_info['path']).unlink()
            except:
                pass
    
    # ─────────────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────────────
    success = runner.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
