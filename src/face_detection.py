"""
Face Detection Module

Detects faces in videos and scenes using MediaPipe (preferred) or OpenCV (fallback).
Supports both video-level and scene-level face detection for B-roll identification.

B-roll scenes (no faces) are preferred when topic matches voiceover content,
as they provide relevant visual coverage without talking heads.
"""

import logging
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detect faces in video files using MediaPipe (preferred) or OpenCV (fallback).
    Caches results per video file for performance.

    MediaPipe is faster and more accurate than OpenCV Haar cascades.
    """

    _instance = None
    _cache: Dict[str, float] = {}  # video_path -> face_score (0-1)
    _scene_cache: Dict[str, Dict[int, float]] = {}  # video_path -> {scene_index: face_score}
    _mediapipe_available = None
    _opencv_available = None
    _mp_face_detection = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._check_backends()

    def _check_backends(self):
        """Check available face detection backends (MediaPipe preferred)"""
        # Check MediaPipe first (preferred)
        if FaceDetector._mediapipe_available is None:
            try:
                import mediapipe as mp
                # Check if solutions attribute exists (some versions don't have it)
                if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
                    FaceDetector._mp_face_detection = mp.solutions.face_detection.FaceDetection(
                        model_selection=0,  # 0 = short-range (within 2m), 1 = full-range
                        min_detection_confidence=0.5
                    )
                    FaceDetector._mediapipe_available = True
                    logger.debug("MediaPipe face detection available")
                else:
                    FaceDetector._mediapipe_available = False
                    logger.debug("MediaPipe installed but solutions.face_detection not available")
            except ImportError:
                FaceDetector._mediapipe_available = False
                logger.debug("MediaPipe not installed, trying OpenCV fallback")
            except Exception as e:
                FaceDetector._mediapipe_available = False
                logger.debug(f"MediaPipe init failed: {e}, trying OpenCV fallback")

        # Check OpenCV fallback
        if FaceDetector._opencv_available is None:
            try:
                import cv2
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if Path(cascade_path).exists():
                    FaceDetector._opencv_available = True
                    logger.debug("OpenCV face detection available")
                else:
                    FaceDetector._opencv_available = False
                    logger.debug("OpenCV cascade file not found")
            except ImportError:
                FaceDetector._opencv_available = False
                logger.debug("OpenCV not installed - face detection disabled")
            except Exception as e:
                FaceDetector._opencv_available = False
                logger.debug(f"OpenCV init failed: {e}")

    def is_available(self) -> bool:
        """Check if face detection is available"""
        return FaceDetector._mediapipe_available or FaceDetector._opencv_available

    def get_face_score(self, video_path: str, cache_dir: str = None) -> float:
        """
        Get face score for a video (0 = no faces, 1 = many faces).
        Uses cached result if available.

        Args:
            video_path: Path to video file
            cache_dir: Directory to store/load face cache

        Returns:
            Face score from 0.0 (no faces) to 1.0 (many faces)
        """
        if not FaceDetector._mediapipe_available and not FaceDetector._opencv_available:
            return 0.5  # Neutral if no backend available

        # Check memory cache
        if video_path in FaceDetector._cache:
            return FaceDetector._cache[video_path]

        # Check disk cache
        if cache_dir:
            cache_path = Path(cache_dir) / '.face_cache.json'
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        disk_cache = json.load(f)
                    if video_path in disk_cache:
                        score = disk_cache[video_path]
                        FaceDetector._cache[video_path] = score
                        return score
                except:
                    pass

        # Compute face score
        if FaceDetector._mediapipe_available:
            score = self._detect_faces_mediapipe(video_path)
        else:
            score = self._detect_faces_opencv(video_path)

        FaceDetector._cache[video_path] = score

        # Save to disk cache
        if cache_dir:
            try:
                cache_path = Path(cache_dir) / '.face_cache.json'
                disk_cache = {}
                if cache_path.exists():
                    with open(cache_path, 'r') as f:
                        disk_cache = json.load(f)
                disk_cache[video_path] = score
                with open(cache_path, 'w') as f:
                    json.dump(disk_cache, f)
            except:
                pass

        return score

    def get_scene_face_score(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        scene_index: int = 0,
        sample_frames: int = 3,
        cache_dir: str = None
    ) -> float:
        """
        Get face score for a specific scene/time range within a video.

        Args:
            video_path: Path to video file
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds
            scene_index: Scene index for caching
            sample_frames: Number of frames to sample within the scene
            cache_dir: Directory for cache

        Returns:
            Face score from 0.0 (no faces = B-roll) to 1.0 (all faces = talking head)
        """
        if not self.is_available():
            return 0.5  # Neutral if no backend

        # Check scene cache
        cache_key = f"{video_path}:{scene_index}"
        if video_path in FaceDetector._scene_cache:
            if scene_index in FaceDetector._scene_cache[video_path]:
                return FaceDetector._scene_cache[video_path][scene_index]

        # Check disk cache for scene faces
        if cache_dir:
            cache_path = Path(cache_dir) / '.scene_face_cache.json'
            if cache_path.exists():
                try:
                    with open(cache_path, 'r') as f:
                        disk_cache = json.load(f)
                    if cache_key in disk_cache:
                        score = disk_cache[cache_key]
                        if video_path not in FaceDetector._scene_cache:
                            FaceDetector._scene_cache[video_path] = {}
                        FaceDetector._scene_cache[video_path][scene_index] = score
                        return score
                except:
                    pass

        # Compute face score for scene
        if FaceDetector._mediapipe_available:
            score = self._detect_faces_in_range_mediapipe(
                video_path, start_time, end_time, sample_frames
            )
        else:
            score = self._detect_faces_in_range_opencv(
                video_path, start_time, end_time, sample_frames
            )

        # Update memory cache
        if video_path not in FaceDetector._scene_cache:
            FaceDetector._scene_cache[video_path] = {}
        FaceDetector._scene_cache[video_path][scene_index] = score

        # Save to disk cache
        if cache_dir:
            try:
                cache_path = Path(cache_dir) / '.scene_face_cache.json'
                disk_cache = {}
                if cache_path.exists():
                    with open(cache_path, 'r') as f:
                        disk_cache = json.load(f)
                disk_cache[cache_key] = score
                with open(cache_path, 'w') as f:
                    json.dump(disk_cache, f)
            except:
                pass

        return score

    def _detect_faces_mediapipe(self, video_path: str, sample_frames: int = 5) -> float:
        """
        Detect faces using MediaPipe (faster and more accurate).

        Returns score from 0.0 (no faces) to 1.0 (faces in all sampled frames).
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.5

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return 0.5

            # Sample frames evenly throughout video
            frames_with_faces = 0
            sample_positions = [int(total_frames * i / (sample_frames + 1)) for i in range(1, sample_frames + 1)]

            for pos in sample_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                results = FaceDetector._mp_face_detection.process(rgb_frame)

                if results.detections and len(results.detections) > 0:
                    frames_with_faces += 1

            cap.release()

            # Return ratio of frames with faces
            return frames_with_faces / sample_frames

        except Exception as e:
            logger.debug(f"MediaPipe face detection error for {video_path}: {e}")
            return 0.5  # Neutral on error

    def _detect_faces_opencv(self, video_path: str, sample_frames: int = 5) -> float:
        """
        Detect faces using OpenCV Haar cascades (fallback).

        Returns score from 0.0 (no faces) to 1.0 (faces in all sampled frames).
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.5

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return 0.5

            # Load face cascade
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            # Sample frames evenly throughout video
            frames_with_faces = 0
            sample_positions = [int(total_frames * i / (sample_frames + 1)) for i in range(1, sample_frames + 1)]

            for pos in sample_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                # Convert to grayscale for detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                if len(faces) > 0:
                    frames_with_faces += 1

            cap.release()

            # Return ratio of frames with faces
            return frames_with_faces / sample_frames

        except Exception as e:
            logger.debug(f"OpenCV face detection error for {video_path}: {e}")
            return 0.5  # Neutral on error

    def _detect_faces_in_range_mediapipe(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_frames: int = 3
    ) -> float:
        """Detect faces within a specific time range using MediaPipe."""
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.5

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                cap.release()
                return 0.5

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            frame_range = end_frame - start_frame

            if frame_range <= 0:
                cap.release()
                return 0.5

            # Sample frames evenly within the scene
            frames_with_faces = 0
            samples_checked = 0

            for i in range(1, sample_frames + 1):
                pos = start_frame + int(frame_range * i / (sample_frames + 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                samples_checked += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = FaceDetector._mp_face_detection.process(rgb_frame)

                if results.detections and len(results.detections) > 0:
                    frames_with_faces += 1

            cap.release()

            if samples_checked == 0:
                return 0.5

            return frames_with_faces / samples_checked

        except Exception as e:
            logger.debug(f"MediaPipe scene face detection error: {e}")
            return 0.5

    def _detect_faces_in_range_opencv(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        sample_frames: int = 3
    ) -> float:
        """Detect faces within a specific time range using OpenCV."""
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.5

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                cap.release()
                return 0.5

            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            frame_range = end_frame - start_frame

            if frame_range <= 0:
                cap.release()
                return 0.5

            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

            frames_with_faces = 0
            samples_checked = 0

            for i in range(1, sample_frames + 1):
                pos = start_frame + int(frame_range * i / (sample_frames + 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                samples_checked += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

                if len(faces) > 0:
                    frames_with_faces += 1

            cap.release()

            if samples_checked == 0:
                return 0.5

            return frames_with_faces / samples_checked

        except Exception as e:
            logger.debug(f"OpenCV scene face detection error: {e}")
            return 0.5


def apply_face_preference(
    candidates: List[Tuple],
    face_preference: str,
    cache_dir: str = None
) -> List[Tuple]:
    """
    Adjust candidate scores based on face preference.

    Args:
        candidates: List of (segment, score) tuples
        face_preference: "more", "none", or "neutral"
        cache_dir: Cache directory for face detection results

    Returns:
        Adjusted candidates with modified scores
    """
    if face_preference == "neutral":
        return candidates

    detector = FaceDetector.get_instance()

    adjusted = []
    for seg, score in candidates:
        video_path = seg.source_file
        face_score = detector.get_face_score(video_path, cache_dir)

        # Apply adjustment based on preference
        if face_preference == "more":
            # Boost videos with faces (face_score: 0-1)
            # More faces = higher boost (up to +0.1)
            adjustment = face_score * 0.1
        elif face_preference == "none":
            # Penalize videos with faces
            # More faces = bigger penalty (up to -0.15)
            adjustment = -face_score * 0.15
        else:
            adjustment = 0

        adjusted_score = min(1.0, max(0.0, score + adjustment))
        adjusted.append((seg, adjusted_score))

    # Re-sort by adjusted score
    adjusted.sort(key=lambda x: x[1], reverse=True)

    return adjusted


def apply_broll_preference(
    candidates: List[Tuple],
    vo_topics: List[str],
    video_topics: Dict[str, List[str]],
    scene_face_scores: Dict[str, Dict[int, float]],
    broll_boost: float = 0.1,
    broll_threshold: float = 0.3
) -> List[Tuple]:
    """
    Boost B-roll scenes (no faces) when topic matches voiceover.

    Args:
        candidates: List of (segment, score) tuples
        vo_topics: Topics from voiceover chapter
        video_topics: Dict of video_path -> topics list
        scene_face_scores: Dict of video_path -> {scene_index: face_score}
        broll_boost: Score boost for matching B-roll (default 0.1 = +10%)
        broll_threshold: Face score below this is considered B-roll (default 0.3)

    Returns:
        Adjusted candidates with B-roll boost applied
    """
    if not vo_topics:
        return candidates

    adjusted = []
    for seg, score in candidates:
        video_path = seg.source_file

        # Check topic overlap
        topics = video_topics.get(video_path, [])
        has_topic_match = False

        for vo_topic in vo_topics:
            vo_lower = vo_topic.lower()
            for vid_topic in topics:
                if vo_lower in vid_topic.lower() or vid_topic.lower() in vo_lower:
                    has_topic_match = True
                    break
            if has_topic_match:
                break

        # Check if scene is B-roll (no faces)
        face_score = 0.5  # Default neutral
        if video_path in scene_face_scores:
            # Try to get scene-specific face score
            scene_idx = getattr(seg, 'scene_index', 0)
            face_score = scene_face_scores[video_path].get(scene_idx, 0.5)

        is_broll = face_score < broll_threshold

        # Apply boost if topic matches AND scene is B-roll
        if has_topic_match and is_broll:
            adjusted_score = min(1.0, score + broll_boost)
            logger.debug(f"B-roll boost: {Path(video_path).name} scene {getattr(seg, 'scene_index', '?')} "
                        f"+{broll_boost:.2f} (face={face_score:.2f}, topics match)")
        else:
            adjusted_score = score

        adjusted.append((seg, adjusted_score))

    # Re-sort by adjusted score
    adjusted.sort(key=lambda x: x[1], reverse=True)

    return adjusted


def is_broll_scene(face_score: float, threshold: float = 0.3) -> bool:
    """Check if a scene is B-roll (no faces) based on face score."""
    return face_score < threshold
