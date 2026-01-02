"""
Scene Detection Module

Detects scene boundaries in videos using PySceneDetect.
Outputs both OTIO files and scene index data for matching.

Integration point: After transcription, before matching.
"""

import os
# Suppress FFmpeg H.264 decoder warnings from OpenCV (must be set before cv2 import)
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"  # AV_LOG_QUIET
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import json
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
from scenedetect import open_video, SceneManager, ContentDetector
import opentimelineio as otio

logger = logging.getLogger(__name__)

# Supported video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpeg', '.mpg', '.3gp', '.mxf'}


def sanitize_path_for_url(path: str) -> str:
    """
    Sanitize a file path for use as a URL in OTIO.
    Removes Windows extended-length path prefix and normalizes slashes.
    """
    path = str(path)
    
    # Remove Windows extended-length path prefix
    if path.startswith('\\\\?\\'):
        path = path[4:]
    elif path.startswith('//?/'):
        path = path[4:]
    
    # Convert backslashes to forward slashes
    path = path.replace('\\', '/')
    
    return path


def get_file_hash(file_path: Path) -> str:
    """Get fast hash based on path + size + mtime"""
    stat = file_path.stat()
    hash_input = f"{file_path.resolve()}|{stat.st_size}|{stat.st_mtime}"
    return hashlib.md5(hash_input.encode()).hexdigest()


@dataclass
class SceneInfo:
    """Information about a detected scene"""
    scene_index: int
    start_frame: int
    end_frame: int
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds
    # Face detection for B-roll identification
    face_score: float = 0.5  # 0.0 = no faces (B-roll), 1.0 = all faces (talking head)
    is_broll: bool = False   # True if face_score < broll_threshold (default 0.3)

    def to_dict(self) -> dict:
        # Convert to native Python types for JSON serialization
        return {
            'scene_index': int(self.scene_index),
            'start_frame': int(self.start_frame),
            'end_frame': int(self.end_frame),
            'start_time': float(self.start_time),
            'end_time': float(self.end_time),
            'duration': float(self.duration),
            'face_score': float(self.face_score),
            'is_broll': bool(self.is_broll)
        }


@dataclass
class VideoSceneData:
    """Scene detection results for a video"""
    video_path: str
    video_name: str
    framerate: float
    total_frames: int
    total_duration: float
    scene_count: int
    scenes: List[SceneInfo]
    otio_path: Optional[str] = None
    # Audio analysis fields
    has_speech: bool = False
    speech_ratio: float = 0.0
    audio_cut_points: List[float] = None
    # Cache validation - file hash based on path+size+mtime
    file_hash: str = ""
    
    def __post_init__(self):
        if self.audio_cut_points is None:
            self.audio_cut_points = []
    
    def to_dict(self) -> dict:
        # Convert numpy types to native Python for JSON serialization
        return {
            'video_path': str(self.video_path),
            'video_name': str(self.video_name),
            'framerate': float(self.framerate),
            'total_frames': int(self.total_frames),
            'total_duration': float(self.total_duration),
            'scene_count': int(self.scene_count),
            'scenes': [s.to_dict() for s in self.scenes],
            'otio_path': str(self.otio_path) if self.otio_path else None,
            'has_speech': bool(self.has_speech),
            'speech_ratio': float(self.speech_ratio),
            'audio_cut_points': [float(x) for x in self.audio_cut_points] if self.audio_cut_points else [],
            'file_hash': str(self.file_hash)
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "VideoSceneData":
        scenes = []
        for s in data.get('scenes', []):
            # Handle both old format (without face_score) and new format
            scene = SceneInfo(
                scene_index=s['scene_index'],
                start_frame=s['start_frame'],
                end_frame=s['end_frame'],
                start_time=s['start_time'],
                end_time=s['end_time'],
                duration=s['duration'],
                face_score=s.get('face_score', 0.5),
                is_broll=s.get('is_broll', False)
            )
            scenes.append(scene)
        return cls(
            video_path=data['video_path'],
            video_name=data['video_name'],
            framerate=data['framerate'],
            total_frames=data['total_frames'],
            total_duration=data['total_duration'],
            scene_count=data['scene_count'],
            scenes=scenes,
            otio_path=data.get('otio_path'),
            has_speech=data.get('has_speech', False),
            speech_ratio=data.get('speech_ratio', 0.0),
            audio_cut_points=data.get('audio_cut_points', []),
            file_hash=data.get('file_hash', '')
        )


# Speed presets - intentionally hardcoded as these are standard templates
# Users select preset via config.scene_detection.preset and can override
# individual settings via config.scene_detection.threshold, etc.
PRESETS = {
    'fast': {
        'downscale': 6,
        'frame_skip': 3,
        'threshold': 30.0,
        'min_scene_length': 20
    },
    'balanced': {
        'downscale': 4,
        'frame_skip': 2,
        'threshold': 27.0,
        'min_scene_length': 15
    },
    'accurate': {
        'downscale': 2,
        'frame_skip': 0,
        'threshold': 25.0,
        'min_scene_length': 10
    }
}


class SceneDetector:
    """
    Scene detection with OTIO export, index generation, and audio analysis.
    
    Features:
    - Visual scene detection using PySceneDetect
    - Audio analysis using librosa (silence/speech detection)
    - OTIO export for DaVinci Resolve
    - Scene index for matching
    """
    
    def __init__(self, config):
        """Initialize with config"""
        self.config = config
        self.scene_config = config.scene_detection
        
        # Get preset settings
        preset = PRESETS.get(self.scene_config.preset, PRESETS['balanced'])
        
        # Allow config overrides (use correct field names from config)
        self.downscale = getattr(self.scene_config, 'downscale_factor', None) or preset['downscale']
        self.frame_skip = getattr(self.scene_config, 'frame_skip', None) or preset['frame_skip']
        self.threshold = getattr(self.scene_config, 'threshold', None) or preset['threshold']
        self.min_scene_length = getattr(self.scene_config, 'min_scene_len', None) or preset['min_scene_length']
        
        # Output paths - use config values with defaults
        output_dir = getattr(config, 'output_dir', 'output')
        cache_dir = getattr(config.cache, 'cache_dir', '.cache') if hasattr(config, 'cache') else '.cache'
        
        self.otio_output_dir = Path(output_dir) / "scene_otio"
        self.scene_index_path = Path(cache_dir) / "scene_index.json"
        
        # Scene index (loaded or created)
        self.scene_index: Dict[str, VideoSceneData] = {}
        self._load_scene_index()
        
        # Setup hardware acceleration
        self._setup_hw_accel()

        # Initialize audio analyzer
        self._init_audio_analyzer()

        # Initialize face detector for B-roll identification
        self._init_face_detector()
    
    def _setup_hw_accel(self):
        """Configure OpenCV for hardware acceleration"""
        use_gpu = getattr(self.scene_config, 'use_gpu', True)
        force_gpu = getattr(self.scene_config, 'force_gpu', False)
        
        self.hw_info = {
            'cuda_available': cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False,
            'opencl_available': cv2.ocl.haveOpenCL()
        }
        
        if not use_gpu:
            logger.info("Scene detection: GPU disabled in config")
            return
        
        # Force CUDA if requested and available
        if force_gpu and self.hw_info['cuda_available']:
            logger.info("Scene detection: Forcing CUDA GPU acceleration")
            self.hw_info['using_cuda'] = True
            # Set CUDA as preferred backend
            try:
                cv2.cuda.setDevice(0)
            except:
                pass
        elif self.hw_info['opencl_available']:
            cv2.ocl.setUseOpenCL(True)
            self.hw_info['opencl_enabled'] = cv2.ocl.useOpenCL()
        
        cuda_status = "CUDA (forced)" if force_gpu and self.hw_info['cuda_available'] else f"CUDA={self.hw_info.get('cuda_available')}"
        logger.info(f"Scene detection HW: {cuda_status}, "
                   f"OpenCL={self.hw_info.get('opencl_enabled', False)}")
    
    def _init_audio_analyzer(self):
        """Initialize audio analyzer for silence/speech detection"""
        self.audio_analyzer = None
        self.audio_enabled = getattr(self.scene_config, 'audio_analysis', True)
        
        if not self.audio_enabled:
            logger.info("Audio analysis disabled in config")
            return
        
        try:
            from .audio_analysis import AudioAnalyzer
            self.audio_analyzer = AudioAnalyzer(self.config)
            if self.audio_analyzer.is_available():
                logger.info("Audio analysis enabled (librosa)")
            else:
                logger.warning("Audio analysis unavailable (librosa not installed)")
                self.audio_analyzer = None
        except ImportError as e:
            logger.warning(f"Could not import audio_analysis module: {e}")
            self.audio_analyzer = None

    def _init_face_detector(self):
        """Initialize face detector for B-roll identification"""
        self.face_detector = None
        self.face_detection_enabled = getattr(self.scene_config, 'detect_faces_per_scene', True)
        self.broll_threshold = getattr(self.config.matching, 'broll_face_threshold', 0.3)

        if not self.face_detection_enabled:
            logger.info("Scene face detection disabled in config")
            return

        try:
            from .face_detection import FaceDetector
            self.face_detector = FaceDetector.get_instance()
            if self.face_detector.is_available():
                logger.info("Scene face detection enabled (for B-roll identification)")
            else:
                logger.warning("Face detection unavailable (MediaPipe/OpenCV not installed)")
                self.face_detector = None
        except ImportError as e:
            logger.warning(f"Could not import face_detection module: {e}")
            self.face_detector = None

    def _load_scene_index(self):
        """Load existing scene index"""
        if self.scene_index_path.exists():
            try:
                with open(self.scene_index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for video_name, scene_data in data.items():
                        self.scene_index[video_name] = VideoSceneData.from_dict(scene_data)
                logger.info(f"Loaded scene index with {len(self.scene_index)} videos")
            except Exception as e:
                logger.warning(f"Could not load scene index: {e}")
                self.scene_index = {}
    
    def _save_scene_index(self):
        """Save scene index to disk"""
        self.scene_index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.scene_index_path, 'w') as f:
            data = {name: sd.to_dict() for name, sd in self.scene_index.items()}
            json.dump(data, f, indent=2, default=self._json_default)
    
    def _json_default(self, obj):
        """Handle numpy types for JSON serialization"""
        import numpy as np
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {type(obj).__name__} is not JSON serializable')
    
    def _detect_scenes_in_video(self, video_path: Path) -> Tuple[List, float, int]:
        """
        Detect scenes in a video using ContentDetector.
        Returns (scene_list, framerate, total_frames)
        """
        # Open video
        video = open_video(str(video_path))
        
        framerate = video.frame_rate
        total_frames = video.duration.get_frames()
        
        # Log resolution info
        width, height = video.frame_size
        if self.downscale > 1:
            proc_w, proc_h = width // self.downscale, height // self.downscale
            logger.debug(f"  {video_path.name}: {width}x{height} → {proc_w}x{proc_h}")
        
        # Create scene manager
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_length
            )
        )
        
        # Detect scenes
        scene_manager.detect_scenes(
            video,
            frame_skip=self.frame_skip,
            show_progress=False
        )
        
        scene_list = scene_manager.get_scene_list()
        
        return scene_list, framerate, total_frames
    
    def _scene_list_to_info(
        self,
        scene_list: List,
        framerate: float,
        total_frames: int
    ) -> List[SceneInfo]:
        """Convert PySceneDetect scene list to SceneInfo objects"""
        scenes = []
        
        if not scene_list:
            # Single scene (entire video)
            total_duration = total_frames / framerate if framerate > 0 else 0
            scenes.append(SceneInfo(
                scene_index=0,
                start_frame=0,
                end_frame=total_frames,
                start_time=0.0,
                end_time=total_duration,
                duration=total_duration
            ))
        else:
            for i, (start, end) in enumerate(scene_list):
                start_frame = start.get_frames()
                end_frame = end.get_frames()
                start_time = start.get_seconds()
                end_time = end.get_seconds()
                
                scenes.append(SceneInfo(
                    scene_index=i,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time
                ))
        
        return scenes
    
    def _create_otio_timeline(
        self,
        video_path: Path,
        scenes: List[SceneInfo],
        framerate: float,
        total_frames: int
    ) -> otio.schema.Timeline:
        """Create OTIO timeline from scene data"""
        timeline = otio.schema.Timeline(name=video_path.stem)
        
        # Create video and audio tracks
        video_track = otio.schema.Track(name="Video", kind=otio.schema.TrackKind.Video)
        audio_track = otio.schema.Track(name="Audio", kind=otio.schema.TrackKind.Audio)
        
        # Parse start timecode
        try:
            global_start = otio.opentime.from_timecode(self.start_timecode, framerate)
        except:
            global_start = otio.opentime.RationalTime(0, framerate)
        
        # Create media reference with sanitized path
        clean_path = sanitize_path_for_url(str(video_path.resolve()))
        media_ref = otio.schema.ExternalReference(
            target_url=clean_path,
            available_range=otio.opentime.TimeRange(
                start_time=global_start,
                duration=otio.opentime.RationalTime(total_frames, framerate)
            )
        )
        
        # Add clips for each scene
        for scene in scenes:
            # Calculate timing
            source_start = otio.opentime.RationalTime(scene.start_frame, framerate) + global_start
            clip_duration = otio.opentime.RationalTime(
                scene.end_frame - scene.start_frame, framerate
            )
            
            source_range = otio.opentime.TimeRange(
                start_time=source_start,
                duration=clip_duration
            )
            
            # Video clip
            video_clip = otio.schema.Clip(
                name=f"{video_path.stem}_scene_{scene.scene_index:03d}",
                media_reference=media_ref.clone(),
                source_range=source_range
            )
            video_track.append(video_clip)
            
            # Audio clip (mirrors video)
            audio_clip = otio.schema.Clip(
                name=f"{video_path.stem}_scene_{scene.scene_index:03d}_audio",
                media_reference=media_ref.clone(),
                source_range=source_range
            )
            audio_track.append(audio_clip)
        
        timeline.tracks.append(video_track)
        timeline.tracks.append(audio_track)
        
        return timeline
    
    def process_video(self, video_path: Path) -> Optional[VideoSceneData]:
        """
        Process a single video: detect scenes, analyze audio, create OTIO, update index.
        """
        video_name = video_path.stem
        
        # Compute file hash for cache validation
        current_hash = get_file_hash(video_path)
        
        # Check if already processed with same file hash
        if video_name in self.scene_index:
            cached = self.scene_index[video_name]
            if cached.file_hash == current_hash:
                logger.debug(f"Skipping {video_name} (cached, hash matches)")
                return cached
            else:
                logger.info(f"  Re-processing {video_name} (file changed)")
        
        try:
            logger.info(f"  Detecting scenes: {video_path.name}")
            
            # Detect scenes
            scene_list, framerate, total_frames = self._detect_scenes_in_video(video_path)
            
            # Convert to SceneInfo objects
            scenes = self._scene_list_to_info(scene_list, framerate, total_frames)
            
            logger.info(f"    → {len(scenes)} scene(s) detected")
            
            # Audio analysis
            has_speech = False
            speech_ratio = 0.0
            audio_cut_points = []
            
            if self.audio_analyzer:
                logger.debug(f"    Analyzing audio...")
                audio_result = self.audio_analyzer.analyze_video(str(video_path))
                if audio_result:
                    has_speech = audio_result.has_speech
                    speech_ratio = audio_result.speech_ratio
                    audio_cut_points = audio_result.suggested_cut_points
                    
                    speech_status = "has speech" if has_speech else "no speech"
                    logger.info(f"    → Audio: {speech_status} ({speech_ratio:.1%}), {len(audio_cut_points)} cut points")

            # Face detection per scene (B-roll identification)
            if self.face_detector:
                broll_count = 0
                cache_dir = str(self.scene_index_path.parent)
                for scene in scenes:
                    face_score = self.face_detector.get_scene_face_score(
                        str(video_path),
                        scene.start_time,
                        scene.end_time,
                        scene.scene_index,
                        sample_frames=3,
                        cache_dir=cache_dir
                    )
                    scene.face_score = face_score
                    scene.is_broll = face_score < self.broll_threshold
                    if scene.is_broll:
                        broll_count += 1
                logger.info(f"    → B-roll: {broll_count}/{len(scenes)} scenes (no faces)")

            # Create OTIO timeline
            self.otio_output_dir.mkdir(parents=True, exist_ok=True)
            timeline = self._create_otio_timeline(video_path, scenes, framerate, total_frames)
            
            # Export OTIO
            otio_path = self.otio_output_dir / f"{video_name}.otio"
            otio.adapters.write_to_file(timeline, str(otio_path))
            logger.debug(f"    → Exported: {otio_path.name}")
            
            # Create VideoSceneData with file hash
            total_duration = total_frames / framerate if framerate > 0 else 0
            scene_data = VideoSceneData(
                video_path=str(video_path),
                video_name=video_name,
                framerate=framerate,
                total_frames=total_frames,
                total_duration=total_duration,
                scene_count=len(scenes),
                scenes=scenes,
                otio_path=str(otio_path),
                has_speech=has_speech,
                speech_ratio=speech_ratio,
                audio_cut_points=audio_cut_points,
                file_hash=current_hash
            )
            
            # Update index and save incrementally
            self.scene_index[video_name] = scene_data
            self._save_scene_index()
            
            return scene_data
            
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            return None
    
    def process_all_videos(
        self,
        video_dir: Path,
        min_duration: float = 0
    ) -> Dict[str, VideoSceneData]:
        """
        Process all videos in directory.
        
        Args:
            video_dir: Directory containing videos
            min_duration: Only process videos longer than this (seconds)
        
        Returns:
            Dict mapping video name to scene data
        """
        video_dir = Path(video_dir)
        
        # Find all video files
        videos = []
        for ext in VIDEO_EXTENSIONS:
            videos.extend(video_dir.rglob(f'*{ext}'))
        
        if not videos:
            logger.warning(f"No videos found in {video_dir}")
            return {}
        
        # Filter by duration if specified
        if min_duration > 0:
            filtered = []
            for v in videos:
                try:
                    video = open_video(str(v))
                    duration = video.duration.get_seconds()
                    if duration >= min_duration:
                        filtered.append(v)
                    else:
                        logger.debug(f"Skipping {v.name} (duration {duration:.1f}s < {min_duration}s)")
                except:
                    filtered.append(v)  # Process anyway if can't check
            videos = filtered
        
        # Pre-check: count cached vs new
        cached_count = 0
        for v in videos:
            video_name = v.stem
            if video_name in self.scene_index:
                current_hash = get_file_hash(v)
                if self.scene_index[video_name].file_hash == current_hash:
                    cached_count += 1
        
        need_processing = len(videos) - cached_count
        if cached_count > 0:
            logger.info(f"Found {cached_count}/{len(videos)} cached scene detections")
        
        if need_processing == 0:
            logger.info("All videos already processed - loading from cache...")
        else:
            logger.info(f"Processing {need_processing} videos for scene detection")
        
        # Process each video
        results = {}
        for i, video_path in enumerate(videos, 1):
            logger.info(f"[{i}/{len(videos)}] {video_path.name}")
            scene_data = self.process_video(video_path)
            if scene_data:
                results[scene_data.video_name] = scene_data
        
        # Final save (in case any were skipped without incremental save)
        self._save_scene_index()
        
        return results
    
    def get_scene_clips(self) -> List[Dict]:
        """
        Convert scene index to clip list for matching.
        Each scene becomes a separate matchable clip.
        
        Returns:
            List of clip dicts with scene-level granularity
        """
        clips = []
        
        for video_name, scene_data in self.scene_index.items():
            for scene in scene_data.scenes:
                clip = {
                    'video_path': scene_data.video_path,
                    'video_name': video_name,
                    'clip_id': f"{video_name}_scene_{scene.scene_index:03d}",
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'duration': scene.duration,
                    'scene_index': scene.scene_index,
                    'framerate': scene_data.framerate,
                    'is_scene_clip': True
                }
                clips.append(clip)
        
        return clips
    
    def enrich_video_metadata(self, video_metadata: Dict) -> Dict:
        """
        Add scene boundaries to existing video metadata.
        
        Args:
            video_metadata: Dict with video_name as key
        
        Returns:
            Enriched metadata with scene_boundaries field
        """
        for video_name, metadata in video_metadata.items():
            if video_name in self.scene_index:
                scene_data = self.scene_index[video_name]
                metadata['scene_boundaries'] = [
                    {
                        'index': s.scene_index,
                        'start': s.start_time,
                        'end': s.end_time,
                        'duration': s.duration
                    }
                    for s in scene_data.scenes
                ]
                metadata['scene_count'] = scene_data.scene_count
            else:
                metadata['scene_boundaries'] = []
                metadata['scene_count'] = 0
        
        return video_metadata
    
    def get_stats(self) -> Dict:
        """Get scene detection statistics"""
        if not self.scene_index:
            return {'videos': 0, 'scenes': 0}
        
        total_scenes = sum(sd.scene_count for sd in self.scene_index.values())
        avg_scenes = total_scenes / len(self.scene_index) if self.scene_index else 0
        
        # Audio stats
        videos_with_speech = sum(1 for sd in self.scene_index.values() if sd.has_speech)
        avg_speech_ratio = sum(sd.speech_ratio for sd in self.scene_index.values()) / len(self.scene_index)
        total_audio_cuts = sum(len(sd.audio_cut_points) for sd in self.scene_index.values())
        
        return {
            'videos_processed': len(self.scene_index),
            'total_scenes': total_scenes,
            'avg_scenes_per_video': round(avg_scenes, 1),
            'preset': self.scene_config.preset,
            'threshold': self.threshold,
            'videos_with_speech': videos_with_speech,
            'avg_speech_ratio': round(avg_speech_ratio, 3),
            'total_audio_cut_points': total_audio_cuts,
            'audio_analysis_enabled': self.audio_analyzer is not None
        }


def detect_scenes(config, video_dir: Path = None, min_duration: float = 0) -> SceneDetector:
    """
    Convenience function to run scene detection.
    
    Args:
        config: Pipeline config
        video_dir: Override video directory (uses config default if None)
        min_duration: Minimum video duration to process (seconds, default 0 = no filter)
    
    Returns:
        SceneDetector instance with populated scene index
    """
    detector = SceneDetector(config)
    
    if video_dir is None:
        # Try to get video directory from config with fallback
        video_dir = getattr(config, 'downloaded_videos_dir', None)
        if video_dir is None and hasattr(config, 'pipeline'):
            video_dir = getattr(config.pipeline, 'video_source_dir', None)
        if video_dir is None:
            video_dir = Path('downloaded_videos')
        video_dir = Path(video_dir)
    
    detector.process_all_videos(video_dir, min_duration=min_duration)
    
    return detector
