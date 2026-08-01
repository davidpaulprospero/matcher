#!/usr/bin/env python3
"""
Vision Processing Module - v2.5 (SRTSegment Fix)

Selective Vision Processing System for Video-to-Voiceover Matching.

This module implements a cost-effective vision processing pipeline that minimizes
API calls while maximizing descriptive quality for video segments.

DECISION FLOW:
==============
1. TRANSCRIPT ANALYSIS (TranscriptAnalyzer)
   - Analyzes transcript segments against detected scenes
   - Calculates word count per scene
   - Determines which scenes have "sparse" transcript coverage

2. VISION DECISION (VideoVisionDecision)
   - Compares transcript coverage against coverage_threshold (default 30%)
   - If coverage >= threshold → SKIP vision API (transcript is sufficient)
   - If coverage < threshold → PROCEED with vision API
   - Identifies specific sparse_scenes that need vision processing

3. SCENE PRIORITIZATION
   - Ranks scenes by transcript poverty (least words = highest priority)
   - Limits processing to max_scenes_per_video (default 50)
   - Extracts frames at scene midpoint for analysis

4. VISION PROCESSING (VisionProcessor)
   - Extracts frame at midpoint of each priority scene
   - Sends frame to vision API (Gemini by default)
   - Caches results to avoid redundant API calls

5. HYBRID DESCRIPTION
   - Combines transcript + vision into unified description
   - Format: "{transcript_text} [Visual: {vision_description}]"
   - Preserves both sources of information

CONFIDENCE THRESHOLDS:
======================
- coverage_threshold: 0.30 (30%)
  - If >= 30% of scenes have >= min_words_per_scene, skip vision
  - Config: config.vision.coverage_threshold

- min_words_per_scene: 5 (configurable)
  - Minimum words in transcript to consider scene "covered"
  - Config: config.vision.min_words_per_scene

- max_scenes_per_video: 50 (configurable)
  - Maximum scenes to process with vision API per video
  - Config: config.vision.max_scenes_per_video

USAGE:
======
from src.vision import process_video_vision_full, TranscriptAnalyzer, VisionProcessor

# Full pipeline (analyzes and processes)
scenes, stats = process_video_vision_full(
    video_path="video.mp4",
    scenes=[{"start_time": 0, "end_time": 5}, ...],
    transcript_segments=[...],
    config=config,
    cache_dir="/path/to/cache"
)

# Individual components
analyzer = TranscriptAnalyzer(config)
decision = analyzer.analyze_video_transcript(video_path, scenes, transcript_segments)

processor = VisionProcessor(config)
description = processor.describe_scene(video_path, scene, cache_dir)

FIX (v2.5): Properly handles both dict and SRTSegment objects using isinstance().
"""

import os
import json
import hashlib
import logging
import subprocess
import time
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Import unified cache (cache consolidation refactor - Jan 7, 2026)
from .cache import BaseCache, CacheEntry, compute_hash
from .downloader.utils import SUBPROCESS_FLAGS

logger = logging.getLogger(__name__)


@dataclass
class SceneAnalysis:
    """Analysis result for a scene.

    Represents the complete analysis of a single video scene, including
    transcript text, vision API description, and the hybrid combined result.

    Attributes:
        scene_index: Zero-based index of the scene in the video
        start_time: Start time in seconds
        end_time: End time in seconds
        transcript_text: Transcript text overlapping this scene
        transcript_word_count: Number of words in the transcript
        needs_vision: Whether vision API was needed for this scene
        reason: Explanation of why vision was/wasn't needed
        vision_description: Description from vision API (if used)
        combined_description: Hybrid: "{transcript} [Visual: {vision}]"

    Example:
        >>> analysis = SceneAnalysis(
        ...     scene_index=0,
        ...     start_time=0.0,
        ...     end_time=5.0,
        ...     transcript_text="Welcome to this tutorial",
        ...     transcript_word_count=4,
        ...     needs_vision=True,
        ...     reason="Low text coverage",
        ...     vision_description="A person standing in front of a whiteboard",
        ...     combined_description="Welcome to this tutorial [Visual: A person standing in front of a whiteboard]"
        ... )
    """
    scene_index: int
    start_time: float
    end_time: float
    transcript_text: str
    transcript_word_count: int
    needs_vision: bool
    reason: str
    vision_description: Optional[str] = None
    combined_description: Optional[str] = None


@dataclass
class VideoVisionDecision:
    """Decision about whether video needs vision processing.

    Contains the decision logic for whether to use the vision API for a video,
    based on transcript coverage analysis.

    Attributes:
        video_path: Path to the video file
        needs_vision: True if vision API should be called, False to skip
        reason: Human-readable explanation of the decision
        transcript_coverage: Ratio 0-1 of scenes with sufficient transcript text
        sparse_scenes: List of scene indices that have sparse transcript coverage
        total_scenes: Total number of scenes in the video

    Decision Logic:
        - needs_vision = True if:
          - transcript_coverage < coverage_threshold (default 0.30), OR
          - Any scene has fewer than min_words_per_scene (default 5)

        - needs_vision = False if:
          - transcript_coverage >= coverage_threshold AND
          - All scenes have >= min_words_per_scene

    Example:
        >>> decision = VideoVisionDecision(
        ...     video_path="/path/to/video.mp4",
        ...     needs_vision=True,
        ...     reason="Low transcript coverage (20%)",
        ...     transcript_coverage=0.20,
        ...     sparse_scenes=[0, 3, 7, 12],
        ...     total_scenes=15
        ... )
    """
    video_path: str
    needs_vision: bool
    reason: str
    transcript_coverage: float  # 0-1, how much of video has transcript
    sparse_scenes: List[int]    # Scene indices that need vision
    total_scenes: int
    

class TranscriptAnalyzer:
    """Analyzes transcripts to determine if vision API is needed.

    This class implements the first stage of the vision processing pipeline:
    analyzing the transcript to determine which scenes need vision API
    augmentation versus which are adequately described by transcript alone.

    Configuration (from config.vision):
        min_words_per_scene: Minimum words required to consider a scene "covered"
            by transcript. Scenes with fewer words are marked as sparse.
            Default: 5
        coverage_threshold: Ratio (0-1) of scenes that must have adequate
            transcript coverage to skip vision API entirely.
            Default: 0.30 (30%)

    Example:
        >>> analyzer = TranscriptAnalyzer(config)
        >>> decision = analyzer.analyze_video_transcript(
        ...     video_path="video.mp4",
        ...     scenes=[{"start_time": 0, "end_time": 5}, ...],
        ...     transcript_segments=[...]
        ... )
        >>> print(decision.needs_vision)  # True if vision needed
    """

    def __init__(self, config: Any):
        """Initialize TranscriptAnalyzer with config.

        Args:
            config: Configuration object with vision settings
        """
        self.config = config
        self.min_words_per_scene = getattr(config.vision, 'min_words_per_scene', 5)
        self.coverage_threshold = getattr(config.vision, 'coverage_threshold', 0.3)
    
    def analyze_video_transcript(
        self,
        video_path: str,
        scenes: List[dict],
        transcript_segments: List[Any]
    ) -> VideoVisionDecision:
        """Analyze if a video needs vision processing based on transcript.

        This is the main entry point for transcript analysis. It calculates
        transcript coverage for each scene and makes a go/no-go decision
        for vision API usage.

        The method:
        1. Iterates through each scene
        2. Finds all transcript segments that overlap with the scene
        3. Counts words in overlapping transcript
        4. Marks scenes with < min_words_per_scene as "sparse"
        5. Returns VideoVisionDecision with needs_vision flag

        Args:
            video_path: Path to the video file
            scenes: List of scene dicts with 'start_time' and 'end_time'
            transcript_segments: List of transcript segments (SRTSegment objects
                or dicts with 'start_time', 'end_time', 'text')

        Returns:
            VideoVisionDecision with:
                - needs_vision: bool - whether to call vision API
                - reason: str - explanation of decision
                - transcript_coverage: float - ratio 0-1 of covered scenes
                - sparse_scenes: List[int] - indices of scenes needing vision
                - total_scenes: int - total scene count

        Example:
            >>> scenes = [
            ...     {"start_time": 0, "end_time": 5},
            ...     {"start_time": 5, "end_time": 10},
            ...     {"start_time": 10, "end_time": 15}
            ... ]
            >>> segments = [
            ...     {"start_time": 0, "end_time": 5, "text": "Welcome to the tutorial"},
            ...     {"start_time": 10, "end_time": 15, "text": "Let's continue"}
            ... ]
            >>> decision = analyzer.analyze_video_transcript("video.mp4", scenes, segments)
            >>> decision.transcript_coverage  # 0.67 (2/3 scenes have text)
        """
        if not scenes:
            return VideoVisionDecision(
                video_path=video_path,
                needs_vision=True,
                reason="No scenes detected",
                transcript_coverage=0.0,
                sparse_scenes=[],
                total_scenes=0
            )
        
        sparse_scenes = []
        total_words = 0
        scenes_with_text = 0
        
        for i, scene in enumerate(scenes):
            scene_start = scene.get('start_time', 0)
            scene_end = scene.get('end_time', scene_start + 5)
            
            # Get transcript text for this scene
            scene_text = []
            for seg in transcript_segments:
                # Handle both dict and SRTSegment objects
                if isinstance(seg, dict):
                    seg_start = seg.get('start_time', 0)
                    seg_end = seg.get('end_time', 0)
                    seg_text = seg.get('text', '')
                else:
                    seg_start = getattr(seg, 'start_time', 0)
                    seg_end = getattr(seg, 'end_time', 0)
                    seg_text = getattr(seg, 'text', '')
                
                # Check overlap
                if seg_end > scene_start and seg_start < scene_end:
                    scene_text.append(seg_text)
            
            combined_text = ' '.join(scene_text).strip()
            word_count = len(combined_text.split()) if combined_text else 0
            total_words += word_count
            
            if word_count >= self.min_words_per_scene:
                scenes_with_text += 1
            else:
                sparse_scenes.append(i)
        
        coverage = scenes_with_text / len(scenes) if scenes else 0
        
        needs_vision = coverage < self.coverage_threshold or len(sparse_scenes) > 0
        
        if not needs_vision:
            reason = f"Good transcript coverage ({coverage:.0%})"
        elif coverage < self.coverage_threshold:
            reason = f"Low transcript coverage ({coverage:.0%})"
        else:
            reason = f"{len(sparse_scenes)} scenes with sparse text"
        
        return VideoVisionDecision(
            video_path=video_path,
            needs_vision=needs_vision,
            reason=reason,
            transcript_coverage=coverage,
            sparse_scenes=sparse_scenes,
            total_scenes=len(scenes)
        )
    
    def get_priority_scenes(
        self,
        scenes: List[dict],
        transcript_segments: List[Any],
        max_scenes: int = None
    ) -> List[int]:
        """Get indices of scenes that most need vision processing.

        Prioritizes scenes with the least transcript coverage, returning
        indices sorted by ascending word count. This ensures the most
        "visually underserved" scenes get processed first when API
        call limits apply.

        Args:
            scenes: List of scene dicts with 'start_time' and 'end_time'
            transcript_segments: List of transcript segments (SRTSegment or dict)
            max_scenes: Maximum scenes to return. Defaults to
                config.vision.max_scenes_per_video (50)

        Returns:
            List of scene indices sorted by priority (lowest word count first)

        Example:
            >>> scenes = [{"start_time": 0, "end_time": 5}, ...]
            >>> segments = [
            ...     {"start_time": 0, "end_time": 2, "text": "Hi"},
            ...     {"start_time": 2, "end_time": 5, "text": "Welcome to the tutorial on Python"}
            ... ]
            >>> priorities = analyzer.get_priority_scenes(scenes, segments, max_scenes=3)
            >>> priorities  # [0] - scene 0 has fewer words
        """
        if max_scenes is None:
            max_scenes = getattr(self.config.vision, 'max_scenes_per_video', 50)
        
        scene_scores = []
        
        for i, scene in enumerate(scenes):
            scene_start = scene.get('start_time', 0)
            scene_end = scene.get('end_time', scene_start + 5)
            
            # Calculate text coverage for this scene
            word_count = 0
            for seg in transcript_segments:
                # Handle both dict and SRTSegment objects
                if isinstance(seg, dict):
                    seg_start = seg.get('start_time', 0)
                    seg_end = seg.get('end_time', 0)
                    seg_text = seg.get('text', '')
                else:
                    seg_start = getattr(seg, 'start_time', 0)
                    seg_end = getattr(seg, 'end_time', 0)
                    seg_text = getattr(seg, 'text', '')
                
                if seg_end > scene_start and seg_start < scene_end:
                    word_count += len(seg_text.split())
            
            scene_scores.append((i, word_count))
        
        # Sort by word count (ascending - least text first)
        scene_scores.sort(key=lambda x: x[1])
        
        # Return indices of lowest-text scenes
        return [idx for idx, _ in scene_scores[:max_scenes]]


class VisionCache(BaseCache):
    """
    Cache for vision API scene descriptions.

    Caches descriptions keyed by video_path + start_time + end_time.
    """

    def _serialize_entry(self, entry: CacheEntry) -> dict:
        """Serialize vision entry"""
        return {
            'data': entry.data,
            'cached_at': entry.cached_at,
            'metadata': entry.metadata
        }

    def _deserialize_entry(self, data: dict) -> CacheEntry:
        """Deserialize vision entry"""
        return CacheEntry(
            data=data['data'],
            cached_at=data['cached_at'],
            key='',
            metadata=data.get('metadata', {})
        )


class VisionProcessor:
    """Processes video frames with vision API.

    This class handles the actual vision API calls for scene analysis.
    It extracts frames from video at specified timestamps and sends them
    to a vision-capable LLM (Gemini by default) for description.

    Configuration (from config.vision):
        provider: Vision API provider ('gemini' or 'openai')
            Default: 'gemini'
        model: Model identifier for the vision provider
            Default: 'gemini-2.5-flash'
        estimated_cost_per_call: Estimated cost per API call in USD
            Default: 0.001 ($0.001 per call)

    Attributes:
        api_calls: Total number of API calls made
        total_cost: Cumulative estimated cost in USD

    Example:
        >>> processor = VisionProcessor(config)
        >>> if processor.is_available():
        ...     description = processor.describe_scene(
        ...         "video.mp4",
        ...         {"start_time": 10, "end_time": 15},
        ...         cache_dir="/cache"
        ...     )
        >>> print(processor.get_stats())
        {'api_calls': 1, 'estimated_cost': 0.001}
    """

    def __init__(self, config: Any):
        """Initialize VisionProcessor with config.

        Args:
            config: Configuration object with vision settings
        """
        self.config = config
        self.provider = getattr(config.vision, 'provider', 'gemini')
        self.model = getattr(config.vision, 'model', 'gemini-2.5-flash')
        self.api_calls = 0
        self.total_cost = 0.0

        # Cost estimate per call (from config)
        self.cost_per_call = getattr(config.vision, 'estimated_cost_per_call', 0.001)
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key for vision provider"""
        if self.provider == 'gemini':
            return os.getenv('GEMINI_API_KEY')
        elif self.provider == 'openai':
            return os.getenv('OPENAI_API_KEY')
        return None

    def is_available(self) -> bool:
        """Check if vision API is available (API key is configured).

        Checks for the presence of required API keys in environment:
        - GEMINI_API_KEY for 'gemini' provider
        - OPENAI_API_KEY for 'openai' provider

        Returns:
            True if API key is configured and provider is available
            False if no API key found

        Example:
            >>> processor = VisionProcessor(config)
            >>> if processor.is_available():
            ...     print("Vision API ready!")
        """
        return self._get_api_key() is not None
    
    def _extract_frame(self, video_path: str, timestamp: float) -> Optional[bytes]:
        """Extract a frame from video at given timestamp"""
        import subprocess
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                temp_path = f.name
            
            cmd = [
                'ffmpeg', '-loglevel', 'error',
                '-ss', str(timestamp),
                '-i', video_path,
                '-vframes', '1',
                '-y', temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30, **SUBPROCESS_FLAGS)
            
            if result.returncode == 0 and Path(temp_path).exists():
                with open(temp_path, 'rb') as f:
                    frame_data = f.read()
                Path(temp_path).unlink()
                return frame_data
                
        except Exception as e:
            logger.debug(f"Frame extraction error: {e}")
        
        return None
    
    def _describe_frame_gemini(self, frame_data: bytes) -> Optional[str]:
        """Describe frame using Gemini Vision"""
        try:
            from src.llm_client import create_client, LLMRequest, ResponseFormat

            api_key = self._get_api_key()
            if not api_key:
                return None

            client = create_client("gemini", api_key=api_key, model=self.model)

            prompt = "Describe this video frame in 2-3 sentences. Focus on: main subjects, actions, setting, mood. Be specific and concise."

            request = LLMRequest(
                prompt=prompt,
                response_format=ResponseFormat.TEXT,
                images=[frame_data],
                image_format="jpeg",
                cache_key_prefix="vision"
            )

            response = client.generate(request)

            self.api_calls += 1
            self.total_cost += self.cost_per_call  # Use config value

            return response.text.strip()

        except Exception as e:
            logger.error(f"Gemini vision error: {e}")
            return None
    
    def describe_scene(
        self,
        video_path: str,
        scene: dict,
        cache_dir: str = None
    ) -> Optional[str]:
        """Get vision description for a scene.

        Extracts a frame from the video at the scene's midpoint and
        sends it to the vision API for description.

        Process:
        1. Calculate midpoint timestamp of scene
        2. Check cache for existing description
        3. If not cached: extract frame with ffmpeg
        4. Send frame to vision API (Gemini)
        5. Cache result if cache_dir provided

        Args:
            video_path: Path to video file
            scene: Scene dict with 'start_time' and 'end_time'
            cache_dir: Optional directory for caching results

        Returns:
            2-3 sentence description of the frame, or None if failed

        Example:
            >>> description = processor.describe_scene(
            ...     "video.mp4",
            ...     {"start_time": 10, "end_time": 15},
            ...     cache_dir="/path/to/cache"
            ... )
            >>> # Returns: "A person standing at a whiteboard explaining
            ... #          concepts. The whiteboard has diagrams and text.
            ... #          The setting appears to be a classroom."
        """
        start_time = scene.get('start_time', 0)
        end_time = scene.get('end_time', start_time + 5)
        mid_time = (start_time + end_time) / 2

        # Check cache using VisionCache (cache consolidation refactor - Jan 7, 2026)
        vision_cache = None
        cache_key = None
        if cache_dir:
            vision_cache = VisionCache(
                cache_dir=Path(cache_dir) / "vision_cache",
                index_name="vision_index.json"
            )
            cache_key = compute_hash(f"{Path(video_path).stem}_{start_time:.1f}_{end_time:.1f}", length=12)

            cached_entry = vision_cache.get(cache_key)
            if cached_entry:
                return cached_entry.data.get('description')

        # Extract and describe frame
        frame_data = self._extract_frame(video_path, mid_time)
        if not frame_data:
            return None

        description = self._describe_frame_gemini(frame_data)

        # Cache result
        if description and vision_cache and cache_key:
            vision_cache.set(cache_key, {
                'video': video_path,
                'start': start_time,
                'end': end_time,
                'description': description
            })

        return description
    
    def get_stats(self) -> dict:
        """Get processing statistics.

        Returns cumulative statistics for all vision API calls made
        by this processor instance.

        Returns:
            dict with:
                - api_calls: Total number of API calls made
                - estimated_cost: Total estimated cost in USD

        Example:
            >>> processor.describe_scene("video1.mp4", scene1)
            >>> processor.describe_scene("video2.mp4", scene2)
            >>> processor.get_stats()
            {'api_calls': 2, 'estimated_cost': 0.002}
        """
        return {
            'api_calls': self.api_calls,
            'estimated_cost': self.total_cost
        }


def process_video_vision(
    video_path: str,
    transcript_segments: List[Any],
    cache: Any,
    config: Any,
    max_scenes_per_video: int = None
) -> List[dict]:
    """
    Process a video with selective vision API calls.
    
    Args:
        video_path: Path to video file
        transcript_segments: List of transcript segments (SRTSegment or dict)
        cache: CacheManager or similar with cache_dir attribute
        config: Configuration object
        max_scenes_per_video: Maximum scenes to process (defaults to config.vision.max_scenes_per_video)
    
    Returns:
        List of scene dicts with descriptions (or empty list if skipped)
    """
    if not getattr(config.vision, 'enabled', False):
        return []
    
    # Get max_scenes from config if not provided
    if max_scenes_per_video is None:
        max_scenes_per_video = getattr(config.vision, 'max_scenes_per_video', 50)
    
    # Get cache directory
    if hasattr(cache, 'cache_dir'):
        cache_dir = cache.cache_dir
    else:
        cache_dir = str(cache) if cache else None
    
    # For now, return transcript segments as "scenes" with combined descriptions
    # This is a simplified version - full implementation would do actual vision API calls
    scenes = []
    
    # Group transcript segments into pseudo-scenes
    for i, seg in enumerate(transcript_segments[:max_scenes_per_video]):
        if isinstance(seg, dict):
            start_time = seg.get('start_time', 0)
            end_time = seg.get('end_time', 0)
            text = seg.get('text', '')
        else:
            start_time = getattr(seg, 'start_time', 0)
            end_time = getattr(seg, 'end_time', 0)
            text = getattr(seg, 'text', '')
        
        if text.strip():
            scenes.append({
                'index': i,
                'start_time': start_time,
                'end_time': end_time,
                'description': text,
                'source': video_path
            })
    
    return scenes


def process_video_vision_full(
    video_path: str,
    scenes: List[dict],
    transcript_segments: List[Any],
    config: Any,
    cache_dir: str = None
) -> Tuple[List[SceneAnalysis], dict]:
    """
    Full vision processing with selective API calls (original implementation).
    
    Args:
        video_path: Path to video file
        scenes: List of scene dicts with start_time, end_time
        transcript_segments: List of transcript segments (SRTSegment or dict)
        config: Configuration object
        cache_dir: Cache directory for vision results
    
    Returns:
        (list of SceneAnalysis, stats dict)
    """
    if not getattr(config.vision, 'enabled', False):
        return [], {'skipped': True, 'reason': 'Vision disabled'}
    
    analyzer = TranscriptAnalyzer(config)
    processor = VisionProcessor(config)
    
    # Analyze transcript coverage
    decision = analyzer.analyze_video_transcript(
        video_path, scenes, transcript_segments
    )
    
    if not decision.needs_vision:
        return [], {
            'skipped': True,
            'reason': decision.reason,
            'coverage': decision.transcript_coverage
        }
    
    # Get priority scenes
    video_duration = scenes[-1].get('end_time', 60) if scenes else 60
    max_scenes = getattr(config.vision, 'max_scenes_per_video', 5)
    
    priority_scenes = analyzer.get_priority_scenes(
        scenes, transcript_segments, max_scenes
    )
    
    # Process priority scenes
    results = []
    for scene_idx in priority_scenes:
        if scene_idx >= len(scenes):
            continue
        
        scene = scenes[scene_idx]
        
        # Get vision description
        description = processor.describe_scene(
            video_path, scene, cache_dir
        )
        
        # Get transcript for this scene
        scene_start = scene.get('start_time', 0)
        scene_end = scene.get('end_time', scene_start + 5)
        
        transcript_text = []
        for seg in transcript_segments:
            # Handle both dict and SRTSegment objects
            if isinstance(seg, dict):
                seg_start = seg.get('start_time', 0)
                seg_end = seg.get('end_time', 0)
                seg_text = seg.get('text', '')
            else:
                seg_start = getattr(seg, 'start_time', 0)
                seg_end = getattr(seg, 'end_time', 0)
                seg_text = getattr(seg, 'text', '')
            
            if seg_end > scene_start and seg_start < scene_end:
                transcript_text.append(seg_text)
        
        combined_transcript = ' '.join(transcript_text).strip()
        
        # Combine transcript and vision
        if description and combined_transcript:
            combined = f"{combined_transcript} [Visual: {description}]"
        elif description:
            combined = f"[Visual: {description}]"
        else:
            combined = combined_transcript
        
        results.append(SceneAnalysis(
            scene_index=scene_idx,
            start_time=scene_start,
            end_time=scene_end,
            transcript_text=combined_transcript,
            transcript_word_count=len(combined_transcript.split()),
            needs_vision=True,
            reason="Low text coverage",
            vision_description=description,
            combined_description=combined
        ))
    
    stats = processor.get_stats()
    stats['scenes_processed'] = len(results)
    stats['coverage'] = decision.transcript_coverage
    
    return results, stats


def get_scene_text(scene: dict, transcript_segments: List[Any]) -> str:
    """
    Get transcript text for a scene.
    Helper function that handles both dict and SRTSegment objects.
    """
    scene_start = scene.get('start_time', 0)
    scene_end = scene.get('end_time', scene_start + 5)
    
    texts = []
    for seg in transcript_segments:
        # Handle both dict and SRTSegment objects
        if isinstance(seg, dict):
            seg_start = seg.get('start_time', 0)
            seg_end = seg.get('end_time', 0)
            seg_text = seg.get('text', '')
        else:
            seg_start = getattr(seg, 'start_time', 0)
            seg_end = getattr(seg, 'end_time', 0)
            seg_text = getattr(seg, 'text', '')
        
        # Check overlap
        if seg_end > scene_start and seg_start < scene_end:
            texts.append(seg_text)
    
    return ' '.join(texts).strip()
