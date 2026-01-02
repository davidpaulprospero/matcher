#!/usr/bin/env python3
"""
Vision Processing Module - v2.5 (SRTSegment Fix)

Selective Vision Processing:
1. LLM PRE-FILTER: Analyzes transcript to determine if vision API is needed.
   - If transcript clearly describes visuals → skip vision API
   - If transcript is sparse/silent → use vision API for those scenes only

2. SCENE PRIORITIZATION: Only processes ambiguous/low-text scenes.

3. COST TRACKING: Logs estimated API costs for vision calls.

4. HYBRID DESCRIPTIONS: Combines transcript + vision for richer metadata.

FIX (v2.5): Properly handles both dict and SRTSegment objects using isinstance().
"""

import os
import json
import hashlib
import logging
import time
import base64
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class SceneAnalysis:
    """Analysis result for a scene"""
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
    """Decision about whether video needs vision processing"""
    video_path: str
    needs_vision: bool
    reason: str
    transcript_coverage: float  # 0-1, how much of video has transcript
    sparse_scenes: List[int]    # Scene indices that need vision
    total_scenes: int
    

class TranscriptAnalyzer:
    """Analyzes transcripts to determine if vision API is needed"""
    
    def __init__(self, config: Any):
        self.config = config
        self.min_words_per_scene = getattr(config.vision, 'min_words_per_scene', 5)
        self.coverage_threshold = getattr(config.vision, 'coverage_threshold', 0.3)
    
    def analyze_video_transcript(
        self,
        video_path: str,
        scenes: List[dict],
        transcript_segments: List[Any]
    ) -> VideoVisionDecision:
        """
        Analyze if a video needs vision processing based on transcript.
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
        """
        Get indices of scenes that most need vision processing.
        Prioritizes scenes with least transcript coverage.
        
        Args:
            scenes: List of scene dicts
            transcript_segments: List of transcript segments
            max_scenes: Maximum scenes to return (defaults to config.vision.max_scenes_per_video)
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


class VisionProcessor:
    """Processes video frames with vision API"""
    
    def __init__(self, config: Any):
        self.config = config
        self.provider = getattr(config.vision, 'provider', 'gemini')
        self.model = getattr(config.vision, 'model', 'gemini-2.0-flash')
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
        """Check if vision API is available (API key is configured)"""
        return self._get_api_key() is not None
    
    def _extract_frame(self, video_path: str, timestamp: float) -> Optional[bytes]:
        """Extract a frame from video at given timestamp"""
        import subprocess
        import tempfile
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                temp_path = f.name
            
            cmd = [
                'ffmpeg', '-ss', str(timestamp),
                '-i', video_path,
                '-vframes', '1',
                '-y', temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
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
            import google.generativeai as genai
            
            api_key = self._get_api_key()
            if not api_key:
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.model)
            
            # Encode frame as base64
            frame_b64 = base64.b64encode(frame_data).decode('utf-8')
            
            response = model.generate_content([
                "Describe this video frame in 2-3 sentences. Focus on: main subjects, actions, setting, mood. Be specific and concise.",
                {"mime_type": "image/jpeg", "data": frame_b64}
            ])
            
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
        """Get vision description for a scene"""
        start_time = scene.get('start_time', 0)
        end_time = scene.get('end_time', start_time + 5)
        mid_time = (start_time + end_time) / 2
        
        # Check cache
        if cache_dir:
            cache_key = f"{Path(video_path).stem}_{start_time:.1f}_{end_time:.1f}"
            cache_file = Path(cache_dir) / "vision_cache" / f"{hashlib.md5(cache_key.encode()).hexdigest()[:12]}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        return data.get('description')
                except:
                    pass
        
        # Extract and describe frame
        frame_data = self._extract_frame(video_path, mid_time)
        if not frame_data:
            return None
        
        description = self._describe_frame_gemini(frame_data)
        
        # Cache result
        if description and cache_dir:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'video': video_path,
                        'start': start_time,
                        'end': end_time,
                        'description': description,
                        'cached_at': time.time()
                    }, f)
            except:
                pass
        
        return description
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
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
