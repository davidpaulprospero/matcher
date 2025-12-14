"""
Vision module for multi-modal matching and scene descriptions
"""

import os
import subprocess
import base64
import logging
from pathlib import Path
from typing import List, Optional, Tuple
import json

from .config import Config
from .utils import SceneInfo, SRTSegment, CacheManager, ProgressBar

logger = logging.getLogger(__name__)


def extract_keyframes(
    video_path: str,
    scenes: List[Tuple[float, float]],  # List of (start_time, end_time)
    cache: CacheManager,
    video_hash: str,
    frames_per_scene: int = 3
) -> List[List[str]]:
    """
    Extract keyframes from video scenes.
    Returns list of keyframe paths for each scene.
    """
    keyframes_dir = cache.keyframes_dir / video_hash
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    
    all_keyframes = []
    
    for scene_idx, (start_time, end_time) in enumerate(scenes):
        scene_keyframes = []
        duration = end_time - start_time
        
        # Calculate frame times (evenly distributed)
        if frames_per_scene == 1:
            times = [start_time + duration / 2]
        else:
            times = [
                start_time + (duration * i / (frames_per_scene - 1))
                for i in range(frames_per_scene)
            ]
        
        for frame_idx, time in enumerate(times):
            output_path = keyframes_dir / f"scene_{scene_idx:04d}_frame_{frame_idx:02d}.jpg"
            
            if not output_path.exists():
                # Extract frame using ffmpeg
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss", str(time),
                    "-i", video_path,
                    "-vframes", "1",
                    "-q:v", "2",
                    str(output_path)
                ]
                
                try:
                    subprocess.run(cmd, capture_output=True, timeout=30)
                except Exception as e:
                    logger.warning(f"Failed to extract keyframe at {time}s: {e}")
                    continue
            
            if output_path.exists():
                scene_keyframes.append(str(output_path))
        
        all_keyframes.append(scene_keyframes)
    
    return all_keyframes


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def describe_scene_gemini(
    keyframe_paths: List[str],
    transcript_text: str,
    api_key: str
) -> Tuple[str, List[str]]:
    """
    Use Gemini Vision to describe a scene.
    Returns (description, visual_keywords)
    """
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Prepare images
    image_parts = []
    for kf_path in keyframe_paths[:3]:  # Max 3 frames
        try:
            image_data = image_to_base64(kf_path)
            image_parts.append({
                "mime_type": "image/jpeg",
                "data": image_data
            })
        except Exception as e:
            logger.warning(f"Failed to load keyframe {kf_path}: {e}")
    
    if not image_parts:
        return "", []
    
    prompt = f"""Analyze these video frames and provide:
1. A brief description of what's shown (1-2 sentences)
2. 5-10 keywords describing the visual content

{"Transcript during this scene: " + transcript_text if transcript_text else ""}

Respond in JSON format:
{{"description": "...", "keywords": ["keyword1", "keyword2", ...]}}"""

    try:
        response = model.generate_content([prompt] + image_parts)
        
        # Parse JSON response
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('description', ''), data.get('keywords', [])
    except Exception as e:
        logger.warning(f"Gemini vision failed: {e}")
    
    return "", []


def describe_scene_openai(
    keyframe_paths: List[str],
    transcript_text: str,
    api_key: str
) -> Tuple[str, List[str]]:
    """
    Use OpenAI Vision to describe a scene.
    Returns (description, visual_keywords)
    """
    from openai import OpenAI
    
    client = OpenAI(api_key=api_key)
    
    # Prepare images
    image_content = []
    for kf_path in keyframe_paths[:3]:
        try:
            image_data = image_to_base64(kf_path)
            image_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_data}",
                    "detail": "low"
                }
            })
        except Exception as e:
            logger.warning(f"Failed to load keyframe {kf_path}: {e}")
    
    if not image_content:
        return "", []
    
    prompt = f"""Analyze these video frames and provide:
1. A brief description of what's shown (1-2 sentences)
2. 5-10 keywords describing the visual content

{"Transcript during this scene: " + transcript_text if transcript_text else ""}

Respond in JSON format:
{{"description": "...", "keywords": ["keyword1", "keyword2", ...]}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *image_content
                    ]
                }
            ],
            max_tokens=300
        )
        
        text = response.choices[0].message.content
        
        import re
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return data.get('description', ''), data.get('keywords', [])
    except Exception as e:
        logger.warning(f"OpenAI vision failed: {e}")
    
    return "", []


def detect_scenes_from_otio(otio_path: str) -> List[Tuple[float, float]]:
    """
    Extract scene cuts from existing OTIO file.
    Returns list of (start_time, end_time) tuples.
    """
    import opentimelineio as otio
    
    scenes = []
    
    try:
        timeline = otio.adapters.read_from_file(otio_path)
        
        for track in timeline.tracks:
            if track.kind == otio.schema.TrackKind.Video:
                current_time = 0.0
                for clip in track:
                    if hasattr(clip, 'source_range') and clip.source_range:
                        duration = clip.source_range.duration.to_seconds()
                        start = clip.source_range.start_time.to_seconds()
                        scenes.append((start, start + duration))
                        current_time += duration
    except Exception as e:
        logger.warning(f"Failed to parse OTIO {otio_path}: {e}")
    
    return scenes


def detect_scenes_pyscenedetect(video_path: str) -> List[Tuple[float, float]]:
    """
    Detect scenes using PySceneDetect.
    Returns list of (start_time, end_time) tuples.
    """
    try:
        from scenedetect import detect, ContentDetector
        
        scene_list = detect(video_path, ContentDetector())
        
        scenes = []
        for scene in scene_list:
            start = scene[0].get_seconds()
            end = scene[1].get_seconds()
            scenes.append((start, end))
        
        return scenes
    except ImportError:
        logger.warning("PySceneDetect not installed. Using fixed intervals.")
        return []
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
        return []


def process_video_vision(
    video_path: str,
    transcript_segments: List[SRTSegment],
    cache: CacheManager,
    config: Config,
    otio_path: Optional[str] = None
) -> List[SceneInfo]:
    """
    Process a video with vision analysis.
    Returns list of SceneInfo objects.
    """
    video_hash = cache.get_file_hash(video_path)
    
    # Check cache
    cached_scenes = cache.get_scenes(video_hash)
    if cached_scenes:
        logger.debug(f"Using cached scenes for {Path(video_path).name}")
        return cached_scenes
    
    # Get scene boundaries
    if otio_path and Path(otio_path).exists():
        scenes_times = detect_scenes_from_otio(otio_path)
        logger.debug(f"Found {len(scenes_times)} scenes from OTIO")
    else:
        scenes_times = detect_scenes_pyscenedetect(video_path)
        logger.debug(f"Detected {len(scenes_times)} scenes with PySceneDetect")
    
    # Fallback: split into fixed intervals
    if not scenes_times:
        from .transcription import get_video_duration
        duration = get_video_duration(video_path)
        interval = 15.0  # 15 second intervals (less granular = fewer API calls)
        scenes_times = [
            (i * interval, min((i + 1) * interval, duration))
            for i in range(int(duration / interval) + 1)
        ]
    
    # Extract keyframes
    if config.vision.enabled:
        keyframes = extract_keyframes(
            video_path, scenes_times, cache, video_hash,
            config.vision.frames_per_scene
        )
    else:
        keyframes = [[] for _ in scenes_times]
    
    # Create SceneInfo objects
    scenes = []
    for scene_idx, ((start_time, end_time), scene_keyframes) in enumerate(zip(scenes_times, keyframes)):
        # Find transcript segment for this scene
        transcript_segment = None
        transcript_text = ""
        for seg in transcript_segments:
            # Check for overlap
            if seg.start_time < end_time and seg.end_time > start_time:
                transcript_segment = seg
                transcript_text = seg.text
                break
        
        scene = SceneInfo(
            video_path=video_path,
            scene_index=scene_idx,
            start_time=start_time,
            end_time=end_time,
            keyframes=scene_keyframes,
            transcript_segment=transcript_segment
        )
        
        scenes.append(scene)
    
    # Generate scene descriptions (if enabled and API available)
    if config.vision.enabled and config.vision.describe_scenes:
        describe_scenes(scenes, transcript_segments, config)
    
    # Cache scenes
    cache.save_scenes(video_hash, scenes)
    
    return scenes


def describe_scenes(
    scenes: List[SceneInfo],
    transcript_segments: List[SRTSegment],
    config: Config
):
    """Add descriptions to scenes using vision API - BATCHED for efficiency"""
    
    if not scenes:
        return
    
    # Filter scenes with keyframes
    scenes_with_keyframes = [s for s in scenes if s.keyframes]
    
    if not scenes_with_keyframes:
        logger.warning("No keyframes available for scene description")
        return
    
    # COST OPTIMIZATION: Sample scenes instead of describing all
    # Default: describe max 10 scenes per video (evenly distributed)
    max_scenes = getattr(config.vision, 'max_scenes_per_video', 10)
    
    if len(scenes_with_keyframes) > max_scenes:
        # Sample evenly distributed scenes
        step = len(scenes_with_keyframes) / max_scenes
        sampled_indices = [int(i * step) for i in range(max_scenes)]
        scenes_to_describe = [scenes_with_keyframes[i] for i in sampled_indices]
        logger.info(f"Sampling {len(scenes_to_describe)} of {len(scenes_with_keyframes)} scenes for vision API")
    else:
        scenes_to_describe = scenes_with_keyframes
    
    # BATCH PROCESSING: Send multiple scenes in one API call
    batch_size = 5  # 5 scenes per API call
    
    logger.info(f"Describing {len(scenes_to_describe)} scenes with vision API (batched)...")
    
    num_batches = (len(scenes_to_describe) + batch_size - 1) // batch_size
    progress = ProgressBar(num_batches, "Describing scenes")
    
    for batch_start in range(0, len(scenes_to_describe), batch_size):
        batch = scenes_to_describe[batch_start:batch_start + batch_size]
        
        if config.vision.provider == "gemini" and config.gemini_api_key:
            results = describe_scenes_batch_gemini(batch, config.gemini_api_key)
        elif config.vision.provider == "openai" and config.openai_api_key:
            # Fallback to individual calls for OpenAI
            results = []
            for scene in batch:
                transcript_text = scene.transcript_segment.text if scene.transcript_segment else ""
                desc, kw = describe_scene_openai(scene.keyframes, transcript_text, config.openai_api_key)
                results.append((desc, kw))
        else:
            results = [("", []) for _ in batch]
        
        # Apply results to scenes
        for scene, (description, keywords) in zip(batch, results):
            scene.description = description
            scene.visual_keywords = keywords
        
        progress.update(1)
    
    progress.close()
    
    # Propagate descriptions to neighboring scenes (fill gaps)
    _propagate_descriptions(scenes, scenes_to_describe)


def describe_scenes_batch_gemini(
    scenes: List[SceneInfo],
    api_key: str
) -> List[Tuple[str, List[str]]]:
    """
    Describe multiple scenes in a single API call.
    Returns list of (description, keywords) tuples.
    """
    import google.generativeai as genai
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    # Build multi-image prompt
    content_parts = []
    
    prompt_text = f"""Analyze these {len(scenes)} video scenes. For EACH scene (numbered), provide:
1. Brief description (1 sentence)
2. 3-5 visual keywords

"""
    
    for i, scene in enumerate(scenes):
        prompt_text += f"SCENE {i+1} (at {scene.start_time:.1f}s):\n"
        
        # Add first keyframe from each scene
        if scene.keyframes:
            try:
                image_data = image_to_base64(scene.keyframes[0])
                content_parts.append({
                    "mime_type": "image/jpeg",
                    "data": image_data
                })
                prompt_text += f"[Image {i+1}]\n"
            except:
                prompt_text += "[No image]\n"
        
        if scene.transcript_segment:
            prompt_text += f"Audio: \"{scene.transcript_segment.text[:50]}...\"\n"
        prompt_text += "\n"
    
    prompt_text += """Respond in JSON format:
{"scenes": [
  {"scene": 1, "description": "...", "keywords": ["kw1", "kw2", ...]},
  {"scene": 2, "description": "...", "keywords": ["kw1", "kw2", ...]},
  ...
]}"""
    
    content_parts.insert(0, prompt_text)
    
    try:
        response = model.generate_content(content_parts)
        
        import re
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            results = []
            for scene_data in data.get('scenes', []):
                results.append((
                    scene_data.get('description', ''),
                    scene_data.get('keywords', [])
                ))
            
            # Pad if we got fewer results
            while len(results) < len(scenes):
                results.append(("", []))
            
            return results[:len(scenes)]
    
    except Exception as e:
        logger.warning(f"Gemini batch vision failed: {e}")
    
    return [("", []) for _ in scenes]


def _propagate_descriptions(all_scenes: List[SceneInfo], described_scenes: List[SceneInfo]):
    """
    Propagate descriptions from sampled scenes to nearby scenes.
    This fills gaps without extra API calls.
    """
    if not described_scenes:
        return
    
    # Create a mapping of described scene indices
    described_indices = {s.scene_index for s in described_scenes}
    
    for scene in all_scenes:
        if scene.scene_index in described_indices:
            continue  # Already has description
        
        # Find nearest described scene
        nearest = min(
            described_scenes,
            key=lambda s: abs(s.scene_index - scene.scene_index)
        )
        
        # Copy description if within 3 scenes
        if abs(nearest.scene_index - scene.scene_index) <= 3:
            scene.description = nearest.description
            scene.visual_keywords = nearest.visual_keywords.copy()
