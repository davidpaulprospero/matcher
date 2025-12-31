"""
Advanced matching module with:
- Tiered LLM matching
- Contextual matching
- Negative matching
- LLM response caching
- Local LLM support
- Smart reuse prevention
- Face detection preference
- Timeline variety enforcement
"""

import logging
import json
import re
import hashlib
from typing import List, Optional, Tuple, Dict, Any, Set
from pathlib import Path
from abc import ABC, abstractmethod
from collections import defaultdict

from .config import Config, get_config
from .utils import (
    SRTSegment, SceneInfo, Match, AlternativeMatch, StrategyMatch, MatchResult,
    CacheManager, ReuseTracker, ProgressBar
)
from .embeddings import find_top_k_similar, cosine_similarity
from .keyword_extractor import find_keyword_matches

logger = logging.getLogger(__name__)


# =============================================================================
# TIMELINE VARIETY ENFORCEMENT
# =============================================================================

class TimelineVarietyTracker:
    """
    Tracks source video usage per track to enforce timeline variety.
    
    Prevents the same source video from appearing more than max_repeats times
    within a timeline_window (e.g., 10 minutes).
    
    This solves the problem of one video dominating 90% of a 30-minute timeline.
    """
    
    def __init__(self, timeline_window: float = 600.0, max_repeats: int = 1):
        """
        Args:
            timeline_window: Time window in seconds (default 600 = 10 minutes)
            max_repeats: Max times same source can appear in window (default 1)
        """
        self.timeline_window = timeline_window
        self.max_repeats = max_repeats
        
        # Track source usage per track: {track_name: [(source_file, timeline_position), ...]}
        self.track_usage: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    
    def get_excluded_sources(self, track: str, current_timeline_pos: float) -> Set[str]:
        """
        Get set of source files that should be excluded for this track at this position.
        
        Args:
            track: Track name (e.g., "V1", "V2", etc.)
            current_timeline_pos: Current position in timeline (seconds from start)
        
        Returns:
            Set of source file paths that should be excluded
        """
        excluded = set()
        window_start = current_timeline_pos - self.timeline_window
        
        # Count occurrences of each source within the window
        source_counts: Dict[str, int] = defaultdict(int)
        
        for source_file, pos in self.track_usage[track]:
            if pos >= window_start and pos < current_timeline_pos:
                source_counts[source_file] += 1
        
        # Exclude sources that have reached max_repeats
        for source_file, count in source_counts.items():
            if count >= self.max_repeats:
                excluded.add(source_file)
        
        return excluded
    
    def record_usage(self, track: str, source_file: str, timeline_pos: float):
        """
        Record that a source file was used on a track at a timeline position.
        
        Args:
            track: Track name (e.g., "V1", "V2", etc.)
            source_file: Path to the source video file
            timeline_pos: Position in timeline where this clip starts (seconds)
        """
        self.track_usage[track].append((source_file, timeline_pos))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about source usage per track."""
        stats = {}
        for track, usages in self.track_usage.items():
            source_counts = defaultdict(int)
            for source_file, _ in usages:
                source_counts[Path(source_file).name] += 1
            
            stats[track] = {
                "total_clips": len(usages),
                "unique_sources": len(source_counts),
                "top_sources": sorted(source_counts.items(), key=lambda x: -x[1])[:5]
            }
        return stats

class FaceDetector:
    """
    Detect faces in video files using MediaPipe (preferred) or OpenCV (fallback).
    Caches results per video file for performance.
    
    MediaPipe is faster and more accurate than OpenCV Haar cascades.
    """
    
    _instance = None
    _cache: Dict[str, float] = {}  # video_path -> face_score (0-1)
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


def apply_face_preference(
    candidates: List[Tuple[SRTSegment, float]],
    face_preference: str,
    cache_dir: str = None
) -> List[Tuple[SRTSegment, float]]:
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


# =============================================================================
# ROBUST JSON PARSING
# =============================================================================

def repair_json(text: str) -> str:
    """
    Attempt to repair common JSON issues from LLM responses.
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Remove any text before the first [ or {
    match = re.search(r'[\[\{]', text)
    if match:
        text = text[match.start():]
    
    # Remove any text after the last ] or }
    for i in range(len(text) - 1, -1, -1):
        if text[i] in ']}':
            text = text[:i + 1]
            break
    
    # Fix trailing commas before ] or }
    text = re.sub(r',\s*([}\]])', r'\1', text)
    
    # Fix missing commas between objects
    text = re.sub(r'}\s*{', '},{', text)
    
    # Fix unescaped quotes inside strings (common in "reason" field)
    # This is tricky - we try to fix obvious cases
    def fix_quotes_in_value(match):
        key = match.group(1)
        value = match.group(2)
        # Escape any unescaped quotes inside the value
        # But not the surrounding quotes
        fixed = value.replace('\\"', '<<ESCAPED>>').replace('"', '\\"').replace('<<ESCAPED>>', '\\"')
        return f'"{key}": "{fixed}"'
    
    # Try to fix strings with unescaped quotes (simplified approach)
    # Match "key": "value with "quotes" inside"
    # This regex looks for string values that might have issues
    
    # Fix newlines inside strings
    text = re.sub(r'(?<!\\)\n', ' ', text)
    
    return text


def parse_llm_json(text: str, expected_count: int = None) -> Optional[List[dict]]:
    """
    Parse JSON from LLM response with multiple fallback strategies.
    
    Args:
        text: Raw LLM response text
        expected_count: Expected number of items (for validation)
    
    Returns:
        Parsed list of dicts, or None if all parsing fails
    """
    if not text or not text.strip():
        return None
    
    # Strategy 1: Direct parse
    try:
        # Find JSON array in response
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Repair and parse
    try:
        repaired = repair_json(text)
        result = json.loads(repaired)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    
    # Strategy 3: Parse individual objects
    try:
        # Find all JSON-like objects
        objects = []
        pattern = r'\{[^{}]*"voiceover"\s*:\s*\d+[^{}]*\}'
        matches = re.finditer(pattern, text, re.DOTALL)
        
        for m in matches:
            try:
                obj_text = m.group()
                # Clean up the object
                obj_text = repair_json(obj_text)
                obj = json.loads(obj_text)
                objects.append(obj)
            except:
                continue
        
        if objects:
            return objects
    except:
        pass
    
    # Strategy 4: Extract with regex (last resort)
    try:
        results = []
        # Pattern to match voiceover, selected, confidence
        pattern = r'"voiceover"\s*:\s*(\d+)[^}]*"selected"\s*:\s*(\d+)[^}]*"confidence"\s*:\s*([\d.]+)'
        matches = re.finditer(pattern, text)
        
        for m in matches:
            results.append({
                'voiceover': int(m.group(1)),
                'selected': int(m.group(2)),
                'confidence': float(m.group(3)),
                'reason': 'regex extracted'
            })
        
        if results:
            return results
    except:
        pass
    
    # Strategy 5: Even simpler regex
    try:
        results = []
        # Look for patterns like: 1, selected: 3, confidence: 0.85
        lines = text.split('\n')
        for line in lines:
            vo_match = re.search(r'voiceover["\s:]+(\d+)', line, re.IGNORECASE)
            sel_match = re.search(r'selected["\s:]+(\d+)', line, re.IGNORECASE)
            conf_match = re.search(r'confidence["\s:]+(\d*\.?\d+)', line, re.IGNORECASE)
            
            if vo_match and sel_match:
                results.append({
                    'voiceover': int(vo_match.group(1)),
                    'selected': int(sel_match.group(1)),
                    'confidence': float(conf_match.group(1)) if conf_match else 0.7,
                    'reason': 'line extracted'
                })
        
        if results:
            return results
    except:
        pass
    
    return None


# =============================================================================
# LLM PROVIDERS
# =============================================================================

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None
    ) -> List[Tuple[int, float, str]]:
        """
        Batch match voiceover segments to candidates.
        Returns list of (selected_idx, confidence, reasoning)
        """
        pass


class GeminiMatcher(LLMProvider):
    """Gemini Flash for matching"""
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None
    ) -> List[Tuple[int, float, str]]:
        
        # Build batch prompt
        batch_sections = []
        for i, (vo_text, candidates) in enumerate(items):
            # Escape quotes in text to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")
            candidates_text = "\n".join([
                f"  {j+1}. [{Path(seg.source_file).stem[:30]}] \"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])
            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean[:100]}\"\nCANDIDATES:\n{candidates_text}")
        
        context_str = f"\nCONTEXT: {context}" if context else ""
        
        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)
        
        prompt = f"""Match each voiceover to its best video candidate based on semantic meaning and topic alignment.
{context_str}{negative_str}

{chr(10).join(batch_sections)}

Respond with ONLY a valid JSON array, no other text. Use simple reasons without special characters:
[{{"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "topic match"}}]"""

        try:
            response = self.model.generate_content(prompt)
            
            # Use robust parser
            results_list = parse_llm_json(response.text, expected_count=len(items))
            
            if results_list:
                outputs = []
                for i, (vo_text, candidates) in enumerate(items):
                    result = next((r for r in results_list if r.get('voiceover') == i + 1), None)
                    if result:
                        selected_idx = result.get('selected', 1) - 1
                        # Clamp to valid range
                        selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                        confidence = result.get('confidence', 0.7)
                        # Clamp confidence to valid range
                        confidence = max(0.0, min(1.0, float(confidence)))
                        outputs.append((
                            selected_idx,
                            confidence,
                            str(result.get('reason', 'matched'))[:50]
                        ))
                    else:
                        # Fallback for missing voiceover entry
                        outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback"))
                
                return outputs
            else:
                logger.warning(f"Gemini: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed after all strategies")
                
        except Exception as e:
            logger.warning(f"Gemini batch error: {e}")
            raise
        
        return [(0, 0.5, "error fallback") for _ in items]


class ClaudeMatcher(LLMProvider):
    """Claude Haiku for matching (secondary/ambiguous)"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model
    
    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None
    ) -> List[Tuple[int, float, str]]:
        
        batch_sections = []
        for i, (vo_text, candidates) in enumerate(items):
            # Escape quotes in text to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")[:100]
            candidates_text = "\n".join([
                f"  {j+1}. [{Path(seg.source_file).stem[:30]}] \"{seg.text[:60].replace(chr(34), chr(39))}{'...' if len(seg.text) > 60 else ''}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])
            batch_sections.append(f"VOICEOVER {i+1}: \"{vo_text_clean}\"\nCANDIDATES:\n{candidates_text}")
        
        context_str = f"\nCONTEXT: {context}" if context else ""
        
        negative_str = ""
        if negative_rules:
            negative_str = "\n\nAVOID:\n" + "\n".join(f"- {rule}" for rule in negative_rules)
        
        prompt = f"""Match each voiceover to its best video candidate. Consider semantic meaning, visual relevance, and topic alignment.
{context_str}{negative_str}

{chr(10).join(batch_sections)}

Respond with ONLY a valid JSON array, no other text. Use simple reasons without special characters:
[{{"voiceover": 1, "selected": 1, "confidence": 0.85, "reason": "topic match"}}]"""

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            
            # Use robust parser
            results_list = parse_llm_json(text, expected_count=len(items))
            
            if results_list:
                outputs = []
                for i, (vo_text, candidates) in enumerate(items):
                    result = next((r for r in results_list if r.get('voiceover') == i + 1), None)
                    if result:
                        selected_idx = result.get('selected', 1) - 1
                        # Clamp to valid range
                        selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                        confidence = result.get('confidence', 0.7)
                        # Clamp confidence to valid range
                        confidence = max(0.0, min(1.0, float(confidence)))
                        outputs.append((
                            selected_idx,
                            confidence,
                            str(result.get('reason', 'matched'))[:50]
                        ))
                    else:
                        outputs.append((0, candidates[0][1] if candidates else 0.5, "parse fallback"))
                
                return outputs
            else:
                logger.warning(f"Claude: Could not parse response, using embedding fallback")
                raise ValueError("JSON parsing failed after all strategies")
                
        except Exception as e:
            logger.warning(f"Claude batch error: {e}")
            raise
        
        return [(0, 0.5, "error fallback") for _ in items]


class LocalLLMMatcher(LLMProvider):
    """Local LLM (Ollama) for finishing touches"""
    
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host
    
    def match_batch(
        self,
        items: List[Tuple[str, List[Tuple[SRTSegment, float]]]],
        context: Optional[str] = None,
        negative_rules: Optional[List[str]] = None
    ) -> List[Tuple[int, float, str]]:
        
        import requests
        
        outputs = []
        
        for vo_text, candidates in items:
            # Escape quotes to avoid JSON issues
            vo_text_clean = vo_text.replace('"', "'")[:100]
            candidates_text = "\n".join([
                f"{j+1}. \"{seg.text[:80].replace(chr(34), chr(39))}\""
                for j, (seg, sim) in enumerate(candidates[:5])
            ])
            
            prompt = f"""Match this voiceover to the best candidate:

VOICEOVER: "{vo_text_clean}"

CANDIDATES:
{candidates_text}

Respond with ONLY valid JSON, no other text: {{"selected": 1, "confidence": 0.85, "reason": "topic match"}}"""

            try:
                response = requests.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    text = response.json().get('response', '')
                    
                    # Use robust parser (wrap in list for compatibility)
                    results_list = parse_llm_json(f"[{text}]") or parse_llm_json(text)
                    
                    if results_list and len(results_list) > 0:
                        data = results_list[0]
                        selected_idx = data.get('selected', 1) - 1
                        selected_idx = max(0, min(selected_idx, len(candidates) - 1))
                        confidence = max(0.0, min(1.0, float(data.get('confidence', 0.5))))
                        outputs.append((
                            selected_idx,
                            confidence,
                            str(data.get('reason', 'local match'))[:50]
                        ))
                        continue
                    
                    # Fallback: try simple regex extraction
                    json_match = re.search(r'\{.*\}', text, re.DOTALL)
                    if json_match:
                        try:
                            data = json.loads(json_match.group())
                            outputs.append((
                                max(0, min(data.get('selected', 1) - 1, len(candidates) - 1)),
                                max(0.0, min(1.0, float(data.get('confidence', 0.5)))),
                                str(data.get('reason', 'local match'))[:50]
                            ))
                            continue
                        except:
                            pass
                            
            except Exception as e:
                logger.debug(f"Local LLM error: {e}")
            
            # Fallback
            outputs.append((0, candidates[0][1] if candidates else 0.5, "local fallback"))
        
        return outputs


# =============================================================================
# TIERED MATCHER
# =============================================================================

class TieredMatcher:
    def __init__(self, config=None, cache=None):
        self.config = config or get_config()
        self.cache = cache  # Store cache for potential future use
        mc = self.config.matching

        self.primary_model = mc.gemini_model
        self.min_confidence = mc.min_confidence
        self.embedding_candidates = mc.embedding_candidates
        self.high_conf_threshold = mc.high_confidence_threshold
        self.low_conf_threshold = mc.low_confidence_threshold
        self.max_clip_reuse = mc.max_clip_reuse
        self.reuse_penalty = mc.reuse_penalty

        # Initialize reuse tracker for clip reuse prevention
        self.reuse_tracker = ReuseTracker(
            max_reuse=mc.max_clip_reuse,
            reuse_penalty=mc.reuse_penalty
        )

        # Initialize providers
        self.primary_provider = None
        self.secondary_provider = None
        self.local_provider = None
        self._init_providers()

    def _should_skip_llm(self, similarity: float) -> bool:
        """Skip LLM if embedding similarity is high enough"""
        return similarity >= self.config.matching.high_confidence_threshold
    
    def _apply_duration_penalty(self, confidence: float, speed_ratio: float) -> float:
        """Apply duration-based penalty from config"""
        mc = self.config.matching
        ideal_min, ideal_max = mc.ideal_speed_range
        soft_min, soft_max = mc.soft_speed_range
        
        if ideal_min <= speed_ratio <= ideal_max:
            return confidence  # No penalty
        elif soft_min <= speed_ratio <= soft_max:
            return confidence - mc.duration_penalty_factor
        else:
            return confidence - (mc.duration_penalty_factor * 2)

    def _init_providers(self):
        """Initialize LLM providers based on config"""
        mc = self.config.matching
        
        # Get model names from config
        gemini_model = getattr(mc, 'gemini_model', 'gemini-2.0-flash')
        anthropic_model = getattr(mc, 'anthropic_model', 'claude-3-haiku-20240307')
        ollama_model = getattr(mc, 'ollama_model', 'llama3.2')
        ollama_host = getattr(mc, 'ollama_host', 'http://localhost:11434')
        
        # Primary provider
        if mc.primary_provider == "gemini" and self.config.gemini_api_key:
            self.primary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Primary LLM: Gemini ({gemini_model})")
        elif mc.primary_provider == "anthropic" and self.config.anthropic_api_key:
            self.primary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Primary LLM: Claude ({anthropic_model})")
        elif self.config.gemini_api_key:
            self.primary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Primary LLM: Gemini ({gemini_model}) (auto)")
        elif self.config.anthropic_api_key:
            self.primary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Primary LLM: Claude ({anthropic_model}) (auto)")
        
        # Secondary provider (for ambiguous matches)
        if mc.secondary_provider == "anthropic" and self.config.anthropic_api_key:
            self.secondary_provider = ClaudeMatcher(self.config.anthropic_api_key, anthropic_model)
            logger.info(f"Secondary LLM: Claude ({anthropic_model})")
        elif mc.secondary_provider == "gemini" and self.config.gemini_api_key:
            self.secondary_provider = GeminiMatcher(self.config.gemini_api_key, gemini_model)
            logger.info(f"Secondary LLM: Gemini ({gemini_model})")
        
        # Local provider (for finishing touches)
        if mc.use_local_for_review:
            try:
                self.local_provider = LocalLLMMatcher(ollama_model, ollama_host)
                # Test connection
                import requests
                requests.get(f"{ollama_host}/api/tags", timeout=2)
                logger.info(f"Local LLM: Ollama ({ollama_model})")
            except:
                logger.info("Local LLM: Not available (Ollama not running)")
                self.local_provider = None
    
    def _get_cache_key(self, vo_text: str, candidates: List[Tuple[SRTSegment, float]]) -> str:
        """Generate cache key for LLM response"""
        content = vo_text + "|" + "|".join(c[0].text for c in candidates[:5])
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cached_response(self, cache_key: str) -> Optional[Tuple[int, float, str]]:
        """Get cached LLM response"""
        if not self.config.matching.cache_llm_responses:
            return None
        
        cached = self.cache.get_llm_response(cache_key)
        if cached:
            return (cached['selected'], cached['confidence'], cached['reasoning'])
        return None
    
    def _cache_response(self, cache_key: str, selected: int, confidence: float, reasoning: str):
        """Cache LLM response"""
        if self.config.matching.cache_llm_responses:
            self.cache.save_llm_response(cache_key, {
                'selected': selected,
                'confidence': confidence,
                'reasoning': reasoning
            })
    
    def _compute_duration_penalty(self, vo_segment: SRTSegment, video_segment: SRTSegment) -> float:
        """
        Compute confidence penalty based on speed change required.
        
        Args:
            vo_segment: Voiceover segment (target duration)
            video_segment: Video segment (source duration)
        
        Returns:
            Penalty value (0.0 = no penalty, higher = worse)
        """
        mc = self.config.matching
        
        if not mc.duration_scoring_enabled:
            return 0.0
        
        vo_duration = vo_segment.end_time - vo_segment.start_time
        vid_duration = video_segment.end_time - video_segment.start_time
        
        if vo_duration <= 0 or vid_duration <= 0:
            return 0.0
        
        # Speed ratio (1.0 = no change, >1 = speed up, <1 = slow down)
        speed_ratio = vid_duration / vo_duration
        
        ideal_min, ideal_max = mc.ideal_speed_range
        soft_min, soft_max = mc.soft_penalty_range
        penalty_factor = mc.duration_penalty_factor
        
        if ideal_min <= speed_ratio <= ideal_max:
            # Within ideal range - no penalty
            return 0.0
        elif soft_min <= speed_ratio <= soft_max:
            # Within soft penalty range - small penalty
            return penalty_factor
        else:
            # Outside both ranges - larger penalty
            return penalty_factor * 2
    
    def _apply_duration_scoring(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]]
    ) -> List[Tuple[SRTSegment, float, float]]:
        """
        Apply duration scoring to candidates.
        
        Returns:
            List of (segment, adjusted_similarity, duration_penalty) tuples
        """
        result = []
        for seg, sim in candidates:
            penalty = self._compute_duration_penalty(vo_segment, seg)
            adjusted = max(0.0, sim - penalty)
            result.append((seg, adjusted, penalty))
        
        # Re-sort by adjusted similarity
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    def match_segment(
        self,
        vo_segment: SRTSegment,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]] = None,
        context_before: Optional[List[SRTSegment]] = None,
        context_after: Optional[List[SRTSegment]] = None
    ) -> MatchResult:
        """
        Match a single voiceover segment.
        Returns MatchResult with primary match and alternatives.
        """
        mc = self.config.matching
        
        # Apply face preference if set
        face_pref = getattr(self, 'face_preference', 'neutral')
        if face_pref != 'neutral':
            cache_dir = self.cache.cache_dir if hasattr(self.cache, 'cache_dir') else None
            candidates = apply_face_preference(candidates, face_pref, cache_dir)
        
        # Apply smart reuse - filter out overused clips and adjust confidence
        valid_candidates = []
        for seg, sim in candidates:
            if self.reuse_tracker.can_use(seg):
                adjusted_sim = self.reuse_tracker.adjust_confidence(seg, sim)
                valid_candidates.append((seg, adjusted_sim))
        
        if not valid_candidates:
            # All candidates overused, use originals with heavy penalty
            valid_candidates = [(seg, sim * 0.5) for seg, sim in candidates[:5]]
        
        # Check for high-confidence embedding match
        top_similarity = valid_candidates[0][1] if valid_candidates else 0
        
        if top_similarity >= mc.skip_llm_threshold:
            # Use embedding match directly
            best_seg = valid_candidates[0][0]
            self.reuse_tracker.record_usage(best_seg)
            
            # Get scene info
            scene = self._get_scene_for_segment(best_seg, scenes)
            
            match = Match(
                voiceover_segment=vo_segment,
                video_segment=best_seg,
                video_scene=scene,
                confidence=top_similarity,
                reasoning=f"High embedding similarity ({top_similarity:.2f})",
                embedding_similarity=top_similarity,
                clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg)
            )
            
            # Get alternatives (prefer different sources)
            alternatives = self._get_alternatives(valid_candidates[1:4], scenes, best_seg)
            
            # Get secondary matches (different video files from V1-V3)
            used_video_files = {best_seg.source_file}
            alt_segments = []
            for alt in alternatives:
                used_video_files.add(alt.video_segment.source_file)
                alt_segments.append(alt.video_segment)
            
            secondary_matches = self._get_secondary_matches(
                valid_candidates,
                scenes,
                used_video_files,
                primary_segment=best_seg,
                alt_segments=alt_segments
            )
            
            return MatchResult(primary_match=match, alternatives=alternatives, secondary_matches=secondary_matches)
        
        # Check cache - but only use if selected clip is still available
        cache_key = self._get_cache_key(vo_segment.text, valid_candidates)
        cached = self._get_cached_response(cache_key)
        
        if cached:
            selected_idx, confidence, reasoning = cached
            selected_idx = min(selected_idx, len(valid_candidates) - 1)
            cached_seg = valid_candidates[selected_idx][0]
            
            # Check if cached selection is still usable (not over-reused)
            if self.reuse_tracker.can_use(cached_seg):
                self.reuse_tracker.record_usage(cached_seg)
                
                scene = self._get_scene_for_segment(cached_seg, scenes)
                
                match = Match(
                    voiceover_segment=vo_segment,
                    video_segment=cached_seg,
                    video_scene=scene,
                    confidence=confidence,
                    reasoning=f"(cached) {reasoning}",
                    embedding_similarity=valid_candidates[selected_idx][1],
                    clip_reuse_count=self.reuse_tracker.get_usage_count(cached_seg)
                )
                
                alternatives = self._get_alternatives(
                    [c for i, c in enumerate(valid_candidates[:4]) if i != selected_idx],
                    scenes,
                    cached_seg
                )
                
                # Get secondary matches (different video files from V1-V3)
                used_video_files = {cached_seg.source_file}
                alt_segments = []
                for alt in alternatives:
                    used_video_files.add(alt.video_segment.source_file)
                    alt_segments.append(alt.video_segment)
                
                secondary_matches = self._get_secondary_matches(
                    valid_candidates,
                    scenes,
                    used_video_files,
                    primary_segment=cached_seg,
                    alt_segments=alt_segments
                )
                
                return MatchResult(primary_match=match, alternatives=alternatives, secondary_matches=secondary_matches)
            # If cached clip is over-reused, fall through to LLM matching
        
        # Build context string
        context = self._build_context(context_before, context_after)
        
        # Get negative rules
        negative_rules = self.config.negative_matching.rules if self.config.negative_matching.enabled else None
        
        # Use LLM
        provider = self.primary_provider
        
        if provider:
            try:
                results = provider.match_batch(
                    [(vo_segment.text, valid_candidates[:5])],
                    context=context,
                    negative_rules=negative_rules
                )
                selected_idx, confidence, reasoning = results[0]
                
                # Check if ambiguous - use secondary provider
                if confidence < mc.ambiguous_threshold and self.secondary_provider:
                    logger.debug(f"Ambiguous match ({confidence:.2f}), using secondary LLM")
                    secondary_results = self.secondary_provider.match_batch(
                        [(vo_segment.text, valid_candidates[:5])],
                        context=context,
                        negative_rules=negative_rules
                    )
                    sec_idx, sec_conf, sec_reason = secondary_results[0]
                    
                    # Use secondary if more confident
                    if sec_conf > confidence:
                        selected_idx, confidence, reasoning = sec_idx, sec_conf, f"(secondary) {sec_reason}"
                
                # Cache the response
                self._cache_response(cache_key, selected_idx, confidence, reasoning)
                
            except Exception as e:
                logger.warning(f"LLM matching failed: {e}")
                # Fallback to embedding similarity
                selected_idx = 0
                confidence = valid_candidates[0][1]
                reasoning = "LLM fallback"
        else:
            # No LLM available
            selected_idx = 0
            confidence = valid_candidates[0][1]
            reasoning = "Embedding similarity only"
        
        # Build result
        selected_idx = min(selected_idx, len(valid_candidates) - 1)
        best_seg = valid_candidates[selected_idx][0]
        self.reuse_tracker.record_usage(best_seg)
        
        scene = self._get_scene_for_segment(best_seg, scenes)
        
        # Check for keyword/visual matches
        # Use getattr for keywords since not all segment types have it
        vo_keywords = getattr(vo_segment, 'keywords', []) or []
        seg_keywords = getattr(best_seg, 'keywords', []) or []
        keyword_boost, is_kw_match, is_vis_match = find_keyword_matches(
            vo_keywords,
            seg_keywords,
            scene.visual_keywords if scene else None
        )
        
        match = Match(
            voiceover_segment=vo_segment,
            video_segment=best_seg,
            video_scene=scene,
            confidence=min(1.0, confidence + keyword_boost),
            reasoning=reasoning,
            is_keyword_match=is_kw_match,
            is_visual_match=is_vis_match,
            embedding_similarity=valid_candidates[selected_idx][1],
            clip_reuse_count=self.reuse_tracker.get_usage_count(best_seg)
        )
        
        # Get alternatives (prefer different sources) - V2-V3
        alternatives = self._get_alternatives(
            [c for i, c in enumerate(valid_candidates[:4]) if i != selected_idx],
            scenes,
            best_seg
        )
        
        # Get secondary matches (must use DIFFERENT video files from V1-V3) - V4-V6
        # Collect all video files used by primary and alternatives
        used_video_files = {best_seg.source_file}
        alt_segments = []
        for alt in alternatives:
            used_video_files.add(alt.video_segment.source_file)
            alt_segments.append(alt.video_segment)
        
        # Search through ALL candidates but exclude video files used by V1-V3
        # Third pass allows same video file but different segment as last resort
        secondary_matches = self._get_secondary_matches(
            valid_candidates,  # Search ALL candidates
            scenes,
            used_video_files,
            primary_segment=best_seg,
            alt_segments=alt_segments
        )
        
        # Check for gap (no good match)
        has_gap = confidence < self.config.matching.confidence_threshold
        gap_reason = f"Low confidence ({confidence:.2f})" if has_gap else ""
        
        return MatchResult(
            primary_match=match,
            alternatives=alternatives,
            secondary_matches=secondary_matches,
            has_gap=has_gap,
            gap_reason=gap_reason
        )
    
    def _get_scene_for_segment(
        self,
        segment: SRTSegment,
        scenes: Optional[Dict[str, List[SceneInfo]]]
    ) -> Optional[SceneInfo]:
        """Find the scene containing this segment"""
        if not scenes:
            return None
        
        video_scenes = scenes.get(segment.source_file, [])
        
        for scene in video_scenes:
            if scene.start_time <= segment.start_time < scene.end_time:
                return scene
        
        return None
    
    def _get_alternatives(
        self,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]],
        primary_match: Optional[SRTSegment] = None
    ) -> List[AlternativeMatch]:
        """Get alternative matches, preferring different sources from primary"""
        alternatives = []
        used_sources = set()
        
        # Add primary match source to exclusion
        if primary_match and primary_match.source_file:
            used_sources.add(primary_match.source_file)
        
        # First pass: prefer candidates from different sources
        for seg, sim in candidates:
            if len(alternatives) >= self.config.output.num_alternatives:
                break
            
            # Prefer different source
            if seg.source_file not in used_sources:
                scene = self._get_scene_for_segment(seg, scenes)
                alternatives.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim,
                    reasoning=f"Alternative (different source: {Path(seg.source_file).stem})"
                ))
                used_sources.add(seg.source_file)
        
        # Second pass: fill remaining slots if not enough different sources
        if len(alternatives) < self.config.output.num_alternatives:
            for seg, sim in candidates:
                if len(alternatives) >= self.config.output.num_alternatives:
                    break
                
                # Check if already added
                if any(alt.video_segment.source_file == seg.source_file and 
                       alt.video_segment.start_time == seg.start_time for alt in alternatives):
                    continue
                
                scene = self._get_scene_for_segment(seg, scenes)
                alternatives.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.9,  # Small penalty for same source
                    reasoning="Alternative (fallback)"
                ))
        
        return alternatives
    
    def _get_secondary_matches(
        self,
        candidates: List[Tuple[SRTSegment, float]],
        scenes: Optional[Dict[str, List[SceneInfo]]],
        excluded_video_files: set,
        primary_segment: Optional[SRTSegment] = None,
        alt_segments: Optional[List[SRTSegment]] = None
    ) -> List[AlternativeMatch]:
        """
        Get secondary matches for V4-V6.
        
        Three-pass approach:
        1. First pass: Different video files from V1-V3, different from each other
        2. Second pass: Different video files from V1-V3, allow same source within V4-V6
        3. Third pass: Allow same video file as V1-V3 but DIFFERENT segment (last resort)
        
        Args:
            candidates: List of (segment, similarity) tuples
            scenes: Scene info dict
            excluded_video_files: Set of video file paths to exclude (used by V1-V3)
            primary_segment: V1 segment (to exclude same exact segment)
            alt_segments: V2-V3 segments (to exclude same exact segments)
        
        Returns:
            List of up to 3 secondary matches
        """
        secondary = []
        used_sources = set()
        num_secondary = 3  # V4, V5, V6
        
        # Collect exact segments used by V1-V3 (to avoid in third pass)
        used_segments = set()
        if primary_segment:
            used_segments.add((primary_segment.source_file, primary_segment.start_time))
        if alt_segments:
            for seg in alt_segments:
                if seg:
                    used_segments.add((seg.source_file, seg.start_time))
        
        # First pass: Different video files from V1-V3, different from each other
        for seg, sim in candidates:
            if len(secondary) >= num_secondary:
                break
            
            # Must be from a video file NOT used by V1-V3
            if seg.source_file in excluded_video_files:
                continue
            
            # Prefer different sources within secondary set too
            if seg.source_file in used_sources:
                continue
            
            scene = self._get_scene_for_segment(seg, scenes)
            
            position = len(secondary)
            label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"
            
            secondary.append(AlternativeMatch(
                video_segment=seg,
                video_scene=scene,
                confidence=sim,
                reasoning=f"{label} (source: {Path(seg.source_file).stem})"
            ))
            used_sources.add(seg.source_file)
        
        # Second pass: Different video files from V1-V3, allow same source within V4-V6
        if len(secondary) < num_secondary:
            for seg, sim in candidates:
                if len(secondary) >= num_secondary:
                    break
                
                # Still must be from a video file NOT used by V1-V3
                if seg.source_file in excluded_video_files:
                    continue
                
                # Check if already added (same file AND same time)
                if any(s.video_segment.source_file == seg.source_file and 
                       s.video_segment.start_time == seg.start_time for s in secondary):
                    continue
                
                scene = self._get_scene_for_segment(seg, scenes)
                
                position = len(secondary)
                label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"
                
                secondary.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.95,  # Small penalty
                    reasoning=f"{label} (same source ok)"
                ))
        
        # Third pass (LAST RESORT): Allow same video file as V1-V3 but DIFFERENT segment
        # Only if we still don't have enough and need to fill gaps
        if len(secondary) < num_secondary:
            for seg, sim in candidates:
                if len(secondary) >= num_secondary:
                    break
                
                # Skip if it's the exact same segment as V1-V3
                seg_key = (seg.source_file, seg.start_time)
                if seg_key in used_segments:
                    continue
                
                # Skip if already in secondary
                if any(s.video_segment.source_file == seg.source_file and 
                       s.video_segment.start_time == seg.start_time for s in secondary):
                    continue
                
                scene = self._get_scene_for_segment(seg, scenes)
                
                position = len(secondary)
                label = "Secondary Primary" if position == 0 else f"Secondary Alt {position}"
                
                secondary.append(AlternativeMatch(
                    video_segment=seg,
                    video_scene=scene,
                    confidence=sim * 0.85,  # Larger penalty for same video file as V1-V3
                    reasoning=f"{label} (fallback - different segment)"
                ))
        
        return secondary
    
    def _build_context(
        self,
        context_before: Optional[List[SRTSegment]],
        context_after: Optional[List[SRTSegment]]
    ) -> Optional[str]:
        """Build context string from surrounding segments"""
        if not context_before and not context_after:
            return None
        
        parts = []
        
        if context_before:
            before_text = " | ".join(s.text[:50] for s in context_before[-2:])
            parts.append(f"Before: {before_text}")
        
        if context_after:
            after_text = " | ".join(s.text[:50] for s in context_after[:2])
            parts.append(f"After: {after_text}")
        
        return " || ".join(parts)
    
    def review_with_local_llm(self, matches: List[MatchResult]) -> List[MatchResult]:
        """
        Use local LLM to review and potentially adjust low-confidence matches.
        """
        if not self.local_provider:
            return matches
        
        low_confidence = [
            (i, m) for i, m in enumerate(matches)
            if m.primary_match.confidence < self.config.matching.ambiguous_threshold
        ]
        
        if not low_confidence:
            return matches
        
        logger.info(f"Reviewing {len(low_confidence)} low-confidence matches with local LLM...")
        
        for idx, match_result in low_confidence:
            # Get the match and its alternatives
            primary = match_result.primary_match
            
            # Build candidates from primary + alternatives
            candidates = [
                (primary.video_segment, primary.confidence),
                *[(alt.video_segment, alt.confidence) for alt in match_result.alternatives]
            ]
            
            try:
                results = self.local_provider.match_batch(
                    [(primary.voiceover_segment.text, candidates)]
                )
                new_idx, new_conf, new_reason = results[0]
                
                # Only update if local LLM is more confident
                if new_conf > primary.confidence:
                    logger.debug(f"Local LLM improved match: {primary.confidence:.2f} -> {new_conf:.2f}")
                    
                    if new_idx == 0:
                        # Same selection, just update confidence
                        primary.confidence = new_conf
                        primary.reasoning = f"(local refined) {new_reason}"
                    else:
                        # Different selection - swap with alternative
                        new_seg = candidates[new_idx][0]
                        match_result.primary_match = Match(
                            voiceover_segment=primary.voiceover_segment,
                            video_segment=new_seg,
                            video_scene=match_result.alternatives[new_idx-1].video_scene if new_idx <= len(match_result.alternatives) else None,
                            confidence=new_conf,
                            reasoning=f"(local selected) {new_reason}",
                            embedding_similarity=candidates[new_idx][1]
                        )
            except Exception as e:
                logger.debug(f"Local LLM review failed: {e}")
        
        return matches


# =============================================================================
# STRATEGY MATCHER - Alternative matching strategies for V4-V7
# =============================================================================

class StrategyMatcher:
    """
    Provides alternative matching strategies for variety tracks V4-V7.
    
    Strategies:
    - visual_first: Prioritize scene descriptions over transcripts
    - different_source: Force selection from different source video
    - keyword_only: Match purely on keyword overlap
    - embedding_diversity: Find maximally different clips from V1-V3
    """
    
    def __init__(self, config: Config, scenes: Optional[Dict[str, List[SceneInfo]]]):
        self.config = config
        self.scenes = scenes or {}
        
        # Handle variety_config as either object or dict
        vc = config.output.variety
        if isinstance(vc, dict):
            # Convert dict to object-like accessor
            class VarietyWrapper:
                def __init__(self, d):
                    self.exclude_same_clip = d.get('exclude_same_clip', True)
                    self.require_different_source = d.get('require_different_source', True)
                    self.min_time_distance = d.get('min_time_distance', 10.0)
                    self.min_embedding_distance = d.get('min_embedding_distance', 0.3)
                    self.enforce_timeline_variety = d.get('enforce_timeline_variety', True)
                    self.timeline_variety_window = d.get('timeline_variety_window', 600.0)
                    self.max_source_repeats_in_window = d.get('max_source_repeats_in_window', 1)
            self.variety_config = VarietyWrapper(vc)
        else:
            self.variety_config = vc
    
    def get_clip_id(self, segment: SRTSegment) -> str:
        """Generate unique clip ID"""
        return f"{segment.source_file}:{segment.start_time:.2f}-{segment.end_time:.2f}"
    
    def is_clip_excluded(
        self,
        candidate: SRTSegment,
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]] = None,
        candidate_embedding: List[float] = None,
        force_different_source: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if candidate violates variety enforcement rules.
        Returns (is_excluded, reason)
        
        Args:
            force_different_source: If True, always require different source file
                                   (used for alternative tracks)
        """
        vc = self.variety_config
        
        # Rule 1: Exclude same clip
        if vc.exclude_same_clip:
            cand_id = self.get_clip_id(candidate)
            for existing in existing_matches:
                if self.get_clip_id(existing) == cand_id:
                    return True, "Same clip already used"
        
        # Rule 2: Require different source (config-based OR forced)
        if vc.require_different_source or force_different_source:
            for existing in existing_matches:
                if candidate.source_file == existing.source_file:
                    # When forcing different source, always reject same source
                    if force_different_source:
                        return True, "Same source file (different source required per track)"
                    # Check time distance within same source
                    if vc.min_time_distance > 0:
                        time_dist = abs(candidate.start_time - existing.start_time)
                        if time_dist < vc.min_time_distance:
                            return True, f"Too close in time ({time_dist:.1f}s < {vc.min_time_distance}s)"
                    else:
                        return True, "Same source file"
        
        # Rule 3: Time window exclusion (for any source) - only if NOT forcing different source
        if not force_different_source and vc.min_time_distance > 0:
            for existing in existing_matches:
                if candidate.source_file == existing.source_file:
                    time_dist = abs(candidate.start_time - existing.start_time)
                    if time_dist < vc.min_time_distance:
                        return True, f"Within time window ({time_dist:.1f}s)"
        
        # Rule 4: Minimum embedding distance
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))
        if vc.min_embedding_distance > 0 and _has_emb(candidate_embedding) and _has_emb(existing_embeddings):
            for existing_emb in existing_embeddings:
                similarity = cosine_similarity(candidate_embedding, existing_emb)
                distance = 1.0 - similarity
                if distance < vc.min_embedding_distance:
                    return True, f"Too similar (dist={distance:.2f})"
        
        return False, ""
    
    def match_visual_first(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy A: Visual-First
        Prioritize scene descriptions and visual keywords over transcript text.
        Falls back to filename/path matching if no scene data.
        """
        vo_text = vo_segment.text.lower()
        vo_kw = getattr(vo_segment, 'keywords', None)
        vo_keywords = set(vo_kw) if vo_kw else set()
        
        # Extract key terms from voiceover for visual matching
        visual_terms = {'earthquake', 'tsunami', 'flood', 'storm', 'fire', 'volcano', 'disaster',
                       'building', 'city', 'water', 'wave', 'destruction', 'damage', 'rescue',
                       'people', 'crowd', 'evacuation', 'explosion', 'collapse', 'rubble'}
        vo_visual_hints = set(w.lower() for w in vo_text.split() if w.lower() in visual_terms)
        
        best_candidate = None
        best_score = -1
        best_reason = ""
        
        for seg, text_sim in all_candidates:
            # Check variety exclusion (force different source for strategy tracks)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, reason = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=True)
            if is_excluded:
                continue
            
            # Get scene for this segment
            scene = self._get_scene_for_segment(seg)
            
            visual_score = 0.0
            text_score = text_sim * 0.3  # Weight text at 30%
            
            if scene and scene.description:
                # Score based on scene description
                desc = scene.description.lower()
                visual_keywords = set(scene.visual_keywords) if scene.visual_keywords else set()
                
                # Keyword overlap with visual keywords
                vo_visual_overlap = len(vo_keywords & visual_keywords)
                visual_score += vo_visual_overlap * 0.15
                
                # Check if voiceover words appear in scene description
                vo_words = set(vo_text.split())
                desc_words = set(desc.split())
                word_overlap = len(vo_words & desc_words)
                visual_score += min(word_overlap * 0.05, 0.3)
            else:
                # Fallback: Use filename/path hints for visual matching
                source_name = Path(seg.source_file).stem.lower() if seg.source_file else ""
                
                # Check if visual terms from voiceover appear in filename
                filename_matches = sum(1 for term in vo_visual_hints if term in source_name)
                visual_score += filename_matches * 0.1
                
                # Check if voiceover words appear in video transcript
                seg_text = seg.text.lower()
                vo_visual_in_seg = sum(1 for term in vo_visual_hints if term in seg_text)
                visual_score += vo_visual_in_seg * 0.08
            
            total_score = visual_score * 0.7 + text_score  # Weight visual at 70%
            
            if total_score > best_score:
                best_score = total_score
                best_candidate = seg
                best_reason = f"Visual match (visual={visual_score:.2f}, text={text_score:.2f})"
        
        if best_candidate and best_score > 0:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=min(best_score, 1.0),
                reasoning=best_reason,
                strategy="visual_first"
            )
        return None
    
    def match_different_source(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy B: Different Source Video
        Force selection from a different video file than existing matches.
        """
        used_sources = set(seg.source_file for seg in existing_matches)
        
        for seg, sim in all_candidates:
            # Must be from different source
            if seg.source_file in used_sources:
                continue
            
            # Check other variety rules (force different source)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, reason = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=True)
            if is_excluded and "source" not in reason.lower():
                continue
            
            return StrategyMatch(
                video_segment=seg,
                video_scene=self._get_scene_for_segment(seg),
                confidence=sim,
                reasoning=f"Different source: {Path(seg.source_file).stem}",
                strategy="different_source"
            )
        
        # Fallback: if no different source available, get best that passes other rules
        for seg, sim in all_candidates:
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, _ = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=False)
            if not is_excluded:
                return StrategyMatch(
                    video_segment=seg,
                    video_scene=self._get_scene_for_segment(seg),
                    confidence=sim * 0.8,  # Penalty for not being different source
                    reasoning=f"Fallback (no different source available)",
                    strategy="different_source"
                )
        
        return None
    
    def match_keyword_only(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None
    ) -> Optional[StrategyMatch]:
        """
        Strategy C: Keyword-Only
        Match purely based on keyword and entity overlap, ignore embeddings.
        Falls back to text word overlap if keywords aren't populated.
        """
        vo_kw = getattr(vo_segment, 'keywords', None) or []
        vo_ent = getattr(vo_segment, 'entities', None) or []
        vo_keywords = set(k.lower() for k in vo_kw)
        
        # Handle entities as either strings or dicts with 'text' key
        vo_entities = set()
        for e in vo_ent:
            if isinstance(e, dict):
                text = e.get('text', '')
                if text:
                    vo_entities.add(text.lower())
            elif isinstance(e, str):
                vo_entities.add(e.lower())
        
        vo_all = vo_keywords | vo_entities
        
        # Fallback: Extract important words from voiceover text if no keywords
        if not vo_all:
            # Extract words 4+ chars, not common stop words
            stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'what', 'when', 'where', 'which', 'their', 'there', 'would', 'could', 'should', 'about', 'after', 'before', 'being', 'other', 'these', 'those', 'through', 'during', 'between'}
            vo_words = set(w.lower() for w in vo_segment.text.split() if len(w) >= 4 and w.lower() not in stop_words)
            vo_all = vo_words
        
        if not vo_all:
            return None
        
        best_candidate = None
        best_score = 0
        best_overlap = []
        
        for seg, _ in all_candidates:
            # Check variety exclusion (force different source for strategy tracks)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, _ = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb, force_different_source=True)
            if is_excluded:
                continue
            
            seg_kw = getattr(seg, 'keywords', None) or []
            seg_ent = getattr(seg, 'entities', None) or []
            seg_keywords = set(k.lower() for k in seg_kw)
            
            # Handle entities as either strings or dicts
            seg_entities = set()
            for e in seg_ent:
                if isinstance(e, dict):
                    text = e.get('text', '')
                    if text:
                        seg_entities.add(text.lower())
                elif isinstance(e, str):
                    seg_entities.add(e.lower())
            
            seg_all = seg_keywords | seg_entities
            
            # Also check text for keyword presence
            seg_text = seg.text.lower()
            text_matches = sum(1 for kw in vo_all if kw in seg_text)
            
            # Extract words from video transcript if no keywords
            if not seg_all:
                seg_words = set(w.lower() for w in seg_text.split() if len(w) >= 4)
                seg_all = seg_words
            
            overlap = vo_all & seg_all
            score = len(overlap) * 0.2 + text_matches * 0.1
            
            if score > best_score:
                best_score = score
                best_candidate = seg
                best_overlap = list(overlap)[:5]
        
        if best_candidate and best_score > 0:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=min(best_score, 1.0),
                reasoning=f"Keyword match: {', '.join(best_overlap) if best_overlap else 'text overlap'}",
                strategy="keyword_only"
            )
        
        return None
    
    def match_embedding_diversity(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]],
        candidate_embeddings: Dict[str, List[float]],
        vo_embedding: List[float]
    ) -> Optional[StrategyMatch]:
        """
        Strategy D: Embedding Diversity
        Find clips that are semantically relevant but maximally DIFFERENT from V1-V3.
        """
        def _has_emb(e):
            return e is not None and (len(e) > 0 if hasattr(e, '__len__') else bool(e))
        if not _has_emb(existing_embeddings) or not candidate_embeddings:
            return None
        
        # Get used sources to enforce different source
        used_sources = set(seg.source_file for seg in existing_matches)
        
        best_candidate = None
        best_score = -1
        best_diversity = 0
        
        for seg, text_sim in all_candidates:
            cand_id = self.get_clip_id(seg)
            cand_emb = candidate_embeddings.get(cand_id)
            
            if not _has_emb(cand_emb):
                continue
            
            # Check basic exclusion (same clip)
            if self.variety_config.exclude_same_clip:
                if any(self.get_clip_id(existing) == cand_id for existing in existing_matches):
                    continue
            
            # Require different source from V1-V3
            if seg.source_file in used_sources:
                continue
            
            # Calculate diversity: average distance from existing matches
            distances = []
            for existing_emb in existing_embeddings:
                sim = cosine_similarity(cand_emb, existing_emb)
                distances.append(1.0 - sim)
            
            avg_diversity = sum(distances) / len(distances) if distances else 0
            
            # Relevance to voiceover (must still be relevant)
            vo_relevance = cosine_similarity(cand_emb, vo_embedding) if _has_emb(vo_embedding) else text_sim
            
            # Combined score: want high relevance AND high diversity
            # Diversity weighted more heavily
            combined_score = vo_relevance * 0.4 + avg_diversity * 0.6
            
            if combined_score > best_score and vo_relevance > 0.3:  # Min relevance threshold
                best_score = combined_score
                best_candidate = seg
                best_diversity = avg_diversity
        
        if best_candidate:
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=best_score,
                reasoning=f"Diverse match (diversity={best_diversity:.2f})",
                strategy="embedding_diversity"
            )
        
        return None
    
    def _get_scene_for_segment(self, segment: SRTSegment) -> Optional[SceneInfo]:
        """Find the scene containing this segment"""
        video_scenes = self.scenes.get(segment.source_file, [])
        
        for scene in video_scenes:
            if scene.start_time <= segment.start_time < scene.end_time:
                return scene
        
        return None
    
    def match_source_rotation(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        existing_matches: List[SRTSegment],
        existing_embeddings: List[List[float]] = None,
        candidate_embeddings: Dict[str, List[float]] = None,
        segment_index: int = 0
    ) -> Optional[StrategyMatch]:
        """
        Strategy: Source Rotation
        Cycles through all source videos systematically for maximum variety.
        
        For segment N, picks the best matching clip from source video (N % num_sources).
        This ensures every source gets used and creates visual variety.
        """
        # Get all unique source videos from candidates
        source_videos = list(set(seg.source_file for seg, _ in all_candidates))
        
        if not source_videos:
            return None
        
        # Sort for consistent ordering
        source_videos.sort()
        
        # Determine which source to use for this segment (round-robin)
        assigned_source_idx = segment_index % len(source_videos)
        assigned_source = source_videos[assigned_source_idx]
        
        # Find best clip from assigned source
        best_candidate = None
        best_score = -1
        
        for seg, sim in all_candidates:
            # Must be from assigned source
            if seg.source_file != assigned_source:
                continue
            
            # Check variety exclusion (except source requirement since we're forcing it)
            cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
            is_excluded, reason = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb)
            if is_excluded and "source" not in reason.lower():
                continue
            
            if sim > best_score:
                best_score = sim
                best_candidate = seg
        
        # If no clip from assigned source passes filters, try next sources in rotation
        if best_candidate is None:
            for offset in range(1, len(source_videos)):
                fallback_idx = (assigned_source_idx + offset) % len(source_videos)
                fallback_source = source_videos[fallback_idx]
                
                for seg, sim in all_candidates:
                    if seg.source_file != fallback_source:
                        continue
                    
                    cand_emb = candidate_embeddings.get(self.get_clip_id(seg)) if candidate_embeddings else None
                    is_excluded, _ = self.is_clip_excluded(seg, existing_matches, existing_embeddings, cand_emb)
                    if is_excluded:
                        continue
                    
                    if sim > best_score:
                        best_score = sim
                        best_candidate = seg
                        assigned_source = fallback_source
                        break
                
                if best_candidate:
                    break
        
        if best_candidate:
            source_name = Path(assigned_source).stem
            return StrategyMatch(
                video_segment=best_candidate,
                video_scene=self._get_scene_for_segment(best_candidate),
                confidence=best_score,
                reasoning=f"Source rotation: {source_name} (idx {segment_index % len(source_videos)})",
                strategy="source_rotation"
            )
        
        return None
    
    def get_strategy_matches(
        self,
        vo_segment: SRTSegment,
        all_candidates: List[Tuple[SRTSegment, float]],
        primary_match: SRTSegment,
        alternatives: List[SRTSegment],
        vo_embedding: List[float],
        candidate_embeddings: Dict[str, List[float]],
        segment_index: int = 0
    ) -> List[StrategyMatch]:
        """
        Get all strategy matches for a voiceover segment.
        Ensures variety across all tracks.
        """
        strategies = self.config.output.strategy_tracks
        
        if not self.config.output.include_strategy_tracks:
            return []
        
        # Collect existing matches (V1 + V2-V3)
        existing_matches = [primary_match] + alternatives
        
        # Collect embeddings of existing matches
        existing_embeddings = []
        for seg in existing_matches:
            seg_id = self.get_clip_id(seg)
            if seg_id in candidate_embeddings:
                existing_embeddings.append(candidate_embeddings[seg_id])
        
        strategy_matches = []
        
        for strategy in strategies:
            # Add previous strategy matches to exclusion list
            all_existing = existing_matches + [sm.video_segment for sm in strategy_matches]
            all_existing_embs = existing_embeddings + [
                candidate_embeddings.get(self.get_clip_id(sm.video_segment), [])
                for sm in strategy_matches
            ]
            # Filter empty - handle both lists and numpy arrays
            def _has_content(e):
                if e is None:
                    return False
                if hasattr(e, '__len__'):
                    return len(e) > 0
                return bool(e)
            all_existing_embs = [e for e in all_existing_embs if _has_content(e)]
            
            match = None
            
            if strategy == "visual_first":
                match = self.match_visual_first(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings
                )
            elif strategy == "different_source":
                match = self.match_different_source(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings
                )
            elif strategy == "keyword_only":
                match = self.match_keyword_only(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings
                )
            elif strategy == "embedding_diversity":
                match = self.match_embedding_diversity(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings, vo_embedding
                )
            elif strategy == "source_rotation":
                match = self.match_source_rotation(
                    vo_segment, all_candidates, all_existing,
                    all_existing_embs, candidate_embeddings, segment_index
                )
            
            if match:
                strategy_matches.append(match)
        
        return strategy_matches


def match_all_segments(
    voiceover_segments: List[SRTSegment],
    video_segments: List[SRTSegment],
    voiceover_embeddings: List[List[float]],
    video_embeddings: List[List[float]],
    scenes: Optional[Dict[str, List[SceneInfo]]],
    config: Config,
    cache: CacheManager,
    embedding_index: Optional[Any] = None,
    face_preference: str = "neutral"
) -> List[MatchResult]:
    """
    Match all voiceover segments to video segments.
    Processes sequentially to ensure accurate reuse tracking.
    Also computes strategy matches for V4-V8 with variety enforcement.
    
    Two-stage matching optimization:
    - Stage 1: Retrieve more candidates from embeddings (embedding_candidates)
    - Stage 2: Send only top candidates to LLM for reranking (llm_rerank_candidates)
    
    Timeline variety enforcement:
    - Prevents same source video from appearing multiple times within a time window
    - Configured via config.output.variety.timeline_variety_window (default 600s = 10 min)
    
    Args:
        face_preference: "more" (prefer faces), "none" (avoid faces), or "neutral"
    """
    # EmbeddingIndex import removed - not needed
    
    matcher = TieredMatcher(config, cache)
    strategy_matcher = StrategyMatcher(config, scenes)
    
    # Store face preference for use during matching
    matcher.face_preference = face_preference
    
    mc = config.matching
    oc = config.output
    vc = oc.variety
    
    logger.info(f"Matching {len(voiceover_segments)} voiceover segments...")
    logger.info(f"  Two-stage matching: embedding_candidates={mc.embedding_candidates}, llm_rerank={mc.llm_rerank_candidates}")
    logger.info(f"  Reuse prevention: max_reuse={mc.max_clip_reuse}, penalty={mc.reuse_penalty}")
    if mc.max_clip_reuse == 1:
        logger.info(f"  Mode: Each clip can only be used ONCE")
    
    if face_preference != "neutral":
        logger.info(f"  Face preference: {face_preference}")
    
    if mc.duration_scoring_enabled:
        logger.info(f"  Duration scoring: ideal={mc.ideal_speed_range}, soft={mc.soft_penalty_range}")
    
    # Timeline variety enforcement
    # Handle vc as either object or dict
    if isinstance(vc, dict):
        timeline_variety_enabled = vc.get('enforce_timeline_variety', True)
        timeline_window = vc.get('timeline_variety_window', 600.0)
        max_repeats = vc.get('max_source_repeats_in_window', 1)
    else:
        timeline_variety_enabled = getattr(vc, 'enforce_timeline_variety', True)
        timeline_window = getattr(vc, 'timeline_variety_window', 600.0)
        max_repeats = getattr(vc, 'max_source_repeats_in_window', 1)
    
    variety_tracker = None
    if timeline_variety_enabled:
        variety_tracker = TimelineVarietyTracker(
            timeline_window=timeline_window,
            max_repeats=max_repeats
        )
        logger.info(f"  Timeline variety: {timeline_window/60:.0f}min window, max {max_repeats} repeat(s) per source")
    
    if oc.include_strategy_tracks:
        logger.info(f"  Strategy tracks: {', '.join(oc.strategy_tracks)}")
        # Handle vc as either object or dict
        if isinstance(vc, dict):
            req_diff = vc.get('require_different_source', True)
            min_time = vc.get('min_time_distance', 10.0)
            min_emb = vc.get('min_embedding_distance', 0.3)
        else:
            req_diff = getattr(vc, 'require_different_source', True)
            min_time = getattr(vc, 'min_time_distance', 10.0)
            min_emb = getattr(vc, 'min_embedding_distance', 0.3)
        logger.info(f"  Variety enforcement: different_source={req_diff}, "
                   f"min_time={min_time}s, min_emb_dist={min_emb}")
    
    # Build candidate embeddings lookup
    candidate_embeddings = {}
    for seg, emb in zip(video_segments, video_embeddings):
        clip_id = strategy_matcher.get_clip_id(seg)
        candidate_embeddings[clip_id] = emb
    
    progress = ProgressBar(len(voiceover_segments), "Matching")
    
    results = []
    
    # Calculate timeline start (first segment start time)
    timeline_start = voiceover_segments[0].start_time if voiceover_segments else 0.0
    
    for i, (vo_seg, vo_emb) in enumerate(zip(voiceover_segments, voiceover_embeddings)):
        # Calculate current timeline position (relative to start)
        current_timeline_pos = vo_seg.start_time - timeline_start
        
        # Stage 1: Get more candidates from embedding search for variety
        num_embedding_candidates = max(mc.embedding_candidates, 20)
        distances, indices = find_top_k_similar(vo_emb, video_embeddings, num_embedding_candidates, index=embedding_index)
        all_candidates = [(video_segments[idx], distances[j]) for j, idx in enumerate(indices)]
        
        # Apply timeline variety filtering for V1 (primary track)
        if variety_tracker:
            excluded_v1 = variety_tracker.get_excluded_sources("V1", current_timeline_pos)
            if excluded_v1:
                # Filter out excluded sources, but keep at least some candidates
                filtered_candidates = [(seg, dist) for seg, dist in all_candidates 
                                       if seg.source_file not in excluded_v1]
                if len(filtered_candidates) >= mc.llm_rerank_candidates:
                    all_candidates = filtered_candidates
                else:
                    # Log that we had to relax the constraint
                    logger.debug(f"Segment {i}: Relaxed variety constraint (only {len(filtered_candidates)} candidates after filter)")
        
        # Stage 2: Send only top candidates to LLM for reranking
        llm_candidates = all_candidates[:mc.llm_rerank_candidates]
        
        # Get context
        context_before = voiceover_segments[max(0, i - mc.context_window):i] if mc.context_window > 0 else None
        context_after = voiceover_segments[i+1:i+1+mc.context_window] if mc.context_window > 0 else None
        
        # Primary match (V1) - use only llm_rerank_candidates for LLM
        result = matcher.match_segment(
            vo_seg, llm_candidates, scenes,
            context_before, context_after
        )
        
        # Record V1 usage for timeline variety
        if variety_tracker and result.primary_match:
            variety_tracker.record_usage(
                "V1", 
                result.primary_match.video_segment.source_file,
                current_timeline_pos
            )
        
        # Record V2, V3 (alternatives) usage
        if variety_tracker and result.alternatives:
            for alt_idx, alt in enumerate(result.alternatives, start=2):
                variety_tracker.record_usage(
                    f"V{alt_idx}",
                    alt.video_segment.source_file,
                    current_timeline_pos
                )
        
        # Strategy matches (V4-V8) - use all embedding candidates for variety
        if oc.include_strategy_tracks:
            # Get alternative segments (V2-V3)
            alt_segments = [alt.video_segment for alt in result.alternatives]
            
            # Apply timeline variety filtering for strategy candidates
            strategy_candidates = all_candidates
            if variety_tracker:
                # For strategy tracks, exclude sources used in V1-V3 AND timeline-excluded sources
                # Use a combined track "V_strategy" for variety tracking
                excluded_strategy = variety_tracker.get_excluded_sources("V_strategy", current_timeline_pos)
                if excluded_strategy:
                    filtered_strategy = [(seg, dist) for seg, dist in all_candidates
                                        if seg.source_file not in excluded_strategy]
                    if len(filtered_strategy) >= 5:  # Need at least some candidates
                        strategy_candidates = filtered_strategy
            
            # Compute strategy matches with variety enforcement
            # Pass segment_index for source_rotation strategy
            strategy_matches = strategy_matcher.get_strategy_matches(
                vo_segment=vo_seg,
                all_candidates=strategy_candidates,  # Use filtered candidates
                primary_match=result.primary_match.video_segment,
                alternatives=alt_segments,
                vo_embedding=vo_emb,
                candidate_embeddings=candidate_embeddings,
                segment_index=i
            )
            
            result.strategy_matches = strategy_matches
            
            # Record strategy track usage
            if variety_tracker:
                for sm in strategy_matches:
                    variety_tracker.record_usage(
                        "V_strategy",
                        sm.video_segment.source_file,
                        current_timeline_pos
                    )
        
        results.append(result)
        
        # Progress with strategy count
        strat_count = len(result.strategy_matches) if result.strategy_matches else 0
        progress.update(1, f"conf: {result.primary_match.confidence:.2f}, strat: {strat_count}")
    
    progress.close()
    
    # Review low-confidence matches with local LLM
    if config.matching.use_local_for_review and matcher.local_provider:
        results = matcher.review_with_local_llm(results)
    
    # Report gaps
    gaps = [r for r in results if r.has_gap]
    if gaps:
        logger.warning(f"Found {len(gaps)} footage gaps (low confidence matches)")
        for gap in gaps[:5]:  # Show first 5
            logger.warning(f"  - \"{gap.primary_match.voiceover_segment.text[:50]}...\" ({gap.gap_reason})")
    
    # Report strategy match stats
    if oc.include_strategy_tracks:
        for strategy in oc.strategy_tracks:
            count = sum(1 for r in results for sm in r.strategy_matches if sm.strategy == strategy)
            logger.info(f"  {strategy}: {count}/{len(results)} segments matched")
    
    # Report timeline variety stats
    if variety_tracker:
        stats = variety_tracker.get_stats()
        logger.info(f"  Timeline variety enforcement:")
        for track, track_stats in stats.items():
            if track_stats["total_clips"] > 0:
                unique_pct = track_stats["unique_sources"] / track_stats["total_clips"] * 100
                logger.info(f"    {track}: {track_stats['unique_sources']} unique sources across {track_stats['total_clips']} clips ({unique_pct:.0f}% variety)")
                if track_stats["top_sources"]:
                    top = track_stats["top_sources"][0]
                    logger.info(f"      Most used: {top[0]} ({top[1]} times)")
    
    return results