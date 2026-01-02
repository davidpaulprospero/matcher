"""
Topic Extraction Module

Extracts topic keywords from video transcripts and detects chapters in voiceover.
Used for chapter-based matching to ensure videos match voiceover topics.
"""

import logging
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class VideoTopics:
    """Topic information for a video"""
    video_path: str
    topics: List[str]
    source_keyword: str = ""  # Original download keyword
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoTopics":
        return cls(
            video_path=data.get('video_path', ''),
            topics=data.get('topics', []),
            source_keyword=data.get('source_keyword', ''),
            confidence=data.get('confidence', 0.0)
        )


class TopicExtractor:
    """
    Extracts topics from video transcripts using LLM.
    Caches results to avoid re-extraction.
    """

    def __init__(self, config, cache_dir: str = ".cache"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.topics_cache_path = self.cache_dir / "video_topics.json"
        self._topics_cache: Dict[str, VideoTopics] = {}
        self._load_cache()

    def _load_cache(self):
        """Load cached topics from disk"""
        if self.topics_cache_path.exists():
            try:
                with open(self.topics_cache_path, 'r') as f:
                    data = json.load(f)
                    for path, topic_data in data.items():
                        self._topics_cache[path] = VideoTopics.from_dict(topic_data)
                logger.debug(f"Loaded {len(self._topics_cache)} cached video topics")
            except Exception as e:
                logger.warning(f"Could not load topics cache: {e}")

    def _save_cache(self):
        """Save topics cache to disk"""
        try:
            data = {path: vt.to_dict() for path, vt in self._topics_cache.items()}
            with open(self.topics_cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save topics cache: {e}")

    def get_cached_topics(self, video_path: str) -> Optional[VideoTopics]:
        """Get cached topics for a video"""
        return self._topics_cache.get(str(video_path))

    def extract_topics_from_transcript(
        self,
        transcript_text: str,
        video_path: str,
        video_title: str = None,
        source_keyword: str = None
    ) -> VideoTopics:
        """
        Extract topic keywords from video transcript using LLM.

        Args:
            transcript_text: Full transcript text
            video_path: Path to video file
            video_title: Optional video title for context
            source_keyword: Original keyword used to download this video

        Returns:
            VideoTopics object with extracted topics
        """
        video_path = str(video_path)

        # Check cache first
        cached = self.get_cached_topics(video_path)
        if cached:
            return cached

        # If transcript is too short, use source keyword as topic
        if not transcript_text or len(transcript_text.strip()) < 50:
            topics = [source_keyword] if source_keyword else []
            result = VideoTopics(
                video_path=video_path,
                topics=topics,
                source_keyword=source_keyword or "",
                confidence=0.5
            )
            self._topics_cache[video_path] = result
            self._save_cache()
            return result

        # Use LLM to extract topics
        try:
            topics = self._extract_with_llm(transcript_text, video_title, source_keyword)
            confidence = 0.9 if topics else 0.3

            result = VideoTopics(
                video_path=video_path,
                topics=topics,
                source_keyword=source_keyword or "",
                confidence=confidence
            )

            self._topics_cache[video_path] = result
            self._save_cache()
            return result

        except Exception as e:
            logger.warning(f"Topic extraction failed for {video_path}: {e}")
            # Fallback to source keyword
            topics = [source_keyword] if source_keyword else []
            return VideoTopics(
                video_path=video_path,
                topics=topics,
                source_keyword=source_keyword or "",
                confidence=0.3
            )

    def _extract_with_llm(
        self,
        transcript_text: str,
        video_title: str = None,
        source_keyword: str = None
    ) -> List[str]:
        """Use LLM to extract topic keywords from transcript"""
        # Truncate transcript if too long
        max_chars = 3000
        if len(transcript_text) > max_chars:
            transcript_text = transcript_text[:max_chars] + "..."

        # Build prompt
        context = ""
        if video_title:
            context += f"Video title: {video_title}\n"
        if source_keyword:
            context += f"Search keyword: {source_keyword}\n"

        prompt = f"""Extract 3-5 topic keywords from this video transcript.
Focus on:
- Main subject/location (e.g., "Boise", "Dallas", "earthquake")
- Key themes (e.g., "downtown", "history", "disaster")
- Specific entities mentioned (e.g., "Capitol Building", "flooding")

{context}
Transcript:
{transcript_text}

Return ONLY a JSON array of lowercase keywords, like: ["boise", "downtown", "history"]
No explanation, just the JSON array."""

        try:
            # Try Gemini first
            if hasattr(self.config, 'gemini_api_key') and self.config.gemini_api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                return self._parse_topics_response(response.text)

            # Fallback: extract from source keyword
            if source_keyword:
                return [kw.strip().lower() for kw in source_keyword.split() if len(kw) > 2]

            return []

        except Exception as e:
            logger.warning(f"LLM topic extraction failed: {e}")
            if source_keyword:
                return [kw.strip().lower() for kw in source_keyword.split() if len(kw) > 2]
            return []

    def _parse_topics_response(self, response: str) -> List[str]:
        """Parse LLM response to extract topic list"""
        try:
            # Try to find JSON array in response
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                topics = json.loads(match.group())
                return [str(t).lower().strip() for t in topics if t]
        except:
            pass

        # Fallback: extract quoted strings
        topics = re.findall(r'"([^"]+)"', response)
        return [t.lower().strip() for t in topics if t]

    def extract_batch(
        self,
        transcripts: Dict[str, str],
        video_metadata: Dict[str, dict] = None
    ) -> Dict[str, VideoTopics]:
        """
        Extract topics for multiple videos.

        Args:
            transcripts: Dict of video_path -> transcript_text
            video_metadata: Optional dict of video_path -> metadata (title, keyword)

        Returns:
            Dict of video_path -> VideoTopics
        """
        results = {}
        video_metadata = video_metadata or {}
        total = len(transcripts)

        for idx, (video_path, transcript) in enumerate(transcripts.items(), 1):
            meta = video_metadata.get(video_path, {})
            title = meta.get('title', '')
            keyword = meta.get('keyword', '')

            result = self.extract_topics_from_transcript(
                transcript_text=transcript,
                video_path=video_path,
                video_title=title,
                source_keyword=keyword
            )
            results[video_path] = result

            if idx % 50 == 0 or idx == total:
                logger.info(f"({idx}/{total}) Topic extraction progress")

        return results


class ChapterDetector:
    """
    Detects chapters/topic sections in voiceover using LLM.
    """

    def __init__(self, config):
        self.config = config

    def detect_chapters(
        self,
        segments: List[dict],
        overall_topic: str = None
    ) -> List[dict]:
        """
        Detect chapter boundaries and topics in voiceover segments.

        Args:
            segments: List of voiceover segment dicts with 'text' field
            overall_topic: Optional overall topic context

        Returns:
            List of Chapter dicts with start_segment_idx, end_segment_idx, title, topics
        """
        if not segments:
            return []

        # Combine segment texts with indices
        indexed_text = "\n".join([
            f"[{i}] {seg.get('text', '')}"
            for i, seg in enumerate(segments)
        ])

        # Use LLM to detect chapters
        try:
            chapters = self._detect_with_llm(indexed_text, len(segments), overall_topic)
            if chapters:
                logger.info(f"Detected {len(chapters)} chapters in voiceover")
                return chapters
        except Exception as e:
            logger.warning(f"Chapter detection failed: {e}")

        # Fallback: treat entire voiceover as one chapter
        return [{
            'chapter_id': 0,
            'start_segment_idx': 0,
            'end_segment_idx': len(segments) - 1,
            'title': overall_topic or 'Main Content',
            'topics': [overall_topic.lower()] if overall_topic else []
        }]

    def _detect_with_llm(
        self,
        indexed_text: str,
        num_segments: int,
        overall_topic: str = None
    ) -> List[dict]:
        """Use LLM to detect chapter boundaries"""
        # Truncate if too long
        max_chars = 8000
        if len(indexed_text) > max_chars:
            indexed_text = indexed_text[:max_chars] + "\n..."

        context = f"Overall topic: {overall_topic}\n" if overall_topic else ""

        prompt = f"""Analyze this voiceover transcript and identify distinct chapters or topic sections.

{context}
Transcript (with segment indices in brackets):
{indexed_text}

Identify 2-5 chapters based on topic changes. For each chapter provide:
- start_segment_idx: first segment index
- end_segment_idx: last segment index
- title: brief chapter title
- topics: 2-4 topic keywords for matching videos

Return ONLY a JSON array like:
[
  {{"start_segment_idx": 0, "end_segment_idx": 3, "title": "Introduction to Boise", "topics": ["boise", "introduction", "idaho"]}},
  {{"start_segment_idx": 4, "end_segment_idx": 8, "title": "Downtown Development", "topics": ["downtown", "development", "buildings"]}}
]

No explanation, just the JSON array."""

        try:
            if hasattr(self.config, 'gemini_api_key') and self.config.gemini_api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                return self._parse_chapters_response(response.text, num_segments)

            return []

        except Exception as e:
            logger.warning(f"LLM chapter detection failed: {e}")
            return []

    def _parse_chapters_response(self, response: str, num_segments: int) -> List[dict]:
        """Parse LLM response to extract chapters"""
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                chapters = json.loads(match.group())

                # Validate and clean chapters
                valid_chapters = []
                for i, ch in enumerate(chapters):
                    if isinstance(ch, dict):
                        start = ch.get('start_segment_idx', 0)
                        end = ch.get('end_segment_idx', num_segments - 1)

                        # Ensure valid range
                        start = max(0, min(start, num_segments - 1))
                        end = max(start, min(end, num_segments - 1))

                        valid_chapters.append({
                            'chapter_id': i,
                            'start_segment_idx': start,
                            'end_segment_idx': end,
                            'title': ch.get('title', f'Chapter {i+1}'),
                            'topics': [t.lower() for t in ch.get('topics', [])]
                        })

                return valid_chapters
        except Exception as e:
            logger.warning(f"Failed to parse chapters response: {e}")

        return []


def compute_topic_overlap(topics1: List[str], topics2: List[str]) -> Tuple[int, float]:
    """
    Compute topic overlap between two topic lists.

    Returns:
        Tuple of (overlap_count, overlap_ratio)
    """
    if not topics1 or not topics2:
        return 0, 0.0

    set1 = set(t.lower().strip() for t in topics1)
    set2 = set(t.lower().strip() for t in topics2)

    overlap = set1 & set2
    overlap_count = len(overlap)

    # Also check for partial matches (e.g., "boise" matches "boise downtown")
    partial_matches = 0
    for t1 in set1:
        for t2 in set2:
            if t1 in t2 or t2 in t1:
                partial_matches += 0.5

    total_overlap = overlap_count + partial_matches
    max_possible = max(len(set1), len(set2))
    overlap_ratio = total_overlap / max_possible if max_possible > 0 else 0.0

    return overlap_count, min(overlap_ratio, 1.0)


def compute_topic_penalty(
    vo_topics: List[str],
    video_topics: List[str],
    max_penalty: float = 0.15,
    min_overlap: int = 1
) -> float:
    """
    Compute confidence penalty based on topic mismatch.

    Args:
        vo_topics: Voiceover segment/chapter topics
        video_topics: Video segment topics
        max_penalty: Maximum penalty to apply
        min_overlap: Minimum overlap required for no penalty

    Returns:
        Penalty value (0.0 = no penalty, max_penalty = full penalty)
    """
    if not vo_topics or not video_topics:
        return 0.0  # No penalty if topics unknown

    overlap_count, overlap_ratio = compute_topic_overlap(vo_topics, video_topics)

    if overlap_count >= min_overlap:
        return 0.0  # Sufficient overlap, no penalty

    if overlap_ratio > 0.3:
        return max_penalty * 0.3  # Partial match, small penalty

    if overlap_ratio > 0:
        return max_penalty * 0.6  # Weak match, medium penalty

    return max_penalty  # No match, full penalty
