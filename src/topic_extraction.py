"""
Topic Extraction Module

Extracts topic keywords from video transcripts and detects chapters in voiceover.
Used for chapter-based matching to ensure videos match voiceover topics.

Enhanced with location-aware chapter detection for travel/geographic content.
"""
from __future__ import annotations

import logging
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING, Any
from dataclasses import dataclass, field, asdict

from .utils import normalize_path

if TYPE_CHECKING:
    from .location_service import GeoLocation, LocationService

logger = logging.getLogger(__name__)


@dataclass
class VideoTopics:
    """Topic information for a video"""
    video_path: str
    topics: List[str]
    source_keyword: str = ""  # Original download keyword
    confidence: float = 0.0
    detected_location: Optional[str] = None  # Location extracted from title/description
    location_data: Optional[Dict] = None  # Resolved GeoLocation as dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VideoTopics":
        return cls(
            video_path=data.get('video_path', ''),
            topics=data.get('topics', []),
            source_keyword=data.get('source_keyword', ''),
            confidence=data.get('confidence', 0.0),
            detected_location=data.get('detected_location'),
            location_data=data.get('location_data'),
        )


@dataclass
class LocationChapter:
    """
    Chapter that focuses on a specific geographic location.

    Used for location-aware matching to ensure videos match
    the correct geographic context (e.g., Paris, France vs Paris, Texas).
    """
    chapter_id: int
    start_segment_idx: int
    end_segment_idx: int
    location_name: str                      # Raw location name: "Paris"
    location_type: str = "city"             # city, country, landmark, region, natural_feature
    visual_keywords: List[str] = field(default_factory=list)   # Landmarks: ["Eiffel Tower", "Louvre"]
    context_keywords: List[str] = field(default_factory=list)  # Themes: ["romantic", "fashion"]
    title: str = ""                         # Chapter title
    topics: List[str] = field(default_factory=list)  # General topic keywords
    location_data: Optional[Dict] = None    # Resolved GeoLocation as dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "LocationChapter":
        return cls(
            chapter_id=data.get('chapter_id', 0),
            start_segment_idx=data.get('start_segment_idx', 0),
            end_segment_idx=data.get('end_segment_idx', 0),
            location_name=data.get('location_name', ''),
            location_type=data.get('location_type', 'city'),
            visual_keywords=data.get('visual_keywords', []),
            context_keywords=data.get('context_keywords', []),
            title=data.get('title', ''),
            topics=data.get('topics', []),
            location_data=data.get('location_data'),
        )

    @property
    def segment_range(self) -> Tuple[int, int]:
        """Get segment index range as tuple"""
        return (self.start_segment_idx, self.end_segment_idx)


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
                        # Normalize path when loading for consistent matching
                        normalized = normalize_path(path)
                        self._topics_cache[normalized] = VideoTopics.from_dict(topic_data)
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
        normalized = normalize_path(video_path)
        return self._topics_cache.get(normalized)

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
        normalized_path = normalize_path(video_path)

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
            self._topics_cache[normalized_path] = result
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

            self._topics_cache[normalized_path] = result
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

    def detect_location_chapters(
        self,
        segments: List[dict],
        location_service: Any = None,
        overall_topic: str = None
    ) -> List[LocationChapter]:
        """
        Detect chapters that focus on specific geographic locations.

        This is designed for travel/location content where chapters cover
        different cities, countries, or landmarks.

        Args:
            segments: List of voiceover segment dicts with 'text' field
            location_service: Optional LocationService for geocoding
            overall_topic: Optional overall topic context

        Returns:
            List of LocationChapter objects for location-focused chapters
        """
        if not segments:
            return []

        # Combine segment texts with indices
        indexed_text = "\n".join([
            f"[{i}] {seg.get('text', '')}"
            for i, seg in enumerate(segments)
        ])

        # Truncate if too long
        max_chars = 8000
        if len(indexed_text) > max_chars:
            indexed_text = indexed_text[:max_chars] + "\n..."

        context = f"Overall topic: {overall_topic}\n" if overall_topic else ""

        prompt = f"""Analyze this voiceover transcript and identify chapters that focus on SPECIFIC LOCATIONS.

{context}
Transcript (with segment indices in brackets):
{indexed_text}

For each chapter that is primarily about a specific LOCATION (city, country, landmark, region), extract:
- start_segment_idx: first segment index
- end_segment_idx: last segment index
- location_name: the main location name (e.g., "Paris", "Grand Canyon", "Japan")
- location_type: one of "city", "country", "landmark", "region", "natural_feature"
- visual_keywords: 2-4 visual landmarks or features associated with this location
- context_keywords: 2-3 thematic keywords (culture, activities, themes)
- title: brief chapter title

ONLY include chapters where a location is the PRIMARY SUBJECT.
Skip chapters about general topics, introductions, or conclusions.

Return ONLY a JSON array like:
[
  {{
    "start_segment_idx": 0,
    "end_segment_idx": 5,
    "location_name": "Paris",
    "location_type": "city",
    "visual_keywords": ["Eiffel Tower", "Louvre", "Notre Dame", "Seine River"],
    "context_keywords": ["romantic", "art", "cuisine"],
    "title": "Exploring Paris"
  }},
  {{
    "start_segment_idx": 6,
    "end_segment_idx": 12,
    "location_name": "Tokyo",
    "location_type": "city",
    "visual_keywords": ["Shibuya Crossing", "Tokyo Tower", "temples"],
    "context_keywords": ["modern", "traditional", "technology"],
    "title": "Tokyo Adventures"
  }}
]

If no chapters focus on specific locations, return an empty array: []
No explanation, just the JSON array."""

        try:
            if hasattr(self.config, 'gemini_api_key') and self.config.gemini_api_key:
                import google.generativeai as genai
                genai.configure(api_key=self.config.gemini_api_key)
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                location_chapters = self._parse_location_chapters_response(
                    response.text, len(segments)
                )

                # Resolve locations if service provided
                if location_service and location_chapters:
                    location_chapters = self._resolve_chapter_locations(
                        location_chapters, location_service
                    )

                if location_chapters:
                    logger.info(f"Detected {len(location_chapters)} location-focused chapters")

                return location_chapters

            return []

        except Exception as e:
            logger.warning(f"Location chapter detection failed: {e}")
            return []

    def _parse_location_chapters_response(
        self,
        response: str,
        num_segments: int
    ) -> List[LocationChapter]:
        """Parse LLM response to extract location chapters"""
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                chapters = json.loads(match.group())

                location_chapters = []
                for i, ch in enumerate(chapters):
                    if isinstance(ch, dict) and ch.get('location_name'):
                        start = ch.get('start_segment_idx', 0)
                        end = ch.get('end_segment_idx', num_segments - 1)

                        # Ensure valid range
                        start = max(0, min(start, num_segments - 1))
                        end = max(start, min(end, num_segments - 1))

                        location_chapters.append(LocationChapter(
                            chapter_id=i,
                            start_segment_idx=start,
                            end_segment_idx=end,
                            location_name=ch.get('location_name', ''),
                            location_type=ch.get('location_type', 'city'),
                            visual_keywords=ch.get('visual_keywords', []),
                            context_keywords=ch.get('context_keywords', []),
                            title=ch.get('title', f"Chapter {i+1}"),
                            topics=[ch.get('location_name', '').lower()] +
                                   [kw.lower() for kw in ch.get('context_keywords', [])],
                        ))

                return location_chapters

        except Exception as e:
            logger.warning(f"Failed to parse location chapters response: {e}")

        return []

    def _resolve_chapter_locations(
        self,
        chapters: List[LocationChapter],
        location_service: Any
    ) -> List[LocationChapter]:
        """
        Resolve location names to GeoLocation data using LocationService.

        Args:
            chapters: List of LocationChapter with raw location names
            location_service: LocationService instance for geocoding

        Returns:
            Updated chapters with resolved location_data
        """
        for chapter in chapters:
            if chapter.location_name and not chapter.location_data:
                try:
                    # Build context from visual and context keywords
                    context = " ".join(chapter.visual_keywords + chapter.context_keywords)

                    # Disambiguate location
                    geo_location = location_service.disambiguate(
                        chapter.location_name,
                        context=context
                    )

                    if geo_location:
                        chapter.location_data = geo_location.to_dict()
                        logger.info(
                            f"Resolved location: '{chapter.location_name}' -> "
                            f"{geo_location.name}, {geo_location.country_name} ({geo_location.country_code})"
                        )

                except Exception as e:
                    logger.debug(f"Could not resolve location '{chapter.location_name}': {e}")

        return chapters


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


def extract_location_from_video_metadata(
    title: str,
    description: str = "",
    source_keyword: str = "",
    config: Any = None
) -> Optional[str]:
    """
    Extract location name from video metadata (title, description, keyword).

    Uses LLM to intelligently extract location from video title patterns like:
    - "4K Walk in Paris"
    - "Tokyo Street Food Tour"
    - "Exploring the Swiss Alps"
    - "New York City Skyline Drone Footage"

    Args:
        title: Video title
        description: Video description (optional, first 500 chars used)
        source_keyword: Original search keyword
        config: Pipeline config with LLM settings

    Returns:
        Extracted location name or None
    """
    if not title:
        return None

    # Try quick pattern matching first (common title patterns)
    location = _extract_location_patterns(title)
    if location:
        return location

    # If config available, use LLM for more complex extraction
    if config and hasattr(config, 'gemini_api_key') and config.gemini_api_key:
        try:
            location = _extract_location_with_llm(title, description, source_keyword, config)
            if location:
                return location
        except Exception as e:
            logger.debug(f"LLM location extraction failed: {e}")

    # Fallback: try to extract from source keyword
    if source_keyword:
        location = _extract_location_patterns(source_keyword)
        if location:
            return location

    return None


def _extract_location_patterns(text: str) -> Optional[str]:
    """
    Extract location from text using common patterns.

    Patterns detected:
    - "in Paris" / "in New York"
    - "Paris, France" / "Tokyo, Japan"
    - "[City] Walking Tour" / "[City] Drone Footage"
    - "Exploring [Location]"
    """
    if not text:
        return None

    text_clean = text.strip()

    # Pattern: "in [Location]" or "of [Location]"
    match = re.search(r'\b(?:in|of|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b', text_clean)
    if match:
        return match.group(1)

    # Pattern: "[City], [Country/State]"
    match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', text_clean)
    if match:
        return f"{match.group(1)}, {match.group(2)}"

    # Pattern: "[Location] Walking Tour" / "[Location] Drone" / "[Location] 4K"
    match = re.search(
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:Walking|Drone|4K|Travel|Tour|Street|City)',
        text_clean
    )
    if match:
        return match.group(1)

    # Pattern: "Exploring [Location]" / "Discover [Location]"
    match = re.search(
        r'(?:Exploring|Discover|Visit|See|Experience)\s+(?:the\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        text_clean
    )
    if match:
        return match.group(1)

    # Pattern: "[Location] footage" / "[Location] video"
    match = re.search(
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+(?:footage|video|aerial|skyline)',
        text_clean,
        re.IGNORECASE
    )
    if match:
        return match.group(1)

    return None


def _extract_location_with_llm(
    title: str,
    description: str,
    source_keyword: str,
    config: Any
) -> Optional[str]:
    """Use LLM to extract location from video metadata"""
    import google.generativeai as genai
    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')

    desc_snippet = description[:500] if description else ""

    prompt = f"""Extract the PRIMARY LOCATION from this video metadata.

Title: "{title}"
Description: "{desc_snippet}"
Search keyword: "{source_keyword}"

If this video is about a specific geographic location (city, country, landmark, etc.),
return ONLY the location name (e.g., "Paris", "Tokyo", "Grand Canyon", "New York City").

If no specific location is identifiable, return "NONE".

Location:"""

    try:
        response = model.generate_content(prompt)
        location = response.text.strip().strip('"').strip("'")

        # Validate response
        if location and location.upper() != "NONE" and len(location) < 50:
            return location

    except Exception as e:
        logger.debug(f"LLM location extraction error: {e}")

    return None


def extract_video_locations_batch(
    videos: List[Dict[str, Any]],
    config: Any = None,
    location_service: Any = None
) -> Dict[str, Any]:
    """
    Extract and resolve locations for multiple videos.

    Args:
        videos: List of video metadata dicts with 'file', 'title', 'keyword' keys
        config: Pipeline config
        location_service: Optional LocationService for geocoding

    Returns:
        Dict mapping video file paths to resolved GeoLocation dicts
    """
    results = {}

    for video in videos:
        file_path = video.get('file', video.get('path', ''))
        title = video.get('title', '')
        keyword = video.get('keyword', '')
        description = video.get('description', '')

        if not file_path:
            continue

        # Extract location from metadata
        location_name = extract_location_from_video_metadata(
            title=title,
            description=description,
            source_keyword=keyword,
            config=config
        )

        if location_name and location_service:
            try:
                # Resolve location using LocationService
                geo_location = location_service.disambiguate(
                    location_name,
                    context=f"{title} {keyword}"
                )
                if geo_location:
                    results[file_path] = geo_location
                    logger.info(f"Video location: '{title[:50]}' -> {geo_location.name}, {geo_location.country_name}")
            except Exception as e:
                logger.debug(f"Could not resolve location for '{title}': {e}")

    logger.info(f"Extracted locations for {len(results)}/{len(videos)} videos")
    return results
