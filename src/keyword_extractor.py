"""
LLM-Based Keyword Extractor for Stock Footage Search

Extracts optimized YouTube search keywords from voiceover text
using Claude/Gemini for intelligent expansion and refinement.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KeywordResult:
    """Result from keyword extraction"""
    keywords: List[str]
    segments_analyzed: int
    extraction_method: str
    entities: List[Dict] = field(default_factory=list)  # Named entities with type info
    topic: str = ""  # Detected main topic


class LLMKeywordExtractor:
    """
    Extracts stock footage search keywords from voiceover segments.
    Uses LLM for intelligent keyword generation optimized for YouTube search.
    """
    
    # Suffixes that improve YouTube footage search results
    FOOTAGE_SUFFIXES = [
        "4K footage",
        "news footage",
        "drone footage",
        "aerial footage",
        "stock footage",
        "documentary footage"
    ]
    
    SEGMENT_KEYWORD_PROMPT = """Extract ONE optimal YouTube search keyword for finding B-roll footage for this voiceover segment.

SEGMENT TEXT:
"{segment_text}"

DOCUMENTARY TOPIC: {topic}

RULES:
1. Return ONE keyword phrase (2-5 words) optimized for YouTube search
2. Focus on VISUAL elements that can be filmed
3. Include location/event names if mentioned
4. Add context words like "footage", "video", "news" if helpful
5. If segment is very short/generic, use the topic context

Return ONLY the keyword phrase, nothing else.

KEYWORD:"""

    BATCH_SEGMENT_KEYWORDS_PROMPT = """Extract ONE search keyword for EACH voiceover segment to find matching B-roll footage.

DOCUMENTARY TOPIC: {topic}

SEGMENTS:
{segments_text}

RULES:
1. Return ONE keyword per segment (2-5 words each)
2. Focus on VISUAL elements - things a camera can capture
3. Include specific names, places, events if mentioned
4. Generic segments should relate to the overall topic
5. Keywords should work for YouTube/stock footage search

OUTPUT FORMAT:
Return a JSON array with one keyword string per segment, in order.
Example: ["flooding aftermath drone", "rescue workers boat", "damaged buildings aerial"]

KEYWORDS:"""
    
    KEYWORD_EXTRACTION_PROMPT = """You are an expert stock footage researcher for disaster documentaries.

Analyze this voiceover script and extract the BEST YouTube search keywords to find relevant B-roll footage.

VOICEOVER TEXT:
{voiceover_text}

CRITICAL EXTRACTION PRIORITIES:
1. **NAMED ENTITIES FIRST** - Extract ALL proper nouns:
   - Person names → search as "[Name] [topic context]" (e.g., "President Marcos typhoon response")
   - Organization names → search as "[Org] [visual action]" (e.g., "Red Cross rescue operation")
   - Place names → search as "[Place] [event type] footage" (e.g., "Manila flooding 2024")
   
2. **DATES & TIME PERIODS** - For each date/year mentioned:
   - Recent events: "[Event] [Year] footage" (e.g., "Japan earthquake 2024")
   - Historical: "[Event] [Year] archive footage" (e.g., "Tohoku tsunami 2011")
   
3. **LOCATIONS WITH CONTEXT** - Be geographically specific:
   - Country + City + Event (e.g., "Philippines Cebu typhoon damage")
   - Region + Specific feature (e.g., "Pacific Ring of Fire volcanic activity")

4. **VISUAL SCENES** - Things a camera can capture:
   - Disasters: destruction, flooding, fires, earthquakes, eruptions
   - Aftermath: damage, debris, collapsed buildings, rescue operations
   - Human elements: evacuation, rescue workers, survivors, relief efforts

RULES:
- Every named person/place/date MUST generate at least one keyword
- Combine entities with the documentary topic for relevance
- Keywords should find FOOTAGE, not articles (add "footage", "video", "news" suffixes)
- Maximum 3-5 words per keyword phrase
- Avoid abstract concepts that don't translate to visuals
- No duplicate or overly similar keywords

OUTPUT FORMAT:
Return ONLY a JSON array of keyword strings, nothing else.
Maximum {max_keywords} keywords total.
Group similar searches to avoid redundancy.

Example for a script about "Typhoon Yolanda hit Tacloban in 2013":
["Typhoon Yolanda Tacloban footage", "Haiyan 2013 destruction", "Philippines typhoon aftermath", "Tacloban flooding aerial", "storm surge Visayas"]

KEYWORDS:"""

    ENTITY_EXTRACTION_PROMPT = """Extract all named entities from this documentary script that would need specific footage.

TEXT:
{text}

Extract and categorize:
1. PEOPLE: Names of officials, experts, victims, responders mentioned
2. PLACES: Cities, regions, countries, specific locations (buildings, landmarks)
3. ORGANIZATIONS: Government agencies, NGOs, companies, institutions
4. DATES: Specific dates, years, time periods mentioned
5. EVENTS: Named disasters, operations, incidents (e.g., "Operation Damayan", "Great East Japan Earthquake")

For each entity, suggest a search keyword that would find relevant footage.

OUTPUT FORMAT - JSON object:
{{
  "people": [
    {{"name": "Person Name", "context": "role/relevance", "search_keyword": "Person Name topic footage"}}
  ],
  "places": [
    {{"name": "Place Name", "context": "what happened there", "search_keyword": "Place event footage"}}
  ],
  "organizations": [
    {{"name": "Org Name", "context": "their role", "search_keyword": "Org action footage"}}
  ],
  "dates": [
    {{"date": "2024", "context": "what event", "search_keyword": "event 2024 footage"}}
  ],
  "events": [
    {{"name": "Event Name", "context": "description", "search_keyword": "Event Name footage"}}
  ]
}}

Only include entities that are RELEVANT to finding documentary footage.
ENTITIES:"""

    KEYWORD_EXPANSION_PROMPT = """You are an expert stock footage researcher.

Given these initial keywords extracted from a documentary script, EXPAND and REFINE them for better YouTube search results.

INITIAL KEYWORDS:
{initial_keywords}

DOCUMENTARY TOPIC: {topic}

CRITICAL RULES:
1. **PRESERVE ALL NAMED ENTITIES** - Do NOT remove keywords containing:
   - Person names (presidents, officials, experts, victims)
   - Specific place names (cities, regions, landmarks)
   - Organization names (agencies, NGOs, companies)
   - Specific dates or years
   - Named events or operations
   
2. EXPAND with variants:
   - Add geographic specificity where missing
   - Add time-based variants (e.g., "2024", "recent", "archive")
   - Add visual variants (e.g., "aerial view", "close up", "timelapse", "news footage")
   - Add related visual phenomena

3. REFINE for searchability:
   - Keep 3-5 words per keyword
   - Ensure keywords find VIDEO FOOTAGE, not articles
   - Remove abstract concepts that don't show visually
   - Remove true duplicates (but keep location/time variants)

OUTPUT FORMAT:
Return ONLY a JSON array of refined keyword strings.
Maximum {max_keywords} keywords total.
KEEP ALL entity-based keywords, then add expanded variants.

REFINED KEYWORDS:"""

    def __init__(self, config):
        """Initialize with config"""
        self.config = config
        self.llm_client = None
        self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client based on config"""
        import os
        llm_config = self.config.llm
        
        # Try to get API key from config first, then environment
        api_key = llm_config.api_key
        
        if llm_config.provider == 'anthropic':
            if not api_key:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("No Anthropic API key found - set ANTHROPIC_API_KEY environment variable")
                return
            try:
                import anthropic
                self.llm_client = anthropic.Anthropic(api_key=api_key)
                self.llm_provider = 'anthropic'
                logger.info("Using Anthropic Claude for keyword extraction")
            except ImportError:
                logger.warning("anthropic package not installed")
        elif llm_config.provider == 'google':
            if not api_key:
                api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("No Gemini API key found - set GEMINI_API_KEY environment variable")
                return
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self.llm_client = genai.GenerativeModel(llm_config.model)
                self.llm_provider = 'google'
                logger.info("Using Google Gemini for keyword extraction")
            except ImportError:
                logger.warning("google-generativeai package not installed")
        
        if not self.llm_client:
            logger.warning("No LLM client available - falling back to TF-IDF extraction")
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM and return response text"""
        try:
            if self.llm_provider == 'anthropic':
                response = self.llm_client.messages.create(
                    model=self.config.llm.model,
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            elif self.llm_provider == 'google':
                response = self.llm_client.generate_content(prompt)
                return response.text
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "[]"
    
    def _parse_keywords_json(self, response: str) -> List[str]:
        """Parse JSON array from LLM response"""
        import json
        
        # Clean up response
        response = response.strip()
        
        # Try to find JSON array in response
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            try:
                keywords = json.loads(match.group())
                if isinstance(keywords, list):
                    return [str(k).strip() for k in keywords if k]
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to parse line by line
        lines = response.split('\n')
        keywords = []
        for line in lines:
            line = line.strip().strip('-').strip('•').strip('"').strip("'").strip(',')
            if line and not line.startswith('[') and not line.startswith(']'):
                keywords.append(line)
        
        return keywords[:50]  # Safety limit
    
    def _combine_voiceover_text(self, segments: List[Dict]) -> str:
        """Combine all voiceover segments into single text"""
        texts = []
        for seg in segments:
            text = seg.get('text', '')
            if text:
                texts.append(text.strip())
        return '\n'.join(texts)
    
    def _detect_topic(self, text: str) -> str:
        """Try to detect the main topic from the text"""
        # Simple heuristic - look for common disaster keywords
        disaster_keywords = {
            'earthquake': 'earthquake',
            'tsunami': 'tsunami',
            'typhoon': 'typhoon',
            'hurricane': 'hurricane',
            'volcano': 'volcanic eruption',
            'eruption': 'volcanic eruption',
            'flood': 'flooding',
            'wildfire': 'wildfire',
            'tornado': 'tornado',
            'landslide': 'landslide',
            'avalanche': 'avalanche'
        }
        
        text_lower = text.lower()
        for keyword, topic in disaster_keywords.items():
            if keyword in text_lower:
                return topic
        
        return "natural disaster documentary"
    
    def extract_keywords(
        self,
        segments: List[Dict],
        max_keywords: int = 50,
        expand: bool = True
    ) -> KeywordResult:
        """
        Extract search keywords from voiceover segments.
        
        Args:
            segments: List of voiceover segments with 'text' field
            max_keywords: Maximum number of keywords to return
            expand: Whether to expand keywords with LLM refinement
        
        Returns:
            KeywordResult with extracted keywords
        """
        if not segments:
            return KeywordResult(keywords=[], segments_analyzed=0, extraction_method="none")
        
        # Combine all text
        full_text = self._combine_voiceover_text(segments)
        
        if not full_text:
            return KeywordResult(keywords=[], segments_analyzed=len(segments), extraction_method="none")
        
        # Detect topic for context
        topic = self._detect_topic(full_text)
        
        # Use LLM if available
        if self.llm_client:
            return self._extract_with_llm(full_text, topic, max_keywords, expand, len(segments))
        else:
            return self._extract_with_tfidf(segments, max_keywords)
    
    def _extract_with_llm(
        self,
        text: str,
        topic: str,
        max_keywords: int,
        expand: bool,
        num_segments: int
    ) -> KeywordResult:
        """Extract keywords using LLM with entity-aware extraction"""
        
        # Step 1: Extract named entities first (people, places, dates, orgs)
        logger.info("Extracting named entities with LLM...")
        entity_keywords, raw_entities = self._extract_entities(text[:8000], topic)
        logger.info(f"Entity extraction: {len(entity_keywords)} entity-based keywords, {len(raw_entities)} entities")
        
        # Step 2: General keyword extraction
        prompt = self.KEYWORD_EXTRACTION_PROMPT.format(
            voiceover_text=text[:8000],  # Limit text length
            max_keywords=max_keywords
        )
        
        logger.info("Extracting general keywords with LLM...")
        response = self._call_llm(prompt)
        initial_keywords = self._parse_keywords_json(response)
        
        logger.info(f"General extraction: {len(initial_keywords)} keywords")
        
        # Step 3: Merge entity keywords FIRST (they're high priority)
        # Entity keywords come first to ensure they're included
        merged_keywords = entity_keywords + [k for k in initial_keywords if k not in entity_keywords]
        
        # Step 4: Expand if requested and we have keywords
        if expand and merged_keywords:
            expand_prompt = self.KEYWORD_EXPANSION_PROMPT.format(
                initial_keywords=merged_keywords[:30],  # Limit for expansion
                topic=topic,
                max_keywords=max_keywords
            )
            
            logger.info("Expanding keywords with LLM...")
            expand_response = self._call_llm(expand_prompt)
            expanded_keywords = self._parse_keywords_json(expand_response)
            
            if expanded_keywords:
                # Keep entity keywords at the front, add expanded ones
                all_keywords = entity_keywords + [k for k in expanded_keywords if k not in entity_keywords]
                keywords = list(dict.fromkeys(all_keywords))[:max_keywords]
            else:
                keywords = list(dict.fromkeys(merged_keywords))[:max_keywords]
        else:
            keywords = list(dict.fromkeys(merged_keywords))[:max_keywords]
        
        logger.info(f"Final keywords: {len(keywords)} (including {len(entity_keywords)} entity-based)")
        
        return KeywordResult(
            keywords=keywords,
            segments_analyzed=num_segments,
            extraction_method="llm_entity_aware",
            entities=raw_entities,
            topic=topic
        )
    
    def _extract_entities(self, text: str, topic: str) -> tuple:
        """
        Extract named entities and convert to search keywords.
        
        Returns:
            Tuple of (keywords: List[str], raw_entities: List[Dict])
        """
        import json
        
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=text)
        raw_entities = []
        
        try:
            response = self._call_llm(prompt)
            
            # Try to parse JSON response
            response = response.strip()
            
            # Find JSON object in response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                entities = json.loads(json_str)
                
                keywords = []
                
                # Extract search keywords from each entity type
                for person in entities.get('people', []):
                    kw = person.get('search_keyword', '')
                    name = person.get('name', '')
                    if kw and len(kw) > 3:
                        keywords.append(kw)
                    if name:
                        raw_entities.append({
                            'text': name,
                            'type': 'PERSON',
                            'search_keyword': kw,
                            'context': person.get('context', '')
                        })
                
                for place in entities.get('places', []):
                    kw = place.get('search_keyword', '')
                    name = place.get('name', '')
                    if kw and len(kw) > 3:
                        keywords.append(kw)
                    if name:
                        raw_entities.append({
                            'text': name,
                            'type': 'GPE',  # Geo-political entity
                            'search_keyword': kw,
                            'context': place.get('context', '')
                        })
                
                for org in entities.get('organizations', []):
                    kw = org.get('search_keyword', '')
                    name = org.get('name', '')
                    if kw and len(kw) > 3:
                        keywords.append(kw)
                    if name:
                        raw_entities.append({
                            'text': name,
                            'type': 'ORG',
                            'search_keyword': kw,
                            'context': org.get('context', '')
                        })
                
                for date in entities.get('dates', []):
                    kw = date.get('search_keyword', '')
                    name = date.get('date', date.get('name', ''))
                    if kw and len(kw) > 3:
                        keywords.append(kw)
                    if name:
                        raw_entities.append({
                            'text': name,
                            'type': 'DATE',
                            'search_keyword': kw,
                            'context': date.get('context', '')
                        })
                
                for event in entities.get('events', []):
                    kw = event.get('search_keyword', '')
                    name = event.get('name', '')
                    if kw and len(kw) > 3:
                        keywords.append(kw)
                    if name:
                        raw_entities.append({
                            'text': name,
                            'type': 'EVENT',
                            'search_keyword': kw,
                            'context': event.get('context', '')
                        })
                
                # Deduplicate keywords while preserving order
                seen = set()
                unique_keywords = []
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower not in seen:
                        seen.add(kw_lower)
                        unique_keywords.append(kw)
                
                return unique_keywords, raw_entities
                
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
        
        return [], []
    
    def _extract_with_tfidf(
        self,
        segments: List[Dict],
        max_keywords: int
    ) -> KeywordResult:
        """Fallback: extract keywords using TF-IDF"""
        from keyword_extractor import KeywordWeightExtractor
        
        logger.info("Falling back to TF-IDF keyword extraction")
        
        # Convert segments to text strings
        texts = []
        for seg in segments:
            text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
            if text:
                texts.append(text.strip())
        
        if not texts:
            logger.warning("No text found in segments for TF-IDF extraction")
            return KeywordResult(
                keywords=[],
                segments_analyzed=len(segments),
                extraction_method="tfidf"
            )
        
        # Override auto_detect_count in config temporarily
        original_count = self.config.keyword_weights.auto_detect_count
        self.config.keyword_weights.auto_detect_count = max_keywords
        
        extractor = KeywordWeightExtractor(self.config)
        
        # Get weighted terms (returns tuple of boost_terms, penalty_terms, tfidf_scores)
        boost_terms, _, tfidf_scores = extractor.extract_weighted_terms(texts)
        
        # Restore original config
        self.config.keyword_weights.auto_detect_count = original_count
        
        # Convert to search keywords (use boost terms which are top TF-IDF terms)
        keywords = list(boost_terms)[:max_keywords]
        
        return KeywordResult(
            keywords=keywords,
            segments_analyzed=len(segments),
            extraction_method="tfidf"
        )
    
    def extract_keyword_per_segment(
        self,
        segments: List[Dict],
        topic: str = ""
    ) -> List[str]:
        """
        Extract ONE keyword per segment for precise B-roll matching.
        
        Args:
            segments: List of segment dicts with 'text' key
            topic: Documentary topic for context
        
        Returns:
            List of keywords (one per segment, in order)
        """
        import json
        
        if not self.llm_client:
            logger.warning("No LLM client - using simple keyword extraction")
            return self._simple_segment_keywords(segments, topic)
        
        # Prepare segments text
        segment_texts = []
        for i, seg in enumerate(segments):
            text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
            text = text.strip()
            if text:
                segment_texts.append(f"{i+1}. \"{text}\"")
            else:
                segment_texts.append(f"{i+1}. [empty]")
        
        # Process in batches to avoid token limits
        batch_size = 50
        all_keywords = []
        
        for batch_start in range(0, len(segment_texts), batch_size):
            batch_end = min(batch_start + batch_size, len(segment_texts))
            batch = segment_texts[batch_start:batch_end]
            
            prompt = self.BATCH_SEGMENT_KEYWORDS_PROMPT.format(
                topic=topic or "documentary",
                segments_text="\n".join(batch)
            )
            
            try:
                response = self._call_llm(prompt)
                response = response.strip()
                
                # Parse JSON array
                start = response.find('[')
                end = response.rfind(']') + 1
                
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    keywords = json.loads(json_str)
                    
                    # Ensure we have one keyword per segment in batch
                    while len(keywords) < (batch_end - batch_start):
                        keywords.append(topic or "documentary footage")
                    
                    all_keywords.extend(keywords[:batch_end - batch_start])
                else:
                    # Fallback for this batch
                    logger.warning(f"Could not parse batch {batch_start}-{batch_end}, using fallback")
                    for i in range(batch_start, batch_end):
                        seg_text = segments[i].get('text', '') if isinstance(segments[i], dict) else getattr(segments[i], 'text', '')
                        all_keywords.append(self._extract_simple_keyword(seg_text, topic))
                        
            except Exception as e:
                logger.warning(f"Batch keyword extraction failed: {e}")
                # Fallback for this batch
                for i in range(batch_start, batch_end):
                    seg_text = segments[i].get('text', '') if isinstance(segments[i], dict) else getattr(segments[i], 'text', '')
                    all_keywords.append(self._extract_simple_keyword(seg_text, topic))
        
        return all_keywords
    
    def _simple_segment_keywords(self, segments: List[Dict], topic: str) -> List[str]:
        """Simple fallback: extract keywords without LLM"""
        keywords = []
        for seg in segments:
            text = seg.get('text', '') if isinstance(seg, dict) else getattr(seg, 'text', '')
            keywords.append(self._extract_simple_keyword(text, topic))
        return keywords
    
    def _extract_simple_keyword(self, text: str, topic: str) -> str:
        """Extract a simple keyword from segment text"""
        if not text or len(text.strip()) < 10:
            return topic or "documentary footage"
        
        # Remove common stopwords and get key terms
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
            'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once', 'here',
            'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'until', 'while', 'although', 'though', 'this', 'that',
            'these', 'those', 'what', 'which', 'who', 'whom', 'whose', 'it', 'its',
            'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
            'his', 'she', 'her', 'i', 'me', 'my'
        }
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter stopwords and get unique meaningful words
        meaningful = []
        seen = set()
        for word in words:
            if word not in stopwords and word not in seen:
                meaningful.append(word)
                seen.add(word)
        
        # Take top 3-4 words
        if meaningful:
            keyword = ' '.join(meaningful[:4])
            # Capitalize proper nouns (words that were capitalized in original)
            original_words = text.split()
            for orig_word in original_words:
                if orig_word[0].isupper() and orig_word.lower() in keyword.lower():
                    keyword = keyword.replace(orig_word.lower(), orig_word)
            return keyword
        
        return topic or "documentary footage"
    
    def add_footage_suffixes(
        self,
        keywords: List[str],
        suffixes: List[str] = None
    ) -> List[str]:
        """
        Optionally add footage-related suffixes to keywords.
        This can improve search results but increases keyword count.
        
        Args:
            keywords: Base keywords
            suffixes: List of suffixes to add (uses defaults if None)
        
        Returns:
            Expanded keyword list with suffixes
        """
        if suffixes is None:
            # Use a subset of default suffixes
            suffixes = ["4K footage", "news footage"]
        
        expanded = []
        for kw in keywords:
            expanded.append(kw)
            for suffix in suffixes:
                expanded.append(f"{kw} {suffix}")
        
        return expanded


def extract_keywords_from_srt(
    srt_path: str,
    config,
    max_keywords: int = 50
) -> KeywordResult:
    """
    Convenience function to extract keywords from SRT file.
    
    Args:
        srt_path: Path to SRT file
        config: Pipeline config
        max_keywords: Maximum keywords to extract
    
    Returns:
        KeywordResult with extracted keywords
    """
    import srt
    
    # Parse SRT
    with open(srt_path, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))
    
    # Convert to segments format
    segments = [{'text': sub.content} for sub in subtitles]
    
    # Extract keywords
    extractor = LLMKeywordExtractor(config)
    return extractor.extract_keywords(segments, max_keywords=max_keywords)


def extract_keyword_per_segment_from_srt(
    srt_path: str,
    config,
    topic: str = ""
) -> List[str]:
    """
    Extract ONE keyword per SRT segment for precise B-roll matching.
    
    Args:
        srt_path: Path to SRT file
        config: Pipeline config
        topic: Documentary topic for context (auto-detected if not provided)
    
    Returns:
        List of keywords (one per segment, in order)
    """
    import srt
    
    # Parse SRT
    with open(srt_path, 'r', encoding='utf-8') as f:
        subtitles = list(srt.parse(f.read()))
    
    # Convert to segments format
    segments = [{'text': sub.content} for sub in subtitles]
    
    # Auto-detect topic from first few segments if not provided
    if not topic:
        first_text = ' '.join([s['text'] for s in segments[:5]])
        # Simple topic detection: use most common nouns
        words = re.findall(r'\b[A-Z][a-z]+\b', first_text)
        if words:
            from collections import Counter
            common = Counter(words).most_common(2)
            topic = ' '.join([w for w, _ in common])
    
    # Extract per-segment keywords
    extractor = LLMKeywordExtractor(config)
    return extractor.extract_keyword_per_segment(segments, topic=topic)

def find_keyword_matches(
    voiceover_keywords: List[str],
    video_keywords: List[str],
    visual_keywords: List[str] = None
) -> Tuple[float, bool, bool]:
    """
    Find keyword overlap between voiceover and video.
    Returns (boost_score, is_keyword_match, is_visual_match)
    """
    vo_set = set(k.lower() for k in voiceover_keywords)
    vid_set = set(k.lower() for k in video_keywords)
    vis_set = set(k.lower() for k in (visual_keywords or []))
    
    # Text keyword match
    text_overlap = len(vo_set & vid_set)
    is_keyword_match = text_overlap > 0
    
    # Visual keyword match
    visual_overlap = len(vo_set & vis_set)
    is_visual_match = visual_overlap > 0
    
    # Calculate boost score
    boost = 0.0
    if is_keyword_match:
        boost += 0.05 * text_overlap  # 5% boost per matching keyword
    if is_visual_match:
        boost += 0.03 * visual_overlap  # 3% boost per visual match
    
    return min(boost, 0.2), is_keyword_match, is_visual_match  # Cap at 20% boost