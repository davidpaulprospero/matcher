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
    
    SEGMENT_KEYWORD_PROMPT = """Extract ONE visual YouTube search keyword for this voiceover segment.

SEGMENT TEXT:
"{segment_text}"

DOCUMENTARY TOPIC: {topic}

CRITICAL: Do NOT copy phrases from the script. Translate to VISUAL search terms.
- Script phrase "the death of entertainment" → Search "buffet prices Las Vegas"
- Script phrase "whale economy" → Search "VIP high roller casino"

RULES:
1. Return ONE keyword phrase (2-5 words) describing FILMABLE content
2. Focus on locations, activities, objects - things a camera can capture
3. Include specific venue/place names if mentioned
4. Add "footage", "4K", "tour", "walkthrough" if helpful
5. If segment is abstract, derive visual content from the topic

Return ONLY the keyword phrase, nothing else.

KEYWORD:"""

    BATCH_SEGMENT_KEYWORDS_PROMPT = """Extract ONE visual search keyword for EACH voiceover segment to find B-roll footage.

DOCUMENTARY TOPIC: {topic}

SEGMENTS:
{segments_text}

=== CRITICAL: DO NOT COPY SCRIPT PHRASES ===
Voiceover segments often contain narrative/poetic language. You must TRANSLATE these into VISUAL search terms.

TRANSLATION EXAMPLES:
- Script: "the death of free entertainment" → Search: "Las Vegas buffet prices sign"
- Script: "pricing out the dream" → Search: "expensive hotel room Las Vegas"
- Script: "the whale economy" → Search: "VIP high roller casino footage"
- Script: "a quiet confession" → Search: "interview talking head footage"
- Script: "everything changed" → Search: "Las Vegas strip before after"

RULES:
1. Each keyword must be 2-5 words describing something FILMABLE
2. Focus on locations, activities, objects, people - things a camera captures
3. If segment mentions a specific place/venue, use that name
4. If segment is abstract/narrative, derive visual content from the topic
5. Add "footage", "4K", "tour", "walkthrough" when it helps searchability

❌ NEVER output keywords like:
- "the quiet confession that changed everything footage"
- "pricing out the dream footage"
- Any phrase that sounds like a documentary title

✅ ALWAYS output keywords like:
- "casino floor slot machines"
- "Las Vegas strip night aerial"
- "hotel buffet food spread"

OUTPUT FORMAT:
Return a JSON array with one keyword string per segment, in order.
Example: ["casino floor walkthrough", "Las Vegas strip aerial", "hotel lobby crowd"]

KEYWORDS:"""
    
    KEYWORD_EXTRACTION_PROMPT = """You are an expert stock footage researcher. Your job is to generate YouTube search keywords that will find DOWNLOADABLE B-roll footage.

VOICEOVER TEXT:
{voiceover_text}

=== CRITICAL WARNING ===
DO NOT extract narrative phrases or section titles from the script!
Voiceovers contain poetic/thematic language that won't find footage.
You must TRANSLATE abstract themes into CONCRETE VISUAL CONTENT.

❌ BAD KEYWORDS (narrative phrases - won't find footage):
- "the quiet confession that changed everything" (script phrase)
- "pricing out the dream" (metaphor)
- "the death of free entertainment" (thematic title)
- "the whale economy" (abstract concept)
- "what really happened" (narrative hook)
- "the truth about" (clickbait phrase)

✅ GOOD KEYWORDS (visual/searchable - will find footage):
- "Las Vegas casino floor walkthrough"
- "Bellagio fountains night 4K"
- "slot machine gameplay footage"
- "Las Vegas buffet tour video"
- "MGM Grand hotel lobby"
- "Fremont Street neon signs"
- "high roller VIP casino"

=== EXTRACTION CATEGORIES ===

1. **SPECIFIC LOCATIONS** - Named venues, landmarks, cities
   - Hotels/Casinos: "Caesars Palace exterior", "Venetian gondola ride"
   - Landmarks: "Las Vegas sign tourists", "High Roller observation wheel"
   - Streets/Areas: "Las Vegas strip timelapse", "Fremont Street canopy"

2. **FILMABLE ACTIVITIES** - Actions a camera can capture
   - "blackjack table dealing footage"
   - "slot machine jackpot win"
   - "Las Vegas pool party crowd"
   - "buffet food spread walkthrough"

3. **ESTABLISHING SHOTS** - Aerials, atmospherics, ambience
   - "Las Vegas aerial night drone"
   - "casino floor crowd ambience"
   - "neon signs Las Vegas strip"
   - "desert highway Nevada"

4. **NAMED ENTITIES** - People, companies, events (with visual context)
   - "Steve Wynn interview footage" (if mentioned)
   - "Cirque du Soleil Las Vegas show"
   - "MGM Resorts properties tour"

=== VALIDATION CHECK ===
Before adding each keyword, ask: "Can a camera film this? Would a videographer understand what to shoot?"
- "the whale economy" → NO (abstract metaphor, not filmable)
- "high roller gambling VIP room" → YES (specific filmable scene)
- "pricing out the dream" → NO (narrative phrase)
- "expensive Las Vegas hotel suite tour" → YES (filmable content)

=== OUTPUT ===
Return ONLY a JSON array of {max_keywords} searchable keywords.
Maximum 5 words per keyword. Every keyword must describe filmable content.

KEYWORDS:"""

    ENTITY_EXTRACTION_PROMPT = """Extract named entities from this script and generate VISUAL search keywords for each.

TEXT:
{text}

=== IMPORTANT: Generate FILMABLE search keywords ===
Each search_keyword must describe something a camera can capture on YouTube.

Extract and categorize:
1. PEOPLE: Names mentioned → search_keyword should find interviews, speeches, or footage of them
2. PLACES: Specific locations → search_keyword should find walkthrough, aerial, or tour footage
3. ORGANIZATIONS: Companies/agencies → search_keyword should find their buildings, events, or operations
4. DATES: Years/periods → search_keyword should find archival or news footage from that time
5. EVENTS: Named events → search_keyword should find coverage or documentary footage

SEARCH KEYWORD EXAMPLES:
- Person "Steve Wynn" → "Steve Wynn interview" or "Wynn Las Vegas opening"
- Place "Las Vegas Strip" → "Las Vegas Strip walkthrough 4K" or "Las Vegas Strip aerial night"
- Organization "MGM Resorts" → "MGM Grand Las Vegas tour" or "MGM casino floor"
- Date "1990s" → "Las Vegas 1990s archive footage" or "vintage Las Vegas casino"
- Event "Fremont Street renovation" → "Fremont Street canopy construction" or "Fremont Experience opening"

OUTPUT FORMAT - JSON object:
{{
  "people": [
    {{"name": "Person Name", "context": "role", "search_keyword": "Person Name interview footage"}}
  ],
  "places": [
    {{"name": "Venue Name", "context": "significance", "search_keyword": "Venue Name walkthrough 4K"}}
  ],
  "organizations": [
    {{"name": "Company", "context": "role", "search_keyword": "Company building tour"}}
  ],
  "dates": [
    {{"date": "1990s", "context": "event", "search_keyword": "location 1990s archive"}}
  ],
  "events": [
    {{"name": "Event", "context": "description", "search_keyword": "Event footage video"}}
  ]
}}

Only include entities where you can generate a SEARCHABLE, VISUAL keyword.
ENTITIES:"""

    KEYWORD_EXPANSION_PROMPT = """You are an expert stock footage researcher. Expand and refine these B-roll search keywords.

INITIAL KEYWORDS:
{initial_keywords}

DOCUMENTARY TOPIC: {topic}

=== FIRST: REMOVE ANY ABSTRACT/NARRATIVE KEYWORDS ===
Delete any keywords that slipped through that are NOT visually searchable:
❌ Remove: "the quiet confession", "pricing out the dream", "death of entertainment"
❌ Remove: "whale economy", "the truth about", "what happened to"
❌ Remove: Any phrase that sounds like a documentary chapter title

=== THEN: EXPAND GOOD KEYWORDS ===
For keywords that ARE visual/searchable, add variants:
- Visual style: "aerial", "drone", "4K", "timelapse", "walkthrough"
- Specificity: "casino" → "Bellagio casino floor", "MGM Grand casino"
- Time of day: "Las Vegas strip" → "Las Vegas strip night", "Las Vegas strip sunset"
- Activity: "hotel" → "hotel lobby", "hotel pool", "hotel room tour"

=== PRESERVE NAMED ENTITIES ===
Keep all keywords with specific names:
- Place names: "Caesars Palace", "Fremont Street", "Bellagio"
- Company names: "MGM Resorts", "Wynn Las Vegas"
- Person names (with visual context): "Steve Wynn interview"

=== QUALITY CHECK ===
Every keyword in output must pass this test:
"Can I find this on YouTube?" and "Would a camera operator know what to film?"

Example refinement:
- INPUT: ["Las Vegas casinos", "whale economy footage", "buffet"]
- OUTPUT: ["Las Vegas casino floor walkthrough", "Bellagio casino interior", "Las Vegas buffet tour 4K", "MGM Grand buffet spread"]
(Note: "whale economy footage" was removed as abstract)

OUTPUT FORMAT:
Return ONLY a JSON array of {max_keywords} refined, searchable keywords.

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
        max_tokens = getattr(self.config.llm, 'max_tokens', 2000)
        try:
            if self.llm_provider == 'anthropic':
                response = self.llm_client.messages.create(
                    model=self.config.llm.model,
                    max_tokens=max_tokens,
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

    def _validate_visual_keywords(self, keywords: List[str]) -> List[str]:
        """
        Filter out abstract/narrative keywords that won't find B-roll footage.

        Returns only keywords that describe filmable, searchable content.
        """
        # Patterns that indicate abstract/narrative content (won't find footage)
        ABSTRACT_PATTERNS = [
            r'\b(the\s+)?quiet\s+confession',
            r'\b(the\s+)?death\s+of\b',
            r'\bpricing\s+out\b',
            r'\bthe\s+truth\s+about\b',
            r'\bthe\s+problem\s+with\b',
            r'\bwhat\s+happened\s+to\b',
            r'\bthe\s+rise\s+and\s+fall\b',
            r'\bchanged\s+everything\b',
            r'\bthe\s+secret\b',
            r'\bthe\s+real\s+reason\b',
            r'\bwhale\s+economy\b',
            r'\bthe\s+end\s+of\b',
            r'\bthe\s+future\s+of\b',
            r'\bthe\s+cost\s+of\b',
            r'\bthe\s+price\s+of\b',
            r'\ba\s+new\s+era\b',
            r'\bthe\s+untold\s+story\b',
            r'\bhidden\s+truth\b',
            r'\bbehind\s+the\s+scenes\b(?!\s+(footage|video|tour))',  # Allow "behind the scenes footage"
        ]

        # Words that strongly indicate visual/filmable content
        VISUAL_INDICATORS = [
            'hotel', 'casino', 'street', 'building', 'aerial', 'drone',
            'walkthrough', 'tour', 'footage', '4k', 'timelapse', 'night',
            'day', 'crowd', 'people', 'exterior', 'interior', 'lobby',
            'pool', 'restaurant', 'bar', 'show', 'performance', 'sign',
            'neon', 'lights', 'skyline', 'view', 'entrance', 'parking',
            'strip', 'boulevard', 'avenue', 'plaza', 'resort', 'tower',
            'fountain', 'buffet', 'slot', 'table', 'game', 'room',
            'suite', 'penthouse', 'rooftop', 'desert', 'highway',
        ]

        validated = []
        filtered_keywords = []  # Track what was filtered for debugging

        for kw in keywords:
            kw_lower = kw.lower()

            # Check if it matches abstract patterns
            is_abstract = False
            for pattern in ABSTRACT_PATTERNS:
                if re.search(pattern, kw_lower):
                    filtered_keywords.append(f"'{kw}' (abstract pattern)")
                    is_abstract = True
                    break

            if is_abstract:
                continue

            # Check if it's too long (likely a script phrase)
            word_count = len(kw.split())
            if word_count > 6:
                filtered_keywords.append(f"'{kw}' (too long: {word_count} words)")
                continue

            # Check for visual indicators or proper nouns (locations/names)
            has_visual = any(ind in kw_lower for ind in VISUAL_INDICATORS)
            words = kw.split()
            has_proper_noun = any(w[0].isupper() for w in words if len(w) > 2)

            # Accept if has visual indicator, proper noun, or is short/specific
            if has_visual or has_proper_noun or word_count <= 3:
                validated.append(kw)
            else:
                # Log but still accept - might be valid
                logger.debug(f"Keyword without visual indicator (kept): '{kw}'")
                validated.append(kw)

        if filtered_keywords:
            logger.info(f"Filtered {len(filtered_keywords)} abstract/narrative keywords: {filtered_keywords}")

        return validated

    def _combine_voiceover_text(self, segments: List[Dict]) -> str:
        """Combine all voiceover segments into single text"""
        texts = []
        for seg in segments:
            text = seg.get('text', '')
            if text:
                texts.append(text.strip())
        return '\n'.join(texts)
    
    def _detect_topic(self, text: str) -> str:
        """Detect the main topic from the text using LLM or heuristics"""
        
        # Try LLM-based topic detection first
        if self.llm_client:
            topic = self._detect_topic_llm(text[:3000])  # Limit text length
            if topic:
                return topic
        
        # Fallback to keyword-based detection
        return self._detect_topic_heuristic(text)
    
    def _detect_topic_llm(self, text: str) -> str:
        """Use LLM to detect the main topic"""
        prompt = f"""Analyze this voiceover transcript and identify the MAIN TOPIC in 2-5 words.

Transcript excerpt:
{text[:2000]}

Respond with ONLY the topic (2-5 words), nothing else. Examples:
- "opioid crisis homelessness"
- "climate change documentary"  
- "wildlife conservation Africa"
- "tech startup journey"
- "World War 2 veterans"

Topic:"""

        try:
            if hasattr(self, 'gemini_model') and self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                topic = response.text.strip().strip('"').strip("'")
                if topic and len(topic) < 100:  # Sanity check
                    logger.info(f"LLM detected topic: {topic}")
                    return topic
            elif hasattr(self, 'anthropic_client') and self.anthropic_client:
                # Use model from config
                anthropic_model = getattr(self.config.llm, 'anthropic_model', 'claude-3-haiku-20240307')
                response = self.anthropic_client.messages.create(
                    model=anthropic_model,
                    max_tokens=100,  # Short response for topic detection
                    messages=[{"role": "user", "content": prompt}]
                )
                topic = response.content[0].text.strip().strip('"').strip("'")
                if topic and len(topic) < 100:
                    logger.info(f"LLM detected topic: {topic}")
                    return topic
        except Exception as e:
            logger.debug(f"LLM topic detection failed: {e}")
        
        return ""
    
    def _detect_topic_heuristic(self, text: str) -> str:
        """Fallback heuristic-based topic detection"""
        text_lower = text.lower()
        
        # Expanded topic keywords
        topic_keywords = {
            # Disasters
            'earthquake': 'earthquake disaster',
            'tsunami': 'tsunami disaster',
            'typhoon': 'typhoon disaster',
            'hurricane': 'hurricane disaster',
            'volcano': 'volcanic eruption',
            'eruption': 'volcanic eruption',
            'flood': 'flooding disaster',
            'wildfire': 'wildfire disaster',
            'tornado': 'tornado disaster',
            'landslide': 'landslide disaster',
            # Social issues
            'homeless': 'homelessness crisis',
            'opioid': 'opioid crisis',
            'fentanyl': 'fentanyl crisis',
            'addiction': 'addiction crisis',
            'poverty': 'poverty documentary',
            'refugee': 'refugee crisis',
            'immigration': 'immigration documentary',
            # Environment
            'climate': 'climate change',
            'pollution': 'pollution documentary',
            'conservation': 'conservation documentary',
            'wildlife': 'wildlife documentary',
            'ocean': 'ocean documentary',
            # Technology
            'artificial intelligence': 'AI technology',
            'startup': 'tech startup',
            'innovation': 'technology innovation',
            # History
            'world war': 'World War documentary',
            'civil war': 'civil war documentary',
            'revolution': 'historical revolution',
            # Health
            'pandemic': 'pandemic documentary',
            'covid': 'COVID-19 documentary',
            'cancer': 'cancer documentary',
            'mental health': 'mental health documentary'
        }
        
        for keyword, topic in topic_keywords.items():
            if keyword in text_lower:
                return topic
        
        # If no specific topic found, try to extract from first sentences
        sentences = text.split('.')[:3]
        if sentences:
            # Look for proper nouns or key subjects
            first_text = ' '.join(sentences)
            # Simple extraction: take significant words
            words = [w for w in first_text.split() if len(w) > 4 and w[0].isupper()]
            if words:
                return ' '.join(words[:3]) + " documentary"
        
        return "documentary video content"
    
    def extract_keywords(
        self,
        segments: List[Dict],
        max_keywords: int = None,
        expand: bool = True
    ) -> KeywordResult:
        """
        Extract search keywords from voiceover segments.
        
        Args:
            segments: List of voiceover segments with 'text' field
            max_keywords: Maximum number of keywords to return (uses config default if None)
            expand: Whether to expand keywords with LLM refinement
        
        Returns:
            KeywordResult with extracted keywords
        """
        # Use config default if not specified
        if max_keywords is None:
            max_keywords = getattr(self.config.keyword, 'max_keywords', 30)
        
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

        # Validate keywords - filter out abstract/narrative phrases
        pre_validation_count = len(keywords)
        keywords = self._validate_visual_keywords(keywords)

        logger.info(f"Final keywords: {len(keywords)} (validated from {pre_validation_count}, {len(entity_keywords)} entity-based)")
        
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
    max_keywords: int = None
) -> KeywordResult:
    """
    Convenience function to extract keywords from SRT file.
    
    Args:
        srt_path: Path to SRT file
        config: Pipeline config
        max_keywords: Maximum keywords to extract (uses config default if None)
    
    Returns:
        KeywordResult with extracted keywords
    """
    import srt
    
    # Use config default if not specified
    if max_keywords is None:
        max_keywords = getattr(config.keyword, 'max_keywords', 30)
    
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
    visual_keywords: List[str] = None,
    keyword_boost: float = 0.05,
    visual_boost: float = 0.03,
    max_boost: float = 0.2
) -> Tuple[float, bool, bool]:
    """
    Find keyword overlap between voiceover and video.
    Returns (boost_score, is_keyword_match, is_visual_match)
    
    Args:
        voiceover_keywords: Keywords from voiceover
        video_keywords: Keywords from video transcript
        visual_keywords: Keywords from vision analysis
        keyword_boost: Boost per matching text keyword (default from config.matching.keyword_boost)
        visual_boost: Boost per matching visual keyword
        max_boost: Maximum total boost cap
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
        boost += keyword_boost * text_overlap
    if is_visual_match:
        boost += visual_boost * visual_overlap
    
    return min(boost, max_boost), is_keyword_match, is_visual_match
