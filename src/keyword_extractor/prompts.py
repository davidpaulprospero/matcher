"""
LLM prompt templates for keyword extraction.

Contains 6 prompts for different extraction tasks:
- SEGMENT_KEYWORD_PROMPT - Extract 1 keyword per segment
- BATCH_SEGMENT_KEYWORDS_PROMPT - Batch segment extraction
- KEYWORD_EXTRACTION_PROMPT - General keyword extraction
- ENTITY_EXTRACTION_PROMPT - Named entity → visual keyword conversion
- KEYWORD_EXPANSION_PROMPT - Refine and expand keywords
- TOPIC_DETECTION_PROMPT - Detect video topic/category
"""

# Prompt 1: Single segment keyword extraction
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

# Prompt 2: Batch segment keyword extraction
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

# Prompt 3: General keyword extraction
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

# Prompt 4: Named entity extraction
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

# Prompt 5: Keyword expansion/refinement
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

# Prompt 6: Grouped segment keyword extraction (N segments → 1 search query)
GROUPED_SEGMENT_KEYWORDS_PROMPT = """Generate ONE YouTube search query for this GROUP of voiceover segments.

DOCUMENTARY TOPIC: {topic}

SEGMENT GROUP ({segment_count} segments):
{segments_text}

=== YOUR TASK ===
Create ONE search query (3-6 words) that would find B-roll footage covering ALL these segments.

=== CRITICAL: TRANSLATE TO VISUAL TERMS ===
DO NOT copy phrases from the script. Translate narrative language to FILMABLE content.

❌ BAD: "the death of entertainment footage" (script phrase)
✅ GOOD: "Las Vegas casino floor 4K"

❌ BAD: "whale economy documentary" (abstract concept)
✅ GOOD: "VIP high roller casino footage"

=== RULES ===
1. Identify the COMMON VISUAL THEME across all segments
2. Focus on LOCATIONS, ACTIVITIES, or OBJECTS that a camera can capture
3. Use specific venue/place names if mentioned
4. Add search-friendly terms: "footage", "4K", "tour", "walkthrough", "aerial"
5. If segments are abstract, derive visual content from the topic

=== OUTPUT ===
Return ONLY the search query string (3-6 words). No explanation.

SEARCH QUERY:"""

# Prompt 7: Topic detection
TOPIC_DETECTION_PROMPT = """Analyze this voiceover transcript and identify the MAIN TOPIC in 2-5 words.

Transcript excerpt:
{text}

Respond with ONLY the topic (2-5 words), nothing else. Examples:
- "opioid crisis homelessness"
- "climate change documentary"
- "wildlife conservation Africa"
- "tech startup journey"
- "World War 2 veterans"

Topic:"""
