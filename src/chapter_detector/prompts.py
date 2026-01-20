"""
LLM prompts for chapter detection.
"""

LISTICLE_DETECTION_PROMPT = '''Analyze this voiceover script and detect if it contains a numbered list or ranking structure.

IMPORTANT: This is ASR-transcribed text, so expect spelling errors and case issues.

Common ASR corrections to apply:
- "Keatsahut", "pizza hat", "pizzahut" → "Pizza Hut"
- "Wallburgers", "wahlburger" → "Wahlburgers"
- "dairy queen", "dairyqueen" → "Dairy Queen"
- "little Caesars", "little caesars" → "Little Caesars"
- "salad and go" → "Salad and Go"
- "mcdonalds", "mcdonald" → "McDonald's"
- "wendys", "wendy" → "Wendy's"
- "burger king" → "Burger King"
- "taco bell" → "Taco Bell"
- "jack in the box" → "Jack in the Box"
- "popeyes", "popeye" → "Popeyes"
- "kfc" → "KFC"
- "subway" → "Subway"
- "boston market" → "Boston Market"

Look for patterns like:
- "Number 15, Denny's" or "Number fifteen, Denny's"
- "#10 on our list"
- "Coming in at number 5"
- "In first place"
- "At number one"

For EACH list item found, extract:
1. The rank/position (as integer)
2. The entity name (original from text)
3. The corrected entity name (fix ASR errors, proper capitalization)
4. The segment index where this item STARTS (look for "Number X" pattern)
5. 3-5 search keywords specific to this item

Script with segment indices:
{indexed_text}

Return valid JSON only:
{{
  "is_listicle": true,
  "list_type": "countdown",
  "total_items": 15,
  "intro_end_segment": 16,
  "items": [
    {{
      "rank": 15,
      "name": "Denny's",
      "corrected_name": "Denny's",
      "start_segment": 17,
      "keywords": ["Denny's restaurant exterior", "Denny's sign", "24 hour diner", "grand slam breakfast"]
    }},
    {{
      "rank": 14,
      "name": "Wallburgers",
      "corrected_name": "Wahlburgers",
      "start_segment": 81,
      "keywords": ["Wahlburgers restaurant", "Mark Wahlberg burger", "celebrity restaurant"]
    }}
  ]
}}

If NOT a listicle, return:
{{
  "is_listicle": false,
  "list_type": null,
  "total_items": 0,
  "items": []
}}

CRITICAL: Return ONLY valid JSON, no markdown, no explanation.'''


CHAPTER_KEYWORDS_PROMPT = '''Generate search keywords for finding video footage of "{entity_name}".

Context: This is for a video about {topic}. The chapter discusses: {description}

Generate 5 search keywords that would find relevant B-roll footage on YouTube or stock video sites.

Requirements:
- First keyword MUST be "{entity_name}" + a visual descriptor (e.g., "restaurant exterior", "storefront", "sign")
- Include brand-specific visual elements
- Include general category terms as fallback
- Keywords should find FOOTAGE, not news or reviews

Return JSON array only:
["keyword 1", "keyword 2", "keyword 3", "keyword 4", "keyword 5"]'''
