"""
Chapter Detection Prompts

Centralized LLM prompts for multi-pass chapter detection.
Each prompt is designed to extract specific information with confidence scores.
"""

# =============================================================================
# PASS 1: INITIAL DETECTION PROMPTS
# =============================================================================

TOPIC_CHAPTER_PROMPT = """Analyze this transcript and identify MAJOR topic transitions.

TRANSCRIPT ({total_segments} segments):
{indexed_text}

For each chapter/topic section, identify:
1. start_segment_idx: Where this topic BEGINS
2. end_segment_idx: Where this topic ENDS (inclusive)
3. title: Brief descriptive title (2-6 words)
4. topics: 2-4 topic keywords (lowercase)
5. boundary_reasoning: WHY you placed the boundary here (1 sentence)
6. confidence: "low", "medium", or "high"

RULES:
- Chapters must be consecutive (no gaps or overlaps)
- Minimum chapter size: {min_segments} segments
- Look for clear topic shifts, not just mention changes
- A topic should be the PRIMARY focus, not just mentioned
- If content is cohesive, return fewer chapters (2-3)
- If content has clear sections, return more (up to {max_chapters})

Return ONLY a JSON array:
[
  {{
    "start_segment_idx": 0,
    "end_segment_idx": 5,
    "title": "Introduction to Topic",
    "topics": ["topic1", "topic2"],
    "boundary_reasoning": "Clear transition from intro to main content",
    "confidence": "high"
  }}
]

If the entire transcript is one cohesive topic, return a single chapter covering all segments."""


LOCATION_CHAPTER_PROMPT = """Analyze this transcript for chapters where a SPECIFIC LOCATION is the PRIMARY SUBJECT.

IMPORTANT DISTINCTION:
- Location as SUBJECT: "Paris is famous for its art museums and romantic atmosphere"
- Location as MENTION: "I was born in Paris but moved to London at age 5"

Only detect chapters where the location IS the subject, not just mentioned.

TRANSCRIPT ({total_segments} segments):
{indexed_text}

For LOCATION-FOCUSED chapters only, extract:
1. start_segment_idx, end_segment_idx (inclusive)
2. location_name: Specific place name (e.g., "Paris", "Grand Canyon", "Japan")
3. location_type: One of "city", "country", "landmark", "region", "natural_feature"
4. visual_keywords: 2-4 specific landmarks or features (e.g., "Eiffel Tower", "Shibuya Crossing")
5. context_keywords: 2-3 thematic keywords (e.g., "romantic", "historic", "modern")
6. title: Chapter title
7. boundary_reasoning: Why this is a location-focused chapter
8. confidence: "low", "medium", or "high"

RULES:
- ONLY include chapters where location IS the primary subject
- Skip general introductions, conclusions, or tangents
- Distinguish same-named places (Paris, France vs Paris, Texas) using context
- Minimum chapter size: {min_segments} segments

Return ONLY a JSON array (empty [] if no location-focused chapters):
[
  {{
    "start_segment_idx": 0,
    "end_segment_idx": 5,
    "location_name": "Paris",
    "location_type": "city",
    "visual_keywords": ["Eiffel Tower", "Louvre", "Notre Dame"],
    "context_keywords": ["romantic", "art", "cuisine"],
    "title": "Exploring Paris",
    "boundary_reasoning": "Segments discuss Paris landmarks and culture",
    "confidence": "high"
  }}
]"""


NARRATIVE_CHAPTER_PROMPT = """Analyze this transcript for NARRATIVE STRUCTURE chapters.

Look for these patterns:
- Introduction / Body / Conclusion
- Temporal markers: "first", "then", "finally", "in the beginning", "at the end"
- List structure: "first point", "secondly", "number one", "the third reason"
- Story arc: setup, conflict, resolution

TRANSCRIPT ({total_segments} segments):
{indexed_text}

For each structural chapter, extract:
1. start_segment_idx, end_segment_idx (inclusive)
2. title: Structural role (e.g., "Introduction", "Main Argument", "Conclusion")
3. topics: 2-4 keywords about the section content
4. boundary_reasoning: Structural marker that indicates this section
5. confidence: "low", "medium", or "high"

RULES:
- Focus on STRUCTURE, not just topic changes
- Look for explicit markers (transition words, list numbers)
- Minimum chapter size: {min_segments} segments

Return ONLY a JSON array:
[
  {{
    "start_segment_idx": 0,
    "end_segment_idx": 3,
    "title": "Introduction",
    "topics": ["overview", "thesis"],
    "boundary_reasoning": "Opening segment introduces main topic",
    "confidence": "medium"
  }}
]"""


# =============================================================================
# PASS 3: VALIDATION PROMPT
# =============================================================================

VALIDATION_PROMPT = """Review these detected chapters for accuracy.

DETECTED CHAPTERS:
{chapters_summary}

FULL TRANSCRIPT:
{indexed_text}

For each chapter, evaluate:
1. chapter_id: The chapter number (0-indexed)
2. boundary_correct: true/false - Are start/end segments accurate?
3. suggested_start: (only if boundary_correct=false) Correct start segment
4. suggested_end: (only if boundary_correct=false) Correct end segment
5. title_accurate: true/false - Does title match content?
6. suggested_title: (only if title_accurate=false) Better title
7. content_coherent: true/false - Is the content within the chapter consistent?
8. notes: Any issues or observations (1 sentence max)

Also identify:
- missed_chapters: Chapters we should have detected but didn't
- merge_suggestions: Chapter IDs that should be combined (e.g., [[0,1], [3,4]])
- split_suggestions: Chapter IDs that should be split, with split point segment index

Return ONLY JSON:
{{
  "validations": [
    {{
      "chapter_id": 0,
      "boundary_correct": true,
      "title_accurate": true,
      "content_coherent": true,
      "notes": ""
    }}
  ],
  "missed_chapters": [
    {{
      "start_segment_idx": 10,
      "end_segment_idx": 15,
      "suggested_title": "Missing Topic",
      "reasoning": "Clear topic shift not captured"
    }}
  ],
  "merge_suggestions": [],
  "split_suggestions": []
}}"""


# =============================================================================
# CONTENT TYPE DETECTION
# =============================================================================

CONTENT_TYPE_PROMPT = """Analyze this transcript and determine its primary content type.

TRANSCRIPT (first 2000 chars):
{text_sample}

Content types:
- "travel": Geographic locations are primary subjects, travel descriptions, destination guides
- "educational": Teaching, tutorials, step-by-step instructions, learning objectives
- "documentary": Investigative, historical, informational about events/people/phenomena
- "narrative": Story-driven, temporal flow, character arcs, fiction or personal stories
- "general": None of the above clearly dominant

Analyze the content and return ONLY JSON:
{{
  "content_type": "travel",
  "confidence": "high",
  "reasoning": "Multiple location names mentioned as subjects, travel verbs present"
}}"""


# =============================================================================
# FALLBACK SUMMARY
# =============================================================================

SUMMARY_TITLE_PROMPT = """Generate a brief title (2-5 words) summarizing this transcript.

TRANSCRIPT (first 1500 chars):
{text_sample}

Return ONLY the title, no quotes or explanation.
Example: "Climate Change Documentary" or "Paris Travel Guide" or "Python Programming Tutorial"

Title:"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_topic_prompt(
    indexed_text: str,
    total_segments: int,
    min_segments: int = 3,
    max_chapters: int = 20
) -> str:
    """Format the topic detection prompt with parameters."""
    return TOPIC_CHAPTER_PROMPT.format(
        indexed_text=indexed_text,
        total_segments=total_segments,
        min_segments=min_segments,
        max_chapters=max_chapters,
    )


def format_location_prompt(
    indexed_text: str,
    total_segments: int,
    min_segments: int = 3
) -> str:
    """Format the location detection prompt with parameters."""
    return LOCATION_CHAPTER_PROMPT.format(
        indexed_text=indexed_text,
        total_segments=total_segments,
        min_segments=min_segments,
    )


def format_narrative_prompt(
    indexed_text: str,
    total_segments: int,
    min_segments: int = 3
) -> str:
    """Format the narrative detection prompt with parameters."""
    return NARRATIVE_CHAPTER_PROMPT.format(
        indexed_text=indexed_text,
        total_segments=total_segments,
        min_segments=min_segments,
    )


def format_validation_prompt(
    chapters_summary: str,
    indexed_text: str
) -> str:
    """Format the validation prompt with detected chapters."""
    return VALIDATION_PROMPT.format(
        chapters_summary=chapters_summary,
        indexed_text=indexed_text,
    )


def format_content_type_prompt(text_sample: str) -> str:
    """Format the content type detection prompt."""
    return CONTENT_TYPE_PROMPT.format(text_sample=text_sample[:2000])


def format_summary_title_prompt(text_sample: str) -> str:
    """Format the summary title generation prompt."""
    return SUMMARY_TITLE_PROMPT.format(text_sample=text_sample[:1500])


def format_chapters_for_validation(chapters: list) -> str:
    """Format chapters list for validation prompt."""
    lines = []
    for ch in chapters:
        if isinstance(ch, dict):
            start = ch.get('start_segment_idx', 0)
            end = ch.get('end_segment_idx', 0)
            title = ch.get('title', 'Untitled')
        else:
            start = ch.start_segment_idx
            end = ch.end_segment_idx
            title = ch.title

        lines.append(f"[Chapter {len(lines)}] segments {start}-{end}: \"{title}\"")

    return "\n".join(lines)
