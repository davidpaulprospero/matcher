"""
Test suite for keyword_extractor package.

Tests cover:
- Models (KeywordResult, PrioritizedKeyword)
- Prompts (6 LLM prompt templates)
- Validator (abstract pattern detection, visual keyword validation)
- Entity Extractor (entity parsing, visual keyword generation)
- Topic Detector (topic detection with LLM)
- Segment Processor (per-segment extraction, batch processing)
- Prioritizer (priority scoring logic)
- Utils (SRT parsing, keyword matching)
- Core (LLMKeywordExtractor integration)
"""
