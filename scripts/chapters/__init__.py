"""Standalone chapter-detection scripts.

Public CLI entry points:
    - `detect_listicle`     - SRT -> listicle groups only
    - `detect_chapters_llm` - SRT -> LLM-detected chapters only
    - `detect_chapters`     - SRT -> unified chapters (listicle seeds the LLM)
    - `split_srt_by_chapter`- SRT + chapter JSON -> per-chapter SRTs

All scripts operate on a single SRT file and emit JSON; they are the building
blocks for the upcoming per-chapter pipelines.
"""
