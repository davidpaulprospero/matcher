"""
Validation Pass

Pass 3: Cross-validate detected chapters with independent LLM call.
"""

import logging
from typing import List, Dict, Any, Optional

from ..models import ChapterCandidate, ValidationResult, MissedChapter
from ..prompts import format_validation_prompt, format_chapters_for_validation
from ..chunking import create_indexed_text

logger = logging.getLogger(__name__)


def run_validation(
    chapters: List[ChapterCandidate],
    segments: List[Dict[str, Any]],
    llm_client: Any,
    config: Any,
) -> List[ChapterCandidate]:
    """
    Validate detected chapters with independent LLM call.

    Args:
        chapters: Chapters to validate
        segments: Original segments
        llm_client: LLM client for API calls
        config: Configuration

    Returns:
        Validated/corrected chapters
    """
    if not chapters:
        return chapters

    if not _should_run_validation(config):
        logger.debug("Validation pass disabled in config")
        return chapters

    # Format chapters for prompt
    chapters_summary = format_chapters_for_validation(chapters)

    # Create indexed text (truncate if needed)
    indexed_text = create_indexed_text(segments)
    max_chars = 8000
    if len(indexed_text) > max_chars:
        indexed_text = indexed_text[:max_chars] + "\n..."

    # Call LLM for validation
    validation_result = _call_validation_llm(
        chapters_summary=chapters_summary,
        indexed_text=indexed_text,
        llm_client=llm_client,
    )

    if validation_result is None:
        logger.warning("Validation LLM call failed, keeping original chapters")
        return chapters

    # Log validation failures if any
    validations = validation_result.get('validations', [])
    for val in validations:
        if not isinstance(val, dict):
            continue

        chapter_id = val.get('chapter_id')
        if not val.get('boundary_correct', True):
            logger.warning(
                f"Chapter validation failure: boundary incorrect for chapter {chapter_id}, "
                f"suggested corrections: start={val.get('suggested_start')}, end={val.get('suggested_end')}"
            )
        if not val.get('title_accurate', True):
            logger.warning(
                f"Chapter validation failure: title inaccurate for chapter {chapter_id}, "
                f"suggested title: '{val.get('suggested_title')}'"
            )
        if not val.get('content_coherent', True):
            logger.warning(
                f"Chapter validation failure: content not coherent for chapter {chapter_id}, "
                f"reason: {val.get('reason', 'unknown')}"
            )

    # Apply validation results
    validated = _apply_validation_results(
        chapters=chapters,
        validation_result=validation_result,
        segments=segments,
    )

    logger.info(f"Validation complete: {len(validated)} chapters")
    return validated


def _call_validation_llm(
    chapters_summary: str,
    indexed_text: str,
    llm_client: Any,
) -> Optional[Dict[str, Any]]:
    """Call LLM for chapter validation."""
    prompt = format_validation_prompt(
        chapters_summary=chapters_summary,
        indexed_text=indexed_text,
    )

    try:
        from src.llm_client import LLMRequest, ResponseFormat

        request = LLMRequest(
            prompt=prompt,
            response_format=ResponseFormat.JSON,
            max_tokens=2000,
            cache_key_prefix="chapter_validation"
        )
        response = llm_client.generate(request)

        if response.parsed_data and isinstance(response.parsed_data, dict):
            return response.parsed_data

    except Exception as e:
        logger.warning(f"Validation LLM call failed: {e}")

    return None


def _apply_validation_results(
    chapters: List[ChapterCandidate],
    validation_result: Dict[str, Any],
    segments: List[Dict[str, Any]],
) -> List[ChapterCandidate]:
    """Apply validation results to chapters."""
    validations = validation_result.get('validations', [])
    missed_chapters = validation_result.get('missed_chapters', [])
    merge_suggestions = validation_result.get('merge_suggestions', [])
    split_suggestions = validation_result.get('split_suggestions', [])

    # Create chapter lookup by ID
    chapter_by_id = {ch.chapter_id: ch for ch in chapters}

    # Apply individual validations
    for val in validations:
        if not isinstance(val, dict):
            continue

        chapter_id = val.get('chapter_id')
        if chapter_id is None or chapter_id not in chapter_by_id:
            continue

        chapter = chapter_by_id[chapter_id]

        # Update boundaries if suggested
        if not val.get('boundary_correct', True):
            suggested_start = val.get('suggested_start')
            suggested_end = val.get('suggested_end')

            if suggested_start is not None:
                chapter.start_segment_idx = max(0, suggested_start)
            if suggested_end is not None:
                chapter.end_segment_idx = max(chapter.start_segment_idx, suggested_end)

            chapter.boundary_reasoning += " [Corrected by validation]"

        # Update title if suggested
        if not val.get('title_accurate', True):
            suggested_title = val.get('suggested_title')
            if suggested_title:
                chapter.title = suggested_title

        # Update confidence based on validation
        if val.get('content_coherent', True):
            chapter.confidence = min(1.0, chapter.confidence + 0.1)
        else:
            chapter.confidence = max(0.1, chapter.confidence - 0.2)

        # Store validation score
        if chapter.confidence_details is None:
            chapter.confidence_details = {}
        chapter.confidence_details['validation_score'] = (
            1.0 if val.get('content_coherent', True) else 0.5
        )

    # Handle merge suggestions
    chapters = list(chapter_by_id.values())

    # Log merge and split suggestions
    if merge_suggestions:
        logger.info(f"Processing {len(merge_suggestions)} merge suggestions from validation")
    if split_suggestions:
        logger.info(f"Processing {len(split_suggestions)} split suggestions from validation")

    chapters = _apply_merge_suggestions(chapters, merge_suggestions)

    # Add missed chapters
    chapters = _add_missed_chapters(chapters, missed_chapters, segments)

    # Sort by start index and renumber
    chapters.sort(key=lambda c: c.start_segment_idx)
    for i, ch in enumerate(chapters):
        ch.chapter_id = i

    return chapters


def _apply_merge_suggestions(
    chapters: List[ChapterCandidate],
    merge_suggestions: List[List[int]],
) -> List[ChapterCandidate]:
    """Merge suggested chapter pairs."""
    if not merge_suggestions:
        return chapters

    # Track which chapters have been merged
    merged_ids = set()
    result = []

    for suggestion in merge_suggestions:
        if not isinstance(suggestion, list) or len(suggestion) < 2:
            continue

        # Find chapters to merge
        to_merge = []
        for ch_id in suggestion:
            for ch in chapters:
                if ch.chapter_id == ch_id and ch_id not in merged_ids:
                    to_merge.append(ch)
                    merged_ids.add(ch_id)

        if len(to_merge) >= 2:
            # Create merged chapter
            merged = _merge_chapters(to_merge)
            result.append(merged)

    # Add non-merged chapters
    for ch in chapters:
        if ch.chapter_id not in merged_ids:
            result.append(ch)

    return result


def _merge_chapters(chapters: List[ChapterCandidate]) -> ChapterCandidate:
    """Merge multiple chapters into one."""
    if not chapters:
        return ChapterCandidate()

    if len(chapters) == 1:
        return chapters[0]

    # Sort by start index
    chapters = sorted(chapters, key=lambda c: c.start_segment_idx)

    # Combine properties
    all_topics = []
    all_visual = []
    all_context = []
    for ch in chapters:
        all_topics.extend(ch.topics)
        all_visual.extend(ch.visual_keywords)
        all_context.extend(ch.context_keywords)

    # Use first chapter's title with "& more" if multiple
    title = chapters[0].title
    if len(chapters) > 1:
        title = f"{title} & {chapters[-1].title}"

    return ChapterCandidate(
        chapter_id=chapters[0].chapter_id,
        start_segment_idx=chapters[0].start_segment_idx,
        end_segment_idx=chapters[-1].end_segment_idx,
        title=title,
        topics=list(dict.fromkeys(all_topics)),  # Dedupe
        location_name=chapters[0].location_name,
        location_type=chapters[0].location_type,
        visual_keywords=list(dict.fromkeys(all_visual)),
        context_keywords=list(dict.fromkeys(all_context)),
        confidence=sum(c.confidence for c in chapters) / len(chapters),
        detection_strategy=chapters[0].detection_strategy,
        boundary_reasoning="Merged from validation suggestion",
    )


def _add_missed_chapters(
    chapters: List[ChapterCandidate],
    missed_chapters: List[Dict[str, Any]],
    segments: List[Dict[str, Any]],
) -> List[ChapterCandidate]:
    """Add chapters that validation detected but initial detection missed."""
    if not missed_chapters:
        return chapters

    for missed in missed_chapters:
        if not isinstance(missed, dict):
            continue

        start = missed.get('start_segment_idx', 0)
        end = missed.get('end_segment_idx', 0)
        title = missed.get('suggested_title', 'Untitled')

        # Check if this range overlaps with existing chapters
        overlaps = False
        for ch in chapters:
            if not (end < ch.start_segment_idx or start > ch.end_segment_idx):
                overlaps = True
                break

        if not overlaps:
            new_chapter = ChapterCandidate(
                chapter_id=len(chapters),
                start_segment_idx=max(0, start),
                end_segment_idx=min(len(segments) - 1, end),
                title=title,
                confidence=0.6,  # Lower confidence for LLM-added chapters
                detection_strategy='validation_added',
                boundary_reasoning=missed.get('reasoning', 'Added by validation pass'),
            )
            chapters.append(new_chapter)
            logger.info(f"Added missed chapter: '{title}' ({start}-{end})")

    return chapters


def _should_run_validation(config: Any) -> bool:
    """Check if validation pass should run."""
    if config is None:
        return True

    if hasattr(config, 'matching'):
        matching = config.matching
        if hasattr(matching, 'chapter_detection'):
            ch_config = matching.chapter_detection
            if isinstance(ch_config, dict):
                return ch_config.get('use_validation_pass', True)
            return getattr(ch_config, 'use_validation_pass', True)

    return True
