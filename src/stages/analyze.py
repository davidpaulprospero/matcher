"""
Analyze Stage - Voiceover Analysis and Keyword Extraction

Stage 1 of the video matching pipeline:
- Loads voiceover segments (SRT or audio transcription)
- Extracts keywords using LLM
- Detects topic context
- Extracts named entities
- Detects chapters and location chapters
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage
from ..cache import AnalyzeCache, AnalyzeResult

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, VoiceoverSegment, DetectedChapter

logger = logging.getLogger(__name__)


@register_stage
class AnalyzeStage(Stage):
    """
    Analyzes voiceover and extracts keywords, entities, and topic context.

    Inputs:
        - state.voiceover_path: Path to voiceover file (SRT or audio)

    Outputs:
        - state.voiceover_segments: List of VoiceoverSegment
        - state.keywords: List of extracted keywords
        - state.topic_context: Detected topic string
        - state.extracted_entities: List of named entities
        - state.location_chapters: Location-focused chapter info
    """

    name = "ANALYZE"
    description = "Analyze voiceover and extract keywords"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the analyze stage"""
        warnings = []

        try:
            # Get voiceover path
            voiceover_path = state.voiceover_path
            if not voiceover_path:
                return StageResult.fail("No voiceover path specified")

            if not Path(voiceover_path).exists():
                return StageResult.fail(f"Voiceover file not found: {voiceover_path}")

            num_keywords = config.keyword.max_keywords

            print(f"\n  ─── Stage 1: ANALYZE VOICEOVER ───")

            # Initialize analyze cache in project directory
            project_dir = getattr(state, 'project_dir', None)
            if project_dir:
                cache_dir = Path(project_dir) / ".cache" / "analyze"
            else:
                cache_dir = Path(".cache") / "analyze"
            analyze_cache = AnalyzeCache(cache_dir)

            # Check cache for existing analysis
            cached_result = analyze_cache.get_by_file(voiceover_path, config)
            if cached_result:
                print(f"  ✓ Using cached analysis for {Path(voiceover_path).name}")
                return self._restore_from_cache(state, cached_result, config, checkpoint)

            # Load voiceover segments
            segments = self._load_voiceover_segments(voiceover_path, config)
            if not segments:
                return StageResult.fail("No segments found in voiceover")

            state.voiceover_segments = segments
            print(f"  ✓ {len(segments)} segments found")

            # Extract keywords
            print(f"\n  Extracting keywords (max {num_keywords})...")
            keywords, entities, topic = self._extract_keywords(
                segments, num_keywords, config
            )

            state.keywords = keywords
            state.extracted_entities = entities
            state.topic_context = topic

            print(f"  ✓ {len(keywords)} keywords extracted")
            if topic:
                print(f"  Detected topic: {topic}")
            if entities:
                print(f"  Entities found: {len(entities)}")

            # Detect listicle/chapter structure using LLM (new)
            chapter_config = getattr(config, 'chapter_detection', None)
            chapter_detection_enabled = getattr(chapter_config, 'enabled', True) if chapter_config else True

            if chapter_detection_enabled:
                # Pass project_dir for proper cache location
                project_dir = getattr(state, 'project_dir', None)
                listicle_chapters = self._detect_listicle_chapters(segments, config, topic, project_dir=project_dir)
                if listicle_chapters:
                    state.chapters = listicle_chapters
                    state.is_listicle = True
                    state.chapter_detection_enabled = True

                    # Add chapter keywords to state.keywords (prioritized)
                    chapter_keywords = []
                    for ch in listicle_chapters:
                        for kw in ch.keywords[:3]:  # Top 3 per chapter
                            if kw not in chapter_keywords and kw not in keywords:
                                chapter_keywords.append(kw)

                    if chapter_keywords:
                        # Prepend chapter keywords (high priority)
                        keywords = chapter_keywords[:30] + [k for k in keywords if k not in chapter_keywords]
                        state.keywords = keywords
                        print(f"  ✓ Added {len(chapter_keywords)} chapter-specific keywords")

            # Detect chapters if enabled (legacy)
            if config.matching.chapter_matching_enabled:
                chapters = self._detect_chapters(segments, topic, config)
                # Chapters are stored on segments, not state directly

            # Detect location chapters if enabled
            location_chapters = self._detect_location_chapters(segments, topic, config)
            state.location_chapters = location_chapters

            # Print summaries for verification
            segment_dicts = [self._segment_to_dict(s) for s in segments]
            self._print_chapter_summary(location_chapters, segment_dicts)
            self._print_entity_summary(entities)
            if state.chapters:
                self._print_listicle_summary(state.chapters)

            # Prepare checkpoint data
            segment_dicts_for_cp = [self._segment_to_dict(s) for s in segments]
            checkpoint_data = {
                'keywords': keywords,
                'segments': segment_dicts_for_cp,
                'topic_context': topic,
                'entities': entities,
                'segment_count': len(segments),
                'location_chapters': [
                    self._location_chapter_to_dict(lc) for lc in location_chapters
                ] if location_chapters else [],
                # New: Listicle chapter detection
                'chapters': [ch.to_dict() for ch in state.chapters] if state.chapters else [],
                'is_listicle': state.is_listicle,
            }

            # Save to analyze cache for future runs
            cache_result = AnalyzeResult(
                keywords=keywords,
                entities=entities,
                topic_context=topic,
                segments=segment_dicts_for_cp,
                chapters=checkpoint_data['chapters'],
                is_listicle=state.is_listicle,
                location_chapters=checkpoint_data['location_chapters'],
            )
            analyze_cache.set_by_file(voiceover_path, config, cache_result)
            print(f"  ✓ Cached analysis for future runs")

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Analyze stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if analyze stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore analyze stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # Restore keywords
            state.keywords = data.get('keywords', [])
            state.topic_context = data.get('topic_context', '')
            state.extracted_entities = data.get('entities', [])

            # Restore segments
            from ..state import VoiceoverSegment
            segments = []
            for seg_dict in data.get('segments', []):
                segments.append(VoiceoverSegment(
                    index=seg_dict.get('index', 0),
                    start=seg_dict.get('start', 0.0),
                    end=seg_dict.get('end', 0.0),
                    text=seg_dict.get('text', ''),
                ))
            state.voiceover_segments = segments

            # Restore location chapters
            state.location_chapters = data.get('location_chapters', [])

            # Restore listicle chapters (new)
            from ..state import DetectedChapter
            chapters_data = data.get('chapters', [])
            if chapters_data:
                state.chapters = [
                    DetectedChapter(
                        name=ch.get('name', ''),
                        corrected_name=ch.get('corrected_name', ''),
                        rank=ch.get('rank'),
                        start_segment=ch.get('start_segment', 0),
                        end_segment=ch.get('end_segment', 0),
                        keywords=ch.get('keywords', []),
                        description=ch.get('description', '')
                    )
                    for ch in chapters_data
                ]
                state.is_listicle = data.get('is_listicle', False)
                logger.info(f"Restored {len(state.chapters)} listicle chapters")

            logger.info(f"Restored ANALYZE: {len(state.keywords)} keywords, "
                       f"{len(state.voiceover_segments)} segments")

            # Print summaries for verification (same as run method)
            segment_dicts = data.get('segments', [])
            self._print_chapter_summary(state.location_chapters, segment_dicts)
            self._print_entity_summary(state.extracted_entities)
            if state.chapters:
                self._print_listicle_summary(state.chapters)

            return True

        except Exception as e:
            logger.warning(f"Failed to restore ANALYZE: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        if not state.voiceover_path:
            return "No voiceover path specified in state"
        if not Path(state.voiceover_path).exists():
            return f"Voiceover file not found: {state.voiceover_path}"
        return None

    def _restore_from_cache(
        self,
        state: 'PipelineState',
        cached: 'AnalyzeResult',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> 'StageResult':
        """
        Restore state from cached analysis result.

        This is used when the same voiceover file has been analyzed before
        with the same config settings.
        """
        from ..state import VoiceoverSegment, DetectedChapter

        # Restore segments
        segments = []
        for seg_dict in cached.segments:
            segments.append(VoiceoverSegment(
                index=seg_dict.get('index', 0),
                start=seg_dict.get('start', 0.0),
                end=seg_dict.get('end', 0.0),
                text=seg_dict.get('text', ''),
            ))
        state.voiceover_segments = segments

        # Restore keywords and entities
        state.keywords = cached.keywords
        state.extracted_entities = cached.entities
        state.topic_context = cached.topic_context

        # Restore location chapters
        state.location_chapters = cached.location_chapters

        # Restore listicle chapters
        if cached.chapters:
            state.chapters = [
                DetectedChapter(
                    name=ch.get('name', ''),
                    corrected_name=ch.get('corrected_name', ''),
                    rank=ch.get('rank'),
                    start_segment=ch.get('start_segment', 0),
                    end_segment=ch.get('end_segment', 0),
                    keywords=ch.get('keywords', []),
                    description=ch.get('description', '')
                )
                for ch in cached.chapters
            ]
            state.is_listicle = cached.is_listicle
            state.chapter_detection_enabled = True

        print(f"  ✓ {len(segments)} segments restored from cache")
        print(f"  ✓ {len(cached.keywords)} keywords")
        if cached.topic_context:
            print(f"  Detected topic: {cached.topic_context}")
        if cached.entities:
            print(f"  Entities found: {len(cached.entities)}")
        if cached.chapters:
            print(f"  ✓ {len(cached.chapters)} chapters restored")

        # Print summaries
        self._print_chapter_summary(cached.location_chapters, cached.segments)
        self._print_entity_summary(cached.entities)
        if state.chapters:
            self._print_listicle_summary(state.chapters)

        # Prepare checkpoint data (same format as normal run)
        checkpoint_data = {
            'keywords': cached.keywords,
            'segments': cached.segments,
            'topic_context': cached.topic_context,
            'entities': cached.entities,
            'segment_count': len(cached.segments),
            'location_chapters': cached.location_chapters,
            'chapters': cached.chapters,
            'is_listicle': cached.is_listicle,
        }

        return StageResult.ok(checkpoint_data, [])

    # === Helper Methods ===

    def _load_voiceover_segments(
        self,
        voiceover_path: str,
        config: 'Config'
    ) -> List['VoiceoverSegment']:
        """Load voiceover segments from file"""
        from ..state import VoiceoverSegment

        path = Path(voiceover_path)
        segments = []

        if path.suffix.lower() == '.srt':
            # Parse SRT file
            segments = self._parse_srt(path)
        elif path.suffix.lower() in ('.mp3', '.wav', '.m4a', '.aac', '.flac'):
            # Transcribe audio file
            segments = self._transcribe_audio(path, config)
        else:
            logger.warning(f"Unknown voiceover format: {path.suffix}")

        return segments

    def _parse_srt(self, path: Path) -> List['VoiceoverSegment']:
        """Parse SRT subtitle file"""
        from ..state import VoiceoverSegment
        import re

        segments = []
        content = path.read_text(encoding='utf-8-sig')

        # SRT format: index, timestamp, text, blank line
        blocks = re.split(r'\n\n+', content.strip())

        for i, block in enumerate(blocks):
            lines = block.strip().split('\n')
            if len(lines) < 3:
                continue

            # Parse timestamp line: 00:00:00,000 --> 00:00:05,000
            timestamp_match = re.match(
                r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
                lines[1]
            )
            if not timestamp_match:
                continue

            h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, timestamp_match.groups())
            start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000

            # Join remaining lines as text
            text = ' '.join(lines[2:]).strip()

            segments.append(VoiceoverSegment(
                index=i,
                start=start,
                end=end,
                text=text,
            ))

        return segments

    def _transcribe_audio(
        self,
        path: Path,
        config: 'Config'
    ) -> List['VoiceoverSegment']:
        """Transcribe audio file to segments"""
        from ..state import VoiceoverSegment

        try:
            from ..transcription import transcribe_voiceover_audio, write_srt

            # Get VAD setting from config - default True for voiceover
            # VAD filters silence accurately, improving gap detection
            vad_filter = getattr(config.transcription, 'vad_filter', True)

            result = transcribe_voiceover_audio(
                str(path),
                model_name=config.transcription.model,
                compute_type=config.transcription.compute_type,
                vad_filter=vad_filter,
            )

            segments = []
            for i, seg in enumerate(result):
                segments.append(VoiceoverSegment(
                    index=i,
                    start=seg.get('start', 0.0),
                    end=seg.get('end', 0.0),
                    text=seg.get('text', ''),
                ))

            # Write SRT file alongside the audio file
            srt_path = path.with_suffix('.srt')
            write_srt(result, str(srt_path))
            logger.info(f"Wrote voiceover SRT: {srt_path}")

            return segments

        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            return []

    def _extract_keywords(
        self,
        segments: List['VoiceoverSegment'],
        max_keywords: int,
        config: 'Config'
    ) -> tuple:
        """Extract keywords from segments using per-segment method"""
        try:
            from ..keyword_extractor import LLMKeywordExtractor

            extractor = LLMKeywordExtractor(config=config)

            # Convert segments to dict format expected by extractor
            segment_dicts = [
                {'index': s.index, 'start': s.start, 'end': s.end, 'text': s.text}
                for s in segments
            ]

            # First, extract overall keywords and entities for context
            result = extractor.extract_keywords(segment_dicts, max_keywords=max_keywords)
            entities = result.entities if result.entities else []
            topic = result.topic if hasattr(result, 'topic') else ''

            # Detect topic from keywords if not provided
            if not topic and result.keywords:
                topic = self._detect_topic_from_keywords(result.keywords, config)

            # Use per-segment extraction for better, more specific keywords
            # This generates ONE keyword per segment instead of general keywords
            print(f"  Using per-segment keyword extraction for {len(segments)} segments...")
            keywords = extractor.extract_keyword_per_segment(segment_dicts, topic=topic)

            # Take unique keywords up to max_keywords limit
            unique_keywords = list(dict.fromkeys(keywords))[:max_keywords]

            logger.info(f"Per-segment extraction: {len(keywords)} total → {len(unique_keywords)} unique keywords")

            return unique_keywords, entities, topic

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            # TF-IDF fallback
            return self._tfidf_fallback(segments, max_keywords, config)

    def _tfidf_fallback(
        self,
        segments: List['VoiceoverSegment'],
        max_keywords: int,
        config: 'Config'
    ) -> tuple:
        """Fallback to TF-IDF keyword extraction"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            text = " ".join([s.text for s in segments])
            vectorizer = TfidfVectorizer(max_features=max_keywords, stop_words='english')
            vectorizer.fit_transform([text])
            keywords = list(vectorizer.get_feature_names_out())

            return keywords, [], ''

        except Exception as e:
            logger.error(f"TF-IDF fallback failed: {e}")
            return [], [], ''

    def _detect_topic_from_keywords(
        self,
        keywords: List[str],
        config: 'Config'
    ) -> str:
        """Detect overall topic from keywords"""
        if not keywords:
            return ''

        # Simple heuristic: join first few keywords
        topic_keywords = keywords[:5]
        return ', '.join(topic_keywords)

    def _detect_listicle_chapters(
        self,
        segments: List['VoiceoverSegment'],
        config: 'Config',
        topic: str = "",
        project_dir: str = None
    ) -> List['DetectedChapter']:
        """
        Detect listicle/ranking structure using LLM-based chapter detection.

        This handles content like "15 Fast Food Chains Dying" where each
        list item is a distinct chapter with its own keywords.
        """
        try:
            from ..chapter_detector import ChapterDetector, detect_chapters
            from ..state import DetectedChapter
            from ..llm_client import create_client_from_config, LLMRequest, ResponseFormat
            from pathlib import Path

            print(f"\n  Detecting listicle/chapter structure...")

            # Create LLM client - use project cache directory
            # project_dir is passed from the stage's run() method
            if project_dir:
                llm_cache_dir = str(Path(project_dir) / ".cache" / "llm_responses")
            else:
                # Fallback to config.cache_dir (may be relative)
                project_cache_dir = getattr(config, 'cache_dir', '.cache')
                llm_cache_dir = f"{project_cache_dir}/llm_responses"
            llm_client = create_client_from_config(config, cache_dir=llm_cache_dir)

            def llm_call_fn(prompt: str) -> str:
                request = LLMRequest(
                    prompt=prompt,
                    response_format=ResponseFormat.JSON,
                    temperature=0.3,
                    max_tokens=4000,
                    cache_key_prefix="chapter_detect",
                    use_cache=True
                )
                response = llm_client.generate(request)
                return response.text

            # Convert segments to dict format
            segment_dicts = [
                {'index': s.index, 'text': s.text, 'start': s.start, 'end': s.end}
                for s in segments
            ]

            # Run detection with topic context for better keywords
            result = detect_chapters(segment_dicts, llm_call_fn, topic=topic)

            if not result.is_listicle or not result.chapters:
                logger.info("No listicle structure detected")
                return []

            # Convert to DetectedChapter dataclass
            detected_chapters = []
            for ch in result.chapters:
                detected_chapter = DetectedChapter(
                    name=ch.name,
                    corrected_name=ch.corrected_name,
                    rank=ch.rank,
                    start_segment=ch.start_segment,
                    end_segment=ch.end_segment,
                    keywords=ch.keywords,
                    description=ch.description
                )
                detected_chapters.append(detected_chapter)

            print(f"  ✓ Detected {len(detected_chapters)} list items (listicle: {result.list_type})")

            return detected_chapters

        except ImportError as e:
            logger.warning(f"Chapter detector not available: {e}")
            return []
        except Exception as e:
            logger.warning(f"Listicle chapter detection failed: {e}")
            logger.exception(e)
            return []

    def _print_listicle_summary(self, chapters: List['DetectedChapter']) -> None:
        """Print summary of detected listicle chapters"""
        if not chapters:
            return

        print(f"\n  Listicle chapters ({len(chapters)} items):")
        for ch in chapters[:5]:  # Show first 5
            rank_str = f"#{ch.rank}" if ch.rank else ""
            kw_preview = ", ".join(ch.keywords[:2]) if ch.keywords else "no keywords"
            print(f"    {rank_str:4s} {ch.corrected_name:20s} (segs {ch.start_segment}-{ch.end_segment}) [{kw_preview}]")

        if len(chapters) > 5:
            print(f"    ... and {len(chapters) - 5} more")

    def _detect_chapters(
        self,
        segments: List['VoiceoverSegment'],
        topic: str,
        config: 'Config'
    ) -> List[Dict[str, Any]]:
        """Detect chapters in voiceover (legacy)"""
        try:
            from ..topic_extraction import ChapterDetector

            detector = ChapterDetector(config)

            segment_dicts = [
                {'index': s.index, 'start': s.start, 'end': s.end, 'text': s.text}
                for s in segments
            ]

            chapters = detector.detect_chapters(segment_dicts, overall_topic=topic)

            if chapters:
                print(f"  ✓ Found {len(chapters)} chapters")

            return chapters

        except Exception as e:
            logger.warning(f"Chapter detection failed: {e}")
            return []

    def _detect_location_chapters(
        self,
        segments: List['VoiceoverSegment'],
        topic: str,
        config: 'Config'
    ) -> List[Any]:
        """Detect location-focused chapters using enhanced multi-pass detection"""
        # Check if location matching is enabled
        location_config = getattr(config.matching, 'location_matching', None)
        if not location_config:
            return []

        if isinstance(location_config, dict):
            enabled = location_config.get('enabled', False)
        else:
            enabled = getattr(location_config, 'enabled', False)

        if not enabled:
            return []

        # Convert segments to dicts
        segment_dicts = [
            {'index': s.index, 'start': s.start, 'end': s.end, 'text': s.text}
            for s in segments
        ]

        # Check if enhanced chapter detection is enabled
        chapter_config = getattr(config.matching, 'chapter_detection', None)
        use_enhanced = True  # Default to enhanced
        if chapter_config:
            if isinstance(chapter_config, dict):
                use_enhanced = chapter_config.get('enabled', True)
            else:
                use_enhanced = getattr(chapter_config, 'enabled', True)

        try:
            from ..location_service import create_location_service
            location_service = create_location_service(config)

            if use_enhanced:
                # Use new multi-pass EnhancedChapterDetector
                return self._detect_chapters_enhanced(
                    segment_dicts, topic, config, location_service
                )
            else:
                # Fall back to legacy ChapterDetector
                return self._detect_chapters_legacy(
                    segment_dicts, topic, config, location_service
                )

        except Exception as e:
            logger.warning(f"Location chapter detection failed: {e}")
            return []

    def _detect_chapters_enhanced(
        self,
        segment_dicts: List[Dict[str, Any]],
        topic: str,
        config: 'Config',
        location_service: Any
    ) -> List[Any]:
        """Use enhanced multi-pass chapter detection"""
        try:
            from ..chapter_detection import EnhancedChapterDetector

            print(f"\n  Detecting chapters (enhanced multi-pass)...")

            detector = EnhancedChapterDetector(config)
            location_chapters = detector.detect_chapters(
                segments=segment_dicts,
                content_type='auto',
                location_service=location_service,
                overall_topic=topic
            )

            if location_chapters:
                print(f"  ✓ Found {len(location_chapters)} chapters")
                # Log confidence info
                for ch in location_chapters:
                    conf = ch.get('confidence', 0.8)
                    title = ch.get('title', 'Untitled')
                    strategy = ch.get('detection_strategy', 'unknown')
                    logger.debug(f"  Chapter '{title}': confidence={conf:.2f}, strategy={strategy}")

            return location_chapters

        except ImportError as e:
            logger.warning(f"Enhanced chapter detection not available: {e}")
            # Fall back to legacy
            return self._detect_chapters_legacy(segment_dicts, topic, config, location_service)

    def _detect_chapters_legacy(
        self,
        segment_dicts: List[Dict[str, Any]],
        topic: str,
        config: 'Config',
        location_service: Any
    ) -> List[Any]:
        """Use legacy single-pass chapter detection"""
        from ..topic_extraction import ChapterDetector

        print(f"\n  Detecting location-focused chapters (legacy)...")

        detector = ChapterDetector(config)
        location_chapters = detector.detect_location_chapters(
            segment_dicts,
            location_service=location_service,
            overall_topic=topic
        )

        if location_chapters:
            print(f"  ✓ Found {len(location_chapters)} location chapters")

        return location_chapters

    def _segment_to_dict(self, segment: 'VoiceoverSegment') -> Dict[str, Any]:
        """Convert segment to dict for checkpointing"""
        return {
            'index': segment.index,
            'start': segment.start,
            'end': segment.end,
            'text': segment.text,
            'duration': segment.duration,
        }

    def _location_chapter_to_dict(self, lc: Any) -> Dict[str, Any]:
        """Convert location chapter to dict for checkpointing"""
        if hasattr(lc, '__dict__'):
            return lc.__dict__
        return dict(lc) if isinstance(lc, dict) else {}

    def _print_chapter_summary(
        self,
        chapters: List[Any],
        segments: List[Any],
        title: str = "Chapter Detection Summary"
    ) -> None:
        """Print a detailed summary of detected chapters for verification."""
        if not chapters:
            # Silently return if no location chapters (listicle chapters printed separately)
            return

        print(f"\n  {'-' * 60}")
        print(f"  [Chapters] {title}")
        print(f"  {'-' * 60}")

        for i, ch in enumerate(chapters):
            # Handle both dict and object forms
            if isinstance(ch, dict):
                ch_title = ch.get('title') or ch.get('location_name') or f'Chapter {i+1}'
                start_idx = ch.get('start_segment_idx', 0)
                end_idx = ch.get('end_segment_idx', 0)
                visual_kw = ch.get('visual_keywords', [])
                context_kw = ch.get('context_keywords', [])
                topics = ch.get('topics', [])
                confidence = ch.get('confidence', 0.0)
                # Direct timestamp fields (if available)
                direct_start = ch.get('start_time')
                direct_end = ch.get('end_time')
            else:
                ch_title = getattr(ch, 'title', None) or getattr(ch, 'location_name', f'Chapter {i+1}')
                start_idx = getattr(ch, 'start_segment_idx', 0)
                end_idx = getattr(ch, 'end_segment_idx', 0)
                visual_kw = getattr(ch, 'visual_keywords', [])
                context_kw = getattr(ch, 'context_keywords', [])
                topics = getattr(ch, 'topics', [])
                confidence = getattr(ch, 'confidence', 0.0)
                direct_start = getattr(ch, 'start_time', None)
                direct_end = getattr(ch, 'end_time', None)

            # Get timestamps - prefer direct fields, fall back to segment lookup
            start_time = direct_start
            end_time = direct_end

            if start_time is None and segments and start_idx < len(segments):
                seg = segments[start_idx]
                start_time = seg.get('start', 0.0) if isinstance(seg, dict) else getattr(seg, 'start', 0.0)

            if end_time is None and segments and end_idx < len(segments):
                seg = segments[end_idx]
                end_time = seg.get('end', 0.0) if isinstance(seg, dict) else getattr(seg, 'end', 0.0)

            # Default to 0 if still None
            start_time = start_time or 0.0
            end_time = end_time or 0.0

            # Format time as HH:MM:SS for longer content, MM:SS for shorter
            def format_time(seconds: float) -> str:
                seconds = float(seconds)
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                if hours > 0:
                    return f"{hours}:{minutes:02d}:{secs:02d}"
                return f"{minutes}:{secs:02d}"

            start_str = format_time(start_time)
            end_str = format_time(end_time)
            duration = end_time - start_time
            duration_str = format_time(duration) if duration > 0 else "?"

            print(f"\n  Chapter {i+1}: {ch_title}")
            print(f"    Time:       {start_str} -> {end_str} ({duration_str})")
            print(f"    Segments:   {start_idx} - {end_idx}")
            print(f"    Confidence: {confidence:.0%}")

            if visual_kw:
                print(f"    Visual:     {', '.join(visual_kw[:5])}")
            if context_kw:
                print(f"    Context:    {', '.join(context_kw[:5])}")
            if topics:
                print(f"    Topics:     {', '.join(topics[:5])}")

        print(f"\n  {'-' * 60}")

    def _print_entity_summary(self, entities: List[Any]) -> None:
        """Print a summary of extracted entities for verification."""
        if not entities:
            print(f"\n  [Entities] None extracted")
            return

        # Group entities by type
        by_type: Dict[str, List[str]] = {}
        for e in entities:
            if isinstance(e, dict):
                etype = e.get('type', 'OTHER')
                name = e.get('text', '')
            else:
                etype = getattr(e, 'type', 'OTHER')
                name = getattr(e, 'text', '')
            if name:
                by_type.setdefault(etype, []).append(name)

        print(f"\n  [Entities] ({len(entities)} total):")
        for etype, names in sorted(by_type.items()):
            print(f"    {etype}: {', '.join(names[:8])}{' ...' if len(names) > 8 else ''}")
