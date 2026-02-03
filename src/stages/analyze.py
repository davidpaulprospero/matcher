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

from . import Stage, StageResult, register_stage, validate_required_state_attrs

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState, VoiceoverSegment

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
        """Execute the analyze stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['voiceover_path'], self.name)

        warnings = []

        try:
            # Get voiceover path
            voiceover_path = state.voiceover_path
            if not voiceover_path:
                return StageResult.fail("No voiceover path specified")

            if not Path(voiceover_path).exists():
                return StageResult.fail(f"Voiceover file not found: {voiceover_path}")

            print(f"\n  ─── Stage 1: ANALYZE VOICEOVER ───")

            # Load voiceover segments
            segments = self._load_voiceover_segments(voiceover_path, config)
            if not segments:
                return StageResult.fail("No segments found in voiceover")

            state.voiceover_segments = segments
            print(f"  ✓ {len(segments)} segments found")

            # Extract keywords
            print(f"\n  Extracting keywords...")
            keywords, entities, topic = self._extract_keywords(segments, config)

            state.keywords = keywords
            state.extracted_entities = entities
            state.topic_context = topic

            print(f"  ✓ {len(keywords)} keywords extracted")
            if topic:
                print(f"  Detected topic: {topic}")
            if entities:
                print(f"  Entities found: {len(entities)}")

            # Detect chapters if enabled
            if config.matching.chapter_matching_enabled:
                chapters = self._detect_chapters(segments, topic, config)
                # Chapters are stored on segments, not state directly

            # Detect location chapters if enabled
            location_chapters = self._detect_location_chapters(segments, topic, config)
            state.location_chapters = location_chapters

            # Prepare checkpoint data
            checkpoint_data = {
                'keywords': keywords,
                'segments': [self._segment_to_dict(s) for s in segments],
                'topic_context': topic,
                'entities': entities,
                'segment_count': len(segments),
                'location_chapters': [
                    self._location_chapter_to_dict(lc) for lc in location_chapters
                ] if location_chapters else [],
            }

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

            logger.info(f"Restored ANALYZE: {len(state.keywords)} keywords, "
                       f"{len(state.voiceover_segments)} segments")
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
            result = extractor.extract_keywords(segment_dicts)
            entities = result.entities if result.entities else []
            topic = result.topic if hasattr(result, 'topic') else ''

            # Detect topic from keywords if not provided
            if not topic and result.keywords:
                topic = self._detect_topic_from_keywords(result.keywords, config)

            # Use grouped extraction: every N segments → 1 search query
            # This balances specificity with search efficiency
            segments_per_query = getattr(config.keyword, 'segments_per_query', 3)
            print(f"  Using grouped keyword extraction ({segments_per_query} segments per query)...")
            keywords = extractor.extract_keywords_grouped(
                segment_dicts,
                topic=topic,
                segments_per_query=segments_per_query
            )

            # Deduplicate while preserving order
            unique_keywords = list(dict.fromkeys(keywords))

            logger.info(f"Grouped extraction: {len(segments)} segments → {len(keywords)} queries → {len(unique_keywords)} unique")

            return unique_keywords, entities, topic

        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            # TF-IDF fallback
            return self._tfidf_fallback(segments, config)

    def _tfidf_fallback(
        self,
        segments: List['VoiceoverSegment'],
        config: 'Config'
    ) -> tuple:
        """Fallback to TF-IDF keyword extraction"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            text = " ".join([s.text for s in segments])
            max_features = getattr(config.keyword, 'tfidf_max_features', 100)
            vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
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

    def _detect_chapters(
        self,
        segments: List['VoiceoverSegment'],
        topic: str,
        config: 'Config'
    ) -> List[Dict[str, Any]]:
        """Detect chapters in voiceover"""
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
