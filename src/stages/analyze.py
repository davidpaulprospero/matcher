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
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage, validate_required_state_attrs
from ..logging_templates import (
    log_stage_start,
    log_stage_complete,
    log_stage_skip,
    log_error_with_context,
    log_progress,
)

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
    DEPENDS_ON = []  # First stage — no dependencies
    PRODUCES = ['voiceover_segments', 'keywords', 'topic_context', 'extracted_entities']

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the analyze stage.

        US-44-002: Validates required state attributes exist.
        """
        # US-167-009: Track stage timing
        stage_start_time = time.time()

        # US-44-002: Validate required attributes exist
        validate_required_state_attrs(state, ['voiceover_path'], self.name)

        warnings = []

        try:
            # Get voiceover path
            voiceover_path = state.voiceover_path
            if not voiceover_path:
                log_error_with_context(logger, "PIPE-004", "No voiceover path specified")
                return StageResult.fail("No voiceover path specified")

            if not Path(voiceover_path).exists():
                log_error_with_context(logger, "PIPE-004", f"Voiceover file not found: {voiceover_path}", path=voiceover_path)
                return StageResult.fail(f"Voiceover file not found: {voiceover_path}")

            log_stage_start(logger, "ANALYZE", total_segments=len(segments) if 'segments' in locals() else None)

            # Load voiceover segments
            segments = self._load_voiceover_segments(voiceover_path, config)
            if not segments:
                log_error_with_context(logger, "PIPE-001", "No segments found in voiceover")
                return StageResult.fail("No segments found in voiceover")

            # Apply test mode segment limit if enabled
            test_mode_max_segments = getattr(config, '_test_mode_max_segments', None)
            if test_mode_max_segments is not None and isinstance(test_mode_max_segments, int) and len(segments) > test_mode_max_segments:
                original_count = len(segments)
                segments = segments[:test_mode_max_segments]
                logger.info(f"Test mode: limited segments from {original_count} to {test_mode_max_segments}")

            state.voiceover_segments = segments
            logger.info(f"Loaded {len(segments)} voiceover segments")

            # Extract keywords
            keywords, entities, topic = self._extract_keywords(segments, config)

            state.keywords = keywords
            state.extracted_entities = entities
            state.topic_context = topic

            log_progress(logger, "ANALYZE", 75.0, 3, 4, keywords=len(keywords), topic=topic, entities=len(entities))

            # Detect chapters if enabled

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

            # US-167-009: Log stage completion with timing
            elapsed = time.time() - stage_start_time
            log_stage_complete(
                logger, "ANALYZE",
                elapsed_seconds=elapsed,
                segments=len(segments),
                keywords=len(keywords),
                entities=len(entities),
                topic=topic
            )

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            log_error_with_context(logger, "PIPE-001", f"Analyze stage failed: {e}")
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

            # US-51-008: Validate checkpoint data schema before restoring
            if not isinstance(data, dict):
                log_error_with_context(logger, "PIPE-002", f"ANALYZE restore: expected dict, got {type(data).__name__}")
                return False

            required_keys = {'keywords', 'segments'}
            missing = required_keys - set(data.keys())
            if missing:
                log_error_with_context(logger, "PIPE-002", f"ANALYZE restore: missing required keys: {missing}")
                return False

            if not isinstance(data['keywords'], list):
                log_error_with_context(logger, "PIPE-002", f"ANALYZE restore: 'keywords' expected list, got {type(data['keywords']).__name__}")
                return False

            if not isinstance(data['segments'], list):
                log_error_with_context(logger, "PIPE-002", f"ANALYZE restore: 'segments' expected list, got {type(data['segments']).__name__}")
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
            log_error_with_context(logger, "PIPE-002", f"Failed to restore ANALYZE: {e}")
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

    def get_input_output_info(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[str, Any]:
        """Get input/output info for dry-run preview"""
        return {
            'inputs': 'voiceover file',
            'outputs': 'voiceover segments',
            'input_count': 1 if state.voiceover_path else 0,
            'output_count': len(state.voiceover_segments) if state.voiceover_segments else None,
        }

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
            segments = self._refresh_stale_srt_from_companion_audio(
                srt_path=path,
                segments=segments,
                config=config,
            )
        elif path.suffix.lower() in ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'):
            # Transcribe audio file
            segments = self._transcribe_audio(path, config)
        else:
            logger.warning(f"Unknown voiceover format: {path.suffix}")

        return segments

    def _refresh_stale_srt_from_companion_audio(
        self,
        srt_path: Path,
        segments: List['VoiceoverSegment'],
        config: 'Config',
    ) -> List['VoiceoverSegment']:
        """
        Refresh stale SRT files when a companion audio file clearly disagrees.

        This guards against stale subtitle files being reused across runs.
        A refresh is triggered only when the SRT/audio duration delta is large,
        to avoid rewriting minor timing drifts.
        """
        companion_audio = self._find_companion_audio_file(srt_path)
        if companion_audio is None:
            return segments

        transcription_cfg = getattr(config, 'transcription', None)
        if isinstance(transcription_cfg, dict):
            auto_refresh_stale_srt = transcription_cfg.get('auto_refresh_stale_srt', True)
        else:
            auto_refresh_stale_srt = getattr(transcription_cfg, 'auto_refresh_stale_srt', True)
        if not auto_refresh_stale_srt:
            return segments

        try:
            from ..transcription import get_audio_duration

            audio_duration = get_audio_duration(str(companion_audio))
            if audio_duration is None:
                return segments

            srt_duration = self._get_segments_end_time(segments)
            duration_delta = abs(audio_duration - srt_duration)
            # Refresh only when mismatch is significant:
            # at least 30s and at least 10% of the audio duration.
            max_allowed_delta = max(30.0, audio_duration * 0.10)

            should_refresh = (
                not segments
                or srt_duration <= 0.0
                or duration_delta > max_allowed_delta
            )
            if not should_refresh:
                return segments

            logger.warning(
                "Detected stale voiceover SRT (srt=%.1fs, audio=%.1fs, delta=%.1fs). "
                "Refreshing from companion audio: %s",
                srt_duration,
                audio_duration,
                duration_delta,
                companion_audio,
            )

            refreshed_segments = self._transcribe_audio(
                companion_audio,
                config,
                output_srt_path=srt_path,
            )
            if refreshed_segments:
                logger.info(
                    "Refreshed stale SRT from companion audio: %s (%d segments)",
                    srt_path,
                    len(refreshed_segments),
                )
                return refreshed_segments
        except Exception as exc:
            logger.warning(f"Failed stale SRT refresh check for {srt_path}: {exc}")

        return segments

    def _find_companion_audio_file(self, srt_path: Path) -> Optional[Path]:
        """Find same-stem audio file that accompanies an SRT file."""
        for ext in ('.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg'):
            candidate = srt_path.with_suffix(ext)
            if candidate.exists():
                return candidate
        return None

    def _get_segments_end_time(self, segments: List['VoiceoverSegment']) -> float:
        """Return the latest segment end time, or 0.0 for empty/invalid lists."""
        if not segments:
            return 0.0

        max_end = 0.0
        for seg in segments:
            end = getattr(seg, 'end', 0.0)
            try:
                end_val = float(end)
            except (TypeError, ValueError):
                end_val = 0.0
            if end_val > max_end:
                max_end = end_val
        return max_end

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
        config: 'Config',
        output_srt_path: Optional[Path] = None,
    ) -> List['VoiceoverSegment']:
        """Transcribe audio file to segments"""
        from ..state import VoiceoverSegment

        try:
            from ..transcription import (
                transcribe_voiceover_audio,
                write_srt,
                compress_segment_gaps,
            )

            # Get VAD setting from config - default True for voiceover
            # VAD filters silence accurately, improving gap detection
            transcription_cfg = getattr(config, 'transcription', None)
            if isinstance(transcription_cfg, dict):
                vad_filter = transcription_cfg.get('vad_filter', True)
                contiguous_timing = transcription_cfg.get('voiceover_contiguous_timing', True)
            else:
                vad_filter = getattr(transcription_cfg, 'vad_filter', True)
                contiguous_timing = getattr(transcription_cfg, 'voiceover_contiguous_timing', True)

            result = transcribe_voiceover_audio(
                str(path),
                model_name=config.transcription.model,
                compute_type=config.transcription.compute_type,
                vad_filter=vad_filter,
            )
            if contiguous_timing and result:
                # Compress gaps while preserving total duration
                # Get original total duration before compression
                original_duration = max(seg.get('end', 0) for seg in result)
                result = compress_segment_gaps(result, target_duration=original_duration)

            segments = []
            for i, seg in enumerate(result):
                segments.append(VoiceoverSegment(
                    index=i,
                    start=seg.get('start', 0.0),
                    end=seg.get('end', 0.0),
                    text=seg.get('text', ''),
                ))

            # Write SRT file (default: alongside audio file).
            # output_srt_path allows refreshing an existing .srt input in-place.
            srt_path = output_srt_path or path.with_suffix('.srt')
            write_srt(result, str(srt_path), force_contiguous_timing=contiguous_timing)
            logger.info(f"Wrote voiceover SRT: {srt_path}")

            return segments

        except Exception as e:
            log_error_with_context(logger, "TRANSCRIBE-001", f"Audio transcription failed: {e}")
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
            logger.info(f"Using grouped keyword extraction ({segments_per_query} segments per query)")
            log_progress(logger, "KEYWORD_EXTRACTION", 50.0, 1, 2, segments_per_query=segments_per_query)
            keywords = extractor.extract_keywords_grouped(
                segment_dicts,
                topic=topic,
                segments_per_query=segments_per_query
            )

            # Deduplicate while preserving order
            unique_keywords = list(dict.fromkeys(keywords))

            log_progress(logger, "KEYWORD_EXTRACTION", 100.0, 2, 2, segments=len(segments), queries=len(keywords), unique=len(unique_keywords))

            return unique_keywords, entities, topic

        except Exception as e:
            log_error_with_context(logger, "PIPE-001", f"Keyword extraction failed: {e}")
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
            log_error_with_context(logger, "PIPE-001", f"TF-IDF fallback failed: {e}")
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
                logger.info(f"Detected {len(chapters)} chapters")

            return chapters

        except Exception as e:
            log_error_with_context(logger, "PIPE-001", f"Chapter detection failed: {e}", topic=topic if topic else None)
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
            log_error_with_context(logger, "PIPE-001", f"Location chapter detection failed: {e}", topic=topic if topic else None)
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

            logger.info("Detecting chapters (enhanced multi-pass)")

            detector = EnhancedChapterDetector(config)
            location_chapters = detector.detect_chapters(
                segments=segment_dicts,
                content_type='auto',
                location_service=location_service,
                overall_topic=topic
            )

            if location_chapters:
                logger.info(f"Found {len(location_chapters)} chapters")
                # Log confidence info
                for ch in location_chapters:
                    conf = ch.get('confidence', 0.8)
                    title = ch.get('title', 'Untitled')
                    strategy = ch.get('detection_strategy', 'unknown')
                    logger.debug(f"Chapter '{title}': confidence={conf:.2f}, strategy={strategy}")

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

        logger.info("Detecting location-focused chapters (legacy)")

        detector = ChapterDetector(config)
        location_chapters = detector.detect_location_chapters(
            segment_dicts,
            location_service=location_service,
            overall_topic=topic
        )

        if location_chapters:
            logger.info(f"Found {len(location_chapters)} location chapters")

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
