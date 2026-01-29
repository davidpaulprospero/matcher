"""
Transcribe Stage - Video/Audio Transcription and Embedding

Stage 3 of the video matching pipeline:
- Transcribes video/audio files using Whisper
- Computes embeddings for semantic search
- Builds embedding index for matching
- Pre-detects faces if face preference is set
- Extracts video topics for chapter matching

Caption-first fallback (US-006):
- Checks for existing caption data in state.text_metadata before transcribing
- Skips transcription for videos with caption_source='youtube'
- Falls back to Whisper transcription when captions are unavailable
- Tracks source in text_metadata: 'youtube', 'whisper', 'manual'
- Logs summary: 'X videos used captions, Y required transcription'
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from . import Stage, StageResult, register_stage
from ..utils import is_embeddings_empty

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@register_stage
class TranscribeStage(Stage):
    """
    Transcribes videos and computes embeddings.

    Inputs:
        - state.downloaded_videos: List of DownloadedVideo
        - state.downloaded_audio: List of AudioDownload (audio-first mode)

    Outputs:
        - state.transcripts: Dict[video_path, List[segments]]
        - state.embeddings: List of embedding vectors
        - state.text_metadata: List of text metadata dicts
        - state.embedding_index: FAISS index
    """

    name = "TRANSCRIBE"
    description = "Transcribe videos and compute embeddings"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the transcribe stage"""
        warnings = []

        try:
            if config.pipeline.skip_transcription:
                print("  >> Skipping transcription - loading from cache...")
                logger.info("Skipping TRANSCRIBE stage (config: skip_transcription=true)")
                cached = self._load_transcripts_from_cache(config)
                state.transcripts = cached
                print(f"  + Loaded {len(cached)} video transcripts from cache")

                # Also load embeddings from cache (required for matching)
                if cached:
                    embeddings_result = self._compute_embeddings(cached, state, config)
                    if embeddings_result:
                        print(f"  + Loaded embedding index ({len(state.embeddings)} vectors)")
                    else:
                        warnings.append("Could not load embeddings from cache")

                return StageResult.ok({'transcripts': {}, 'from_cache': True}, warnings)

            print(f"\n  --- Stage 3: TRANSCRIBE & INDEX ---")

            # Check for existing caption data (caption-first mode)
            # US-006: Skip transcription for videos with existing YouTube captions
            videos_with_captions, caption_stats = self._get_videos_with_captions(state, config)
            if caption_stats['total'] > 0:
                print(f"  Found {caption_stats['total']} videos with existing captions")
                logger.info(f"Caption-first fallback: {caption_stats['youtube']} YouTube captions, "
                           f"{caption_stats['needs_transcription']} need transcription")

            # Determine video files to transcribe
            video_files = self._get_video_files(state, config)

            if not video_files:
                print("  ! No video files found")
                warnings.append("No video files to transcribe")
                return StageResult.ok({'transcripts': {}}, warnings)

            # Filter out videos that already have caption data (US-006)
            files_needing_transcription = self._filter_videos_with_captions(
                video_files, videos_with_captions, config
            )
            caption_count = len(video_files) - len(files_needing_transcription)

            print(f"  Found {len(video_files)} files to process")
            if caption_count > 0:
                print(f"  Skipping {caption_count} files (already have YouTube captions)")

            # Transcription - only transcribe files that don't have captions
            transcripts = self._transcribe_videos(files_needing_transcription, config)
            state.transcripts = transcripts

            # Add caption_source='whisper' to transcribed segments (US-006)
            self._mark_transcription_source(transcripts, state, 'whisper')

            # Log caption vs transcription summary (US-006)
            transcription_count = len(files_needing_transcription)
            if caption_count > 0 or transcription_count > 0:
                print(f"  + Source summary: {caption_count} videos used captions, "
                      f"{transcription_count} required transcription")
                logger.info(f"Caption-first summary: {caption_count} captions, "
                           f"{transcription_count} transcriptions")

            print(f"  + Transcribed {len(transcripts)} videos")

            # Handle silent videos
            self._handle_silent_videos(files_needing_transcription, transcripts, config)

            # Compute embeddings
            embeddings_result = self._compute_embeddings(transcripts, state, config)
            if embeddings_result:
                print(f"  + Built embedding index ({len(state.embeddings)} vectors)")

            # Pre-detect faces if needed
            if state.face_preference != 'neutral':
                self._predetect_faces(video_files, config)

            # Extract video topics for chapter matching
            if config.matching.chapter_matching_enabled:
                self._extract_video_topics(transcripts, config)

            # Prepare checkpoint data
            checkpoint_data = {
                'transcript_count': len(transcripts),
                'embedding_count': 0 if is_embeddings_empty(state.embeddings) else len(state.embeddings),
                'video_files': [str(vf) for vf in video_files],
                # US-006: Caption-first statistics
                'caption_count': caption_count,
                'transcription_count': transcription_count,
                'caption_source_stats': caption_stats,
            }

            return StageResult.ok(checkpoint_data, warnings)

        except Exception as e:
            logger.exception(f"Transcribe stage failed: {e}")
            return StageResult.fail(str(e), warnings)

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if transcribe stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore transcribe stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # Transcripts need to be loaded from cache, not checkpoint
            # (they're too large to store in checkpoint JSON)
            # Note: text_metadata is rebuilt in can_skip() via _rebuild_text_metadata()

            # Load embeddings from cache (critical for MATCH stage)
            if config and state.transcripts:
                embeddings_result = self._compute_embeddings(state.transcripts, state, config)
                if embeddings_result:
                    logger.info(f"Restored embeddings: {len(state.embeddings)} vectors")
                else:
                    logger.warning("Could not restore embeddings from cache")
                    return False

            logger.info(f"Restored TRANSCRIBE metadata from checkpoint")
            return True

        except Exception as e:
            logger.warning(f"Failed to restore TRANSCRIBE: {e}")
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        # Need either downloaded videos or audio files
        has_videos = len(state.downloaded_videos) > 0
        has_audio = len(state.downloaded_audio) > 0

        if not has_videos and not has_audio:
            return "No videos or audio files available for transcription"
        return None

    # === Helper Methods ===

    def _get_video_files(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> List[Path]:
        """Determine which files to transcribe"""
        # Check if audio-first mode with audio files
        if state.downloaded_audio:
            return [Path(ad.file) for ad in state.downloaded_audio]

        # Use downloaded videos
        if state.downloaded_videos:
            return [Path(dv.file) for dv in state.downloaded_videos]

        # Fallback: scan directory
        videos_dir = Path(config.downloaded_videos_dir)
        video_files = (
            list(videos_dir.rglob('*.mp4')) +
            list(videos_dir.rglob('*.webm')) +
            list(videos_dir.rglob('*.mp3'))
        )
        return video_files

    def _get_videos_with_captions(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> tuple[set, Dict[str, int]]:
        """
        Get set of video IDs/paths that already have caption data.

        US-006: Check state.text_metadata for entries with caption_source='youtube'.
        These videos don't need Whisper transcription.

        Returns:
            Tuple of (set of video identifiers with captions, stats dict)
        """
        videos_with_captions = set()
        stats = {
            'total': 0,
            'youtube': 0,
            'whisper': 0,
            'manual': 0,
            'needs_transcription': 0,
        }

        # Check if caption-first mode is enabled
        caption_config = getattr(config.download, 'caption_first', None)
        if not caption_config or not getattr(caption_config, 'enabled', False):
            # Caption-first not enabled, no captions to check
            return videos_with_captions, stats

        # Check if fallback to transcription is disabled
        fallback_enabled = getattr(caption_config, 'fallback_to_transcription', True)
        if not fallback_enabled:
            logger.info("Caption-first fallback disabled, will not transcribe missing captions")

        # Scan text_metadata for entries with caption_source
        for entry in state.text_metadata:
            if not isinstance(entry, dict):
                continue

            caption_source = entry.get('caption_source')
            video_id = entry.get('video_path') or entry.get('source_file')

            if caption_source == 'youtube':
                if video_id:
                    videos_with_captions.add(video_id)
                stats['youtube'] += 1
            elif caption_source == 'whisper':
                stats['whisper'] += 1
            elif caption_source == 'manual':
                stats['manual'] += 1

        stats['total'] = len(videos_with_captions)
        # Count unique video IDs in text_metadata that don't have YouTube captions
        all_video_ids = set()
        for entry in state.text_metadata:
            if isinstance(entry, dict):
                video_id = entry.get('video_path') or entry.get('source_file')
                if video_id:
                    all_video_ids.add(video_id)

        stats['needs_transcription'] = len(all_video_ids - videos_with_captions)

        return videos_with_captions, stats

    def _filter_videos_with_captions(
        self,
        video_files: List[Path],
        videos_with_captions: set,
        config: 'Config'
    ) -> List[Path]:
        """
        Filter out video files that already have caption data.

        US-006: Skip transcription for videos with valid YouTube captions.

        Args:
            video_files: List of video file paths to potentially transcribe
            videos_with_captions: Set of video IDs that have captions
            config: Configuration object

        Returns:
            Filtered list of video files that need transcription
        """
        import re

        # Check if caption-first mode is enabled
        caption_config = getattr(config.download, 'caption_first', None)
        if not caption_config or not getattr(caption_config, 'enabled', False):
            # Caption-first not enabled, transcribe everything
            return video_files

        # Check if fallback to transcription is enabled
        if not getattr(caption_config, 'fallback_to_transcription', True):
            # Fallback disabled, still filter but log differently
            logger.info("Fallback disabled - videos without captions will have no transcript")

        if not videos_with_captions:
            return video_files

        filtered = []
        for vf in video_files:
            # Extract video ID from path
            video_id = self._extract_video_id_from_path(str(vf))

            # Check if this video has captions
            if video_id and video_id in videos_with_captions:
                logger.debug(f"Skipping transcription for {vf.name} (has YouTube captions)")
                continue

            # Also check full path match
            if str(vf) in videos_with_captions:
                logger.debug(f"Skipping transcription for {vf.name} (has YouTube captions)")
                continue

            filtered.append(vf)

        skipped = len(video_files) - len(filtered)
        if skipped > 0:
            logger.info(f"Skipped {skipped} videos with existing captions")

        return filtered

    def _extract_video_id_from_path(self, path: str) -> Optional[str]:
        """
        Extract YouTube video ID from a file path.

        Args:
            path: File path (may contain video ID in filename)

        Returns:
            11-character video ID if found, None otherwise
        """
        import re
        from pathlib import Path

        filename = Path(path).stem

        # YouTube IDs are exactly 11 alphanumeric chars with _-
        match = re.search(r'([A-Za-z0-9_-]{11})', filename)
        if match:
            return match.group(1)

        return None

    def _mark_transcription_source(
        self,
        transcripts: Dict[str, List[Any]],
        state: 'PipelineState',
        source: str
    ):
        """
        Mark transcribed segments with their source type.

        US-006: Track caption_source in text_metadata.

        Args:
            transcripts: Dict of video_path -> segments
            state: Pipeline state
            source: Source type ('whisper', 'youtube', 'manual')
        """
        # Mark segments in transcripts dict
        for video_path, segments in transcripts.items():
            for seg in segments:
                if isinstance(seg, dict):
                    seg['caption_source'] = source
                elif hasattr(seg, '__dict__'):
                    seg.caption_source = source

        # Also update any matching entries in text_metadata
        for entry in state.text_metadata:
            if isinstance(entry, dict):
                # If entry doesn't have caption_source, check if it's from a transcribed video
                if 'caption_source' not in entry:
                    video_path = entry.get('video_path')
                    if video_path and video_path in transcripts:
                        entry['caption_source'] = source

    def _transcribe_videos(
        self,
        video_files: List[Path],
        config: 'Config'
    ) -> Dict[str, List[Any]]:
        """Transcribe video files"""
        transcripts = {}

        if config.pipeline.parallel_transcription:
            try:
                from ..transcription import transcribe_videos_parallel, DeltaAwareIndex

                print(f"  Using parallel transcription ({config.transcription.max_workers} workers)")

                # Delta-aware indexing
                use_delta = getattr(config.pipeline, 'delta_indexing', True)
                if use_delta:
                    delta_index = DeltaAwareIndex(config.cache.cache_dir)
                    video_files_str = [str(v) for v in video_files]
                    video_files_str = delta_index.get_new_videos(video_files_str)
                    video_files = [Path(v) for v in video_files_str]
                    print(f"  Delta indexing: {len(video_files)} new videos")

                if not video_files:
                    print("  + All videos already transcribed (cached)")
                    return {}

                transcripts = transcribe_videos_parallel(
                    video_paths=[str(v) for v in video_files],
                    cache=config.cache.cache_dir,
                    config=config,
                    max_workers=config.transcription.max_workers,
                    show_progress=True
                )
            except ImportError:
                logger.warning("Parallel transcription not available, using sequential")
                transcripts = self._transcribe_sequential(video_files, config)
        else:
            transcripts = self._transcribe_sequential(video_files, config)

        return transcripts

    def _transcribe_sequential(
        self,
        video_files: List[Path],
        config: 'Config'
    ) -> Dict[str, List[Any]]:
        """Sequential transcription fallback"""
        from ..transcription import transcribe_video, TranscriptCache

        print("  Using sequential transcription")
        cache = TranscriptCache(config.cache.cache_dir)

        transcripts = {}
        for vf in video_files:
            try:
                segments = transcribe_video(
                    str(vf),
                    cache=cache,
                    model_name=config.transcription.model,
                    language=config.transcription.language if config.transcription.language != 'auto' else None,
                    vad_filter=False,  # Disable VAD for YouTube videos - speech quality varies
                    min_silence_duration_ms=getattr(config.transcription, 'min_silence_duration_ms', 200),
                    speech_pad_ms=getattr(config.transcription, 'speech_pad_ms', 10)
                )
                transcripts[str(vf)] = segments
            except Exception as e:
                logger.warning(f"Failed to transcribe {vf}: {e}")

        return transcripts

    def _handle_silent_videos(
        self,
        video_files: List[Path],
        transcripts: Dict[str, List[Any]],
        config: 'Config'
    ):
        """Generate descriptions for silent/B-roll videos"""
        silent_config = getattr(config, 'silent_video', None)
        if not silent_config or not getattr(silent_config, 'enabled', True):
            return

        # Find videos with no transcription or very few words
        min_words = getattr(silent_config, 'min_words_threshold', 10)
        silent_videos = []
        for vf in video_files:
            vf_str = str(vf)
            if vf_str not in transcripts or not transcripts.get(vf_str):
                silent_videos.append(vf)
            else:
                # Check word count
                segments = transcripts.get(vf_str, [])
                word_count = sum(
                    len(seg.get('text', '').split()) if isinstance(seg, dict)
                    else len(getattr(seg, 'text', '').split())
                    for seg in segments
                )
                if word_count < min_words:
                    silent_videos.append(vf)

        if not silent_videos:
            return

        logger.info(f"Found {len(silent_videos)} silent videos for description generation")
        print(f"  Generating descriptions for {len(silent_videos)} silent videos...")

        # Generate LLM descriptions
        descriptions = self._generate_llm_descriptions(silent_videos, transcripts, config)

        # Update transcripts with generated descriptions
        for video_path, description in descriptions.items():
            if description:
                # Create a synthetic transcript segment with the description
                transcripts[video_path] = [{
                    'text': description,
                    'start_time': 0.0,
                    'end_time': 30.0,  # Assume 30s default duration
                    'is_generated': True,
                    'source': 'llm_description'
                }]
                logger.debug(f"Generated description for {Path(video_path).name}: {description[:50]}...")

        generated_count = sum(1 for d in descriptions.values() if d)
        print(f"  + Generated {generated_count}/{len(silent_videos)} descriptions")

    def _generate_llm_descriptions(
        self,
        silent_videos: List[Path],
        transcripts: Dict[str, List[Any]],
        config: 'Config'
    ) -> Dict[str, str]:
        """
        Generate semantic descriptions for silent videos using Vision API or LLM fallback.

        Uses Vision API if available (GEMINI_API_KEY), otherwise falls back to
        generating descriptions from filename keywords.

        Args:
            silent_videos: List of video paths with no/sparse transcription
            transcripts: Dict of existing transcripts (will be updated)
            config: Configuration object

        Returns:
            Dict mapping video_path to generated description
        """
        descriptions = {}
        silent_config = getattr(config, 'silent_video', None)

        use_vision = getattr(silent_config, 'use_vision_api', True) if silent_config else True
        use_llm_fallback = getattr(silent_config, 'use_llm_fallback', True) if silent_config else True

        # Try Vision API first
        if use_vision:
            try:
                from ..vision import VisionProcessor

                processor = VisionProcessor(config)
                if processor.is_available():
                    cache_dir = getattr(config.cache, 'cache_dir', '.cache')

                    for vf in silent_videos:
                        video_path = str(vf)
                        # Create a pseudo-scene covering the video
                        scene = {'start_time': 0.0, 'end_time': 30.0}

                        description = processor.describe_scene(video_path, scene, cache_dir)
                        if description:
                            descriptions[video_path] = description

                    stats = processor.get_stats()
                    if stats.get('api_calls', 0) > 0:
                        logger.info(f"Vision API: {stats['api_calls']} calls, ~${stats['estimated_cost']:.4f}")

            except ImportError:
                logger.debug("Vision module not available, using fallback")
            except Exception as e:
                logger.warning(f"Vision API error: {e}, using fallback")

        # LLM fallback for videos without descriptions
        if use_llm_fallback:
            missing_videos = [vf for vf in silent_videos if str(vf) not in descriptions]

            if missing_videos:
                descriptions.update(self._generate_filename_descriptions(missing_videos, config))

        return descriptions

    def _generate_filename_descriptions(
        self,
        videos: List[Path],
        config: 'Config'
    ) -> Dict[str, str]:
        """
        Generate descriptions from video filenames using LLM.

        Falls back to simple keyword extraction if LLM unavailable.

        Args:
            videos: List of video paths
            config: Configuration object

        Returns:
            Dict mapping video_path to generated description
        """
        import re
        descriptions = {}

        # Try LLM-based description
        try:
            from ..llm_client import create_client, LLMRequest, ResponseFormat

            # Get LLM config
            llm_config = getattr(config, 'llm', None)
            api_key = None
            model = "gemini-2.0-flash"

            if llm_config:
                provider = getattr(llm_config, 'provider', 'gemini')
                model = getattr(llm_config, 'model', model)
                import os
                if provider == 'gemini':
                    api_key = os.getenv('GEMINI_API_KEY')
                elif provider == 'anthropic':
                    api_key = os.getenv('ANTHROPIC_API_KEY')

            if api_key:
                client = create_client("gemini", api_key=api_key, model=model)

                for vf in videos:
                    video_path = str(vf)
                    filename = vf.stem

                    # Clean filename to keywords
                    keywords = re.sub(r'[_\-\.]', ' ', filename)
                    keywords = re.sub(r'\s+', ' ', keywords).strip()

                    prompt = f"""Generate a brief 2-sentence description for a stock video based on these keywords from its filename:
Keywords: {keywords}

Describe what the video likely shows, focusing on: subjects, actions, setting.
Return ONLY the description, no other text."""

                    request = LLMRequest(
                        prompt=prompt,
                        response_format=ResponseFormat.TEXT,
                        max_tokens=100,
                        temperature=0.3,
                        cache_key_prefix="silent_video_desc",
                        use_cache=True
                    )

                    try:
                        response = client.generate(request)
                        if response.text:
                            descriptions[video_path] = response.text.strip()
                    except Exception as e:
                        logger.debug(f"LLM description failed for {filename}: {e}")

                return descriptions

        except ImportError:
            logger.debug("LLM client not available, using simple extraction")
        except Exception as e:
            logger.debug(f"LLM description error: {e}")

        # Simple fallback: extract keywords from filename
        for vf in videos:
            video_path = str(vf)
            filename = vf.stem

            # Clean filename
            keywords = re.sub(r'[_\-\.]', ' ', filename)
            keywords = re.sub(r'\s+', ' ', keywords).strip()

            if keywords:
                descriptions[video_path] = f"[Silent video: {keywords}]"

        return descriptions

    def _compute_embeddings(
        self,
        transcripts: Dict[str, List[Any]],
        state: 'PipelineState',
        config: 'Config'
    ) -> bool:
        """Compute embeddings for transcript segments"""
        if not config.pipeline.parallel_embedding:
            return False

        try:
            from ..embeddings import compute_embeddings, build_embedding_index, get_embedding_provider
            from ..utils import CacheManager

            print(f"  Computing embeddings (batch size: {config.embedding.batch_size})")

            # Collect texts
            texts = []
            for video_path, segments in transcripts.items():
                for seg in segments:
                    if hasattr(seg, 'text'):
                        texts.append({
                            'text': seg.text,
                            'video_path': video_path,
                            'start_time': seg.start_time,
                            'end_time': seg.end_time
                        })
                    else:
                        texts.append({
                            'text': seg.get('text', ''),
                            'video_path': video_path,
                            'start_time': seg.get('start_time', 0),
                            'end_time': seg.get('end_time', 0)
                        })

            state.text_metadata = texts
            text_strings = [t['text'] for t in texts]

            if not text_strings:
                print("  ! No text segments to embed")
                state.embeddings = []
                return False

            # Get provider and cache
            provider = get_embedding_provider(config)
            cache = CacheManager(config.cache.cache_dir)

            # Compute embeddings
            state.embeddings = compute_embeddings(
                texts=text_strings,
                provider=provider,
                cache=cache,
                cache_key="video_segments"
            )

            # Build index
            if state.embeddings is not None and len(state.embeddings) > 0:
                state.embedding_index = build_embedding_index(state.embeddings, config=config)
                return True

            return False

        except ImportError as e:
            logger.warning(f"Embedding computation not available: {e}")
            return False
        except Exception as e:
            logger.error(f"Embedding computation failed: {e}")
            return False

    def _predetect_faces(
        self,
        video_files: List[Path],
        config: 'Config'
    ):
        """Pre-detect faces in videos and cache results"""
        try:
            from ..face_detection import FaceDetector

            detector = FaceDetector.get_instance()
            cache_dir = config.cache.cache_dir

            # Check how many need detection
            uncached = [
                vf for vf in video_files
                if str(vf) not in detector._cache
            ]

            if not uncached:
                print(f"  + Face detection: all {len(video_files)} videos cached")
                return

            print(f"  Detecting faces in {len(uncached)} videos...")

            faces_found = 0
            for i, vf in enumerate(uncached):
                video_path = str(vf)
                score = detector.get_face_score(video_path, cache_dir)
                if score > 0.2:
                    faces_found += 1

                if (i + 1) % 10 == 0:
                    print(f"    Processed {i + 1}/{len(uncached)}...", end='\r')

            print(f"  + Face detection: {faces_found}/{len(uncached)} videos have faces")

        except Exception as e:
            logger.warning(f"Face pre-detection failed: {e}")

    def _extract_video_topics(
        self,
        transcripts: Dict[str, List[Any]],
        config: 'Config'
    ):
        """Extract topics from video transcripts"""
        if not transcripts:
            return

        try:
            from ..topic_extraction import TopicExtractor

            extractor = TopicExtractor(config, config.cache.cache_dir)

            transcripts_dict = {}
            for video_path, segments in transcripts.items():
                if segments:
                    texts = []
                    for seg in segments:
                        if hasattr(seg, 'text'):
                            texts.append(seg.text)
                        elif isinstance(seg, dict):
                            texts.append(seg.get('text', ''))
                    transcripts_dict[video_path] = ' '.join(texts)

            if transcripts_dict:
                logger.info(f"Extracting topics from {len(transcripts_dict)} videos")
                # Topic extraction happens in extractor

        except Exception as e:
            logger.warning(f"Video topic extraction failed: {e}")

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if transcribe stage can be skipped"""
        # If we can skip via checkpoint, rebuild text_metadata from cache
        if checkpoint.should_skip_stage(self.name):
            # CRITICAL: Rebuild text_metadata when skipping
            # This is needed for --match-only mode to have B-roll flags
            logger.info("===== TRANSCRIBE can_skip: TRUE, rebuilding text_metadata =====")
            logger.info(f"  Before rebuild: text_metadata has {len(state.text_metadata)} entries")

            # Skip rebuild if text_metadata was preloaded (match-only mode)
            if len(state.text_metadata) > 0:
                logger.info("  Skipping rebuild - text_metadata already preloaded")
            else:
                self._rebuild_text_metadata(state, checkpoint)
                logger.info(f"  After rebuild: text_metadata has {len(state.text_metadata)} entries")

            # CRITICAL: After rebuilding text_metadata, re-run SCENE_DETECTION merge
            # to restore B-roll flags that were lost during rebuild
            self._restore_broll_flags(state, checkpoint)

            broll_before = sum(1 for m in state.text_metadata if isinstance(m, dict) and m.get('is_broll'))
            logger.info(f"  B-roll entries after rebuild: {broll_before}")
            return True
        logger.info("===== TRANSCRIBE can_skip: FALSE =====")
        return False

    def _rebuild_text_metadata(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ):
        """
        Rebuild state.text_metadata from cached transcripts.

        This is critical for --match-only mode: when TRANSCRIBE is skipped,
        state.text_metadata would be empty, losing all B-roll flags added
        by SCENE_DETECTION.

        Strategy:
        1. Load cached transcripts
        2. Rebuild text_metadata list from transcripts
        3. Let SCENE_DETECTION update it with is_broll flags
        """
        try:
            from ..config import Config
            config_path = checkpoint.checkpoint_path.parent / "project_config.yaml"
            if not config_path.exists():
                config_path = Path("config.yaml")

            # Load config to get cache dir
            import yaml
            try:
                from yaml import CSafeLoader as SafeLoader
            except ImportError:
                from yaml import SafeLoader

            with open(config_path) as f:
                config_data = yaml.load(f, Loader=SafeLoader)

            cache_dir = config_data.get('cache', {}).get('cache_dir', '.cache')
            cache_dir = checkpoint.checkpoint_path.parent / cache_dir

            # Load cached transcripts
            from ..transcription import TranscriptCache
            trans_cache = TranscriptCache(str(cache_dir))

            # Rebuild transcripts dict
            state.transcripts = {}
            texts = []

            # Get all cached transcript files
            import glob
            import json
            transcript_files = glob.glob(str(cache_dir / "transcriptions" / "*.json"))

            for tf in transcript_files:
                try:
                    with open(tf) as f:
                        data = json.load(f)
                        if isinstance(data, list) and data:
                            # Find video path (stored in first segment usually)
                            # Note: field is 'source_file' in cache, not 'video_path'
                            video_path = None
                            for seg in data:
                                if isinstance(seg, dict):
                                    video_path = seg.get('source_file') or seg.get('video_path')
                                    if video_path:
                                        break

                            if video_path:
                                state.transcripts[video_path] = data

                                # Rebuild text_metadata
                                for seg in data:
                                    if isinstance(seg, dict):
                                        texts.append({
                                            'text': seg.get('text', ''),
                                            'video_path': video_path,
                                            'start_time': seg.get('start_time', 0),
                                            'end_time': seg.get('end_time', 0)
                                        })
                except Exception as e:
                    logger.debug(f"Could not load transcript {tf}: {e}")
                    continue

            state.text_metadata = texts
            logger.info(f"Rebuilt text_metadata: {len(texts)} entries from {len(state.transcripts)} videos")

        except Exception as e:
            logger.warning(f"Failed to rebuild text_metadata: {e}")
            logger.warning("B-roll detection may not work in --match-only mode")
            state.text_metadata = []

    def _load_transcripts_from_cache(self, config: 'Config') -> Dict[str, List[Any]]:
        """Load transcripts from cache when skipping transcription"""
        try:
            from ..transcription import TranscriptCache

            cache = TranscriptCache(config.cache.cache_dir)

            # Scan for video files
            videos_dir = Path(config.downloaded_videos_dir)
            video_files = (
                list(videos_dir.rglob('*.mp4')) +
                list(videos_dir.rglob('*.webm'))
            )

            transcripts = {}
            for vf in video_files:
                cached = cache.get(str(vf))
                if cached:
                    transcripts[str(vf)] = cached

            logger.info(f"Loaded {len(transcripts)} transcripts from cache")
            return transcripts

        except Exception as e:
            logger.warning(f"Failed to load transcripts from cache: {e}")
            return {}

    def _restore_broll_flags(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ):
        """
        Restore B-roll flags to text_metadata after rebuild.

        When text_metadata is rebuilt from transcripts, it loses the is_broll
        flags that were added by SCENE_DETECTION. This method re-runs the
        scene data merge to restore those flags.
        """
        try:
            # Check if SCENE_DETECTION was completed (has checkpoint data)
            scene_data = checkpoint.get_stage_data("SCENE_DETECTION")
            if not scene_data or scene_data.get('skipped'):
                logger.info("  SCENE_DETECTION not completed, skipping B-roll flag restore")
                return

            # Check if we have scene data available
            if not state.text_metadata:
                logger.info("  No text_metadata to restore B-roll flags to")
                return

            # Import scene detection stage to use its merge method
            from .scene_detection import SceneDetectionStage

            # Load config for scene detector using the proper public API
            from ..config import load_config
            config_path = checkpoint.checkpoint_path.parent / "project_config.yaml"
            if not config_path.exists():
                config_path = Path("config.yaml")

            if config_path.exists():
                config = load_config(str(config_path))
            else:
                logger.warning("  Cannot restore B-roll flags: config not found")
                return

            # Load scene data from SceneDetector's cache
            from ..scene_detection import SceneDetector

            scene_detector = SceneDetector(config)
            scene_data_dict = scene_detector.scene_index

            if not scene_data_dict:
                logger.info("  No scene data available for B-roll flag restore")
                return

            # Use SceneDetectionStage's merge method to restore B-roll flags
            stage = SceneDetectionStage()
            broll_before = sum(1 for m in state.text_metadata if isinstance(m, dict) and m.get('is_broll'))
            logger.info(f"  Restoring B-roll flags: {broll_before} entries have is_broll before merge")

            stage._merge_scene_data_to_transcripts(state, scene_data_dict, config)

            broll_after = sum(1 for m in state.text_metadata if isinstance(m, dict) and m.get('is_broll'))
            logger.info(f"  Restored B-roll flags: {broll_after} entries now have is_broll")

        except Exception as e:
            logger.warning(f"  Failed to restore B-roll flags: {e}")
            import traceback
            logger.debug(traceback.format_exc())
