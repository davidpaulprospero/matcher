"""
Transcribe Stage - Video/Audio Transcription and Embedding

Stage 3 of the video matching pipeline:
- Transcribes video/audio files using Whisper
- Computes embeddings for semantic search
- Builds embedding index for matching
- Pre-detects faces if face preference is set
- Extracts video topics for chapter matching
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
        - state.transcripts: Dict[video_id, List[segments]] (from CAPTION stage)
        - state.videos_need_audio: List of video IDs needing Whisper (caption-first mode)

    Outputs:
        - state.transcripts: Dict[video_path, List[segments]] (updated with Whisper transcripts)
        - state.embeddings: List of embedding vectors
        - state.text_metadata: List of text metadata dicts
        - state.embedding_index: FAISS index

    In caption-first mode, only transcribes videos in state.videos_need_audio
    (those without captions). Caption transcripts already exist in state.transcripts.
    Embeddings are computed for ALL transcripts (caption + Whisper).
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

                # Merge cached transcripts with existing caption transcripts (don't overwrite!)
                caption_count = len(state.transcripts)
                state.transcripts.update(cached)
                print(f"  + Loaded {len(cached)} video transcripts from cache (+ {caption_count} from captions)")

                # Compute embeddings for ALL transcripts (including caption-sourced)
                if state.transcripts:
                    embeddings_result = self._compute_embeddings(state.transcripts, state, config)
                    if embeddings_result:
                        print(f"  + Loaded embedding index ({len(state.embeddings)} vectors)")
                    else:
                        warnings.append("Could not load embeddings from cache")

                return StageResult.ok({'transcripts': {}, 'from_cache': True}, warnings)

            print(f"\n  --- Stage 3: TRANSCRIBE & INDEX ---")

            # Determine video files to transcribe
            video_files = self._get_video_files(state, config)

            # Caption-first mode: if no files need transcription but we have caption transcripts,
            # proceed directly to embedding computation
            caption_count = len(state.transcripts)
            if not video_files:
                if caption_count > 0:
                    # All videos have captions - skip Whisper, just compute embeddings
                    print(f"  All {caption_count} videos have captions - skipping Whisper transcription")
                    logger.info(f"Caption-first: all {caption_count} videos have caption transcripts")
                    transcripts = {}
                else:
                    print("  ! No video files found")
                    warnings.append("No video files to transcribe")
                    return StageResult.ok({'transcripts': {}}, warnings)
            else:
                print(f"  Found {len(video_files)} files to process")

                # Transcription
                transcripts = self._transcribe_videos(video_files, config)

                # Merge Whisper transcripts with existing caption transcripts (don't overwrite!)
                # Caption transcripts are added by CaptionStage before this stage runs
                state.transcripts.update(transcripts)
                print(f"  + Transcribed {len(transcripts)} videos (+ {caption_count} from captions)")

            # Handle silent videos
            self._handle_silent_videos(video_files, transcripts, config)

            # Compute embeddings for ALL transcripts (both caption and Whisper)
            embeddings_result = self._compute_embeddings(state.transcripts, state, config)
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
        # Need either downloaded videos, audio files, OR existing transcripts from captions
        has_videos = len(state.downloaded_videos) > 0
        has_audio = len(state.downloaded_audio) > 0
        has_caption_transcripts = len(state.transcripts) > 0

        # In caption-first mode, we might have transcripts without downloads
        # (when all videos have captions)
        if not has_videos and not has_audio and not has_caption_transcripts:
            return "No videos, audio files, or caption transcripts available"
        return None

    # === Helper Methods ===

    def _get_video_files(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> List[Path]:
        """Determine which files to transcribe"""
        video_files = []

        # Get base video directory for resolving relative paths
        videos_dir = Path(config.downloaded_videos_dir)

        # Check if audio-first mode with audio files
        if state.downloaded_audio:
            # Resolve relative paths against videos_dir
            video_files = []
            for ad in state.downloaded_audio:
                p = Path(ad.file)
                video_files.append(p if p.is_absolute() else videos_dir / p)
        # Use downloaded videos
        elif state.downloaded_videos:
            # Resolve relative paths against videos_dir
            video_files = []
            for dv in state.downloaded_videos:
                p = Path(dv.file)
                video_files.append(p if p.is_absolute() else videos_dir / p)
        else:
            # Fallback: scan directory
            videos_dir = Path(config.downloaded_videos_dir)
            video_files = (
                list(videos_dir.rglob('*.mp4')) +
                list(videos_dir.rglob('*.webm')) +
                list(videos_dir.rglob('*.mp3'))
            )

        # Caption-first mode: filter out videos that already have transcripts from captions
        if state.videos_need_audio or state.caption_downloads:
            video_files = self._filter_captioned_videos(video_files, state, config)

        return video_files

    def _filter_captioned_videos(
        self,
        video_files: List[Path],
        state: 'PipelineState',
        config: 'Config'
    ) -> List[Path]:
        """Filter out videos that already have transcripts from captions"""
        # Build set of video IDs that need Whisper (no captions available)
        videos_need_whisper = set(state.videos_need_audio)

        # If caption-first mode is active, only transcribe videos that need audio fallback
        caption_config = getattr(config.download, 'caption_first', None)
        # Handle both dict and object config patterns (Rule 6)
        if isinstance(caption_config, dict):
            is_enabled = caption_config.get('enabled', False)
        else:
            is_enabled = getattr(caption_config, 'enabled', False) if caption_config else False
        if is_enabled:
            if state.videos_need_audio:
                # Only transcribe videos explicitly marked as needing audio
                filtered = []
                for vf in video_files:
                    video_id = self._extract_video_id(str(vf))
                    if video_id and video_id in videos_need_whisper:
                        filtered.append(vf)
                        logger.debug(f"  Including {vf.name} for Whisper (no caption)")
                    elif video_id:
                        logger.debug(f"  Skipping {vf.name} (has caption)")

                skipped = len(video_files) - len(filtered)
                if skipped > 0:
                    print(f"  Skipping {skipped} videos with captions (using cached transcripts)")
                return filtered

        return video_files

    def _extract_video_id(self, path_or_url: str) -> Optional[str]:
        """Extract YouTube video ID from path or URL"""
        import re
        patterns = [
            r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'([a-zA-Z0-9_-]{11})(?:\.mp[34]|\.webm|\.mkv|\.m4a)?$',
        ]
        for pattern in patterns:
            match = re.search(pattern, path_or_url)
            if match:
                return match.group(1)
        return None

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

        # Find videos with no transcription
        silent_videos = [
            vf for vf in video_files
            if str(vf) not in transcripts or not transcripts.get(str(vf))
        ]

        if not silent_videos:
            return

        logger.info(f"Found {len(silent_videos)} silent videos for description generation")
        # TODO: Implement LLM-based description generation for silent videos

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

            # Check if caption-first mode is enabled
            # In caption-first mode, caption segments ARE the primary candidates
            # and should NOT be filtered out during matching
            caption_first_enabled = False
            try:
                # Handle both dict and object access (Rule 6)
                download_cfg = config.download
                if hasattr(download_cfg, 'caption_first'):
                    cf_config = download_cfg.caption_first
                    if hasattr(cf_config, 'enabled'):
                        caption_first_enabled = cf_config.enabled
                    elif isinstance(cf_config, dict):
                        caption_first_enabled = cf_config.get('enabled', False)
                elif isinstance(download_cfg, dict):
                    cf_config = download_cfg.get('caption_first', {})
                    caption_first_enabled = cf_config.get('enabled', False) if cf_config else False
                logger.info(f"Caption-first mode enabled: {caption_first_enabled}")
            except Exception as e:
                logger.warning(f"Could not check caption_first config: {e}")

            for video_path, segments in transcripts.items():
                # Check if this is a caption-only video (no actual file downloaded)
                # Video IDs are 11 characters, alphanumeric with optional hyphens
                # BUT: In caption-first mode, don't mark as caption_only - they ARE primary candidates
                is_caption_only = False
                if not caption_first_enabled:
                    if len(video_path) == 11 and video_path.replace('_', '').replace('-', '').isalnum():
                        is_caption_only = True
                    elif not Path(video_path).exists():
                        # Also check if path doesn't exist (may be video_id only)
                        is_caption_only = True

                for seg in segments:
                    if hasattr(seg, 'text'):
                        # TranscriptSegment object (from Whisper)
                        texts.append({
                            'text': seg.text,
                            'video_path': video_path,
                            'start_time': seg.start_time,
                            'end_time': seg.end_time,
                            'transcript_source': getattr(seg, 'transcript_source', 'whisper'),
                            'caption_only': is_caption_only,
                        })
                    else:
                        # Dict segment (from captions or cache)
                        # Handle both 'start'/'end' (captions) and 'start_time'/'end_time' (cache) keys
                        texts.append({
                            'text': seg.get('text', ''),
                            'video_path': video_path,
                            'start_time': seg.get('start_time', seg.get('start', 0)),
                            'end_time': seg.get('end_time', seg.get('end', 0)),
                            'transcript_source': seg.get('transcript_source', 'whisper'),
                            'caption_only': is_caption_only,
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
        Rebuild state.text_metadata from cached transcripts AND captions.

        This is critical for --match-only mode: when TRANSCRIBE is skipped,
        state.text_metadata would be empty, losing all B-roll flags added
        by SCENE_DETECTION.

        Strategy:
        1. Load cached transcripts (from .cache/transcriptions/)
        2. Load cached captions (from .cache/captions/) - CRITICAL for caption-first mode
        3. Rebuild text_metadata list from both sources
        4. Let SCENE_DETECTION update it with is_broll flags
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

            # Get all cached transcript files (Whisper transcriptions)
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
                                # Store transcript with the key from cache
                                state.transcripts[video_path] = data

                                # Rebuild text_metadata
                                # Handle both 'start'/'end' (captions) and 'start_time'/'end_time' (cache)
                                for seg in data:
                                    if isinstance(seg, dict):
                                        texts.append({
                                            'text': seg.get('text', ''),
                                            'video_path': video_path,
                                            'start_time': seg.get('start_time', seg.get('start', 0)),
                                            'end_time': seg.get('end_time', seg.get('end', 0)),
                                            'transcript_source': seg.get('transcript_source', 'whisper'),
                                        })
                except Exception as e:
                    logger.debug(f"Could not load transcript {tf}: {e}")
                    continue

            whisper_count = len(texts)

            # CRITICAL: Also load caption-based transcripts (caption-first mode)
            # These are stored as SRT files in .cache/captions/
            caption_dir = cache_dir / "captions"
            if caption_dir.exists():
                from ..downloader.caption_fetcher import CaptionFetcher
                fetcher = CaptionFetcher(str(cache_dir))

                caption_files = list(caption_dir.glob("*.srt"))
                logger.info(f"Found {len(caption_files)} caption files in cache")

                for cf in caption_files:
                    try:
                        # Extract video_id from filename (format: {video_id}.{lang}.srt)
                        parts = cf.stem.split('.')
                        if len(parts) >= 2:
                            video_id = parts[0]
                        else:
                            video_id = cf.stem

                        # Skip if already have this video from transcriptions
                        if video_id in state.transcripts:
                            continue

                        # Parse the SRT file
                        segments = fetcher.parse_caption_file(str(cf))
                        if segments:
                            # Store with video_id as key (not file path - we don't have files yet)
                            state.transcripts[video_id] = segments

                            # Add to text_metadata with video_id as video_path
                            # This is what MATCH stage needs to create valid matches
                            # NOTE: Don't mark as caption_only - in caption-first mode these ARE
                            # the primary candidates and shouldn't be filtered out during matching
                            for seg in segments:
                                texts.append({
                                    'text': seg.get('text', ''),
                                    'video_path': video_id,  # Use video_id as path in caption-first mode
                                    'start_time': seg.get('start', seg.get('start_time', 0)),
                                    'end_time': seg.get('end', seg.get('end_time', 0)),
                                    'transcript_source': 'caption',
                                })
                    except Exception as e:
                        logger.debug(f"Could not load caption {cf}: {e}")
                        continue

            caption_count = len(texts) - whisper_count

            state.text_metadata = texts
            logger.info(f"Rebuilt text_metadata: {len(texts)} entries ({whisper_count} whisper, {caption_count} captions) from {len(state.transcripts)} videos")

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
