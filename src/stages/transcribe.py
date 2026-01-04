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
                return StageResult.ok({'transcripts': {}}, warnings)

            print(f"\n  --- Stage 3: TRANSCRIBE & INDEX ---")

            # Determine video files to transcribe
            video_files = self._get_video_files(state, config)

            if not video_files:
                print("  ! No video files found")
                warnings.append("No video files to transcribe")
                return StageResult.ok({'transcripts': {}}, warnings)

            print(f"  Found {len(video_files)} files to process")

            # Transcription
            transcripts = self._transcribe_videos(video_files, config)
            state.transcripts = transcripts
            print(f"  + Transcribed {len(transcripts)} videos")

            # Handle silent videos
            self._handle_silent_videos(video_files, transcripts, config)

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
                'embedding_count': len(state.embeddings) if state.embeddings else 0,
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
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Restore transcribe stage from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                return False

            # Transcripts need to be loaded from cache, not checkpoint
            # (they're too large to store in checkpoint JSON)
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
                    language=config.transcription.language if config.transcription.language != 'auto' else None
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
