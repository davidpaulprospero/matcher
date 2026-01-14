"""
B-roll Match Stage - Match Silent Scenes to Voiceover

Stage runs after main matching:
- Detects silent/minimal-speech scenes (< 10 words)
- Enriches scene descriptions (Vision API or filename keywords)
- Matches silent scenes to voiceover segments for V8 track
- Multi-strategy scoring: embedding + keyword + entity + source boost

Created during B-roll improvement refactoring (Jan 2026).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from dataclasses import dataclass

from . import Stage, StageResult, register_stage

if TYPE_CHECKING:
    from ..config import Config
    from ..checkpoint import CheckpointManager
    from ..state import PipelineState

logger = logging.getLogger(__name__)


@dataclass
class BrollScene:
    """A silent/B-roll scene detected from video transcripts"""
    source_file: str
    start_time: float
    end_time: float
    word_count: int
    transcript: str
    description: str = ""  # Vision-enriched or keyword-extracted
    embedding: Any = None  # Numpy array
    source: str = ""  # 'youtube', 'pexels', 'pixabay', 'broll'


@dataclass
class BrollMatch:
    """A match between voiceover segment and B-roll scene"""
    segment_index: int
    scene: BrollScene
    score: float
    embedding_score: float = 0.0
    keyword_score: float = 0.0
    entity_score: float = 0.0
    source_boost: float = 0.0


@register_stage
class BrollMatchStage(Stage):
    """
    Matches silent scenes to voiceover segments for V8 track.

    Inputs:
        - state.transcripts: Video transcripts
        - state.text_metadata: Segment metadata with word counts
        - state.voiceover_segments: Voiceover to match against
        - state.keywords: Keywords for keyword overlap scoring
        - state.extracted_entities: Entities for entity matching

    Outputs:
        - state.broll_matches: List of BrollMatch objects

    Configuration (config.broll):
        - min_words_threshold: Scenes with fewer words = B-roll (default: 10)
        - embedding_weight: Weight for embedding similarity (default: 0.4)
        - keyword_weight: Weight for keyword overlap (default: 0.35)
        - entity_weight: Weight for entity matching (default: 0.25)
        - source_boost: Score boosts per source {youtube: 0.1, pexels: 0.05}
        - min_match_score: Threshold for "good" match (default: 0.3)
        - always_match: Always pick best B-roll (default: true)
        - max_matches_per_segment: Top N options per segment (default: 3)
    """

    name = "BROLL_MATCH"
    description = "Match silent scenes to voiceover for V8 track"

    def run(
        self,
        state: 'PipelineState',
        config: 'Config',
        checkpoint: 'CheckpointManager'
    ) -> StageResult:
        """Execute the B-roll matching stage"""
        warnings = []

        # Check master enable
        broll_config = getattr(config, 'broll', None)
        if not broll_config or not getattr(broll_config, 'enabled', True):
            logger.debug("B-roll disabled in config")
            return StageResult.ok({
                'skipped': True,
                'reason': 'broll_disabled'
            })

        # Check if we have voiceover segments
        if not state.voiceover_segments:
            logger.debug("No voiceover segments to match against")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_voiceover'
            })

        # Check if we have transcripts
        if not state.transcripts and not state.text_metadata:
            logger.debug("No transcripts available for B-roll detection")
            return StageResult.ok({
                'skipped': True,
                'reason': 'no_transcripts'
            })

        print(f"\n  ─── Stage: B-ROLL MATCHING ───")

        try:
            # Detect silent scenes
            silent_scenes = self._detect_silent_scenes(state, broll_config)

            if not silent_scenes:
                logger.info("No silent scenes detected")
                print("  No silent scenes found")
                return StageResult.ok({
                    'silent_count': 0,
                    'match_count': 0
                })

            print(f"  Detected {len(silent_scenes)} silent scenes")

            # Check if Vision API should be used
            use_vision = self._should_use_vision(config, broll_config)

            # Enrich descriptions
            if use_vision:
                enriched_count = self._enrich_with_vision(
                    silent_scenes, config, broll_config
                )
                print(f"  ✓ Vision API enriched {enriched_count} scenes")
            else:
                enriched_count = self._enrich_with_keywords(
                    silent_scenes, state, broll_config
                )
                print(f"  ✓ Keyword-enriched {enriched_count} scenes")

            # Create embeddings for scenes
            self._create_scene_embeddings(silent_scenes, config)

            # Match scenes to voiceover segments
            matches = self._match_scenes_to_voiceover(
                silent_scenes,
                state,
                config,
                broll_config
            )

            print(f"  ✓ Matched {len(matches)} scenes to voiceover")

            # Store in state
            if not hasattr(state, 'broll_matches'):
                state.broll_matches = []

            for match in matches:
                state.broll_matches.append({
                    'segment_index': match.segment_index,
                    'source_file': match.scene.source_file,
                    'start_time': match.scene.start_time,
                    'end_time': match.scene.end_time,
                    'score': match.score,
                    'embedding_score': match.embedding_score,
                    'keyword_score': match.keyword_score,
                    'entity_score': match.entity_score,
                    'description': match.scene.description,
                    'source': match.scene.source
                })

            # Build checkpoint data
            checkpoint_data = {
                'silent_count': len(silent_scenes),
                'match_count': len(matches),
                'vision_used': use_vision,
                'matches': [
                    {
                        'segment_index': m.segment_index,
                        'source_file': m.scene.source_file,
                        'score': m.score
                    }
                    for m in matches
                ]
            }

            return StageResult.ok(checkpoint_data, warnings=warnings)

        except Exception as e:
            logger.error(f"B-roll matching failed: {e}", exc_info=True)
            print(f"  ⚠ B-roll matching failed: {e}")
            return StageResult.fail(str(e))

    def can_skip(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager'
    ) -> bool:
        """Check if this stage can be skipped"""
        return checkpoint.should_skip_stage(self.name)

    def restore(
        self,
        state: 'PipelineState',
        checkpoint: 'CheckpointManager',
        config: 'Config' = None
    ) -> bool:
        """Restore B-roll matches from checkpoint"""
        try:
            data = checkpoint.get_stage_data(self.name)
            if not data:
                logger.warning(f"No checkpoint data for {self.name}")
                return False

            # Initialize broll_matches if needed
            if not hasattr(state, 'broll_matches'):
                state.broll_matches = []

            # Restore matches from checkpoint
            for match in data.get('matches', []):
                state.broll_matches.append({
                    'segment_index': match.get('segment_index', 0),
                    'source_file': match.get('source_file', ''),
                    'score': match.get('score', 0.0),
                    'start_time': match.get('start_time', 0.0),
                    'end_time': match.get('end_time', 0.0),
                    'description': match.get('description', ''),
                    'source': match.get('source', '')
                })

            logger.info(f"Restored {self.name}: {len(state.broll_matches)} matches")
            return True

        except Exception as e:
            logger.error(f"Failed to restore {self.name}: {e}", exc_info=True)
            return False

    def validate_inputs(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Optional[str]:
        """Validate inputs before running"""
        return None

    def _detect_silent_scenes(
        self,
        state: 'PipelineState',
        broll_config
    ) -> List[BrollScene]:
        """
        Detect silent/minimal-speech scenes from transcripts.

        A scene is considered B-roll if:
        1. Word count < min_words_threshold (default: 10)
        2. Contains only ignore markers ([Music], [Silence], etc.)
        """
        min_words = getattr(broll_config, 'min_words_threshold', 10)
        ignore_markers = getattr(broll_config, 'ignore_markers', [
            '[Music]', '[Applause]', '[Silence]'
        ])

        silent_scenes = []

        # Check text_metadata for word counts and is_broll flag
        for meta in state.text_metadata:
            is_broll = meta.get('is_broll', False)
            text = meta.get('text', '')
            source_file = meta.get('source_file', '')
            start_time = meta.get('start_time', 0.0)
            end_time = meta.get('end_time', 0.0)

            # Skip if already marked as B-roll (from face detection)
            if is_broll:
                # Still include it for matching
                silent_scenes.append(BrollScene(
                    source_file=source_file,
                    start_time=start_time,
                    end_time=end_time,
                    word_count=0,
                    transcript=text,
                    source=self._get_video_source(source_file, state)
                ))
                continue

            # Count words (excluding ignore markers)
            clean_text = text
            for marker in ignore_markers:
                clean_text = clean_text.replace(marker, '')

            word_count = len(clean_text.split())

            # Check if scene is silent
            if word_count < min_words:
                silent_scenes.append(BrollScene(
                    source_file=source_file,
                    start_time=start_time,
                    end_time=end_time,
                    word_count=word_count,
                    transcript=text,
                    source=self._get_video_source(source_file, state)
                ))

        # Also check transcripts dict for videos with no segments
        for video_name, segments in state.transcripts.items():
            if not segments:
                # Video has no transcript = fully silent
                # Find corresponding downloaded video
                video_file = self._find_video_file(video_name, state)
                if video_file:
                    silent_scenes.append(BrollScene(
                        source_file=video_file,
                        start_time=0.0,
                        end_time=0.0,  # Will be determined later
                        word_count=0,
                        transcript='',
                        source=self._get_video_source(video_file, state)
                    ))

        # Handle long videos: sample scenes
        long_threshold = getattr(broll_config, 'long_video_threshold', 600)
        sample_interval = getattr(broll_config, 'sample_interval_seconds', 120)
        max_scenes = getattr(broll_config, 'max_scenes_per_video', 15)

        silent_scenes = self._sample_long_video_scenes(
            silent_scenes,
            long_threshold,
            sample_interval,
            max_scenes
        )

        return silent_scenes

    def _get_video_source(
        self,
        video_file: str,
        state: 'PipelineState'
    ) -> str:
        """Determine source of video (youtube, pexels, pixabay, broll)"""
        path = Path(video_file)
        name_lower = path.stem.lower()

        # Check filename patterns
        if name_lower.startswith('pexels_'):
            return 'pexels'
        if name_lower.startswith('pixabay_'):
            return 'pixabay'

        # Check downloaded_videos source
        for vid in state.downloaded_videos:
            vid_file = vid.file if hasattr(vid, 'file') else vid.get('file', '')
            if Path(vid_file).stem == path.stem:
                source = vid.source if hasattr(vid, 'source') else vid.get('source', '')
                if source:
                    return source

        # Check if in broll directory
        if 'broll' in str(path.parent).lower():
            return 'broll'

        return 'youtube'

    def _find_video_file(
        self,
        video_name: str,
        state: 'PipelineState'
    ) -> Optional[str]:
        """Find full video file path from video name"""
        for vid in state.downloaded_videos:
            vid_file = vid.file if hasattr(vid, 'file') else vid.get('file', '')
            if Path(vid_file).stem == video_name:
                return vid_file
        return None

    def _sample_long_video_scenes(
        self,
        scenes: List[BrollScene],
        long_threshold: int,
        sample_interval: int,
        max_scenes: int
    ) -> List[BrollScene]:
        """Sample scenes from long videos to avoid overwhelming"""
        # Group by source file
        by_file: Dict[str, List[BrollScene]] = {}
        for scene in scenes:
            if scene.source_file not in by_file:
                by_file[scene.source_file] = []
            by_file[scene.source_file].append(scene)

        result = []
        for file, file_scenes in by_file.items():
            if len(file_scenes) > max_scenes:
                # Sample evenly
                step = len(file_scenes) // max_scenes
                result.extend(file_scenes[::step][:max_scenes])
            else:
                result.extend(file_scenes)

        return result

    def _should_use_vision(
        self,
        config: 'Config',
        broll_config
    ) -> bool:
        """Determine if Vision API should be used"""
        # Check if vision is enabled in broll config
        vision_enabled = getattr(broll_config, 'vision_enabled', True)
        if not vision_enabled:
            return False

        # Check audio-first mode (no video frames available)
        audio_config = getattr(config.download, 'audio_first', None)
        if audio_config and getattr(audio_config, 'enabled', False):
            logger.info("Audio-first mode: skipping Vision API for B-roll")
            return False

        # Check global vision config
        vision_config = getattr(config, 'vision', None)
        if not vision_config or not getattr(vision_config, 'enabled', False):
            return False

        return True

    def _enrich_with_vision(
        self,
        scenes: List[BrollScene],
        config: 'Config',
        broll_config
    ) -> int:
        """Enrich scene descriptions using Vision API"""
        try:
            from ..vision import describe_video_frame

            vision_config = getattr(config, 'vision', None)
            frames_per_scene = getattr(broll_config, 'frames_per_scene', 3)
            cache_descriptions = getattr(broll_config, 'cache_descriptions', True)

            enriched_count = 0
            for scene in scenes:
                if scene.description:
                    continue

                try:
                    # Get frame description
                    description = describe_video_frame(
                        scene.source_file,
                        scene.start_time,
                        config=vision_config
                    )
                    if description:
                        scene.description = description
                        enriched_count += 1
                except Exception as e:
                    logger.warning(f"Vision API failed for {scene.source_file}: {e}")
                    # Fallback to keywords
                    scene.description = self._extract_keywords_from_filename(scene.source_file)

            return enriched_count

        except ImportError:
            logger.warning("Vision module not available, using keyword fallback")
            return self._enrich_with_keywords(scenes, None, broll_config)

    def _enrich_with_keywords(
        self,
        scenes: List[BrollScene],
        state: Optional['PipelineState'],
        broll_config
    ) -> int:
        """Enrich scene descriptions using filename keywords"""
        enriched_count = 0
        for scene in scenes:
            if scene.description:
                continue

            description = self._extract_keywords_from_filename(scene.source_file)
            if description:
                scene.description = description
                enriched_count += 1

        return enriched_count

    def _extract_keywords_from_filename(self, filepath: str) -> str:
        """Extract descriptive keywords from filename"""
        name = Path(filepath).stem

        # Remove common prefixes
        for prefix in ['pexels_', 'pixabay_', 'entity_', 'yt_', 'broll_']:
            if name.lower().startswith(prefix):
                name = name[len(prefix):]

        # Replace underscores and hyphens with spaces
        name = re.sub(r'[_\-]+', ' ', name)

        # Remove video IDs (long alphanumeric strings)
        name = re.sub(r'\b[a-zA-Z0-9]{11,}\b', '', name)

        # Remove numbers at end (timestamps, etc.)
        name = re.sub(r'\s*\d+\s*$', '', name)

        # Clean up whitespace
        name = ' '.join(name.split())

        return name.strip()

    def _create_scene_embeddings(
        self,
        scenes: List[BrollScene],
        config: 'Config'
    ):
        """Create embeddings for scene descriptions"""
        try:
            from sentence_transformers import SentenceTransformer

            # Load model
            embedding_config = getattr(config, 'embedding', None)
            model_name = getattr(embedding_config, 'model', 'all-MiniLM-L6-v2') if embedding_config else 'all-MiniLM-L6-v2'

            model = SentenceTransformer(model_name)

            # Create embeddings for scenes with descriptions
            for scene in scenes:
                if scene.description:
                    scene.embedding = model.encode(scene.description)

        except ImportError:
            logger.warning("sentence_transformers not available, skipping embeddings")
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")

    def _match_scenes_to_voiceover(
        self,
        scenes: List[BrollScene],
        state: 'PipelineState',
        config: 'Config',
        broll_config
    ) -> List[BrollMatch]:
        """Match B-roll scenes to voiceover segments using multi-strategy scoring"""
        # Get scoring weights
        embedding_weight = getattr(broll_config, 'embedding_weight', 0.4)
        keyword_weight = getattr(broll_config, 'keyword_weight', 0.35)
        entity_weight = getattr(broll_config, 'entity_weight', 0.25)

        # Get source boosts
        source_boost_config = getattr(broll_config, 'source_boost', None)
        source_boosts = {
            'youtube': getattr(source_boost_config, 'youtube', 0.1) if source_boost_config else 0.1,
            'pexels': getattr(source_boost_config, 'pexels', 0.05) if source_boost_config else 0.05,
            'pixabay': getattr(source_boost_config, 'pixabay', 0.0) if source_boost_config else 0.0,
            'broll': 0.1  # B-roll downloads get boost
        }

        # Get match config
        min_score = getattr(broll_config, 'min_match_score', 0.3)
        always_match = getattr(broll_config, 'always_match', True)
        max_per_segment = getattr(broll_config, 'max_matches_per_segment', 3)

        # Get keywords and entities for scoring
        keywords = set(k.lower() for k in state.keywords)
        entity_names = set()
        for entity in state.extracted_entities:
            name = entity.get('name', '') if isinstance(entity, dict) else str(entity)
            if name:
                entity_names.add(name.lower())

        # Create voiceover embeddings if needed
        vo_embeddings = self._get_voiceover_embeddings(state, config)

        matches = []

        for seg in state.voiceover_segments:
            segment_matches = []

            for scene in scenes:
                # Calculate embedding score
                emb_score = self._calculate_embedding_score(
                    vo_embeddings.get(seg.index),
                    scene.embedding
                )

                # Calculate keyword overlap score
                kw_score = self._calculate_keyword_score(
                    seg.text,
                    scene.description,
                    keywords
                )

                # Calculate entity match score
                ent_score = self._calculate_entity_score(
                    seg.text,
                    scene.description,
                    entity_names
                )

                # Calculate source boost
                boost = source_boosts.get(scene.source, 0.0)

                # Combined score
                total_score = (
                    emb_score * embedding_weight +
                    kw_score * keyword_weight +
                    ent_score * entity_weight +
                    boost
                )

                segment_matches.append(BrollMatch(
                    segment_index=seg.index,
                    scene=scene,
                    score=total_score,
                    embedding_score=emb_score,
                    keyword_score=kw_score,
                    entity_score=ent_score,
                    source_boost=boost
                ))

            # Sort by score and take top matches
            segment_matches.sort(key=lambda m: m.score, reverse=True)
            top_matches = segment_matches[:max_per_segment]

            # Apply min score threshold or always_match
            for match in top_matches:
                if match.score >= min_score or always_match:
                    matches.append(match)
                    break  # Only take best match per segment for V8

        return matches

    def _get_voiceover_embeddings(
        self,
        state: 'PipelineState',
        config: 'Config'
    ) -> Dict[int, Any]:
        """Get or create embeddings for voiceover segments"""
        embeddings = {}

        try:
            from sentence_transformers import SentenceTransformer

            embedding_config = getattr(config, 'embedding', None)
            model_name = getattr(embedding_config, 'model', 'all-MiniLM-L6-v2') if embedding_config else 'all-MiniLM-L6-v2'

            model = SentenceTransformer(model_name)

            for seg in state.voiceover_segments:
                embeddings[seg.index] = model.encode(seg.text)

        except ImportError:
            logger.warning("sentence_transformers not available")
        except Exception as e:
            logger.error(f"Voiceover embedding failed: {e}")

        return embeddings

    def _calculate_embedding_score(
        self,
        vo_embedding: Any,
        scene_embedding: Any
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        if vo_embedding is None or scene_embedding is None:
            return 0.0

        try:
            import numpy as np
            from numpy.linalg import norm

            # Cosine similarity
            similarity = np.dot(vo_embedding, scene_embedding) / (
                norm(vo_embedding) * norm(scene_embedding)
            )
            return float(max(0, similarity))

        except Exception:
            return 0.0

    def _calculate_keyword_score(
        self,
        vo_text: str,
        scene_description: str,
        keywords: set
    ) -> float:
        """Calculate keyword overlap score"""
        if not scene_description:
            return 0.0

        vo_words = set(vo_text.lower().split())
        scene_words = set(scene_description.lower().split())

        # Keywords present in both voiceover and scene
        vo_keywords = vo_words & keywords
        scene_keywords = scene_words & keywords
        overlap = vo_keywords & scene_keywords

        if not vo_keywords:
            # Fallback: direct word overlap
            overlap = vo_words & scene_words
            if len(vo_words) == 0:
                return 0.0
            return len(overlap) / len(vo_words)

        return len(overlap) / len(vo_keywords) if vo_keywords else 0.0

    def _calculate_entity_score(
        self,
        vo_text: str,
        scene_description: str,
        entity_names: set
    ) -> float:
        """Calculate entity match score"""
        if not scene_description or not entity_names:
            return 0.0

        vo_lower = vo_text.lower()
        scene_lower = scene_description.lower()

        # Count entities mentioned in both
        vo_entities = sum(1 for e in entity_names if e in vo_lower)
        scene_entities = sum(1 for e in entity_names if e in scene_lower)

        if vo_entities == 0:
            return 0.0

        matching = sum(1 for e in entity_names if e in vo_lower and e in scene_lower)
        return matching / vo_entities
