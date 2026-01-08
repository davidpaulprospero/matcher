"""
Shared utility functions for media downloads.

Extracted from entity_images.py top-level functions (Jan 7, 2026):
- build_entity_query() (lines 1331-1372)
- check_local_entity_images() (lines 1375-1426)
- map_entities_to_segments() (lines 1701-1726)
- restore_entity_images_from_disk() (lines 1729-1818)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import EntityImageResult

logger = logging.getLogger(__name__)


def build_entity_query(entity: Dict, topic: str = "") -> str:
    """
    Build a trimmed search query from entity + context + topic.

    Extracted from entity_images.py lines 1331-1372.

    Args:
        entity: Dict with 'text', 'type', 'context' keys
        topic: Overall documentary topic

    Returns:
        Trimmed search query string
    """
    name = entity.get('text', '')
    context = entity.get('context', '')
    etype = entity.get('type', '')

    if not name:
        return ""

    # Build query parts
    parts = [name]

    # Add context (trimmed)
    if context:
        # Take first 2-3 words of context
        context_words = context.split()[:3]
        parts.extend(context_words)

    # Add topic keywords (trimmed)
    if topic:
        # Extract 2-3 key words from topic
        topic_words = topic.replace('documentary', '').replace('footage', '').split()
        topic_words = [w for w in topic_words if len(w) > 3][:2]
        parts.extend(topic_words)

    # Join and clean
    query = ' '.join(parts)

    # Limit length
    if len(query) > 60:
        query = ' '.join(query.split()[:6])

    return query


def check_local_entity_images(
    images_dir: str,
    entity_name: str,
    entity_type: str = ""
) -> List[str]:
    """
    Check if entity images already exist locally.

    Extracted from entity_images.py lines 1375-1426.

    Scans for .entity.json files matching this entity and returns
    paths to existing image files.

    Args:
        images_dir: Directory containing downloaded images
        entity_name: Entity name to search for
        entity_type: Optional entity type filter

    Returns:
        List of existing image file paths for this entity
    """
    images_path = Path(images_dir)
    if not images_path.exists():
        logger.debug(f"check_local_entity_images: Directory not found: {images_dir}")
        return []

    matching_images = []

    # Find .entity.json files for this entity
    meta_files = list(images_path.glob("*.entity.json"))

    for meta_path in meta_files:
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            meta_entity_name = meta.get('entity_name', '')
            # Match by entity name (case-insensitive)
            if meta_entity_name.lower() == entity_name.lower():
                # Find corresponding image file
                # meta_path is like "28207239.entity.json", need to get "28207239"
                base_name = meta_path.stem  # "28207239.entity"
                if base_name.endswith('.entity'):
                    base_name = base_name[:-7]  # Strip ".entity" -> "28207239"

                for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']:
                    img_path = meta_path.parent / f"{base_name}{ext}"
                    if img_path.exists():
                        matching_images.append(str(img_path))
                        break
        except (json.JSONDecodeError, IOError):
            continue

    return matching_images


def map_entities_to_segments(
    entities: List[Dict],
    voiceover_segments: List[Dict]
) -> Dict[str, List[int]]:
    """
    Map entity names to segment indices where they appear.

    Extracted from entity_images.py lines 1701-1726.

    Args:
        entities: List of entity dicts with 'text' key
        voiceover_segments: List of segment dicts with 'text' key

    Returns:
        Dict mapping entity name to list of segment indices
    """
    entity_segments: Dict[str, List[int]] = {}

    for entity in entities:
        entity_name = entity.get('text', '')
        if not entity_name:
            continue

        entity_segments[entity_name] = []

        for i, seg in enumerate(voiceover_segments):
            seg_text = seg.get('text', '').lower() if isinstance(seg, dict) else getattr(seg, 'text', '').lower()

            if entity_name.lower() in seg_text:
                entity_segments[entity_name].append(i)

    return entity_segments


def restore_entity_images_from_disk(
    images_dir: str,
    voiceover_segments: Optional[List[Dict]] = None
) -> Dict[str, 'EntityImageResult']:
    """
    Scan images directory for .entity.json metadata files and reconstruct
    EntityImageResult objects. Used for checkpoint resume.

    Extracted from entity_images.py lines 1729-1818.

    Args:
        images_dir: Path to project images directory
        voiceover_segments: Optional voiceover segments for segment mapping

    Returns:
        Dict mapping entity_name -> EntityImageResult
    """
    from .models import EntityImageResult

    images_path = Path(images_dir)
    if not images_path.exists():
        logger.info(f"Images directory not found: {images_dir}")
        return {}

    # Group images by entity name
    entity_images: Dict[str, List[Tuple[str, Dict]]] = {}  # name -> [(image_path, metadata), ...]

    # Find all .entity.json metadata files
    meta_files = list(images_path.glob("**/*.entity.json"))
    logger.info(f"Found {len(meta_files)} entity metadata files in {images_dir}")

    for meta_path in meta_files:
        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            entity_name = metadata.get('entity_name', '')
            if not entity_name:
                continue

            # Find corresponding image file (same name, different extension)
            # meta_path is like "img1.entity.json", we need "img1.jpg"
            image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.gif']
            image_path = None

            # Remove .entity.json suffix to get base name
            base_name = str(meta_path).replace('.entity.json', '')

            for ext in image_extensions:
                candidate = Path(base_name + ext)
                if candidate.exists():
                    image_path = str(candidate)
                    break

            if not image_path:
                logger.debug(f"No image file found for metadata: {meta_path}")
                continue

            if entity_name not in entity_images:
                entity_images[entity_name] = []

            entity_images[entity_name].append((image_path, metadata))

        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read metadata file {meta_path}: {e}")
            continue

    # Build EntityImageResult objects
    results: Dict[str, EntityImageResult] = {}

    for entity_name, image_list in entity_images.items():
        if not image_list:
            continue

        # Use first metadata entry for entity info
        first_meta = image_list[0][1]

        results[entity_name] = EntityImageResult(
            entity_name=entity_name,
            entity_type=first_meta.get('entity_type', ''),
            context=first_meta.get('context', ''),
            query=first_meta.get('query', ''),
            images=[img_path for img_path, _ in image_list],
            segment_indices=[]
        )

    # Map to voiceover segments if provided
    if voiceover_segments and results:
        # Build simple entity list for mapping function
        entities = [{'text': name, 'type': r.entity_type} for name, r in results.items()]
        entity_segments = map_entities_to_segments(entities, voiceover_segments)

        for entity_name, result in results.items():
            result.segment_indices = entity_segments.get(entity_name, [])

    total_images = sum(len(r.images) for r in results.values())
    logger.info(f"Restored {total_images} images for {len(results)} entities from disk")

    return results
