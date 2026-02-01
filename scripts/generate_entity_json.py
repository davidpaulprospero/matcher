#!/usr/bin/env python3
"""
Generate .entity.json metadata files for existing entity images.

Smart mode tries to auto-match images to entities using:
1. checkpoint.json - if entity_images stage data exists
2. saved_keywords.json - entities extracted from voiceover
3. .meta.json files - description/tags may contain entity names
4. Fuzzy matching on file content

Usage:
    # Auto-detect from project data and config files
    python generate_entity_json.py <project_dir>

    # With install directory (to read config.yaml)
    python generate_entity_json.py <project_dir> --install-dir <path>

    # Specify images directory explicitly
    python generate_entity_json.py <project_dir> --images-dir <path>

    # Interactive mode for unmatched images
    python generate_entity_json.py <project_dir> --interactive

    # Use a manual entity map
    python generate_entity_json.py <project_dir> --entity-map entity_map.json

Entity map JSON format (optional):
{
    "image_filename_without_ext": {
        "entity_name": "Person Name",
        "entity_type": "PERSON"
    }
}
"""

import json
import os
import sys
import re
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB"""
    return path.stat().st_size / (1024 * 1024)


def load_yaml_config(path: Path) -> Optional[Dict]:
    """Load a YAML config file"""
    if not HAS_YAML:
        return None
    if not path.exists():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load {path}: {e}")
        return None


def find_images_dir_from_config(
    project_dir: Path,
    install_dir: Path = None
) -> Optional[Path]:
    """
    Find entity images directory by reading config files.

    Checks:
    1. config.yaml in install_dir
    2. Combines image_search.root_dir + folder derived from project name
    """
    if not HAS_YAML:
        print("Warning: PyYAML not installed, cannot read config files")
        return None

    # Load install config
    install_config = {}
    if install_dir:
        install_config = load_yaml_config(install_dir / "config.yaml") or {}

    # Get image_search settings from install config
    image_search = install_config.get('image_search', {})

    root_dir = image_search.get('root_dir', '')
    folder_name = image_search.get('folder_name', 'images')

    if root_dir:
        # Use root_dir + project folder name pattern
        root_path = Path(root_dir)
        if root_path.exists():
            # Project folder name is typically the last part of project_dir
            project_name = project_dir.name

            # Check for exact match
            images_path = root_path / project_name
            if images_path.exists():
                return images_path

            # Check for partial match (project name might be truncated)
            for subdir in root_path.iterdir():
                if subdir.is_dir() and project_name.startswith(subdir.name[:10]):
                    return subdir

            # Check if project name appears in any subdir
            for subdir in root_path.iterdir():
                if subdir.is_dir() and subdir.name in project_name:
                    return subdir

    return None


def fuzzy_match(s1: str, s2: str) -> float:
    """Calculate fuzzy similarity between two strings (0.0-1.0)"""
    s1 = s1.lower().strip()
    s2 = s2.lower().strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


def load_checkpoint(project_dir: Path) -> Optional[Dict]:
    """Load checkpoint.json if it exists"""
    checkpoint_path = project_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    return None


def load_saved_keywords(project_dir: Path) -> Optional[Dict]:
    """Load saved_keywords.json if it exists"""
    keywords_path = project_dir / "saved_keywords.json"
    if keywords_path.exists():
        try:
            with open(keywords_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load saved_keywords: {e}")
    return None


def extract_entities_from_project(project_dir: Path) -> List[Dict]:
    """Extract entities from checkpoint or saved_keywords"""
    entities = []

    # Try checkpoint first
    checkpoint = load_checkpoint(project_dir)
    if checkpoint:
        # Check entity_images stage
        entity_images = checkpoint.get('entity_images', {})
        if entity_images.get('results'):
            for name, data in entity_images['results'].items():
                entities.append({
                    'name': name,
                    'type': data.get('entity_type', 'PERSON'),
                    'query': data.get('query', ''),
                    'source': 'checkpoint'
                })
            print(f"Found {len(entities)} entities from checkpoint.json")
            return entities

        # Check analyze stage
        analyze = checkpoint.get('analyze', {})
        if analyze.get('entities'):
            for ent in analyze['entities']:
                entities.append({
                    'name': ent.get('text', ''),
                    'type': ent.get('type', 'PERSON'),
                    'query': '',
                    'source': 'checkpoint_analyze'
                })
            print(f"Found {len(entities)} entities from checkpoint analyze stage")
            return entities

    # Try saved_keywords
    saved = load_saved_keywords(project_dir)
    if saved:
        # Check for entities at top level
        if saved.get('entities'):
            for ent in saved['entities']:
                entities.append({
                    'name': ent.get('text', ''),
                    'type': ent.get('type', 'PERSON'),
                    'query': ent.get('search_keyword', ''),
                    'source': 'saved_keywords'
                })
            print(f"Found {len(entities)} entities from saved_keywords.json")
            return entities

        # Check for entities inside presets (saved_keywords structure)
        presets = saved.get('presets', {})
        if presets:
            # Use the most recent preset (sorted by name which is timestamp)
            latest_preset_name = sorted(presets.keys())[-1] if presets else None
            if latest_preset_name:
                preset = presets[latest_preset_name]
                preset_entities = preset.get('entities', [])
                for ent in preset_entities:
                    entities.append({
                        'name': ent.get('text', ''),
                        'type': ent.get('type', 'PERSON'),
                        'query': ent.get('search_keyword', ''),
                        'source': f'saved_keywords/{latest_preset_name}'
                    })
                print(f"Found {len(entities)} entities from saved_keywords.json (preset: {latest_preset_name})")
                return entities

    return entities


def match_image_to_entity(
    img_path: Path,
    meta_path: Optional[Path],
    entities: List[Dict],
    threshold: float = 0.6
) -> Optional[Dict]:
    """Try to match an image to an entity using various heuristics"""

    # Load meta.json if it exists
    meta_data = {}
    if meta_path and meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
        except:
            pass

    # Build searchable text from meta
    search_text = " ".join([
        meta_data.get('description', ''),
        meta_data.get('photographer', ''),
        " ".join(meta_data.get('tags', [])),
        img_path.stem  # filename without extension
    ]).lower()

    best_match = None
    best_score = 0.0

    for entity in entities:
        entity_name = entity['name'].lower()

        # Check if entity name appears in search text
        if entity_name in search_text:
            return entity  # Exact match in text

        # Fuzzy match against description
        score = fuzzy_match(entity_name, meta_data.get('description', ''))

        # Also try matching first/last name for PERSON entities
        if entity['type'] == 'PERSON' and ' ' in entity_name:
            name_parts = entity_name.split()
            for part in name_parts:
                if len(part) > 2 and part in search_text:
                    score = max(score, 0.7)  # Partial name match

        if score > best_score and score >= threshold:
            best_score = score
            best_match = entity

    return best_match


def find_images_dir(project_dir: Path, install_dir: Path = None) -> Optional[Path]:
    """Find the entity images directory in a project"""

    # First try config-based detection
    config_path = find_images_dir_from_config(project_dir, install_dir)
    if config_path and config_path.exists():
        # Check if it has image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        has_images = any(
            f.suffix.lower() in image_extensions
            for f in config_path.iterdir() if f.is_file()
        )
        if has_images:
            print(f"Found images dir from config: {config_path}")
            return config_path

    # Fallback to common locations
    candidates = [
        project_dir / "output" / "entity_images",
        project_dir / "entity_images",
        project_dir,  # Images directly in project dir
    ]

    for candidate in candidates:
        if candidate.exists():
            # Check if it has image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
            has_images = any(
                f.suffix.lower() in image_extensions
                for f in candidate.iterdir() if f.is_file()
            )
            if has_images:
                return candidate

    return None


def generate_entity_json(
    project_dir: str,
    images_dir: str = None,
    entity_map: dict = None,
    interactive: bool = False,
    install_dir: str = None
):
    """Generate .entity.json files for images missing metadata."""
    project_path = Path(project_dir)
    install_path = Path(install_dir) if install_dir else Path(__file__).parent

    if not project_path.exists():
        print(f"Error: Project directory not found: {project_dir}")
        return

    # Find images directory
    if images_dir:
        images_path = Path(images_dir)
    else:
        images_path = find_images_dir(project_path, install_path)
        if not images_path:
            print(f"Error: Could not find images directory in {project_dir}")
            print("Use --images-dir to specify the path, or --install-dir to locate config.yaml")
            return

    print(f"Images directory: {images_path}")

    # Extract entities from project data
    entities = extract_entities_from_project(project_path)

    # Find all image files (recursive search)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    image_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.gif']:
        image_files.extend(images_path.rglob(pattern))

    print(f"Found {len(image_files)} image files")
    print(f"Found {len(entities)} entities from project data")

    generated = 0
    skipped = 0
    unmatched = []

    for img_path in sorted(image_files):
        # Check if .entity.json already exists
        entity_json_path = img_path.parent / f"{img_path.stem}.entity.json"

        if entity_json_path.exists():
            skipped += 1
            continue

        # Try to find corresponding .meta.json
        meta_path = img_path.parent / f"{img_path.stem}.meta.json"

        entity_name = None
        entity_type = "PERSON"
        query = ""

        # Check manual entity map first
        if entity_map and img_path.stem in entity_map:
            entry = entity_map[img_path.stem]
            entity_name = entry.get('entity_name')
            entity_type = entry.get('entity_type', 'PERSON')
            query = f"{entity_name} (from entity map)"

        # Try auto-matching
        if not entity_name and entities:
            matched = match_image_to_entity(img_path, meta_path, entities)
            if matched:
                entity_name = matched['name']
                entity_type = matched['type']
                query = matched.get('query', f"{entity_name} (auto-matched)")
                print(f"  Auto-matched: {img_path.name} -> {entity_name} ({entity_type})")

        # Interactive prompt for unmatched
        if not entity_name:
            if interactive:
                print(f"\nImage: {img_path.name}")

                # Show meta info if available
                if meta_path.exists():
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        if meta.get('description'):
                            print(f"  Description: {meta['description']}")
                    except:
                        pass

                # Show available entities
                if entities:
                    print(f"  Available entities: {', '.join(e['name'] for e in entities[:10])}")

                entity_name = input("  Entity name (or 'skip'/'quit'): ").strip()

                if entity_name.lower() == 'quit':
                    break
                if entity_name.lower() == 'skip' or not entity_name:
                    unmatched.append(img_path.name)
                    continue

                entity_type = input("  Entity type [PERSON/GPE/ORG/EVENT] (default: PERSON): ").strip().upper()
                if not entity_type:
                    entity_type = "PERSON"
                query = f"{entity_name} (manually entered)"
            else:
                unmatched.append(img_path.name)
                continue

        # Generate the .entity.json
        metadata = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "query": query,
            "source": "generated",
            "file_size_mb": round(get_file_size_mb(img_path), 2)
        }

        with open(entity_json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Created: {entity_json_path.name}")
        generated += 1

    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Generated: {generated}")
    print(f"  Already existed: {skipped}")
    print(f"  Unmatched: {len(unmatched)}")

    if unmatched:
        print(f"\nUnmatched images (run with --interactive to add manually):")
        for name in unmatched[:20]:
            print(f"  - {name}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    project_dir = sys.argv[1]
    images_dir = None
    entity_map = None
    interactive = False
    install_dir = None

    # Parse arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--images-dir' and i + 1 < len(sys.argv):
            images_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--install-dir' and i + 1 < len(sys.argv):
            install_dir = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--entity-map' and i + 1 < len(sys.argv):
            map_file = sys.argv[i + 1]
            try:
                with open(map_file, 'r', encoding='utf-8') as f:
                    entity_map = json.load(f)
                print(f"Loaded entity map with {len(entity_map)} entries")
            except Exception as e:
                print(f"Warning: Could not load entity map: {e}")
            i += 2
        elif sys.argv[i] == '--interactive':
            interactive = True
            i += 1
        else:
            i += 1

    generate_entity_json(project_dir, images_dir, entity_map, interactive, install_dir)


if __name__ == '__main__':
    main()
