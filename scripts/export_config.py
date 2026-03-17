#!/usr/bin/env python3
"""
Project Configuration Export/Import Utility

Exports and imports project configurations including keywords, presets, and settings.

Usage:
    python export_config.py --project "E:/Projects/MyProject" --export --output config_export.json
    python export_config.py --project "E:/Projects/MyProject" --export --encrypt --output config_export.enc
    python export_config.py --project "E:/Projects/MyProject" --import --input config_export.json
    python export_config.py --project "E:/Projects/MyProject" --export --include-cache --output config_export.json
"""

import os
import sys
import json
import argparse
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Change to project root so relative paths work correctly
os.chdir(project_root)

# Import standardized output functions
from script_utils import print_header, print_ok, print_warn, print_error, set_verbosity, get_verbosity


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_keywords(project_dir: Path) -> Dict[str, Any]:
    """Export keyword presets from a project."""
    keywords_path = project_dir / "saved_keywords.json"

    if not keywords_path.exists():
        return {"presets": {}, "count": 0}

    try:
        with open(keywords_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract just the presets
        presets = data.get('presets', {})
        return {
            "presets": presets,
            "count": len(presets),
            "exported_at": datetime.now().isoformat()
        }
    except Exception as e:
        print_warn(f"Failed to export keywords: {e}")
        return {"presets": {}, "count": 0, "error": str(e)}


def export_checkpoint(project_dir: Path) -> Dict[str, Any]:
    """Export checkpoint metadata (not full data) for stage configuration."""
    checkpoint_path = project_dir / "checkpoint.json"

    if not checkpoint_path.exists():
        return {"exists": False}

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)

        # Export only configuration-related fields, not full video data
        return {
            "exists": True,
            "last_completed_stage": checkpoint.get("last_completed_stage"),
            "stages_completed": checkpoint.get("stages_completed", []),
            "stage_timings": checkpoint.get("stage_timings", {}),
            "video_ids_count": len(checkpoint.get("video_ids", [])),
            "matches_count": len(checkpoint.get("matches", [])),
            "exported_at": datetime.now().isoformat()
        }
    except Exception as e:
        print_warn(f"Failed to export checkpoint metadata: {e}")
        return {"exists": False, "error": str(e)}


def export_config_settings(project_dir: Path) -> Dict[str, Any]:
    """Export pipeline settings and stage configurations from config.yaml."""
    config_path = project_dir / "config.yaml"

    if not config_path.exists():
        # Try parent directory
        config_path = project_root / "config.yaml"

    if not config_path.exists():
        return {"exists": False}

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)

        # Extract relevant pipeline settings
        pipeline_settings = {}

        # Pipeline settings
        if 'pipeline' in config_data:
            pipeline = config_data['pipeline']
            pipeline_settings['pipeline'] = {
                'enabled': pipeline.get('enabled', True),
                'checkpoint': pipeline.get('checkpoint', {}),
                'health_check_interval': pipeline.get('health_check_interval', {}),
                'drift_rules': pipeline.get('drift_rules', {}),
            }

        # Matching settings
        if 'matching' in config_data:
            matching = config_data['matching']
            pipeline_settings['matching'] = {
                'min_confidence': matching.get('min_confidence'),
                'high_confidence_threshold': matching.get('high_confidence_threshold'),
                'max_clip_reuse': matching.get('max_clip_reuse'),
                'chapter_matching_enabled': matching.get('chapter_matching_enabled'),
                'temporal_coherence_enabled': matching.get('temporal_coherence_enabled'),
            }

        # Download settings
        if 'download' in config_data:
            download = config_data['download']
            pipeline_settings['download'] = {
                'enabled': download.get('enabled'),
                'max_workers': download.get('max_workers'),
                'retry_budget': download.get('retry_budget', {}),
                'mullvad': download.get('mullvad', {}),
            }

        # Video search settings
        if 'video_search' in config_data:
            vs = config_data['video_search']
            pipeline_settings['video_search'] = {
                'max_results': vs.get('max_results'),
                'use_chapter_queries': vs.get('use_chapter_queries'),
                'listicle_topic_as_search_terms': vs.get('listicle_topic_as_search_terms'),
            }

        return {
            "exists": True,
            "settings": pipeline_settings,
            "exported_at": datetime.now().isoformat()
        }
    except Exception as e:
        print_warn(f"Failed to export config settings: {e}")
        return {"exists": False, "error": str(e)}


def export_cache_metadata(project_dir: Path) -> Dict[str, Any]:
    """Export cache metadata (sizes, counts, ages) without actual cache data."""
    cache_dir = project_dir / ".cache"
    cache_metadata = {
        "exists": False,
        "total_size_bytes": 0,
        "subdirs": {}
    }

    if not cache_dir.exists():
        return cache_metadata

    cache_metadata["exists"] = True

    def get_dir_size(path: Path) -> int:
        """Calculate total size of directory."""
        total = 0
        try:
            for entry in path.rglob('*'):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except (OSError, IOError):
                        pass
        except (OSError, IOError):
            pass
        return total

    # Get sizes of cache subdirectories
    subdirs = ['transcriptions', 'embeddings', 'llm_responses', 'scene_detection']
    for subdir in subdirs:
        subdir_path = cache_dir / subdir
        if subdir_path.exists():
            size = get_dir_size(subdir_path)
            cache_metadata["subdirs"][subdir] = {
                "size_bytes": size,
                "size_human": format_size(size),
                "file_count": sum(1 for _ in subdir_path.rglob('*') if _.is_file())
            }
            cache_metadata["total_size_bytes"] += size

    cache_metadata["total_size_human"] = format_size(cache_metadata["total_size_bytes"])
    cache_metadata["exported_at"] = datetime.now().isoformat()

    return cache_metadata


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# =============================================================================
# IMPORT FUNCTIONS
# =============================================================================

def import_keywords(project_dir: Path, keywords_data: Dict[str, Any]) -> bool:
    """Import keyword presets into a project."""
    keywords_path = project_dir / "saved_keywords.json"

    existing_presets = {}

    # Load existing presets if file exists
    if keywords_path.exists():
        try:
            with open(keywords_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_presets = existing_data.get('presets', {})
        except Exception as e:
            print_warn(f"Could not read existing keywords: {e}")

    # Merge with imported presets
    imported_presets = keywords_data.get('presets', {})
    merged_presets = {**existing_presets, **imported_presets}

    # Save merged presets
    try:
        data = {
            'version': '1.0',
            'updated_at': datetime.now().isoformat(),
            'presets': merged_presets
        }
        with open(keywords_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        print_ok(f"Imported {len(imported_presets)} keyword preset(s)")
        return True
    except Exception as e:
        print_error(f"Failed to import keywords: {e}")
        return False


def import_config_settings(project_dir: Path, config_data: Dict[str, Any]) -> bool:
    """Import pipeline settings into config.yaml."""
    config_path = project_dir / "config.yaml"

    if not config_path.exists():
        # Try parent directory
        config_path = project_root / "config.yaml"

    if not config_path.exists():
        print_error(f"config.yaml not found in {project_dir}")
        return False

    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            current_config = yaml.safe_load(f)

        # Merge settings (imported settings take precedence for nested values)
        settings = config_data.get('settings', {})

        for section, values in settings.items():
            if section not in current_config:
                current_config[section] = {}
            if isinstance(values, dict):
                current_config[section].update(values)

        # Write back
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(current_config, f, default_flow_style=False, sort_keys=False)

        print_ok(f"Imported pipeline settings")
        return True
    except Exception as e:
        print_error(f"Failed to import config settings: {e}")
        return False


# =============================================================================
# ENCRYPTION FUNCTIONS
# =============================================================================

def encrypt_data(data: str, password: str) -> str:
    """Encrypt data using a simple base64 + hash-based key derivation."""
    # Derive key from password (simplified - use proper crypto in production)
    key = hashlib.sha256(password.encode()).digest()

    # Simple XOR encryption with key
    data_bytes = data.encode('utf-8')
    key_bytes = (key * (len(data_bytes) // len(key) + 1))[:len(data_bytes)]

    encrypted = bytes(a ^ b for a, b in zip(data_bytes, key_bytes))

    # Encode as base64
    return base64.b64encode(encrypted).decode('ascii')


def decrypt_data(encrypted_data: str, password: str) -> Optional[str]:
    """Decrypt data encrypted with encrypt_data."""
    try:
        # Derive same key
        key = hashlib.sha256(password.encode()).digest()

        # Decode from base64
        encrypted = base64.b64decode(encrypted_data.encode('ascii'))

        # XOR decryption
        key_bytes = (key * (len(encrypted) // len(key) + 1))[:len(encrypted)]
        decrypted = bytes(a ^ b for a, b in zip(encrypted, key_bytes))

        return decrypted.decode('utf-8')
    except Exception as e:
        print_error(f"Decryption failed: {e}")
        return None


# =============================================================================
# MAIN EXPORT/IMPORT FUNCTIONS
# =============================================================================

def export_project(project_dir: Path, output_path: Path, encrypt: bool = False,
                   include_cache: bool = False) -> bool:
    """Export project configuration to a file."""
    print_header("EXPORTING PROJECT CONFIG")

    project_dir = Path(project_dir)
    if not project_dir.exists():
        print_error(f"Project directory not found: {project_dir}")
        return False

    print_ok(f"Project: {project_dir}")

    # Collect export data
    export_data = {
        "version": "1.0",
        "exported_at": datetime.now().isoformat(),
        "project_dir": str(project_dir),
    }

    # Export keywords
    print_ok("Exporting keywords...")
    export_data["keywords"] = export_keywords(project_dir)

    # Export checkpoint metadata
    print_ok("Exporting checkpoint metadata...")
    export_data["checkpoint"] = export_checkpoint(project_dir)

    # Export config settings
    print_ok("Exporting config settings...")
    export_data["config"] = export_config_settings(project_dir)

    # Export cache metadata if requested
    if include_cache:
        print_ok("Exporting cache metadata...")
        export_data["cache"] = export_cache_metadata(project_dir)
    else:
        export_data["cache"] = {"included": False}

    # Handle encryption
    if encrypt:
        print_ok("Encrypting export...")
        password = input("Enter encryption password: ")
        confirm = input("Confirm password: ")
        if password != confirm:
            print_error("Passwords do not match")
            return False

        json_data = json.dumps(export_data, indent=2)
        encrypted = encrypt_data(json_data, password)
        output_data = {
            "encrypted": True,
            "data": encrypted,
            "version": "1.0"
        }
    else:
        output_data = export_data

    # Write output
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)

        print_ok(f"Exported to: {output_path}")
        return True
    except Exception as e:
        print_error(f"Failed to write export file: {e}")
        return False


def import_project(project_dir: Path, input_path: Path) -> bool:
    """Import project configuration from a file."""
    print_header("IMPORTING PROJECT CONFIG")

    project_dir = Path(project_dir)
    if not project_dir.exists():
        print_error(f"Project directory not found: {project_dir}")
        return False

    # Read input
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
    except Exception as e:
        print_error(f"Failed to read import file: {e}")
        return False

    # Handle decryption
    if input_data.get("encrypted", False):
        print_ok("Decrypting import...")
        password = input("Enter decryption password: ")
        encrypted = input_data.get("data", "")
        decrypted = decrypt_data(encrypted, password)
        if decrypted is None:
            return False
        try:
            import_data = json.loads(decrypted)
        except json.JSONDecodeError:
            print_error("Invalid decrypted data")
            return False
    else:
        import_data = input_data

    print_ok(f"Importing to: {project_dir}")

    # Import keywords
    if "keywords" in import_data and import_data["keywords"].get("presets"):
        print_ok("Importing keywords...")
        import_keywords(project_dir, import_data["keywords"])

    # Import config settings
    if "config" in import_data and import_data["config"].get("settings"):
        print_ok("Importing config settings...")
        import_config_settings(project_dir, import_data["config"])

    print_ok("Import complete!")
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export and import project configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Export project config:
    python export_config.py --project "E:/Projects/MyProject" --export --output config.json

  Export with encryption:
    python export_config.py --project "E:/Projects/MyProject" --export --encrypt --output config.enc

  Export with cache metadata:
    python export_config.py --project "E:/Projects/MyProject" --export --include-cache --output config.json

  Import project config:
    python export_config.py --project "E:/Projects/MyProject" --import --input config.json
        """
    )

    parser.add_argument('--project', '-p', type=str, required=True,
                        help='Project directory path')
    parser.add_argument('--export', action='store_true',
                        help='Export project configuration')
    parser.add_argument('--import', dest='import_config', action='store_true',
                        help='Import project configuration')
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (for export)')
    parser.add_argument('--input', '-i', type=str,
                        help='Input file path (for import)')
    parser.add_argument('--encrypt', action='store_true',
                        help='Encrypt export with password')
    parser.add_argument('--include-cache', action='store_true',
                        help='Include cache metadata in export')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress non-essential output')

    args = parser.parse_args()

    # Set verbosity
    if args.quiet:
        set_verbosity(0)
    elif args.verbose:
        set_verbosity(2)
    else:
        set_verbosity(1)

    # Validate arguments
    if args.export and not args.output:
        print_error("--output required for export")
        sys.exit(1)

    if args.import_config and not args.input:
        print_error("--input required for import")
        sys.exit(1)

    if not args.export and not args.import_config:
        print_error("Must specify either --export or --import")
        sys.exit(1)

    project_dir = Path(args.project)

    # Execute export or import
    if args.export:
        output_path = Path(args.output)
        success = export_project(
            project_dir=project_dir,
            output_path=output_path,
            encrypt=args.encrypt,
            include_cache=args.include_cache
        )
        sys.exit(0 if success else 1)
    else:
        input_path = Path(args.input)
        success = import_project(
            project_dir=project_dir,
            input_path=input_path
        )
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
