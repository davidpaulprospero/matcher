"""
Cache management CLI commands.

Provides unified cache management across all cache types:
- Global video cache (~/.matcher_global_cache/)
- Entity image cache (~/.matcher_entity_cache/)
- Project-level caches (.cache/)
"""

from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional, Any

if TYPE_CHECKING:
    from ..config import Config

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Statistics for a single cache"""
    name: str
    cache_type: str
    total_entries: int
    total_size_mb: float
    location: str
    oldest_entry: Optional[str] = None
    newest_entry: Optional[str] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClearResult:
    """Result of clearing a cache"""
    cache_type: str
    entries_removed: int
    bytes_freed: int
    dry_run: bool
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        action = "Would remove" if self.dry_run else "Removed"
        size_mb = self.bytes_freed / (1024 * 1024)
        msg = f"{action} {self.entries_removed} entries ({size_mb:.2f} MB) from {self.cache_type}"
        if self.errors:
            msg += f"\nErrors: {len(self.errors)}"
        return msg


@dataclass
class CleanupResult:
    """Result of cleanup operation"""
    expired_removed: int
    orphaned_removed: int
    bytes_freed: int
    dry_run: bool
    details: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        action = "Would remove" if self.dry_run else "Removed"
        size_mb = self.bytes_freed / (1024 * 1024)
        return (
            f"{action} {self.expired_removed} expired + {self.orphaned_removed} orphaned entries "
            f"({size_mb:.2f} MB freed)"
        )


class CacheManager:
    """Unified cache management for all cache types"""

    CACHE_TYPES = ['global', 'entity', 'transcripts', 'embeddings', 'llm', 'vision', 'scenes']

    def __init__(self, config: 'Config', project_dir: Optional[Path] = None):
        self.config = config
        self.project_dir = project_dir or Path.cwd()
        self._cache_paths = self._resolve_cache_paths()

    def _resolve_cache_paths(self) -> Dict[str, Path]:
        """Resolve all cache directory paths"""
        paths = {}

        # Global caches (user home directory)
        global_cache_config = getattr(self.config, 'global_cache', None)
        if global_cache_config:
            cache_dir = getattr(global_cache_config, 'cache_dir', '~/.matcher_global_cache')
            paths['global'] = Path(cache_dir).expanduser()

        entity_cache_config = getattr(
            getattr(self.config, 'image_search', None), 'entity_cache', None
        )
        if entity_cache_config:
            cache_dir = getattr(entity_cache_config, 'cache_dir', '~/.matcher_entity_cache')
            paths['entity'] = Path(cache_dir).expanduser()

        # Project-level caches
        cache_config = getattr(self.config, 'cache', None)
        base_cache_dir = Path(getattr(cache_config, 'cache_dir', '.cache'))
        if not base_cache_dir.is_absolute():
            base_cache_dir = self.project_dir / base_cache_dir

        paths['transcripts'] = base_cache_dir / 'transcriptions'
        paths['embeddings'] = base_cache_dir / 'embeddings'
        paths['llm'] = base_cache_dir / 'llm_responses'
        paths['vision'] = base_cache_dir / 'vision_cache'
        paths['scenes'] = base_cache_dir / 'scene_detection'

        return paths

    def get_all_stats(self) -> Dict[str, CacheStats]:
        """Collect stats from all cache types"""
        stats = {}

        for cache_type, cache_path in self._cache_paths.items():
            if not cache_path.exists():
                stats[cache_type] = CacheStats(
                    name=cache_type.title(),
                    cache_type=cache_type,
                    total_entries=0,
                    total_size_mb=0.0,
                    location=str(cache_path),
                    additional_info={'status': 'not_created'}
                )
                continue

            # Get size and entry count
            total_size = 0
            entry_count = 0
            oldest_time = None
            newest_time = None

            for root, dirs, files in os.walk(cache_path):
                for f in files:
                    file_path = Path(root) / f
                    try:
                        stat = file_path.stat()
                        total_size += stat.st_size
                        entry_count += 1

                        mtime = datetime.fromtimestamp(stat.st_mtime)
                        if oldest_time is None or mtime < oldest_time:
                            oldest_time = mtime
                        if newest_time is None or mtime > newest_time:
                            newest_time = mtime
                    except OSError:
                        pass

            stats[cache_type] = CacheStats(
                name=cache_type.title(),
                cache_type=cache_type,
                total_entries=entry_count,
                total_size_mb=total_size / (1024 * 1024),
                location=str(cache_path),
                oldest_entry=oldest_time.isoformat() if oldest_time else None,
                newest_entry=newest_time.isoformat() if newest_time else None,
            )

            # Add type-specific info
            if cache_type == 'global':
                stats[cache_type].additional_info = self._get_global_cache_info(cache_path)
            elif cache_type == 'entity':
                stats[cache_type].additional_info = self._get_entity_cache_info(cache_path)
            elif cache_type == 'llm':
                stats[cache_type].additional_info = self._get_llm_cache_info(cache_path)

        return stats

    def _get_global_cache_info(self, cache_path: Path) -> Dict[str, Any]:
        """Get additional info for global video cache"""
        info = {}
        registry_dir = cache_path / 'video_registry'
        if registry_dir.exists():
            json_files = list(registry_dir.glob('*.json'))
            info['registered_videos'] = len([f for f in json_files if f.name != 'index.json'])

        # Count topics and keywords from indices
        topic_index = cache_path / 'topics' / 'topic_index.json'
        keyword_index = cache_path / 'keywords' / 'keyword_index.json'

        if topic_index.exists():
            try:
                import json
                with open(topic_index) as f:
                    topics = json.load(f)
                info['indexed_topics'] = len(topics)
            except Exception:
                pass

        if keyword_index.exists():
            try:
                import json
                with open(keyword_index) as f:
                    keywords = json.load(f)
                info['indexed_keywords'] = len(keywords)
            except Exception:
                pass

        return info

    def _get_entity_cache_info(self, cache_path: Path) -> Dict[str, Any]:
        """Get additional info for entity image cache"""
        info = {}
        images_dir = cache_path / 'images'
        if images_dir.exists():
            entity_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
            info['cached_entities'] = len(entity_dirs)
            total_images = sum(len(list(d.glob('*'))) for d in entity_dirs)
            info['total_images'] = total_images
        return info

    def _get_llm_cache_info(self, cache_path: Path) -> Dict[str, Any]:
        """Get additional info for LLM response cache"""
        info = {}
        # Count by provider
        for provider_dir in cache_path.iterdir():
            if provider_dir.is_dir():
                count = len(list(provider_dir.glob('*.json')))
                info[f'{provider_dir.name}_responses'] = count
        return info

    def clear_cache(self, cache_type: str, dry_run: bool = False) -> ClearResult:
        """Clear specified cache type"""
        if cache_type == 'all':
            # Clear all caches
            total_entries = 0
            total_bytes = 0
            errors = []

            for ct in self.CACHE_TYPES:
                result = self._clear_single_cache(ct, dry_run)
                total_entries += result.entries_removed
                total_bytes += result.bytes_freed
                errors.extend(result.errors)

            return ClearResult(
                cache_type='all',
                entries_removed=total_entries,
                bytes_freed=total_bytes,
                dry_run=dry_run,
                errors=errors
            )

        return self._clear_single_cache(cache_type, dry_run)

    def _clear_single_cache(self, cache_type: str, dry_run: bool) -> ClearResult:
        """Clear a single cache type"""
        if cache_type not in self._cache_paths:
            return ClearResult(
                cache_type=cache_type,
                entries_removed=0,
                bytes_freed=0,
                dry_run=dry_run,
                errors=[f"Unknown cache type: {cache_type}"]
            )

        cache_path = self._cache_paths[cache_type]
        if not cache_path.exists():
            return ClearResult(
                cache_type=cache_type,
                entries_removed=0,
                bytes_freed=0,
                dry_run=dry_run
            )

        # Calculate size before clearing
        total_size = 0
        entry_count = 0
        for root, dirs, files in os.walk(cache_path):
            for f in files:
                try:
                    total_size += (Path(root) / f).stat().st_size
                    entry_count += 1
                except OSError:
                    pass

        errors = []
        if not dry_run:
            try:
                shutil.rmtree(cache_path)
                cache_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(str(e))

        return ClearResult(
            cache_type=cache_type,
            entries_removed=entry_count,
            bytes_freed=total_size,
            dry_run=dry_run,
            errors=errors
        )

    def cleanup_all(self, dry_run: bool = False) -> CleanupResult:
        """Run cleanup on all caches (remove expired/orphaned entries)"""
        expired_removed = 0
        orphaned_removed = 0
        bytes_freed = 0
        details = {}

        # Cleanup global cache (orphaned entries for deleted videos)
        global_result = self._cleanup_global_cache(dry_run)
        orphaned_removed += global_result.get('orphaned', 0)
        bytes_freed += global_result.get('bytes_freed', 0)
        details['global'] = global_result.get('orphaned', 0)

        # Cleanup entity cache (expired entries)
        entity_result = self._cleanup_entity_cache(dry_run)
        expired_removed += entity_result.get('expired', 0)
        orphaned_removed += entity_result.get('orphaned', 0)
        bytes_freed += entity_result.get('bytes_freed', 0)
        details['entity'] = entity_result.get('expired', 0) + entity_result.get('orphaned', 0)

        # Cleanup LLM cache (expired entries)
        llm_result = self._cleanup_llm_cache(dry_run)
        expired_removed += llm_result.get('expired', 0)
        bytes_freed += llm_result.get('bytes_freed', 0)
        details['llm'] = llm_result.get('expired', 0)

        return CleanupResult(
            expired_removed=expired_removed,
            orphaned_removed=orphaned_removed,
            bytes_freed=bytes_freed,
            dry_run=dry_run,
            details=details
        )

    def _cleanup_global_cache(self, dry_run: bool) -> Dict[str, int]:
        """Cleanup orphaned entries in global cache"""
        cache_path = self._cache_paths.get('global')
        if not cache_path or not cache_path.exists():
            return {'orphaned': 0, 'bytes_freed': 0}

        try:
            from ..global_cache import GlobalCacheManager

            global_cache_config = getattr(self.config, 'global_cache', None)
            if not global_cache_config:
                return {'orphaned': 0, 'bytes_freed': 0}

            cache = GlobalCacheManager(global_cache_config)
            if hasattr(cache, 'cleanup_orphaned_entries'):
                if dry_run:
                    # Just count orphaned without removing
                    count = cache.count_orphaned_entries() if hasattr(cache, 'count_orphaned_entries') else 0
                    return {'orphaned': count, 'bytes_freed': 0}
                else:
                    removed = cache.cleanup_orphaned_entries()
                    return {'orphaned': removed, 'bytes_freed': 0}
        except Exception as e:
            logger.warning(f"Global cache cleanup failed: {e}")

        return {'orphaned': 0, 'bytes_freed': 0}

    def _cleanup_entity_cache(self, dry_run: bool) -> Dict[str, int]:
        """Cleanup expired/orphaned entries in entity cache"""
        cache_path = self._cache_paths.get('entity')
        if not cache_path or not cache_path.exists():
            return {'expired': 0, 'orphaned': 0, 'bytes_freed': 0}

        try:
            from ..entity_cache import EntityCache

            entity_cache_config = getattr(
                getattr(self.config, 'image_search', None), 'entity_cache', None
            )
            if not entity_cache_config:
                return {'expired': 0, 'orphaned': 0, 'bytes_freed': 0}

            cache = EntityCache(entity_cache_config)
            if hasattr(cache, 'cleanup'):
                if dry_run:
                    # Try to count without removing
                    return {'expired': 0, 'orphaned': 0, 'bytes_freed': 0}
                else:
                    removed = cache.cleanup()
                    return {'expired': 0, 'orphaned': removed, 'bytes_freed': 0}
        except Exception as e:
            logger.warning(f"Entity cache cleanup failed: {e}")

        return {'expired': 0, 'orphaned': 0, 'bytes_freed': 0}

    def _cleanup_llm_cache(self, dry_run: bool) -> Dict[str, int]:
        """Cleanup expired entries in LLM cache"""
        cache_path = self._cache_paths.get('llm')
        if not cache_path or not cache_path.exists():
            return {'expired': 0, 'bytes_freed': 0}

        # Get TTL from config
        llm_config = getattr(self.config, 'llm', None)
        cache_config = getattr(llm_config, 'cache', None) if llm_config else None
        ttl_hours = getattr(cache_config, 'ttl_hours', 24) if cache_config else 24

        if ttl_hours <= 0:
            return {'expired': 0, 'bytes_freed': 0}

        expired_count = 0
        bytes_freed = 0
        cutoff_time = datetime.now().timestamp() - (ttl_hours * 3600)

        for provider_dir in cache_path.iterdir():
            if not provider_dir.is_dir():
                continue

            for cache_file in provider_dir.glob('*.json'):
                try:
                    stat = cache_file.stat()
                    if stat.st_mtime < cutoff_time:
                        bytes_freed += stat.st_size
                        expired_count += 1
                        if not dry_run:
                            cache_file.unlink()
                except OSError:
                    pass

        return {'expired': expired_count, 'bytes_freed': bytes_freed}

    def list_entries(self, cache_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List cache entries with metadata"""
        if cache_type == 'global':
            return self._list_global_entries(limit)
        elif cache_type == 'entity':
            return self._list_entity_entries(limit)
        else:
            return []

    def _list_global_entries(self, limit: int) -> List[Dict[str, Any]]:
        """List entries in global video cache"""
        cache_path = self._cache_paths.get('global')
        if not cache_path or not cache_path.exists():
            return []

        entries = []
        registry_dir = cache_path / 'video_registry'
        if not registry_dir.exists():
            return []

        import json
        for entry_file in list(registry_dir.glob('*.json'))[:limit]:
            if entry_file.name == 'index.json':
                continue
            try:
                with open(entry_file) as f:
                    data = json.load(f)
                entries.append({
                    'filename': data.get('filename', 'unknown'),
                    'duration': data.get('duration', 0),
                    'topics': data.get('topics', [])[:3],
                    'keywords': data.get('keywords', [])[:3],
                    'file_exists': data.get('file_exists', False),
                    'usage_count': data.get('usage_count', 0),
                    'last_used': data.get('last_used', ''),
                    'current_path': data.get('current_path', ''),
                })
            except Exception:
                pass

        return entries

    def _list_entity_entries(self, limit: int) -> List[Dict[str, Any]]:
        """List entries in entity image cache"""
        cache_path = self._cache_paths.get('entity')
        if not cache_path or not cache_path.exists():
            return []

        entries = []
        index_file = cache_path / 'entity_cache_index.json'
        if not index_file.exists():
            return []

        import json
        try:
            with open(index_file) as f:
                index_data = json.load(f)

            for entity_name, entry in list(index_data.items())[:limit]:
                data = entry.get('data', entry)  # Handle both old and new formats
                entries.append({
                    'entity_name': data.get('entity_name', entity_name),
                    'entity_type': data.get('entity_type', 'unknown'),
                    'image_count': len(data.get('images', [])),
                    'source_project': data.get('source_project', ''),
                    'cached_at': data.get('cached_at', ''),
                })
        except Exception:
            pass

        return entries


def display_cache_stats(stats: Dict[str, CacheStats]) -> None:
    """Format and print cache statistics"""
    print("\n" + "=" * 70)
    print("  CACHE STATISTICS")
    print("=" * 70)

    total_size = 0
    total_entries = 0

    for cache_type, stat in stats.items():
        total_size += stat.total_size_mb
        total_entries += stat.total_entries

        status = ""
        if stat.additional_info.get('status') == 'not_created':
            status = " (not created)"

        print(f"\n  {stat.name}{status}")
        print(f"  " + "-" * 40)
        print(f"    Location: {stat.location}")
        print(f"    Entries:  {stat.total_entries:,}")
        print(f"    Size:     {stat.total_size_mb:.2f} MB")

        if stat.oldest_entry and stat.newest_entry:
            print(f"    Oldest:   {stat.oldest_entry[:10]}")
            print(f"    Newest:   {stat.newest_entry[:10]}")

        # Type-specific info
        for key, value in stat.additional_info.items():
            if key != 'status':
                print(f"    {key.replace('_', ' ').title()}: {value}")

    print(f"\n  " + "=" * 40)
    print(f"  TOTAL: {total_entries:,} entries, {total_size:.2f} MB")
    print()


def display_cache_list(entries: List[Dict[str, Any]], cache_type: str) -> None:
    """Format and print cache entry list"""
    print(f"\n  {cache_type.upper()} CACHE ENTRIES ({len(entries)} shown)")
    print("  " + "-" * 60)

    if not entries:
        print("  No entries found")
        return

    if cache_type == 'global':
        for i, entry in enumerate(entries, 1):
            status = "✓" if entry.get('file_exists') else "✗"
            print(f"\n  {i}. [{status}] {entry.get('filename', 'unknown')}")
            print(f"     Duration: {entry.get('duration', 0):.0f}s | Used: {entry.get('usage_count', 0)}x")
            topics = ", ".join(entry.get('topics', []))
            if topics:
                print(f"     Topics: {topics}")
            keywords = ", ".join(entry.get('keywords', []))
            if keywords:
                print(f"     Keywords: {keywords}")
            if entry.get('current_path'):
                print(f"     Path: {entry.get('current_path')}")

    elif cache_type == 'entity':
        for i, entry in enumerate(entries, 1):
            print(f"\n  {i}. {entry.get('entity_name', 'unknown')} ({entry.get('entity_type', '')})")
            print(f"     Images: {entry.get('image_count', 0)}")
            if entry.get('source_project'):
                print(f"     Source: {entry.get('source_project')}")
            if entry.get('cached_at'):
                print(f"     Cached: {entry.get('cached_at')[:10]}")

    print()
