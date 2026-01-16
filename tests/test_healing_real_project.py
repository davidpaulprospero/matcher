"""
Real-world healing simulations against an actual project directory.

This test suite runs healing simulations against real project data
at E:/Edit Job/test/jan_test/audio__2026-01-01

Usage:
    pytest tests/test_healing_real_project.py -v
    pytest tests/test_healing_real_project.py -v -k "checkpoint"
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Project path
PROJECT_DIR = Path("E:/Edit Job/test/jan_test/audio__2026-01-01")


def project_exists():
    """Check if the test project exists."""
    return PROJECT_DIR.exists() and (PROJECT_DIR / "checkpoint.json").exists()


# Skip all tests if project doesn't exist
pytestmark = pytest.mark.skipif(
    not project_exists(),
    reason=f"Test project not found at {PROJECT_DIR}"
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def project_dir():
    """Return the test project directory."""
    return PROJECT_DIR


@pytest.fixture
def real_config(project_dir):
    """Load real configuration from the project."""
    from src.config import Config, load_config

    # Load default config first
    config = load_config()

    # Try to load project config overrides
    project_config_path = project_dir / "project_config.yaml"
    if project_config_path.exists():
        import yaml
        with open(project_config_path, 'r') as f:
            overrides = yaml.safe_load(f) or {}

        # Apply overrides (simplified)
        for section, values in overrides.items():
            if hasattr(config, section) and isinstance(values, dict):
                for key, value in values.items():
                    if hasattr(getattr(config, section), key):
                        setattr(getattr(config, section), key, value)

    return config


@pytest.fixture
def real_checkpoint(project_dir):
    """Load real checkpoint data from the project."""
    checkpoint_path = project_dir / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@pytest.fixture
def backup_checkpoint(project_dir):
    """Load backup checkpoint if available."""
    backup_path = project_dir / "checkpoint.backup.json"
    if backup_path.exists():
        with open(backup_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


@pytest.fixture
def real_state(project_dir, real_config, real_checkpoint):
    """Create a pipeline state from real project data."""
    from dataclasses import dataclass, field
    from typing import List, Optional

    @dataclass
    class RealSegment:
        index: int
        start: float
        end: float
        text: str
        duration: float

    @dataclass
    class RealMatch:
        video_path: str
        start_time: float
        end_time: float
        score: float
        segment_index: int
        segment: Optional[RealSegment] = None

    # Create state object
    state = Mock()
    state.config = real_config
    state.project_dir = project_dir

    # Load segments from checkpoint
    segments = []
    if real_checkpoint and 'analyze' in real_checkpoint:
        for seg_data in real_checkpoint['analyze'].get('segments', []):
            segments.append(RealSegment(
                index=seg_data.get('index', 0),
                start=seg_data.get('start', 0),
                end=seg_data.get('end', 0),
                text=seg_data.get('text', ''),
                duration=seg_data.get('duration', 0)
            ))

    state.voiceover_segments = segments

    # Create matches (empty for now, will be populated from checkpoint if available)
    state.matches = []

    # Load downloaded videos info from cache
    state.downloaded_videos = []
    transcription_cache = project_dir / ".cache" / "transcriptions"
    if transcription_cache.exists():
        for json_file in transcription_cache.glob("*.json"):
            video = Mock()
            video.file = str(json_file.with_suffix('.mp4'))
            video.video_id = json_file.stem
            video.duration = 120.0  # Default
            state.downloaded_videos.append(video)

    return state


# ==============================================================================
# CHECKPOINT HEALER SIMULATIONS
# ==============================================================================


class TestCheckpointHealerReal:
    """Test checkpoint healing against real project."""

    def test_checkpoint_integrity(self, project_dir, real_checkpoint):
        """Verify checkpoint file integrity."""
        assert real_checkpoint is not None, "Checkpoint should exist"
        assert 'version' in real_checkpoint
        assert 'created_at' in real_checkpoint
        assert 'last_completed_stage' in real_checkpoint

        print(f"\nCheckpoint Info:")
        print(f"  Version: {real_checkpoint.get('version')}")
        print(f"  Last stage: {real_checkpoint.get('last_completed_stage')}")
        print(f"  Created: {real_checkpoint.get('created_at')}")
        print(f"  Updated: {real_checkpoint.get('updated_at')}")

    def test_backup_restoration_simulation(self, project_dir, real_config, backup_checkpoint):
        """Simulate checkpoint corruption and backup restoration."""
        from src.agents.healers.checkpoint import CheckpointHealer

        healer = CheckpointHealer(real_config, project_dir)

        # Create temporary corrupted checkpoint (don't modify real one)
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Copy real checkpoint to temp
            shutil.copy(project_dir / "checkpoint.json", temp_dir / "checkpoint.json")
            if (project_dir / "checkpoint.backup.json").exists():
                shutil.copy(project_dir / "checkpoint.backup.json", temp_dir / "checkpoint.backup.json")

            # Create a healer for temp dir
            temp_healer = CheckpointHealer(real_config, temp_dir)

            # Corrupt the temp checkpoint
            (temp_dir / "checkpoint.json").write_text("{{corrupted}}", encoding='utf-8')

            # Create mock state
            state = Mock()
            state.config = real_config
            state.project_dir = temp_dir

            # Simulate JSONDecodeError
            error = json.JSONDecodeError("Simulated corruption", "", 0)

            # Try to heal
            result = temp_healer.fix(error, state, "RESUME")

            print(f"\nBackup Restoration:")
            print(f"  Success: {result.success}")
            print(f"  Action: {result.action.value}")
            print(f"  Message: {result.message}")

            if result.success:
                # Verify checkpoint was restored
                with open(temp_dir / "checkpoint.json", 'r') as f:
                    restored = json.load(f)
                    assert 'version' in restored

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_rebuild_detection(self, project_dir, real_config):
        """Test that healer can detect cached data for rebuild."""
        from src.agents.healers.checkpoint import CheckpointHealer

        healer = CheckpointHealer(real_config, project_dir)

        # Check what caches exist
        cache_dir = project_dir / ".cache"
        available_caches = []

        if (cache_dir / "transcriptions").exists():
            trans_count = len(list((cache_dir / "transcriptions").glob("*.json")))
            if trans_count > 0:
                available_caches.append(f"transcriptions ({trans_count} files)")

        if (cache_dir / "embeddings").exists():
            embed_count = len(list((cache_dir / "embeddings").glob("*.npy")))
            if embed_count > 0:
                available_caches.append(f"embeddings ({embed_count} files)")

        if (cache_dir / "scene_detection").exists():
            scene_count = len(list((cache_dir / "scene_detection").glob("*.json")))
            if scene_count > 0:
                available_caches.append(f"scene_detection ({scene_count} files)")

        if (cache_dir / "llm_responses").exists():
            llm_count = len(list((cache_dir / "llm_responses").glob("*.json")))
            if llm_count > 0:
                available_caches.append(f"llm_responses ({llm_count} files)")

        print(f"\nAvailable Caches for Rebuild:")
        for cache in available_caches:
            print(f"  - {cache}")

        assert len(available_caches) > 0, "Should have some cached data"


# ==============================================================================
# OTIO HEALER SIMULATIONS
# ==============================================================================


class TestOTIOHealerReal:
    """Test OTIO healing against real project data."""

    def test_segment_duration_validation(self, real_state, real_config, project_dir):
        """Validate segment durations from real checkpoint."""
        from src.agents.healers.otio import OTIOHealer

        healer = OTIOHealer(real_config, project_dir)

        issues_found = []
        for segment in real_state.voiceover_segments:
            if segment.duration <= 0:
                issues_found.append(f"Segment {segment.index}: zero/negative duration")
            if segment.duration > healer.MAX_CLIP_DURATION:
                issues_found.append(f"Segment {segment.index}: exceeds max duration")
            if segment.end <= segment.start:
                issues_found.append(f"Segment {segment.index}: end <= start")

        print(f"\nSegment Duration Validation:")
        print(f"  Total segments: {len(real_state.voiceover_segments)}")
        print(f"  Issues found: {len(issues_found)}")
        for issue in issues_found[:5]:  # Show first 5
            print(f"    - {issue}")

        # This is informational - real data might have issues
        # The point is to verify the healer can handle them

    def test_gap_mode_detection(self, real_state, real_config, project_dir):
        """Test gap mode handling with real segments."""
        from src.agents.healers.otio import OTIOHealer

        healer = OTIOHealer(real_config, project_dir)

        # Calculate total voiceover duration
        if real_state.voiceover_segments:
            first_start = min(s.start for s in real_state.voiceover_segments)
            last_end = max(s.end for s in real_state.voiceover_segments)
            total_duration = last_end - first_start

            # Calculate gaps
            segments = sorted(real_state.voiceover_segments, key=lambda s: s.start)
            total_gap = 0
            for i in range(len(segments) - 1):
                gap = segments[i + 1].start - segments[i].end
                if gap > 0:
                    total_gap += gap

            gap_percentage = (total_gap / total_duration * 100) if total_duration > 0 else 0

            print(f"\nGap Analysis:")
            print(f"  Total duration: {total_duration:.2f}s")
            print(f"  Total gap time: {total_gap:.2f}s")
            print(f"  Gap percentage: {gap_percentage:.1f}%")
            print(f"  Current gap_mode: {real_config.output.gap_mode}")

            # Simulate gap overflow error
            if gap_percentage > 50:
                error = ValueError(f"Gap overflow: {gap_percentage:.1f}% gaps")
                result = healer.fix(error, real_state, "OUTPUT")
                print(f"  Gap overflow fix: {result.message}")

    def test_media_reference_check(self, project_dir, real_config, real_state):
        """Check media references in real project."""
        from src.agents.healers.otio import OTIOHealer

        healer = OTIOHealer(real_config, project_dir)

        # Check for video files
        video_dirs = [
            project_dir / "downloaded_videos",
            project_dir / ".cache" / "videos",
            Path("E:/v"),  # Short path root
        ]

        existing_videos = []
        missing_videos = []

        for video in real_state.downloaded_videos:
            video_path = Path(video.file)
            found = False

            # Check if file exists directly or in known locations
            if video_path.exists():
                found = True
                existing_videos.append(str(video_path))
            else:
                # Search in video directories
                for vdir in video_dirs:
                    if vdir.exists():
                        matches = list(vdir.glob(f"*{video.video_id}*"))
                        if matches:
                            found = True
                            existing_videos.append(str(matches[0]))
                            break

            if not found:
                missing_videos.append(video.video_id)

        print(f"\nMedia Reference Check:")
        print(f"  Total videos: {len(real_state.downloaded_videos)}")
        print(f"  Found: {len(existing_videos)}")
        print(f"  Missing: {len(missing_videos)}")
        if missing_videos[:3]:
            print(f"  Missing IDs: {missing_videos[:3]}...")


# ==============================================================================
# API HEALER SIMULATIONS
# ==============================================================================


class TestAPIHealerReal:
    """Test API healing with real configuration."""

    def test_api_key_detection(self, real_config, project_dir):
        """Check which API keys are available."""
        from src.agents.healers.api import APIHealer

        healer = APIHealer(real_config, project_dir)

        # Check environment variables
        api_keys = {
            'GEMINI_API_KEY': os.environ.get('GEMINI_API_KEY'),
            'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY'),
            'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY'),
            'PEXELS_API_KEY': os.environ.get('PEXELS_API_KEY'),
            'PIXABAY_API_KEY': os.environ.get('PIXABAY_API_KEY'),
        }

        print(f"\nAPI Key Status:")
        for key, value in api_keys.items():
            status = "SET" if value else "NOT SET"
            print(f"  {key}: {status}")

        # Test provider fallback
        current_provider = getattr(real_config.llm, 'provider', 'gemini')
        fallback_providers = healer.PROVIDER_FALLBACK.get(current_provider, [])
        print(f"\nProvider Fallback Chain:")
        print(f"  Current: {current_provider}")
        print(f"  Fallbacks: {fallback_providers}")

    def test_rate_limit_backoff_simulation(self, real_config, project_dir, real_state):
        """Simulate rate limiting and verify backoff."""
        from src.agents.healers.api import APIHealer

        healer = APIHealer(real_config, project_dir)

        # Record backoff times
        backoff_times = []

        with patch('time.sleep') as mock_sleep:
            def record_backoff(duration):
                backoff_times.append(duration)
            mock_sleep.side_effect = record_backoff

            # Simulate 3 rate limits
            for i in range(3):
                error = Exception("Error 429: Rate limit exceeded")
                result = healer.fix(error, real_state, "ANALYZE")
                assert result.success

        print(f"\nRate Limit Backoff Simulation:")
        print(f"  Attempts: {len(backoff_times)}")
        for i, bt in enumerate(backoff_times):
            print(f"  Attempt {i + 1}: {bt:.1f}s wait")

        # Verify exponential backoff
        if len(backoff_times) >= 2:
            assert backoff_times[1] >= backoff_times[0], "Backoff should increase"


# ==============================================================================
# DISK HEALER SIMULATIONS
# ==============================================================================


class TestDiskHealerReal:
    """Test disk healing with real project data."""

    def test_disk_space_check(self, project_dir):
        """Check available disk space for project."""
        import shutil

        total, used, free = shutil.disk_usage(project_dir)

        print(f"\nDisk Space ({project_dir.drive or project_dir.anchor}):")
        print(f"  Total: {total / (1024**3):.1f} GB")
        print(f"  Used: {used / (1024**3):.1f} GB")
        print(f"  Free: {free / (1024**3):.1f} GB")
        print(f"  Usage: {used / total * 100:.1f}%")

        # Check if we'd trigger disk warnings
        free_gb = free / (1024**3)
        if free_gb < 1.0:
            print(f"  WARNING: Less than 1 GB free - would trigger critical alert")
        elif free_gb < 5.0:
            print(f"  WARNING: Less than 5 GB free - would trigger warning")
        else:
            print(f"  OK: Sufficient disk space")

    def test_cache_cleanup_potential(self, project_dir, real_config):
        """Calculate how much space cache cleanup would recover."""
        cache_dir = project_dir / ".cache"

        cache_sizes = {}
        total_recoverable = 0

        cleanable_caches = ['llm_responses', 'vision_cache']

        for cache_name in cleanable_caches:
            cache_path = cache_dir / cache_name
            if cache_path.exists():
                size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
                cache_sizes[cache_name] = size
                total_recoverable += size

        print(f"\nCache Cleanup Potential:")
        for name, size in cache_sizes.items():
            print(f"  {name}: {size / (1024**2):.1f} MB")
        print(f"  Total recoverable: {total_recoverable / (1024**2):.1f} MB")


# ==============================================================================
# PATH HEALER SIMULATIONS
# ==============================================================================


class TestPathHealerReal:
    """Test path healing with real project paths."""

    def test_path_length_analysis(self, project_dir):
        """Analyze path lengths in real project."""
        from src.agents.healers.path import PathHealer

        max_path_length = 0
        long_paths = []
        WINDOWS_MAX = 260

        for item in project_dir.rglob('*'):
            path_len = len(str(item))
            if path_len > max_path_length:
                max_path_length = path_len

            if path_len > WINDOWS_MAX - 50:  # Within 50 chars of limit
                long_paths.append((path_len, str(item)[-80:]))

        print(f"\nPath Length Analysis:")
        print(f"  Project root: {len(str(project_dir))} chars")
        print(f"  Longest path: {max_path_length} chars")
        print(f"  Windows limit: {WINDOWS_MAX} chars")
        print(f"  Margin: {WINDOWS_MAX - max_path_length} chars")

        if long_paths:
            print(f"  Long paths ({len(long_paths)}):")
            for length, path in sorted(long_paths, reverse=True)[:3]:
                print(f"    {length} chars: ...{path}")

    def test_unicode_filename_check(self, project_dir):
        """Check for unicode characters in filenames."""
        unicode_files = []
        illegal_chars = []

        ILLEGAL_CHARS = '<>:"|?*'

        for item in project_dir.rglob('*'):
            name = item.name
            # Check for non-ASCII
            if not name.isascii():
                unicode_files.append(name)
            # Check for illegal chars
            if any(c in name for c in ILLEGAL_CHARS):
                illegal_chars.append(name)

        print(f"\nFilename Character Analysis:")
        print(f"  Unicode filenames: {len(unicode_files)}")
        print(f"  Illegal character files: {len(illegal_chars)}")

        if unicode_files[:3]:
            # Safe print with ASCII encoding for console
            safe_examples = [f.encode('ascii', 'replace').decode('ascii') for f in unicode_files[:3]]
            print(f"  Unicode examples: {safe_examples}")
        if illegal_chars[:3]:
            safe_examples = [f.encode('ascii', 'replace').decode('ascii') for f in illegal_chars[:3]]
            print(f"  Illegal char examples: {safe_examples}")


# ==============================================================================
# ORCHESTRATOR SIMULATIONS
# ==============================================================================


class TestOrchestratorReal:
    """Test orchestrator with real project."""

    def test_preflight_check(self, project_dir, real_config, real_state):
        """Run preflight checks against real project."""
        from src.agents.orchestrator import HealingOrchestrator
        from src.agents.strategy import HealingStrategy

        strategy = HealingStrategy.conservative()

        # Create orchestrator with watcher/llm disabled for testing
        real_config.healing = Mock()
        real_config.healing.enabled = True
        real_config.healing.strategy = "conservative"
        real_config.healing.max_attempts_per_stage = 3
        real_config.healing.max_total_heals = 20
        real_config.healing.heal_delay = 0.1

        real_config.healing.watcher = Mock()
        real_config.healing.watcher.enabled = False
        real_config.healing.watcher.host = "http://localhost:11434"
        real_config.healing.watcher.model = "llama3.2"
        real_config.healing.watcher.fallback_model = "llama3.1"
        real_config.healing.watcher.provider = "ollama"
        real_config.healing.watcher.recheck_interval_seconds = 300.0
        real_config.healing.watcher.max_failures = 3

        real_config.healing.llm_healer = Mock()
        real_config.healing.llm_healer.enabled = False
        real_config.healing.llm_healer.provider = "anthropic"
        real_config.healing.llm_healer.timeout = 60.0
        real_config.healing.llm_healer.max_failures = 3
        real_config.healing.llm_healer.recheck_interval_seconds = 300.0

        real_config.healing.logging = Mock()
        real_config.healing.logging.enabled = False
        real_config.healing.logging.log_dir = "logs"
        real_config.healing.logging.json_log = True

        orchestrator = HealingOrchestrator(real_config, project_dir, strategy)

        # Create proper state for preflight
        real_state.matches = []
        real_state.downloaded_videos = []

        issues = orchestrator.run_preflight(real_state)

        print(f"\nPreflight Check Results:")
        print(f"  Total issues: {len(issues)}")

        by_severity = {}
        for issue in issues:
            sev = issue.severity
            by_severity[sev] = by_severity.get(sev, 0) + 1

        for severity, count in sorted(by_severity.items()):
            print(f"  {severity}: {count}")

        # Show first few issues
        print(f"\nTop Issues:")
        for issue in issues[:5]:
            auto_fix = " (auto-fixable)" if issue.auto_fixable else ""
            print(f"  [{issue.severity}] {issue.message}{auto_fix}")

    def test_healer_registry(self, project_dir, real_config):
        """Check which healers are available."""
        from src.agents.healers import HEALER_REGISTRY

        print(f"\nRegistered Healers ({len(HEALER_REGISTRY)}):")
        for healer_cls in HEALER_REGISTRY:
            healer = healer_cls(real_config, project_dir)
            print(f"  - {healer.name}: {healer.description}")
            print(f"    Patterns: {len(healer.error_patterns)} | Exceptions: {len(healer.exception_types)}")


# ==============================================================================
# END-TO-END SIMULATION
# ==============================================================================


class TestEndToEndSimulation:
    """End-to-end healing simulation with real project."""

    def test_simulated_pipeline_run(self, project_dir, real_config, real_state):
        """Simulate a pipeline run with various errors."""
        from src.agents.healers.api import APIHealer
        from src.agents.healers.otio import OTIOHealer
        from src.agents.healers.checkpoint import CheckpointHealer
        from src.agents.healers.download import DownloadHealer

        # Disable watcher for tests
        real_config.healing = Mock()
        real_config.healing.watcher = Mock()
        real_config.healing.watcher.enabled = False
        real_config.healing.llm_healer = Mock()
        real_config.healing.llm_healer.enabled = False

        healers = [
            APIHealer(real_config, project_dir),
            OTIOHealer(real_config, project_dir),
            CheckpointHealer(real_config, project_dir),
            DownloadHealer(real_config, project_dir),
        ]

        # Simulate errors that might occur during pipeline
        simulated_errors = [
            ("ANALYZE", Exception("Error 429: Rate limit exceeded")),
            ("DOWNLOAD", Exception("Video unavailable")),
            ("TRANSCRIBE", TimeoutError("Timeout after 120s")),
            ("OUTPUT", ValueError("Duration must be positive")),
        ]

        print(f"\n{'='*60}")
        print("SIMULATED PIPELINE RUN")
        print(f"{'='*60}")

        results = []
        for stage, error in simulated_errors:
            print(f"\n[{stage}] Simulated error: {type(error).__name__}")
            print(f"  Message: {str(error)[:60]}")

            # Find healer that can handle this error
            handled = False
            for healer in healers:
                if healer.can_handle(error, stage):
                    print(f"  Handler: {healer.name}")

                    with patch('time.sleep'):
                        result = healer.fix(error, real_state, stage)

                    print(f"  Success: {result.success}")
                    print(f"  Action: {result.action.value}")
                    print(f"  Message: {result.message[:60]}...")

                    results.append({
                        'stage': stage,
                        'error': type(error).__name__,
                        'healer': healer.name,
                        'success': result.success,
                        'action': result.action.value
                    })
                    handled = True
                    break

            if not handled:
                print(f"  Handler: NONE - error would abort pipeline")
                results.append({
                    'stage': stage,
                    'error': type(error).__name__,
                    'healer': None,
                    'success': False,
                    'action': 'abort'
                })

        print(f"\n{'='*60}")
        print("SIMULATION SUMMARY")
        print(f"{'='*60}")
        successes = sum(1 for r in results if r['success'])
        print(f"  Errors handled: {successes}/{len(results)}")
        for r in results:
            status = "HEALED" if r['success'] else "FAILED"
            healer = r['healer'] or "none"
            print(f"  [{status}] {r['stage']}: {r['error']} -> {healer}")


# ==============================================================================
# MAIN
# ==============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
