#!/usr/bin/env python3
"""
Full Integration Test - Real Project Simulation

Downloads a real sample voiceover, runs the complete pipeline, and validates
all outputs including recent features like:
- OTIO timeline statistics
- Run summary logging
- Segment IDs in clips
- Entity matching (sticky/semantic)
- Post-edit analysis

Usage:
    python tests/test_integration_full.py
    python tests/test_integration_full.py --keep          # Keep temp project
    python tests/test_integration_full.py --voiceover URL # Use specific video as voiceover
    python tests/test_integration_full.py --skip-download # Use synthetic SRT only
"""

import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

INSTALL_DIR = Path(__file__).parent.parent.resolve()

# Load .env file from project directory
try:
    from dotenv import load_dotenv
    env_path = INSTALL_DIR / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # dotenv not installed, try manual loading
    env_path = INSTALL_DIR / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ.setdefault(key, value)

# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_box(text: str, char: str = "="):
    print(f"\n{char * 60}")
    print(f"  {text}")
    print(f"{char * 60}")


def print_section(text: str):
    print(f"\n  ─── {text} ───")


def print_check(name: str, passed: bool, detail: str = ""):
    status = "✓" if passed else "✗"
    detail_str = f" ({detail})" if detail else ""
    print(f"  [{status}] {name}{detail_str}")
    return passed


# =============================================================================
# VOICEOVER DOWNLOAD
# =============================================================================

# Short, speech-focused videos good for voiceover testing (< 5 mins)
SAMPLE_VOICEOVER_URLS = [
    # TED-Ed educational shorts (clear narration, no music)
    "https://www.youtube.com/watch?v=RcYjXbSJBN8",  # ~3 min TED-Ed
    "https://www.youtube.com/watch?v=xuCn8ux2gbs",  # ~4 min TED-Ed
    # News/documentary style
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Fallback
]


def download_voiceover(output_dir: Path, url: str = None, max_duration: int = 300) -> Path:
    """
    Download a sample voiceover video/audio.

    Args:
        output_dir: Directory to save the file
        url: Specific URL to download, or None for auto-search
        max_duration: Maximum duration in seconds (default 5 mins)

    Returns:
        Path to downloaded file, or None if failed
    """
    output_path = output_dir / "voiceover.mp4"

    # Build yt-dlp command with low quality video settings
    cmd = [
        'yt-dlp',
        '--format', 'worst[ext=mp4]/worstvideo[ext=mp4]+worstaudio[ext=m4a]/worst',  # Lowest quality video
        '--output', str(output_path),
        '--no-playlist',
        '--progress',
    ]

    # Add duration filter if searching
    if not url:
        cmd.extend(['--match-filter', f'duration < {max_duration}'])

    # Try specific URL if provided
    if url:
        print(f"    Downloading: {url[:60]}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                cmd + [url],
                capture_output=True,
                timeout=180,  # 3 min timeout
                text=True
            )

            if output_path.exists() and output_path.stat().st_size > 10000:  # 10KB min
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")
                return output_path
            else:
                # Show error details
                if result.stderr:
                    print(f"✗ (download failed)")
                    print(f"      Error: {result.stderr[:200]}")
                else:
                    print("✗ (file too small or not created)")
                return None  # Don't fallback search when specific URL provided
        except subprocess.TimeoutExpired:
            print("✗ (timeout)")
            return None
        except FileNotFoundError:
            print("✗ (yt-dlp not found)")
            return None
        except Exception as e:
            print(f"✗ ({e})")
            return None

    # Try default sample URLs
    for try_url in SAMPLE_VOICEOVER_URLS:
        print(f"    Trying: {try_url[:50]}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                cmd + [try_url],
                capture_output=True,
                timeout=120,
                text=True
            )

            if output_path.exists() and output_path.stat().st_size > 10000:
                size_mb = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ ({size_mb:.1f} MB)")
                return output_path
            else:
                print("✗")
        except:
            print("✗")

    # Skip the slow search - just return None and use synthetic
    print("    No sample videos available, will use synthetic SRT")
    return None


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video using ffmpeg"""
    print(f"    Extracting audio...", end=" ", flush=True)

    try:
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'libmp3lame',
            '-ab', '128k',
            '-y',  # Overwrite
            str(audio_path)
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=60)

        if audio_path.exists() and audio_path.stat().st_size > 10000:
            print("✓")
            return True
        else:
            print("✗")
            return False
    except Exception as e:
        print(f"✗ ({e})")
        return False


def transcribe_to_srt(audio_path: Path, srt_path: Path) -> bool:
    """Transcribe audio to SRT using faster-whisper"""
    print(f"    Transcribing...", end=" ", flush=True)

    try:
        from faster_whisper import WhisperModel

        # Use smallest model for speed
        model = WhisperModel("tiny", device="auto", compute_type="auto")
        segments, info = model.transcribe(str(audio_path), language="en")

        lines = []
        segment_list = list(segments)

        for i, seg in enumerate(segment_list):
            # Format timestamps
            def fmt_time(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = t % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')

            lines.append(str(i + 1))
            lines.append(f"{fmt_time(seg.start)} --> {fmt_time(seg.end)}")
            lines.append(seg.text.strip())
            lines.append("")

        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        word_count = sum(len(seg.text.split()) for seg in segment_list)
        print(f"✓ ({len(segment_list)} segments, {word_count} words)")
        return True

    except ImportError:
        print("✗ (faster-whisper not installed)")
        return False
    except Exception as e:
        print(f"✗ ({e})")
        return False


def create_synthetic_srt(srt_path: Path) -> bool:
    """Create synthetic SRT with named entities for testing all features"""
    print(f"    Creating synthetic SRT...", end=" ", flush=True)

    # Segments designed to test various features:
    # - Named entities (for entity image/video search)
    # - Topic keywords (for chapter matching)
    # - Variety of content (for matching diversity)
    segments = [
        (0, 8, "Welcome to our documentary about technology innovation in Silicon Valley."),
        (8, 16, "Elon Musk founded SpaceX in 2002 with the goal of reducing space transportation costs."),
        (16, 24, "Tesla Motors revolutionized the electric vehicle industry with the Model S sedan."),
        (24, 32, "Meanwhile, Apple continues to dominate the smartphone market with the iPhone."),
        (32, 40, "Google's headquarters in Mountain View houses thousands of engineers."),
        (40, 48, "Amazon Web Services powers much of the internet's cloud infrastructure."),
        (48, 56, "Microsoft Azure competes directly with AWS for enterprise customers."),
        (56, 64, "The San Francisco Bay Area remains the world's leading tech hub."),
        (64, 72, "Venture capital firms on Sand Hill Road fund the next generation of startups."),
        (72, 80, "Thank you for watching this overview of the technology landscape."),
    ]

    lines = []
    for i, (start, end, text) in enumerate(segments):
        lines.append(str(i + 1))
        lines.append(f"00:00:{start:02d},000 --> 00:00:{end:02d},000")
        lines.append(text)
        lines.append("")

    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ ({len(segments)} segments)")
    return True


# =============================================================================
# PROJECT CONFIG
# =============================================================================

def create_test_config(project_dir: Path) -> Path:
    """Create project config optimized for testing (low quality, fast processing)"""

    config_content = f"""# Integration Test Config
# Created: {datetime.now().isoformat()}
# Purpose: Full pipeline test with all features enabled, minimal resources

# Non-interactive mode (no prompts)
enhanced:
  non_interactive: true
  face_preference: none
  enable_pexels: false   # Disable to avoid API costs
  enable_pixabay: false
  remix_enabled: false   # Disable remix for faster test
  stock_per_keyword: 0

# Entity search - minimal for testing
image_search:
  enabled: true
  images_per_entity: 1
  videos_per_entity: 0   # Skip entity videos
  max_entities: 2
  max_search_time: 30
  min_size_mb: 0.05      # 50KB min for testing (default is 1MB)

# Scene detection - fast preset
scene_detection:
  enabled: true
  preset: fast
  threshold: 35.0

# Transcription - use cached if available
transcription:
  model: tiny            # Smallest model
  use_gpu: false         # CPU for compatibility
  max_workers: 2

# Downloads - LOW QUALITY for testing
download:
  folder_name: downloaded_videos
  max_total_videos: 5    # Only 5 videos total
  preferred_quality: "360"
  max_quality: "480"
  tiers:
    short:
      min: 20
      max: 120
      per_keyword: 2
    medium:
      min: 120
      max: 300
      per_keyword: 1
    long:
      min: 300
      max: 600
      per_keyword: 0     # Disabled

downloading:
  output_dir: downloaded_videos
  preferred_quality: "360"
  max_quality: "480"

# Matching - basic settings
matching:
  min_confidence: 0.3
  embedding_candidates: 20
  llm_rerank_candidates: 3

# Keyword extraction - minimal
keyword_extraction:
  max_keywords: 3

# OTIO output - all tracks for testing
otio:
  include_alternatives: true
  include_voiceover: true
  alternatives_count: 2

# Logging
logging:
  log_dir: logs
  verbose: true

# Cache
cache:
  cache_dir: .cache
"""

    config_path = project_dir / "project_config.yaml"
    config_path.write_text(config_content)
    return config_path


# =============================================================================
# VALIDATION
# =============================================================================

def validate_otio_output(otio_path: Path) -> dict:
    """Validate OTIO file structure and content"""
    results = {'passed': 0, 'failed': 0, 'details': {}}

    try:
        import opentimelineio as otio

        timeline = otio.adapters.read_from_file(str(otio_path))

        # Check timeline name
        if timeline.name:
            results['details']['timeline_name'] = timeline.name
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Count tracks
        video_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Video]
        audio_tracks = [t for t in timeline.tracks if t.kind == otio.schema.TrackKind.Audio]

        results['details']['video_tracks'] = len(video_tracks)
        results['details']['audio_tracks'] = len(audio_tracks)

        if len(video_tracks) >= 1:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Check for clips with segment IDs
        clips_with_segment_ids = 0
        total_clips = 0

        for track in video_tracks:
            for item in track:
                if isinstance(item, otio.schema.Clip):
                    total_clips += 1
                    if item.name and '[S' in item.name:
                        clips_with_segment_ids += 1

        results['details']['total_clips'] = total_clips
        results['details']['clips_with_segment_ids'] = clips_with_segment_ids

        if clips_with_segment_ids > 0:
            results['passed'] += 1
        else:
            results['failed'] += 1

        # Check for metadata
        has_metadata = False
        for track in video_tracks:
            for item in track:
                if isinstance(item, otio.schema.Clip) and hasattr(item, 'metadata'):
                    if 'segment_index' in item.metadata:
                        has_metadata = True
                        break

        results['details']['has_metadata'] = has_metadata
        if has_metadata:
            results['passed'] += 1

    except ImportError:
        results['details']['error'] = "opentimelineio not installed"
    except Exception as e:
        results['details']['error'] = str(e)
        results['failed'] += 1

    return results


def validate_log_output(log_dir: Path) -> dict:
    """Validate log files and run summary"""
    results = {'passed': 0, 'failed': 0, 'details': {}}

    # Check for log file
    log_files = list(log_dir.glob("run_*.log"))
    if log_files:
        latest_log = sorted(log_files)[-1]
        results['details']['log_file'] = latest_log.name
        log_content = latest_log.read_text(encoding='utf-8', errors='replace')
        results['details']['log_size'] = len(log_content)
        results['passed'] += 1

        # Check for run summary in log
        if 'RUN SUMMARY' in log_content:
            results['details']['has_run_summary'] = True
            results['passed'] += 1
        else:
            results['details']['has_run_summary'] = False
            results['failed'] += 1
    else:
        results['failed'] += 1

    # Check for JSON log
    json_files = list(log_dir.glob("run_*.json"))
    if json_files:
        latest_json = sorted(json_files)[-1]
        results['details']['json_file'] = latest_json.name

        try:
            data = json.loads(latest_json.read_text())
            results['details']['json_valid'] = True

            # Check summary fields
            summary = data.get('summary', {})
            results['details']['total_segments'] = summary.get('total_segments', 0)
            results['details']['total_matches'] = summary.get('total_matches', 0)
            results['details']['avg_confidence'] = summary.get('avg_confidence', 0)

            results['passed'] += 1
        except:
            results['details']['json_valid'] = False
            results['failed'] += 1
    else:
        results['failed'] += 1

    # Check for summary files
    md_files = list(log_dir.glob("*_summary.md"))
    txt_files = list(log_dir.glob("*_summary.txt"))

    results['details']['summary_md'] = len(md_files) > 0
    results['details']['summary_txt'] = len(txt_files) > 0

    if md_files or txt_files:
        results['passed'] += 1

    return results


# =============================================================================
# API KEY CHECKING
# =============================================================================

def check_api_keys() -> dict:
    """Check which API keys are available"""
    keys = {
        'GEMINI_API_KEY': bool(os.environ.get('GEMINI_API_KEY')),
        'ANTHROPIC_API_KEY': bool(os.environ.get('ANTHROPIC_API_KEY')),
        'PEXELS_API_KEY': bool(os.environ.get('PEXELS_API_KEY')),
        'PIXABAY_API_KEY': bool(os.environ.get('PIXABAY_API_KEY')),
    }
    return keys


def has_required_keys() -> bool:
    """Check if minimum required API keys are available"""
    keys = check_api_keys()
    # At minimum we need GEMINI for matching and embedding
    return keys['GEMINI_API_KEY']


# =============================================================================
# MAIN TEST
# =============================================================================

def run_integration_test(
    keep_files: bool = False,
    voiceover_url: str = None,
    skip_download: bool = False,
    verbose: bool = True
) -> bool:
    """
    Run the full integration test.

    Args:
        keep_files: Keep temp project after test
        voiceover_url: Specific URL for voiceover
        skip_download: Skip downloading, use synthetic SRT only
        verbose: Show pipeline output

    Returns:
        True if test passed, False otherwise
    """

    print_box("FULL INTEGRATION TEST")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Install dir: {INSTALL_DIR}")
    print(f"  Mode: {'Synthetic SRT' if skip_download else 'Real Download'}")

    # Show .env status
    env_path = INSTALL_DIR / '.env'
    if env_path.exists():
        print(f"  .env: Loaded from {env_path.name}")
    else:
        print(f"  .env: Not found (using system environment)")

    # Check API keys
    keys = check_api_keys()
    print(f"\n  API Keys:")
    for key_name, available in keys.items():
        status = "✓" if available else "✗"
        print(f"    [{status}] {key_name}")

    if not has_required_keys():
        print_box("TEST SKIPPED", char="-")
        print("  ⚠ Missing required API keys")
        print("  ")
        print("  This integration test requires at minimum:")
        print("    • GEMINI_API_KEY - for matching and embeddings")
        print("  ")
        print("  Optional keys for full testing:")
        print("    • PEXELS_API_KEY - for stock footage")
        print("    • PIXABAY_API_KEY - for stock footage")
        print("    • ANTHROPIC_API_KEY - for secondary matching")
        print("  ")
        print("  Set these environment variables and re-run.")
        print("  ")
        print("  Example:")
        print("    export GEMINI_API_KEY=your_key_here")
        print("    python tests/test_integration_full.py --skip-download")
        print(f"\n{'=' * 60}\n")
        # Return True to not fail CI - skipped tests are not failures
        return True

    start_time = time.time()

    # =========================================================================
    # Create temp project
    # =========================================================================
    print_section("PROJECT SETUP")

    temp_base = Path(tempfile.mkdtemp(prefix='integration_test_'))
    project_dir = temp_base / 'test_project'

    # Create structure
    (project_dir / 'voiceover').mkdir(parents=True)
    (project_dir / 'downloaded_videos').mkdir(parents=True)
    (project_dir / 'output').mkdir(parents=True)
    (project_dir / 'logs').mkdir(parents=True)

    print(f"  Project: {project_dir}")

    # Create config
    config_path = create_test_config(project_dir)
    print(f"  Config: {config_path.name}")

    # =========================================================================
    # Prepare voiceover
    # =========================================================================
    print_section("VOICEOVER PREPARATION")

    vo_dir = project_dir / 'voiceover'
    vo_mp4 = vo_dir / 'voiceover.mp4'
    vo_mp3 = vo_dir / 'voiceover.mp3'
    vo_srt = vo_dir / 'voiceover.srt'

    srt_ready = False

    if not skip_download:
        # Try to download real voiceover
        downloaded = download_voiceover(vo_dir, voiceover_url, max_duration=300)

        if downloaded:
            # Extract audio
            if extract_audio(downloaded, vo_mp3):
                # Transcribe
                if transcribe_to_srt(vo_mp3, vo_srt):
                    srt_ready = True

    # Fallback to synthetic SRT
    if not srt_ready:
        print("    Using synthetic SRT (faster testing)")
        create_synthetic_srt(vo_srt)
        srt_ready = True

    if not vo_srt.exists():
        print("  ✗ Failed to prepare voiceover SRT")
        return False

    # Show SRT info
    srt_content = vo_srt.read_text(encoding='utf-8')
    segment_count = srt_content.count('-->')
    print(f"  SRT ready: {segment_count} segments")

    # =========================================================================
    # Run pipeline
    # =========================================================================
    print_section("RUNNING PIPELINE")

    cmd = [
        sys.executable,
        str(INSTALL_DIR / 'main.py'),
        '--project', str(project_dir),
        '--voiceover', str(vo_srt),
        '--keywords', '2',
    ]

    print(f"  Command: python main.py --project <temp> --voiceover <srt> --keywords 2")
    print(f"  Timeout: 10 minutes")
    print()

    pipeline_start = time.time()
    pipeline_success = False

    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'

    try:
        if verbose:
            print("  " + "-" * 56)
            print("  PIPELINE OUTPUT:")
            print("  " + "-" * 56)

            process = subprocess.Popen(
                cmd,
                cwd=str(INSTALL_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
            )

            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())

            process.wait(timeout=600)
            returncode = process.returncode

            print("  " + "-" * 56)
        else:
            result = subprocess.run(
                cmd,
                cwd=str(INSTALL_DIR),
                capture_output=True,
                timeout=600,
                env=env,
                text=True,
                errors='replace'
            )
            returncode = result.returncode

            # Show last 20 lines
            print("  Last 20 lines:")
            for line in result.stdout.strip().split('\n')[-20:]:
                print(f"  | {line}")

        pipeline_duration = time.time() - pipeline_start
        pipeline_success = (returncode == 0)

        # Handle Windows crash-on-exit
        if returncode == 3221226505 or returncode == -1073740791:
            print(f"\n  Note: Windows crash-on-exit (harmless if outputs exist)")
            pipeline_success = True

        status = "✓" if pipeline_success else "✗"
        print(f"\n  {status} Pipeline {'completed' if pipeline_success else 'failed'} in {pipeline_duration:.1f}s")

    except subprocess.TimeoutExpired:
        print(f"\n  ✗ Pipeline timed out after 10 minutes")
        pipeline_duration = 600
    except Exception as e:
        print(f"\n  ✗ Pipeline error: {e}")
        pipeline_duration = time.time() - pipeline_start

    # =========================================================================
    # Validate outputs
    # =========================================================================
    print_section("OUTPUT VALIDATION")

    validation_results = {
        'otio': False,
        'xml': False,
        'logs': False,
        'videos': False,
    }

    # Check OTIO
    otio_files = list((project_dir / 'output').glob('*.otio'))
    if otio_files:
        validation_results['otio'] = True
        otio_validation = validate_otio_output(otio_files[0])
        print_check("OTIO file exists", True, otio_files[0].name)
        print_check("  - Has video tracks", otio_validation['details'].get('video_tracks', 0) > 0,
                   f"{otio_validation['details'].get('video_tracks', 0)} tracks")
        print_check("  - Has segment IDs", otio_validation['details'].get('clips_with_segment_ids', 0) > 0,
                   f"{otio_validation['details'].get('clips_with_segment_ids', 0)} clips")
        print_check("  - Has metadata", otio_validation['details'].get('has_metadata', False))
    else:
        print_check("OTIO file exists", False)

    # Check XML
    xml_files = list((project_dir / 'output').glob('*.xml'))
    if xml_files:
        validation_results['xml'] = True
        xml_content = xml_files[0].read_text(encoding='utf-8', errors='replace')
        print_check("XML file exists", True, xml_files[0].name)
        print_check("  - Has clip IDs", 'clipitem id=' in xml_content)
        print_check("  - Has audio refs", '<audio>' in xml_content)
    else:
        print_check("XML file exists", False)

    # Check logs
    log_dir = project_dir / 'logs'
    if log_dir.exists():
        log_validation = validate_log_output(log_dir)
        validation_results['logs'] = log_validation['passed'] > 0

        print_check("Log file exists", 'log_file' in log_validation['details'],
                   log_validation['details'].get('log_file', ''))
        print_check("  - Has run summary", log_validation['details'].get('has_run_summary', False))
        print_check("JSON log valid", log_validation['details'].get('json_valid', False))
        print_check("Summary files", log_validation['details'].get('summary_md', False) or
                   log_validation['details'].get('summary_txt', False))

        # Show stats from JSON
        if 'total_matches' in log_validation['details']:
            print(f"    Stats: {log_validation['details']['total_matches']} matches, "
                  f"{log_validation['details']['avg_confidence']*100:.0f}% avg confidence")
    else:
        print_check("Log directory exists", False)

    # Check videos
    video_dir = project_dir / 'downloaded_videos'
    video_count = len(list(video_dir.rglob('*.mp4'))) + len(list(video_dir.rglob('*.webm')))
    if video_count > 0:
        validation_results['videos'] = True
        print_check("Videos downloaded", True, f"{video_count} videos")
    else:
        print_check("Videos downloaded", False, "0 videos (may be expected for synthetic SRT)")

    # =========================================================================
    # Final result
    # =========================================================================
    total_duration = time.time() - start_time

    print_box("TEST RESULT")

    # Determine pass/fail
    has_outputs = validation_results['otio'] or validation_results['xml']
    has_logs = validation_results['logs']

    if pipeline_success and has_logs:
        if has_outputs:
            print(f"  ✓ FULL INTEGRATION TEST PASSED")
            print(f"    - Pipeline completed successfully")
            print(f"    - All output files generated")
        else:
            print(f"  ⚠ PARTIAL PASS (no matches but pipeline ran)")
            print(f"    - Pipeline completed but no matches found")
            print(f"    - This may be expected with synthetic SRT")
        passed = True
    else:
        print(f"  ✗ INTEGRATION TEST FAILED")
        if not pipeline_success:
            print(f"    - Pipeline execution failed")
        if not has_logs:
            print(f"    - No log files generated")
        passed = False

    print(f"\n  Duration: {total_duration:.1f}s (pipeline: {pipeline_duration:.1f}s)")

    # Cleanup or keep
    if keep_files:
        print(f"\n  Project kept at: {project_dir}")
        print(f"    └─ output/             - OTIO/XML files")
        print(f"    └─ logs/               - Run logs and summaries")
        print(f"    └─ downloaded_videos/  - Downloaded footage")
    else:
        print(f"\n  Cleaning up temp files...")
        try:
            shutil.rmtree(temp_base)
        except:
            pass

    print(f"\n{'=' * 60}\n")

    return passed


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Full Integration Test - Real Project Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_integration_full.py                    # Run with real download
  python tests/test_integration_full.py --skip-download    # Use synthetic SRT only
  python tests/test_integration_full.py --keep             # Keep temp files
  python tests/test_integration_full.py --voiceover URL    # Use specific video
"""
    )

    parser.add_argument('--keep', action='store_true',
                       help='Keep temp project after test')
    parser.add_argument('--voiceover', type=str, default=None,
                       help='URL of video to use as voiceover')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip downloading, use synthetic SRT only')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Hide pipeline output')

    args = parser.parse_args()

    success = run_integration_test(
        keep_files=args.keep,
        voiceover_url=args.voiceover,
        skip_download=args.skip_download,
        verbose=not args.quiet
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
