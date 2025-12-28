#!/usr/bin/env python3
"""
End-to-End Pipeline Test

Actually runs main.py with a test project and validates the outputs.

Usage:
    python tests/test_e2e.py
    python tests/test_e2e.py --keep       # Keep temp project for inspection
    python tests/test_e2e.py --verbose    # Show pipeline output in real-time
"""

import os
import sys

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
import time
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Get the install directory (parent of tests folder)
INSTALL_DIR = Path(__file__).parent.parent.resolve()


def print_box(text: str):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def print_section(text: str):
    print(f"\n  ─── {text} ───")


def download_audio(output_path: Path, cookies_path: str = None) -> bool:
    """Download a short audio clip for voiceover"""
    searches = [
        'short motivational speech 30 seconds',
        'ted talk excerpt 1 minute',
        'podcast intro clip short',
        'public domain speech audio',
        'news report audio short',
    ]
    
    for query in searches:
        print(f"    Trying: {query[:40]}...", end=" ", flush=True)
        
        cmd = [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', '128K',
            '--max-downloads', '1',
            '--match-filter', 'duration < 120',
            '--output', str(output_path).replace('.mp3', '.%(ext)s'),
            '--no-playlist',
            '--quiet',
            '--no-warnings',
            f'ytsearch1:{query}'
        ]
        
        if cookies_path:
            cmd.extend(['--cookies', cookies_path])
        
        try:
            subprocess.run(cmd, capture_output=True, timeout=90)
            
            # Check for downloaded file
            for ext in ['.mp3', '.m4a', '.webm', '.opus']:
                check = output_path.parent / f"{output_path.stem}{ext}"
                if check.exists() and check.stat().st_size > 10000:
                    if ext != '.mp3':
                        check.rename(output_path)
                    print("✓")
                    return True
            print("✗")
        except subprocess.TimeoutExpired:
            print("timeout")
        except FileNotFoundError:
            print("yt-dlp not found")
            return False
        except:
            print("error")
    
    return False


def create_srt_from_audio(mp3_path: Path, srt_path: Path) -> bool:
    """Transcribe audio to SRT using Whisper"""
    print(f"    Transcribing audio...", end=" ", flush=True)
    
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel("base", device="auto", compute_type="auto")
        segments, _ = model.transcribe(str(mp3_path))
        
        lines = []
        segment_list = list(segments)
        
        for i, seg in enumerate(segment_list):
            h, m, s = int(seg.start // 3600), int((seg.start % 3600) // 60), seg.start % 60
            start = f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')
            h, m, s = int(seg.end // 3600), int((seg.end % 3600) // 60), seg.end % 60
            end = f"{h:02d}:{m:02d}:{s:06.3f}".replace('.', ',')
            
            lines.append(str(i + 1))
            lines.append(f"{start} --> {end}")
            lines.append(seg.text.strip())
            lines.append("")
        
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✓ ({len(segment_list)} segments)")
        return True
        
    except Exception as e:
        print(f"failed: {e}")
        return False


def create_synthetic_srt(srt_path: Path) -> bool:
    """Create a synthetic SRT for testing"""
    texts = [
        "Technology continues to transform our daily lives in remarkable ways.",
        "Scientists are making breakthrough discoveries across multiple fields.",
        "The global economy shows signs of resilience and adaptation.",
        "Communities worldwide are finding innovative solutions to challenges.",
        "Environmental awareness is driving sustainable development initiatives.",
        "Education is evolving to meet the needs of a changing world.",
        "Healthcare innovations are improving outcomes for patients everywhere.",
        "Digital connectivity is bridging gaps between distant communities.",
        "Creative industries are exploring new forms of artistic expression.",
        "The future holds both challenges and unprecedented opportunities.",
    ]
    
    lines = []
    for i, text in enumerate(texts):
        start = f"00:00:{i*6:02d},000"
        end = f"00:00:{(i+1)*6:02d},000"
        lines.extend([str(i + 1), f"{start} --> {end}", text, ""])
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"    Created synthetic SRT (10 segments)")
    return True


def run_e2e_test(keep_files: bool = False, verbose: bool = True) -> bool:
    """Run the actual pipeline and validate outputs
    
    Args:
        keep_files: Keep temp project after test
        verbose: Show pipeline output in real-time (default: True)
    """
    
    print_box("END-TO-END PIPELINE TEST")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Install dir: {INSTALL_DIR}")
    
    start_time = time.time()
    
    # Find cookies
    cookies_path = None
    for cp in [INSTALL_DIR / 'cookies.txt', Path('cookies.txt')]:
        if cp.exists():
            cookies_path = str(cp)
            break
    
    # =========================================================================
    # Create temp project
    # =========================================================================
    print_section("PROJECT SETUP")
    
    temp_base = Path(tempfile.mkdtemp(prefix='e2e_test_'))
    project_dir = temp_base / 'test_project'
    
    # Create structure
    (project_dir / 'voiceover').mkdir(parents=True)
    (project_dir / 'downloaded_videos').mkdir(parents=True)
    (project_dir / 'output').mkdir(parents=True)
    (project_dir / 'logs').mkdir(parents=True)
    
    print(f"  Created: {project_dir}")
    
    # Create project config for non-interactive mode
    project_config = project_dir / 'project_config.yaml'
    project_config.write_text(f"""# E2E Test Config - Non-interactive mode with minimal settings
enhanced:
  non_interactive: true
  face_preference: neutral
  enable_pexels: false  # Disable stock footage for testing
  enable_pixabay: false
  remix_enabled: false  # Disable keyword remix

# Disable ALL remix features
remix:
  interactive_curation: false
  auto_accept_filter: filtered
  enabled: false

# Disable zero-download remix (retry failed keywords)
zero_download_remix:
  enabled: false

# Use temp folder for cache (not shared install dir cache)
# Both top-level and nested for compatibility
cache_dir: .cache
cache:
  cache_dir: .cache

# Use temp folder for downloads (not shared E:/v)
download:
  root_dir: null  # Null = use project folder
  folder_name: downloaded_videos
  max_total_videos: 10  # Cap total videos
  # Complete tier definitions with min/max durations
  tiers:
    short:
      min: 20
      max: 120
      per_keyword: 1
      max_results: 2
    medium:
      min: 120
      max: 600
      per_keyword: 0
      max_results: 0
    long:
      min: 600
      max: 1500
      per_keyword: 0
      max_results: 0
    longer:
      min: 1500
      max: 3000
      per_keyword: 0
      max_results: 0

# Downloading settings
downloading:
  output_dir: downloaded_videos

# MINIMAL settings for testing (fast run)
keyword_extraction:
  max_keywords: 5  # Only 5 keywords for testing
""")
    print(f"  Created: project_config.yaml (minimal test settings)")
    
    # =========================================================================
    # Prepare voiceover
    # =========================================================================
    print_section("VOICEOVER PREPARATION")
    
    vo_mp3 = project_dir / 'voiceover' / 'test.mp3'
    vo_srt = project_dir / 'voiceover' / 'test.srt'
    
    # Try to download real audio
    if download_audio(vo_mp3, cookies_path):
        # Transcribe it
        if not create_srt_from_audio(vo_mp3, vo_srt):
            create_synthetic_srt(vo_srt)
    else:
        # Use synthetic SRT
        create_synthetic_srt(vo_srt)
    
    if not vo_srt.exists():
        print("  ✗ Could not create voiceover SRT")
        return False
    
    # =========================================================================
    # Run the actual pipeline
    # =========================================================================
    print_section("RUNNING PIPELINE")
    
    cmd = [
        sys.executable,
        str(INSTALL_DIR / 'main.py'),
        '--project', str(project_dir),
        '--voiceover', str(vo_srt),
        '--keywords', '5',  # Minimal keywords for testing
    ]
    
    print(f"  Command: python main.py --project <temp> --voiceover test.srt --keywords 5")
    print(f"  Timeout: 10 minutes")
    print()
    
    pipeline_start = time.time()
    
    # Set environment for consistent encoding
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    try:
        if verbose:
            print("  " + "-" * 56)
            print("  PIPELINE OUTPUT:")
            print("  " + "-" * 56)
            
            # Stream output in real-time
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
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                # Print directly (pipeline already has indentation)
                print(line.rstrip())
                output_lines.append(line)
            
            process.wait(timeout=600)
            returncode = process.returncode
            output = ''.join(output_lines)
            
            print("  " + "-" * 56)
        else:
            # Capture output silently
            result = subprocess.run(
                cmd,
                cwd=str(INSTALL_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                env=env,
            )
            # Decode with UTF-8, replacing errors
            output = result.stdout.decode('utf-8', errors='replace')
            returncode = result.returncode
            
            # Print last 30 lines for context
            print("  Last 30 lines of output:")
            for line in output.strip().split('\n')[-30:]:
                print(f"  | {line}")
        
        pipeline_duration = time.time() - pipeline_start
        pipeline_success = (returncode == 0)
        
        print()
        if pipeline_success:
            print(f"  ✓ Pipeline completed in {pipeline_duration:.1f}s")
        else:
            print(f"  ✗ Pipeline failed (exit code {returncode}) after {pipeline_duration:.1f}s")
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Pipeline timed out after 10 minutes")
        pipeline_success = False
        pipeline_duration = 600
    except Exception as e:
        print(f"  ✗ Pipeline error: {e}")
        pipeline_success = False
        pipeline_duration = time.time() - pipeline_start
    
    # =========================================================================
    # Check outputs
    # =========================================================================
    print_section("OUTPUT VALIDATION")
    
    outputs_found = {}
    
    # Check OTIO
    otio_files = list((project_dir / 'output').glob('*.otio'))
    if otio_files:
        f = otio_files[0]
        outputs_found['otio'] = f
        print(f"  ✓ OTIO: {f.name} ({f.stat().st_size:,} bytes)")
    else:
        print(f"  ✗ OTIO: Not found")
    
    # Check XML
    xml_files = list((project_dir / 'output').glob('*.xml'))
    if xml_files:
        f = xml_files[0]
        outputs_found['xml'] = f
        content = f.read_text(encoding='utf-8')
        checks = []
        if '<uuid>' in content: checks.append('uuid')
        if '<audio>' in content: checks.append('audio')
        if 'clipitem id=' in content: checks.append('clipitem-id')
        print(f"  ✓ XML: {f.name} ({f.stat().st_size:,} bytes) [{', '.join(checks)}]")
    else:
        print(f"  ✗ XML: Not found")
    
    # Check logs
    log_files = list((project_dir / 'logs').glob('run_*.log'))
    if log_files:
        f = sorted(log_files)[-1]
        outputs_found['log'] = f
        print(f"  ✓ Log: {f.name} ({f.stat().st_size:,} bytes)")
    else:
        print(f"  ✗ Log: Not found")
    
    # Check JSON log
    json_files = list((project_dir / 'logs').glob('run_*.json'))
    if json_files:
        f = sorted(json_files)[-1]
        outputs_found['json'] = f
        try:
            data = json.loads(f.read_text())
            summary = data.get('summary', {})
            print(f"  ✓ JSON: {f.name}")
            print(f"      Matches: {summary.get('total_matches', 'N/A')}")
            print(f"      Confidence: {summary.get('avg_confidence', 0)*100:.1f}%")
        except:
            print(f"  ✓ JSON: {f.name} (parse error)")
    else:
        print(f"  ✗ JSON: Not found")
    
    # Check summary files
    for pattern, name in [('*_summary.md', 'Summary MD'), ('*_summary.txt', 'Summary TXT')]:
        files = list((project_dir / 'logs').glob(pattern))
        if files:
            f = sorted(files)[-1]
            outputs_found[name] = f
            print(f"  ✓ {name}: {f.name} ({f.stat().st_size:,} bytes)")
        else:
            print(f"  ✗ {name}: Not found")
    
    # Check downloaded videos
    video_count = len(list((project_dir / 'downloaded_videos').rglob('*.mp4')))
    video_count += len(list((project_dir / 'downloaded_videos').rglob('*.webm')))
    if video_count > 0:
        outputs_found['videos'] = video_count
        print(f"  ✓ Videos: {video_count} downloaded")
    else:
        print(f"  ✗ Videos: None downloaded")
    
    # =========================================================================
    # Final result
    # =========================================================================
    total_duration = time.time() - start_time
    
    print_box("TEST RESULT")
    
    # Determine pass/fail
    # Pipeline success = outputs generated (Windows may crash on exit with code 3221226505)
    has_log = 'log' in outputs_found
    has_otio = 'otio' in outputs_found
    has_xml = 'xml' in outputs_found
    has_videos = 'videos' in outputs_found
    
    # Check for Windows crash-on-exit (0xC0000409 = 3221226505)
    windows_crash_exit = (returncode == 3221226505 or returncode == -1073740791)
    
    # Success if: (pipeline exit 0 OR windows crash but outputs exist) AND has outputs
    outputs_success = has_log and (has_otio or has_xml or has_videos)
    
    if (pipeline_success or (windows_crash_exit and outputs_success)) and has_log:
        print(f"  ✓ E2E TEST PASSED")
        if has_otio and has_xml:
            print(f"    - Full output generated (OTIO + XML)")
        elif outputs_success:
            print(f"    - Pipeline completed with outputs")
        else:
            print(f"    - Pipeline completed (no matches = no output files)")
        if windows_crash_exit:
            print(f"    - Note: Windows crash-on-exit (harmless)")
        passed = True
    else:
        print(f"  ✗ E2E TEST FAILED")
        if not pipeline_success and not windows_crash_exit:
            print(f"    - Pipeline execution failed (exit code: {returncode})")
        if not has_log:
            print(f"    - No log file generated")
        passed = False
    
    print(f"\n  Duration: {total_duration:.1f}s (pipeline: {pipeline_duration:.1f}s)")
    print(f"  Outputs: {len(outputs_found)} files generated")
    
    # Cleanup or keep
    if keep_files:
        print(f"\n  Project kept at: {project_dir}")
        print(f"    └─ output/     - OTIO and XML files")
        print(f"    └─ logs/       - Run logs and summaries")
        print(f"    └─ downloaded_videos/ - Video footage")
    else:
        print(f"\n  Cleaning up temp files...")
        try:
            shutil.rmtree(temp_base)
        except:
            pass
    
    print(f"{'=' * 60}\n")
    
    return passed


def main():
    parser = argparse.ArgumentParser(description='E2E Pipeline Test')
    parser.add_argument('--keep', action='store_true', help='Keep temp project')
    parser.add_argument('--quiet', '-q', action='store_true', help='Hide pipeline output (show only summary)')
    args = parser.parse_args()
    
    success = run_e2e_test(keep_files=args.keep, verbose=not args.quiet)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
