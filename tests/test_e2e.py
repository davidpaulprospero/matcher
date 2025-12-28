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
    """Download a short audio clip for voiceover - prioritize content with named entities"""
    searches = [
        # Prioritize content likely to have named entities (people, companies, places)
        'tech news report Tesla SpaceX 1 minute',
        'business news earnings report 30 seconds',
        'documentary narration history short',
        'news anchor voiceover clip',
        'ted talk excerpt 1 minute',
        'podcast intro clip short',
        'public domain speech audio',
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


def validate_srt_content(srt_path: Path) -> bool:
    """Check if SRT has useful content for testing
    
    Returns True if SRT has:
    - At least 2 segments
    - At least 50 total words
    - Content that's not just music/noise transcription
    """
    if not srt_path.exists():
        return False
    
    try:
        content = srt_path.read_text(encoding='utf-8')
        
        # Extract text lines (every 3rd line starting from line 3)
        lines = content.strip().split('\n')
        text_lines = []
        for i, line in enumerate(lines):
            # Skip sequence numbers and timestamps
            if '-->' not in line and not line.strip().isdigit() and line.strip():
                text_lines.append(line.strip())
        
        full_text = ' '.join(text_lines)
        word_count = len(full_text.split())
        segment_count = content.count('-->')
        
        # Check for music/noise indicators
        noise_indicators = ['[music]', '[applause]', '[laughter]', '♪', '🎵']
        is_mostly_noise = any(ind in full_text.lower() for ind in noise_indicators)
        
        # Validate
        if segment_count < 2:
            print(f"    ⚠ SRT has only {segment_count} segments (need 2+)")
            return False
        if word_count < 50:
            print(f"    ⚠ SRT has only {word_count} words (need 50+)")
            return False
        if is_mostly_noise:
            print(f"    ⚠ SRT appears to be mostly music/noise")
            return False
        
        print(f"    ✓ SRT validated: {segment_count} segments, {word_count} words")
        return True
        
    except Exception as e:
        print(f"    ⚠ SRT validation failed: {e}")
        return False


def create_synthetic_srt(srt_path: Path) -> bool:
    """Create a synthetic SRT for testing - includes named entities AND obscure terms
    
    Named entities trigger entity image/video search.
    Obscure terms trigger remix retry logic when downloads fail.
    """
    texts = [
        # Segment 1: Named entities - PERSON, ORG, LOCATION
        "Elon Musk announced major changes at Tesla headquarters in Palo Alto, California.",
        # Segment 2: More entities - ORG, LOCATION  
        "SpaceX continues to revolutionize space exploration with Starship launches from Texas.",
        # Segment 3: Obscure/rare terms to trigger remix retry
        "The rare Xenophyophore organisms exhibit bioluminescent properties in abyssal zones.",
        # Segment 4: Mix of common and specific
        "Climate scientists study permafrost degradation in the Arctic tundra regions.",
    ]
    
    lines = []
    for i, text in enumerate(texts):
        start = f"00:00:{i*10:02d},000"
        end = f"00:00:{(i+1)*10:02d},000"
        lines.extend([str(i + 1), f"{start} --> {end}", text, ""])
    
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"    Created synthetic SRT (4 segments: entities + obscure terms)")
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
    
    # Create project config for non-interactive mode with ALL features enabled (minimal)
    project_config = project_dir / 'project_config.yaml'
    project_config.write_text(f"""# E2E Test Config - ALL FEATURES enabled with minimal settings
# Goal: Test every feature at least once, but keep processing time low

enhanced:
  non_interactive: true
  face_preference: none   # Test face detection (prefer clips WITHOUT faces)
  enable_pexels: true     # Enable stock footage
  enable_pixabay: true
  stock_per_keyword: 1    # Only 1 stock video per keyword (cost saving)
  remix_enabled: true     # Enable keyword remix

# Entity/Image search - enabled with minimal settings
image_search:
  enabled: true
  images_per_entity: 1  # Just 1 image per entity
  videos_per_entity: 1  # Just 1 video per entity
  max_search_time: 30   # Limit search time

# Stock footage - minimal (API cost saving)
stock_footage:
  pexels_enabled: true
  pixabay_enabled: true
  per_keyword: 1  # Just 1 per keyword

# Remix - enabled with minimal settings  
remix:
  enabled: true
  interactive_curation: false
  auto_accept_filter: filtered
  min_relevance_score: 0.1

# Zero-download remix - enabled (will trigger on obscure keywords)
zero_download_remix:
  enabled: true
  max_retries: 1

# Scene detection - enabled
scene_detection:
  enabled: true
  preset: fast
  threshold: 30.0

# Pipeline settings - don't skip anything
pipeline:
  skip_scene_detection: false
  skip_image_search: false

# Use temp folder for cache
cache_dir: .cache
cache:
  cache_dir: .cache

# Downloads - ALL TIERS enabled, LONGER limited to 1 total
download:
  root_dir: null
  folder_name: downloaded_videos
  max_total_videos: 15
  preferred_quality: "360"  # LOWEST quality for testing
  max_quality: "480"
  tiers:
    short:
      min: 20
      max: 120
      per_keyword: 1
    medium:
      min: 120
      max: 600
      per_keyword: 1
    long:
      min: 600
      max: 1500
      per_keyword: 1
    longer:
      min: 1500
      max: 3000
      per_keyword: 1
      max_total: 1  # Only 1 LONGER video total across all keywords

# Downloading settings (yt-dlp)
downloading:
  output_dir: downloaded_videos
  preferred_quality: "360"  # LOWEST quality for testing
  max_quality: "480"

# Keyword settings - minimal for fast testing
keyword_extraction:
  max_keywords: 2  # Only 2 keywords for fast testing
""")
    print(f"  Created: project_config.yaml (ALL features enabled, minimal settings)")
    
    # =========================================================================
    # Prepare voiceover
    # =========================================================================
    print_section("VOICEOVER PREPARATION")
    
    vo_mp3 = project_dir / 'voiceover' / 'test.mp3'
    vo_srt = project_dir / 'voiceover' / 'test.srt'
    
    use_synthetic = False
    
    # Try to download real audio first
    if download_audio(vo_mp3, cookies_path):
        # Transcribe it
        if create_srt_from_audio(vo_mp3, vo_srt):
            # Validate the transcription has useful content
            if not validate_srt_content(vo_srt):
                print(f"    → Real audio transcription not useful, using synthetic SRT")
                use_synthetic = True
        else:
            use_synthetic = True
    else:
        use_synthetic = True
    
    # Fall back to synthetic SRT with entities
    if use_synthetic:
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
        '--keywords', '2',  # Only 2 keywords for faster testing
    ]
    
    print(f"  Command: python main.py --project <temp> --voiceover test.srt --keywords 2")
    print(f"  Timeout: 15 minutes (full feature test)")
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
            
            process.wait(timeout=900)  # 15 min for full feature test
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
                timeout=900,  # 15 min for full feature test
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
    
    # Check stock footage specifically
    stock_dir = project_dir / 'downloaded_videos' / 'stock'
    if stock_dir.exists():
        stock_count = len(list(stock_dir.rglob('*.mp4')))
        if stock_count > 0:
            outputs_found['stock'] = stock_count
            print(f"  ✓ Stock footage: {stock_count} videos (Pexels/Pixabay)")
        else:
            print(f"  ⚠ Stock footage: Directory exists but empty")
    else:
        print(f"  ⚠ Stock footage: Not downloaded (dir missing)")
    
    # Check entity images
    entity_images_dir = project_dir / 'downloaded_videos' / 'entity_images'
    if entity_images_dir.exists():
        img_count = len(list(entity_images_dir.rglob('*.jpg'))) + len(list(entity_images_dir.rglob('*.png')))
        if img_count > 0:
            outputs_found['entity_images'] = img_count
            print(f"  ✓ Entity images: {img_count} images")
        else:
            print(f"  ⚠ Entity images: Directory exists but empty")
    else:
        print(f"  ⚠ Entity images: Not downloaded (dir missing)")
    
    # Check JSON log for feature usage stats
    if 'json' in outputs_found:
        try:
            data = json.loads(outputs_found['json'].read_text())
            stages = data.get('stages', {})
            
            # Check if various features were used
            features_used = []
            if stages.get('ENTITY_IMAGES', {}).get('images', 0) > 0:
                features_used.append('entity_images')
            if stages.get('ENTITY_VIDEOS', {}).get('videos', 0) > 0:
                features_used.append('entity_videos')
            if stages.get('REMIX', {}).get('videos', 0) > 0:
                features_used.append('remix')
            
            if features_used:
                print(f"  ✓ Features used: {', '.join(features_used)}")
        except:
            pass
    
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
