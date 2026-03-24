"""Download DVIDS Hub videos matching a project's voiceover SRT topic.

Searches dvidshub.net for videos related to SRT content, downloads them,
and segments any file exceeding 5MB into smaller chunks (~10s target).
Accumulates at least 60s total duration.

Usage:
    python scripts/download_dvids.py "E:\\Edit Job\\Degold\\DeepSeaReports\\PROJECT"
    python scripts/download_dvids.py "E:\\Edit Job\\Degold\\PROJECT" --dry-run
    python scripts/download_dvids.py "E:\\Edit Job\\Degold\\PROJECT" --query "naval operations"
    python scripts/download_dvids.py "E:\\Edit Job\\Degold\\PROJECT" --min-clips 30
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import requests

_script_path = os.path.abspath(__file__)
PROJECT_ROOT = Path(_script_path).parent.parent.resolve()
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))
os.chdir(PROJECT_ROOT)

from script_utils import (
    print_ok, print_info, print_warn, print_error, print_header,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DVIDS_SEARCH_URL = "https://api.dvidshub.net/search"
DVIDS_PAGE_LIMIT = 50
DEFAULT_MAX_SEGMENT_MB = 5.0
DEFAULT_TARGET_SEG_DUR = 10.0
DEFAULT_MIN_CLIPS = 20
DEFAULT_MAX_RESULTS = 10
DOWNLOAD_TIMEOUT = 120
RATE_LIMIT_DELAY = 0.5

ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can",
    "could", "did", "do", "does", "each", "for", "from", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "just", "like", "may", "me", "might", "more", "most", "much",
    "my", "no", "nor", "not", "now", "of", "on", "one", "only", "or", "other",
    "our", "out", "over", "own", "said", "same", "she", "should", "so", "some",
    "still", "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "up",
    "upon", "us", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "would", "you", "your",
    "also", "about", "after", "all", "am", "any", "because", "before", "being",
    "between", "both", "come", "day", "don", "down", "even", "every", "few",
    "first", "get", "give", "go", "going", "gone", "good", "got", "great",
    "guy", "guys", "gonna", "gotta", "know", "let", "ll", "long", "look",
    "make", "man", "many", "men", "new", "off", "oh", "ok", "okay", "old",
    "once", "part", "people", "put", "re", "right", "say", "see", "take",
    "tell", "thing", "think", "time", "two", "use", "ve", "want", "way",
    "well", "went", "what", "yeah", "year", "yes",
}


# ---------------------------------------------------------------------------
# .env helpers
# ---------------------------------------------------------------------------

def parse_env_file(path: Path) -> dict[str, str]:
    """Parse .env file without loading global environment."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def resolve_api_key(cli_key: Optional[str]) -> str:
    """Resolve DVIDS API key from CLI flag, env var, or .env file."""
    if cli_key:
        return cli_key
    env_key = os.getenv("DVIDS_API_KEY", "")
    if env_key:
        return env_key
    env_data = parse_env_file(PROJECT_ROOT / ".env")
    if env_data.get("DVIDS_API_KEY"):
        return env_data["DVIDS_API_KEY"]
    return ""


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

def parse_srt(srt_path: Path) -> list[dict]:
    """Parse SRT subtitle file into segments.

    Adapted from src/stages/analyze.py:479-517.
    """
    content = srt_path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\n+", content.strip())
    segments = []

    for i, block in enumerate(blocks):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        timestamp_match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})",
            lines[1],
        )
        if not timestamp_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, timestamp_match.groups())
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
        text = " ".join(lines[2:]).strip()
        segments.append({"index": i, "start": start, "end": end, "text": text})

    return segments


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

def extract_topics(segments: list[dict], max_queries: int = 5) -> list[str]:
    """Extract search queries from SRT text via word frequency and n-grams."""
    full_text = " ".join(seg["text"] for seg in segments)

    # Strip HTML tags and bracketed annotations like [Music], [Applause]
    full_text = re.sub(r"<[^>]+>", " ", full_text)
    full_text = re.sub(r"\[[^\]]*\]", " ", full_text)
    full_text = full_text.lower()

    # Tokenize
    words = re.findall(r"[a-z]+", full_text)
    content_words = [w for w in words if w not in ENGLISH_STOPWORDS and len(w) >= 3]

    if len(content_words) < 5:
        # Very short SRT - use the entire cleaned text as one query
        cleaned = re.sub(r"\s+", " ", full_text).strip()
        return [cleaned] if cleaned else []

    # Count unigrams
    unigram_counts = Counter(content_words)

    # Build bigrams and trigrams from consecutive content words
    bigrams: list[str] = []
    trigrams: list[str] = []
    for i in range(len(content_words) - 1):
        bigrams.append(f"{content_words[i]} {content_words[i + 1]}")
    for i in range(len(content_words) - 2):
        trigrams.append(
            f"{content_words[i]} {content_words[i + 1]} {content_words[i + 2]}"
        )

    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)

    # Score phrases: trigrams weighted 3x, bigrams 2x, unigrams 1x
    scored: dict[str, float] = {}
    for phrase, count in trigram_counts.items():
        if count >= 2:
            scored[phrase] = count * 3
    for phrase, count in bigram_counts.items():
        if count >= 2:
            scored[phrase] = scored.get(phrase, 0) + count * 2
    for word, count in unigram_counts.most_common(20):
        if count >= 3:
            scored[word] = scored.get(word, 0) + count

    # Sort by score descending
    ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)

    # Deduplicate: skip phrases that are substrings of higher-ranked ones
    queries: list[str] = []
    for phrase, _score in ranked:
        if len(queries) >= max_queries:
            break
        if any(phrase in existing for existing in queries):
            continue
        queries.append(phrase)

    # Fallback: if no multi-word phrases scored, use top unigrams
    if not queries:
        for word, _count in unigram_counts.most_common(max_queries):
            queries.append(word)

    return queries


# ---------------------------------------------------------------------------
# DVIDS API client
# ---------------------------------------------------------------------------

class DvidsDownloader:
    """Standalone DVIDS Hub API client for searching and downloading videos.

    Uses the DVIDS search API for discovery, then resolves download URLs by
    scraping the video page for the direct CloudFront MP4 link, or falling
    back to ffmpeg HLS download from the API-provided stream URL.
    """

    def __init__(
        self,
        api_key: str,
        output_dir: Path,
        prefer_hd: bool = True,
        landscape_only: bool = True,
        min_duration: float = 3.0,
        max_duration: float = 120.0,
    ):
        self.api_key = api_key
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefer_hd = prefer_hd
        self.landscape_only = landscape_only
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between API requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def search(self, query: str, max_results: int = 20) -> list[dict]:
        """Search DVIDS for videos matching query.

        Returns list of dicts with id, title, duration, url, hls_url, hd, quality.
        """
        results: list[dict] = []
        pages_needed = (max_results + DVIDS_PAGE_LIMIT - 1) // DVIDS_PAGE_LIMIT
        collected = 0

        for page in range(1, pages_needed + 1):
            if collected >= max_results:
                break

            per_page = min(DVIDS_PAGE_LIMIT, max_results - collected)
            params = {
                "q": query,
                "type": "video",
                "max_results": per_page,
                "page": page,
                "api_key": self.api_key,
            }
            # Note: DVIDS API does not reliably support aspect_ratio filter,
            # so we skip it here. Landscape filtering is done post-search
            # based on the video page metadata if needed.

            self._rate_limit()
            try:
                resp = self.session.get(DVIDS_SEARCH_URL, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                print_warn(f"DVIDS search error (page {page}): {exc}")
                break

            search_results = data.get("results", [])
            if not search_results:
                break

            for item in search_results:
                if collected >= max_results:
                    break

                duration = item.get("duration", 0)
                if duration and (
                    duration < self.min_duration or duration > self.max_duration
                ):
                    continue

                is_hd = item.get("hd", False)
                quality = "hd" if is_hd else "sd"

                results.append({
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "duration": float(duration),
                    "url": item.get("url", ""),
                    "hls_url": item.get("hls_url", ""),
                    "hd": is_hd,
                    "quality": quality,
                    "description": item.get("short_description", ""),
                })
                collected += 1

            # Stop paging if we've seen all results
            page_info = data.get("page_info", {})
            total = page_info.get("total_results", 0)
            if page * DVIDS_PAGE_LIMIT >= total:
                break

        return results

    def _resolve_mp4_url(self, page_url: str) -> Optional[str]:
        """Scrape the DVIDS video page to find the direct CloudFront MP4 URL."""
        self._rate_limit()
        try:
            resp = self.session.get(page_url, timeout=20)
            resp.raise_for_status()
            # Look for <source> tag with .mp4 URL
            match = re.search(
                r'<source[^>]+src=["\']([^"\']+\.mp4)["\']', resp.text
            )
            if match:
                return match.group(1)
            # Fallback: find any cloudfront MP4 URL
            cf_match = re.search(
                r'(https://[^"\s]+cloudfront\.net[^"\s]+\.mp4)', resp.text
            )
            if cf_match:
                return cf_match.group(1)
        except requests.RequestException:
            pass
        return None

    def _download_via_hls(self, hls_url: str, output_path: Path) -> bool:
        """Download video from HLS stream using ffmpeg."""
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", hls_url,
                "-c", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        return result.returncode == 0 and output_path.exists()

    def download(self, video: dict) -> Optional[Path]:
        """Download a video. Returns path on success, None on failure.

        Strategy:
        1. Scrape the video page for a direct CloudFront MP4 URL
        2. Fallback: use ffmpeg to download from HLS stream URL
        """
        video_id = video["id"]
        numeric_id = video_id.replace("video:", "") if ":" in video_id else video_id
        filename = f"d{numeric_id}.mp4"
        output_path = self.output_dir / filename

        if output_path.exists() and output_path.stat().st_size > 0:
            print_info(f"  Already downloaded: {filename}")
            return output_path

        page_url = video.get("url", "")
        hls_url = video.get("hls_url", "")

        # Strategy 1: Direct MP4 from video page
        if page_url:
            mp4_url = self._resolve_mp4_url(page_url)
            if mp4_url:
                self._rate_limit()
                try:
                    resp = self.session.get(
                        mp4_url, timeout=DOWNLOAD_TIMEOUT, stream=True
                    )
                    resp.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    if output_path.exists() and output_path.stat().st_size > 0:
                        size_mb = output_path.stat().st_size / (1024 * 1024)
                        print_ok(f"  Downloaded: {filename} ({size_mb:.1f} MB)")
                        return output_path
                except requests.RequestException as exc:
                    print_warn(f"  Direct download failed ({video_id}): {exc}")
                    if output_path.exists():
                        output_path.unlink()

        # Strategy 2: HLS via ffmpeg
        if hls_url:
            print_info(f"  Trying HLS download for {video_id}...")
            if self._download_via_hls(hls_url, output_path):
                if output_path.exists() and output_path.stat().st_size > 0:
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    print_ok(f"  Downloaded (HLS): {filename} ({size_mb:.1f} MB)")
                    return output_path

        print_warn(f"  Could not download {video_id}")
        if output_path.exists():
            output_path.unlink()
        return None

    def cleanup(self):
        """Close the HTTP session."""
        self.session.close()


# ---------------------------------------------------------------------------
# Video segmentation
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available."""
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    return True


def probe_video(path: Path) -> dict:
    """Get video format info via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(path),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout).get("format", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def trim_to_clip(
    input_path: Path,
    output_dir: Path,
    max_mb: float = DEFAULT_MAX_SEGMENT_MB,
    target_dur: float = DEFAULT_TARGET_SEG_DUR,
) -> Optional[Path]:
    """Trim a video to a single short clip under max_mb.

    Takes the first target_dur seconds (or less to fit under max_mb).
    Returns the trimmed clip path, or None on failure.
    """
    file_size = input_path.stat().st_size
    max_bytes = max_mb * 1024 * 1024

    if file_size <= max_bytes:
        return input_path

    # Probe for duration and bitrate
    fmt = probe_video(input_path)
    duration = float(fmt.get("duration", 0))
    bitrate_bps = int(fmt.get("bit_rate", 0))

    if not duration or not bitrate_bps:
        if duration:
            bitrate_bps = int(file_size * 8 / duration)
        else:
            print_warn(f"  Cannot probe {input_path.name} - skipping trim")
            return input_path

    # Calculate clip duration that fits under size limit
    bytes_per_sec = bitrate_bps / 8
    max_dur_for_size = max_bytes / bytes_per_sec
    clip_dur = min(target_dur, max_dur_for_size * 0.9)
    clip_dur = max(clip_dur, 3.0)

    clip_path = output_dir / f"{input_path.stem}_clip.mp4"

    # Try copy-mode trim first (fast, lossless)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-t", f"{clip_dur:.1f}",
            "-c", "copy",
            "-movflags", "+faststart",
            str(clip_path),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )

    if result.returncode == 0 and clip_path.exists() and clip_path.stat().st_size <= max_bytes:
        input_path.unlink(missing_ok=True)
        size_mb = clip_path.stat().st_size / (1024 * 1024)
        print_ok(f"  Trimmed to {clip_dur:.0f}s clip ({size_mb:.1f} MB)")
        return clip_path

    # Copy-mode clip still too large (sparse keyframes) - re-encode
    if clip_path.exists():
        for _attempt in range(5):
            try:
                clip_path.unlink()
                break
            except PermissionError:
                import time
                time.sleep(0.5)

    audio_bitrate_bps = 128_000
    total_budget_bps = int(max_bytes * 8 / clip_dur)
    target_video_bps = int(total_budget_bps * 0.60 - audio_bitrate_bps)
    target_video_bps = max(target_video_bps, 500_000)
    target_video_kbps = target_video_bps // 1000

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(input_path),
            "-t", f"{clip_dur:.1f}",
            "-c:v", "libx264",
            "-b:v", f"{target_video_kbps}k",
            "-maxrate", f"{target_video_kbps}k",
            "-bufsize", f"{target_video_kbps}k",
            "-preset", "fast",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(clip_path),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )

    if result.returncode != 0 or not clip_path.exists():
        print_warn(f"  Trim failed for {input_path.name}")
        if clip_path.exists():
            clip_path.unlink()
        return input_path

    input_path.unlink(missing_ok=True)
    size_mb = clip_path.stat().st_size / (1024 * 1024)
    print_ok(f"  Trimmed + re-encoded to {clip_dur:.0f}s clip ({size_mb:.1f} MB)")
    return clip_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download DVIDS Hub videos matching a project's voiceover SRT topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "project",
        help="Project directory path",
    )
    parser.add_argument(
        "--srt",
        help="Explicit SRT file path (default: {project}/voiceover.srt)",
    )
    parser.add_argument(
        "--query",
        help="Explicit search query (bypasses SRT topic extraction)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (default: {project}/dvids/)",
    )
    parser.add_argument(
        "--max-segment-mb",
        type=float,
        default=DEFAULT_MAX_SEGMENT_MB,
        help=f"Maximum clip file size in MB (default: {DEFAULT_MAX_SEGMENT_MB})",
    )
    parser.add_argument(
        "--target-seg-dur",
        type=float,
        default=DEFAULT_TARGET_SEG_DUR,
        help=f"Target clip duration in seconds (default: {DEFAULT_TARGET_SEG_DUR})",
    )
    parser.add_argument(
        "--min-clips",
        type=int,
        default=DEFAULT_MIN_CLIPS,
        help=f"Minimum number of clips to download (default: {DEFAULT_MIN_CLIPS})",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"Max DVIDS search results per query (default: {DEFAULT_MAX_RESULTS})",
    )
    parser.add_argument(
        "--no-prefer-hd",
        action="store_true",
        help="Do not prefer HD renditions (use provider file order)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be downloaded without downloading",
    )
    parser.add_argument(
        "--api-key",
        help="Explicit DVIDS API key (overrides env)",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve project directory
    project_dir = Path(args.project).resolve()
    if not project_dir.is_dir():
        print_error(f"Project directory not found: {project_dir}")
        sys.exit(1)

    # Resolve API key
    api_key = resolve_api_key(args.api_key)
    if not api_key:
        print_error(
            "DVIDS API key not found. Set DVIDS_API_KEY in .env, "
            "environment, or pass --api-key"
        )
        sys.exit(1)

    # Resolve SRT file
    if args.srt:
        srt_path = Path(args.srt).resolve()
    else:
        srt_path = project_dir / "voiceover.srt"

    # Resolve output directory
    output_dir = Path(args.output_dir).resolve() if args.output_dir else project_dir / "dvids"

    # Determine search queries
    if args.query:
        queries = [args.query]
        print_info(f"Using explicit query: {args.query}")
    else:
        if not srt_path.is_file():
            print_error(
                f"SRT file not found: {srt_path}\n"
                "  Use --srt to specify a different path, or --query for manual search"
            )
            sys.exit(1)
        print_info(f"Parsing SRT: {srt_path}")
        segments = parse_srt(srt_path)
        if not segments:
            print_error("No segments found in SRT file")
            sys.exit(1)
        print_info(f"  Found {len(segments)} SRT segments")
        queries = extract_topics(segments)
        if not queries:
            print_error("Could not extract topics from SRT. Use --query as fallback")
            sys.exit(1)

    print_header("SEARCH QUERIES")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    # Check ffmpeg availability
    if not args.dry_run and not check_ffmpeg():
        print_error(
            "ffmpeg/ffprobe not found in PATH.\n"
            "  Install from https://ffmpeg.org/download.html and ensure it's in PATH."
        )
        sys.exit(1)

    # Initialize downloader
    dvids = DvidsDownloader(
        api_key=api_key,
        output_dir=output_dir,
        prefer_hd=not args.no_prefer_hd,
    )

    # Main download loop
    total_duration = 0.0
    downloaded_ids: set[str] = set()
    clips: list[Path] = []
    videos_downloaded = 0

    try:
        for qi, query in enumerate(queries, 1):
            if len(downloaded_ids) >= args.min_clips:
                break

            print_header(f"QUERY {qi}/{len(queries)}: {query}")
            results = dvids.search(query, max_results=args.max_results)

            if not results:
                print_warn("No results found")
                continue

            print_ok(f"Found {len(results)} videos")

            for video in results:
                if len(downloaded_ids) >= args.min_clips:
                    break
                if video["id"] in downloaded_ids:
                    continue

                vid_dur = video["duration"]
                vid_quality = video["quality"]
                vid_title = video.get("title", "")[:60]
                clip_dur = min(args.target_seg_dur, vid_dur)

                if args.dry_run:
                    print(f"  [{vid_quality.upper()}] {vid_title} ({vid_dur:.0f}s -> {clip_dur:.0f}s clip)")
                    downloaded_ids.add(video["id"])
                    total_duration += clip_dur
                    continue

                print(f"  Downloading: {vid_title} ({vid_dur:.0f}s, {vid_quality})")
                path = dvids.download(video)
                if not path:
                    continue

                downloaded_ids.add(video["id"])
                videos_downloaded += 1

                clip = trim_to_clip(
                    path, output_dir, args.max_segment_mb, args.target_seg_dur,
                )
                if clip:
                    clips.append(clip)
                    total_duration += clip_dur

    finally:
        dvids.cleanup()

    # Summary
    print_header("SUMMARY")
    if args.dry_run:
        print(f"  Videos found:   {len(downloaded_ids)}")
        print(f"  Total duration: {total_duration:.1f}s")
        print(f"  Output:         {output_dir}")
    else:
        total_size_mb = sum(f.stat().st_size for f in clips if f.exists()) / (1024 * 1024)
        print(f"  Clips:          {len(clips)} (from {videos_downloaded} videos)")
        print(f"  Total duration: {total_duration:.1f}s")
        print(f"  Total size:     {total_size_mb:.1f} MB")
        print(f"  Output:         {output_dir}")

    if len(downloaded_ids) < args.min_clips:
        print_warn(
            f"  Only found {len(downloaded_ids)} clips (target: {args.min_clips}). "
            "Try broader search terms with --query."
        )

    if not args.dry_run and clips:
        print_ok("Done!")


if __name__ == "__main__":
    main()
