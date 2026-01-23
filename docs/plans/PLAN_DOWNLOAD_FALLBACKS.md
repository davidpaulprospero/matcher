# Plan: Download Fallback System for Rate Limit Resilience

> **Version:** 2.0 | **Updated:** 2026-01-22 | **Status:** Draft

## Problem Statement

yt-dlp frequently hits YouTube rate limits (HTTP 429), causing pipeline failures. We need a tiered fallback system that automatically switches to alternative download methods when the primary method fails.

## Goals

1. Reduce pipeline failures from rate limits by 90%+
2. Maintain caption-first mode compatibility
3. Keep existing yt-dlp as primary (best quality/features)
4. Add fallbacks that require minimal configuration
5. Support both video downloads and caption extraction
6. Provide ASR fallback when no captions exist

---

## Architecture Overview

### Caption Extraction Chain (6 Tiers)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CaptionFallbackChain                             │
├─────────────────────────────────────────────────────────────────────┤
│  Tier 1: yt-dlp --write-subs (primary)                              │
│    ↓ 429/rate limit                                                 │
│  Tier 2: youtube-transcript-api (different endpoint, rarely limited)│
│    ↓ TranscriptsDisabled                                            │
│  Tier 3: Direct Innertube/timedtext URL fetch                       │
│    ↓ blocked                                                        │
│  Tier 4: Invidious API (rotate 10+ instances)                       │
│    ↓ all instances down                                             │
│  Tier 5: Piped API (rotate instances)                               │
│    ↓ no captions available on YouTube                               │
│  Tier 6: Local Whisper ASR (download audio + transcribe)            │
└─────────────────────────────────────────────────────────────────────┘
```

### Video Download Chain (4 Tiers)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VideoFallbackChain                               │
├─────────────────────────────────────────────────────────────────────┤
│  Tier 1: yt-dlp (primary)                                           │
│    ↓ 429/rate limit                                                 │
│  Tier 2: Invidious adaptiveFormats stream URLs                      │
│    ↓ blocked                                                        │
│  Tier 3: Piped videoStreams URLs                                    │
│    ↓ all blocked                                                    │
│  Tier 4: Cobalt API (self-hosted)                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Caption Fallbacks (HIGH PRIORITY)

#### 1A: youtube-transcript-api Integration

**Why it works when yt-dlp fails:** Uses a different YouTube endpoint (`/api/timedtext`) that has separate rate limiting from the video player API.

**Files to create/modify:**
- `src/downloader/caption_fallback.py` (new)
- `src/stages/transcribe.py` (integrate)

**Dependencies:**
```bash
pip install youtube-transcript-api
```

**Implementation:**

```python
# src/downloader/caption_fallback.py
"""
Multi-tier caption fallback system.

Extraction priority:
1. yt-dlp (existing)
2. youtube-transcript-api (different endpoint)
3. Direct Innertube timedtext fetch
4. Invidious API
5. Piped API
6. Whisper ASR (last resort)
"""

import re
import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class CaptionTier(Enum):
    """Caption extraction tiers in priority order."""
    YTDLP = auto()
    TRANSCRIPT_API = auto()
    INNERTUBE_DIRECT = auto()
    INVIDIOUS = auto()
    PIPED = auto()
    WHISPER_ASR = auto()


@dataclass
class CaptionSegment:
    """Single caption segment with timing."""
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class CaptionResult:
    """Result of caption extraction attempt."""
    success: bool
    tier_used: CaptionTier
    segments: list[CaptionSegment] = field(default_factory=list)
    language: str = "en"
    is_auto_generated: bool = False
    error: Optional[str] = None


class TranscriptAPIFetcher:
    """Tier 2: youtube-transcript-api wrapper."""

    def __init__(self, preferred_languages: list[str] = None):
        self.preferred_languages = preferred_languages or ['en', 'en-US', 'en-GB']
        self._api = None

    @property
    def api(self):
        if self._api is None:
            from youtube_transcript_api import YouTubeTranscriptApi
            self._api = YouTubeTranscriptApi
        return self._api

    def fetch(self, video_id: str) -> CaptionResult:
        """Fetch transcript using youtube-transcript-api."""
        try:
            from youtube_transcript_api._errors import (
                TranscriptsDisabled,
                NoTranscriptFound,
                VideoUnavailable
            )

            transcript_list = self.api.list_transcripts(video_id)

            # Strategy 1: Manual transcript in preferred language
            for lang in self.preferred_languages:
                try:
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    data = transcript.fetch()
                    return CaptionResult(
                        success=True,
                        tier_used=CaptionTier.TRANSCRIPT_API,
                        segments=[CaptionSegment(s['text'], s['start'], s['duration']) for s in data],
                        language=lang,
                        is_auto_generated=False
                    )
                except NoTranscriptFound:
                    continue

            # Strategy 2: Auto-generated in preferred language
            for lang in self.preferred_languages:
                try:
                    transcript = transcript_list.find_generated_transcript([lang])
                    data = transcript.fetch()
                    return CaptionResult(
                        success=True,
                        tier_used=CaptionTier.TRANSCRIPT_API,
                        segments=[CaptionSegment(s['text'], s['start'], s['duration']) for s in data],
                        language=lang,
                        is_auto_generated=True
                    )
                except NoTranscriptFound:
                    continue

            # Strategy 3: Any transcript, translated to English
            try:
                available = list(transcript_list)
                if available:
                    transcript = available[0]
                    if transcript.language_code not in self.preferred_languages:
                        transcript = transcript.translate('en')
                    data = transcript.fetch()
                    return CaptionResult(
                        success=True,
                        tier_used=CaptionTier.TRANSCRIPT_API,
                        segments=[CaptionSegment(s['text'], s['start'], s['duration']) for s in data],
                        language='en',
                        is_auto_generated=transcript.is_generated
                    )
            except Exception:
                pass

            return CaptionResult(
                success=False,
                tier_used=CaptionTier.TRANSCRIPT_API,
                error="No transcript found in any language"
            )

        except TranscriptsDisabled:
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.TRANSCRIPT_API,
                error="Transcripts disabled for this video"
            )
        except VideoUnavailable:
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.TRANSCRIPT_API,
                error="Video unavailable"
            )
        except Exception as e:
            logger.warning(f"transcript-api failed for {video_id}: {e}")
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.TRANSCRIPT_API,
                error=str(e)
            )


class InnertubeDirectFetcher:
    """Tier 3: Direct Innertube/timedtext URL extraction."""

    TIMEDTEXT_FORMATS = ['srv3', 'json3', 'vtt']

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout, follow_redirects=True)

    def fetch(self, video_id: str) -> CaptionResult:
        """Extract captions directly from YouTube page."""
        try:
            # Fetch video page
            url = f"https://www.youtube.com/watch?v={video_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            response = self.client.get(url, headers=headers)

            if response.status_code != 200:
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.INNERTUBE_DIRECT,
                    error=f"HTTP {response.status_code}"
                )

            # Extract captionTracks from player_response
            caption_url = self._extract_caption_url(response.text)
            if not caption_url:
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.INNERTUBE_DIRECT,
                    error="No caption tracks found in player_response"
                )

            # Fetch caption content (prefer srv3 XML format)
            for fmt in self.TIMEDTEXT_FORMATS:
                fetch_url = self._add_format_param(caption_url, fmt)
                cap_response = self.client.get(fetch_url)

                if cap_response.status_code == 200:
                    segments = self._parse_captions(cap_response.text, fmt)
                    if segments:
                        return CaptionResult(
                            success=True,
                            tier_used=CaptionTier.INNERTUBE_DIRECT,
                            segments=segments,
                            language='en',
                            is_auto_generated='asr' in caption_url.lower()
                        )

            return CaptionResult(
                success=False,
                tier_used=CaptionTier.INNERTUBE_DIRECT,
                error="Failed to fetch/parse caption content"
            )

        except Exception as e:
            logger.warning(f"Innertube direct failed for {video_id}: {e}")
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.INNERTUBE_DIRECT,
                error=str(e)
            )

    def _extract_caption_url(self, page_html: str) -> Optional[str]:
        """Extract English caption URL from player_response."""
        # Pattern 1: captionTracks in JSON
        match = re.search(r'"captionTracks":\s*(\[.*?\])', page_html)
        if match:
            try:
                tracks = json.loads(match.group(1))
                # Prefer English, then any
                for track in tracks:
                    lang = track.get('languageCode', '')
                    if lang.startswith('en'):
                        return track.get('baseUrl')
                # Fallback to first track
                if tracks:
                    return tracks[0].get('baseUrl')
            except json.JSONDecodeError:
                pass

        # Pattern 2: Direct timedtext URL
        match = re.search(r'"(https://www\.youtube\.com/api/timedtext[^"]+)"', page_html)
        if match:
            return match.group(1).replace('\\u0026', '&')

        return None

    def _add_format_param(self, url: str, fmt: str) -> str:
        """Add or replace format parameter in URL."""
        if 'fmt=' in url:
            return re.sub(r'fmt=[^&]+', f'fmt={fmt}', url)
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}fmt={fmt}"

    def _parse_captions(self, content: str, fmt: str) -> list[CaptionSegment]:
        """Parse caption content based on format."""
        segments = []

        if fmt == 'srv3':
            # XML format: <text start="0.0" dur="2.5">Caption text</text>
            for match in re.finditer(r'<text[^>]*start="([^"]+)"[^>]*dur="([^"]+)"[^>]*>([^<]*)</text>', content):
                start = float(match.group(1))
                dur = float(match.group(2))
                text = self._decode_html_entities(match.group(3))
                segments.append(CaptionSegment(text, start, dur))

        elif fmt == 'json3':
            # JSON format
            try:
                data = json.loads(content)
                for event in data.get('events', []):
                    if 'segs' in event:
                        text = ''.join(seg.get('utf8', '') for seg in event['segs'])
                        start = event.get('tStartMs', 0) / 1000
                        dur = event.get('dDurationMs', 0) / 1000
                        if text.strip():
                            segments.append(CaptionSegment(text, start, dur))
            except json.JSONDecodeError:
                pass

        elif fmt == 'vtt':
            # WebVTT format
            for match in re.finditer(
                r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n([^\n]+)',
                content
            ):
                start = self._vtt_to_seconds(match.group(1))
                end = self._vtt_to_seconds(match.group(2))
                text = match.group(3).strip()
                segments.append(CaptionSegment(text, start, end - start))

        return segments

    @staticmethod
    def _decode_html_entities(text: str) -> str:
        """Decode HTML entities in caption text."""
        import html
        return html.unescape(text)

    @staticmethod
    def _vtt_to_seconds(timestamp: str) -> float:
        """Convert VTT timestamp to seconds."""
        parts = timestamp.replace(',', '.').split(':')
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(parts[0])


class InvidiousCaptionFetcher:
    """Tier 4: Invidious API caption fetching with instance rotation."""

    # Updated instance list - check https://api.invidious.io/instances.json for current
    DEFAULT_INSTANCES = [
        "https://invidious.snopyta.org",
        "https://yewtu.be",
        "https://invidious.kavin.rocks",
        "https://vid.puffyan.us",
        "https://invidious.namazso.eu",
        "https://invidious.nerdvpn.de",
        "https://inv.riverside.rocks",
        "https://invidious.flokinet.to",
        "https://invidious.esmailelbob.xyz",
        "https://invidious.projectsegfau.lt",
    ]

    def __init__(self, instances: list[str] = None, timeout: float = 30.0):
        self.instances = instances or self.DEFAULT_INSTANCES.copy()
        self.timeout = timeout
        self.failed_instances: dict[str, float] = {}  # instance -> failure timestamp
        self.current_index = 0
        self.cooldown_seconds = 300  # 5 minutes

    def fetch(self, video_id: str) -> CaptionResult:
        """Fetch captions via Invidious API with instance rotation."""
        import time

        for _ in range(len(self.instances)):
            instance = self._get_next_instance()
            if not instance:
                break

            try:
                url = f"{instance}/api/v1/videos/{video_id}"
                response = httpx.get(url, timeout=self.timeout)

                if response.status_code == 429:
                    self.failed_instances[instance] = time.time()
                    continue

                if response.status_code != 200:
                    continue

                data = response.json()
                captions = data.get('captions', [])

                # Find English caption
                for cap in captions:
                    lang = cap.get('language_code', '') or cap.get('languageCode', '')
                    if lang.startswith('en'):
                        cap_url = cap.get('url')
                        if cap_url:
                            # URL might be relative
                            if cap_url.startswith('/'):
                                cap_url = f"{instance}{cap_url}"

                            cap_response = httpx.get(cap_url, timeout=self.timeout)
                            if cap_response.status_code == 200:
                                segments = self._parse_invidious_captions(cap_response.text)
                                if segments:
                                    return CaptionResult(
                                        success=True,
                                        tier_used=CaptionTier.INVIDIOUS,
                                        segments=segments,
                                        language=lang
                                    )

            except httpx.TimeoutException:
                self.failed_instances[instance] = time.time()
                continue
            except Exception as e:
                logger.debug(f"Invidious {instance} failed: {e}")
                continue

        return CaptionResult(
            success=False,
            tier_used=CaptionTier.INVIDIOUS,
            error="All Invidious instances failed"
        )

    def _get_next_instance(self) -> Optional[str]:
        """Get next healthy instance."""
        import time
        now = time.time()

        for _ in range(len(self.instances)):
            instance = self.instances[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.instances)

            # Check if instance is on cooldown
            if instance in self.failed_instances:
                if now - self.failed_instances[instance] < self.cooldown_seconds:
                    continue
                else:
                    del self.failed_instances[instance]

            return instance

        return None

    def _parse_invidious_captions(self, content: str) -> list[CaptionSegment]:
        """Parse caption content from Invidious."""
        segments = []

        # Try XML format first
        for match in re.finditer(r'<text[^>]*start="([^"]+)"[^>]*dur="([^"]+)"[^>]*>([^<]*)</text>', content):
            start = float(match.group(1))
            dur = float(match.group(2))
            text = match.group(3)
            segments.append(CaptionSegment(text, start, dur))

        if segments:
            return segments

        # Try VTT format
        for match in re.finditer(
            r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n([^\n]+)',
            content
        ):
            start = InnertubeDirectFetcher._vtt_to_seconds(match.group(1))
            end = InnertubeDirectFetcher._vtt_to_seconds(match.group(2))
            text = match.group(3).strip()
            segments.append(CaptionSegment(text, start, end - start))

        return segments


class PipedCaptionFetcher:
    """Tier 5: Piped API caption fetching."""

    DEFAULT_INSTANCES = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.tokhmi.xyz",
        "https://pipedapi.moomoo.me",
        "https://pipedapi.syncpundit.io",
        "https://api.piped.yt",
        "https://pipedapi.in.projectsegfau.lt",
    ]

    def __init__(self, instances: list[str] = None, timeout: float = 30.0):
        self.instances = instances or self.DEFAULT_INSTANCES.copy()
        self.timeout = timeout
        self.current_index = 0

    def fetch(self, video_id: str) -> CaptionResult:
        """Fetch captions via Piped API."""
        for instance in self._rotate_instances():
            try:
                url = f"{instance}/streams/{video_id}"
                response = httpx.get(url, timeout=self.timeout)

                if response.status_code != 200:
                    continue

                data = response.json()
                subtitles = data.get('subtitles', [])

                # Find English subtitle
                for sub in subtitles:
                    code = sub.get('code', '')
                    if code.startswith('en'):
                        sub_url = sub.get('url')
                        if sub_url:
                            sub_response = httpx.get(sub_url, timeout=self.timeout)
                            if sub_response.status_code == 200:
                                segments = self._parse_vtt(sub_response.text)
                                if segments:
                                    return CaptionResult(
                                        success=True,
                                        tier_used=CaptionTier.PIPED,
                                        segments=segments,
                                        language=code,
                                        is_auto_generated=sub.get('autoGenerated', False)
                                    )

            except Exception as e:
                logger.debug(f"Piped {instance} failed: {e}")
                continue

        return CaptionResult(
            success=False,
            tier_used=CaptionTier.PIPED,
            error="All Piped instances failed"
        )

    def _rotate_instances(self):
        """Yield instances in rotating order."""
        for i in range(len(self.instances)):
            idx = (self.current_index + i) % len(self.instances)
            yield self.instances[idx]
        self.current_index = (self.current_index + 1) % len(self.instances)

    def _parse_vtt(self, content: str) -> list[CaptionSegment]:
        """Parse VTT subtitle content."""
        segments = []
        for match in re.finditer(
            r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\s*\n([^\n]+)',
            content
        ):
            start = InnertubeDirectFetcher._vtt_to_seconds(match.group(1))
            end = InnertubeDirectFetcher._vtt_to_seconds(match.group(2))
            text = match.group(3).strip()
            # Remove VTT tags like <c> </c>
            text = re.sub(r'<[^>]+>', '', text)
            if text:
                segments.append(CaptionSegment(text, start, end - start))
        return segments


class CaptionFallbackChain:
    """
    Orchestrates multi-tier caption extraction with automatic fallback.

    Usage:
        chain = CaptionFallbackChain(config)
        result = chain.fetch(video_id)
        if result.success:
            print(f"Got {len(result.segments)} segments via {result.tier_used.name}")
    """

    def __init__(self, config=None, enable_whisper_fallback: bool = True):
        self.config = config
        self.enable_whisper_fallback = enable_whisper_fallback

        # Initialize fetchers
        languages = ['en', 'en-US', 'en-GB']
        if config and hasattr(config, 'download'):
            languages = getattr(config.download, 'caption_languages', languages)

        self.transcript_api = TranscriptAPIFetcher(languages)
        self.innertube_direct = InnertubeDirectFetcher()
        self.invidious = InvidiousCaptionFetcher()
        self.piped = PipedCaptionFetcher()

        # Adaptive behavior tracking
        self.tier_stats: dict[CaptionTier, dict] = {
            tier: {'success': 0, 'failure': 0} for tier in CaptionTier
        }
        self.consecutive_ytdlp_failures = 0

    def fetch(self, video_id: str, skip_ytdlp: bool = False) -> CaptionResult:
        """
        Fetch captions with automatic fallback through all tiers.

        Args:
            video_id: YouTube video ID (11 characters)
            skip_ytdlp: Skip Tier 1 (yt-dlp), useful when already rate-limited

        Returns:
            CaptionResult with segments if successful
        """
        # Tier 1: yt-dlp (handled externally, caller indicates if we should skip)
        # This method handles Tiers 2-6

        fetchers = [
            (CaptionTier.TRANSCRIPT_API, self.transcript_api),
            (CaptionTier.INNERTUBE_DIRECT, self.innertube_direct),
            (CaptionTier.INVIDIOUS, self.invidious),
            (CaptionTier.PIPED, self.piped),
        ]

        last_error = None
        for tier, fetcher in fetchers:
            result = fetcher.fetch(video_id)

            if result.success:
                self.tier_stats[tier]['success'] += 1
                logger.info(f"Caption fetch succeeded via {tier.name} for {video_id}")
                return result

            self.tier_stats[tier]['failure'] += 1
            last_error = result.error
            logger.debug(f"Caption fetch failed via {tier.name}: {result.error}")

        # Tier 6: Whisper ASR (if enabled and all else failed)
        if self.enable_whisper_fallback:
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.WHISPER_ASR,
                error=f"All caption sources failed. Last error: {last_error}. Consider Whisper ASR fallback."
            )

        return CaptionResult(
            success=False,
            tier_used=CaptionTier.PIPED,
            error=f"All caption sources failed. Last error: {last_error}"
        )

    def get_stats(self) -> dict:
        """Get success/failure statistics for each tier."""
        return {
            tier.name: stats for tier, stats in self.tier_stats.items()
        }

    def to_srt(self, segments: list[CaptionSegment]) -> str:
        """Convert caption segments to SRT format."""
        lines = []
        for i, seg in enumerate(segments, 1):
            start_ts = self._seconds_to_srt(seg.start)
            end_ts = self._seconds_to_srt(seg.end)
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    def to_vtt(self, segments: list[CaptionSegment]) -> str:
        """Convert caption segments to WebVTT format."""
        lines = ["WEBVTT", ""]
        for seg in segments:
            start_ts = self._seconds_to_vtt(seg.start)
            end_ts = self._seconds_to_vtt(seg.end)
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _seconds_to_srt(seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _seconds_to_vtt(seconds: float) -> str:
        """Convert seconds to VTT timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
```

---

#### 1B: Integration with Existing Caption-First Mode

**Files to modify:**
- `src/stages/transcribe.py`
- `src/downloader/core.py`

```python
# In src/stages/transcribe.py - add fallback integration

from src.downloader.caption_fallback import CaptionFallbackChain, CaptionResult

class TranscribeStage:
    def __init__(self, config):
        self.config = config
        self.caption_chain = CaptionFallbackChain(config)
        self._ytdlp_rate_limited = False

    async def get_captions(self, video_id: str) -> CaptionResult:
        """Get captions with automatic fallback."""

        # Try yt-dlp first (existing code) unless rate-limited
        if not self._ytdlp_rate_limited:
            result = await self._try_ytdlp_captions(video_id)
            if result.success:
                return result

            # Check if rate limited
            if "429" in (result.error or "") or "rate" in (result.error or "").lower():
                self._ytdlp_rate_limited = True
                logger.warning("yt-dlp rate limited, switching to fallback chain")

        # Use fallback chain (Tiers 2-6)
        return self.caption_chain.fetch(video_id, skip_ytdlp=True)
```

---

### Phase 2: Video Stream Fallbacks

**Files to create:**
- `src/downloader/video_fallback.py` (new)

```python
# src/downloader/video_fallback.py
"""
Video download fallback system using alternative stream sources.
"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class VideoTier(Enum):
    YTDLP = auto()
    INVIDIOUS = auto()
    PIPED = auto()
    COBALT = auto()


@dataclass
class StreamInfo:
    """Video stream metadata."""
    url: str
    quality: str
    format: str
    filesize: Optional[int] = None
    has_audio: bool = True


@dataclass
class VideoResult:
    """Result of video download attempt."""
    success: bool
    tier_used: VideoTier
    file_path: Optional[Path] = None
    stream_info: Optional[StreamInfo] = None
    error: Optional[str] = None


class InvidiousStreamFetcher:
    """Fetch video streams from Invidious API."""

    DEFAULT_INSTANCES = [
        "https://invidious.snopyta.org",
        "https://yewtu.be",
        "https://invidious.kavin.rocks",
        "https://vid.puffyan.us",
        "https://invidious.namazso.eu",
    ]

    def __init__(self, instances: list[str] = None, timeout: float = 30.0):
        self.instances = instances or self.DEFAULT_INSTANCES.copy()
        self.timeout = timeout
        self.current_index = 0

    def get_streams(self, video_id: str, preferred_quality: str = "720p") -> list[StreamInfo]:
        """Get available video streams."""
        for instance in self._rotate_instances():
            try:
                url = f"{instance}/api/v1/videos/{video_id}"
                response = httpx.get(url, timeout=self.timeout)

                if response.status_code != 200:
                    continue

                data = response.json()
                streams = []

                # Parse adaptive formats (video-only and audio-only)
                for fmt in data.get('adaptiveFormats', []):
                    streams.append(StreamInfo(
                        url=fmt.get('url', ''),
                        quality=fmt.get('qualityLabel', 'unknown'),
                        format=fmt.get('container', 'mp4'),
                        filesize=fmt.get('contentLength'),
                        has_audio=fmt.get('type', '').startswith('audio/')
                    ))

                # Parse format streams (combined video+audio)
                for fmt in data.get('formatStreams', []):
                    streams.append(StreamInfo(
                        url=fmt.get('url', ''),
                        quality=fmt.get('qualityLabel', 'unknown'),
                        format=fmt.get('container', 'mp4'),
                        has_audio=True
                    ))

                if streams:
                    # Sort by quality preference
                    return self._sort_by_quality(streams, preferred_quality)

            except Exception as e:
                logger.debug(f"Invidious {instance} stream fetch failed: {e}")
                continue

        return []

    def _rotate_instances(self):
        for i in range(len(self.instances)):
            idx = (self.current_index + i) % len(self.instances)
            yield self.instances[idx]
        self.current_index = (self.current_index + 1) % len(self.instances)

    def _sort_by_quality(self, streams: list[StreamInfo], preferred: str) -> list[StreamInfo]:
        """Sort streams with preferred quality first."""
        def quality_key(s: StreamInfo) -> int:
            if preferred in s.quality:
                return 0
            # Extract numeric quality
            import re
            match = re.search(r'(\d+)p', s.quality)
            if match:
                return abs(int(match.group(1)) - int(preferred.replace('p', '')))
            return 9999

        return sorted(streams, key=quality_key)


class PipedStreamFetcher:
    """Fetch video streams from Piped API."""

    DEFAULT_INSTANCES = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.tokhmi.xyz",
        "https://pipedapi.moomoo.me",
    ]

    def __init__(self, instances: list[str] = None, timeout: float = 30.0):
        self.instances = instances or self.DEFAULT_INSTANCES.copy()
        self.timeout = timeout
        self.current_index = 0

    def get_streams(self, video_id: str, preferred_quality: str = "720p") -> list[StreamInfo]:
        """Get available video streams."""
        for instance in self._rotate_instances():
            try:
                url = f"{instance}/streams/{video_id}"
                response = httpx.get(url, timeout=self.timeout)

                if response.status_code != 200:
                    continue

                data = response.json()
                streams = []

                for vs in data.get('videoStreams', []):
                    streams.append(StreamInfo(
                        url=vs.get('url', ''),
                        quality=vs.get('quality', 'unknown'),
                        format=vs.get('format', 'mp4'),
                        filesize=vs.get('contentLength'),
                        has_audio=vs.get('videoOnly', True) == False
                    ))

                if streams:
                    return streams

            except Exception as e:
                logger.debug(f"Piped {instance} stream fetch failed: {e}")
                continue

        return []

    def _rotate_instances(self):
        for i in range(len(self.instances)):
            idx = (self.current_index + i) % len(self.instances)
            yield self.instances[idx]
        self.current_index = (self.current_index + 1) % len(self.instances)


class VideoFallbackChain:
    """
    Orchestrates video download with automatic fallback.

    Note: yt-dlp (Tier 1) is handled externally. This class handles Tiers 2-4.
    """

    def __init__(self, config=None):
        self.config = config
        self.invidious = InvidiousStreamFetcher()
        self.piped = PipedStreamFetcher()

    def download(
        self,
        video_id: str,
        output_dir: Path,
        preferred_quality: str = "720p"
    ) -> VideoResult:
        """Download video with automatic fallback."""

        # Tier 2: Invidious
        streams = self.invidious.get_streams(video_id, preferred_quality)
        if streams:
            result = self._download_stream(streams[0], video_id, output_dir)
            if result.success:
                result.tier_used = VideoTier.INVIDIOUS
                return result

        # Tier 3: Piped
        streams = self.piped.get_streams(video_id, preferred_quality)
        if streams:
            result = self._download_stream(streams[0], video_id, output_dir)
            if result.success:
                result.tier_used = VideoTier.PIPED
                return result

        return VideoResult(
            success=False,
            tier_used=VideoTier.PIPED,
            error="All video stream sources failed"
        )

    def _download_stream(
        self,
        stream: StreamInfo,
        video_id: str,
        output_dir: Path
    ) -> VideoResult:
        """Download video from stream URL."""
        try:
            output_path = output_dir / f"{video_id}.{stream.format}"

            with httpx.stream("GET", stream.url, timeout=300.0, follow_redirects=True) as response:
                if response.status_code != 200:
                    return VideoResult(
                        success=False,
                        tier_used=VideoTier.INVIDIOUS,
                        error=f"HTTP {response.status_code}"
                    )

                with open(output_path, 'wb') as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)

            return VideoResult(
                success=True,
                tier_used=VideoTier.INVIDIOUS,
                file_path=output_path,
                stream_info=stream
            )

        except Exception as e:
            logger.error(f"Stream download failed: {e}")
            return VideoResult(
                success=False,
                tier_used=VideoTier.INVIDIOUS,
                error=str(e)
            )
```

---

### Phase 3: Configuration Schema

**Files to modify:**
- `src/config/sections/download.py`
- `config.yaml`

```yaml
# config.yaml additions
download:
  fallback:
    enabled: true

    # Caption extraction fallbacks
    caption:
      # Tier 2: youtube-transcript-api
      transcript_api:
        enabled: true
        preferred_languages: ["en", "en-US", "en-GB", "en-AU"]

      # Tier 3: Direct Innertube/timedtext
      innertube_direct:
        enabled: true
        timeout: 30

      # Tier 4: Invidious API
      invidious:
        enabled: true
        instances: []  # Empty = use defaults
        timeout: 30
        cooldown_seconds: 300

      # Tier 5: Piped API
      piped:
        enabled: true
        instances: []  # Empty = use defaults
        timeout: 30

      # Tier 6: Whisper ASR fallback
      whisper_fallback:
        enabled: true
        # Uses existing Whisper config

    # Video stream fallbacks
    video:
      # Tier 2: Invidious streams
      invidious:
        enabled: true
        instances: []
        preferred_quality: "720p"
        timeout: 30

      # Tier 3: Piped streams
      piped:
        enabled: true
        instances: []
        timeout: 30

      # Tier 4: Cobalt (requires self-hosting)
      cobalt:
        enabled: false
        url: ""  # e.g., "http://localhost:9000"

    # Adaptive behavior
    behavior:
      skip_ytdlp_after_consecutive_failures: 3
      ytdlp_cooldown_seconds: 600  # 10 minutes
      log_tier_stats: true
```

```python
# src/config/sections/download.py additions

@dataclass
class CaptionFallbackConfig:
    transcript_api: dict = field(default_factory=lambda: {
        'enabled': True,
        'preferred_languages': ['en', 'en-US', 'en-GB']
    })
    innertube_direct: dict = field(default_factory=lambda: {
        'enabled': True,
        'timeout': 30
    })
    invidious: dict = field(default_factory=lambda: {
        'enabled': True,
        'instances': [],
        'timeout': 30,
        'cooldown_seconds': 300
    })
    piped: dict = field(default_factory=lambda: {
        'enabled': True,
        'instances': [],
        'timeout': 30
    })
    whisper_fallback: dict = field(default_factory=lambda: {
        'enabled': True
    })


@dataclass
class VideoFallbackConfig:
    invidious: dict = field(default_factory=lambda: {
        'enabled': True,
        'instances': [],
        'preferred_quality': '720p',
        'timeout': 30
    })
    piped: dict = field(default_factory=lambda: {
        'enabled': True,
        'instances': [],
        'timeout': 30
    })
    cobalt: dict = field(default_factory=lambda: {
        'enabled': False,
        'url': ''
    })


@dataclass
class FallbackBehaviorConfig:
    skip_ytdlp_after_consecutive_failures: int = 3
    ytdlp_cooldown_seconds: int = 600
    log_tier_stats: bool = True


@dataclass
class FallbackConfig:
    enabled: bool = True
    caption: CaptionFallbackConfig = field(default_factory=CaptionFallbackConfig)
    video: VideoFallbackConfig = field(default_factory=VideoFallbackConfig)
    behavior: FallbackBehaviorConfig = field(default_factory=FallbackBehaviorConfig)
```

---

### Phase 4: Testing

**Files to create:**
- `tests/test_caption_fallback.py`
- `tests/test_video_fallback.py`

```python
# tests/test_caption_fallback.py
"""Tests for caption fallback system."""

import pytest
from src.downloader.caption_fallback import (
    CaptionFallbackChain,
    TranscriptAPIFetcher,
    InnertubeDirectFetcher,
    InvidiousCaptionFetcher,
    PipedCaptionFetcher,
    CaptionTier,
    CaptionSegment,
)

# Known video with captions
TEST_VIDEO_ID = "dQw4w9WgXcQ"


class TestTranscriptAPIFetcher:
    """Tests for youtube-transcript-api wrapper."""

    def test_fetch_english_captions(self):
        """Test fetching English captions."""
        fetcher = TranscriptAPIFetcher(['en'])
        result = fetcher.fetch(TEST_VIDEO_ID)

        assert result.success
        assert result.tier_used == CaptionTier.TRANSCRIPT_API
        assert len(result.segments) > 0
        assert all(isinstance(s, CaptionSegment) for s in result.segments)

    def test_invalid_video_id(self):
        """Test handling of invalid video ID."""
        fetcher = TranscriptAPIFetcher()
        result = fetcher.fetch("invalid_id_xxx")

        assert not result.success
        assert result.error is not None


class TestInnertubeDirectFetcher:
    """Tests for direct Innertube/timedtext extraction."""

    def test_extract_caption_url(self):
        """Test caption URL extraction from page."""
        fetcher = InnertubeDirectFetcher()
        result = fetcher.fetch(TEST_VIDEO_ID)

        # May succeed or fail depending on YouTube's current behavior
        # Just ensure no exceptions
        assert result.tier_used == CaptionTier.INNERTUBE_DIRECT


class TestInvidiousCaptionFetcher:
    """Tests for Invidious API caption fetching."""

    def test_instance_rotation(self):
        """Test that instances are rotated on failure."""
        fetcher = InvidiousCaptionFetcher()
        initial_index = fetcher.current_index

        # Fetch (may succeed or fail)
        fetcher.fetch(TEST_VIDEO_ID)

        # Index should have changed
        assert fetcher.current_index != initial_index or len(fetcher.instances) == 1


class TestCaptionFallbackChain:
    """Integration tests for full fallback chain."""

    def test_fetch_with_fallback(self):
        """Test that fallback chain finds captions via some tier."""
        chain = CaptionFallbackChain()
        result = chain.fetch(TEST_VIDEO_ID)

        # Should succeed via at least one tier
        assert result.success
        assert len(result.segments) > 0

    def test_srt_conversion(self):
        """Test SRT format conversion."""
        chain = CaptionFallbackChain()
        segments = [
            CaptionSegment("Hello world", 0.0, 2.5),
            CaptionSegment("Testing captions", 2.5, 3.0),
        ]

        srt = chain.to_srt(segments)

        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:02,500" in srt
        assert "Hello world" in srt

    def test_vtt_conversion(self):
        """Test WebVTT format conversion."""
        chain = CaptionFallbackChain()
        segments = [
            CaptionSegment("Hello world", 0.0, 2.5),
        ]

        vtt = chain.to_vtt(segments)

        assert "WEBVTT" in vtt
        assert "00:00:00.000 --> 00:00:02.500" in vtt

    def test_stats_tracking(self):
        """Test that tier statistics are tracked."""
        chain = CaptionFallbackChain()
        chain.fetch(TEST_VIDEO_ID)

        stats = chain.get_stats()

        # At least one tier should have been attempted
        total_attempts = sum(
            s['success'] + s['failure']
            for s in stats.values()
        )
        assert total_attempts > 0
```

---

## Rollout Plan

| Phase | Scope | Priority | Files | Risk |
|-------|-------|----------|-------|------|
| 1A | youtube-transcript-api | HIGH | `caption_fallback.py` | Low |
| 1B | Innertube direct fetch | HIGH | `caption_fallback.py` | Low |
| 1C | Invidious/Piped captions | MEDIUM | `caption_fallback.py` | Low |
| 2 | Video stream fallbacks | MEDIUM | `video_fallback.py` | Low |
| 3 | Config schema | LOW | `download.py`, `config.yaml` | Low |
| 4 | Integration + tests | HIGH | `transcribe.py`, tests | Medium |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Pipeline failures from 429 | ~30% | <5% |
| Caption extraction success | ~85% | >98% |
| Fallback latency overhead | N/A | <10s avg |
| Videos requiring Whisper ASR | N/A | <5% |

---

## Dependencies

```bash
pip install youtube-transcript-api httpx
```

Add to `requirements.txt`:
```
youtube-transcript-api>=0.6.0
httpx>=0.25.0
```

---

## File Summary

| File | Status | Description |
|------|--------|-------------|
| `src/downloader/caption_fallback.py` | NEW | Multi-tier caption extraction |
| `src/downloader/video_fallback.py` | NEW | Multi-tier video download |
| `src/config/sections/download.py` | MODIFY | Add fallback config classes |
| `config.yaml` | MODIFY | Add fallback settings |
| `src/stages/transcribe.py` | MODIFY | Integrate caption fallback |
| `src/downloader/core.py` | MODIFY | Integrate video fallback |
| `tests/test_caption_fallback.py` | NEW | Caption fallback tests |
| `tests/test_video_fallback.py` | NEW | Video fallback tests |

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Cobalt self-hosting? | Optional Tier 4, disabled by default |
| Instance health monitoring? | Built into fetchers with cooldown |
| Supadata integration? | Skipped - youtube-transcript-api is better |
| Audio-first compatibility? | Video fallbacks support audio extraction |
| Third-party ASR APIs? | Not needed - local Whisper sufficient |

---

## References

### Research Documents
- `docs/research/ytdlp_alternatives.md`
- `docs/research/caption_apis.md`
- `docs/research/youtube_transcription_methods.md`
- `docs/research/transcription_apis.md`
- `docs/research/youtube_internal_apis.md`
- `docs/research/newpipe_invidious.md`
- `docs/research/video_segment_services.md`

### External Documentation
- youtube-transcript-api: https://github.com/jdepoix/youtube-transcript-api
- Invidious API: https://docs.invidious.io/api/
- Piped API: https://docs.piped.video/docs/api-documentation/
- YouTube Innertube: https://tyrrrz.me/blog/reverse-engineering-youtube-revisited
