"""
Multi-tier caption fallback system for YouTube videos.

Extraction priority:
1. yt-dlp (handled externally)
2. youtube-transcript-api (different endpoint, rarely rate-limited)
3. Direct Innertube/timedtext URL fetch
4. Invidious API (10+ instances with rotation)
5. Piped API (6+ instances with rotation)
6. Whisper ASR (last resort, handled externally)

Usage:
    chain = CaptionFallbackChain(config)
    result = chain.fetch(video_id)
    if result.success:
        srt_content = chain.to_srt(result.segments)
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

import httpx

from src.downloader.fallback_logging import (
    FallbackLogger,
    get_tier_logger,
    log_chain_result,
    log_chain_start,
    log_http_error,
    log_operation,
)
from src.downloader.http_client import create_httpx_client, get_proxy_for_httpx
from src.downloader.rate_limit_handler import RateLimitHandler, get_rate_limit_handler

if TYPE_CHECKING:
    from src.config import Config

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

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "start": self.start,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CaptionSegment":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            start=data["start"],
            duration=data["duration"],
        )


@dataclass
class CaptionResult:
    """Result of caption extraction attempt."""

    success: bool
    tier_used: CaptionTier
    segments: list[CaptionSegment] = field(default_factory=list)
    language: str = "en"
    is_auto_generated: bool = False
    error: Optional[str] = None

    @property
    def text(self) -> str:
        """Get full transcript text without timestamps."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def duration(self) -> float:
        """Get total duration covered by captions."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end


class TranscriptAPIFetcher:
    """Tier 2: youtube-transcript-api wrapper.

    Uses a different YouTube endpoint (/api/timedtext) that has
    separate rate limiting from the video player API used by yt-dlp.
    """

    def __init__(self, preferred_languages: list[str] | None = None):
        self.preferred_languages = preferred_languages or ["en", "en-US", "en-GB"]
        self._api = None
        self._errors = None
        self.log = get_tier_logger("TRANSCRIPT_API")

    @property
    def api(self):
        """Lazy load the API to avoid import errors if not installed."""
        if self._api is None:
            try:
                from youtube_transcript_api import YouTubeTranscriptApi

                self._api = YouTubeTranscriptApi
            except ImportError:
                logger.warning("youtube-transcript-api not installed")
                self._api = False
        return self._api

    @property
    def errors(self):
        """Lazy load error classes."""
        if self._errors is None:
            try:
                from youtube_transcript_api._errors import (
                    NoTranscriptFound,
                    TranscriptsDisabled,
                    VideoUnavailable,
                )

                self._errors = {
                    "TranscriptsDisabled": TranscriptsDisabled,
                    "NoTranscriptFound": NoTranscriptFound,
                    "VideoUnavailable": VideoUnavailable,
                }
            except ImportError:
                self._errors = {}
        return self._errors

    def fetch(self, video_id: str) -> CaptionResult:
        """Fetch transcript using youtube-transcript-api."""
        self.log.start_operation("fetch", video_id)
        self.log.info("Attempting TIER_2_TRANSCRIPT_API fetch...")

        if not self.api:
            self.log.failure("youtube-transcript-api not installed", reason="ModuleNotFound")
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.TRANSCRIPT_API,
                error="youtube-transcript-api not installed",
            )

        try:
            self.log.debug("Listing available transcripts...")
            transcript_list = self.api.list_transcripts(video_id)
            available_langs = [t.language_code for t in transcript_list]
            self.log.debug(f"Available transcript languages: {available_langs}")

            # Strategy 1: Manual transcript in preferred language
            self.log.debug("Strategy 1: Looking for manual transcript...")
            last_error = None
            for lang in self.preferred_languages:
                try:
                    self.log.debug(f"Trying manual transcript for language: {lang}")
                    transcript = transcript_list.find_manually_created_transcript([lang])
                    self.log.debug(f"Found manual transcript ({lang}), fetching content...")
                    data = transcript.fetch()
                    segments = [
                        CaptionSegment(s["text"], s["start"], s["duration"])
                        for s in data
                    ]
                    self.log.success(f"Got {len(segments)} segments (manual, {lang})")
                    return CaptionResult(
                        success=True,
                        tier_used=CaptionTier.TRANSCRIPT_API,
                        segments=segments,
                        language=lang,
                        is_auto_generated=False,
                    )
                except Exception as e:
                    error_str = str(e)
                    # Check for rate limiting
                    if "429" in error_str or "Too Many Requests" in error_str:
                        self.log.rate_limited("YouTube")
                        self.log.failure("Fetch aborted", reason="HTTP 429 Rate Limited")
                        return CaptionResult(
                            success=False,
                            tier_used=CaptionTier.TRANSCRIPT_API,
                            error="Rate limited by YouTube (429)",
                        )
                    last_error = error_str
                    self.log.debug(f"No manual {lang}: {error_str[:80]}")
                    continue

            # Strategy 2: Auto-generated in preferred language
            self.log.debug("Strategy 2: Looking for auto-generated transcript...")
            for lang in self.preferred_languages:
                try:
                    self.log.debug(f"Trying auto-generated transcript for language: {lang}")
                    transcript = transcript_list.find_generated_transcript([lang])
                    self.log.debug(f"Found auto-generated transcript ({lang}), fetching content...")
                    data = transcript.fetch()
                    segments = [
                        CaptionSegment(s["text"], s["start"], s["duration"])
                        for s in data
                    ]
                    self.log.success(f"Got {len(segments)} segments (auto-generated, {lang})")
                    return CaptionResult(
                        success=True,
                        tier_used=CaptionTier.TRANSCRIPT_API,
                        segments=segments,
                        language=lang,
                        is_auto_generated=True,
                    )
                except Exception as e:
                    error_str = str(e)
                    # Check for rate limiting
                    if "429" in error_str or "Too Many Requests" in error_str:
                        self.log.rate_limited("YouTube")
                        self.log.failure("Fetch aborted", reason="HTTP 429 Rate Limited")
                        return CaptionResult(
                            success=False,
                            tier_used=CaptionTier.TRANSCRIPT_API,
                            error="Rate limited by YouTube (429)",
                        )
                    last_error = error_str
                    self.log.debug(f"No auto-generated {lang}: {error_str[:80]}")
                    continue

            # Strategy 3: Any transcript, translated to English
            self.log.debug("Strategy 3: Trying any available transcript + translation...")
            try:
                available = list(transcript_list)
                if available:
                    transcript = available[0]
                    self.log.debug(f"Using transcript in {transcript.language_code}")
                    if transcript.language_code not in self.preferred_languages:
                        self.log.debug(f"Translating from {transcript.language_code} to en...")
                        transcript = transcript.translate("en")
                    data = transcript.fetch()
                    segments = [
                        CaptionSegment(s["text"], s["start"], s["duration"])
                        for s in data
                    ]
                    self.log.success(f"Got {len(segments)} segments (translated to en)")
                    return CaptionResult(
                        success=True,
                        tier_used=CaptionTier.TRANSCRIPT_API,
                        segments=segments,
                        language="en",
                        is_auto_generated=transcript.is_generated,
                    )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "Too Many Requests" in error_str:
                    self.log.rate_limited("YouTube")
                    self.log.failure("Fetch aborted", reason="HTTP 429 Rate Limited")
                    return CaptionResult(
                        success=False,
                        tier_used=CaptionTier.TRANSCRIPT_API,
                        error="Rate limited by YouTube (429)",
                    )
                last_error = error_str
                self.log.debug(f"Translation fallback failed: {error_str[:80]}")

            # Return with last known error if we have one
            error_msg = last_error if last_error else "No transcript found in any language"
            self.log.failure("All strategies failed", reason=error_msg[:100])
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.TRANSCRIPT_API,
                error=error_msg,
            )

        except Exception as e:
            error_str = str(e)
            error_type = type(e).__name__

            # Check for rate limiting in the exception message
            if "429" in error_str or "Too Many Requests" in error_str:
                self.log.rate_limited("YouTube")
                self.log.failure("Fetch aborted", reason="HTTP 429 Rate Limited")
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.TRANSCRIPT_API,
                    error="Rate limited by YouTube (429)",
                )
            elif error_type == "TranscriptsDisabled":
                self.log.failure("Transcripts disabled for video", reason="TranscriptsDisabled")
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.TRANSCRIPT_API,
                    error="Transcripts disabled for this video",
                )
            elif error_type == "VideoUnavailable":
                self.log.failure("Video unavailable", reason="VideoUnavailable")
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.TRANSCRIPT_API,
                    error="Video unavailable",
                )
            else:
                self.log.error(f"Unexpected failure", error=e)
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.TRANSCRIPT_API,
                    error=str(e),
                )


class InnertubeDirectFetcher:
    """Tier 3: Direct Innertube/timedtext URL extraction.

    Extracts caption URLs directly from YouTube's player_response JSON
    embedded in the video page, then fetches captions in srv3/json3/vtt format.
    """

    TIMEDTEXT_FORMATS = ["srv3", "json3", "vtt"]

    def __init__(
        self,
        timeout: float = 30.0,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ):
        self.timeout = timeout
        self.handler = rate_limit_handler
        self._client: httpx.Client | None = None
        self._proxy_url: str | None = None  # Track current proxy for refresh
        self.log = get_tier_logger("INNERTUBE")

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client with proxy support."""
        if self._client is None:
            self._client = create_httpx_client(
                handler=self.handler,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    def _refresh_client(self, proxy: str | None = None) -> None:
        """Refresh client after proxy rotation."""
        if self._client is not None:
            self._client.close()
            self._client = None

        # If explicit proxy provided, create client with it
        if proxy:
            self._client = create_httpx_client(proxy=proxy, timeout=30.0)

    def fetch(
        self,
        video_id: str,
        proxy: str | None = None,
        worker_id: int | None = None
    ) -> CaptionResult:
        """Extract captions directly from YouTube page.

        Args:
            video_id: YouTube video ID
            proxy: Optional proxy URL to use (overrides rate_limit_handler proxy)
            worker_id: Optional worker ID for logging
        """
        log_prefix = f"[worker_{worker_id}]" if worker_id is not None else ""
        self.log.start_operation("fetch", video_id)
        self.log.info(f"{log_prefix} Attempting TIER_3_INNERTUBE_DIRECT fetch (proxy: {proxy or 'default'})...")

        # Use explicit proxy if provided, otherwise use default client
        if proxy and (self._client is None or self._proxy_url != proxy):
            self._refresh_client(proxy)
            self._proxy_url = proxy

        try:
            # Fetch video page
            url = f"https://www.youtube.com/watch?v={video_id}"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            }
            self.log.request("GET", url)
            response = self.client.get(url, headers=headers)
            self.log.response(response.status_code, len(response.text))

            if response.status_code == 429:
                self.log.rate_limited("YouTube")
                self.log.failure("Page fetch rate limited", reason="HTTP 429")
                if self.handler:
                    self.handler.on_rate_limit("innertube_page")
                    self._refresh_client()  # Get new proxy
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.INNERTUBE_DIRECT,
                    error="Rate limited by YouTube (429)",
                )

            if response.status_code != 200:
                self.log.failure("Page fetch failed", reason=f"HTTP {response.status_code}")
                log_http_error(self.log, url, response.status_code)
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.INNERTUBE_DIRECT,
                    error=f"HTTP {response.status_code}",
                )

            # Report success to handler
            if self.handler:
                self.handler.on_success()

            # Extract captionTracks from player_response
            self.log.debug("Extracting captionTracks from player_response...")
            caption_url, language = self._extract_caption_url(response.text)
            if not caption_url:
                self.log.failure("No caption tracks found", reason="Missing captionTracks in player_response")
                return CaptionResult(
                    success=False,
                    tier_used=CaptionTier.INNERTUBE_DIRECT,
                    error="No caption tracks found in player_response",
                )

            self.log.debug(f"Found caption URL for language: {language or 'unknown'}")
            is_asr = "asr" in caption_url.lower() or "kind=asr" in caption_url.lower()
            self.log.debug(f"Caption type: {'auto-generated (ASR)' if is_asr else 'manual'}")

            # Fetch caption content (prefer srv3 XML format)
            last_error = "No formats succeeded"
            for fmt in self.TIMEDTEXT_FORMATS:
                fetch_url = self._add_format_param(caption_url, fmt)
                self.log.debug(f"Trying format: {fmt}")
                try:
                    self.log.request("GET", fetch_url)
                    cap_response = self.client.get(fetch_url)
                    self.log.response(cap_response.status_code, len(cap_response.text))

                    # Check for rate limiting
                    if cap_response.status_code == 429:
                        self.log.rate_limited("YouTube")
                        last_error = "Rate limited by YouTube (429)"
                        if self.handler:
                            self.handler.on_rate_limit("innertube_caption")
                            self._refresh_client()
                        break  # Don't try other formats, we're rate limited

                    if cap_response.status_code == 200:
                        # Check for empty or error response
                        if not cap_response.text or len(cap_response.text) < 50:
                            last_error = f"Empty response for {fmt} format"
                            self.log.debug(f"Empty response for {fmt} (size: {len(cap_response.text)})")
                            continue

                        segments = self._parse_captions(cap_response.text, fmt)
                        if segments:
                            self.log.parsing(fmt, len(segments))
                            self.log.success(f"Got {len(segments)} segments ({fmt}, {language or 'en'})")
                            return CaptionResult(
                                success=True,
                                tier_used=CaptionTier.INNERTUBE_DIRECT,
                                segments=segments,
                                language=language or "en",
                                is_auto_generated=is_asr,
                            )
                        else:
                            last_error = f"Could not parse {fmt} format"
                            self.log.debug(f"Parse failed for {fmt}")
                    else:
                        last_error = f"HTTP {cap_response.status_code} for {fmt} format"
                        log_http_error(self.log, fetch_url, cap_response.status_code)

                except Exception as e:
                    self.log.debug(f"Exception fetching {fmt}: {type(e).__name__}: {str(e)[:60]}")
                    last_error = str(e)
                    continue

            self.log.failure("All formats failed", reason=last_error[:100])
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.INNERTUBE_DIRECT,
                error=f"Failed to fetch caption content: {last_error}",
            )

        except Exception as e:
            self.log.error("Unexpected failure", error=e)
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.INNERTUBE_DIRECT,
                error=str(e),
            )

    def _extract_caption_url(self, page_html: str) -> tuple[str | None, str | None]:
        """Extract English caption URL from player_response.

        Returns (caption_url, language_code) or (None, None) if not found.
        """
        # Pattern 1: captionTracks in JSON (with DOTALL for multiline)
        match = re.search(r'"captionTracks":\s*(\[.*?\])', page_html, re.DOTALL)
        if match:
            try:
                tracks = json.loads(match.group(1))
                # Prefer English, then any
                for track in tracks:
                    lang = track.get("languageCode", "")
                    if lang.startswith("en"):
                        base_url = track.get("baseUrl", "")
                        if base_url:
                            return base_url.replace("\\u0026", "&"), lang
                # Fallback to first track
                if tracks:
                    base_url = tracks[0].get("baseUrl", "")
                    lang = tracks[0].get("languageCode", "")
                    if base_url:
                        return base_url.replace("\\u0026", "&"), lang
            except json.JSONDecodeError:
                pass

        # Pattern 2: Direct timedtext URL
        match = re.search(
            r'"(https://www\.youtube\.com/api/timedtext[^"]+)"', page_html
        )
        if match:
            return match.group(1).replace("\\u0026", "&"), None

        return None, None

    def _add_format_param(self, url: str, fmt: str) -> str:
        """Add or replace format parameter in URL."""
        if "fmt=" in url:
            return re.sub(r"fmt=[^&]+", f"fmt={fmt}", url)
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}fmt={fmt}"

    def _parse_captions(self, content: str, fmt: str) -> list[CaptionSegment]:
        """Parse caption content based on format."""
        segments: list[CaptionSegment] = []

        if fmt == "srv3":
            # XML format: <text start="0.0" dur="2.5">Caption text</text>
            for match in re.finditer(
                r'<text[^>]*start="([^"]+)"[^>]*dur="([^"]+)"[^>]*>([^<]*)</text>',
                content,
            ):
                try:
                    start = float(match.group(1))
                    dur = float(match.group(2))
                    text = html.unescape(match.group(3))
                    if text.strip():
                        segments.append(CaptionSegment(text, start, dur))
                except (ValueError, IndexError):
                    continue

        elif fmt == "json3":
            # JSON format
            try:
                data = json.loads(content)
                for event in data.get("events", []):
                    if "segs" in event:
                        text = "".join(seg.get("utf8", "") for seg in event["segs"])
                        start = event.get("tStartMs", 0) / 1000
                        dur = event.get("dDurationMs", 0) / 1000
                        if text.strip():
                            segments.append(CaptionSegment(text, start, dur))
            except json.JSONDecodeError:
                pass

        elif fmt == "vtt":
            # WebVTT format
            segments = self._parse_vtt(content)

        return segments

    def _parse_vtt(self, content: str) -> list[CaptionSegment]:
        """Parse WebVTT subtitle content."""
        segments: list[CaptionSegment] = []

        # Match timestamp lines and following text
        pattern = r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n([^\n]+)"
        for match in re.finditer(pattern, content):
            try:
                start = self._vtt_to_seconds(match.group(1))
                end = self._vtt_to_seconds(match.group(2))
                text = match.group(3).strip()
                # Remove VTT tags like <c> </c>
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    segments.append(CaptionSegment(text, start, end - start))
            except (ValueError, IndexError):
                continue

        return segments

    @staticmethod
    def _vtt_to_seconds(timestamp: str) -> float:
        """Convert VTT timestamp to seconds."""
        timestamp = timestamp.replace(",", ".")
        parts = timestamp.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(parts[0])


class InvidiousCaptionFetcher:
    """Tier 4: Invidious API caption fetching with instance rotation.

    Invidious is a privacy-respecting YouTube frontend that provides
    an API for accessing video metadata including captions.
    """

    # Updated instance list - check https://api.invidious.io/instances.json
    # Note: Instance availability varies; the system will rotate through them
    DEFAULT_INSTANCES = [
        "https://iv.melmac.space",  # Tested working 2025-01
        "https://invidious.fdn.fr",
        "https://invidious.perennialte.ch",
        "https://yewtu.be",
        "https://invidious.kavin.rocks",
        "https://vid.puffyan.us",
        "https://invidious.namazso.eu",
        "https://inv.nadeko.net",
        "https://invidious.lunar.icu",
        "https://invidious.privacydev.net",
    ]

    def __init__(
        self,
        instances: list[str] | None = None,
        timeout: float = 30.0,
        cooldown_seconds: float = 300.0,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ):
        self.instances = instances or self.DEFAULT_INSTANCES.copy()
        self.timeout = timeout
        self.cooldown_seconds = cooldown_seconds
        self.failed_instances: dict[str, float] = {}  # instance -> failure timestamp
        self.current_index = 0
        self.log = get_tier_logger("INVIDIOUS")
        self.handler = rate_limit_handler
        self._proxy = get_proxy_for_httpx(rate_limit_handler)

    def _refresh_proxy(self, proxy: str | None = None) -> None:
        """Refresh proxy after rotation."""
        if proxy:
            self._proxy = proxy
        else:
            self._proxy = get_proxy_for_httpx(self.handler)

    def fetch(
        self,
        video_id: str,
        proxy: str | None = None,
        worker_id: int | None = None
    ) -> CaptionResult:
        """Fetch captions via Invidious API with instance rotation.

        Args:
            video_id: YouTube video ID
            proxy: Optional proxy URL to use (overrides default)
            worker_id: Optional worker ID for logging
        """
        log_prefix = f"[worker_{worker_id}]" if worker_id is not None else ""
        self.log.start_operation("fetch", video_id)
        self.log.info(f"{log_prefix} Attempting TIER_4_INVIDIOUS fetch (proxy: {proxy or self._proxy or 'direct'})...")

        # Use explicit proxy if provided
        effective_proxy = proxy if proxy else self._proxy

        instances_tried = 0
        instances_skipped = 0
        last_error = "No instances available"

        for _ in range(len(self.instances)):
            instance = self._get_next_instance()
            if not instance:
                instances_skipped += 1
                continue

            instances_tried += 1
            self.log.start_operation("fetch", video_id, instance=instance)
            self.log.instance_status(instance, "trying")

            try:
                url = f"{instance}/api/v1/videos/{video_id}"
                self.log.request("GET", url)
                response = httpx.get(url, timeout=self.timeout, proxy=effective_proxy)
                self.log.response(response.status_code, len(response.text))

                if response.status_code == 429:
                    self.failed_instances[instance] = time.time()
                    self.log.rate_limited(instance)
                    self.log.instance_status(instance, "rate_limited -> cooldown")
                    last_error = f"Rate limited by {instance}"
                    if self.handler and not proxy:  # Don't refresh if using explicit proxy
                        self.handler.on_rate_limit(f"invidious_{instance}")
                        self._refresh_proxy()
                        effective_proxy = self._proxy
                    continue

                if response.status_code != 200:
                    log_http_error(self.log, url, response.status_code)
                    self.log.instance_status(instance, f"error_{response.status_code}")
                    last_error = f"HTTP {response.status_code} from {instance}"
                    continue

                data = response.json()
                captions = data.get("captions", [])
                self.log.debug(f"Found {len(captions)} caption tracks in response")

                if not captions:
                    self.log.debug("No captions in API response")
                    last_error = f"No captions returned by {instance}"
                    continue

                # Find English caption
                for cap in captions:
                    lang = cap.get("language_code", "") or cap.get("languageCode", "")
                    if lang.startswith("en"):
                        cap_url = cap.get("url")
                        if cap_url:
                            # URL might be relative
                            if cap_url.startswith("/"):
                                cap_url = f"{instance}{cap_url}"

                            self.log.debug(f"Fetching caption content for {lang}...")
                            self.log.request("GET", cap_url)
                            cap_response = httpx.get(cap_url, timeout=self.timeout, proxy=self._proxy)
                            self.log.response(cap_response.status_code, len(cap_response.text))

                            if cap_response.status_code == 429:
                                self.log.rate_limited(instance)
                                last_error = f"Caption fetch rate limited by {instance}"
                                if self.handler:
                                    self.handler.on_rate_limit(f"invidious_caption_{instance}")
                                    self._refresh_proxy()
                                break

                            if cap_response.status_code == 200:
                                segments = self._parse_captions(cap_response.text)
                                if segments:
                                    self.log.parsing("xml/vtt", len(segments))
                                    self.log.success(f"Got {len(segments)} segments from {instance} ({lang})")
                                    if self.handler:
                                        self.handler.on_success()
                                    return CaptionResult(
                                        success=True,
                                        tier_used=CaptionTier.INVIDIOUS,
                                        segments=segments,
                                        language=lang,
                                    )
                                else:
                                    self.log.debug("Caption parse returned no segments")
                                    last_error = f"Parse failed from {instance}"
                            else:
                                last_error = f"Caption HTTP {cap_response.status_code} from {instance}"

                self.log.debug(f"No English captions from {instance}")
                last_error = f"No English captions from {instance}"

            except httpx.TimeoutException:
                self.failed_instances[instance] = time.time()
                self.log.warning(f"Timeout from {instance} -> cooldown")
                self.log.instance_status(instance, "timeout -> cooldown")
                last_error = f"Timeout from {instance}"
                continue
            except Exception as e:
                self.log.debug(f"Exception from {instance}: {type(e).__name__}: {str(e)[:60]}")
                last_error = f"{type(e).__name__} from {instance}: {str(e)[:40]}"
                continue

        self.log.failure(
            f"All instances failed (tried: {instances_tried}, skipped: {instances_skipped})",
            reason=last_error[:100]
        )
        return CaptionResult(
            success=False,
            tier_used=CaptionTier.INVIDIOUS,
            error=f"All Invidious instances failed: {last_error}",
        )

    def _get_next_instance(self) -> str | None:
        """Get next healthy instance."""
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

    def _parse_captions(self, content: str) -> list[CaptionSegment]:
        """Parse caption content from Invidious."""
        segments: list[CaptionSegment] = []

        # Try XML format first (srv3)
        for match in re.finditer(
            r'<text[^>]*start="([^"]+)"[^>]*dur="([^"]+)"[^>]*>([^<]*)</text>',
            content,
        ):
            try:
                start = float(match.group(1))
                dur = float(match.group(2))
                text = html.unescape(match.group(3))
                if text.strip():
                    segments.append(CaptionSegment(text, start, dur))
            except (ValueError, IndexError):
                continue

        if segments:
            return segments

        # Try VTT format
        pattern = r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n([^\n]+)"
        for match in re.finditer(pattern, content):
            try:
                start = InnertubeDirectFetcher._vtt_to_seconds(match.group(1))
                end = InnertubeDirectFetcher._vtt_to_seconds(match.group(2))
                text = match.group(3).strip()
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    segments.append(CaptionSegment(text, start, end - start))
            except (ValueError, IndexError):
                continue

        return segments


class PipedCaptionFetcher:
    """Tier 5: Piped API caption fetching.

    Piped is a privacy-focused YouTube frontend built on the NewPipe extractor.
    Note: Piped instances can be unreliable; this is a best-effort fallback.
    """

    DEFAULT_INSTANCES = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.adminforge.de",
        "https://pipedapi.r4fo.com",
        "https://pipedapi.leptons.xyz",
        "https://api.piped.privacydev.net",
        "https://pipedapi.us.projectsegfau.lt",
    ]

    def __init__(
        self,
        instances: list[str] | None = None,
        timeout: float = 30.0,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ):
        self.instances = instances or self.DEFAULT_INSTANCES.copy()
        self.timeout = timeout
        self.current_index = 0
        self.log = get_tier_logger("PIPED")
        self.handler = rate_limit_handler
        self._proxy = get_proxy_for_httpx(rate_limit_handler)

    def _refresh_proxy(self, proxy: str | None = None) -> None:
        """Refresh proxy after rotation."""
        if proxy:
            self._proxy = proxy
        else:
            self._proxy = get_proxy_for_httpx(self.handler)

    def fetch(
        self,
        video_id: str,
        proxy: str | None = None,
        worker_id: int | None = None
    ) -> CaptionResult:
        """Fetch captions via Piped API.

        Args:
            video_id: YouTube video ID
            proxy: Optional proxy URL to use (overrides default)
            worker_id: Optional worker ID for logging
        """
        log_prefix = f"[worker_{worker_id}]" if worker_id is not None else ""
        self.log.start_operation("fetch", video_id)
        self.log.info(f"{log_prefix} Attempting TIER_5_PIPED fetch (proxy: {proxy or self._proxy or 'direct'})...")

        # Use explicit proxy if provided
        effective_proxy = proxy if proxy else self._proxy

        instances_tried = 0
        last_error = "No instances available"

        for instance in self._rotate_instances():
            instances_tried += 1
            self.log.start_operation("fetch", video_id, instance=instance)
            self.log.instance_status(instance, "trying")

            try:
                url = f"{instance}/streams/{video_id}"
                self.log.request("GET", url)
                response = httpx.get(url, timeout=self.timeout, proxy=effective_proxy)
                self.log.response(response.status_code, len(response.text))

                if response.status_code == 429:
                    self.log.rate_limited(instance)
                    last_error = f"Rate limited by {instance}"
                    if self.handler and not proxy:  # Don't refresh if using explicit proxy
                        self.handler.on_rate_limit(f"piped_{instance}")
                        self._refresh_proxy()
                        effective_proxy = self._proxy
                    continue

                if response.status_code != 200:
                    log_http_error(self.log, url, response.status_code)
                    last_error = f"HTTP {response.status_code} from {instance}"
                    continue

                data = response.json()
                subtitles = data.get("subtitles", [])
                self.log.debug(f"Found {len(subtitles)} subtitle tracks in response")

                if not subtitles:
                    self.log.debug("No subtitles in API response")
                    last_error = f"No subtitles from {instance}"
                    continue

                # Find English subtitle
                for sub in subtitles:
                    code = sub.get("code", "")
                    if code.startswith("en"):
                        sub_url = sub.get("url")
                        if sub_url:
                            self.log.debug(f"Fetching subtitle content for {code}...")
                            self.log.request("GET", sub_url)
                            sub_response = httpx.get(sub_url, timeout=self.timeout, proxy=self._proxy)
                            self.log.response(sub_response.status_code, len(sub_response.text))

                            if sub_response.status_code == 429:
                                self.log.rate_limited("subtitle endpoint")
                                last_error = f"Subtitle fetch rate limited"
                                if self.handler:
                                    self.handler.on_rate_limit(f"piped_subtitle_{instance}")
                                    self._refresh_proxy()
                                break

                            if sub_response.status_code == 200:
                                segments = self._parse_vtt(sub_response.text)
                                if segments:
                                    auto_gen = sub.get("autoGenerated", False)
                                    self.log.parsing("vtt", len(segments))
                                    self.log.success(
                                        f"Got {len(segments)} segments from {instance} "
                                        f"({code}, {'auto' if auto_gen else 'manual'})"
                                    )
                                    if self.handler:
                                        self.handler.on_success()
                                    return CaptionResult(
                                        success=True,
                                        tier_used=CaptionTier.PIPED,
                                        segments=segments,
                                        language=code,
                                        is_auto_generated=auto_gen,
                                    )
                                else:
                                    self.log.debug("VTT parse returned no segments")
                                    last_error = f"Parse failed from {instance}"
                            else:
                                last_error = f"Subtitle HTTP {sub_response.status_code}"

                self.log.debug(f"No English subtitles from {instance}")
                last_error = f"No English subtitles from {instance}"

            except httpx.TimeoutException:
                self.log.warning(f"Timeout from {instance}")
                last_error = f"Timeout from {instance}"
                continue
            except Exception as e:
                self.log.debug(f"Exception from {instance}: {type(e).__name__}: {str(e)[:60]}")
                last_error = f"{type(e).__name__} from {instance}"
                continue

        self.log.failure(f"All instances failed (tried: {instances_tried})", reason=last_error[:100])
        return CaptionResult(
            success=False,
            tier_used=CaptionTier.PIPED,
            error=f"All Piped instances failed: {last_error}",
        )

    def _rotate_instances(self):
        """Yield instances in rotating order."""
        for i in range(len(self.instances)):
            idx = (self.current_index + i) % len(self.instances)
            yield self.instances[idx]
        self.current_index = (self.current_index + 1) % len(self.instances)

    def _parse_vtt(self, content: str) -> list[CaptionSegment]:
        """Parse VTT subtitle content."""
        segments: list[CaptionSegment] = []

        pattern = r"(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n([^\n]+)"
        for match in re.finditer(pattern, content):
            try:
                start = InnertubeDirectFetcher._vtt_to_seconds(match.group(1))
                end = InnertubeDirectFetcher._vtt_to_seconds(match.group(2))
                text = match.group(3).strip()
                # Remove VTT tags like <c> </c>
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    segments.append(CaptionSegment(text, start, end - start))
            except (ValueError, IndexError):
                continue

        return segments


class CaptionFallbackChain:
    """
    Orchestrates multi-tier caption extraction with automatic fallback.

    Usage:
        chain = CaptionFallbackChain(config)
        result = chain.fetch(video_id)
        if result.success:
            print(f"Got {len(result.segments)} segments via {result.tier_used.name}")
            srt_content = chain.to_srt(result.segments)
    """

    def __init__(
        self,
        config: "Config | None" = None,
        enable_whisper_fallback: bool = True,
        preferred_languages: list[str] | None = None,
    ):
        self.config = config
        self.enable_whisper_fallback = enable_whisper_fallback
        self.log = get_tier_logger("CHAIN")

        # Initialize rate limit handler for proxy/VPN support
        self.rate_limit_handler = get_rate_limit_handler(config)

        # Determine languages
        languages = preferred_languages or ["en", "en-US", "en-GB", "en-AU"]
        if config and hasattr(config, "download"):
            fallback_cfg = getattr(config.download, "fallback", None)
            if fallback_cfg and hasattr(fallback_cfg, "caption"):
                cap_cfg = fallback_cfg.caption
                if hasattr(cap_cfg, "transcript_api"):
                    ta_cfg = cap_cfg.transcript_api
                    if isinstance(ta_cfg, dict):
                        languages = ta_cfg.get("preferred_languages", languages)
                    elif hasattr(ta_cfg, "preferred_languages"):
                        languages = ta_cfg.preferred_languages

        # Initialize fetchers with rate limit handler for proxy support
        self.transcript_api = TranscriptAPIFetcher(languages)  # No proxy support (library internal)
        self.innertube_direct = InnertubeDirectFetcher(
            rate_limit_handler=self.rate_limit_handler
        )
        self.invidious = InvidiousCaptionFetcher(
            rate_limit_handler=self.rate_limit_handler
        )
        self.piped = PipedCaptionFetcher(
            rate_limit_handler=self.rate_limit_handler
        )

        # Adaptive behavior tracking
        self.tier_stats: dict[CaptionTier, dict[str, int]] = {
            tier: {"success": 0, "failure": 0} for tier in CaptionTier
        }

    def fetch(
        self,
        video_id: str,
        skip_tiers: list[CaptionTier] | None = None,
        proxy: str | None = None,
        worker_id: int | None = None
    ) -> CaptionResult:
        """
        Fetch captions with automatic fallback through all tiers.

        Args:
            video_id: YouTube video ID (11 characters)
            skip_tiers: List of tiers to skip (e.g., [CaptionTier.YTDLP] when rate-limited)
            proxy: Optional proxy URL to use for HTTP requests (for parallel workers)
            worker_id: Optional worker ID for logging (for parallel workers)

        Returns:
            CaptionResult with segments if successful
        """
        skip_tiers = skip_tiers or []
        log_prefix = f"[worker_{worker_id}]" if worker_id is not None else ""

        # Log chain start
        log_chain_start(video_id, [t.name for t in skip_tiers] if skip_tiers else None)
        self.log.start_operation("fallback_chain", video_id)

        # Define fetcher chain (Tier 1 yt-dlp is handled externally)
        # Check config for enabled tiers
        invidious_enabled = True
        piped_enabled = True
        if self.config and hasattr(self.config, "download"):
            fallback_cfg = getattr(self.config.download, "fallback", None)
            self.log.debug(f"{log_prefix} fallback_cfg type: {type(fallback_cfg)}, has caption: {hasattr(fallback_cfg, 'caption') if fallback_cfg else 'N/A'}")
            if fallback_cfg and hasattr(fallback_cfg, "caption"):
                cap_cfg = fallback_cfg.caption
                self.log.debug(f"{log_prefix} cap_cfg type: {type(cap_cfg)}, has invidious_enabled: {hasattr(cap_cfg, 'invidious_enabled')}")
                if hasattr(cap_cfg, "invidious_enabled"):
                    invidious_enabled = cap_cfg.invidious_enabled
                    self.log.debug(f"{log_prefix} invidious_enabled from config: {invidious_enabled}")
                if hasattr(cap_cfg, "piped_enabled"):
                    piped_enabled = cap_cfg.piped_enabled
                    self.log.debug(f"{log_prefix} piped_enabled from config: {piped_enabled}")
        else:
            self.log.debug(f"{log_prefix} No config or download attr - config type: {type(self.config)}")

        fetchers: list[tuple[CaptionTier, object]] = [
            (CaptionTier.TRANSCRIPT_API, self.transcript_api),
            (CaptionTier.INNERTUBE_DIRECT, self.innertube_direct),
        ]
        if invidious_enabled:
            fetchers.append((CaptionTier.INVIDIOUS, self.invidious))
        else:
            self.log.debug(f"{log_prefix} INVIDIOUS disabled in config")
        if piped_enabled:
            fetchers.append((CaptionTier.PIPED, self.piped))
        else:
            self.log.debug(f"{log_prefix} PIPED disabled in config")

        last_error = None
        tiers_tried = 0
        tiers_skipped = 0

        for tier, fetcher in fetchers:
            if tier in skip_tiers:
                self.log.debug(f"{log_prefix} Skipping {tier.name} (in skip_tiers)")
                tiers_skipped += 1
                continue

            tiers_tried += 1
            self.log.info(f"{log_prefix} === Trying {tier.name} (tier {tiers_tried}) ===")

            # Pass proxy to HTTP-based fetchers (TranscriptAPI uses internal library)
            if tier == CaptionTier.TRANSCRIPT_API:
                result = fetcher.fetch(video_id)
            else:
                # HTTP-based fetchers accept proxy parameter
                result = fetcher.fetch(video_id, proxy=proxy, worker_id=worker_id)

            if result.success:
                self.tier_stats[tier]["success"] += 1

                # Report success to rate limit handler
                self.rate_limit_handler.on_success()

                # Log detailed success
                log_chain_result(video_id, True, tier.name, self.tier_stats)
                self.log.success(
                    f"Chain succeeded via {tier.name} | "
                    f"{len(result.segments)} segments | "
                    f"Language: {result.language} | "
                    f"Auto-generated: {result.is_auto_generated}"
                )
                return result

            self.tier_stats[tier]["failure"] += 1
            last_error = result.error or "Unknown"

            # Check if it was a rate limit and handle it
            if "429" in last_error or "rate limit" in last_error.lower():
                self.rate_limit_handler.on_rate_limit(f"{tier.name} for {video_id}")

            self.log.warning(f"{tier.name} failed: {last_error[:100]}")

        # All tiers failed
        log_chain_result(video_id, False, "NONE", self.tier_stats)
        self.log.failure(
            f"All tiers exhausted (tried: {tiers_tried}, skipped: {tiers_skipped})",
            reason=last_error[:100] if last_error else "All tiers failed"
        )

        if self.enable_whisper_fallback:
            self.log.info("Whisper ASR fallback is available as last resort")
            return CaptionResult(
                success=False,
                tier_used=CaptionTier.WHISPER_ASR,
                error=f"All caption sources failed. Last error: {last_error}. "
                "Consider Whisper ASR fallback.",
            )

        return CaptionResult(
            success=False,
            tier_used=CaptionTier.PIPED,
            error=f"All caption sources failed. Last error: {last_error}",
        )

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get success/failure statistics for each tier."""
        return {tier.name: stats.copy() for tier, stats in self.tier_stats.items()}

    def reset_stats(self) -> None:
        """Reset tier statistics."""
        for tier in self.tier_stats:
            self.tier_stats[tier] = {"success": 0, "failure": 0}

    @staticmethod
    def to_srt(segments: list[CaptionSegment]) -> str:
        """Convert caption segments to SRT format."""
        lines: list[str] = []
        for i, seg in enumerate(segments, 1):
            start_ts = CaptionFallbackChain._seconds_to_srt(seg.start)
            end_ts = CaptionFallbackChain._seconds_to_srt(seg.end)
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_vtt(segments: list[CaptionSegment]) -> str:
        """Convert caption segments to WebVTT format."""
        lines = ["WEBVTT", ""]
        for seg in segments:
            start_ts = CaptionFallbackChain._seconds_to_vtt(seg.start)
            end_ts = CaptionFallbackChain._seconds_to_vtt(seg.end)
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def to_transcript_list(segments: list[CaptionSegment]) -> list[dict]:
        """Convert to list of dicts matching youtube-transcript-api format."""
        return [
            {"text": seg.text, "start": seg.start, "duration": seg.duration}
            for seg in segments
        ]

    @staticmethod
    def _seconds_to_srt(seconds: float) -> str:
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _seconds_to_vtt(seconds: float) -> str:
        """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# Convenience function for quick usage
def fetch_captions(
    video_id: str,
    preferred_languages: list[str] | None = None,
) -> CaptionResult:
    """
    Convenience function to fetch captions for a video.

    Args:
        video_id: YouTube video ID
        preferred_languages: List of preferred language codes

    Returns:
        CaptionResult with segments if successful
    """
    chain = CaptionFallbackChain(preferred_languages=preferred_languages)
    return chain.fetch(video_id)
