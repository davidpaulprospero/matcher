"""
Multi-tier video download fallback system for YouTube videos.

Download priority:
1. yt-dlp (handled externally)
2. Invidious API (10+ instances with rotation)
3. Piped API (6+ instances with rotation)
4. Cobalt API (self-hosted, optional)

Usage:
    chain = VideoFallbackChain(config)
    result = chain.download(video_id, output_dir)
    if result.success:
        print(f"Downloaded to {result.file_path}")
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import httpx

from src.downloader.fallback_logging import (
    FallbackLogger,
    get_tier_logger,
    log_http_error,
)
from src.downloader.http_client import get_proxy_for_httpx
from src.downloader.rate_limit_handler import RateLimitHandler, get_rate_limit_handler

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class VideoTier(Enum):
    """Video download tiers in priority order."""

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
    bitrate: Optional[int] = None
    fps: Optional[int] = None

    def __str__(self) -> str:
        size_str = f", {self.filesize // 1024 // 1024}MB" if self.filesize else ""
        return f"{self.quality} {self.format}{size_str}"


@dataclass
class VideoResult:
    """Result of video download attempt."""

    success: bool
    tier_used: VideoTier
    file_path: Optional[Path] = None
    stream_info: Optional[StreamInfo] = None
    error: Optional[str] = None
    download_time: Optional[float] = None


class InvidiousStreamFetcher:
    """Tier 2: Fetch video streams from Invidious API.

    Invidious provides direct stream URLs from YouTube via their API.
    """

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
        self.failed_instances: dict[str, float] = {}
        self.current_index = 0
        self.log = get_tier_logger("VIDEO_INVIDIOUS")
        self.handler = rate_limit_handler
        self._proxy = get_proxy_for_httpx(rate_limit_handler)

    def _refresh_proxy(self) -> None:
        """Refresh proxy after rotation."""
        self._proxy = get_proxy_for_httpx(self.handler)

    def get_streams(
        self, video_id: str, preferred_quality: str = "720p"
    ) -> list[StreamInfo]:
        """Get available video streams."""
        self.log.start_operation("get_streams", video_id)
        self.log.info(f"Fetching streams (preferred: {preferred_quality})...")

        instances_tried = 0
        last_error = "No instances available"

        for instance in self._rotate_instances():
            instances_tried += 1
            self.log.start_operation("get_streams", video_id, instance=instance)
            self.log.instance_status(instance, "trying")

            try:
                url = f"{instance}/api/v1/videos/{video_id}"
                self.log.request("GET", url)
                response = httpx.get(url, timeout=self.timeout, proxy=self._proxy)
                self.log.response(response.status_code, len(response.text))

                if response.status_code == 429:
                    self.failed_instances[instance] = time.time()
                    self.log.rate_limited(instance)
                    self.log.instance_status(instance, "rate_limited -> cooldown")
                    last_error = f"Rate limited by {instance}"
                    if self.handler:
                        self.handler.on_rate_limit(f"video_invidious_{instance}")
                        self._refresh_proxy()
                    continue

                if response.status_code != 200:
                    log_http_error(self.log, url, response.status_code)
                    last_error = f"HTTP {response.status_code} from {instance}"
                    continue

                data = response.json()
                streams: list[StreamInfo] = []

                # Parse adaptive formats (video-only and audio-only)
                adaptive_count = len(data.get("adaptiveFormats", []))
                for fmt in data.get("adaptiveFormats", []):
                    fmt_type = fmt.get("type", "")
                    streams.append(
                        StreamInfo(
                            url=fmt.get("url", ""),
                            quality=fmt.get("qualityLabel", "unknown"),
                            format=fmt.get("container", "mp4"),
                            filesize=self._parse_int(fmt.get("contentLength")),
                            has_audio=fmt_type.startswith("audio/"),
                            bitrate=self._parse_int(fmt.get("bitrate")),
                            fps=self._parse_int(fmt.get("fps")),
                        )
                    )

                # Parse format streams (combined video+audio)
                format_count = len(data.get("formatStreams", []))
                for fmt in data.get("formatStreams", []):
                    streams.append(
                        StreamInfo(
                            url=fmt.get("url", ""),
                            quality=fmt.get("qualityLabel", "unknown"),
                            format=fmt.get("container", "mp4"),
                            has_audio=True,
                        )
                    )

                if streams:
                    sorted_streams = self._sort_by_quality(streams, preferred_quality)
                    combined = sum(1 for s in sorted_streams if s.has_audio and "audio" not in s.quality.lower())
                    self.log.success(
                        f"Got {len(sorted_streams)} streams from {instance} "
                        f"(adaptive: {adaptive_count}, format: {format_count}, combined: {combined})"
                    )
                    if self.handler:
                        self.handler.on_success()
                    return sorted_streams
                else:
                    self.log.debug(f"No streams in response from {instance}")
                    last_error = f"No streams from {instance}"

            except httpx.TimeoutException:
                self.failed_instances[instance] = time.time()
                self.log.warning(f"Timeout from {instance} -> cooldown")
                last_error = f"Timeout from {instance}"
                continue
            except Exception as e:
                self.log.debug(f"Exception from {instance}: {type(e).__name__}: {str(e)[:60]}")
                last_error = f"{type(e).__name__} from {instance}"
                continue

        self.log.failure(f"All instances failed (tried: {instances_tried})", reason=last_error[:100])
        return []

    def _rotate_instances(self):
        """Yield healthy instances in rotating order."""
        now = time.time()
        for i in range(len(self.instances)):
            idx = (self.current_index + i) % len(self.instances)
            instance = self.instances[idx]

            # Skip instances on cooldown
            if instance in self.failed_instances:
                if now - self.failed_instances[instance] < self.cooldown_seconds:
                    continue
                else:
                    del self.failed_instances[instance]

            yield instance

        self.current_index = (self.current_index + 1) % len(self.instances)

    def _sort_by_quality(
        self, streams: list[StreamInfo], preferred: str
    ) -> list[StreamInfo]:
        """Sort streams with preferred quality first, prioritizing combined streams."""
        preferred_height = int(re.sub(r"\D", "", preferred) or "720")

        def quality_key(s: StreamInfo) -> tuple[int, int, int]:
            # Priority 1: Has audio (combined streams first)
            has_audio_score = 0 if s.has_audio else 1

            # Priority 2: Quality match
            match = re.search(r"(\d+)p", s.quality)
            if match:
                height = int(match.group(1))
                quality_diff = abs(height - preferred_height)
            else:
                quality_diff = 9999

            # Priority 3: Bitrate (higher is better, so negate)
            bitrate_score = -(s.bitrate or 0)

            return (has_audio_score, quality_diff, bitrate_score)

        return sorted(streams, key=quality_key)

    @staticmethod
    def _parse_int(value) -> Optional[int]:
        """Safely parse integer from various types."""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None


class PipedStreamFetcher:
    """Tier 3: Fetch video streams from Piped API.

    Piped is a privacy-focused YouTube frontend built on NewPipe extractor.
    """

    DEFAULT_INSTANCES = [
        "https://pipedapi.kavin.rocks",
        "https://pipedapi.tokhmi.xyz",
        "https://pipedapi.moomoo.me",
        "https://pipedapi.syncpundit.io",
        "https://api.piped.yt",
        "https://pipedapi.in.projectsegfau.lt",
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
        self.log = get_tier_logger("VIDEO_PIPED")
        self.handler = rate_limit_handler
        self._proxy = get_proxy_for_httpx(rate_limit_handler)

    def _refresh_proxy(self) -> None:
        """Refresh proxy after rotation."""
        self._proxy = get_proxy_for_httpx(self.handler)

    def get_streams(
        self, video_id: str, preferred_quality: str = "720p"
    ) -> list[StreamInfo]:
        """Get available video streams."""
        self.log.start_operation("get_streams", video_id)
        self.log.info(f"Fetching streams (preferred: {preferred_quality})...")

        instances_tried = 0
        last_error = "No instances available"

        for instance in self._rotate_instances():
            instances_tried += 1
            self.log.start_operation("get_streams", video_id, instance=instance)
            self.log.instance_status(instance, "trying")

            try:
                url = f"{instance}/streams/{video_id}"
                self.log.request("GET", url)
                response = httpx.get(url, timeout=self.timeout, proxy=self._proxy)
                self.log.response(response.status_code, len(response.text))

                if response.status_code == 429:
                    self.log.rate_limited(instance)
                    last_error = f"Rate limited by {instance}"
                    if self.handler:
                        self.handler.on_rate_limit(f"video_piped_{instance}")
                        self._refresh_proxy()
                    continue

                if response.status_code != 200:
                    log_http_error(self.log, url, response.status_code)
                    last_error = f"HTTP {response.status_code} from {instance}"
                    continue

                data = response.json()
                streams: list[StreamInfo] = []

                # Video streams
                video_count = len(data.get("videoStreams", []))
                for vs in data.get("videoStreams", []):
                    streams.append(
                        StreamInfo(
                            url=vs.get("url", ""),
                            quality=vs.get("quality", "unknown"),
                            format=vs.get("format", "mp4"),
                            filesize=self._parse_int(vs.get("contentLength")),
                            has_audio=not vs.get("videoOnly", True),
                            bitrate=self._parse_int(vs.get("bitrate")),
                            fps=self._parse_int(vs.get("fps")),
                        )
                    )

                # Audio streams (for potential muxing)
                audio_count = len(data.get("audioStreams", []))
                for audio in data.get("audioStreams", []):
                    streams.append(
                        StreamInfo(
                            url=audio.get("url", ""),
                            quality=f"{audio.get('quality', 'unknown')} audio",
                            format=audio.get("format", "m4a"),
                            filesize=self._parse_int(audio.get("contentLength")),
                            has_audio=True,
                            bitrate=self._parse_int(audio.get("bitrate")),
                        )
                    )

                if streams:
                    sorted_streams = self._sort_streams(streams, preferred_quality)
                    combined = sum(1 for s in sorted_streams if s.has_audio and "audio" not in s.quality.lower())
                    self.log.success(
                        f"Got {len(sorted_streams)} streams from {instance} "
                        f"(video: {video_count}, audio: {audio_count}, combined: {combined})"
                    )
                    if self.handler:
                        self.handler.on_success()
                    return sorted_streams
                else:
                    self.log.debug(f"No streams in response from {instance}")
                    last_error = f"No streams from {instance}"

            except httpx.TimeoutException:
                self.log.warning(f"Timeout from {instance}")
                last_error = f"Timeout from {instance}"
                continue
            except Exception as e:
                self.log.debug(f"Exception from {instance}: {type(e).__name__}: {str(e)[:60]}")
                last_error = f"{type(e).__name__} from {instance}"
                continue

        self.log.failure(f"All instances failed (tried: {instances_tried})", reason=last_error[:100])
        return []

    def _rotate_instances(self):
        """Yield instances in rotating order."""
        for i in range(len(self.instances)):
            idx = (self.current_index + i) % len(self.instances)
            yield self.instances[idx]
        self.current_index = (self.current_index + 1) % len(self.instances)

    def _sort_streams(
        self, streams: list[StreamInfo], preferred: str
    ) -> list[StreamInfo]:
        """Sort streams by quality preference."""
        preferred_height = int(re.sub(r"\D", "", preferred) or "720")

        def quality_key(s: StreamInfo) -> tuple[int, int]:
            # Audio-only streams go last
            if "audio" in s.quality.lower():
                return (1, 0)

            match = re.search(r"(\d+)p", s.quality)
            if match:
                height = int(match.group(1))
                return (0, abs(height - preferred_height))
            return (0, 9999)

        return sorted(streams, key=quality_key)

    @staticmethod
    def _parse_int(value) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None


class CobaltFetcher:
    """Tier 4: Cobalt API for video downloads (optional, requires self-hosting)."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 60.0,
        rate_limit_handler: Optional[RateLimitHandler] = None,
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.log = get_tier_logger("VIDEO_COBALT")
        self.handler = rate_limit_handler
        self._proxy = get_proxy_for_httpx(rate_limit_handler)

    def _refresh_proxy(self) -> None:
        """Refresh proxy after rotation."""
        self._proxy = get_proxy_for_httpx(self.handler)

    def get_download_url(
        self, video_id: str, quality: str = "720"
    ) -> Optional[StreamInfo]:
        """Get download URL from Cobalt API."""
        self.log.start_operation("get_download_url", video_id)

        if not self.base_url:
            self.log.debug("Cobalt not configured (no base_url)")
            return None

        self.log.info(f"Fetching download URL (quality: {quality})...")

        try:
            url = f"{self.base_url}/api/json"
            payload = {
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "vQuality": quality,
                "filenamePattern": "basic",
            }
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }

            self.log.request("POST", url)
            response = httpx.post(
                url, json=payload, headers=headers, timeout=self.timeout, proxy=self._proxy
            )
            self.log.response(response.status_code, len(response.text))

            if response.status_code == 429:
                self.log.rate_limited("Cobalt")
                if self.handler:
                    self.handler.on_rate_limit("video_cobalt")
                    self._refresh_proxy()
                return None

            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "")
                self.log.debug(f"Cobalt response status: {status}")

                if status in ("stream", "redirect"):
                    download_url = data.get("url", "")
                    if download_url:
                        self.log.success(f"Got download URL ({quality}p)")
                        if self.handler:
                            self.handler.on_success()
                        return StreamInfo(
                            url=download_url,
                            quality=f"{quality}p",
                            format="mp4",
                            has_audio=True,
                        )
                    else:
                        self.log.debug("Cobalt returned empty URL")
                else:
                    self.log.debug(f"Cobalt status not ready: {status}")
            else:
                log_http_error(self.log, url, response.status_code)

        except httpx.TimeoutException:
            self.log.warning("Cobalt request timed out")
        except Exception as e:
            self.log.debug(f"Cobalt exception: {type(e).__name__}: {str(e)[:60]}")

        self.log.failure("Failed to get download URL")
        return None


class VideoFallbackChain:
    """
    Orchestrates video download with automatic fallback.

    Note: yt-dlp (Tier 1) is handled externally. This class handles Tiers 2-4.

    Usage:
        chain = VideoFallbackChain(config)
        result = chain.download(video_id, output_dir)
        if result.success:
            print(f"Downloaded to {result.file_path} via {result.tier_used.name}")
    """

    def __init__(
        self,
        config: "Config | None" = None,
        preferred_quality: str = "720p",
        cobalt_url: str | None = None,
    ):
        self.config = config
        self.preferred_quality = preferred_quality
        self.log = FallbackLogger("VIDEO_CHAIN")

        # Initialize rate limit handler for proxy/VPN support
        self.rate_limit_handler = get_rate_limit_handler(config)

        # Initialize fetchers with rate limit handler for proxy support
        self.invidious = InvidiousStreamFetcher(
            rate_limit_handler=self.rate_limit_handler
        )
        self.piped = PipedStreamFetcher(
            rate_limit_handler=self.rate_limit_handler
        )
        self.cobalt = CobaltFetcher(
            base_url=cobalt_url,
            rate_limit_handler=self.rate_limit_handler
        )

        # Stats tracking
        self.tier_stats: dict[VideoTier, dict[str, int]] = {
            tier: {"success": 0, "failure": 0} for tier in VideoTier
        }

    def download(
        self,
        video_id: str,
        output_dir: Path,
        preferred_quality: str | None = None,
        skip_tiers: list[VideoTier] | None = None,
    ) -> VideoResult:
        """
        Download video with automatic fallback.

        Args:
            video_id: YouTube video ID
            output_dir: Directory to save the video
            preferred_quality: Quality preference (e.g., "720p", "1080p")
            skip_tiers: Tiers to skip

        Returns:
            VideoResult with file path if successful
        """
        quality = preferred_quality or self.preferred_quality
        skip_tiers = skip_tiers or []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.log.start_operation("download", video_id)
        self.log.info(f"Starting video fallback chain (quality: {quality}, output: {output_dir})")
        if skip_tiers:
            self.log.debug(f"Skipping tiers: {[t.name for t in skip_tiers]}")

        tiers_tried = 0
        last_error = "No tiers available"

        # Tier 2: Invidious
        if VideoTier.INVIDIOUS not in skip_tiers:
            tiers_tried += 1
            self.log.info("=== Trying INVIDIOUS (Tier 2) ===")
            streams = self.invidious.get_streams(video_id, quality)
            if streams:
                combined_streams = [s for s in streams if s.has_audio and s.url]
                self.log.debug(f"Found {len(combined_streams)} combined streams to try")

                for i, stream in enumerate(combined_streams[:3]):  # Try up to 3
                    self.log.debug(f"Trying stream {i+1}: {stream}")
                    result = self._download_stream(
                        stream, video_id, output_dir, VideoTier.INVIDIOUS
                    )
                    if result.success:
                        self.tier_stats[VideoTier.INVIDIOUS]["success"] += 1
                        self.log.success(f"Downloaded via INVIDIOUS: {result.file_path}")
                        return result
                    last_error = result.error or "Download failed"

            self.tier_stats[VideoTier.INVIDIOUS]["failure"] += 1
            self.log.warning("INVIDIOUS failed")
        else:
            self.log.debug("INVIDIOUS skipped")

        # Tier 3: Piped
        if VideoTier.PIPED not in skip_tiers:
            tiers_tried += 1
            self.log.info("=== Trying PIPED (Tier 3) ===")
            streams = self.piped.get_streams(video_id, quality)
            if streams:
                video_streams = [s for s in streams if s.has_audio and s.url and "audio" not in s.quality.lower()]
                self.log.debug(f"Found {len(video_streams)} video streams to try")

                for i, stream in enumerate(video_streams[:3]):  # Try up to 3
                    self.log.debug(f"Trying stream {i+1}: {stream}")
                    result = self._download_stream(
                        stream, video_id, output_dir, VideoTier.PIPED
                    )
                    if result.success:
                        self.tier_stats[VideoTier.PIPED]["success"] += 1
                        self.log.success(f"Downloaded via PIPED: {result.file_path}")
                        return result
                    last_error = result.error or "Download failed"

            self.tier_stats[VideoTier.PIPED]["failure"] += 1
            self.log.warning("PIPED failed")
        else:
            self.log.debug("PIPED skipped")

        # Tier 4: Cobalt (if configured)
        if VideoTier.COBALT not in skip_tiers and self.cobalt.base_url:
            tiers_tried += 1
            self.log.info("=== Trying COBALT (Tier 4) ===")
            stream = self.cobalt.get_download_url(video_id, quality.replace("p", ""))
            if stream and stream.url:
                result = self._download_stream(
                    stream, video_id, output_dir, VideoTier.COBALT
                )
                if result.success:
                    self.tier_stats[VideoTier.COBALT]["success"] += 1
                    self.log.success(f"Downloaded via COBALT: {result.file_path}")
                    return result
                last_error = result.error or "Download failed"

            self.tier_stats[VideoTier.COBALT]["failure"] += 1
            self.log.warning("COBALT failed")
        elif VideoTier.COBALT not in skip_tiers:
            self.log.debug("COBALT not configured (no base_url)")

        self.log.failure(f"All tiers exhausted (tried: {tiers_tried})", reason=last_error[:100] if last_error else "Unknown")
        return VideoResult(
            success=False,
            tier_used=VideoTier.PIPED,
            error=f"All video stream sources failed: {last_error}",
        )

    def _download_stream(
        self,
        stream: StreamInfo,
        video_id: str,
        output_dir: Path,
        tier: VideoTier,
    ) -> VideoResult:
        """Download video from stream URL."""
        start_time = time.time()
        output_path = output_dir / f"{video_id}.{stream.format}"

        self.log.debug(f"Starting download: {stream.quality} {stream.format}")
        self.log.request("GET (stream)", stream.url)

        # Get proxy from handler for stream download
        proxy = get_proxy_for_httpx(self.rate_limit_handler)

        try:
            with httpx.stream(
                "GET",
                stream.url,
                timeout=httpx.Timeout(10.0, read=300.0),
                follow_redirects=True,
                proxy=proxy,
            ) as response:
                if response.status_code == 429:
                    self.log.rate_limited(f"{tier.name} stream")
                    if self.rate_limit_handler:
                        self.rate_limit_handler.on_rate_limit(f"video_stream_{tier.name}")
                    return VideoResult(
                        success=False,
                        tier_used=tier,
                        error="Rate limited (429)",
                    )

                if response.status_code != 200:
                    self.log.debug(f"Stream HTTP {response.status_code}")
                    return VideoResult(
                        success=False,
                        tier_used=tier,
                        error=f"HTTP {response.status_code}",
                    )

                # Get content length for progress
                total_size = int(response.headers.get("content-length", 0))
                self.log.debug(f"Content-Length: {total_size // 1024 // 1024}MB" if total_size else "Content-Length: unknown")
                downloaded = 0
                last_log = 0

                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=65536):
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Log progress for large files (every 10MB)
                        if total_size > 0 and (downloaded - last_log) >= 10 * 1024 * 1024:
                            pct = (downloaded / total_size) * 100
                            self.log.debug(
                                f"Progress: {pct:.1f}% ({downloaded // 1024 // 1024}MB)"
                            )
                            last_log = downloaded

            download_time = time.time() - start_time
            file_size_mb = downloaded / 1024 / 1024
            speed_mbps = (file_size_mb * 8) / download_time if download_time > 0 else 0

            self.log.success(
                f"Download complete: {file_size_mb:.1f}MB in {download_time:.1f}s "
                f"({speed_mbps:.1f} Mbps) | {stream.quality}"
            )

            # Report success to rate limit handler
            if self.rate_limit_handler:
                self.rate_limit_handler.on_success()

            return VideoResult(
                success=True,
                tier_used=tier,
                file_path=output_path,
                stream_info=stream,
                download_time=download_time,
            )

        except httpx.TimeoutException:
            self.log.warning(f"Download timeout after {time.time() - start_time:.1f}s")
            # Clean up partial file
            if output_path.exists():
                partial_size = output_path.stat().st_size
                self.log.debug(f"Cleaning up partial file: {partial_size // 1024 // 1024}MB")
                output_path.unlink()
            return VideoResult(
                success=False,
                tier_used=tier,
                error="Download timeout",
            )
        except Exception as e:
            self.log.error(f"Stream download failed", error=e)
            # Clean up partial file
            if output_path.exists():
                try:
                    partial_size = output_path.stat().st_size
                    self.log.debug(f"Cleaning up partial file: {partial_size // 1024 // 1024}MB")
                    output_path.unlink()
                except Exception:
                    pass
            return VideoResult(
                success=False,
                tier_used=tier,
                error=str(e),
            )

    def get_stats(self) -> dict[str, dict[str, int]]:
        """Get success/failure statistics for each tier."""
        return {tier.name: stats.copy() for tier, stats in self.tier_stats.items()}

    def reset_stats(self) -> None:
        """Reset tier statistics."""
        for tier in self.tier_stats:
            self.tier_stats[tier] = {"success": 0, "failure": 0}


# Convenience function
def download_video(
    video_id: str,
    output_dir: Path | str,
    preferred_quality: str = "720p",
) -> VideoResult:
    """
    Convenience function to download a video.

    Args:
        video_id: YouTube video ID
        output_dir: Directory to save the video
        preferred_quality: Quality preference

    Returns:
        VideoResult with file path if successful
    """
    chain = VideoFallbackChain(preferred_quality=preferred_quality)
    return chain.download(video_id, Path(output_dir))
