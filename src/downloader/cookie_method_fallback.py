"""
Cookie method fallback chain for YouTube downloads.

When yt-dlp gets 403/auth errors and retries exhaust on one cookie method,
automatically falls back to the next method in the chain:

    [browser:firefox] → [file:main.txt] → [file:backup1.txt] → [no-cookies] → GIVE UP

On success, proactively rotates to the next authenticated method (round-robin)
so no single cookie source gets rate-limited across keywords.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config.sections.download import DownloadConfig

logger = logging.getLogger(__name__)


@dataclass
class CookieMethod:
    """A single cookie method in the fallback chain."""
    kind: str  # "browser", "file", or "none"
    value: str  # browser name, file path, or ""
    label: str  # human-readable label for logging

    def get_cmd_args(self) -> List[str]:
        """Return yt-dlp command-line args for this method."""
        if self.kind == "browser":
            return ['--cookies-from-browser', self.value]
        elif self.kind == "file":
            return ['--cookies', self.value]
        else:
            return []


class CookieMethodFallback:
    """
    Manages an ordered fallback chain of cookie methods.

    Built automatically from existing config values (cookies_from_browser,
    cookie_rotation.cookie_files, cookies_path). Always appends 'none' as
    last resort.

    Usage:
        fallback = CookieMethodFallback(download_config)

        # Get current method's yt-dlp args
        args = fallback.get_cmd_args()

        # On auth error after retries exhausted
        if fallback.advance():
            # retry with new method
        else:
            # all methods exhausted, give up

        # On successful download
        fallback.mark_success()

        # Before next download
        fallback.reset_for_next_download()
    """

    def __init__(self, download_config: 'DownloadConfig'):
        self._chain, self._skipped_files = self._build_chain(download_config)
        self._current_index = 0
        self._last_success_index: Optional[int] = None

        # Log chain health summary
        health = self.get_chain_health()
        logger.info(
            f"Cookie fallback chain: {health['valid_methods']}/{health['total_methods']} methods available"
        )
        if len(self._chain) > 1:
            labels = [m.label for m in self._chain]
            logger.info(f"Cookie fallback chain order: {' -> '.join(labels)}")
        else:
            logger.debug(f"Cookie fallback chain: {self._chain[0].label if self._chain else 'empty'}")

    @staticmethod
    def _build_chain(config: 'DownloadConfig') -> tuple:
        """Build ordered method list from config values.

        Returns:
            Tuple of (chain, skipped_files) where skipped_files is a list
            of paths that were skipped due to not existing or not being readable.
        """
        chain: List[CookieMethod] = []
        seen_files: set = set()
        skipped_files: List[str] = []

        # 1. Browser cookies (highest priority)
        browser = getattr(config, 'cookies_from_browser', '')
        if browser:
            chain.append(CookieMethod(
                kind="browser",
                value=browser,
                label=f"browser:{browser}"
            ))

        # 2. Cookie rotation files (each file is a separate fallback method)
        rotation_config = getattr(config, 'cookie_rotation', None)
        if rotation_config:
            cookie_files = getattr(rotation_config, 'cookie_files', []) or []
            for cf in cookie_files:
                resolved = str(Path(cf))
                if resolved in seen_files:
                    continue
                p = Path(cf)
                if p.exists() and os.access(str(p), os.R_OK):
                    seen_files.add(resolved)
                    chain.append(CookieMethod(
                        kind="file",
                        value=str(cf),
                        label=f"file:{p.name}"
                    ))
                else:
                    skipped_files.append(str(cf))
                    logger.warning(f"Cookie file not found, skipping: {cf}")

        # 3. Static cookies_path (if not already covered by rotation files)
        cookies_path = getattr(config, 'cookies_path', '')
        if cookies_path:
            resolved = str(Path(cookies_path))
            if resolved not in seen_files:
                p = Path(cookies_path)
                if p.exists() and os.access(str(p), os.R_OK):
                    seen_files.add(resolved)
                    chain.append(CookieMethod(
                        kind="file",
                        value=cookies_path,
                        label=f"file:{p.name}"
                    ))
                else:
                    skipped_files.append(cookies_path)
                    logger.warning(f"Cookie file not found, skipping: {cookies_path}")

        # 4. No cookies (last resort — always present)
        chain.append(CookieMethod(
            kind="none",
            value="",
            label="no-cookies"
        ))

        return chain, skipped_files

    @property
    def current_method(self) -> CookieMethod:
        """Get the current cookie method."""
        return self._chain[self._current_index]

    @property
    def is_exhausted(self) -> bool:
        """True if all methods have been tried."""
        return self._current_index >= len(self._chain)

    @property
    def methods_remaining(self) -> int:
        """Number of methods left to try (including current)."""
        return max(0, len(self._chain) - self._current_index)

    def get_cmd_args(self) -> List[str]:
        """Return yt-dlp args for the current method."""
        if self.is_exhausted:
            return []
        return self._chain[self._current_index].get_cmd_args()

    def advance(self) -> bool:
        """
        Advance to the next method in the chain.

        Returns:
            True if there's a next method to try, False if exhausted.
        """
        prev = self._chain[self._current_index] if self._current_index < len(self._chain) else None
        self._current_index += 1

        if self.is_exhausted:
            logger.warning("Cookie method fallback exhausted — all methods tried")
            return False

        current = self._chain[self._current_index]
        logger.info(
            f"Cookie method fallback: {prev.label if prev else '?'} -> {current.label} "
            f"({self.methods_remaining} remaining)"
        )
        return True

    def mark_success(self) -> None:
        """Remember success and proactively rotate to next method for next download.

        Spreads load across cookie sources (round-robin) so no single source
        gets rate-limited. The error fallback chain still works from wherever
        the rotation puts us.
        """
        if not self.is_exhausted:
            current_label = self.current_method.label
            # Advance to next authenticated method for the next download
            next_idx = self._next_authenticated_index(self._current_index)
            self._last_success_index = next_idx
            next_label = self._chain[next_idx].label
            if next_idx != self._current_index:
                logger.info(f"Cookie rotation: {current_label} succeeded → next download starts at {next_label}")
            else:
                logger.debug(f"Cookie method success recorded: {current_label} (only method available)")

    def _next_authenticated_index(self, from_index: int) -> int:
        """Get next method index that has cookies, wrapping around.

        Skips 'no-cookies' (kind='none') since rotating to no-auth
        would be counterproductive for rate limit spreading.
        """
        n = len(self._chain)
        for i in range(1, n):
            candidate = (from_index + i) % n
            if self._chain[candidate].kind != "none":
                return candidate
        # Only one authenticated method (or only no-cookies), stay put
        return from_index

    def reset_for_next_download(self) -> None:
        """Reset to last working method (or first if none succeeded yet)."""
        if self._last_success_index is not None:
            self._current_index = self._last_success_index
        else:
            self._current_index = 0

    def get_status(self) -> dict:
        """Get status for logging/debugging."""
        return {
            "chain": [m.label for m in self._chain],
            "current_index": self._current_index,
            "current_method": self.current_method.label if not self.is_exhausted else "exhausted",
            "last_success": self._chain[self._last_success_index].label if self._last_success_index is not None else None,
            "is_exhausted": self.is_exhausted,
        }

    def get_chain_health(self) -> Dict[str, object]:
        """Get chain health summary.

        Returns:
            Dict with total_methods, valid_methods count, and skipped_files list.
        """
        return {
            "total_methods": len(self._chain) + len(self._skipped_files),
            "valid_methods": len(self._chain),
            "skipped_files": list(self._skipped_files),
        }
