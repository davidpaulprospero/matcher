"""Cookie rotation manager for YouTube rate limit bypass.

Rotates between multiple YouTube account cookies to distribute rate limit
load and avoid account-level blocking. Default behavior is to rotate after
EVERY request (success or failure) to proactively distribute load.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path
import time
import threading
import logging

logger = logging.getLogger(__name__)

DEFAULT_COOKIES_DIR = "cookies"


@dataclass
class CookieAccount:
    """Represents a YouTube account's cookie configuration."""

    label: str
    cookies_path: Optional[Path] = None
    browser: Optional[str] = None  # firefox, chrome, edge, brave

    # Runtime state
    is_cooling_down: bool = False
    cooldown_until: float = 0.0
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0


class CookieManager:
    """Manages rotation between multiple YouTube account cookies.

    Thread-safe singleton that rotates cookies after every request
    to distribute load across accounts and avoid rate limits.

    Usage:
        # Initialize (typically done once at startup)
        config = {'enabled': True, 'accounts': [...]}
        manager = CookieManager.get_instance(config)

        # Get cookie args for yt-dlp
        args = manager.get_cookies_args()

        # Report results
        manager.report_success()  # or manager.report_rate_limit()
    """

    _instance: Optional["CookieManager"] = None
    _lock = threading.Lock()

    def __init__(self, config: dict, project_root: Path = None):
        """Initialize CookieManager with configuration.

        Args:
            config: Dictionary with keys:
                - enabled: bool
                - cookies_dir: str (relative to project_root)
                - rotate_on_success: bool (default True)
                - cooldown_seconds: int (default 300)
                - accounts: List[dict] with label, cookies_path, browser
            project_root: Path to resolve cookies_dir against
        """
        self.enabled = config.get("enabled", False)
        self.rotate_on_success = config.get("rotate_on_success", True)
        self.cooldown_seconds = config.get("cooldown_seconds", 300)
        self.accounts: List[CookieAccount] = []
        self.current_index = 0

        # Resolve cookies directory
        cookies_dir_name = config.get("cookies_dir", DEFAULT_COOKIES_DIR)
        if project_root:
            self.cookies_dir = Path(project_root) / cookies_dir_name
        else:
            self.cookies_dir = Path.cwd() / cookies_dir_name

        # Create cookies directory if needed
        if self.enabled and not self.cookies_dir.exists():
            self.cookies_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[cookies] Created directory: {self.cookies_dir}")

        # Load accounts
        for acc in config.get("accounts", []):
            cookies_path = None
            if acc.get("cookies_path"):
                cookies_path = self.cookies_dir / acc["cookies_path"]

            self.accounts.append(
                CookieAccount(
                    label=acc.get("label", f"account_{len(self.accounts)}"),
                    cookies_path=cookies_path,
                    browser=acc.get("browser"),
                )
            )

        # Log initialization
        if self.enabled and self.accounts:
            labels = [acc.label for acc in self.accounts]
            logger.info(
                f"[cookies] Initialized {len(self.accounts)} accounts: {', '.join(labels)}"
            )
            logger.info(
                f"[cookies] rotate_on_success={self.rotate_on_success}, cooldown={self.cooldown_seconds}s"
            )
        elif self.enabled:
            logger.warning("[cookies] Enabled but no accounts configured")

    @classmethod
    def get_instance(
        cls, config: dict = None, project_root: Path = None
    ) -> Optional["CookieManager"]:
        """Get or create singleton instance.

        Args:
            config: Configuration dict (required on first call)
            project_root: Path to resolve cookies_dir against

        Returns:
            CookieManager instance or None if not initialized
        """
        with cls._lock:
            if cls._instance is None and config:
                cls._instance = cls(config, project_root)
            return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def get_current_account(self) -> Optional[CookieAccount]:
        """Get current account, skipping those in cooldown.

        Returns:
            Current available CookieAccount, or None if all in cooldown
        """
        if not self.accounts:
            return None

        for _ in range(len(self.accounts)):
            account = self.accounts[self.current_index]

            if account.is_cooling_down:
                if time.time() > account.cooldown_until:
                    account.is_cooling_down = False
                    logger.debug(f"[cookies] '{account.label}' cooldown expired")
                else:
                    remaining = int(account.cooldown_until - time.time())
                    logger.debug(
                        f"[cookies] Skipping '{account.label}' ({remaining}s cooldown)"
                    )
                    self._rotate_index_internal()
                    continue

            logger.debug(f"[cookies] Using '{account.label}'")
            return account

        # All accounts in cooldown
        logger.warning(f"[cookies] All {len(self.accounts)} accounts in cooldown!")
        return None

    def get_cookies_args(self) -> List[str]:
        """Get yt-dlp cookie arguments for current account.

        Returns:
            List of yt-dlp arguments, e.g. ['--cookies', '/path/to/cookies.txt']
            or ['--cookies-from-browser', 'firefox']
        """
        account = self.get_current_account()
        if not account:
            return []

        if account.browser:
            return ["--cookies-from-browser", account.browser]
        elif account.cookies_path and account.cookies_path.exists():
            return ["--cookies", str(account.cookies_path)]
        elif account.cookies_path:
            logger.warning(f"[cookies] File not found: {account.cookies_path}")

        return []

    def report_success(self):
        """Report successful request, rotate to next account if enabled."""
        account = self.get_current_account()
        if account:
            account.request_count += 1
            account.success_count += 1
            logger.debug(
                f"[cookies] Success on '{account.label}' ({account.success_count}/{account.request_count})"
            )

        if self.rotate_on_success:
            self._rotate_index()

    def report_rate_limit(self):
        """Report rate limit, put account in cooldown and rotate."""
        account = self.get_current_account()
        if account:
            account.request_count += 1
            account.failure_count += 1
            account.is_cooling_down = True
            account.cooldown_until = time.time() + self.cooldown_seconds

            fail_rate = (
                (account.failure_count / account.request_count * 100)
                if account.request_count > 0
                else 0
            )
            logger.warning(
                f"[cookies] '{account.label}' rate limited! "
                f"{account.failure_count}/{account.request_count} ({fail_rate:.1f}%), "
                f"cooldown {self.cooldown_seconds}s"
            )

        self._rotate_index()

    def _rotate_index(self):
        """Move to next account (round-robin) with logging."""
        if self.accounts:
            old = self.accounts[self.current_index].label
            self.current_index = (self.current_index + 1) % len(self.accounts)
            new = self.accounts[self.current_index].label
            logger.info(f"[cookies] Rotated: {old} -> {new}")

    def _rotate_index_internal(self):
        """Move to next account without logging (used during cooldown skip)."""
        if self.accounts:
            self.current_index = (self.current_index + 1) % len(self.accounts)

    def list_available_cookies(self) -> List[str]:
        """List cookie files in cookies directory.

        Returns:
            List of .txt filenames in the cookies directory
        """
        if not self.cookies_dir.exists():
            return []
        return [f.name for f in self.cookies_dir.glob("*.txt")]

    def get_stats(self) -> Dict:
        """Get usage statistics.

        Returns:
            Dictionary with account statistics
        """
        return {
            "accounts": [
                {
                    "label": acc.label,
                    "requests": acc.request_count,
                    "successes": acc.success_count,
                    "failures": acc.failure_count,
                    "cooling_down": acc.is_cooling_down,
                }
                for acc in self.accounts
            ]
        }

    def log_summary(self):
        """Log usage summary at end of stage."""
        if not self.accounts:
            return

        total_requests = sum(acc.request_count for acc in self.accounts)
        total_successes = sum(acc.success_count for acc in self.accounts)
        total_failures = sum(acc.failure_count for acc in self.accounts)

        logger.info("=" * 50)
        logger.info("[cookies] ROTATION SUMMARY")
        logger.info(
            f"[cookies] Requests: {total_requests} | Success: {total_successes} | Failures: {total_failures}"
        )

        if total_requests > 0:
            logger.info(
                f"[cookies] Success rate: {total_successes / total_requests * 100:.1f}%"
            )
            logger.info("[cookies] Distribution:")
            for acc in self.accounts:
                pct = acc.request_count / total_requests * 100
                bar = "\u2588" * int(pct / 5)  # Unicode block character
                status = "COOLING" if acc.is_cooling_down else "ACTIVE"
                logger.info(f"[cookies]   {acc.label}: {bar} {pct:.1f}% [{status}]")
        logger.info("=" * 50)

    # === Parallel Worker Support ===

    def get_account_count(self) -> int:
        """Get number of configured accounts."""
        return len(self.accounts)

    def get_available_accounts(self) -> List[CookieAccount]:
        """Get list of accounts not in cooldown.

        Returns:
            List of CookieAccount objects that are available for use
        """
        available = []
        current_time = time.time()

        for account in self.accounts:
            # Check if cooldown expired
            if account.is_cooling_down and current_time > account.cooldown_until:
                account.is_cooling_down = False

            if not account.is_cooling_down:
                available.append(account)

        return available

    def get_account_by_index(self, index: int) -> Optional[CookieAccount]:
        """Get a specific account by index (for worker assignment).

        Args:
            index: Account index (0-based)

        Returns:
            CookieAccount or None if index invalid
        """
        if 0 <= index < len(self.accounts):
            return self.accounts[index]
        return None

    def get_cookies_args_for_account(self, account: CookieAccount) -> List[str]:
        """Get yt-dlp cookie arguments for a specific account.

        Args:
            account: The CookieAccount to get args for

        Returns:
            List of yt-dlp arguments
        """
        if not account:
            return []

        if account.browser:
            return ["--cookies-from-browser", account.browser]
        elif account.cookies_path and account.cookies_path.exists():
            return ["--cookies", str(account.cookies_path)]
        elif account.cookies_path:
            logger.warning(f"[cookies] File not found: {account.cookies_path}")

        return []

    def report_success_for_account(self, account: CookieAccount):
        """Report successful request for a specific account.

        Args:
            account: The account that succeeded
        """
        if account:
            with self._lock:
                account.request_count += 1
                account.success_count += 1
            logger.debug(
                f"[cookies] Worker success on '{account.label}' "
                f"({account.success_count}/{account.request_count})"
            )

    def report_rate_limit_for_account(self, account: CookieAccount) -> Optional[CookieAccount]:
        """Report rate limit for a specific account, return next available.

        Args:
            account: The account that was rate limited

        Returns:
            Next available CookieAccount, or None if all in cooldown
        """
        if account:
            with self._lock:
                account.request_count += 1
                account.failure_count += 1
                account.is_cooling_down = True
                account.cooldown_until = time.time() + self.cooldown_seconds

            fail_rate = (
                (account.failure_count / account.request_count * 100)
                if account.request_count > 0
                else 0
            )
            logger.warning(
                f"[cookies] Worker '{account.label}' rate limited! "
                f"{account.failure_count}/{account.request_count} ({fail_rate:.1f}%), "
                f"cooldown {self.cooldown_seconds}s"
            )

        # Find next available account
        available = self.get_available_accounts()
        if available:
            # Return first available that isn't the rate-limited one
            for acc in available:
                if acc != account:
                    logger.info(f"[cookies] Worker switching: {account.label} -> {acc.label}")
                    return acc
            # If only the same one available (shouldn't happen), return it
            return available[0] if available else None

        logger.warning(f"[cookies] All accounts in cooldown!")
        return None
