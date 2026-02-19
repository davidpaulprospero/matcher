"""
Cookie rotation for YouTube rate limit evasion.

Rotates between multiple cookie files when hitting rate limits or
authentication errors. Each cookie file should be exported from a
different browser profile or YouTube account.

Also supports browser-based cookie extraction (US-143-005):
- Extract cookies directly from browser profiles
- Multiple browser support (Chrome, Firefox, Edge, Safari, etc.)
- Profile-specific cookie extraction
- Automatic cookie freshness validation and refresh
"""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..config.sections.download import CookieRotationConfig

logger = logging.getLogger(__name__)


class BrowserCookieExtractor:
    """
    Extract cookies directly from browser profiles (US-143-005).

    Supports multiple browsers and profiles:
    - Chrome, Firefox, Edge, Safari, Opera, Brave
    - Profile-specific extraction
    - Domain filtering (e.g., only YouTube cookies)

    Uses browser-specific methods:
    - Chrome/Edge/Brave: Copy SQLite database and query with Python
    - Firefox: JSON file or SQLite
    - Safari: Uses safaridriver or sqlite3

    Usage:
        extractor = BrowserCookieExtractor(timeout=30)
        cookies = extractor.extract_cookies(browser="chrome", profile="Default", domain=".youtube.com")
    """

    # Browser to OS-specific cookie database paths
    BROWSER_PATHS = {
        "chrome": {
            "linux": "~/.config/google-chrome/Default/Cookies",
            "darwin": "~/Library/Application Support/Google/Chrome/Default/Cookies",
            "win32": "%LOCALAPPDATA%/Google/Chrome/User Data/Default/Cookies",
        },
        "edge": {
            "linux": "~/.config/microsoft-edge/Default/Cookies",
            "darwin": "~/Library/Application Support/Microsoft Edge/Default/Cookies",
            "win32": "%LOCALAPPDATA%/Microsoft/Edge/User Data/Default/Cookies",
        },
        "brave": {
            "linux": "~/.config/BraveSoftware/Brave-Browser/Default/Cookies",
            "darwin": "~/Library/Application Support/BraveSoftware/Brave-Browser/Default/Cookies",
            "win32": "%LOCALAPPDATA%/BraveSoftware/Brave-Browser/User Data/Default/Cookies",
        },
        "firefox": {
            "linux": "~/.mozilla/firefox",
            "darwin": "~/Library/Application Support/Firefox/Profiles",
            "win32": "%APPDATA%/Mozilla/Firefox/Profiles",
        },
        "opera": {
            "linux": "~/.config/opera/Cookies",
            "darwin": "~/Library/Application Support/com.operasoftware.Opera/Cookies",
            "win32": "%APPDATA%/Opera Software/Opera Stable/Cookies",
        },
    }

    def __init__(self, timeout: int = 30):
        """
        Initialize browser cookie extractor.

        Args:
            timeout: Maximum seconds to wait for extraction
        """
        self.timeout = timeout

    def extract_cookies(
        self,
        browser: str,
        profile: str = "Default",
        domain: str = ".youtube.com",
    ) -> Optional[str]:
        """
        Extract cookies from specified browser profile.

        Args:
            browser: Browser name (chrome, firefox, edge, brave, opera, safari)
            profile: Profile name (e.g., "Default", "Profile1")
            domain: Domain filter (e.g., ".youtube.com" for YouTube cookies)

        Returns:
            Path to extracted cookie file in Netscape format, or None if extraction failed
        """
        logger.info(f"Extracting cookies from {browser} profile '{profile}' for domain '{domain}'")

        # Dispatch to browser-specific extraction
        try:
            if browser.lower() == "chrome":
                return self._extract_chrome(profile, domain)
            elif browser.lower() == "firefox":
                return self._extract_firefox(profile, domain)
            elif browser.lower() == "edge":
                return self._extract_edge(profile, domain)
            elif browser.lower() == "brave":
                return self._extract_brave(profile, domain)
            elif browser.lower() == "opera":
                return self._extract_opera(profile, domain)
            elif browser.lower() == "safari":
                return self._extract_safari(profile, domain)
            else:
                logger.warning(f"Unsupported browser: {browser}")
                return None
        except Exception as e:
            logger.error(f"Failed to extract cookies from {browser}: {e}")
            return None

    def _get_browser_cookie_path(self, browser: str, profile: str = "Default") -> Optional[Path]:
        """Get the browser cookie database path for the current OS."""
        import platform

        os_name = platform.system().lower()
        browser_key = browser.lower()

        if browser_key not in self.BROWSER_PATHS:
            return None

        path_template = self.BROWSER_PATHS[browser_key].get(os_name)
        if not path_template:
            return None

        # Expand path
        path_str = os.path.expanduser(os.path.expandvars(path_template))
        base_path = Path(path_str)

        # For Firefox, we need to find the profile directory
        if browser_key == "firefox":
            if not base_path.exists():
                return None
            # Find profile directory matching name pattern
            for item in base_path.iterdir():
                if item.is_dir() and (profile in item.name or item.name.endswith(".default")):
                    cookies = item / "cookies.sqlite"
                    if cookies.exists():
                        return cookies
            return None

        # For other browsers with profile subdirectory
        if profile != "Default" and browser_key in ("chrome", "edge", "brave"):
            # Check for profile-specific path
            profile_path = Path(str(base_path).replace("Default", profile))
            if profile_path.exists():
                cookies = profile_path / "Cookies"
                if cookies.exists():
                    return cookies

        return base_path if base_path.exists() else None

    def _extract_chrome(self, profile: str, domain: str) -> Optional[str]:
        """Extract cookies from Chrome/Chromium."""
        cookies_path = self._get_browser_cookie_path("chrome", profile)
        if not cookies_path:
            logger.warning(f"Chrome cookie database not found")
            return None

        return self._extract_from_sqlite(cookies_path, domain)

    def _extract_edge(self, profile: str, domain: str) -> Optional[str]:
        """Extract cookies from Edge."""
        cookies_path = self._get_browser_cookie_path("edge", profile)
        if not cookies_path:
            logger.warning(f"Edge cookie database not found")
            return None

        return self._extract_from_sqlite(cookies_path, domain)

    def _extract_brave(self, profile: str, domain: str) -> Optional[str]:
        """Extract cookies from Brave."""
        cookies_path = self._get_browser_cookie_path("brave", profile)
        if not cookies_path:
            logger.warning(f"Brave cookie database not found")
            return None

        return self._extract_from_sqlite(cookies_path, domain)

    def _extract_firefox(self, profile: str, domain: str) -> Optional[str]:
        """Extract cookies from Firefox."""
        cookies_path = self._get_browser_cookie_path("firefox", profile)
        if not cookies_path:
            logger.warning(f"Firefox cookie database not found")
            return None

        return self._extract_from_sqlite(cookies_path, domain)

    def _extract_opera(self, profile: str, domain: str) -> Optional[str]:
        """Extract cookies from Opera."""
        cookies_path = self._get_browser_cookie_path("opera", profile)
        if not cookies_path:
            logger.warning(f"Opera cookie database not found")
            return None

        return self._extract_from_sqlite(cookies_path, domain)

    def _extract_safari(self, profile: str, domain: str) -> Optional[str]:
        """Extract cookies from Safari (macOS only)."""
        import platform

        if platform.system() != "Darwin":
            logger.warning("Safari cookie extraction only supported on macOS")
            return None

        # Try using sqlite3 directly
        safari_cookies = Path.home() / "Library" / "Cookies" / "Cookies.binarycookies"
        if not safari_cookies.exists():
            logger.warning("Safari cookie database not found")
            return None

        # For now, return None as binary format requires special parsing
        # Could implement binarycookies parsing if needed
        logger.warning("Safari binarycookies format not yet supported")
        return None

    def _extract_from_sqlite(self, cookies_db: Path, domain: str) -> Optional[str]:
        """
        Extract cookies from SQLite database to Netscape format.

        Args:
            cookies_db: Path to cookies.sqlite or Cookies file
            domain: Domain filter

        Returns:
            Path to temporary cookie file in Netscape format
        """
        # Copy database to avoid locking issues
        temp_dir = Path(tempfile.mkdtemp(prefix="cookies_"))
        temp_db = temp_dir / "cookies_copy.sqlite"

        try:
            shutil.copy2(cookies_db, temp_db)
        except (OSError, IOError) as e:
            logger.warning(f"Could not copy cookie database: {e}")
            return None

        # Extract using Python's sqlite3
        import sqlite3

        try:
            conn = sqlite3.connect(temp_db)
            cursor = conn.cursor()

            # Query cookies for domain
            # Table structure varies: older Chrome uses 'cookies', newer uses 'network'
            tables = ["cookies", "network"]
            cookies_data = None

            for table in tables:
                try:
                    cursor.execute(
                        f"SELECT host, name, value, path, expires, isSecure FROM {table} "
                        f"WHERE host LIKE ?",
                        (f"%{domain}",),
                    )
                    cookies_data = cursor.fetchall()
                    if cookies_data:
                        break
                except sqlite3.OperationalError:
                    continue

            conn.close()

            if not cookies_data:
                logger.debug(f"No cookies found for domain {domain}")
                return None

            # Write to Netscape format
            cookie_file = temp_dir / "cookies.txt"
            with open(cookie_file, "w", encoding="utf-8") as f:
                f.write("# Netscape HTTP Cookie File\n")
                f.write("# Generated by Matcher\n\n")

                for host, name, value, path, expires, is_secure in cookies_data:
                    # Convert expires (Unix timestamp) to Netscape format
                    # 0 = session cookie
                    expiry = int(expires) if expires else 0
                    secure = "TRUE" if is_secure else "FALSE"

                    # Netscape format: domain flag path secure expiry name value
                    f.write(f"{host}\tTRUE\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")

            logger.info(f"Extracted {len(cookies_data)} cookies to {cookie_file}")
            return str(cookie_file)

        except Exception as e:
            logger.error(f"Failed to extract cookies from SQLite: {e}")
            return None
        finally:
            # Cleanup temp database
            try:
                temp_db.unlink()
            except OSError:
                pass

    def get_available_browsers(self) -> List[str]:
        """Check which browsers have cookie databases available."""
        available = []

        for browser in self.BROWSER_PATHS.keys():
            if self._get_browser_cookie_path(browser):
                available.append(browser)

        logger.debug(f"Available browsers with cookies: {available}")
        return available

    def get_browser_profiles(self, browser: str) -> List[str]:
        """Get available profiles for a given browser."""
        import platform

        os_name = platform.system().lower()
        browser_key = browser.lower()

        if browser_key not in self.BROWSER_PATHS:
            return []

        path_template = self.BROWSER_PATHS[browser_key].get(os_name)
        if not path_template:
            return []

        base_path = Path(os.path.expanduser(os.path.expandvars(path_template)))

        if not base_path.exists():
            return []

        if browser_key == "firefox":
            # Firefox profiles are directories
            profiles = []
            for item in base_path.iterdir():
                if item.is_dir() and (".default" in item.name or ".release" in item.name):
                    profiles.append(item.name)
            return profiles
        else:
            # Other browsers typically have Default + Profile1, Profile2, etc.
            # For now, just return Default
            return ["Default"]


class CookieRotator:
    """
    Manages rotation between multiple cookie files for YouTube downloads.

    Features:
    - Multiple rotation strategies (on_error, round_robin, random)
    - Health tracking: success/failure rates per cookie for intelligent prioritization
    - Cooldown tracking to avoid reusing rate-limited cookies too soon
    - Error pattern matching for automatic rotation triggers
    - Session-level rotation limits
    - Startup validation of cookie file existence and readability

    Usage:
        rotator = CookieRotator(config.download.cookie_rotation)
        cookie_path = rotator.get_current_cookie()

        # On success:
        rotator.mark_success(cookie_path)

        # On error:
        if rotator.should_rotate(error_message):
            new_cookie = rotator.rotate()
    """

    def __init__(self, config: 'CookieRotationConfig', continue_on_partial: bool = True):
        """
        Initialize the cookie rotator.

        Args:
            config: CookieRotationConfig with rotation settings
            continue_on_partial: If True (default), allow startup with some invalid cookies.
                                 If False, raise ValueError when any cookie is invalid.
        """
        self.config = config
        self._current_index = 0
        self._rotation_count = 0
        self._continue_on_partial = continue_on_partial

        # Track failed cookies with timestamps for cooldown
        # {cookie_path: timestamp_when_failed}
        self._failed_cookies: Dict[str, float] = {}

        # Health tracking: success/failure counts per cookie (US-93-010)
        # {cookie_path: {"success": int, "failure": int}}
        self._cookie_health: Dict[str, Dict[str, int]] = {}

        # Minimum attempts before evaluating health (prevents premature removal)
        self._health_min_attempts = getattr(config, 'health_min_attempts', 5)

        # Success rate threshold for auto-removal (US-93-010)
        # Cookies with success rate below this threshold after min_attempts will be removed
        self._success_rate_threshold = getattr(config, 'success_rate_threshold', 0.3)

        # Maximum consecutive failures before auto-removal
        self._max_consecutive_failures = getattr(config, 'max_consecutive_failures', 3)

        # Cookie expiration tracking (US-113-011)
        # Hours before expiry to trigger warning/rotation
        self._expiry_warning_threshold_hours = getattr(
            config, 'cookie_expiry_warning_threshold_hours', 24
        )
        # Enable proactive rotation before expiry
        self._rotate_before_expiry = getattr(config, 'rotate_before_expiry', True)

        # US-136-005: Proactive rotation threshold
        # If 0, use expiry_warning_threshold_hours as default
        proactive_threshold = getattr(config, 'proactive_rotation_threshold_hours', 0)
        self._proactive_rotation_threshold_hours = (
            proactive_threshold if proactive_threshold > 0
            else self._expiry_warning_threshold_hours
        )

        # Cookie health monitoring (US-114-006)
        # Minimum success rate for health warnings (below this triggers warning, not auto-retire)
        self._cookie_min_success_rate = getattr(config, 'cookie_min_success_rate', 0.6)
        # Health check interval (check every N downloads)
        self._cookie_health_check_interval = getattr(config, 'cookie_health_check_interval', 10)
        # Track downloads since last health check
        self._downloads_since_health_check = 0
        # Track if health check warning was already logged (avoid spam)
        self._health_warning_logged: Dict[str, bool] = {}

        # Track cookie expiration times (parsed from cookie files)
        # {cookie_path: expiry_timestamp}
        self._cookie_expiry: Dict[str, float] = {}

        # Track when cookies were loaded (for age calculation)
        # {cookie_path: load_timestamp}
        self._cookie_load_time: Dict[str, float] = {}

        # Track cookie age vs success correlation (US-113-011)
        # {cookie_path: [success_age_ranges]} - age in hours when successes occurred
        self._cookie_age_success: Dict[str, List[float]] = {}

        # Validate and filter to existing AND readable cookie files
        self._cookie_files, self._invalid_cookies = self._validate_cookie_files(
            config.cookie_files
        )

        # Log summary of validation
        total = len(config.cookie_files)
        valid = len(self._cookie_files)
        invalid = len(self._invalid_cookies)

        logger.info(
            f"Cookie rotator: {valid}/{total} cookies valid ({invalid} invalid)"
        )

        if invalid > 0 and self._continue_on_partial and self._invalid_cookies:
            reasons = "; ".join(
                f"{Path(p).name}: {r}" for p, r in self._invalid_cookies.items()
            )
            logger.warning(
                f"{invalid} cookie file(s) failed validation: {reasons}"
            )

        if not self._cookie_files:
            logger.warning("Cookie rotator has no valid cookie files!")

        # Raise error if continue_on_partial=False and some cookies are invalid
        if not continue_on_partial and invalid > 0:
            reasons = "; ".join([f"{p}: {r}" for p, r in self._invalid_cookies.items()])
            raise ValueError(f"Cookie validation failed: {reasons}")

        # US-143-005: Browser cookie extraction and freshness validation
        # Initialize browser cookie extractor
        self._browser_extractor: Optional[BrowserCookieExtractor] = None
        self._extracted_cookies: Dict[str, str] = {}  # {browser_profile_key: cookie_path}
        self._browser_cookie_timestamps: Dict[str, float] = {}  # When cookie was extracted

        # Cookie freshness validation setting
        self._cookie_freshness_validation = getattr(config, 'cookie_freshness_validation', True)
        # Auto-refresh expired cookies
        self._auto_refresh_cookies = getattr(config, 'auto_refresh_cookies', True)
        # Extraction timeout
        self._browser_extraction_timeout = getattr(config, 'browser_extraction_timeout', 30)
        # Freshness threshold for extracted cookies (US-144-005)
        self._freshness_threshold_minutes = getattr(config, 'freshness_threshold_minutes', 5)

        # Browser profiles from config
        self._browser_profiles = getattr(config, 'browser_profiles', [])
        # Browser priority order
        self._browser_priority_order = getattr(config, 'browser_priority_order', [])

        # Auto-extract cookies from browsers if configured (US-143-005)
        auto_extract = getattr(config, 'auto_extract_cookies', False)
        if auto_extract and self._browser_profiles:
            self._auto_extract_browser_cookies()

    def _auto_extract_browser_cookies(self) -> None:
        """
        Automatically extract cookies from configured browser profiles (US-143-005).

        Adds extracted cookies to the rotation pool.
        """
        if not self._browser_profiles:
            logger.debug("No browser profiles configured for auto-extraction")
            return

        # Initialize extractor
        if self._browser_extractor is None:
            self._browser_extractor = BrowserCookieExtractor(timeout=self._browser_extraction_timeout)

        # Sort profiles by browser priority
        sorted_profiles = sorted(
            self._browser_profiles,
            key=lambda p: (
                self._browser_priority_order.index(p.get('browser', ''))
                if p.get('browser') in self._browser_priority_order
                else len(self._browser_priority_order)
            )
        )

        for profile_config in sorted_profiles:
            browser = profile_config.get('browser', '')
            profile = profile_config.get('profile', 'Default')
            domain = profile_config.get('domain', '.youtube.com')

            if not browser:
                logger.warning(f"Browser profile missing browser name: {profile_config}")
                continue

            profile_key = f"{browser}:{profile}"
            logger.info(f"Auto-extracting cookies from {profile_key} for {domain}")

            try:
                cookie_path = self._browser_extractor.extract_cookies(
                    browser=browser,
                    profile=profile,
                    domain=domain,
                )

                if cookie_path and Path(cookie_path).exists():
                    # Add to rotation pool
                    if cookie_path not in self._cookie_files:
                        self._cookie_files.append(cookie_path)
                        self._extracted_cookies[profile_key] = cookie_path
                        self._browser_cookie_timestamps[cookie_path] = time.time()
                        logger.info(f"Added extracted cookie to rotation pool: {profile_key}")
                    else:
                        logger.debug(f"Cookie already in pool: {cookie_path}")
                else:
                    logger.warning(f"Failed to extract cookies from {profile_key}")

            except Exception as e:
                logger.error(f"Error extracting cookies from {profile_key}: {e}")

    def _refresh_browser_cookie(self, cookie_path: str) -> Optional[str]:
        """
        Refresh a cookie by re-extracting from the browser (US-143-005).

        Args:
            cookie_path: Path to the expired cookie file

        Returns:
            Path to new cookie file, or None if refresh failed
        """
        if not self._auto_refresh_cookies:
            return None

        # Find which browser profile this cookie came from
        profile_key = None
        for key, path in self._extracted_cookies.items():
            if path == cookie_path:
                profile_key = key
                break

        if not profile_key:
            logger.debug(f"Cookie not from browser extraction: {cookie_path}")
            return None

        # Parse browser and profile
        parts = profile_key.split(':')
        if len(parts) != 2:
            return None

        browser, profile = parts

        # Re-extract
        if self._browser_extractor is None:
            self._browser_extractor = BrowserCookieExtractor(timeout=self._browser_extraction_timeout)

        try:
            new_cookie_path = self._browser_extractor.extract_cookies(
                browser=browser,
                profile=profile,
                domain=".youtube.com",
            )

            if new_cookie_path and Path(new_cookie_path).exists():
                # Update tracking
                self._extracted_cookies[profile_key] = new_cookie_path
                self._browser_cookie_timestamps[new_cookie_path] = time.time()

                # Remove old cookie from rotation if still there
                if cookie_path in self._cookie_files:
                    self._cookie_files.remove(cookie_path)

                # Add new cookie
                if new_cookie_path not in self._cookie_files:
                    self._cookie_files.append(new_cookie_path)

                logger.info(f"Refreshed cookie from {profile_key}")
                return new_cookie_path

        except Exception as e:
            logger.error(f"Failed to refresh cookie from {profile_key}: {e}")

        return None

    def validate_cookie_freshness(self, cookie_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate cookie freshness before use (US-143-005).

        Checks if cookie is expired or approaching expiration.

        Args:
            cookie_path: Path to cookie file

        Returns:
            Tuple of (is_fresh, reason). If is_fresh is False, reason explains why.
        """
        if not self._cookie_freshness_validation:
            return True, None

        return self._check_cookie_expiry(cookie_path)

    def _validate_cookie_files(
        self, cookie_files: List[str]
    ) -> tuple[List[str], Dict[str, str]]:
        """
        Validate cookie file paths: check existence AND readability.

        Args:
            cookie_files: List of cookie file paths from config

        Returns:
            Tuple of:
            - List of validated, readable cookie file paths
            - Dict of invalid paths mapped to failure reasons
        """
        valid_files: List[str] = []
        invalid_files: Dict[str, str] = {}

        for cookie_path in cookie_files:
            path = Path(cookie_path)

            # Check existence
            if not path.exists():
                reason = "file not found"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
                continue

            # Check if it's a file (not directory)
            if not path.is_file():
                reason = "not a file"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
                continue

            # Check readability
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    # Read first 100 bytes to verify readability
                    f.read(100)
                valid_files.append(str(path))
                logger.debug(f"Cookie file validated: {cookie_path}")
            except PermissionError:
                reason = "permission denied"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
            except UnicodeDecodeError:
                # Cookie files should be plain text, but try binary read as fallback
                try:
                    with open(path, 'rb') as f:
                        f.read(100)
                    valid_files.append(str(path))
                    logger.debug(f"Cookie file validated (binary): {cookie_path}")
                except (OSError, IOError) as e:
                    reason = f"read error: {e}"
                    invalid_files[cookie_path] = reason
                    logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")
            except (OSError, IOError) as e:
                reason = f"read error: {e}"
                invalid_files[cookie_path] = reason
                logger.warning(f"Cookie file validation failed: {cookie_path} ({reason})")

        return valid_files, invalid_files

    def _parse_cookie_expiry(self, cookie_path: str) -> Tuple[Optional[float], Optional[str]]:
        """
        Parse cookie file and extract the earliest expiration timestamp.

        Supports multiple formats:
        1. Netscape cookie format:
           # Netscape HTTP Cookie File
           .youtube.com    TRUE    /   FALSE   1735689600  SID  ...
           (domain, flag, path, secure, expiry, name, value)

        2. JSON format (from browser extensions like "Get cookies.txt LOCALLY"):
           [{"domain": ".youtube.com", "expirationDate": 1735689600, ...}, ...]

        3. Line-delimited JSON (LDJSON/NDJSON):
           {"domain": ".youtube.com", "expirationDate": 1735689600, ...}
           {"domain": ".google.com", "expirationDate": 1738281600, ...}

        Returns:
            Tuple of (expiry_timestamp, error_message).
            Returns (None, None) if no expiry found or parse error.
        """
        try:
            path = Path(cookie_path)
            if not path.exists():
                return None, "file not found"

            min_expiry: Optional[float] = None

            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Try parsing as JSON first (for JSON/LDJSON formats)
            if content.strip().startswith('[') or content.strip().startswith('{'):
                min_expiry = self._parse_json_cookie_expiry(content)
                if min_expiry is not None:
                    return min_expiry, None

            # Fall back to Netscape format parsing
            for line in content.splitlines():
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                # Netscape format: domain, flag, path, secure, expiry, name, value
                if len(parts) >= 6:
                    try:
                        expiry = float(parts[4])
                        # 0 means session cookie (no expiry)
                        if expiry > 0:
                            if min_expiry is None or expiry < min_expiry:
                                min_expiry = expiry
                    except (ValueError, IndexError):
                        continue

            if min_expiry is not None:
                return min_expiry, None
            else:
                # No expiry found - could be all session cookies
                return None, "no expiry found (session cookies)"

        except Exception as e:
            return None, f"parse error: {e}"

    def _parse_json_cookie_expiry(self, content: str) -> Optional[float]:
        """
        Parse expiry from JSON or LDJSON cookie format.

        JSON format (from browser extensions):
        [{"domain": ".youtube.com", "expirationDate": 1735689600, ...}, ...]

        LDJSON format (line-delimited):
        {"domain": ".youtube.com", "expirationDate": 1735689600, ...}
        {"domain": ".google.com", "expirationDate": 1738281600, ...}

        Args:
            content: Raw file content

        Returns:
            Earliest expiry timestamp, or None if not found
        """
        import json

        min_expiry: Optional[float] = None

        # Try parsing as JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                for cookie in data:
                    if isinstance(cookie, dict):
                        expiry = cookie.get('expirationDate')
                        if expiry is not None:
                            try:
                                exp_float = float(expiry)
                                if min_expiry is None or exp_float < min_expiry:
                                    min_expiry = exp_float
                            except (ValueError, TypeError):
                                continue
                if min_expiry is not None:
                    return min_expiry
        except json.JSONDecodeError:
            pass

        # Try parsing as LDJSON (line-delimited JSON)
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cookie = json.loads(line)
                if isinstance(cookie, dict):
                    expiry = cookie.get('expirationDate')
                    if expiry is not None:
                        try:
                            exp_float = float(expiry)
                            if min_expiry is None or exp_float < min_expiry:
                                min_expiry = exp_float
                        except (ValueError, TypeError):
                            continue
            except json.JSONDecodeError:
                continue

        return min_expiry

    def _check_cookie_expiry(
        self,
        cookie_path: str,
        use_proactive_threshold: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if cookie is expired or approaching expiration.

        Args:
            cookie_path: Path to cookie file
            use_proactive_threshold: If True, use proactive rotation threshold instead of warning threshold

        Returns:
            Tuple of (is_valid, reason). If is_valid is False, reason explains why.
        """
        # Load expiry if not already tracked
        if cookie_path not in self._cookie_expiry:
            expiry, error = self._parse_cookie_expiry(cookie_path)
            if error and "session cookies" not in error:
                logger.debug(f"Cookie expiry parse warning for {cookie_path}: {error}")
            self._cookie_expiry[cookie_path] = expiry if expiry else float('inf')

        # Track load time for age correlation
        if cookie_path not in self._cookie_load_time:
            self._cookie_load_time[cookie_path] = time.time()

        expiry = self._cookie_expiry.get(cookie_path)
        if expiry is None or expiry == float('inf'):
            # No expiry (session cookie) - always valid
            return True, None

        current_time = time.time()
        time_until_expiry = expiry - current_time

        # Check if expired
        if time_until_expiry <= 0:
            return False, "expired"

        # Check if approaching expiration
        # Use proactive threshold if specified, otherwise use warning threshold
        threshold_hours = (
            self._proactive_rotation_threshold_hours
            if use_proactive_threshold
            else self._expiry_warning_threshold_hours
        )
        threshold_seconds = threshold_hours * 3600
        if time_until_expiry <= threshold_seconds:
            hours_left = time_until_expiry / 3600
            return True, f"expiring soon ({hours_left:.1f}h left)"

        return True, None

    def _is_expiring_soon(
        self,
        cookie_path: str,
        use_proactive_threshold: bool = True
    ) -> bool:
        """Check if cookie is approaching expiration threshold.

        Args:
            cookie_path: Path to cookie file
            use_proactive_threshold: If True (default), use proactive rotation threshold.
                                    If False, use warning threshold.
        """
        is_valid, reason = self._check_cookie_expiry(
            cookie_path,
            use_proactive_threshold=use_proactive_threshold
        )
        return reason is not None and "expiring soon" in reason

    def _is_extracted_cookie_stale(self, cookie_path: str) -> bool:
        """
        Check if extracted cookie is too old (US-144-005).

        This checks how recently the cookie was EXTRACTED from the browser,
        not when the cookie itself expires. If the extracted cookie is older
        than freshness_threshold_minutes, it's considered stale.

        Args:
            cookie_path: Path to cookie file

        Returns:
            True if cookie was extracted and is now stale
        """
        # Only check extracted cookies
        if cookie_path not in self._browser_cookie_timestamps:
            return False

        extract_time = self._browser_cookie_timestamps.get(cookie_path)
        if extract_time is None:
            return False

        age_minutes = (time.time() - extract_time) / 60
        return age_minutes > self._freshness_threshold_minutes

    def get_cookie_age_hours(self, cookie_path: str) -> float:
        """Get the age of a cookie in hours (time since first loaded)."""
        if cookie_path not in self._cookie_load_time:
            self._cookie_load_time[cookie_path] = time.time()
        return (time.time() - self._cookie_load_time[cookie_path]) / 3600

    @property
    def is_enabled(self) -> bool:
        """Check if rotation is enabled and has valid cookies."""
        return self.config.enabled and len(self._cookie_files) > 0

    @property
    def available_cookies(self) -> int:
        """Get count of cookies not in cooldown."""
        return sum(1 for cf in self._cookie_files if self.is_available(cf))

    def _is_cookie_file_valid(self, cookie_path: str) -> bool:
        """
        Check if a cookie file still exists and is non-empty.

        Used for mid-session validation to detect files deleted or
        truncated after initial startup validation.

        Args:
            cookie_path: Path to cookie file

        Returns:
            True if file exists and has content, False otherwise
        """
        path = Path(cookie_path)
        if not path.exists():
            return False
        if not path.is_file():
            return False
        try:
            return path.stat().st_size > 0
        except OSError:
            return False

    def _remove_invalid_cookie(self, cookie_path: str, reason: str) -> None:
        """
        Remove a cookie file from the active list after mid-session invalidation.

        Args:
            cookie_path: Path to the invalid cookie file
            reason: Reason for removal (for logging)
        """
        if cookie_path in self._cookie_files:
            self._cookie_files.remove(cookie_path)
            self._invalid_cookies[cookie_path] = reason
            # Clean up cooldown tracking for removed cookie (prevents unbounded dict growth)
            self._failed_cookies.pop(cookie_path, None)
            logger.warning(f"Cookie file invalidated mid-session: {cookie_path} ({reason})")
            # Adjust current index if needed
            if self._current_index >= len(self._cookie_files):
                self._current_index = 0

    def get_current_cookie(self) -> Optional[str]:
        """
        Get the current active cookie file path.

        Performs mid-session validation to detect deleted or empty cookie files.

        Returns:
            Path to current cookie file, or None if no valid cookies
        """
        if not self._cookie_files:
            return None

        # Ensure index is valid
        if self._current_index >= len(self._cookie_files):
            self._current_index = 0

        current = self._cookie_files[self._current_index]

        # Mid-session validation: check if file still exists and is non-empty
        if not self._is_cookie_file_valid(current):
            reason = "deleted" if not Path(current).exists() else "empty file"
            self._remove_invalid_cookie(current, reason)
            # Try to find another valid cookie
            return self._find_valid_cookie()

        # If current cookie is in cooldown, try to find an available one
        if not self.is_available(current):
            available = self._find_available_cookie()
            if available:
                self._current_index = self._cookie_files.index(available)
                return available
            # All cookies in cooldown - return current anyway (will fail but that's expected)
            logger.warning("All cookies in cooldown, using current cookie anyway")

        # Cookie expiration check (US-113-011)
        is_valid, expiry_reason = self._check_cookie_expiry(current)
        if not is_valid:
            # Cookie is expired - try to refresh from browser if extracted (US-143-005)
            refreshed = self._refresh_browser_cookie(current)
            if refreshed:
                logger.info(f"Cookie expired, refreshed from browser: {Path(current).name}")
                # Use the refreshed cookie
                self._current_index = self._cookie_files.index(refreshed)
                return refreshed

            # Refresh failed - remove expired cookie and find another

        # Check if extracted cookie is too old (US-144-005)
        # This is about how recently the cookie was EXTRACTED, not its own expiry
        if self._is_extracted_cookie_stale(current):
            logger.info(f"Extracted cookie is stale (older than {self._freshness_threshold_minutes} min), refreshing")
            refreshed = self._refresh_browser_cookie(current)
            if refreshed:
                logger.info(f"Refreshed stale extracted cookie: {Path(current).name}")
                self._current_index = self._cookie_files.index(refreshed)
                return refreshed
            # If refresh fails, continue with current cookie (it may still work)
            logger.warning(f"Cookie expired: {Path(current).name}, using another")
            self._remove_invalid_cookie(current, "expired")
            # Find another valid cookie without calling rotate()
            return self._find_valid_cookie()

        # Proactive rotation for expiring soon cookies
        if self._rotate_before_expiry and self._expiry_warning_threshold_hours > 0:
            if self._is_expiring_soon(current):
                # Check if there's a fresher (not expiring soon) option available
                fresher_count = sum(
                    1 for cf in self._cookie_files
                    if cf != current and not self._is_expiring_soon(cf)
                )
                if fresher_count > 0:
                    logger.info(
                        f"Cookie expiring soon: {Path(current).name} ({expiry_reason}), "
                        f"proactively rotating to fresh cookie"
                    )
                    # Directly select another cookie without calling rotate() to avoid recursion
                    # Find a fresher cookie
                    for cf in self._cookie_files:
                        if cf != current and not self._is_expiring_soon(cf):
                            self._current_index = self._cookie_files.index(cf)
                            logger.debug(f"Proactive rotation selected: {Path(cf).name}")
                            return cf
                else:
                    # No fresher alternative - log warning about approaching expiry
                    logger.warning(
                        f"Cookie expiring soon: {Path(current).name} ({expiry_reason}), "
                        f"no fresher alternative available - will use anyway"
                    )

        return current

    def _find_valid_cookie(self) -> Optional[str]:
        """Find the first cookie that still exists, is non-empty, not expired, and not in cooldown."""
        # Remove any invalidated cookies first (deleted or empty)
        to_remove = []
        for cookie_path in self._cookie_files:
            if not self._is_cookie_file_valid(cookie_path):
                reason = "deleted" if not Path(cookie_path).exists() else "empty file"
                to_remove.append((cookie_path, reason))

        for cookie_path, reason in to_remove:
            self._remove_invalid_cookie(cookie_path, reason)

        # Also remove expired cookies (US-113-011)
        expired_to_remove = []
        for cookie_path in self._cookie_files:
            is_valid, reason = self._check_cookie_expiry(cookie_path)
            if not is_valid:
                expired_to_remove.append((cookie_path, reason))

        for cookie_path, reason in expired_to_remove:
            self._remove_invalid_cookie(cookie_path, reason)

        if not self._cookie_files:
            logger.warning("All cookie files deleted, empty, or expired — no cookies available")
            return None

        # Find available (not in cooldown)
        available = self._find_available_cookie()
        if available:
            self._current_index = self._cookie_files.index(available)
            return available

        # All valid cookies in cooldown — return first valid anyway
        self._current_index = 0
        return self._cookie_files[0]

    def _find_available_cookie(self) -> Optional[str]:
        """Find the first cookie not in cooldown."""
        for cookie_path in self._cookie_files:
            if self.is_available(cookie_path):
                return cookie_path
        return None

    def is_available(self, cookie_path: str) -> bool:
        """
        Check if a cookie is available (not in cooldown).

        Args:
            cookie_path: Path to cookie file

        Returns:
            True if cookie can be used, False if in cooldown
        """
        if cookie_path not in self._failed_cookies:
            return True

        failed_time = self._failed_cookies[cookie_path]
        elapsed = time.time() - failed_time

        if elapsed >= self.config.cooldown_seconds:
            # Cooldown expired, remove from failed list
            del self._failed_cookies[cookie_path]
            logger.debug(f"Cookie cooldown expired: {cookie_path}")
            return True

        remaining = self.config.cooldown_seconds - elapsed
        logger.debug(f"Cookie in cooldown ({remaining:.0f}s remaining): {cookie_path}")
        return False

    def should_rotate(self, error_message: str) -> bool:
        """
        Check if the error message should trigger cookie rotation.

        Args:
            error_message: Error string from yt-dlp or download process

        Returns:
            True if error matches rotation triggers
        """
        if not self.is_enabled:
            return False

        error_lower = error_message.lower()

        for pattern in self.config.rotate_on_errors:
            if pattern.lower() in error_lower:
                logger.debug(f"Error matches rotation trigger: {pattern}")
                return True

        return False

    def rotate(self) -> Optional[str]:
        """
        Rotate to the next cookie file.

        Marks current cookie as failed (enters cooldown) and switches
        to the next available cookie based on rotation strategy.

        Returns:
            Path to new cookie file, or None if rotation exhausted
        """
        if not self._cookie_files:
            return None

        # Check rotation limit
        if self.config.max_rotations_per_session > 0:
            if self._rotation_count >= self.config.max_rotations_per_session:
                logger.warning(
                    f"Max rotations reached ({self.config.max_rotations_per_session}), "
                    "cannot rotate further"
                )
                return None

        # Mark current cookie as failed
        current = self.get_current_cookie()
        if current:
            self.mark_failed(current)

        # Select next cookie based on strategy
        new_cookie = self._select_next_cookie()

        if new_cookie:
            self._rotation_count += 1
            logger.info(
                f"Rotated cookie ({self._rotation_count}): {Path(current).name if current else 'none'} "
                f"-> {Path(new_cookie).name}"
            )

        return new_cookie

    def _select_next_cookie(self) -> Optional[str]:
        """
        Select the next cookie based on rotation strategy.

        Returns:
            Path to next cookie, or None if all exhausted
        """
        strategy = self.config.rotation_strategy

        if strategy == "random":
            return self._select_random()
        elif strategy == "round_robin":
            return self._select_round_robin()
        elif strategy == "health":  # US-93-010: prioritize by health score
            return self._select_by_health()
        else:  # "on_error" - same as round_robin
            return self._select_round_robin()

    def _select_round_robin(self) -> Optional[str]:
        """Select next available cookie in sequence."""
        if not self._cookie_files:
            return None

        start_index = self._current_index
        tried = 0

        while tried < len(self._cookie_files):
            self._current_index = (self._current_index + 1) % len(self._cookie_files)
            candidate = self._cookie_files[self._current_index]

            # Mid-session validation: skip deleted/empty files
            if not self._is_cookie_file_valid(candidate):
                reason = "deleted" if not Path(candidate).exists() else "empty file"
                self._remove_invalid_cookie(candidate, reason)
                if not self._cookie_files:
                    return None
                # Adjust tried count since list shrank
                continue

            if self.is_available(candidate):
                return candidate

            tried += 1

        # All cookies exhausted
        logger.warning("All cookies in cooldown or exhausted")
        return None

    def _select_random(self) -> Optional[str]:
        """Select a random available cookie."""
        # Mid-session validation: filter out deleted/empty files
        to_remove = []
        for cf in self._cookie_files:
            if not self._is_cookie_file_valid(cf):
                reason = "deleted" if not Path(cf).exists() else "empty file"
                to_remove.append((cf, reason))
        for cf, reason in to_remove:
            self._remove_invalid_cookie(cf, reason)

        available = [cf for cf in self._cookie_files if self.is_available(cf)]

        if not available:
            logger.warning("No available cookies for random selection")
            return None

        selected = random.choice(available)
        self._current_index = self._cookie_files.index(selected)
        return selected

    def mark_success(self, cookie_path: str) -> None:
        """
        Mark a cookie as successful (US-93-010).

        Also tracks cookie age for success correlation analysis (US-113-011).
        Performs health checks at configured intervals (US-114-006).

        Args:
            cookie_path: Path to cookie file that succeeded
        """
        if cookie_path not in self._cookie_health:
            self._cookie_health[cookie_path] = {"success": 0, "failure": 0}

        self._cookie_health[cookie_path]["success"] += 1

        # Track cookie age on success (US-113-011)
        age_hours = self.get_cookie_age_hours(cookie_path)
        if cookie_path not in self._cookie_age_success:
            self._cookie_age_success[cookie_path] = []
        self._cookie_age_success[cookie_path].append(age_hours)

        # Clear from failed cookies on success
        if cookie_path in self._failed_cookies:
            del self._failed_cookies[cookie_path]

        # Track downloads and perform health check at intervals (US-114-006)
        self._downloads_since_health_check += 1
        if self._cookie_health_check_interval > 0:
            if self._downloads_since_health_check >= self._cookie_health_check_interval:
                self._perform_health_check()
                self._downloads_since_health_check = 0
        else:
            # Check on every download if interval is 0
            self._perform_health_check()
            self._downloads_since_health_check = 0

        logger.debug(f"Cookie marked successful: {cookie_path} "
                     f"(health: {self._cookie_health[cookie_path]}, age: {age_hours:.1f}h)")

    def mark_failed(self, cookie_path: str) -> None:
        """
        Mark a cookie as failed (enters cooldown).

        Args:
            cookie_path: Path to cookie file that failed
        """
        self._failed_cookies[cookie_path] = time.time()

        # Track failure for health scoring (US-93-010)
        if cookie_path not in self._cookie_health:
            self._cookie_health[cookie_path] = {"success": 0, "failure": 0}
        self._cookie_health[cookie_path]["failure"] += 1

        # Check for auto-removal of consistently failing cookies (US-93-010)
        self._check_and_remove_failing_cookie(cookie_path)

        logger.debug(f"Cookie marked failed (cooldown {self.config.cooldown_seconds}s): {cookie_path}")

    def _check_and_remove_failing_cookie(self, cookie_path: str) -> None:
        """
        Check if a cookie should be automatically removed due to poor health (US-93-010).

        Removes cookie if:
        - Has exceeded max consecutive failures, OR
        - Has success rate below threshold after minimum attempts
        """
        if cookie_path not in self._cookie_health:
            return

        health = self._cookie_health[cookie_path]
        total_attempts = health["success"] + health["failure"]

        # Check consecutive failures (if tracking)
        consecutive_failures = getattr(self, '_consecutive_failures', {}).get(cookie_path, 0)
        if consecutive_failures >= self._max_consecutive_failures:
            logger.warning(
                f"Cookie auto-removed due to {consecutive_failures} consecutive failures: {cookie_path}"
            )
            self._remove_invalid_cookie(cookie_path, f"consecutive_failures:{consecutive_failures}")
            return

        # Check success rate threshold after minimum attempts
        if total_attempts >= self._health_min_attempts:
            success_rate = health["success"] / total_attempts
            if success_rate < self._success_rate_threshold:
                logger.warning(
                    f"Cookie auto-removed due to low success rate: {success_rate:.1%} "
                    f"(below {self._success_rate_threshold:.0%} threshold) after {total_attempts} attempts: {cookie_path}"
                )
                self._remove_invalid_cookie(cookie_path, f"low_success_rate:{success_rate:.2f}")

    def _perform_health_check(self) -> None:
        """
        Perform periodic health check on all cookies (US-114-006).

        Logs warnings for cookies with success rate below cookie_min_success_rate
        threshold. This is a monitoring-only check that doesn't auto-retire -
        auto-retirement is handled by _check_and_remove_failing_cookie.
        """
        for cookie_path in list(self._cookie_files):
            if cookie_path not in self._cookie_health:
                continue

            health = self._cookie_health[cookie_path]
            total_attempts = health["success"] + health["failure"]

            # Only check cookies with enough attempts
            if total_attempts < self._health_min_attempts:
                continue

            success_rate = health["success"] / total_attempts

            # Log warning if below min success rate threshold (but don't auto-retire)
            if success_rate < self._cookie_min_success_rate:
                # Only log once to avoid spam
                if not self._health_warning_logged.get(cookie_path, False):
                    logger.warning(
                        f"Cookie health warning: {Path(cookie_path).name} success rate "
                        f"{success_rate:.1%} below minimum threshold {self._cookie_min_success_rate:.0%} "
                        f"({total_attempts} attempts)"
                    )
                    self._health_warning_logged[cookie_path] = True

        # Reset health warning flags for cookies that have improved
        for cookie_path in list(self._health_warning_logged.keys()):
            if cookie_path in self._cookie_health:
                health = self._cookie_health[cookie_path]
                total_attempts = health["success"] + health["failure"]
                if total_attempts > 0:
                    success_rate = health["success"] / total_attempts
                    if success_rate >= self._cookie_min_success_rate:
                        # Cookie has improved, allow logging again if it drops
                        self._health_warning_logged[cookie_path] = False

    def get_health_score(self, cookie_path: str) -> float:
        """
        Get health score for a cookie (0.0 to 1.0) (US-93-010).

        Returns:
            Health score based on success rate. Returns 0.5 for unknown cookies.
        """
        if cookie_path not in self._cookie_health:
            return 0.5  # Unknown cookies get neutral score

        health = self._cookie_health[cookie_path]
        total = health["success"] + health["failure"]

        if total == 0:
            return 0.5

        return health["success"] / total

    def get_comprehensive_health_score(self, cookie_path: str) -> float:
        """
        Get comprehensive health score combining success rate and expiry status.

        This provides a holistic view of cookie health by factoring in:
        - Historical success rate (70% weight)
        - Expiry proximity (30% weight) - cookies expiring soon get lower scores

        Args:
            cookie_path: Path to cookie file

        Returns:
            Health score from 0.0 to 1.0, where 1.0 is best
        """
        # Base score from success rate (70% weight)
        success_score = self.get_health_score(cookie_path)
        success_weight = 0.7

        # Expiry score (30% weight)
        expiry_weight = 0.3
        expiry_score = 1.0  # Default to full score

        # Check expiry status
        is_valid, reason = self._check_cookie_expiry(cookie_path)
        expiry = self._cookie_expiry.get(cookie_path)

        if expiry and expiry != float('inf'):
            current_time = time.time()
            hours_until_expiry = (expiry - current_time) / 3600

            if hours_until_expiry <= 0:
                # Already expired - no score
                expiry_score = 0.0
            elif hours_until_expiry <= 24:
                # Less than 24 hours - decreasing score
                expiry_score = max(0.0, hours_until_expiry / 24)
            elif hours_until_expiry <= 168:  # 7 days
                # Between 24 hours and 7 days - slight penalty
                expiry_score = 0.5 + (0.5 * min(1.0, (hours_until_expiry - 24) / (168 - 24)))
            # More than 7 days - full score

        # Combined score
        combined = (success_score * success_weight) + (expiry_score * expiry_weight)
        return combined

    def get_cookie_health_report(self) -> Dict:
        """Get detailed health report for all cookies (US-93-010)."""
        report = {}
        for cookie_path in self._cookie_files:
            health = self._cookie_health.get(cookie_path, {"success": 0, "failure": 0})
            total = health["success"] + health["failure"]
            success_rate = health["success"] / total if total > 0 else 0.0

            report[cookie_path] = {
                "success": health["success"],
                "failure": health["failure"],
                "total_attempts": total,
                "success_rate": success_rate,
                "health_score": self.get_health_score(cookie_path),
                "in_cooldown": not self.is_available(cookie_path),
            }

        return report

    def _select_by_health(self) -> Optional[str]:
        """
        Select the best available cookie based on health score (US-93-010).

        Returns:
            Path to highest-health available cookie, or None if all exhausted
        """
        available = [
            (cf, self.get_health_score(cf))
            for cf in self._cookie_files
            if self.is_available(cf)
        ]

        if not available:
            logger.warning("No available cookies for health-based selection")
            return None

        # Sort by health score descending (highest first)
        available.sort(key=lambda x: x[1], reverse=True)

        selected = available[0][0]
        score = available[0][1]

        logger.info(
            f"Cookie health-based selection: {Path(selected).name} "
            f"(health score: {score:.1%}, {len(available)} available)"
        )

        self._current_index = self._cookie_files.index(selected)
        return selected

    def reset(self) -> None:
        """Reset rotation state (clear cooldowns and counters, but preserve health for session continuity)."""
        self._current_index = 0
        self._rotation_count = 0
        self._failed_cookies.clear()
        logger.info("Cookie rotator reset (health data preserved for session)")

    def get_expiry_report(self) -> Dict:
        """Get expiry status report for all cookies (US-113-011)."""
        report = {}
        current_time = time.time()

        for cookie_path in self._cookie_files:
            # Ensure expiry is loaded
            is_valid, reason = self._check_cookie_expiry(cookie_path)
            expiry = self._cookie_expiry.get(cookie_path)

            age_hours = self.get_cookie_age_hours(cookie_path)
            success_ages = self._cookie_age_success.get(cookie_path, [])

            if expiry and expiry != float('inf'):
                hours_until_expiry = (expiry - current_time) / 3600
                report[cookie_path] = {
                    "age_hours": age_hours,
                    "expiry_hours_until": hours_until_expiry,
                    "expiry_timestamp": expiry,
                    "is_expired": not is_valid,
                    "is_expiring_soon": reason is not None and "expiring soon" in reason,
                    "expiry_status": reason,
                    "success_age_samples": success_ages[-5:] if success_ages else [],  # Last 5
                }
            else:
                # Session cookie (no expiry)
                report[cookie_path] = {
                    "age_hours": age_hours,
                    "expiry_hours_until": None,
                    "expiry_timestamp": None,
                    "is_expired": False,
                    "is_expiring_soon": False,
                    "expiry_status": "session_cookie",
                    "success_age_samples": success_ages[-5:] if success_ages else [],
                }

        return report

    def get_status(self) -> Dict:
        """
        Get current rotation status for logging/debugging.

        Returns:
            Dict with rotation status info including:
            - total: All configured cookie files (valid + invalid)
            - valid: Cookie files that passed validation (exist + readable)
            - invalid: Count of cookie files that failed validation
            - cooldown: Count of valid cookies currently in cooldown
            - exhausted: True if no cookies are available (all invalid or in cooldown)
            - health_report: Detailed health info per cookie (US-93-010)
            - expiry_report: Expiry status per cookie (US-113-011)
            - Plus legacy keys for backward compatibility
        """
        total = len(self._cookie_files) + len(self._invalid_cookies)
        valid = len(self._cookie_files)
        invalid = len(self._invalid_cookies)
        cooldown = valid - self.available_cookies
        exhausted = self.available_cookies == 0

        return {
            "enabled": self.is_enabled,
            "total": total,
            "valid": valid,
            "invalid": invalid,
            "cooldown": cooldown,
            "exhausted": exhausted,
            # Health tracking (US-93-010)
            "health_report": self.get_cookie_health_report(),
            "success_rate_threshold": self._success_rate_threshold,
            # Health monitoring (US-114-006)
            "cookie_min_success_rate": self._cookie_min_success_rate,
            "cookie_health_check_interval": self._cookie_health_check_interval,
            "downloads_since_health_check": self._downloads_since_health_check,
            # Expiry tracking (US-113-011)
            "expiry_report": self.get_expiry_report(),
            "expiry_warning_threshold_hours": self._expiry_warning_threshold_hours,
            "rotate_before_expiry": self._rotate_before_expiry,
            # Proactive rotation threshold (US-136-005)
            "proactive_rotation_threshold_hours": self._proactive_rotation_threshold_hours,
            # Browser cookie extraction (US-143-005)
            "browser_extraction": {
                "enabled": bool(self._browser_profiles),
                "browser_profiles": self._browser_profiles,
                "browser_priority_order": self._browser_priority_order,
                "extracted_cookies": self._extracted_cookies,
                "cookie_freshness_validation": self._cookie_freshness_validation,
                "auto_refresh_cookies": self._auto_refresh_cookies,
                "freshness_threshold_minutes": self._freshness_threshold_minutes,
            },
            # Legacy keys for backward compatibility
            "total_cookies": total,
            "valid_cookies": valid,
            "available_cookies": self.available_cookies,
            "current_cookie": self.get_current_cookie(),
            "rotation_count": self._rotation_count,
            "max_rotations": self.config.max_rotations_per_session,
            "strategy": self.config.rotation_strategy,
            "cookies_in_cooldown": list(self._failed_cookies.keys()),
            "invalid_cookies": dict(self._invalid_cookies),
        }

    def can_rotate(self) -> bool:
        """
        Check if rotation is possible.

        Returns:
            True if there are available cookies to rotate to
        """
        if not self.is_enabled:
            return False

        # Check rotation limit
        if self.config.max_rotations_per_session > 0:
            if self._rotation_count >= self.config.max_rotations_per_session:
                return False

        # Check if there are available cookies (besides current)
        return self.available_cookies > 1 or (
            self.available_cookies == 1 and
            not self.is_available(self.get_current_cookie() or "")
        )
