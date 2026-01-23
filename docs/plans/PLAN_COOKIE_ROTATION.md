# Plan: Cookie Rotation for YouTube Rate Limit Bypass

## Overview

Rotate between multiple YouTube account cookies to distribute rate limit load and avoid account-level blocking.

**Key Behavior:** Rotate after EVERY request (success or failure) to proactively distribute load before rate limits hit.

| Strategy | Behavior | Result |
|----------|----------|--------|
| Rotate on error only | Use same account until 429 | Account gets flagged, 5min cooldown |
| **Rotate every request** | Round-robin across accounts | Each account stays under radar |

With 3 accounts: A handles requests 1,4,7... B handles 2,5,8... C handles 3,6,9... → Each sees 1/3 traffic.

---

## Directory Structure

```
project_root/
├── cookies/                    # Cookie files (gitignored)
│   ├── main.txt
│   ├── backup1.txt
│   └── backup2.txt
├── config.yaml
└── src/downloader/
    └── cookie_manager.py       # NEW
```

---

## Config Schema

```yaml
download:
  cookie_rotation:
    enabled: true
    cookies_dir: "cookies"        # Relative to project root
    rotate_on_success: true       # DEFAULT: Rotate after EVERY request
    cooldown_seconds: 300         # 5 min cooldown on 429

    accounts:
      - label: "main"
        cookies_path: "main.txt"  # Relative to cookies_dir

      - label: "backup1"
        cookies_path: "backup1.txt"

      - label: "backup2"
        browser: "firefox"        # Live extraction (no file needed)
```

---

## Implementation

### New File: `src/downloader/cookie_manager.py`

```python
"""Cookie rotation manager for YouTube rate limit bypass."""

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
    """

    _instance: Optional['CookieManager'] = None
    _lock = threading.Lock()

    def __init__(self, config: dict, project_root: Path = None):
        self.enabled = config.get('enabled', False)
        self.rotate_on_success = config.get('rotate_on_success', True)
        self.cooldown_seconds = config.get('cooldown_seconds', 300)
        self.accounts: List[CookieAccount] = []
        self.current_index = 0

        # Resolve cookies directory
        cookies_dir_name = config.get('cookies_dir', DEFAULT_COOKIES_DIR)
        if project_root:
            self.cookies_dir = project_root / cookies_dir_name
        else:
            self.cookies_dir = Path.cwd() / cookies_dir_name

        # Create cookies directory if needed
        if self.enabled and not self.cookies_dir.exists():
            self.cookies_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[cookies] Created directory: {self.cookies_dir}")

        # Load accounts
        for acc in config.get('accounts', []):
            cookies_path = None
            if acc.get('cookies_path'):
                cookies_path = self.cookies_dir / acc['cookies_path']

            self.accounts.append(CookieAccount(
                label=acc.get('label', f'account_{len(self.accounts)}'),
                cookies_path=cookies_path,
                browser=acc.get('browser')
            ))

        # Log initialization
        if self.enabled and self.accounts:
            labels = [acc.label for acc in self.accounts]
            logger.info(f"[cookies] Initialized {len(self.accounts)} accounts: {', '.join(labels)}")
            logger.info(f"[cookies] rotate_on_success={self.rotate_on_success}, cooldown={self.cooldown_seconds}s")
        elif self.enabled:
            logger.warning("[cookies] Enabled but no accounts configured")

    @classmethod
    def get_instance(cls, config: dict = None, project_root: Path = None) -> Optional['CookieManager']:
        """Get or create singleton instance."""
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
        """Get current account, skipping those in cooldown."""
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
                    logger.debug(f"[cookies] Skipping '{account.label}' ({remaining}s cooldown)")
                    self._rotate_index()
                    continue

            logger.debug(f"[cookies] Using '{account.label}'")
            return account

        # All accounts in cooldown
        logger.warning(f"[cookies] All {len(self.accounts)} accounts in cooldown!")
        return None

    def get_cookies_args(self) -> List[str]:
        """Get yt-dlp cookie arguments for current account."""
        account = self.get_current_account()
        if not account:
            return []

        if account.browser:
            return ['--cookies-from-browser', account.browser]
        elif account.cookies_path and account.cookies_path.exists():
            return ['--cookies', str(account.cookies_path)]
        elif account.cookies_path:
            logger.warning(f"[cookies] File not found: {account.cookies_path}")

        return []

    def report_success(self):
        """Report successful request, rotate to next account."""
        account = self.get_current_account()
        if account:
            account.request_count += 1
            account.success_count += 1
            logger.debug(f"[cookies] Success on '{account.label}' ({account.success_count}/{account.request_count})")

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

            fail_rate = (account.failure_count / account.request_count * 100) if account.request_count > 0 else 0
            logger.warning(
                f"[cookies] '{account.label}' rate limited! "
                f"{account.failure_count}/{account.request_count} ({fail_rate:.1f}%), "
                f"cooldown {self.cooldown_seconds}s"
            )

        self._rotate_index()

    def _rotate_index(self):
        """Move to next account (round-robin)."""
        if self.accounts:
            old = self.accounts[self.current_index].label
            self.current_index = (self.current_index + 1) % len(self.accounts)
            new = self.accounts[self.current_index].label
            logger.info(f"[cookies] Rotated: {old} -> {new}")

    def list_available_cookies(self) -> List[str]:
        """List cookie files in cookies directory."""
        if not self.cookies_dir.exists():
            return []
        return [f.name for f in self.cookies_dir.glob("*.txt")]

    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            'accounts': [
                {
                    'label': acc.label,
                    'requests': acc.request_count,
                    'successes': acc.success_count,
                    'failures': acc.failure_count,
                    'cooling_down': acc.is_cooling_down,
                }
                for acc in self.accounts
            ]
        }

    def log_summary(self):
        """Log usage summary at end of stage."""
        total_requests = sum(acc.request_count for acc in self.accounts)
        total_successes = sum(acc.success_count for acc in self.accounts)
        total_failures = sum(acc.failure_count for acc in self.accounts)

        logger.info("=" * 50)
        logger.info("[cookies] ROTATION SUMMARY")
        logger.info(f"[cookies] Requests: {total_requests} | Success: {total_successes} | Failures: {total_failures}")

        if total_requests > 0:
            logger.info(f"[cookies] Success rate: {total_successes / total_requests * 100:.1f}%")
            logger.info("[cookies] Distribution:")
            for acc in self.accounts:
                pct = acc.request_count / total_requests * 100
                bar = "█" * int(pct / 5)
                status = "COOLING" if acc.is_cooling_down else "ACTIVE"
                logger.info(f"[cookies]   {acc.label}: {bar} {pct:.1f}% [{status}]")
        logger.info("=" * 50)
```

### Modify: `src/downloader/utils.py`

```python
def get_cookies_args(config: 'Config') -> List[str]:
    """Get yt-dlp cookie arguments, with rotation support."""

    # Check for cookie rotation
    rotation_config = getattr(config.download, 'cookie_rotation', None)
    if rotation_config and getattr(rotation_config, 'enabled', False):
        from .cookie_manager import CookieManager
        manager = CookieManager.get_instance(vars(rotation_config) if hasattr(rotation_config, '__dict__') else rotation_config)
        if manager:
            return manager.get_cookies_args()

    # Fallback to existing single-cookie logic
    # ... existing code ...
```

### Modify: `src/stages/download.py`

```python
from ..downloader.cookie_manager import CookieManager

# After successful download:
cm = CookieManager.get_instance()
if cm and cm.enabled:
    cm.report_success()

# On rate limit (in error handling):
if 'rate-limited' in stderr_lower or '429' in stderr:
    cm = CookieManager.get_instance()
    if cm and cm.enabled:
        cm.report_rate_limit()

# At end of stage:
cm = CookieManager.get_instance()
if cm and cm.enabled:
    cm.log_summary()
```

### Modify: `src/downloader/caption_fetcher.py`

Same integration as download.py for caption fetching.

### Add: `src/config/sections/download.py`

```python
@dataclass
class CookieRotationConfig:
    enabled: bool = False
    cookies_dir: str = "cookies"
    rotate_on_success: bool = True
    cooldown_seconds: int = 300
    accounts: List[Dict] = field(default_factory=list)
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `cookies/` | NEW directory (gitignored) |
| `src/downloader/cookie_manager.py` | NEW - CookieManager class |
| `src/downloader/utils.py` | Integrate CookieManager |
| `src/stages/download.py` | Report success/failure, log summary |
| `src/downloader/caption_fetcher.py` | Report success/failure |
| `src/config/sections/download.py` | Add CookieRotationConfig |
| `config.yaml` | Add cookie_rotation section |
| `.gitignore` | Add `cookies/` |

---

## Setup

### 1. Create cookies directory

```bash
mkdir cookies
echo "cookies/" >> .gitignore
```

### 2. Export cookies from YouTube accounts

Use browser extension "Get cookies.txt LOCALLY" to export from each account:

```
cookies/
├── main.txt      # Primary account
├── backup1.txt   # Second account
└── backup2.txt   # Third account
```

**Important:** Each file must be from a **different YouTube account**.

### 3. Or use browser extraction

```yaml
accounts:
  - label: "firefox"
    browser: "firefox"    # Works while browser open
  - label: "chrome"
    browser: "chrome"     # Browser must be closed
```

### 4. Configure

```yaml
download:
  cookie_rotation:
    enabled: true
    cookies_dir: "cookies"
    rotate_on_success: true
    cooldown_seconds: 300
    accounts:
      - label: "main"
        cookies_path: "main.txt"
      - label: "backup1"
        cookies_path: "backup1.txt"
      - label: "backup2"
        cookies_path: "backup2.txt"
```

---

## Example Log Output

```
[cookies] Initialized 3 accounts: main, backup1, backup2
[cookies] rotate_on_success=True, cooldown=300s

# Request 1: main → success → rotate
[cookies] Using 'main'
[cookies] Success on 'main' (1/1)
[cookies] Rotated: main -> backup1

# Request 2: backup1 → success → rotate
[cookies] Using 'backup1'
[cookies] Success on 'backup1' (1/1)
[cookies] Rotated: backup1 -> backup2

# Request 3: backup2 → success → rotate back to main
[cookies] Using 'backup2'
[cookies] Success on 'backup2' (1/1)
[cookies] Rotated: backup2 -> main

# ... round-robin continues ...

# Request 44: main hits rate limit
[cookies] 'main' rate limited! 1/15 (6.7%), cooldown 300s
[cookies] Rotated: main -> backup1

# Request 45: main in cooldown, skip to backup1
[cookies] Skipping 'main' (298s cooldown)
[cookies] Using 'backup1'

# End of stage summary
==================================================
[cookies] ROTATION SUMMARY
[cookies] Requests: 45 | Success: 42 | Failures: 3
[cookies] Success rate: 93.3%
[cookies] Distribution:
[cookies]   main: ██████ 33.3% [COOLING]
[cookies]   backup1: ██████ 33.3% [ACTIVE]
[cookies]   backup2: ██████ 33.3% [ACTIVE]
==================================================
```

---

## Validation

### Checklist

| Check | Verify |
|-------|--------|
| Rotation on every success | `Rotated: X -> Y` after each download |
| Round-robin pattern | main → backup1 → backup2 → main → ... |
| Equal distribution | Summary shows ~33% per account |
| Rate limit cooldown | `rate limited!` + `cooldown Ns` |
| Cooldown skip | `Skipping 'X' (Ns cooldown)` |

### Grep commands

```bash
# Check rotation happening
grep "\[cookies\] Rotated" output/logs/run_*.log | head -20

# Check rate limits
grep "\[cookies\].*rate limited" output/logs/run_*.log

# Check distribution
grep "\[cookies\].*Distribution" -A5 output/logs/run_*.log
```

---

## Unit Tests

### File: `tests/test_cookie_rotation.py`

```python
"""Tests for cookie rotation manager."""

import pytest
import time
from pathlib import Path
from src.downloader.cookie_manager import CookieManager, CookieAccount


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton between tests."""
    CookieManager.reset_instance()
    yield
    CookieManager.reset_instance()


@pytest.fixture
def cookies_dir(tmp_path):
    """Create temp cookies directory with test files."""
    cookies = tmp_path / "cookies"
    cookies.mkdir()
    for name in ["main.txt", "backup1.txt", "backup2.txt"]:
        (cookies / name).write_text("# Netscape cookie")
    return cookies


@pytest.fixture
def config(cookies_dir):
    """Standard 3-account config."""
    return {
        'enabled': True,
        'cookies_dir': str(cookies_dir),
        'rotate_on_success': True,
        'cooldown_seconds': 300,
        'accounts': [
            {'label': 'main', 'cookies_path': 'main.txt'},
            {'label': 'backup1', 'cookies_path': 'backup1.txt'},
            {'label': 'backup2', 'cookies_path': 'backup2.txt'},
        ]
    }


class TestInit:
    def test_loads_accounts(self, config):
        cm = CookieManager(config)
        assert len(cm.accounts) == 3
        assert cm.accounts[0].label == 'main'

    def test_empty_config(self):
        cm = CookieManager({})
        assert cm.enabled is False
        assert len(cm.accounts) == 0

    def test_browser_account(self):
        cm = CookieManager({'enabled': True, 'accounts': [{'label': 'ff', 'browser': 'firefox'}]})
        assert cm.accounts[0].browser == 'firefox'


class TestRotation:
    def test_cycles_through_accounts(self, config):
        cm = CookieManager(config)
        assert cm.get_current_account().label == 'main'
        cm._rotate_index()
        assert cm.get_current_account().label == 'backup1'
        cm._rotate_index()
        assert cm.get_current_account().label == 'backup2'
        cm._rotate_index()
        assert cm.get_current_account().label == 'main'  # Wraps

    def test_rotates_on_success(self, config):
        cm = CookieManager(config)
        assert cm.get_current_account().label == 'main'
        cm.report_success()
        assert cm.get_current_account().label == 'backup1'

    def test_no_rotate_when_disabled(self, config):
        config['rotate_on_success'] = False
        cm = CookieManager(config)
        cm.report_success()
        cm.report_success()
        assert cm.get_current_account().label == 'main'


class TestCooldown:
    def test_rate_limit_triggers_cooldown(self, config):
        cm = CookieManager(config)
        cm.report_rate_limit()
        assert cm.accounts[0].is_cooling_down is True
        assert cm.accounts[0].cooldown_until > time.time()

    def test_skips_cooling_account(self, config):
        cm = CookieManager(config)
        cm.accounts[0].is_cooling_down = True
        cm.accounts[0].cooldown_until = time.time() + 300
        assert cm.get_current_account().label == 'backup1'

    def test_cooldown_expiry(self, config):
        cm = CookieManager(config)
        cm.accounts[0].is_cooling_down = True
        cm.accounts[0].cooldown_until = time.time() - 1  # Expired
        assert cm.get_current_account().label == 'main'
        assert cm.accounts[0].is_cooling_down is False

    def test_all_cooling_returns_none(self, config):
        cm = CookieManager(config)
        for acc in cm.accounts:
            acc.is_cooling_down = True
            acc.cooldown_until = time.time() + 300
        assert cm.get_current_account() is None


class TestCookiesArgs:
    def test_returns_cookies_flag(self, config, cookies_dir):
        cm = CookieManager(config)
        args = cm.get_cookies_args()
        assert args == ['--cookies', str(cookies_dir / 'main.txt')]

    def test_returns_browser_flag(self):
        cm = CookieManager({'enabled': True, 'accounts': [{'browser': 'firefox'}]})
        assert cm.get_cookies_args() == ['--cookies-from-browser', 'firefox']

    def test_missing_file_returns_empty(self, config):
        config['accounts'][0]['cookies_path'] = 'nonexistent.txt'
        cm = CookieManager(config)
        assert cm.get_cookies_args() == []


class TestStatistics:
    def test_tracks_success(self, config):
        cm = CookieManager(config)
        cm.report_success()
        assert cm.accounts[0].success_count == 1
        assert cm.accounts[0].request_count == 1

    def test_tracks_failure(self, config):
        cm = CookieManager(config)
        cm.report_rate_limit()
        assert cm.accounts[0].failure_count == 1

    def test_distributed_with_rotation(self, config):
        cm = CookieManager(config)
        for _ in range(9):
            cm.report_success()
        # 3 requests each due to rotation
        assert all(acc.request_count == 3 for acc in cm.accounts)


class TestSingleton:
    def test_returns_same_instance(self, config):
        cm1 = CookieManager.get_instance(config)
        cm2 = CookieManager.get_instance()
        assert cm1 is cm2

    def test_none_without_config(self):
        assert CookieManager.get_instance() is None
```

### Run Tests

```bash
pytest tests/test_cookie_rotation.py -v
pytest tests/test_cookie_rotation.py -v --cov=src/downloader/cookie_manager
```

---

## Notes

- **`rotate_on_success: true` is the default** — rotates after EVERY request
- Browser extraction: Firefox works while open, Chrome/Edge must be closed
- 3 accounts = 3x rate limit capacity with rotation
- Cookies directory is gitignored for security
