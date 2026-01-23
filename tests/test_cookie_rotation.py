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
        (cookies / name).write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t0\ttest\tvalue")
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
    """Test CookieManager initialization."""

    def test_loads_accounts(self, config):
        """Should load all configured accounts."""
        cm = CookieManager(config)
        assert len(cm.accounts) == 3
        assert cm.accounts[0].label == 'main'
        assert cm.accounts[1].label == 'backup1'
        assert cm.accounts[2].label == 'backup2'

    def test_empty_config(self):
        """Should handle empty config gracefully."""
        cm = CookieManager({})
        assert cm.enabled is False
        assert len(cm.accounts) == 0

    def test_no_accounts(self):
        """Should handle config with no accounts."""
        cm = CookieManager({'enabled': True, 'accounts': []})
        assert cm.enabled is True
        assert len(cm.accounts) == 0

    def test_browser_account(self):
        """Should handle browser-based account config."""
        cm = CookieManager({
            'enabled': True,
            'accounts': [{'label': 'ff', 'browser': 'firefox'}]
        })
        assert cm.accounts[0].browser == 'firefox'
        assert cm.accounts[0].cookies_path is None

    def test_default_label_generation(self, cookies_dir):
        """Should generate default labels when not provided."""
        config = {
            'enabled': True,
            'cookies_dir': str(cookies_dir),
            'accounts': [
                {'cookies_path': 'main.txt'},
                {'cookies_path': 'backup1.txt'},
            ]
        }
        cm = CookieManager(config)
        assert cm.accounts[0].label == 'account_0'
        assert cm.accounts[1].label == 'account_1'

    def test_creates_cookies_dir(self, tmp_path):
        """Should create cookies directory if it doesn't exist."""
        cookies_dir = tmp_path / "new_cookies"
        config = {
            'enabled': True,
            'cookies_dir': str(cookies_dir),
            'accounts': []
        }
        CookieManager(config)
        assert cookies_dir.exists()


class TestRotation:
    """Test round-robin rotation behavior."""

    def test_cycles_through_accounts(self, config):
        """Should cycle through accounts in order."""
        cm = CookieManager(config)
        assert cm.get_current_account().label == 'main'
        cm._rotate_index()
        assert cm.get_current_account().label == 'backup1'
        cm._rotate_index()
        assert cm.get_current_account().label == 'backup2'
        cm._rotate_index()
        assert cm.get_current_account().label == 'main'  # Wraps around

    def test_rotates_on_success(self, config):
        """Should rotate after success when enabled."""
        cm = CookieManager(config)
        assert cm.get_current_account().label == 'main'
        cm.report_success()
        assert cm.get_current_account().label == 'backup1'
        cm.report_success()
        assert cm.get_current_account().label == 'backup2'

    def test_no_rotate_when_disabled(self, config):
        """Should NOT rotate after success when disabled."""
        config['rotate_on_success'] = False
        cm = CookieManager(config)
        cm.report_success()
        cm.report_success()
        assert cm.get_current_account().label == 'main'

    def test_single_account_rotation(self, cookies_dir):
        """Should handle single account gracefully."""
        config = {
            'enabled': True,
            'cookies_dir': str(cookies_dir),
            'rotate_on_success': True,
            'accounts': [{'label': 'only', 'cookies_path': 'main.txt'}]
        }
        cm = CookieManager(config)
        assert cm.get_current_account().label == 'only'
        cm._rotate_index()
        assert cm.get_current_account().label == 'only'


class TestCooldown:
    """Test rate limit handling and cooldown behavior."""

    def test_rate_limit_triggers_cooldown(self, config):
        """Should put account in cooldown on rate limit."""
        cm = CookieManager(config)
        cm.report_rate_limit()
        assert cm.accounts[0].is_cooling_down is True
        assert cm.accounts[0].cooldown_until > time.time()
        assert cm.accounts[0].failure_count == 1

    def test_rate_limit_rotates(self, config):
        """Should rotate to next account after rate limit."""
        cm = CookieManager(config)
        assert cm.get_current_account().label == 'main'
        cm.report_rate_limit()
        assert cm.get_current_account().label == 'backup1'

    def test_skips_cooling_account(self, config):
        """Should skip accounts in cooldown."""
        cm = CookieManager(config)
        cm.accounts[0].is_cooling_down = True
        cm.accounts[0].cooldown_until = time.time() + 300
        assert cm.get_current_account().label == 'backup1'

    def test_cooldown_expiry(self, config):
        """Should return account after cooldown expires."""
        cm = CookieManager(config)
        cm.accounts[0].is_cooling_down = True
        cm.accounts[0].cooldown_until = time.time() - 1  # Already expired
        assert cm.get_current_account().label == 'main'
        assert cm.accounts[0].is_cooling_down is False

    def test_all_cooling_returns_none(self, config):
        """Should return None when all accounts cooling down."""
        cm = CookieManager(config)
        for acc in cm.accounts:
            acc.is_cooling_down = True
            acc.cooldown_until = time.time() + 300
        assert cm.get_current_account() is None

    def test_cooldown_seconds_configurable(self, config):
        """Should use configured cooldown seconds."""
        config['cooldown_seconds'] = 60
        cm = CookieManager(config)
        before = time.time()
        cm.report_rate_limit()
        after = time.time()
        assert cm.accounts[0].cooldown_until >= before + 60
        assert cm.accounts[0].cooldown_until <= after + 60


class TestCookiesArgs:
    """Test yt-dlp argument generation."""

    def test_returns_cookies_flag(self, config, cookies_dir):
        """Should return --cookies arg for file-based account."""
        cm = CookieManager(config)
        args = cm.get_cookies_args()
        expected_path = str(cookies_dir / 'main.txt')
        assert args == ['--cookies', expected_path]

    def test_returns_browser_flag(self):
        """Should return --cookies-from-browser arg for browser account."""
        cm = CookieManager({
            'enabled': True,
            'accounts': [{'label': 'ff', 'browser': 'firefox'}]
        })
        assert cm.get_cookies_args() == ['--cookies-from-browser', 'firefox']

    def test_missing_file_returns_empty(self, config, cookies_dir):
        """Should return empty if cookie file doesn't exist."""
        config['accounts'][0]['cookies_path'] = 'nonexistent.txt'
        cm = CookieManager(config)
        assert cm.get_cookies_args() == []

    def test_no_accounts_returns_empty(self):
        """Should return empty if no accounts configured."""
        cm = CookieManager({'enabled': True, 'accounts': []})
        assert cm.get_cookies_args() == []

    def test_all_cooling_returns_empty(self, config):
        """Should return empty if all accounts cooling down."""
        cm = CookieManager(config)
        for acc in cm.accounts:
            acc.is_cooling_down = True
            acc.cooldown_until = time.time() + 300
        assert cm.get_cookies_args() == []

    def test_uses_absolute_path(self, config, cookies_dir):
        """Should return absolute paths in yt-dlp args."""
        cm = CookieManager(config)
        args = cm.get_cookies_args()
        assert len(args) == 2
        assert args[0] == '--cookies'
        assert Path(args[1]).is_absolute()


class TestStatistics:
    """Test statistics tracking."""

    def test_tracks_success(self, config):
        """Should increment success count."""
        cm = CookieManager(config)
        cm.report_success()
        assert cm.accounts[0].success_count == 1
        assert cm.accounts[0].request_count == 1

    def test_tracks_failure(self, config):
        """Should increment failure count on rate limit."""
        cm = CookieManager(config)
        cm.report_rate_limit()
        assert cm.accounts[0].failure_count == 1
        assert cm.accounts[0].request_count == 1

    def test_distributed_with_rotation(self, config):
        """Should distribute requests evenly with rotation."""
        cm = CookieManager(config)
        for _ in range(9):
            cm.report_success()
        # 3 requests each due to rotation
        assert all(acc.request_count == 3 for acc in cm.accounts)

    def test_get_stats(self, config):
        """Should return correct stats dictionary."""
        cm = CookieManager(config)
        cm.report_success()
        cm.report_success()
        cm.report_rate_limit()

        stats = cm.get_stats()
        assert len(stats['accounts']) == 3
        assert stats['accounts'][0]['label'] == 'main'
        assert stats['accounts'][0]['successes'] == 1
        assert stats['accounts'][2]['failures'] == 1


class TestSingleton:
    """Test singleton pattern."""

    def test_returns_same_instance(self, config):
        """Should return same instance on subsequent calls."""
        cm1 = CookieManager.get_instance(config)
        cm2 = CookieManager.get_instance()
        assert cm1 is cm2

    def test_none_without_config(self):
        """Should return None if no config provided on first call."""
        assert CookieManager.get_instance() is None

    def test_reset_clears_instance(self, config):
        """Should clear instance on reset."""
        cm1 = CookieManager.get_instance(config)
        CookieManager.reset_instance()
        assert CookieManager.get_instance() is None


class TestListAvailableCookies:
    """Test cookie file listing."""

    def test_lists_cookie_files(self, config, cookies_dir):
        """Should list all .txt files in cookies directory."""
        cm = CookieManager(config)
        available = cm.list_available_cookies()
        assert 'main.txt' in available
        assert 'backup1.txt' in available
        assert 'backup2.txt' in available

    def test_empty_dir(self, tmp_path):
        """Should return empty list for empty directory."""
        cookies_dir = tmp_path / "empty_cookies"
        cookies_dir.mkdir()
        config = {
            'enabled': True,
            'cookies_dir': str(cookies_dir),
            'accounts': []
        }
        cm = CookieManager(config)
        assert cm.list_available_cookies() == []


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_report_success_no_accounts(self):
        """Should handle report_success with no accounts."""
        cm = CookieManager({'enabled': True, 'accounts': []})
        # Should not raise
        cm.report_success()

    def test_report_rate_limit_no_accounts(self):
        """Should handle report_rate_limit with no accounts."""
        cm = CookieManager({'enabled': True, 'accounts': []})
        # Should not raise
        cm.report_rate_limit()

    def test_mixed_cookie_and_browser(self, cookies_dir):
        """Should handle mix of cookie file and browser accounts."""
        config = {
            'enabled': True,
            'cookies_dir': str(cookies_dir),
            'rotate_on_success': True,
            'accounts': [
                {'label': 'file', 'cookies_path': 'main.txt'},
                {'label': 'browser', 'browser': 'firefox'},
            ]
        }
        cm = CookieManager(config)

        # First account - file
        args = cm.get_cookies_args()
        assert args == ['--cookies', str(cookies_dir / 'main.txt')]

        cm._rotate_index()

        # Second account - browser
        args = cm.get_cookies_args()
        assert args == ['--cookies-from-browser', 'firefox']

    def test_single_account_all_cooling(self, cookies_dir):
        """Single account in cooldown should return None."""
        config = {
            'enabled': True,
            'cookies_dir': str(cookies_dir),
            'accounts': [{'label': 'only', 'cookies_path': 'main.txt'}]
        }
        cm = CookieManager(config)
        cm.report_rate_limit()
        assert cm.get_current_account() is None
