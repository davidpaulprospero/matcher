# Plan: yt-dlp Authentication Preflight Check

## Problem

The pipeline can fail deep into the DOWNLOAD stage when YouTube authentication isn't working:
- Browser cookies expired or unavailable
- Firefox profile not accessible
- YouTube rate limiting already active
- Tier 4 (browser cookies) assumed but not actually working

Currently, these issues only surface after spending time on earlier stages (ANALYZE, ENTITY_IMAGES, etc.), wasting user time.

## Solution

Add a **yt-dlp authentication preflight check** that runs before any downloads, verifying:
1. Current tier settings can successfully authenticate with YouTube
2. Browser cookies are accessible (if Tier 4+ is configured)
3. A test video can be accessed (metadata only, no download)

## Implementation

### Phase 1: Add Preflight Check to Orchestrator

**File: `src/agents/orchestrator.py`**

Add new method `_check_ytdlp_auth()` that:

```python
def _check_ytdlp_auth(self) -> List[PreflightIssue]:
    """
    Verify yt-dlp can authenticate with YouTube using current tier settings.

    Tests by fetching metadata for a known public video (no download).
    """
    issues = []

    # Get current bypass tier config
    download_config = getattr(self.config, 'download', None)
    if not download_config:
        return issues

    bypass_config = getattr(download_config, 'rate_limit_bypass', None)
    if not bypass_config:
        return issues

    # Get current tier
    current_tier = getattr(bypass_config, '_current_tier', 1)
    start_tier = getattr(bypass_config, 'start_tier', 1)

    # Build yt-dlp command based on tier
    test_video_id = "dQw4w9WgXcQ"  # Known stable public video
    cmd = self._build_ytdlp_auth_test_cmd(current_tier, test_video_id)

    # Run test
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            # Authentication failed
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"

            if "Sign in" in error_msg or "cookies" in error_msg.lower():
                issues.append(PreflightIssue(
                    category="ytdlp_auth",
                    severity="critical",
                    message=f"YouTube authentication failed at Tier {current_tier}. "
                            f"Browser cookies may be expired or inaccessible. "
                            f"Error: {error_msg}",
                    auto_fixable=True,  # Can try escalating tier
                ))
            elif "429" in error_msg or "rate" in error_msg.lower():
                issues.append(PreflightIssue(
                    category="ytdlp_auth",
                    severity="warning",
                    message=f"YouTube rate limiting detected at Tier {current_tier}. "
                            f"Downloads may be slow or fail.",
                    auto_fixable=False,
                ))
            else:
                issues.append(PreflightIssue(
                    category="ytdlp_auth",
                    severity="warning",
                    message=f"yt-dlp test failed at Tier {current_tier}: {error_msg}",
                    auto_fixable=True,
                ))
        else:
            logger.info(f"[preflight] yt-dlp auth OK at Tier {current_tier}")

    except subprocess.TimeoutExpired:
        issues.append(PreflightIssue(
            category="ytdlp_auth",
            severity="warning",
            message=f"yt-dlp auth test timed out at Tier {current_tier}",
            auto_fixable=True,
        ))
    except Exception as e:
        issues.append(PreflightIssue(
            category="ytdlp_auth",
            severity="warning",
            message=f"yt-dlp auth test error: {str(e)}",
            auto_fixable=False,
        ))

    return issues


def _build_ytdlp_auth_test_cmd(self, tier: int, video_id: str) -> List[str]:
    """Build yt-dlp command for auth testing based on tier."""
    cmd = [
        'yt-dlp',
        '--skip-download',
        '--print', 'title',
        '--no-warnings',
        f'https://www.youtube.com/watch?v={video_id}'
    ]

    # Add tier-specific options
    if tier >= 4:
        # Tier 4: Browser cookies
        cmd.extend(['--cookies-from-browser', 'firefox'])

    if tier >= 2:
        # Tier 2+: Client impersonation
        cmd.extend(['--extractor-args', 'youtube:player-client=web'])

    return cmd
```

### Phase 2: Add Auto-Fix via Full Sweep Tier Testing

**File: `src/agents/orchestrator.py`**

Full sweep logic: Start at configured tier, go UP to max, then DOWN to 1.

```python
def _fix_ytdlp_auth_issue(self, issue: PreflightIssue) -> bool:
    """
    Attempt to fix yt-dlp auth issue by sweeping all tiers.

    Full sweep order (if start_tier=4, max_tier=7):
      4 (failed) -> 5 -> 6 -> 7 -> 3 -> 2 -> 1

    Finds the first working tier and updates config.
    """
    download_config = getattr(self.config, 'download', None)
    if not download_config:
        return False

    bypass_config = getattr(download_config, 'rate_limit_bypass', None)
    if not bypass_config:
        return False

    start_tier = getattr(bypass_config, 'start_tier', 1)
    current_tier = getattr(bypass_config, '_current_tier', start_tier)
    max_tier = getattr(bypass_config, 'max_tier', 7)

    # Build sweep order: UP from current, then DOWN from current-1
    tiers_to_try = []

    # First: go UP (current+1 to max)
    for t in range(current_tier + 1, max_tier + 1):
        tiers_to_try.append(t)

    # Then: go DOWN (current-1 to 1)
    for t in range(current_tier - 1, 0, -1):
        tiers_to_try.append(t)

    logger.info(f"[preflight] Full sweep: trying tiers {tiers_to_try}")

    for tier in tiers_to_try:
        logger.info(f"[preflight] Testing Tier {tier}...")

        if self._test_ytdlp_auth_at_tier(tier):
            # Found working tier
            bypass_config._current_tier = tier
            logger.info(f"[preflight] Tier {tier} works! Updated config.")
            return True
        else:
            logger.debug(f"[preflight] Tier {tier} failed")

    # All tiers failed
    logger.error(f"[preflight] All tiers (1-{max_tier}) failed authentication")
    return False


def _test_ytdlp_auth_at_tier(self, tier: int) -> bool:
    """Test if a specific tier can authenticate with YouTube."""
    test_video_id = "dQw4w9WgXcQ"  # Known stable public video
    cmd = self._build_ytdlp_auth_test_cmd(tier, test_video_id)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except:
        return False
```

### Phase 3: Integration with run_preflight()

Update `run_preflight()` to call the new check:

```python
def run_preflight(self, state: 'PipelineState') -> List[PreflightIssue]:
    # ... existing checks ...

    # Check yt-dlp basic availability
    ytdlp_issues = self._check_ytdlp()
    issues.extend(ytdlp_issues)

    # NEW: Check yt-dlp authentication (cookies, rate limits)
    ytdlp_auth_issues = self._check_ytdlp_auth()
    issues.extend(ytdlp_auth_issues)

    # ... rest of checks ...
```

### Phase 4: Config Options

**File: `config.yaml`**

```yaml
healing:
  preflight:
    check_ytdlp_auth: true           # Enable auth preflight check
    ytdlp_auth_timeout: 30           # Timeout for auth test (seconds)
    ytdlp_test_video: "dQw4w9WgXcQ"  # Video ID for testing (public, stable)
    auto_escalate_tier: true         # Auto-escalate on auth failure
```

## Test Scenarios

1. **Tier 4 with valid cookies**: Should pass immediately
2. **Tier 4 with expired cookies**: Should detect and escalate to Tier 5+
3. **Tier 1 (no cookies)**: Should detect auth requirement and escalate to Tier 4
4. **Rate limited**: Should warn but not block (downloads may still work slowly)
5. **No yt-dlp installed**: Should fail with clear error

## Benefits

1. **Early detection**: Auth issues caught before wasting time on other stages
2. **Auto-recovery**: Automatic tier escalation finds working config
3. **Clear feedback**: User knows exactly what's wrong (cookies expired, etc.)
4. **No downloads**: Test uses `--skip-download` + `--print title` (metadata only)

## Risks

1. Test video could become unavailable (use well-known stable video)
2. Auth test itself could trigger rate limiting (mitigate with single fast test)
3. Browser profile access could fail silently (handle subprocess errors)

## Timeline

- Phase 1-2: Core implementation (~2 hours)
- Phase 3: Integration (~30 min)
- Phase 4: Config + docs (~30 min)

## Files to Modify

1. `src/agents/orchestrator.py` - Add `_check_ytdlp_auth()`, update `run_preflight()`
2. `config.yaml` - Add preflight config options
3. `src/config/sections/infrastructure.py` - Add preflight config dataclass
4. `CLAUDE.md` - Document new preflight check
