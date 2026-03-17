"""
Tests for budget exhaustion cascade behavior in RateLimitBudget.

Verifies the cascade logic when budget resources are progressively exhausted:
- Cookie rotation exhaustion triggers VPN escalation advice
- Full resource exhaustion triggers abort_keyword advice
- Exact boundary behavior for rotation and backoff budgets
- Checkpoint round-trip preserves exhaustion state

Sprint 14, US-010: Add budget exhaustion cascade behavior tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloader.rate_limit_budget import RateLimitBudget


# ============================================================================
# AC1: can_rotate()=False triggers skip_to_vpn in get_budget_advice()
# ============================================================================

class TestRotationExhaustionTriggersVpnAdvice:
    """When cookie rotations are exhausted but VPN is available,
    get_budget_advice() should return 'skip_to_vpn'."""

    @pytest.mark.fast
    def test_skip_to_vpn_when_rotations_exhausted(self):
        """Exhausting rotations with VPN available returns skip_to_vpn."""
        budget = RateLimitBudget()
        budget.max_rotations = 3
        budget.max_vpn_switches = 5

        for i in range(3):
            budget.record_rotation(keyword=f"kw_{i}")

        assert budget.can_rotate() is False
        assert budget.can_switch_vpn() is True
        assert budget.get_budget_advice() == "skip_to_vpn"

    @pytest.mark.fast
    def test_continue_when_rotations_available(self):
        """With rotations still available, advice is continue."""
        budget = RateLimitBudget()
        budget.max_rotations = 5
        budget.max_vpn_switches = 3

        budget.record_rotation(keyword="kw1")
        budget.record_rotation(keyword="kw2")

        assert budget.can_rotate() is True
        assert budget.get_budget_advice() == "continue"

    @pytest.mark.fast
    def test_skip_to_vpn_includes_correct_state(self):
        """skip_to_vpn advice is based on can_rotate() and can_switch_vpn()."""
        budget = RateLimitBudget()
        budget.max_rotations = 2
        budget.max_vpn_switches = 10

        # Exhaust rotations
        budget.record_rotation()
        budget.record_rotation()

        advice = budget.get_budget_advice()
        assert advice == "skip_to_vpn"
        # Confirm underlying state
        assert budget.can_rotate() is False
        assert budget.can_switch_vpn() is True


# ============================================================================
# AC2: can_switch_vpn()=False after can_rotate()=False -> abort_keyword
# ============================================================================

class TestFullExhaustionTriggersAbort:
    """When both cookie rotations AND VPN switches are exhausted,
    get_budget_advice() should return 'abort_keyword'."""

    @pytest.mark.fast
    def test_abort_keyword_when_all_exhausted(self):
        """Exhausting both rotations and VPN returns abort_keyword."""
        budget = RateLimitBudget()
        budget.max_rotations = 2
        budget.max_vpn_switches = 1

        # Exhaust rotations
        budget.record_rotation(keyword="kw1")
        budget.record_rotation(keyword="kw2")

        # Exhaust VPN
        budget.record_vpn_switch(keyword="kw1")

        assert budget.can_rotate() is False
        assert budget.can_switch_vpn() is False
        assert budget.get_budget_advice() == "abort_keyword"

    @pytest.mark.fast
    def test_abort_keyword_cascade_progression(self):
        """Advice progresses: continue -> skip_to_vpn -> abort_keyword."""
        budget = RateLimitBudget()
        budget.max_rotations = 2
        budget.max_vpn_switches = 1

        # Phase 1: continue
        assert budget.get_budget_advice() == "continue"

        # Phase 2: exhaust rotations -> skip_to_vpn
        budget.record_rotation()
        budget.record_rotation()
        assert budget.get_budget_advice() == "skip_to_vpn"

        # Phase 3: exhaust VPN -> abort_keyword
        budget.record_vpn_switch()
        assert budget.get_budget_advice() == "abort_keyword"

    @pytest.mark.fast
    def test_abort_keyword_with_backoff_still_available(self):
        """abort_keyword even if backoff budget remains (backoff doesn't help
        when all escalation resources are gone)."""
        budget = RateLimitBudget()
        budget.max_rotations = 1
        budget.max_vpn_switches = 1
        budget.max_backoff_time = 999.0  # Plenty of backoff

        budget.record_rotation()
        budget.record_vpn_switch()

        # Backoff is available, but rotations and VPN are gone
        assert budget.can_backoff(5.0) is True
        assert budget.can_rotate() is False
        assert budget.can_switch_vpn() is False
        assert budget.get_budget_advice() == "abort_keyword"


# ============================================================================
# AC3: record_rotation() N times then can_rotate()=False at exact boundary
# ============================================================================

class TestRotationBudgetExactBoundary:
    """record_rotation() succeeds N times then can_rotate() returns False,
    verifying exact boundary (max_rotations consumed)."""

    @pytest.mark.fast
    def test_exact_boundary_small_budget(self):
        """With max_rotations=3, can_rotate() flips at exactly 3."""
        budget = RateLimitBudget()
        budget.max_rotations = 3

        for i in range(3):
            assert budget.can_rotate() is True, f"Should allow rotation {i+1}"
            budget.record_rotation(keyword=f"kw_{i}")

        assert budget.can_rotate() is False
        assert budget.rotations_used == 3
        assert budget.rotations_remaining() == 0

    @pytest.mark.fast
    def test_exact_boundary_single_rotation(self):
        """With max_rotations=1, one rotation exhausts budget."""
        budget = RateLimitBudget()
        budget.max_rotations = 1

        assert budget.can_rotate() is True
        budget.record_rotation()
        assert budget.can_rotate() is False

    @pytest.mark.fast
    def test_exact_boundary_large_budget(self):
        """With max_rotations=100, boundary is at exactly 100."""
        budget = RateLimitBudget()
        budget.max_rotations = 100

        for i in range(99):
            budget.record_rotation()
        assert budget.can_rotate() is True
        assert budget.rotations_remaining() == 1

        budget.record_rotation()
        assert budget.can_rotate() is False
        assert budget.rotations_remaining() == 0

    @pytest.mark.fast
    def test_over_budget_still_false(self):
        """Recording beyond max_rotations still returns False."""
        budget = RateLimitBudget()
        budget.max_rotations = 2

        budget.record_rotation()
        budget.record_rotation()
        budget.record_rotation()  # Over limit

        assert budget.can_rotate() is False
        assert budget.rotations_used == 3
        assert budget.rotations_remaining() == 0

    @pytest.mark.fast
    def test_mid_download_boundary_triggers_vpn_advice(self):
        """At the exact rotation boundary, advice shifts to skip_to_vpn."""
        budget = RateLimitBudget()
        budget.max_rotations = 5
        budget.max_vpn_switches = 3

        for i in range(4):
            budget.record_rotation()
            assert budget.get_budget_advice() == "continue"

        # 5th rotation exhausts budget
        budget.record_rotation()
        assert budget.get_budget_advice() == "skip_to_vpn"


# ============================================================================
# AC4: record_backoff_time() cumulative > max_backoff_time -> can_backoff()=False
# ============================================================================

class TestBackoffBudgetExhaustion:
    """record_backoff() with cumulative > max_backoff_time causes
    can_backoff() to return False at threshold."""

    @pytest.mark.fast
    def test_backoff_exhaustion_at_threshold(self):
        """can_backoff() returns False when cumulative equals max."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 60.0

        budget.record_backoff(30.0)
        assert budget.can_backoff(30.0) is True  # 30 + 30 = 60 <= 60

        budget.record_backoff(30.0)
        # Now at 60.0 total, can_backoff with any additional should be False
        assert budget.can_backoff(0.1) is False  # 60 + 0.1 = 60.1 > 60
        assert budget.can_backoff(0.0) is True   # 60 + 0 = 60 <= 60

    @pytest.mark.fast
    def test_backoff_exhaustion_incremental(self):
        """Incremental small backoffs exhaust budget correctly."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 10.0

        for _ in range(10):
            budget.record_backoff(1.0)

        assert budget.backoff_time_spent == 10.0
        assert budget.can_backoff(0.1) is False
        assert budget.backoff_time_remaining() == 0.0

    @pytest.mark.fast
    def test_backoff_exhaustion_over_threshold(self):
        """Recording backoff over threshold; can_backoff stays False."""
        budget = RateLimitBudget()
        budget.max_backoff_time = 20.0

        budget.record_backoff(25.0)  # Over limit in one shot

        assert budget.backoff_time_spent == 25.0
        assert budget.can_backoff(0.0) is False  # 25 > 20
        assert budget.backoff_time_remaining() == 0.0

    @pytest.mark.fast
    def test_backoff_does_not_affect_rotation_advice(self):
        """Backoff exhaustion alone doesn't change get_budget_advice()
        (advice is about rotation/VPN, not backoff)."""
        budget = RateLimitBudget()
        budget.max_rotations = 5
        budget.max_vpn_switches = 3
        budget.max_backoff_time = 10.0

        budget.record_backoff(10.0)  # Exhaust backoff

        assert budget.can_backoff(1.0) is False
        assert budget.can_rotate() is True
        assert budget.get_budget_advice() == "continue"


# ============================================================================
# AC5: from_checkpoint() restores budget state correctly
# ============================================================================

class TestCheckpointRestoresBudgetState:
    """Serialize with 5 rotations used, deserialize, verify
    rotations_used=5 and can_rotate() reflects remaining budget."""

    @pytest.mark.fast
    def test_restore_5_rotations_used(self):
        """Round-trip with 5 rotations preserves exact count."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0

        for i in range(5):
            budget.record_rotation(keyword=f"kw_{i}")

        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.rotations_used == 5
        assert restored.max_rotations == 10
        assert restored.can_rotate() is True
        assert restored.rotations_remaining() == 5

    @pytest.mark.fast
    def test_restore_exhausted_rotations(self):
        """Round-trip with exhausted rotations preserves can_rotate()=False."""
        budget = RateLimitBudget()
        budget.max_rotations = 3

        for i in range(3):
            budget.record_rotation(keyword=f"kw_{i}")

        assert budget.can_rotate() is False

        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.rotations_used == 3
        assert restored.max_rotations == 3
        assert restored.can_rotate() is False

    @pytest.mark.fast
    def test_restore_preserves_advice(self):
        """Round-trip preserves get_budget_advice() output."""
        budget = RateLimitBudget()
        budget.max_rotations = 2
        budget.max_vpn_switches = 1

        budget.record_rotation()
        budget.record_rotation()
        # skip_to_vpn
        assert budget.get_budget_advice() == "skip_to_vpn"

        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)
        assert restored.get_budget_advice() == "skip_to_vpn"

    @pytest.mark.fast
    def test_restore_preserves_abort_keyword(self):
        """Round-trip preserves abort_keyword advice state."""
        budget = RateLimitBudget()
        budget.max_rotations = 1
        budget.max_vpn_switches = 1

        budget.record_rotation()
        budget.record_vpn_switch()
        assert budget.get_budget_advice() == "abort_keyword"

        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)
        assert restored.get_budget_advice() == "abort_keyword"

    @pytest.mark.fast
    def test_restore_keywords_and_escalation_level(self):
        """Round-trip preserves keywords_rate_limited and last_escalation_level."""
        budget = RateLimitBudget()
        budget.max_rotations = 10
        budget.max_vpn_switches = 5
        budget.max_backoff_time = 300.0

        for i in range(5):
            budget.record_rotation(keyword=f"kw_{i}")
        budget.record_backoff(45.0, keyword="kw_backoff")

        data = budget.to_dict()
        restored = RateLimitBudget.from_dict(data)

        assert restored.keywords_rate_limited == budget.keywords_rate_limited
        assert len(restored.keywords_rate_limited) == 6
        assert restored.last_escalation_level == "backoff"
        assert restored.backoff_time_spent == 45.0
