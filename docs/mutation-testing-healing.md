# Mutation Testing Report: Healing.Tests.ps1

**Date:** 2026-01-28
**File under test:** `scripts/ralph/lib/healing.ps1`
**Test file:** `scripts/ralph/tests/Healing.Tests.ps1`
**Suite size:** 89 tests (68 original + 21 edge-case additions)

## Summary

| Metric | Value |
|--------|-------|
| Mutations applied | 10 |
| Mutations killed | 10 |
| Kill rate | **100%** |
| Surviving mutants | 0 |

## Methodology

Each mutation deliberately breaks one behavior in `healing.ps1`, then runs the corresponding test to verify detection. Source is reverted after each mutation.

## Mutations

### Mutation 1: Remove `ConfigErrors` from `HasErrors` calculation

**Function:** `Invoke-FastHealthCheck`
**Change:** Removed `$result.ConfigErrors.Count -gt 0` from the `HasErrors` boolean expression.
**Killed by:** `detects config validation failures`

```
Expected $true, but got $false.
```

**Why it matters:** Config validation failures would be silently ignored, letting the pipeline run with broken YAML.

---

### Mutation 2: Break merge conflict regex

**Function:** `Invoke-FastHealthCheck`
**Change:** `'\t(.+)$'` -> `'\t\t(.+)$'` (require double-tab instead of single-tab)
**Killed by:** `detects merge conflicts from git ls-files`

```
Expected the actual value to be greater than 0, but got 0.
```

**Why it matters:** Git merge conflicts would go undetected, producing corrupt merged code.

---

### Mutation 3: Remove `elseif` fallback for ERROR lines without dash separator

**Function:** `Invoke-CollectionHealthCheck`
**Change:** Removed the `elseif ($_ -match '^ERROR\s+(.+)$')` branch that handles `ERROR path` lines (no ` - ErrorType`).
**Killed by:** `parses ERROR line with no dash separator`

```
Expected $true, but got $false.
```

**Why it matters:** Some pytest collection errors omit the dash-separated error type. Without the fallback, these errors are silently dropped from the count.

---

### Mutation 4: Replace catch with `throw` in FullHealthCheck

**Function:** `Invoke-FullHealthCheck`
**Change:** `catch { $result.Skipped = $true }` -> `catch { throw }`
**Killed by:** `handles pytest exception gracefully`

```
RuntimeException: pytest crashed
```

**Why it matters:** If pytest isn't installed or crashes, the health check should degrade gracefully (`Skipped = $true`) instead of aborting the entire healing pipeline.

---

### Mutation 5: Remove CONFIG line formatting in diagnostics

**Function:** `Build-TierDiagnostics`
**Change:** Removed the `if ($TierResult.ConfigErrors)` block that formats config errors into the diagnostics string.
**Killed by:** `formats config errors`

```
Expected like wildcard '*CONFIG: Config validation failed*' to match 'Tier 1 diagnostics:', but it did not match.
```

**Why it matters:** Config errors would vanish from the diagnostics passed to Claude's healing prompt, making the root cause invisible to the healer.

---

### Mutation 6: Remove try/catch in Test-HealingInProgress

**Function:** `Test-HealingInProgress`
**Change:** Removed `try/catch` around `ConvertFrom-Json`, letting corrupt JSON propagate as an exception.
**Killed by:** `returns false when state file has corrupt JSON`

```
ArgumentException: Invalid JSON primitive: not.
```

**Why it matters:** A corrupted `healing_state.json` (e.g., from interrupted writes) would crash the entire Ralph loop instead of treating it as "no healing in progress."

---

### Mutation 7: Rename tier 2 from "Import/Collection" to "Dependency Check"

**Function:** `Build-HealingPrompt`
**Change:** `2 = "Import/Collection"` -> `2 = "Dependency Check"` in the `$tierNames` hashtable.
**Killed by:** `maps all three tier names correctly`

```
Expected like wildcard '*Import/Collection*' to match '...Dependency Check...', but it did not match.
```

**Why it matters:** Tier names are embedded in the healing prompt sent to Claude. Wrong names would confuse the healer about what type of error it's fixing.

---

### Mutation 8: Remove output truncation

**Function:** `Build-HealingPrompt`
**Change:** `$truncated = $PreviousOutput.Substring(0, [Math]::Min(1000, $PreviousOutput.Length))` -> `$truncated = $PreviousOutput`
**Killed by:** `truncates long previous output to 1000 chars`

```
Expected the actual value to be less than 2500, but got 2996.
```

**Why it matters:** Without truncation, verbose Claude output from failed attempts would bloat the retry prompt, wasting tokens and potentially exceeding context limits.

---

### Mutation 9: Remove try/catch in Get-HealingSummary foreach loop

**Function:** `Get-HealingSummary`
**Change:** Removed `try/catch` around JSONL line parsing in the foreach loop.
**Killed by:** `skips corrupt JSONL lines without crashing`

```
ArgumentException: Invalid JSON primitive: this.
```

**Why it matters:** A single corrupt line in `healing_log.jsonl` (e.g., from a concurrent write or disk issue) would crash the entire sprint report generation.

---

### Mutation 10: Remove null-state guard in Log-HealingEvent

**Function:** `Log-HealingEvent`
**Change:** `if ($script:State) { ... }` guard removed; session context written unconditionally.
**Killed by:** `handles missing $State gracefully`

```
Expected 'sessionId' to not be found in collection @('iteration', 'timestamp', 'data', 'event', 'sessionId'), but it was found.
```

**Why it matters:** `Log-HealingEvent` can be called before the Ralph session state is initialized (e.g., during config validation). Without the guard, it writes null/empty values that pollute the audit log.

---

## How to Reproduce

Run a specific mutation manually:

```powershell
# 1. Break the code (example: remove ConfigErrors from HasErrors)
# Edit scripts/ralph/lib/healing.ps1 - remove ConfigErrors line from HasErrors

# 2. Run the test that should catch it
$config = New-PesterConfiguration
$config.Run.Path = 'scripts/ralph/tests/Healing.Tests.ps1'
$config.Filter.FullName = '*config validation*'
$config.Output.Verbosity = 'Detailed'
Invoke-Pester -Configuration $config

# 3. Verify it fails, then revert
git checkout -- scripts/ralph/lib/healing.ps1
```

## Coverage Map

| Function | Tests | Mutations | Kill Rate |
|----------|-------|-----------|-----------|
| `Invoke-FastHealthCheck` | 9 | 2 | 100% |
| `Invoke-CollectionHealthCheck` | 6 | 1 | 100% |
| `Invoke-FullHealthCheck` | 7 | 1 | 100% |
| `Invoke-TieredHealthCheck` | 9 | 0 | - |
| `Build-TierDiagnostics` | 8 | 1 | 100% |
| `Log-HealingEvent` | 6 | 1 | 100% |
| `Suspend-SprintForHealing` | 4 | 0 | - |
| `Resume-SprintFromHealing` | 4 | 0 | - |
| `Test-HealingInProgress` | 4 | 1 | 100% |
| `Build-HealingPrompt` | 5 | 2 | 100% |
| `Invoke-HealingSession` | 9 | 0 | - |
| `Invoke-PostIterationHealing` | 9 | 0 | - |
| `Get-HealingSummary` | 7 | 1 | 100% |
| **Total** | **89** | **10** | **100%** |
