# Ralph Loop Refactor Design

**Date:** 2026-01-27
**Scope:** Scripts + state layer (no test restructuring)
**Approach:** Hybrid decomposition — domain modules + thin orchestrator

## Problem

ralph.ps1 is 7,405 lines with 60+ functions, extensive script-scoped global state, and dual-format support for legacy queue schemas. This makes it difficult to modify individual systems, test in isolation, and reason about state flow.

## Design

### Module Decomposition

Ralph.ps1 splits into 6 domain modules in `scripts/ralph/lib/`, dot-sourced at startup. Ralph.ps1 shrinks from 7,400 to ~800 lines, keeping only mode routing and the main execution loop.

```
scripts/ralph/
├── ralph.ps1              # ~800 lines: startup, mode routing, main loop
├── lib/
│   ├── sprint.ps1         # ~600 lines: PRD generation, sprint lifecycle, archive, history
│   ├── scoring.ps1        # ~400 lines: Ralph's Choice, focus area scoring, git activity
│   ├── queue.ps1          # ~300 lines: interview queue processing, progress tracking
│   ├── metrics.ps1        # ~350 lines: CSV metrics, iteration tracking, cost attribution
│   ├── quality.ps1        # ~500 lines: quality review, test baselines, regression detection
│   └── prompts.ps1        # ~300 lines: Claude prompt building, context injection, retry context
├── launcher.ps1           # unchanged
├── interview.ps1          # minor: imports from lib/queue.ps1 for shared format handling
├── watch.ps1              # unchanged
├── watch-lib.ps1          # minor: removes legacy queue format handling
└── status.ps1             # unchanged
```

**Dot-sourcing pattern:**
```powershell
# ralph.ps1 top
$libPath = Join-Path $PSScriptRoot 'lib'
. "$libPath\sprint.ps1"
. "$libPath\scoring.ps1"
. "$libPath\queue.ps1"
. "$libPath\metrics.ps1"
. "$libPath\quality.ps1"
. "$libPath\prompts.ps1"
```

### Function-to-Module Mapping

**lib/sprint.ps1** — Sprint lifecycle:
- `New-SeedPRD`, `Save-SprintArchive`, `Update-SprintHistory`, `Get-SprintHistory`
- `Get-RalphConfig`, `Get-ConfigValue`
- `Start-NewSprint`, `Complete-Sprint`, `Get-SprintReport`, `New-SprintReport`
- `Get-SprintRetrospective`

**lib/scoring.ps1** — Ralph's Choice:
- `Get-AllFocusAreaScores`, `Get-FocusAreaScore`, `Get-GitActivityByArea`
- `Get-StayOrSwitchDecision`, `Get-NeglectedAreas`, `Get-CategoryBalance`
- `Show-FocusAreaRecommendations`

**lib/queue.ps1** — Interview queue:
- `Get-InterviewFocusAreas`, `Update-QueueProgress`, `Get-NextQueuedFocusArea`
- `Test-QueueComplete`, `Show-CompletionChoice`
- `Get-Queue`, `Save-Queue` (new state accessors)

**lib/metrics.ps1** — Tracking:
- `Add-MetricRow`, `Get-MetricsSummary`, `Get-SessionMetrics`
- `Export-Metrics`, `Get-CostAttribution`
- `Get-PhaseTimings`, `Update-ProgressLog`

**lib/quality.ps1** — Quality gates:
- `Invoke-QualityReview`, `Compare-TestBaseline`, `Save-TestBaseline`
- `Test-RegressionDetected`, `Get-TestResults`
- `Invoke-ReviewAgent`, `Get-ReviewVerdict`

**lib/prompts.ps1** — Claude interaction:
- `Build-StoryPrompt`, `Build-ExplorationPrompt`, `Build-RetryPrompt`
- `Get-StoryFailureContext`, `Get-ExplorationContext`
- `Format-AcceptanceCriteria`, `Format-InterviewContext`

**Stays in ralph.ps1** (~800 lines):
- Parameter block and startup validation
- Mode detection and routing (Start-StandardLoop, Start-TrueAutoLoop, etc.)
- Main story execution loop (invoke Claude, check result, advance)
- Watch window spawning
- Graceful shutdown signal handling
- The dot-source block

### State Layer Redesign

**1. Unified queue format** — Remove legacy `{ "queue": [], "completedAreas": [], "currentIndex": N }`. Only the new format:

```json
{
  "focusAreas": [
    { "id": "pipeline", "completed": false },
    { "id": "testing", "completed": true, "completedAt": "2026-01-27T..." }
  ],
  "interviewContext": "...",
  "interviewDetails": "...",
  "sessionId": "...",
  "createdAt": "..."
}
```

Removes ~150 lines of dual-format branching across ralph.ps1, interview.ps1, and watch-lib.ps1.

**2. State access functions** — Each module owns its state file:

| Module | File | Functions |
|--------|------|-----------|
| sprint.ps1 | prd.json | `Get-Sprint`, `Save-Sprint`, `Update-StoryStatus` |
| queue.ps1 | queue.json | `Get-Queue`, `Save-Queue`, `Update-QueueProgress` |
| metrics.ps1 | metrics.csv | `Add-MetricRow`, `Get-MetricsSummary` |
| sprint.ps1 | sprint_history.json | `Get-SprintHistory`, `Update-SprintHistory` |
| quality.ps1 | test_baseline.json | `Get-TestBaseline`, `Save-TestBaseline` |

**3. Write safety** — Atomic writes via temp-then-rename:

```powershell
function Save-StateFile {
    param([string]$Path, [object]$Data)
    $tempPath = "$Path.tmp"
    $Data | ConvertTo-Json -Depth 10 | Set-Content -Path $tempPath -Encoding UTF8
    Move-Item -Path $tempPath -Destination $Path -Force
}
```

### Script-Scoped State Consolidation

Replace scattered `$script:*` variables with a single `$script:State` hashtable:

```powershell
# Before (scattered globals)
$script:SessionId = New-Guid
$script:IterationCount = 0
$script:LastFocusAreaId = ''

# After (single state object)
$script:State = @{
    SessionId        = New-Guid
    IterationCount   = 0
    LastFocusAreaId  = ''
    StoriesSinceExploration = 0
    SprintExplorationContext = ''
}
```

Module functions receive `$State` as a parameter instead of reaching into `$script:` scope.

## Migration Phases

Each phase is independently shippable. No phase depends on the next.

### Phase 1: Extract modules (mechanical move)

Move functions from ralph.ps1 into `lib/*.ps1` files. Zero behavioral changes. Every function keeps its exact name, parameters, and logic. Only change in ralph.ps1: add dot-source block, delete moved function bodies.

**Verification:** Full Pester suite. Tests find functions via dot-sourcing transitively.

**Risk:** Tests that dot-source ralph.ps1 directly need ralph.ps1's dot-sourcing to load lib/ transitively. Verify before cutting.

### Phase 2: State layer cleanup

- Add `Save-StateFile` helper
- Replace raw `Get-Content | ConvertFrom-Json` with module-owned accessors
- Delete legacy queue format support
- Update watch-lib.ps1 to remove legacy handling

**Verification:** Pester tests + manual smoke test (launcher -> interview -> queue mode -> watch dashboard).

### Phase 3: Script-scoped state reduction

- Replace `$script:*` variables with `$script:State` hashtable
- Module functions receive `$State` as parameter
- Update tests to construct state hashtables instead of mocking script scope

**Verification:** Full Pester suite. Tests that mock `$script:*` variables updated.

## What Doesn't Change

- **launcher.ps1** (515 lines) — already clean
- **watch.ps1** (181 lines) — display only
- **status.ps1** (99 lines) — read only
- **ralph-config.json** schema — no structural changes
- **clients.json** schema — untouched
- **Test file organization** — tests stay in `scripts/ralph/tests/`, organized by concern
- **Batch files** — unchanged
- **Function names** — no renames, only physical relocation
