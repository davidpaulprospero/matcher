# Ralph Loop Refactor — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Decompose ralph.ps1 (7,444 lines, 82 functions) into 6 domain modules, unify state file access, and remove legacy queue format.

**Architecture:** Hybrid decomposition — domain modules in `scripts/ralph/lib/` dot-sourced by a slim ralph.ps1 orchestrator. Three independent phases: extract → state cleanup → state consolidation.

**Tech Stack:** PowerShell 5.1+, Pester 5.x (test framework), JSON state files

---

## Critical Context

### Test Sourcing Pattern

Tests use **AST parsing** to extract functions from ralph.ps1 — they do NOT dot-source the file directly. Two patterns exist:

**Pattern A — Full AST + Global Scope** (11 test files):
```powershell
$ast = [System.Management.Automation.Language.Parser]::ParseFile($script:RalphScript, [ref]$null, [ref]$null)
$functions = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
foreach ($func in $functions) {
    $globalFuncDef = $func.Extent.Text -replace '^function\s+([A-Za-z0-9_-]+)', 'function global:$1'
    Invoke-Expression $globalFuncDef
}
```

**Pattern B — Selective AST** (4 test files):
```powershell
$neededFunctions = @('Function1', 'Function2')
foreach ($fn in $functions) {
    if ($neededFunctions -contains $fn.Name) { . ([scriptblock]::Create($fn.Extent.Text)) }
}
```

**Impact:** After extraction, AST-parsing ralph.ps1 alone won't find moved functions. We need a test helper that parses `lib/*.ps1` files too.

### Complete Function-to-Module Mapping (All 82 Functions)

**lib/sprint.ps1** (15 functions):
| Function | Lines | Description |
|----------|-------|-------------|
| `Get-RalphConfig` | 77-108 | Load ralph-config.json with defaults |
| `Get-SprintHistory` | 117-136 | Load sprint history or return empty |
| `Update-SprintHistory` | 139-198 | Update cumulative sprint history |
| `Save-SprintArchive` | 201-318 | Archive PRD with retrospective |
| `Test-ShouldGenerateNewPRD` | 321-373 | Determine if new PRD needed |
| `Write-JsonNoBom` | 681-700 | Write JSON without UTF-8 BOM |
| `New-SeedPRD` | 925-1005 | Create minimal PRD with US-001 |
| `New-SprintReport` | 2273-2430 | Generate sprint report markdown |
| `Get-SprintRetrospective` | 3088-3211 | Analyze sprint for lessons learned |
| `Get-RetrospectiveContext` | 3213-3262 | Format retrospective into prompt context |
| `Get-IndependentStories` | 3973-4029 | Stories with met dependencies |
| `Build-DependencyGraph` | 4240-4294 | Build story dependency graph |
| `Get-ExecutableStories` | 4296-4339 | Stories ready to execute |
| `Update-LearningDb` | 4031-4068 | Update cross-session learning DB |
| `Get-LearningContext` | 4070-4126 | Get relevant learning context |

**lib/scoring.ps1** (7 functions):
| Function | Lines | Description |
|----------|-------|-------------|
| `Get-GitActivityByArea` | 380-439 | Count commits per focus area |
| `Get-FocusAreaScore` | 442-538 | Weighted score (activity+neglect+balance) |
| `Get-AllFocusAreaScores` | 540-567 | Score all areas, return sorted |
| `Get-StayOrSwitchDecision` | 569-675 | Decide stay/switch after sprint |
| `Get-OptimalNextStory` | 2727-2822 | Determine optimal next story |
| `Get-StoryFileTouches` | 2824-2877 | Count file touches per story |
| `Test-FileConflict` | 2879-2961 | Test story file conflicts |

**lib/queue.ps1** (7 functions):
| Function | Lines | Description |
|----------|-------|-------------|
| `Get-QueueData` | 758-775 | Read and parse queue.json |
| `Get-InterviewFocusAreas` | 777-804 | Get incomplete focus areas |
| `Get-InterviewContext` | 806-814 | Get interview context string |
| `Update-ContextFromPRD` | 816-840 | Update queue context from PRD |
| `Get-NextQueuedFocusArea` | 842-856 | Get next incomplete area ID |
| `Update-QueueProgress` | 858-915 | Mark area completed |
| `Show-CompletionChoice` | 1011-1060 | Prompt for next action |

**lib/metrics.ps1** (27 functions):
| Function | Lines | Description |
|----------|-------|-------------|
| `Get-GitState` | 1066-1092 | Capture current git state |
| `Get-FileOperations` | 1094-1145 | File ops between git states |
| `Get-GitCommits` | 1147-1191 | Commits since hash |
| `Append-Jsonl` | 1194-1220 | Append to JSONL file |
| `Log-ClaudeInvocation` | 1222-1316 | Log Claude CLI invocation |
| `Log-IterationManifest` | 1318-1459 | Create iteration manifest |
| `Log-FileOperations` | 1461-1478 | Log file ops to file |
| `Log-GitOperations` | 1480-1521 | Log git ops to file |
| `Log-StoryVerification` | 1523-1615 | Log story verification |
| `Append-SessionTimeline` | 1617-1625 | Append to session timeline |
| `Get-SprintTokenBudget` | 2432-2484 | Calculate token budget |
| `Measure-CodebaseHealth` | 3356-3440 | Track codebase health metrics |
| `Compare-HealthMetrics` | 3442-3501 | Compare health snapshots |
| `Test-TokenBudget` | 3503-3536 | Check token budget exceeded |
| `Get-StoryProgress` | 3778-3838 | Get story completion progress |
| `Save-StoryProgress` | 3840-3911 | Save story progress |
| `Log-StateTransition` | 4345-4353 | Log state machine transitions |
| `Measure-PhaseTimings` | 4359-4411 | Estimate phase timings |
| `Log-ErrorEvolution` | 4417-4426 | Log error patterns |
| `Log-ConfigChange` | 4433-4442 | Log config changes |
| `Log-TestDetails` | 4445-4512 | Log test results |
| `Get-ProcessMetrics` | 4515-4541 | Get CPU/memory metrics |
| `Log-ResourceUsage` | 4543-4582 | Log resource usage |
| `Get-PromptEffectiveness` | 4585-4614 | Measure prompt effectiveness |
| `Log-PromptEffectiveness` | 4616-4625 | Log prompt effectiveness |
| `Log-Skip` | 4628-4638 | Log story skip reason |
| `Record-Metric` | 6107-6194 | Record story metric to CSV |

**lib/quality.ps1** (12 functions):
| Function | Lines | Description |
|----------|-------|-------------|
| `Invoke-QualityReview` | 1631-1803 | LLM-as-judge quality review |
| `Get-StoryFailureContext` | 1805-1912 | Build retry context |
| `Get-DiffQualityScore` | 1914-2005 | Analyze diff quality |
| `Get-TestBaseline` | 2007-2063 | Capture test baseline |
| `Compare-TestBaseline` | 2065-2136 | Compare against baseline |
| `Update-TestBaseline` | 2138-2176 | Update baseline |
| `Import-HumanFeedback` | 2178-2203 | Read feedback file |
| `Get-FeedbackForStory` | 2205-2271 | Get feedback for story |
| `Format-ReviewPrompt` | 2490-2596 | Format review prompt |
| `Search-CriterionEvidence` | 2963-3086 | Search for acceptance evidence |
| `Test-CanRollback` | 4128-4187 | Check rollback safety |
| `Invoke-StoryRollback` | 4189-4238 | Rollback story changes |

**lib/prompts.ps1** (9 functions):
| Function | Lines | Description |
|----------|-------|-------------|
| `Get-RelevantFilesForStory` | 3268-3354 | Find files relevant to story |
| `Get-PromptEffectivenessHistory` | 3538-3575 | Query prompt effectiveness |
| `Get-PromptRecommendation` | 3577-3641 | Recommend prompt improvements |
| `Build-StoryPrompt` | 3643-3772 | Build complete story prompt |
| `Build-ResumePrompt` | 3913-3971 | Build resume prompt |
| `Invoke-ClaudeExploration` | 4644-4750 | Claude codebase exploration |
| `Invoke-FocusAreaExploration` | 4752-4882 | Claude focus area exploration |
| `Test-ShouldExplore` | 4884-4946 | Check if exploration needed |
| `Invoke-PeriodicExplorationIfNeeded` | 4948-4983 | Conditionally invoke exploration |

**Stays in ralph.ps1** (~27 functions + main block, ~2,500 lines):
| Function | Lines | Description |
|----------|-------|-------------|
| `Write-RalphBanner` | 706-732 | Display banner |
| `Write-IterationBanner` | 734-752 | Display iteration banner |
| `Invoke-ClaudeProcess` | 4985-5397 | Low-level Claude invocation |
| `Get-ClaudePath` | 5399-5434 | Resolve Claude CLI path |
| `Invoke-ClaudeForFocusArea` | 5436-5542 | Generate PRD via Claude |
| `Invoke-BatchPreFlight` | 5544-5655 | Batch pre-flight check |
| `Confirm-CommitMatchesStory` | 5657-5775 | Verify commit matches story |
| `Test-StoryAlreadyCommitted` | 5775-5834 | Check if story committed |
| `Complete-StoryAutomatically` | 5836-5912 | Auto-complete committed story |
| `Invoke-ClaudeForStory` | 5914-5955 | Invoke Claude for story |
| `Get-TestResults` | 5992-6039 | Run pytest, extract results |
| `Get-ErrorCategory` | 6041-6064 | Categorize error |
| `Get-EstimatedTokens` | 6066-6105 | Estimate token count |
| `Test-ShouldAbort` | 6200-6222 | Check abort conditions |
| `Test-MaxIterations` | 6224-6248 | Check max iterations |
| `Test-GracefulStopRequested` | 6254-6288 | Check graceful stop |
| `Clear-GracefulStopSignal` | 6290-6303 | Remove stop signal |
| `Request-GracefulStop` | 6305-6321 | Request graceful stop |
| `Get-SprintStatus` | 6327-6377 | Sprint completion status |
| `Start-InterviewQueueLoop` | 6383-6552 | Interview queue mode |
| `Start-TrueAutoLoop` | 6554-6631 | TrueAuto mode |
| `Start-StandardLoop` | 6633-6817 | Standard mode |
| `Show-RalphsReasoning` | 6823-6985 | Display scoring breakdown |
| `Get-RalphsChoiceUserInput` | 6987-7038 | Get user input |
| `Write-RalphsChoiceLog` | 7040-7066 | Log decisions |
| `Start-RalphsChoiceLoop` | 7068-7187 | Ralph's Choice mode |
| `Start-RalphsChoiceAutoLoop` | 7189-7391 | Ralph's Choice Auto mode |
| Main execution block | 7393-7444 | Parameter routing |

### Legacy Queue Format Locations

| File | Function | Lines | Type | What to Remove |
|------|----------|-------|------|----------------|
| ralph.ps1 | `Get-InterviewFocusAreas` | 797-799 | READ | `if ($queue.queue -and $queue.completedAreas)` branch |
| ralph.ps1 | `Update-QueueProgress` | 889-898 | WRITE | `$queue.completedAreas` and `$queue.currentIndex` updates |
| watch-lib.ps1 | `Show-QueueStatus` | 24, 35-36, 64-77 | READ | `$isLegacyFormat` branch + legacy display block |

---

## Phase 1: Extract Modules (Mechanical Move)

### Task 1: Create test helper and lib directory

**Files:**
- Create: `scripts/ralph/lib/` (directory)
- Create: `scripts/ralph/tests/test-helper.ps1`

**Step 1: Create directory**

```bash
mkdir -p scripts/ralph/lib
```

**Step 2: Write test helper**

Create `scripts/ralph/tests/test-helper.ps1` with a function that AST-parses all lib files + ralph.ps1:

```powershell
function Import-RalphFunctions {
    param(
        [string]$RalphDir,
        [switch]$GlobalScope,
        [string[]]$OnlyFunctions
    )

    $libDir = Join-Path $RalphDir 'lib'
    $scripts = @()

    # Source lib modules first (order matters for dependencies)
    if (Test-Path $libDir) {
        $scripts += Get-ChildItem -Path $libDir -Filter '*.ps1' | ForEach-Object { $_.FullName }
    }

    # Then ralph.ps1 itself
    $ralphScript = Join-Path $RalphDir 'ralph.ps1'
    if (Test-Path $ralphScript) {
        $scripts += $ralphScript
    }

    foreach ($scriptPath in $scripts) {
        $ast = [System.Management.Automation.Language.Parser]::ParseFile($scriptPath, [ref]$null, [ref]$null)
        $functions = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)

        foreach ($func in $functions) {
            if ($OnlyFunctions -and $OnlyFunctions -notcontains $func.Name) { continue }

            $funcDef = $func.Extent.Text
            if ($GlobalScope) {
                $globalFuncDef = $funcDef -replace '^function\s+([A-Za-z0-9_-]+)', 'function global:$1'
                Invoke-Expression $globalFuncDef
            } else {
                . ([scriptblock]::Create($funcDef))
            }
        }
    }
}
```

**Step 3: Commit**

```bash
git add scripts/ralph/lib scripts/ralph/tests/test-helper.ps1
git commit -m "refactor: add lib/ directory and test helper for Ralph module extraction"
```

---

### Task 2: Extract lib/sprint.ps1

**Files:**
- Create: `scripts/ralph/lib/sprint.ps1`
- Modify: `scripts/ralph/ralph.ps1` (delete moved functions)

**Step 1: Create lib/sprint.ps1**

Extract these 15 functions from ralph.ps1 (cut, not copy) into `scripts/ralph/lib/sprint.ps1`:

```
Get-RalphConfig              (lines 77-108)
Get-SprintHistory            (lines 117-136)
Update-SprintHistory         (lines 139-198)
Save-SprintArchive           (lines 201-318)
Test-ShouldGenerateNewPRD    (lines 321-373)
Write-JsonNoBom              (lines 681-700)
New-SeedPRD                  (lines 925-1005)
New-SprintReport             (lines 2273-2430)
Get-SprintRetrospective      (lines 3088-3211)
Get-RetrospectiveContext      (lines 3213-3262)
Get-IndependentStories       (lines 3973-4029)
Update-LearningDb            (lines 4031-4068)
Get-LearningContext          (lines 4070-4126)
Build-DependencyGraph        (lines 4240-4294)
Get-ExecutableStories        (lines 4296-4339)
```

Add file header:
```powershell
# scripts/ralph/lib/sprint.ps1
# Sprint lifecycle: PRD generation, archive, history, learning, dependencies
```

**Step 2: Delete those function bodies from ralph.ps1**

Remove the 15 function blocks from ralph.ps1. Do NOT add placeholder comments — just delete.

**Step 3: Verify syntax**

```powershell
pwsh -Command "[System.Management.Automation.Language.Parser]::ParseFile('scripts/ralph/lib/sprint.ps1', [ref]$null, [ref]$errors); if ($errors) { $errors | ForEach-Object { Write-Error $_ } } else { Write-Host 'OK' }"
```

**Step 4: Commit**

```bash
git add scripts/ralph/lib/sprint.ps1 scripts/ralph/ralph.ps1
git commit -m "refactor: extract 15 sprint functions to lib/sprint.ps1"
```

---

### Task 3: Extract lib/scoring.ps1

**Files:**
- Create: `scripts/ralph/lib/scoring.ps1`
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Create lib/scoring.ps1**

Extract these 7 functions:

```
Get-GitActivityByArea        (lines 380-439)
Get-FocusAreaScore           (lines 442-538)
Get-AllFocusAreaScores       (lines 540-567)
Get-StayOrSwitchDecision     (lines 569-675)
Get-OptimalNextStory         (lines 2727-2822)
Get-StoryFileTouches         (lines 2824-2877)
Test-FileConflict            (lines 2879-2961)
```

Header:
```powershell
# scripts/ralph/lib/scoring.ps1
# Ralph's Choice: focus area scoring, git activity analysis, story optimization
```

**Step 2: Delete from ralph.ps1**

**Step 3: Verify syntax**

```powershell
pwsh -Command "[System.Management.Automation.Language.Parser]::ParseFile('scripts/ralph/lib/scoring.ps1', [ref]$null, [ref]$errors); if ($errors) { $errors | ForEach-Object { Write-Error $_ } } else { Write-Host 'OK' }"
```

**Step 4: Commit**

```bash
git add scripts/ralph/lib/scoring.ps1 scripts/ralph/ralph.ps1
git commit -m "refactor: extract 7 scoring functions to lib/scoring.ps1"
```

---

### Task 4: Extract lib/queue.ps1

**Files:**
- Create: `scripts/ralph/lib/queue.ps1`
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Create lib/queue.ps1**

Extract these 7 functions:

```
Get-QueueData                (lines 758-775)
Get-InterviewFocusAreas      (lines 777-804)
Get-InterviewContext         (lines 806-814)
Update-ContextFromPRD        (lines 816-840)
Get-NextQueuedFocusArea      (lines 842-856)
Update-QueueProgress         (lines 858-915)
Show-CompletionChoice        (lines 1011-1060)
```

Also move the backward-compatible aliases if present (lines ~918-919).

Header:
```powershell
# scripts/ralph/lib/queue.ps1
# Interview queue: focus area processing, progress tracking, completion flow
```

**Step 2: Delete from ralph.ps1**

**Step 3: Verify syntax**

**Step 4: Commit**

```bash
git add scripts/ralph/lib/queue.ps1 scripts/ralph/ralph.ps1
git commit -m "refactor: extract 7 queue functions to lib/queue.ps1"
```

---

### Task 5: Extract lib/metrics.ps1

**Files:**
- Create: `scripts/ralph/lib/metrics.ps1`
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Create lib/metrics.ps1**

Extract these 27 functions (the largest module):

```
Get-GitState                 (lines 1066-1092)
Get-FileOperations           (lines 1094-1145)
Get-GitCommits               (lines 1147-1191)
Append-Jsonl                 (lines 1194-1220)
Log-ClaudeInvocation         (lines 1222-1316)
Log-IterationManifest        (lines 1318-1459)
Log-FileOperations           (lines 1461-1478)
Log-GitOperations            (lines 1480-1521)
Log-StoryVerification        (lines 1523-1615)
Append-SessionTimeline       (lines 1617-1625)
Get-SprintTokenBudget        (lines 2432-2484)
Measure-CodebaseHealth       (lines 3356-3440)
Compare-HealthMetrics        (lines 3442-3501)
Test-TokenBudget             (lines 3503-3536)
Get-StoryProgress            (lines 3778-3838)
Save-StoryProgress           (lines 3840-3911)
Log-StateTransition          (lines 4345-4353)
Measure-PhaseTimings         (lines 4359-4411)
Log-ErrorEvolution           (lines 4417-4426)
Log-ConfigChange             (lines 4433-4442)
Log-TestDetails              (lines 4445-4512)
Get-ProcessMetrics           (lines 4515-4541)
Log-ResourceUsage            (lines 4543-4582)
Get-PromptEffectiveness      (lines 4585-4614)
Log-PromptEffectiveness      (lines 4616-4625)
Log-Skip                     (lines 4628-4638)
Record-Metric                (lines 6107-6194)
```

Header:
```powershell
# scripts/ralph/lib/metrics.ps1
# Metrics, logging, instrumentation: CSV tracking, git state, health, tokens
```

**Step 2: Delete from ralph.ps1**

**Step 3: Verify syntax**

**Step 4: Commit**

```bash
git add scripts/ralph/lib/metrics.ps1 scripts/ralph/ralph.ps1
git commit -m "refactor: extract 27 metrics functions to lib/metrics.ps1"
```

---

### Task 6: Extract lib/quality.ps1

**Files:**
- Create: `scripts/ralph/lib/quality.ps1`
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Create lib/quality.ps1**

Extract these 12 functions:

```
Invoke-QualityReview         (lines 1631-1803)
Get-StoryFailureContext      (lines 1805-1912)
Get-DiffQualityScore         (lines 1914-2005)
Get-TestBaseline             (lines 2007-2063)
Compare-TestBaseline         (lines 2065-2136)
Update-TestBaseline          (lines 2138-2176)
Import-HumanFeedback         (lines 2178-2203)
Get-FeedbackForStory         (lines 2205-2271)
Format-ReviewPrompt          (lines 2490-2596)
Search-CriterionEvidence     (lines 2963-3086)
Test-CanRollback             (lines 4128-4187)
Invoke-StoryRollback         (lines 4189-4238)
```

Header:
```powershell
# scripts/ralph/lib/quality.ps1
# Quality gates: review, baselines, regression, feedback, rollback
```

**Step 2: Delete from ralph.ps1**

**Step 3: Verify syntax**

**Step 4: Commit**

```bash
git add scripts/ralph/lib/quality.ps1 scripts/ralph/ralph.ps1
git commit -m "refactor: extract 12 quality functions to lib/quality.ps1"
```

---

### Task 7: Extract lib/prompts.ps1

**Files:**
- Create: `scripts/ralph/lib/prompts.ps1`
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Create lib/prompts.ps1**

Extract these 9 functions:

```
Get-RelevantFilesForStory         (lines 3268-3354)
Get-PromptEffectivenessHistory    (lines 3538-3575)
Get-PromptRecommendation          (lines 3577-3641)
Build-StoryPrompt                 (lines 3643-3772)
Build-ResumePrompt                (lines 3913-3971)
Invoke-ClaudeExploration          (lines 4644-4750)
Invoke-FocusAreaExploration       (lines 4752-4882)
Test-ShouldExplore                (lines 4884-4946)
Invoke-PeriodicExplorationIfNeeded (lines 4948-4983)
```

Header:
```powershell
# scripts/ralph/lib/prompts.ps1
# Prompt building: story prompts, exploration, context injection, effectiveness
```

**Step 2: Delete from ralph.ps1**

**Step 3: Verify syntax**

**Step 4: Commit**

```bash
git add scripts/ralph/lib/prompts.ps1 scripts/ralph/ralph.ps1
git commit -m "refactor: extract 9 prompt functions to lib/prompts.ps1"
```

---

### Task 8: Add dot-source block to ralph.ps1

**Files:**
- Modify: `scripts/ralph/ralph.ps1` (add dot-source block after script variable declarations)

**Step 1: Add dot-source block**

Insert after the script variable declarations (around line 65, after `$script:SessionLogDir`), before any function definitions:

```powershell
# Load domain modules
$script:LibPath = Join-Path $PSScriptRoot 'lib'
. "$script:LibPath\sprint.ps1"
. "$script:LibPath\scoring.ps1"
. "$script:LibPath\queue.ps1"
. "$script:LibPath\metrics.ps1"
. "$script:LibPath\quality.ps1"
. "$script:LibPath\prompts.ps1"
```

**Step 2: Verify ralph.ps1 parses cleanly**

```powershell
pwsh -Command "[System.Management.Automation.Language.Parser]::ParseFile('scripts/ralph/ralph.ps1', [ref]$null, [ref]$errors); if ($errors) { $errors | ForEach-Object { Write-Error $_ } } else { Write-Host 'OK' }"
```

**Step 3: Commit**

```bash
git add scripts/ralph/ralph.ps1
git commit -m "refactor: add dot-source block for lib modules in ralph.ps1"
```

---

### Task 9: Update test sourcing to use test-helper.ps1

**Files:**
- Modify: ALL test files that AST-parse ralph.ps1

**Step 1: Update tests using Pattern A (Full AST + Global Scope)**

These 9 test files need their BeforeAll block updated to use `Import-RalphFunctions`:

| Test File | Current Source |
|-----------|---------------|
| `Ralph.Tests.ps1` | ralph.ps1 |
| `QualityGate.Tests.ps1` | ralph.ps1 |
| `E2E.Tests.ps1` | ralph.ps1 |
| `Archive.Tests.ps1` | ralph.ps1 |
| `SprintCompletion.Tests.ps1` | ralph.ps1 |
| `PreFlight.Tests.ps1` | ralph.ps1 |

Replace the AST parsing block in each BeforeAll with:

```powershell
# Source test helper
. (Join-Path $PSScriptRoot 'test-helper.ps1')
Import-RalphFunctions -RalphDir $script:RalphDir -GlobalScope
```

**Step 2: Update tests using Pattern B (Selective AST)**

These 4 test files use selective sourcing and need a similar update:

| Test File | Current Source |
|-----------|---------------|
| `ParallelExecution.Tests.ps1` | ralph.ps1 (selective) |
| `AdaptivePrompt.Tests.ps1` | ralph.ps1 (selective) |
| `ReviewAgent.Tests.ps1` | ralph.ps1 (selective) |
| `GracefulStop.Tests.ps1` | ralph.ps1 (selective) |

Replace with:

```powershell
. (Join-Path $PSScriptRoot 'test-helper.ps1')
Import-RalphFunctions -RalphDir $script:RalphDir -OnlyFunctions @('Function1', 'Function2', ...)
```

Keep the existing `$neededFunctions` lists — just change the sourcing mechanism.

**Step 3: Leave unchanged**

These test files source OTHER scripts (not ralph.ps1) and need no changes:
- `SmartQueue.Tests.ps1` → sources launcher.ps1
- `Launcher.Tests.ps1` → sources launcher.ps1
- `Interview.Tests.ps1` → sources interview.ps1
- `Watch.Tests.ps1` → no sourcing (inline helpers)
- `Exploration.Tests.ps1` → inline function definitions

**Step 4: Run full Pester suite**

```powershell
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```

All tests must pass. If any fail, the issue is a function that wasn't moved correctly or a `$script:` variable scope issue. Fix individually.

**Step 5: Commit**

```bash
git add scripts/ralph/tests/
git commit -m "refactor: update test sourcing to use test-helper.ps1 for modular imports"
```

---

### Task 10: Phase 1 verification and cleanup

**Step 1: Verify ralph.ps1 line count**

```powershell
(Get-Content scripts/ralph/ralph.ps1).Count
# Expected: ~2,500 lines (27 functions + main block)
```

**Step 2: Verify all lib modules parse**

```powershell
Get-ChildItem scripts/ralph/lib/*.ps1 | ForEach-Object {
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$null, [ref]$errors)
    if ($errors) { Write-Error "$($_.Name): $errors" } else { Write-Host "$($_.Name): OK" }
}
```

**Step 3: Run full Pester suite**

```powershell
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```

**Step 4: Verify function count**

```powershell
$totalFuncs = 0
Get-ChildItem scripts/ralph/lib/*.ps1 | ForEach-Object {
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($_.FullName, [ref]$null, [ref]$null)
    $count = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true).Count
    Write-Host "$($_.Name): $count functions"
    $totalFuncs += $count
}
$ralphAst = [System.Management.Automation.Language.Parser]::ParseFile('scripts/ralph/ralph.ps1', [ref]$null, [ref]$null)
$ralphCount = $ralphAst.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true).Count
Write-Host "ralph.ps1: $ralphCount functions"
Write-Host "Total: $($totalFuncs + $ralphCount) (expected: 82)"
```

**Step 5: Commit tag**

```bash
git commit --allow-empty -m "refactor: Phase 1 complete — ralph.ps1 decomposed into 6 modules"
```

---

## Phase 2: State Layer Cleanup

### Task 11: Add Save-StateFile utility to lib/sprint.ps1

**Files:**
- Modify: `scripts/ralph/lib/sprint.ps1`

**Step 1: Add Save-StateFile function**

Add at the top of `lib/sprint.ps1` (after the header comment):

```powershell
function Save-StateFile {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][object]$Data,
        [int]$Depth = 10
    )
    $tempPath = "$Path.tmp"
    $Data | ConvertTo-Json -Depth $Depth | Set-Content -Path $tempPath -Encoding UTF8
    Move-Item -Path $tempPath -Destination $Path -Force
}
```

**Step 2: Update Write-JsonNoBom to use Save-StateFile internally**

```powershell
function Write-JsonNoBom {
    param([string]$Path, [string]$Content)
    $tempPath = "$Path.tmp"
    [System.IO.File]::WriteAllText($tempPath, $Content)
    Move-Item -Path $tempPath -Destination $Path -Force
}
```

**Step 3: Commit**

```bash
git add scripts/ralph/lib/sprint.ps1
git commit -m "refactor: add Save-StateFile atomic write utility"
```

---

### Task 12: Add state accessors to lib/sprint.ps1

**Files:**
- Modify: `scripts/ralph/lib/sprint.ps1`

**Step 1: Add Get-Sprint and Save-Sprint functions**

```powershell
function Get-Sprint {
    param([string]$Path = $script:PrdFile)
    if (-not (Test-Path $Path)) { return $null }
    try {
        Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Warning "Failed to parse sprint file: $Path"
        return $null
    }
}

function Save-Sprint {
    param(
        [Parameter(Mandatory)][object]$Sprint,
        [string]$Path = $script:PrdFile
    )
    Save-StateFile -Path $Path -Data $Sprint
}

function Update-StoryStatus {
    param(
        [Parameter(Mandatory)][string]$StoryId,
        [bool]$Passes = $false,
        [string]$Notes = '',
        [string]$Path = $script:PrdFile
    )
    $sprint = Get-Sprint -Path $Path
    if (-not $sprint) { return }
    foreach ($story in $sprint.userStories) {
        if ($story.id -eq $StoryId) {
            $story.passes = $Passes
            if ($Notes) { $story.notes = $Notes }
        }
    }
    Save-Sprint -Sprint $sprint -Path $Path
}
```

**Step 2: Commit**

```bash
git add scripts/ralph/lib/sprint.ps1
git commit -m "refactor: add sprint state accessors (Get-Sprint, Save-Sprint, Update-StoryStatus)"
```

---

### Task 13: Add state accessors to lib/queue.ps1

**Files:**
- Modify: `scripts/ralph/lib/queue.ps1`

**Step 1: Add Get-Queue and Save-Queue functions**

```powershell
function Get-Queue {
    param([string]$Path = $script:QueueFile)
    if (-not (Test-Path $Path)) { return $null }
    try {
        Get-Content -Path $Path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Warning "Failed to parse queue file: $Path"
        return $null
    }
}

function Save-Queue {
    param(
        [Parameter(Mandatory)][object]$Queue,
        [string]$Path = $script:QueueFile
    )
    Save-StateFile -Path $Path -Data $Queue
}
```

**Step 2: Update existing functions to use Get-Queue/Save-Queue**

Replace raw `Get-Content | ConvertFrom-Json` calls in `Get-QueueData`, `Get-InterviewFocusAreas`, `Update-QueueProgress`, etc. with calls to `Get-Queue` and `Save-Queue`.

**Step 3: Commit**

```bash
git add scripts/ralph/lib/queue.ps1
git commit -m "refactor: add queue state accessors and wire existing functions"
```

---

### Task 14: Remove legacy queue format from ralph.ps1 / queue.ps1

**Files:**
- Modify: `scripts/ralph/lib/queue.ps1`

**Step 1: Remove legacy branch from Get-InterviewFocusAreas**

Delete the legacy format handling (the `if ($queue.queue -and $queue.completedAreas)` branch, lines ~797-799 equivalent in lib/queue.ps1).

**Step 2: Remove legacy updates from Update-QueueProgress**

Delete the `completedAreas` and `currentIndex` update blocks (lines ~889-898 equivalent).

**Step 3: Run Pester tests**

```powershell
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```

**Step 4: Commit**

```bash
git add scripts/ralph/lib/queue.ps1
git commit -m "refactor: remove legacy queue format support from queue module"
```

---

### Task 15: Remove legacy queue format from watch-lib.ps1

**Files:**
- Modify: `scripts/ralph/watch-lib.ps1`

**Step 1: Remove legacy format detection and display**

In `Show-QueueStatus`:
- Remove `$isLegacyFormat` variable (line ~24)
- Remove the legacy `else` branch (lines ~35-36 count calculation)
- Remove legacy display block (lines ~64-77)
- Simplify to only handle `focusAreas` format

If a queue.json with legacy format is encountered, display a warning:
```powershell
if (-not $queue.focusAreas) {
    Write-Host "  [!] Legacy queue format detected - run interview.ps1 to create new format" -ForegroundColor Yellow
    return
}
```

**Step 2: Run Watch.Tests.ps1**

```powershell
Invoke-Pester -Path 'scripts/ralph/tests/Watch.Tests.ps1' -Output Detailed
```

**Step 3: Commit**

```bash
git add scripts/ralph/watch-lib.ps1
git commit -m "refactor: remove legacy queue format from watch-lib.ps1"
```

---

### Task 16: Phase 2 verification

**Step 1: Run full Pester suite**

```powershell
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```

**Step 2: Manual smoke test**

Run through this sequence manually:
1. `.\scripts\ralph\launcher.ps1` → Select mode → Verify focus area menu works
2. Start interview mode → Create queue → Verify queue.json uses new format only
3. Check watch dashboard displays queue correctly

**Step 3: Commit tag**

```bash
git commit --allow-empty -m "refactor: Phase 2 complete — state layer unified, legacy queue format removed"
```

---

## Phase 3: Script-Scoped State Consolidation

### Task 17: Create $script:State hashtable

**Files:**
- Modify: `scripts/ralph/ralph.ps1`

**Step 1: Replace scattered $script:* declarations**

Replace the variable block (lines ~41-57) with a single hashtable:

```powershell
$script:State = @{
    SessionId                = (Get-Date -Format 'yyyy-MM-dd_HHmmss')
    IterationCount           = 0
    ConsecutiveFailures      = 0
    SessionStartTime         = Get-Date
    CurrentMode              = ''
    CurrentRetryCount        = 0
    LastFocusAreaId          = ''
    LastStoryId              = ''
    StoriesSinceExploration  = 0
    LastExplorationSummary   = ''
    LastExplorationTime      = $null
    SprintExplorationContext = ''
    LastExplorationCommit    = ''
}
```

Keep path variables as separate `$script:*` variables (they're config, not mutable state):
```powershell
$script:ProjectRoot
$script:RalphDir
$script:QueueFile
$script:ConfigFile
$script:PrdFile
$script:ProgressFile
$script:MetricsFile
$script:PromptFile
$script:LogDir
$script:ArchiveDir
$script:SprintHistoryFile
$script:ExplorationContextFile
$script:SessionLogDir
$script:Config
$script:LibPath
```

**Step 2: Update ralph.ps1 functions to use $script:State**

Find-and-replace in ralph.ps1:
- `$script:SessionId` → `$script:State.SessionId`
- `$script:IterationCount` → `$script:State.IterationCount`
- `$script:ConsecutiveFailures` → `$script:State.ConsecutiveFailures`
- `$script:CurrentMode` → `$script:State.CurrentMode`
- `$script:CurrentRetryCount` → `$script:State.CurrentRetryCount`
- `$script:LastFocusAreaId` → `$script:State.LastFocusAreaId`
- `$script:LastStoryId` → `$script:State.LastStoryId`
- `$script:StoriesSinceExploration` → `$script:State.StoriesSinceExploration`
- `$script:LastExplorationSummary` → `$script:State.LastExplorationSummary`
- `$script:LastExplorationTime` → `$script:State.LastExplorationTime`
- `$script:SprintExplorationContext` → `$script:State.SprintExplorationContext`
- `$script:LastExplorationCommit` → `$script:State.LastExplorationCommit`
- `$script:SessionStartTime` → `$script:State.SessionStartTime`

**Step 3: Update lib module functions that reference $script:State variables**

Search all `lib/*.ps1` files for `$script:SessionId`, `$script:IterationCount`, etc. and update to `$script:State.SessionId`, etc.

**Step 4: Commit**

```bash
git add scripts/ralph/ralph.ps1 scripts/ralph/lib/
git commit -m "refactor: consolidate script-scoped state into State hashtable"
```

---

### Task 18: Update tests for $script:State

**Files:**
- Modify: Test files that mock `$script:*` state variables

**Step 1: Update test variable setup**

Tests that set `$script:SessionId = '...'` (and `$global:SessionId`) need to change to:

```powershell
$script:State = @{
    SessionId = 'test-session'
    IterationCount = 0
    ConsecutiveFailures = 0
    CurrentRetryCount = 0
    LastFocusAreaId = ''
    LastStoryId = ''
    StoriesSinceExploration = 0
    LastExplorationSummary = ''
    LastExplorationTime = $null
    SprintExplorationContext = ''
    LastExplorationCommit = ''
    SessionStartTime = Get-Date
    CurrentMode = 'Standard'
}
$global:State = $script:State
```

**Affected test files:**
- `Ralph.Tests.ps1` — mocks SessionId, IterationCount, CurrentRetryCount
- `E2E.Tests.ps1` — mocks SessionId, IterationCount, ConsecutiveFailures, CurrentRetryCount
- `ParallelExecution.Tests.ps1` — mocks SessionId
- `Archive.Tests.ps1` — (check for state var usage)
- `SprintCompletion.Tests.ps1` — (check for state var usage)

**Step 2: Run Pester suite**

```powershell
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```

Fix any failures caused by the state variable migration.

**Step 3: Commit**

```bash
git add scripts/ralph/tests/
git commit -m "refactor: update tests for consolidated State hashtable"
```

---

### Task 19: Phase 3 verification

**Step 1: Full Pester suite**

```powershell
Invoke-Pester -Path 'scripts/ralph/tests' -Output Detailed
```

**Step 2: Verify no orphaned $script:* state references**

```powershell
# Should return no matches for mutable state vars (path vars are OK)
$stateVars = @('SessionId', 'IterationCount', 'ConsecutiveFailures', 'CurrentMode',
    'CurrentRetryCount', 'LastFocusAreaId', 'LastStoryId', 'StoriesSinceExploration',
    'LastExplorationSummary', 'LastExplorationTime', 'SprintExplorationContext',
    'LastExplorationCommit', 'SessionStartTime')

foreach ($var in $stateVars) {
    $matches = Select-String -Path scripts/ralph/*.ps1, scripts/ralph/lib/*.ps1 -Pattern "\`$script:$var\b" -SimpleMatch
    if ($matches) {
        Write-Warning "Orphaned reference: `$script:$var"
        $matches | ForEach-Object { Write-Host "  $($_.Filename):$($_.LineNumber)" }
    }
}
```

All mutable state references should use `$script:State.$var` pattern.

**Step 3: Final commit**

```bash
git commit --allow-empty -m "refactor: Phase 3 complete — all script state consolidated into State hashtable"
```

---

## Post-Refactor Checklist

- [ ] ralph.ps1 is ~2,500 lines (down from 7,444)
- [ ] 6 lib modules parse without errors
- [ ] Full Pester suite passes (all 138+ tests)
- [ ] No legacy queue format code remains
- [ ] All state file writes use atomic temp-then-rename
- [ ] No orphaned `$script:*` mutable state variables
- [ ] Manual smoke test: launcher → interview → queue → watch all work
