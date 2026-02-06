# Carlini-Inspired Ralph Improvements

**Date:** 2026-02-07
**Status:** Design Document (no code changes)
**Reference:** [Building a C Compiler with Claude](https://www.anthropic.com/engineering/building-c-compiler) by Nicholas Carlini

## Context

Nicholas Carlini's article describes running 16 parallel Claude agents that autonomously built a 100K-line Rust C compiler in 2 weeks. The core thesis: **environmental design beats prompt engineering** — invest in tests, feedback loops, and infrastructure rather than better prompts.

Ralph already has strong foundations (test gating via `Compare-TestBaseline`, 3-tier healing in `healing.ps1`, learning system in `learning.ps1`/`sprint.ps1`, phase tracking in `metrics.ps1`). This plan closes the gaps between Ralph's current architecture and Carlini's proven patterns.

---

## Current Architecture Reference

Before proposing changes, here's what exists today:

| Component | Location | Key Functions |
|-----------|----------|---------------|
| Metrics | `lib/metrics.ps1` | `Record-Metric` (24-col CSV), `Measure-PhaseTimings`, `Get-TestResults`, `Confirm-CriteriaEvidence` |
| Healing | `lib/healing.ps1` | `Invoke-TieredHealthCheck` (3-tier), `Invoke-HealingSession` (max 3 attempts), `Invoke-PostIterationHealing` |
| Learning | `lib/learning.ps1` | `Get-RecurringErrors`, `Get-PromptMetrics`, `Save-SprintLearning`, `Get-LearningInsights` |
| Quality | `lib/quality.ps1` | `Compare-TestBaseline`, `Update-TestBaseline`, `Get-DiffQualityScore` |
| Prompts | `lib/prompts.ps1` | `Build-StoryPrompt` (adaptive, 12+ sections), `Get-PromptRecommendation` |
| Sprint | `lib/sprint.ps1` | `Save-SprintArchive`, `Invoke-StoryDecomposition`, `Mark-AsHardStory`, `Update-LearningDb`, `Get-LearningContext` |
| Resolution | `lib/claude.ps1` | `Resolve-ClaudeResult` (evidence gate, regression detection, rollback, learning DB) |
| Watch | `watch.ps1` | Dashboard with session stats, cost, errors, productivity, phase breakdown |
| Feature flags | `config/ralph-config.json` | `flags.phaseTracking`, `flags.autoGeneratePesterTests`, `flags.llmAsJudgeQuality`, `flags.acceptanceDrivenBackpressure` |

---

## Improvement Areas (Priority Order)

### 0. Sprint Diagnostics Output (S)

**Carlini Principle:** "Pre-compute summary statistics so Claude doesn't recompute."

**Current Gap:** Ralph tracks metrics in CSV (`session/metrics.csv`) and `state/sprint_history.json`, but there's no unified diagnostics view. After enabling a feature flag, you have to manually dig through metrics to measure impact. `watch.ps1` shows session-level stats but doesn't show which flags are active or their measured effect.

**What Exists Today:**
- `Record-Metric` writes 24-column CSV rows per iteration
- `Save-SprintArchive` captures `_archiveMetadata` (completed/total stories, reason) + health snapshot via `Measure-CodebaseHealth`
- `Update-SprintHistory` tracks per-sprint stats (stories completed/total, focusArea, archiveFile)
- `watch.ps1` reads CSV and shows session stats, cost, errors, productivity, phase breakdown
- `ralph-config.json` has 4 flags under `flags`: `autoGeneratePesterTests`, `llmAsJudgeQuality`, `phaseTracking`, `acceptanceDrivenBackpressure`

**Design:**

Add `Get-SprintDiagnostics` function in `lib/metrics.ps1` that:

1. Reads current `ralph-config.json` flags (all keys under `flags` + new improvement flags)
2. Reads `session/metrics.csv` for current sprint's iteration data
3. Reads `state/sprint_history.json` for rolling 3-sprint comparison
4. Computes derived metrics: first-attempt success rate, healing session count, regression count
5. Emits structured JSON to `session/diagnostics.json`

```powershell
function Get-SprintDiagnostics {
    param(
        [int]$SprintNumber = 0,
        [object]$Prd = $null
    )
    # Returns hashtable with: active_flags, metrics, comparison, flag_impact
}
```

**Output schema:**

```json
{
  "sprint": 70,
  "timestamp": "2026-02-07T14:30:00",
  "active_flags": {
    "regressionGuard": true,
    "testSubsampling": false,
    "learningInjection": false,
    "roleSpecialization": false,
    "adaptivePruning": false,
    "deltaDebugging": false,
    "parallelExecution": false
  },
  "metrics": {
    "stories_attempted": 12,
    "stories_passed_first_try": 8,
    "stories_passed_with_retry": 3,
    "stories_failed": 1,
    "first_attempt_success_rate": 0.67,
    "overall_success_rate": 0.92,
    "avg_duration_minutes": 4.2,
    "healing_sessions": 2,
    "regressions_caught": 0,
    "regressions_missed": 0
  },
  "comparison": {
    "vs_last_3_sprints": {
      "first_attempt_rate_delta": "+0.05",
      "duration_delta": "-0.8min",
      "healing_delta": "-1"
    }
  },
  "flag_impact": {
    "regressionGuard": {
      "regressions_caught": 0,
      "reverts_triggered": 0,
      "overhead_seconds": 45
    }
  }
}
```

**Integration points:**

1. **`Save-SprintArchive`** (`lib/sprint.ps1:250`): Call `Get-SprintDiagnostics` after health snapshot. Add diagnostics to `_archiveMetadata` so sprint archives include a diagnostics snapshot for historical comparison.

2. **`watch.ps1`** (line ~167, after metrics dashboard): Add a one-line diagnostics summary reading `session/diagnostics.json`:
   ```
   [DIAG] Flags: regressionGuard | 1st-try: 67% (+5%) | Heal: 2 (-1) | Regress: 0
   ```

3. **`ralph-config.json`**: Extend `flags` with new improvement flags (initially all `false`):
   ```json
   "flags": {
     "autoGeneratePesterTests": false,
     "llmAsJudgeQuality": false,
     "phaseTracking": true,
     "acceptanceDrivenBackpressure": false,
     "regressionGuard": false,
     "testSubsampling": false,
     "learningInjection": false,
     "roleSpecialization": false,
     "adaptivePruning": false,
     "deltaDebugging": false,
     "parallelExecution": false
   }
   ```

**Deriving metrics from existing data:**
- `stories_passed_first_try`: From CSV, count stories where `retry_count` = 0 and `success` = true
- `healing_sessions`: Count rows where `story_id` starts with `HEALING-`
- `regressions_caught`: From `session/healing_log.jsonl`, count events with `event: "regression_detected"`
- Comparison: Read last 3 entries from `state/sprint_history.json`, compute deltas

**Files touched:** `lib/metrics.ps1`, `watch.ps1`, `lib/sprint.ps1` (archive integration), `config/ralph-config.json`

**Effort:** S — Reads existing metrics + config flags, formats output. No new data collection needed.

---

### 1. Oracle-Based Regression Guard (S)

**Carlini Principle:** Use a known-good reference (GCC) to verify outputs without manual inspection.

**Current Gap:** Ralph already has `Compare-TestBaseline` in `lib/quality.ps1:468` which detects regressions after story completion, and `Resolve-ClaudeResult` in `lib/claude.ps1:886-904` handles auto-rollback. However, the baseline is only captured/updated **after** successful stories (line 960-962). There's no **pre-story** baseline capture — if the baseline file is stale or was corrupted, regression detection is blind.

**What Exists Today:**
- `Compare-TestBaseline` (quality.ps1:468): Parses test results, compares against `config/test_baseline.json`, detects regressions with subset-run awareness
- `Update-TestBaseline` (quality.ps1:560): Updates baseline only when current run covers >= existing test count
- `Resolve-ClaudeResult` (claude.ps1:886-904): On success, calls `Compare-TestBaseline`, auto-rollbacks via `Invoke-StoryRollback` if regression detected
- Limitation: Baseline staleness — if 5 stories pass without updating (e.g., subset runs), baseline drifts

**Design:**

Add `Test-RegressionBaseline` function in `lib/healing.ps1`:

```powershell
function Test-RegressionBaseline {
    <#
    .SYNOPSIS
        Capture fresh test baseline before each story starts.
        Run pytest tests/ --tb=no -q, parse pass count, save to session/test_baseline.json.
    .DESCRIPTION
        Called at the START of each story (before Claude runs), not at the end.
        This ensures Compare-TestBaseline always has a fresh reference.
        If the pre-story run itself fails, log warning but don't block story execution.
    .RETURNS
        Hashtable: total, passed, failed, capturedAt
    #>
    param()

    $pytestArgs = "tests/ --tb=no -q --no-header"
    # Run pytest, parse "X passed, Y failed"
    # Save to session/test_baseline.json (not config/ — session-scoped)
    # Return results hashtable
}
```

**Integration into `Invoke-ClaudeForStory`** (`ralph.ps1:498`):

```powershell
# Before building prompt, capture fresh baseline
if ($config.flags.regressionGuard) {
    $preBaseline = Test-RegressionBaseline
    if ($preBaseline) {
        Write-Host "  Baseline: $($preBaseline.passed)/$($preBaseline.total) tests passing" -ForegroundColor DarkGray
    }
}
```

**Post-story validation change in `Resolve-ClaudeResult`** (`claude.ps1:886`):

When `regressionGuard` flag is enabled, the existing `Compare-TestBaseline` call uses the freshly-captured pre-story baseline instead of the potentially-stale `config/test_baseline.json`. The key difference: baseline is always fresh (captured minutes ago, not hours/days ago).

**Revert behavior:** On regression detection, the existing `Invoke-StoryRollback` handles revert. Add diagnostic context to the retry prompt:

```powershell
# In failure context for retried story
"REGRESSION: This story caused test count to drop from $baseline.passed to $current.passed.
The following tests now fail that previously passed: [parsed from pytest diff]"
```

**New config:**

```json
"flags": {
  "regressionGuard": true
}
```

No new config section needed — uses existing `selfHealing.pytestArgs` for the baseline run, existing `regression.blockOnRegression` / `regression.autoRollback` for behavior.

**Files touched:** `lib/healing.ps1` (add `Test-RegressionBaseline`), `ralph.ps1` (wire into `Invoke-ClaudeForStory`), `lib/claude.ps1` (use fresh baseline in `Resolve-ClaudeResult`)

**Effort:** S — ~50 lines. Highest impact/effort ratio. Leverages existing `Compare-TestBaseline` and rollback infrastructure.

---

### 2. Deterministic Test Subsampling (S)

**Carlini Principle:** `--fast` flag running 1-10% random sample, deterministic per-agent but random across runs.

**Current Gap:** Each iteration runs the focus area's test suite. `selfHealing.fastPytestArgs` already uses `-x -p randomly --randomly-seed=random` for fast health checks, but story iterations always run the same test scope. There's no "quick feedback" mode for mid-implementation iterations vs "thorough validation" for completion attempts.

**What Exists Today:**
- `selfHealing.pytestArgs`: `"tests/ --tb=short -q --no-header"` (full suite)
- `selfHealing.fastPytestArgs`: `"tests/ --tb=line -q --no-header -x -p randomly --randomly-seed=random"` (fast, random order, stop-on-first-fail)
- Focus area test patterns in `focusAreaTests` config section
- `Invoke-TieredHealthCheck` already has a cadence system (`fullRunCadence: 3`) for when to run full vs fast checks

**Design:**

Add `Get-SubsampledTestCommand` function in `lib/healing.ps1`:

```powershell
function Get-SubsampledTestCommand {
    <#
    .SYNOPSIS
        Generate pytest command with deterministic subsampling.
    .PARAMETER StoryId
        Story ID used as seed for deterministic selection
    .PARAMETER IsCompletionAttempt
        If true, always run full suite (no subsampling)
    .PARAMETER IterationNumber
        Current iteration number (full suite every N iterations)
    .RETURNS
        String: pytest command with appropriate flags
    #>
    param(
        [string]$StoryId,
        [bool]$IsCompletionAttempt = $false,
        [int]$IterationNumber = 0
    )

    $config = Get-RalphConfig
    $subsamplingConfig = $config.selfHealing.subsampling

    # Full suite conditions:
    # 1. Flag disabled
    # 2. Completion attempt (story claims "done")
    # 3. Every N iterations (fullSuiteEvery)
    # 4. After healing session

    # Subsampled: deterministic 20% using story hash as seed
    # pytest --randomly-seed={hash(StoryId)} -x --limit=N
    # where N = ceil(total_test_count * ratio)
}
```

**Seed determinism:** Hash the story ID to get a stable seed. Same story always exercises the same 20% of tests, ensuring reproducible failures:

```powershell
$seed = [Math]::Abs($StoryId.GetHashCode())
# pytest tests/ --randomly-seed=$seed -x --limit=$subsetCount
```

**Integration:** Wire into `Invoke-ClaudeForStory` (`ralph.ps1:498`). The subsample is for Ralph's own validation checks, not for what Claude runs internally. Specifically:
- Post-iteration health checks (`Invoke-PostIterationHealing`) use subsampled command
- Completion validation uses full suite

**New config:**

```json
"selfHealing": {
  "subsampling": {
    "enabled": true,
    "ratio": 0.2,
    "fullSuiteEvery": 3,
    "fullSuiteOnCompletion": true,
    "fullSuiteAfterHealing": true
  }
}
```

**Validation:** If subsampled tests pass but full suite fails, mark story as "needs full validation" (not failed). This prevents false confidence from a lucky subset while avoiding penalty for subset-only validation.

**Files touched:** `lib/healing.ps1` (add `Get-SubsampledTestCommand`, modify `Invoke-TieredHealthCheck` to accept subsampling), `config/ralph-config.json`

**Effort:** S — Mostly pytest flag orchestration. Leverages existing `--randomly-seed` and `-x` patterns.

---

### 3. Feedback-Driven Learning Loop (M)

**Carlini Principle:** Watch for failure modes, then design tests/environment to prevent recurrence.

**Current Gap:** `learning.ps1` captures error patterns (`Get-RecurringErrors`, `Get-PromptMetrics`) and `Save-SprintLearning` records them in `state/learning_db.json`. `Get-LearningContext` (sprint.ps1:1007) queries relevant entries. But **none of this feeds back into story prompts**. Sprint N learnings don't influence sprint N+1 behavior.

The learning DB has data. The prompt builder has sections. They're not connected.

**What Exists Today:**
- `Save-SprintLearning` (learning.ps1:368): Captures 4 types of learning per sprint (story analysis, error patterns, prompt effectiveness, co-changed files), calls `Update-LearningDb`
- `Get-LearningInsights` (learning.ps1:454): Aggregates common error types, success rates, top adjustments from last N entries
- `Get-LearningContext` (sprint.ps1:1007): Returns last 10 relevant entries by focus area or error category
- `Get-LearningAdjustments` (learning.ps1:303): Determines config adjustments (but these are logged, not applied to prompts)
- `Build-StoryPrompt` (prompts.ps1:198): Has 7+ sections but no learning injection section

**Design:**

Add `Get-LearningInjection` function in `lib/learning.ps1`:

```powershell
function Get-LearningInjection {
    <#
    .SYNOPSIS
        Generate warning text from learning DB for injection into story prompts.
    .DESCRIPTION
        Reads learning_db.json, extracts top-3 recurring error patterns for the
        current focus area, formats as actionable warnings.
        Only returns warnings that appeared 3+ times in last 20 entries.
    .PARAMETER FocusArea
        Current focus area for filtering
    .PARAMETER MaxWarnings
        Maximum number of warnings to return (default: 3)
    .RETURNS
        String: formatted warning section, or empty string if no relevant warnings
    #>
    param(
        [string]$FocusArea = "",
        [int]$MaxWarnings = 3
    )

    $insights = Get-LearningInsights -FocusArea $FocusArea -LastN 20
    if (-not $insights.hasData) { return "" }

    # Filter to errors with 3+ occurrences
    $significantErrors = $insights.commonErrorTypes.GetEnumerator() |
        Where-Object { $_.Value -ge 3 } |
        Sort-Object Value -Descending |
        Select-Object -First $MaxWarnings

    if ($significantErrors.Count -eq 0) { return "" }

    $lines = @()
    $lines += ""
    $lines += "## Known Issues (from learning database)"
    foreach ($error in $significantErrors) {
        $errorType = $error.Key
        $count = $error.Value
        # Map error types to actionable warnings
        $warning = switch -Wildcard ($errorType) {
            "ImportError"    { "This codebase has frequent ImportError from circular imports. Always verify imports with ``python -m py_compile`` before committing." }
            "TestFailure"    { "Tests in this area are fragile. Run the full test suite, not just your new tests." }
            "Timeout"        { "Previous stories in this area timed out. Keep changes small and focused." }
            "SyntaxError"    { "Syntax errors are common. Run ``python -m py_compile`` on every file you modify." }
            "AttributeError" { "Dict-vs-object bugs are common. Check isinstance() when accessing attributes from cache/JSON." }
            default          { "Pattern '$errorType' appeared $count times recently. Be extra careful." }
        }
        $lines += "- WARNING ($count recent occurrences): $warning"
    }

    return ($lines -join "`n")
}
```

**Integration into `Build-StoryPrompt`** (prompts.ps1:198):

Insert after Section 5.5 (sprint progress context, line 287) and before Section 6 (resume context, line 289):

```powershell
# Section 5.7: Learning injection (known issues from past sprints)
if ($config.flags.learningInjection) {
    $learningWarnings = Get-LearningInjection -FocusArea $FocusArea
    if ($learningWarnings) {
        $promptParts += $learningWarnings
    }
}
```

**Effectiveness tracking:** After each story, record whether the injected warnings' error types recurred:

```powershell
# In Resolve-ClaudeResult, after recording metric
if ($config.flags.learningInjection) {
    $errorCategory = Get-ErrorCategory -Output $Ctx.ClaudeOutput
    Update-LearningDb -Entry @{
        type = "injection_result"
        storyId = $Ctx.StoryId
        focusArea = $Ctx.FocusAreaId
        injectedWarnings = @($activeWarningTypes)  # What we warned about
        actualError = $errorCategory               # What actually happened
        prevented = $errorCategory -notin $activeWarningTypes  # Did warnings help?
    }
}
```

**New config:**

```json
"flags": {
  "learningInjection": true
}
```

**Files touched:** `lib/learning.ps1` (add `Get-LearningInjection`), `lib/prompts.ps1` (inject into `Build-StoryPrompt`), `lib/claude.ps1` (effectiveness tracking in `Resolve-ClaudeResult`), `config/ralph-config.json`

**Effort:** M — Requires learning DB query logic + prompt integration + effectiveness tracking.

---

### 4. Role-Based Story Specialization (M)

**Carlini Principle:** Different agents assigned specific roles — duplicate elimination, performance optimization, design critique, documentation.

**Current Gap:** Every story gets the same prompt structure in `Build-StoryPrompt` regardless of type. A bug fix story gets the same framing as a new feature or a refactoring task. The only differentiation is seed stories vs regular stories (prompts.ps1:304-361).

**What Exists Today:**
- `Build-StoryPrompt` has seed story detection (line 304) but no other role awareness
- Story objects have `title`, `acceptanceCriteria`, `notes`, `priority`, `passes` — no `role` field
- `stallDetection` config has per-focus-area timeout overrides but not per-role overrides
- `Invoke-StoryDecomposition` generates placeholder sub-stories with no role awareness

**Design:**

**Role classification** — Add `Get-StoryRole` function in `lib/sprint.ps1`:

```powershell
function Get-StoryRole {
    <#
    .SYNOPSIS
        Classify story into role based on title and criteria keywords.
    .RETURNS
        String: 'bugfix', 'feature', 'refactor', 'test', 'docs', 'performance'
    #>
    param([object]$Story)

    $title = ($Story.title ?? "").ToLower()
    $criteria = ($Story.acceptanceCriteria -join " ").ToLower()
    $combined = "$title $criteria"

    # Priority-ordered keyword matching
    if ($combined -match 'fix|bug|error|broken|crash|regression') { return 'bugfix' }
    if ($combined -match 'refactor|reorganize|restructure|clean.?up|simplify|extract') { return 'refactor' }
    if ($combined -match 'test|coverage|assertion|spec|pester') { return 'test' }
    if ($combined -match 'doc|readme|comment|guide|tutorial') { return 'docs' }
    if ($combined -match 'perf|optimi|speed|latency|memory|cache|benchmark') { return 'performance' }
    return 'feature'  # Default
}
```

**Role-specific prompt prefixes** — Add to `Build-StoryPrompt`:

```powershell
# After story header (line 311), before notes/criteria
if ($config.flags.roleSpecialization) {
    $role = Get-StoryRole -Story $Story
    $rolePrefix = switch ($role) {
        'bugfix'      { "You are a debugging specialist. Reproduce the bug first with a failing test, then fix. Use the /bugfix workflow: reproduce -> diagnose -> fix -> verify -> scan for similar." }
        'refactor'    { "You are a refactoring specialist. Preserve ALL existing behavior. Run the full test suite BEFORE and AFTER changes. If any test fails after your change, revert immediately." }
        'test'        { "You are a test engineer. Focus on edge cases, error paths, and mutation-resistant assertions. Tests should fail for the right reasons when code is broken." }
        'docs'        { "You are a documentation specialist. Be concise. Update only what's changed. Don't add boilerplate." }
        'performance' { "You are a performance specialist. Measure before and after. Include benchmark numbers in your commit message. No premature optimization." }
        'feature'     { "" }  # Default: no special prefix
    }
    if ($rolePrefix) {
        $promptParts += ""
        $promptParts += "## Role: $($role.ToUpper())"
        $promptParts += $rolePrefix
    }
}
```

**Role-based timeout overrides** — In `config/ralph-config.json`:

```json
"roles": {
  "bugfix":      { "timeout": 420,  "maxIterations": 3 },
  "refactor":    { "timeout": 720,  "maxIterations": 4 },
  "test":        { "timeout": 360,  "maxIterations": 2 },
  "docs":        { "timeout": 240,  "maxIterations": 2 },
  "performance": { "timeout": 600,  "maxIterations": 3 },
  "feature":     { "timeout": 600,  "maxIterations": 5 }
}
```

**Metrics per role:** Add `role` column to `Record-Metric` CSV. Track success rate per role in diagnostics:

```json
"metrics_by_role": {
  "bugfix":  { "attempted": 4, "passed": 3, "rate": 0.75 },
  "feature": { "attempted": 6, "passed": 5, "rate": 0.83 }
}
```

**Files touched:** `lib/sprint.ps1` (add `Get-StoryRole`), `lib/prompts.ps1` (role prefix in `Build-StoryPrompt`), `lib/metrics.ps1` (add `role` column to `Record-Metric`), `ralph.ps1` (role-based timeout in `Invoke-ClaudeForStory`), `config/ralph-config.json`

**Effort:** M — Prompt templates + classification logic + metrics integration.

---

### 5. Adaptive Context Pruning (M)

**Carlini Principle:** Avoid printing "thousands of useless bytes." Pre-compute summary statistics. Log critical info to files with grep-friendly format.

**Current Gap:** `Build-StoryPrompt` has fixed structure with `maxLength: 10000` (from `Get-PromptRecommendation`). When the prompt exceeds this, it truncates from the END (line 366-368), which means story instructions (at the end for regular stories) get cut. There's no intelligence about WHAT to trim or priority-based allocation.

**What Exists Today:**
- `Build-StoryPrompt` appends sections in order: failure context → feedback → retrospective → conflict warnings → file hints → sprint progress → resume context → story details
- Truncation is dumb: `$prompt.Substring(0, $maxLength - 50)` — cuts at byte boundary
- `Get-PromptRecommendation` reads `prompt_effectiveness.jsonl` to toggle sections on/off, but doesn't do budget allocation
- Seed stories already prioritize: instructions+criteria FIRST, notes LAST (safe to truncate)
- Regular stories: notes → criteria → instructions (instructions at end = cut first on overflow)

**Design:**

**Token budget allocation** — Replace dumb truncation with priority-based budget:

```powershell
function Build-BudgetedPrompt {
    <#
    .SYNOPSIS
        Assemble prompt sections with priority-based budget allocation.
    .DESCRIPTION
        Each section gets a priority (1=critical, 4=optional) and budget share.
        When total exceeds maxLength, trim lowest-priority sections first.
        Within a section, compress rather than cut (e.g., summarize 5 errors as 1 line).
    #>
    param(
        [array]$Sections,   # @{ name; content; priority; compressor }
        [int]$MaxLength = 10000
    )
}
```

**Priority assignments:**

| Priority | Section | Budget % | Compression Strategy |
|----------|---------|----------|---------------------|
| 1 (critical) | Story details + criteria + instructions | 40% | Never compress |
| 2 (important) | Failure context (retries) | 20% | Group by error category, 1 line each |
| 3 (useful) | File hints, sprint progress, learning warnings | 20% | Trim to top 5 files, 3 warnings |
| 4 (optional) | Retrospective, conflict warnings, resume context | 20% | Drop entirely if over budget |

**Failure context compression** — Add `Compress-FailureContext` function:

```powershell
function Compress-FailureContext {
    <#
    .SYNOPSIS
        Compress multiple failure traces into categorized one-liners.
    .DESCRIPTION
        Instead of 5 full error traces (500+ chars each), emit:
        "Failed 5x: 3 TestFailure (test_cache.py), 2 ImportError (src/config/)"
    #>
    param([array]$Failures)

    # Group by error category, format as single line
    # Include file name for each category, not full trace
}
```

**Integration:** Replace the current truncation block (prompts.ps1:365-368) with `Build-BudgetedPrompt`. The section-building code (lines 230-361) populates section objects instead of appending to `$promptParts` directly.

**Metrics:** Track prompt size vs success rate in diagnostics. Over time, adjust budget ratios based on correlation data.

**New config:**

```json
"prompts": {
  "maxLength": 10000,
  "budgetAllocation": {
    "storyDetails": 0.40,
    "failureContext": 0.20,
    "fileHintsAndLearning": 0.20,
    "supplementary": 0.20
  }
}
```

**Files touched:** `lib/prompts.ps1` (refactor `Build-StoryPrompt` to use budget allocation, add `Build-BudgetedPrompt`, `Compress-FailureContext`), `config/ralph-config.json`

**Effort:** M — Prompt restructuring with budget math. Must not break existing prompt flow.

---

### 6. Delta Debugging for Stuck Stories (L)

**Carlini Principle:** When Linux kernel became one monolithic task, all 16 agents hit identical bugs. Solution: decompose into independent sub-problems.

**Current Gap:** `Invoke-StoryDecomposition` (sprint.ps1:1363) already exists and decomposes hard stories into 2-4 sub-stories. However:
1. It's only triggered **between sprints** (in `Start-TrueAutoLoop`, `Start-RalphsChoiceAutoLoop`) via `Get-HardStoriesArchive`
2. It requires a story to be marked as "hard" first (3 failures or 30-min timeout)
3. Decomposed stories are placeholder titles ("Sub-task 1 from US-005") — no actual criterion splitting
4. Stories are decomposed as a monolithic block, not per-criterion

**What Exists Today:**
- `Mark-AsHardStory` (sprint.ps1:1207): Marks story after 3 failures or 30-min timeout
- `Invoke-StoryDecomposition` (sprint.ps1:1363): LLM-based decomposition into 2-4 sub-stories
- `Add-DecomposedStoriesToPRD` (sprint.ps1:1499): Inserts sub-stories into PRD
- `Get-DecomposedStoryId` (sprint.ps1:1316): Generates `US-D` / `US-DD` IDs with nesting levels
- Nesting limit: 3 levels (US → US-D → US-DD → abandoned)

**Design:**

**In-sprint criterion-level decomposition** — Add `Split-StuckStory` in `lib/sprint.ps1`:

```powershell
function Split-StuckStory {
    <#
    .SYNOPSIS
        Decompose a stuck story into per-criterion sub-stories during the sprint.
    .DESCRIPTION
        After 2 consecutive failures, parse the acceptance criteria list and
        create sub-stories, one per criterion. Each sub-story is independently
        testable and committable.

        Unlike Invoke-StoryDecomposition (between-sprint, LLM-based, 2-4 generic),
        this is in-sprint, deterministic, and 1:1 criterion-to-story.
    .PARAMETER StoryId
        Story to split
    .PARAMETER Story
        Story object from PRD
    .RETURNS
        Array of sub-story objects, or empty if story has <2 criteria
    #>
    param(
        [string]$StoryId,
        [object]$Story
    )

    $criteria = @($Story.acceptanceCriteria)
    if ($criteria.Count -lt 2) { return @() }  # Can't split single-criterion stories

    $subStories = @()
    for ($i = 0; $i -lt $criteria.Count; $i++) {
        $subId = Get-DecomposedStoryId -ParentId $StoryId -NestingLevel 1 -Sequence ($i + 1)
        $subStories += @{
            id = $subId
            title = "[DELTA] Criterion $($i+1) of $StoryId`: $($criteria[$i].Substring(0, [Math]::Min(60, $criteria[$i].Length)))"
            acceptanceCriteria = @($criteria[$i])
            priority = $Story.priority
            passes = $false
            notes = "Delta-split from $StoryId. Original failure context preserved. Focus ONLY on this single criterion."
            decomposedFrom = $StoryId
            nestingLevel = 1
        }
    }

    return $subStories
}
```

**Trigger logic** — Wire into `Invoke-ClaudeForStory` (`ralph.ps1:498`):

```powershell
# After 2 consecutive failures on same story
if ($config.flags.deltaDebugging -and $storyFailureCount -ge $triggerAfterFailures) {
    $subStories = Split-StuckStory -StoryId $StoryId -Story $Story
    if ($subStories.Count -gt 0) {
        Add-DecomposedStoriesToPRD -Stories $subStories
        # Mark parent as "delta_split" (not failed, not passed)
        Update-StoryStatus -StoryId $StoryId -Passes $false -Notes "Delta-split into $($subStories.Count) sub-stories"
        # Continue to next story — sub-stories will be picked up by Get-SprintStatus
    }
}
```

**Parent completion tracking:** Parent story is marked complete when ALL sub-stories pass. Add check in `Get-SprintStatus` (`ralph.ps1:789`):

```powershell
# Check if delta-split parent's children are all complete
foreach ($story in $stories | Where-Object { $_.notes -match 'Delta-split' }) {
    $childStories = $stories | Where-Object { $_.decomposedFrom -eq $story.id }
    if ($childStories.Count -gt 0 -and ($childStories | Where-Object { -not $_.passes }).Count -eq 0) {
        $story.passes = $true  # All children passed
    }
}
```

**New config:**

```json
"flags": {
  "deltaDebugging": true
},
"healing": {
  "deltaDebugging": {
    "enabled": true,
    "triggerAfterFailures": 2,
    "minCriteriaToSplit": 2
  }
}
```

**Key difference from existing decomposition:** `Invoke-StoryDecomposition` creates generic placeholder sub-stories between sprints using LLM. `Split-StuckStory` creates precise, criterion-level sub-stories **during** the sprint, deterministically, without LLM calls. They complement each other:

| | `Split-StuckStory` (new) | `Invoke-StoryDecomposition` (existing) |
|---|---|---|
| When | During sprint, after 2 failures | Between sprints, from hard stories archive |
| How | Deterministic criterion splitting | LLM-generated decomposition |
| Granularity | 1 criterion = 1 sub-story | 2-4 thematic sub-stories |
| LLM cost | None | 1 Claude call |

**Files touched:** `lib/sprint.ps1` (add `Split-StuckStory`), `ralph.ps1` (trigger in `Invoke-ClaudeForStory`, parent tracking in `Get-SprintStatus`), `config/ralph-config.json`

**Effort:** L — Story lifecycle changes, sub-story tracking, parent completion logic. Must coexist with existing decomposition system.

---

### 7. Parallel Story Execution (XL)

**Carlini Principle:** 16 agents working simultaneously. Lock-file task claiming. Git synchronization.

**Current Gap:** `ralph-config.json` has `parallel.enabled: false` and `parallel.maxConcurrent: 2`. Config exists but implementation is entirely absent. All loops in `lib/loops.ps1` are sequential.

**What Exists Today:**
- Config: `parallel: { enabled: false, maxConcurrent: 2 }` (placeholders only)
- `Get-IndependentStories` (sprint.ps1:915): Finds stories with no file overlap — already designed for parallel use
- `Build-DependencyGraph` (sprint.ps1:1059): Explicit + implicit dependency tracking
- `Get-ExecutableStories` (sprint.ps1:1115): Returns stories whose dependencies are met
- CLAUDE.md Rule 35: Never use `[System.Threading.Thread]` — use `[powershell]::Create()` with `.AddArgument()`
- CLAUDE.md Rule 36: Quick Edit Mode disabled to prevent console freezes

**Design:**

**Phase 1: Git worktree isolation** — Each agent gets its own working directory:

```powershell
# Create worktrees for parallel agents
git worktree add .worktrees/agent-1 ralph/sprint-70 --no-checkout
git worktree add .worktrees/agent-2 ralph/sprint-70 --no-checkout
```

**Phase 2: Lock-file claiming** — Before starting a story, agent creates a lock:

```powershell
function Claim-Story {
    param([string]$StoryId, [int]$AgentId)

    $lockDir = Join-Path $script:RalphDir "session\locks"
    $lockFile = Join-Path $lockDir "$StoryId.lock"

    # Atomic create — fails if another agent already claimed
    try {
        [System.IO.File]::Open($lockFile, [System.IO.FileMode]::CreateNew).Close()
        Set-Content $lockFile "$AgentId|$(Get-Date -Format 'o')"
        return $true
    } catch {
        return $false  # Already claimed
    }
}
```

**Phase 3: Parallel execution loop** — New `Start-ParallelLoop` in `lib/parallel.ps1`:

```powershell
function Start-ParallelLoop {
    param([int]$MaxConcurrent = 2)

    # 1. Get executable stories (dependencies met)
    # 2. Filter to independent stories (no file overlap)
    # 3. Claim stories via lock files
    # 4. Launch [powershell]::Create() runspaces (NOT threads — Rule 35)
    # 5. Each runspace: cd to worktree → run Claude → commit
    # 6. On completion: merge worktree → run regression baseline → accept/reject
    # 7. On conflict: reject, requeue story
}
```

**Merge strategy:**

```
Agent completes story → git add + commit in worktree
                      → git checkout main-branch
                      → git merge --no-ff worktree-branch
                      → Test-RegressionBaseline (from #1)
                      → If regression: git merge --abort, requeue
                      → If clean: accept, update PRD
```

**Story independence detection:** Use `Get-IndependentStories` (already exists) + `Get-RelevantFilesForStory` to compute file relevance sets. Stories with disjoint sets can run in parallel.

**Dependencies:** Requires #1 (regression guard) for merge validation and #2 (subsampling) for fast post-merge checks. Without regression detection, parallel merges can silently corrupt the codebase.

**New config:**

```json
"parallel": {
  "enabled": true,
  "maxConcurrent": 2,
  "worktreeDir": ".worktrees",
  "mergeStrategy": "no-ff",
  "lockTimeout": 1800,
  "requireIndependentFiles": true
}
```

**Key risks:**
1. PowerShell runspace management is fragile — `[powershell]::Create()` with `RunspacePool` needed
2. Git worktree on Windows has known issues with path lengths
3. Console output from parallel agents will interleave — need per-agent log files
4. `session/metrics.csv` concurrent writes need the existing non-locking pattern from `Write-MetricsRow`

**Files touched:** New `lib/parallel.ps1` (parallel loop, worktree management, lock claiming), `lib/claude.ps1` (multi-process Claude invocation), `ralph.ps1` (worktree setup/teardown), `config/ralph-config.json`

**Effort:** XL — Fundamental architecture change. Git worktree management, concurrent state, merge conflicts. Start with 2 agents and validate before scaling.

---

## Implementation Roadmap

```
Phase 0 (Sprint 70): Observability
  [S] #0 Sprint Diagnostics Output  ← measure BEFORE changing

Phase 1 (Sprint 71-72): Foundation
  [S] #1 Oracle-Based Regression Guard
  [S] #2 Deterministic Test Subsampling

Phase 2 (Sprint 73-74): Intelligence
  [M] #3 Feedback-Driven Learning Loop
  [M] #4 Role-Based Story Specialization

Phase 3 (Sprint 75-76): Optimization
  [M] #5 Adaptive Context Pruning
  [L] #6 Delta Debugging for Stuck Stories

Phase 4 (Sprint 77+): Scale
  [XL] #7 Parallel Story Execution
```

**Rationale for ordering:**
- #0 first because you can't measure improvement without a baseline dashboard
- #1 and #2 are pure infrastructure with no prompt changes — lowest risk, highest payoff
- #3 and #4 modify prompt content — need regression guard (#1) in place first
- #5 restructures prompt assembly — needs stable prompt sections from #3/#4
- #6 changes story lifecycle — needs all prior systems stable
- #7 is a fundamental architecture change — needs everything else working

---

## Verification Strategy

**Diagnostics-first approach:**
1. Ship #0 Sprint Diagnostics first → establish 3-sprint measurement baseline
2. Each subsequent improvement gets its own feature flag in `ralph-config.json`
3. Enable ONE flag at a time, measure for 3 sprints, then enable next
4. `session/diagnostics.json` tracks which flags are active + their measured impact

**Key metrics to track:**
| Metric | Source | Target |
|--------|--------|--------|
| First-attempt success rate | metrics.csv (retry_count=0 AND success=true) | >70% |
| Stories per hour | metrics.csv (stories / session duration) | >3 |
| Healing sessions per sprint | healing_log.jsonl count | <2 |
| Regression count | diagnostics.json flag_impact | 0 |
| Prompt size vs success correlation | prompt length + success | Optimize ratio |

**Rollback plan:** Each flag can be independently disabled in `ralph-config.json` without code changes. If a flag degrades metrics over 3 sprints, disable it and investigate.

---

## Carlini Principles Mapping

| Carlini Principle | Ralph Improvement | Existing Foundation | New Work |
|---|---|---|---|
| Pre-compute summary statistics | #0 Sprint diagnostics | `Record-Metric`, `Measure-CodebaseHealth` | Unified diagnostics view |
| Environmental design > prompts | #1 Regression guard, #2 Subsampling | `Compare-TestBaseline`, tiered health checks | Pre-story baseline, subsampled validation |
| High-quality test suites | #1 Oracle baseline, #2 Fast feedback | `Invoke-TieredHealthCheck` (3 tiers) | Fresh baselines, deterministic subsets |
| Deterministic subsampling | #2 Test subsampling | `--randomly-seed` in fastPytestArgs | Per-story deterministic seed |
| Self-documentation of failures | #3 Learning loop closure | `Save-SprintLearning`, `Get-LearningInsights` | Close loop: learning → prompt injection |
| Role-based specialization | #4 Story roles | Seed story detection | Full role classification + prompt prefixes |
| Context window management | #5 Adaptive pruning | `maxLength` truncation | Priority-based budget allocation |
| Single-giant-task anti-pattern | #6 Delta debugging | `Invoke-StoryDecomposition` (between sprints) | In-sprint criterion-level splitting |
| Parallel agents | #7 Parallel execution | Config placeholders, `Get-IndependentStories` | Full worktree-based parallel execution |
