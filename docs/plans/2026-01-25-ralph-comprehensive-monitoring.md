# Ralph Loop Comprehensive Monitoring Plan

> **Goal:** "Everything that can be documented will be documented"

## Executive Summary

The Ralph Loop has functional execution but significant monitoring gaps. This plan addresses 50+ identified gaps across logging, metrics, state tracking, and documentation.

---

## Current State Analysis

### What IS Being Monitored

| Component | Data Captured |
|-----------|---------------|
| **Session** | SessionId, IterationCount, ConsecutiveFailures, StartTime |
| **Execution** | Duration (min), Success/Fail, Timeout, ExitCode |
| **Git** | Lines added/deleted (stats only) |
| **Tests** | Pass/fail counts via pattern matching |
| **Errors** | Category (SyntaxError, TestFailure, etc.) |
| **Tokens** | Estimated from output length |
| **Retries** | Count per focus area/story |
| **Mode** | Standard, Interview, TrueAuto |
| **Prompts** | Saved to prompt_N.txt files |
| **Output** | claude_out_N.log, claude_err_N.log |

### What IS NOT Being Monitored (Gaps)

| Category | Gap Count | Impact |
|----------|-----------|--------|
| Claude CLI Invocation | 8 gaps | Cannot reproduce/debug executions |
| Timing Breakdown | 5 gaps | Cannot optimize slow phases |
| File Operations | 4 gaps | Cannot audit what Claude changed |
| Git Operations | 5 gaps | Cannot correlate commits to iterations |
| State Transitions | 6 gaps | Cannot debug state machine issues |
| Error Details | 4 gaps | Cannot root-cause failures |
| Test Details | 4 gaps | Only counts, not which tests |
| Resource Usage | 4 gaps | Cannot detect leaks/bottlenecks |
| Story Verification | 3 gaps | Cannot prove acceptance criteria met |
| Cost Attribution | 4 gaps | Cannot optimize spending |

---

## Implementation Plan

### Phase 1: Core Instrumentation (Priority 1) ✅ COMPLETE

#### Task 1.1: Claude CLI Invocation Logging
**File:** `ralph.ps1` - Both `Invoke-ClaudeForFocusArea` and `Invoke-ClaudeForStory`

Add to logs:
```
claude_invocation_N.json:
{
  "iteration": 1,
  "timestamp": "2026-01-25T11:03:57Z",
  "command": {
    "executable": "C:\\Users\\daves\\AppData\\Roaming\\npm\\claude.cmd",
    "resolvedPath": "...",
    "arguments": ["--print", "--dangerously-skip-permissions", "--allowedTools=..."],
    "workingDirectory": "D:\\_Projects\\voiceover-matcher-subtitle",
    "environmentOverrides": {}
  },
  "prompt": {
    "file": "prompt_1.txt",
    "lineCount": 15,
    "charCount": 892,
    "type": "prd_generation"
  },
  "execution": {
    "processId": 12345,
    "startedAt": "2026-01-25T11:03:57Z",
    "endedAt": "2026-01-25T11:05:23Z",
    "durationMs": 86000,
    "exitCode": 0,
    "exitSignal": null,
    "timedOut": false
  }
}
```

#### Task 1.2: Structured Iteration Manifest
**File:** `ralph.ps1` - Create after each iteration

Add to logs:
```
iteration_N_manifest.json:
{
  "iteration": 1,
  "storyId": "US-001",
  "focusArea": "quality",
  "sprint": 4,
  "branch": "ralph/sprint-4",
  "status": "completed",
  "timestamps": {
    "queued": "2026-01-25T11:00:00Z",
    "started": "2026-01-25T11:03:57Z",
    "completed": "2026-01-25T11:05:23Z"
  },
  "prompt": {
    "file": "prompt_1.txt",
    "hash": "abc123..."
  },
  "output": {
    "stdout": "claude_out_1.log",
    "stderr": "claude_err_1.log",
    "exitCode": 0
  },
  "git": {
    "beforeCommit": "abc123",
    "afterCommit": "def456",
    "filesModified": ["src/foo.py", "tests/test_foo.py"],
    "linesAdded": 45,
    "linesDeleted": 12
  },
  "tests": {
    "ran": true,
    "passed": 15,
    "failed": 0,
    "skipped": 2,
    "duration": 12.5
  },
  "metrics": {
    "tokensEstimated": 8500,
    "retryCount": 1
  }
}
```

#### Task 1.3: Acceptance Criteria Verification Logging
**File:** `ralph.ps1` - When marking story as passed

Add verification checklist:
```
story_US-001_verification.json:
{
  "storyId": "US-001",
  "title": "Add cache warm-up utility function",
  "verifiedAt": "2026-01-25T11:05:23Z",
  "iteration": 1,
  "acceptanceCriteria": [
    {
      "criterion": "Function exists in src/cache.py",
      "verified": true,
      "evidence": "grep found 'def warm_up_cache' at line 234"
    },
    {
      "criterion": "Unit tests pass",
      "verified": true,
      "evidence": "pytest tests/test_cache.py::test_warm_up - PASSED"
    },
    {
      "criterion": "No regressions",
      "verified": true,
      "evidence": "Full test suite: 156 passed, 0 failed"
    }
  ],
  "overallVerified": true
}
```

#### Task 1.4: File Operation Tracking
**File:** `ralph.ps1` - Capture before/after git status

```
file_operations_N.json:
{
  "iteration": 1,
  "filesCreated": [
    {"path": "src/new_module.py", "size": 1234}
  ],
  "filesModified": [
    {"path": "src/existing.py", "linesAdded": 15, "linesDeleted": 3}
  ],
  "filesDeleted": [],
  "totalFilesChanged": 2
}
```

#### Task 1.5: Git Operation Logging
**File:** `ralph.ps1` - After each iteration with commits

```
git_operations_N.json:
{
  "iteration": 1,
  "branch": "ralph/sprint-4",
  "commits": [
    {
      "hash": "abc123def456",
      "message": "feat: add cache warm-up utility",
      "timestamp": "2026-01-25T11:05:20Z",
      "filesChanged": 2,
      "insertions": 45,
      "deletions": 12
    }
  ],
  "beforeState": {
    "hash": "previous123",
    "clean": true
  },
  "afterState": {
    "hash": "abc123def456",
    "clean": true
  }
}
```

---

### Phase 2: Enhanced Metrics (Priority 2) ✅ COMPLETE

#### Task 2.1: Timing Breakdown
**File:** `ralph.ps1` - Add phase timing

New metrics columns:
```csv
...,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms
```

Capture in code:
```powershell
$phaseTimings = @{
    read = 0
    analyze = 0
    implement = 0
    test = 0
    commit = 0
}
# Parse Claude output for phase markers or estimate from log timestamps
```

#### Task 2.2: Per-Focus-Area Metrics Aggregation
**File:** `watch.ps1` - Add focus area dashboard section

```
FOCUS AREA HEALTH:
  pipeline:  85% success | 4.2 min avg | $0.03 avg cost | 12 stories
  quality:   72% success | 6.1 min avg | $0.05 avg cost | 8 stories
  testing:   91% success | 3.8 min avg | $0.02 avg cost | 15 stories
```

#### Task 2.3: Session Timeline Log
**File:** `ralph.ps1` - Append to timeline file

```
session_timeline.jsonl (append-only):
{"ts":"2026-01-25T11:00:00Z","event":"session_start","session":"2026-01-25_110000"}
{"ts":"2026-01-25T11:00:01Z","event":"config_loaded","focusArea":"quality"}
{"ts":"2026-01-25T11:00:02Z","event":"prd_loaded","sprint":4,"stories":12}
{"ts":"2026-01-25T11:03:57Z","event":"iteration_start","iteration":1,"story":"US-001"}
{"ts":"2026-01-25T11:05:23Z","event":"iteration_complete","iteration":1,"success":true}
{"ts":"2026-01-25T11:05:25Z","event":"story_verified","story":"US-001","passed":true}
```

#### Task 2.4: Error Evolution Tracking
**File:** `watch.ps1` - Add error history section

```
ERROR PATTERNS:
  SyntaxError: 0 (was 3 yesterday, trending down)
  TestFailure: 2 (stable)
  APIError: 1 (new, appeared 10 min ago)
```

#### Task 2.5: State Machine Logging
**File:** All scripts - Add state transition logging

```powershell
function Log-StateTransition {
    param([string]$From, [string]$To, [string]$Reason)
    $entry = @{
        timestamp = Get-Date -Format "o"
        from = $From
        to = $To
        reason = $Reason
    }
    $entry | ConvertTo-Json -Compress | Add-Content "state_transitions.jsonl"
}
```

---

### Phase 3: Detailed Tracking (Priority 3) ✅ COMPLETE

#### Task 3.1: Configuration Change Audit Trail
**File:** `ralph.ps1` - Track config modifications

```
config_audit.jsonl:
{"ts":"...","field":"autonomy.maxIterations","old":50,"new":100,"reason":"user request"}
```

#### Task 3.2: Test Failure Detail Logging
**File:** `ralph.ps1` - Parse pytest output

```
test_details_N.json:
{
  "iteration": 1,
  "testRun": {
    "command": "pytest tests/ -v",
    "duration": 12.5,
    "exitCode": 0
  },
  "results": [
    {"name": "test_cache.py::test_warm_up", "status": "passed", "duration": 0.5},
    {"name": "test_cache.py::test_eviction", "status": "passed", "duration": 0.3}
  ],
  "summary": {
    "passed": 15,
    "failed": 0,
    "skipped": 2,
    "errors": 0
  }
}
```

#### Task 3.3: Resource Usage Monitoring
**File:** `ralph.ps1` - Sample during execution

```powershell
function Get-ProcessMetrics {
    param([int]$ProcessId)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($proc) {
        @{
            cpu = $proc.CPU
            memoryMB = [math]::Round($proc.WorkingSet64 / 1MB, 2)
            handles = $proc.HandleCount
            threads = $proc.Threads.Count
        }
    }
}
```

#### Task 3.4: Prompt Effectiveness Scoring
**File:** `ralph.ps1` - Correlate prompts with outcomes

New metric: `prompt_effectiveness_score`
- Success on first try = 1.0
- Success after retry = 0.5
- Failure = 0.0

Aggregate by prompt type (prd_generation, story_work, etc.)

#### Task 3.5: Skip/Blocker Tracking
**File:** `ralph.ps1`, `metrics.csv`

New columns:
```csv
...,was_skipped,skip_reason,blocker_type,blocker_resolved
```

---

### Phase 4: Documentation & Dashboards (Priority 4)

#### Task 4.1: Cost Attribution Dashboard
**File:** `watch.ps1` - Add cost breakdown

```
COST BREAKDOWN (this sprint):
  Total: $0.45
  By Focus Area:
    quality: $0.18 (40%)
    testing: $0.12 (27%)
    pipeline: $0.15 (33%)
  By Story:
    US-001: $0.08
    US-002: $0.05
    ...
```

#### Task 4.2: Anomaly Detection Flags
**File:** `metrics.csv`, `watch.ps1`

New columns:
```csv
...,anomaly_duration,anomaly_cost,anomaly_errors
```

Dashboard alerts:
```
ANOMALIES DETECTED:
  ! Duration spike: Iteration 5 took 15 min (avg 4 min)
  ! Cost spike: Last hour $0.20 (daily avg $0.05/hr)
```

#### Task 4.3: Operational Playbook
**File:** `scripts/ralph/docs/OPERATIONS.md`

Contents:
- How to recover from partial failure
- How to resume from specific iteration
- How to skip a failing story
- How to merge interrupted sessions
- How to force-retry a focus area

#### Task 4.4: Troubleshooting Guide
**File:** `scripts/ralph/docs/TROUBLESHOOTING.md`

Contents:
- "Why did iteration time out?" decision tree
- "How to debug a failed story"
- "Why metrics show N errors but I see Y"
- Common error patterns and fixes

#### Task 4.5: Metrics Interpretation Guide
**File:** `scripts/ralph/docs/METRICS.md`

Contents:
- What is "good" vs "bad" for each metric
- How metrics correlate
- What to do if metric X spikes
- Benchmarks by focus area

---

## New Files Created

| File | Purpose |
|------|---------|
| `logs/{session}/claude_invocation_N.json` | Full CLI invocation details |
| `logs/{session}/iteration_N_manifest.json` | Structured iteration metadata |
| `logs/{session}/story_*_verification.json` | Acceptance criteria audit |
| `logs/{session}/file_operations_N.json` | File change tracking |
| `logs/{session}/git_operations_N.json` | Git commit details |
| `logs/{session}/test_details_N.json` | Individual test results |
| `logs/{session}/session_timeline.jsonl` | Minute-by-minute events |
| `state_transitions.jsonl` | State machine audit trail |
| `config_audit.jsonl` | Configuration change history |
| `docs/OPERATIONS.md` | Recovery playbook |
| `docs/TROUBLESHOOTING.md` | Debug decision trees |
| `docs/METRICS.md` | Metrics interpretation |

---

## Metrics Schema v2

```csv
# Current columns (keep)
timestamp,session,sprint,story_id,mode,duration_min,success,timeout,focus_area,tokens_used,error_category,hour_of_day,test_results,retry_count,lines_added,lines_deleted

# New columns (add)
,phase_read_ms,phase_analyze_ms,phase_implement_ms,phase_test_ms,phase_commit_ms
,files_created,files_modified,files_deleted
,git_commits,git_branch,git_before_hash,git_after_hash
,tests_passed,tests_failed,tests_skipped,tests_duration
,prompt_type,prompt_hash,prompt_effectiveness
,was_skipped,skip_reason,blocker_type,blocker_resolved
,anomaly_duration,anomaly_cost,anomaly_errors
,cost_dollars,cpu_percent_avg,memory_mb_avg
```

---

## Watch Dashboard v2 Layout

```
======================================================================
  RALPH LOOP MONITOR - 11:04:25
======================================================================

  QUEUE MODE [===========-------------------] 40% (2/5 areas)
    [DONE] pipeline
    [DONE] testing
    [>>  ] quality (current)
    [    ] rate-limiting
    [    ] otio

----------------------------------------------------------------------
  Sprint: 4 | Branch: ralph/sprint-4 | Focus: quality
----------------------------------------------------------------------
  [DONE] US-001: Add cache warm-up utility function
  [DONE] US-002: Add parallel download worker config
  [>>  ] US-003: Add embedding batch size validation
  [    ] US-004: Add checkpoint load time logging
  ...
  [================================================--] 92% (11/12)

----------------------------------------------------------------------
  CURRENT PROMPT (prompt_3.txt, 0.1 min ago)
----------------------------------------------------------------------
    You are working on story US-003 from scripts/ralph/prd.json...
    1. Read the story and acceptance criteria
    2. Implement the required changes
    ...
    Full prompt: D:\...\logs\2026-01-25_110000\prompt_3.txt

----------------------------------------------------------------------
  METRICS DASHBOARD
----------------------------------------------------------------------
  Session: 2026-01-25_110000 | Iterations: 3 | Success: 100%

  TIMING:                      COST:
    Total: 12 min               Tokens: 25,000
    Avg/story: 4 min            Est. cost: $0.08
    Phases: R:10% A:30% I:45% T:15%

  PRODUCTIVITY:                RESILIENCE:
    +120 lines / -45 lines      Avg retries: 1.2
    3 files modified            Max retries: 2
    2 commits

  TESTS:                       ERRORS:
    156 passed, 0 failed        (none this session)
    Coverage: 78%

  FOCUS AREA HEALTH:
    quality: 100% (3/3) | 4.0 min avg | $0.03 avg

----------------------------------------------------------------------
  TIMELINE (last 5 events):
    11:04:20 iteration_complete iteration=3 success=true
    11:04:18 tests_passed count=156
    11:04:15 git_commit hash=abc123
    11:00:02 iteration_start iteration=3 story=US-003
    10:56:00 story_verified story=US-002 passed=true

----------------------------------------------------------------------
  Git: 2 files modified (run.bat, src/cache.py)
======================================================================
```

---

## Implementation Order

1. **Day 1:** Task 1.1 (CLI logging), Task 1.2 (manifests)
2. **Day 2:** Task 1.3 (verification), Task 1.4 (file ops), Task 1.5 (git ops)
3. **Day 3:** Task 2.1 (timing), Task 2.3 (timeline)
4. **Day 4:** Task 2.2 (focus area metrics), Task 2.4 (error evolution)
5. **Day 5:** Task 3.1-3.5 (detailed tracking)
6. **Day 6:** Task 4.1-4.2 (dashboards), Task 4.3-4.5 (docs)

---

## Success Criteria

- [ ] Every Claude invocation has full audit trail
- [ ] Every iteration has structured manifest
- [ ] Every story completion has verification evidence
- [ ] Every file change is tracked
- [ ] Every git operation is logged
- [ ] Timing breakdown available for optimization
- [ ] Error patterns detectable over time
- [ ] Cost attributable to focus areas/stories
- [ ] Operational playbooks enable self-service recovery
- [ ] Metrics guide enables data-driven decisions
