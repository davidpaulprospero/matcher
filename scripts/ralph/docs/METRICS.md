# Ralph Loop Metrics Interpretation Guide

Understanding what Ralph Loop metrics mean and how to use them effectively.

## Key Metrics

### Success Rate

**What it measures:** Percentage of iterations that completed without errors.

| Value | Interpretation | Action |
|-------|----------------|--------|
| 90%+ | Excellent | System running smoothly |
| 70-89% | Good | Some retries needed, acceptable |
| 50-69% | Concerning | Investigate error patterns |
| <50% | Poor | Stop and diagnose root cause |

**What affects it:**
- Story complexity
- Test stability
- API reliability
- Prompt quality

### Duration (minutes per story)

**What it measures:** Time from iteration start to completion.

| Value | Interpretation | Action |
|-------|----------------|--------|
| <5 min | Very fast | Simple tasks, good prompts |
| 5-10 min | Normal | Standard development work |
| 10-15 min | Slow | Complex stories or exploration |
| >15 min | Very slow | Consider splitting story |

**Benchmarks by focus area:**

| Focus Area | Expected Duration | Notes |
|------------|-------------------|-------|
| testing | 5-8 min | Short focused changes |
| speed | 8-12 min | Profiling adds time |
| quality | 6-10 min | Refactoring |
| config | 4-6 min | Simple changes |
| otio | 10-15 min | Complex integrations |

### Token Usage

**What it measures:** Estimated Claude API token consumption.

| Per Iteration | Interpretation |
|---------------|----------------|
| <10,000 | Simple task |
| 10,000-30,000 | Normal complexity |
| 30,000-50,000 | Heavy exploration/reading |
| >50,000 | Very complex or stuck in loop |

**Cost calculation:** `$0.003 per 1,000 tokens` (approximate)

### Timeout Rate

**What it measures:** Percentage of iterations killed for exceeding time limit.

| Value | Interpretation | Action |
|-------|----------------|--------|
| 0% | Ideal | No hangs |
| <5% | Acceptable | Occasional complex tasks |
| 5-15% | High | Review timeout settings |
| >15% | Critical | Stories too complex |

### Retry Count

**What it measures:** Number of attempts before success.

| Value | Interpretation |
|-------|----------------|
| 1 | First try success (ideal) |
| 2-3 | Minor issues self-corrected |
| 4-5 | Significant challenges |
| >5 | Story may need human help |

### Prompt Effectiveness

**What it measures:** How well prompts lead to success.

| Score | Meaning |
|-------|---------|
| 1.0 | Success on first try |
| 0.5 | Success after 1-3 retries |
| 0.25 | Success after many retries |
| 0.0 | Never succeeded |

**By prompt type:**

| Type | Expected | If Lower |
|------|----------|----------|
| story_work | 0.6+ | Review story clarity |
| prd_generation | 0.8+ | Check focus area definition |

---

## Metric Correlations

### High Duration + High Tokens = Exploration
Claude is reading lots of files to understand the codebase. Normal for first tasks in a new area.

### High Duration + Low Tokens = Slow Tools
Claude is waiting for tool execution (tests, builds). Check system resources.

### Low Duration + High Retries = Simple Fix, Hard to Find
The change is small but Claude needed multiple attempts. Story may need clearer requirements.

### High Errors + Same Category = Systematic Issue
Repeated errors of the same type indicate a root cause to fix (broken test, API issue, etc.)

---

## Phase Timing Breakdown

When available, phase timing shows where time is spent:

| Phase | What It Includes | If High |
|-------|------------------|---------|
| Read | File reading, code exploration | Complex codebase or unclear requirements |
| Analyze | Planning, thinking | Complex story logic |
| Implement | Code writing, edits | Large changes needed |
| Test | Running pytest | Many tests or slow test suite |
| Commit | Git operations | Many files changed |

**Ideal distribution:**
- Read: 15-25%
- Analyze: 10-20%
- Implement: 40-50%
- Test: 15-25%
- Commit: 5-10%

---

## Anomaly Indicators

### Duration Spike
**Trigger:** Single iteration 3x+ longer than average.

**Possible causes:**
- Story unexpectedly complex
- Claude exploring unrelated code
- System resource contention

**Action:** Review the iteration manifest and output log.

### Cost Spike
**Trigger:** Recent token usage 2x+ higher than session average.

**Possible causes:**
- Large files being read
- Many tools being called
- Retry loops

**Action:** Check prompt_N.txt and claude_out_N.log.

### High Failure Rate
**Trigger:** 4+ of last 5 iterations failed.

**Possible causes:**
- Broken test suite
- API issues
- Malformed stories

**Action:** Check test_details_N.json and error_category breakdown.

### Timeout Trend
**Trigger:** 3+ timeouts in last 10 iterations.

**Possible causes:**
- Timeout too short for focus area
- System overloaded
- Hung Claude process

**Action:** Increase autonomy.iterationTimeout or check system resources.

---

## Dashboard Sections Explained

### COST ATTRIBUTION
Shows where money is going. High costs in one area may indicate:
- Complex work (expected)
- Inefficient prompts (fixable)
- Retry loops (investigate)

### FOCUS AREA HEALTH
Per-area success rate and timing. Use to:
- Identify problem areas
- Set expectations for estimates
- Prioritize prompt improvements

### ANOMALY CHECK
Real-time issue detection. Green = good, Red = investigate.

### PHASE BREAKDOWN
Where Claude spends time. Helps optimize:
- Prompt clarity (reduces Analyze)
- Story scope (reduces Implement)
- Test efficiency (reduces Test)

### RESOURCE USAGE
Memory and CPU trends. High values indicate:
- System strain
- Potential instability
- Need for breaks between sprints

### PROMPT EFFECTIVENESS
How well your prompts work. Low scores mean:
- Stories need clearer requirements
- Focus areas need better definitions
- Consider different prompt templates

---

## Optimization Targets

| Metric | Target | Priority |
|--------|--------|----------|
| Success Rate | >80% | High |
| Avg Duration | <10 min | Medium |
| Timeout Rate | <5% | High |
| Prompt Effectiveness | >0.6 | Medium |
| Retry Count | <3 avg | Low |

---

## Exporting Metrics

### To CSV
```powershell
# Current session
Import-Csv scripts/ralph/metrics.csv |
    Where-Object { $_.session -eq "2026-01-25_120000" } |
    Export-Csv session_export.csv -NoTypeInformation

# By focus area
Import-Csv scripts/ralph/metrics.csv |
    Group-Object focus_area |
    ForEach-Object {
        $_.Name + ": " + ($_.Group | Measure-Object -Property duration_min -Average).Average.ToString("F1") + " min avg"
    }
```

### To JSON for Analysis
```powershell
Import-Csv scripts/ralph/metrics.csv |
    ConvertTo-Json -Depth 3 |
    Set-Content metrics_analysis.json
```
