# Experimental Feature Flags

> Control experimental features via `ralph-config.json` flags section.

## Current Flags

| Flag | Default | Description |
|------|---------|-------------|
| `autoGeneratePesterTests` | false | Auto-generate Pester tests from story acceptance criteria |
| `llmAsJudgeQuality` | false | Use Claude to verify story completion quality after each story |
| `phaseTracking` | true | Track PLAN/EXECUTE/REVIEW phases in metrics |
| `acceptanceDrivenBackpressure` | false | Inject acceptance criteria into story prompts for explicit verification |

## Usage

```json
{
  "flags": {
    "autoGeneratePesterTests": false,
    "llmAsJudgeQuality": false,
    "phaseTracking": true,
    "acceptanceDrivenBackpressure": false
  }
}
```

## Flag Descriptions

### autoGeneratePesterTests

When enabled, Ralph will generate Pester test stubs from story acceptance criteria before implementation.

**Status**: Planned
**Risk**: Low

### llmAsJudgeQuality

Uses a separate Claude session as a quality gate to verify story completion meets acceptance criteria. After a story succeeds (exit code 0), Ralph invokes a read-only Claude session with the git diff and acceptance criteria. The review agent scores the work on correctness, completeness, quality, and safety. If the score is below `review.minScoreToPass` (default 6), the story is flagged for revision.

**Status**: Active (Phase 1)
**Risk**: Medium (adds latency per story, ~60-180s review time)
**Config**: `review.enabled`, `review.model`, `review.timeout`, `review.minScoreToPass`
**Dependencies**: `scripts/ralph/review-prompt.md`

### phaseTracking

Tracks which phase (PLAN, EXECUTE, REVIEW) each iteration is in for metrics.

**Status**: Active
**Risk**: None

### acceptanceDrivenBackpressure

Injects acceptance criteria directly into the story prompt so Claude sees exactly what needs to be verified. This replaces the generic one-liner prompt with a structured prompt containing:
- Story title and ID
- All acceptance criteria as a checklist
- Instruction to verify each criterion before marking as complete

**Status**: Active (Phase 1)
**Risk**: Medium (may slow iteration velocity, increases prompt length)

## Configuration Sections

### review

Controls the independent code review agent (Story 1.1).

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | false | Enable LLM-as-judge quality review |
| `model` | "sonnet" | Model to use for review (sonnet recommended for cost) |
| `timeout` | 180 | Seconds before review times out |
| `minScoreToPass` | 6 | Minimum overall score (1-10) to pass |

### quality

Controls diff-based quality metrics (Story 1.4).

| Key | Default | Description |
|-----|---------|-------------|
| `minTestRatio` | 0.2 | Minimum ratio of test lines to implementation lines |
| `maxDiffLines` | 800 | Maximum diff size before flagging |

### regression

Controls test regression detection (Story 1.5).

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | true | Enable test baseline comparison |
| `blockOnRegression` | true | Block story completion on regression |
| `autoRollback` | false | Auto-revert on regression (Phase 4) |

### budget

Controls token budget tracking (Story 1.8).

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | false | Enable token budget enforcement |
| `maxTokensPerSprint` | 500000 | Maximum tokens per sprint |
| `warnAtPercent` | 80 | Warn when budget reaches this percentage |

### prompts

Controls adaptive prompt templates (Phase 3).

| Key | Default | Description |
|-----|---------|-------------|
| `adaptive` | false | Enable adaptive prompt building |
| `maxLength` | 4000 | Maximum prompt length in characters |

### parallel

Controls parallel story execution (Phase 4).

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | false | Enable parallel story execution |
| `maxConcurrent` | 2 | Maximum concurrent stories |

### health

Controls codebase health tracking (Phase 3).

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | true | Enable health metric collection |
| `trackCoverage` | false | Track test coverage (requires pytest-cov) |
| `trackTechDebt` | true | Track complexity and tech debt indicators |

### ordering

Controls smart story ordering (Phase 2).

| Key | Default | Description |
|-----|---------|-------------|
| `smart` | false | Enable DAG-aware story ordering |
| `preferTestsFirst` | true | Prefer test-creating stories before implementation |

## Adding New Flags

1. Add to `ralph-config.json` under `flags`
2. Document in this file with status, risk, and config references
3. Check flag value in code:
   ```powershell
   $config = Get-Content ralph-config.json | ConvertFrom-Json
   if ($config.flags.myNewFlag) { ... }
   ```
4. Add corresponding config section if the flag needs detailed settings
