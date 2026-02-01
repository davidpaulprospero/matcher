# Ralph Loop Agent Configuration

> Per Huntley's playbook: Keep agent instructions under 60 lines.

## Primary Agent: Story Executor

- **Role**: Execute one user story per iteration
- **Context**: Fresh window per story (no context rot)
- **Backpressure**: Tests must pass before story marked complete

## Behaviors

| Behavior | Setting |
|----------|---------|
| Auto-commit | After tests pass |
| Retry on fail | Up to 2 times |
| Fast fail | After 3 consecutive failures |

## Constraints

1. One story at a time
2. Tests gate completion
3. No multi-story planning
4. Fresh context each iteration

## Phases

```
PLAN -> EXECUTE -> REVIEW
  |        |         |
  v        v         v
PRD     Stories    Verify
```

## Exit Conditions

- Sprint complete (all stories pass)
- Max iterations reached
- Fast fail triggered
- User interrupt (Ctrl+C)
