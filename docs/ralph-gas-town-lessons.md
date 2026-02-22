# Ralph Loop: Lessons from Gas Town

> Research date: 2026-02-17
> Source: [steveyegge/gastown](https://github.com/steveyegge/gastown)

## Overview

Gas Town is a multi-agent orchestration system for Claude Code with persistent work tracking. It addresses accountability, quality, efficiency, and scaling challenges by treating agent work as structured data with full attribution.

## Key Resources

- **Main Repository**: https://github.com/steveyegge/gastown
- **Documentation**:
  - [Overview](https://raw.githubusercontent.com/steveyegge/gastown/main/docs/overview.md)
  - [Reference](https://raw.githubusercontent.com/steveyegge/gastown/main/docs/reference.md)
  - [Why These Features](https://raw.githubusercontent.com/steveyegge/gastown/main/docs/why-these-features.md)
  - [Convoy Concept](https://raw.githubusercontent.com/steveyegge/gastown/main/docs/concepts/convoy.md)
  - [Molecules Concept](https://raw.githubusercontent.com/steveyegge/gastown/main/docs/concepts/molecules.md)

---

## What Ralph Can Learn from Gas Town

### 1. Molecules (Workflow Templates)

| Gas Town | Ralph (current) | Improvement |
|----------|-----------------|-------------|
| Formula → Protomolecule → Molecule | Manual story creation | Pre-defined workflow templates with auto-generated step stories |

**Benefit:** Instead of manually creating 8-12 stories per sprint, Ralph could have "molecule templates" that auto-generate the full story tree.

**Example:** A "bugfix" molecule template would auto-create:
- Reproduce issue
- Diagnose root cause
- Implement fix
- Add tests
- Verify fix

### 2. Real-Time Step Tracking

- **Gas Town**: `bd mol current` shows progress with checkmarks/arrows
- **Ralph**: Manual status updates

**Improvement:** Ralph could emit a real-time progress view showing current story, completed stories, remaining.

### 3. Git Worktree Persistence

- **Gas Town**: Hooks stored in git worktrees, survives crashes
- **Ralph**: JSON files (`prd.json`, `queue.json`)

**Improvement:** Ralph could use git-backed storage for better atomicity, history, and crash recovery.

### 4. Agent CVs / Performance History

- **Gas Town**: Agents build track records, capability-based routing
- **Ralph**: No agent performance tracking

**Improvement:** Track which Claude models, prompt strategies succeed, build "Ralph's playbook" of what works.

### 5. Validation Gates

- **Gas Town**: Structured quality verification with attribution
- **Ralph**: Basic story completion

**Improvement:** Add explicit validation steps (tests pass, manual review) before story closes.

### 6. Federation (Multi-Project Visibility)

- **Gas Town**: Cross-repo tracking
- **Ralph**: Single project focus

**Improvement:** Ralph could track work across multiple projects/sprints.

### 7. Propulsion Principle

- **Gas Town**: "If you find something on your hook, YOU RUN IT"
- **Ralph**: Waits for user confirmation

**Improvement:** Ralph could auto-execute certain tasks without waiting.

---

## Gas Town Architecture Reference

### Core Components

| Role | Description |
|------|-------------|
| **Mayor** | Global AI coordinator (like Ralph) |
| **Polecats** | Ephemeral worker agents |
| **Crew** | Persistent long-lived workers |
| **Convoys** | Work tracking units (like Ralph's sprints) |
| **Beads** | Git-backed issues (like Ralph's stories) |
| **Hooks** | Git worktree-based persistent storage |
| **Witness** | Per-rig polecat lifecycle manager |
| **Deacon** | Background supervisor daemon |

### Key Principles

1. **Propulsion Principle**: "If you find something on your hook, YOU RUN IT" — execute assignments immediately
2. **Identity via cwd**: Agent identity based on working directory
3. **Git-backed persistence**: Work survives crashes/restarts via git worktrees
4. **Scale**: Designed for 20-30 agents
5. **Attribution is mandatory**: Every action is tracked and attributable

### Design Philosophy

1. Attribution is mandatory
2. Work is structured, queryable data
3. History determines trust
4. Scale is assumed from day one
5. Verification over trust

---

## Recommended Improvements (Priority Order)

1. **Molecule Templates** — Biggest productivity boost. Pre-built sprint templates.
2. **Real-Time Progress UI** — Better visibility during execution
3. **Agent Performance Tracking** — Learn what works across sessions
4. **Git-Backed State** — More robust persistence

---

## Gas Town Command Reference

```bash
gt mayor attach           # Start Mayor session
gt convoy create "Name"   # Create work convoy
gt sling <bead-id> <rig>  # Assign work to agent
gt convoy list            # Track progress
bd mol current            # Show workflow progress
gt hook                   # Check assigned work
gt done                   # Signal completion
```
