# Mutation Testing Guide

This document explains how to use mutation testing in the voiceover-matcher-subtitle project.

## What is Mutation Testing?

Mutation testing is a technique that evaluates the quality of your test suite by making small changes (mutations) to your source code and checking if your tests catch them.

- **Killed mutant**: Tests failed = good test coverage
- **Survived mutant**: Tests passed = potential test gap
- **Timeout mutant**: Tests took too long = counted as caught

A high mutation score (percentage of killed mutants) indicates effective tests.

## Quick Start

```bash
# Run mutation testing on default target (src/matching/scoring.py)
python scripts/run_mutation_tests.py

# Show results from last run
python scripts/run_mutation_tests.py --results-only

# Show surviving mutants (test gaps)
python scripts/run_mutation_tests.py --show-survivors

# CI mode with threshold check
python scripts/run_mutation_tests.py --ci --threshold 0.70
```

## Configuration

Mutation testing is configured in `setup.cfg`:

```ini
[mutmut]
# Source modules to mutate
paths_to_mutate =
    src/matching/scoring.py
    src/matching/similarity.py

# Test runner command
runner = pytest tests/test_matching*.py -x -q --tb=no

# Paths to exclude
paths_to_exclude =
    src/matching/__pycache__
    src/matching/tests
```

## Runner Script Options

The `scripts/run_mutation_tests.py` script provides several options:

| Option | Description |
|--------|-------------|
| `--module`, `-m` | Target module to mutate (default: src/matching/scoring.py) |
| `--threshold`, `-t` | Mutation score threshold (default: 0.70) |
| `--tests`, `-T` | Test pattern for killing mutants |
| `--all-matching` | Run on all matching modules |
| `--html` | Generate HTML report |
| `--show-survivors` | Show surviving mutants |
| `--results-only` | Show results without re-running |
| `--ci` | CI mode - exit code 1 if below threshold |
| `--save`, `-s` | Save results to JSON |
| `--verbose`, `-v` | Verbose output |

## Baseline Results

### src/matching/scoring.py (2026-01-29)

Initial baseline from partial run (first 31 mutants tested):

| Metric | Value |
|--------|-------|
| Total mutants | ~1138 |
| Tested | 31 |
| Killed | 14 |
| Survived | 17 |
| Untested | 1107 |
| Mutation Score | 45.2% (partial) |

**Note:** Full mutation testing takes significant time (~30+ minutes for scoring.py).
A complete baseline run is recommended for accurate metrics. Run with:
```bash
python scripts/run_mutation_tests.py --save --verbose
```

### Surviving Mutants (Test Gaps)

The following mutant IDs survived in scoring.py (lines need additional test coverage):
- Mutants 5, 7-9, 11-16, 20, 22-24, 28-29, 31

To investigate a surviving mutant:
```bash
PYTHONIOENCODING=utf-8 python -m mutmut show 5
```

## Common Mutation Types

mutmut applies these mutation operators:

| Operator | Description | Example |
|----------|-------------|---------|
| **SDL** | Statement deletion | Remove a line of code |
| **AOR** | Arithmetic operator | `+` -> `-`, `*` -> `/` |
| **ROR** | Relational operator | `<` -> `<=`, `==` -> `!=` |
| **BCR** | Boolean constant | `True` -> `False` |
| **SDL** | String mutation | `"text"` -> `"XXtextXX"` |
| **NE** | Negate expression | `x` -> `not x` |

## Best Practices

### 1. Focus on Critical Code

Mutation testing is CPU-intensive. Prioritize:
- Core business logic (matching algorithms)
- Scoring functions
- Configuration parsing
- Error handling paths

### 2. Investigate Survivors

When a mutant survives:
1. View the mutation: `mutmut show <id>`
2. Understand what changed
3. Ask: "Should my tests catch this?"
4. Add targeted test if needed

### 3. Acceptable Survivors

Some survivors are acceptable:
- Logging statements
- Debug output
- Error message text changes
- Performance optimizations

### 4. Set Realistic Thresholds

- **70%**: Minimum acceptable for critical code
- **80%**: Good coverage target
- **90%+**: Excellent but may require excessive testing

## CI Integration

The project includes an optional CI job for mutation testing.
See `.github/workflows/tests.yml` for the mutation testing job configuration.

### Running in CI

```yaml
# Example CI job configuration
mutation-testing:
  runs-on: ubuntu-latest
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    - run: pip install -r requirements-dev.txt
    - run: python scripts/run_mutation_tests.py --ci --threshold 0.70
```

**Note:** Mutation testing is time-intensive. Consider running as:
- Nightly scheduled job
- Manual trigger only
- On main branch merges only

## Troubleshooting

### Unicode Errors on Windows

If you see encoding errors in the terminal:
```bash
PYTHONIOENCODING=utf-8 python -m mutmut results
```

### Cache Issues

Clear the mutmut cache to start fresh:
```bash
rm .mutmut-cache
python -m mutmut run
```

### Slow Tests

If mutation testing is too slow:
1. Target smaller modules: `--module src/matching/specific_file.py`
2. Use faster test subset: `--tests "tests/test_scoring.py"`
3. Run overnight or as scheduled CI job

## Related Documentation

- [tests/README.md](README.md) - Test suite overview
- [CLAUDE.md](../CLAUDE.md) - Project conventions
- [mutmut docs](https://mutmut.readthedocs.io/) - Official documentation
