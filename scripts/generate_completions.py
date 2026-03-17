#!/usr/bin/env python3
"""
Shell completion generator for the matcher CLI.

Generates bash, zsh, and fish completion scripts for the matcher pipeline CLI.
Supports --install flag to install completions to appropriate system locations.

Usage:
    python scripts/generate_completions.py --shell bash
    python scripts/generate_completions.py --shell zsh
    python scripts/generate_completions.py --shell fish
    python scripts/generate_completions.py --install           # Auto-detect shell
    python scripts/generate_completions.py --install bash     # Install specific shell
"""

import argparse
import os
import sys
from pathlib import Path

# Import standardized output functions
from script_utils import print_ok, print_warn, print_error, print_info, print_header
# Add project root and scripts directory to path for imports
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))
os.chdir(project_root)


# CLI argument definitions (mirrored from src/cli/args.py for standalone execution)
CLI_ARGS = {
    'flags': [
        ('match-only', 'Skip download, only match existing footage'),
        ('output-only', 'Regenerate OTIO/EDL/XML only'),
        ('resume', 'Resume interrupted pipeline run'),
        ('fresh', 'Force fresh start'),
        ('dry-run', 'Preview pipeline execution plan'),
        ('force-rematch', 'Force rematch all videos'),
        ('list-keywords', 'List saved keyword presets'),
        ('validate-config', 'Validate config file'),
        ('validate-config-json', 'Validate config and output as JSON'),
        ('dry-run-config', 'Load and validate config'),
        ('non-interactive', 'Run in non-interactive mode'),
        ('refresh-entities', 'Force re-download entity images'),
        ('caption-first', 'DEPRECATED: Caption-first is default'),
        ('no-caption-fallback', 'DEPRECATED: Removed in v4.0'),
        ('validate-captions', 'Validate caption-first config'),
        ('cleanup-caption-cache', 'Remove stale caption cache'),
        ('cleanup-caption-cache-dry-run', 'Preview cleanup without deleting'),
        ('rebuild-transcript-cache', 'Force rebuild transcript cache'),
        ('benchmark-transcript-cache', 'Benchmark transcript cache'),
        ('reset-budget', 'Reset retry budget counters'),
        ('dump-dependency-graph', 'Output DOT graph'),
        ('verbose-progress', 'Enable detailed progress'),
        ('rate-limit-stats', 'Display rate limit diagnostics'),
        ('progress', 'Show real-time progress'),
        ('health-check', 'Run pipeline health diagnostics'),
        ('error-summary', 'Display aggregated errors'),
        ('error-stats', 'Display error frequency analytics'),
        ('escalation-status', 'Display tier health dashboard'),
        ('trace-events', 'Enable event logging'),
        ('trace', 'Enable verbose tracing'),
        ('checkpoint-info', 'Display checkpoint info'),
        ('checkpoint-info-verbose', 'Detailed checkpoint info'),
        ('checkpoint-diff-json', 'Output diff as JSON'),
        ('validate-only', 'Run stage contract validation'),
        ('export-config-include-sensitive', 'Include sensitive in export'),
        ('config-history', 'Show config version history'),
        ('redact', 'Redact sensitive data on export'),
        ('circuit-status', 'Display circuit breaker state'),
        ('validate-checkpoint', 'Validate checkpoint integrity'),
        ('validate-checkpoint-json', 'Output validation as JSON'),
        ('diagnostic-view', 'Launch diagnostic log viewer'),
        ('config-diff', 'Show config changes'),
        ('list-checkpoints', 'List checkpoint index entries'),
        ('list-templates', 'List available config templates'),
    ],
    'options': [
        ('voiceover', '-v', 'Path to voiceover file'),
        ('keywords', '-k', 'Number of keywords to extract'),
        ('project', '-p', 'Project directory'),
        ('config', '-c', 'Path to config file'),
        ('use-template', None, 'Config template name'),
        ('use-keywords', None, 'Use saved keywords'),
        ('save-keywords', None, 'Save keywords as preset'),
        ('hot-backup', None, 'Enable hot backup'),
        ('query-checkpoints-by-stage', None, 'Query by stage'),
        ('query-checkpoints-by-date', None, 'Query by date range'),
        ('save-matching-fixtures', None, 'Save matching inputs'),
        ('caption-language', None, 'Caption language code'),
        ('test-fetch', None, 'Fetch N sample captions'),
        ('cleanup-caption-cache-days', None, 'Override cache age'),
        ('transcription-model-version', None, 'Pin Whisper version'),
        ('export-metrics', None, 'Export rate limit metrics'),
        ('export-caption-metrics', None, 'Export caption metrics'),
        ('export-resource-metrics', None, 'Export resource metrics'),
        ('export-download-metrics', None, 'Export download metrics'),
        ('dependency-graph-format', None, 'Graph output format'),
        ('export-pipeline-graph', None, 'Export pipeline graph'),
        ('checkpoint-diff', None, 'Compare two checkpoints'),
        ('checkpoint-history', None, 'Show run history'),
        ('checkpoint-export', None, 'Export checkpoint'),
        ('checkpoint-import', None, 'Import checkpoint'),
        ('export-stages', None, 'Stages to export'),
        ('checkpoint-import-target', None, 'Import target dir'),
        ('export-config', None, 'Export current config'),
        ('export-config-sections', None, 'Sections to export'),
        ('export-config-format', None, 'Config output format'),
        ('error-stats-json', None, 'Export error stats'),
        ('pipeline-mode', None, 'Pipeline execution mode'),
    ],
    'choices': {
        'pipeline-mode': ['fast', 'full', 'test'],
        'dependency-graph-format': ['dot', 'summary'],
        'export-config-format': ['yaml', 'json', 'diff'],
    },
    'subcommands': [],  # No subcommands currently
}


def generate_bash_completion():
    """Generate bash completion script."""
    script = '''#!/bin/bash
# bash completion for matcher CLI
# Auto-generated by scripts/generate_completions.py

_matcher_completion() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Main options
    opts="
        --voiceover --keywords --project --config --use-template
        --match-only --output-only --resume --fresh --dry-run
        --force-rematch --use-keywords --save-keywords --list-keywords
        --validate-config --validate-config-json --dry-run-config
        --non-interactive --save-matching-fixtures --refresh-entities
        --export-metrics --export-caption-metrics --export-resource-metrics
        --export-download-metrics --caption-first --no-caption-fallback
        --caption-language --validate-captions --test-fetch
        --cleanup-caption-cache --cleanup-caption-cache-days
        --cleanup-caption-cache-dry-run --rebuild-transcript-cache
        --benchmark-transcript-cache --transcription-model-version
        --reset-budget --dump-dependency-graph --dependency-graph-format
        --export-pipeline-graph --verbose-progress --rate-limit-stats
        --progress --health-check --error-summary --error-stats
        --error-stats-json --escalation-status --trace-events --trace
        --checkpoint-info --checkpoint-info-verbose --checkpoint-diff
        --checkpoint-diff-json --checkpoint-history --validate-only
        --export-config --export-config-include-sensitive
        --export-config-sections --export-config-format --config-history
        --checkpoint-export --checkpoint-import --export-stages
        --checkpoint-import-target --redact --circuit-status
        --validate-checkpoint --validate-checkpoint-json
        --diagnostic-view --config-diff --list-checkpoints
        --query-checkpoints-by-stage --query-checkpoints-by-date
        --hot-backup --list-templates
        -v -k -p -c
    "

    # Options that require arguments
    case "${prev}" in
        --project|-p)
            _filedir -d
            return 0
            ;;
        --config|-c)
            _filedir "@(.yaml|.yml)"
            return 0
            ;;
        --voiceover|-v)
            _filedir "@(.srt|.mp3|.wav|.mp4)"
            return 0
            ;;
        --use-keywords|--save-keywords)
            # Could complete with saved keyword presets
            return 0
            ;;
        --pipeline-mode)
            COMPREPLY=($(compgen -W "fast full test" -- "${cur}"))
            return 0
            ;;
        --dependency-graph-format|--export-config-format)
            case "${prev}" in
                --dependency-graph-format)
                    COMPREPLY=($(compgen -W "dot summary" -- "${cur}"))
                    ;;
                --export-config-format)
                    COMPREPLY=($(compgen -W "yaml json diff" -- "${cur}"))
                    ;;
            esac
            return 0
            ;;
        --hot-backup)
            if [[ "${cur}" != -* ]]; then
                _filedir -d
            fi
            return 0
            ;;
        --export-metrics|--export-caption-metrics|--export-resource-metrics|--export-download-metrics|--error-stats-json)
            _filedir "@(.json)"
            return 0
            ;;
        --export-pipeline-graph)
            _filedir "@(.dot|.png|.svg)"
            return 0
            ;;
        --export-config)
            _filedir "@(.yaml|.yml|.json)"
            return 0
            ;;
    esac

    # Handle short options
    if [[ "${cur}" == -* ]]; then
        COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
        return 0
    fi

    # Default to option completion
    COMPREPLY=($(compgen -W "${opts}" -- "${cur}"))
    return 0
}

complete -F _matcher_completion python
complete -F _matcher_completion python3
complete -F _matcher_completion main.py
complete -F _matcher_completion matcher
'''
    return script


def generate_zsh_completion():
    """Generate zsh completion script."""
    script = '''#compdef matcher python main.py
# zsh completion for matcher CLI
# Auto-generated by scripts/generate_completions.py

local -a main_opts
main_opts=(
    '--voiceover[Path to voiceover file]:file:_files -g "*.{srt,mp3,wav,mp4}"'
    '--keywords[Number of keywords to extract]:number:'
    '--project[Project directory]:directory:_directories'
    '--config[Path to config file]:file:_files -g "*.{yaml,yml}"'
    '--use-template[Config template name]:template:'
    '--match-only[Skip download, only match existing footage]'
    '--output-only[Regenerate OTIO/EDL/XML only]'
    '--resume[Resume interrupted pipeline run]'
    '--fresh[Force fresh start]'
    '--dry-run[Preview pipeline execution plan]'
    '--force-rematch[Force rematch all videos]'
    '--use-keywords[Use saved keywords]:preset:'
    '--save-keywords[Save keywords as preset]:name:'
    '--list-keywords[List saved keyword presets]'
    '--validate-config[Validate config file]'
    '--validate-config-json[Validate config and output as JSON]'
    '--dry-run-config[Load and validate config]'
    '--non-interactive[Run in non-interactive mode]'
    '--save-matching-fixtures[Save matching inputs]:path:_files'
    '--refresh-entities[Force re-download entity images]'
    '--export-metrics[Export rate limit metrics]:path:_files'
    '--export-caption-metrics[Export caption metrics]:path:_files'
    '--export-resource-metrics[Export resource metrics]:path:_files'
    '--export-download-metrics[Export download metrics]:path:_files'
    '--caption-first[DEPRECATED: Caption-first is default]'
    '--no-caption-fallback[DEPRECATED: Removed in v4.0]'
    '--caption-language[Caption language code]:code:'
    '--validate-captions[Validate caption-first config]'
    '--test-fetch[Fetch N sample captions]:number:'
    '--cleanup-caption-cache[Remove stale caption cache]'
    '--cleanup-caption-cache-days[Override cache age]:days:'
    '--cleanup-caption-cache-dry-run[Preview cleanup without deleting]'
    '--rebuild-transcript-cache[Force rebuild transcript cache]'
    '--benchmark-transcript-cache[Benchmark transcript cache]'
    '--transcription-model-version[Pin Whisper version]:version:'
    '--reset-budget[Reset retry budget counters]'
    '--dump-dependency-graph[Output DOT graph]'
    '--dependency-graph-format[Graph output format]:format:(dot summary)'
    '--export-pipeline-graph[Export pipeline graph]:path:_files'
    '--verbose-progress[Enable detailed progress]'
    '--rate-limit-stats[Display rate limit diagnostics]'
    '--progress[Show real-time progress]'
    '--health-check[Run pipeline health diagnostics]'
    '--error-summary[Display aggregated errors]'
    '--error-stats[Display error frequency analytics]'
    '--error-stats-json[Export error stats]:path:_files'
    '--escalation-status[Display tier health dashboard]'
    '--trace-events[Enable event logging]'
    '--trace[Enable verbose tracing]'
    '--checkpoint-info[Display checkpoint info]'
    '--checkpoint-info-verbose[Detailed checkpoint info]'
    '--checkpoint-diff[Compare two checkpoints]:checkpoint1: _files'
    '--checkpoint-diff-json[Output diff as JSON]'
    '--checkpoint-history[Show run history]:limit:'
    '--validate-only[Run stage contract validation]'
    '--export-config[Export current config]:path:_files'
    '--export-config-include-sensitive[Include sensitive in export]'
    '--export-config-sections[Sections to export]:sections:'
    '--export-config-format[Config output format]:format:(yaml json diff)'
    '--config-history[Show config version history]'
    '--checkpoint-export[Export checkpoint]:path:_files'
    '--checkpoint-import[Import checkpoint]:path:_files'
    '--export-stages[Stages to export]:stages:'
    '--checkpoint-import-target[Import target dir]:directory:_directories'
    '--redact[Redact sensitive data on export]'
    '--circuit-status[Display circuit breaker state]'
    '--validate-checkpoint[Validate checkpoint integrity]'
    '--validate-checkpoint-json[Output validation as JSON]'
    '--diagnostic-view[Launch diagnostic log viewer]'
    '--config-diff[Show config changes]'
    '--list-checkpoints[List checkpoint index entries]'
    '--query-checkpoints-by-stage[Query by stage]:stage:'
    '--query-checkpoints-by-date[Query by date range]:start end:'
    '--hot-backup[Enable hot backup]:path:'
    '--list-templates[List available config templates]'
    '-v[Path to voiceover file]:file:_files -g "*.{srt,mp3,wav,mp4}"'
    '-k[Number of keywords to extract]:number:'
    '-p[Project directory]:directory:_directories'
    '-c[Path to config file]:file:_files -g "*.{yaml,yml}"'
)

if [[ -n ${cur} ]]; then
    if [[ ${cur} == -* ]]; then
        # Complete options starting with -
        _describe 'option' main_opts
    else
        # Complete positional arguments
        _describe 'option' main_opts
    fi
else
    # No current word, show all options
    _describe 'option' main_opts
fi

return 0
'''
    return script


def generate_fish_completion():
    """Generate fish completion script."""
    script = '''# fish completion for matcher CLI
# Auto-generated by scripts/generate_completions.py

complete -c matcher -f
complete -c python -f -a 'main.py'
complete -c python3 -f -a 'main.py'

# Main options
complete -c matcher -l voiceover -s v -d 'Path to voiceover file (SRT, MP3, WAV, MP4)' -r -f -a '(complete -f; and ls *.srt *.mp3 *.wav *.mp4 2>/dev/null)'
complete -c matcher -l keywords -s k -d 'Number of keywords to extract'
complete -c matcher -l project -s p -d 'Project directory' -r -f -a '(complete -d)'
complete -c matcher -l config -s c -d 'Path to config file' -r -f -a '(complete -f; and ls *.yaml *.yml 2>/dev/null)'
complete -c matcher -l use-template -d 'Config template name (fast, quality, debug)'
complete -c matcher -l match-only -d 'Skip download, only match existing footage'
complete -c matcher -l output-only -d 'Regenerate OTIO/EDL/XML only'
complete -c matcher -l resume -d 'Resume interrupted pipeline run'
complete -c matcher -l fresh -d 'Force fresh start'
complete -c matcher -l dry-run -d 'Preview pipeline execution plan'
complete -c matcher -l force-rematch -d 'Force rematch all videos'
complete -c matcher -l use-keywords -d 'Use saved keywords preset'
complete -c matcher -l save-keywords -d 'Save keywords as preset'
complete -c matcher -l list-keywords -d 'List saved keyword presets'
complete -c matcher -l validate-config -d 'Validate config file'
complete -c matcher -l validate-config-json -d 'Validate config and output as JSON'
complete -c matcher -l dry-run-config -d 'Load and validate config'
complete -c matcher -l non-interactive -d 'Run in non-interactive mode'
complete -c matcher -l save-matching-fixtures -d 'Save matching inputs to fixture file' -r -f -a '(complete -f)'
complete -c matcher -l refresh-entities -d 'Force re-download entity images'
complete -c matcher -l export-metrics -d 'Export rate limit metrics to JSON' -r -f -a '(complete -f; and ls *.json 2>/dev/null)'
complete -c matcher -l export-caption-metrics -d 'Export caption metrics to JSON' -r -f -a '(complete -f; and ls *.json 2>/dev/null)'
complete -c matcher -l export-resource-metrics -d 'Export resource metrics to JSON' -r -f -a '(complete -f; and ls *.json 2>/dev/null)'
complete -c matcher -l export-download-metrics -d 'Export download metrics to JSON' -r -f -a '(complete -f; and ls *.json 2>/dev/null)'
complete -c matcher -l caption-first -d 'DEPRECATED: Caption-first is now default'
complete -c matcher -l no-caption-fallback -d 'DEPRECATED: Removed in v4.0'
complete -c matcher -l caption-language -d 'Preferred caption language (ISO 639-1)'
complete -c matcher -l validate-captions -d 'Validate caption-first configuration'
complete -c matcher -l test-fetch -d 'Fetch N sample captions for testing'
complete -c matcher -l cleanup-caption-cache -d 'Remove stale caption cache entries'
complete -c matcher -l cleanup-caption-cache-days -d 'Override max_cache_age_days for cleanup'
complete -c matcher -l cleanup-caption-cache-dry-run -d 'Preview cleanup without deleting'
complete -c matcher -l rebuild-transcript-cache -d 'Force rebuild of transcript cache'
complete -c matcher -l benchmark-transcript-cache -d 'Benchmark transcript cache startup'
complete -c matcher -l transcription-model-version -d 'Pin Whisper model version (v3, v2, v3-turbo)'
complete -c matcher -l reset-budget -d 'Reset retry budget counters'
complete -c matcher -l dump-dependency-graph -d 'Output DOT graph representation'
complete -c matcher -l dependency-graph-format -d 'Output format (dot, summary)' -x -a 'dot summary'
complete -c matcher -l export-pipeline-graph -d 'Export pipeline graph to PATH' -r -f -a '(complete -f; and ls *.dot *.png *.svg 2>/dev/null)'
complete -c matcher -l verbose-progress -d 'Enable detailed progress output'
complete -c matcher -l rate-limit-stats -d 'Display rate limit diagnostics'
complete -c matcher -l progress -d 'Show real-time progress'
complete -c matcher -l health-check -d 'Run pipeline health diagnostics'
complete -c matcher -l error-summary -d 'Display aggregated errors'
complete -c matcher -l error-stats -d 'Display error frequency analytics'
complete -c matcher -l error-stats-json -d 'Export error stats as JSON' -r -f -a '(complete -f; and ls *.json 2>/dev/null)'
complete -c matcher -l escalation-status -d 'Display tier health dashboard'
complete -c matcher -l trace-events -d 'Enable event logging'
complete -c matcher -l trace -d 'Enable verbose tracing'
complete -c matcher -l pipeline-mode -d 'Pipeline mode (fast, full, test)' -x -a 'fast full test'
complete -c matcher -l checkpoint-info -d 'Display checkpoint information'
complete -c matcher -l checkpoint-info-verbose -d 'Detailed checkpoint information'
complete -c matcher -l checkpoint-diff -d 'Compare two checkpoint files' -r -a '(complete -f)'
complete -c matcher -l checkpoint-diff-json -d 'Output diff in JSON format'
complete -c matcher -l checkpoint-history -d 'Display checkpoint run history'
complete -c matcher -l validate-only -d 'Run stage contract validation'
complete -c matcher -l export-config -d 'Export current config to file' -r -f -a '(complete -f; and ls *.yaml *.yml *.json 2>/dev/null)'
complete -c matcher -l export-config-include-sensitive -d 'Include sensitive fields in export'
complete -c matcher -l export-config-sections -d 'Comma-separated sections to export'
complete -c matcher -l export-config-format -d 'Output format (yaml, json, diff)' -x -a 'yaml json diff'
complete -c matcher -l config-history -d 'Show config version history'
complete -c matcher -l checkpoint-export -d 'Export checkpoint to file' -r -f -a '(complete -f)'
complete -c matcher -l checkpoint-import -d 'Import checkpoint from file' -r -f -a '(complete -f)'
complete -c matcher -l export-stages -d 'Comma-separated stages to export'
complete -c matcher -l checkpoint-import-target -d 'Target project directory for import' -r -f -a '(complete -d)'
complete -c matcher -l redact -d 'Redact sensitive data on export'
complete -c matcher -l circuit-status -d 'Display circuit breaker state'
complete -c matcher -l validate-checkpoint -d 'Validate checkpoint integrity'
complete -c matcher -l validate-checkpoint-json -d 'Output validation as JSON'
complete -c matcher -l diagnostic-view -d 'Launch diagnostic log viewer'
complete -c matcher -l config-diff -d 'Show config changes'
complete -c matcher -l list-checkpoints -d 'List checkpoint index entries'
complete -c matcher -l query-checkpoints-by-stage -d 'Query by stage name'
complete -c matcher -l query-checkpoints-by-date -d 'Query by date range'
complete -c matcher -l hot-backup -d 'Enable hot backup'
complete -c matcher -l list-templates -d 'List available config templates'
'''
    return script


def get_install_location(shell):
    """Get the appropriate install location for the shell."""
    home = Path.home()

    if shell == 'bash':
        # Try multiple locations
        for loc in [
            home / '.bash_completion',
            home / '.bash_completions',
            home / '.config' / 'bash_completion',
        ]:
            if loc.exists() or (loc.parent.exists() and os.access(loc.parent, os.W_OK)):
                return loc

        # Default to .bash_completion
        return home / '.bash_completion'

    elif shell == 'zsh':
        # Try compsys directory
        compdir = home / '.zsh' / 'completions'
        if compdir.exists() or os.access(home / '.zsh', os.W_OK):
            return compdir

        # Try completions dir
        compdir = home / '.config' / 'zsh' / 'completions'
        if compdir.exists() or os.access(compdir.parent, os.W_OK):
            return compdir

        # Default
        return home / '.zsh' / 'completions'

    elif shell == 'fish':
        # Fish uses functions directory
        config_dir = home / '.config' / 'fish'
        compdir = config_dir / 'completions'
        return compdir

    return None


def install_completion(shell, force=False):
    """Install completion script for the specified shell."""
    script_content = {
        'bash': generate_bash_completion,
        'zsh': generate_zsh_completion,
        'fish': generate_fish_completion,
    }.get(shell)()

    if shell == 'bash':
        target = get_install_location('bash')
        if target:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Append to existing or create new
            if target.exists() and not force:
                with open(target, 'a') as f:
                    f.write('\n# Matcher CLI completion\n')
                    f.write(script_content)
                print(f"Appended bash completion to {target}")
            else:
                with open(target, 'w') as f:
                    f.write(script_content)
                print(f"Installed bash completion to {target}")
            print(f"\nAdd to your ~/.bashrc to load:")
            print(f"  source {target}")
        else:
            print("Could not determine bash completion location")
            return False

    elif shell == 'zsh':
        target_dir = get_install_location('zsh')
        if target_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / '_matcher'
            with open(target, 'w') as f:
                f.write(script_content)
            print(f"Installed zsh completion to {target}")
            print(f"\nAdd to your ~/.zshrc to load:")
            print(f"  fpath+={target_dir}")
            print(f"  autoload -Uz compinit && compinit")
        else:
            print("Could not determine zsh completion location")
            return False

    elif shell == 'fish':
        target_dir = get_install_location('fish')
        if target_dir:
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / 'matcher.fish'
            with open(target, 'w') as f:
                f.write(script_content)
            print(f"Installed fish completion to {target}")
            print(f"\nFish loads completions automatically from:")
            print(f"  {target_dir}")
        else:
            print("Could not determine fish completion location")
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate shell completion scripts for matcher CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    %(prog)s --shell bash                    # Generate bash completion
    %(prog)s --shell zsh                      # Generate zsh completion
    %(prog)s --shell fish                     # Generate fish completion
    %(prog)s --install                        # Auto-install for current shell
    %(prog)s --install bash                   # Install bash completion
    %(prog)s --install zsh                    # Install zsh completion
    %(prog)s --install fish                   # Install fish completion
        '''
    )

    parser.add_argument(
        '--shell',
        choices=['bash', 'zsh', 'fish'],
        help='Generate completion for specified shell'
    )

    parser.add_argument(
        '--install',
        nargs='?',
        const='auto',
        choices=['auto', 'bash', 'zsh', 'fish'],
        help='Install completion to appropriate location'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing completion files'
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: stdout)'
    )

    args = parser.parse_args()

    # Handle install mode
    if args.install:
        shell = args.install

        if shell == 'auto':
            # Detect current shell
            shell = os.environ.get('SHELL', '')
            if 'bash' in shell:
                shell = 'bash'
            elif 'zsh' in shell:
                shell = 'zsh'
            elif 'fish' in shell:
                shell = 'fish'
            else:
                print("Could not detect shell. Please specify explicitly:")
                print("  --install bash")
                print("  --install zsh")
                print("  --install fish")
                return 1

        print(f"Installing {shell} completion...")
        success = install_completion(shell, args.force)
        return 0 if success else 1

    # Generate mode
    if not args.shell:
        parser.print_help()
        return 1

    # Generate the completion script
    generators = {
        'bash': generate_bash_completion,
        'zsh': generate_zsh_completion,
        'fish': generate_fish_completion,
    }

    script = generators[args.shell]()

    if args.output:
        with open(args.output, 'w') as f:
            f.write(script)
        print(f"Wrote {args.shell} completion to {args.output}")
    else:
        print(script)

    return 0


if __name__ == '__main__':
    sys.exit(main())
