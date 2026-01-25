"""
Tests for CLI argument parser.

US-002: Add CLI argument parser unit tests

Tests for src/cli/args.py covering:
- parse_arguments() with various flag combinations
- Default values
- Short and long flag variants
"""

import pytest

# Mark all tests in this module as unit tests
pytestmark = pytest.mark.unit
import sys
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestParseArgumentsVoiceover:
    """Tests for --voiceover / -v flag."""

    def test_voiceover_long_flag(self):
        """Test --voiceover returns correct path."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--voiceover', 'script.srt']):
            args = parse_arguments()
            assert args.voiceover == 'script.srt'

    def test_voiceover_short_flag(self):
        """Test -v returns correct path."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '-v', 'audio.mp3']):
            args = parse_arguments()
            assert args.voiceover == 'audio.mp3'

    def test_voiceover_default_none(self):
        """Test voiceover defaults to None when not specified."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.voiceover is None


class TestParseArgumentsProject:
    """Tests for --project / -p flag."""

    def test_project_long_flag(self):
        """Test --project returns correct path."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--project', '/path/to/project']):
            args = parse_arguments()
            assert args.project == '/path/to/project'

    def test_project_short_flag(self):
        """Test -p returns correct path."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '-p', 'E:\\Projects\\Test']):
            args = parse_arguments()
            assert args.project == 'E:\\Projects\\Test'

    def test_project_default_none(self):
        """Test project defaults to None when not specified."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.project is None


class TestParseArgumentsMatchOnlyOutputOnly:
    """Tests for --match-only and --output-only flags."""

    def test_match_only_flag_set(self):
        """Test --match-only sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--match-only']):
            args = parse_arguments()
            assert args.match_only is True

    def test_match_only_default_false(self):
        """Test match_only defaults to False."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.match_only is False

    def test_match_only_with_voiceover(self):
        """Test --match-only can be combined with --voiceover."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--voiceover', 'test.srt', '--match-only']):
            args = parse_arguments()
            assert args.match_only is True
            assert args.voiceover == 'test.srt'

    def test_output_only_flag_set(self):
        """Test --output-only sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--output-only']):
            args = parse_arguments()
            assert args.output_only is True

    def test_output_only_default_false(self):
        """Test output_only defaults to False."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.output_only is False

    def test_output_only_with_project(self):
        """Test --output-only can be combined with --project."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--project', '/path/to/project', '--output-only']):
            args = parse_arguments()
            assert args.output_only is True
            assert args.project == '/path/to/project'

    def test_match_only_and_output_only_both_set(self):
        """Test both --match-only and --output-only can be set (parser allows, main.py validates)."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--match-only', '--output-only']):
            args = parse_arguments()
            assert args.match_only is True
            assert args.output_only is True


class TestParseArgumentsResumeAndFresh:
    """Tests for --resume and --fresh flags."""

    def test_resume_flag_set(self):
        """Test --resume sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--resume']):
            args = parse_arguments()
            assert args.resume is True

    def test_fresh_flag_set(self):
        """Test --fresh sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--fresh']):
            args = parse_arguments()
            assert args.fresh is True

    def test_resume_default_false(self):
        """Test resume defaults to False."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.resume is False

    def test_fresh_default_false(self):
        """Test fresh defaults to False."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.fresh is False

    def test_resume_and_fresh_both_set(self):
        """Test both --resume and --fresh can be set (parser allows, main.py handles precedence).

        Note: These are logically mutually exclusive (can't resume AND start fresh),
        but the parser allows both. main.py handles this by first checking --fresh
        (deletes checkpoint), then checking --resume (would load deleted checkpoint).
        """
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--resume', '--fresh']):
            args = parse_arguments()
            assert args.resume is True
            assert args.fresh is True


class TestParseArgumentsMutuallyExclusiveOptions:
    """Tests for mutually exclusive or conflicting option combinations.

    The parser allows these combinations but main.py may handle them specially.
    These tests document the parser's behavior for conflicting options.
    """

    def test_resume_and_fresh_parser_allows(self):
        """Parser allows both --resume and --fresh (main.py handles precedence)."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--resume', '--fresh']):
            args = parse_arguments()
            assert args.resume is True
            assert args.fresh is True

    def test_match_only_and_output_only_parser_allows(self):
        """Parser allows both --match-only and --output-only."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--match-only', '--output-only']):
            args = parse_arguments()
            assert args.match_only is True
            assert args.output_only is True

    def test_use_keywords_and_save_keywords_parser_allows(self):
        """Parser allows both --use-keywords and --save-keywords."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--use-keywords', 'preset1', '--save-keywords', 'preset2']):
            args = parse_arguments()
            assert args.use_keywords == 'preset1'
            assert args.save_keywords == 'preset2'

    def test_validate_config_with_other_flags(self):
        """Parser allows --validate-config with other flags."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--validate-config', '--project', '/path']):
            args = parse_arguments()
            assert args.validate_config is True
            assert args.project == '/path'


class TestParseArgumentsConfig:
    """Tests for --config / -c flag."""

    def test_config_long_flag(self):
        """Test --config returns correct path."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--config', 'custom.yaml']):
            args = parse_arguments()
            assert args.config == 'custom.yaml'

    def test_config_short_flag(self):
        """Test -c returns correct path."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '-c', 'other.yaml']):
            args = parse_arguments()
            assert args.config == 'other.yaml'

    def test_config_default_value(self):
        """Test config defaults to 'config.yaml'."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.config == 'config.yaml'


class TestParseArgumentsKeywords:
    """Tests for --keywords / -k flag."""

    def test_keywords_long_flag(self):
        """Test --keywords returns correct integer."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--keywords', '30']):
            args = parse_arguments()
            assert args.keywords == 30

    def test_keywords_short_flag(self):
        """Test -k returns correct integer."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '-k', '15']):
            args = parse_arguments()
            assert args.keywords == 15

    def test_keywords_default_none(self):
        """Test keywords defaults to None."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.keywords is None


class TestParseArgumentsNonInteractive:
    """Tests for --non-interactive flag."""

    def test_non_interactive_set(self):
        """Test --non-interactive sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--non-interactive']):
            args = parse_arguments()
            assert args.non_interactive is True

    def test_non_interactive_default_false(self):
        """Test non_interactive defaults to False."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.non_interactive is False


class TestParseArgumentsKeywordPresets:
    """Tests for --save-keywords, --use-keywords, --list-keywords."""

    def test_save_keywords_without_name(self):
        """Test --save-keywords without name uses 'auto'."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--save-keywords']):
            args = parse_arguments()
            assert args.save_keywords == 'auto'

    def test_save_keywords_with_name(self):
        """Test --save-keywords with custom name."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--save-keywords', 'mypreset']):
            args = parse_arguments()
            assert args.save_keywords == 'mypreset'

    def test_use_keywords_without_name(self):
        """Test --use-keywords without name uses 'latest'."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--use-keywords']):
            args = parse_arguments()
            assert args.use_keywords == 'latest'

    def test_use_keywords_with_name(self):
        """Test --use-keywords with preset name."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--use-keywords', 'mypreset']):
            args = parse_arguments()
            assert args.use_keywords == 'mypreset'

    def test_list_keywords_flag(self):
        """Test --list-keywords sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--list-keywords']):
            args = parse_arguments()
            assert args.list_keywords is True


class TestParseArgumentsCombinations:
    """Tests for valid argument combinations."""

    def test_full_workflow_args(self):
        """Test typical full workflow arguments."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', [
            'main.py',
            '--voiceover', 'script.srt',
            '--project', '/path/to/project',
            '--config', 'custom.yaml',
            '--keywords', '25',
            '--non-interactive'
        ]):
            args = parse_arguments()
            assert args.voiceover == 'script.srt'
            assert args.project == '/path/to/project'
            assert args.config == 'custom.yaml'
            assert args.keywords == 25
            assert args.non_interactive is True

    def test_resume_workflow_args(self):
        """Test resume workflow arguments."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', [
            'main.py',
            '--project', '/path/to/project',
            '--resume',
            '--non-interactive'
        ]):
            args = parse_arguments()
            assert args.project == '/path/to/project'
            assert args.resume is True
            assert args.non_interactive is True


class TestParseArgumentsSpecialFlags:
    """Tests for special purpose flags."""

    def test_validate_config_flag(self):
        """Test --validate-config sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--validate-config']):
            args = parse_arguments()
            assert args.validate_config is True

    def test_refresh_entities_flag(self):
        """Test --refresh-entities sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--refresh-entities']):
            args = parse_arguments()
            assert args.refresh_entities is True

    def test_force_rematch_flag(self):
        """Test --force-rematch sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--force-rematch']):
            args = parse_arguments()
            assert args.force_rematch is True

    def test_save_matching_fixtures(self):
        """Test --save-matching-fixtures stores path."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--save-matching-fixtures', 'fixtures/test.json']):
            args = parse_arguments()
            assert args.save_matching_fixtures == 'fixtures/test.json'


class TestParseArgumentsCaptionFirst:
    """Tests for caption-first mode CLI flags (US-010)."""

    def test_caption_first_flag_set(self):
        """Test --caption-first sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--caption-first']):
            args = parse_arguments()
            assert args.caption_first is True

    def test_caption_first_default_false(self):
        """Test caption_first defaults to False when not specified."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.caption_first is False

    def test_caption_language_flag(self):
        """Test --caption-language stores language code."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--caption-language', 'es']):
            args = parse_arguments()
            assert args.caption_language == 'es'

    def test_caption_language_default_none(self):
        """Test caption_language defaults to None when not specified."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.caption_language is None

    def test_no_caption_fallback_flag_set(self):
        """Test --no-caption-fallback sets flag to True."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--no-caption-fallback']):
            args = parse_arguments()
            assert args.no_caption_fallback is True

    def test_no_caption_fallback_default_false(self):
        """Test no_caption_fallback defaults to False when not specified."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.no_caption_fallback is False

    def test_caption_first_with_language(self):
        """Test --caption-first combined with --caption-language."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--caption-first', '--caption-language', 'fr']):
            args = parse_arguments()
            assert args.caption_first is True
            assert args.caption_language == 'fr'

    def test_caption_first_with_no_fallback(self):
        """Test --caption-first combined with --no-caption-fallback."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', ['main.py', '--caption-first', '--no-caption-fallback']):
            args = parse_arguments()
            assert args.caption_first is True
            assert args.no_caption_fallback is True

    def test_all_caption_flags_combined(self):
        """Test all caption-first flags combined."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', [
            'main.py',
            '--caption-first',
            '--caption-language', 'de',
            '--no-caption-fallback'
        ]):
            args = parse_arguments()
            assert args.caption_first is True
            assert args.caption_language == 'de'
            assert args.no_caption_fallback is True

    def test_caption_first_with_project_and_voiceover(self):
        """Test caption-first flags with typical workflow arguments."""
        from src.cli.args import parse_arguments

        with patch.object(sys, 'argv', [
            'main.py',
            '--voiceover', 'script.srt',
            '--project', '/path/to/project',
            '--caption-first',
            '--caption-language', 'en',
            '--non-interactive'
        ]):
            args = parse_arguments()
            assert args.voiceover == 'script.srt'
            assert args.project == '/path/to/project'
            assert args.caption_first is True
            assert args.caption_language == 'en'
            assert args.non_interactive is True

    def test_caption_language_various_codes(self):
        """Test --caption-language with various ISO 639-1 codes."""
        from src.cli.args import parse_arguments

        codes = ['en', 'es', 'fr', 'de', 'ja', 'zh', 'pt', 'ru']
        for code in codes:
            with patch.object(sys, 'argv', ['main.py', '--caption-language', code]):
                args = parse_arguments()
                assert args.caption_language == code, f"Failed for code: {code}"
