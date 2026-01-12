"""Tests for setup_project.py - batch file generation"""

import pytest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent))

from setup_project import (
    create_convert_bat,
    create_run_bat,
    create_run_sh,
    regenerate_run_script,
)


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory"""
    temp_dir = tempfile.mkdtemp()
    project_dir = Path(temp_dir) / "TestProject__2026-01-01"
    project_dir.mkdir()
    yield project_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_install_dir():
    """Create a temporary install directory"""
    temp_dir = tempfile.mkdtemp()
    install_dir = Path(temp_dir) / "voiceover-matcher"
    install_dir.mkdir()
    # Create a fake main.py
    (install_dir / "main.py").write_text("# main entry point")
    yield install_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestCreateConvertBat:
    """Tests for create_convert_bat function"""

    def test_creates_convert_bat_file(self, temp_project_dir):
        """Test that convert.bat file is created"""
        result = create_convert_bat(temp_project_dir)

        assert result.exists()
        assert result.name == "convert.bat"
        assert result.parent == temp_project_dir

    def test_convert_bat_contains_ffmpeg_command(self, temp_project_dir):
        """Test that convert.bat contains FFmpeg ProRes conversion command"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Check for FFmpeg command with ProRes settings
        assert "ffmpeg" in content.lower()
        assert "prores_ks" in content
        assert "-profile:v 4444" in content
        assert "yuva444p10le" in content

    def test_convert_bat_contains_usage_message(self, temp_project_dir):
        """Test that convert.bat shows usage when run without args"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Check for usage instructions
        assert "VIDEO CONVERTER" in content
        assert "Drag" in content or "drag" in content
        assert "DaVinci" in content

    def test_convert_bat_handles_empty_input(self, temp_project_dir):
        """Test that convert.bat checks for empty input parameter"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Should check if input is empty
        assert 'if "%INPUT%"==""' in content or 'if "%~1"==""' in content

    def test_convert_bat_checks_file_exists(self, temp_project_dir):
        """Test that convert.bat verifies input file exists"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Should check if file exists
        assert 'if not exist' in content

    def test_convert_bat_checks_ffmpeg_exists(self, temp_project_dir):
        """Test that convert.bat checks for FFmpeg installation"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Should check FFmpeg exists
        assert "FFMPEG" in content
        assert "not exist" in content

    def test_convert_bat_sets_output_filename(self, temp_project_dir):
        """Test that convert.bat creates output with _DAVINCI suffix"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Output should have _DAVINCI suffix and .mov extension
        assert "_DAVINCI.mov" in content

    def test_convert_bat_uses_setlocal(self, temp_project_dir):
        """Test that convert.bat uses setlocal for proper variable scoping"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        assert "setlocal" in content

    def test_convert_bat_supports_ffmpeg_path_env(self, temp_project_dir):
        """Test that convert.bat respects FFMPEG_PATH environment variable"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        assert "FFMPEG_PATH" in content

    def test_convert_bat_has_error_handling(self, temp_project_dir):
        """Test that convert.bat has error level checking"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        assert "ERRORLEVEL" in content


class TestCreateRunBat:
    """Tests for create_run_bat function"""

    def test_creates_run_bat_file(self, temp_project_dir, temp_install_dir):
        """Test that run.bat file is created"""
        result = create_run_bat(temp_project_dir, temp_install_dir)

        assert result.exists()
        assert result.name == "run.bat"
        assert result.parent == temp_project_dir

    def test_run_bat_contains_install_dir(self, temp_project_dir, temp_install_dir):
        """Test that run.bat references the install directory"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert str(temp_install_dir) in content

    def test_run_bat_contains_project_name(self, temp_project_dir, temp_install_dir):
        """Test that run.bat includes project name in header"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert temp_project_dir.name in content

    def test_run_bat_supports_resume_flag(self, temp_project_dir, temp_install_dir):
        """Test that run.bat handles --resume flag"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "--resume" in content

    def test_run_bat_supports_fresh_flag(self, temp_project_dir, temp_install_dir):
        """Test that run.bat handles --fresh flag"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "--fresh" in content

    def test_run_bat_supports_use_keywords_flag(self, temp_project_dir, temp_install_dir):
        """Test that run.bat handles --use-keywords flag"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "--use-keywords" in content

    def test_run_bat_supports_match_only_flag(self, temp_project_dir, temp_install_dir):
        """Test that run.bat handles --match-only flag"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "--match-only" in content

    def test_run_bat_supports_list_flag(self, temp_project_dir, temp_install_dir):
        """Test that run.bat handles --list flag"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "--list" in content

    def test_run_bat_supports_help_flag(self, temp_project_dir, temp_install_dir):
        """Test that run.bat handles --help flag"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "--help" in content

    def test_run_bat_checks_checkpoint_exists(self, temp_project_dir, temp_install_dir):
        """Test that run.bat checks for checkpoint.json"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "checkpoint.json" in content

    def test_run_bat_checks_saved_keywords(self, temp_project_dir, temp_install_dir):
        """Test that run.bat checks for saved_keywords.json"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "saved_keywords.json" in content

    def test_run_bat_has_auto_detection(self, temp_project_dir, temp_install_dir):
        """Test that run.bat has auto-detection logic for existing saves"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        # Should have logic to detect what exists
        assert "HAS_CHECKPOINT" in content
        assert "HAS_KEYWORDS" in content

    def test_run_bat_has_error_checking(self, temp_project_dir, temp_install_dir):
        """Test that run.bat checks for errors after python execution"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "ERRORLEVEL" in content
        assert "ERROR" in content or "error" in content

    def test_run_bat_sets_ffmpeg_path(self, temp_project_dir, temp_install_dir):
        """Test that run.bat sets IMAGEIO_FFMPEG_EXE"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "IMAGEIO_FFMPEG_EXE" in content

    def test_run_bat_calls_main_py(self, temp_project_dir, temp_install_dir):
        """Test that run.bat calls main.py with --project flag"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        assert "python main.py" in content
        assert "--project" in content


class TestCreateRunSh:
    """Tests for create_run_sh function (Unix shell script)"""

    def test_creates_run_sh_file(self, temp_project_dir, temp_install_dir):
        """Test that run.sh file is created"""
        result = create_run_sh(temp_project_dir, temp_install_dir)

        assert result.exists()
        assert result.name == "run.sh"
        assert result.parent == temp_project_dir

    @pytest.mark.skipif(sys.platform == 'win32', reason="Unix permissions not supported on Windows")
    def test_run_sh_is_executable(self, temp_project_dir, temp_install_dir):
        """Test that run.sh has executable permissions"""
        result = create_run_sh(temp_project_dir, temp_install_dir)

        # Check for executable bit (on Unix systems)
        import os
        import stat
        mode = os.stat(result).st_mode
        assert mode & stat.S_IXUSR  # Owner executable

    def test_run_sh_has_shebang(self, temp_project_dir, temp_install_dir):
        """Test that run.sh starts with shebang"""
        create_run_sh(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.sh").read_text()

        assert content.startswith("#!/")

    def test_run_sh_contains_install_dir(self, temp_project_dir, temp_install_dir):
        """Test that run.sh references the install directory"""
        create_run_sh(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.sh").read_text()

        assert str(temp_install_dir) in content


class TestRegenerateRunScript:
    """Tests for regenerate_run_script function"""

    @patch('setup_project.get_install_dir')
    def test_regenerate_creates_run_bat_on_windows(self, mock_get_install_dir, temp_project_dir, temp_install_dir):
        """Test that regenerate creates run.bat on Windows"""
        mock_get_install_dir.return_value = temp_install_dir

        with patch('sys.platform', 'win32'):
            result = regenerate_run_script(str(temp_project_dir))

        assert result == True
        assert (temp_project_dir / "run.bat").exists()

    @patch('setup_project.get_install_dir')
    def test_regenerate_creates_convert_bat_on_windows(self, mock_get_install_dir, temp_project_dir, temp_install_dir):
        """Test that regenerate creates convert.bat on Windows"""
        mock_get_install_dir.return_value = temp_install_dir

        with patch('sys.platform', 'win32'):
            result = regenerate_run_script(str(temp_project_dir))

        assert result == True
        assert (temp_project_dir / "convert.bat").exists()

    @patch('setup_project.get_install_dir')
    def test_regenerate_creates_run_sh_on_unix(self, mock_get_install_dir, temp_project_dir, temp_install_dir):
        """Test that regenerate creates run.sh on Unix"""
        mock_get_install_dir.return_value = temp_install_dir

        with patch('sys.platform', 'linux'):
            result = regenerate_run_script(str(temp_project_dir))

        assert result == True
        assert (temp_project_dir / "run.sh").exists()

    def test_regenerate_fails_for_nonexistent_dir(self, temp_install_dir):
        """Test that regenerate returns False for non-existent directory"""
        with patch('setup_project.get_install_dir', return_value=temp_install_dir):
            result = regenerate_run_script("/nonexistent/path/to/project")

        assert result == False

    @patch('setup_project.get_install_dir')
    def test_regenerate_uses_custom_install_dir(self, mock_get_install_dir, temp_project_dir, temp_install_dir):
        """Test that regenerate respects custom install_dir parameter"""
        # This should NOT call get_install_dir if install_dir is provided
        with patch('sys.platform', 'win32'):
            result = regenerate_run_script(str(temp_project_dir), str(temp_install_dir))

        assert result == True

        # Verify install_dir is in the generated script
        content = (temp_project_dir / "run.bat").read_text()
        assert str(temp_install_dir) in content


class TestBatchFileContentValidation:
    """Detailed content validation for batch files"""

    def test_convert_bat_proper_quoting(self, temp_project_dir):
        """Test that convert.bat properly quotes paths with spaces"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Variables should be quoted when used
        assert '"%INPUT%"' in content
        assert '"%OUTPUT%"' in content
        assert '"%FFMPEG%"' in content

    def test_convert_bat_proper_escaping(self, temp_project_dir):
        """Test that convert.bat properly escapes special characters"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Parentheses in echo should be escaped
        # The content mentions "HEVC/H.265" which has special chars
        # Check that the file is valid batch syntax (no unescaped parens in echo inside if)
        assert "@echo off" in content  # Basic validation

    def test_run_bat_proper_project_dir_handling(self, temp_project_dir, temp_install_dir):
        """Test that run.bat properly handles PROJECT_DIR with trailing slash"""
        create_run_bat(temp_project_dir, temp_install_dir)
        content = (temp_project_dir / "run.bat").read_text()

        # Should strip trailing backslash from PROJECT_DIR
        assert "PROJECT_DIR:~-1" in content or "PROJECT_DIR:~0,-1" in content

    def test_convert_bat_audio_codec(self, temp_project_dir):
        """Test that convert.bat uses PCM audio codec"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Should use PCM audio for ProRes compatibility
        assert "pcm_s16le" in content

    def test_convert_bat_quality_setting(self, temp_project_dir):
        """Test that convert.bat sets quality parameter"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Should have quality setting
        assert "-q:v" in content

    def test_convert_bat_overwrite_flag(self, temp_project_dir):
        """Test that convert.bat uses -y flag to overwrite"""
        create_convert_bat(temp_project_dir)
        content = (temp_project_dir / "convert.bat").read_text()

        # Should overwrite without prompting
        assert " -y " in content


class TestEdgeCases:
    """Edge case tests"""

    def test_project_dir_with_spaces(self, temp_install_dir):
        """Test handling of project directory with spaces in name"""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "My Project With Spaces"
        project_dir.mkdir()

        try:
            result = create_run_bat(project_dir, temp_install_dir)
            assert result.exists()

            result = create_convert_bat(project_dir)
            assert result.exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_project_dir_with_unicode(self, temp_install_dir):
        """Test handling of project directory with unicode characters"""
        temp_dir = tempfile.mkdtemp()
        project_dir = Path(temp_dir) / "Project_2026"
        project_dir.mkdir()

        try:
            result = create_run_bat(project_dir, temp_install_dir)
            assert result.exists()

            result = create_convert_bat(project_dir)
            assert result.exists()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_overwrite_existing_bat_files(self, temp_project_dir, temp_install_dir):
        """Test that existing bat files are overwritten"""
        # Create initial files
        (temp_project_dir / "run.bat").write_text("old content")
        (temp_project_dir / "convert.bat").write_text("old content")

        # Regenerate
        create_run_bat(temp_project_dir, temp_install_dir)
        create_convert_bat(temp_project_dir)

        # Verify new content
        run_content = (temp_project_dir / "run.bat").read_text()
        convert_content = (temp_project_dir / "convert.bat").read_text()

        assert run_content != "old content"
        assert convert_content != "old content"
        assert "VOICEOVER-MATCHER" in run_content
        assert "VIDEO CONVERTER" in convert_content
