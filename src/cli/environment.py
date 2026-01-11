"""
Environment setup utilities for the matcher pipeline.

Extracted from main.py (Jan 2026).
"""

from pathlib import Path

# Determine install directory (where main.py lives)
# This needs to be the parent of src/cli/
INSTALL_DIR = Path(__file__).parent.parent.parent.resolve()


def strip_extended_path_prefix(path: Path) -> Path:
    r"""
    Strip Windows extended-length path prefix (\\?\) from a Path.
    This prefix can cause issues with some applications.

    Args:
        path: Path that may have extended-length prefix

    Returns:
        Path with prefix stripped
    """
    path_str = str(path)

    prefixes = ['\\\\?\\', '\\\\.\\', '//?/', '//./']
    for prefix in prefixes:
        if path_str.startswith(prefix):
            path_str = path_str[len(prefix):]
            break

    return Path(path_str)


def load_environment(project_dir: Path = None):
    """
    Load .env file from install dir, then optionally from project dir.

    Environment loading order (later overrides earlier):
    1. Global .env from install directory
    2. Project-specific .env (if project_dir specified)
    3. Current working directory .env (fallback)

    Args:
        project_dir: Optional project directory to load .env from
    """
    try:
        from dotenv import load_dotenv

        # Load global .env from install directory
        global_env = INSTALL_DIR / '.env'
        if global_env.exists():
            load_dotenv(global_env)
            print(f"  ✓ Loaded global environment from {global_env}")

        # Load project-specific .env (overrides global)
        if project_dir:
            project_env = project_dir / '.env'
            if project_env.exists():
                load_dotenv(project_env, override=True)
                print(f"  ✓ Loaded project environment from {project_env}")

        # Fallback: try current working directory
        if not global_env.exists() and Path('.env').exists():
            load_dotenv('.env')
            print(f"  ✓ Loaded environment from .env")

    except ImportError:
        # python-dotenv not installed
        pass
