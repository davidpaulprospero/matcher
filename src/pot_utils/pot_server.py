"""
PO Token Server Management

Ensures the bgutil PO Token server is running for YouTube access.
The server is required for caption/subtitle fetching and video downloads.
"""

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Default server configuration
POT_SERVER_HOST = "127.0.0.1"
POT_SERVER_PORT = 4416
POT_SERVER_SCRIPT = Path.home() / "bgutil-ytdlp-pot-provider" / "server" / "build" / "main.js"


def is_pot_server_running(host: str = POT_SERVER_HOST, port: int = POT_SERVER_PORT) -> bool:
    """Check if the PO Token server is running by attempting to connect."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception as e:
        logger.debug(f"Error checking PO Token server: {e}")
        return False


def start_pot_server(
    script_path: Path = POT_SERVER_SCRIPT,
    host: str = POT_SERVER_HOST,
    port: int = POT_SERVER_PORT,
    timeout: float = 10.0
) -> bool:
    """
    Start the PO Token server if not already running.

    Returns True if server is running (either already was or successfully started).
    """
    # Check if already running
    if is_pot_server_running(host, port):
        logger.info(f"PO Token server already running on {host}:{port}")
        return True

    # Verify script exists
    if not script_path.exists():
        logger.warning(f"PO Token server script not found: {script_path}")
        logger.warning("Run: git clone https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git ~/bgutil-ytdlp-pot-provider")
        logger.warning("Then: cd ~/bgutil-ytdlp-pot-provider/server && npm install && npx tsc")
        return False

    # Start the server
    logger.info(f"Starting PO Token server from {script_path}...")

    try:
        # Use different startup methods based on platform
        if sys.platform == "win32":
            # Windows: use subprocess with CREATE_NEW_PROCESS_GROUP
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            process = subprocess.Popen(
                ["node", str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                startupinfo=startupinfo,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
            )
        else:
            # Unix: use nohup-style detachment
            process = subprocess.Popen(
                ["node", str(script_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if is_pot_server_running(host, port):
                logger.info(f"PO Token server started successfully (PID: {process.pid})")
                return True
            time.sleep(0.5)

        logger.error(f"PO Token server failed to start within {timeout}s")
        return False

    except FileNotFoundError:
        logger.error("Node.js not found. Please install Node.js to use the PO Token server.")
        return False
    except Exception as e:
        logger.error(f"Failed to start PO Token server: {e}")
        return False


def ensure_pot_server(required: bool = False) -> bool:
    """
    Ensure PO Token server is running. Start it if not.

    Args:
        required: If True, raise an exception if server cannot be started.

    Returns:
        True if server is running, False otherwise.
    """
    if is_pot_server_running():
        return True

    success = start_pot_server()

    if not success and required:
        raise RuntimeError(
            "PO Token server is required but could not be started. "
            "Please start it manually: node ~/bgutil-ytdlp-pot-provider/server/build/main.js"
        )

    return success


def get_pot_server_status() -> dict:
    """Get detailed status of the PO Token server."""
    running = is_pot_server_running()
    script_exists = POT_SERVER_SCRIPT.exists()

    return {
        "running": running,
        "host": POT_SERVER_HOST,
        "port": POT_SERVER_PORT,
        "script_exists": script_exists,
        "script_path": str(POT_SERVER_SCRIPT),
    }
