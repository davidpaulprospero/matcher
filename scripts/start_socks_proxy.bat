@echo off
REM ============================================
REM SOCKS5 Proxy via SSH Tunnel
REM ============================================
REM
REM This creates a local SOCKS5 proxy on port 1080
REM that routes traffic through your remote server.
REM
REM SETUP:
REM 1. Edit the variables below with your server details
REM 2. Run this script
REM 3. Keep the window open while using the proxy
REM
REM ============================================

REM === EDIT THESE VALUES ===
set SSH_USER=your_username
set SSH_HOST=your.server.com
set SSH_PORT=22
set LOCAL_PROXY_PORT=1080

REM === Optional: SSH key path (leave empty for password auth) ===
set SSH_KEY=

echo.
echo ========================================
echo Starting SOCKS5 Proxy
echo ========================================
echo.
echo Server: %SSH_USER%@%SSH_HOST%:%SSH_PORT%
echo Local proxy: socks5://127.0.0.1:%LOCAL_PROXY_PORT%
echo.
echo Press Ctrl+C to stop the proxy.
echo.

if "%SSH_KEY%"=="" (
    ssh -D %LOCAL_PROXY_PORT% -C -N %SSH_USER%@%SSH_HOST% -p %SSH_PORT%
) else (
    ssh -D %LOCAL_PROXY_PORT% -C -N -i "%SSH_KEY%" %SSH_USER%@%SSH_HOST% -p %SSH_PORT%
)

pause
