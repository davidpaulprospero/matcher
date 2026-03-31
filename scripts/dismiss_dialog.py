"""Find and dismiss any Premiere Pro dialog windows."""
import ctypes
import ctypes.wintypes

user32 = ctypes.windll.user32
EnumWindows = user32.EnumWindows
GetWindowText = user32.GetWindowTextW
GetWindowTextLength = user32.GetWindowTextLengthW
IsWindowVisible = user32.IsWindowVisible
SetForegroundWindow = user32.SetForegroundWindow
PostMessage = user32.PostMessageW

WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
BN_CLICKED = 0
IDOK = 1

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

windows = []

def enum_callback(hwnd, lparam):
    if IsWindowVisible(hwnd):
        length = GetWindowTextLength(hwnd)
        if length > 0:
            buff = ctypes.create_unicode_buffer(length + 1)
            GetWindowText(hwnd, buff, length + 1)
            title = buff.value
            if title:
                windows.append((hwnd, title))
    return True

EnumWindows(WNDENUMPROC(enum_callback), 0)

# Find Premiere-related windows
for hwnd, title in windows:
    if any(k in title.lower() for k in ['premiere', 'import', 'error', 'failure', 'warning']):
        print(f"  hwnd={hwnd}, title='{title}'")

# Try to dismiss known dialog titles
dialog_titles = ['File Import Failure', 'Import', 'Error', 'Warning']
for hwnd, title in windows:
    for dt in dialog_titles:
        if dt.lower() in title.lower():
            print(f"\nDismissing: '{title}' (hwnd={hwnd})")
            SetForegroundWindow(hwnd)
            # Send WM_CLOSE
            PostMessage(hwnd, WM_CLOSE, 0, 0)
            print("  Sent WM_CLOSE")
