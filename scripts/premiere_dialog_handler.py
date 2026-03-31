"""Detect, read, and handle Premiere Pro dialogs."""
import ctypes
import ctypes.wintypes
import json
import requests
import time

user32 = ctypes.windll.user32

PANEL_URL = "http://127.0.0.1:3000"

# Win32 constants
WM_CLOSE = 0x0010
WM_COMMAND = 0x0111
BM_CLICK = 0x00F5
GW_CHILD = 5
IDOK = 1

EnumWindows = user32.EnumWindows
EnumChildWindows = user32.EnumChildWindows
GetWindowText = user32.GetWindowTextW
GetWindowTextLength = user32.GetWindowTextLengthW
GetClassName = user32.GetClassNameW
IsWindowVisible = user32.IsWindowVisible
SetForegroundWindow = user32.SetForegroundWindow
PostMessage = user32.PostMessageW
SendMessage = user32.SendMessageW
FindWindowEx = user32.FindWindowExW
GetDlgItem = user32.GetDlgItem

WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.wintypes.BOOL, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)


def get_window_text(hwnd):
    length = GetWindowTextLength(hwnd)
    if length == 0:
        return ""
    buff = ctypes.create_unicode_buffer(length + 1)
    GetWindowText(hwnd, buff, length + 1)
    return buff.value


def get_class_name(hwnd):
    buff = ctypes.create_unicode_buffer(256)
    GetClassName(hwnd, buff, 256)
    return buff.value


def find_all_windows():
    """Find all visible windows."""
    results = []
    def callback(hwnd, _):
        if IsWindowVisible(hwnd):
            title = get_window_text(hwnd)
            if title:
                results.append((hwnd, title, get_class_name(hwnd)))
        return True
    EnumWindows(WNDENUMPROC(callback), 0)
    return results


def find_child_windows(parent_hwnd):
    """Find all child windows/controls of a parent."""
    children = []
    def callback(hwnd, _):
        title = get_window_text(hwnd)
        cls = get_class_name(hwnd)
        children.append((hwnd, title, cls))
        return True
    EnumChildWindows(parent_hwnd, WNDENUMPROC(callback), 0)
    return children


def check_premiere_blocked():
    """Check if Premiere has a blocking dialog open."""
    try:
        requests.get(PANEL_URL, timeout=2)
    except:
        return None, "Premiere panel not running"

    try:
        requests.post(PANEL_URL, json={"to_eval": "1;"}, timeout=3)
        return False, "Premiere is responsive (no dialog)"
    except requests.exceptions.Timeout:
        return True, "Premiere is blocked (dialog open)"
    except:
        return None, "Connection error"


def find_premiere_dialogs():
    """Find dialog windows that belong to Premiere."""
    windows = find_all_windows()
    dialogs = []

    # Known dialog titles
    dialog_patterns = [
        'file import failure', 'import files', 'edl information',
        'error', 'warning', 'confirm', 'save', 'missing',
        'codec', 'render', 'export'
    ]

    for hwnd, title, cls in windows:
        title_lower = title.lower()
        # Check if it's a known dialog
        if any(p in title_lower for p in dialog_patterns):
            children = find_child_windows(hwnd)
            dialog_info = {
                "hwnd": hwnd,
                "title": title,
                "class": cls,
                "controls": []
            }
            for c_hwnd, c_text, c_cls in children:
                if c_text or c_cls in ("Button", "Static", "Edit", "ComboBox"):
                    dialog_info["controls"].append({
                        "hwnd": c_hwnd,
                        "text": c_text,
                        "class": c_cls
                    })
            dialogs.append(dialog_info)

    return dialogs


def click_button(hwnd, button_text="OK"):
    """Find and click a button in a dialog."""
    children = find_child_windows(hwnd)
    for c_hwnd, c_text, c_cls in children:
        if c_text.lower() == button_text.lower() and "button" in c_cls.lower():
            SendMessage(c_hwnd, BM_CLICK, 0, 0)
            return True
    # Fallback: send Enter key
    SetForegroundWindow(hwnd)
    import time
    time.sleep(0.2)
    user32.keybd_event(0x0D, 0, 0, 0)  # VK_RETURN down
    user32.keybd_event(0x0D, 0, 2, 0)  # VK_RETURN up
    return True


def handle_dialog(dialog_info):
    """Decide how to handle a specific dialog."""
    title = dialog_info["title"].lower()
    controls = dialog_info["controls"]
    hwnd = dialog_info["hwnd"]

    print(f"\nDialog: '{dialog_info['title']}'")
    print(f"  Controls: {[c['text'] for c in controls if c['text']]}")

    if "edl information" in title:
        # NTSC is default and correct - just click OK
        print("  -> EDL Information: clicking OK (NTSC selected)")
        click_button(hwnd, "OK")
        return "clicked_ok"

    elif "file import failure" in title:
        print("  -> Import failure: dismissing")
        PostMessage(hwnd, WM_CLOSE, 0, 0)
        return "dismissed"

    elif "import" in title:
        print("  -> Import dialog: clicking OK")
        click_button(hwnd, "OK")
        return "clicked_ok"

    else:
        print(f"  -> Unknown dialog, closing")
        PostMessage(hwnd, WM_CLOSE, 0, 0)
        return "closed"


if __name__ == "__main__":
    import sys

    # Check if blocked
    blocked, msg = check_premiere_blocked()
    print(f"Blocked: {blocked} - {msg}")

    # Find dialogs
    dialogs = find_premiere_dialogs()
    if not dialogs:
        print("No dialogs found.")
    else:
        for d in dialogs:
            handle_dialog(d)

    # Re-check
    time.sleep(0.5)
    blocked2, msg2 = check_premiere_blocked()
    print(f"\nAfter handling: {blocked2} - {msg2}")
