"""Read and interact with Windows dialogs using UI Automation."""
import ctypes
import ctypes.wintypes
from ctypes import POINTER, byref, HRESULT
import comtypes
import comtypes.client

def read_premiere_dialogs():
    """Find and read any open Premiere Pro dialog windows."""
    # Use UI Automation
    uia = comtypes.client.CreateObject(
        "{ff48dba4-60ef-4201-aa87-54103eef594e}",  # CUIAutomation
        interface=comtypes.gen.UIAutomationClient.IUIAutomation
    )

    root = uia.GetRootElement()

    # Find all top-level windows
    condition = uia.CreateTrueCondition()
    walker = uia.CreateTreeWalker(condition)

    # Search for Premiere-related windows with dialog-like names
    dialog_keywords = ['import', 'edl', 'error', 'failure', 'warning', 'information', 'premiere']

    # Enumerate top-level windows
    child = walker.GetFirstChildElement(root)
    found_dialogs = []

    while child:
        try:
            name = child.CurrentName
            class_name = child.CurrentClassName
            control_type = child.CurrentControlType

            if name and any(k in name.lower() for k in dialog_keywords):
                print(f"\n=== Dialog: '{name}' (class: {class_name}, type: {control_type}) ===")
                found_dialogs.append(child)

                # Read all child controls
                read_controls(child, walker, indent=1)
        except Exception:
            pass

        try:
            child = walker.GetNextSiblingElement(child)
        except Exception:
            break

    if not found_dialogs:
        print("No Premiere dialogs found.")

    return found_dialogs

def read_controls(element, walker, indent=0):
    """Recursively read all controls in a dialog."""
    prefix = "  " * indent
    child = None
    try:
        child = walker.GetFirstChildElement(element)
    except Exception:
        return

    while child:
        try:
            name = child.CurrentName or ""
            control_type = child.CurrentControlType
            class_name = child.CurrentClassName or ""

            # Map control type IDs to names
            type_names = {
                50000: "Button",
                50001: "Calendar",
                50002: "CheckBox",
                50003: "ComboBox",
                50004: "Edit",
                50005: "Hyperlink",
                50006: "Image",
                50007: "ListItem",
                50008: "List",
                50009: "Menu",
                50010: "MenuBar",
                50011: "MenuItem",
                50012: "ProgressBar",
                50013: "RadioButton",
                50014: "ScrollBar",
                50015: "Slider",
                50016: "Spinner",
                50017: "StatusBar",
                50020: "Tab",
                50021: "TabItem",
                50022: "Text",
                50025: "ToolBar",
                50026: "ToolTip",
                50027: "Tree",
                50028: "TreeItem",
                50030: "Group",
                50031: "Thumb",
                50032: "DataGrid",
                50033: "DataItem",
                50034: "Document",
                50035: "SplitButton",
                50036: "Window",
                50037: "Pane",
                50038: "Header",
                50039: "HeaderItem",
                50040: "Table",
            }
            type_name = type_names.get(control_type, f"Type_{control_type}")

            if name or type_name in ("Button", "RadioButton", "CheckBox", "Edit", "Text", "ComboBox"):
                # For radio buttons, check if selected
                extra = ""
                if type_name == "RadioButton":
                    try:
                        pattern = child.GetCurrentPattern(10015)  # SelectionItemPattern
                        if pattern:
                            sel = pattern.QueryInterface(comtypes.gen.UIAutomationClient.IUIAutomationSelectionItemPattern)
                            extra = " [SELECTED]" if sel.CurrentIsSelected else " [ ]"
                    except Exception:
                        pass

                if type_name == "ComboBox":
                    try:
                        val_pattern = child.GetCurrentPattern(10002)  # ValuePattern
                        if val_pattern:
                            val = val_pattern.QueryInterface(comtypes.gen.UIAutomationClient.IUIAutomationValuePattern)
                            extra = f" = '{val.CurrentValue}'"
                    except Exception:
                        pass

                print(f"{prefix}[{type_name}] {name}{extra}")

            # Recurse into children
            read_controls(child, walker, indent + 1)

        except Exception as e:
            pass

        try:
            child = walker.GetNextSiblingElement(child)
        except Exception:
            break


if __name__ == "__main__":
    # Generate the UIAutomation type library
    try:
        comtypes.client.GetModule("UIAutomationCore.dll")
    except Exception:
        pass

    read_premiere_dialogs()
