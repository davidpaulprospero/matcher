# Re-export symbols from the legacy src/utils.py module so that
# "from src.utils import X" keeps working now that src/utils/ is a package.
import importlib.util as _iu
import sys as _sys
from pathlib import Path as _Path

_utils_file = str(_Path(__file__).resolve().parent.parent / "utils.py")
_spec = _iu.spec_from_file_location("src._utils_compat", _utils_file)
_mod = _iu.module_from_spec(_spec)
_sys.modules["src._utils_compat"] = _mod
_spec.loader.exec_module(_mod)

# Pull all public names into this package namespace
_all_names = [n for n in dir(_mod) if not n.startswith("_")]
globals().update({n: getattr(_mod, n) for n in _all_names})
