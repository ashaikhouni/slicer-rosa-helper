"""Shared bridge to access Loader (`RosaHelperLogic`) core services.

This keeps legacy module dependencies centralized while Phase 8 cleanup
finishes extracting remaining service methods out of `RosaHelperLogic`.
"""

import importlib.util
import os

_LOADER_CORE_INSTANCE = None


def get_loader_core(module_dir):
    """Load and cache `RosaHelperLogic` from extension source tree."""
    global _LOADER_CORE_INSTANCE
    if _LOADER_CORE_INSTANCE is not None:
        return _LOADER_CORE_INSTANCE

    helper_path = os.path.join(os.path.dirname(module_dir), "RosaHelper", "RosaHelper.py")
    spec = importlib.util.spec_from_file_location("_rosahelper_logic_bridge", helper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load Loader core bridge (`RosaHelperLogic`).")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _LOADER_CORE_INSTANCE = module.RosaHelperLogic()
    return _LOADER_CORE_INSTANCE

