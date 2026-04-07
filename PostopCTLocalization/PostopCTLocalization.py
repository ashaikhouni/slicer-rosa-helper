"""3D Slicer module for postop CT localization workflows.

This file is intentionally thin: Slicer entrypoint + composition of workflow
modules. Workflow-specific UI and logic live under `postop_ct_localization/`.
"""

import importlib
import os
import sys

from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CANDIDATES = [
    MODULE_DIR,
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),
    os.path.join(MODULE_DIR, "CommonLib"),
]
for path in PATH_CANDIDATES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _reload_local_support_modules():
    """Hot-reload local helper modules when the scripted module is reimported."""
    importlib.invalidate_caches()
    roots = tuple(os.path.abspath(path) for path in PATH_CANDIDATES if os.path.isdir(path))
    prefixes = ("rosa_core", "rosa_scene", "shank_engine", "rosa_workflow", "postop_ct_localization")
    to_reload = []
    for name, module in list(sys.modules.items()):
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue
        module_path = os.path.abspath(module_file)
        if any(module_path.startswith(root + os.sep) or module_path == root for root in roots):
            to_reload.append((name.count("."), name, module))
    for _, _, module in sorted(to_reload, key=lambda item: item[0], reverse=True):
        importlib.reload(module)


_reload_local_support_modules()

from postop_ct_localization.deep_core_debug import DeepCoreDebugLogicMixin, DeepCoreDebugWidgetMixin
from postop_ct_localization.de_novo import DeNovoLogicMixin, DeNovoWidgetMixin
from postop_ct_localization.gui import PostopCTLocalizationWidgetBaseMixin
from postop_ct_localization.guided_fit import GuidedFitWidgetMixin
from postop_ct_localization.logic_common import PostopCTLocalizationLogicBaseMixin
from postop_ct_localization.manual_fit import ManualFitWidgetMixin


class PostopCTLocalization(ScriptedLoadableModule):
    """Slicer metadata for unified postop CT localization module."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "01 Postop CT Localization"
        self.parent.categories = ["ROSA.02 Localization"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Unified postop CT localization: Guided Fit (planned trajectories) and "
            "De Novo Detect (CT-only)."
        )


class PostopCTLocalizationWidget(
    PostopCTLocalizationWidgetBaseMixin,
    GuidedFitWidgetMixin,
    DeepCoreDebugWidgetMixin,
    DeNovoWidgetMixin,
    ManualFitWidgetMixin,
    ScriptedLoadableModuleWidget,
):
    """Widget exposing guided-fit, deep-core debug, manual, and de novo workflows."""

    def _create_logic(self):
        return PostopCTLocalizationLogic()


class PostopCTLocalizationLogic(
    PostopCTLocalizationLogicBaseMixin,
    DeepCoreDebugLogicMixin,
    DeNovoLogicMixin,
    ScriptedLoadableModuleLogic,
):
    """Logic composed from workflow-specific mixins."""

    pass
