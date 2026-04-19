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

from postop_ct_localization.deep_core_visualization import DeepCoreVisualizationLogicMixin
from postop_ct_localization.deep_core_widget import ContactPitchV1WidgetMixin
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
            "Unified postop CT localization: Auto Fit (contact_pitch_v1 "
            "CT-only detector), Guided Fit (planned trajectories), and "
            "Manual Fit."
        )


class PostopCTLocalizationWidget(
    PostopCTLocalizationWidgetBaseMixin,
    GuidedFitWidgetMixin,
    ContactPitchV1WidgetMixin,
    ManualFitWidgetMixin,
    ScriptedLoadableModuleWidget,
):
    """Widget exposing Auto Fit (contact_pitch_v1), Guided Fit, and Manual Fit."""

    def _create_logic(self):
        return PostopCTLocalizationLogic()


class PostopCTLocalizationLogic(
    PostopCTLocalizationLogicBaseMixin,
    DeepCoreVisualizationLogicMixin,
    ScriptedLoadableModuleLogic,
):
    """Logic composed from workflow-specific mixins."""

    pass
