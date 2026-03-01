"""Small reusable MRML scene utility helpers."""

from __future__ import annotations

from __main__ import slicer


def find_node_by_name(node_name: str, class_name: str):
    """Return first node with exact name and class, or ``None``."""
    for node in slicer.util.getNodesByClass(class_name):
        if node.GetName() == node_name:
            return node
    return None


def widget_current_text(widget) -> str:
    """Return `currentText` value for Qt widgets that expose it."""
    attr = getattr(widget, "currentText", "")
    value = attr() if callable(attr) else attr
    return str(value) if value is not None else ""


def get_or_create_linear_transform(name: str):
    """Return a named linear transform node, creating one if needed."""
    node = find_node_by_name(name, "vtkMRMLLinearTransformNode")
    if node is None:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
    return node


def preselect_base_volume(selector, workflow_node) -> None:
    """Set selector to workflow BaseVolume when available."""
    if selector is None or workflow_node is None:
        return
    base = workflow_node.GetNodeReference("BaseVolume")
    if base is not None and hasattr(selector, "setCurrentNode"):
        selector.setCurrentNode(base)
