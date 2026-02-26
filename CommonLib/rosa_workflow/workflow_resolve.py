"""Input resolution helpers for module interoperability via workflow contract."""

from __main__ import slicer

from .workflow_state import WorkflowState
from .workflow_registry import table_to_dict_rows


def resolve_or_create_workflow(context_hint=None):
    """Convenience wrapper to get shared workflow node."""
    return WorkflowState().resolve_or_create_workflow_node(context_hint=context_hint)


def resolve_role_nodes(role, workflow_node=None):
    """Return role-bound nodes from workflow or empty list."""
    state = WorkflowState()
    wf = workflow_node or state.resolve_or_create_workflow_node()
    return state.role_nodes(role, workflow_node=wf)


def resolve_default_volume(modality=None, prefer_postop=False, workflow_node=None):
    """Resolve best volume node using workflow defaults then registry fallback."""
    state = WorkflowState()
    wf = workflow_node or state.resolve_or_create_workflow_node()
    if prefer_postop:
        node = wf.GetNodeReference("PostopCT")
        if node is not None:
            return node
    node = wf.GetNodeReference("BaseVolume")
    if node is not None and (modality is None or modality.upper() == "MRI"):
        return node

    table = wf.GetNodeReference("ImageRegistryTable")
    rows = table_to_dict_rows(table) if table is not None else []
    target_modality = (modality or "").upper().strip()
    for row in reversed(rows):
        if target_modality and (row.get("modality", "").upper() != target_modality):
            continue
        node = slicer.mrmlScene.GetNodeByID(row.get("node_id", ""))
        if node is not None:
            return node
    return None


def resolve_module_inputs(spec, workflow_node=None):
    """Resolve a module input spec against workflow roles/defaults.

    `spec` format:
      {
        "<output_key>": {"role": "RoleName", "mode": "single|many", "required": bool},
        "<output_key2>": {"default_volume": {"modality": "CT|MRI|OTHER", "prefer_postop": bool}}
      }
    """
    state = WorkflowState()
    wf = workflow_node or state.resolve_or_create_workflow_node()
    resolved = {}
    for key, rule in (spec or {}).items():
        rule = rule or {}
        if "role" in rule:
            role = str(rule.get("role", ""))
            mode = str(rule.get("mode", "single")).lower()
            nodes = state.role_nodes(role, workflow_node=wf)
            value = nodes if mode == "many" else (nodes[0] if nodes else None)
            if rule.get("required", False) and ((mode == "many" and not nodes) or (mode != "many" and value is None)):
                raise RuntimeError(f"Missing required workflow role: {role}")
            resolved[key] = value
            continue
        if "default_volume" in rule:
            opts = rule.get("default_volume", {}) or {}
            node = resolve_default_volume(
                modality=opts.get("modality", None),
                prefer_postop=bool(opts.get("prefer_postop", False)),
                workflow_node=wf,
            )
            if rule.get("required", False) and node is None:
                raise RuntimeError(f"Missing required default volume for '{key}'")
            resolved[key] = node
            continue
        resolved[key] = None
    return resolved
