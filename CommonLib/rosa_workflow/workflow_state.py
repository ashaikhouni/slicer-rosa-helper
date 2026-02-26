"""Shared workflow state node utilities."""

import hashlib
import os
import uuid
from datetime import datetime, timezone

from __main__ import slicer

from .workflow_registry import (
    ensure_image_registry_table,
    ensure_transform_registry_table,
    table_to_dict_rows,
)

WORKFLOW_NODE_NAME = "RosaWorkflow"
WORKFLOW_VERSION = "1.0"


def _utc_now_iso():
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()


def _hash_signature(*parts):
    """Create deterministic signature hash for provenance and dedup."""
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part or "").encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


class WorkflowState:
    """Small facade around `vtkMRMLScriptedModuleNode` workflow state."""

    def __init__(self, node_name=WORKFLOW_NODE_NAME):
        self.node_name = node_name

    def resolve_or_create_workflow_node(self, context_hint=None):
        """Get existing workflow node or create a new one."""
        node = None
        for candidate in slicer.util.getNodesByClass("vtkMRMLScriptedModuleNode"):
            if candidate.GetName() == self.node_name:
                node = candidate
                break
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScriptedModuleNode", self.node_name)
            node.SetModuleName("RosaWorkflow")
        if not node.GetParameter("ContextId"):
            node.SetParameter("ContextId", str(context_hint or uuid.uuid4()))
        if not node.GetParameter("WorkflowVersion"):
            node.SetParameter("WorkflowVersion", WORKFLOW_VERSION)
        if not node.GetParameter("BaseSpaceName"):
            node.SetParameter("BaseSpaceName", "ROSA_BASE")
        if not node.GetParameter("PreferredPrimaryAtlas"):
            node.SetParameter("PreferredPrimaryAtlas", "thomas_if_available")
        if not node.GetParameter("DefaultExportPrefix"):
            node.SetParameter("DefaultExportPrefix", "ROSA_Contacts")
        node.SetParameter("LastUpdatedUTC", _utc_now_iso())
        ensure_image_registry_table(node)
        ensure_transform_registry_table(node)
        return node

    def context_id(self, workflow_node=None):
        """Return context ID for current workflow."""
        node = workflow_node or self.resolve_or_create_workflow_node()
        return node.GetParameter("ContextId")

    def set_param(self, key, value, workflow_node=None):
        """Set workflow parameter."""
        node = workflow_node or self.resolve_or_create_workflow_node()
        node.SetParameter(str(key), str(value))
        node.SetParameter("LastUpdatedUTC", _utc_now_iso())

    def get_param(self, key, default="", workflow_node=None):
        """Get workflow parameter."""
        node = workflow_node or self.resolve_or_create_workflow_node()
        value = node.GetParameter(str(key))
        return default if value is None or value == "" else value

    def set_single_role(self, role, node, workflow_node=None):
        """Set single node reference role."""
        wf = workflow_node or self.resolve_or_create_workflow_node()
        node_id = node.GetID() if node is not None else None
        wf.SetNodeReferenceID(str(role), node_id)
        wf.SetParameter("LastUpdatedUTC", _utc_now_iso())

    def clear_role(self, role, workflow_node=None):
        """Remove all node references under a role."""
        wf = workflow_node or self.resolve_or_create_workflow_node()
        count = wf.GetNumberOfNodeReferences(role)
        for idx in range(count - 1, -1, -1):
            wf.RemoveNthNodeReferenceID(role, idx)
        wf.SetParameter("LastUpdatedUTC", _utc_now_iso())

    def add_role_node(self, role, node, workflow_node=None):
        """Append unique multi-reference node for a role."""
        if node is None:
            return
        wf = workflow_node or self.resolve_or_create_workflow_node()
        node_id = node.GetID()
        for idx in range(wf.GetNumberOfNodeReferences(role)):
            existing = wf.GetNthNodeReferenceID(role, idx)
            if existing == node_id:
                return
        wf.AddNodeReferenceID(role, node_id)
        wf.SetParameter("LastUpdatedUTC", _utc_now_iso())

    def set_role_nodes(self, role, nodes, workflow_node=None):
        """Replace a multi-reference role with provided nodes."""
        wf = workflow_node or self.resolve_or_create_workflow_node()
        self.clear_role(role, workflow_node=wf)
        for node in nodes or []:
            self.add_role_node(role, node, workflow_node=wf)
        wf.SetParameter("LastUpdatedUTC", _utc_now_iso())

    def role_nodes(self, role, workflow_node=None):
        """Return list of live MRML nodes in role."""
        wf = workflow_node or self.resolve_or_create_workflow_node()
        out = []
        for idx in range(wf.GetNumberOfNodeReferences(role)):
            node = wf.GetNthNodeReference(role, idx)
            if node is not None:
                out.append(node)
        return out

    def tag_node(self, node, role, source="", space="", signature="", workflow_node=None):
        """Set standard workflow provenance attributes on node."""
        if node is None:
            return
        wf = workflow_node or self.resolve_or_create_workflow_node()
        node.SetAttribute("Rosa.ContextId", self.context_id(workflow_node=wf))
        node.SetAttribute("Rosa.Role", str(role or ""))
        node.SetAttribute("Rosa.Source", str(source or ""))
        node.SetAttribute("Rosa.Space", str(space or ""))
        node.SetAttribute("Rosa.Signature", str(signature or ""))
        node.SetAttribute("Rosa.Managed", "1")

    def image_signature(self, source_path, space_name):
        """Build canonical signature used for image dedup policy."""
        path = os.path.abspath(source_path) if source_path else ""
        return _hash_signature(path, space_name)

    def find_registered_image_by_signature(self, signature, workflow_node=None):
        """Return matching volume node from image registry signature."""
        wf = workflow_node or self.resolve_or_create_workflow_node()
        table = wf.GetNodeReference("ImageRegistryTable")
        if table is None:
            return None
        for row in table_to_dict_rows(table):
            if str(row.get("signature", "")) != str(signature):
                continue
            node_id = row.get("node_id", "")
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                return node
        return None

