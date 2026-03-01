"""Publish helpers for modules writing into the shared ROSA workflow contract."""

from __main__ import slicer

from .workflow_registry import (
    ensure_image_registry_table,
    ensure_transform_registry_table,
    upsert_row,
)
from .workflow_state import WorkflowState


def _modality_for_node(node):
    """Infer coarse modality bucket from node class/name."""
    if node is None:
        return "OTHER"
    if node.IsA("vtkMRMLScalarVolumeNode") or node.IsA("vtkMRMLLabelMapVolumeNode"):
        name = (node.GetName() or "").lower()
        if "ct" in name:
            return "CT"
        if "t1" in name or "t2" in name or "mprage" in name or "mri" in name:
            return "MRI"
        return "OTHER"
    return "OTHER"


class WorkflowPublisher:
    """High-level publish operations for MRML workflow roles and registries."""

    def __init__(self, workflow_state=None):
        self.state = workflow_state or WorkflowState()

    def _ensure_subject_hierarchy_folder(self, parent_item_id, folder_name):
        """Create/reuse one subject-hierarchy folder under parent."""
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return 0
        if hasattr(sh_node, "GetItemChildWithName"):
            existing = sh_node.GetItemChildWithName(parent_item_id, folder_name)
            if existing:
                return existing
        return sh_node.CreateFolderItem(parent_item_id, folder_name)

    def _place_transform_in_hierarchy(self, transform_node):
        """Place transform node under `RosaWorkflow/Transforms` folder in subject hierarchy."""
        if transform_node is None:
            return
        sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        if sh_node is None:
            return
        scene_item = sh_node.GetSceneItemID()
        root = self._ensure_subject_hierarchy_folder(scene_item, "RosaWorkflow")
        transforms_root = self._ensure_subject_hierarchy_folder(root, "Transforms")
        item = sh_node.GetItemByDataNode(transform_node)
        if not item:
            try:
                item = sh_node.CreateItem(transforms_root, transform_node)
            except Exception:
                item = sh_node.GetItemByDataNode(transform_node)
        if item:
            sh_node.SetItemParent(item, transforms_root)

    def register_volume(
        self,
        volume_node,
        source_type,
        source_path="",
        space_name="ROSA_BASE",
        role=None,
        is_default_base=False,
        is_default_postop=False,
        is_derived=False,
        derived_from_node_id="",
        series_uid="",
        workflow_node=None,
    ):
        """Register one volume in image registry and optional role(s)."""
        if volume_node is None:
            return None
        wf = workflow_node or self.state.resolve_or_create_workflow_node()
        image_table = ensure_image_registry_table(wf)
        source_path = source_path or ""
        signature = self.state.image_signature(source_path=source_path, space_name=space_name)

        # Reuse policy: if same source+space already exists, prefer that node.
        existing = self.state.find_registered_image_by_signature(signature, workflow_node=wf)
        if existing is not None and existing.GetID() != volume_node.GetID():
            volume_node = existing

        parent_tfm = volume_node.GetParentTransformNode()
        row = {
            "node_id": volume_node.GetID(),
            "label": volume_node.GetName() or "",
            "modality": _modality_for_node(volume_node),
            "source_type": source_type or "",
            "source_path": source_path,
            "space_name": space_name or "",
            "is_default_base": "1" if is_default_base else "0",
            "is_default_postop_ct": "1" if is_default_postop else "0",
            "series_uid": series_uid or "",
            "parent_transform_id": "" if parent_tfm is None else parent_tfm.GetID(),
            "is_derived": "1" if is_derived else "0",
            "derived_from_node_id": derived_from_node_id or "",
            "signature": signature,
        }
        upsert_row(image_table, row, key_column="node_id")
        if is_default_base or is_default_postop:
            table = image_table.GetTable()
            for r in range(table.GetNumberOfRows()):
                node_id = str(table.GetValue(r, 0))
                if is_default_base:
                    table.SetValue(r, 6, "1" if node_id == volume_node.GetID() else "0")
                if is_default_postop:
                    table.SetValue(r, 7, "1" if node_id == volume_node.GetID() else "0")
            table.Modified()
            image_table.Modified()
        self.state.tag_node(
            volume_node,
            role=role or "Volume",
            source=source_type,
            space=space_name,
            signature=signature,
            workflow_node=wf,
        )
        if role:
            self.state.add_role_node(role, volume_node, workflow_node=wf)
        if is_default_base:
            self.set_default_role("BaseVolume", volume_node, workflow_node=wf)
        if is_default_postop:
            self.set_default_role("PostopCT", volume_node, workflow_node=wf)
        return volume_node

    def register_transform(
        self,
        transform_node,
        from_space,
        to_space,
        transform_type="linear",
        status="active",
        quality_metric="",
        role=None,
        workflow_node=None,
    ):
        """Register one transform in transform registry and optional role."""
        if transform_node is None:
            return
        wf = workflow_node or self.state.resolve_or_create_workflow_node()
        transform_table = ensure_transform_registry_table(wf)
        row = {
            "transform_node_id": transform_node.GetID(),
            "from_space": from_space or "",
            "to_space": to_space or "",
            "transform_type": transform_type or "",
            "quality_metric": str(quality_metric or ""),
            "status": status or "",
        }
        upsert_row(transform_table, row, key_column="transform_node_id")
        self.state.tag_node(
            transform_node,
            role=role or "Transform",
            source="transform",
            space=f"{from_space}->{to_space}",
            signature=transform_node.GetID(),
            workflow_node=wf,
        )
        if role:
            self.state.set_single_role(role, transform_node, workflow_node=wf)
        self._place_transform_in_hierarchy(transform_node)

    def publish_nodes(self, role, nodes, source="", space_name="", workflow_node=None):
        """Publish multiple existing nodes under one role with provenance tags."""
        wf = workflow_node or self.state.resolve_or_create_workflow_node()
        self.state.set_role_nodes(role, nodes, workflow_node=wf)
        for node in nodes or []:
            if node is None:
                continue
            self.state.tag_node(
                node=node,
                role=role,
                source=source,
                space=space_name,
                signature=node.GetID(),
                workflow_node=wf,
            )

    def publish_artifact(self, role, node_or_ids, source="", space_name="", workflow_node=None):
        """Publish one or many nodes under a workflow role.

        Parameters accept a single node/ID or a list/tuple/set of nodes/IDs.
        """
        wf = workflow_node or self.state.resolve_or_create_workflow_node()
        if node_or_ids is None:
            self.state.clear_role(role, workflow_node=wf)
            return
        values = node_or_ids
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        nodes = []
        for value in values:
            node = value
            if isinstance(value, str):
                node = slicer.mrmlScene.GetNodeByID(value)
            if node is not None:
                nodes.append(node)
        self.publish_nodes(
            role=role,
            nodes=nodes,
            source=source,
            space_name=space_name,
            workflow_node=wf,
        )

    def set_default_role(self, role, node_or_id, workflow_node=None):
        """Set one default-role node (`BaseVolume` or `PostopCT`) and sync image registry flags."""
        wf = workflow_node or self.state.resolve_or_create_workflow_node()
        node = node_or_id
        if isinstance(node_or_id, str):
            node = slicer.mrmlScene.GetNodeByID(node_or_id)
        if node is None:
            self.state.set_single_role(role, None, workflow_node=wf)
            return
        self.state.set_single_role(role, node, workflow_node=wf)
        image_table = wf.GetNodeReference("ImageRegistryTable")
        if image_table is None:
            return
        table = image_table.GetTable()
        target_id = node.GetID()
        for row in range(table.GetNumberOfRows()):
            value = table.GetValue(row, 0)
            node_id = str(value.ToString()) if hasattr(value, "ToString") else str(value)
            if role == "BaseVolume":
                table.SetValue(row, 6, "1" if node_id == target_id else "0")
            elif role == "PostopCT":
                table.SetValue(row, 7, "1" if node_id == target_id else "0")
        table.Modified()
        image_table.Modified()

    def set_default_base_volume(self, volume_node, workflow_node=None):
        """Set default base volume role and update image registry flags."""
        self.set_default_role("BaseVolume", volume_node, workflow_node=workflow_node)


def register_volume(node, metadata=None, workflow_node=None, publisher=None):
    """Function-style wrapper for volume registration API."""
    pub = publisher or WorkflowPublisher()
    meta = metadata or {}
    return pub.register_volume(
        volume_node=node,
        source_type=meta.get("source_type", "import"),
        source_path=meta.get("source_path", ""),
        space_name=meta.get("space_name", "ROSA_BASE"),
        role=meta.get("role", None),
        is_default_base=bool(meta.get("is_default_base", False)),
        is_default_postop=bool(meta.get("is_default_postop", False)),
        is_derived=bool(meta.get("is_derived", False)),
        derived_from_node_id=meta.get("derived_from_node_id", ""),
        series_uid=meta.get("series_uid", ""),
        workflow_node=workflow_node,
    )


def set_default_role(role, node_or_id, workflow_node=None, publisher=None):
    """Function-style wrapper for setting workflow default role."""
    pub = publisher or WorkflowPublisher()
    pub.set_default_role(role=role, node_or_id=node_or_id, workflow_node=workflow_node)


def publish_artifact(role, node_or_ids, source="", space_name="", workflow_node=None, publisher=None):
    """Function-style wrapper for publishing role artifacts."""
    pub = publisher or WorkflowPublisher()
    pub.publish_artifact(
        role=role,
        node_or_ids=node_or_ids,
        source=source,
        space_name=space_name,
        workflow_node=workflow_node,
    )
