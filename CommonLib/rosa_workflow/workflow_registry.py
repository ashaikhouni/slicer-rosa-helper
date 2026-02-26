"""Registry table helpers used by the shared ROSA workflow contract."""

from __main__ import slicer, vtk

IMAGE_REGISTRY_COLUMNS = [
    "node_id",
    "label",
    "modality",
    "source_type",
    "source_path",
    "space_name",
    "is_default_base",
    "is_default_postop_ct",
    "series_uid",
    "parent_transform_id",
    "is_derived",
    "derived_from_node_id",
    "signature",
]

TRANSFORM_REGISTRY_COLUMNS = [
    "transform_node_id",
    "from_space",
    "to_space",
    "transform_type",
    "quality_metric",
    "status",
]


def _normalize_text(value):
    """Return safe string representation for table storage."""
    if value is None:
        return ""
    return str(value)


def _ensure_table_columns(table_node, columns):
    """Ensure table node has exact named string columns in order."""
    table = table_node.GetTable()
    existing = [table.GetColumnName(i) for i in range(table.GetNumberOfColumns())]
    if existing == list(columns):
        return

    # Rebuild to keep schema deterministic and simple.
    while table.GetNumberOfColumns() > 0:
        table.RemoveColumn(0)
    for name in columns:
        arr = vtk.vtkStringArray()
        arr.SetName(name)
        table.AddColumn(arr)


def ensure_table_node(workflow_node, role_name, node_name, columns):
    """Return table node bound to role, creating one when missing."""
    table_node = workflow_node.GetNodeReference(role_name)
    if table_node is None:
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", node_name)
        workflow_node.SetNodeReferenceID(role_name, table_node.GetID())
    _ensure_table_columns(table_node, columns)
    return table_node


def ensure_image_registry_table(workflow_node):
    """Create or return image registry table node."""
    return ensure_table_node(
        workflow_node=workflow_node,
        role_name="ImageRegistryTable",
        node_name="RosaWorkflow_ImageRegistry",
        columns=IMAGE_REGISTRY_COLUMNS,
    )


def ensure_transform_registry_table(workflow_node):
    """Create or return transform registry table node."""
    return ensure_table_node(
        workflow_node=workflow_node,
        role_name="TransformRegistryTable",
        node_name="RosaWorkflow_TransformRegistry",
        columns=TRANSFORM_REGISTRY_COLUMNS,
    )


def find_row_by_value(table_node, key_column, key_value):
    """Return row index matching one column value, else -1."""
    table = table_node.GetTable()
    try:
        key_col = IMAGE_REGISTRY_COLUMNS.index(key_column)
    except ValueError:
        try:
            key_col = TRANSFORM_REGISTRY_COLUMNS.index(key_column)
        except ValueError:
            key_col = table.GetColumnByName(key_column)
            if key_col is None:
                return -1
    if isinstance(key_col, int):
        col_obj = table.GetColumn(key_col)
    else:
        col_obj = key_col
    if col_obj is None:
        return -1
    want = _normalize_text(key_value)
    for row in range(table.GetNumberOfRows()):
        if _normalize_text(col_obj.GetValue(row)) == want:
            return row
    return -1


def upsert_row(table_node, data, key_column):
    """Insert/update one row by key column value and return row index."""
    table = table_node.GetTable()
    row = find_row_by_value(table_node, key_column, data.get(key_column, ""))
    if row < 0:
        row = table.GetNumberOfRows()
        table.InsertNextBlankRow()
    for col in range(table.GetNumberOfColumns()):
        name = table.GetColumnName(col)
        table.SetValue(row, col, _normalize_text(data.get(name, "")))
    table.Modified()
    table_node.Modified()
    return row


def table_to_dict_rows(table_node):
    """Return list of row dictionaries for easier filtering."""
    if table_node is None:
        return []
    table = table_node.GetTable()
    columns = [table.GetColumnName(i) for i in range(table.GetNumberOfColumns())]
    rows = []
    for r in range(table.GetNumberOfRows()):
        rows.append({name: str(table.GetValue(r, c)) for c, name in enumerate(columns)})
    return rows

