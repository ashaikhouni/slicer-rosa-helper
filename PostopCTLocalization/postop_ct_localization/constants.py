GUIDED_SOURCE_OPTIONS = [
    ("working", "Working (active)"),
    ("imported_rosa", "Imported ROSA"),
    ("imported_external", "Imported External"),
    ("manual", "Manual (scene)"),
    ("guided_fit", "Guided Fit"),
    ("deep_core", "Deep Core"),
    ("de_novo", "De Novo"),
    ("planned_rosa", "Planned ROSA"),
]

DE_NOVO_MODE_SPECS = (
    {
        "pipeline_key": "blob_ransac_v1",
        "tab_label": "Voxel Fit",
        "show_blob_controls": False,
    },
    {
        "pipeline_key": "blob_em_v2",
        "tab_label": "Blob Fit",
        "show_blob_controls": True,
    },
)
