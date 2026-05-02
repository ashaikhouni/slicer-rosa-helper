"""rosa_core public API.

Lazy re-exports (PEP 562). Importing this package runs only this file;
heavy submodules (numpy/scipy/SimpleITK consumers — ``contact_fit``,
``contact_peak_fit``, ``case_loader``, ``qc``, ``electrode_models``)
are imported on first attribute access via ``__getattr__``.

Why: callers reaching pure-Python policy modules (e.g.
``from rosa_core.atlas_assignment_policy import ...``) would otherwise
pull NumPy as a side effect of the package init, breaking minimal
environments and pure-Python tests. See the regression test in
``tests/rosa_core/test_lazy_init.py``.
"""

from __future__ import annotations

# Map public attribute name -> (submodule, attribute). Submodules
# import their own heavy deps lazily, so adding entries here is free.
# Listed alphabetically by attribute name.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # transforms — pure Python.
    "apply_affine":                                       ("transforms", "apply_affine"),
    "invert_4x4":                                         ("transforms", "invert_4x4"),
    "is_identity_4x4":                                    ("transforms", "is_identity_4x4"),
    "lps_to_ras_matrix":                                  ("transforms", "lps_to_ras_matrix"),
    "lps_to_ras_point":                                   ("transforms", "lps_to_ras_point"),
    "to_itk_affine_text":                                 ("transforms", "to_itk_affine_text"),
    # ros_parser — pure Python.
    "parse_ros_file":                                     ("ros_parser", "parse_ros_file"),
    # case_loader — most helpers are pure-Python; load_rosa_volume_as_sitk
    # pulls SITK lazily inside its body.
    "build_effective_matrices":                           ("case_loader", "build_effective_matrices"),
    "choose_reference_volume":                            ("case_loader", "choose_reference_volume"),
    "find_ros_file":                                      ("case_loader", "find_ros_file"),
    "load_rosa_volume_as_sitk":                           ("case_loader", "load_rosa_volume_as_sitk"),
    "resolve_analyze_volume":                             ("case_loader", "resolve_analyze_volume"),
    "resolve_reference_index":                            ("case_loader", "resolve_reference_index"),
    # assignments — light.
    "electrode_length_mm":                                ("assignments", "electrode_length_mm"),
    "suggest_model_id_for_trajectory":                    ("assignments", "suggest_model_id_for_trajectory"),
    "trajectory_length_mm":                               ("assignments", "trajectory_length_mm"),
    # contacts — light.
    "build_assignment_template":                          ("contacts", "build_assignment_template"),
    "contacts_to_fcsv_rows":                              ("contacts", "contacts_to_fcsv_rows"),
    "generate_contacts":                                  ("contacts", "generate_contacts"),
    "load_assignments":                                   ("contacts", "load_assignments"),
    "save_assignment_template":                           ("contacts", "save_assignment_template"),
    "save_contacts_markups_json":                         ("contacts", "save_contacts_markups_json"),
    "save_contacts_rosa_json":                            ("contacts", "save_contacts_rosa_json"),
    # contact_fit — numpy.
    "fit_electrode_axis_and_tip":                         ("contact_fit", "fit_electrode_axis_and_tip"),
    "refine_fit_batch_with_exclusive_terminal_assignment": ("contact_fit", "refine_fit_batch_with_exclusive_terminal_assignment"),
    # contact_peak_fit — numpy + SimpleITK (LoG helper).
    "PeakFitResult":                                      ("contact_peak_fit", "PeakFitResult"),
    "candidate_ids_for_vendors":                          ("contact_peak_fit", "candidate_ids_for_vendors"),
    "detect_contacts_on_axis":                            ("contact_peak_fit", "detect_contacts_on_axis"),
    "detect_peaks_1d":                                    ("contact_peak_fit", "detect_peaks_1d"),
    "fit_best_electrode":                                 ("contact_peak_fit", "fit_best_electrode"),
    "ras_contacts_to_contact_records":                    ("contact_peak_fit", "ras_contacts_to_contact_records"),
    "sample_axis_profile":                                ("contact_peak_fit", "sample_axis_profile"),
    # curry_export — light.
    "contacts_to_pom_points":                             ("curry_export", "contacts_to_pom_points"),
    "trajectory_endpoints_to_pom_points":                 ("curry_export", "trajectory_endpoints_to_pom_points"),
    "write_curry_pom":                                    ("curry_export", "write_curry_pom"),
    # electrode_models — light (pure Python).
    "default_electrode_library_path":                     ("electrode_models", "default_electrode_library_path"),
    "load_electrode_library":                             ("electrode_models", "load_electrode_library"),
    "model_map":                                          ("electrode_models", "model_map"),
    # exporters — light.
    "build_fcsv_rows":                                    ("exporters", "build_fcsv_rows"),
    "build_markups_lines":                                ("exporters", "build_markups_lines"),
    "build_markups_document":                             ("exporters", "build_markups_document"),
    # qc — numpy.
    "compute_qc_metrics":                                 ("qc", "compute_qc_metrics"),
    # types — pure Python.
    "AssignmentRow":                                      ("types", "AssignmentRow"),
    "AssignmentTemplate":                                 ("types", "AssignmentTemplate"),
    "ContactFitResult":                                   ("types", "ContactFitResult"),
    "ContactRecord":                                      ("types", "ContactRecord"),
    "DisplayRecord":                                      ("types", "DisplayRecord"),
    "ElectrodeLibrary":                                   ("types", "ElectrodeLibrary"),
    "ElectrodeModel":                                     ("types", "ElectrodeModel"),
    "FCSVRow":                                            ("types", "FCSVRow"),
    "Matrix4x4":                                          ("types", "Matrix4x4"),
    "Point3D":                                            ("types", "Point3D"),
    "QCMetricsRow":                                       ("types", "QCMetricsRow"),
    "RosParseResult":                                     ("types", "RosParseResult"),
    "TokenBlock":                                         ("types", "TokenBlock"),
    "TrajectoryRecord":                                   ("types", "TrajectoryRecord"),
}

__all__ = sorted(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module 'rosa_core' has no attribute {name!r}")
    module_name, attr = spec
    import importlib
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attr)
    # Cache so subsequent lookups skip __getattr__.
    globals()[name] = value
    return value


def __dir__():
    return list(__all__)
