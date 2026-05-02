"""Slicer ↔ SimpleITK adapter — the boundary between volume nodes and rosa_detect.

The detection algorithm package (``rosa_detect``) takes plain inputs:
SITK images plus 4×4 numpy IJK↔RAS matrices. Slicer modules carry
``vtkMRMLScalarVolumeNode`` instances that wrap both the voxel array
and a (potentially Slicer-transformed) IJK→RAS matrix. The bridge
between the two used to live in ``rosa_detect.service`` — but that put
Slicer / VTK concepts inside the supposedly headless algorithm package
(``rosa_detect/__init__.py`` claims "no Slicer / VTK / Qt deps", and
the lazy ``from __main__ import vtk`` inside service.py made that a
half-truth).

This module is the proper home: a Slicer-side adapter that any
Slicer-coupled caller imports to translate volume nodes into the plain
inputs ``rosa_detect.service.run_contact_pitch_v1`` actually needs.

Public API:

    vtk_matrix4x4_to_numpy(m_vtk) -> ndarray
    volume_node_geometry(volume_node) -> (ijk_to_ras, ras_to_ijk)
    image_from_volume_node(volume_node) -> (sitk_image, ijk_to_ras, ras_to_ijk)
    prepare_detection_context(volume_node, *, run_id=None, config=None,
                              extras=None) -> DetectionContext
"""

from __future__ import annotations

from typing import Any

import numpy as np

from __main__ import slicer, vtk

from rosa_detect.service import stamp_ijk_to_ras_on_sitk


def vtk_matrix4x4_to_numpy(m_vtk) -> np.ndarray:
    """Convert a vtkMatrix4x4 (or any object with ``GetElement(r, c)``)
    to a numpy 4×4 float matrix.
    """
    mat = np.eye(4, dtype=float)
    for r in range(4):
        for c in range(4):
            mat[r, c] = float(m_vtk.GetElement(r, c))
    return mat


def volume_node_geometry(volume_node) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(ijk_to_ras_4x4, ras_to_ijk_4x4)`` for a Slicer volume node.

    Single source of truth for "extract the IJK↔RAS matrices from a
    volume node" — Slicer modules used to roll their own copy of this
    pattern; everyone goes through this helper now so a future change
    in matrix source (e.g. honoring the parent transform stack) lands
    in one place.
    """
    m_ijk = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(m_ijk)
    m_ras = vtk.vtkMatrix4x4()
    volume_node.GetRASToIJKMatrix(m_ras)
    return vtk_matrix4x4_to_numpy(m_ijk), vtk_matrix4x4_to_numpy(m_ras)


def image_from_volume_node(volume_node) -> tuple[Any, np.ndarray, np.ndarray]:
    """Build a SITK image from a Slicer volume node, with the node's
    IJK→RAS matrix stamped into origin/direction (LPS-flipped via
    ``rosa_detect.service.stamp_ijk_to_ras_on_sitk``).

    Returns ``(sitk_image, ijk_to_ras, ras_to_ijk)`` ready to feed into
    ``rosa_detect.guided_fit_engine.compute_features`` or pack into a
    ``DetectionContext`` for ``run_contact_pitch_v1`` (use
    ``prepare_detection_context`` for the latter).
    """
    import SimpleITK as sitk

    arr_kji = np.asarray(
        slicer.util.arrayFromVolume(volume_node), dtype=np.float32,
    )
    img = sitk.GetImageFromArray(arr_kji)
    img.SetSpacing(tuple(float(v) for v in volume_node.GetSpacing()))
    ijk_to_ras, ras_to_ijk = volume_node_geometry(volume_node)
    stamp_ijk_to_ras_on_sitk(img, ijk_to_ras)
    return img, ijk_to_ras, ras_to_ijk


def prepare_detection_context(
    volume_node,
    *,
    run_id: str | None = None,
    config: dict | None = None,
    extras: dict | None = None,
) -> dict[str, Any]:
    """Build a complete ``DetectionContext`` for ``run_contact_pitch_v1``
    from a Slicer volume node.

    This is the recommended entry point Slicer modules should use —
    pre-builds the SITK image + matrices via ``image_from_volume_node``
    and packs them into the ctx as the ``img`` / ``ijk_to_ras_4x4`` /
    ``ras_to_ijk_4x4`` keys ``rosa_detect.service.load_image_and_matrices``
    consumes directly. Keeps the algorithm package free of any
    "volume node" indirection.

    When the volume has a storage node pointing at a readable NIfTI,
    also populate ``ctx['ct'].path`` (a SimpleNamespace shim with a
    ``path`` attribute matching ``VolumeRef``'s shape) so
    diagnostics / fingerprint logging can show the source path.
    """
    import os
    from types import SimpleNamespace

    img, ijk_to_ras, ras_to_ijk = image_from_volume_node(volume_node)

    ct_ref = None
    try:
        storage = volume_node.GetStorageNode()
        src = storage.GetFileName() if storage is not None else ""
        if src and os.path.exists(src):
            ct_ref = SimpleNamespace(
                volume_id=volume_node.GetName(), path=str(src),
            )
    except Exception:
        ct_ref = None

    ctx: dict[str, Any] = {
        "run_id": run_id or f"contact_pitch_{volume_node.GetName()}",
        "img": img,
        "ijk_to_ras_4x4": ijk_to_ras,
        "ras_to_ijk_4x4": ras_to_ijk,
        "config": dict(config or {}),
        "extras": dict(extras or {}),
    }
    if ct_ref is not None:
        ctx["ct"] = ct_ref
    return ctx
