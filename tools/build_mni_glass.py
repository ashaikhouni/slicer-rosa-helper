"""One-time build of the shared MNI glass brain for the cohort viewer.

Marching-cubes the CerebrA atlas envelope (`CerebrA.nii.gz > 0` = brain only, no
skull — cleaner than Otsu on the whole-head T1) in MNI152NLin2009cSym space, and
writes a translucent GLB the cohort page loads once as the pooled backdrop.
Vertices are MNI RAS mm — the same frame the warped contacts land in — so it
drops into the scene with no per-view transform.

Run from the repo root (needs the [mesh] extra for scikit-image):
    python tools/build_mni_glass.py
Commit the result: CommonLib/rosa_core/resources/atlases/templates/mni152_2009c_sym_glass.glb
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "CommonLib"))
sys.path.insert(0, str(ROOT / "cli"))

import numpy as np
import nibabel as nib
from scipy import ndimage

from rosa_core import brain_mesh
from rosa_agent.io.glb_writer import GLBScene, write_glb

RES = ROOT / "CommonLib" / "rosa_core" / "resources" / "atlases"
CEREBRA = RES / "cerebra" / "CerebrA.nii.gz"
OUT = RES / "templates" / "mni152_2009c_sym_glass.glb"


def main() -> int:
    img = nib.load(str(CEREBRA))
    data = np.asarray(img.dataobj)
    mask = data > 0                                   # brain envelope (all labels)
    mask = ndimage.binary_closing(mask, iterations=2)  # bridge label seams
    mask = ndimage.binary_fill_holes(mask)             # solid envelope (fill ventricles)
    mimg = nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header)

    surf = brain_mesh.surface_from_mask(
        mimg, smooth_sigma=1.6, step_size=2, taubin_iterations=15, largest_component=True)

    scene = GLBScene()
    glass = scene.add_material(
        "mni_glass", (0.60, 0.70, 0.82, 0.17),   # cool blue-grey, ~17% — a coverage backdrop
        metallic=0.0, roughness=0.9, double_sided=True, alpha_mode="BLEND")
    scene.add_surface("mni_brain", surf.vertices_ras.astype(np.float32), surf.faces, glass,
                      normals=surf.vertex_normals)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = write_glb(OUT, scene)
    print(f"[mni-glass] verts={len(surf.vertices_ras)} faces={len(surf.faces)} "
          f"bytes={n} → {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
