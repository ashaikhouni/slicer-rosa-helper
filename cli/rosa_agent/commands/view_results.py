"""rosa-agent view-results — visualize ALREADY-COMPUTED results, no pipeline re-run.

Unlike ``export-view`` (which runs detection + placement on a raw case),
``view-results`` takes a result set you already have — ``contacts.tsv`` (+
``trajectories.tsv``) from ``fit-rosa`` / ``place`` / ``pipeline``, or
hand-edited contacts — plus the CT they live in, and renders them into the same
browser viewer in seconds. Open the output with ``rosa-agent view`` or drop it
into the GitHub Pages viewer.

Two ways to point it at the data:

* **Results directory** (positional) — auto-scans for ``contacts.tsv``,
  ``trajectories.tsv`` and a CT (``work/postop_ct.nii.gz`` → ``ct.nii.gz`` →
  any ``*.nii.gz``). Ideal for a ``fit-rosa`` output dir::

      rosa-agent view-results CASE/fitrosa_qc -o /tmp/view

* **Explicit paths** — override any of them::

      rosa-agent view-results --ct ct.nii.gz --contacts c.tsv \\
          --trajectories t.tsv -o /tmp/view

FreeSurfer is optional (``--freesurfer-dir`` adds the brain mesh). The contacts /
trajectories must already be in the CT's RAS frame — this command does not
register or re-detect; it displays what you give it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _pick(explicit: str, results_dir: Path | None, *candidates: str) -> Path | None:
    if explicit:
        return Path(explicit)
    if results_dir is not None:
        for c in candidates:
            p = results_dir / c
            if p.exists():
                return p
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent view-results",
        description="Render an existing result set (CT + contacts/trajectories) "
                    "into the browser viewer — no detection re-run.",
    )
    parser.add_argument(
        "results_dir", nargs="?", default="",
        help="results directory to auto-scan (contacts.tsv, trajectories.tsv, "
             "and a CT under work/postop_ct.nii.gz / ct.nii.gz). Optional when "
             "the explicit flags are given.",
    )
    parser.add_argument("--output", "-o", required=True, help="output directory")
    parser.add_argument("--ct", default="", help="CT NIfTI (overrides auto-scan)")
    parser.add_argument("--contacts", default="", help="contacts.tsv (overrides auto-scan)")
    parser.add_argument("--trajectories", default="", help="trajectories.tsv (overrides auto-scan)")
    parser.add_argument("--labels", default="", help="optional labels.tsv for sidebar atlas labels")
    parser.add_argument(
        "--freesurfer-dir", default="",
        help="optional FreeSurfer recon dir — adds the brain mesh (registered to the CT)",
    )
    parser.add_argument(
        "--brain-volume", default="",
        help="optional volume ALREADY in the CT frame (an MRI resampled to CT) — "
             "brain-extracted + marching-cubed into the subject's own translucent "
             "brain surface. Ignored if --freesurfer-dir is given.",
    )
    parser.add_argument(
        "--brain-mask-cache", default="",
        help="path to cache the brain mask; if it exists it's reused (skip "
             "SynthStrip), else --brain-volume is extracted and saved here — so "
             "the mask is computed once per case, not per label.",
    )
    parser.add_argument(
        "--brain-gyri", action="store_true",
        help="mesh the MRI's gray/white matter (Otsu, no FreeSurfer) so the "
             "brain surface shows gyri instead of a smooth envelope — heavier "
             "mesh (~100k+ verts). Requires --brain-volume.",
    )
    parser.add_argument(
        "--brain-native-volume", default="",
        help="the NATIVE MRI (1mm, not resampled to CT). Meshes gyri in the "
             "native frame (clean, isotropic) and transforms the surface into "
             "CT — much cleaner than meshing the CT-resampled MRI. Preferred "
             "over --brain-volume; pair with --brain-to-ct-transform.",
    )
    parser.add_argument(
        "--brain-to-ct-transform", default="",
        help="cached MRI→CT rigid transform (t1_to_ct.tfm from the label step). "
             "Reused to push the native brain surface into CT; if absent, "
             "view-results registers the native MRI to the CT itself.",
    )
    parser.add_argument(
        "--brain-surface-cache", default="",
        help="path to cache the built brain surface (.npz). The surface is "
             "atlas-independent, so it's meshed once per case and reused — "
             "labeling another atlas skips the re-mesh entirely.",
    )
    parser.add_argument("--ct-window", default="",
                        help="HU window 'lo,hi' for the CT volume (default '-150,1500')")
    parser.add_argument("--subject-label", default="", help="label shown in the viewer header")
    parser.add_argument("--shaft-radius-mm", type=float, default=0.35)
    parser.add_argument("--contact-band-radius-mm", type=float, default=0.55)
    parser.add_argument("--contact-band-length-mm", type=float, default=2.0)
    args = parser.parse_args(argv)

    try:
        from rosa_agent.commands.export_view import run_view_results, _parse_window
    except ImportError as exc:
        _stderr(f"error: rosa_core / rosa_agent unavailable ({exc})")
        return 2

    rd = Path(args.results_dir) if args.results_dir else None
    if rd is not None and not rd.is_dir():
        _stderr(f"error: results dir not found: {rd}")
        return 2

    contacts = _pick(args.contacts, rd, "contacts.tsv")
    trajectories = _pick(args.trajectories, rd, "trajectories.tsv")
    ct = _pick(args.ct, rd, "work/postop_ct.nii.gz", "ct.nii.gz")
    if ct is None and rd is not None:
        # Last resort: first *.nii.gz in the dir that isn't a viewer artifact.
        for p in sorted(rd.glob("*.nii.gz")):
            if p.name not in ("ct_in_view.nii.gz", "t1_in_ct.nii.gz"):
                ct = p
                break
    labels = _pick(args.labels, rd, "labels.tsv")

    if contacts is None:
        _stderr("error: no contacts.tsv (give --contacts or a results dir containing it)")
        return 2
    if not contacts.exists():
        _stderr(f"error: contacts not found: {contacts}")
        return 2
    if ct is None:
        _stderr("error: no CT found (give --ct, or a results dir with work/postop_ct.nii.gz / ct.nii.gz)")
        return 2
    if not ct.exists():
        _stderr(f"error: CT not found: {ct}")
        return 2

    try:
        run_view_results(
            args.output,
            ct_path=ct,
            contacts_tsv=contacts,
            trajectories_tsv=trajectories,
            labels_tsv=labels,
            freesurfer_dir=args.freesurfer_dir,
            brain_volume=args.brain_volume or None,
            brain_mask_cache=args.brain_mask_cache or None,
            brain_gyri=bool(args.brain_gyri),
            brain_native_volume=args.brain_native_volume or None,
            brain_to_ct_transform=args.brain_to_ct_transform or None,
            brain_surface_cache=args.brain_surface_cache or None,
            ct_window=_parse_window(args.ct_window),
            subject_label=args.subject_label,
            shaft_radius_mm=float(args.shaft_radius_mm),
            contact_band_radius_mm=float(args.contact_band_radius_mm),
            contact_band_length_mm=float(args.contact_band_length_mm),
        )
    except (FileNotFoundError, ValueError) as exc:
        _stderr(f"error: {exc}")
        return 2

    _stderr(f"[view-results] viewer ready: {args.output}")
    _stderr(f"[view-results] open with:  rosa-agent view {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
