"""rosa-agent stamp-mni — make a case MNI-ready in one shot.

Given ``<case_dir>`` (contacts.tsv + regcache/) and optionally ``--t1`` (the
patient MRI), this:

  1. ensures the T1→MNI transform is cached (registers if ``--t1`` is given and
     the transform is missing — the pipeline already produced the CT→T1 leg),
  2. warps contacts CT→MNI  → ``regcache/contacts_mni.tsv``,
  3. samples every bundled MNI-native atlas (CerebrA + Iglesias thalamus) at each
     MNI coordinate → ``regcache/contacts_labels_mni.tsv``.

Self-gates to a no-op when the case isn't MNI-poolable (no CT→T1 transform yet,
or no T1 to register the MNI leg). Idempotent and cheap once the transform is
cached, so it's safe to run after detection, after labeling, on MRI import, and
on every edit-save — it rebuilds wholesale from the current contacts, so moves /
adds / deletes propagate and the MNI coords + labels never drift.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent stamp-mni",
        description="Warp a case's contacts to MNI + label them against the "
                    "bundled MNI-native atlases (CerebrA + Iglesias).",
    )
    parser.add_argument("case_dir", help="the case directory (contacts.tsv + regcache/)")
    parser.add_argument("--t1", default="",
                        help="patient MRI — register T1→MNI if that transform isn't cached yet")
    parser.add_argument("--refine", action="store_true",
                        help="refine the T1→MNI affine with a B-spline nonlinear warp "
                             "(SimpleITK, torch-free, ~30 s) — better subcortical accuracy. "
                             "Requires --t1; recomputes the transform.")
    args = parser.parse_args(argv)

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        print(f"error: case dir not found: {case_dir}", file=sys.stderr)
        return 2

    try:
        from rosa_core import cohort, mni_label
    except ImportError as exc:
        print(f"error: rosa_core unavailable ({exc})", file=sys.stderr)
        return 2

    regcache = case_dir / "regcache"
    log = lambda m: print(m, file=sys.stderr)   # noqa: E731

    if args.t1:
        cohort.ensure_mni_transform(regcache, args.t1, refine=args.refine, log=log)

    if not cohort.mni_transforms_present(regcache):
        print(f"[stamp-mni] {case_dir.name}: not MNI-poolable — need "
              f"{cohort.CT_TO_T1_TFM} + {cohort.T1_TO_MNI_TFM}; skipping", file=sys.stderr)
        return 0

    n_labels = mni_label.stamp_case(case_dir)
    print(f"[stamp-mni] {case_dir.name}: warped + {n_labels} atlas labels → "
          f"{regcache / 'contacts_labels_mni.tsv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
