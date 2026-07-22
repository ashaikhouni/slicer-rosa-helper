"""rosa-agent cohort-export — warp a case's contacts into MNI for cohort pooling.

Reads ``<case_dir>/contacts.tsv`` + the two cached ITK affines in
``<case_dir>/regcache/`` (``t1_to_ct.tfm`` and
``mni_MNI152NLin2009cSym_to_t1.tfm``), warps every contact CT→T1→MNI, and writes
``<case_dir>/regcache/contacts_mni.tsv`` — the per-case cache the cohort MNI
viewer pools across subjects.

No-op (exit 0) when the case isn't poolable — i.e. it wasn't labeled with
CerebrA through its MRI, so the MNI transform was never cached. That makes it
safe to run as an unconditional step after any label job.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent cohort-export",
        description="Warp a case's contacts into MNI152NLin2009cSym "
                    "(regcache/contacts_mni.tsv) for cohort pooling.",
    )
    parser.add_argument("case_dir", help="the case directory (holds contacts.tsv + regcache/)")
    parser.add_argument("--output", "-o", default="",
                        help="output TSV (default: <case_dir>/regcache/contacts_mni.tsv)")
    args = parser.parse_args(argv)

    case_dir = Path(args.case_dir)
    if not case_dir.is_dir():
        print(f"error: case dir not found: {case_dir}", file=sys.stderr)
        return 2

    try:
        from rosa_core import cohort
    except ImportError as exc:
        print(f"error: rosa_core unavailable ({exc})", file=sys.stderr)
        return 2

    if not cohort.mni_transforms_present(case_dir / "regcache"):
        print(f"[cohort-export] {case_dir.name}: not MNI-poolable "
              f"(no {cohort.T1_TO_MNI_TFM}); skipping", file=sys.stderr)
        return 0

    if args.output:
        rows = cohort.contacts_mni_rows(case_dir)
        if rows:
            cohort.write_tsv(rows, args.output)
        n = len(rows)
        out = args.output
    else:
        n = cohort.export_case(case_dir)
        out = case_dir / "regcache" / "contacts_mni.tsv"

    print(f"[cohort-export] {case_dir.name}: {n} contacts → {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
