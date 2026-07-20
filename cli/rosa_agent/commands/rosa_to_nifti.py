"""rosa-agent rosa-to-nifti — bake a ROSA folder into ready-to-use NIfTI inputs.

A ROSA case folder contains:

  * a ``.ros`` file declaring display volumes + trajectories +
    display→reference transforms,
  * Analyze-format volumes under ``DICOM/<uid>/<name>.img/.hdr``.

The display volumes are individually un-aligned on disk — the
display→reference transform stamped into the ``.ros`` file is what
puts them in a common RAS frame (typically the first display). This
command applies that transform and writes each display as a NIfTI
whose IJK→RAS header matches the ROSA reference frame, so the output
files can drop straight into anything that reads NIfTI (Slicer,
ITK-SNAP, ``rosa-agent place``, atlas tools, …).

It also writes the planned trajectories as a TSV in the same RAS
frame, ready to feed back through ``rosa-agent place --seeds``.

Usage::

    rosa-agent rosa-to-nifti --rosa-folder s57_rosa/ --output s57_unpacked/
    # ... then:
    rosa-agent place \\
        --ct s57_unpacked/post.nii.gz \\
        --seeds s57_unpacked/seeds.tsv \\
        --output s57_qc/ --library dixi

By default every display volume is exported. Pass ``--volume`` (one
or more times) to export a subset.

Output directory layout::

    output/
      manifest.json              ← what was loaded + matrix provenance
      seeds.tsv                  ← trajectories in RAS frame; place --seeds compatible
      <display_name>.nii.gz      ← one per exported display
      ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rosa-agent rosa-to-nifti",
        description=(
            "Convert a ROSA case folder (.ros + Analyze) into NIfTI "
            "volumes + a seeds TSV in the ROSA reference RAS frame, "
            "ready for `rosa-agent place`."
        ),
    )
    parser.add_argument("--rosa-folder", required=True,
                        help="ROSA case folder (contains the .ros file + DICOM/Analyze tree)")
    parser.add_argument("--output", required=True,
                        help="output directory (created if missing)")
    parser.add_argument("--volume", action="append", default=None,
                        help="display volume name to export (e.g. 'post'). "
                             "Repeat to select more than one. Default: all displays.")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-volume progress prints")
    args = parser.parse_args(argv)

    log = (lambda _msg: None) if args.quiet else _stderr

    rosa_folder = Path(args.rosa_folder)
    if not rosa_folder.is_dir():
        _stderr(f"error: --rosa-folder is not a directory: {rosa_folder}")
        return 2

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy-import the heavy stuff (SITK + parser).
    try:
        import SimpleITK as sitk  # noqa: F401
        from rosa_core import load_rosa_volume_as_sitk
        from rosa_core.case_loader import find_ros_file
        from rosa_core.ros_parser import parse_ros_file
    except ImportError as exc:
        _stderr(f"error: rosa_core / SimpleITK unavailable ({exc})")
        return 2

    # Resolve the .ros file once + figure out which displays to export.
    try:
        ros_path = find_ros_file(str(rosa_folder))
        parsed = parse_ros_file(ros_path)
    except (ValueError, FileNotFoundError) as exc:
        _stderr(f"error: couldn't read .ros file under {rosa_folder}: {exc}")
        return 2
    all_displays = [d.get("volume") for d in parsed.get("displays") or []]
    if not all_displays:
        _stderr(f"error: .ros at {ros_path} declared no display volumes")
        return 2

    if args.volume:
        wanted = list(dict.fromkeys(args.volume))   # dedup, preserve order
        unknown = [v for v in wanted if v not in all_displays]
        if unknown:
            _stderr(
                f"error: unknown --volume name(s): {unknown}. "
                f"Available in {Path(ros_path).name}: {all_displays}"
            )
            return 2
        targets = wanted
    else:
        targets = list(all_displays)

    log(
        f"[rosa-to-nifti] {rosa_folder} → {out_dir}  "
        f"(.ros={Path(ros_path).name}; exporting {len(targets)} of "
        f"{len(all_displays)} display volumes)"
    )

    # Export each target volume. The SAME load_rosa_volume_as_sitk call
    # the Slicer side uses — single source of truth for the .ros math.
    export_records: list[dict] = []
    skipped: list[dict] = []
    # Display 0 defines the reference RAS frame every other display is aligned to.
    reference_name = all_displays[0]
    reference_volume = reference_name
    rosa_trajectories: list[dict] | None = None
    for vname in targets:
        try:
            sitk_img, meta = load_rosa_volume_as_sitk(
                str(rosa_folder), volume_name=vname,
            )
        except (ValueError, FileNotFoundError) as exc:
            if vname == reference_name:
                # The reference display defines the common frame — can't proceed.
                _stderr(f"error: the reference display {vname!r} could not be loaded: {exc}")
                return 2
            # A .ros can reference a display whose Analyze volume isn't present
            # (never exported, or dropped in de-identification). Skip it and bake
            # the rest so the case still imports with the volumes that ARE there.
            _stderr(f"warning: skipping display {vname!r} — the .ros references it but its "
                    f"Analyze volume was not found ({exc})")
            skipped.append({"display_name": vname, "reason": str(exc)})
            continue

        out_nifti = out_dir / f"{_safe_stem(vname)}.nii.gz"
        sitk.WriteImage(sitk_img, str(out_nifti))
        log(
            f"[rosa-to-nifti]  wrote {out_nifti.name}  "
            f"(display_index={meta['display_index']}, "
            f"is_reference={meta['is_reference']})"
        )
        export_records.append({
            "display_name": vname,
            "display_index": int(meta["display_index"]),
            "is_reference": bool(meta["is_reference"]),
            "source_analyze_path": meta.get("loaded_volume_path"),
            "output_nifti_path": str(out_nifti),
            "display_to_reference_ras": meta["display_to_reference_ras"],
        })
        # Trajectories are global to the .ros file; just grab them
        # once from any successful load.
        if rosa_trajectories is None:
            rosa_trajectories = list(meta.get("trajectories") or [])
        if str(meta.get("reference_volume") or "") and meta["reference_volume"]:
            reference_volume = meta["reference_volume"]

    if not export_records:
        _stderr(f"error: none of the {len(targets)} display volume(s) could be "
                f"loaded from {rosa_folder}")
        return 2
    if skipped:
        log(f"[rosa-to-nifti] skipped {len(skipped)} display(s) with no Analyze volume: "
            f"{[s['display_name'] for s in skipped]}")

    # Trajectories → seeds.tsv (place --seeds compatible).
    seeds_path = out_dir / "seeds.tsv"
    n_seeds = _write_seeds_tsv(seeds_path, rosa_trajectories or [])
    log(
        f"[rosa-to-nifti] wrote {seeds_path.name}  ({n_seeds} trajectories "
        f"in RAS frame of {reference_volume!r})"
    )

    # Manifest tying it all together.
    manifest = {
        "manifest_version": "1.0",
        "tool": "rosa-to-nifti",
        "rosa_folder": str(rosa_folder.resolve()),
        "ros_file": str(Path(ros_path).resolve()),
        "reference_volume": reference_volume,
        "exported_volumes": export_records,
        "skipped_volumes": skipped,
        "seeds_tsv": str(seeds_path),
        "n_planned_trajectories": n_seeds,
        "all_displays_in_ros": all_displays,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"[rosa-to-nifti] wrote {manifest_path.name}")

    print(
        f"rosa_folder={rosa_folder} "
        f"output={out_dir} "
        f"exported={len(export_records)} "
        f"skipped={len(skipped)} "
        f"seeds={n_seeds} "
        f"reference={reference_volume!r}"
    )
    return 0


def _safe_stem(name: str) -> str:
    """File-safe stem (allow alnum, dot, dash, underscore)."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


def _write_seeds_tsv(path: Path, trajectories: list[dict]) -> int:
    """Write a place-compatible seeds TSV in RAS coordinates.

    Schema matches what ``rosa_agent.io.trajectory_io.read_seeds_tsv``
    accepts: ``name``, ``start_x/y/z``, ``end_x/y/z``. No
    ``electrode_model`` column — .ros trajectories are surgical plans,
    not implant records, so the placer dispatches to mode 5 (library
    matching) on these by default.
    """
    cols = ["name", "start_x", "start_y", "start_z",
            "end_x", "end_y", "end_z"]
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        n = 0
        for t in trajectories:
            try:
                sx, sy, sz = (float(v) for v in t["start_ras"])
                ex, ey, ez = (float(v) for v in t["end_ras"])
            except (KeyError, TypeError, ValueError):
                continue
            f.write(
                "\t".join([
                    str(t.get("name") or f"T{n + 1:02d}"),
                    f"{sx:.6f}", f"{sy:.6f}", f"{sz:.6f}",
                    f"{ex:.6f}", f"{ey:.6f}", f"{ez:.6f}",
                ]) + "\n"
            )
            n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(main())
