#!/usr/bin/env python3
"""Command-line utilities built on top of `rosa_core`.

Subcommands support inspection and export of trajectory/transform data from a
ROSA `.ros` file without launching Slicer.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB_DIR = os.path.join(REPO_ROOT, "RosaHelper", "Lib")
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from rosa_core import (  # noqa: E402
    build_assignment_template,
    build_effective_matrices,
    build_fcsv_rows,
    build_markups_lines,
    contacts_to_fcsv_rows,
    default_electrode_library_path,
    generate_contacts,
    invert_4x4,
    lps_to_ras_matrix,
    load_assignments,
    load_electrode_library,
    model_map,
    parse_ros_file,
    resolve_reference_index,
    save_assignment_template,
    save_contacts_markups_json,
    save_contacts_rosa_json,
    to_itk_affine_text,
)
from rosa_core.exporters import save_fcsv, save_markups_json  # noqa: E402


def cmd_list(args):
    """Print display metadata and trajectory count."""
    parsed = parse_ros_file(args.ros)
    displays = parsed["displays"]
    trajectories = parsed["trajectories"]

    print(f"ROS: {args.ros}")
    print(f"Displays: {len(displays)}")
    for disp in displays:
        print(
            "volume={v} imagery={i} serie_uid={u} imagery_3dref={r}".format(
                v=disp.get("volume", ""),
                i=disp.get("imagery_name", ""),
                u=disp.get("serie_uid", ""),
                r=disp.get("imagery_3dref", ""),
            )
        )
        for row in disp["matrix"]:
            print("  " + " ".join(f"{x: .6f}" for x in row))
    print(f"Trajectories: {len(trajectories)}")


def _find_display_index(parsed, volume_name):
    """Return display index for a volume name, case-insensitive."""
    for i, disp in enumerate(parsed["displays"]):
        if disp["volume"].lower() == volume_name.lower():
            return i
    names = ", ".join(d["volume"] for d in parsed["displays"]) or "none"
    raise ValueError(f"Volume '{volume_name}' not found. Available: {names}")


def cmd_markups(args):
    """Export trajectory lines to Slicer Markups JSON format."""
    parsed = parse_ros_file(args.ros)
    displays = parsed["displays"]
    root_index = resolve_reference_index(displays, args.root_volume)
    effective_lps = build_effective_matrices(displays, root_index=root_index)
    transform = None
    if args.volume_name:
        idx = _find_display_index(parsed, args.volume_name)
        transform = invert_4x4(effective_lps[idx])

    markups = build_markups_lines(
        parsed["trajectories"],
        to_ras=(args.coord == "ras"),
        display_to_dicom=transform,
    )
    save_markups_json(args.out, markups)
    print(f"Wrote {args.out} ({len(markups)} trajectories)")


def cmd_fcsv(args):
    """Export trajectory endpoints to Slicer FCSV format."""
    parsed = parse_ros_file(args.ros)
    rows = build_fcsv_rows(
        parsed["trajectories"],
        to_ras=(args.coord == "ras"),
        same_label_pair=args.same_label_pair,
    )
    save_fcsv(args.out, rows)
    print(f"Wrote {args.out} ({len(rows)} points)")


def cmd_tfm(args):
    """Export one display transform to ITK `.tfm` text format."""
    parsed = parse_ros_file(args.ros)
    displays = parsed["displays"]
    root_index = resolve_reference_index(displays, args.root_volume)
    effective_lps = build_effective_matrices(displays, root_index=root_index)
    matrix = effective_lps[_find_display_index(parsed, args.volume_name)]
    if args.ras:
        matrix = lps_to_ras_matrix(matrix)
    if args.invert:
        matrix = invert_4x4(matrix)
    text = to_itk_affine_text(matrix)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {args.out}")


def cmd_contacts_template(args):
    """Create editable trajectory->electrode assignment template."""
    parsed = parse_ros_file(args.ros)
    template = build_assignment_template(
        parsed["trajectories"],
        default_model_id=args.default_model_id or "",
        default_tip_at=args.default_tip_at,
    )
    save_assignment_template(args.out, template)
    print(f"Wrote {args.out} ({len(template['assignments'])} trajectory rows)")


def cmd_contacts_generate(args):
    """Generate contact points from assignments and electrode library."""
    parsed = parse_ros_file(args.ros)
    lib = load_electrode_library(args.electrode_library)
    models = model_map(lib)
    assignments = load_assignments(args.assignments)

    contacts = generate_contacts(parsed["trajectories"], models, assignments)
    metadata = {
        "ros_path": args.ros,
        "electrode_library": str(args.electrode_library or default_electrode_library_path()),
    }
    save_contacts_rosa_json(args.out_rosa_json, contacts, metadata=metadata)
    print(f"Wrote {args.out_rosa_json} ({len(contacts)} contacts in ROSA_LPS)")

    if args.out_fcsv:
        rows = contacts_to_fcsv_rows(contacts, to_ras=True)
        save_fcsv(args.out_fcsv, rows)
        print(f"Wrote {args.out_fcsv} ({len(rows)} points in RAS)")

    if args.out_markups:
        save_contacts_markups_json(args.out_markups, contacts, to_ras=True, node_name="contacts")
        print(f"Wrote {args.out_markups} (markups fiducials)")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="ROSA export helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List volumes and trajectory counts")
    p_list.add_argument("--ros", required=True)
    p_list.set_defaults(func=cmd_list)

    p_markups = sub.add_parser("markups", help="Export Slicer .mrk.json")
    p_markups.add_argument("--ros", required=True)
    p_markups.add_argument("--out", required=True)
    p_markups.add_argument("--coord", choices=["ras", "lps"], default="ras")
    p_markups.add_argument("--volume-name", help="Optional display volume for inverse TRdicomRdisplay")
    p_markups.add_argument(
        "--root-volume",
        help="Root reference volume name used to compose IMAGERY_3DREF chains (default: first display)",
    )
    p_markups.set_defaults(func=cmd_markups)

    p_fcsv = sub.add_parser("fcsv", help="Export Slicer .fcsv")
    p_fcsv.add_argument("--ros", required=True)
    p_fcsv.add_argument("--out", required=True)
    p_fcsv.add_argument("--coord", choices=["ras", "lps"], default="ras")
    p_fcsv.add_argument(
        "--same-label-pair",
        action="store_true",
        help="Use same label for entry/target point pair (DEETO style)",
    )
    p_fcsv.set_defaults(func=cmd_fcsv)

    p_tfm = sub.add_parser("tfm", help="Export volume transform as ITK .tfm")
    p_tfm.add_argument("--ros", required=True)
    p_tfm.add_argument("--volume-name", required=True)
    p_tfm.add_argument(
        "--root-volume",
        help="Root reference volume name used to compose IMAGERY_3DREF chains (default: first display)",
    )
    p_tfm.add_argument("--out", required=True)
    p_tfm.add_argument(
        "--ras",
        action="store_true",
        help="Convert ROSA LPS matrix to RAS before writing",
    )
    p_tfm.add_argument(
        "--invert",
        action="store_true",
        help="Invert matrix before writing",
    )
    p_tfm.set_defaults(func=cmd_tfm)

    p_contacts_template = sub.add_parser(
        "contacts-template",
        help="Create an assignments template for trajectory -> electrode model",
    )
    p_contacts_template.add_argument("--ros", required=True)
    p_contacts_template.add_argument("--out", required=True)
    p_contacts_template.add_argument("--default-model-id")
    p_contacts_template.add_argument("--default-tip-at", choices=["entry", "target"], default="target")
    p_contacts_template.set_defaults(func=cmd_contacts_template)

    p_contacts_generate = sub.add_parser(
        "contacts-generate",
        help="Generate contact locations from assignments",
    )
    p_contacts_generate.add_argument("--ros", required=True)
    p_contacts_generate.add_argument("--assignments", required=True)
    p_contacts_generate.add_argument(
        "--electrode-library",
        help="Path to electrode models JSON (defaults to bundled DIXI library)",
    )
    p_contacts_generate.add_argument("--out-rosa-json", required=True)
    p_contacts_generate.add_argument("--out-fcsv")
    p_contacts_generate.add_argument("--out-markups")
    p_contacts_generate.set_defaults(func=cmd_contacts_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
