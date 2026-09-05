"""Plan-vs-actual trajectory accuracy for a case with a surgical plan.

Compares each PLANNED trajectory (the ROSA ``.ros`` plan, persisted per case as
``ros_plan.tsv`` in the post-op CT frame) against the ACTUAL localized trajectory
(``trajectories.tsv`` + ``contacts.tsv``), keyed by shank name. Reuses the tested
metric engine in :mod:`rosa_core.qc` — entry error, target error, axis angle, and
per-contact radial (lateral) deviation — plus a median/p95/max case summary.

Two things this layer adds on top of ``qc``:
  * reads the case's TSVs into the ``{name: {...}}`` dicts ``qc`` expects, and
  * resolves the SEEG bolt↔tip axis flip (a ``.ros`` plan may store entry/target
    the opposite way round from the detected line) BEFORE differencing endpoints,
    so entry error is entry-vs-entry and target error is tip-vs-tip.

No plan on disk → returns ``None`` (the "no plan, no error" contract). The metrics
are radial/angular so they're frame-invariant; plan and actual only have to share
one consistent frame, which the ``ros_plan.tsv``-in-CT-frame invariant guarantees.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

__all__ = ["compute_plan_accuracy", "HEADLINE_METRIC"]

# The single per-case number surfaced in summaries / the cohort table.
HEADLINE_METRIC = "target_error_mm"


def _read_lines(path: Path) -> dict[str, dict]:
    """``{name: {"start": [x,y,z], "end": [x,y,z]}}`` from a trajectories/plan TSV."""
    out: dict[str, dict] = {}
    if not path.is_file():
        return out
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
             if not ln.lstrip().startswith("#")]
    for r in csv.DictReader(lines, delimiter="\t"):
        name = (r.get("name") or "").strip()
        try:
            out[name] = {
                "start": [float(r["start_x"]), float(r["start_y"]), float(r["start_z"])],
                "end": [float(r["end_x"]), float(r["end_y"]), float(r["end_z"])],
            }
        except (KeyError, ValueError, TypeError):
            continue
    return out


def _read_contacts(path: Path) -> dict[str, list[list[float]]]:
    """``{trajectory: [[x,y,z], ...]}`` from a contacts TSV."""
    out: dict[str, list[list[float]]] = {}
    if not path.is_file():
        return out
    with open(path, encoding="utf-8", newline="") as f:
        for c in csv.DictReader(f, delimiter="\t"):
            try:
                out.setdefault(str(c.get("trajectory") or ""), []).append(
                    [float(c["x"]), float(c["y"]), float(c["z"])])
            except (KeyError, ValueError, TypeError):
                continue
    return out


def _oriented_plan(plan: dict, actual: dict) -> dict:
    """Plan line with entry/target oriented to match ``actual``'s direction, so
    entry↔entry and tip↔tip (a ``.ros`` plan can point bolt→tip either way)."""
    pa = [plan["end"][i] - plan["start"][i] for i in range(3)]
    aa = [actual["end"][i] - actual["start"][i] for i in range(3)]
    if (pa[0] * aa[0] + pa[1] * aa[1] + pa[2] * aa[2]) < 0.0:
        return {"start": plan["end"], "end": plan["start"]}
    return plan


def compute_plan_accuracy(plan_tsv, traj_tsv, contacts_tsv) -> dict[str, Any] | None:
    """Per-shank plan-vs-actual accuracy + a case summary, or ``None`` if there is
    no plan. All three inputs are TSV paths in the SAME (CT) frame.

    Returns::

        {"rows": [ {trajectory, entry_error_mm, target_error_mm, angle_deg,
                    mean/max/rms_contact_radial_mm, n_contacts}, ... ],
         "summary": {metric: {median, p95, max}, ...},
         "headline_tpe_mm": <median target_error_mm>,
         "n_plan": int, "n_matched": int}
    """
    from rosa_core.qc import compute_plan_vs_placement_qc, summarize_plan_vs_placement_qc

    plan = _read_lines(Path(plan_tsv))
    if not plan:
        return None                       # no plan → no error (by contract)
    actual = _read_lines(Path(traj_tsv))
    contacts = _read_contacts(Path(contacts_tsv))

    planned_by_name: dict[str, dict] = dict(plan)          # every plan shank (unmatched → None row)
    fitted_by_name: dict[str, dict] = {}
    for name, al in actual.items():
        if name not in plan:
            continue                       # an actual with no plan counterpart isn't scored
        planned_by_name[name] = _oriented_plan(plan[name], al)
        fitted_by_name[name] = {"start": al["start"], "end": al["end"],
                                "contacts": contacts.get(name, [])}

    rows = compute_plan_vs_placement_qc(planned_by_name, fitted_by_name)
    summary = summarize_plan_vs_placement_qc(rows)
    return {
        "rows": rows,
        "summary": summary,
        "headline_tpe_mm": summary.get(HEADLINE_METRIC, {}).get("median"),
        "n_plan": len(plan),
        "n_matched": len(fitted_by_name),
    }
