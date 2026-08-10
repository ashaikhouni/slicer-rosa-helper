"""ReviewDoc — build an editable result from a run, apply edits, export.

A completed run leaves ``contacts.tsv`` (engine ``CONTACT_COLUMNS``) in its job
dir. :func:`build_review_doc` turns that into a :class:`ReviewDoc` (shanks →
contacts) the UI can review; :func:`apply_edit` applies accept/reject/relabel
edits; :func:`export_contacts` writes the corrected set back out.

Reuses the engine's TSV reader + column contract (the app depends on the
engine), so the format stays a single source of truth. Only pure-data edits
here (accept/reject/relabel); model-change / tip-nudge *re-derive* edits need
the engine's placement and are a follow-up.
"""
from __future__ import annotations

from pathlib import Path

# App depends on the engine (never the reverse). Reuse its reader + columns.
from rosa_agent.io.trajectory_io import (
    CONTACT_COLUMNS, read_tsv_rows, write_tsv_rows,
)

from .models import ReviewContact, ReviewDoc, ReviewEdit, ReviewOp, ReviewShank

# Corrected export carries the anatomical region alongside the engine columns.
REVIEW_CONTACT_COLUMNS = tuple(CONTACT_COLUMNS) + ("region",)
# Region may arrive under any of these column names from the labeling step.
_REGION_KEYS = ("region", "fs_region", "thomas_region", "atlas_label", "label_region")


def _find_contacts_tsv(job_dir: Path) -> Path | None:
    exact = job_dir / "contacts.tsv"
    if exact.is_file():
        return exact
    hits = sorted(job_dir.glob("*contacts*.tsv"))
    return hits[0] if hits else None


def _region_of(row: dict) -> str | None:
    for k in _REGION_KEYS:
        v = row.get(k)
        if v:
            return v
    return None


def build_review_doc(job_dir: str | Path) -> ReviewDoc:
    """Build a ReviewDoc from a job dir's ``contacts.tsv`` (+ trajectories.tsv).

    Raises FileNotFoundError when the run produced no contacts.
    """
    job_dir = Path(job_dir)
    tsv = _find_contacts_tsv(job_dir)
    if tsv is None:
        raise FileNotFoundError(f"no contacts.tsv in {job_dir}")

    # Per-shank electrode model from trajectories.tsv, if present.
    traj_model: dict[str, str | None] = {}
    traj_tsv = job_dir / "trajectories.tsv"
    if traj_tsv.is_file():
        for tr in read_tsv_rows(traj_tsv):
            name = tr.get("name")
            if name:
                traj_model[name] = tr.get("electrode_model") or None

    shanks: dict[str, ReviewShank] = {}
    order: list[str] = []
    for row in read_tsv_rows(tsv):
        shank = (row.get("trajectory") or "").strip()
        if not shank:
            continue
        if shank not in shanks:
            shanks[shank] = ReviewShank(name=shank, model=traj_model.get(shank))
            order.append(shank)
        try:
            idx = int(float(row.get("contact_index") or 0))
            contact = ReviewContact(
                shank=shank,
                index=idx,
                name=(row.get("label") or f"{shank}{idx}"),
                x=float(row["x"]), y=float(row["y"]), z=float(row["z"]),
                model=(row.get("electrode_model") or None),
                region=_region_of(row),
            )
        except (KeyError, ValueError):
            continue  # skip malformed rows rather than fail the whole doc
        shanks[shank].contacts.append(contact)
        if shanks[shank].model is None and contact.model:
            shanks[shank].model = contact.model

    return ReviewDoc(shanks=[shanks[n] for n in order])


def apply_edit(doc: ReviewDoc, edit: ReviewEdit) -> None:
    """Apply one edit in place. Raises ValueError on an invalid target/op."""
    shank = next((s for s in doc.shanks if s.name == edit.shank), None)
    if shank is None:
        raise ValueError(f"unknown shank {edit.shank!r}")

    if edit.op == ReviewOp.accept_shank:
        shank.accepted = True
        return
    if edit.op == ReviewOp.reject_shank:
        shank.accepted = False
        return

    if edit.index is None:
        raise ValueError(f"{edit.op.value} requires 'index'")
    contact = next((c for c in shank.contacts if c.index == edit.index), None)
    if contact is None:
        raise ValueError(f"unknown contact {edit.shank}[{edit.index}]")

    if edit.op == ReviewOp.accept_contact:
        contact.accepted = True
    elif edit.op == ReviewOp.reject_contact:
        contact.accepted = False
    elif edit.op == ReviewOp.relabel_contact:
        if edit.region is None:
            raise ValueError("relabel_contact requires 'region'")
        contact.region = edit.region


def export_contacts(doc: ReviewDoc, out_path: str | Path) -> int:
    """Write the corrected contacts (accepted shanks' accepted contacts) to a
    TSV. Returns the number of contacts written."""
    rows = []
    for shank in doc.shanks:
        if not shank.accepted:
            continue
        for c in shank.contacts:
            if not c.accepted:
                continue
            rows.append({
                "trajectory": c.shank, "label": c.name, "contact_index": c.index,
                "x": f"{c.x:.6f}", "y": f"{c.y:.6f}", "z": f"{c.z:.6f}",
                "peak_detected": "", "electrode_model": c.model or "",
                "region": c.region or "",
            })
    write_tsv_rows(out_path, rows, REVIEW_CONTACT_COLUMNS)
    return len(rows)


class ReviewStore:
    """Per-job ReviewDoc registry, persisted to ``<job_dir>/review.json``."""

    def __init__(self) -> None:
        self._docs: dict[str, ReviewDoc] = {}

    def get_or_build(self, job_id: str, job_dir: str | Path) -> ReviewDoc:
        if job_id not in self._docs:
            # Restore a persisted review (accept/reject/relabel + approved atlas
            # labels) across restarts; only rebuild from contacts.tsv when there
            # is no saved review yet.
            saved = Path(job_dir) / "review.json"
            if saved.is_file():
                try:
                    self._docs[job_id] = ReviewDoc.model_validate_json(saved.read_text(encoding="utf-8"))
                    return self._docs[job_id]
                except Exception:  # noqa: BLE001 — fall back to a fresh build
                    pass
            self._docs[job_id] = build_review_doc(job_dir)
            self._persist(job_id, job_dir)
        return self._docs[job_id]

    def apply(self, job_id: str, job_dir: str | Path, ops: list[ReviewEdit]) -> ReviewDoc:
        doc = self.get_or_build(job_id, job_dir)
        for op in ops:
            apply_edit(doc, op)
        self._persist(job_id, job_dir)
        return doc

    def rebuild_preserving_labels(self, job_id: str, job_dir: str | Path,
                                  renames: dict[str, str] | None = None) -> ReviewDoc:
        """After a geometry edit rewrote contacts.tsv: rebuild the doc from the
        new contacts, carrying over anatomical labels by (shank, contact_index).
        ``renames`` maps new→old shank name so labels follow a rename. Removed
        shanks / contacts drop out; added ones start unlabeled."""
        job_dir = Path(job_dir)
        renames = renames or {}
        prev = self._docs.get(job_id)
        if prev is None and (job_dir / "review.json").is_file():
            try:
                prev = ReviewDoc.model_validate_json((job_dir / "review.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                prev = None
        regions = {(s.name, c.index): c.region
                   for s in (prev.shanks if prev else []) for c in s.contacts if c.region}
        doc = build_review_doc(job_dir)                 # from the NEW contacts.tsv
        for s in doc.shanks:
            src = renames.get(s.name, s.name)           # follow a rename to old labels
            for c in s.contacts:
                r = regions.get((src, c.index))
                if r:
                    c.region = r
        self._docs[job_id] = doc
        self._persist(job_id, job_dir)
        return doc

    def _persist(self, job_id: str, job_dir: str | Path) -> None:
        try:
            (Path(job_dir) / "review.json").write_text(
                self._docs[job_id].model_dump_json(indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 — persistence is best-effort audit
            pass


__all__ = [
    "build_review_doc", "apply_edit", "export_contacts", "ReviewStore",
    "REVIEW_CONTACT_COLUMNS",
]
