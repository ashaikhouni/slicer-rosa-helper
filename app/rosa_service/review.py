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
import math
import os
import tempfile

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
        contact.region_stale = False


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


class ReviewPersistenceError(RuntimeError):
    """A review could not be read or durably saved; never silently rebuild it."""


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Replace a UTF-8 file only after the complete new contents reach disk."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as f:
            tmp = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


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
                except Exception as exc:
                    raise ReviewPersistenceError("Saved review could not be read. Restore it from backup before continuing.") from exc
            doc = build_review_doc(job_dir)
            self._persist(job_id, job_dir, doc)
            self._docs[job_id] = doc
        return self._docs[job_id]

    def apply(self, job_id: str, job_dir: str | Path, ops: list[ReviewEdit]) -> ReviewDoc:
        doc = self.get_or_build(job_id, job_dir).model_copy(deep=True)
        for op in ops:
            apply_edit(doc, op)
        self._persist(job_id, job_dir, doc)
        self._docs[job_id] = doc
        return doc

    def rebuild_preserving_labels(self, job_id: str, job_dir: str | Path,
                                  renames: dict[str, str] | None = None) -> ReviewDoc:
        """Carry inclusion decisions across edits; retain only geometrically valid labels.

        A rename preserves identity through new→old names. Moved contacts keep
        their inclusion decisions, but their anatomical labels need review again.
        """
        job_dir = Path(job_dir)
        renames = renames or {}
        prev = self._docs.get(job_id)
        if prev is None and (job_dir / "review.json").is_file():
            try:
                prev = ReviewDoc.model_validate_json((job_dir / "review.json").read_text(encoding="utf-8"))
            except Exception as exc:
                raise ReviewPersistenceError("Saved review could not be read; geometry edit was not accepted.") from exc
        old_shanks = {s.name: s for s in (prev.shanks if prev else [])}
        doc = build_review_doc(job_dir)
        for s in doc.shanks:
            old = old_shanks.get(renames.get(s.name, s.name))
            if old is None:
                continue
            s.accepted = old.accepted
            contacts = {c.index: c for c in old.contacts}
            for c in s.contacts:
                prior = contacts.get(c.index)
                if prior is None:
                    continue
                c.accepted = prior.accepted
                unchanged = c.model == prior.model and all(
                    math.isclose(a, b, abs_tol=1e-5, rel_tol=0)
                    for a, b in zip((c.x, c.y, c.z), (prior.x, prior.y, prior.z)))
                c.region = prior.region if unchanged else None
                c.region_stale = prior.region_stale or (not unchanged and bool(prior.region))
        self._persist(job_id, job_dir, doc)
        self._docs[job_id] = doc
        return doc

    def _persist(self, job_id: str, job_dir: str | Path, doc: ReviewDoc) -> None:
        try:
            atomic_write_text(Path(job_dir) / "review.json", doc.model_dump_json(indent=2))
        except OSError as exc:
            raise ReviewPersistenceError("Review could not be saved. Check disk space and folder permissions, then retry.") from exc


__all__ = [
    "build_review_doc", "apply_edit", "export_contacts", "ReviewStore",
    "REVIEW_CONTACT_COLUMNS",
]
