"""Per-case summaries for the Cases list.

The Cases home shows enough to tell cases apart without opening them — how many
electrodes and contacts, whether an MRI/labels are in, when it was made. Those
counts aren't on JobStatus (they'd cost a file read on every status poll), so we
compute them here, only for the (infrequently-fetched) case list, by reading the
case's TSVs + review.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def ct_fingerprint(ct_path: str | Path) -> str | None:
    """SHA-256 of the CT file's bytes, or None if it's missing.

    Identity of the scan: two cases built from the same CT file share this, so
    the New-case / Import flows can spot "you already have a case for this CT"
    (re-uploading the same file yields identical bytes → identical hash). Cheap,
    streamed — read once at case creation, stored in the manifest.
    """
    ct_path = Path(ct_path)
    if not ct_path.is_file():
        return None
    h = hashlib.sha256()
    with open(ct_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_data_rows(path: Path) -> int:
    """Row count minus the header, 0 if the file is missing."""
    if not path.is_file():
        return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        n = sum(1 for _ in f)
    return max(n - 1, 0)


def _distinct_trajectories(contacts: Path) -> int:
    """Shank count from contacts.tsv when trajectories.tsv is absent."""
    if not contacts.is_file():
        return 0
    seen: set[str] = set()
    with open(contacts, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            t = (row.get("trajectory") or "").strip()
            if t:
                seen.add(t)
    return len(seen)


def _is_labeled(job_dir: Path) -> bool:
    """True once any contact carries an anatomical region in the saved review."""
    rj = job_dir / "review.json"
    if not rj.is_file():
        return False
    try:
        doc = json.loads(rj.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a partial review.json isn't "labeled"
        return False
    return any(c.get("region")
               for s in doc.get("shanks", []) for c in s.get("contacts", []))


# The cohort pools contacts in CerebrA's space (MNI152NLin2009cSym); a case is
# "MNI-poolable" once its CerebrA-through-T1 label run has cached both legs of
# CT→T1→MNI in regcache/ (see docs/design/cohort_view.md §4.2).
_MNI_POOL_TFM = "mni_MNI152NLin2009cSym_to_t1.tfm"


def summarize_case(status, job_dir: str | Path, *, ct_hash=None, atlas=None) -> dict:
    """A Cases-list row: identity + electrode/contact counts + MRI/label/atlas
    state. ``ct_hash`` (PHI-safe scan identity, from the manifest params) and
    ``atlas`` (the case's active label atlas) are resolved by the caller, which
    has the job params + the runner; here we only add the derived ``mni_eligible``
    (both CT→T1→MNI transforms cached)."""
    job_dir = Path(job_dir)
    contacts = job_dir / "contacts.tsv"
    regcache = job_dir / "regcache"
    mni_eligible = ((regcache / "t1_to_ct.tfm").is_file()
                    and (regcache / _MNI_POOL_TFM).is_file())
    return {
        "id": status.id,
        "label": status.label,
        "kind": status.kind,
        "created_at": status.created_at,
        "has_mri": bool(status.t1),
        "n_contacts": _count_data_rows(contacts),
        "n_shanks": _count_data_rows(job_dir / "trajectories.tsv") or _distinct_trajectories(contacts),
        "labeled": _is_labeled(job_dir),
        "ct_hash": ct_hash,
        "atlas": atlas,
        "mni_eligible": mni_eligible,
    }


__all__ = ["summarize_case", "ct_fingerprint"]
