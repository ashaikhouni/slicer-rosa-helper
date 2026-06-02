"""De-identify ROSA ``.ros`` files and extract planned trajectories.

A ``.ros`` file is line-oriented: a bracketed tag on its own line followed by
its value on the next line(s). The patient-identifying fields are:

  ``[PATIENT_NAME]``      patient name            -> replaced with a subject id
  ``[PATIENT_BIRTHDAY]``  date of birth           -> redacted (``***``)
  ``[SERIE_DATE]``        acquisition date        -> redacted (``***``)
  ``[SERIE_UID]``         DICOM SeriesInstanceUID -> PSEUDONYMISED (see below)
  ``[VOLUME]``            ``\\DICOM\\<uid>\\<name>`` path embedding the UID

Everything else (display transforms, ``IMAGERY_NAME``, modality/config, the
planned ``[TRAJECTORY]`` block, ``SECURITY_ZONE`` / ``ELLIPS`` geometry, frame /
atlas / registration) is non-identifying and is preserved verbatim.

**UIDs are pseudonymised, not blanked.** A DICOM Series UID is an identifier,
but blanking every UID to ``***`` (as the old MATLAB ``deidentifyRosFile.m``
did) destroys the linkage — you can no longer tell which display references
which series. Instead each distinct UID is mapped to a *consistent*
deterministic token (``UID-<12 hex>`` = a truncated SHA-1), applied everywhere
it appears, so a display's ``SERIE_UID`` value and the UID in its ``VOLUME``
path still match. The real-UID -> token (and name -> subject-id) mapping is
returned so the caller can persist a PRIVATE re-link key — that key is PHI and
must be kept, never shared.

CRLF line endings (the usual ROSA convention) are preserved.
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path

from .ros_parser import parse_ros_file

# Image volumes we carry into a de-identified folder. Analyze/NIfTI/NRRD headers
# have no patient-demographics fields (verified empty on real cases), so these
# are safe to copy as-is. `DICOMFiles.zip` (raw DICOM = full PHI) is NOT here.
_IMAGE_SUFFIXES = (".img", ".hdr", ".nii", ".nii.gz", ".nrrd", ".mgz", ".mha")


def _is_image_file(name: str) -> bool:
    low = name.lower()
    return any(low.endswith(s) for s in _IMAGE_SUFFIXES)


def _read_ros_text(path) -> str:
    """Read .ros text WITHOUT universal-newline translation, so the original
    CRLF line endings (the ROSA convention) survive into the de-identified file.
    ``Path.read_text`` would collapse ``\\r\\n`` -> ``\\n`` on read."""
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        return f.read()


def _write_ros_text(path, text: str) -> None:
    """Write .ros text verbatim (``newline=""`` so the kept CRLF isn't
    re-translated — on Windows the default would turn ``\\r\\n`` into ``\\r\\r\\n``)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)

# Tags whose VALUE line is redacted outright (no linkage value).
_REDACT_TO = {
    "PATIENT_BIRTHDAY": "***",
    "SERIE_DATE": "***",
}
# A DICOM UID: dot-separated integer groups, many of them. The >=4-dot guard
# keeps this from matching trajectory coords ("-11.40"), ROSANNA_OPTIONS
# ("001001000001"), or security-zone triples ("0.00 10.00 10.00").
_UID_RE = re.compile(r"\d+(?:\.\d+){4,}")


def _split_terminator(line: str) -> tuple[str, str]:
    """Split a line into (content, line-terminator) so we can rewrite the
    content while preserving the original ``\\r\\n`` / ``\\n`` / ``""`` ending."""
    content = line.rstrip("\r\n")
    return content, line[len(content):]


def deidentify_ros_text(
    text: str,
    *,
    subject_id: str = "ANON",
    pseudonymize_uids: bool = True,
) -> tuple[str, dict]:
    """De-identify ``.ros`` text. Returns ``(clean_text, mapping)``.

    ``mapping`` is the PRIVATE re-link key:
        ``{"names": {real_name: subject_id},
           "uids":  {real_uid: token},
           "dates": [(tag, real_value), ...]}``
    """
    uid_to_token: dict[str, str] = {}
    mapping: dict = {"names": {}, "uids": {}, "dates": []}

    def token_for(uid: str) -> str:
        uid = uid.strip()
        if uid not in uid_to_token:
            tok = "UID-" + hashlib.sha1(uid.encode("utf-8")).hexdigest()[:12]
            uid_to_token[uid] = tok
            mapping["uids"][uid] = tok
        return uid_to_token[uid]

    def sub_uids(s: str) -> str:
        if not pseudonymize_uids:
            return _UID_RE.sub("***", s)
        return _UID_RE.sub(lambda m: token_for(m.group(0)), s)

    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        out.append(line)
        stripped = line.strip()
        tag = stripped[1:-1] if stripped.startswith("[") and stripped.endswith("]") else None
        # The value is the next line — but only if it isn't itself a tag
        # (a tag with no value, e.g. [BEGIN], must not consume its successor).
        nxt = lines[i + 1] if i + 1 < n else None
        nxt_is_tag = bool(nxt) and nxt.strip().startswith("[")

        if tag is not None and nxt is not None and not nxt_is_tag and tag in (
            "PATIENT_NAME", "SERIE_UID", *_REDACT_TO
        ):
            content, term = _split_terminator(nxt)
            value = content.strip()
            if tag == "PATIENT_NAME":
                if value:
                    mapping["names"][value] = subject_id
                new = subject_id
            elif tag == "SERIE_UID":
                new = token_for(value) if pseudonymize_uids else "***"
            else:  # PATIENT_BIRTHDAY / SERIE_DATE
                if value:
                    mapping["dates"].append((tag, value))
                new = _REDACT_TO[tag]
            out.append(new + term)
            i += 2
            continue

        if tag == "VOLUME" and nxt is not None and not nxt_is_tag:
            content, term = _split_terminator(nxt)
            out.append(sub_uids(content) + term)
            i += 2
            continue

        i += 1

    return "".join(out), mapping


def deidentify_ros_file(
    in_path: str | Path,
    out_path: str | Path | None = None,
    *,
    subject_id: str | None = None,
    keymap_path: str | Path | None = None,
    pseudonymize_uids: bool = True,
) -> tuple[Path, dict]:
    """De-identify a ``.ros`` file.

    Args:
        in_path: the identifying ``.ros``.
        out_path: clean output. Default ``<dir>/<subject_id>.ros`` — note the
            ORIGINAL filename often contains the patient name, so the default
            output name is derived from ``subject_id``, not the input stem.
        subject_id: replaces the patient name + names the output. Default: the
            input file's PARENT folder name (the de-identified subject label you
            already use), else ``"ANON"``.
        keymap_path: where to write the PRIVATE real->token mapping (JSON).
            Default ``<out_dir>/<subject_id>_deid_keymap.json``. This file is
            PHI — keep it, never share it. Pass ``False`` to skip writing it.
        pseudonymize_uids: ``True`` = consistent ``UID-<hash>`` tokens (linkage
            preserved); ``False`` = blank UIDs to ``***`` (legacy behaviour).

    Returns ``(out_path, mapping)``.
    """
    import json

    in_path = Path(in_path).expanduser()
    if subject_id is None:
        parent = in_path.parent.name
        subject_id = parent if re.fullmatch(r"[A-Za-z0-9_-]{1,32}", parent or "") else "ANON"
    if out_path is None:
        out_path = in_path.parent / f"{subject_id}.ros"
    out_path = Path(out_path)

    text = _read_ros_text(in_path)
    clean, mapping = deidentify_ros_text(
        text, subject_id=subject_id, pseudonymize_uids=pseudonymize_uids,
    )
    _write_ros_text(out_path, clean)

    if keymap_path is not False:
        if keymap_path is None:
            keymap_path = out_path.with_name(f"{subject_id}_deid_keymap.json")
        Path(keymap_path).write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return out_path, mapping


def extract_trajectories(ros_path: str | Path) -> list[dict]:
    """Planned trajectories from a ``.ros`` as ``{name, entry, target}`` (the
    raw ROSA-frame coordinates as stored in the file). Trajectory names are
    anatomical labels, not PHI."""
    parsed = parse_ros_file(ros_path)
    return [
        {"name": t["name"], "entry": list(t["start"]), "target": list(t["end"])}
        for t in (parsed.get("trajectories") or [])
    ]


def write_trajectories_csv(ros_path: str | Path, out_csv: str | Path | None = None) -> Path:
    """Write the planned trajectories to a CSV (name + entry/target XYZ + length).

    Coordinates are the raw values stored in the ``.ros`` (ROSA/LPS frame). The
    CSV carries no patient identifiers, so it is safe to share."""
    import math

    ros_path = Path(ros_path)
    trajs = extract_trajectories(ros_path)
    if out_csv is None:
        out_csv = ros_path.with_name(f"{ros_path.stem}_trajectories.csv")
    out_csv = Path(out_csv)
    cols = ["name", "entry_x", "entry_y", "entry_z",
            "target_x", "target_y", "target_z", "length_mm"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for t in trajs:
            s, e = t["entry"], t["target"]
            length = math.dist(s, e)
            w.writerow({
                "name": t["name"],
                "entry_x": f"{s[0]:.4f}", "entry_y": f"{s[1]:.4f}", "entry_z": f"{s[2]:.4f}",
                "target_x": f"{e[0]:.4f}", "target_y": f"{e[1]:.4f}", "target_z": f"{e[2]:.4f}",
                "length_mm": f"{length:.4f}",
            })
    return out_csv


def _find_ros(case_dir: Path) -> Path:
    cands = sorted(case_dir.glob("*.ros"))
    # Prefer an un-deidentified original over any prior *_clean / <id>.ros output.
    originals = [p for p in cands if "_clean" not in p.stem]
    chosen = (originals or cands)
    if not chosen:
        raise FileNotFoundError(f"no .ros file in {case_dir}")
    return chosen[0]


def deidentify_ros_folder(
    case_dir: str | Path,
    out_dir: str | Path,
    *,
    subject_id: str | None = None,
    keymap_path: str | Path | None = None,
    pseudonymize_uids: bool = True,
    trajectories_csv: bool = True,
    log=None,
) -> tuple[Path, dict, dict]:
    """De-identify a whole ROSA case folder into a fresh, shareable ``out_dir``.

    - De-identifies the ``.ros`` → ``out_dir/<subject_id>.ros``.
    - Copies the image volumes (``.img``/``.hdr``/…) with each ``DICOM/<uid>/``
      subfolder RENAMED to the same UID token the ``.ros`` now uses — so the
      de-identified ``.ros`` still loads them (display↔series linkage intact).
    - DROPS ``DICOMFiles.zip`` (raw DICOM = full PHI) and does not copy
      screenshots / ``qc`` outputs (only the whitelisted image files + the .ros
      are emitted).
    - Writes the planned-trajectory CSV (PHI-free) into ``out_dir`` by default.
    - NEVER mutates the original. The PRIVATE keymap is written OUTSIDE
      ``out_dir`` (it is PHI — keep it, never share).

    Returns ``(out_dir, mapping, summary)`` where ``summary`` counts dirs
    renamed / images copied / zips dropped / screenshots skipped.
    """
    import json

    _log = log if callable(log) else (lambda _m: None)
    case_dir = Path(case_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    ros = _find_ros(case_dir)

    if subject_id is None:
        subject_id = case_dir.name if re.fullmatch(r"[A-Za-z0-9_-]{1,32}", case_dir.name or "") else "ANON"

    clean, mapping = deidentify_ros_text(
        _read_ros_text(ros),
        subject_id=subject_id, pseudonymize_uids=pseudonymize_uids,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_ros_text(out_dir / f"{subject_id}.ros", clean)

    uid_to_token = dict(mapping["uids"])

    def token_for(uid: str) -> str:
        if uid not in uid_to_token:
            tok = "UID-" + hashlib.sha1(uid.encode("utf-8")).hexdigest()[:12]
            uid_to_token[uid] = tok
            mapping["uids"][uid] = tok
        return uid_to_token[uid]

    summary = {"dirs_renamed": 0, "images_copied": 0, "zips_dropped": 0, "screenshots_skipped": 0}

    src_dicom = case_dir / "DICOM"
    if src_dicom.is_dir():
        for sub in sorted(p for p in src_dicom.iterdir() if p.is_dir()):
            name = sub.name
            if name in uid_to_token:
                token = uid_to_token[name]
            elif _UID_RE.fullmatch(name):
                token = token_for(name)        # UID present on disk but not in the .ros
            else:
                token = name                   # not a UID dir — keep its name
            if token != name:
                summary["dirs_renamed"] += 1
            dst = out_dir / "DICOM" / token
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(p for p in sub.iterdir() if p.is_file()):
                if f.name.startswith("."):
                    continue
                if f.suffix.lower() == ".zip":
                    summary["zips_dropped"] += 1
                    continue
                if _is_image_file(f.name):
                    shutil.copy2(f, dst / f.name)
                    summary["images_copied"] += 1
                # any other loose file is skipped (defensive — don't carry unknowns)

    # Deliberately not copied: screenshots / QC. Count them for the report.
    for pat in ("*.png", "*.jpg", "*.jpeg"):
        summary["screenshots_skipped"] += len(list(case_dir.rglob(pat)))

    if trajectories_csv:
        write_trajectories_csv(ros, out_csv=out_dir / f"{subject_id}_trajectories.csv")

    if keymap_path is not False:
        if keymap_path is None:
            # PHI key — write OUTSIDE the shareable out_dir (as a sibling).
            keymap_path = out_dir.with_name(f"{out_dir.name}_deid_keymap.json")
        Path(keymap_path).write_text(json.dumps(mapping, indent=2), encoding="utf-8")

    _log(f"[deidentify-ros] folder -> {out_dir}: {summary['dirs_renamed']} UID dirs renamed, "
         f"{summary['images_copied']} image file(s) copied, {summary['zips_dropped']} DICOM zip(s) dropped, "
         f"{summary['screenshots_skipped']} screenshot(s) excluded")
    return out_dir, mapping, summary
