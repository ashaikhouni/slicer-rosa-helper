"""FastAPI service — the desktop app's contract over the rosa-agent engine.

The web UI (served at ``/``) talks only to this versioned HTTP+JSON API (under
``/api/v1``), never to the engine CLI directly. Endpoints:

  * ``GET  /``                            — the single-page wizard UI
  * ``GET  /healthz``                     — liveness + engine link
  * ``POST /api/v1/uploads``              — drag-drop a CT → a local path
  * ``POST /api/v1/jobs``                 — create + start a job (JobSpec)
  * ``GET  /api/v1/jobs``                 — list jobs (newest first)
  * ``GET  /api/v1/jobs/{id}``            — job status + artifacts
  * ``GET  /api/v1/jobs/{id}/logs``       — live log stream (SSE)
  * ``DELETE /api/v1/jobs/{id}``          — cancel a running/queued job
  * ``GET/PATCH /api/v1/jobs/{id}/review``       — editable ReviewDoc
  * ``POST /api/v1/jobs/{id}/review/export``     — write corrected TSV
  * ``GET  /api/v1/jobs/{id}/files/{path}``      — download a job file
  * ``GET  /api/v1/jobs/{id}/viewer/{path}``     — the 3D viewer
  * ``GET  /api/v1/atlases``                     — bundled atlases (label picker)
  * ``POST /api/v1/jobs/{id}/label``             — label a run's contacts (MRI+atlas)
  * ``GET  /api/v1/jobs/{id}/labels``            — proposed labels + reg QC
  * ``POST /api/v1/jobs/{id}/labels/approve``    — apply labels to parent's ReviewDoc

Jobs run as supervised subprocesses in an auditable per-job dir (see
``jobs.JobRunner``).
"""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import (
    FileResponse, RedirectResponse, Response, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .editor_payload import ensure_cache
from .jobs import JobNotFound, JobRunner, _engine_base
from .models import (
    DicomRequest, ImportRequest, JobSpec, JobStatus, LabelRequest, ReviewDoc,
    ReviewEdit, ReviewOp, ReviewPatch,
)
from .review import ReviewStore, export_contacts

API_VERSION = "v1"


def _engine_info() -> dict:
    try:
        import importlib.metadata as md
        version = md.version("rosa-agent")
    except Exception:  # noqa: BLE001
        version = "unknown"
    try:
        import rosa_core  # noqa: F401  — the engine import must succeed
        engine_ok = True
    except Exception:  # noqa: BLE001
        engine_ok = False
    return {"engine": "rosa-agent", "engine_version": version, "engine_import_ok": engine_ok}


def _read_proposed_labels(job_dir: Path) -> list[dict]:
    """Parse a label job's ``contacts_labeled.tsv`` into per-contact regions.

    The engine ``label`` command writes the region under ``closest_label``,
    keyed by ``trajectory`` + ``contact_index``. Raises FileNotFoundError if
    the job hasn't produced it (e.g. still running / failed).
    """
    from rosa_agent.io.trajectory_io import read_tsv_rows
    tsv = job_dir / "contacts_labeled.tsv"
    if not tsv.is_file():
        raise FileNotFoundError("no contacts_labeled.tsv (label job not finished?)")
    out: list[dict] = []
    for r in read_tsv_rows(tsv):
        region = (r.get("closest_label") or "").strip()
        if region in ("", "Unknown", "None"):
            region = None
        try:
            idx = int(float(r.get("contact_index") or 0))
        except ValueError:
            continue
        out.append({
            "shank": (r.get("trajectory") or "").strip(),
            "index": idx,
            "name": r.get("contact_label") or "",
            "region": region,
        })
    return out


def _regions_from_review(case_dir) -> dict:
    """Map channel name → anatomical region from a case's saved review.json
    (the labeled/edited state), for the cohort viewer's per-contact tooltip."""
    import json
    rj = Path(case_dir) / "review.json"
    if not rj.is_file():
        return {}
    try:
        doc = json.loads(rj.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    out = {}
    for shank in doc.get("shanks", []):
        for c in shank.get("contacts", []):
            if c.get("name") and c.get("region"):
                out[c["name"]] = c["region"]
    return out


def _cohort_hue(case_id: str) -> str:
    """A stable, distinct-ish per-subject color (hashed case id → HSV → hex), so a
    subject keeps its color as the cohort grows."""
    import colorsys
    import hashlib
    h = (int(hashlib.md5(case_id.encode()).hexdigest()[:8], 16) % 997) / 997.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.96)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))


def create_app(*, work_root: str | Path | None = None, max_concurrent: int = 1) -> FastAPI:
    if work_root is None:
        work_root = os.environ.get("ROSA_APP_WORKDIR") \
            or tempfile.mkdtemp(prefix="rosa-app-jobs-")
    runner = JobRunner(work_root, max_concurrent=int(
        os.environ.get("ROSA_APP_MAX_CONCURRENT", max_concurrent)))

    reviews = ReviewStore()

    app = FastAPI(title="ROSA app service", version=API_VERSION)
    app.state.runner = runner
    app.state.reviews = reviews

    @app.middleware("http")
    async def _no_store(request, call_next):
        # A local app under active iteration: never let the browser serve a
        # stale JS/HTML/viewer from cache (no CDN benefit on localhost). Avoids
        # "I edited the UI but the browser runs the old one" confusion.
        resp = await call_next(request)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _job_or_404(job_id: str):
        try:
            return runner.get(job_id)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}") from exc

    def _dedup_or_hash(ct_path: str, force: bool) -> str | None:
        """Fingerprint the CT; unless ``force``, 409 if a finished case already
        uses the same CT (so the UI can offer to open it). Returns the hash to
        stamp on the new case's params (enables future dedup)."""
        from .cases import ct_fingerprint
        ct_hash = ct_fingerprint(ct_path)
        if ct_hash and not force:
            for st in runner.list():                 # newest-first
                if st.kind in ("pipeline", "import") and st.state == "succeeded" \
                        and runner.get(st.id).params.get("ct_hash") == ct_hash:
                    raise HTTPException(status_code=409, detail={
                        "message": f"This CT already has a case "
                                   f"({st.label or st.id[:8]}).",
                        "existing": {"id": st.id, "label": st.label,
                                     "created_at": st.created_at}})
        return ct_hash

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "api": API_VERSION, **_engine_info()}

    # NOTE: these are ``async`` on purpose. Sync (``def``) endpoints run in a
    # threadpool with no running event loop, so scheduling the job task (and all
    # runner state access) must happen on the loop thread.
    @app.post(f"/api/{API_VERSION}/jobs", response_model=JobStatus, status_code=201)
    async def create_job(spec: JobSpec) -> JobStatus:
        # A new pipeline case: refuse a duplicate CT (unless params.force) and
        # stamp the CT hash so future runs can spot the duplicate.
        if spec.kind == "pipeline" and spec.params.get("ct"):
            spec.params["ct_hash"] = _dedup_or_hash(
                spec.params["ct"], bool(spec.params.get("force")))
        try:
            job = runner.create(spec)
        except ValueError as exc:            # unknown kind / bad params
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return job.status()

    @app.delete(f"/api/{API_VERSION}/cases/{{job_id}}", status_code=204)
    async def delete_case(job_id: str) -> Response:
        """Delete a finished case (and its label jobs + workdirs)."""
        try:
            runner.delete(job_id)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=f"no case {job_id!r}") from exc
        except ValueError as exc:            # still running
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return Response(status_code=204)

    @app.post(f"/api/{API_VERSION}/jobs/import", status_code=201)
    async def import_localization(req: ImportRequest) -> dict:
        """Create a reviewable case from a localization computed elsewhere.

        Validates that the contacts actually fit the CT (parity content check)
        before creating a view-results-only job. A clearly-wrong CT/TSV pairing
        (contacts outside the volume / off metal) is rejected with 422; a weak
        but usable match returns a ``check`` the UI can ask the user to confirm.
        """
        from .import_check import check_localization
        for what, p in (("CT", req.ct), ("contacts TSV", req.contacts),
                        ("trajectories TSV", req.trajectories)):
            if not p or not Path(p).is_file():
                raise HTTPException(status_code=422, detail=f"{what} not found: {p!r}")
        try:
            check = check_localization(req.ct, req.contacts)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if check["verdict"] == "red":
            raise HTTPException(status_code=422,
                                detail={"message": check["reason"], "check": check})
        # Valid pairing — now refuse a duplicate CT (unless force) and stamp the hash.
        ct_hash = _dedup_or_hash(req.ct, req.force)
        spec = JobSpec(kind="import", params={
            "ct": req.ct, "contacts": req.contacts, "trajectories": req.trajectories,
            "label": req.label or "case", "surface": req.surface or "auto",
            "ct_hash": ct_hash, **({"t1": req.t1} if req.t1 else {})})
        try:
            job = runner.create(spec)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"job": job.status().model_dump(), "check": check}

    @app.get(f"/api/{API_VERSION}/jobs", response_model=list[JobStatus])
    async def list_jobs() -> list[JobStatus]:
        return runner.list()

    @app.get(f"/api/{API_VERSION}/cases")
    async def list_cases() -> list[dict]:
        """Reviewable cases for the home screen — pipeline/import runs that
        finished, each enriched with electrode/contact counts + MRI/label state
        so the list is informative without opening a case. Newest first."""
        from .cases import summarize_case
        # newest-first, so the first succeeded label child seen per case is its
        # currently-active atlas (drives the Atlas column + MNI-poolable state).
        atlas_by_case: dict[str, str] = {}
        for st in runner.list():
            if st.kind == "label" and st.state == "succeeded" and st.parent:
                atlas_by_case.setdefault(st.parent, st.atlas or "")
        out = []
        for st in runner.list():                 # newest-first
            if st.kind in ("pipeline", "import") and st.state == "succeeded":
                job = runner.get(st.id)
                out.append(summarize_case(
                    st, job.workdir,
                    ct_hash=job.params.get("ct_hash"),
                    atlas=atlas_by_case.get(st.id)))
        return out

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}", response_model=JobStatus)
    async def get_job(job_id: str) -> JobStatus:
        try:
            return runner.get(job_id).status()
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}") from exc

    @app.delete(f"/api/{API_VERSION}/jobs/{{job_id}}", response_model=JobStatus)
    async def cancel_job(job_id: str) -> JobStatus:
        try:
            job = await runner.cancel(job_id)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}") from exc
        return job.status()

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/logs")
    async def job_logs(job_id: str) -> StreamingResponse:
        try:
            runner.get(job_id)
        except JobNotFound as exc:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}") from exc

        async def event_stream():
            async for line in runner.stream_logs(job_id):
                yield f"data: {line}\n\n"
            yield "event: end\ndata: \n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ---- review & edit ---------------------------------------------

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/review", response_model=ReviewDoc)
    async def get_review(job_id: str) -> ReviewDoc:
        job = _job_or_404(job_id)
        try:
            return reviews.get_or_build(job_id, job.workdir)
        except FileNotFoundError as exc:      # run produced no contacts yet
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.patch(f"/api/{API_VERSION}/jobs/{{job_id}}/review", response_model=ReviewDoc)
    async def patch_review(job_id: str, patch: ReviewPatch) -> ReviewDoc:
        job = _job_or_404(job_id)
        try:
            return reviews.apply(job_id, job.workdir, patch.ops)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:             # bad edit target / op
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(f"/api/{API_VERSION}/jobs/{{job_id}}/review/export")
    async def export_review(job_id: str) -> dict:
        job = _job_or_404(job_id)
        try:
            doc = reviews.get_or_build(job_id, job.workdir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        out = job.workdir / "contacts_reviewed.tsv"
        n = export_contacts(doc, out)
        return {"path": str(out), "rel_path": out.name, "n_contacts": n}

    # ---- anatomical labeling (MRI + bundled atlas → proposed labels) ----

    @app.get(f"/api/{API_VERSION}/atlases")
    async def list_atlases() -> dict:
        """Atlases available for labeling (for the picker). Prepends FastSurfer —
        a subject-specific labeler (native aparc+aseg, no MNI warp) — when its
        runtime is detected; otherwise it's greyed out (bundled MNI atlases are
        the always-available path)."""
        try:
            from rosa_core import bundled_atlases
            atlases = list(bundled_atlases.list_atlases())
            try:
                from rosa_detect.services.fastsurfer_seg import find_fastsurfer
                fs_available = find_fastsurfer()[0] is not None
            except Exception:  # noqa: BLE001
                fs_available = False
            atlases.insert(0, {
                "id": "fastsurfer", "name": "FastSurfer (native)",
                "available": fs_available, "license_tier": "permissive",
                "license": "Apache-2.0 (FastSurfer, no FreeSurfer license for seg-only)",
                "coverage": "Whole-brain cortical (DKT) + subcortical, subject-specific",
                "is_default": False,
            })
            # deepmriprep native-space atlases (id "dmp:<name>"): already in the
            # patient's T1 space, so labeled via a T1→CT rigid — no MNI warp. Only
            # offered when a deepmriprep runtime is reachable.
            try:
                from rosa_detect.services.deepmriprep_seg import (
                    deepmriprep_available, DEEPMRIPREP_ATLAS_INFO,
                )
                dmp_ok = deepmriprep_available()
            except Exception:  # noqa: BLE001
                dmp_ok, DEEPMRIPREP_ATLAS_INFO = False, {}
            for name, (disp, coverage, tier) in DEEPMRIPREP_ATLAS_INFO.items():
                atlases.append({
                    "id": f"dmp:{name}", "name": f"{disp} (deepmriprep)",
                    "available": dmp_ok, "license_tier": tier,
                    "license": "deepmriprep code MIT; atlas data academic — cite the source atlas",
                    "coverage": coverage, "is_default": False,
                })
            # THOMAS: a BYO deep-structure atlas. The user loads a THOMAS output
            # folder; its reference MRI is registered → the CT and the nuclei are
            # labeled + rendered as 3-D meshes. Always offered (needs no runtime),
            # flagged byo so the UI knows to prompt for the folder.
            atlases.append({
                "id": "thomas", "name": "THOMAS thalamic nuclei (load folder…)",
                "available": True, "byo": True, "license_tier": "byo",
                "license": "your own THOMAS segmentation output",
                "coverage": "Deep: thalamic + subcortical nuclei (per hemisphere)",
                "is_default": False,
            })
            return {"atlases": atlases,
                    "default": bundled_atlases.load_manifest()["default"]}
        except Exception as exc:  # noqa: BLE001 — engine/resources missing
            raise HTTPException(status_code=500, detail=f"atlas registry error: {exc}") from exc

    @app.get(f"/api/{API_VERSION}/electrode-models")
    async def list_electrode_models() -> dict:
        """The bundled electrode library, grouped by type — so the new-case
        screen can offer per-type checkboxes to constrain detection/placement to
        the electrode types a site actually uses (passed to `contacts
        --electrode-types`)."""
        try:
            from rosa_core.electrode_models import (
                load_electrode_library, label_map, manufacturer_of)
            lib = load_electrode_library()
            top = lib.get("vendor")
            groups: dict[str, list] = {}
            manu: dict[str, str] = {}
            for m in lib.get("models", []):
                t = str(m.get("type") or "?")
                groups.setdefault(t, []).append(
                    {"id": m.get("id"), "contact_count": m.get("contact_count")})
                # type codes are manufacturer-specific → one manufacturer per type
                manu.setdefault(t, (manufacturer_of(m, top) or "").split()[0])
            types = [{"type": t, "manufacturer": manu.get(t, ""), "count": len(ms),
                      "models": ms,
                      "contact_counts": sorted({x["contact_count"] for x in ms if x["contact_count"]})}
                     for t, ms in groups.items()]
            types.sort(key=lambda x: (x["manufacturer"], x["type"]))  # cluster by manufacturer
            # id → "Manufacturer Reference" (e.g. "DIXI D08-05AM") for friendly
            # display of the placed electrode model in the review list.
            return {"vendor": top, "series": lib.get("series"),
                    "types": types, "labels": label_map(lib)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"electrode library error: {exc}") from exc

    @app.post(f"/api/{API_VERSION}/jobs/{{job_id}}/label",
              response_model=JobStatus, status_code=201)
    async def create_label_job(job_id: str, req: LabelRequest) -> JobStatus:
        """Start a labeling job for a pipeline run's contacts (through the MRI).

        Registration goes MNI→T1(MRI)→CT; labels are *proposed* and only reach
        the ReviewDoc once approved (``/labels/approve``).
        """
        parent = _job_or_404(job_id)
        contacts = parent.workdir / "contacts.tsv"
        if not contacts.is_file():
            raise HTTPException(status_code=409,
                                detail="parent job has no contacts.tsv (run a pipeline job first)")
        ct = parent.params.get("ct")
        if not ct:
            raise HTTPException(status_code=409, detail="parent job has no CT recorded")
        is_thomas = str(req.atlas or "").startswith("thomas")
        # THOMAS is a BYO atlas: it brings its own reference MRI (registered → CT),
        # so a patient T1 is optional (it only adds the cortical Ghost surface). It
        # needs the uploaded folder instead.
        thomas_dir = None
        if is_thomas:
            if not req.thomas_dir:
                raise HTTPException(status_code=422,
                                    detail="THOMAS atlas needs a folder (thomas_dir)")
            tdir = Path(req.thomas_dir).expanduser()
            if not ((tdir / "left").is_dir() or (tdir / "right").is_dir()):
                raise HTTPException(
                    status_code=422,
                    detail="not a THOMAS output dir — expected a left/ or right/ subfolder")
            thomas_dir = str(tdir)
        # The MRI: the request's, else the one provided at case creation (parent
        # pipeline's t1). Either lets labeling proceed without re-uploading.
        t1 = req.t1 or parent.params.get("t1")
        if not t1 and not is_thomas:
            raise HTTPException(status_code=409,
                                detail="no MRI (T1): provide one, or create the case with an MRI")
        spec = JobSpec(kind="label", params={
            "parent": job_id, "contacts": str(contacts), "ct": ct,
            "t1": t1, "atlas": req.atlas, "thomas_dir": thomas_dir,
            # Reuse the parent pipeline's surface backend so labeling meshes/loads
            # the SAME brain_surface_*.npz rather than a different one.
            "surface": parent.params.get("surface", "auto"),
            # Cache registrations in the parent case dir so labeling more atlases
            # reuses T1→CT (once/case) + MNI→T1 (once/space) instead of re-running.
            "regcache": str(parent.workdir / "regcache")})
        try:
            job = runner.create(spec)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return job.status()

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/labels")
    async def proposed_labels(job_id: str) -> dict:
        """Proposed labels from a finished label job (not yet applied)."""
        job = _job_or_404(job_id)
        try:
            contacts = _read_proposed_labels(job.workdir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        n_labeled = sum(1 for c in contacts if c["region"])
        return {
            "parent": job.params.get("parent"),
            "atlas": job.params.get("atlas"),
            "n_contacts": len(contacts),
            "n_labeled": n_labeled,
            "has_mri_qc": (job.workdir / "mri_in_ct.nii.gz").is_file(),
            "has_mni_qc": (job.workdir / "ct_in_mni.nii.gz").is_file()
            and (job.workdir / "mri_in_mni.nii.gz").is_file(),
            "contacts": contacts,
        }

    @app.post(f"/api/{API_VERSION}/jobs/{{job_id}}/labels/approve",
              response_model=ReviewDoc)
    async def approve_labels(job_id: str) -> ReviewDoc:
        """Apply a label job's proposed regions to the parent's ReviewDoc."""
        job = _job_or_404(job_id)
        parent_id = job.params.get("parent")
        if not parent_id:
            raise HTTPException(status_code=409, detail="label job has no parent recorded")
        parent = _job_or_404(parent_id)
        try:
            contacts = _read_proposed_labels(job.workdir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        edits = [ReviewEdit(op=ReviewOp.relabel_contact, shank=c["shank"],
                            index=c["index"], region=c["region"])
                 for c in contacts if c["region"]]
        try:
            return reviews.apply(parent_id, parent.workdir, edits)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # NOTE: sync ``def`` on purpose — Starlette runs it in a threadpool, so the
    # CPU-bound render (numpy + PNG) does NOT block the event loop (job polls,
    # SSE logs stay responsive even while sliders are dragged). It only reads
    # runner state (no job scheduling), which is safe off the loop.
    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/qc")
    def registration_qc(job_id: str, axis: int = 2, frac: float = 0.5,
                        mode: str = "color", value: float = 0.5,
                        direction: str = "h", space: str = "ct") -> Response:
        """Render a composited CT↔MRI slice (PNG) so registration can be eyed.

        ``space``: ``ct`` slices in the CT's native frame; ``mni`` slices in the
        atlas (MNI, AC-PC aligned) grid for standard neuroanatomical planes.
        Compositing (opacity/wipe/color) is server-side, so the UI just swaps
        one image per plane.
        """
        job = _job_or_404(job_id)
        if space == "atlas":
            # Verify the MNI→T1 (atlas) registration: overlay the bundled MNI
            # TEMPLATE the atlas is defined on against the patient MRI warped
            # into MNI. Both live on the same template grid, so they composite
            # directly. Alignment here = accurate atlas label placement.
            from rosa_core import bundled_atlases
            atlas = job.params.get("atlas") or "cerebra"
            try:
                assets = bundled_atlases.resolve(atlas)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=409,
                                    detail=f"atlas template unavailable: {exc}") from exc
            ct = str(assets.template_path)            # MNI template (shown as CT/magenta)
            mri = job.workdir / "mri_in_mni.nii.gz"   # patient MRI in MNI (green)
            if not mri.is_file() or not Path(ct).is_file():
                raise HTTPException(status_code=409, detail="no atlas-registration QC for this job")
        elif space == "acpc":
            # Rigid AC-PC reorientation (built by POST /acpc, cached in regcache):
            # CT + MRI resampled onto the upright MNI grid so the check slices in
            # standard neuroanatomical planes without deforming the geometry.
            rc = job.workdir / "regcache"
            ct = str(rc / "ct_in_acpc.nii.gz")
            mri = rc / "mri_in_acpc.nii.gz"
            if not mri.is_file() or not Path(ct).is_file():
                raise HTTPException(status_code=409,
                                    detail="no AC-PC reorientation yet (POST /acpc first)")
        elif space == "mni":
            ct = str(job.workdir / "ct_in_mni.nii.gz")
            mri = job.workdir / "mri_in_mni.nii.gz"
            if not mri.is_file() or not Path(ct).is_file():
                raise HTTPException(status_code=409, detail="no AC-PC (MNI) QC for this job")
        else:
            mri = job.workdir / "mri_in_ct.nii.gz"
            if not mri.is_file():
                # Fall back to the pipeline's cached MRI-in-CT so the CT↔MRI check
                # works straight from case creation — before any labeling.
                alt = job.workdir / "regcache" / "brain_mri_in_ct.nii.gz"
                if alt.is_file():
                    mri = alt
            ct = job.params.get("ct")
            if not mri.is_file() or not ct or not Path(ct).is_file():
                raise HTTPException(status_code=409,
                                    detail="no CT↔MRI registration yet (add an MRI at case creation, or run a label)")
        try:
            from rosa_core.qc_render import render_registration_qc
            png = render_registration_qc(ct, str(mri), axis=int(axis),
                                         frac=float(frac), mode=str(mode),
                                         value=float(value), direction=str(direction))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"QC render failed: {exc}") from exc
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    # Sync def → threadpool (a ~20-40s rigid registration; must not block the
    # event loop). Idempotent: the reoriented volumes are cached in regcache.
    @app.post(f"/api/{API_VERSION}/jobs/{{job_id}}/acpc")
    def build_acpc(job_id: str) -> dict:
        """Build the rigid AC-PC reorientation for a case (upright slice planes).

        Resamples the CT + MRI onto the AC-PC-aligned MNI grid so the
        registration check reads as standard axial/coronal/sagittal planes
        instead of the tilted acquisition. Needs the case MRI + the cached
        ``t1_to_ct.tfm`` (present for MRI cases). Returns
        ``{ready: bool, reason?}`` — ``ready=False`` (not an error) when the case
        simply lacks the inputs, so the UI can fall back to the native view.
        """
        job = _job_or_404(job_id)
        rc = job.workdir / "regcache"
        out_ct = rc / "ct_in_acpc.nii.gz"
        out_mri = rc / "mri_in_acpc.nii.gz"
        if out_ct.is_file() and out_mri.is_file():
            return {"ready": True, "cached": True}
        ct = job.params.get("ct")
        t1 = job.params.get("t1")
        tfm = rc / "t1_to_ct.tfm"        # reused if present, else computed + cached
        mask = rc / "brain_mask_native.nii.gz"
        if not (ct and Path(ct).is_file()):
            raise HTTPException(status_code=409, detail="case CT not found")
        if not (t1 and Path(t1).is_file()):
            return {"ready": False, "reason": "no MRI on this case"}
        try:
            from rosa_core.acpc import reorient_to_acpc
            rc.mkdir(parents=True, exist_ok=True)
            info = reorient_to_acpc(ct, t1, str(tfm), str(out_ct), str(out_mri),
                                    brain_mask_path=str(mask) if mask.is_file() else None)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"AC-PC reorient failed: {exc}") from exc
        return {"ready": True, "cached": False, "metric": info["metric"]}

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/files/{{path:path}}")
    async def job_file(job_id: str, path: str) -> FileResponse:
        """Download a file from the job dir (e.g. the exported TSV)."""
        job = _job_or_404(job_id)
        root = job.workdir.resolve()
        target = (root / path).resolve()
        if target != root and root not in target.parents:
            raise HTTPException(status_code=400, detail="invalid path")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"not found: {path}")
        return FileResponse(target, filename=target.name)

    # ---- viewer (served-mode static dir produced by view-results) ----

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/viewer")
    async def viewer_root(job_id: str) -> RedirectResponse:
        _job_or_404(job_id)
        # Trailing slash so the viewer's relative fetches (scene.glb, …) resolve.
        return RedirectResponse(
            url=f"/api/{API_VERSION}/jobs/{job_id}/viewer/", status_code=307)

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/viewer/{{path:path}}")
    async def viewer_files(job_id: str, path: str = "") -> FileResponse:
        job = _job_or_404(job_id)
        viewer_dir = (job.workdir / "viewer").resolve()
        if not viewer_dir.is_dir():
            raise HTTPException(status_code=404, detail="no viewer for this job (run a pipeline job)")
        target = (viewer_dir / (path or "index.html")).resolve()
        # Path-traversal guard: target must stay inside the viewer dir.
        if target != viewer_dir and viewer_dir not in target.parents:
            raise HTTPException(status_code=400, detail="invalid path")
        if not target.is_file():
            raise HTTPException(status_code=404, detail=f"not found: {path or 'index.html'}")
        return FileResponse(target)

    # ---- trajectory editor (client-side reslicer; CT + plan served locally) ----

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/editor")
    async def editor_root(job_id: str) -> RedirectResponse:
        _job_or_404(job_id)
        return RedirectResponse(
            url=f"/api/{API_VERSION}/jobs/{job_id}/editor/", status_code=307)

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/editor/")
    async def editor_index(job_id: str) -> FileResponse:
        _job_or_404(job_id)
        page = Path(__file__).resolve().parent / "web" / "editor" / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="editor asset missing")
        return FileResponse(page)

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/editor/plan")
    async def editor_plan(job_id: str) -> FileResponse:
        job = _job_or_404(job_id)
        try:
            ensure_cache(job.workdir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return FileResponse(job.workdir / "editor_plan.json", media_type="application/json")

    @app.get(f"/api/{API_VERSION}/jobs/{{job_id}}/editor/volume")
    async def editor_volume(job_id: str) -> FileResponse:
        job = _job_or_404(job_id)
        try:
            ensure_cache(job.workdir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return FileResponse(job.workdir / "editor_ct.i16", media_type="application/octet-stream")

    @app.post(f"/api/{API_VERSION}/jobs/{{job_id}}/editor/plan")
    async def save_editor_plan(job_id: str, plan: dict) -> dict:
        """Persist an edited plan: rewrite trajectories.tsv + regenerate
        contacts.tsv from the combs, carry labels onto the new contacts, and
        kick off a viewer rebuild so Review reflects the edit."""
        from .editor_writeback import write_plan
        job = _job_or_404(job_id)
        try:
            summary = write_plan(job.workdir, plan)
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=f"bad plan: {exc}") from exc
        reviews.rebuild_preserving_labels(job_id, job.workdir, renames=summary.get("renames"))
        ct = job.params.get("ct")
        rebuild = None
        if ct:                              # rebuild the 3D scene from the new TSVs
            params = {
                "case_dir": str(job.workdir), "ct": ct,
                "label": job.params.get("label", "case"),
                "surface": job.params.get("surface", "auto"),
                **({"t1": job.params["t1"]} if job.params.get("t1") else {})}
            # Carry the case's atlas through the rebuild so it doesn't vanish after
            # an edit — especially THOMAS deep-structure meshes, which have no
            # cached-surface fallback. The atlas artifacts live in the case's
            # newest succeeded label job's workdir; re-pass what that job passed.
            for st in runner.list():        # newest-first → the currently-shown atlas
                if st.kind != "label" or st.parent != job_id or st.state != "succeeded":
                    continue
                atlas = str(st.atlas or "")
                labelmap = runner.get(st.id).workdir / "atlas_in_ct.nii.gz"
                if atlas and atlas != "fastsurfer" and labelmap.is_file():
                    is_thomas = atlas.startswith("thomas")
                    params["atlas_labelmap"] = str(labelmap)
                    params["atlas_name"] = "THOMAS" if is_thomas else atlas
                    if is_thomas:
                        params["structure_meshes"] = str(labelmap)
                        lut = labelmap.with_name("atlas_in_ct.nii.gz.lut.json")
                        if lut.is_file():
                            params["structure_lut"] = str(lut)
                break                       # only the newest succeeded label job
            spec = JobSpec(kind="rebuild", params=params)
            try:
                rebuild = runner.create(spec).status().model_dump()
            except ValueError:
                rebuild = None
        return {**summary, "rebuild_job": rebuild}

    # ---- uploads (browser drag-drop → a local path a job can consume) ----

    @app.post(f"/api/{API_VERSION}/uploads")
    async def upload(file: UploadFile = File(...)) -> dict:
        uploads = Path(work_root) / "_uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        name = Path(file.filename or "upload.nii.gz").name  # strip any path
        dest = uploads / f"{uuid.uuid4().hex[:8]}_{name}"
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
        return {"path": str(dest), "name": name, "bytes": dest.stat().st_size}

    @app.post(f"/api/{API_VERSION}/uploads/dir")
    async def upload_dir(files: list[UploadFile] = File(...)) -> dict:
        """Upload a folder tree (browser directory pick) and return its root path.

        Each file's *filename* carries its relative path (the browser sends
        ``webkitRelativePath`` there, e.g. ``T1/left/11-CM.nii.gz``); we rebuild
        the tree under ``_uploads/<uuid>/`` so a directory-consuming job
        (import-thomas) can point at the root. The browser side sends only the
        files that job needs, so this stays lean.
        """
        if not files:
            raise HTTPException(status_code=400, detail="no files uploaded")
        base = Path(work_root) / "_uploads" / uuid.uuid4().hex[:8]
        roots: set[str] = set()
        n = 0
        for f in files:
            # The filename IS the relative path; drop traversal/absolute segments.
            parts = [p for p in Path(str(f.filename or "")).parts if p not in ("..", "/", "")]
            if not parts:
                continue
            roots.add(parts[0])
            dest = base.joinpath(*parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            n += 1
        if not n:
            raise HTTPException(status_code=400, detail="no usable files in upload")
        # Single common top folder → return it; otherwise the upload root.
        root = base / next(iter(roots)) if len(roots) == 1 else base
        return {"path": str(root), "n_files": n}

    @app.post(f"/api/{API_VERSION}/dicom-to-nifti")
    async def dicom_to_nifti(req: DicomRequest) -> dict:
        """Convert a DICOM series folder (on this machine) to a de-identified
        NIfTI and return its path, ready to use as the case CT. Dropping DICOM →
        NIfTI strips the PHI headers; the source DICOM is left untouched."""
        src = Path(req.dicom_dir).expanduser()
        if not src.is_dir():
            raise HTTPException(status_code=422, detail=f"DICOM folder not found: {src}")
        out_dir = Path(work_root) / "_uploads" / uuid.uuid4().hex[:8]
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "ct.nii.gz"
        argv = [*_engine_base(), "dicom-to-nifti", str(src), "-o", str(out)]
        if req.series_uid:
            argv += ["--series-uid", req.series_uid]
        proc = await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        stdout, _ = await proc.communicate()
        log = (stdout or b"").decode(errors="replace")
        if proc.returncode != 0 or not out.is_file():
            raise HTTPException(status_code=422,
                                detail=f"DICOM conversion failed: {log.strip()[-400:]}")
        return {"path": str(out), "name": "ct.nii.gz", "deidentified": True, "log": log}

    # ---- cohort MNI group viewer: shared three.js assets + pooled contacts ----
    import rosa_agent
    three_dir = Path(rosa_agent.__file__).resolve().parent / "commands" / "viewer_assets" / "three"
    if three_dir.is_dir():                        # mounted before "/" so it wins
        app.mount("/assets/three", StaticFiles(directory=str(three_dir)), name="three")

    @app.get(f"/api/{API_VERSION}/cohort/brain.glb")
    async def cohort_brain() -> FileResponse:
        """The shared MNI152NLin2009cSym glass brain (prebuilt, committed)."""
        import rosa_core
        glb = (Path(rosa_core.__file__).resolve().parent / "resources"
               / "atlases" / "templates" / "mni152_2009c_sym_glass.glb")
        if not glb.is_file():
            raise HTTPException(status_code=404, detail="MNI glass brain not built")
        return FileResponse(str(glb), media_type="model/gltf-binary")

    @app.get(f"/api/{API_VERSION}/cohort/contacts")
    async def cohort_contacts() -> dict:
        """Every MNI-poolable case's contacts pooled in MNI152NLin2009cSym for the
        group viewer. Lazily (re)computes each case's contacts_mni cache and joins
        the anatomical region from its review.json."""
        from rosa_core import cohort as cohort_mod
        subjects = []
        for st in runner.list():                  # newest-first
            if not (st.kind in ("pipeline", "import") and st.state == "succeeded"):
                continue
            case_dir = runner.get(st.id).workdir
            if not cohort_mod.mni_transforms_present(case_dir / "regcache"):
                continue
            try:
                rows = cohort_mod.ensure_contacts_mni(case_dir)
            except Exception:                     # noqa: BLE001 — one bad case can't sink the cohort
                continue
            if not rows:
                continue
            regions = _regions_from_review(case_dir)
            contacts = [[r["mni_x"], r["mni_y"], r["mni_z"],
                         regions.get(r.get("name", "")) or "", r.get("hemisphere", ""),
                         r.get("name", "")]
                        for r in rows]
            subjects.append({"id": st.id, "label": st.label or st.id[:8],
                             "color": _cohort_hue(st.id), "contacts": contacts})
        return {"space": cohort_mod.POOL_SPACE, "subjects": subjects}

    # ---- the web UI (single-page wizard), served at / ----
    # Mounted LAST so the /api and /healthz routes above take precedence; the
    # SPA + its assets are served for everything else. html=True serves
    # index.html at /.
    web_dir = Path(__file__).resolve().parent / "web"
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

    return app


# Module-level app for ``uvicorn rosa_service.app:app``.
app = create_app()
