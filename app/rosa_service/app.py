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

from .jobs import JobNotFound, JobRunner
from .models import (
    JobSpec, JobStatus, LabelRequest, ReviewDoc, ReviewEdit, ReviewOp, ReviewPatch,
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

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "api": API_VERSION, **_engine_info()}

    # NOTE: these are ``async`` on purpose. Sync (``def``) endpoints run in a
    # threadpool with no running event loop, so scheduling the job task (and all
    # runner state access) must happen on the loop thread.
    @app.post(f"/api/{API_VERSION}/jobs", response_model=JobStatus, status_code=201)
    async def create_job(spec: JobSpec) -> JobStatus:
        try:
            job = runner.create(spec)
        except ValueError as exc:            # unknown kind / bad params
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return job.status()

    @app.get(f"/api/{API_VERSION}/jobs", response_model=list[JobStatus])
    async def list_jobs() -> list[JobStatus]:
        return runner.list()

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
            return {"atlases": atlases,
                    "default": bundled_atlases.load_manifest()["default"]}
        except Exception as exc:  # noqa: BLE001 — engine/resources missing
            raise HTTPException(status_code=500, detail=f"atlas registry error: {exc}") from exc

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
        # The MRI: the request's, else the one provided at case creation (parent
        # pipeline's t1). Either lets labeling proceed without re-uploading.
        t1 = req.t1 or parent.params.get("t1")
        if not t1:
            raise HTTPException(status_code=409,
                                detail="no MRI (T1): provide one, or create the case with an MRI")
        spec = JobSpec(kind="label", params={
            "parent": job_id, "contacts": str(contacts), "ct": ct,
            "t1": t1, "atlas": req.atlas,
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
        elif space == "mni":
            ct = str(job.workdir / "ct_in_mni.nii.gz")
            mri = job.workdir / "mri_in_mni.nii.gz"
            if not mri.is_file() or not Path(ct).is_file():
                raise HTTPException(status_code=409, detail="no AC-PC (MNI) QC for this job")
        else:
            mri = job.workdir / "mri_in_ct.nii.gz"
            ct = job.params.get("ct")
            if not mri.is_file() or not ct or not Path(ct).is_file():
                raise HTTPException(status_code=409,
                                    detail="no registration QC (label job unfinished or no MRI)")
        try:
            from rosa_core.qc_render import render_registration_qc
            png = render_registration_qc(ct, str(mri), axis=int(axis),
                                         frac=float(frac), mode=str(mode),
                                         value=float(value), direction=str(direction))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"QC render failed: {exc}") from exc
        return Response(content=png, media_type="image/png",
                        headers={"Cache-Control": "no-store"})

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
