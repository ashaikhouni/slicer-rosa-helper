"""FastAPI service — the desktop app's contract over the rosa-agent engine.

The UI talks only to this versioned HTTP+JSON API (under ``/api/v1``), never to
the engine CLI directly. Endpoints:

  * ``GET  /healthz``                     — liveness + engine link
  * ``POST /api/v1/jobs``                 — create + start a job (JobSpec)
  * ``GET  /api/v1/jobs``                 — list jobs (newest first)
  * ``GET  /api/v1/jobs/{id}``            — job status + artifacts
  * ``GET  /api/v1/jobs/{id}/logs``       — live log stream (SSE)
  * ``DELETE /api/v1/jobs/{id}``          — cancel a running/queued job

Jobs run as supervised subprocesses in an auditable per-job dir (see
``jobs.JobRunner``). The ReviewDoc DTO + static viewer serving land later.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .jobs import JobNotFound, JobRunner
from .models import JobSpec, JobStatus, ReviewDoc, ReviewPatch
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

    return app


# Module-level app for ``uvicorn rosa_service.app:app``.
app = create_app()
