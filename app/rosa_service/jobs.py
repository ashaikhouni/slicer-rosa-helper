"""Job runner — each job is a supervised subprocess with an auditable job dir.

Design (per the app plan):
  * **Subprocess per job** — cancellation is a clean ``terminate()``, a native
    crash (SimpleITK/torch) can't take down the service, and logs are captured
    per job.
  * **Bounded concurrency** — a semaphore caps concurrent jobs (a big CT +
    torch is GB-scale RAM; don't let the UI fire unbounded work). Extra jobs sit
    ``queued`` until a slot frees.
  * **Auditable job dir** — every job gets ``<work_root>/<id>/`` holding
    ``job.log`` + a ``manifest.json`` (id, kind, argv, state, times, exit code)
    written on completion. Outputs the engine writes there become artifacts.
  * **Live logs** — stdout+stderr stream to subscribers (SSE) and to the log
    file simultaneously.

The UI never sends a command line: it sends a :class:`JobSpec` (kind + params)
and :func:`build_command` maps it to argv here, so command construction stays
server-side (no injection surface) and decoupled from the engine's CLI flags.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

from .models import Artifact, JobSpec, JobState, JobStatus


class JobNotFound(KeyError):
    pass


def build_command(spec: JobSpec, workdir: Path) -> list[str]:
    """Map a :class:`JobSpec` to the argv to run in ``workdir``.

    Only known kinds are runnable — the UI can't inject arbitrary commands.
    ``selftest*`` kinds are fast synthetic jobs used to exercise the runner
    (they need no data and run in CI). ``pipeline`` maps to the real engine.
    """
    kind = spec.kind
    if kind == "selftest":
        steps = int(spec.params.get("steps", 3))
        script = (
            "import time\n"
            f"for i in range({steps}):\n"
            f"    print('step', i + 1, 'of', {steps}, flush=True)\n"
            "    time.sleep(0.02)\n"
            "print('done', flush=True)\n"
        )
        return [sys.executable, "-u", "-c", script]
    if kind == "selftest-hang":
        # Never exits on its own — used to test cancellation.
        return [sys.executable, "-u", "-c", "import time\nwhile True:\n    time.sleep(0.05)\n"]
    if kind == "selftest-fail":
        return [sys.executable, "-u", "-c", "import sys\nprint('boom', flush=True)\nsys.exit(3)\n"]
    if kind == "pipeline":
        ct = spec.params.get("ct")
        if not ct:
            raise ValueError("pipeline job requires params.ct")
        # Real engine run. `python -m rosa_agent` works whether the engine is
        # pip-installed or on PYTHONPATH; in the frozen app this becomes the
        # frozen exe re-invoked. Outputs land in the job dir.
        argv = [sys.executable, "-u", "-m", "rosa_agent", "pipeline", str(ct),
                "--out-dir", str(workdir)]
        for flag in ("t1", "ref_volume", "mask_backend"):
            if spec.params.get(flag):
                argv += [f"--{flag.replace('_', '-')}", str(spec.params[flag])]
        return argv
    raise ValueError(f"unknown job kind: {kind!r}")


class _Job:
    def __init__(self, job_id: str, spec: JobSpec, argv: list[str], workdir: Path):
        self.id = job_id
        self.kind = spec.kind
        self.argv = argv
        self.workdir = workdir
        self.state = JobState.queued
        self.created_at = time.time()
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.exit_code: int | None = None
        self.error: str | None = None
        self.lines: list[str] = []
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel_requested = False
        self._finished = False
        self._cond = asyncio.Condition()

    # ---- artifacts / status ----------------------------------------

    def artifacts(self) -> list[Artifact]:
        out: list[Artifact] = []
        if not self.workdir.exists():
            return out
        for p in sorted(self.workdir.rglob("*")):
            if p.is_file() and p.name != "job.log" and p.name != "manifest.json":
                out.append(Artifact(name=p.name,
                                    rel_path=str(p.relative_to(self.workdir)),
                                    bytes=p.stat().st_size))
        return out

    def status(self) -> JobStatus:
        return JobStatus(
            id=self.id, kind=self.kind, state=self.state,
            created_at=self.created_at, started_at=self.started_at,
            ended_at=self.ended_at, exit_code=self.exit_code, error=self.error,
            artifacts=self.artifacts(),
        )

    def _write_manifest(self) -> None:
        manifest = {
            "id": self.id, "kind": self.kind, "argv": self.argv,
            "state": self.state.value, "created_at": self.created_at,
            "started_at": self.started_at, "ended_at": self.ended_at,
            "exit_code": self.exit_code, "error": self.error,
        }
        try:
            (self.workdir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 — manifest is best-effort audit
            pass


class JobRunner:
    """Owns the job registry, the concurrency gate, and the work root."""

    def __init__(self, work_root: str | Path, *, max_concurrent: int = 1):
        self.work_root = Path(work_root)
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, _Job] = {}
        self._sem = asyncio.Semaphore(max(1, int(max_concurrent)))

    # ---- lookup -----------------------------------------------------

    def get(self, job_id: str) -> _Job:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise JobNotFound(job_id) from exc

    def list(self) -> list[JobStatus]:
        return [j.status() for j in
                sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)]

    # ---- create / run ----------------------------------------------

    def create(self, spec: JobSpec) -> _Job:
        job_id = uuid.uuid4().hex[:12]
        workdir = self.work_root / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        argv = build_command(spec, workdir)  # raises ValueError on bad spec
        job = _Job(job_id, spec, argv, workdir)
        self._jobs[job_id] = job
        # Requires a running loop — create() is called from async endpoints.
        asyncio.create_task(self._run(job))
        return job

    async def _run(self, job: _Job) -> None:
        async with self._sem:
            if job._cancel_requested:            # cancelled while queued
                await self._finish(job, JobState.cancelled)
                return
            job.state = JobState.running
            job.started_at = time.time()
            log_path = job.workdir / "job.log"
            try:
                proc = await asyncio.create_subprocess_exec(
                    *job.argv, cwd=str(job.workdir),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                job._proc = proc
                with open(log_path, "wb") as logf:
                    assert proc.stdout is not None
                    async for raw in proc.stdout:
                        logf.write(raw)
                        logf.flush()
                        line = raw.decode("utf-8", "replace").rstrip("\n")
                        async with job._cond:
                            job.lines.append(line)
                            job._cond.notify_all()
                rc = await proc.wait()
                job.exit_code = rc
                if job._cancel_requested:
                    await self._finish(job, JobState.cancelled)
                else:
                    await self._finish(job, JobState.succeeded if rc == 0 else JobState.failed)
            except Exception as exc:  # noqa: BLE001
                job.error = f"{type(exc).__name__}: {exc}"
                await self._finish(job, JobState.failed)

    async def _finish(self, job: _Job, state: JobState) -> None:
        job.state = state
        job.ended_at = time.time()
        job._write_manifest()
        async with job._cond:
            job._finished = True
            job._cond.notify_all()

    # ---- cancel -----------------------------------------------------

    async def cancel(self, job_id: str) -> _Job:
        job = self.get(job_id)
        job._cancel_requested = True
        if job._proc is not None and job._proc.returncode is None:
            job._proc.terminate()
        elif job.state == JobState.queued:
            # Not started yet — _run will observe the flag and finish it.
            pass
        return job

    # ---- log streaming (SSE) ---------------------------------------

    async def stream_logs(self, job_id: str):
        """Yield log lines: replay what's buffered, then live lines until done."""
        job = self.get(job_id)
        idx = 0
        while True:
            async with job._cond:
                while idx >= len(job.lines) and not job._finished:
                    await job._cond.wait()
                new = job.lines[idx:]
                idx = len(job.lines)
                finished = job._finished
            for line in new:
                yield line
            if finished and idx >= len(job.lines):
                return


__all__ = ["JobRunner", "JobNotFound", "build_command"]
