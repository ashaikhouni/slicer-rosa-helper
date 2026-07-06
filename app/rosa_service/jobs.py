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


def build_command(spec: JobSpec, workdir: Path) -> list[list[str]]:
    """Map a :class:`JobSpec` to an ordered list of subprocess STEPS.

    Returns a list of argv lists; the runner executes them in order, fail-fast.
    Single-command kinds return one step; ``pipeline`` returns the real engine
    chain (``detect`` → ``contacts`` → ``view-results``). Only known kinds are
    runnable, so the UI can't inject arbitrary commands. ``selftest*`` kinds are
    fast synthetic jobs used to exercise the runner (no data, run in tests).
    """
    py = sys.executable
    kind = spec.kind
    if kind == "selftest":
        n = int(spec.params.get("steps", 3))
        script = (
            "import time\n"
            f"for i in range({n}):\n"
            f"    print('step', i + 1, 'of', {n}, flush=True)\n"
            "    time.sleep(0.02)\n"
            "print('done', flush=True)\n"
        )
        return [[py, "-u", "-c", script]]
    if kind == "selftest-hang":
        # Never exits on its own — used to test cancellation.
        return [[py, "-u", "-c", "import time\nwhile True:\n    time.sleep(0.05)\n"]]
    if kind == "selftest-fail":
        return [[py, "-u", "-c", "import sys\nprint('boom', flush=True)\nsys.exit(3)\n"]]
    if kind == "selftest-multi":
        return [
            [py, "-u", "-c", "print('step-A done', flush=True)"],
            [py, "-u", "-c", "print('step-B done', flush=True)"],
        ]
    if kind == "selftest-multi-fail":
        # First step fails → the runner must not run the second.
        return [
            [py, "-u", "-c", "import sys\nprint('step-A', flush=True)\nsys.exit(2)"],
            [py, "-u", "-c", "print('step-B should not run', flush=True)"],
        ]
    if kind == "selftest-emit":
        # Write a small synthetic contacts.tsv into the job dir (cwd), so the
        # review flow can be exercised end-to-end without a real pipeline run.
        script = (
            "import csv\n"
            "cols=['trajectory','label','contact_index','x','y','z','peak_detected','electrode_model','region']\n"
            "rows=[]\n"
            "for sh,(base,reg) in {'LAC':(0.0,'Amygdala'),'LPC':(20.0,'Hippocampus')}.items():\n"
            "    for i in range(1,4):\n"
            "        rows.append({'trajectory':sh,'label':f'{sh}{i}','contact_index':i,"
            "'x':base+i,'y':i*2.0,'z':5.0,'peak_detected':'1','electrode_model':'DIXI-15AM','region':reg})\n"
            "with open('contacts.tsv','w',newline='') as f:\n"
            "    w=csv.DictWriter(f,fieldnames=cols,delimiter='\\t'); w.writeheader(); w.writerows(rows)\n"
            "print('emitted',len(rows),'contacts',flush=True)\n"
        )
        return [[py, "-u", "-c", script]]
    if kind == "selftest-label":
        # Synthetic contacts_labeled.tsv matching selftest-emit's contacts, so
        # the label → proposed → approve flow is testable without real
        # registration (SITK). closest_label carries the region, keyed by
        # trajectory + contact_index — the same contract the engine emits.
        script = (
            "import csv\n"
            "cols=['trajectory','contact_label','contact_index','closest_label']\n"
            "rows=[]\n"
            "for sh,reg in {'LAC':'Left Amygdala','LPC':'Left Hippocampus'}.items():\n"
            "    for i in range(1,4):\n"
            "        rows.append({'trajectory':sh,'contact_label':f'{sh}{i}',"
            "'contact_index':i,'closest_label':reg})\n"
            "with open('contacts_labeled.tsv','w',newline='') as f:\n"
            "    w=csv.DictWriter(f,fieldnames=cols,delimiter='\\t'); w.writeheader(); w.writerows(rows)\n"
            "print('labeled',len(rows),'contacts',flush=True)\n"
        )
        return [[py, "-u", "-c", script]]
    if kind == "pipeline":
        ct = spec.params.get("ct")
        if not ct:
            raise ValueError("pipeline job requires params.ct")
        ct = str(ct)
        label = str(spec.params.get("label") or "case")
        traj = str(workdir / "trajectories.tsv")
        contacts = str(workdir / "contacts.tsv")
        viewer = str(workdir / "viewer")
        # `python -m rosa_agent` works whether the engine is pip-installed or on
        # PYTHONPATH; in the frozen app this becomes the frozen exe re-invoked.
        # detect (CT → trajectories) → contacts (+CT → contacts) → view-results
        # (→ served viewer dir: index.html + scene.glb + scene_meta.json + CT).
        base = [py, "-u", "-m", "rosa_agent"]
        return [
            base + ["detect", ct, "--out", traj],
            base + ["contacts", traj, ct, "--out", contacts],
            # --brain-volume ct: brain-extract (SynthStrip) + marching-cubes the
            # CT into the subject's OWN translucent brain surface (accurate — the
            # patient's actual anatomy, not a template warp) with electrodes
            # penetrating it. Adds ~1-2 min (SynthStrip) to the run.
            base + ["view-results", str(workdir), "--output", viewer,
                    "--ct", ct, "--contacts", contacts, "--trajectories", traj,
                    "--brain-volume", ct, "--subject-label", label],
        ]
    if kind == "label":
        # Anatomical labeling of an existing pipeline run's contacts against a
        # bundled MNI atlas, routed through the patient's T1 (MRI). Produces a
        # proposed labeling (contacts_labeled.tsv) + an MRI-in-CT QC volume;
        # the app applies the labels to the parent's ReviewDoc only on approval.
        contacts = spec.params.get("contacts")
        ct = spec.params.get("ct")
        t1 = spec.params.get("t1")
        if not (contacts and ct and t1):
            raise ValueError("label job requires params.contacts, params.ct, params.t1")
        atlas = str(spec.params.get("atlas") or "cerebra")
        out = str(workdir / "contacts_labeled.tsv")
        mri_qc = str(workdir / "mri_in_ct.nii.gz")
        ct_mni = str(workdir / "ct_in_mni.nii.gz")
        mri_mni = str(workdir / "mri_in_mni.nii.gz")
        base = [py, "-u", "-m", "rosa_agent"]
        return [base + ["label", str(contacts),
                        "--bundled-atlas", atlas,
                        "--target-volume", str(ct),
                        "--intermediate-volume", str(t1),
                        "--save-registered-mri", mri_qc,
                        "--save-ct-in-mni", ct_mni,
                        "--save-mri-in-mni", mri_mni,
                        "-o", out]]
    raise ValueError(f"unknown job kind: {kind!r}")


def _step_label(argv: list[str]) -> str:
    """A short human label for a step, e.g. ``rosa_agent detect`` → ``detect``."""
    if "-m" in argv:
        i = argv.index("-m")
        return " ".join(argv[i + 1:i + 3]) if len(argv) > i + 2 else argv[i + 1]
    return "run"


class _Job:
    def __init__(self, job_id: str, spec: JobSpec, steps: list[list[str]], workdir: Path):
        self.id = job_id
        self.kind = spec.kind
        self.params = dict(spec.params)   # kept so e.g. a label job can find its parent's CT
        self.steps = steps
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
            "id": self.id, "kind": self.kind, "params": self.params, "steps": self.steps,
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
        self._rehydrate()

    def _rehydrate(self) -> None:
        """Rebuild finished jobs from prior runs' manifests, so a restart keeps
        their results viewable/reviewable (in-memory registry survives)."""
        from .models import JobSpec
        for manifest in sorted(self.work_root.glob("*/manifest.json")):
            try:
                m = json.loads(manifest.read_text(encoding="utf-8"))
                jid = m.get("id")
                if not jid or jid in self._jobs:
                    continue
                job = _Job(jid, JobSpec(kind=m.get("kind", "unknown"),
                                        params=m.get("params", {})),
                           m.get("steps", []), manifest.parent)
                job.state = JobState(m.get("state", "succeeded"))
                job.created_at = m.get("created_at") or 0.0
                job.started_at = m.get("started_at")
                job.ended_at = m.get("ended_at")
                job.exit_code = m.get("exit_code")
                job.error = m.get("error")
                job._finished = True   # completed → logs won't stream, routes read disk
                self._jobs[jid] = job
            except Exception:  # noqa: BLE001 — skip an unreadable/partial manifest
                continue

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
        steps = build_command(spec, workdir)  # raises ValueError on bad spec
        job = _Job(job_id, spec, steps, workdir)
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
            n = len(job.steps)
            try:
                with open(log_path, "wb") as logf:
                    for i, argv in enumerate(job.steps, 1):
                        if job._cancel_requested:
                            await self._finish(job, JobState.cancelled)
                            return
                        if n > 1:
                            await self._emit(job, logf, f"[step {i}/{n}] {_step_label(argv)}")
                        proc = await asyncio.create_subprocess_exec(
                            *argv, cwd=str(job.workdir),
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        job._proc = proc
                        assert proc.stdout is not None
                        async for raw in proc.stdout:
                            logf.write(raw)
                            logf.flush()
                            async with job._cond:
                                job.lines.append(raw.decode("utf-8", "replace").rstrip("\n"))
                                job._cond.notify_all()
                        rc = await proc.wait()
                        job.exit_code = rc
                        if job._cancel_requested:
                            await self._finish(job, JobState.cancelled)
                            return
                        if rc != 0:                    # fail-fast: stop the chain
                            await self._finish(job, JobState.failed)
                            return
                await self._finish(job, JobState.succeeded)
            except Exception as exc:  # noqa: BLE001
                job.error = f"{type(exc).__name__}: {exc}"
                await self._finish(job, JobState.failed)

    async def _emit(self, job: "_Job", logf, line: str) -> None:
        """Write a synthetic (non-subprocess) line to the log + subscribers."""
        logf.write((line + "\n").encode())
        logf.flush()
        async with job._cond:
            job.lines.append(line)
            job._cond.notify_all()

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
