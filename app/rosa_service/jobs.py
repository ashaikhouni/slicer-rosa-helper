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
import shutil
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

from .models import Artifact, JobSpec, JobState, JobStatus


class JobNotFound(KeyError):
    pass


@lru_cache(maxsize=1)
def _deepbet_available() -> bool:
    """True when a deepbet-capable python is reachable (probed once per process).

    deepbet is the fast MIT T1 strip; when present the pipeline pre-extracts the
    native brain mask with it (vs SynthStrip). Memoised: the probe spawns a
    subprocess, and this is called on the event loop while building a job.
    """
    try:
        from rosa_detect.services.deepbet_strip import deepbet_available
        return deepbet_available()
    except Exception:  # noqa: BLE001 — no engine / probe failure → treat as absent
        return False


def _fastsurfer_available() -> bool:
    """True when the FastSurfer seg runtime is detected (the anatomic mesh source)."""
    try:
        from rosa_detect.services.fastsurfer_seg import find_fastsurfer
        return find_fastsurfer()[0] is not None
    except Exception:  # noqa: BLE001 — treat any import/probe failure as "no FastSurfer"
        return False


@lru_cache(maxsize=1)
def _deepmriprep_available() -> bool:
    """True when a deepmriprep-capable python is reachable (memoised subprocess probe)."""
    try:
        from rosa_detect.services.deepmriprep_seg import deepmriprep_available
        return deepmriprep_available()
    except Exception:  # noqa: BLE001
        return False


def _brainchop_available() -> bool:
    """True when the bundled brainchop GM/WM ONNX + onnxruntime are present (torch-free)."""
    try:
        from rosa_detect.services.brainchop_onnx import brainchop_available
        return brainchop_available()
    except Exception:  # noqa: BLE001
        return False


# Which surface backend each source name maps to a cache filename.
_SURFACE_CACHE_NAME = {
    "fastsurfer": "brain_surface_fs.npz",
    "deepmriprep": "brain_surface_dm.npz",
    "brainchop": "brain_surface_bc.npz",
    "otsu": "brain_surface.npz",
}


def _resolve_surface_source(surface_source: str | None) -> str:
    """Resolve the requested surface source to a concrete backend, degrading
    gracefully to what's installed. ``auto`` = FastSurfer if available, else
    deepmriprep if available, else Otsu. An explicit choice that isn't reachable
    falls back to ``auto`` rather than failing the run mid-mesh.
    """
    src = (surface_source or "auto").lower()
    if src == "fastsurfer":
        return "fastsurfer" if _fastsurfer_available() else _resolve_surface_source("auto")
    if src == "deepmriprep":
        return "deepmriprep" if _deepmriprep_available() else _resolve_surface_source("auto")
    if src == "brainchop":
        return "brainchop" if _brainchop_available() else _resolve_surface_source("auto")
    if src == "otsu":
        return "otsu"
    # auto: prefer BYO recons (FastSurfer > deepmriprep), then the bundled
    # torch-free brainchop MeshNet, then the Otsu grayscale-iso fallback.
    if _fastsurfer_available():
        return "fastsurfer"
    if _deepmriprep_available():
        return "deepmriprep"
    if _brainchop_available():
        return "brainchop"
    return "otsu"


def _brain_surface_view_flags(t1: str, regcache: Path, *,
                              include_transform: bool,
                              surface_source: str | None = "auto") -> list[str]:
    """``view-results`` flags that build/reuse the subject brain SURFACE from the
    MRI, shared by the pipeline (case-creation) and label jobs so their caches
    line up: the native brain mask, the T1→CT transform, and the meshed surface
    all live in the same ``regcache`` and are produced once.

    ``surface_source`` picks the mesh backend (``fastsurfer`` / ``deepmriprep`` /
    ``otsu`` / ``auto``), resolved against what's installed. The cache filename
    tracks the backend so one source never masquerades as another on a cache hit.

    ``include_transform`` adds ``--brain-to-ct-transform`` (the label job runs the
    ``label`` step first, which SAVES that transform); the pipeline omits it and
    lets view-results register T1→CT ephemerally (saved later, on the first label).
    """
    regcache = Path(regcache)
    src = _resolve_surface_source(surface_source)
    brain_mask = regcache / "brain_mask_native.nii.gz"
    brain_surface = regcache / _SURFACE_CACHE_NAME[src]
    flags = ["--brain-native-volume", str(t1)]
    if include_transform:
        flags += ["--brain-to-ct-transform", str(regcache / "t1_to_ct.tfm")]
    flags += ["--brain-mask-cache", str(brain_mask),
              "--brain-surface-cache", str(brain_surface)]
    if src == "fastsurfer":
        fs_aseg = (regcache / "fastsurfer" / "subject" / "mri"
                   / "aparc.DKTatlas+aseg.deep.mgz")
        flags += (["--fastsurfer-aseg", str(fs_aseg)] if fs_aseg.is_file()
                  else ["--fastsurfer"])
    elif src == "deepmriprep":
        dm_p0 = regcache / "deepmriprep" / "p0.nii.gz"
        flags += (["--deepmriprep-tissue", str(dm_p0)] if dm_p0.is_file()
                  else ["--deepmriprep"])
    elif src == "brainchop":
        flags += ["--brainchop"]
    return flags


def _engine_base() -> list[str]:
    """argv prefix to invoke the rosa_agent engine, in dev AND in the frozen app.

    Dev / pip-installed: ``python -u -m rosa_agent``. Frozen (PyInstaller): the
    packaged binary is a multi-call sidecar, so ``sys.executable`` is that binary
    and it re-invokes itself as ``<binary> engine`` (see app/desktop/sidecar_main.py).
    The frozen engine sets line-buffered stdout itself, so no ``-u`` is needed.
    """
    if getattr(sys, "frozen", False):
        return [sys.executable, "engine"]
    return [sys.executable, "-u", "-m", "rosa_agent"]


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
        # Optional patient MRI (T1) provided at case creation. When present, the
        # pipeline brings it in from the start: extract the native brain mask and
        # mesh the subject brain SURFACE so the electrode view shows anatomy
        # immediately, and cache both (+ the T1→CT transform on first label) in
        # ``regcache`` so a later label job reuses them instead of re-running.
        t1 = spec.params.get("t1")
        t1 = str(t1) if t1 else None
        # Surface backend the MRI mesh uses: 'auto' (FastSurfer→deepmriprep→Otsu),
        # 'fastsurfer', 'deepmriprep', or 'otsu'. Recorded on the job so the later
        # label job reuses the SAME cache (see create_label_job).
        surface = str(spec.params.get("surface") or "auto")
        traj = str(workdir / "trajectories.tsv")
        contacts = str(workdir / "contacts.tsv")
        viewer = str(workdir / "viewer")
        regcache = workdir / "regcache"
        # `python -m rosa_agent` works whether the engine is pip-installed or on
        # PYTHONPATH; in the frozen app this becomes the frozen exe re-invoked.
        # detect (CT → trajectories) → contacts (+CT → contacts) → view-results
        # (→ served viewer dir: index.html + scene.glb + scene_meta.json + CT).
        base = _engine_base()
        native_mask = regcache / "brain_mask_native.nii.gz"
        mask_in_ct = regcache / "brain_mask_in_ct.nii.gz"
        t1_to_ct = regcache / "t1_to_ct.tfm"
        steps: list[list[str]] = []
        # When a T1 is present AND deepbet is reachable, one brain-extract step
        # extracts the native brain mask (for the surface) AND registers it into
        # the CT frame:
        #   * --mask-in-target → the CT-frame mask the CONTACTS placement anchor
        #     consumes (fast + MRI-accurate, replacing its SynthStrip-or-hull
        #     default), and
        #   * --save-transform → the T1→CT transform view-results reuses, so the
        #     surface (and any later label) don't re-register.
        # deepbet is T1-only; without it we skip this and everything below falls
        # back to the CT-only behaviour (contacts uses its own mask default;
        # view-results extracts + registers on its own).
        use_mri_mask = bool(t1 and _deepbet_available())
        if use_mri_mask:
            steps.append(base + ["brain-extract", t1, "-o", str(native_mask),
                                  "--backend", "deepbet",
                                  "--register-to", ct,
                                  "--save-transform", str(t1_to_ct),
                                  "--mask-in-target", str(mask_in_ct)])
        contacts_step = base + ["contacts", traj, ct, "--out", contacts]
        # Constrain the electrode library to the types a site uses (from the
        # new-case screen's checkboxes) — restricts model matching + speeds it up.
        etypes = spec.params.get("electrode_types")
        if etypes:
            contacts_step += ["--electrode-types", str(etypes)]
        if use_mri_mask:
            # Feed the MRI-derived intracranial mask to the placement anchor.
            contacts_step += ["--brain-mask", str(mask_in_ct)]
        steps += [
            base + ["detect", ct, "--out", traj],
            contacts_step,
        ]
        # Without an MRI the CT stays MIP-only (fast, no surface). With an MRI,
        # add the brain-surface flags so view-results meshes the subject surface;
        # reuse the cached T1→CT transform when brain-extract already saved it.
        view = base + ["view-results", str(workdir), "--output", viewer,
                       "--ct", ct, "--contacts", contacts, "--trajectories", traj,
                       "--subject-label", label]
        if t1:
            view += _brain_surface_view_flags(t1, regcache,
                                              include_transform=use_mri_mask,
                                              surface_source=surface)
        steps.append(view)
        # Make MNI first-class from detection: register T1→MNI + warp contacts to
        # MNI + label them against the bundled MNI atlases (CerebrA + Iglesias).
        # Gated on use_mri_mask so the CT→T1 leg (t1_to_ct.tfm) exists to compose;
        # stamp-mni self-gates to a no-op otherwise. Runs after contacts + extract.
        if use_mri_mask:
            steps.append(base + ["stamp-mni", str(workdir), "--t1", t1])
        return steps
    if kind == "import":
        # Import a localization computed elsewhere (a batch CLI run) for review:
        # skip detect + contacts (the TSVs already exist) and run ONLY
        # view-results against the provided CT + TSVs. Everything downstream
        # (review doc, editor payload) reads workdir/contacts.tsv +
        # trajectories.tsv, so we stage the provided files there first — the same
        # loader/paths as a pipeline job, so an import reviews identically.
        ct = spec.params.get("ct")
        contacts_src = spec.params.get("contacts")
        traj_src = spec.params.get("trajectories")
        if not (ct and contacts_src and traj_src):
            raise ValueError(
                "import job requires params.ct, params.contacts, params.trajectories")
        ct, contacts_src, traj_src = str(ct), str(contacts_src), str(traj_src)
        label = str(spec.params.get("label") or "case")
        t1 = spec.params.get("t1")
        t1 = str(t1) if t1 else None
        surface = str(spec.params.get("surface") or "auto")
        traj = str(workdir / "trajectories.tsv")
        contacts = str(workdir / "contacts.tsv")
        viewer = str(workdir / "viewer")
        regcache = workdir / "regcache"
        base = _engine_base()
        steps = []
        # 1) stage the provided TSVs into the job dir (steps run cwd=workdir).
        stage = ("import shutil, sys\n"
                 "shutil.copyfile(sys.argv[1], 'contacts.tsv')\n"
                 "shutil.copyfile(sys.argv[2], 'trajectories.tsv')\n"
                 "print('staged import TSVs', flush=True)\n")
        steps.append([py, "-u", "-c", stage, contacts_src, traj_src])
        # 2) optional MRI brain-extract (surface + CT-frame mask), as in pipeline.
        use_mri_mask = bool(t1 and _deepbet_available())
        if use_mri_mask:
            steps.append(base + ["brain-extract", t1,
                                  "-o", str(regcache / "brain_mask_native.nii.gz"),
                                  "--backend", "deepbet", "--register-to", ct,
                                  "--save-transform", str(regcache / "t1_to_ct.tfm"),
                                  "--mask-in-target", str(regcache / "brain_mask_in_ct.nii.gz")])
        # 3) build the 3D viewer from the imported localization.
        view = base + ["view-results", str(workdir), "--output", viewer,
                       "--ct", ct, "--contacts", contacts, "--trajectories", traj,
                       "--subject-label", label]
        if t1:
            view += _brain_surface_view_flags(t1, regcache,
                                              include_transform=use_mri_mask,
                                              surface_source=surface)
        steps.append(view)
        return steps
    if kind == "rebuild":
        # Rebuild a case's 3D viewer in place from its (edited) TSVs — view-results
        # only. Geometry changed but the brain surface didn't, so reuse the cached
        # surface + T1→CT transform (no re-strip / re-mesh).
        case_dir = spec.params.get("case_dir")
        ct = spec.params.get("ct")
        if not (case_dir and ct):
            raise ValueError("rebuild job requires params.case_dir and params.ct")
        case_dir, ct = str(case_dir), str(ct)
        label = str(spec.params.get("label") or "case")
        t1 = spec.params.get("t1")
        t1 = str(t1) if t1 else None
        surface = str(spec.params.get("surface") or "auto")
        cd = Path(case_dir)
        view = [*_engine_base(), "view-results", case_dir,
                "--output", str(cd / "viewer"), "--ct", ct,
                "--contacts", str(cd / "contacts.tsv"),
                "--trajectories", str(cd / "trajectories.tsv"),
                "--subject-label", label]
        if t1:
            view += _brain_surface_view_flags(
                t1, cd / "regcache",
                include_transform=(cd / "regcache" / "t1_to_ct.tfm").is_file(),
                surface_source=surface)
        # Re-attach the atlas the case was labeled with (resolved by app.py from
        # the case's label job). The cortical surface survives via its cache but
        # the atlas TINT does not, and THOMAS deep-structure meshes have no cache
        # fallback at all — so without these flags they vanish on every rebuild.
        atlas_labelmap = spec.params.get("atlas_labelmap")
        if atlas_labelmap:
            view += ["--atlas-labelmap", str(atlas_labelmap)]
            if spec.params.get("atlas_name"):
                view += ["--atlas-name", str(spec.params["atlas_name"])]
        structure_meshes = spec.params.get("structure_meshes")
        if structure_meshes:
            view += ["--structure-meshes", str(structure_meshes)]
            if spec.params.get("structure_lut"):
                view += ["--structure-lut", str(spec.params["structure_lut"])]
        steps = [view]
        # Re-stamp MNI on every edit-save so the MNI coords + multi-atlas labels
        # never drift: it rebuilds wholesale from the edited contacts.tsv, so moves
        # / adds / deletes propagate. No re-registration (the transform is stable);
        # self-gates to a no-op when the case isn't poolable.
        if t1:
            steps.append([*_engine_base(), "stamp-mni", case_dir, "--t1", t1])
        return steps
    if kind == "label":
        # Anatomical labeling of an existing pipeline run's contacts against a
        # bundled MNI atlas, routed through the patient's T1 (MRI). Produces a
        # proposed labeling (contacts_labeled.tsv) + an MRI-in-CT QC volume;
        # the app applies the labels to the parent's ReviewDoc only on approval.
        contacts = spec.params.get("contacts")
        ct = spec.params.get("ct")
        t1 = spec.params.get("t1")
        atlas = str(spec.params.get("atlas") or "cerebra")
        is_thomas = atlas.startswith("thomas")
        # THOMAS brings its OWN reference MRI (it registers that → the CT), so a
        # patient T1 is optional — it only adds the cortical Ghost surface for
        # context. Every other atlas needs the patient T1 to route registration.
        if not (contacts and ct) or (not t1 and not is_thomas):
            raise ValueError("label job requires params.contacts, params.ct, params.t1")
        # Same surface backend the parent pipeline chose, so this label reuses the
        # already-built brain_surface_*.npz instead of meshing a different one.
        surface = str(spec.params.get("surface") or "auto")
        out = str(workdir / "contacts_labeled.tsv")
        mri_qc = str(workdir / "mri_in_ct.nii.gz")
        ct_mni = str(workdir / "ct_in_mni.nii.gz")
        mri_mni = str(workdir / "mri_in_mni.nii.gz")
        atlas_in_ct = str(workdir / "atlas_in_ct.nii.gz")   # warped atlas → colors the surface
        # Per-case registration cache (in the parent pipeline job's dir) — T1→CT
        # once per case, MNI→T1 once per template space, so labeling more atlases
        # reuses the transforms instead of re-registering.
        regcache = str(spec.params.get("regcache") or (workdir / "regcache"))
        # Parent case dir + its viewer, so the 2nd step regenerates the 3D view
        # with the subject brain SURFACE (with gyri), electrodes penetrating it.
        # The surface is meshed in the NATIVE MRI frame (clean, isotropic) and
        # pushed into CT via the T1→CT transform the label step just cached —
        # meshing the MRI resampled to the anisotropic CT grid looks bumpy.
        # Native brain mask (SynthStrip) is cached in regcache → once per case.
        parent_dir = Path(contacts).parent
        parent_traj = str(parent_dir / "trajectories.tsv")
        parent_viewer = str(parent_dir / "viewer")
        base = _engine_base()
        # FastSurfer = subject-specific labeler (native aparc+aseg, no MNI). The
        # bundled MNI atlases warp a template. The 3D brain MESH is atlas-INDEPENDENT
        # (FastSurfer recon whenever available, else the Otsu surface) — meshed once
        # per case (cached in regcache), recolored per atlas below. The surface,
        # native brain mask and T1→CT transform caches are shared with the pipeline
        # job via ``_brain_surface_view_flags`` so nothing is re-run per atlas.
        thomas_lut = str(workdir / "atlas_in_ct.nii.gz.lut.json")  # THOMAS mesh LUT
        if is_thomas:
            # THOMAS = a BYO deep-structure atlas: register its own reference T1 →
            # the CT (no patient-T1 route), warp the nuclei labelmap in, label the
            # contacts by nucleus, and save the MRI-in-CT QC for the reg check.
            thomas_dir = spec.params.get("thomas_dir")
            if not thomas_dir:
                raise ValueError("thomas atlas requires params.thomas_dir")
            label_step = base + ["label", str(contacts),
                                 "--thomas", str(thomas_dir),
                                 "--target-volume", str(ct),
                                 "--save-registered-mri", mri_qc,
                                 "--save-atlas-labelmap", atlas_in_ct,
                                 "--registration-cache", regcache, "-o", out]
        elif atlas == "fastsurfer":
            label_step = base + ["label", str(contacts), "--fastsurfer",
                                  "--target-volume", str(ct),
                                  "--intermediate-volume", str(t1),
                                  "--save-registered-mri", mri_qc,
                                  "--registration-cache", regcache, "-o", out]
        elif atlas.startswith("dmp:"):
            # deepmriprep native-space atlas: labeled via a T1→CT rigid (no MNI),
            # so no MNI-space QC volumes. Still saves the CT-frame labelmap for the
            # surface coloring + the MRI-in-CT for the Native registration QC.
            label_step = base + ["label", str(contacts),
                                 "--deepmriprep-atlas", atlas[len("dmp:"):],
                                 "--target-volume", str(ct),
                                 "--intermediate-volume", str(t1),
                                 "--save-registered-mri", mri_qc,
                                 "--save-atlas-labelmap", atlas_in_ct,
                                 "--registration-cache", regcache, "-o", out]
        else:
            label_step = base + ["label", str(contacts),
                                 "--bundled-atlas", atlas,
                                 "--target-volume", str(ct),
                                 "--intermediate-volume", str(t1),
                                 "--save-registered-mri", mri_qc,
                                 "--save-ct-in-mni", ct_mni,
                                 "--save-mri-in-mni", mri_mni,
                                 # Warped atlas labelmap → colors the brain surface by
                                 # this atlas (view-results samples it below).
                                 "--save-atlas-labelmap", atlas_in_ct,
                                 "--registration-cache", regcache, "-o", out]
        steps = [label_step]
        # view-results runs on EVERY label: it (re)builds or reuses the cached
        # anatomic mesh, then colors it for the active atlas.
        view_step = base + ["view-results", str(parent_dir), "--output", parent_viewer,
                            "--ct", str(ct), "--contacts", str(contacts),
                            "--trajectories", parent_traj]
        # Brain SURFACE flags (native volume + T1→CT transform the label step just
        # saved + cached mask + cached mesh). On a cache hit view-results loads the
        # mesh and ignores the FastSurfer flags. Shared with the pipeline job.
        # THOMAS may run without a patient MRI → then there's no cortex surface.
        if t1:
            view_step += _brain_surface_view_flags(str(t1), Path(regcache),
                                                   include_transform=True,
                                                   surface_source=surface)
        # COLOR: the FastSurfer labeler keeps its own DKT parcellation; every other
        # atlas recolors that same mesh (+ tints the slices) by its warped labelmap.
        if atlas != "fastsurfer":
            view_step += ["--atlas-labelmap", atlas_in_ct,
                          "--atlas-name", ("THOMAS" if is_thomas else atlas)]
        # THOMAS additionally renders its nuclei as 3-D meshes (the deep anatomy a
        # cortical atlas can't paint), colored by the same LUT that tints the slices.
        if is_thomas:
            view_step += ["--structure-meshes", atlas_in_ct, "--structure-lut", thomas_lut]
        steps.append(view_step)
        # This label run cached the CT→T1→MNI transforms (CerebrA-through-T1), so
        # stamp the case: warp contacts to MNI + label them against every bundled
        # MNI atlas (CerebrA + Iglesias). Self-gates to a no-op for non-poolable
        # atlas paths. regcache lives in the parent case dir.
        if t1:
            steps.append([*_engine_base(), "stamp-mni",
                          str(Path(regcache).parent), "--t1", str(t1)])
        return steps
    if kind == "stamp":
        # On-demand MNI (re)stamp — used for the "Refine (nonlinear)" action. Runs
        # stamp-mni on the case dir; --refine swaps the affine T1→MNI for a
        # B-spline composite (~30 s) and re-warps + re-labels off it.
        case_dir = spec.params.get("case_dir")
        if not case_dir:
            raise ValueError("stamp job requires params.case_dir")
        argv = [*_engine_base(), "stamp-mni", str(case_dir)]
        t1 = spec.params.get("t1")
        if t1:
            argv += ["--t1", str(t1)]
        if spec.params.get("refine"):
            argv += ["--refine"]
        return [argv]
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
            parent=self.params.get("parent"), atlas=self.params.get("atlas"),
            t1=self.params.get("t1"), label=self.params.get("label"),
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

    def delete(self, job_id: str) -> None:
        """Delete a finished case: drop it (and any child label jobs) from the
        registry and remove their workdirs. Refuses a job that's still
        running/queued — cancel it first."""
        job = self.get(job_id)
        if not job.state.terminal:
            raise ValueError(f"job {job_id} is {job.state.value}; cancel it before deleting")
        # A case's label jobs are children (params.parent == job_id) — remove them too.
        victims = [job_id] + [jid for jid, j in self._jobs.items()
                              if j.params.get("parent") == job_id]
        for jid in victims:
            j = self._jobs.pop(jid, None)
            if j is not None:
                shutil.rmtree(j.workdir, ignore_errors=True)

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
