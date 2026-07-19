"""Service DTOs — the versioned contract the UI depends on.

These Pydantic models are the *stable surface* the web/Electron UI binds to. The
UI never sees engine/CLI argument shapes: it POSTs a :class:`JobSpec` (a ``kind``
+ validated ``params``), and the service maps that to an engine command
internally (see ``jobs.build_command``). So the engine/CLI can change without
breaking the UI — only this contract must stay stable.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class JobState(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobState.succeeded, JobState.failed, JobState.cancelled)


class JobSpec(BaseModel):
    """What the UI asks the service to run."""

    kind: str = Field(..., description="job kind, e.g. 'pipeline' (mapped to an engine command internally)")
    params: dict = Field(default_factory=dict, description="validated, kind-specific parameters")


class Artifact(BaseModel):
    name: str
    rel_path: str
    bytes: int


class JobStatus(BaseModel):
    """Everything the UI needs to render a job's state."""

    id: str
    kind: str
    state: JobState
    created_at: float
    started_at: float | None = None
    ended_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    # Surfaced for the UI: a label job's parent pipeline id + its atlas + the MRI
    # it used, so a reload can find the case's label job, re-enable the
    # registration tab, and keep atlas-switching working (relabel reuses the
    # cached registration) without re-uploading the MRI.
    parent: str | None = None
    atlas: str | None = None
    t1: str | None = None
    # The human case label (params.label), so the Cases list can name a case
    # without opening it.
    label: str | None = None


class ImportRequest(BaseModel):
    """Import a localization computed elsewhere (batch CLI run) for review.

    ``ct`` + ``contacts`` + ``trajectories`` are absolute paths already on this
    machine (uploaded via ``/uploads`` or typed). The service validates that the
    contacts actually fit the CT before creating the (view-results-only) job.
    """

    ct: str
    contacts: str
    trajectories: str
    t1: str | None = None
    label: str | None = None
    surface: str | None = None
    # Bypass the "this CT already has a case" duplicate check (import anyway).
    force: bool = False


# ---------------------------------------------------------------------
# Review & edit — the editable-results contract the review UI binds to.
# ---------------------------------------------------------------------


class ReviewContact(BaseModel):
    """One contact in the reviewable result."""

    shank: str
    index: int                       # contact_index (1-based, along the shank)
    name: str                        # channel name, e.g. "LAC1" (TSV `label`)
    x: float
    y: float
    z: float
    model: str | None = None         # electrode model
    region: str | None = None        # anatomical label (editable via relabel)
    accepted: bool = True            # reject to drop from export


class ReviewShank(BaseModel):
    """One electrode (trajectory) and its contacts."""

    name: str
    model: str | None = None
    accepted: bool = True            # reject to drop the whole shank
    contacts: list[ReviewContact] = Field(default_factory=list)


class ReviewDoc(BaseModel):
    """The editable result of a run — what the review-and-edit UI reads/patches."""

    subject_id: str | None = None
    ct_path: str | None = None
    shanks: list[ReviewShank] = Field(default_factory=list)


class ReviewOp(str, Enum):
    accept_contact = "accept_contact"
    reject_contact = "reject_contact"
    relabel_contact = "relabel_contact"
    accept_shank = "accept_shank"
    reject_shank = "reject_shank"


class ReviewEdit(BaseModel):
    """A single edit. ``index`` is required for ``*_contact`` ops; ``region``
    for ``relabel_contact``."""

    op: ReviewOp
    shank: str
    index: int | None = None
    region: str | None = None


class ReviewPatch(BaseModel):
    ops: list[ReviewEdit] = Field(default_factory=list)


class DicomRequest(BaseModel):
    """Convert a DICOM series folder (on this machine) to a de-identified NIfTI.

    ``dicom_dir`` is an absolute folder path (from Electron's native directory
    picker, or typed). Converting to NIfTI drops the DICOM PHI headers, so the
    returned file is safe to store + distribute.
    """

    dicom_dir: str
    series_uid: str | None = None


class RosMatchRequest(BaseModel):
    """Match a case's detected trajectories against a ROSA surgical plan.

    ``ros_text`` is the raw text of a ``.ros`` planning file (read client-side and
    uploaded — no native path needed). Empty → use the plan stashed at import
    (``ros_plan.tsv``, for ROSA-imported cases). Matching is pure line geometry,
    so only the plan's trajectory names + entry/target are used; no images."""

    ros_text: str = ""


class RosaPrepareRequest(BaseModel):
    """Stage a ROSA export folder for import: de-identify, bake to NIfTI, and
    return the display volumes (with thumbnails + CT/MRI guesses) to confirm."""

    rosa_dir: str
    label: str | None = None


class RosaCreateRequest(BaseModel):
    """Create a case from a staged ROSA import — the clinician-chosen post-op CT
    (+ optional non-contrast T1), plus the plan to stash for later labeling."""

    ct_path: str
    t1_path: str | None = None
    label: str | None = None
    seeds_path: str | None = None


class LabelRequest(BaseModel):
    """Kick off anatomical labeling of a pipeline job's contacts.

    ``t1`` is the uploaded MRI path (from ``POST /uploads``); ``atlas`` is a
    bundled atlas id (see ``GET /atlases``). The CT + contacts come from the
    parent pipeline job. ``t1`` may be omitted when the case was created with an
    MRI — the service falls back to the parent pipeline's ``t1``.
    """

    t1: str | None = None
    atlas: str = "cerebra"
    # THOMAS is a BYO atlas (``atlas="thomas"``): the uploaded folder of nucleus
    # masks + a reference MRI. Ignored for every other atlas.
    thomas_dir: str | None = None


__all__ = [
    "JobState", "JobSpec", "Artifact", "JobStatus",
    "ReviewContact", "ReviewShank", "ReviewDoc",
    "ReviewOp", "ReviewEdit", "ReviewPatch", "LabelRequest", "DicomRequest",
]
