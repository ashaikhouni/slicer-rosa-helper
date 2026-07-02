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


__all__ = ["JobState", "JobSpec", "Artifact", "JobStatus"]
