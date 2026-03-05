"""Artifact writing hooks for detection pipelines."""

from __future__ import annotations

import csv
import json
import os
from typing import Any

from .contracts import ArtifactRecord, DetectionResult, to_jsonable


class NullArtifactWriter:
    """No-op artifact writer used when outputs are disabled."""

    def __init__(self, run_id: str = ""):
        self.run_id = str(run_id or "")

    def write_json(self, name: str, payload: Any) -> str:
        return ""

    def write_text(self, name: str, text: str) -> str:
        return ""

    def write_bytes(self, name: str, data: bytes) -> str:
        return ""

    def write_csv_rows(self, name: str, header: list[str], rows: list[list[Any]]) -> str:
        return ""


class FileArtifactWriter:
    """Filesystem-backed artifact writer with stable run-folder layout."""

    def __init__(self, artifact_root: str, run_id: str):
        self.artifact_root = os.path.abspath(artifact_root)
        self.run_id = str(run_id)
        self.run_dir = os.path.join(self.artifact_root, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

    def _resolve_path(self, name: str) -> str:
        rel = str(name or "").lstrip("/")
        path = os.path.abspath(os.path.join(self.run_dir, rel))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return path

    def write_json(self, name: str, payload: Any) -> str:
        path = self._resolve_path(name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(payload), f, indent=2)
        return path

    def write_text(self, name: str, text: str) -> str:
        path = self._resolve_path(name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(text))
        return path

    def write_bytes(self, name: str, data: bytes) -> str:
        path = self._resolve_path(name)
        with open(path, "wb") as f:
            f.write(bytes(data))
        return path

    def write_csv_rows(self, name: str, header: list[str], rows: list[list[Any]]) -> str:
        path = self._resolve_path(name)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            for row in rows:
                writer.writerow([to_jsonable(v) for v in row])
        return path



def add_artifact(records: list[ArtifactRecord], *, kind: str, path: str, description: str = "", stage: str = "") -> None:
    """Append one artifact record with stable field names."""

    if not path:
        return
    records.append(
        {
            "kind": str(kind),
            "path": str(path),
            "description": str(description),
            "stage": str(stage),
        }
    )



def write_standard_artifacts(
    writer: NullArtifactWriter | FileArtifactWriter | None,
    result: DetectionResult,
    *,
    blobs: list[dict[str, Any]] | None = None,
    pipeline_payload: dict[str, Any] | None = None,
) -> list[ArtifactRecord]:
    """Write standard run artifacts when writer is available."""

    artifacts: list[ArtifactRecord] = []
    if writer is None:
        return artifacts

    diagnostics_path = writer.write_json("diagnostics.json", result.get("diagnostics", {}))
    add_artifact(artifacts, kind="diagnostics_json", path=diagnostics_path, stage="finalize")

    shanks_path = writer.write_json("shanks.json", result.get("trajectories", []))
    add_artifact(artifacts, kind="shanks_json", path=shanks_path, stage="finalize")

    contacts_path = writer.write_json("contacts.json", result.get("contacts", []))
    add_artifact(artifacts, kind="contacts_json", path=contacts_path, stage="finalize")

    if blobs:
        header = sorted({k for blob in blobs for k in blob.keys()})
        rows = [[blob.get(k, "") for k in header] for blob in blobs]
        blob_path = writer.write_csv_rows("blobs.csv", header, rows)
        add_artifact(artifacts, kind="blobs_csv", path=blob_path, stage="finalize")

    if pipeline_payload is not None:
        payload_path = writer.write_json("pipeline_payload.json", pipeline_payload)
        add_artifact(artifacts, kind="pipeline_payload_json", path=payload_path, stage="finalize")

    return artifacts
