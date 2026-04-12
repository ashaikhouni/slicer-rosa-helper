# shank_engine — SEEG Detection Pipeline Framework

Last updated: 2026-04-11

## Overview

`shank_engine` is the pipeline framework for all SEEG electrode trajectory
detection algorithms. It provides:

- **Typed contracts** for inputs, outputs, and intermediate data
- **Protocol-based component interfaces** (duck typing, no forced inheritance)
- **A base class** with stage execution, timing, diagnostics, and error handling
- **A registry** for named pipeline lookup and instantiation

All detection pipelines — blob-based, graph-based, de novo, and deep core —
are `BaseDetectionPipeline` subclasses registered in the same
`PipelineRegistry`.

## Architecture

```
DetectionContext (input)
       |
       v
 BaseDetectionPipeline.run(ctx)
       |
       |-- run_stage("gating",     fn) --> masks, depth maps
       |-- run_stage("extraction",  fn) --> BlobRecord[]
       |-- run_stage("scoring",     fn) --> scored BlobRecord[]
       |-- run_stage("initialize",  fn) --> ShankModel[]
       |-- run_stage("refine",      fn) --> refined ShankModel[]
       |-- run_stage("select",      fn) --> final ShankModel[]
       |-- run_stage("contacts",    fn) --> ContactRecord[]
       |
       v
 DetectionResult (output)
```

Not every pipeline uses every stage. The stage names and count are up to
the implementation. `run_stage` provides timing, error mapping, and
diagnostics regardless of what the stage does.

## Key Files

| File | Purpose |
|------|---------|
| `contracts.py` | `DetectionContext`, `DetectionResult`, `BlobRecord`, `ShankModel`, `ContactRecord`, `DetectionDiagnostics` |
| `interfaces.py` | Protocol definitions: `DetectionPipeline`, `GatingComponent`, `BlobExtractor`, `BlobScorer`, `Initializer`, `ShankRefiner`, `ModelSelector`, `ContactDetector`, `ArtifactWriter` |
| `pipelines/base.py` | `BaseDetectionPipeline` — stage execution, result scaffolding, component resolution, error handling |
| `diagnostics.py` | `DiagnosticsCollector` — counters, timing, reason codes, notes |
| `artifacts.py` | `ArtifactWriter`, `FileArtifactWriter`, `NullArtifactWriter` |
| `registry.py` | `PipelineRegistry`, `ComponentRegistry`, `get_default_registry()` |
| `bootstrap.py` | `register_builtin_pipelines()` — registers all 14 built-in pipelines |
| `pipelines/*.py` | Concrete pipeline implementations |

## Contracts

### DetectionContext (input)

```python
class DetectionContext(TypedDict, total=False):
    run_id: str                            # Unique execution ID
    ct: VolumeRef                          # Volume metadata
    arr_kji: Any                           # numpy array in K,J,I order
    spacing_xyz: tuple[float, float, float]
    config: dict[str, Any]                 # Pipeline parameters
    params: dict[str, Any]                 # Legacy alias for config
    ijk_kji_to_ras_fn: Any                 # Coordinate transform
    ras_to_ijk_fn: Any                     # Inverse transform
    center_ras: list[float]                # Head center
    components: dict[str, Any]             # Component overrides
    extras: dict[str, Any]                 # Arbitrary extra data
    artifact_writer: Any                   # ArtifactWriter instance
    logger: Any                            # Logging callback
```

### DetectionResult (output)

```python
class DetectionResult(TypedDict, total=False):
    schema_version: str                    # "seeg_detection_result.v1"
    pipeline_id: str
    pipeline_version: str
    run_id: str
    status: Literal["ok", "error"]
    trajectories: list[dict[str, Any]]     # Detected shank lines
    contacts: list[dict[str, Any]]         # Detected contact centers
    diagnostics: DetectionDiagnostics      # Timing, counts, notes
    artifacts: list[ArtifactRecord]        # Output file references
    warnings: list[str]
    error: dict[str, Any] | None
    meta: dict[str, Any]
```

### Data Records

```python
@dataclass(frozen=True)
class BlobRecord:          # One candidate metal blob
    blob_id: int
    centroid_ras: tuple[float, float, float]
    centroid_kji: tuple[float, float, float]
    voxel_count: int
    # + peak_hu, mean_hu, pca_axis_ras, length_mm, diameter_mm, elongation,
    #   depth_min_mm, depth_max_mm, scores, meta, tags

@dataclass(frozen=True)
class ShankModel:          # One detected trajectory model
    shank_id: str
    kind: Literal["line", "polyline", "spline", "unknown"]
    params: dict[str, Any]
    support: dict[str, float]
    assigned_blob_ids: tuple[int, ...]

@dataclass(frozen=True)
class ContactRecord:       # One detected contact center
    shank_id: str
    contact_id: str
    position_ras: tuple[float, float, float]
    confidence: float
```

## Component Interfaces

All interfaces are `Protocol` classes — implement the methods and the
component is valid, no inheritance required.

| Protocol | Method | Signature |
|----------|--------|-----------|
| `DetectionPipeline` | `run` | `(ctx) -> DetectionResult` |
| `GatingComponent` | `compute` | `(ctx, state) -> dict` |
| `BlobExtractor` | `extract` | `(ctx, state) -> list[BlobRecord]` |
| `BlobScorer` | `score` | `(ctx, blobs, state) -> list[BlobRecord]` |
| `Initializer` | `initialize` | `(ctx, blobs, state) -> list[ShankModel]` |
| `ShankRefiner` | `refine` | `(ctx, blobs, models, state) -> (list[ShankModel], dict)` |
| `ModelSelector` | `select` | `(ctx, blobs, models, state) -> list[ShankModel]` |
| `ContactDetector` | `detect` | `(ctx, blobs, models, state) -> list[ContactRecord]` |
| `ArtifactWriter` | `write_json` | `(name, payload) -> str` |

Pipelines resolve components at runtime via `resolve_component(ctx, key)`,
which checks `ctx.components[key]` first, then `self.default_components[key]`.

## BaseDetectionPipeline

All concrete pipelines inherit from `BaseDetectionPipeline` (in
`pipelines/base.py`). It provides:

### Result scaffolding

```python
result = self.make_result(ctx)            # Pre-filled DetectionResult
diag = self.diagnostics(result)           # DiagnosticsCollector wrapper
```

### Stage execution

```python
output = self.run_stage(
    ctx=ctx, result=result, diagnostics=diag,
    stage_name="my_stage",
    fn=lambda: self._do_something(ctx, cfg),
)
```

`run_stage` automatically:
- Times the stage and records `stage.my_stage.ms` in diagnostics
- Catches exceptions and wraps them as `StageExecutionError`
- Records failure reason codes

### Component resolution

```python
scorer = self.resolve_component(ctx, "blob_scorer")
# Checks ctx.components["blob_scorer"], then self.default_components["blob_scorer"]
```

### Error handling and finalization

```python
try:
    # ... stages ...
except Exception as exc:
    self.fail(ctx=ctx, result=result, diagnostics=diag, stage="...", exc=exc)
return self.finalize(result, diag, t_start)
# finalize() records total_ms and JSON-sanitizes the result
```

## DiagnosticsCollector

```python
diag.inc("blob_count", 42)               # Increment counter
diag.set_count("selected_count", 5)      # Set counter
diag.set_timing("fit_ms", 123.4)         # Record timing
diag.add_reason("rejected:bone", 3)      # Reason code
diag.note("skipped extension for short proposals")  # Narrative
diag.set_param("threshold_hu", 1900.0)   # Record config used
diag.set_extra("debug_array", [...])     # Custom data
```

## Pipeline Registry

```python
from shank_engine.registry import get_default_registry
from shank_engine.bootstrap import register_builtin_pipelines

registry = get_default_registry()
register_builtin_pipelines(registry)

# List available pipelines
registry.keys()  # ['blob_consensus_v1', 'blob_em_v2', ...]

# Run by name
result = registry.run("blob_em_v2", ctx)

# Or create and configure
pipeline = registry.create_pipeline("de_novo_seed_extend_v2")
result = pipeline.run(ctx)
```

## Registered Pipelines

| Key | Class | Strategy |
|-----|-------|----------|
| `blob_ransac_v1` | `BlobRansacV1Pipeline` | Legacy shank_core RANSAC adapter |
| `blob_em_v2` | `BlobEMV2Pipeline` | EM-based blob assignment and refinement |
| `blob_consensus_v1` | `BlobConsensusV1Pipeline` | Multi-strategy consensus voting |
| `blob_persistence_v1` | `BlobPersistenceV1Pipeline` | Multi-threshold persistence scoring |
| `blob_persistence_v2` | `BlobPersistenceV2Pipeline` | Persistence with lineage tracking |
| `hybrid_bead_string_v1` | `HybridBeadStringV1Pipeline` | Bead-chain and string-fit hybrid |
| `de_novo_hypothesis_select_v1` | `DeNovoHypothesisSelectV1Pipeline` | Hypothesis generation + global selection |
| `de_novo_seed_extend_v2` | `DeNovoSeedExtendV2Pipeline` | Seed-extend with refinement pass |
| `shank_axis_v1` | `ShankAxisV1Pipeline` | Geometry-first axis optimization |
| `shank_cluster_v1` | `ShankClusterV1Pipeline` | Orientation clustering |
| `shank_graph_v1` | `ShankGraphV1Pipeline` | Graph-based model fusion |
| `shank_grow_v1` | `ShankGrowV1Pipeline` | Iterative shank growth |
| `shank_hypothesis_v1` | `ShankHypothesisV1Pipeline` | Local hypothesis generation |
| `shank_stitch_v1` | `ShankStitchV1Pipeline` | Segment stitching |

## Adding a New Pipeline

1. Create `pipelines/my_pipeline_v1.py`:

```python
from ..pipelines.base import BaseDetectionPipeline

class MyPipelineV1(BaseDetectionPipeline):
    pipeline_id = "my_pipeline_v1"
    pipeline_version = "1.0.0"

    def run(self, ctx):
        import time
        t_start = time.perf_counter()
        result = self.make_result(ctx)
        diag = self.diagnostics(result)
        cfg = self._config(ctx)

        try:
            masks = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="gating",
                fn=lambda: self._build_masks(ctx, cfg),
            )
            proposals = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="proposals",
                fn=lambda: self._generate(ctx, masks, cfg),
            )
            selected = self.run_stage(
                ctx=ctx, result=result, diagnostics=diag,
                stage_name="selection",
                fn=lambda: self._select(proposals, cfg),
            )
            result["trajectories"] = selected
        except Exception as exc:
            self.fail(ctx=ctx, result=result, diagnostics=diag,
                      stage="unknown", exc=exc)

        return self.finalize(result, diag, t_start)
```

2. Register in `bootstrap.py`:

```python
from .pipelines.my_pipeline_v1 import MyPipelineV1
registry.register_pipeline("my_pipeline_v1", MyPipelineV1, overwrite=True)
```

3. Run:

```python
result = registry.run("my_pipeline_v1", ctx)
```

## Design Principles

- **No Slicer dependencies.** All contracts and pipeline code are pure Python.
  Volume data enters via `ctx.arr_kji` and coordinate transforms via
  `ctx.ijk_kji_to_ras_fn`.
- **Protocol-based interfaces.** No forced inheritance for components.
  Implement the method signature and it works.
- **Composable stages.** Each `run_stage` call is independent. Pipelines
  choose which stages to run and in what order.
- **Structured diagnostics.** Every pipeline emits the same diagnostics
  schema — counters, timing, reason codes, notes — making comparison
  across pipelines straightforward.
- **JSON-serializable results.** `sanitize_result()` converts numpy arrays
  and dataclasses to JSON-safe builtins before returning.
- **Lazy registration.** Pipeline classes are imported only when
  `register_builtin_pipelines()` is called, keeping import time low.
