# ROSA Cohort Feature — Design

*Two features: (A) a cohort registry that becomes the home page, and (B) an MNI
group-contact viewer. Transform directions + compose order are CONFIRMED against
the code; the PHI rule is the tightened version. The three.js import-map detail
in §4.3 is reasoned (its adversarial check returned a degenerate result) —
validate empirically when implementing.*

---

## 1. TL;DR

The request is **two features, not one**: (A) a **cohort registry** that replaces the home-page card grid with a sortable table, and (B) an **MNI cohort viewer** — one shared glass brain with every subject's contacts pooled by affine registration, select-to-highlight. The data store stays the **filesystem job store**; we add exactly one derived per-case file (`regcache/contacts_mni.tsv`), one committed resource mesh, two endpoints, and four `summarize_case` fields. SQLite is a later, disposable *index*, gated on cross-case region queries — not now.

**The three questions, answered directly:**

- **Registry on the home page? Yes — it *is* the home page.** Evolve `#panel-cases` in place (card list → table). Pure reads over the `summarize_case` seam that already exists; lowest-cost, highest-frequency win.
- **MNI viewer on the home page? No — its own mode.** Reached by a nav button → `showStep("cohort")`. Different mental mode (exploration vs. management), applies only to the CerebrA-labeled *subset*, and must not tax the one screen that has to paint instantly on every launch.
- **Too heavy? No.** Runtime cost is trivial: dozens of subjects × ~150 contacts = low-thousands of points = one `THREE.Points` draw call + one static glass mesh. The only real costs are engineering (new page + endpoints) and honest atlas-path *gating* — scoping concerns, not framerate concerns.

---

## 2. Data store

### 2.1 Source of truth: the filesystem job store (do NOT add a DB now)

`<work_root>/<12-hex-id>/manifest.json` per job, rehydrated on restart by globbing `<work_root>/*/manifest.json` (`jobs.py` `_rehydrate`). Already a rehydratable, PHI-boundary-respecting registry backing store. A SQL engine would add a migration surface, a second source of truth that can diverge from the TSVs, and a PyInstaller freeze burden — for a cohort of dozens. **Everything below is a derived cache: disposable, rebuildable from TSVs + tfms, so it can never *be* the thing that's wrong.**

### 2.2 Derived layer (a): per-case contacts-in-MNI cache

- **File:** `<case_dir>/regcache/contacts_mni.tsv`, one row per contact. `regcache/` is the right home — it is the per-case cache dir, already holds the two source tfms, and `runner.delete` tears it down with the case (no orphan index).
- **Columns:** `trajectory, contact_index, name, mni_x, mni_y, mni_z, hemisphere, region, accepted`.
  - `mni_x/y/z` = warped coordinates (§4.1).
  - `hemisphere` = `"R" if mni_x > 0 else "L"` — no `hemisphere` field exists in `ReviewDoc` (`models.py:88-99`), so derive it; MNI-x sign about the AC midline is the honest place.
  - `region` + `accepted` read from **`review.json`** (`ReviewContact` carries `region`, `accepted`, `x/y/z`, `shank`, `index`). Use review.json (approved/edited state; respects rejects + relabels) as primary; fall back to `contacts.tsv` for coords only when a case has tfms but no review yet (region → null).
- **When computed — eager AND lazy:**
  - **Eagerly** at label time: append a `cohort-export` step to the label job (`jobs.py:485-508`), right after `view-results`, when both tfms are fresh in `regcache/`. Keeps the cohort view instant.
  - **Lazily** on first cohort view: the endpoint (§4.3) computes-and-caches any eligible case with a missing/stale file → pre-existing and re-labeled cases self-heal with no backfill migration.
- **Invalidation:** rebuild when `mtime(contacts_mni.tsv) < max(mtime(review.json), mtime(contacts.tsv), mtime(t1_to_ct.tfm), mtime(mni_MNI152NLin2009cSym_to_t1.tfm))`. Cheap `stat`s only; a relabel bumps review.json's mtime → auto-recompute.
- **Gate:** emit only when **both** `regcache/t1_to_ct.tfm` and `regcache/mni_MNI152NLin2009cSym_to_t1.tfm` exist (§4.2).

### 2.3 Derived layer (b): the shared MNI glass brain — prebuild ONCE, commit as a resource

**Prebuild offline; do not compute-on-first-view.** The pool template never changes (`mni152_2009c_sym_T1.nii.gz`, the CerebrA space), and the mesh is byte-identical for every user forever. Per-view marching-cubes would add a first-view latency spike and pull scikit-image onto the frozen runtime for nothing.

- **Build tool (one-time, `tools/build_mni_glass.py`):** Otsu-threshold `mni152_2009c_sym_T1.nii.gz` → fill holes → largest component → `brain_mesh.surface_from_mask(mask, smooth_sigma≈1.2, step_size=2, taubin_iterations≈12)` (`brain_mesh.py:564`) → `glb_writer` with the `alphaMode:BLEND` glass material (the pattern in `export_view.py`). Eyeball once.
- **Commit to:** `CommonLib/rosa_core/resources/atlases/templates/mni152_2009c_sym_glass.glb`. Vertices are MNI RAS mm — the same frame the warped contacts land in — so it drops into the scene with **no per-view transform**.

### 2.4 Growth path (c): SQLite as a derived, disposable index — later, not now

Stay on TSVs until a query is **cross-case AND region-selective** (flat per-case files can't answer without a full scan) — e.g. "which subjects sampled left hippocampus?", cohort-wide region coverage counts/heatmaps, or N in the hundreds where per-request re-parsing gets slow. Then add **`<work_root>/cohort.sqlite` as a pure index, never a source of truth** — rebuilt by re-reading every eligible case's `contacts_mni.tsv`, deletable/regenerable at any time, so it can never diverge (nothing writes to it but the rebuild):

```sql
CREATE TABLE contact (
    subject_id     TEXT    NOT NULL,   -- JobStatus.label (clinician-typed)
    case_id        TEXT    NOT NULL,   -- 12-hex job id (stable join key)
    trajectory     TEXT    NOT NULL,
    contact_index  INTEGER NOT NULL,
    name           TEXT,               -- channel name, e.g. "LAC1"
    mni_x          REAL    NOT NULL,
    mni_y          REAL    NOT NULL,
    mni_z          REAL    NOT NULL,
    hemisphere     TEXT,               -- 'L' | 'R'  (sign of mni_x)
    region         TEXT,               -- from review.json; NULL if unlabeled
    accepted       INTEGER NOT NULL,   -- 0/1
    PRIMARY KEY (case_id, trajectory, contact_index)
);
CREATE INDEX ix_contact_region      ON contact(region);
CREATE INDEX ix_contact_hemi_region ON contact(hemisphere, region);
```

The registry and viewer endpoints never *depend* on this table existing — it's a query accelerator they fall back off of. **Refresh model (decided):** the DB is rebuilt/synced from the filesystem **on request** (a "rebuild index" action, or a cheap mtime-diff scan that reindexes only changed case folders) — not a live filesystem watcher.

**Forward note — the index will soon hold non-image data (per user).** When SQLite arrives it won't only mirror `contacts_mni.tsv`; it will carry clinical metadata entered *in the app* (seizure semiology, outcomes, EEG findings, …). That data is **NOT derivable from the case files**, so at that point the DB stops being purely disposable and splits into two kinds of tables:

- **Derived tables** (`contact`, per-case summaries) — rebuildable from the case folders; safe to drop and regenerate.
- **Primary tables** (clinical variables) — a *source of truth* in their own right; they need their own persistence + backup story (and ideally a per-case sidecar file, e.g. `<case_dir>/clinical.json`, so the "folder = source of truth / backup" invariant still holds and the DB can rebuild even the clinical tables from folders). Design the schema with this split explicit when we get there.

---

## 3. Feature A — Cohort registry (the home page)

Replace the `caseCard` list (`app.js:1103-1121`, rendered by `renderCases()` `app.js:1082`) with a sortable table in the **same** `#panel-cases` (`index.html:23`). The `+ New case` / `Import localization` actions, search box, and all/detected/imported filter stay put. A render swap inside one panel, not a new screen.

### Columns

| Column | Source | Notes |
|---|---|---|
| Subject ID | `label` | clinician-typed subject id (**not** an enforced pseudonym — §7) |
| Kind | `kind` | detected / imported badge |
| MRI | `has_mri` | check / dash (`bool(status.t1)`) |
| # electrodes | `n_shanks` | right-aligned |
| # contacts | `n_contacts` | right-aligned |
| Labeled | `labeled` | check / dash |
| **Atlas** *(new)* | newest succeeded child label job's `params.atlas` | CerebrA / FastSurfer / THOMAS / — |
| **MNI-poolable** *(new)* | derived: both tfms in `regcache/` | the gate for Feature B; an actionable dot |
| Date | `created_at` | already formatted client-side |
| **Scan id** *(new)* | short `ct_hash` (first 8) | PHI-safe scan fingerprint; disambiguates same-named subjects |

Drop a "CT" column — every case has one; it's a checkmark that says nothing.

**The only backend change is three derived fields in `summarize_case` (`cases.py:70-83`)**, which already has `JobStatus` + `job_dir`:
- `ct_hash`: from `manifest.json` `params.ct_hash` (in manifest, not the DTO).
- `atlas`: newest succeeded child label job (`params.parent == case_id`), read `params.atlas` — same lookup `app.py:660-674` already does for rebuilds.
- `mni_eligible`: both `regcache/*.tfm` `is_file()`.

No new endpoint, no new data model for Feature A.

### Interaction
- **Click row → `openCase(c.id)`** (`app.js:1136`); keep the inline delete `×` as a trailing-cell action with `stopPropagation`.
- **Sort** any column client-side over the already-fetched `state.cases` (same instant no-refetch pattern search/filter use). Default newest-first.
- **Search** — add `ct_hash` to the existing `label`+`id` match set so a pasted scan hash finds its case.
- Empty / no-results states reused verbatim.

---

## 4. Feature B — MNI cohort viewer (its own mode)

### 4.1 The verified warp pipeline — CONFIRMED

For a contact in CT-RAS `p_ct`:

```python
from rosa_core.registration import (
    load_transform, transform_to_4x4_ras, apply_transform_to_points_ras)

A = transform_to_4x4_ras(load_transform(regcache/"t1_to_ct.tfm"))
#   register_rigid_mi(fixed=CT, moving=T1)  →  A maps  CT-RAS → T1-RAS
B = transform_to_4x4_ras(load_transform(regcache/"mni_MNI152NLin2009cSym_to_t1.tfm"))
#   register_affine_mi(fixed=T1, moving=MNI) →  B maps  T1-RAS → MNI-RAS

p_mni = apply_transform_to_points_ras(p_ct, B @ A)   # apply A first, then B
```

- **Compose order = `B @ A`.** `apply_transform_to_points_ras` left-multiplies (`out = (M @ h.T).T`), so `M = B @ A` applies A (CT→T1) then B (T1→MNI). Corroborated by `atlas_provider_headless.py:212-215` ("CT→MNI = affine ∘ rigid, apply rigid first").
- **No `inv()` on either leg.** `transform_to_4x4_ras` returns the *fixed→moving* RAS matrix; both files are stored in exactly the direction we consume. Confirmed by `acpc.py:71-72,113`.
- **Precedent caveat:** `fit_rosa.py:441-456` is the right *pattern* but line 452's `reg.matrix_ras_4x4` does not exist on `RegistrationResult` (it exposes `fixed_to_moving_ras_4x4` / `moving_to_fixed_ras_4x4`). Illustrative only; the construction above does not depend on it.
- **Filename:** `mni_MNI152NLin2009cSym_to_t1.tfm` (CerebrA's `space` = `MNI152NLin2009cSym`).

### 4.2 Gating (honest)
- **Eligible ⇔** both `regcache/t1_to_ct.tfm` and `regcache/mni_MNI152NLin2009cSym_to_t1.tfm` present — the CerebrA-through-T1 path.
- **Excluded until a CerebrA pass:** FastSurfer / deepmriprep / THOMAS are native-space (only `t1_to_ct.tfm`); a CerebrA run *without* a patient T1 (direct MNI→CT affine) persists no reusable tfm. `mni_eligible:false` lets the UI offer "run a CerebrA pass to pool this subject."
- **Pool-space coherence:** consume ONLY the `MNI152NLin2009cSym` tfm. Harvard-Oxford/Schaefer (`NLin6Asym`) and thalamus (`2009aSym`) write differently-named tfms in different MNI variants — mixing injects a few-mm cross-space offset. Standardize the pool on 2009c-Sym = CerebrA default = the committed glass brain's space.

### 4.3 The scene, serving, and endpoints
- **One shared MNI glass brain**: the committed `mni152_2009c_sym_glass.glb` (§2.3), vendored three.js + GLTFLoader, `alphaMode:BLEND` reused from `export_view.py`.
- **Per-subject contact clouds — one `THREE.Points` with a per-vertex color buffer.** Dozens × ~150 = a few thousand points = one draw call; per-vertex color/size gives subject-tinting and select-highlight for free. **New client-side code** — today contacts are baked as individual GLB cylinder nodes; there is no `THREE.Points`/`InstancedMesh` in app code. The cohort page reads a compact per-subject array and builds the cloud in JS; it does not reuse the per-case GLB baking path.
- **Serving**: a new same-origin iframe page `app/rosa_service/web/cohort/index.html`, served like the editor (small route → `FileResponse`). Add **one shared static mount** — `app.mount("/assets/three", StaticFiles(directory=viewer_assets/three))` — **before** the catch-all SPA mount at `/` (`app.py:755`), same as `/api`.
  - **Import-map — use ABSOLUTE values, not relative** (this is the corrected detail): the cohort page lives at a different URL path than the shared mount, so a relative `./three/` would resolve against the *page* URL, not `/assets/three/`. Write the import map as:
    ```html
    <script async src="/assets/three/es-module-shims.js"></script>
    <script type="importmap">
    { "imports": { "three": "/assets/three/three.module.js",
                   "three/addons/": "/assets/three/addons/" } }</script>
    ```
    es-module-shims is still required; bare specifiers are still rejected. (`_IMPORTMAP_LOCAL` uses relative `./three/` only because the per-case viewer copies `three/` *beside* the page.) **Validate empirically** — this leg was not code-verified.

**Endpoints:**

`GET /api/v1/cohort/contacts` — iterate `runner.list()` filtered exactly like `list_cases`; for each `mni_eligible` case ensure `regcache/contacts_mni.tsv` is fresh (lazy compute-and-cache via `warp_review_to_mni` on miss/stale), then read it:

```json
{ "space": "MNI152NLin2009cSym",
  "subjects": [
    {"id": "3fa9c1e2b077", "label": "T14", "color": "#3b82f6",
     "contacts": [[x, y, z, "Left-Hippocampus", "L"], ...]} ] }
```
Per-subject deterministic `color` = golden-ratio hue keyed on `case_id` (cf. `brain_mesh._label_hue_rgb`).

`GET /api/v1/cohort/brain.glb` — a one-line `FileResponse` of the committed glass mesh.

### 4.4 Select-to-highlight (reuse, don't reinvent)
Reuse from the per-case viewer: OrbitControls rig + frame-to-bounds, the selection beacon + `selectContact()`, and the `rosa:selected` / `rosa:select` postMessage bridge. The "list" is the cohort table / subject legend: clicking a subject posts `rosa:select {subject}` into the cohort iframe, which **brightens + enlarges that subject's points and dims the rest** (a write into the Points color/size attribute) with optional camera focus. Stable id = `[case_id, trajectory, contact_index]`. Hover tooltip: `subject · electrode · contact · region`.

---

## 5. Phasing — three independently shippable phases

**Phase 1 — Registry table (ship first; highest value/effort).** Card list → sortable table over the existing `/cases` + `summarize_case` seam; add Atlas / MNI-poolable / short-ct_hash in that one function. Zero registration, zero 3D. ~a day.

**Phase 2 — `contacts_mni` caching + `warp_review_to_mni` (invisible plumbing that unblocks Phase 3).** Add the `rosa_core` warp function, the `cohort-export` CLI subcommand, the label-job step + lazy endpoint recompute. Backfill existing CerebrA cases lazily on first view. Ships with no visible change beyond the poolable dot going green.

**Phase 3 — The shared-MNI group scene (the payoff).** Bake `mni152_2009c_sym_glass.glb`, add `web/cohort/` + `/assets/three` mount + the two `/api/v1/cohort/*` endpoints, wire the nav button + `showStep("cohort")` (add `"cohort"` to `nosteps`), render the glass brain + `THREE.Points`, reuse OrbitControls + beacon + postMessage bridge. No registration in the request path.

---

## 6. File-level touch map

**Backend — `app/rosa_service/`**
- `cases.py` `summarize_case` (`:70-83`) — add `ct_hash`, `atlas`, `mni_eligible`. **Phase 1.** No path fields (§7).
- `app.py` — `GET /api/v1/cohort/contacts`; `GET /api/v1/cohort/brain.glb`; serve `web/cohort/index.html` (cf. `:604-615`); `app.mount("/assets/three", StaticFiles(...))` before the SPA mount `:755`. **Phase 3.**
- `jobs.py` — append `cohort-export` step to the label job (after `view_step`, `:485-508`). **Phase 2.**

**Core / CLI**
- New `CommonLib/rosa_core/cohort.py`: `warp_review_to_mni(case_dir, pool_space="MNI152NLin2009cSym") -> rows` — reads review.json (fallback contacts.tsv), loads both tfms, composes `B @ A`, derives hemisphere, joins region. **Single source of the warp logic.** **Phase 2.**
- New CLI `rosa-agent cohort-export <case_dir> -o regcache/contacts_mni.tsv`. **Phase 2.**
- New one-time `tools/build_mni_glass.py`; new committed resource `.../templates/mni152_2009c_sym_glass.glb`. **Phase 3.**

**Frontend — `app/rosa_service/web/`**
- `index.html` (`#panel-cases`) + `app.js` (`renderCases`, `caseCard`) — table + `ct_hash` search. **Phase 1.**
- `app.js` `showStep` — add `"cohort"` to `nosteps`; new nav button. **Phase 3.**
- New `app/rosa_service/web/cohort/index.html` (+ JS). **Phase 3.**

---

## 7. Caveats & open decisions

- **Affine-only accuracy.** The warp is 12-DOF (MNI→T1) ∘ 6-DOF (T1→CT) — affine, no SyN/deformable anywhere. Cross-subject MNI positions are good to a cortical-registration tolerance (several mm near cortex): a *coverage / where-did-we-sample* overview, **not** a basis for millimetric cross-subject localization. The per-contact `region` shown is still the **native-space** atlas label at the true CT coordinate — only the *display position* is warped. Permanent line under the canvas: *"Contacts pooled in MNI by affine registration — a coverage overview, not millimetric localization."*
- **Atlas-path gating.** Only CerebrA-labeled-with-T1 cases are poolable. State "N of M cases poolable" — never let a partial cohort silently imply "this is everyone."
- **PHI strategy (decided 2026-07-16): convert everything to NIfTI on import and store it *inside the case folder under neutral names* (`ct.nii.gz`, `t1.nii.gz`).** This is the real fix and it composes three wins:
  - **DICOM tag PHI** (patient name, MRN, DOB, dates, institution) — gone the moment the image is NIfTI. This is the big bucket. (`POST /dicom-to-nifti` already drops DICOM headers on conversion; make it the enforced default and copy the result into the case dir.)
  - **Path/filename leak** — killed, because the app then references `<case_dir>/ct.nii.gz`, not the source path. This *supersedes* the earlier "never surface a path" rule and is strictly better; keep surfacing only `label` + short `ct_hash` regardless.
  - **Self-contained folders** — each case becomes a portable, backup-worthy directory (exactly what the "folder = source of truth / backup" model wants). Today CT/T1 are external paths NOT copied in; this changes that.
  - **Two residuals NIfTI conversion does NOT remove — be honest about the threat model:**
    1. **NIfTI header free-text** (`descrip`, `aux_file`, `db_name`, `intent_name`) — some converters (dcm2niix) write the series description here. Trivial: zero these on conversion.
    2. **The face in the voxels** — a head CT/MRI is identifiable by 3-D surface reconstruction; format conversion cannot touch this. **DECIDED (PARKED, not yet built): shear on storage, after registration.** Flow: import → convert to NIfTI → register CT→T1 + T1→MNI on the **faced originals** (cache the transforms) → **Quickshear**-deface the CT + MRI using the brain mask we already compute (deepbet/brainchop for T1; `brain_mask_in_ct.nii.gz` for CT) → store **only** the sheared, neutral-named NIfTIs in the case folder; discard the faced originals. Net: registration is full-fidelity (transforms computed pre-shear) and the folder is defaced + PHI-safe **at rest**, not just at export. No new heavy dep — Quickshear places a shear plane with margin outside the brain hull (keeps skull/bolt context; removes only face + air; never touches brain, skull base, or electrodes). **Residual:** any *later* re-registration (relabel to a new atlas space, AC-PC recompute) runs on the stored *defaced* image → possible ~sub-mm bias; acceptable since primary transforms are cached and re-registration is rare.
  - `ct_hash` (SHA-256 over CT bytes) stays the PHI-free scan identity. **`label` is clinician-typed free text, NOT an enforced pseudonym** — PHI-safe only by site convention; consider a soft warning if it looks like a real name/MRN. The real-name↔pseudonym keymap stays in the separate `deidentify-ros` CLI, outside the app.
- **Open decisions:**
  1. `label` sanitization — leave as free text (status quo), or add a soft warning if it looks like a real name/MRN?
  2. "Run CerebrA to pool" affordance — should a `mni_eligible:false` row offer a one-click CerebrA-relabel, or just display the excluded state?
  3. SQLite trigger — confirm we defer the index until an actual cross-case region query is needed (§2.4).
