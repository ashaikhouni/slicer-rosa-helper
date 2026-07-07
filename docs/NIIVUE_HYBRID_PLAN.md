# NiiVue hybrid viewer — plan

**Decision (from the spike):** do **not** migrate the whole viewer to NiiVue. NiiVue is *volume-first*; our viewer is *electrode-first* (clickable contacts, red-highlight, shafts, snap-slices, accept/reject, list↔3D sync), which NiiVue does weakly (connectome spheres). And two WebGL contexts can't share a depth buffer, so NiiVue can't compose *into* the 3D scene.

**Target = split by strength, as separate panels:**

- **three.js owns the 3D** — electrodes + selection, the parcellated brain mesh, DVR, MIP, camera. Keep it; it's the crown jewel. (DVR just landed.)
- **NiiVue owns the 2D** — the multiplanar slice panels **and** the registration QC overlay. This retires the hand-rolled slice canvases *and* the server-side `qc_render.py` PNG compositor.
- **The app shell owns coordination** — the review sidebar + a thin RAS-coordinate event bus wiring the two panels.

```
 ┌──────────────────────────┬───────────────────────────┐
 │  three.js 3D (Panel A)   │   NiiVue multiplanar (B)   │
 │  electrodes+mesh+DVR+MIP │   ax / cor / sag + xhair   │
 │  click contact ─────────────► set crosshair @ RAS     │
 │  slice planes ◄───────────── crosshair moved @ RAS    │
 ├──────────────────────────┴───────────────────────────┤
 │  Registration tab → NiiVue two-volume overlay (B')    │
 │  (CT+MRI, opacity/colormap; replaces qc_render.py)    │
 ├───────────────────────────────────────────────────────┤
 │  Review sidebar (app shell) — accept/reject/label     │
 │  row click ──► select(contact) ──► A + B respond      │
 └───────────────────────────────────────────────────────┘
```

## The enabling refactor: separate the 3D scene out

Today `export_view.py` emits **one** HTML page with everything fused (3D + hand-rolled slice canvases + controls). Step 1 is to extract the 3D scene into a **self-contained module with a documented message API**, so it stops owning the slice panels and can be driven by an external bus. Nothing else can happen cleanly until this decoupling exists.

**3D scene message API (already ~half there via `rosa:select` / `rosa:visibility`):**

| Direction | Message | Meaning |
|---|---|---|
| in  | `select {contact, shank}` | highlight + focus a contact |
| in  | `visibility {hideShanks, hideContacts}` | hide rejected |
| in  | `locate {ras:[x,y,z]}` | move slice planes / crosshair to a point |
| out | `selected {contact, shank, ras}` | user picked a contact in 3D |
| out | `located {ras}` | user moved the 3D crosshair/planes |

## Interaction: RAS mm is the lingua franca

Everything syncs on **RAS millimetre coordinates** + **contact identity** (`shank`+`index`) — never pixels or voxels. Contacts, the mesh, and the volumes all live in the CT/contact RAS frame, so:

- click contact in A → `selected{ras}` → B `nv.scene.crosshairPos = mm2frac(ras)`; sidebar highlights row.
- move NiiVue crosshair in B → `located{ras}` → A moves slice planes (+ optionally highlights nearest contact).
- sidebar row click → `select{contact}` → A focuses, B crosshairs.
- accept/reject → `visibility` → A hides the electrode.

A tiny pub/sub bus does this — `postMessage` if panels stay in iframes, or a shared module if single-page.

## What moves where

| Feature | Now | Target |
|---|---|---|
| 3D electrodes + selection/highlight | three.js | **three.js (keep)** |
| Brain mesh + parcellation + DVR + MIP | three.js | **three.js (keep)** |
| 2D slice panels (ax/cor/sag) | hand-rolled `<canvas>` | **NiiVue** |
| Registration QC overlay | server PNGs (`qc_render.py`) | **NiiVue** client overlay |
| Review sidebar (accept/reject/label) | app | **app (keep)** |
| crosshair ↔ 3D sync | none / ad-hoc | **RAS event bus** |

## Data contract (already produced)

- three.js: `scene.glb` (mesh + electrodes) in RAS.
- NiiVue: NIfTI volumes in the CT/contact frame — `ct_in_view`, `mri_in_view` (brain), and for QC the paired `*_in_mni`. NiiVue reads `.nii.gz` natively; no new export needed beyond what the pipeline already writes.

## Phases

1. **Extract the 3D scene** into a self-contained module + finalize the message API (in/out above). Pure refactor of `export_view` — no behaviour change. *(the "separate the 3D out" ask)*
2. **NiiVue slice panels** replace the hand-rolled canvases; wire crosshair ↔ 3D via the bus. Validate sync on a real case.
3. **Registration QC → NiiVue** two-volume overlay (opacity/colormap/scroll); delete `qc_render.py` + the server compositor + the Safari-repaint workarounds.
4. Layout/resize/colormap polish; single-page vs iframe decision.

## Gains / non-goals

- **Gain:** interactive client-side QC (retire the server renderer); proper linked multiplanar; the 3D keeps every electrode feature (no downgrade).
- **Non-goal:** NiiVue owning the 3D electrode scene, or compositing NiiVue *into* three.js (depth-buffer conflict).

**Sequencing:** do this only after the three.js version is otherwise complete (DVR ✓). It's a UI-architecture refactor, best tackled once the 3D feature set is stable.
