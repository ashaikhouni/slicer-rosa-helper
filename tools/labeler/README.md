# SEEG GT labeler

Flask-based ground-truth annotation tool for SEEG datasets that follow the
`contact_label_dataset` layout (`ct/`, `rosa_helper_import/`, plus
per-subject `T{N}/masks/` and `T{N}/snap/` dirs produced by
`notebooks/seeded_fit/05_snap_ct_dataset.ipynb`).

## Launch

```bash
cd slicer-rosa-helper/tools
python -m labeler --dataset /Users/ammar/Dropbox/thalamus_subjects/seeg_localization/contact_label_dataset
```

Opens `http://localhost:5057/` in your browser. Per-trajectory annotations
are persisted **immediately** on Save to:

```
<dataset>/gt/trajectories_gt.tsv   ← one row per trajectory (canonical)
<dataset>/gt/contacts_gt.tsv       ← auto-derived from model + corrected tip
```

## Workflow per trajectory

1. Click a subject in the left sidebar.
2. Click a trajectory (first unlabeled is auto-selected).
3. Pick the electrode model from the **vendor + model** dropdowns.
4. Click on the slab to set landmarks:
   - **Tip** (deepest contact center — this anchors auto-derived contacts)
   - **Bone inner edge** (intracranial start; defines the picker window)
   - **Bolt start** (bolt outer edge, air side; optional)
   - **Bolt end** (bolt inner edge, brain side; optional)
5. The green dots that appear on the slab are the **auto-derived contacts**
   — the electrode-model offsets walked back from your tip along the
   fitted axis. They should overlay the visible contacts on the slab.
6. Click **Save** to persist. Auto-advances to the next trajectory.

## Keyboard shortcuts

- `s` or `Enter` — save + advance
- `→` or `n` — next trajectory
- `←` or `p` — previous trajectory
- `1` / `2` / `3` / `4` — activate the tip / bone / bolt-start / bolt-end landmark
- Right-click on slab — clear all landmarks for this trajectory

## Resume support

GT TSVs are loaded on startup. Re-opening shows checkmarks for already-
labeled trajectories. Clicking a labeled trajectory re-populates all
saved landmarks for review/edit.

## Curved-shank limitation (v1)

Auto-contact placement assumes the electrode is **straight** along the
fitted axis. For slightly curved shanks the green dots will diverge from
the visible contacts on the slab — flag in the notes; per-contact manual
override is a v2 feature.

## Architecture notes

- Slab views are **regenerated on demand** from the cached canonical CT
  + brain mask (in `<dataset>/T{N}/masks/`), giving native mm
  coordinates. First trajectory in a subject takes ~3-5 s; subsequent
  ones are near-instant (subject features cached in process memory).
- Auto-derived contacts use the
  `contact_center_offsets_from_tip_mm` from
  `CommonLib/rosa_core/resources/electrodes/electrode_models.json` and
  re-zero so the deepest contact sits AT the corrected tip.
- `contacts_gt.tsv` is regenerated atomically after every save —
  always consistent with the trajectories table.
