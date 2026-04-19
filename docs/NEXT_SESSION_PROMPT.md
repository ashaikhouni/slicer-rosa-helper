# Next-session resume prompt

Paste the block below as the first message to a new Claude Code
session in the `rosa_viewer` project.

---

Work on the **Contacts & Trajectory View** module
([`ContactsTrajectoryView/ContactsTrajectoryView.py`](ContactsTrajectoryView/ContactsTrajectoryView.py)).

Start by reading the project memory files, especially
`project_contact_pitch_v1.md` (production state) and
`project_contact_pitch_v1_next.md` (the finished plan), plus
[`docs/CONTACT_PITCH_V1_HANDOFF.md`](docs/CONTACT_PITCH_V1_HANDOFF.md)
for the detector. Auto Fit / Guided Fit / Manual Fit upstream are
clean (49 / 49 matched, 0 FP across T22 / T2 / T21 / T1); this next
session is purely about what happens AFTER a trajectory line is
published.

**Problem statement**: the module is named "Contacts & Trajectory
View" but it doesn't actually *detect* contacts from the CT — it
*synthesizes* them by placing fiducials at the assigned electrode
model's nominal pitch intervals along the fitted line. If the
electrode is slightly curved, a contact has drifted, or the model
assignment is wrong, the "contacts" don't match the real imaged
positions.

**Goal**: give the user two clear modes for producing contact
fiducials, both verifiable against the CT image.

## Mode A — trust the assigned electrode (manual / model-driven)

Current behaviour, but made explicit. Source of the electrode model:

1. User selects a model in the table (current UI), or
2. `Rosa.BestModelId` stamped on the line node by Auto Fit /
   Guided Fit (`contact_pitch_v1_fit.py` `suggest_model_id`).

Fit contacts from the library's `contact_center_offsets_from_tip_mm`
geometry along the line. User can override the tip position + tip
shift via the existing `TipAt` / `TipShift` columns.

## Mode B — peak-driven (automatic)

1. Enhance contrast along the trajectory axis. Options to try
   probe-first, pick the cleanest:
   - 1-D LoG σ=1 sampled on-axis (already computed by Auto Fit —
     we can stash `log_sigma1_kji` on the line node attribute or
     re-compute).
   - 1-D top-hat filter on raw CT HU.
   - Derivative of HU profile, peaks = local maxima of `|d²HU/ds²|`.
2. Peak-pick along the axis from skull_entry → end_ras. Each peak is
   a candidate contact position (arc-length `s`).
3. Match the peak sequence against every electrode in the
   library's selected vendor set; pick the model whose offset
   pattern matches best (residual `|s_detected − s_model|` sum).
4. Emit contact fiducials at the DETECTED peak positions (not the
   model's nominal) so downstream QC reflects real anatomy.

Fallback to Mode A when peaks are too sparse / noisy.

## Concrete deliverables

1. **Add a "Detection mode" combo** to the Contacts tab: *Model-driven*
   (Mode A, default) / *Peak-driven* (Mode B).
2. **Write a contact-peak-detection engine** in
   [`CommonLib/rosa_core/`](CommonLib/rosa_core/) (new module,
   probably `contact_peak_fit.py`) that takes:
   - RAS axis endpoints
   - CT array + IJK↔RAS matrices
   - Optional LoG volume (skip re-compute)
   - Vendor filter
   and returns: (model_id, per-contact positions, fit score).
3. **Wire it into `_run_contact_generation`** alongside the existing
   `rosa_core.generate_contacts` synthesis path.
4. **Update QC** to compare peak-detected contacts vs. the assigned
   model's nominal positions — flag drifts > 1 mm.

## Reference points

- Module file: [`ContactsTrajectoryView/ContactsTrajectoryView.py`](ContactsTrajectoryView/ContactsTrajectoryView.py)
  (~1,170 LOC). Widget + Logic in the same file.
- Contact synthesis today: `rosa_core.generate_contacts` called
  from `_run_contact_generation`.
- Electrode library: [`CommonLib/resources/electrodes/electrode_models.json`](CommonLib/resources/electrodes/electrode_models.json).
  Key field per model: `contact_center_offsets_from_tip_mm`
  (list of floats, arc-lengths from tip).
- Auto Fit already emits `Rosa.BestModelId` on every line node it
  publishes. Guided Fit does too. Manual lines don't — those need
  a model assignment (either explicit or peak-driven).
- Feature arrays from Auto Fit: `<CT>_ContactPitch_LoG_sigma1` is
  already registered as a Slicer volume by the widget. The engine
  can read it instead of recomputing.

## Verification

Regression suite still has to pass (`tests.deep_core.test_pipeline_dataset_contact_pitch_v1`
covers T22 default + auto + T2); no changes needed there.

For the new contact-peak engine, add a targeted test that seeds an
Auto-Fit trajectory and compares peak positions against the
subject's `contacts.tsv` ground truth from the dataset (`contact_label_dataset/labels/`).
Aim for ≤ 1.5 mm median per-contact error on T22 / T2 as the
baseline.

## Things NOT to touch

- `contact_pitch_v1_fit.py` — the Auto Fit pipeline is stable at
  4 / 4 subjects clean; don't drift those gates.
- `guided_fit_engine.py` — same.
- Trajectory origin naming (`auto_fit` / `guided_fit` / `manual`,
  roles `AutoFitTrajectoryLines` etc.) — stabilized.

Follow `feedback_probe_first.md` for any new filter choice —
probe a single subject's CT with candidate filters before wiring
anything into the module.
