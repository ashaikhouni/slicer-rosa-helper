# Next-session resume prompt

Paste the block below as the first message to a new Claude Code
session in the `rosa_viewer` project.

---

Resume the `contact_pitch_v1` SEEG detector work. Start by reading
the project memory files, especially `project_contact_pitch_v1.md`
(current state) and `project_contact_pitch_v1_next.md` (the plan we
agreed). Also read `slicer-rosa-helper/docs/CONTACT_PITCH_V1_HANDOFF.md`
for the algorithm.

**Current state**: contact_pitch_v1 is in production. T22 9/9 0 FP,
T2 12/12 0 FP, T1/T3 visually clean. Progress reporting is plumbed.
Committed at `2090358`.

**Top priority this session**: library-driven walker + electrode
classification. Two concrete deliverables:

1. **Multi-family walker.** Replace the hardcoded `PITCH_MM = 3.5` in
   [`PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py`](PostopCTLocalization/postop_ct_localization/contact_pitch_v1_fit.py)
   with a search over each selected family's offset signature from
   [`CommonLib/resources/electrodes/electrode_models.json`](CommonLib/resources/electrodes/electrode_models.json).
   UI: add a multi-select in the "Contact Pitch v1" tab
   ([`deep_core_widget.py` `_build_contact_pitch_v1_tab`](PostopCTLocalization/postop_ct_localization/deep_core_widget.py))
   so users pick which families are in play. DIXI-BM (9 mm insulation
   jump) and DIXI-CM (13 mm insulation jump) need the walker to
   accept those specific gaps as legal steps, not just a looser
   uniform tolerance.

2. **Electrode classification step.** After stage-1 + extension, for
   each trajectory build the 1D LoG (or peak-HU) profile along the
   intracranial portion, then find the best-fitting model from the
   selected families. Emit classified electrode id, per-contact RAS
   positions (arc-length-snapped), and fit score. This is the actual
   user deliverable.

**Verification command** (should still pass after every change):

```bash
cd /Users/ammar/Dropbox/rosa_viewer/slicer-rosa-helper
/Users/ammar/miniforge3/envs/shankdetect/bin/python3 \
    -m unittest tests.deep_core.test_pipeline_dataset_contact_pitch_v1
```

T22 baseline: ≥ 8/9 matched, ≤ 10 FP. T2: ≥ 12/12, ≤ 10 FP.

Follow `feedback_probe_first.md` if you need to iterate on the
walker — probe-first, don't change the pipeline until the probe
matches expectations. And `feedback_mixin_helpers.md` / `feedback_factory_indentation.md`
still apply to the Slicer extension code.

Secondary ideas (from `project_contact_pitch_v1_next.md`) are Frangi
cost analysis and a blob-graph probe — skip unless (1) is already
done or user redirects.
