# Refactor Regression Checklist

Last updated: 2026-03-01

Use this checklist after structural changes.

## 1) Loader

- Load ROSA case from `01 Loader`.
- Verify base volume and additional ROSA volumes appear and align.
- Verify trajectories are published (`Imported ROSA`, planned lines).

## 2) Contact Workflow

- Open `02 Contacts & Trajectory View`.
- Generate contacts and models.
- Edit one trajectory line and run `Update From Edited Trajectories`.
- Verify contacts/models update in place.

## 3) Postop CT Localization

- Guided fit path:
  - detect candidates
  - fit selected/all
  - verify guided-fit trajectories publish and become selectable in Contacts module
- De novo path:
  - run detect
  - verify generated trajectories appear and can generate contacts

## 4) QC Metrics

- Verify QC table populates after contact generation.
- Verify QC disables when planned/final prerequisites are absent.

## 5) Atlas Sources + Labeling

- Load/register FreeSurfer and THOMAS sources.
- Verify image/transform registry rows are populated.
- Run atlas labeling and verify assignment table rows are created.

## 6) Navigation Burn

- Burn a THOMAS nucleus into selected MRI.
- Verify output volume alignment.
- Optional: export burned output as DICOM and re-import for sanity.

## 7) Export Center

- Run `full_bundle`; verify manifest and expected outputs.
- Run `contacts_only`; verify only contact-centric outputs are written.
- Verify selected export frame is reflected in coordinate columns.

## 8) Sanity Script

```bash
cd <repo>
python3 tools/phase8_sanity.py
```

Expected:
- compile failures: 0
- legacy bridge references only allowed in docs/tooling guards
