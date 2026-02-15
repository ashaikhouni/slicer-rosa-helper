# Refactor Regression Checklist

Use this checklist after each refactor step to confirm behavior remains unchanged.

## 1) ROSA Load
- Load case folder with `.ros` + `DICOM/`
- Confirm reference volume is centered and other volumes are transformed/hardened
- Confirm trajectories and `Plan_*` lines are created

## 2) Contact Workflow
- Open `V1 Contact Labels`
- Generate contacts and electrode models
- Edit one trajectory and run `Update From Edited Trajectories`
- Confirm contacts/models update in place

## 3) Auto-fit Workflow
- Select postop CT and run `Detect Candidates`
- Run `Fit Selected` and `Apply Fit to Trajectories`
- Confirm preview `AutoFit_*` line is removed after apply
- Confirm contacts regenerate automatically

## 4) QC Metrics
- Confirm `Trajectory QC Metrics` table populates after contact generation
- Confirm table disables if no contacts are generated

## 5) Export
- Run `Export Aligned NIfTI + Coordinates/QC`
- Confirm output includes:
  - aligned NIfTI volumes
  - `*_aligned_world_coords.txt`
  - `*_planned_trajectory_points.csv`
  - `*_qc_metrics.csv`

## 6) FreeSurfer Integration
- Add recon-all MRI to scene via Slicer `Add Data`
- Register FS MRI -> ROSA
- Load `pial` surfaces from subject `surf/`
- Confirm surfaces transform to ROSA frame
- Annotation test A (nibabel): enable annotation and verify applied count > 0
- Annotation test B (fallback): set `RosaHelper.nib = None` and verify fallback applies annotations when extension is available

## 7) Final Smoke
- Restart Slicer and run one complete load->contacts->export cycle
