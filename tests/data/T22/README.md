# T22 fixtures

Single-subject fixtures used by PaCER-style picker probes and other
T22-specific tests that need both the post-op CT and per-contact
ground-truth model labels.

## Files

| file | tracked? | source | what it is |
|---|---|---|---|
| `T22_aligned_world_coords.txt` | yes | manual export, 2026-04-30 | per-contact GT with `model_id`. 9 trajectories (X01–X09), 117 contacts. RAS = post-registered frame. **This is the only place per-contact model GT exists for T22.** |
| `T22_contacts.tsv` | yes | dataset `contact_label_dataset/labels/` | auto-snapped contact labels (every row `coord_source=World`, `snap_status=unchanged`, `move_mm=0`). Not human-curated — see `project_contact_placement_native_ct_plan.md`. |
| `T22_shanks.tsv` | yes | dataset `contact_label_dataset/shanks/` | per-shank entry/target labels (the SEEG dataset's shank-level GT). |
| `T22_ct.nii.gz` | **NO** (gitignored) | dataset `post_registered_ct/T22_post_registered.nii.gz` (symlinked from `contact_label_dataset/ct/T22_ct.nii.gz`) | post-op CT, 18 MB. Refresh from dataset when the dataset moves. |

## Trajectories + GT models

Per `T22_aligned_world_coords.txt`:

| trajectory | n_contacts | model_id |
|---|---|---|
| X01 | 15 | DIXI-15CM |
| X02 | 15 | DIXI-15CM |
| X03 | 15 | DIXI-15CM |
| X04 | 15 | DIXI-15CM |
| X05 | 15 | DIXI-15AM |
| X06 | 12 | DIXI-12AM |
| X07 | 10 | DIXI-10AM |
| X08 | 10 | DIXI-10AM |
| X09 | 10 | DIXI-10AM |

## Refreshing the CT

```
cp /Users/ammar/Dropbox/thalamus_subjects/seeg_localization/post_registered_ct/T22_post_registered.nii.gz \
   tests/data/T22/T22_ct.nii.gz
```

(Or symlink, whichever you prefer for local dev.)
