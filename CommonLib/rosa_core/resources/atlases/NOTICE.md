# Bundled atlas third-party notices

The files under this directory are third-party neuroimaging atlases and
templates, redistributed here under their own licenses (not the project's).
Each is listed in `atlases.json`. This NOTICE satisfies the attribution
requirement of those licenses; keep it alongside the data when redistributing.

Only permissively-licensed, redistributable resources are bundled. Atlases
with non-commercial or all-rights-reserved terms (FreeSurfer thalamic nuclei,
AAL3, THOMAS, Morel, JHU, SUIT/Diedrichsen cerebellum) are deliberately **not**
bundled and remain bring-your-own.

---

## CerebrA — `cerebra/CerebrA.nii.gz`, `cerebra/CerebrA_LabelDetails.csv`

- **License:** CC0 1.0 Universal (public-domain dedication of the atlas data).
- **Citation:** Manera AL, Dadar M, Fonov V, Collins DL. *CerebrA, registration
  and the human brain atlas.* Scientific Data 7:237 (2020).
  https://doi.org/10.1038/s41597-020-0557-9
- **Source:** https://gin.g-node.org/anamanera/CerebrA
- **Modification note:** the bundled labelmap is the upstream `CerebrA.nii`
  recast from float32 to uint8 (max label value 102) and gzipped. Label values
  and the voxel-to-world affine are preserved exactly; only the storage dtype
  changed. CC0 imposes no restriction on modification or redistribution.

## MNI ICBM152 2009c Nonlinear Symmetric — `templates/mni152_2009c_sym_T1.nii.gz`

- **License:** MNI / Louis Collins BSD-style permissive grant — "Permission to
  use, copy, modify, and distribute … for any purpose and without fee is hereby
  granted, provided that the above copyright notice appear in all copies."
  Copyright (C) 1993–2004 Louis Collins, McConnell Brain Imaging Centre,
  Montreal Neurological Institute, McGill University.
- **Citation:** Fonov VS, Evans AC, Botteron K, Almli CR, McKinstry RC,
  Collins DL. *Unbiased average age-appropriate atlases for pediatric studies.*
  NeuroImage 54(1):313–327 (2011). https://doi.org/10.1016/j.neuroimage.2010.07.033
- **Source:** TemplateFlow `tpl-MNI152NLin2009cSym` (res-1 T1w), which
  redistributes the McGill BIC template under the license above.
  https://github.com/templateflow/tpl-MNI152NLin2009cSym
