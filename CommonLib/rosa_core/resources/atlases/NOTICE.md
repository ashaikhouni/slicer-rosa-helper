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

## Thalamic nuclei (MIAL / Najdenovska) — `thalamus_mial/ThalamicNuclei.nii.gz`, `thalamus_mial/Thalamic_Nuclei-ColorLUT.txt`

- **License:** CC BY-SA 4.0 (atlas data on Zenodo). ShareAlike — keep this notice
  and the LUT with the files; the bundled labelmap is unchanged except a dtype
  recast (float→uint8, max label 14) that preserves label values and the affine.
- **Citation:** Najdenovska E, Alemán-Gómez Y, Battistella G, et al. *In-vivo
  probabilistic atlas of human thalamic nuclei based on diffusion-weighted MRI.*
  Scientific Data 5:180270 (2018). https://doi.org/10.1038/sdata.2018.270
- **Source:** Zenodo https://doi.org/10.5281/zenodo.1405484 (MNI152 2009a
  nonlinear symmetric space).

## MNI ICBM152 2009a Nonlinear Symmetric — `templates/mni152_2009a_sym_T1.nii.gz`

- **License:** MNI / Louis Collins BSD-style permissive grant (same as the 2009c
  template below).
- **Citation:** Fonov VS, Evans AC, Botteron K, Almli CR, McKinstry RC, Collins DL.
  *Unbiased average age-appropriate atlases for pediatric studies.* NeuroImage
  54(1):313–327 (2011).
- **Source:** TemplateFlow `tpl-MNI152NLin2009aSym` (res-1 T1w), downsampled here
  to 2 mm and rescaled to uint8 to keep it small (MI registration is intensity-
  structure based, so this does not affect alignment).

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
