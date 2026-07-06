# Bundled atlas third-party notices

The files under this directory are third-party neuroimaging atlases and
templates, redistributed here under their own licenses (not the project's).
Each is listed in `atlases.json`. This NOTICE satisfies the attribution
requirement of those licenses; keep it alongside the data when redistributing.

Atlases are tagged with a `license_tier` in `atlases.json`:

- **permissive** (CC0 / CC-BY / MIT / BSD) — CerebrA, thalamus_mial, Harvard-Oxford,
  Schaefer, all templates. Redistributable including commercially.
- **noncommercial** (CC-BY-NC / FreeSurfer) — SUIT cerebellum, FreeSurfer/Iglesias
  thalamic nuclei. Bundled here for **non-commercial research use** (the
  FieldTrip/MNE model); do not use commercially, and keep this notice with them.

All-rights-reserved / registration-gated atlases (THOMAS, Brainnetome) grant no
redistribution right and remain bring-your-own. The atlas DATA licenses are
independent of the app's own (permissive) code license.

---

## FreeSurfer / Iglesias thalamic nuclei — `thalamus_iglesias/`  *(non-commercial)*

- **License:** FreeSurfer Software License (permissive Part-A/B grant; retain the
  MGH copyright notice). Bundled under the non-commercial tier; a commercial
  redistribution should confirm the atlas-data terms with MGH.
- **Citation:** Iglesias JE, Insausti R, Lerma-Usabiaga G, et al. *A probabilistic
  atlas of the human thalamic nuclei combining ex vivo MRI and histology.*
  NeuroImage 183:314–326 (2018).
- **Source:** FreeSurfer `SubfieldAtlasesICBMspace` (Thalamus.zip); the 4D
  probability maps argmaxed to a discrete max-prob labelmap. MNI152 2009c-Sym space.

## SUIT cerebellum (Diedrichsen 2009) — `suit_cerebellum/`  *(non-commercial)*

- **License:** CC BY-NC 3.0 (Diedrichsen lab). Non-commercial use only; attribution
  required; keep this notice with the files.
- **Citation:** Diedrichsen J, Balsters JH, Flavell J, Cussans E, Ramnani N. *A
  probabilistic MR atlas of the human cerebellum.* NeuroImage 46(1):39–46 (2009).
- **Source:** DiedrichsenLab/cerebellar_atlases `Diedrichsen_2009` (space-MNI dseg).
  FSL MNI152 space.

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

## Harvard-Oxford — `harvard_oxford/HarvardOxford.nii.gz`, `harvard_oxford/HarvardOxford_labels.tsv`

- **License:** CC BY-SA 4.0 (FSL Harvard-Oxford atlas). ShareAlike — keep this
  notice with the files.
- **Citation:** Harvard-Oxford cortical & subcortical structural atlases, Harvard
  Center for Morphometric Analysis (Makris N, Goldstein JM, Kennedy D, et al.);
  distributed with FSL.
- **Source:** `nilearn.datasets.fetch_atlas_harvard_oxford` (cort + sub
  maxprob-thr25-2mm), merged into one labelmap (cortical 1–48 + true subcortical
  49–63; WM/cortex/ventricle filler labels dropped). FSL MNI152 space.

## Schaefer 2018 — `schaefer/Schaefer400-7.nii.gz`, `schaefer/Schaefer400-7_labels.tsv`

- **License:** MIT-style (Schaefer/Yeo, CBIG). Redistribution + commercial use
  with attribution.
- **Citation:** Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ,
  Eickhoff SB, Yeo BTT. *Local-Global Parcellation of the Human Cerebral Cortex.*
  Cerebral Cortex 28:3095–3114 (2018).
- **Source:** `nilearn.datasets.fetch_atlas_schaefer_2018` (400 parcels, 7
  networks, 2 mm). FSL MNI152 space.

## MNI ICBM152 Nonlinear 6th-gen (FSL) — `templates/mni152_nlin6_sym_T1.nii.gz`

- **License:** MNI / Louis Collins BSD-style permissive.
- **Source:** TemplateFlow `tpl-MNI152NLin6Asym` (res-02 T1w), rescaled to uint8.
  Shared registration template for Harvard-Oxford + Schaefer (both FSL MNI152).

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
