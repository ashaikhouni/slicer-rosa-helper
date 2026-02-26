"""Widget mixin for ROSA Helper UI workflows.

This file groups UI section builders and feature handlers so the module
entrypoint stays focused on wiring.
"""

import os

from __main__ import ctk, qt, slicer

from rosa_core import (
    compute_qc_metrics,
    electrode_length_mm,
    generate_contacts,
    load_electrode_library,
    lps_to_ras_point,
    model_map,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_workflow.export_profiles import get_export_profile, profile_names


class RosaHelperWidgetMixin:
    """UI helper and workflow handlers mixed into `RosaHelperWidget`."""

    def _build_contact_ui(self):
        """Create V1 contact labeling controls."""
        self.contactSection = ctk.ctkCollapsibleButton()
        self.contactSection.text = "V1 Contact Labels"
        self.contactSection.collapsed = True
        self.layout.addWidget(self.contactSection)

        contact_layout = qt.QFormLayout(self.contactSection)

        self.contactTable = qt.QTableWidget()
        self.contactTable.setColumnCount(6)
        self.contactTable.setHorizontalHeaderLabels(
            [
                "Trajectory",
                "Traj Length (mm)",
                "Electrode Model",
                "Elec Length (mm)",
                "Tip At",
                "Tip Shift (mm)",
            ]
        )
        self.contactTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(2, qt.QHeaderView.Stretch)
        self.contactTable.horizontalHeader().setSectionResizeMode(3, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(4, qt.QHeaderView.ResizeToContents)
        self.contactTable.horizontalHeader().setSectionResizeMode(5, qt.QHeaderView.ResizeToContents)
        self.contactTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        contact_layout.addRow(self.contactTable)

        defaults_row = qt.QHBoxLayout()
        self.defaultModelCombo = qt.QComboBox()
        self.defaultModelCombo.setMinimumContentsLength(14)
        defaults_row.addWidget(self.defaultModelCombo)
        self.applyModelAllButton = qt.QPushButton("Apply model to all")
        self.applyModelAllButton.clicked.connect(self.onApplyModelAllClicked)
        defaults_row.addWidget(self.applyModelAllButton)
        defaults_row.addStretch(1)
        contact_layout.addRow("Default model", defaults_row)

        self.contactsNodeNameEdit = qt.QLineEdit("ROSA_Contacts")
        contact_layout.addRow("Output markups", self.contactsNodeNameEdit)

        self.createModelsCheck = qt.QCheckBox("Create electrode models")
        self.createModelsCheck.setChecked(True)
        self.createModelsCheck.setToolTip(
            "Create per-electrode 3D model nodes (shaft + contacts) after contact generation."
        )
        contact_layout.addRow("Model option", self.createModelsCheck)

        self.generateContactsButton = qt.QPushButton("Generate Contact Fiducials")
        self.generateContactsButton.clicked.connect(self.onGenerateContactsClicked)
        self.generateContactsButton.setEnabled(False)
        contact_layout.addRow(self.generateContactsButton)

        self.updateContactsButton = qt.QPushButton("Update From Edited Trajectories")
        self.updateContactsButton.clicked.connect(self.onUpdateContactsClicked)
        self.updateContactsButton.setEnabled(False)
        self.updateContactsButton.setToolTip(
            "Recompute contacts/models from current trajectory line markups without creating new nodes."
        )
        contact_layout.addRow(self.updateContactsButton)

        self.bundleExportDirEdit = qt.QLineEdit()
        self.bundleExportDirEdit.setPlaceholderText("Required (no automatic default path)")
        contact_layout.addRow("Aligned export folder", self.bundleExportDirEdit)

        self.exportFrameSelector = slicer.qMRMLNodeComboBox()
        self.exportFrameSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.exportFrameSelector.noneEnabled = True
        self.exportFrameSelector.addEnabled = False
        self.exportFrameSelector.removeEnabled = False
        self.exportFrameSelector.setMRMLScene(slicer.mrmlScene)
        self.exportFrameSelector.setToolTip(
            "Contact XYZ export frame. If empty, defaults to ROSA base/reference volume."
        )
        contact_layout.addRow("Export coordinate frame", self.exportFrameSelector)

        self.exportProfileCombo = qt.QComboBox()
        for name in profile_names():
            self.exportProfileCombo.addItem(name)
        default_idx = self.exportProfileCombo.findText("full_bundle")
        if default_idx >= 0:
            self.exportProfileCombo.setCurrentIndex(default_idx)
        self.exportProfileCombo.setToolTip("Controls which artifact types are written to export output.")
        contact_layout.addRow("Export profile", self.exportProfileCombo)

        self.exportBundleButton = qt.QPushButton("Export Aligned NIfTI + Coordinates/QC")
        self.exportBundleButton.clicked.connect(self.onExportAlignedBundleClicked)
        self.exportBundleButton.setEnabled(False)
        contact_layout.addRow(self.exportBundleButton)

    def _build_trajectory_view_ui(self):
        """Create controls to align a slice view to a selected trajectory."""
        self.trajectoryViewSection = ctk.ctkCollapsibleButton()
        self.trajectoryViewSection.text = "Trajectory Slice View"
        self.trajectoryViewSection.collapsed = True
        self.layout.addWidget(self.trajectoryViewSection)

        view_layout = qt.QFormLayout(self.trajectoryViewSection)

        self.trajectorySelector = qt.QComboBox()
        view_layout.addRow("Trajectory", self.trajectorySelector)

        self.sliceViewSelector = qt.QComboBox()
        self.sliceViewSelector.addItems(["Red", "Yellow", "Green"])
        view_layout.addRow("Slice view", self.sliceViewSelector)

        self.sliceModeSelector = qt.QComboBox()
        self.sliceModeSelector.addItems(["long", "down"])
        self.sliceModeSelector.setToolTip(
            "'long': slice along electrode shaft, 'down': view along shaft axis."
        )
        view_layout.addRow("Mode", self.sliceModeSelector)

        self.showPlannedCheck = qt.QCheckBox("Show planned trajectories")
        self.showPlannedCheck.setChecked(False)
        self.showPlannedCheck.toggled.connect(self.onShowPlannedToggled)
        view_layout.addRow(self.showPlannedCheck)

        self.alignSliceButton = qt.QPushButton("Align Slice to Trajectory")
        self.alignSliceButton.setEnabled(False)
        self.alignSliceButton.clicked.connect(self.onAlignSliceClicked)
        view_layout.addRow(self.alignSliceButton)

    def _build_freesurfer_ui(self):
        """Create controls for FreeSurfer MRI registration and surface import."""
        self.freesurferSection = ctk.ctkCollapsibleButton()
        self.freesurferSection.text = "FreeSurfer Integration (V1)"
        self.freesurferSection.collapsed = True
        self.layout.addWidget(self.freesurferSection)

        fs_layout = qt.QFormLayout(self.freesurferSection)

        self.fsFixedSelector = slicer.qMRMLNodeComboBox()
        self.fsFixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fsFixedSelector.noneEnabled = True
        self.fsFixedSelector.addEnabled = False
        self.fsFixedSelector.removeEnabled = False
        self.fsFixedSelector.setMRMLScene(slicer.mrmlScene)
        self.fsFixedSelector.setToolTip("ROSA base/reference volume (fixed image)")
        fs_layout.addRow("ROSA base volume", self.fsFixedSelector)

        self.fsMovingSelector = slicer.qMRMLNodeComboBox()
        self.fsMovingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.fsMovingSelector.noneEnabled = True
        self.fsMovingSelector.addEnabled = False
        self.fsMovingSelector.removeEnabled = False
        self.fsMovingSelector.setMRMLScene(slicer.mrmlScene)
        self.fsMovingSelector.setToolTip("MRI used for recon-all (moving image)")
        fs_layout.addRow("FreeSurfer MRI", self.fsMovingSelector)

        self.fsInitModeCombo = qt.QComboBox()
        self.fsInitModeCombo.addItems(["useGeometryAlign", "useMomentsAlign"])
        self.fsInitModeCombo.setCurrentText("useGeometryAlign")
        fs_layout.addRow("Init mode", self.fsInitModeCombo)

        self.fsTransformNameEdit = qt.QLineEdit("FS_to_ROSA")
        fs_layout.addRow("Output transform", self.fsTransformNameEdit)

        self.fsRegisterButton = qt.QPushButton("Register FS MRI -> ROSA")
        self.fsRegisterButton.clicked.connect(self.onRegisterFSMRIToRosaClicked)
        fs_layout.addRow(self.fsRegisterButton)

        self.fsSubjectDirSelector = ctk.ctkPathLineEdit()
        self.fsSubjectDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.fsSubjectDirSelector.setToolTip(
            "FreeSurfer subject directory (contains surf/) or direct surf/ directory"
        )
        if hasattr(self.fsSubjectDirSelector, "currentPathChanged"):
            self.fsSubjectDirSelector.currentPathChanged.connect(
                lambda _p: self._refresh_fs_parcellation_combo()
            )
        fs_layout.addRow("FreeSurfer subject", self.fsSubjectDirSelector)

        fs_parc_row = qt.QHBoxLayout()
        self.fsParcellationCombo = qt.QComboBox()
        self.fsParcellationCombo.addItem("all available")
        fs_parc_row.addWidget(self.fsParcellationCombo)
        self.fsRefreshParcellationButton = qt.QPushButton("Refresh")
        self.fsRefreshParcellationButton.clicked.connect(self.onRefreshFSParcellationClicked)
        fs_parc_row.addWidget(self.fsRefreshParcellationButton)
        fs_layout.addRow("Parcellation volume", fs_parc_row)

        self.fsApplyTransformVolumesCheck = qt.QCheckBox("Apply FS->ROSA transform to parcellations")
        self.fsApplyTransformVolumesCheck.setChecked(True)
        fs_layout.addRow(self.fsApplyTransformVolumesCheck)

        self.fsHardenVolumeCheck = qt.QCheckBox("Harden parcellation transforms")
        self.fsHardenVolumeCheck.setChecked(False)
        self.fsHardenVolumeCheck.setToolTip(
            "Leave off to preserve integer label values. Enable only when a hardened copy is required."
        )
        fs_layout.addRow(self.fsHardenVolumeCheck)

        self.fsLoadParcellationButton = qt.QPushButton("Load Parcellation Volumes")
        self.fsLoadParcellationButton.clicked.connect(self.onLoadFSParcellationsClicked)
        fs_layout.addRow(self.fsLoadParcellationButton)

        self.fsApplyParcellationLUTCheck = qt.QCheckBox("Apply LUT to parcellation volumes")
        self.fsApplyParcellationLUTCheck.setChecked(True)
        self.fsApplyParcellationLUTCheck.setToolTip(
            "Use FreeSurferLabels or selected LUT for aparc/aseg volume colorization in slice views."
        )
        fs_layout.addRow(self.fsApplyParcellationLUTCheck)

        self.fsCreateParcellation3DCheck = qt.QCheckBox("Create 3D geometry from parcellations")
        self.fsCreateParcellation3DCheck.setChecked(False)
        self.fsCreateParcellation3DCheck.setToolTip(
            "Create segmentation closed-surface nodes from loaded parcellation volumes for 3D display."
        )
        fs_layout.addRow(self.fsCreateParcellation3DCheck)

        self.fsSurfaceSetCombo = qt.QComboBox()
        self.fsSurfaceSetCombo.addItems(["pial", "white", "pial+white", "inflated"])
        self.fsSurfaceSetCombo.setCurrentText("pial")
        fs_layout.addRow("Surface set", self.fsSurfaceSetCombo)

        self.fsUseAnnotationCheck = qt.QCheckBox("Load annotation scalars")
        self.fsUseAnnotationCheck.setChecked(True)
        fs_layout.addRow(self.fsUseAnnotationCheck)

        self.fsAnnotationCombo = qt.QComboBox()
        self.fsAnnotationCombo.addItems(["aparc", "aparc.a2009s", "aparc.DKTatlas", "custom"])
        self.fsAnnotationCombo.setCurrentText("aparc")
        fs_layout.addRow("Annotation", self.fsAnnotationCombo)

        self.fsCustomAnnotationEdit = qt.QLineEdit()
        self.fsCustomAnnotationEdit.setPlaceholderText("Custom annotation name (without lh./rh. and .annot)")
        fs_layout.addRow("Custom annot", self.fsCustomAnnotationEdit)

        self.fsLUTPathSelector = ctk.ctkPathLineEdit()
        self.fsLUTPathSelector.filters = ctk.ctkPathLineEdit.Files
        self.fsLUTPathSelector.setToolTip(
            "Optional color LUT text file (for example FreeSurferColorLUT.txt). "
            "If empty, a default FreeSurfer color table node is used when available."
        )
        fs_layout.addRow("Annotation LUT", self.fsLUTPathSelector)

        self.fsApplyTransformCheck = qt.QCheckBox("Apply FS->ROSA transform")
        self.fsApplyTransformCheck.setChecked(True)
        fs_layout.addRow(self.fsApplyTransformCheck)

        self.fsHardenSurfaceCheck = qt.QCheckBox("Harden loaded surface transforms")
        self.fsHardenSurfaceCheck.setChecked(False)
        fs_layout.addRow(self.fsHardenSurfaceCheck)

        self.fsLoadSurfacesButton = qt.QPushButton("Load FreeSurfer Surfaces")
        self.fsLoadSurfacesButton.clicked.connect(self.onLoadFSSurfacesClicked)
        fs_layout.addRow(self.fsLoadSurfacesButton)

    def _build_thomas_ui(self):
        """Create controls for THOMAS thalamus mask registration/import."""
        self.thomasSection = ctk.ctkCollapsibleButton()
        self.thomasSection.text = "THOMAS Thalamus Integration (V1)"
        self.thomasSection.collapsed = True
        self.layout.addWidget(self.thomasSection)

        th_layout = qt.QFormLayout(self.thomasSection)

        self.thomasFixedSelector = slicer.qMRMLNodeComboBox()
        self.thomasFixedSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.thomasFixedSelector.noneEnabled = True
        self.thomasFixedSelector.addEnabled = False
        self.thomasFixedSelector.removeEnabled = False
        self.thomasFixedSelector.setMRMLScene(slicer.mrmlScene)
        self.thomasFixedSelector.setToolTip("ROSA base/reference volume (fixed image)")
        th_layout.addRow("ROSA base volume", self.thomasFixedSelector)

        self.thomasMovingSelector = slicer.qMRMLNodeComboBox()
        self.thomasMovingSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.thomasMovingSelector.noneEnabled = True
        self.thomasMovingSelector.addEnabled = False
        self.thomasMovingSelector.removeEnabled = False
        self.thomasMovingSelector.setMRMLScene(slicer.mrmlScene)
        self.thomasMovingSelector.setToolTip("MRI used to run THOMAS segmentation")
        th_layout.addRow("THOMAS MRI", self.thomasMovingSelector)

        self.thomasInitModeCombo = qt.QComboBox()
        self.thomasInitModeCombo.addItems(["useGeometryAlign", "useMomentsAlign"])
        self.thomasInitModeCombo.setCurrentText("useGeometryAlign")
        th_layout.addRow("Init mode", self.thomasInitModeCombo)

        self.thomasTransformNameEdit = qt.QLineEdit("THOMAS_to_ROSA")
        th_layout.addRow("Output transform", self.thomasTransformNameEdit)

        self.thomasRegisterButton = qt.QPushButton("Register THOMAS MRI -> ROSA")
        self.thomasRegisterButton.clicked.connect(self.onRegisterThomasMRIToRosaClicked)
        th_layout.addRow(self.thomasRegisterButton)

        self.thomasDicomDirSelector = ctk.ctkPathLineEdit()
        self.thomasDicomDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.thomasDicomDirSelector.setToolTip(
            "Optional: DICOM series directory for navigation MRI import (single series folder preferred)."
        )
        th_layout.addRow("Nav MRI DICOM dir", self.thomasDicomDirSelector)

        self.thomasImportDicomButton = qt.QPushButton("Import DICOM MRI")
        self.thomasImportDicomButton.clicked.connect(self.onImportThomasDicomClicked)
        th_layout.addRow(self.thomasImportDicomButton)

        self.thomasMaskDirSelector = ctk.ctkPathLineEdit()
        self.thomasMaskDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.thomasMaskDirSelector.setToolTip(
            "Directory containing THOMAS outputs (searched recursively for *THALAMUS*.nii/.nii.gz)"
        )
        th_layout.addRow("THOMAS output dir", self.thomasMaskDirSelector)

        self.thomasApplyTransformCheck = qt.QCheckBox("Apply THOMAS->ROSA transform")
        self.thomasApplyTransformCheck.setChecked(True)
        th_layout.addRow(self.thomasApplyTransformCheck)

        self.thomasHardenCheck = qt.QCheckBox("Harden loaded thalamus transforms")
        self.thomasHardenCheck.setChecked(True)
        th_layout.addRow(self.thomasHardenCheck)

        self.thomasLoadMasksButton = qt.QPushButton("Load THOMAS Thalamus Masks")
        self.thomasLoadMasksButton.clicked.connect(self.onLoadThomasMasksClicked)
        th_layout.addRow(self.thomasLoadMasksButton)

        self.thomasBurnInputSelector = slicer.qMRMLNodeComboBox()
        self.thomasBurnInputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.thomasBurnInputSelector.noneEnabled = True
        self.thomasBurnInputSelector.addEnabled = False
        self.thomasBurnInputSelector.removeEnabled = False
        self.thomasBurnInputSelector.setMRMLScene(slicer.mrmlScene)
        self.thomasBurnInputSelector.setToolTip(
            "MRI volume to burn selected THOMAS nucleus into (typically DICOM-loaded navigation MRI)."
        )
        th_layout.addRow("Burn input MRI", self.thomasBurnInputSelector)

        self.thomasBurnAutoRegisterCheck = qt.QCheckBox("Auto-register THOMAS MRI -> Burn input")
        self.thomasBurnAutoRegisterCheck.setChecked(False)
        self.thomasBurnAutoRegisterCheck.setToolTip(
            "Advanced fallback. Enable only when Burn input MRI is not already aligned to ROSA base."
        )
        th_layout.addRow(self.thomasBurnAutoRegisterCheck)

        self.thomasBurnSideCombo = qt.QComboBox()
        self.thomasBurnSideCombo.addItems(["Left", "Right", "Both"])
        self.thomasBurnSideCombo.setCurrentText("Both")
        th_layout.addRow("Nucleus side", self.thomasBurnSideCombo)

        nucleus_row = qt.QHBoxLayout()
        self.thomasBurnNucleusCombo = qt.QComboBox()
        self.thomasBurnNucleusCombo.setEditable(True)
        self.thomasBurnNucleusCombo.addItem("CM")
        self.thomasBurnNucleusCombo.setCurrentText("CM")
        nucleus_row.addWidget(self.thomasBurnNucleusCombo)
        self.thomasRefreshNucleiButton = qt.QPushButton("Refresh")
        self.thomasRefreshNucleiButton.clicked.connect(self.onRefreshThomasNucleiClicked)
        nucleus_row.addWidget(self.thomasRefreshNucleiButton)
        th_layout.addRow("Nucleus", nucleus_row)

        self.thomasBurnFillValueSpin = qt.QDoubleSpinBox()
        self.thomasBurnFillValueSpin.setRange(-32768.0, 32767.0)
        self.thomasBurnFillValueSpin.setDecimals(1)
        self.thomasBurnFillValueSpin.setValue(1200.0)
        self.thomasBurnFillValueSpin.setSingleStep(50.0)
        self.thomasBurnFillValueSpin.setSuffix(" HU")
        th_layout.addRow("Burn fill value", self.thomasBurnFillValueSpin)

        self.thomasBurnOutputNameEdit = qt.QLineEdit("THOMAS_Burned_MRI")
        self.thomasBurnOutputNameEdit.setToolTip("Name for the generated burned scalar volume.")
        th_layout.addRow("Output volume name", self.thomasBurnOutputNameEdit)

        self.thomasBurnButton = qt.QPushButton("Register + Burn Nucleus")
        self.thomasBurnButton.clicked.connect(self.onBurnThomasNucleusClicked)
        th_layout.addRow(self.thomasBurnButton)

        self.thomasDicomExportDirSelector = ctk.ctkPathLineEdit()
        self.thomasDicomExportDirSelector.filters = ctk.ctkPathLineEdit.Dirs
        self.thomasDicomExportDirSelector.setToolTip(
            "Output directory for exported burned DICOM series (one file per slice)."
        )
        th_layout.addRow("DICOM export dir", self.thomasDicomExportDirSelector)

        self.thomasDicomSeriesDescriptionEdit = qt.QLineEdit("THOMAS_BURNED")
        self.thomasDicomSeriesDescriptionEdit.setToolTip("SeriesDescription tag for exported burned series.")
        th_layout.addRow("DICOM series description", self.thomasDicomSeriesDescriptionEdit)

        self.thomasBurnExportButton = qt.QPushButton("Register + Burn + Export DICOM")
        self.thomasBurnExportButton.clicked.connect(self.onBurnAndExportThomasDicomClicked)
        th_layout.addRow(self.thomasBurnExportButton)

    def _build_atlas_labeling_ui(self):
        """Create contact-to-atlas labeling controls."""
        self.atlasLabelSection = ctk.ctkCollapsibleButton()
        self.atlasLabelSection.text = "Atlas Contact Labeling (V1)"
        self.atlasLabelSection.collapsed = True
        self.layout.addWidget(self.atlasLabelSection)

        form = qt.QFormLayout(self.atlasLabelSection)

        self.atlasFSVolumeCombo = qt.QComboBox()
        self.atlasFSVolumeCombo.addItem("(none)")
        form.addRow("FreeSurfer atlas", self.atlasFSVolumeCombo)

        self.atlasThomasCombo = qt.QComboBox()
        self.atlasThomasCombo.addItem("(none)")
        form.addRow("Thalamus atlas", self.atlasThomasCombo)

        self.atlasWMVolumeCombo = qt.QComboBox()
        self.atlasWMVolumeCombo.addItem("(none)")
        form.addRow("White matter atlas", self.atlasWMVolumeCombo)

        self.atlasPreferThomasCheck = qt.QCheckBox("Prefer THOMAS when available")
        self.atlasPreferThomasCheck.setChecked(True)
        form.addRow(self.atlasPreferThomasCheck)

        self.atlasRefreshButton = qt.QPushButton("Refresh Atlas Sources")
        self.atlasRefreshButton.clicked.connect(self.onRefreshAtlasSourcesClicked)
        form.addRow(self.atlasRefreshButton)

        self.atlasAssignButton = qt.QPushButton("Assign Contacts to Atlas")
        self.atlasAssignButton.clicked.connect(self.onAssignContactsToAtlasClicked)
        form.addRow(self.atlasAssignButton)

    def _build_qc_ui(self):
        """Create QC table that updates automatically after contact generation."""
        self.qcSection = ctk.ctkCollapsibleButton()
        self.qcSection.text = "Trajectory QC Metrics"
        self.qcSection.collapsed = True
        self.layout.addWidget(self.qcSection)

        qc_layout = qt.QFormLayout(self.qcSection)
        self.qcStatusLabel = qt.QLabel("QC metrics are unavailable until contacts are generated.")
        qc_layout.addRow(self.qcStatusLabel)

        self.qcTable = qt.QTableWidget()
        self.qcTable.setColumnCount(8)
        self.qcTable.setHorizontalHeaderLabels(
            [
                "Trajectory",
                "Entry Radial (mm)",
                "Target Radial (mm)",
                "Mean Contact Radial (mm)",
                "Max Contact Radial (mm)",
                "RMS Contact Radial (mm)",
                "Angle (deg)",
                "Matched Contacts",
            ]
        )
        self.qcTable.horizontalHeader().setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        for col in range(1, 8):
            self.qcTable.horizontalHeader().setSectionResizeMode(col, qt.QHeaderView.ResizeToContents)
        self.qcTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        self.qcTable.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        qc_layout.addRow(self.qcTable)
        self._set_qc_enabled(False, "QC metrics are unavailable until contacts are generated.")

    def _build_autofit_ui(self):
        """Create controls for automatic postop CT alignment of trajectories."""
        self.autoFitSection = ctk.ctkCollapsibleButton()
        self.autoFitSection.text = "Auto Align to Postop CT (V1)"
        self.autoFitSection.collapsed = True
        self.layout.addWidget(self.autoFitSection)

        form = qt.QFormLayout(self.autoFitSection)

        self.postopCTSelector = slicer.qMRMLNodeComboBox()
        self.postopCTSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.postopCTSelector.noneEnabled = True
        self.postopCTSelector.addEnabled = False
        self.postopCTSelector.removeEnabled = False
        self.postopCTSelector.setMRMLScene(slicer.mrmlScene)
        form.addRow("Postop CT", self.postopCTSelector)

        self.autoThresholdSpin = qt.QDoubleSpinBox()
        self.autoThresholdSpin.setRange(-2000.0, 6000.0)
        self.autoThresholdSpin.setDecimals(1)
        self.autoThresholdSpin.setValue(1800.0)
        self.autoThresholdSpin.setSingleStep(50.0)
        form.addRow("Threshold (HU)", self.autoThresholdSpin)

        self.resetCTWindowButton = qt.QPushButton("Reset CT Window")
        self.resetCTWindowButton.clicked.connect(self.onResetCTWindowClicked)
        form.addRow(self.resetCTWindowButton)

        self.autoRoiRadiusSpin = qt.QDoubleSpinBox()
        self.autoRoiRadiusSpin.setRange(0.5, 20.0)
        self.autoRoiRadiusSpin.setDecimals(2)
        self.autoRoiRadiusSpin.setValue(2.5)
        self.autoRoiRadiusSpin.setSingleStep(0.25)
        self.autoRoiRadiusSpin.setSuffix(" mm")
        form.addRow("ROI radius", self.autoRoiRadiusSpin)

        self.autoMaxAngleSpin = qt.QDoubleSpinBox()
        self.autoMaxAngleSpin.setRange(0.0, 90.0)
        self.autoMaxAngleSpin.setDecimals(1)
        self.autoMaxAngleSpin.setValue(12.0)
        self.autoMaxAngleSpin.setSingleStep(0.5)
        self.autoMaxAngleSpin.setSuffix(" deg")
        form.addRow("Max angle", self.autoMaxAngleSpin)

        self.autoMaxDepthShiftSpin = qt.QDoubleSpinBox()
        self.autoMaxDepthShiftSpin.setRange(0.0, 50.0)
        self.autoMaxDepthShiftSpin.setDecimals(2)
        self.autoMaxDepthShiftSpin.setValue(20.0)
        self.autoMaxDepthShiftSpin.setSingleStep(0.5)
        self.autoMaxDepthShiftSpin.setSuffix(" mm")
        form.addRow("Max depth shift", self.autoMaxDepthShiftSpin)

        self.autoFitTrajectorySelector = qt.QComboBox()
        form.addRow("Selected trajectory", self.autoFitTrajectorySelector)

        btn_row = qt.QHBoxLayout()
        self.detectCandidatesButton = qt.QPushButton("Detect Candidates")
        self.detectCandidatesButton.clicked.connect(self.onDetectCandidatesClicked)
        btn_row.addWidget(self.detectCandidatesButton)
        self.fitSelectedButton = qt.QPushButton("Fit Selected")
        self.fitSelectedButton.clicked.connect(self.onFitSelectedClicked)
        btn_row.addWidget(self.fitSelectedButton)
        self.fitAllButton = qt.QPushButton("Fit All")
        self.fitAllButton.clicked.connect(self.onFitAllClicked)
        btn_row.addWidget(self.fitAllButton)
        form.addRow(btn_row)

        self.applyFitButton = qt.QPushButton("Apply Fit to Trajectories")
        self.applyFitButton.clicked.connect(self.onApplyFitClicked)
        form.addRow(self.applyFitButton)
        self._set_autofit_buttons_enabled(False)

    def _set_autofit_buttons_enabled(self, enabled):
        """Enable/disable auto-fit action buttons."""
        enabled = bool(enabled)
        self.fitSelectedButton.setEnabled(enabled)
        self.fitAllButton.setEnabled(enabled)
        self.applyFitButton.setEnabled(enabled and bool(self.autoFitResults))

    def _autofit_candidate_count(self):
        """Return number of detected postop CT candidate points."""
        candidates = self.autoFitCandidatesLPS
        if candidates is None:
            return 0
        size_attr = getattr(candidates, "shape", None)
        if size_attr is not None:
            return int(candidates.shape[0])
        return len(candidates)

    def _load_electrode_library(self):
        """Load bundled electrode models used by V1 contact generation."""
        try:
            data = load_electrode_library()
            self.modelsById = model_map(data)
            self.modelIds = sorted(self.modelsById.keys())
        except Exception as exc:
            self.modelsById = {}
            self.modelIds = []
            self.log(f"[electrodes] failed to load model library: {exc}")
            return

        self.defaultModelCombo.clear()
        self.defaultModelCombo.addItem("")
        for model_id in self.modelIds:
            self.defaultModelCombo.addItem(model_id)
        self.log(f"[electrodes] loaded {len(self.modelIds)} models")

    def _build_model_combo(self):
        """Create a combo box populated with electrode model IDs."""
        combo = qt.QComboBox()
        combo.addItem("")
        for model_id in self.modelIds:
            combo.addItem(model_id)
        return combo

    def _build_tip_at_combo(self):
        """Create tip anchor selector for contact placement."""
        combo = qt.QComboBox()
        combo.addItems(["target", "entry"])
        return combo

    def _build_tip_shift_spinbox(self):
        """Create spinbox for tip shift in millimeters."""
        spin = qt.QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-50.0, 50.0)
        spin.setSingleStep(0.25)
        spin.setValue(0.0)
        spin.setSuffix(" mm")
        return spin

    def _widget_text(self, widget):
        """Return current text from Qt widgets across Python bindings."""
        if widget is None:
            return ""
        text_attr = getattr(widget, "currentText", "")
        return text_attr() if callable(text_attr) else text_attr

    def _widget_value(self, widget):
        """Return numeric value from Qt widgets across Python bindings."""
        if widget is None:
            return 0.0
        value_attr = getattr(widget, "value", 0.0)
        value = value_attr() if callable(value_attr) else value_attr
        return float(value)

    def _set_readonly_text_item(self, row, column, text):
        """Set a read-only table item string value."""
        item = qt.QTableWidgetItem(text)
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.contactTable.setItem(row, column, item)

    def _set_qc_enabled(self, enabled, message=""):
        """Enable/disable QC section and update status message."""
        self.qcSection.setEnabled(bool(enabled))
        self.qcTable.setEnabled(bool(enabled))
        if message:
            self.qcStatusLabel.setText(message)

    def _set_qc_table_item(self, row, column, text):
        """Set one read-only item in QC table."""
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.qcTable.setItem(row, column, item)

    def _collect_planned_trajectory_map(self):
        """Read planned line markups (`Plan_*`) into trajectory dictionaries."""
        return self.logic.trajectory_scene.collect_planned_trajectory_map()

    def _compute_qc_rows(self):
        """Compute per-trajectory QC metrics using planned vs current contacts/lines."""
        if not self.lastGeneratedContacts:
            return [], "QC disabled: generate contacts first."

        planned_map = self._collect_planned_trajectory_map()
        if not planned_map:
            return [], "QC disabled: no planned trajectories (Plan_*) found."

        if not self.lastAssignments.get("assignments"):
            return [], "QC disabled: no electrode assignments available."

        final_map = self._build_trajectory_map_with_scene_overrides()
        assignments = self.lastAssignments

        try:
            planned_contacts = generate_contacts(list(planned_map.values()), self.modelsById, assignments)
        except Exception as exc:
            return [], f"QC disabled: failed to generate planned contacts ({exc})."
        rows = compute_qc_metrics(
            planned_trajectories_by_name=planned_map,
            final_trajectories_by_name=final_map,
            planned_contacts=planned_contacts,
            final_contacts=self.lastGeneratedContacts,
        )

        if not rows:
            return [], "QC disabled: no matching planned/final trajectories with contacts."
        return rows, f"QC metrics computed for {len(rows)} trajectories."

    def _refresh_qc_metrics(self):
        """Refresh QC table state from current trajectories, assignments, and contacts."""
        rows, status = self._compute_qc_rows()
        self.lastQCMetricsRows = rows
        self.qcTable.setRowCount(0)
        if not rows:
            self._set_qc_enabled(False, status)
            return

        self._set_qc_enabled(True, status)
        self.qcTable.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            self._set_qc_table_item(row_index, 0, row["trajectory"])
            self._set_qc_table_item(row_index, 1, f"{row['entry_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 2, f"{row['target_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 3, f"{row['mean_contact_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 4, f"{row['max_contact_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 5, f"{row['rms_contact_radial_mm']:.2f}")
            self._set_qc_table_item(row_index, 6, f"{row['angle_deg']:.2f}")
            self._set_qc_table_item(row_index, 7, str(row["matched_contacts"]))

    def _electrode_length_mm(self, model_id):
        """Return electrode exploration length in millimeters for model ID."""
        model = self.modelsById.get(model_id, {})
        return electrode_length_mm(model)

    def _update_electrode_length_cell(self, row):
        """Refresh per-row electrode length after model selection change."""
        model_combo = self.contactTable.cellWidget(row, 2)
        model_id = self._widget_text(model_combo).strip()
        length_text = ""
        if model_id:
            length_text = f"{self._electrode_length_mm(model_id):.2f}"
        self._set_readonly_text_item(row, 3, length_text)

    def _bind_model_length_update(self, model_combo, row):
        """Bind combo change signal to row-specific electrode length update."""
        if hasattr(model_combo, "currentTextChanged"):
            model_combo.currentTextChanged.connect(
                lambda _text, row_index=row: self._update_electrode_length_cell(row_index)
            )
        else:
            model_combo.currentIndexChanged.connect(
                lambda _idx, row_index=row: self._update_electrode_length_cell(row_index)
            )

    def _trajectory_length_mm(self, trajectory):
        """Return Euclidean distance between entry/start and target/end in millimeters."""
        return trajectory_length_mm(trajectory)

    def _suggest_model_id_for_trajectory(self, trajectory):
        """Select model with closest length, constrained to +/- 5 mm from trajectory."""
        return suggest_model_id_for_trajectory(
            trajectory=trajectory,
            models_by_id=self.modelsById,
            model_ids=self.modelIds,
            tolerance_mm=5.0,
        )

    def _populate_contact_table(self, trajectories):
        """Fill assignment table with one row per trajectory."""
        self.contactTable.setRowCount(0)
        auto_assigned = 0
        for row, traj in enumerate(trajectories):
            self.contactTable.insertRow(row)
            self._set_readonly_text_item(row, 0, traj["name"])
            traj_len = self._trajectory_length_mm(traj)
            self._set_readonly_text_item(row, 1, f"{traj_len:.2f}")
            model_combo = self._build_model_combo()
            self._bind_model_length_update(model_combo, row)
            suggested_model = self._suggest_model_id_for_trajectory(traj)
            if suggested_model:
                idx = model_combo.findText(suggested_model)
                if idx >= 0:
                    model_combo.setCurrentIndex(idx)
                    auto_assigned += 1
            self.contactTable.setCellWidget(row, 2, model_combo)
            self._set_readonly_text_item(row, 3, "")
            self._update_electrode_length_cell(row)
            self.contactTable.setCellWidget(row, 4, self._build_tip_at_combo())
            self.contactTable.setCellWidget(row, 5, self._build_tip_shift_spinbox())

        enabled = bool(trajectories) and bool(self.modelsById)
        self.generateContactsButton.setEnabled(enabled)
        self.updateContactsButton.setEnabled(enabled)
        self.applyModelAllButton.setEnabled(enabled)
        self.exportBundleButton.setEnabled(bool(self.loadedVolumeNodeIDs))
        if trajectories:
            self.log(f"[contacts] ready for {len(trajectories)} trajectories")
            self.log(f"[contacts] auto-assigned models for {auto_assigned}/{len(trajectories)}")
        else:
            self.log("[contacts] no trajectories found in case")

    def _populate_trajectory_selector(self, trajectories):
        """Populate trajectory selector used for slice alignment."""
        self.trajectorySelector.clear()
        for traj in trajectories:
            self.trajectorySelector.addItem(traj["name"])
        self.alignSliceButton.setEnabled(bool(trajectories))

    def _populate_autofit_trajectory_selector(self, trajectories):
        """Populate selected-trajectory dropdown in auto-fit section."""
        self.autoFitTrajectorySelector.clear()
        for traj in trajectories:
            self.autoFitTrajectorySelector.addItem(traj["name"])

    def _trajectory_by_name(self, name):
        """Find trajectory dictionary by name."""
        for traj in self.loadedTrajectories:
            if traj["name"] == name:
                return traj
        return None

    def _find_line_markup_node(self, name):
        """Return trajectory line markup node by exact name."""
        return self.logic.trajectory_scene.find_line_markup_node(name)

    def _trajectory_from_line_node(self, name, node):
        """Extract trajectory start/end from a line node as ROSA/LPS points."""
        return self.logic.trajectory_scene.trajectory_from_line_node(name, node)

    def _build_trajectory_map_with_scene_overrides(self):
        """Return trajectory map using current scene markups when available."""
        return self.logic.trajectory_scene.build_trajectory_map_with_scene_overrides(
            self.loadedTrajectories
        )

    def _refresh_trajectory_lengths_from_scene(self):
        """Refresh trajectory length column from currently edited line markups."""
        traj_map = self._build_trajectory_map_with_scene_overrides()
        for row in range(self.contactTable.rowCount):
            name_item = self.contactTable.item(row, 0)
            if not name_item:
                continue
            name = name_item.text()
            traj = traj_map.get(name)
            if traj is None:
                continue
            self._set_readonly_text_item(row, 1, f"{self._trajectory_length_mm(traj):.2f}")

    def _run_contact_generation(self, log_context="generate", allow_last_assignments=False):
        """Compute contacts from table assignments and current trajectory markup positions."""
        assignments = self._collect_assignments()
        if not assignments["assignments"]:
            if allow_last_assignments and self.lastAssignments.get("assignments"):
                assignments = self.lastAssignments
                self.log(f"[contacts:{log_context}] using last non-empty electrode assignments")
            else:
                raise ValueError("Select at least one electrode model in the assignment table.")
        else:
            self.lastAssignments = assignments

        traj_map = self._build_trajectory_map_with_scene_overrides()
        ordered_names = []
        for row in range(self.contactTable.rowCount):
            item = self.contactTable.item(row, 0)
            if item:
                ordered_names.append(item.text())
        self.loadedTrajectories = [traj_map[name] for name in ordered_names if name in traj_map]
        self._refresh_trajectory_lengths_from_scene()

        try:
            contacts = generate_contacts(self.loadedTrajectories, self.modelsById, assignments)
        except Exception as exc:
            raise ValueError(str(exc))

        node_prefix = self.contactsNodeNameEdit.text.strip() or "ROSA_Contacts"
        nodes = self.logic.create_contacts_fiducials_nodes_by_trajectory(
            contacts,
            node_prefix=node_prefix,
        )

        model_nodes = {}
        if self.createModelsCheck.checked:
            model_nodes = self.logic.create_electrode_models_by_trajectory(
                contacts=contacts,
                trajectories_by_name=traj_map,
                models_by_id=self.modelsById,
                node_prefix=node_prefix,
            )

        self.lastGeneratedContacts = contacts
        self.lastAtlasAssignmentRows = []
        assignment_rows = list(assignments.get("assignments", []))
        for row in assignment_rows:
            traj_name = row.get("trajectory", "")
            traj = traj_map.get(traj_name)
            row["trajectory_length_mm"] = 0.0 if traj is None else self._trajectory_length_mm(traj)
            row["electrode_length_mm"] = self._electrode_length_mm(row.get("model_id", ""))
            row["source"] = "contacts"
        self.exportBundleButton.setEnabled(True)
        self.log(
            f"[contacts:{log_context}] updated {len(contacts)} points across {len(nodes)} electrode nodes"
        )
        if model_nodes:
            self.log(f"[models:{log_context}] updated {len(model_nodes)} electrode model pairs")
        self._refresh_qc_metrics()
        self.logic.publish_contacts_outputs(
            contact_nodes_by_traj=nodes,
            model_nodes_by_traj=model_nodes,
            assignment_rows=assignment_rows,
            qc_rows=self.lastQCMetricsRows,
            workflow_node=getattr(self, "workflowNode", None),
        )
        if self.lastQCMetricsRows:
            self.log(f"[qc:{log_context}] computed metrics for {len(self.lastQCMetricsRows)} trajectories")

    def _assignment_map(self):
        """Return mapping `trajectory_name -> assignment_row` for selected models."""
        amap = {}
        for row in self._collect_assignments().get("assignments", []):
            amap[row["trajectory"]] = row
        return amap

    def _set_preview_line(self, trajectory_name, start_lps, end_lps):
        """Create/update a preview line showing the fitted trajectory."""
        self.logic.trajectory_scene.set_preview_line(
            trajectory_name=trajectory_name,
            start_lps=start_lps,
            end_lps=end_lps,
            node_prefix="AutoFit_",
        )

    def _remove_autofit_preview_lines(self, trajectory_names=None):
        """Remove AutoFit preview line nodes."""
        self.logic.trajectory_scene.remove_preview_lines(
            trajectory_names=trajectory_names,
            node_prefix="AutoFit_",
        )

    def _fit_trajectories(self, names):
        """Run auto-fit for selected trajectory names and store successful results."""
        if self._autofit_candidate_count() == 0:
            raise ValueError("No CT candidates available. Run 'Detect Candidates' first.")
        try:
            import importlib
            import rosa_core.contact_fit as contact_fit

            contact_fit = importlib.reload(contact_fit)
            fit_electrode_axis_and_tip = contact_fit.fit_electrode_axis_and_tip
        except Exception as exc:
            raise ValueError(f"Auto-fit dependency unavailable: {exc}")

        traj_map = self._build_trajectory_map_with_scene_overrides()
        assign_map = self._assignment_map()
        roi_radius = self._widget_value(self.autoRoiRadiusSpin)
        max_angle = self._widget_value(self.autoMaxAngleSpin)
        max_shift = self._widget_value(self.autoMaxDepthShiftSpin)

        success_count = 0
        for name in names:
            traj = traj_map.get(name)
            if traj is None:
                self.log(f"[autofit] {name}: skipped (trajectory not found)")
                continue
            assignment = assign_map.get(name)
            if assignment is None:
                self.log(f"[autofit] {name}: skipped (no electrode model assigned)")
                continue
            model = self.modelsById.get(assignment["model_id"])
            if not model:
                self.log(f"[autofit] {name}: skipped (unknown model {assignment['model_id']})")
                continue

            fit = fit_electrode_axis_and_tip(
                candidate_points_lps=self.autoFitCandidatesLPS,
                planned_entry_lps=traj["start"],
                planned_target_lps=traj["end"],
                contact_offsets_mm=model["contact_center_offsets_from_tip_mm"],
                tip_at=assignment.get("tip_at", "target"),
                roi_radius_mm=roi_radius,
                max_angle_deg=max_angle,
                max_depth_shift_mm=max_shift,
            )
            if fit.get("success"):
                self.autoFitResults[name] = fit
                success_count += 1
                self._set_preview_line(name, fit["entry_lps"], fit["target_lps"])
                self.log(
                    "[autofit] {name}: angle={a:.2f} deg depth={s:.2f} mm lateral={l:.2f} mm "
                    "residual={r:.2f} mm points={n} slabs={sl}/{si}".format(
                        name=name,
                        a=float(fit.get("angle_deg", 0.0)),
                        s=float(fit.get("tip_shift_mm", 0.0)),
                        l=float(fit.get("lateral_shift_mm", 0.0)),
                        r=float(fit.get("residual_mm", 0.0)),
                        n=int(fit.get("points_in_roi", 0)),
                        sl=int(fit.get("slab_centroids", 0)),
                        si=int(fit.get("slab_inliers", 0)),
                    )
                )
            else:
                self.log(f"[autofit] {name}: failed ({fit.get('reason', 'unknown')})")

        self.applyFitButton.setEnabled(bool(self.autoFitResults))
        self.log(f"[autofit] fitted {success_count}/{len(names)} trajectories")

    def onDetectCandidatesClicked(self):
        """Extract postop CT high-intensity candidate points for auto-fit."""
        volume_node = self.postopCTSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Select a postop CT volume.")
            return
        threshold = self._widget_value(self.autoThresholdSpin)
        try:
            points_lps = self.logic.extract_threshold_candidates_lps(volume_node, threshold=threshold)
        except Exception as exc:
            self.log(f"[autofit] detect error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self._remove_autofit_preview_lines()
        self.autoFitCandidatesLPS = points_lps
        self.autoFitResults = {}
        self.autoFitCandidateVolumeNodeID = volume_node.GetID()
        if hasattr(self.logic, "workflow_publish"):
            self.logic.workflow_publish.register_volume(
                volume_node=volume_node,
                source_type="import",
                source_path="",
                space_name="ROSA_BASE",
                role="AdditionalCTVolumes",
                is_default_postop=True,
                workflow_node=getattr(self, "workflowNode", None),
            )
        count = int(points_lps.shape[0]) if hasattr(points_lps, "shape") else len(points_lps)
        self._set_autofit_buttons_enabled(count > 0)
        self.log(f"[autofit] detected {count} candidate points at threshold {threshold:.1f}")

        try:
            self.logic.show_volume_in_all_slice_views(volume_node)
            self.logic.apply_ct_window_from_threshold(volume_node, threshold=threshold)
        except Exception as exc:
            self.log(f"[autofit] display warning: {exc}")

    def onResetCTWindowClicked(self):
        """Reset selected postop CT window/level to Slicer auto preset."""
        volume_node = self.postopCTSelector.currentNode()
        if volume_node is None:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Select a postop CT volume.")
            return
        try:
            self.logic.reset_ct_window(volume_node)
            self.log("[autofit] CT window reset to auto")
        except Exception as exc:
            self.log(f"[autofit] CT window reset warning: {exc}")

    def onFitSelectedClicked(self):
        """Fit only the currently selected trajectory using detected candidates."""
        name = self._widget_text(self.autoFitTrajectorySelector).strip()
        if not name:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Select a trajectory to fit.")
            return
        try:
            self._fit_trajectories([name])
        except Exception as exc:
            self.log(f"[autofit] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))

    def onFitAllClicked(self):
        """Fit all trajectories that have an assigned electrode model."""
        names = [row.get("trajectory") for row in self._collect_assignments().get("assignments", [])]
        names = [n for n in names if n]
        if not names:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Assign at least one electrode model before fitting.",
            )
            return
        try:
            self._fit_trajectories(names)
        except Exception as exc:
            self.log(f"[autofit] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))

    def onApplyFitClicked(self):
        """Apply successful auto-fit results to trajectory line markups and refresh contacts."""
        if not self.autoFitResults:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "No successful fit results to apply.",
            )
            return

        applied = 0
        for name, fit in self.autoFitResults.items():
            if not fit.get("success"):
                continue
            node = self._find_line_markup_node(name)
            if node is None or node.GetNumberOfControlPoints() < 2:
                continue
            start_ras = lps_to_ras_point(fit["entry_lps"])
            end_ras = lps_to_ras_point(fit["target_lps"])
            node.SetNthControlPointPositionWorld(0, start_ras[0], start_ras[1], start_ras[2])
            node.SetNthControlPointPositionWorld(1, end_ras[0], end_ras[1], end_ras[2])
            applied += 1

        self.log(f"[autofit] applied fitted trajectories: {applied}")
        self._remove_autofit_preview_lines()
        self.autoFitResults = {}
        self.applyFitButton.setEnabled(False)
        try:
            self._run_contact_generation(log_context="autofit", allow_last_assignments=True)
        except Exception as exc:
            self.log(f"[autofit] contact update error: {exc}")
            qt.QMessageBox.critical(
                slicer.util.mainWindow(),
                "ROSA Helper",
                f"Applied fit to trajectories, but contact update failed:\n{exc}",
            )

    def onApplyModelAllClicked(self):
        """Apply selected default electrode model to all rows."""
        model_id = self._widget_text(self.defaultModelCombo).strip()
        if not model_id:
            return
        for row in range(self.contactTable.rowCount):
            combo = self.contactTable.cellWidget(row, 2)
            if combo:
                idx = combo.findText(model_id)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _collect_assignments(self):
        """Collect non-empty assignments from table rows."""
        rows = []
        for row in range(self.contactTable.rowCount):
            traj_item = self.contactTable.item(row, 0)
            if not traj_item:
                continue
            model_combo = self.contactTable.cellWidget(row, 2)
            tip_at_combo = self.contactTable.cellWidget(row, 4)
            tip_shift_spin = self.contactTable.cellWidget(row, 5)
            model_id = self._widget_text(model_combo).strip()
            if not model_id:
                continue
            rows.append(
                {
                    "trajectory": traj_item.text(),
                    "model_id": model_id,
                    "tip_at": self._widget_text(tip_at_combo) or "target",
                    "tip_shift_mm": self._widget_value(tip_shift_spin),
                    "xyz_offset_mm": [0.0, 0.0, 0.0],
                }
            )
        return {"schema_version": "1.0", "assignments": rows}

    def onGenerateContactsClicked(self):
        """Generate contact fiducials from trajectory assignments."""
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Load a ROSA case first.")
            return

        try:
            self._run_contact_generation(log_context="generate")
        except Exception as exc:
            self.log(f"[contacts] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

    def onUpdateContactsClicked(self):
        """Recalculate contact and model nodes from edited trajectory markups."""
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Load a ROSA case first.")
            return
        if not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Generate contacts once first, then use update for edited trajectories.",
            )
            return

        try:
            self._run_contact_generation(log_context="update")
        except Exception as exc:
            self.log(f"[contacts:update] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

    def onExportAlignedBundleClicked(self):
        """Export aligned volumes as NIfTI and contact coordinates in same frame."""
        selected_profile_name = self._widget_text(self.exportProfileCombo).strip() or "full_bundle"
        profile, resolved_profile_name = get_export_profile(selected_profile_name)
        if not any(bool(v) for v in profile.values()):
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "Selected export profile is empty.")
            return
        if profile.get("include_volumes", False) and not self.loadedVolumeNodeIDs:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "No loaded volumes available to export.",
            )
            return
        needs_contacts = bool(
            profile.get("include_contacts", False)
            or profile.get("include_qc", False)
            or profile.get("include_atlas", False)
        )
        if needs_contacts and not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "This export profile requires generated contacts.",
            )
            return
        needs_trajectories = bool(profile.get("include_planned", False) or profile.get("include_final", False))
        if needs_trajectories and not self.loadedTrajectories:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "This export profile requires loaded trajectories.",
            )
            return

        out_dir = self.bundleExportDirEdit.text.strip()
        if not out_dir:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select an output folder for aligned export.",
            )
            return

        node_prefix = self.contactsNodeNameEdit.text.strip() or "ROSA_Contacts"
        self._refresh_qc_metrics()
        planned_map = self._collect_planned_trajectory_map()
        final_map = self._build_trajectory_map_with_scene_overrides()
        try:
            result = self.logic.export_aligned_bundle(
                volume_node_ids=self.loadedVolumeNodeIDs,
                contacts=self.lastGeneratedContacts,
                out_dir=out_dir,
                node_prefix=node_prefix,
                planned_trajectories=planned_map,
                final_trajectories=final_map,
                qc_rows=self.lastQCMetricsRows,
                atlas_rows=self.lastAtlasAssignmentRows,
                output_frame_node=self.exportFrameSelector.currentNode(),
                export_profile=resolved_profile_name,
            )
        except Exception as exc:
            self.log(f"[bundle] export warning: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.log(
            f"[bundle] exported {result['volume_count']} NIfTI volumes "
            f"and coordinate/QC files to {result['out_dir']}"
        )
        self.log(f"[bundle] manifest: {result.get('manifest_path', '')}")
        if self.lastAtlasAssignmentRows:
            self.log(f"[bundle] atlas assignment CSV: {result.get('atlas_assignment_path', '')}")

    def onAlignSliceClicked(self):
        """Align selected slice view to selected trajectory."""
        traj_name = self._widget_text(self.trajectorySelector).strip()
        if not traj_name:
            qt.QMessageBox.warning(slicer.util.mainWindow(), "ROSA Helper", "No trajectory selected.")
            return

        trajectory = self._trajectory_by_name(traj_name)
        if trajectory is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                f"Trajectory '{traj_name}' not found in loaded case.",
            )
            return

        start_ras = lps_to_ras_point(trajectory["start"])
        end_ras = lps_to_ras_point(trajectory["end"])
        slice_view = self._widget_text(self.sliceViewSelector) or "Red"
        mode = self._widget_text(self.sliceModeSelector) or "long"

        try:
            self.logic.align_slice_to_trajectory(start_ras, end_ras, slice_view=slice_view, mode=mode)
        except Exception as exc:
            self.log(f"[slice] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.log(f"[slice] aligned {slice_view} view to {traj_name} ({mode})")

    def onShowPlannedToggled(self, checked):
        """Toggle visibility of stored planned trajectory lines."""
        self.logic.set_planned_trajectory_visibility(bool(checked))

    def _preselect_freesurfer_reference_volume(self):
        """Default FS fixed volume selector to loaded ROSA reference volume."""
        if self.referenceVolumeName and self.referenceVolumeName in self.loadedVolumeNodeIDs:
            node_id = self.loadedVolumeNodeIDs[self.referenceVolumeName]
            self.fsFixedSelector.setCurrentNodeID(node_id)
            return
        if self.loadedVolumeNodeIDs:
            first_name = sorted(self.loadedVolumeNodeIDs.keys())[0]
            self.fsFixedSelector.setCurrentNodeID(self.loadedVolumeNodeIDs[first_name])

    def _refresh_fs_parcellation_combo(self):
        """Refresh FS parcellation dropdown from selected subject directory."""
        subject_dir = self.fsSubjectDirSelector.currentPath
        current = self._widget_text(self.fsParcellationCombo).strip()
        self.fsParcellationCombo.clear()
        self.fsParcellationCombo.addItem("all available")
        if not subject_dir:
            return
        try:
            listing = self.logic.list_freesurfer_parcellation_candidates(subject_dir)
        except Exception as exc:
            self.log(f"[fs] parcellation scan error: {exc}")
            return
        available = listing.get("available", [])
        for item in available:
            name = item.get("name")
            if name:
                self.fsParcellationCombo.addItem(name)
        if current and self.fsParcellationCombo.findText(current) >= 0:
            self.fsParcellationCombo.setCurrentText(current)

    def _refresh_atlas_source_options(self):
        """Refresh atlas source dropdowns from currently loaded FS/THOMAS nodes."""
        fs_current = self._widget_text(self.atlasFSVolumeCombo).strip() if hasattr(self, "atlasFSVolumeCombo") else ""
        wm_current = self._widget_text(self.atlasWMVolumeCombo).strip() if hasattr(self, "atlasWMVolumeCombo") else ""
        th_current = self._widget_text(self.atlasThomasCombo).strip() if hasattr(self, "atlasThomasCombo") else ""

        if hasattr(self, "atlasFSVolumeCombo"):
            self.atlasFSVolumeCombo.clear()
            self.atlasFSVolumeCombo.addItem("(none)")
        if hasattr(self, "atlasWMVolumeCombo"):
            self.atlasWMVolumeCombo.clear()
            self.atlasWMVolumeCombo.addItem("(none)")
        if hasattr(self, "atlasThomasCombo"):
            self.atlasThomasCombo.clear()
            self.atlasThomasCombo.addItem("(none)")

        fs_nodes = []
        for node_id in getattr(self, "fsParcellationVolumeNodeIDs", []) or []:
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                fs_nodes.append(node)
        if not fs_nodes:
            for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                name = node.GetName() or ""
                if name.startswith("FSVOL_"):
                    fs_nodes.append(node)

        for node in fs_nodes:
            name = node.GetName() or ""
            self.atlasFSVolumeCombo.addItem(name, node.GetID())
            lower = name.lower()
            if "wmparc" in lower:
                self.atlasWMVolumeCombo.addItem(name, node.GetID())

        thomas_nodes = self._loaded_thomas_segmentation_nodes()
        if thomas_nodes:
            self.atlasThomasCombo.addItem("All loaded THOMAS", "__ALL__")
            for node in thomas_nodes:
                self.atlasThomasCombo.addItem(node.GetName() or "THOMAS", node.GetID())

        # restore selection where possible
        if fs_current and self.atlasFSVolumeCombo.findText(fs_current) >= 0:
            self.atlasFSVolumeCombo.setCurrentText(fs_current)
        if wm_current and self.atlasWMVolumeCombo.findText(wm_current) >= 0:
            self.atlasWMVolumeCombo.setCurrentText(wm_current)
        if th_current and self.atlasThomasCombo.findText(th_current) >= 0:
            self.atlasThomasCombo.setCurrentText(th_current)

    def _preselect_thomas_reference_volume(self):
        """Default THOMAS fixed volume selector to loaded ROSA reference volume."""
        if self.referenceVolumeName and self.referenceVolumeName in self.loadedVolumeNodeIDs:
            node_id = self.loadedVolumeNodeIDs[self.referenceVolumeName]
            self.thomasFixedSelector.setCurrentNodeID(node_id)
            return
        if self.loadedVolumeNodeIDs:
            first_name = sorted(self.loadedVolumeNodeIDs.keys())[0]
            self.thomasFixedSelector.setCurrentNodeID(self.loadedVolumeNodeIDs[first_name])

    def _preselect_thomas_burn_volume(self):
        """Default burn input selector to ROSA reference volume when available."""
        if self.referenceVolumeName and self.referenceVolumeName in self.loadedVolumeNodeIDs:
            node_id = self.loadedVolumeNodeIDs[self.referenceVolumeName]
            self.thomasBurnInputSelector.setCurrentNodeID(node_id)
            return
        if self.loadedVolumeNodeIDs:
            first_name = sorted(self.loadedVolumeNodeIDs.keys())[0]
            self.thomasBurnInputSelector.setCurrentNodeID(self.loadedVolumeNodeIDs[first_name])

    def _loaded_thomas_segmentation_nodes(self):
        """Return currently tracked THOMAS segmentation nodes present in scene."""
        nodes = []
        for node_id in getattr(self, "thomasSegmentationNodeIDs", []) or []:
            node = slicer.mrmlScene.GetNodeByID(node_id)
            if node is not None:
                nodes.append(node)
        if nodes:
            return nodes
        # Fallback by name for scenes loaded before tracking IDs.
        for side in ("Left", "Right"):
            node = self.logic._find_node_by_name(f"THOMAS_{side}_Structures", "vtkMRMLSegmentationNode")
            if node is not None:
                nodes.append(node)
        return nodes

    def _refresh_thomas_nucleus_combo(self):
        """Populate nucleus picker from currently loaded THOMAS segments."""
        nuclei = self.logic.collect_thomas_nuclei(self._loaded_thomas_segmentation_nodes())
        current = self._widget_text(self.thomasBurnNucleusCombo).strip()
        self.thomasBurnNucleusCombo.clear()
        if nuclei:
            for nucleus in nuclei:
                self.thomasBurnNucleusCombo.addItem(nucleus)
            if current and current in nuclei:
                self.thomasBurnNucleusCombo.setCurrentText(current)
            else:
                default = "CM" if "CM" in nuclei else nuclei[0]
                self.thomasBurnNucleusCombo.setCurrentText(default)
            return
        self.thomasBurnNucleusCombo.addItem("CM")
        self.thomasBurnNucleusCombo.setCurrentText(current or "CM")

    def _get_or_create_fs_transform_node(self):
        """Return existing or newly created linear transform for FS->ROSA mapping."""
        name = self.fsTransformNameEdit.text.strip() or "FS_to_ROSA"
        node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        return node

    def _get_or_create_thomas_transform_node(self):
        """Return existing or newly created linear transform for THOMAS->ROSA mapping."""
        name = self.thomasTransformNameEdit.text.strip() or "THOMAS_to_ROSA"
        node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        return node

    def _get_or_create_thomas_dicom_transform_node(self):
        """Return linear transform node for imported DICOM MRI -> ROSA base."""
        base = self.thomasTransformNameEdit.text.strip() or "THOMAS_to_ROSA"
        name = f"{base}_DICOM_to_ROSA"
        node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        return node

    def _get_or_create_thomas_burn_transform_node(self):
        """Return dedicated linear transform node for THOMAS->burn-input mapping."""
        base = self.thomasTransformNameEdit.text.strip() or "THOMAS_to_ROSA"
        name = f"{base}_for_burn"
        node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
        if node is None:
            node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLinearTransformNode", name)
        return node

    def _selected_fs_annotation_name(self):
        """Return selected FreeSurfer annotation name or None when disabled."""
        if not bool(self.fsUseAnnotationCheck.checked):
            return None
        value = (self._widget_text(self.fsAnnotationCombo) or "").strip()
        if not value:
            return None
        if value.lower() == "custom":
            custom = self.fsCustomAnnotationEdit.text.strip()
            return custom or None
        return value

    def onRegisterFSMRIToRosaClicked(self):
        """Run linear registration from FreeSurfer MRI to selected ROSA base volume."""
        fixed_node = self.fsFixedSelector.currentNode()
        moving_node = self.fsMovingSelector.currentNode()
        if fixed_node is None or moving_node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select both ROSA base volume and FreeSurfer MRI volume.",
            )
            return
        if fixed_node.GetID() == moving_node.GetID():
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Fixed and moving volumes must be different nodes.",
            )
            return

        transform_node = self._get_or_create_fs_transform_node()
        init_mode = self._widget_text(self.fsInitModeCombo) or "useGeometryAlign"
        self.log(
            f"[fs] registration start: moving={moving_node.GetName()} -> fixed={fixed_node.GetName()} "
            f"(init={init_mode})"
        )
        try:
            self.logic.run_brainsfit_rigid_registration(
                fixed_volume_node=fixed_node,
                moving_volume_node=moving_node,
                output_transform_node=transform_node,
                initialize_mode=init_mode,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[fs] registration error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.fsToRosaTransformNodeID = transform_node.GetID()
        if hasattr(self.logic, "workflow_publish"):
            self.logic.workflow_publish.register_transform(
                transform_node=transform_node,
                from_space="FS_NATIVE",
                to_space="ROSA_BASE",
                transform_type="linear",
                status="active",
                role="FSToBaseTransform",
                workflow_node=getattr(self, "workflowNode", None),
            )
        self.log(f"[fs] registration done: transform={transform_node.GetName()}")

    def onRefreshFSParcellationClicked(self):
        """Refresh detected FS parcellation options from selected subject directory."""
        self._refresh_fs_parcellation_combo()
        self.log("[fs] parcellation options refreshed")

    def onRefreshAtlasSourcesClicked(self):
        """Refresh atlas-source dropdowns from current scene nodes."""
        self._refresh_atlas_source_options()
        self.log("[atlas] source options refreshed")

    def _atlas_selected_volume_node(self, combo):
        """Return selected scalar volume node from atlas combo."""
        if combo is None:
            return None
        text = (self._widget_text(combo) or "").strip()
        if text == "(none)" or not text:
            return None
        data = combo.currentData if hasattr(combo, "currentData") else None
        node_id = data() if callable(data) else data
        if node_id:
            node = slicer.mrmlScene.GetNodeByID(str(node_id))
            if node is not None:
                return node
        return self.logic._find_node_by_name(text, "vtkMRMLScalarVolumeNode")

    def _atlas_selected_thomas_nodes(self):
        """Return selected THOMAS segmentation nodes for atlas assignment."""
        combo = self.atlasThomasCombo
        if combo is None:
            return []
        text = (self._widget_text(combo) or "").strip()
        if text == "(none)" or not text:
            return []
        data = combo.currentData if hasattr(combo, "currentData") else None
        selected = data() if callable(data) else data
        if selected == "__ALL__":
            return self._loaded_thomas_segmentation_nodes()
        if selected:
            node = slicer.mrmlScene.GetNodeByID(str(selected))
            return [node] if node is not None else []
        node = self.logic._find_node_by_name(text, "vtkMRMLSegmentationNode")
        return [node] if node is not None else []

    def onAssignContactsToAtlasClicked(self):
        """Assign generated contacts to selected atlas sources and store rows for export."""
        if not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Generate contacts first.",
            )
            return

        fs_node = self._atlas_selected_volume_node(self.atlasFSVolumeCombo)
        wm_node = self._atlas_selected_volume_node(self.atlasWMVolumeCombo)
        th_nodes = self._atlas_selected_thomas_nodes()
        if fs_node is None and wm_node is None and not th_nodes:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select at least one atlas source (FreeSurfer, THOMAS, or WM).",
            )
            return

        reference_node = self.fsFixedSelector.currentNode()
        if reference_node is None and self.referenceVolumeName in self.loadedVolumeNodeIDs:
            reference_node = slicer.mrmlScene.GetNodeByID(self.loadedVolumeNodeIDs[self.referenceVolumeName])
        try:
            rows = self.logic.assign_contacts_to_atlases(
                contacts=self.lastGeneratedContacts,
                freesurfer_volume_node=fs_node,
                thomas_segmentation_nodes=th_nodes,
                wm_volume_node=wm_node,
                reference_volume_node=reference_node,
                prefer_thomas=bool(self.atlasPreferThomasCheck.checked),
            )
        except Exception as exc:
            self.log(f"[atlas] assignment error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.lastAtlasAssignmentRows = rows
        self.logic.publish_atlas_assignment_rows(
            atlas_rows=rows,
            workflow_node=getattr(self, "workflowNode", None),
        )
        self.log(f"[atlas] assigned {len(rows)} contacts")

    def onLoadFSParcellationsClicked(self):
        """Load selected FreeSurfer parcellation volume(s) from recon-all mri/."""
        subject_dir = self.fsSubjectDirSelector.currentPath
        if not subject_dir:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select FreeSurfer subject directory.",
            )
            return

        selected = (self._widget_text(self.fsParcellationCombo) or "").strip()
        selected_names = None if selected == "all available" else [selected]
        create_3d = bool(self.fsCreateParcellation3DCheck.checked)
        apply_lut = bool(self.fsApplyParcellationLUTCheck.checked)
        lut_path = self.fsLUTPathSelector.currentPath.strip() if self.fsLUTPathSelector.currentPath else ""

        apply_transform = bool(self.fsApplyTransformVolumesCheck.checked)
        transform_node = None
        if apply_transform:
            if self.fsToRosaTransformNodeID:
                transform_node = slicer.mrmlScene.GetNodeByID(self.fsToRosaTransformNodeID)
            if transform_node is None:
                name = self.fsTransformNameEdit.text.strip() or "FS_to_ROSA"
                transform_node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
            if transform_node is None:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "FS->ROSA transform is not available. Run registration first or disable transform application.",
                )
                return

        try:
            result = self.logic.load_freesurfer_parcellation_volumes(
                subject_dir=subject_dir,
                selected_names=selected_names,
                color_lut_path=lut_path or None,
                apply_color_table=apply_lut,
                create_3d_geometry=create_3d,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[fs] parcellation load error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        loaded_nodes = result.get("loaded_nodes", [])
        loaded_paths = result.get("loaded_paths", [])
        loaded_seg_nodes = result.get("loaded_segmentation_nodes", [])
        missing_paths = result.get("missing_paths", [])
        failed_paths = result.get("failed_paths", [])
        available = result.get("available_names", [])
        if apply_transform and transform_node is not None and (loaded_nodes or loaded_seg_nodes):
            targets = list(loaded_nodes) + list(loaded_seg_nodes)
            self.logic.apply_transform_to_nodes(
                nodes=targets,
                transform_node=transform_node,
                harden=bool(self.fsHardenVolumeCheck.checked),
            )
            self.log(
                f"[fs] applied transform {transform_node.GetName()} to "
                f"{len(loaded_nodes)} parcellation volumes and {len(loaded_seg_nodes)} 3D segmentations"
            )

        if available:
            self.log(f"[fs] available parcellations: {', '.join(available)}")
        if missing_paths:
            self.log(f"[fs] missing parcellation files: {len(missing_paths)}")
            for path in missing_paths:
                self.log(f"[fs] missing: {path}")
        if failed_paths:
            self.log(f"[fs] failed parcellation files: {len(failed_paths)}")
            for path in failed_paths:
                self.log(f"[fs] failed: {path}")

        self.fsParcellationVolumeNodeIDs = [node.GetID() for node in loaded_nodes]
        self.fsParcellationSegNodeIDs = [node.GetID() for node in loaded_seg_nodes]
        if hasattr(self.logic, "workflow_publish"):
            wf = getattr(self, "workflowNode", None)
            for node, path in zip(loaded_nodes, loaded_paths):
                name = (node.GetName() or "").lower()
                role = "WMParcellationVolumes" if "wmparc" in name else "FSParcellationVolumes"
                self.logic.workflow_publish.register_volume(
                    volume_node=node,
                    source_type="freesurfer",
                    source_path=path,
                    space_name="ROSA_BASE" if apply_transform else "FS_NATIVE",
                    role=role,
                    workflow_node=wf,
                )
        self.log(f"[fs] loaded {len(loaded_nodes)} parcellation volumes")
        if loaded_seg_nodes:
            self.log(f"[fs] created {len(loaded_seg_nodes)} parcellation 3D segmentation nodes")

        # Keep anatomy visualization stable: loading parcellation volumes can switch
        # slice background away from the ROSA base/fixed MRI.
        fixed_node = self.fsFixedSelector.currentNode()
        if fixed_node is not None:
            self.logic.show_volume_in_all_slice_views(fixed_node)
            self.log(f"[fs] restored background volume: {fixed_node.GetName()}")
        self._refresh_atlas_source_options()

    def onLoadFSSurfacesClicked(self):
        """Load FreeSurfer surfaces and optionally apply FS->ROSA transform."""
        subject_dir = self.fsSubjectDirSelector.currentPath
        if not subject_dir:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select FreeSurfer subject directory.",
            )
            return

        surface_set = self._widget_text(self.fsSurfaceSetCombo) or "pial"
        annotation_name = self._selected_fs_annotation_name()
        lut_path = self.fsLUTPathSelector.currentPath.strip() if self.fsLUTPathSelector.currentPath else ""
        apply_transform = bool(self.fsApplyTransformCheck.checked)
        transform_node = None
        if apply_transform:
            if self.fsToRosaTransformNodeID:
                transform_node = slicer.mrmlScene.GetNodeByID(self.fsToRosaTransformNodeID)
            if transform_node is None:
                name = self.fsTransformNameEdit.text.strip() or "FS_to_ROSA"
                transform_node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
            if transform_node is None:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "FS->ROSA transform is not available. Run registration first or disable transform application.",
                )
                return

        try:
            result = self.logic.load_freesurfer_surfaces(
                subject_dir=subject_dir,
                surface_set=surface_set,
                annotation_name=annotation_name,
                color_lut_path=lut_path or None,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[fs] surface load error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return
        loaded_nodes = result.get("loaded_nodes", [])
        missing_paths = result.get("missing_surface_paths", [])
        failed_paths = result.get("failed_surface_paths", [])
        missing_annotation_paths = result.get("missing_annotation_paths", [])
        annotated_count = int(result.get("annotated_nodes", 0))
        color_node_name = result.get("color_node_name", "")

        if apply_transform and transform_node is not None and loaded_nodes:
            self.logic.apply_transform_to_model_nodes(
                model_nodes=loaded_nodes,
                transform_node=transform_node,
                harden=bool(self.fsHardenSurfaceCheck.checked),
            )
            self.log(
                f"[fs] applied transform {transform_node.GetName()} to {len(loaded_nodes)} surfaces"
            )

        if missing_paths:
            self.log(f"[fs] missing surface files: {len(missing_paths)}")
            for path in missing_paths:
                self.log(f"[fs] missing: {path}")
        if failed_paths:
            self.log(f"[fs] failed to load surfaces: {len(failed_paths)}")
            for path in failed_paths:
                self.log(f"[fs] failed: {path}")
        if missing_annotation_paths:
            self.log(f"[fs] missing annotation files: {len(missing_annotation_paths)}")
            for path in missing_annotation_paths:
                self.log(f"[fs] missing annot: {path}")
        if annotation_name:
            self.log(
                f"[fs] annotation mode '{annotation_name}' applied to {annotated_count}/{len(loaded_nodes)} loaded surfaces"
            )
            if color_node_name:
                self.log(f"[fs] annotation color node: {color_node_name}")

        self.log(f"[fs] loaded {len(loaded_nodes)} surface models")

    def onRegisterThomasMRIToRosaClicked(self):
        """Run rigid registration from THOMAS MRI to selected ROSA base volume."""
        fixed_node = self.thomasFixedSelector.currentNode()
        moving_node = self.thomasMovingSelector.currentNode()
        if fixed_node is None or moving_node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select both ROSA base volume and THOMAS MRI volume.",
            )
            return
        if fixed_node.GetID() == moving_node.GetID():
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Fixed and moving volumes must be different nodes.",
            )
            return

        transform_node = self._get_or_create_thomas_transform_node()
        init_mode = self._widget_text(self.thomasInitModeCombo) or "useGeometryAlign"
        self.log(
            f"[thomas] registration start: moving={moving_node.GetName()} -> fixed={fixed_node.GetName()} "
            f"(init={init_mode})"
        )
        try:
            self.logic.run_brainsfit_rigid_registration(
                fixed_volume_node=fixed_node,
                moving_volume_node=moving_node,
                output_transform_node=transform_node,
                initialize_mode=init_mode,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[thomas] registration error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.thomasToRosaTransformNodeID = transform_node.GetID()
        if hasattr(self.logic, "workflow_publish"):
            self.logic.workflow_publish.register_transform(
                transform_node=transform_node,
                from_space="THOMAS_NATIVE",
                to_space="ROSA_BASE",
                transform_type="linear",
                status="active",
                role="THOMASToBaseTransform",
                workflow_node=getattr(self, "workflowNode", None),
            )
        self.log(f"[thomas] registration done: transform={transform_node.GetName()}")

    def onImportThomasDicomClicked(self):
        """Import a DICOM scalar volume from selected directory for burn workflow."""
        dicom_dir = self.thomasDicomDirSelector.currentPath
        if not dicom_dir:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select a DICOM series directory first.",
            )
            return
        try:
            volume_node = self.logic.load_dicom_scalar_volume_from_directory(
                dicom_dir=dicom_dir,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[thomas] DICOM import error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        fixed_node = self.thomasFixedSelector.currentNode()
        if fixed_node is not None and fixed_node.GetID() != volume_node.GetID():
            transform_node = self._get_or_create_thomas_dicom_transform_node()
            init_mode = self._widget_text(self.thomasInitModeCombo) or "useGeometryAlign"
            self.log(
                f"[thomas] DICOM registration start: moving={volume_node.GetName()} -> "
                f"fixed={fixed_node.GetName()} (init={init_mode})"
            )
            try:
                self.logic.run_brainsfit_rigid_registration(
                    fixed_volume_node=fixed_node,
                    moving_volume_node=volume_node,
                    output_transform_node=transform_node,
                    initialize_mode=init_mode,
                    logger=self.log,
                )
            except Exception as exc:
                self.log(f"[thomas] DICOM registration error: {exc}")
                qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
                return
            self.logic.apply_transform_to_nodes(
                nodes=[volume_node],
                transform_node=transform_node,
                harden=True,
            )
            self.thomasDicomToRosaTransformNodeID = transform_node.GetID()
            self.log(
                f"[thomas] DICOM registration done: {volume_node.GetName()} aligned to {fixed_node.GetName()}"
            )
        else:
            self.log("[thomas] DICOM registration skipped (no ROSA base selected or same node)")

        if volume_node is not None:
            self.thomasBurnInputSelector.setCurrentNode(volume_node)
            self.thomasImportedDicomNodeID = volume_node.GetID()
            if hasattr(self.logic, "workflow_publish"):
                self.logic.workflow_publish.register_volume(
                    volume_node=volume_node,
                    source_type="dicom",
                    source_path=dicom_dir,
                    space_name="ROSA_BASE",
                    role="AdditionalMRIVolumes",
                    workflow_node=getattr(self, "workflowNode", None),
                )
            self.log(f"[thomas] DICOM MRI ready: {volume_node.GetName()}")

    def onRefreshThomasNucleiClicked(self):
        """Refresh available nucleus list from loaded THOMAS segmentation nodes."""
        self._refresh_thomas_nucleus_combo()
        self.log("[thomas] nucleus list refreshed")

    def onLoadThomasMasksClicked(self):
        """Load THOMAS thalamus masks and optionally map them to ROSA space."""
        thomas_dir = self.thomasMaskDirSelector.currentPath
        if not thomas_dir:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select THOMAS output directory.",
            )
            return

        apply_transform = bool(self.thomasApplyTransformCheck.checked)
        transform_node = None
        if apply_transform:
            if self.thomasToRosaTransformNodeID:
                transform_node = slicer.mrmlScene.GetNodeByID(self.thomasToRosaTransformNodeID)
            if transform_node is None:
                name = self.thomasTransformNameEdit.text.strip() or "THOMAS_to_ROSA"
                transform_node = self.logic._find_node_by_name(name, "vtkMRMLLinearTransformNode")
            if transform_node is None and self.fsToRosaTransformNodeID:
                transform_node = slicer.mrmlScene.GetNodeByID(self.fsToRosaTransformNodeID)
            if transform_node is None:
                fs_name = self.fsTransformNameEdit.text.strip() or "FS_to_ROSA"
                transform_node = self.logic._find_node_by_name(fs_name, "vtkMRMLLinearTransformNode")
            if transform_node is None:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "THOMAS->ROSA transform is not available. Run THOMAS registration first or disable transform application.",
                )
                return

        try:
            result = self.logic.load_thomas_thalamus_masks(
                thomas_dir=thomas_dir,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[thomas] load error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        loaded_nodes = result.get("loaded_nodes", [])
        loaded_paths = result.get("loaded_mask_paths", [])
        failed_paths = result.get("failed_mask_paths", [])
        missing_paths = result.get("missing_mask_paths", [])
        skipped_paths = result.get("skipped_mask_paths", [])
        if apply_transform and transform_node is not None and loaded_nodes:
            self.logic.apply_transform_to_nodes(
                nodes=loaded_nodes,
                transform_node=transform_node,
                harden=bool(self.thomasHardenCheck.checked),
            )
            self.log(
                f"[thomas] applied transform {transform_node.GetName()} to {len(loaded_nodes)} segmentations"
            )

        if missing_paths:
            self.log(f"[thomas] missing mask files: {len(missing_paths)}")
            for path in missing_paths:
                self.log(f"[thomas] missing: {path}")
        if failed_paths:
            self.log(f"[thomas] failed to load masks: {len(failed_paths)}")
            for path in failed_paths:
                self.log(f"[thomas] failed: {path}")
        if skipped_paths:
            self.log(f"[thomas] skipped masks: {len(skipped_paths)} (EXTRAS/cropped/resampled/full)")
        self.log(f"[thomas] loaded {len(loaded_nodes)} THOMAS segmentation nodes")
        for path in loaded_paths:
            self.log(f"[thomas] loaded: {path}")
        self.thomasSegmentationNodeIDs = [node.GetID() for node in loaded_nodes]
        if hasattr(self.logic, "workflow_publish"):
            self.logic.workflow_publish.publish_nodes(
                role="THOMASSegmentations",
                nodes=loaded_nodes,
                source="thomas",
                space_name="ROSA_BASE" if apply_transform else "THOMAS_NATIVE",
                workflow_node=getattr(self, "workflowNode", None),
            )
        self._refresh_thomas_nucleus_combo()
        self._refresh_atlas_source_options()

    def onBurnThomasNucleusClicked(self):
        """Run optional registration and burn selected THOMAS nucleus into MRI volume."""
        burn_input_node = self.thomasBurnInputSelector.currentNode()
        moving_node = self.thomasMovingSelector.currentNode()
        if burn_input_node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select Burn input MRI volume.",
            )
            return None
        if moving_node is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select THOMAS MRI volume (moving image).",
            )
            return None

        seg_nodes = self._loaded_thomas_segmentation_nodes()
        burn_seg_nodes = list(seg_nodes)
        temp_seg_nodes = []
        temp_volume_nodes = []
        burn_moving_node = moving_node
        burn_input_for_burn = burn_input_node

        # Burn workflow may be run in scenes where users already parented volumes
        # under transforms. Use temporary hardened copies so original nodes are untouched.
        volumes_logic = slicer.modules.volumes.logic()
        transform_logic = slicer.vtkSlicerTransformLogic()
        if volumes_logic is None:
                qt.QMessageBox.critical(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "Volumes logic is unavailable.",
                )
                return None
        if moving_node.GetTransformNodeID():
            burn_moving_node = volumes_logic.CloneVolume(
                slicer.mrmlScene,
                moving_node,
                "__THOMAS_BURN_MOVING",
            )
            temp_volume_nodes.append(burn_moving_node)
            transform_logic.hardenTransform(burn_moving_node)
            self.log(f"[thomas] using hardened temp moving volume: {burn_moving_node.GetName()}")
        if burn_input_node.GetTransformNodeID():
            burn_input_for_burn = volumes_logic.CloneVolume(
                slicer.mrmlScene,
                burn_input_node,
                "__THOMAS_BURN_FIXED",
            )
            temp_volume_nodes.append(burn_input_for_burn)
            transform_logic.hardenTransform(burn_input_for_burn)
            self.log(f"[thomas] using hardened temp fixed volume: {burn_input_for_burn.GetName()}")

        # Optional re-registration directly to burn input volume for a one-click workflow.
        if bool(self.thomasBurnAutoRegisterCheck.checked):
            thomas_dir = self.thomasMaskDirSelector.currentPath
            if not thomas_dir:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "Set THOMAS output dir to run auto-register burn workflow.",
                )
                return None

            # Use temporary raw THOMAS segmentations for burn registration only.
            # This prevents visible scene THOMAS nodes from being moved/overwritten.
            for node in slicer.util.getNodesByClass("vtkMRMLSegmentationNode"):
                name = node.GetName() or ""
                if name.startswith("__THOMAS_BURN_"):
                    slicer.mrmlScene.RemoveNode(node)
            try:
                refreshed = self.logic.load_thomas_thalamus_masks(
                    thomas_dir=thomas_dir,
                    logger=self.log,
                    replace_existing=False,
                    node_name_prefix="__THOMAS_BURN_",
                )
                burn_seg_nodes = refreshed.get("loaded_nodes", [])
                temp_seg_nodes = list(burn_seg_nodes)
                self.log(
                    f"[thomas] burn workflow loaded temporary raw masks: "
                    f"{len(burn_seg_nodes)} segmentation nodes"
                )
            except Exception as exc:
                self.log(f"[thomas] burn mask load failed: {exc}")
                qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
                return None
            if not burn_seg_nodes:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "ROSA Helper",
                    "No THOMAS masks available for burn workflow.",
                )
                return None

            transform_node = self._get_or_create_thomas_burn_transform_node()
            init_mode = self._widget_text(self.thomasInitModeCombo) or "useGeometryAlign"
            self.log(
                f"[thomas] burn registration start: moving={burn_moving_node.GetName()} -> "
                f"fixed={burn_input_for_burn.GetName()} (init={init_mode})"
            )
            try:
                self.logic.run_brainsfit_rigid_registration(
                    fixed_volume_node=burn_input_for_burn,
                    moving_volume_node=burn_moving_node,
                    output_transform_node=transform_node,
                    initialize_mode=init_mode,
                    logger=self.log,
                )
            except Exception as exc:
                self.log(f"[thomas] burn registration error: {exc}")
                qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
                return None
            # Harden on temporary nodes to guarantee downstream labelmap export uses the
            # registered geometry even if parent transforms are ignored by export code paths.
            self.logic.apply_transform_to_nodes(nodes=burn_seg_nodes, transform_node=transform_node, harden=True)
            self.log(
                f"[thomas] applied burn transform {transform_node.GetName()} "
                f"to {len(burn_seg_nodes)} temporary segmentation nodes (hardened)"
            )
        elif not burn_seg_nodes:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Load THOMAS masks first.",
            )
            return None

        side = (self._widget_text(self.thomasBurnSideCombo) or "Both").strip()
        nucleus = (self._widget_text(self.thomasBurnNucleusCombo) or "").strip()
        fill_value = self._widget_value(self.thomasBurnFillValueSpin)
        output_name = self.thomasBurnOutputNameEdit.text.strip() or "THOMAS_Burned_MRI"

        out_volume = None
        try:
            out_volume = self.logic.burn_thomas_nucleus_to_volume(
                segmentation_nodes=burn_seg_nodes,
                input_volume_node=burn_input_for_burn,
                nucleus=nucleus,
                side=side,
                fill_value=fill_value,
                output_name=output_name,
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[thomas] burn failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return None
        finally:
            for node in temp_seg_nodes:
                if node is not None and node.GetScene() is not None:
                    slicer.mrmlScene.RemoveNode(node)
            for node in temp_volume_nodes:
                if node is not None and node.GetScene() is not None:
                    slicer.mrmlScene.RemoveNode(node)

        self.logic.place_node_under_same_study(out_volume, burn_input_node, logger=self.log)
        self.logic.show_volume_in_all_slice_views(out_volume)
        if hasattr(self.logic, "workflow_publish"):
            self.logic.workflow_publish.register_volume(
                volume_node=out_volume,
                source_type="derived",
                source_path="",
                space_name="ROSA_BASE",
                role="DerivedVolumes",
                is_derived=True,
                derived_from_node_id=burn_input_node.GetID() if burn_input_node is not None else "",
                workflow_node=getattr(self, "workflowNode", None),
            )
        self.log(f"[thomas] burn complete: {out_volume.GetName()}")
        return out_volume

    def onBurnAndExportThomasDicomClicked(self):
        """Run burn workflow then export the created volume as classic DICOM series."""
        reference_volume = None
        if getattr(self, "thomasImportedDicomNodeID", None):
            reference_volume = slicer.mrmlScene.GetNodeByID(self.thomasImportedDicomNodeID)
        if reference_volume is None:
            reference_volume = self.thomasBurnInputSelector.currentNode()
        if reference_volume is None:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select Burn input MRI volume first.",
            )
            return

        out_volume = self.onBurnThomasNucleusClicked()
        if out_volume is None:
            return

        export_dir = (self.thomasDicomExportDirSelector.currentPath or "").strip()
        if not export_dir:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "ROSA Helper",
                "Select DICOM export directory first.",
            )
            return

        series_description = (self.thomasDicomSeriesDescriptionEdit.text or "").strip()
        if not series_description:
            nucleus = (self._widget_text(self.thomasBurnNucleusCombo) or "NUCLEUS").strip().upper()
            side = (self._widget_text(self.thomasBurnSideCombo) or "Both").strip().upper()
            series_description = f"THOMAS_{nucleus}_{side}_BURNED"
            self.thomasDicomSeriesDescriptionEdit.setText(series_description)

        try:
            series_dir = self.logic.export_scalar_volume_to_dicom_series(
                volume_node=out_volume,
                reference_volume_node=reference_volume,
                export_dir=export_dir,
                series_description=series_description,
                modality="MR",
                logger=self.log,
            )
        except Exception as exc:
            self.log(f"[thomas] DICOM export failed: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "ROSA Helper", str(exc))
            return

        self.log(f"[thomas] DICOM export complete: {series_dir}")
