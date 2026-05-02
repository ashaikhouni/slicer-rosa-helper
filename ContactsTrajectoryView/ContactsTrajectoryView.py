"""3D Slicer module for contact generation and trajectory-oriented viewing.

Last updated: 2026-03-01
"""

import datetime
import math
import os
import sys

from __main__ import ctk, qt, slicer, vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH_CANDIDATES = [
    os.path.join(os.path.dirname(MODULE_DIR), "CommonLib"),
    os.path.join(MODULE_DIR, "CommonLib"),
]
for path in PATH_CANDIDATES:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)

from rosa_core import (
    candidate_ids_for_vendors,
    compute_qc_metrics,
    detect_contacts_on_axis,
    electrode_length_mm,
    generate_contacts,
    load_electrode_library,
    lps_to_ras_point,
    model_map,
    ras_contacts_to_contact_records,
    suggest_model_id_for_trajectory,
    trajectory_length_mm,
)
from rosa_core.electrode_classifier import classify_electrode_model
from rosa_scene import (
    ElectrodeSceneService,
    LayoutService,
    TrajectoryFocusController,
    TrajectorySceneService,
    widget_current_text,
)
from rosa_workflow import WorkflowPublisher, WorkflowState

TRAJECTORY_SOURCE_OPTIONS = [
    ("working", "Working (active)"),
    ("imported_rosa", "Imported ROSA"),
    ("imported_external", "Imported External"),
    ("manual", "Manual (scene)"),
    ("guided_fit", "Guided Fit"),
    ("auto_fit", "Auto Fit"),
    ("planned_rosa", "Planned ROSA"),
]


class ContactsTrajectoryView(ScriptedLoadableModule):
    """Slicer module metadata for contact + trajectory workflows."""

    def __init__(self, parent):
        super().__init__(parent)
        self.parent.title = "02 Contacts & Trajectory View"
        self.parent.categories = ["ROSA.02 Localization"]
        self.parent.dependencies = []
        self.parent.contributors = ["Ammar Shaikhouni", "Codex"]
        self.parent.helpText = (
            "Generate electrode contacts from trajectory lines and align slice views to trajectories "
            "using the shared RosaWorkflow MRML contract."
        )


class ContactsTrajectoryViewWidget(ScriptedLoadableModuleWidget):
    """UI for electrode model assignment, contact generation, and slice alignment."""

    def setup(self):
        super().setup()
        # Drop Slicer's default min/max width constraints on the module
        # panel dock so the user can drag the splitter narrower (or
        # wider than the viewport-imposed cap). Same trick as
        # PostopCTLocalization; one-time global tweak per session.
        try:
            mw = slicer.util.mainWindow()
            if mw is not None:
                for dock in mw.findChildren(qt.QDockWidget):
                    if dock.objectName() == "PanelDockWidget":
                        dock.setMinimumWidth(0)
                        dock.setMaximumWidth(16777215)
                        inner = dock.widget()
                        if inner is not None:
                            inner.setMinimumWidth(0)
                            inner.setMaximumWidth(16777215)
                        break
        except Exception:
            pass

        self.logic = ContactsTrajectoryViewLogic()
        self.workflowState = self.logic.workflow_state
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()

        self.modelsById = {}
        self.modelIds = []
        self.loadedTrajectories = []
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self.lastPeakDriftFlags = []
        # User-chosen model per trajectory name. Persists across
        # `_populate_contact_table` rebuilds (Refresh, source change,
        # rename) so the manual choice doesn't get clobbered by
        # Auto Fit's `Rosa.BestModelId` suggestion.
        self._userModelOverrides: dict[str, str] = {}
        self._syncingSourceCombo = False
        self._syncingFocusControls = False
        self._syncingVolumeSelectors = False
        self._pendingFollow = False
        self._pendingFocusLayoutApply = False
        self._updatingContactTable = False
        self._renamingTrajectory = False
        self._workflowObserverTag = None
        self._workflowObserverNode = None
        self._workflowRefreshPending = False
        self._workflowRefreshInFlight = False

        top_form = qt.QFormLayout()
        top_form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        self.layout.addLayout(top_form)

        refresh_row = qt.QHBoxLayout()
        self.refreshButton = qt.QPushButton("Refresh")
        self.refreshButton.clicked.connect(self.onRefreshClicked)
        refresh_row.addWidget(self.refreshButton)
        self.showPlannedCheck = qt.QCheckBox("Show planned")
        self.showPlannedCheck.setChecked(False)
        self.showPlannedCheck.toggled.connect(self.onShowPlannedToggled)
        refresh_row.addWidget(self.showPlannedCheck)
        # Reduce 3D-scene clutter — selected row's trajectory + contacts
        # only, hide the rest. Default OFF so existing
        # multi-shank-overview workflows are preserved.
        self.isolateSelectedCheck = qt.QCheckBox("Isolate selected")
        self.isolateSelectedCheck.setToolTip(
            "When checked, only the selected row's trajectory + "
            "contacts + electrode model are shown in the 3D scene; "
            "all other shanks are hidden."
        )
        self.isolateSelectedCheck.setChecked(False)
        self.isolateSelectedCheck.toggled.connect(self.onIsolateSelectedToggled)
        refresh_row.addWidget(self.isolateSelectedCheck)
        # TL ↔ TD slice intersections are auto-enabled by the layout
        # service when the trajectory-focus 2×3 layout is applied
        # (and auto-disabled on layout restore). No per-module
        # checkbox needed.
        refresh_row.addStretch(1)
        top_form.addRow(refresh_row)

        self.summaryLabel = qt.QLabel("Workflow not scanned yet.")
        self.summaryLabel.wordWrap = True
        self.summaryLabel.setMinimumWidth(0)
        self.summaryLabel.setSizePolicy(
            qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred,
        )
        top_form.addRow("Workflow summary", self.summaryLabel)

        self.trajectorySourceCombo = qt.QComboBox()
        for key, label in TRAJECTORY_SOURCE_OPTIONS:
            self.trajectorySourceCombo.addItem(label, key)
        self.trajectorySourceCombo.currentIndexChanged.connect(self.onTrajectorySourceChanged)
        self.trajectorySourceCombo.setMinimumContentsLength(8)
        self.trajectorySourceCombo.setSizeAdjustPolicy(
            qt.QComboBox.AdjustToMinimumContentsLengthWithIcon,
        )
        self.trajectorySourceCombo.setSizePolicy(
            qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed,
        )
        top_form.addRow("Trajectory source", self.trajectorySourceCombo)

        self._build_contact_ui()
        self._build_qc_ui()
        self._build_slice_view_ui()

        self.statusText = qt.QPlainTextEdit()
        self.statusText.setReadOnly(True)
        self.statusText.setMaximumBlockCount(1500)
        self.layout.addWidget(self.statusText)
        self.layout.addStretch(1)

        self._load_electrode_library()
        self.logic.layout_service.sanitize_focus_layout_state()
        self._ensure_workflow_observer()
        self.onRefreshClicked()

    def enter(self):
        """Refresh module state when entering this module."""
        self._ensure_workflow_observer()
        self.onRefreshClicked()

    def cleanup(self):
        self._remove_workflow_observer()
        parent_cleanup = getattr(super(), "cleanup", None)
        if callable(parent_cleanup):
            parent_cleanup()

    def _remove_workflow_observer(self):
        node = getattr(self, "_workflowObserverNode", None)
        tag = getattr(self, "_workflowObserverTag", None)
        if node is not None and tag is not None:
            try:
                node.RemoveObserver(tag)
            except Exception:
                pass
        self._workflowObserverNode = None
        self._workflowObserverTag = None

    def _ensure_workflow_observer(self):
        node = self.workflowState.resolve_or_create_workflow_node()
        if node is None:
            self._remove_workflow_observer()
            return
        if node is getattr(self, "_workflowObserverNode", None) and getattr(self, "_workflowObserverTag", None) is not None:
            return
        self._remove_workflow_observer()
        self._workflowObserverNode = node
        self._workflowObserverTag = node.AddObserver(vtk.vtkCommand.ModifiedEvent, self._on_workflow_node_modified)

    def _on_workflow_node_modified(self, caller=None, event=None):
        if self._workflowRefreshPending or self._workflowRefreshInFlight:
            return
        self._workflowRefreshPending = True
        qt.QTimer.singleShot(0, self._refresh_from_workflow_change)

    def _refresh_from_workflow_change(self):
        self._workflowRefreshPending = False
        if self._workflowRefreshInFlight:
            return
        self._workflowRefreshInFlight = True
        try:
            self._sync_source_combo_from_workflow()
            self.onRefreshClicked()
        finally:
            self._workflowRefreshInFlight = False

    def _workflow_active_source(self):
        if self.workflowNode is None:
            return ""
        return str(self.workflowNode.GetParameter("ActiveTrajectorySource") or "").strip().lower()

    def _set_workflow_active_source(self, source_key):
        if self.workflowNode is None:
            return
        key = str(source_key or "").strip().lower()
        self.workflowNode.SetParameter("ActiveTrajectorySource", key)

    def _sync_source_combo_from_workflow(self):
        key = self._workflow_active_source()
        if not key:
            return
        idx = self.trajectorySourceCombo.findData(key)
        if idx < 0 or idx == self.trajectorySourceCombo.currentIndex:
            return
        self._syncingSourceCombo = True
        try:
            self.trajectorySourceCombo.setCurrentIndex(idx)
        finally:
            self._syncingSourceCombo = False

    def _workflow_param_bool(self, key, default=False):
        if self.workflowNode is None:
            return bool(default)
        value = str(self.workflowNode.GetParameter(key) or "").strip().lower()
        if not value:
            return bool(default)
        return value in ("1", "true", "yes", "on")

    def _set_workflow_param_bool(self, key, enabled):
        if self.workflowNode is None:
            return
        self.workflowNode.SetParameter(key, "true" if bool(enabled) else "false")

    def _sync_focus_controls_from_workflow(self):
        autofollow = self._workflow_param_bool("TrajectoryFocusAutoFollow", True)
        self._syncingFocusControls = True
        try:
            self.autoFollowTrajectoryCheck.setChecked(bool(autofollow))
        finally:
            self._syncingFocusControls = False

    def _sync_focus_volume_selectors_from_workflow(self):
        if self.workflowNode is None:
            return
        self._syncingVolumeSelectors = True
        try:
            self.focusBaseSelector.setCurrentNode(self.workflowNode.GetNodeReference("BaseVolume"))
            self.focusPostopSelector.setCurrentNode(self.workflowNode.GetNodeReference("PostopCT"))
        finally:
            self._syncingVolumeSelectors = False

    def _role_has_nodes(self, role):
        return len(self.workflowState.role_nodes(role, workflow_node=self.workflowNode)) > 0

    def _default_source_when_unset(self):
        # Preferred startup source is imported ROSA if available.
        if self._role_has_nodes("ImportedTrajectoryLines"):
            return "imported_rosa"
        if self._role_has_nodes("GuidedFitTrajectoryLines"):
            return "guided_fit"
        if self._role_has_nodes("AutoFitTrajectoryLines"):
            return "auto_fit"
        if self._role_has_nodes("ImportedExternalTrajectoryLines"):
            return "imported_external"
        if self._role_has_nodes("PlannedTrajectoryLines"):
            return "planned_rosa"
        if self._role_has_nodes("WorkingTrajectoryLines"):
            return "working"
        return "working"

    def log(self, message):
        """Append one message line to module log."""
        self.statusText.appendPlainText(str(message))
        self.statusText.ensureCursorVisible()
        try:
            slicer.app.processEvents()
        except Exception:
            pass

    def _widget_value(self, widget):
        if widget is None:
            return 0.0
        value_attr = getattr(widget, "value", 0.0)
        value = value_attr() if callable(value_attr) else value_attr
        return float(value)

    def _build_contact_ui(self):
        """Create contact generation controls."""
        section = ctk.ctkCollapsibleButton()
        section.text = "Contact Labels"
        section.collapsed = False
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)

        self.contactTable = qt.QTableWidget()
        self.contactTable.setColumnCount(8)
        # Short header labels to fit narrow columns; tooltips on the
        # header items carry the full meaning for the user to hover.
        _contact_headers = [
            ("Use",     "Use this trajectory for contact generation"),
            ("Traj",    "Trajectory name"),
            ("Len mm",  "Trajectory length (entry → tip) in mm"),
            ("Model",   "Electrode model picked for this trajectory"),
            ("# C",     "Number of contacts (read from the picked model)"),
            ("Elec mm", "Active electrode length in mm"),
            ("Tip At",  "Where the tip sits relative to the trajectory line "
                        "(target vs entry)"),
            ("Shift",   "Tip shift along the axis in mm"),
        ]
        self.contactTable.setHorizontalHeaderLabels(
            [label for label, _ in _contact_headers]
        )
        for col, (_label, tip) in enumerate(_contact_headers):
            item = self.contactTable.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)
        # All columns Interactive with explicit narrow defaults — no
        # column stretches, so the table doesn't sprawl when the panel
        # is wide. User can drag column borders to resize. Horizontal
        # scrollbar engages when the panel is narrower than total
        # column width. setMinimumWidth(0) + Ignored size policy let
        # the table shrink below its natural content width — without
        # this the cell-widget sizeHints (model combo + spinboxes)
        # pin the table (and hence the Slicer panel) to a wide
        # minimum.
        header = self.contactTable.horizontalHeader()
        for col in range(self.contactTable.columnCount):
            header.setSectionResizeMode(col, qt.QHeaderView.Interactive)
        for col, width in (
            (0, 40),    # Use
            (1, 110),   # Trajectory
            (2, 80),    # Traj Length
            (3, 140),   # Electrode Model
            (4, 70),    # # Contacts
            (5, 90),    # Elec Length
            (6, 70),    # Tip At
            (7, 90),    # Tip Shift
        ):
            self.contactTable.setColumnWidth(col, width)
        header.setStretchLastSection(False)
        self.contactTable.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        # Expanding+Preferred + small min width keeps the table
        # shrinkable AND shows the horizontal scrollbar when columns
        # exceed viewport. (Ignored policy disabled the scrollbar.)
        self.contactTable.setMinimumWidth(120)
        self.contactTable.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
        self.contactTable.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self.contactTable.setSelectionMode(qt.QAbstractItemView.SingleSelection)
        self.contactTable.cellClicked.connect(self.onContactTableCellClicked)
        self.contactTable.currentCellChanged.connect(self.onContactTableCurrentCellChanged)
        self.contactTable.itemChanged.connect(self.onContactTableItemChanged)
        form.addRow(self.contactTable)

        defaults_row = qt.QHBoxLayout()
        self.defaultModelCombo = qt.QComboBox()
        self.defaultModelCombo.setMinimumContentsLength(14)
        defaults_row.addWidget(self.defaultModelCombo)
        self.applyModelAllButton = qt.QPushButton("Apply model to all")
        self.applyModelAllButton.clicked.connect(self.onApplyModelAllClicked)
        defaults_row.addWidget(self.applyModelAllButton)
        defaults_row.addStretch(1)
        form.addRow("Default model", defaults_row)

        # Restrict the model library used by the unified picker fallback
        # for rows that arrive without a stamped `Rosa.BestModelId`
        # (legacy planned / imported trajectories). Auto / Guided /
        # Manual Fit each have their own copy of this combo, so
        # restrictions apply per-module.
        from rosa_core.electrode_classifier import PITCH_STRATEGY_OPTIONS
        self.contactsPitchStrategyCombo = qt.QComboBox()
        for label, key in PITCH_STRATEGY_OPTIONS:
            self.contactsPitchStrategyCombo.addItem(label, key)
        self.contactsPitchStrategyCombo.setCurrentIndex(0)  # default: Dixi AM
        self.contactsPitchStrategyCombo.setToolTip(
            "Restrict the model library used by the unified electrode-"
            "model picker fallback for unstamped rows. The Default "
            "model dropdown above is unrestricted; this combo only "
            "affects the fallback picker."
        )
        form.addRow("Pitch strategy", self.contactsPitchStrategyCombo)

        # Model-driven: synthesize contacts at the assigned electrode
        # model's nominal offsets along the fitted line.
        # Peak-driven: sample LoG sigma=1 along the axis, pick peaks,
        # match to the library, emit contacts at the detected peak
        # positions. Falls back to model-driven per-trajectory when
        # the axis doesn't produce enough peaks to resolve a model.
        self.detectionModeCombo = qt.QComboBox()
        self.detectionModeCombo.addItem("Model-driven (nominal)", "model_driven")
        self.detectionModeCombo.addItem("Peak-driven (CT peaks)", "peak_driven")
        form.addRow("Detection mode", self.detectionModeCombo)

        self.contactsNodeNameEdit = qt.QLineEdit("ROSA_Contacts")
        form.addRow("Output node prefix", self.contactsNodeNameEdit)

        self.createModelsCheck = qt.QCheckBox("Create electrode models")
        self.createModelsCheck.setChecked(True)
        form.addRow("Model option", self.createModelsCheck)

        # When the cylinders are visible they obscure the contact
        # fiducial glyphs in slice viewers, making click-to-drag
        # painful for GT correction. Toggle hides every published
        # ``ElectrodeShaftModelNodes`` + ``ElectrodeContactModelNodes``
        # so the glyphs are pickable; cylinders remain in the scene
        # so re-checking the box doesn't require re-generating.
        self.showModelsCheck = qt.QCheckBox("Show electrode models")
        self.showModelsCheck.setChecked(True)
        self.showModelsCheck.setToolTip(
            "Uncheck to hide the cylinder visualizations so contact "
            "fiducial glyphs become clickable for hand-correction in "
            "slice views."
        )
        self.showModelsCheck.toggled.connect(self.onShowModelsToggled)
        form.addRow("", self.showModelsCheck)

        button_row = qt.QHBoxLayout()
        self.generateContactsButton = qt.QPushButton("Generate Contacts")
        self.generateContactsButton.clicked.connect(self.onGenerateContactsClicked)
        self.generateContactsButton.setEnabled(False)
        button_row.addWidget(self.generateContactsButton)
        self.updateContactsButton = qt.QPushButton("Update From Edited")
        self.updateContactsButton.clicked.connect(self.onUpdateContactsClicked)
        self.updateContactsButton.setEnabled(False)
        button_row.addWidget(self.updateContactsButton)
        form.addRow(button_row)

        # GT-export row. The auto-snap dataset *_contacts.tsv files are
        # NOT human-curated (every row reads coord_source=World,
        # snap_status=unchanged, move_mm=0); the placement test gate's
        # 1.5 / 1.25 mm error budget tells us nothing real. This button
        # writes the user's hand-corrected fiducial positions to a
        # parallel labels_gt/<sid>_contacts_gt.tsv that the test
        # framework can prefer over the auto-snap. See
        # ``project_contact_gt_annotation_workflow.md`` for the full
        # plan.
        gt_row = qt.QHBoxLayout()
        self.saveContactsAsGtButton = qt.QPushButton("Save Contacts as GT")
        self.saveContactsAsGtButton.setToolTip(
            "Write the current contact-fiducial positions to "
            "labels_gt/<subject>_contacts_gt.tsv. Run Generate "
            "Contacts first, hand-correct any wrong fiducials in the "
            "slice viewers, then click here. The TSV reuses the "
            "labels/ schema with coord_source=manual + a verified-at "
            "timestamp; move_mm is the delta from the auto-generated "
            "position."
        )
        self.saveContactsAsGtButton.setEnabled(False)
        self.saveContactsAsGtButton.clicked.connect(self.onSaveContactsAsGtClicked)
        gt_row.addWidget(self.saveContactsAsGtButton)
        gt_row.addStretch(1)

        # Curry .pom export lives in the dedicated Export Center
        # module — a single home for all bundle-style outputs keeps
        # the CTV contact panel focused on contact generation.
        form.addRow(gt_row)

    def _build_qc_ui(self):
        """Create QC metric table."""
        self.qcSection = ctk.ctkCollapsibleButton()
        self.qcSection.text = "Trajectory QC Metrics"
        self.qcSection.collapsed = True
        self.layout.addWidget(self.qcSection)
        qf = qt.QFormLayout(self.qcSection)
        qf.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)

        self.qcStatusLabel = qt.QLabel("QC disabled: generate contacts first.")
        self.qcStatusLabel.wordWrap = True
        qf.addRow(self.qcStatusLabel)

        self.qcTable = qt.QTableWidget()
        self.qcTable.setColumnCount(8)
        _qc_headers = [
            ("Traj",      "Trajectory name"),
            ("Entry RE",  "Entry residual error in mm"),
            ("Target RE", "Target residual error in mm"),
            ("Mean RE",   "Mean residual error in mm across all contacts"),
            ("Max RE",    "Max residual error in mm"),
            ("RMS RE",    "Root-mean-square residual error in mm"),
            ("Angle°",    "Angle deviation from the planned trajectory in degrees"),
            ("N",         "Number of contacts"),
        ]
        self.qcTable.setHorizontalHeaderLabels(
            [label for label, _ in _qc_headers]
        )
        for col, (_label, tip) in enumerate(_qc_headers):
            item = self.qcTable.horizontalHeaderItem(col)
            if item is not None:
                item.setToolTip(tip)
        # Interactive columns + scrollable + shrinkable size policy so
        # the QC table doesn't pin the panel to its full content width.
        qc_header = self.qcTable.horizontalHeader()
        for col in range(self.qcTable.columnCount):
            qc_header.setSectionResizeMode(col, qt.QHeaderView.Interactive)
        for col, width in (
            (0, 100),   # Trajectory
            (1, 80),    # Entry RE
            (2, 80),    # Target RE
            (3, 80),    # Mean RE
            (4, 70),    # Max RE
            (5, 70),    # RMS RE
            (6, 70),    # Angle
            (7, 40),    # N
        ):
            self.qcTable.setColumnWidth(col, width)
        qc_header.setStretchLastSection(False)
        self.qcTable.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        self.qcTable.setMinimumWidth(120)
        self.qcTable.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Preferred)
        self.qcTable.setSelectionMode(qt.QAbstractItemView.NoSelection)
        qf.addRow(self.qcTable)
        self._set_qc_enabled(False, "QC disabled: generate contacts first.")

    def _build_slice_view_ui(self):
        """Create trajectory-focus layout controls."""
        section = ctk.ctkCollapsibleButton()
        section.text = "Trajectory Focus View"
        section.collapsed = False
        self.layout.addWidget(section)
        form = qt.QFormLayout(section)
        form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)

        # Two MRML node selectors plus a button on a single row pinned
        # the panel to ~500 px. Stack each selector on its own form
        # row so the panel can narrow to a single combo width.
        self.focusBaseSelector = slicer.qMRMLNodeComboBox()
        self.focusBaseSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.focusBaseSelector.noneEnabled = True
        self.focusBaseSelector.addEnabled = False
        self.focusBaseSelector.removeEnabled = False
        self.focusBaseSelector.renameEnabled = False
        self.focusBaseSelector.setMRMLScene(slicer.mrmlScene)
        self.focusBaseSelector.setMinimumWidth(0)
        form.addRow("Base", self.focusBaseSelector)

        self.focusPostopSelector = slicer.qMRMLNodeComboBox()
        self.focusPostopSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.focusPostopSelector.noneEnabled = True
        self.focusPostopSelector.addEnabled = False
        self.focusPostopSelector.removeEnabled = False
        self.focusPostopSelector.renameEnabled = False
        self.focusPostopSelector.setMRMLScene(slicer.mrmlScene)
        self.focusPostopSelector.setMinimumWidth(0)
        form.addRow("Postop", self.focusPostopSelector)

        self.applyFocusVolumesButton = qt.QPushButton("Set Base/Postop")
        self.applyFocusVolumesButton.clicked.connect(self.onApplyFocusVolumesClicked)
        form.addRow(self.applyFocusVolumesButton)

        focus_row = qt.QHBoxLayout()
        self.applyFocusLayoutButton = qt.QPushButton("Apply 2×3 Layout")
        self.applyFocusLayoutButton.clicked.connect(self.onApplyFocusLayoutClicked)
        focus_row.addWidget(self.applyFocusLayoutButton)
        self.restoreLayoutButton = qt.QPushButton("Restore Layout")
        self.restoreLayoutButton.clicked.connect(self.onResetFocusLayoutClicked)
        focus_row.addWidget(self.restoreLayoutButton)
        focus_row.addStretch(1)
        form.addRow(focus_row)

        self.autoFollowTrajectoryCheck = qt.QCheckBox("Auto-follow")
        self.autoFollowTrajectoryCheck.setChecked(True)
        self.autoFollowTrajectoryCheck.toggled.connect(self.onAutoFollowToggled)
        form.addRow(self.autoFollowTrajectoryCheck)

    def _set_readonly_text_item(self, row, column, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.contactTable.setItem(row, column, item)

    def _set_editable_trajectory_item(self, row, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() | qt.Qt.ItemIsEditable)
        item.setData(qt.Qt.UserRole, str(text))
        self.contactTable.setItem(row, 1, item)

    def _set_qc_enabled(self, enabled, message=""):
        self.qcSection.setEnabled(bool(enabled))
        self.qcTable.setEnabled(bool(enabled))
        if message:
            self.qcStatusLabel.setText(message)

    def _set_qc_table_item(self, row, column, text):
        item = qt.QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~qt.Qt.ItemIsEditable)
        self.qcTable.setItem(row, column, item)

    def _build_model_combo(self):
        combo = qt.QComboBox()
        combo.setMinimumContentsLength(14)
        combo.addItem("")
        for model_id in self.modelIds:
            combo.addItem(model_id)
        return combo

    def _build_tip_at_combo(self):
        combo = qt.QComboBox()
        combo.addItems(["target", "entry"])
        return combo

    def _build_tip_shift_spinbox(self):
        spin = qt.QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(-50.0, 50.0)
        spin.setSingleStep(0.25)
        spin.setValue(0.0)
        spin.setSuffix(" mm")
        return spin

    def _build_contact_count_spinbox(self):
        """Integer spinbox for the # Contacts column. Auto-populated
        when an electrode model is assigned; editable independently for
        peak-driven model-free emission. ``0`` = "use the model" (or
        emit all detected peaks if no model is assigned).
        """
        spin = qt.QSpinBox()
        spin.setRange(0, 32)
        spin.setValue(0)
        spin.setToolTip(
            "Number of contacts to emit in peak-driven mode.\n"
            "  - 0 (default): use the assigned model's contact count "
            "(or emit all detected peaks if no model is assigned).\n"
            "  - >0: emit exactly N contacts as the strongest peaks "
            "along the axis. Lets you skip the model assignment when "
            "the count is known but the exact electrode pattern isn't."
        )
        return spin

    def _build_use_checkbox(self):
        check = qt.QCheckBox()
        check.setChecked(True)
        return check

    def _row_is_selected(self, row):
        check = self.contactTable.cellWidget(row, 0)
        return bool(check and check.checked)

    def _electrode_length_mm(self, model_id):
        model = self.modelsById.get(model_id, {})
        return electrode_length_mm(model)

    def _bind_model_length_update(self, model_combo, row):
        # Both the electrode-length cell AND the # Contacts spinbox
        # auto-update when the model selection changes. The user
        # override is recorded too so the choice survives the next
        # ``_populate_contact_table`` rebuild — without this, Refresh
        # / source change / rename all silently revert to Auto Fit's
        # ``Rosa.BestModelId`` suggestion.
        def _on_model_change(_arg, row_index=row):
            if not self._updatingContactTable:
                self._record_user_model_override(row_index)
            self._update_electrode_length_cell(row_index)
            self._update_contact_count_cell(row_index)

        if hasattr(model_combo, "currentTextChanged"):
            model_combo.currentTextChanged.connect(_on_model_change)
        else:
            model_combo.currentIndexChanged.connect(_on_model_change)

    def _record_user_model_override(self, row):
        name_item = self.contactTable.item(row, 1)
        if name_item is None:
            return
        name = str(name_item.text() or "").strip()
        if not name:
            return
        combo = self.contactTable.cellWidget(row, 3)
        model_id = widget_current_text(combo).strip() if combo else ""
        if model_id:
            self._userModelOverrides[name] = model_id
        else:
            self._userModelOverrides.pop(name, None)

    def _update_electrode_length_cell(self, row):
        model_combo = self.contactTable.cellWidget(row, 3)
        model_id = widget_current_text(model_combo).strip()
        length_text = ""
        if model_id:
            length_text = f"{self._electrode_length_mm(model_id):.2f}"
        self._set_readonly_text_item(row, 5, length_text)

    def _update_contact_count_cell(self, row):
        """Sync the # Contacts spinbox to the currently-selected
        electrode model. Overwrites any prior user value — the model
        change is the implicit "I want this many contacts" signal; the
        user can re-edit afterward to override.
        """
        spin = self.contactTable.cellWidget(row, 4)
        if spin is None:
            return
        model_combo = self.contactTable.cellWidget(row, 3)
        model_id = widget_current_text(model_combo).strip()
        n = 0
        if model_id:
            offsets = self.modelsById.get(model_id, {}).get(
                "contact_center_offsets_from_tip_mm",
            ) or []
            n = len(offsets)
        spin.blockSignals(True)
        try:
            spin.setValue(int(n))
        finally:
            spin.blockSignals(False)

    def _get_contact_count(self, row):
        spin = self.contactTable.cellWidget(row, 4)
        if spin is None:
            return 0
        try:
            return int(spin.value)
        except Exception:
            return 0

    def _load_electrode_library(self):
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

    def _populate_contact_table(self, trajectories):
        self._updatingContactTable = True
        self.contactTable.setRowCount(0)
        auto_assigned = 0
        for row, traj in enumerate(trajectories):
            self.contactTable.insertRow(row)
            self.contactTable.setCellWidget(row, 0, self._build_use_checkbox())
            self._set_editable_trajectory_item(row, traj["name"])
            self._set_readonly_text_item(row, 2, f"{trajectory_length_mm(traj):.2f}")

            model_combo = self._build_model_combo()
            self._bind_model_length_update(model_combo, row)
            # Precedence: user's manual override > Auto-Fit suggestion >
            # unified PaCER-style picker run live (CT-aware). Without this,
            # manually-picked models silently revert to `best_model_id`
            # on every populate cycle, and rows imported from sources
            # that didn't stamp `best_model_id` (legacy planned/imported
            # without picker) get the unified picker as a one-shot here.
            suggested_model = self._userModelOverrides.get(traj["name"], "")
            if not suggested_model:
                suggested_model = str(traj.get("best_model_id") or "").strip()
            if not suggested_model:
                pick = self._classify_with_unified_picker(traj)
                suggested_model = str(pick or "")
            if not suggested_model:
                # Last-ditch length-only fallback (no CT volume available).
                suggested_model = suggest_model_id_for_trajectory(
                    trajectory=traj,
                    models_by_id=self.modelsById,
                    model_ids=self.modelIds,
                    tolerance_mm=5.0,
                )
            if suggested_model:
                idx = model_combo.findText(suggested_model)
                if idx >= 0:
                    model_combo.setCurrentIndex(idx)
                    auto_assigned += 1
            self.contactTable.setCellWidget(row, 3, model_combo)
            self.contactTable.setCellWidget(row, 4, self._build_contact_count_spinbox())
            self._set_readonly_text_item(row, 5, "")
            self._update_electrode_length_cell(row)
            self._update_contact_count_cell(row)
            self.contactTable.setCellWidget(row, 6, self._build_tip_at_combo())
            self.contactTable.setCellWidget(row, 7, self._build_tip_shift_spinbox())

        enabled = bool(trajectories) and bool(self.modelsById)
        self.generateContactsButton.setEnabled(enabled)
        self.updateContactsButton.setEnabled(enabled)
        self.applyModelAllButton.setEnabled(enabled)
        if trajectories:
            self.log(f"[contacts] ready for {len(trajectories)} trajectories")
            self.log(f"[contacts] auto-assigned models for {auto_assigned}/{len(trajectories)}")
            self.contactTable.selectRow(0)
        else:
            self.log("[contacts] no trajectories available")
        self._updatingContactTable = False

    def onContactTableItemChanged(self, item):
        if self._updatingContactTable or self._renamingTrajectory:
            return
        if item is None or item.column() != 1:
            return
        row = int(item.row())
        if row < 0 or row >= len(self.loadedTrajectories):
            return

        old_name = str(item.data(qt.Qt.UserRole) or self.loadedTrajectories[row].get("name", "")).strip()
        new_name = (item.text() or "").strip()
        if not old_name:
            return
        if not new_name:
            self._renamingTrajectory = True
            item.setText(old_name)
            self._renamingTrajectory = False
            return
        if new_name == old_name:
            return

        for r in range(self.contactTable.rowCount):
            if r == row:
                continue
            other = self.contactTable.item(r, 1)
            if other and (other.text() or "").strip().lower() == new_name.lower():
                self._renamingTrajectory = True
                item.setText(old_name)
                self._renamingTrajectory = False
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Contacts & Trajectory View",
                    f"Trajectory name '{new_name}' already exists.",
                )
                return

        node_id = self.loadedTrajectories[row].get("node_id", "")
        renamed = self.logic.rename_trajectory(
            node_id=node_id,
            new_name=new_name,
        )
        if not renamed:
            self._renamingTrajectory = True
            item.setText(old_name)
            self._renamingTrajectory = False
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                f"Failed to rename trajectory '{old_name}'.",
            )
            return

        self.loadedTrajectories[row]["name"] = new_name
        item.setData(qt.Qt.UserRole, new_name)
        # Carry the model override across the rename so the manual
        # choice doesn't get dropped just because the row got a new
        # display name.
        if old_name in self._userModelOverrides:
            self._userModelOverrides[new_name] = self._userModelOverrides.pop(old_name)
        self.log(f"[trajectory] renamed {old_name} -> {new_name}")
        self._refresh_summary()
        self._schedule_follow_selected_trajectory()

    def _collect_assignments(self):
        rows = []
        for row in range(self.contactTable.rowCount):
            if not self._row_is_selected(row):
                continue
            traj_item = self.contactTable.item(row, 1)
            if not traj_item:
                continue
            model_combo = self.contactTable.cellWidget(row, 3)
            tip_at_combo = self.contactTable.cellWidget(row, 6)
            tip_shift_spin = self.contactTable.cellWidget(row, 7)
            model_id = widget_current_text(model_combo).strip()
            n_contacts = self._get_contact_count(row)
            # Rows without a model are kept for the peak-driven path —
            # the engine emits all detected peaks (or the strongest
            # ``n_contacts`` of them when set). Model-driven generation
            # still requires a model and skips no-model rows
            # downstream.
            rows.append(
                {
                    "trajectory": traj_item.text(),
                    "model_id": model_id,
                    "n_contacts_target": int(n_contacts) if n_contacts > 0 else None,
                    "tip_at": widget_current_text(tip_at_combo) or "target",
                    "tip_shift_mm": self._widget_value(tip_shift_spin),
                    "xyz_offset_mm": [0.0, 0.0, 0.0],
                }
            )
        return {"schema_version": "1.0", "assignments": rows}

    def _build_trajectory_map_with_scene_overrides(self):
        out = {}
        for traj in self.loadedTrajectories:
            node = slicer.mrmlScene.GetNodeByID(traj.get("node_id", ""))
            scene_traj = self.logic.trajectory_scene.trajectory_from_line_node(traj.get("name", ""), node)
            out[traj["name"]] = scene_traj if scene_traj is not None else traj
        return out

    def _compute_qc_rows(self):
        if not self.lastGeneratedContacts:
            return [], "QC disabled: generate contacts first."
        planned_map = self.logic.collect_planned_trajectory_map(workflow_node=self.workflowNode)
        if not planned_map:
            return [], "QC disabled: no planned trajectories (Plan_*) found."
        if not self.lastAssignments.get("assignments"):
            return [], "QC disabled: no electrode assignments available."

        final_map = self._build_trajectory_map_with_scene_overrides()
        try:
            planned_contacts = generate_contacts(
                list(planned_map.values()),
                self.modelsById,
                self.lastAssignments,
            )
        except Exception as exc:
            return [], f"QC disabled: failed to generate planned contacts ({exc})."

        rows = compute_qc_metrics(
            planned_trajectories_by_name=planned_map,
            final_trajectories_by_name=final_map,
            planned_contacts=planned_contacts,
            final_contacts=self.lastGeneratedContacts,
            include_unmatched_planned=True,
        )
        if not rows:
            return [], "QC disabled: no planned trajectories available for comparison."
        return rows, f"QC metrics computed for {len(rows)} planned trajectories."

    def _fmt_qc_metric(self, value):
        if value is None:
            return "NA"
        try:
            return f"{float(value):.2f}"
        except Exception:
            return "NA"

    def _refresh_qc_metrics(self):
        rows, status = self._compute_qc_rows()
        self.lastQCMetricsRows = rows
        self.qcTable.setRowCount(0)
        if not rows:
            self._set_qc_enabled(False, status)
            return
        self._set_qc_enabled(True, status)
        self.qcTable.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self._set_qc_table_item(r, 0, row["trajectory"])
            self._set_qc_table_item(r, 1, self._fmt_qc_metric(row.get("entry_radial_mm")))
            self._set_qc_table_item(r, 2, self._fmt_qc_metric(row.get("target_radial_mm")))
            self._set_qc_table_item(r, 3, self._fmt_qc_metric(row.get("mean_contact_radial_mm")))
            self._set_qc_table_item(r, 4, self._fmt_qc_metric(row.get("max_contact_radial_mm")))
            self._set_qc_table_item(r, 5, self._fmt_qc_metric(row.get("rms_contact_radial_mm")))
            self._set_qc_table_item(r, 6, self._fmt_qc_metric(row.get("angle_deg")))
            self._set_qc_table_item(r, 7, str(row["matched_contacts"]))

    def _refresh_summary(self):
        planned = self.logic.collect_planned_trajectory_map(workflow_node=self.workflowNode)
        source_key = self.trajectorySourceCombo.currentData
        source_key = source_key() if callable(source_key) else source_key
        summary = (
            f"source={source_key or 'working'}, "
            f"loaded trajectories={len(self.loadedTrajectories)}, "
            f"planned trajectories={len(planned)}, "
            f"contacts={len(self.lastGeneratedContacts)}"
        )
        self.summaryLabel.setText(summary)

    def _apply_source_visibility(self, source_key):
        key = str(source_key or "working").strip().lower()
        group_map = {
            "working": ["imported_rosa", "imported_external", "manual", "guided_fit", "auto_fit"],
            "imported_rosa": ["imported_rosa"],
            "imported_external": ["imported_external"],
            "manual": ["manual"],
            "guided_fit": ["guided_fit"],
            "auto_fit": ["auto_fit"],
            "planned_rosa": ["planned_rosa"],
        }
        groups = group_map.get(key, group_map["working"])
        self.logic.trajectory_scene.show_only_groups(groups)
        # Planned visibility remains user-controlled except when planned source is selected directly.
        if key == "planned_rosa":
            self.logic.electrode_scene.set_planned_trajectory_visibility(True)
        else:
            self.logic.electrode_scene.set_planned_trajectory_visibility(bool(self.showPlannedCheck.checked))

    def onRefreshClicked(self):
        self.workflowNode = self.workflowState.resolve_or_create_workflow_node()
        self.logic.layout_service.sanitize_focus_layout_state()
        if not self._workflow_active_source():
            self._set_workflow_active_source(self._default_source_when_unset())
        self._sync_focus_controls_from_workflow()
        self._sync_source_combo_from_workflow()
        self._sync_focus_volume_selectors_from_workflow()
        source_key = self._selected_source_key()
        self.loadedTrajectories = self.logic.collect_trajectories_by_source(
            source_key=source_key,
            workflow_node=self.workflowNode,
        )
        self.lastGeneratedContacts = []
        self.lastAssignments = {"schema_version": "1.0", "assignments": []}
        self.lastQCMetricsRows = []
        self._populate_contact_table(self.loadedTrajectories)
        self._apply_source_visibility(source_key)
        self._refresh_qc_metrics()
        self._refresh_summary()
        self.log(f"[refresh] source={source_key} trajectories={len(self.loadedTrajectories)}")
        if not self.logic.layout_service.has_focus_slice_views():
            self._schedule_apply_focus_layout()
        self._apply_focus_slice_layers()
        self._schedule_follow_selected_trajectory()

    def _selected_source_key(self):
        data = self.trajectorySourceCombo.currentData
        value = data() if callable(data) else data
        return str(value or "working")

    def onTrajectorySourceChanged(self, _idx):
        if self._syncingSourceCombo:
            return
        self._set_workflow_active_source(self._selected_source_key())
        self.onRefreshClicked()

    def onApplyModelAllClicked(self):
        model_id = widget_current_text(self.defaultModelCombo).strip()
        if not model_id:
            return
        for row in range(self.contactTable.rowCount):
            combo = self.contactTable.cellWidget(row, 3)
            if combo:
                idx = combo.findText(model_id)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

    def _selected_detection_mode(self):
        data = self.detectionModeCombo.currentData
        value = data() if callable(data) else data
        return str(value or "model_driven").strip().lower()

    def _resolve_postop_ct_node(self):
        """Return the PostopCT volume node (workflow role) or None."""
        nodes = self.workflowState.role_nodes(
            "PostopCT", workflow_node=self.workflowNode,
        )
        return nodes[0] if nodes else None

    def _classify_with_unified_picker(self, traj):
        """Run the unified electrode-model picker on one trajectory dict.

        PaCER template-correlation against the canonical-resampled CT
        when available, else the legacy length-only fallback (which the
        caller does directly). Returns a model_id string or empty.
        """
        try:
            ct_node = self._resolve_postop_ct_node()
            if ct_node is None:
                return ""
            import SimpleITK as sitk
            from shank_core.io import image_ijk_ras_matrices
            from rosa_detect.contact_pitch_v1_fit import prepare_volume
            arr = slicer.util.arrayFromVolume(ct_node)
            img = sitk.GetImageFromArray(arr)
            i2r, r2i = image_ijk_ras_matrices(ct_node)
            img, _i2r_canon, r2i_canon = prepare_volume(img, i2r, r2i)
            ct_arr_kji = sitk.GetArrayFromImage(img).astype("float32")
            start_ras = traj.get("start_ras") or traj.get("start")
            end_ras = traj.get("end_ras") or traj.get("end")
            if not start_ras or not end_ras:
                return ""
            strategy = "auto"
            combo = getattr(self, "contactsPitchStrategyCombo", None)
            if combo is not None:
                data = combo.currentData
                if isinstance(data, str) and data:
                    strategy = data
            pick = classify_electrode_model(
                start_ras=start_ras, end_ras=end_ras,
                pitch_strategy=strategy,
                ct_volume_kji=ct_arr_kji,
                ras_to_ijk_mat=r2i_canon,
            )
            if pick is None:
                return ""
            return str(pick.get("model_id") or "")
        except Exception:
            return ""

    def _resolve_log_volume_node(self, ct_node):
        """Look up the Auto-Fit-stashed ``<CT>_ContactPitch_LoG_sigma1``
        volume node in the scene if it exists, matching by base name of
        ``ct_node``. Returns None when missing.
        """
        if ct_node is None:
            return None
        base = ct_node.GetName() or ""
        if not base:
            return None
        candidate_name = f"{base}_ContactPitch_LoG_sigma1"
        found = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        for node in found:
            if node.GetName() == candidate_name:
                return node
        return None

    def _compute_log_volume_from_ct(self, ct_node):
        """Compute LoG sigma=1 on the CT node's image data, in memory."""
        import numpy as np
        import SimpleITK as sitk
        arr_kji = slicer.util.arrayFromVolume(ct_node)
        if arr_kji is None:
            raise RuntimeError("CT volume has no array data")
        img = sitk.GetImageFromArray(np.asarray(arr_kji, dtype=np.float32))
        spacing = [0.0, 0.0, 0.0]
        ct_node.GetSpacing(spacing)
        img.SetSpacing(tuple(float(s) for s in spacing))
        log_img = sitk.LaplacianRecursiveGaussian(img, sigma=1.0)
        return sitk.GetArrayFromImage(log_img).astype(np.float32)

    def _ras_to_ijk_matrix_np(self, volume_node):
        """Return the 4x4 RAS→IJK matrix of a scalar volume as numpy."""
        import numpy as np
        m = vtk.vtkMatrix4x4()
        volume_node.GetRASToIJKMatrix(m)
        out = np.zeros((4, 4), dtype=float)
        for i in range(4):
            for j in range(4):
                out[i, j] = m.GetElement(i, j)
        return out

    def _candidate_ids_from_default_combo(self):
        """Return the electrode ids matching the vendor token of the
        ``defaultModelCombo`` text, or all ids when the combo is blank.
        Gives a lightweight vendor filter without adding more UI.
        """
        default = widget_current_text(self.defaultModelCombo).strip()
        if not default:
            return candidate_ids_for_vendors(
                self.modelsById, vendors=["DIXI", "PMT", "AdTech"],
            )
        vendor = default.split("-", 1)[0]
        return candidate_ids_for_vendors(self.modelsById, vendors=[vendor])

    def _generate_contacts_peak_driven(self, trajectories, assignments, log_context):
        """Peak-driven contact detection path. Returns
        ``(contacts, peak_fit_by_traj, updated_assignments)``.

        For each trajectory:
          1. Resolve the CT volume + LoG volume (precomputed by Auto
             Fit, or computed here via SITK).
          2. Convert trajectory LPS endpoints to RAS.
          3. Run ``detect_contacts_on_axis`` restricted to the user-
             assigned model when one is set; otherwise let the engine
             choose the best-matching library model.
          4. Emit contact records at the detected peak positions;
             fall back to the model-driven nominal-offset synthesis
             when the engine rejects the fit.
        """
        ct_node = self._resolve_postop_ct_node()
        if ct_node is None:
            raise ValueError(
                "PostopCT volume not available in the workflow — peak-driven "
                "mode needs the CT to sample. Assign it via the Focus view "
                "selectors or run Auto Fit first."
            )
        log_node = self._resolve_log_volume_node(ct_node)
        if log_node is not None:
            import numpy as np
            log_arr = np.asarray(slicer.util.arrayFromVolume(log_node))
            self.log(
                f"[contacts:{log_context}] reusing LoG volume '{log_node.GetName()}'"
            )
        else:
            self.log(
                f"[contacts:{log_context}] no cached LoG volume — computing sigma=1 on CT"
            )
            log_arr = self._compute_log_volume_from_ct(ct_node)
        ras_to_ijk = self._ras_to_ijk_matrix_np(ct_node)

        candidate_ids = self._candidate_ids_from_default_combo()
        assignment_by_name = {
            row.get("trajectory", ""): row for row in assignments.get("assignments", [])
        }
        contacts = []
        peak_fit_by_traj = {}
        fallback_names = []
        for traj in trajectories:
            name = traj.get("name", "")
            row = assignment_by_name.get(name)
            tip_at = (row.get("tip_at") if row else "target") or "target"
            assigned_model = (row.get("model_id") if row else "") or ""
            n_contacts_target = row.get("n_contacts_target") if row else None
            # Two modes: model-driven (with assigned model, peaks are
            # snapped to the model's slot pattern) and model-free
            # (no model — emit detected peaks directly, optionally
            # capped at ``n_contacts_target`` strongest peaks).
            use_model_free = not assigned_model
            start_lps = traj.get("start") or [0.0, 0.0, 0.0]
            end_lps = traj.get("end") or [0.0, 0.0, 0.0]
            # Trajectories travel from entry → target; the deep tip is
            # the target end by convention (same as model-driven mode).
            entry_ras = lps_to_ras_point(list(start_lps))
            target_ras = lps_to_ras_point(list(end_lps))
            try:
                result = detect_contacts_on_axis(
                    start_ras=entry_ras,
                    end_ras=target_ras,
                    log_volume_kji=log_arr,
                    ras_to_ijk_mat=ras_to_ijk,
                    models_by_id=self.modelsById,
                    candidate_ids=candidate_ids,
                    restrict_to_model_id=assigned_model or None,
                    model_free=use_model_free,
                    n_contacts_target=n_contacts_target if use_model_free else None,
                )
            except Exception as exc:
                self.log(
                    f"[contacts:{log_context}] peak fit failed for {name}: {exc}"
                )
                result = None

            peak_fit_by_traj[name] = result
            if result is not None and result.model_id:
                records = ras_contacts_to_contact_records(
                    result, traj, tip_at_for_schema=tip_at,
                )
                if row is not None and not use_model_free:
                    # Engine may have chosen a different model than
                    # the combo — reflect that in the stored
                    # assignment so downstream consumers see the
                    # winner. (Skipped in model-free mode: there's no
                    # model selection to write back, ``result.model_id``
                    # is the "manual" sentinel.)
                    row["model_id"] = result.model_id
                contacts.extend(records)
                if use_model_free:
                    self.log(
                        f"[contacts:{log_context}] {name}: model-free peak fit "
                        f"({result.n_matched} peaks emitted from "
                        f"{result.n_peaks_found} detected)"
                    )
                else:
                    self.log(
                        f"[contacts:{log_context}] {name}: peak fit "
                        f"{result.model_id} "
                        f"({result.n_matched}/{result.n_model_slots} peaks, "
                        f"mean res {result.mean_residual_mm:.2f} mm)"
                    )
                continue

            # Fallback: synthesize at the user-assigned model's nominal
            # offsets. Require an assigned model to fall back; empty
            # means the user hasn't picked anything for this row.
            reason = (
                result.rejected_reason
                if result is not None
                else "engine_error"
            )
            fallback_names.append(f"{name} ({reason})")
            if not assigned_model:
                continue
            fallback = generate_contacts(
                [traj], self.modelsById,
                {"schema_version": "1.0", "assignments": [row]} if row else {
                    "schema_version": "1.0",
                    "assignments": [{
                        "trajectory": name,
                        "model_id": assigned_model,
                        "tip_at": tip_at,
                        "tip_shift_mm": 0.0,
                        "xyz_offset_mm": [0.0, 0.0, 0.0],
                    }],
                },
            )
            for rec in fallback:
                rec["peak_detected"] = False  # nominal
            contacts.extend(fallback)

        if fallback_names:
            self.log(
                f"[contacts:{log_context}] peak→nominal fallback for: "
                + ", ".join(fallback_names)
            )
        return contacts, peak_fit_by_traj, assignments

    def _sync_model_combos_from_assignments(self, assignments):
        """Push assignment model_id back into the row model combos so
        the user sees which model the engine picked in peak-driven mode.
        """
        by_name = {
            row.get("trajectory", ""): row.get("model_id", "")
            for row in assignments.get("assignments", [])
        }
        for row_index in range(self.contactTable.rowCount):
            traj_item = self.contactTable.item(row_index, 1)
            if traj_item is None:
                continue
            model_id = by_name.get(traj_item.text(), "")
            if not model_id:
                continue
            combo = self.contactTable.cellWidget(row_index, 3)
            if combo is None:
                continue
            idx = combo.findText(model_id)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _compute_peak_vs_nominal_drift(self, peak_fit_by_traj,
                                        trajectories_by_name, assignments,
                                        drift_threshold_mm=1.0):
        """Compute per-slot drift between peak-detected positions and
        the same model's nominal positions along the assigned axis.

        Returns a list of dicts (one per trajectory) with keys:
          trajectory, model_id, max_drift_mm, mean_drift_mm,
          n_slots_above_threshold, n_slots_detected.
        """
        import numpy as np
        assignment_by_name = {
            row.get("trajectory", ""): row for row in assignments.get("assignments", [])
        }
        out = []
        for name, result in peak_fit_by_traj.items():
            if result is None or not result.model_id:
                continue
            # Drift is "peak vs model nominal" — undefined for the
            # model-free path (``model_id="manual"``), since there's
            # no slot pattern to compare against.
            if result.model_id not in self.modelsById:
                continue
            traj = trajectories_by_name.get(name)
            row = assignment_by_name.get(name)
            if traj is None or row is None:
                continue
            nominal = generate_contacts([traj], self.modelsById, {
                "schema_version": "1.0",
                "assignments": [{
                    "trajectory": name,
                    "model_id": result.model_id,
                    "tip_at": row.get("tip_at", "target"),
                    "tip_shift_mm": 0.0,
                    "xyz_offset_mm": [0.0, 0.0, 0.0],
                }],
            })
            # nominal positions are LPS; peak positions are RAS.
            # Convert both to the same frame by flipping the peak ones.
            drift = []
            for idx, (peak_ras, detected) in enumerate(zip(
                result.positions_ras, result.peak_detected,
            )):
                if not detected or idx >= len(nominal):
                    continue
                peak_lps = lps_to_ras_point(list(peak_ras))
                nom_lps = nominal[idx]["position_lps"]
                d = float(np.linalg.norm(
                    np.asarray(peak_lps) - np.asarray(nom_lps)
                ))
                drift.append(d)
            drift_arr = np.asarray(drift, dtype=float) if drift else np.array([])
            out.append({
                "trajectory": name,
                "model_id": result.model_id,
                "max_drift_mm": float(drift_arr.max()) if drift_arr.size else 0.0,
                "mean_drift_mm": float(drift_arr.mean()) if drift_arr.size else 0.0,
                "n_slots_above_threshold": int((drift_arr > drift_threshold_mm).sum()),
                "n_slots_detected": int(drift_arr.size),
            })
        return out

    def _run_contact_generation(self, log_context="generate", allow_last_assignments=False):
        selected_rows = [row for row in range(self.contactTable.rowCount) if self._row_is_selected(row)]
        if not selected_rows:
            raise ValueError("Select at least one trajectory in the Use column.")

        assignments = self._collect_assignments()
        if not assignments["assignments"]:
            if allow_last_assignments and self.lastAssignments.get("assignments"):
                assignments = self.lastAssignments
                self.log(f"[contacts:{log_context}] using last non-empty assignments")
            else:
                raise ValueError("Select electrode models for the selected trajectories.")
        else:
            self.lastAssignments = assignments

        traj_map = self._build_trajectory_map_with_scene_overrides()
        ordered_names = []
        for row in selected_rows:
            item = self.contactTable.item(row, 1)
            if item:
                ordered_names.append(item.text())
        selected_trajectories = [traj_map[name] for name in ordered_names if name in traj_map]

        mode = self._selected_detection_mode()
        if mode == "peak_driven":
            contacts, peak_fit_by_traj, assignments = self._generate_contacts_peak_driven(
                trajectories=selected_trajectories,
                assignments=assignments,
                log_context=log_context,
            )
            self._sync_model_combos_from_assignments(assignments)
            self.lastAssignments = assignments
            self.lastPeakDriftFlags = self._compute_peak_vs_nominal_drift(
                peak_fit_by_traj=peak_fit_by_traj,
                trajectories_by_name=traj_map,
                assignments=assignments,
            )
        else:
            contacts = generate_contacts(selected_trajectories, self.modelsById, assignments)
            self.lastPeakDriftFlags = []

        node_prefix = self.contactsNodeNameEdit.text.strip() or "ROSA_Contacts"
        contact_nodes = self.logic.electrode_scene.create_contacts_fiducials_nodes_by_trajectory(
            contacts,
            node_prefix=node_prefix,
        )
        model_nodes = {}
        if self.createModelsCheck.checked:
            model_nodes = self.logic.electrode_scene.create_electrode_models_by_trajectory(
                contacts=contacts,
                trajectories_by_name=traj_map,
                models_by_id=self.modelsById,
                node_prefix=node_prefix,
            )

        self.lastGeneratedContacts = contacts
        # GT-export becomes available once we have something to save.
        self.saveContactsAsGtButton.setEnabled(bool(contacts))
        assignment_rows = list(assignments.get("assignments", []))
        for row in assignment_rows:
            traj_name = row.get("trajectory", "")
            traj = traj_map.get(traj_name)
            row["trajectory_length_mm"] = 0.0 if traj is None else trajectory_length_mm(traj)
            row["electrode_length_mm"] = electrode_length_mm(self.modelsById.get(row.get("model_id", ""), {}))
            row["source"] = "contacts"

        self._refresh_qc_metrics()
        qc_rows_for_publish = []
        for row in self.lastQCMetricsRows:
            out = dict(row)
            for key in (
                "entry_radial_mm",
                "target_radial_mm",
                "mean_contact_radial_mm",
                "max_contact_radial_mm",
                "rms_contact_radial_mm",
                "angle_deg",
            ):
                if out.get(key) is None:
                    out[key] = "NA"
            qc_rows_for_publish.append(out)
        self.logic.electrode_scene.publish_contacts_outputs(
            contact_nodes_by_traj=contact_nodes,
            model_nodes_by_traj=model_nodes,
            assignment_rows=assignment_rows,
            qc_rows=qc_rows_for_publish,
            workflow_node=self.workflowNode,
        )
        self.log(
            f"[contacts:{log_context}] updated {len(contacts)} points across {len(contact_nodes)} electrode nodes"
        )
        if model_nodes:
            self.log(f"[models:{log_context}] updated {len(model_nodes)} electrode model pairs")
        if self.lastPeakDriftFlags:
            for row in self.lastPeakDriftFlags:
                flag = "⚠ drift>1mm" if row["n_slots_above_threshold"] > 0 else "ok"
                self.log(
                    f"[peak-vs-nominal:{log_context}] {row['trajectory']} "
                    f"model={row['model_id']} max={row['max_drift_mm']:.2f} mm "
                    f"mean={row['mean_drift_mm']:.2f} mm "
                    f"n={row['n_slots_detected']} {flag}"
                )
        if self.lastQCMetricsRows:
            self.log(f"[qc:{log_context}] computed metrics for {len(self.lastQCMetricsRows)} trajectories")
        self._refresh_summary()
        # Auto-apply the trajectory-aligned dual-view layout if the
        # user hasn't already done so. Editing contacts requires both
        # the long-axis view (drag along axis = depth) AND the
        # perpendicular view (drag perpendicular = lateral); without
        # the focus layout the user has no slice plane that's both
        # aligned with the shank and showing the cross-section.
        if not self.logic.layout_service.has_focus_slice_views():
            self._schedule_apply_focus_layout()
            self.log(
                f"[contacts:{log_context}] applied trajectory focus "
                "layout — drag contacts in TrajectoryLong (depth) or "
                "TrajectoryDown (lateral) views; click a row to align"
            )
        self._schedule_follow_selected_trajectory()

    def onGenerateContactsClicked(self):
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                "No trajectories are available in workflow/scene.",
            )
            return
        try:
            self._run_contact_generation(log_context="generate")
        except Exception as exc:
            self.log(f"[contacts] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contacts & Trajectory View", str(exc))

    def onUpdateContactsClicked(self):
        if not self.loadedTrajectories:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                "No trajectories are available in workflow/scene.",
            )
            return
        if not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                "Generate contacts once first, then use update.",
            )
            return
        try:
            self._run_contact_generation(log_context="update")
        except Exception as exc:
            self.log(f"[contacts:update] error: {exc}")
            qt.QMessageBox.critical(slicer.util.mainWindow(), "Contacts & Trajectory View", str(exc))

    # ---- GT-export ----------------------------------------------------

    _GT_TSV_COLUMNS = (
        "subject_id", "channel", "shank", "contact_index",
        "x", "y", "z",
        "coord_source", "snap_status", "move_mm", "peak_value",
        "ct_path", "source_contacts_file", "verified_at_iso",
    )

    def _suggest_subject_id(self):
        """Best-effort subject-ID guess from the loaded postop CT name.

        Dataset CT volumes are named ``<sid>_ct.nii.gz`` so the volume
        node name typically ends in ``_ct``. Strip the suffix; otherwise
        return the volume name verbatim.
        """
        ct_node = self._resolve_postop_ct_node()
        if ct_node is None:
            return ""
        name = (ct_node.GetName() or "").strip()
        if name.endswith("_ct"):
            name = name[:-3]
        return name

    def _resolve_subject_id(self):
        """Prompt the user for a subject ID, pre-filled with the best
        guess from the postop CT name. Returns ``""`` on cancel.
        """
        suggested = self._suggest_subject_id()
        text, ok = qt.QInputDialog.getText(
            slicer.util.mainWindow(),
            "Save Contacts as GT",
            "Subject ID (used as labels_gt/<sid>_contacts_gt.tsv):",
            qt.QLineEdit.Normal,
            suggested,
        )
        if not ok:
            return ""
        return str(text or "").strip()

    def _gt_output_path(self, subject_id):
        """Resolve labels_gt directory under $ROSA_SEEG_DATASET (or the
        configured fallback). Creates the directory on first call.
        """
        dataset_root = os.environ.get(
            "ROSA_SEEG_DATASET",
            "/Users/ammar/Dropbox/thalamus_subjects/seeg_localization",
        )
        gt_dir = os.path.join(dataset_root, "contact_label_dataset", "labels_gt")
        os.makedirs(gt_dir, exist_ok=True)
        return os.path.join(gt_dir, f"{subject_id}_contacts_gt.tsv")

    def _ct_path_for_gt(self):
        ct_node = self._resolve_postop_ct_node()
        if ct_node is None:
            return ""
        storage = ct_node.GetStorageNode()
        if storage is None:
            return ""
        return str(storage.GetFileName() or "")

    def _build_gt_rows(self, subject_id):
        """Walk the current ContactFiducials nodes, pair each control
        point with its auto-generated nominal position from
        ``self.lastGeneratedContacts``, and produce one TSV row per
        contact. ``move_mm`` is the RAS distance from the auto position
        to the user-edited position; the user implicitly verifies every
        contact by clicking save.
        """
        nominal_by_key = {}
        for c in self.lastGeneratedContacts or []:
            traj_name = str(c.get("trajectory") or "")
            try:
                idx = int(c.get("index", 0))
            except (TypeError, ValueError):
                continue
            pos_lps = c.get("position_lps")
            if pos_lps is None:
                continue
            nominal_ras = lps_to_ras_point([float(v) for v in list(pos_lps)])
            nominal_by_key[(traj_name, idx)] = nominal_ras

        ct_path = self._ct_path_for_gt()
        timestamp_iso = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        contact_nodes = self.workflowState.role_nodes(
            "ContactFiducials", workflow_node=self.workflowNode,
        )
        rows = []
        for node in contact_nodes:
            traj_name = str(node.GetAttribute("Rosa.TrajectoryName") or "").strip()
            if not traj_name:
                # Fall back to derived shank from node name; if that's
                # not available either, skip — we can't write a row
                # without a shank.
                node_name = str(node.GetName() or "")
                if node_name.startswith("ROSA_Contacts_"):
                    traj_name = node_name[len("ROSA_Contacts_"):]
            if not traj_name:
                continue
            n_pts = node.GetNumberOfControlPoints()
            for i in range(n_pts):
                pos = [0.0, 0.0, 0.0]
                node.GetNthControlPointPositionWorld(i, pos)
                label = node.GetNthControlPointLabel(i) or f"{traj_name}{i + 1}"
                idx = i + 1
                nominal = nominal_by_key.get((traj_name, idx))
                if nominal is None:
                    move_mm = 0.0
                else:
                    move_mm = math.sqrt(
                        sum((pos[j] - nominal[j]) ** 2 for j in range(3))
                    )
                rows.append({
                    "subject_id": subject_id,
                    "channel": label,
                    "shank": traj_name,
                    "contact_index": idx,
                    "x": f"{pos[0]:.6f}",
                    "y": f"{pos[1]:.6f}",
                    "z": f"{pos[2]:.6f}",
                    "coord_source": "manual",
                    "snap_status": "verified",
                    "move_mm": f"{move_mm:.6f}",
                    "peak_value": "",
                    "ct_path": ct_path,
                    "source_contacts_file": "",
                    "verified_at_iso": timestamp_iso,
                })
        return rows

    @classmethod
    def _write_gt_tsv(cls, path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\t".join(cls._GT_TSV_COLUMNS) + "\n")
            for row in rows:
                f.write(
                    "\t".join(str(row.get(col, "")) for col in cls._GT_TSV_COLUMNS) + "\n"
                )

    def onSaveContactsAsGtClicked(self):
        if not self.lastGeneratedContacts:
            qt.QMessageBox.warning(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                "Generate contacts first, hand-correct the fiducials in "
                "the slice viewers, then click Save Contacts as GT.",
            )
            return
        subject_id = self._resolve_subject_id()
        if not subject_id:
            return
        try:
            rows = self._build_gt_rows(subject_id)
            if not rows:
                qt.QMessageBox.warning(
                    slicer.util.mainWindow(),
                    "Contacts & Trajectory View",
                    "No ContactFiducials nodes found in the workflow. "
                    "Generate contacts first.",
                )
                return
            out_path = self._gt_output_path(subject_id)
            existed = os.path.exists(out_path)
            self._write_gt_tsv(out_path, rows)
        except Exception as exc:
            self.log(f"[contacts:save_gt] error: {exc}")
            qt.QMessageBox.critical(
                slicer.util.mainWindow(),
                "Contacts & Trajectory View",
                f"Could not write GT TSV: {exc}",
            )
            return
        verb = "Overwrote" if existed else "Wrote"
        self.log(
            f"[contacts:save_gt] {verb.lower()} {len(rows)} contacts → {out_path}"
        )
        qt.QMessageBox.information(
            slicer.util.mainWindow(),
            "Contacts & Trajectory View",
            f"{verb} {len(rows)} contacts to {out_path}",
        )

    def _selected_trajectory_name_from_table(self):
        row = self.contactTable.currentRow()
        if row < 0:
            selected = self.contactTable.selectedItems()
            if selected:
                row = selected[0].row()
        if row < 0 and self.contactTable.rowCount > 0:
            row = 0
        if row < 0:
            return ""
        item = self.contactTable.item(row, 1)
        return (item.text() if item else "").strip()

    def _selected_trajectory(self):
        traj_name = self._selected_trajectory_name_from_table()
        if not traj_name:
            return None
        traj_map = self._build_trajectory_map_with_scene_overrides()
        return traj_map.get(traj_name)

    def _highlight_selected_trajectory(self):
        scope_ids = [traj.get("node_id", "") for traj in self.loadedTrajectories if traj.get("node_id")]
        selected = self._selected_trajectory()
        self.logic.focus_controller.focus_selected(
            trajectory=selected,
            scope_node_ids=scope_ids,
            jump_cardinal=False,
            align_focus_views=False,
        )

    def _resolve_primary_volumes(self):
        base_nodes = self.workflowState.role_nodes("BaseVolume", workflow_node=self.workflowNode)
        post_nodes = self.workflowState.role_nodes("PostopCT", workflow_node=self.workflowNode)
        base_node = base_nodes[0] if base_nodes else None
        post_node = post_nodes[0] if post_nodes else None
        return base_node, post_node

    def _apply_focus_slice_layers(self):
        base_node, post_node = self._resolve_primary_volumes()
        if post_node is not None and base_node is not None:
            bg_node, fg_node, fg_opacity = post_node, base_node, 0.5
        elif post_node is not None:
            bg_node, fg_node, fg_opacity = post_node, None, 0.0
        elif base_node is not None:
            bg_node, fg_node, fg_opacity = base_node, None, 0.0
        else:
            bg_node, fg_node, fg_opacity = None, None, 0.0

        for view_name in (
            "Red",
            "Yellow",
            "Green",
            self.logic.layout_service.TRAJECTORY_LONG_VIEW,
            self.logic.layout_service.TRAJECTORY_DOWN_VIEW,
        ):
            self.logic.electrode_scene.set_slice_view_layers(
                slice_view=view_name,
                background_node=bg_node,
                foreground_node=fg_node,
                foreground_opacity=fg_opacity,
            )

    def _align_focus_views_for_selected(self, align_trajectory_views=True):
        trajectory = self._selected_trajectory()
        scope_ids = [traj.get("node_id", "") for traj in self.loadedTrajectories if traj.get("node_id")]
        try:
            return bool(
                self.logic.focus_controller.focus_selected(
                    trajectory=trajectory,
                    scope_node_ids=scope_ids,
                    jump_cardinal=True,
                    align_focus_views=bool(align_trajectory_views),
                    focus="entry",
                )
            )
        except Exception as exc:
            self.log(f"[focus] alignment fallback: {exc}")
            return False

    def _follow_selected_trajectory_if_enabled(self):
        if self._syncingFocusControls:
            return
        self._highlight_selected_trajectory()
        if not bool(self.autoFollowTrajectoryCheck.checked):
            return
        align_trajectory_views = bool(self.logic.layout_service.has_focus_slice_views())
        self._align_focus_views_for_selected(align_trajectory_views=align_trajectory_views)

    def _schedule_follow_selected_trajectory(self):
        if self._pendingFollow:
            return
        self._pendingFollow = True
        qt.QTimer.singleShot(0, self._run_scheduled_follow)

    def _run_scheduled_follow(self):
        self._pendingFollow = False
        self._follow_selected_trajectory_if_enabled()

    def _schedule_apply_focus_layout(self):
        if self._pendingFocusLayoutApply:
            return
        self._pendingFocusLayoutApply = True
        qt.QTimer.singleShot(0, self._run_scheduled_apply_focus_layout)

    def _run_scheduled_apply_focus_layout(self):
        self._pendingFocusLayoutApply = False
        self.onApplyFocusLayoutClicked()

    def onContactTableCellClicked(self, _row, _col):
        self._schedule_follow_selected_trajectory()
        self._apply_isolation_if_enabled()

    def onContactTableCurrentCellChanged(self, _currentRow, _currentColumn, _previousRow, _previousColumn):
        self._schedule_follow_selected_trajectory()
        self._apply_isolation_if_enabled()

    def onIsolateSelectedToggled(self, checked):
        if not bool(checked):
            # Restore everything — empty target set means show-all.
            try:
                self.logic.electrode_scene.apply_trajectory_isolation(set())
            except Exception:
                pass
            return
        self._apply_isolation_if_enabled()



    def _apply_isolation_if_enabled(self):
        if not getattr(self, "isolateSelectedCheck", None):
            return
        if not bool(self.isolateSelectedCheck.checked):
            return
        names = self._selected_trajectory_names_for_isolation()
        try:
            self.logic.electrode_scene.apply_trajectory_isolation(names)
        except Exception:
            pass

    def _selected_trajectory_names_for_isolation(self):
        """Return the set of trajectory names currently selected in
        the contact table. Falls back to the current row when the
        selection model is in single-selection mode (default).
        """
        names = set()
        try:
            sel = self.contactTable.selectionModel()
            for idx in sel.selectedRows() if sel else []:
                row = idx.row()
                item = self.contactTable.item(row, 1)
                if item:
                    txt = (item.text() or "").strip()
                    if txt:
                        names.add(txt)
            if not names:
                row = self.contactTable.currentRow()
                if row >= 0:
                    item = self.contactTable.item(row, 1)
                    if item:
                        txt = (item.text() or "").strip()
                        if txt:
                            names.add(txt)
        except Exception:
            pass
        return names

    def onAutoFollowToggled(self, checked):
        self._set_workflow_param_bool("TrajectoryFocusAutoFollow", bool(checked))
        self._highlight_selected_trajectory()
        if bool(checked):
            self._schedule_follow_selected_trajectory()

    def onApplyFocusVolumesClicked(self):
        base_node = self.focusBaseSelector.currentNode()
        postop_node = self.focusPostopSelector.currentNode()
        if base_node is not None:
            self.logic.workflow_publish.set_default_role("BaseVolume", base_node, workflow_node=self.workflowNode)
        if postop_node is not None:
            self.logic.workflow_publish.set_default_role("PostopCT", postop_node, workflow_node=self.workflowNode)
        else:
            self.logic.workflow_publish.set_default_role("PostopCT", None, workflow_node=self.workflowNode)
        self._apply_focus_slice_layers()
        self._schedule_follow_selected_trajectory()
        self.log(
            "[focus] set defaults: "
            f"base={(base_node.GetName() if base_node else 'unset')}, "
            f"postop={(postop_node.GetName() if postop_node else 'unset')}"
        )

    def onApplyFocusLayoutClicked(self):
        self.logic.layout_service.sanitize_focus_layout_state()
        if not self.logic.layout_service.apply_trajectory_focus_layout():
            self.log("[focus] failed to apply trajectory focus layout")
            return
        self._apply_focus_slice_layers()
        self._schedule_follow_selected_trajectory()
        self.log("[focus] trajectory focus layout applied")

    def onResetFocusLayoutClicked(self):
        if self.logic.layout_service.restore_previous_layout():
            self.log("[focus] restored previous layout")

    def onAlignSliceClicked(self):
        # Deprecated manual align action retained for backward compatibility.
        self._follow_selected_trajectory_if_enabled()

    def onShowPlannedToggled(self, checked):
        self.logic.electrode_scene.set_planned_trajectory_visibility(bool(checked))

    def onShowModelsToggled(self, checked):
        """Toggle visibility of every published electrode-model node
        (cylinders for shaft + contact tubes). Hiding them makes the
        contact fiducial glyphs clickable in slice viewers without
        deleting the cylinders from the scene.
        """
        visible = bool(checked)
        for role in (
            "ElectrodeShaftModelNodes",
            "ElectrodeContactModelNodes",
        ):
            for node in self.workflowState.role_nodes(role, workflow_node=self.workflowNode):
                display = node.GetDisplayNode()
                if display is not None:
                    display.SetVisibility(visible)
                    if hasattr(display, "SetVisibility2D"):
                        display.SetVisibility2D(visible)
                    if hasattr(display, "SetVisibility3D"):
                        display.SetVisibility3D(visible)


class ContactsTrajectoryViewLogic(ScriptedLoadableModuleLogic):
    """Logic for extracting trajectories and publishing contact scene outputs."""

    def __init__(self):
        super().__init__()
        self.workflow_state = WorkflowState()
        self.workflow_publish = WorkflowPublisher(self.workflow_state)
        self.trajectory_scene = TrajectorySceneService()
        self.layout_service = LayoutService()
        self.electrode_scene = ElectrodeSceneService(
            workflow_state=self.workflow_state,
            workflow_publish=self.workflow_publish,
        )
        self.focus_controller = TrajectoryFocusController(
            trajectory_scene=self.trajectory_scene,
            electrode_scene=self.electrode_scene,
            layout_service=self.layout_service,
        )

    def _collect_trajectories_from_role(self, role, workflow_node=None):
        """Return trajectories from one workflow role."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        trajectories = []
        role_nodes = self.workflow_state.role_nodes(role, workflow_node=wf)
        for node in role_nodes:
            traj = self.trajectory_scene.trajectory_from_line_node("", node)
            if traj is not None:
                trajectories.append(traj)
        trajectories.sort(key=lambda item: item.get("name", ""))
        return trajectories

    def collect_trajectories_by_source(self, source_key="working", workflow_node=None):
        """Return trajectories for one selected source group."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        source = str(source_key or "working").strip().lower()

        if source == "working":
            trajectories = self._collect_trajectories_from_role("WorkingTrajectoryLines", workflow_node=wf)
            if trajectories:
                return trajectories
            rows = self.trajectory_scene.collect_working_trajectory_rows(
                groups=["imported_rosa", "imported_external", "manual", "guided_fit", "auto_fit"]
            )
            fallback = []
            for row in rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is None:
                    continue
                traj = self.trajectory_scene.trajectory_from_line_node(row["name"], node)
                if traj is not None:
                    fallback.append(traj)
            fallback.sort(key=lambda item: item.get("name", ""))
            return fallback

        if source == "imported_rosa":
            return self._collect_trajectories_from_role("ImportedTrajectoryLines", workflow_node=wf)
        if source == "guided_fit":
            return self._collect_trajectories_from_role("GuidedFitTrajectoryLines", workflow_node=wf)
        if source == "auto_fit":
            return self._collect_trajectories_from_role("AutoFitTrajectoryLines", workflow_node=wf)
        if source == "imported_external":
            return self._collect_trajectories_from_role("ImportedExternalTrajectoryLines", workflow_node=wf)
        if source == "planned_rosa":
            return self._collect_trajectories_from_role("PlannedTrajectoryLines", workflow_node=wf)
        if source == "manual":
            rows = self.trajectory_scene.collect_working_trajectory_rows(groups=["manual"])
            nodes = []
            trajectories = []
            for row in rows:
                node = slicer.mrmlScene.GetNodeByID(row["node_id"])
                if node is None:
                    continue
                self.trajectory_scene.set_trajectory_metadata(
                    node=node,
                    trajectory_name=row["name"],
                    group="manual",
                    origin=node.GetAttribute("Rosa.TrajectoryOrigin") or "manual",
                )
                nodes.append(node)
                traj = self.trajectory_scene.trajectory_from_line_node(row["name"], node)
                if traj is not None:
                    trajectories.append(traj)
            self.workflow_publish.publish_nodes(
                role="ManualTrajectoryLines",
                nodes=nodes,
                source="manual",
                space_name="ROSA_BASE",
                workflow_node=wf,
            )
            self.trajectory_scene.place_trajectory_nodes_in_hierarchy(
                context_id=self.workflow_state.context_id(workflow_node=wf),
                nodes=nodes,
            )
            trajectories.sort(key=lambda item: item.get("name", ""))
            return trajectories

        return []

    def collect_planned_trajectory_map(self, workflow_node=None):
        """Return planned trajectories from role if available, else from Plan_* line nodes."""
        wf = workflow_node or self.workflow_state.resolve_or_create_workflow_node()
        planned = {}
        role_nodes = self.workflow_state.role_nodes("PlannedTrajectoryLines", workflow_node=wf)
        for node in role_nodes:
            name = self.trajectory_scene.logical_name_from_node(node) or (node.GetName() or "")
            traj = self.trajectory_scene.trajectory_from_line_node(name, node)
            if traj is not None:
                planned[name] = traj
        if planned:
            return planned
        return self.trajectory_scene.collect_planned_trajectory_map()

    def rename_trajectory(self, node_id, new_name):
        """Rename one working trajectory node (planned trajectories are left unchanged)."""
        node = slicer.mrmlScene.GetNodeByID(str(node_id or ""))
        if node is None:
            return False
        return bool(self.trajectory_scene.rename_trajectory_node(node, new_name))
