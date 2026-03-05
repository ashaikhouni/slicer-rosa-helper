"""Shared Slicer layout utilities for trajectory-focused review views."""

from __future__ import annotations

from __main__ import slicer


class LayoutService:
    """Manage custom layout definitions and layout switching."""

    TRAJECTORY_FOCUS_LAYOUT_ID = 862
    LEGACY_FOCUS_LAYOUT_IDS = (761,)
    TRAJECTORY_LONG_VIEW = "TrajectoryLong"
    TRAJECTORY_DOWN_VIEW = "TrajectoryDown"

    # DTD-safe nested layout:
    # top row: Red / Yellow / Green
    # bottom row: TrajectoryLong / TrajectoryDown / 3D
    _TRAJECTORY_FOCUS_LAYOUT_XML = f"""
<layout type="vertical" split="true">
  <item splitSize="500">
    <layout type="horizontal">
      <item>
        <view class="vtkMRMLSliceNode" singletontag="Red">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">R</property>
          <property name="viewcolor" action="default">#F34A33</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLSliceNode" singletontag="Yellow">
          <property name="orientation" action="default">Sagittal</property>
          <property name="viewlabel" action="default">Y</property>
          <property name="viewcolor" action="default">#EDD54C</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLSliceNode" singletontag="Green">
          <property name="orientation" action="default">Coronal</property>
          <property name="viewlabel" action="default">G</property>
          <property name="viewcolor" action="default">#6EB04B</property>
        </view>
      </item>
    </layout>
  </item>
  <item splitSize="500">
    <layout type="horizontal">
      <item>
        <view class="vtkMRMLSliceNode" singletontag="{TRAJECTORY_LONG_VIEW}">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">TL</property>
          <property name="viewcolor" action="default">#33A1FF</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLSliceNode" singletontag="{TRAJECTORY_DOWN_VIEW}">
          <property name="orientation" action="default">Axial</property>
          <property name="viewlabel" action="default">TD</property>
          <property name="viewcolor" action="default">#8E66FF</property>
        </view>
      </item>
      <item>
        <view class="vtkMRMLViewNode" singletontag="1">
          <property name="viewlabel" action="default">1</property>
        </view>
      </item>
    </layout>
  </item>
</layout>
"""

    def __init__(self):
        """Track layout transitions so modules can restore user arrangement."""
        # Persisted only for current Slicer session/module lifetime.
        self._previous_layout_id: int | None = None

    def _layout_node(self):
        """Return the active ``vtkMRMLLayoutNode`` or ``None`` when unavailable."""
        lm = slicer.app.layoutManager()
        if lm is None:
            return None
        logic = lm.layoutLogic()
        if logic is None:
            return None
        return logic.GetLayoutNode()

    def ensure_trajectory_focus_layout(self) -> bool:
        """Register trajectory-focus layout XML if not already present."""
        layout_node = self._layout_node()
        if layout_node is None:
            return False
        # Use AddLayoutDescription consistently; SetLayoutDescription can be unreliable
        # across Slicer builds for custom IDs.
        layout_node.AddLayoutDescription(
            self.TRAJECTORY_FOCUS_LAYOUT_ID,
            self._TRAJECTORY_FOCUS_LAYOUT_XML,
        )
        desc = layout_node.GetLayoutDescription(self.TRAJECTORY_FOCUS_LAYOUT_ID)
        return bool(desc)

    def sanitize_focus_layout_state(self) -> bool:
        """Ensure focus layout is registered before any module uses or restores it."""
        layout_node = self._layout_node()
        if layout_node is None:
            return False
        # Important: register custom layout before querying current arrangement.
        # Querying current arrangement first can emit "Can't find layout:<id>"
        # if a previous session left the custom id active but undefined.
        if not self.ensure_trajectory_focus_layout():
            return False
        return True

    def apply_trajectory_focus_layout(self) -> bool:
        """Switch to trajectory-focus layout and remember prior layout id."""
        layout_node = self._layout_node()
        if layout_node is None:
            return False
        if not self.ensure_trajectory_focus_layout():
            layout_node.SetViewArrangement(int(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView))
            return False
        current = int(layout_node.GetViewArrangement())
        if current != self.TRAJECTORY_FOCUS_LAYOUT_ID:
            self._previous_layout_id = current
        try:
            layout_node.SetViewArrangement(self.TRAJECTORY_FOCUS_LAYOUT_ID)
            return True
        except Exception:
            # Fail safe to FourUp if custom layout fails.
            layout_node.SetViewArrangement(int(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView))
            return False

    def is_focus_layout_active(self) -> bool:
        """Return True when current arrangement is trajectory-focus layout."""
        layout_node = self._layout_node()
        if layout_node is None:
            return False
        self.ensure_trajectory_focus_layout()
        return int(layout_node.GetViewArrangement()) == self.TRAJECTORY_FOCUS_LAYOUT_ID

    def has_focus_slice_views(self) -> bool:
        """Return True when both custom trajectory slice widgets are currently available."""
        lm = slicer.app.layoutManager()
        if lm is None:
            return False
        return (lm.sliceWidget(self.TRAJECTORY_LONG_VIEW) is not None) and (
            lm.sliceWidget(self.TRAJECTORY_DOWN_VIEW) is not None
        )

    def restore_previous_layout(self) -> bool:
        """Restore previously active layout, or fallback to FourUp."""
        layout_node = self._layout_node()
        if layout_node is None:
            return False
        target = (
            int(self._previous_layout_id)
            if self._previous_layout_id is not None
            else int(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        )
        layout_node.SetViewArrangement(target)
        return True
