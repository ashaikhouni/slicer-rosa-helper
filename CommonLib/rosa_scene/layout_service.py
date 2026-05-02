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
            # Auto-enable TL ↔ TD slice intersections — Slicer 5.4+
            # supports per-slice scoping so R/Y/G stay clean. Lets the
            # user always see where the perpendicular cross-section
            # sits along the long axis without remembering to toggle.
            try:
                self.set_focus_slice_intersection_visibility(True)
            except Exception:
                pass
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

    def set_focus_slice_intersection_visibility(self, visible: bool) -> bool:
        """Toggle slice-intersection rendering scoped to the
        trajectory-focus slice ports (TrajectoryLong = blue,
        TrajectoryDown = purple).

        Modern-Slicer scoped path: master toggle via
        `vtkMRMLApplicationLogic.SetIntersectingSlicesEnabled(op, bool)`
        with the `IntersectingSlicesOperation` enum, plus per-slice
        flags via `vtkMRMLSliceDisplayNode.SetIntersectingSlicesVisibility`
        (accessed through the slice LOGIC, not the slice node).
        Standard R/Y/G ports get the per-slice flag set OFF so their
        planes don't render cutting lines in TL/TD; TL/TD get it ON
        so they render each other.

        Falls back to the deprecated `SliceCompositeNode.SetSliceIntersectionVisibility`
        only when the modern display-node API isn't reachable on the
        running build.
        """
        lm = slicer.app.layoutManager()
        if lm is None:
            return False
        # Master toggle (Slicer 5.4+: 2-arg signature).
        try:
            app_logic = slicer.app.applicationLogic()
            cls = type(app_logic) if app_logic is not None else None
            if app_logic is not None and cls is not None:
                op_vis = getattr(cls, "IntersectingSlicesVisibility", None)
                if op_vis is not None and hasattr(
                    app_logic, "SetIntersectingSlicesEnabled"
                ):
                    app_logic.SetIntersectingSlicesEnabled(op_vis, bool(visible))
                if visible:
                    op_inter = getattr(cls, "IntersectingSlicesInteractive", None)
                    if op_inter is not None and hasattr(
                        app_logic, "SetIntersectingSlicesEnabled"
                    ):
                        app_logic.SetIntersectingSlicesEnabled(op_inter, True)
        except Exception:
            pass
        # Per-slice flags. Trajectory ports ON (when enabling); R/Y/G
        # explicitly OFF so they don't render cutting lines into TL/TD.
        focus_tags = (self.TRAJECTORY_LONG_VIEW, self.TRAJECTORY_DOWN_VIEW)
        all_tags = ("Red", "Yellow", "Green") + focus_tags
        applied = 0
        for tag in all_tags:
            want_on = (tag in focus_tags) and bool(visible)
            widget = lm.sliceWidget(tag)
            if widget is None:
                continue
            try:
                slice_logic = (
                    widget.sliceLogic() if hasattr(widget, "sliceLogic") else None
                )
                display_node = (
                    slice_logic.GetSliceDisplayNode()
                    if slice_logic is not None
                    and hasattr(slice_logic, "GetSliceDisplayNode")
                    else None
                )
                if display_node is not None and hasattr(
                    display_node, "SetIntersectingSlicesVisibility"
                ):
                    display_node.SetIntersectingSlicesVisibility(
                        1 if want_on else 0,
                    )
                    applied += 1
                    continue
                # Legacy composite-node fallback. Emits a deprecation
                # warning but functions on builds without the modern
                # display-node API.
                composite = widget.mrmlSliceCompositeNode()
                if composite is not None and hasattr(
                    composite, "SetSliceIntersectionVisibility"
                ):
                    composite.SetSliceIntersectionVisibility(
                        1 if want_on else 0,
                    )
                    applied += 1
            except Exception:
                continue
        return applied > 0

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
        # Turn off TL/TD intersections when leaving the focus layout —
        # the TL/TD slice ports won't be visible anyway, but clearing
        # their per-slice flags keeps the scene state tidy.
        try:
            self.set_focus_slice_intersection_visibility(False)
        except Exception:
            pass
        return True
