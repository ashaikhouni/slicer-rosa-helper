"""Provider registry for atlas contact-labeling sources."""

from __future__ import annotations

from .atlas_providers import ThomasSegmentationAtlasProvider, VolumeLabelAtlasProvider


class AtlasProviderRegistry:
    """Build atlas-provider instances from workflow-selected nodes."""

    def __init__(self, utils):
        self.utils = utils
        self._provider_cache = {}

    @staticmethod
    def _transform_signature(node):
        if node is None:
            return ("", 0)
        return (node.GetID(), int(node.GetMTime()))

    def _node_signature(self, node):
        if node is None:
            return ("", 0, ("", 0))
        return (
            node.GetID(),
            int(node.GetMTime()),
            self._transform_signature(node.GetParentTransformNode()),
        )

    def _volume_provider_key(self, source_id, volume_node):
        return ("volume", str(source_id), self._node_signature(volume_node))

    def _thomas_provider_key(self, segmentation_nodes, reference_volume_node):
        seg_signatures = sorted(self._node_signature(node) for node in (segmentation_nodes or []) if node is not None)
        return ("thomas", tuple(seg_signatures), self._node_signature(reference_volume_node))

    def _get_or_create(self, key, builder):
        provider = self._provider_cache.get(key)
        if provider is None:
            provider = builder()
            self._provider_cache[key] = provider
        return provider

    def build_default_providers(
        self,
        freesurfer_volume_node=None,
        thomas_segmentation_nodes=None,
        wm_volume_node=None,
        reference_volume_node=None,
    ):
        active_keys = set()
        providers = {}
        if freesurfer_volume_node is not None:
            key = self._volume_provider_key("freesurfer", freesurfer_volume_node)
            active_keys.add(key)
            providers["freesurfer"] = self._get_or_create(
                key,
                lambda: VolumeLabelAtlasProvider(
                    source_id="freesurfer",
                    display_name="FreeSurfer",
                    volume_node=freesurfer_volume_node,
                    utils=self.utils,
                ),
            )
        if thomas_segmentation_nodes:
            key = self._thomas_provider_key(thomas_segmentation_nodes, reference_volume_node)
            active_keys.add(key)
            providers["thomas"] = self._get_or_create(
                key,
                lambda: ThomasSegmentationAtlasProvider(
                    segmentation_nodes=thomas_segmentation_nodes,
                    reference_volume_node=reference_volume_node,
                    utils=self.utils,
                ),
            )
        if wm_volume_node is not None:
            key = self._volume_provider_key("wm", wm_volume_node)
            active_keys.add(key)
            providers["wm"] = self._get_or_create(
                key,
                lambda: VolumeLabelAtlasProvider(
                    source_id="wm",
                    display_name="WhiteMatter",
                    volume_node=wm_volume_node,
                    utils=self.utils,
                ),
            )

        # Keep cache bounded to currently active provider signatures.
        if self._provider_cache:
            self._provider_cache = {key: value for key, value in self._provider_cache.items() if key in active_keys}
        return providers
