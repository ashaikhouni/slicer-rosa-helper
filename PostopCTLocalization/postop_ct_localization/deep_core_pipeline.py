"""Staged Deep Core pipeline.

This module is the typed internal boundary for Deep Core.  The pipeline
uses composed stage classes (``MaskStage``, ``SupportStage``,
``ProposalStage``) that communicate through typed dataclass outputs.

The legacy ``DeepCoreMaskResult``/``DeepCoreSupportResult``/
``DeepCoreProposalResult`` wrappers are kept so that ``deep_core_widget.py``
callers do not need to change immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .deep_core_annulus import AnnulusSampler
from .deep_core_config import DeepCoreConfig, deep_core_default_config
from .deep_core_stages import (
    MaskStage,
    MaskStageOutput,
    ProposalStage,
    ProposalStageOutput,
    SupportStage,
    SupportStageOutput,
)


# ---------------------------------------------------------------------------
# Legacy result wrappers (backward compat for widget/viz code)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DeepCoreMaskResult:
    """Legacy wrapper around MaskStageOutput."""

    volume_node: Any
    volume_node_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    _typed: MaskStageOutput | None = field(default=None, repr=False)

    @property
    def typed(self) -> MaskStageOutput | None:
        return self._typed

    @property
    def stats(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._typed.stats
        return dict(self.payload.get("stats") or {})

    @property
    def combined_payload(self) -> dict[str, Any]:
        return dict(self.payload or {})

    def to_legacy_payload(self) -> dict[str, Any]:
        return self.combined_payload


@dataclass(frozen=True)
class DeepCoreSupportResult:
    """Legacy wrapper around SupportStageOutput."""

    mask_result: DeepCoreMaskResult
    payload: dict[str, Any] = field(default_factory=dict)
    _typed: SupportStageOutput | None = field(default=None, repr=False)

    @property
    def typed(self) -> SupportStageOutput | None:
        return self._typed

    @property
    def volume_node(self) -> Any:
        return self.mask_result.volume_node

    @property
    def volume_node_id(self) -> str:
        return self.mask_result.volume_node_id

    @property
    def stats(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._typed.stats
        return dict(self.payload.get("stats") or {})

    @property
    def combined_payload(self):
        """Return the typed output directly when available — it supports
        ``.get()`` so callers that do ``combined_payload.get("key")``
        work without materialising a dict."""
        if self._typed is not None:
            return self._typed
        merged = dict(self.mask_result.combined_payload)
        merged.update(dict(self.payload or {}))
        return merged

    def to_legacy_payload(self) -> dict[str, Any]:
        if self._typed is not None:
            return self._typed.to_payload()
        merged = dict(self.mask_result.combined_payload)
        merged.update(dict(self.payload or {}))
        return merged


@dataclass(frozen=True)
class DeepCoreProposalResult:
    """Legacy wrapper around ProposalStageOutput."""

    support_result: DeepCoreSupportResult
    payload: dict[str, Any] = field(default_factory=dict)
    _typed: ProposalStageOutput | None = field(default=None, repr=False)

    @property
    def typed(self) -> ProposalStageOutput | None:
        return self._typed

    @property
    def volume_node(self) -> Any:
        return self.support_result.volume_node

    @property
    def volume_node_id(self) -> str:
        return self.support_result.volume_node_id

    @property
    def combined_payload(self) -> dict[str, Any]:
        if self._typed is not None:
            # Merge support typed output with proposal payload — only
            # materialise the proposal dict (small), not the full support.
            out = dict(self.payload or {})
            return out
        merged = dict(self.support_result.combined_payload)
        merged.update(dict(self.payload or {}))
        return merged

    def to_legacy_payload(self) -> dict[str, Any]:
        if self._typed is not None:
            out = self._typed.support.to_payload()
            out.update(self._typed.to_payload())
            return out
        merged = dict(self.support_result.combined_payload)
        merged.update(dict(self.payload or {}))
        return merged


# ---------------------------------------------------------------------------
# Composed pipeline
# ---------------------------------------------------------------------------

class DeepCorePipeline:
    """Orchestrator that runs Deep Core stages via composition."""

    def __init__(self, volume_accessor, host_logic=None):
        self._vol = volume_accessor
        self._host_logic = host_logic
        self._annulus = AnnulusSampler(volume_accessor)
        self._mask_stage = MaskStage(volume_accessor)
        self._support_stage = SupportStage(volume_accessor, self._annulus)
        self._proposal_stage = ProposalStage(volume_accessor, self._annulus)

    def run_debug(
        self,
        volume_node,
        config=None,
        show_support_diagnostics=True,
    ) -> DeepCoreSupportResult:
        """Run mask and support stages and return typed stage results."""

        cfg = config if config is not None else deep_core_default_config()

        # --- mask stage ---
        mask_output = self._mask_stage.run(volume_node, config=cfg.mask)

        # Create debug viz volumes if accessor supports it
        smooth_node = None
        distance_node = None
        if hasattr(self._vol, "update_scalar_volume"):
            vol_name = (
                volume_node.GetName()
                if hasattr(volume_node, "GetName")
                else "volume"
            )
            smooth_node = self._vol.update_scalar_volume(
                reference_volume_node=volume_node,
                node_name=f"{vol_name}_HullSmooth",
                array_kji=mask_output.smoothed_hull_kji,
            )
            distance_node = self._vol.update_scalar_volume(
                reference_volume_node=volume_node,
                node_name=f"{vol_name}_HeadDistanceMm",
                array_kji=mask_output.head_distance_map_kji,
            )

        # Build legacy mask payload (adds viz node refs)
        mask_payload = mask_output.to_payload()
        mask_payload["volume_node_id"] = (
            str(volume_node.GetID())
            if hasattr(volume_node, "GetID")
            else ""
        )
        mask_payload["smoothed_hull_volume_node"] = smooth_node
        mask_payload["head_distance_volume_node"] = distance_node

        mask_result = DeepCoreMaskResult(
            volume_node=volume_node,
            volume_node_id=mask_payload["volume_node_id"],
            payload=mask_payload,
            _typed=mask_output,
        )

        # --- support stage ---
        support_output = self._support_stage.run(
            volume_node=volume_node,
            mask=mask_output,
            support_config=cfg.support,
            annulus_config=cfg.annulus,
            internal_config=cfg.internal,
            show_support_diagnostics=bool(show_support_diagnostics),
        )

        # Build legacy support payload (support-only keys, not mask keys)
        support_payload = {
            k: v
            for k, v in support_output.to_payload().items()
            if k not in mask_payload
        }
        support_payload["stats"] = dict(support_output.stats)

        return DeepCoreSupportResult(
            mask_result=mask_result,
            payload=support_payload,
            _typed=support_output,
        )

    def run_proposals(
        self,
        volume_node,
        config=None,
        debug_result=None,
    ) -> DeepCoreProposalResult:
        """Run support->proposal stages and return typed proposal outputs."""

        cfg = config if config is not None else deep_core_default_config()

        if debug_result is not None and debug_result.typed is not None:
            support_output = debug_result.typed
        elif debug_result is not None:
            # Legacy DeepCoreSupportResult without typed field —
            # fall back to running support stage fresh.
            support_result = debug_result
            support_output = None
        else:
            support_result = None
            support_output = None

        if support_output is None:
            # Need to run mask + support stages
            mask_output = self._mask_stage.run(volume_node, config=cfg.mask)
            support_output = self._support_stage.run(
                volume_node=volume_node,
                mask=mask_output,
                support_config=cfg.support,
                annulus_config=cfg.annulus,
                internal_config=cfg.internal,
                show_support_diagnostics=False,
            )

        proposal_output = self._proposal_stage.run(
            volume_node=volume_node,
            support=support_output,
            config=cfg,
        )

        # Wrap in legacy types for backward compat
        if debug_result is None:
            mask_payload = support_output.mask.to_payload()
            mask_result = DeepCoreMaskResult(
                volume_node=volume_node,
                volume_node_id=(
                    str(volume_node.GetID())
                    if hasattr(volume_node, "GetID")
                    else ""
                ),
                payload=mask_payload,
                _typed=support_output.mask,
            )
            support_payload = {
                k: v
                for k, v in support_output.to_payload().items()
                if k not in mask_payload
            }
            support_payload["stats"] = dict(support_output.stats)
            debug_result = DeepCoreSupportResult(
                mask_result=mask_result,
                payload=support_payload,
                _typed=support_output,
            )

        return DeepCoreProposalResult(
            support_result=debug_result,
            payload=proposal_output.to_payload(),
            _typed=proposal_output,
        )
