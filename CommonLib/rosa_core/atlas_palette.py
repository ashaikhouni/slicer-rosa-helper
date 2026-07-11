"""Meaningful per-atlas region colors for surface / label rendering.

Resolves a ``{label_id: (r, g, b)}`` palette for an atlas, in priority order:

  1. **publisher LUT** — the atlas ships its own RGB color table (e.g. the
     MIAL thalamic atlas's ``Thalamic_Nuclei-ColorLUT.txt``); use it verbatim.
  2. **network palette** — Yeo/Schaefer parcels grouped by functional network
     (the network is encoded in the ROI name, e.g. ``17Networks_LH_VisCent_…``),
     so same-network parcels share a hue family instead of 400 random hues.
  3. **FreeSurfer palette** — labels whose ids are FreeSurfer-compatible (an
     ``aparc+aseg`` or a Mindboggle-id atlas) map through the FreeSurfer LUT.
  4. **golden-ratio fallback** — a stable, well-spread distinct hue per id, for
     an atlas with no structure and no shipped colors.

The resolver is **pure**: the caller supplies any already-parsed publisher /
FreeSurfer LUTs, so this module makes no assumptions about resource paths and
stays trivially testable. ``parse_color_lut`` is provided for the common
FreeSurfer-style ``id name R G B [A]`` file.
"""
from __future__ import annotations

import colorsys
import re
from typing import Dict, Optional, Tuple

RGB = Tuple[int, int, int]


def golden_hue(label: int) -> RGB:
    """A stable, well-spread color for an integer id (golden-ratio hue), so the
    same label reads the same color across cases without an atlas color table."""
    h = (int(label) * 0.6180339887498949) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def parse_color_lut(path) -> Dict[int, RGB]:
    """Parse a FreeSurfer-style color LUT: ``id name R G B [A]`` per line,
    whitespace-separated, ``#`` comments and blank lines skipped. Returns
    ``{id: (r, g, b)}`` (alpha dropped)."""
    out: Dict[int, RGB] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 5 and p[0].lstrip("-").isdigit():
                try:
                    out[int(p[0])] = (int(p[2]), int(p[3]), int(p[4]))
                except ValueError:
                    continue
    return out


# "17Networks_LH_VisCent_Striate_1" / "7Networks_RH_Default_2" → the network token
_NET_RE = re.compile(r"\d+Networks_(?:LH|RH)_([A-Za-z]+)")


def _network_of(name: str) -> Optional[str]:
    m = _NET_RE.search(name or "")
    return m.group(1) if m else None


def is_network_atlas(label_names: Dict[int, str]) -> bool:
    """True if the ROI names look like a Yeo/Schaefer network parcellation
    (a majority carry a ``…Networks_LH/RH_<Network>…`` token)."""
    if not label_names:
        return False
    hits = sum(1 for n in label_names.values() if _network_of(n))
    return hits >= max(2, 0.5 * len(label_names))


def network_palette(label_names: Dict[int, str]) -> Dict[int, RGB]:
    """Color each parcel by its Yeo network: one evenly-spaced base hue per
    network, with a small per-parcel value ripple so same-network parcels form a
    distinguishable family rather than one flat block. Deterministic."""
    nets = sorted({_network_of(n) for n in label_names.values() if _network_of(n)})
    base = {net: i / max(1, len(nets)) for i, net in enumerate(nets)}
    seen: Dict[str, int] = {}
    out: Dict[int, RGB] = {}
    for lid in sorted(label_names):
        net = _network_of(label_names[lid])
        if net is None:
            out[lid] = golden_hue(lid)
            continue
        k = seen.get(net, 0)
        seen[net] = k + 1
        v = 0.80 + 0.18 * ((k % 3) - 1)          # 0.62 / 0.80 / 0.98 ripple
        r, g, b = colorsys.hsv_to_rgb(base[net], 0.62, max(0.30, min(1.0, v)))
        out[lid] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def build_atlas_palette(
    label_names: Dict[int, str],
    *,
    publisher_lut: Optional[Dict[int, RGB]] = None,
    freesurfer_lut: Optional[Dict[int, RGB]] = None,
) -> Dict[int, RGB]:
    """Return ``{id: (r, g, b)}`` for the atlas's labels (tiers in module docstring).

    ``publisher_lut`` / ``freesurfer_lut`` are already-parsed ``{id: (r,g,b)}``
    maps the caller supplies when the atlas ships colors / uses FreeSurfer ids.
    Labels a chosen source doesn't cover fall back to a golden-ratio hue, so the
    result always covers every id in ``label_names``.
    """
    names = {int(k): v for k, v in (label_names or {}).items()}
    if not names:
        return {}

    # 1. publisher LUT (verbatim colors, golden fill for any gaps)
    if publisher_lut:
        pl = {int(k): tuple(v) for k, v in publisher_lut.items()}
        return {lid: pl.get(lid, golden_hue(lid)) for lid in names}

    # 2. Yeo / Schaefer network coloring
    if is_network_atlas(names):
        return network_palette(names)

    # 3. FreeSurfer-compatible ids (aparc+aseg, Mindboggle) — only if they match
    if freesurfer_lut:
        fs = {int(k): tuple(v) for k, v in freesurfer_lut.items()}
        matched = [lid for lid in names if lid in fs]
        if len(matched) >= max(2, 0.5 * len(names)):
            return {lid: fs.get(lid, golden_hue(lid)) for lid in names}

    # 4. golden-ratio fallback
    return {lid: golden_hue(lid) for lid in names}


__all__ = [
    "build_atlas_palette", "network_palette", "is_network_atlas",
    "parse_color_lut", "golden_hue",
]
