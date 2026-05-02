"""Strategy-agnostic primitives shared across detection algorithms.

These are the pieces that any reasonable detection strategy
(``contact_pitch_v1`` today, a future ``contact_pitch_v2`` or a CNN)
might want to reuse. Each is pure NumPy / SciPy / SimpleITK — no Slicer
dependencies, no algorithm-specific tuning.

  * ``preprocessing`` — canonicalize-to-1mm + anti-alias σ + HU clamp;
    LoG / Frangi feature volumes; head hull + intracranial mask.
  * ``bolt_anchor`` — extract bolt CCs from a metal-evidence cloud and
    snap a trajectory's start to the bolt CC closest to its axis.

Strategies are free to import from here. Anything strategy-specific
(walker tolerances, scoring weights, etc.) lives inside the strategy's
own module instead.
"""
