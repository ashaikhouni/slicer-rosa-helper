"""GT labeler — Flask-based ground-truth annotation tool for SEEG datasets.

Run with::

    python -m labeler --dataset /path/to/contact_label_dataset

Opens a local Flask server + browser tab. Per-trajectory annotations are
saved immediately to ``<dataset>/gt/trajectories_gt.tsv`` (one row per
trajectory) and ``<dataset>/gt/contacts_gt.tsv`` (auto-derived from
electrode model + corrected tip).
"""
