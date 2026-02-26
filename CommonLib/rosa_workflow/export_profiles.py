"""Export profile presets for the central export workflow."""

EXPORT_PROFILES = {
    "contacts_only": {
        "include_contacts": True,
        "include_planned": False,
        "include_final": False,
        "include_qc": False,
        "include_atlas": False,
        "include_volumes": False,
    },
    "trajectories_only": {
        "include_contacts": False,
        "include_planned": True,
        "include_final": True,
        "include_qc": False,
        "include_atlas": False,
        "include_volumes": False,
    },
    "registered_volumes_only": {
        "include_contacts": False,
        "include_planned": False,
        "include_final": False,
        "include_qc": False,
        "include_atlas": False,
        "include_volumes": True,
    },
    "atlas_only": {
        "include_contacts": False,
        "include_planned": False,
        "include_final": False,
        "include_qc": False,
        "include_atlas": True,
        "include_volumes": False,
    },
    "qc_only": {
        "include_contacts": False,
        "include_planned": False,
        "include_final": False,
        "include_qc": True,
        "include_atlas": False,
        "include_volumes": False,
    },
    "full_bundle": {
        "include_contacts": True,
        "include_planned": True,
        "include_final": True,
        "include_qc": True,
        "include_atlas": True,
        "include_volumes": True,
    },
}


def profile_names():
    """Return stable sorted export profile names."""
    return sorted(EXPORT_PROFILES.keys())


def get_export_profile(name, fallback="full_bundle"):
    """Return profile dictionary by name, using fallback when unknown."""
    profile_name = str(name or "").strip()
    if profile_name in EXPORT_PROFILES:
        return dict(EXPORT_PROFILES[profile_name]), profile_name
    return dict(EXPORT_PROFILES.get(fallback, {})), fallback


def merge_export_profile(name, overrides=None, fallback="full_bundle"):
    """Return profile with optional boolean overrides applied."""
    profile, profile_name = get_export_profile(name=name, fallback=fallback)
    for key, value in (overrides or {}).items():
        if key in profile:
            profile[key] = bool(value)
    return profile, profile_name


def export_profile(profile_name, output_frame_node_id=None, options=None):
    """Return normalized export profile payload used by exporters.

    This helper does not write files; it resolves and merges profile settings so
    modules can pass one canonical payload into export logic.
    """
    profile, resolved_name = merge_export_profile(profile_name, overrides=options or {})
    return {
        "profile_name": resolved_name,
        "output_frame_node_id": output_frame_node_id or "",
        "options": profile,
    }
