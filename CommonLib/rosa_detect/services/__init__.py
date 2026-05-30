"""Shared services for SEEG processing (brain-mask backends, etc.).

Lifted out of the CLI (cli/rosa_agent/services) 2026-05-30 so the brain-mask
backend (synthstrip / log-watershed / auto selector) is usable by the headless
CLI AND the Slicer/place_seeg path through one shared interface.
"""
