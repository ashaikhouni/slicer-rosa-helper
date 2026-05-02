"""rosa_agent — pure-Python CLI agent for the ROSA viewer SEEG pipeline.

Loads a ROSA folder, runs detection, places contacts, labels against atlases,
all from the terminal. No Slicer / VTK / Qt imports.

Subcommands are wired in ``main.py``; ``python -m rosa_agent --help`` lists
them.
"""

__all__ = ["main"]
