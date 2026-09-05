# Localization experiments and integration policy

Recorded 2026-09-05 during the desktop PR reconciliation.

The production heuristic localizer remains the default. The graph approaches
remain experiments for comparative evaluation; their worktrees are preserved,
including uncommitted files. Do not merge or delete them as routine branch cleanup.

| Worktree / branch | Base commit | State at inventory |
| --- | --- | --- |
| `slicer-rosa-helper-graph-pr` / `feat/component-graph-v2-provider-pr` | `b2905e8` | Modified placement/CLI code and untracked graph provider, comparison tests, tools, and analysis |
| `rosa-seed-graph-experiments` / `seed-graph-experiments` | `da2c437` | Modified classifier/refinement code and untracked graph experiments, evaluation tools, and output |

These worktrees contain uncommitted work; their branch commits alone are **not**
a backup of that work. Before retiring either worktree, inventory and preserve
source/test changes separately from generated results and any patient data.
Do not bulk-add experimental output to GitHub.

## Shared verification contract

Both approaches propose trajectories and contact positions. Human inspection
and correction remain the last localization step. Trajectory detection success
does not establish individual-contact accuracy, especially where image evidence
is ambiguous. The application must preserve inclusion/exclusion decisions and
invalidate anatomical labels when contact geometry changes.

Before promoting an experimental provider, compare it with the production
provider on the same held-out cases and adjudicated references. Evaluate
trajectory recovery, contact position/count errors, false contacts, runtime,
and reviewer correction time separately. Record CT resolution, electrode model,
and difficult artifact cases. Ground-truth uncertainty and excluded cases must
be reported explicitly.

Keep the engine's existing provider boundary. A graph provider can eventually
be an optional candidate source feeding the same placement and review workflow;
there is no requirement to replace the production detector wholesale.
