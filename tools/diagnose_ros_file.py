"""Diagnose a `.ros` file — count tokens, surface anything that looks
like trajectories. Use this when `[done] loaded N volumes, created 0
trajectories` shows up unexpectedly.

Run:
    /Users/ammar/miniforge3/envs/shankdetect/bin/python3 \\
      tools/diagnose_ros_file.py "/path/to/case.ros"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "CommonLib"))

from rosa_core.ros_parser import extract_tokens, parse_ros_text


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: diagnose_ros_file.py <path-to.ros>", file=sys.stderr)
        return 1
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8", errors="ignore")
    print(f"size: {len(text)} chars")
    tokens = extract_tokens(text)
    print(f"tokens: {len(tokens)}")
    # Histogram of all token names.
    from collections import Counter
    hist = Counter(t["token"] for t in tokens)
    print("\ntop 25 tokens by count:")
    for name, n in sorted(hist.items(), key=lambda p: -p[1])[:25]:
        print(f"  {name:<30} {n}")

    parsed = parse_ros_text(text)
    print(f"\ndisplays: {len(parsed['displays'])}")
    for d in parsed["displays"][:10]:
        print(f"  [{d.get('index')}] {d.get('volume')} ref={d.get('imagery_3dref')}")
    print(f"\ntrajectories parsed: {len(parsed['trajectories'])}")
    if parsed["trajectories"]:
        print("first 5 trajectories:")
        for t in parsed["trajectories"][:5]:
            print(f"  {t.get('name')}: start={t.get('start')} end={t.get('end')}")

    print(f"\ntrajectory_candidates: {parsed.get('trajectory_candidates', 0)}")
    rejects = parsed.get("trajectory_rejects") or []
    if rejects:
        print(f"trajectory rejects (parser couldn't read): {len(rejects)}")
        for r in rejects[:5]:
            print(f"  [{r['token']}] n_lines={r['n_lines']} "
                  f"fields_line2={r['n_fields_line2']} "
                  f"first='{r['first_line'][:80]}'")

    # Look for anything that could be trajectory-like.
    traj_like = sorted(
        (k for k in hist
         if any(s in k.upper()
                for s in ("TRAJ", "ELECTRODE", "PLAN", "POINT", "TARGET", "ENTRY"))),
        key=lambda k: -hist[k],
    )
    if traj_like:
        print("\ntrajectory-like token candidates (not yet parsed):")
        for k in traj_like:
            print(f"  {k:<30} {hist[k]}")
        # Print the content of the first instance of the top candidate.
        top = traj_like[0]
        for tok in tokens:
            if tok["token"] == top:
                print(f"\nfirst [{top}] payload (first 600 chars):")
                content = tok["content"][:600]
                print(content)
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
