"""Diff consecutive pass dumps. Shell- and platform-independent.

The core workflow of stage 2 is "what did each pass actually do", and the answer
is a diff between dump N and dump N+1. Don't rely on the shell for this:
PowerShell's `diff` is an alias for Compare-Object, which compares OBJECTS, so
`diff a.txt b.txt` compares the two filename STRINGS and never opens the files.

Usage:
    python scripts/diffdump.py                  # every consecutive pair in ./dumps
    python scripts/diffdump.py ./dumps          # ... in a given directory
    python scripts/diffdump.py a.txt b.txt      # one specific pair
"""

from __future__ import annotations

import difflib
import os
import sys


def read(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.readlines()


def show(a: str, b: str) -> int:
    """Unified diff of two dump files. Returns the number of changed lines."""
    diff = list(difflib.unified_diff(
        read(a), read(b), fromfile=os.path.basename(a), tofile=os.path.basename(b),
        n=1, lineterm=""))
    changed = sum(1 for line in diff
                  if line[:1] in "+-" and not line.startswith(("+++", "---")))
    print("=" * 78)
    print(f"{os.path.basename(a)}\n  ->  {os.path.basename(b)}")
    print("=" * 78)
    if not diff:
        print("  (no change)\n")
        return 0
    for line in diff:
        print(line.rstrip("\n"))
    print(f"\n  {changed} lines changed\n")
    return changed


def pairs(dump_dir: str) -> None:
    files = sorted(f for f in os.listdir(dump_dir) if f.endswith(".txt"))
    # group by module name: "<module>.<index>.<label>.txt"
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(f.split(".")[0], []).append(f)
    if not groups:
        sys.exit(f"no .txt dumps in {dump_dir}. Run with MOCKXLA_DUMP={dump_dir}")
    for module, fs in groups.items():
        print(f"\n######## {module} ########")
        for a, b in zip(fs, fs[1:]):
            show(os.path.join(dump_dir, a), os.path.join(dump_dir, b))


def main(argv: list[str]) -> None:
    if len(argv) == 2:
        show(argv[0], argv[1])
    else:
        pairs(argv[0] if argv else "dumps")


if __name__ == "__main__":
    main(sys.argv[1:])
