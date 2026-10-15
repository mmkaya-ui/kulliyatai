"""Audit the git history for private or generated data.

    python scripts/audit_repo.py        # exit 1 if any commit contains a forbidden path or a large blob

Checks every object reachable from any ref (full history, all branches/tags):

* no path under ``my_pdfs/``, ``vector_db*/`` or ``.env``;
* no blob larger than ``MAX_BLOB_BYTES``, except the files listed in
  ``ALLOWED_LARGE`` (the bundled public-domain PDFs and the font).

In a shallow clone only the fetched commits can be checked; the script says so.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_BLOB_BYTES = 200_000
FORBIDDEN = re.compile(r"^(my_pdfs/|vector_db[^/]*/|\.env$)")
ALLOWED_LARGE = re.compile(r"^(sample_docs/[^/]+\.pdf|scripts/fonts/DejaVuSans\.ttf)$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def audit() -> tuple[list[str], dict]:
    objects = git("rev-list", "--objects", "--all").splitlines()
    paths = {}
    for line in objects:
        sha, _, path = line.partition(" ")
        if path:
            paths.setdefault(sha, set()).add(path)
    batch = subprocess.run(
        ["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        cwd=ROOT,
        input="\n".join(paths),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    problems, largest = [], (0, "")
    for row in batch:
        sha, kind, size = row.split()
        if kind != "blob":
            continue
        for p in paths.get(sha, ()):
            if FORBIDDEN.match(p):
                problems.append(f"forbidden path in history: {p}")
            if int(size) > MAX_BLOB_BYTES and not ALLOWED_LARGE.match(p):
                problems.append(f"large blob ({int(size):,} bytes): {p}")
            if int(size) > largest[0]:
                largest = (int(size), p)
    stats = {
        "commits": len(git("rev-list", "--all").split()),
        "shallow": git("rev-parse", "--is-shallow-repository").strip() == "true",
        "objects": len(objects),
        "largest_blob": {"bytes": largest[0], "path": largest[1]},
    }
    return problems, stats


def main() -> int:
    problems, stats = audit()
    print(f"commits checked: {stats['commits']}{' (shallow clone: partial history)' if stats['shallow'] else ''}")
    print(f"largest blob: {stats['largest_blob']['bytes']:,} bytes ({stats['largest_blob']['path']})")
    for p in problems:
        print("PROBLEM", p)
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
