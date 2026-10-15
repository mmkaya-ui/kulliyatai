"""No private or generated data anywhere in git history (see docs/git-history.md)."""

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _in_git_repo() -> bool:
    if not shutil.which("git"):
        return False
    r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


@pytest.mark.skipif(not _in_git_repo(), reason="not a git checkout")
def test_history_has_no_private_or_large_files():
    spec = importlib.util.spec_from_file_location("audit_repo", ROOT / "scripts" / "audit_repo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    problems, stats = mod.audit()
    assert problems == [], problems
    assert stats["commits"] >= 1
