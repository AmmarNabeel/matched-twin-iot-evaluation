"""Run provenance: git commit hash and dirty flag, and the clean-tree rule (Stage 2C).

git_dirty is True if `git status --porcelain` lists ANY change, tracked or untracked (ignored paths such as data/ and results/raw/
are not listed by git). Scientific runs (every split except SMOKE) refuse to start from a dirty tree; SMOKE engineering runs may
start dirty only with an explicit flag, and the dirty state is then recorded in the metadata.
"""
from __future__ import annotations

import subprocess

SCIENTIFIC_SPLITS = ("TRAIN", "CAL", "VAL", "TEST-SEEN", "TEST-HELD-OUT")


class DirtyTreeError(RuntimeError):
    pass


def git_state(root: str) -> dict:
    head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", root, "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    return {"commit": head, "git_dirty": bool(status.strip()), "dirty_paths": [ln[3:] for ln in status.splitlines()]}


def require_clean(state: dict, split: str, allow_dirty_smoke: bool = False) -> None:
    if not state["git_dirty"]:
        return
    if split == "SMOKE" and allow_dirty_smoke:
        return
    raise DirtyTreeError(f"refusing to start a {split} run from a dirty working tree: {state['dirty_paths'][:10]}")
