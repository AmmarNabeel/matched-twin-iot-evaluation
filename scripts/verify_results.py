"""Compare newly computed results (outputs/) with the reference results shipped in results/.

usage: python scripts/verify_results.py [final] [stats_review] [exploratory] [figures]

Numbers are compared with a relative tolerance of 1e-9; text and booleans must be equal. Exit code 0 means that everything that was compared agrees.
A folder that has not been recomputed yet (no outputs/ copy) is skipped and reported as such.
"""
import csv
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import matched_config as mc  # noqa: E402

SKIP_KEYS = {"git", "started_utc", "finished_utc", "timestamp_utc"}
TOL = 1e-9


def diff(a, b, path=""):
    """Yields a description of every difference between two parsed JSON objects."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in SKIP_KEYS:
                continue
            if k not in a or k not in b:
                yield f"{path}/{k}: missing on one side"
            else:
                yield from diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            yield f"{path}: length {len(a)} vs {len(b)}"
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                yield from diff(x, y, f"{path}[{i}]")
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b:
            yield f"{path}: {a!r} vs {b!r}"
    elif not math.isclose(a, b, rel_tol=TOL, abs_tol=TOL):
        yield f"{path}: {a!r} vs {b!r}"


def csv_diff(p, q):
    with open(p, newline="") as f, open(q, newline="") as g:
        ra, rb = list(csv.reader(f)), list(csv.reader(g))
    if len(ra) != len(rb):
        return [f"{os.path.basename(p)}: {len(ra)} vs {len(rb)} rows"]
    out = []
    for i, (x, y) in enumerate(zip(ra, rb)):
        for j, (u, v) in enumerate(zip(x, y)):
            try:
                same = math.isclose(float(u), float(v), rel_tol=TOL, abs_tol=TOL)
            except ValueError:
                same = u == v
            if not same:
                out.append(f"{os.path.basename(p)} row {i} col {j}: {u} vs {v}")
    return out


def compare_folder(name):
    new, ref = os.path.join(mc.OUT_ROOT, name), os.path.join(mc.REF_ROOT, name)
    if not os.path.isdir(new):
        print(f"{name}: not recomputed (no {os.path.relpath(new, ROOT)}), skipped")
        return 0, 0
    n_files, n_bad = 0, 0
    for dirpath, _, files in os.walk(ref):
        for fn in sorted(files):
            if not fn.endswith((".json", ".csv")):
                continue
            r = os.path.join(dirpath, fn)
            n = os.path.join(new, os.path.relpath(r, ref))
            if not os.path.exists(n):
                continue  # not every reference file is produced by every script
            n_files += 1
            if fn.endswith(".json"):
                with open(r) as f, open(n) as g:
                    problems = list(diff(json.load(f), json.load(g)))
            else:
                problems = csv_diff(r, n)
            if problems:
                n_bad += 1
                print(f"  DIFFERENT {os.path.relpath(r, mc.REF_ROOT)}: {len(problems)} differences, first: {problems[0]}")
    print(f"{name}: {n_files} files compared, {n_bad} differ")
    return n_files, n_bad


def main(names):
    total_bad, total = 0, 0
    for name in names:
        n, b = compare_folder(name)
        total += n
        total_bad += b
    print("OK: all compared files agree with the reference results" if total and not total_bad else ("nothing compared" if not total else "FAILED"))
    return 0 if total and not total_bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["final", "stats_review", "exploratory"]))
