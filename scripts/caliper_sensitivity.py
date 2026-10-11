"""Descriptive caliper sensitivity (read-only; saved session scores only). Recomputes retention and PMP for tighter calipers (multiples of the primary caliper).
The primary analysis is unchanged. Output: outputs/matched_drift_v1/stats_review/caliper_sensitivity.json"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import matched_config as mc  # noqa: E402

cfg = mc.load()
lock = mc.load_lock(cfg)
FIN = mc.result_dir("final")
DET = ["D1_rf_isotonic", "D2_novelty_only", "D3_isolation_forest", "D4_logistic_regression"]
MULT = [1.0, 0.75, 0.5, 0.25]
rng = np.random.default_rng(20261012)
B = 4000
out = {}
for tag, name in (("primary_m1m2", "m12"), ("stress_m3", "m3")):
    R = json.load(open(os.path.join(FIN, f"{tag}_results.json")))
    by = {}
    with open(os.path.join(FIN, f"{tag}_session_scores.csv"), newline="") as f:
        for r in csv.DictReader(f):
            by.setdefault(r["pair_id"], {})[int(r["y"])] = r
    pids = sorted(by)
    fam = np.array([int(by[p][1]["family"]) for p in pids])
    lnm = np.array([float(by[p][1]["ln_N_primary"]) for p in pids])
    lnb = np.array([float(by[p][0]["ln_N_primary"]) for p in pids])
    wins = {}
    for d in DET:
        sm = np.array([float(by[p][1][d]) for p in pids])
        sb = np.array([float(by[p][0][d]) for p in pids])
        wins[d] = (sm > sb) + 0.5 * (sm == sb)
    fams = sorted(set(fam))
    fidx = {x: np.flatnonzero(fam == x) for x in fams}
    bi = rng.integers(0, len(fams), size=(B, len(fams)))
    out[name] = {"n_pairs": len(pids), "calipers": {}}
    for m in MULT:
        c = lock["caliper_primary"] * m
        keep = np.abs(lnm - lnb) <= c
        nf = np.array([keep[fidx[x]].sum() for x in fams], float)
        row = {"caliper": c, "retained_pairs": int(keep.sum()), "retention": float(keep.mean()), "families_with_pairs": int((nf > 0).sum()), "min_pairs_per_family": int(nf.min()), "PMP": {}}
        for d in DET:
            Wf = np.array([(wins[d][fidx[x]] * keep[fidx[x]]).sum() for x in fams])
            den = nf[bi].sum(1)
            ok = den > 0
            b = Wf[bi].sum(1)[ok] / den[ok]
            row["PMP"][d] = {"estimate": float(wins[d][keep].mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
        row["D1_minus_D2"] = row["PMP"]["D1_rf_isotonic"]["estimate"] - row["PMP"]["D2_novelty_only"]["estimate"]
        out[name]["calipers"][str(m)] = row
    # consistency with the locked analysis at multiple 1.0
    assert abs(out[name]["calipers"]["1.0"]["PMP"]["D1_rf_isotonic"]["estimate"] - R["detectors"]["D1_rf_isotonic"]["PMP_primary"]["estimate"]) < 1e-9
    assert abs(out[name]["calipers"]["1.0"]["retention"] - R["retention"]["primary"]) < 1e-12
os.makedirs(os.path.join(mc.OUT_ROOT, "stats_review"), exist_ok=True)
with open(os.path.join(mc.OUT_ROOT, "stats_review", "caliper_sensitivity.json"), "w") as fh:
    json.dump(out, fh, indent=1)
for st, v in out.items():
    print(st)
    for m, r in v["calipers"].items():
        print(f"  x{m}: c={r['caliper']:.3f} retained {r['retained_pairs']} ({100*r['retention']:.1f}%) fams {r['families_with_pairs']} minpairs {r['min_pairs_per_family']}  D1 {r['PMP']['D1_rf_isotonic']['estimate']:.3f}  D2 {r['PMP']['D2_novelty_only']['estimate']:.3f}  D3 {r['PMP']['D3_isolation_forest']['estimate']:.3f}  D4 {r['PMP']['D4_logistic_regression']['estimate']:.3f}  D1-D2 {r['D1_minus_D2']:.3f}")
