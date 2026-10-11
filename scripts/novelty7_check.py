"""Descriptive supplementary check (read-only): an unsupervised novelty score on all seven features, compared with the magnitude-based novelty comparator D2,
on the pairs retained by the PRIMARY caliper. Not part of the locked analysis and not used for C1 or C2.
Output: outputs/matched_drift_v1/stats_review/novelty7_check.json (and .txt)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import matched as M, matched_config as mc, pilot_model as pm  # noqa: E402

cfg = mc.load()
lock = mc.load_lock(cfg)
OUT = os.path.join(mc.OUT_ROOT, "stats_review")
os.makedirs(OUT, exist_ok=True)
tr = pm.load_split(mc.SOURCE_ROOT, cfg["source_data"]["train_split"])
nov7, nov4 = M.build_novelty(tr, cfg, "secondary"), M.build_novelty(tr, cfg, "primary")
rng = np.random.default_rng(20261011)
B = 4000
res, lines = {}, []
for stage, label, base, n in (("m12", "MFIN", cfg["seed_blocks"]["final_m12"]["base"], lock["F_m12"]), ("m3", "MM3", cfg["seed_blocks"]["final_m3"]["base"], lock["F_m3"])):
    data = M.load_families(os.path.join(mc.RAW_ROOT, stage), label, list(range(base, base + n)))
    pairs = M.make_pairs(data)
    l4, l7 = nov4.ln_n(data["X"]), nov7.ln_n(data["X"])
    keep = M.caliper_keep(l4, pairs, lock["caliper_primary"])
    fam = data["family"].astype(int)
    fams = sorted(set(fam))
    pf = pairs["family"].astype(int)
    pidx = {f: np.flatnonzero(pf == f) for f in fams}
    res[stage] = {}
    bi = rng.integers(0, len(fams), size=(B, len(fams)))
    for nm, s in (("D2_magnitude_novelty", l4), ("novelty_all_7_features", l7)):
        w = M.pair_wins(s, pairs)
        Wf = np.array([(w[pidx[f]] * keep[pidx[f]]).sum() for f in fams])
        nf = np.array([keep[pidx[f]].sum() for f in fams], float)
        bs = Wf[bi].sum(1) / nf[bi].sum(1)
        res[stage][nm] = {"PMP": float(w[keep].mean()), "PMP_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "AUC_pooled": float(M.auc_mw(s, data["y"]))}
        label_ = "novelty 4 magnitude features (D2)" if nm.startswith("D2") else "novelty all 7 features"
        lines.append(f"{stage} {label_} PMP on primary-retained pairs {res[stage][nm]['PMP']:.3f} pooled AUC {res[stage][nm]['AUC_pooled']:.3f}")
json.dump(res, open(os.path.join(OUT, "novelty7_check.json"), "w"), indent=1)
open(os.path.join(OUT, "novelty7_check.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
