"""Independent recomputation of the primary matched-drift results from the released per-run feature tables and ground-truth files.

Deliberately imports NOTHING from postdrift_auth: hashing, random forest + isotonic fit, novelty, pairing, PMP, AUC and the family bootstrap are
re-implemented here from the file formats alone. Compared with results/matched_drift_v1/final/*.json and *_session_scores.csv.
Any difference above 1e-9 is a STOP condition (exit code 2). Writes outputs/matched_drift_v1/independent/reproduction_comparison.csv/.json.
"""
import csv
import glob
import hashlib
import json
import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("MTE_DATA_DIR", os.path.join(ROOT, "data"))
RAW_FAM = os.path.join(RAW, "matched_drift_v1")
FIN = os.path.join(ROOT, "results/matched_drift_v1/final")
OUT = os.path.join(ROOT, "outputs/matched_drift_v1/independent")
os.makedirs(OUT, exist_ok=True)
FEAT = ["iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio", "topic_novelty", "sub_novel_count", "reconnects_60s"]


def draw(run_seed, comp, idx):
    n = int.from_bytes(hashlib.blake2b(b"postdrift_auth/derived-draw/v1|" + f"{run_seed}|{comp}|{idx}".encode(), digest_size=8).digest(), "big")
    return (n >> 11) / float(1 << 53)


def derived_seed(run_seed, comp):
    return int.from_bytes(hashlib.blake2b(b"postdrift_auth/derived-draw/v1|" + f"{run_seed}|{comp}|seed".encode(), digest_size=8).digest(), "big")


def read_run(d, with_labels=True):
    gt = {}
    if with_labels:
        for l in open(os.path.join(d, "ground_truth.jsonl")):
            g = json.loads(l)
            gt[g["session_id"]] = g
    rows = []
    with open(os.path.join(d, "extract_out", "features.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["client_id"], [float(r[k]) for k in FEAT]))
    return rows, gt


def source(root, split):
    X, y, scn = [], [], []
    for d in sorted(x for x in glob.glob(os.path.join(root, split, f"{split}-*")) if os.path.isdir(x) and ".FAILED_attempt" not in x):
        rows, gt = read_run(d)
        for cid, v in rows:
            X.append(v); y.append(gt[cid]["y"]); scn.append(gt[cid]["scenario"])
    return np.array(X), np.array(y), np.array(scn)


cfg = json.load(open(os.path.join(ROOT, "configs/matched_drift_v1.json")))
lock = json.load(open(os.path.join(ROOT, "configs/matched_drift_v1_lock.json")))
p2 = json.load(open(os.path.join(ROOT, cfg["pilot_v2_config"])))
seed_an = cfg["analysis_seed"]
src_root = os.path.join(RAW, "source_regime")
Xtr, ytr, str_ = source(src_root, "TRAIN")
Xca, yca, _ = source(src_root, "CAL")
rfc = p2["rf"]
rf = RandomForestClassifier(n_estimators=rfc["n_estimators"], min_samples_leaf=rfc["min_samples_leaf"], max_features=rfc["max_features"], max_depth=rfc["max_depth"],
                            class_weight=rfc["class_weight"], bootstrap=rfc["bootstrap"], n_jobs=1, random_state=derived_seed(seed_an, "rf/full") % (2 ** 31)).fit(Xtr, ytr)
ic = p2["isotonic"]
iso = IsotonicRegression(increasing=ic["increasing"], y_min=ic["y_min"], y_max=ic["y_max"], out_of_bounds=ic["out_of_bounds"]).fit(rf.predict_proba(Xca)[:, 1], yca)
d1 = lambda X: iso.predict(rf.predict_proba(X)[:, 1])  # noqa: E731
nv = cfg["novelty"]
ref = Xtr[np.isin(str_, nv["reference_scenarios"])][:, :4]
mu, sd = ref.mean(axis=0), np.maximum(ref.std(axis=0, ddof=1), 1e-6)
zref = (ref - mu) / sd


def ln_n(X):
    d = cdist((X[:, :4] - mu) / sd, zref)
    return np.log(np.maximum(np.sort(d, axis=1)[:, :nv["k"]].mean(axis=1), 1e-6))


def analyse(stage, label, tag, n_fam, base_seed):
    fam_seeds = [base_seed + i for i in range(n_fam)]
    fam, arm_, cid_, pid, y, scn, X = [], [], [], [], [], [], []
    for sdd in fam_seeds:
        for arm in "AB":
            rows, gt = read_run(os.path.join(RAW_FAM, stage, f"{label}-{sdd}-{arm}"))
            for cid, v in rows:
                fam.append(sdd); arm_.append(arm); cid_.append(cid); pid.append(gt[cid]["pair_id"]); y.append(gt[cid]["y"]); scn.append(gt[cid]["scenario"]); X.append(v)
    fam, y, X, scn = np.array(fam), np.array(y), np.array(X), np.array(scn)
    by = {}
    for i, p in enumerate(pid):
        by.setdefault(p, []).append(i)
    ben, mal, pf, ms = [], [], [], []
    for p in sorted(by):
        a, b = by[p]
        bi, mi = (a, b) if y[a] == 0 else (b, a)
        assert y[bi] == 0 and y[mi] == 1
        ben.append(bi); mal.append(mi); pf.append(fam[bi]); ms.append(scn[mi])
    ben, mal, pf, ms = np.array(ben), np.array(mal), np.array(pf), np.array(ms)
    s1 = d1(X); lnN = ln_n(X)
    keep = np.abs(lnN[mal] - lnN[ben]) <= lock["caliper_primary"]
    win = lambda s: (s[mal] > s[ben]).astype(float) + 0.5 * (s[mal] == s[ben])  # noqa: E731
    w1, w2 = win(s1), win(lnN)
    fams = sorted(set(fam)); n = len(fams)
    boot = np.array([[int(draw(seed_an, f"{tag}/boot/{b}", j) * n) for j in range(n)] for b in range(cfg["primary_analysis"]["n_boot"])])
    pair_idx = {f: np.flatnonzero(pf == f) for f in fams}; sess_idx = {f: np.flatnonzero(fam == f) for f in fams}

    def pmp_boot(w, k):
        s_ = np.array([(w[pair_idx[f]] * k[pair_idx[f]]).sum() for f in fams]); m_ = np.array([k[pair_idx[f]].sum() for f in fams])
        num, den = s_[boot].sum(axis=1), m_[boot].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def auc(s, yy):
        r = rankdata(s); n1, n0 = int(yy.sum()), int((1 - yy).sum())
        return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    ci = lambda b: [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]  # noqa: E731
    res = {"retention": float(keep.mean()), "retained_pairs": int(keep.sum())}
    b1 = pmp_boot(w1, keep)
    res["D1_PMP"] = float(w1[keep].mean()); res["D1_PMP_ci"] = ci(b1)
    auc_b = np.array([auc(s1[np.concatenate([sess_idx[fams[j]] for j in row])], y[np.concatenate([sess_idx[fams[j]] for j in row])]) for row in boot])
    res["D1_AUC_pooled"] = float(auc(s1, y)); res["D1_AUC_ci"] = ci(auc_b)
    res["Delta_match"] = res["D1_AUC_pooled"] - res["D1_PMP"]; res["Delta_match_ci"] = ci(auc_b - b1)
    b2 = pmp_boot(w2, keep)
    res["D2_PMP"] = float(w2[keep].mean()); res["D1_minus_D2"] = float(w1[keep].mean() - w2[keep].mean()); res["D1_minus_D2_ci"] = ci(b1 - b2)
    sig = (X[mal][:, 4:] != 0).any(axis=1)
    for nm, flag in (("signature_present", True), ("signature_absent", False)):
        k = keep & (sig == flag)
        res[f"D1_PMP_{nm}"] = float(w1[k].mean()); res[f"D1_PMP_{nm}_ci"] = ci(pmp_boot(w1, k)); res[f"retained_{nm}"] = int(k.sum())
    for t in sorted(set(ms)):
        k = keep & (ms == t)
        res[f"D1_PMP_{t}"] = float(w1[k].mean()); res[f"D1_PMP_{t}_ci"] = ci(pmp_boot(w1, k))
    # scores from FEATURES ONLY (no ground-truth file is opened): label independence of the scoring path
    Xf = []
    for sdd in fam_seeds:
        for arm in "AB":
            rows, _ = read_run(os.path.join(RAW_FAM, stage, f"{label}-{sdd}-{arm}"), with_labels=False)
            Xf += [v for _, v in rows]
    res["scores_from_features_only_identical"] = bool(np.array_equal(d1(np.array(Xf)), s1))
    return res, s1, lnN, (fam, arm_, cid_)


rows_out, worst, stop = [], 0.0, False
for stage, label, tag, n_fam, base in (("m12", "MFIN", "primary_m1m2", lock["F_m12"], cfg["seed_blocks"]["final_m12"]["base"]), ("m3", "MM3", "stress_m3", lock["F_m3"], cfg["seed_blocks"]["final_m3"]["base"])):
    res, s1, lnN, ids = analyse(stage, label, tag, n_fam, base)
    stored = json.load(open(os.path.join(FIN, f"{tag}_results.json")))
    d1s, d2s = stored["detectors"]["D1_rf_isotonic"], stored["detectors"]["D2_novelty_only"]
    comp = [("retention", res["retention"], stored["retention"]["primary"]), ("D1 PMP", res["D1_PMP"], d1s["PMP_primary"]["estimate"]),
            ("D1 PMP CI low", res["D1_PMP_ci"][0], d1s["PMP_primary"]["ci95_two_sided"][0]), ("D1 PMP CI high", res["D1_PMP_ci"][1], d1s["PMP_primary"]["ci95_two_sided"][1]),
            ("D1 pooled AUC", res["D1_AUC_pooled"], d1s["AUC_pooled"]["estimate"]), ("D1 AUC CI low", res["D1_AUC_ci"][0], d1s["AUC_pooled"]["ci95_two_sided"][0]),
            ("D1 AUC CI high", res["D1_AUC_ci"][1], d1s["AUC_pooled"]["ci95_two_sided"][1]), ("Delta_match", res["Delta_match"], d1s["Delta_match"]["estimate"]),
            ("Delta_match CI low", res["Delta_match_ci"][0], d1s["Delta_match"]["ci95_two_sided"][0]), ("Delta_match CI high", res["Delta_match_ci"][1], d1s["Delta_match"]["ci95_two_sided"][1]),
            ("D2 PMP", res["D2_PMP"], d2s["PMP_primary"]["estimate"]), ("D1 - D2", res["D1_minus_D2"], stored["P2_D1_minus_D2_gt_0"]["estimate"]),
            ("D1 - D2 CI low", res["D1_minus_D2_ci"][0], stored["P2_D1_minus_D2_gt_0"]["ci95_two_sided"][0]), ("D1 - D2 CI high", res["D1_minus_D2_ci"][1], stored["P2_D1_minus_D2_gt_0"]["ci95_two_sided"][1]),
            ("D1 PMP, structural present", res["D1_PMP_signature_present"], d1s["by_structural_signature"]["signature_present"]["estimate"]),
            ("D1 PMP, structural absent", res["D1_PMP_signature_absent"], d1s["by_structural_signature"]["signature_absent"]["estimate"]),
            ("retained pairs, structural present", res["retained_signature_present"], d1s["by_structural_signature"]["signature_present"]["retained_pairs"]),
            ("retained pairs, structural absent", res["retained_signature_absent"], d1s["by_structural_signature"]["signature_absent"]["retained_pairs"])]
    for t, v in d1s["by_malicious_type"].items():
        comp += [(f"D1 PMP {t}", res[f"D1_PMP_{t}"], v["estimate"]), (f"D1 PMP {t} CI low", res[f"D1_PMP_{t}_ci"][0], v["ci95_two_sided"][0]), (f"D1 PMP {t} CI high", res[f"D1_PMP_{t}_ci"][1], v["ci95_two_sided"][1])]
    # row-level: stored session scores and ln N
    sc = list(csv.DictReader(open(os.path.join(FIN, f"{tag}_session_scores.csv"))))
    st_s1 = np.array([float(r["D1_rf_isotonic"]) for r in sc]); st_ln = np.array([float(r["ln_N_primary"]) for r in sc])
    comp += [("max |D1 score - stored|, all sessions", float(np.max(np.abs(st_s1 - s1))), 0.0), ("max |ln N - stored|, all sessions", float(np.max(np.abs(st_ln - lnN))), 0.0)]
    for name, mine, theirs in comp:
        diff = abs(mine - theirs); worst = max(worst, diff)
        rows_out.append({"set": tag, "quantity": name, "independent": mine, "stored": theirs, "abs_diff": diff, "ok_1e-9": diff <= 1e-9})
    rows_out.append({"set": tag, "quantity": "scores recomputed from features only (no ground-truth file opened) identical to stored", "independent": res["scores_from_features_only_identical"], "stored": True, "abs_diff": 0.0 if res["scores_from_features_only_identical"] else 1.0,
                     "ok_1e-9": res["scores_from_features_only_identical"]})
    if not res["scores_from_features_only_identical"]:
        stop = True
with open(os.path.join(OUT, "reproduction_comparison.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader(); w.writerows(rows_out)
bad = [r for r in rows_out if not r["ok_1e-9"]]
json.dump({"n_quantities": len(rows_out), "n_exact_or_within_1e-9": len(rows_out) - len(bad), "max_abs_diff": worst, "discrepancies": bad, "stop_condition": bool(bad) or stop}, open(os.path.join(OUT, "reproduction_summary.json"), "w"), indent=1, default=str)
print(f"{len(rows_out)} quantities compared; within 1e-9: {len(rows_out) - len(bad)}; maximum absolute difference {worst:.3g}")
for r in rows_out:
    if r["set"] == "primary_m1m2" and r["quantity"] in ("retention", "D1 PMP", "D1 PMP CI low", "D1 PMP CI high", "D1 pooled AUC", "D2 PMP", "D1 - D2", "D1 PMP, structural present", "D1 PMP, structural absent") or r["set"] == "stress_m3" and r["quantity"].startswith("D1 PMP") or "max |" in r["quantity"]:
        print(f"  {r['set']:13s} {r['quantity'][:46]:46s} independent {r['independent']!s:>22.22} stored {r['stored']!s:>22.22} diff {r['abs_diff']:.2g}")
sys.exit(2 if (bad or stop) else 0)
