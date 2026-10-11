"""Statistical method review (read-only; NOT part of the locked analysis).
Recomputes the family-level quantities behind PMP from the saved session scores, then evaluates candidate inferential procedures to decide
which, if any, add information beyond the locked family-clustered bootstrap. Uses no new data and changes no detector, feature, caliper or result.
Writes results/matched_drift_v1/stats_review/stat_review.json.   Run: python scripts/stat_review.py
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import matched as M  # noqa: E402
from postdrift_auth import matched_config as mc  # noqa: E402

FIN = mc.result_dir("final")
OUT = os.path.join(mc.OUT_ROOT, "stats_review")
os.makedirs(OUT, exist_ok=True)
cfg = mc.load()
lock = mc.load_lock(cfg)
DET = ["D1_rf_isotonic", "D2_novelty_only", "D3_isolation_forest", "D4_logistic_regression"]
RNG = np.random.default_rng(20261009)  # the review's own seed; the locked analysis seeds are not reused
B = 20000


def load(stage):
    label, tag, base, n = ("MFIN", "primary_m1m2", cfg["seed_blocks"]["final_m12"]["base"], lock["F_m12"]) if stage == "m12" else ("MM3", "stress_m3", cfg["seed_blocks"]["final_m3"]["base"], lock["F_m3"])
    data = M.load_families(os.path.join(mc.RAW_ROOT, stage), label, list(range(base, base + n)))
    pairs = M.make_pairs(data)
    sc = {}
    with open(os.path.join(FIN, f"{tag}_session_scores.csv"), newline="") as f:
        for r in csv.DictReader(f):
            sc[(int(r["family"]), r["arm"], r["session_id"])] = {"ln": float(r["ln_N_primary"]), **{d: float(r[d]) for d in DET}}
    keys = list(zip(data["family"].astype(int), data["arm"], data["session_id"]))
    ln = np.array([sc[k]["ln"] for k in keys])
    S = {d: np.array([sc[k][d] for k in keys]) for d in DET}
    keep = M.caliper_keep(ln, pairs, lock["caliper_primary"])
    fam = pairs["family"].astype(int)
    out = {"families": sorted(set(fam.tolist())), "keep": keep, "fam": fam, "mal_scn": pairs["mal_scenario"], "X": data["X"], "pairs": pairs, "S": S, "ln": ln}
    wins = {}
    for d in DET:
        sm, sb = S[d][pairs["mal"]], S[d][pairs["ben"]]
        wins[d] = (sm > sb) + 0.5 * (sm == sb)
    out["wins"] = wins
    return out


def fam_table(o, mask=None, wins=None):
    """Per-family numerators and denominators over the retained pairs (optionally a sub-mask)."""
    m = o["keep"] if mask is None else (o["keep"] & mask)
    fams = o["families"]
    n = np.array([m[o["fam"] == f].sum() for f in fams], float)
    W = {d: np.array([wins[d][m & (o["fam"] == f)].sum() for f in fams]) for d in wins}
    return fams, n, W


def ratio(Wf, nf):
    return float(Wf.sum() / nf.sum())


def boot_ci(Wf, nf, B=B):
    G = len(nf)
    idx = RNG.integers(0, G, size=(B, G))
    est = Wf[idx].sum(1) / nf[idx].sum(1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def robust_ci(Wf, nf, null=0.0):
    """Cluster-robust (linearised ratio, CR1) interval with t(G-1) reference. Wf may be a difference of wins."""
    G = len(nf)
    th = Wf.sum() / nf.sum()
    r = Wf - th * nf
    se = math.sqrt(G / (G - 1) * (r ** 2).sum()) / nf.sum()
    t = stats.t.ppf(0.975, G - 1)
    return {"estimate": float(th), "se": float(se), "ci95": [float(th - t * se), float(th + t * se)], "t_vs_null": float((th - null) / se), "df": G - 1}


def simple_ci(Wf, nf, B=B):
    """Interval if the retained pairs were (wrongly) treated as independent: Wald interval for the pooled proportion."""
    th = ratio(Wf, nf)
    se = math.sqrt(th * (1 - th) / nf.sum())
    return [th - 1.96 * se, th + 1.96 * se]


def sign_test(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    n = pos + neg
    p1 = float(stats.binom.sf(pos - 1, n, 0.5)) if n else 1.0
    return {"positive": pos, "negative": neg, "zero": int((d == 0).sum()), "p_one_sided": p1, "p_two_sided": float(min(1.0, 2 * min(p1, stats.binom.sf(neg - 1, n, 0.5))))}


def signflip_p(d, B2=2_000_000):
    """Cluster-level sign-flip test of sum(d) with Monte Carlo flips; exact lower bound 2^-G is returned when the observed sum is the maximum."""
    G = len(d)
    obs = d.sum()
    if np.all(d >= 0) and np.sum(d > 0) == G:
        return {"p_one_sided_exact": 2.0 ** -G, "note": "all clusters positive: observed sum is the maximum of the 2^G sign patterns"}
    s = RNG.choice([-1.0, 1.0], size=(B2 // 100, G))
    cnt = 0
    for _ in range(100):
        s = RNG.choice([-1.0, 1.0], size=(B2 // 100, G))
        cnt += int(((s * d).sum(1) >= obs - 1e-12).sum())
    return {"p_one_sided_mc": (cnt + 1) / (B2 + 1), "flips": B2}


def block(o, name, stage_note):
    res = {"note": stage_note}
    fams, n, W = fam_table(o, wins=o["wins"])
    G = len(fams)
    res["G"] = G
    res["retained_pairs"] = int(n.sum())
    res["retained_per_family"] = {"min": int(n.min()), "median": float(np.median(n)), "max": int(n.max())}
    res["detectors"] = {}
    for d in DET:
        Wf = W[d]
        pf = Wf / n
        res["detectors"][d] = {
            "PMP_pooled_ratio": ratio(Wf, n), "boot_ci_review": boot_ci(Wf, n), "robust_t_ci": robust_ci(Wf, n, null=0.5 * 1.0) if False else robust_ci(Wf - 0.5 * n, n, 0.0) | {"note": "interval is for PMP-0.5; add 0.5"},
            "naive_wald_ci_pairs_independent": simple_ci(Wf, n),
            "family_mean_unweighted": float(pf.mean()), "family_mean_t_ci": [float(pf.mean() - stats.t.ppf(0.975, G - 1) * pf.std(ddof=1) / math.sqrt(G)), float(pf.mean() + stats.t.ppf(0.975, G - 1) * pf.std(ddof=1) / math.sqrt(G))],
            "family_median": float(np.median(pf)), "family_min": float(pf.min()), "family_max": float(pf.max()),
            "families_above_0.5": int((pf > 0.5).sum()), "families_below_0.5": int((pf < 0.5).sum()), "sign_test_vs_0.5": sign_test(pf - 0.5),
            "tie_share_retained_pairs": float(np.mean((o["S"][d][o["pairs"]["mal"]] == o["S"][d][o["pairs"]["ben"]])[o["keep"]])),
        }
        # shift robust CI to PMP scale
        c = res["detectors"][d]["robust_t_ci"]
        res["detectors"][d]["robust_t_ci_pmp"] = [c["ci95"][0] + 0.5, c["ci95"][1] + 0.5]
    # design effect of clustering for D1
    Wf = W["D1_rf_isotonic"]
    se_cl = (np.array(res["detectors"]["D1_rf_isotonic"]["boot_ci_review"])[1] - np.array(res["detectors"]["D1_rf_isotonic"]["boot_ci_review"])[0]) / 3.92
    th = ratio(Wf, n)
    se_srs = math.sqrt(th * (1 - th) / n.sum())
    res["design_effect_D1"] = {"se_cluster_bootstrap": float(se_cl), "se_if_pairs_independent": float(se_srs), "design_effect": float((se_cl / se_srs) ** 2), "effective_pairs": float(n.sum() / ((se_cl / se_srs) ** 2))}
    # contrasts of D1 with the others (paired within pair)
    res["contrasts_D1_minus"] = {}
    for d in DET[1:]:
        dW = W["D1_rf_isotonic"] - W[d]
        pf = W["D1_rf_isotonic"] / n - W[d] / n
        wil = stats.wilcoxon(pf, alternative="greater") if np.any(pf != 0) else None
        res["contrasts_D1_minus"][d] = {"pooled_difference": ratio(dW, n), "boot_ci_review": boot_ci(dW, n), "robust_t_ci": robust_ci(dW, n, 0.0), "family_sign_test": sign_test(pf), "wilcoxon_signed_rank_greater_p": float(wil.pvalue) if wil is not None else None,
                                          "signflip": signflip_p(dW), "families_D1_greater": int((pf > 0).sum())}
    ps = [res["contrasts_D1_minus"][d]["wilcoxon_signed_rank_greater_p"] for d in DET[1:]]
    order = np.argsort(ps)
    holm = [None] * 3
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (3 - rank) * ps[i]))
        holm[i] = run
    for d, h in zip(DET[1:], holm):
        res["contrasts_D1_minus"][d]["wilcoxon_p_holm_3_contrasts"] = h
    # C1 sign-flip
    res["C1_signflip_D1"] = signflip_p(W["D1_rf_isotonic"] - 0.5 * n)
    # omnibus Friedman across detectors on family PMP (reference only)
    fr = stats.friedmanchisquare(*[W[d] / n for d in DET])
    res["friedman_reference_only"] = {"statistic": float(fr.statistic), "p": float(fr.pvalue), "blocks": G, "treatments": 4}
    return res, (fams, n, W)


def main():
    out = {}
    o12 = load("m12")
    r12, (fams, n, W) = block(o12, "m12", "M1/M2 families: confirmatory estimands C1, C2")
    out["m12"] = r12
    # check against the locked results
    P = json.load(open(os.path.join(FIN, "primary_m1m2_results.json")))
    for d in DET:
        assert abs(r12["detectors"][d]["PMP_pooled_ratio"] - P["detectors"][d]["PMP_primary"]["estimate"]) < 1e-9, d
    # structural stratum and within-M2 comparison (exploratory)
    X = o12["X"]
    mal = o12["pairs"]["mal"]
    sp = (X[mal][:, 4] > 0) | (X[mal][:, 5] > 0) | (X[mal][:, 6] > 0)
    kp = o12["keep"]
    assert int((sp & kp).sum()) == P["detectors"]["D1_rf_isotonic"]["by_structural_signature"]["signature_present"]["retained_pairs"] and int((~sp & kp).sum()) == P["detectors"]["D1_rf_isotonic"]["by_structural_signature"]["signature_absent"]["retained_pairs"]
    st = {"retained_pairs_by_type_and_signature": {}}
    for t in ("M1", "M2"):
        for lab, m in (("present", sp), ("absent", ~sp)):
            st["retained_pairs_by_type_and_signature"][f"{t}_{lab}"] = int((kp & m & (o12["mal_scn"] == t)).sum())
    d1w = o12["wins"]["D1_rf_isotonic"]
    for lab, m in (("M2_present", sp & (o12["mal_scn"] == "M2")), ("M2_absent", ~sp & (o12["mal_scn"] == "M2"))):
        fm, nn, WW = fam_table(o12, mask=m, wins={"D1": d1w, "D2": o12["wins"]["D2_novelty_only"]})
        ok = nn > 0
        st[lab] = {"pairs": int(nn.sum()), "families_with_pairs": int(ok.sum()), "PMP_D1": ratio(WW["D1"][ok], nn[ok]), "boot_ci_review": boot_ci(WW["D1"][ok], nn[ok]), "PMP_D2": ratio(WW["D2"][ok], nn[ok])}
    # within-M2 difference present - absent, cluster bootstrap over families (families contribute to both strata)
    fa = o12["families"]
    npres = np.array([(kp & sp & (o12["mal_scn"] == "M2") & (o12["fam"] == f)).sum() for f in fa], float)
    nabs = np.array([(kp & ~sp & (o12["mal_scn"] == "M2") & (o12["fam"] == f)).sum() for f in fa], float)
    Wp = np.array([d1w[kp & sp & (o12["mal_scn"] == "M2") & (o12["fam"] == f)].sum() for f in fa])
    Wa = np.array([d1w[kp & ~sp & (o12["mal_scn"] == "M2") & (o12["fam"] == f)].sum() for f in fa])
    G = len(fa)
    idx = RNG.integers(0, G, size=(B, G))
    diff = Wp[idx].sum(1) / np.maximum(npres[idx].sum(1), 1) - Wa[idx].sum(1) / np.maximum(nabs[idx].sum(1), 1)
    st["within_M2_difference_present_minus_absent"] = {"estimate": float(Wp.sum() / npres.sum() - Wa.sum() / nabs.sum()), "boot_ci_review": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))], "caveat": "stratum is defined by the realized features of the malicious twin, which the score itself uses; the comparison is a description, not a causal contrast"}
    # D1 score level in M2 absent: share of ties
    out["structural_within_M2"] = st
    # M3 (descriptive)
    o3 = load("m3")
    r3, _ = block(o3, "m3", "M3 families: descriptive (retention below the pre-specified minimum)")
    out["m3"] = r3
    json.dump(out, open(os.path.join(OUT, "stat_review.json"), "w"), indent=1, default=float)
    print("written", os.path.join(OUT, "stat_review.json"))


if __name__ == "__main__":
    main()
