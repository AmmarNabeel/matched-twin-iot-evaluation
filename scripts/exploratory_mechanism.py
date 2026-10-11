"""Exploratory mechanism analysis (EXPLORATORY: motivated by the final result; not part of the confirmatory analysis).

Permutation importance and masking of the frozen model, auxiliary random forests trained only on the source regime, feature and score distributions of the
twins, and the counterfactual feature swaps on the retained M3 pairs. Uses the released feature tables only. Writes outputs/matched_drift_v1/exploratory/.
usage: python scripts/exploratory_mechanism.py
"""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import matched_config as mc  # noqa: E402

RAW = mc.RAW_ROOT
OUT = os.path.join(mc.OUT_ROOT, "exploratory")
os.makedirs(OUT, exist_ok=True)


def write_csv(name, rows, cols=None):
    cols = cols or list(rows[0])
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

def dump(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

FEAT = ["iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio", "topic_novelty", "sub_novel_count", "reconnects_60s"]

def _block(stage):
    return {"m12": (4100000, 4100037), "m3": (5100000, 5100020), "dry": (3200000, 3200003)}[stage]

def part_d():
    """EXPLORATORY (motivated by the final result). Frozen model and development-trained auxiliary models only; no model is trained on final labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from postdrift_auth import matched as M, matched_analysis as ma, matched_config as mc, pilot_model as pm
    cfg = mc.load(); lock = mc.load_lock(cfg)
    src = ma.Source(cfg)
    D1 = src.det["D1_rf_isotonic"]
    pcfg = cfg["_pilot_v2_cfg"]
    out = {"label": "EXPLORATORY / POST HOC: motivated by the final result; not part of the confirmatory analysis"}
    sets = {}
    for stage, label, kind in (("m12", "MFIN", "M12"), ("m3", "MM3", "M3")):
        seeds = list(range(*_block(stage)))
        data = M.load_families(os.path.join(RAW, stage), label, seeds)
        pairs = M.make_pairs(data)
        ln_p = src.nov_p.ln_n(data["X"])
        keep = M.caliper_keep(ln_p, pairs, lock["caliper_primary"])
        sets[stage] = (data, pairs, keep, M.FamilyStats(data, pairs))
    rng = np.random.default_rng(2026)

    def metrics(stage, score, B=0):
        data, pairs, keep, fs = sets[stage]
        w = M.pair_wins(score, pairs)
        return float(w[keep].mean()), float(M.auc_mw(score, data["y"]))

    def metrics_ci(stage, score, B=2000):
        data, pairs, keep, fs = sets[stage]
        w = M.pair_wins(score, pairs)
        boot = M.boot_family_index(cfg, fs.fams, f"audit/{stage}", B)
        b = fs.pmp_boot(w, keep, boot)
        return float(w[keep].mean()), [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))], float(M.auc_mw(score, data["y"]))

    groups = {"reconnects_60s": [6], "topic_novelty": [4], "sub_novel_count": [5], "structural (all three)": [4, 5, 6], "iat_cv": [0], "max_gap_ratio": [1], "log_rate_ratio": [2], "log_payload_ratio": [3],
              "magnitude (all four)": [0, 1, 2, 3]}
    # 1a. permutation importance of the FROZEN model on the final matched set (rows permuted jointly within a group; 30 repeats)
    # 1b. masking: group set to the median of the benign source sessions
    benign_med = np.median(src.train["X"][src.train["y"] == 0], axis=0)
    imp = {}
    for stage in ("m12", "m3"):
        data = sets[stage][0]
        base_pmp, base_auc = metrics(stage, D1(data["X"]))
        imp[stage] = {"baseline_PMP": base_pmp, "baseline_AUC_pooled": base_auc, "groups": {}}
        for g, idx in groups.items():
            drops_p, drops_a = [], []
            for _ in range(30):
                X = data["X"].copy()
                X[:, idx] = X[rng.permutation(len(X))][:, idx]
                p_, a_ = metrics(stage, D1(X))
                drops_p.append(base_pmp - p_); drops_a.append(base_auc - a_)
            Xm = data["X"].copy(); Xm[:, idx] = benign_med[idx]
            pm_, am_ = metrics(stage, D1(Xm))
            imp[stage]["groups"][g] = {"permutation_PMP_drop_mean": float(np.mean(drops_p)), "permutation_PMP_drop_sd": float(np.std(drops_p, ddof=1)), "permutation_AUC_drop_mean": float(np.mean(drops_a)),
                                       "mask_to_benign_median_PMP": pm_, "mask_PMP_drop": base_pmp - pm_, "mask_AUC_drop": base_auc - am_}
    out["frozen_model_importance"] = imp
    rows = [{"stage": st, "group": g, **v} for st, d in imp.items() for g, v in d["groups"].items()]
    write_csv("mechanism_frozen_model_importance.csv", rows)
    # 2. auxiliary models trained ONLY on the development source data (TRAIN-v2, isotonic on CAL-v2), evaluated unchanged on the final matched sets
    subsets = {"all 7 (= frozen D1, retrained check)": list(range(7)), "magnitude four only (structural three removed)": [0, 1, 2, 3], "structural three only": [4, 5, 6], "minus reconnects_60s": [0, 1, 2, 3, 4, 5],
               "minus topic_novelty": [0, 1, 2, 3, 5, 6], "minus sub_novel_count": [0, 1, 2, 3, 4, 6]}
    aux = {}
    for name, idx in subsets.items():
        rf, iso = pm.fit_pipeline(pcfg, src.train["X"][:, idx], src.train["y"], src.cal["X"][:, idx], src.cal["y"], pm.rs_state(pcfg, "rf/full"))
        aux[name] = {}
        for stage in ("m12", "m3"):
            X = sets[stage][0]["X"][:, idx]
            pmp, ci, auc = metrics_ci(stage, pm.score(rf, iso, X))
            aux[name][stage] = {"PMP": pmp, "PMP_ci95_family_clustered": ci, "AUC_pooled": auc}
    out["auxiliary_models_trained_on_TRAIN_CAL_only"] = aux
    write_csv("mechanism_auxiliary_models.csv", [{"model": n, "stage": st, "PMP": v["PMP"], "PMP_ci_low": v["PMP_ci95_family_clustered"][0], "PMP_ci_high": v["PMP_ci95_family_clustered"][1], "AUC_pooled": v["AUC_pooled"]}
                                                 for n, d in aux.items() for st, v in d.items()])
    # 3. feature and score distributions
    def grp_stats(X, s_):
        r = {}
        for j, n in enumerate(FEAT):
            q = np.percentile(X[:, j], [25, 50, 75])
            r[n] = {"mean": float(X[:, j].mean()), "median": float(q[1]), "iqr": [float(q[0]), float(q[2])], "share_zero": float((X[:, j] == 0).mean())}
        r["D1_score"] = {"mean": float(s_.mean()), "median": float(np.median(s_)), "iqr": [float(np.percentile(s_, 25)), float(np.percentile(s_, 75))]}
        r["share_all_structural_zero"] = float((X[:, 4:] == 0).all(axis=1).mean()); r["n"] = int(len(X))
        return r
    G = {}
    for stage in ("m12", "m3"):
        data, pairs, keep, fs = sets[stage]
        X = data["X"]; s_ = D1(X)
        if stage == "m12":
            G["benign twin (M1/M2 families)"] = (X[pairs["ben"]], s_[pairs["ben"]])
            for t in ("M1", "M2"):
                m = pairs["mal_scenario"] == t
                G[f"malicious {t} twin"] = (X[pairs["mal"][m]], s_[pairs["mal"][m]])
        else:
            G["benign twin (M3 families)"] = (X[pairs["ben"]], s_[pairs["ben"]])
            G["malicious M3 twin"] = (X[pairs["mal"]], s_[pairs["mal"]])
    # source regime context (development data; CAL is only used here for description)
    for scn in ("S0", "B1", "M1", "M2"):
        m = src.cal["scenario"] == scn
        G[f"source CAL {scn}"] = (src.cal["X"][m], D1(src.cal["X"][m]))
    out["group_distributions"] = {k: grp_stats(*v) for k, v in G.items()}
    rows = []
    for k, d in out["group_distributions"].items():
        for n in FEAT + ["D1_score"]:
            rows.append({"group": k, "variable": n, "n": d["n"], "mean": round(d[n]["mean"], 4), "median": round(d[n]["median"], 4), "iqr_low": round(d[n]["iqr"][0], 4), "iqr_high": round(d[n]["iqr"][1], 4)})
    write_csv("mechanism_group_distributions.csv", rows)
    # 3b. why is M3 ranked below its benign twin? swaps with the frozen model, retained M3 pairs
    data, pairs, keep, fs = sets["m3"]
    X = data["X"]; mal, ben = pairs["mal"][keep], pairs["ben"][keep]
    base = D1(X[mal]); sb = D1(X[ben])
    swaps = {"observed (malicious twin vs benign twin)": float(np.mean((base > sb) + 0.5 * (base == sb)))}
    for name, idx in (("malicious twin with the BENIGN twin's magnitude four features", [0, 1, 2, 3]), ("malicious twin with the BENIGN twin's rate and payload ratios only", [2, 3]),
                      ("malicious twin with the BENIGN twin's timing features iat_cv and max_gap_ratio", [0, 1]), ("malicious twin with the BENIGN twin's structural three", [4, 5, 6])):
        Xh = X[mal].copy(); Xh[:, idx] = X[ben][:, idx]
        sh = D1(Xh)
        swaps[name] = float(np.mean((sh > sb) + 0.5 * (sh == sb)))
    out["M3_counterfactual_swaps_frozen_model_retained_pairs"] = swaps
    ret = {}
    for nm, idxs in (("benign twin", ben), ("malicious M3 twin", mal)):
        ret[nm] = {"median_abs_log_rate_ratio": float(np.median(np.abs(X[idxs][:, 2]))), "median_abs_log_payload_ratio": float(np.median(np.abs(X[idxs][:, 3]))), "median_iat_cv": float(np.median(X[idxs][:, 0])),
                   "median_max_gap_ratio": float(np.median(X[idxs][:, 1])), "median_D1_score": float(np.median(D1(X[idxs]))), "mean_D1_score": float(np.mean(D1(X[idxs]))), "share_all_structural_zero": float((X[idxs][:, 4:] == 0).all(axis=1).mean())}
    out["M3_retained_pairs_magnitudes"] = ret
    # dose response of the frozen score to |log_rate_ratio| and to log_payload_ratio magnitude
    def binned(Xg, sg, j, edges):
        v = np.abs(Xg[:, j]); b = np.digitize(v, edges)
        return [{"bin": f"{(edges[i-1] if i else 0):.2f}-{(edges[i] if i < len(edges) else float('inf')):.2f}", "n": int((b == i).sum()), "mean_D1": float(sg[b == i].mean()) if (b == i).any() else None} for i in range(len(edges) + 1)]
    out["score_by_abs_log_rate_ratio"] = {k: binned(*G[k], 2, [0.05, 0.15, 0.3, 0.6]) for k in ("benign twin (M3 families)", "malicious M3 twin", "source CAL S0", "source CAL B1", "source CAL M2")}
    dump("mechanism_exploratory.json", out)
    # figures
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, stage, ttl in ((axes[0], "m12", "M1/M2 families"), (axes[1], "m3", "M3 families")):
        names = list(groups); vals = [imp[stage]["groups"][g]["permutation_PMP_drop_mean"] for g in names]; err = [imp[stage]["groups"][g]["permutation_PMP_drop_sd"] for g in names]
        ax.barh(range(len(names)), vals, xerr=err, color="#2a78d6"); ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis(); ax.axvline(0, color="#999", lw=0.8)
        ax.set_title(f"{ttl} (baseline PMP {imp[stage]['baseline_PMP']:.2f})", fontsize=9, loc="left"); ax.set_xlabel("drop in PMP when permuted (negative = PMP rises); 30 permutations, +- SD")
    fig.suptitle("EXPLORATORY: frozen-model permutation importance", fontsize=10, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mechanism_importance.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.2, 4.6)); names = list(subsets); y = np.arange(len(names))
    for off, st, col, lab in ((-0.15, "m12", "#2a78d6", "M1/M2 families"), (0.15, "m3", "#eb6834", "M3 families")):
        est = [aux[n][st]["PMP"] for n in names]; lo = [aux[n][st]["PMP_ci95_family_clustered"][0] for n in names]; hi = [aux[n][st]["PMP_ci95_family_clustered"][1] for n in names]
        ax.errorbar(est, y + off, xerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt="o", color=col, capsize=3, label=lab)
    ax.axvline(0.5, color="#bbb"); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis(); ax.set_xlabel("PMP on retained pairs (95% family-clustered CI)")
    ax.set_title("EXPLORATORY: auxiliary RF models trained on TRAIN/CAL only, evaluated unchanged", fontsize=9, loc="left"); ax.legend(frameon=False, loc="lower left"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mechanism_auxiliary_models.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6)); order = ["benign twin (M3 families)", "malicious M3 twin", "benign twin (M1/M2 families)", "malicious M1 twin", "malicious M2 twin"]
    cols = ["#9ec5f0", "#eb6834", "#2a78d6", "#d9a400", "#7a5195"]
    for ax, (j, n) in zip(axes.ravel(), list(enumerate(FEAT)) + [(None, "D1_score")]):
        data_ = [(G[k][1] if j is None else G[k][0][:, j]) for k in order]
        ax.boxplot(data_, tick_labels=["ben\n(M3 fam)", "M3", "ben\n(M1/M2 fam)", "M1", "M2"], showfliers=False, patch_artist=True, boxprops=dict(facecolor="#eef"))
        ax.set_title(n, fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle("EXPLORATORY: feature and score distributions of the twins (outliers hidden)", fontsize=10, x=0.01, ha="left"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig_mechanism_distributions.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    part_d()
    print("written to", OUT)
