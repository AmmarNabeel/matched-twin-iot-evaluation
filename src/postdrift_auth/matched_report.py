"""Tables and figures for the matched-drift experiment, generated from the saved analysis outputs only."""
from __future__ import annotations

import csv
import os

import numpy as np

TITLES = {"primary_m1m2": "M1/M2 twin families (37 families)", "stress_m3": "Held-out M3 twin families (20 families)"}
NAMES = {"D1_rf_isotonic": "D1 RF + isotonic (frozen)", "D2_novelty_only": "D2 novelty only", "D3_isolation_forest": "D3 Isolation Forest", "D4_logistic_regression": "D4 logistic regression"}


def _row(prefix, s):
    return {f"{prefix}": round(s["estimate"], 4), f"{prefix}_ci_low": round(s["ci95_two_sided"][0], 4), f"{prefix}_ci_high": round(s["ci95_two_sided"][1], 4)}


def write_tables(res: dict, out_dir: str, tag: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    files = []

    def w(name, rows):
        p = os.path.join(out_dir, f"{tag}_{name}.csv")
        with open(p, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        files.append(p)

    rows = []
    for k, d in res["detectors"].items():
        rows.append({"detector": NAMES[k], **_row("PMP", d["PMP_primary"]), **_row("AUC_pooled", d["AUC_pooled"]), **_row("Delta_match", d["Delta_match"]),
                     **_row("PMP_all_pairs", d["PMP_all_pairs"]), **_row("PMP_7feature_novelty", d["PMP_secondary_7feature_novelty"])})
    w("detectors", rows)
    rows = [{"detector": NAMES[k], "malicious_type": t, **_row("PMP", s)} for k, d in res["detectors"].items() for t, s in d["by_malicious_type"].items()]
    w("by_malicious_type", rows)
    rows = [{"detector": NAMES[k], "stratum": t, "retained_pairs": s["retained_pairs"], **_row("PMP", s)} for k, d in res["detectors"].items() for t, s in d["by_structural_signature"].items()]
    w("by_structural_signature", rows)
    r = res["retention"]
    rows = [{"malicious_type": t, "pairs": v["pairs"], "retained": v["retained"], "retention": round(v["retained"] / max(v["pairs"], 1), 4)} for t, v in r["by_malicious_type"].items()]
    rows.append({"malicious_type": "ALL", "pairs": res["n_pairs"], "retained": res["n_pairs"] - r["unmatched_pairs"], "retention": round(r["primary"], 4)})
    w("retention", rows)
    w("tests", [{"test": "P1 PMP(D1) > 0.5 (lower 97.5%)", "value": round(res["P1_D1_PMP_gt_0.5"]["lower_97.5"], 4), "holds": res["P1_D1_PMP_gt_0.5"]["holds"]},
                {"test": "P2 PMP(D1) - PMP(D2) > 0 (lower 97.5%)", "value": round(res["P2_D1_minus_D2_gt_0"]["lower_one_sided_97.5"], 4), "holds": res["P2_D1_minus_D2_gt_0"]["holds"]},
                {"test": "ceiling PMP (lower 97.5%)", "value": round(res["ceiling_lofo_rf"]["lower_one_sided_97.5"], 4), "holds": ""}])
    return files


def write_figures(res: dict, arrays: dict, out_dir: str, tag: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blue, orange, ink, grid = "#2a78d6", "#eb6834", "#0b0b0b", "#d9d8d3"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": grid, "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(out_dir, exist_ok=True)
    files = []
    names = list(res["detectors"])
    # 1. matched versus conventional performance
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = np.arange(len(names))
    for off, key, col, mk, lab in ((-0.12, "AUC_pooled", blue, "o", "pooled AUC (conventional, all pairs)"), (0.12, "PMP_primary", orange, "s", "PMP (novelty-matched pairs)")):
        est = [res["detectors"][n][key]["estimate"] for n in names]
        lo = [res["detectors"][n][key]["ci95_two_sided"][0] for n in names]
        hi = [res["detectors"][n][key]["ci95_two_sided"][1] for n in names]
        ax.errorbar(est, y + off, xerr=[np.subtract(est, lo), np.subtract(hi, est)], fmt=mk, color=col, capsize=3, label=lab)
    ax.axvline(0.5, color=grid, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([NAMES[n] for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("ranking probability (0.5 = chance); bars: 95% family-clustered interval")
    ax.set_title(f"{TITLES.get(tag, tag)}: conventional versus matched performance", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    ax.grid(axis="x", color=grid, linewidth=0.6)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_pmp_vs_auc.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    files.append(p)
    # 2. family-level PMP of D1
    fam = res["wins_per_family_D1"]
    vals = [v[0] / v[1] for v in fam.values() if v[1] > 0]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(np.random.default_rng(0).uniform(-0.08, 0.08, len(vals)), vals, "o", color=blue, alpha=0.8)
    ax.axhline(0.5, color=grid)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_ylabel("PMP of D1 in one family")
    ax.set_title(f"{TITLES.get(tag, tag)}: PMP of D1 per family", loc="left", fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_family_pmp.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(p)
    # 3. novelty of benign versus malicious twins, matched pairs marked
    d, ln, pairs, keep = arrays["data"], arrays["ln_p"], arrays["pairs"], arrays["keep_p"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.scatter(ln[pairs["ben"]][~keep], ln[pairs["mal"]][~keep], s=10, c=grid, marker="x", label="unmatched pair")
    ax.scatter(ln[pairs["ben"]][keep], ln[pairs["mal"]][keep], s=12, c=blue, marker="o", alpha=0.6, label="retained pair")
    lim = [min(ln.min(), -1), ln.max()]
    ax.plot(lim, lim, color=ink, linewidth=0.8)
    ax.set_xlabel("ln N of the benign twin")
    ax.set_ylabel("ln N of the malicious twin")
    ax.set_title(f"{TITLES.get(tag, tag)}: novelty of twin pairs", loc="left", fontsize=9)
    ax.legend(frameon=False)
    fig.tight_layout()
    p = os.path.join(out_dir, f"{tag}_novelty_pairs.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    files.append(p)
    return files
