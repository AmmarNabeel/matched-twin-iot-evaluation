"""Publication figures for the matched-drift paper (main Figures 1-5, supplementary S1-S3).

Reads only saved outputs (results/matched_drift_v1/final and .../exploratory) and, for per-pair feature values, the raw feature and ground-truth files of the
final runs (read-only; nothing is refitted and no number is recomputed except medians/percentiles of feature values). Writes vector PDF and SVG plus
300 dpi PNG to outputs/matched_drift_v1/figures/, and every plotted value to outputs/matched_drift_v1/figures/figure_data.json.
Run:  python scripts/reproduce_figures.py
"""
import csv
import json
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import matched as M  # noqa: E402
from postdrift_auth import matched_config as mc  # noqa: E402

FIN, AUD = mc.result_dir("final"), mc.result_dir("exploratory")
OUT = os.path.join(mc.OUT_ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
J = lambda p: json.load(open(p))  # noqa: E731
cfg = mc.load()
lock = mc.load_lock(cfg)
P, M3 = J(os.path.join(FIN, "primary_m1m2_results.json")), J(os.path.join(FIN, "stress_m3_results.json"))
MECH = J(os.path.join(AUD, "mechanism_exploratory.json"))
IMP, AUX, SW = MECH["frozen_model_importance"], MECH["auxiliary_models_trained_on_TRAIN_CAL_only"], MECH["M3_counterfactual_swaps_frozen_model_retained_pairs"]
MIN_RET = cfg["novelty"]["min_retention"]
DET = ["D1_rf_isotonic", "D2_novelty_only", "D3_isolation_forest", "D4_logistic_regression"]
DNAME = {"D1_rf_isotonic": "D1 random forest (frozen)", "D2_novelty_only": "D2 novelty only", "D3_isolation_forest": "D3 Isolation Forest", "D4_logistic_regression": "D4 logistic regression"}
FEAT = ["iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio", "topic_novelty", "sub_novel_count", "reconnects_60s"]
BLUE, ORANGE, INK, GRID = "#0072B2", "#D55E00", "#222222", "#d0d0d0"  # Okabe-Ito blue and vermillion (colour-blind safe); shapes and fill also differ
FIGDATA = {}

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 8, "legend.fontsize": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42, "svg.fonttype": "none"})


def save(fig, name):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def forest(ax, rows, series, xlim, ref=0.5, ref_label="0.5 = chance", offsets=None):
    """rows: list of labels; series: list of dict(name, est, lo, hi, marker, fill, color)."""
    y = np.arange(len(rows))
    k = len(series)
    offs = offsets or np.linspace(-0.16 * (k - 1), 0.16 * (k - 1), k)
    for s, off in zip(series, offs):
        est, lo, hi = np.array(s["est"], float), np.array(s["lo"], float), np.array(s["hi"], float)
        ax.errorbar(est, y + off, xerr=[est - lo, hi - est], fmt=s["marker"], color=s["color"], mfc=s["color"] if s["fill"] else "white", mec=s["color"], ms=5.2, capsize=2.2, lw=1.0, label=s["name"], zorder=3)
    if ref is not None:
        ax.axvline(ref, color=INK, lw=0.8, ls="--", zorder=1)
        ax.text(ref, -0.72, ref_label, ha="center", va="bottom", fontsize=6.8, color=INK, bbox=dict(fc="white", ec="none", pad=0.6))
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.set_ylim(len(rows) - 0.4, -0.85)
    ax.set_xlim(*xlim)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.tick_params(axis="y", length=0)


def tag(ax, text, color=INK, x=0.0):
    ax.text(x, 1.04, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold", color=color)


# ------------------------------------------------------------------ Figure 1: framework (no numbers)
def fig1():
    """Matched-twin design: shared pre-drift session, two post-drift branches, matched (PMP) and pooled (AUC) summaries and their gap (no numbers)."""
    fig, ax = plt.subplots(figsize=(7.2, 3.55))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set_xlim(0, 100)
    ax.set_ylim(-5, 50)
    ax.axis("off")
    rng_ = np.random.default_rng(3)
    LB, LO, GR, MUTE = "#cfe3f3", "#f6d8c3", "#e4e4e4", "#444444"
    box = lambda x, y, w, h, fc, ec, lw=1.0: ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.8", fc=fc, ec=ec, lw=lw))  # noqa: E731
    arr = lambda p, q, col=INK, ls="-", rad=0.0, lw=0.9: ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw, color=col, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))  # noqa: E731
    # shared pre-drift session
    box(1, 20, 18, 10, GR, INK, 0.8)
    xs = np.cumsum(rng_.uniform(0.7, 1.2, 30)) + 1.6
    ax.vlines(xs[xs < 18.5], 22, 28, color="#555555", lw=0.7)
    ax.text(10, 32, "Shared pre-drift session\nof one identity", ha="center", va="bottom", fontsize=7.4)
    ax.text(10, 18.2, "same events, link condition,\nonset and trigger time", ha="center", va="top", fontsize=6.6, color=MUTE)
    # onset / trigger
    for xv, lab in ((21, r"onset $\tau$"), (46.5, r"trigger $t_a=\tau+30$ s")):
        ax.plot([xv, xv], [3.5, 44.5], ls="--", lw=0.8, color=INK)
        ax.text(xv, 46, lab, ha="center", fontsize=7.2)
    # post-drift branches
    box(22.5, 33, 23, 8, LB, BLUE)
    box(22.5, 7, 23, 8, LO, ORANGE)
    xb = np.cumsum(rng_.uniform(1.5, 2.2, 14)) + 23
    xm = np.cumsum(rng_.uniform(0.8, 1.3, 26)) + 23
    ax.vlines(xb[xb < 44.8], 35, 39, color=BLUE, lw=0.8)
    ax.vlines(xm[xm < 44.8], 8.6, 12.6, color=ORANGE, lw=0.8)
    xd = xm[xm < 44.8][17]
    ax.plot([xd], [13.6], marker="D", ms=4.4, color=ORANGE, mec="#7a3a00", mew=0.5, zorder=4)
    ax.text(34, 41.6, "Benign twin: B1 reconfiguration", ha="center", va="bottom", fontsize=6.8, color=BLUE)
    ax.text(34, 5.4, "Malicious twin: M1, M2 (M3 held out)", ha="center", va="top", fontsize=6.8, color="#a84800")
    arr((19.3, 28), (22.2, 35))
    arr((19.3, 22), (22.2, 13))
    ax.text(34, 29.2, "post-drift behaviour\ndrawn independently", ha="center", va="center", fontsize=6.4, color=MUTE)
    # shared baseline enters both scores
    ax.plot([19.3, 52.5], [25, 25], ls=":", lw=1.0, color="#777777")
    arr((52.5, 25), (52.5, 32.3), "#777777", ls=":", lw=1.0)
    arr((52.5, 25), (52.5, 15.7), "#777777", ls=":", lw=1.0)
    ax.text(33, 22.4, "pre-drift baseline enters both scores", ha="center", va="center", fontsize=6.4, color=MUTE, style="italic")
    # score boxes
    for yy, col, lab in ((37, BLUE, r"$s^{\mathrm{ben}},\ \ln N^{\mathrm{ben}}$"), (11, ORANGE, r"$s^{\mathrm{mal}},\ \ln N^{\mathrm{mal}}$")):
        box(48.2, yy - 4, 10.2, 8, "white", col)
        ax.text(53.3, yy + 0.8, lab, ha="center", va="center", fontsize=7)
        ax.text(53.3, yy - 2.2, "decision window", ha="center", va="center", fontsize=5.8, color=MUTE)
        ax.plot([58.4, 61.5], [yy, yy], color=col, lw=0.9)
    ax.plot([61.5, 61.5], [11, 37], color=INK, lw=0.9)
    # matched and pooled summaries
    box(65, 25, 19, 16, "#eef3f9", INK)
    ax.text(74.5, 33, "Novelty matching\n" + r"$d=\ln N^{\mathrm{mal}}-\ln N^{\mathrm{ben}}$" + "\n" + r"keep pair if $|d|\leq c$" + "\n\n" + r"PMP $=\Pr(s^{\mathrm{mal}}>s^{\mathrm{ben}})$" + "\non retained pairs", ha="center", va="center", fontsize=6.5, linespacing=1.25)
    box(65, 3, 19, 16, "#faf1e8", INK)
    ax.text(74.5, 11, "Pooled comparison\nall malicious vs. all benign\nsessions, no pairing\n\nPooled AUC", ha="center", va="center", fontsize=6.5, linespacing=1.25)
    arr((61.5, 31), (64.8, 31))
    arr((61.5, 14), (64.8, 14))
    box(87, 15.5, 12.6, 12, "white", INK)
    ax.text(93.3, 21.5, "Gap between\nthe two\n" + r"$\Delta_{\mathrm{match}}$" + "\n= AUC - PMP", ha="center", va="center", fontsize=6.6, linespacing=1.2)
    arr((84.3, 28), (86.8, 25.5), rad=-0.1)
    arr((84.3, 15), (86.8, 17.5), rad=0.1)
    # legend
    ax.vlines([1.6], -3.6, -0.6, color="#555555", lw=0.9)
    ax.text(3.3, -2.1, "one PUBLISH event; tick spacing = publish timing in one example realization", ha="left", va="center", fontsize=6.5, color=MUTE)
    ax.plot([60.5], [-2.1], marker="D", ms=4.4, color=ORANGE, mec="#7a3a00", mew=0.5)
    ax.text(62.3, -2.1, "structural cue (reconnect or new topic)", ha="left", va="center", fontsize=6.5, color=MUTE)
    save(fig, "fig1_framework")


def _scatter(ax, pdx, res, ttl):
    data, pairs, ln, d1_, keep = pdx
    c = lock["caliper_primary"]
    b, m = ln[pairs["ben"]], ln[pairs["mal"]]
    ax.scatter(b[~keep], m[~keep], s=5, c="#9a9a9a", marker="x", lw=0.45, label="unmatched pair")
    ax.scatter(b[keep], m[keep], s=6, c=BLUE, marker="o", alpha=0.55, lw=0, label="retained pair")
    xs = np.array([-1.6, 4.8])
    ax.plot(xs, xs, color=INK, lw=0.7)
    ax.plot(xs, xs + c, color=INK, lw=0.6, ls="--")
    ax.plot(xs, xs - c, color=INK, lw=0.6, ls="--")
    ax.set_xlim(-1.6, 4.8)
    ax.set_ylim(-1.6, 4.8)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\ln N$ of the benign twin")
    ax.set_title(f"{ttl}: {100 * res['retention']['primary']:.1f}% of pairs retained", loc="left", fontsize=7.5)
    ax.grid(color=GRID, lw=0.4)


def fig2(pd12, pd3):
    """Matching and primary result: (a, b) novelty of the twins with the caliper, (c) pooled AUC and matched PMP per score."""
    fig = plt.figure(figsize=(6.6, 5.15))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.5, wspace=0.28, left=0.07, right=0.985, top=0.95, bottom=0.09)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _scatter(a, pd12, P, "M1/M2 families")
    _scatter(b, pd3, M3, "M3 families (descriptive)")
    a.set_ylabel(r"$\ln N$ of the malicious twin")
    a.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper left", fontsize=6.8)
    a.text(0.0, 1.14, "(a)", transform=a.transAxes, fontweight="bold")
    b.text(0.0, 1.14, "(b)", transform=b.transAxes, fontweight="bold")
    c = fig.add_subplot(gs[1, :])
    y = np.arange(4)
    for i, k in enumerate(DET):
        au, pm_ = P["detectors"][k]["AUC_pooled"], P["detectors"][k]["PMP_primary"]
        c.plot([pm_["estimate"], au["estimate"]], [i, i], color="#888888", lw=2.0, zorder=1)
        c.errorbar(au["estimate"], i, xerr=[[au["estimate"] - au["ci95_two_sided"][0]], [au["ci95_two_sided"][1] - au["estimate"]]], fmt="o", color=BLUE, mfc="white", ms=6, capsize=2.5, lw=1.0, zorder=3, label="Pooled AUC, all sessions (conventional)" if i == 0 else None)
        c.errorbar(pm_["estimate"], i, xerr=[[pm_["estimate"] - pm_["ci95_two_sided"][0]], [pm_["ci95_two_sided"][1] - pm_["estimate"]]], fmt="s", color=ORANGE, ms=5.5, capsize=2.5, lw=1.0, zorder=3, label="PMP, novelty-matched pairs" if i == 0 else None)
        c.text(1.003, i, rf"$\Delta_{{\mathrm{{match}}}}={P['detectors'][k]['Delta_match']['estimate']:.3f}$", va="center", fontsize=7.5, transform=c.get_yaxis_transform())
    c.axvline(0.5, color=INK, lw=0.8, ls="--")
    c.text(0.5, -0.78, "0.5 = chance", ha="center", va="bottom", fontsize=6.8, bbox=dict(fc="white", ec="none", pad=0.6))
    c.set_yticks(y)
    c.set_yticklabels([DNAME[k] for k in DET])
    c.set_ylim(3.5, -0.9)
    c.set_xlim(0.45, 1.0)
    c.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    c.tick_params(axis="y", length=0)
    c.set_xlabel("Probability that the malicious twin is ranked above the benign twin (M1/M2 families)")
    c.legend(loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, frameon=False)
    c.text(0.0, 1.06, "(c) Pooled AUC and PMP of the four scores (D1, D2 confirmatory; D3, D4 descriptive)", transform=c.transAxes, fontweight="bold", fontsize=7.5)
    FIGDATA["fig2"] = {"auc": {k: P["detectors"][k]["AUC_pooled"] for k in DET}, "pmp": {k: P["detectors"][k]["PMP_primary"] for k in DET}, "P1": P["P1_D1_PMP_gt_0.5"], "P2": P["P2_D1_minus_D2_gt_0"], "retained_m12": int(pd12[4].sum()), "retained_m3": int(pd3[4].sum())}
    save(fig, "fig2_primary")


def fig3():
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 3.3), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.85})
    sig = {k: P["detectors"][k]["by_structural_signature"] for k in DET}
    pres = [sig[k]["signature_present"] for k in DET]
    absn = [sig[k]["signature_absent"] for k in DET]
    npres, nabs = pres[0]["retained_pairs"], absn[0]["retained_pairs"]
    ser = [dict(name=f"Structural change present (n = {npres} pairs)", est=[s["estimate"] for s in pres], lo=[s["ci95_two_sided"][0] for s in pres], hi=[s["ci95_two_sided"][1] for s in pres], marker="s", fill=True, color=ORANGE),
           dict(name=f"Structural change absent (n = {nabs} pairs)", est=[s["estimate"] for s in absn], lo=[s["ci95_two_sided"][0] for s in absn], hi=[s["ci95_two_sided"][1] for s in absn], marker="o", fill=False, color=BLUE)]
    forest(a, ["D1", "D2", "D3", "D4"], ser, (0.3, 1.02))
    a.set_xlabel("PMP on retained pairs")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, ncol=1)
    tag(a, "(a) By structural change: DESCRIPTIVE")
    names = ["all 7 (= frozen D1, retrained check)", "structural three only", "magnitude four only (structural three removed)", "minus reconnects_60s", "minus topic_novelty", "minus sub_novel_count"]
    lab = ["all seven features", "structural three only", "magnitude four only", "without reconnects_60s", "without topic_novelty", "without sub_novel_count"]
    est = [AUX[n]["m12"]["PMP"] for n in names]
    lo = [AUX[n]["m12"]["PMP_ci95_family_clustered"][0] for n in names]
    hi = [AUX[n]["m12"]["PMP_ci95_family_clustered"][1] for n in names]
    forest(b, lab, [dict(name="Auxiliary random forest, TRAIN/CAL only", est=est, lo=lo, hi=hi, marker="D", fill=True, color=INK)], (0.3, 1.02))
    b.set_xlabel("PMP on retained pairs")
    tag(b, "(b) Feature subsets: EXPLORATORY")
    FIGDATA["fig3"] = {"present": pres, "absent": absn, "aux_m12": dict(zip(lab, zip(est, lo, hi)))}
    save(fig, "fig3_structural")


# ------------------------------------------------------------------ Figure 4: permutation
def fig4():
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    order = ["structural (all three)", "reconnects_60s", "topic_novelty", "sub_novel_count", "magnitude (all four)", "iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio"]
    lab = ["structural, all three", "   reconnects_60s", "   topic_novelty", "   sub_novel_count", "magnitude, all four", "   iat_cv", "   max_gap_ratio", "   log_rate_ratio", "   log_payload_ratio"]
    ser = []
    for st, name, mk, fill, col in (("m12", f"M1/M2 families (baseline PMP {IMP['m12']['baseline_PMP']:.2f})", "s", True, ORANGE), ("m3", f"M3 families, retention below the minimum (baseline PMP {IMP['m3']['baseline_PMP']:.2f})", "o", False, BLUE)):
        est = [-IMP[st]["groups"][g]["permutation_PMP_drop_mean"] for g in order]
        sd = [IMP[st]["groups"][g]["permutation_PMP_drop_sd"] for g in order]
        ser.append(dict(name=name, est=est, lo=np.array(est) - sd, hi=np.array(est) + sd, marker=mk, fill=fill, color=col))
    forest(ax, lab, ser, (-0.45, 0.42), ref=0.0, ref_label="no change")
    ax.set_xlabel("Change in PMP after permuting the feature group (permuted − baseline; bars: ±1 SD over 30 permutations)")
    ax.xaxis.label.set_size(7.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.16), frameon=False, ncol=1)
    ax.annotate("PMP falls", xy=(-0.43, 8.55), fontsize=6.8, color="#555555", ha="left")
    ax.annotate("PMP rises", xy=(0.40, 8.55), fontsize=6.8, color="#555555", ha="right")
    tag(ax, "Frozen D1, permutation of feature groups: EXPLORATORY")
    FIGDATA["fig4"] = {"order": order, "m12": {g: IMP["m12"]["groups"][g] for g in order}, "m3": {g: IMP["m3"]["groups"][g] for g in order}}
    save(fig, "fig4_permutation")


# ------------------------------------------------------------------ per-pair data for Figures 5c, S2, S3
def pair_data(stage):
    label, tagn, base, n = ("MFIN", "primary_m1m2", cfg["seed_blocks"]["final_m12"]["base"], lock["F_m12"]) if stage == "m12" else ("MM3", "stress_m3", cfg["seed_blocks"]["final_m3"]["base"], lock["F_m3"])
    data = M.load_families(os.path.join(mc.RAW_ROOT, stage), label, list(range(base, base + n)))
    pairs = M.make_pairs(data)
    sc = {}
    with open(os.path.join(FIN, f"{tagn}_session_scores.csv"), newline="") as f:
        for r in csv.DictReader(f):
            sc[(int(r["family"]), r["arm"], r["session_id"])] = (float(r["ln_N_primary"]), float(r["D1_rf_isotonic"]))
    ln = np.array([sc[(int(f), a, s)][0] for f, a, s in zip(data["family"], data["arm"], data["session_id"])])
    d1 = np.array([sc[(int(f), a, s)][1] for f, a, s in zip(data["family"], data["arm"], data["session_id"])])
    keep = M.caliper_keep(ln, pairs, lock["caliper_primary"])
    return data, pairs, ln, d1, keep


def dotrange(ax, x, vals, color, marker, fill):
    q = np.percentile(vals, [5, 25, 50, 75, 95])
    ax.plot([x, x], [q[0], q[4]], color=color, lw=0.8)
    ax.plot([x, x], [q[1], q[3]], color=color, lw=3.2, solid_capstyle="butt")
    ax.plot([x], [q[2]], marker, color=color, mfc=color if fill else "white", mec=color if not fill else "white", mew=0.9, ms=6, zorder=4)
    return q.tolist()


# ------------------------------------------------------------------ Figure 5: M3 stress test
def fig5(pd3):
    data, pairs, ln, d1, keep = pd3
    fig = plt.figure(figsize=(7.16, 5.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.8, wspace=0.6, top=0.93, bottom=0.07, left=0.08, right=0.98)
    a = fig.add_subplot(gs[0, 0:2])
    b = fig.add_subplot(gs[0, 2:4])
    r12 = [P["detectors"][k]["PMP_primary"] for k in DET]
    r3 = [M3["detectors"][k]["PMP_primary"] for k in DET]
    ser = [dict(name="M1/M2 families (retention above the minimum)", est=[s["estimate"] for s in r12], lo=[s["ci95_two_sided"][0] for s in r12], hi=[s["ci95_two_sided"][1] for s in r12], marker="s", fill=True, color=ORANGE),
           dict(name=f"M3 families: DESCRIPTIVE (retention {100 * M3['retention']['primary']:.1f}% < {100 * MIN_RET:.0f}%)", est=[s["estimate"] for s in r3], lo=[s["ci95_two_sided"][0] for s in r3], hi=[s["ci95_two_sided"][1] for s in r3], marker="o", fill=False, color=BLUE)]
    forest(a, ["D1", "D2", "D3", "D4"], ser, (0.0, 1.0), ref_label="0.5")
    a.set_xlabel("PMP on retained pairs")
    a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), frameon=False, ncol=1, fontsize=7)
    tag(a, "(a) Ranking of the malicious twin")
    names = ["observed (malicious twin vs benign twin)", "malicious twin with the BENIGN twin's magnitude four features", "malicious twin with the BENIGN twin's rate and payload ratios only",
             "malicious twin with the BENIGN twin's timing features iat_cv and max_gap_ratio", "malicious twin with the BENIGN twin's structural three"]
    lab = ["observed", "magnitude four", "rate and payload", "iat_cv, max_gap", "structural three"]
    vals = [SW[n] for n in names]
    yy = np.arange(len(vals))
    b.axvline(0.5, color=INK, lw=0.8, ls="--")
    b.text(0.5, -0.72, "0.5", ha="center", va="bottom", fontsize=6.8, bbox=dict(fc="white", ec="none", pad=0.6))
    b.plot(vals, yy, "o", color=BLUE, mfc="white", ms=6, zorder=3)
    b.plot([vals[0], vals[0]], [-0.4, len(vals) - 0.6], color=GRID, lw=0.8, zorder=0)
    b.set_yticks(yy)
    b.set_yticklabels(lab, fontsize=7)
    b.set_ylim(len(vals) - 0.4, -0.85)
    b.set_xlim(0.0, 1.0)
    b.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    b.tick_params(axis="y", length=0)
    b.set_xlabel("PMP of frozen D1, retained M3 pairs\n(rows: block taken from the benign twin; point estimates)")
    tag(b, "(b) Swapping feature blocks: EXPLORATORY")
    # (c) distributions of retained M3 pairs
    mal, ben = pairs["mal"][keep], pairs["ben"][keep]
    X = data["X"]
    panels = [("|log_rate_ratio|", np.abs(X[:, 2])), ("|log_payload_ratio|", np.abs(X[:, 3])), ("iat_cv", X[:, 0]), ("D1 score", d1)]
    dist = {}
    for j, (nm, v) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        qb = dotrange(ax, 0, v[ben], BLUE, "o", False)
        qm = dotrange(ax, 1, v[mal], ORANGE, "s", True)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["benign\ntwin", "M3\ntwin"])
        ax.set_title(nm, fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_ylim(bottom=min(0, ax.get_ylim()[0]))
        dist[nm] = {"benign twin": qb, "M3 twin": qm}
        if j == 0:
            ax.set_ylabel("median, IQR, 5-95%")
    top_c = gs[1, 0].get_position(fig).y1
    fig.text(0.08, top_c + 0.055, "(c) Retained M3 pairs: magnitude features and score; EXPLORATORY", fontsize=8, fontweight="bold", ha="left")
    FIGDATA["fig5"] = {"pmp_m12": r12, "pmp_m3": r3, "swaps": SW, "distributions_retained_m3_pairs": dist, "n_retained_m3_pairs": int(keep.sum())}
    save(fig, "fig5_m3_stress")


# ------------------------------------------------------------------ supplementary
def figs1():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, res, ttl in ((axes[0], P, f"M1/M2 families ({P['n_families']})"), (axes[1], M3, f"M3 families ({M3['n_families']}), DESCRIPTIVE")):
        w = res["wins_per_family_D1"]
        v = np.sort([x[0] / x[1] for x in w.values() if x[1] > 0])
        ax.plot(np.arange(len(v)), v, "o", color=BLUE if res is M3 else ORANGE, mfc="white" if res is M3 else ORANGE, ms=4.5)
        ax.axhline(0.5, color=INK, lw=0.8, ls="--")
        est = res["detectors"]["D1_rf_isotonic"]["PMP_primary"]
        ax.axhline(est["estimate"], color="#777777", lw=0.9)
        ax.axhspan(est["ci95_two_sided"][0], est["ci95_two_sided"][1], color="#bbbbbb", alpha=0.35, lw=0)
        ax.set_xlabel("family, sorted by PMP")
        ax.set_ylim(0, 1.02)
        ax.set_xticks([])
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_title(ttl, loc="left", fontsize=8.5)
    axes[0].set_ylabel("PMP of D1 within one family")
    axes[0].text(0.01, 0.52, "0.5 = chance", fontsize=6.8, transform=axes[0].get_yaxis_transform())
    FIGDATA["figS1"] = {"m12": P["wins_per_family_D1"], "m3": M3["wins_per_family_D1"]}
    save(fig, "figS1_family_pmp")


def figs2(pd12, pd3):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True, gridspec_kw={"wspace": 0.1})
    c = lock["caliper_primary"]
    for ax, (data, pairs, ln, d1, keep), ttl, res in ((axes[0], pd12, "M1/M2 families", P), (axes[1], pd3, "M3 families (DESCRIPTIVE)", M3)):
        b, m = ln[pairs["ben"]], ln[pairs["mal"]]
        ax.scatter(b[~keep], m[~keep], s=6, c="#9a9a9a", marker="x", lw=0.5, label="unmatched pair")
        ax.scatter(b[keep], m[keep], s=7, c=BLUE, marker="o", alpha=0.55, lw=0, label="retained pair")
        xs = np.array([-1.6, 4.8])
        ax.plot(xs, xs, color=INK, lw=0.7)
        ax.plot(xs, xs + c, color=INK, lw=0.6, ls="--")
        ax.plot(xs, xs - c, color=INK, lw=0.6, ls="--")
        ax.set_xlim(-1.6, 4.8)
        ax.set_ylim(-1.6, 4.8)
        ax.set_aspect("equal")
        ax.set_xlabel("ln N of the benign twin")
        ax.set_title(f"{ttl}: {100 * res['retention']['primary']:.1f}% retained", loc="left", fontsize=8.5)
        ax.grid(color=GRID, lw=0.4)
    axes[0].set_ylabel("ln N of the malicious twin")
    axes[0].legend(frameon=False, loc="upper left", fontsize=7)
    FIGDATA["figS1"] = {"caliper_primary": c, "retained_m12": int(pd12[4].sum()), "retained_m3": int(pd3[4].sum())}
    save(fig, "figS1_novelty_pairs")


def figs3(pd12, pd3):
    groups = [("B", pd12, "ben", None), ("M1", pd12, "mal", "M1"), ("M2", pd12, "mal", "M2"), ("B'", pd3, "ben", None), ("M3", pd3, "mal", None)]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.6), gridspec_kw={"hspace": 0.62, "wspace": 0.5})
    out = {}
    cols = [BLUE, ORANGE, ORANGE, BLUE, ORANGE]
    marks = ["o", "s", "s", "o", "s"]
    fills = [False, True, True, False, True]
    for ax, (j, nm) in zip(axes.ravel(), list(enumerate(FEAT)) + [(None, "D1 score")]):
        out[nm] = {}
        for i, (lab, pdx, role, scn) in enumerate(groups):
            data, pairs, ln, d1, keep = pdx
            idx = pairs[role]
            if scn:
                idx = idx[pairs["mal_scenario"] == scn]
            v = d1[idx] if j is None else data["X"][idx, j]
            out[nm][lab.replace("\n", " ")] = dotrange(ax, i, v, cols[i], marks[i], fills[i])
        ax.set_xticks(range(5))
        ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
        ax.set_title(nm, fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xlim(-0.6, 4.6)
    fig.text(0.5, -0.02, "B: benign twin in M1/M2 families; M1, M2: malicious twins; B': benign twin in M3 families; M3: malicious M3 twin.  Median, IQR (thick) and 5-95% range (thin), all pairs.", ha="center", fontsize=7)
    FIGDATA["figS2"] = out
    save(fig, "figS2_feature_distributions")


if __name__ == "__main__":
    pd12, pd3 = pair_data("m12"), pair_data("m3")
    assert int(pd12[4].sum()) == round(P["retention"]["primary"] * P["n_pairs"]) and int(pd3[4].sum()) == round(M3["retention"]["primary"] * M3["n_pairs"]), "retained-pair counts differ from the locked analysis"
    fig1(); fig2(pd12, pd3); fig3(); fig4(); fig5(pd3); figs1(); figs3(pd12, pd3)
    json.dump(FIGDATA, open(os.path.join(OUT, "figure_data.json"), "w"), indent=1, default=float)
    print("figures written to", OUT, sorted(os.listdir(OUT)))
