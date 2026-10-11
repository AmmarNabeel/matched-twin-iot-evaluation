"""Matched-drift analysis drivers: dry-run report (writes the lock) and the locked final analysis. Reads saved run outputs only."""
from __future__ import annotations

import datetime
import json
import os

import numpy as np

from . import matched as M, matched_config as mc, pilot_model as pm
from .matched_quality import check_matched_quality
from .scenarios import TWIN_KINDS


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


class Source:
    """Source regime and frozen detectors (Pilot-v2 TRAIN-v2 / CAL-v2); built once."""

    def __init__(self, cfg: dict):
        root = mc.SOURCE_ROOT
        self.train, self.cal = pm.load_split(root, cfg["source_data"]["train_split"]), pm.load_split(root, cfg["source_data"]["cal_split"])
        self.det = M.fit_detectors(self.train, self.cal, cfg)
        self.nov_p, self.nov_s = M.build_novelty(self.train, cfg, "primary"), M.build_novelty(self.train, cfg, "secondary")


def _scores(src: Source, data: dict, ln_p: np.ndarray) -> dict:
    X = data["X"]
    return {"D1_rf_isotonic": src.det["D1_rf_isotonic"](X), "D2_novelty_only": ln_p, "D3_isolation_forest": src.det["D3_isolation_forest"](X),
            "D4_logistic_regression": src.det["D4_logistic_regression"](X)}


def dry_run_report(cfg: dict, write_lock: bool = False) -> dict:
    seeds = mc.family_seeds(cfg, "dry")
    root = os.path.join(mc.RAW_ROOT, "dry")
    blk = cfg["seed_blocks"]["dry_run"]
    q = check_matched_quality(root, blk["label"], seeds, blk["kind"], os.path.join(root, "matched_manifest.json*"))
    _dump(os.path.join(mc.OUT_ROOT, "dry_run", "data_quality_report.json"), q)
    if not q["passed"]:
        return {"stopped": "dry-run data-quality gate failed", "failed": [c for c in q["checks"] if not c["ok"]]}
    src = Source(cfg)
    data = M.load_families(root, blk["label"], seeds)
    pairs = M.make_pairs(data)
    ln_p, ln_s = src.nov_p.ln_n(data["X"]), src.nov_s.ln_n(data["X"])
    benign = data["y"] == 0
    sd_p, sd_s = float(np.std(ln_p[benign], ddof=1)), float(np.std(ln_s[benign], ddof=1))
    c_p, c_s = cfg["novelty"]["caliper_sd_multiple"] * sd_p, cfg["novelty"]["caliper_sd_multiple"] * sd_s
    keep_p, keep_s = M.caliper_keep(ln_p, pairs, c_p), M.caliper_keep(ln_s, pairs, c_s)
    fs = M.FamilyStats(data, pairs)
    wins = M.pair_wins(src.det["D1_rf_isotonic"](data["X"]), pairs)
    s_f, n_f = fs.pmp_by_family(wins, keep_p)
    pmp_f = s_f / np.maximum(n_f, 1)
    ss = M.sample_size(pmp_f[n_f > 0], float(n_f.mean()), cfg)
    ret = float(keep_p.mean())
    gates = {"all_runs_all_ok_and_pair_invariants": q["passed"], "retention_primary": ret, "retention_ok": ret >= cfg["dry_run_gates"]["retention_at_least"]}
    by_type = {t: {"pairs": int((pairs["mal_scenario"] == t).sum()), "retained": int(keep_p[pairs["mal_scenario"] == t].sum())} for t in sorted(set(pairs["mal_scenario"]))}
    rep = {"seeds": seeds, "n_pairs": int(len(keep_p)), "sd_ln_n_benign_primary": sd_p, "sd_ln_n_benign_secondary": sd_s, "caliper_primary": c_p, "caliper_secondary": c_s,
           "retention_primary": ret, "retention_secondary": float(keep_s.mean()), "retained_by_malicious_type": by_type, "gates": gates, "sample_size": ss,
           "note": "dry-run detector results are used ONLY for the family-level variability; these families never enter the final analysis"}
    _dump(os.path.join(mc.OUT_ROOT, "dry_run", "dry_run_report.json"), rep)
    if write_lock:
        if not (gates["retention_ok"] and gates["all_runs_all_ok_and_pair_invariants"]):
            return {"stopped": "dry-run gates not met; lock not written", "gates": gates, "report": rep}
        lock = {"config_sha256": cfg["_sha256"], "caliper_primary": c_p, "caliper_secondary": c_s, "sd_ln_n_benign_primary": sd_p, "sd_ln_n_benign_secondary": sd_s,
                "dry_run_retention_primary": ret, "F_m12": ss["F_m12"], "F_m3": ss["F_m3"], "sample_size": ss, "dry_run_seeds": seeds,
                "dry_run_commit": q["commits"], "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        _dump(mc.LOCK_PATH, lock)
        rep["lock_written"] = True
    return rep


def analyze_families(cfg: dict, lock: dict, src: Source, data: dict, label: str, kind: str, B: int | None = None) -> dict:
    B = B or cfg["primary_analysis"]["n_boot"]
    pairs = M.make_pairs(data)
    fs = M.FamilyStats(data, pairs)
    ln_p, ln_s = src.nov_p.ln_n(data["X"]), src.nov_s.ln_n(data["X"])
    keep_p, keep_s = M.caliper_keep(ln_p, pairs, lock["caliper_primary"]), M.caliper_keep(ln_s, pairs, lock["caliper_secondary"])
    ones = np.ones(len(keep_p), dtype=bool)
    boot = M.boot_family_index(cfg, fs.fams, f"{label}/boot", B)
    scores = _scores(src, data, ln_p)
    sig = (data["X"][pairs["mal"]][:, M.STRUCT_IDX] != 0).any(axis=1)
    out = {"label": label, "kind": kind, "n_families": len(fs.fams), "n_pairs": int(len(keep_p)), "n_sessions": int(len(data["y"])),
           "retention": {"primary": float(keep_p.mean()), "secondary_7feature": float(keep_s.mean()),
                         "by_malicious_type": {t: {"pairs": int((pairs["mal_scenario"] == t).sum()), "retained": int(keep_p[pairs["mal_scenario"] == t].sum())} for t in sorted(set(pairs["mal_scenario"]))},
                         "per_family_min": float(min(keep_p[fs.pair_idx[f]].mean() for f in fs.fams)), "unmatched_pairs": int((~keep_p).sum())},
           "calipers": {"primary": lock["caliper_primary"], "secondary": lock["caliper_secondary"]}, "detectors": {}}
    wins_all = {}
    for name, sc in scores.items():
        w = M.pair_wins(sc, pairs)
        wins_all[name] = w
        pmp_pt, pmp_b = float(w[keep_p].mean()) if keep_p.any() else float("nan"), fs.pmp_boot(w, keep_p, boot)
        auc_pt, auc_b = M.auc_mw(sc, data["y"]), fs.auc_boot(sc, boot)
        d = {"PMP_primary": M.summarize(pmp_pt, pmp_b), "PMP_all_pairs": M.summarize(float(w.mean()), fs.pmp_boot(w, ones, boot)),
             "AUC_pooled": M.summarize(auc_pt, auc_b), "Delta_match": M.summarize(auc_pt - pmp_pt, auc_b - pmp_b),
             "PMP_secondary_7feature_novelty": M.summarize(float(w[keep_s].mean()) if keep_s.any() else float("nan"), fs.pmp_boot(w, keep_s, boot))}
        d["by_malicious_type"] = {t: M.summarize(float(w[keep_p & (pairs["mal_scenario"] == t)].mean()) if (keep_p & (pairs["mal_scenario"] == t)).any() else float("nan"),
                                                 fs.pmp_boot(w, keep_p & (pairs["mal_scenario"] == t), boot)) for t in sorted(set(pairs["mal_scenario"]))}
        d["by_structural_signature"] = {k: M.summarize(float(w[keep_p & (sig == flag)].mean()) if (keep_p & (sig == flag)).any() else float("nan"), fs.pmp_boot(w, keep_p & (sig == flag), boot))
                                        | {"retained_pairs": int((keep_p & (sig == flag)).sum())} for k, flag in (("signature_present", True), ("signature_absent", False))}
        out["detectors"][name] = d
    d1, d2 = wins_all["D1_rf_isotonic"], wins_all["D2_novelty_only"]
    diff_b = fs.pmp_boot(d1, keep_p, boot) - fs.pmp_boot(d2, keep_p, boot)
    out["P1_D1_PMP_gt_0.5"] = {"lower_97.5": out["detectors"]["D1_rf_isotonic"]["PMP_primary"]["lower_one_sided_97.5"],
                               "holds": bool(out["detectors"]["D1_rf_isotonic"]["PMP_primary"]["lower_one_sided_97.5"] > 0.5)}
    diff = M.summarize(float(d1[keep_p].mean() - d2[keep_p].mean()), diff_b)
    out["P2_D1_minus_D2_gt_0"] = {**diff, "holds": bool(diff["lower_one_sided_97.5"] > 0)}
    ceil_scores = M.ceiling_scores_lofo(data, cfg)
    wc = M.pair_wins(ceil_scores, pairs)
    out["ceiling_lofo_rf"] = M.summarize(float(wc[keep_p].mean()), fs.pmp_boot(wc, keep_p, boot))
    out["wins_per_family_D1"] = {int(f): [float(d1[fs.pair_idx[f]][keep_p[fs.pair_idx[f]]].sum()), int(keep_p[fs.pair_idx[f]].sum())] for f in fs.fams}
    return out, {"pairs": pairs, "keep_p": keep_p, "ln_p": ln_p, "ln_s": ln_s, "scores": scores, "data": data}


def decision(cfg: dict, res: dict) -> dict:
    mr = cfg["novelty"]["min_retention"]
    ret_ok = res["retention"]["primary"] >= mr
    p1, p2 = res["P1_D1_PMP_gt_0.5"]["holds"], res["P2_D1_minus_D2_gt_0"]["holds"]
    ceil_ok = res["ceiling_lofo_rf"]["lower_one_sided_97.5"] >= 0.60
    upper = res["detectors"]["D1_rf_isotonic"]["PMP_primary"]["upper_one_sided_97.5"]
    if p1 and p2 and ret_ok and ceil_ok:
        label = "SUPPORTED"
    elif ret_ok and upper < 0.55:
        label = "NOT_SUPPORTED"
    else:
        label = "INCONCLUSIVE"
    return {"label": label, "P1": p1, "P2": p2, "retention_ok": ret_ok, "retention": res["retention"]["primary"], "min_retention": mr, "ceiling_ok": ceil_ok,
            "ceiling_lower_97.5": res["ceiling_lofo_rf"]["lower_one_sided_97.5"], "D1_PMP_upper_97.5": upper}
