"""Matched-drift analysis (configs/matched_drift_v1.json). All definitions here are frozen before any final data exist.

Terminology: PMP = paired malicious-over-benign win probability = P(score of the malicious twin > score of its benign twin) over novelty-matched
twin pairs, ties count 1/2. AUC_pooled = ordinary AUC over all benign and malicious sessions of the same families (no matching).
Delta_match = AUC_pooled - PMP is a DESCRIPTIVE gap: the part of the conventional ranking performance that does not remain under the matched comparison.
"""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np
from scipy.stats import chi2, norm, rankdata
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import features, pilot_model as pm
from .seeds import derived_seed, draw

FEATURES = list(features.FEATURE_NAMES)
MAG_IDX = [FEATURES.index(n) for n in ("iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio")]
ALL_IDX = list(range(len(FEATURES)))
STRUCT_IDX = [FEATURES.index(n) for n in ("topic_novelty", "sub_novel_count", "reconnects_60s")]


# ---------------------------------------------------------------- data
def run_dir(root: str, label: str, seed: int, arm: str) -> str:
    return os.path.join(root, f"{label}-{seed}-{arm}")


def load_families(root: str, label: str, seeds: list[int]) -> dict:
    """Sessions of both arms of the given families. Y/scenario are joined from the ground truth AFTER extraction."""
    cols = {k: [] for k in ("family", "arm", "session_id", "pair_id", "y", "scenario")}
    rows = []
    for sd in seeds:
        for arm in ("A", "B"):
            d = run_dir(root, label, sd, arm)
            with open(os.path.join(d, "ground_truth.jsonl")) as f:
                gt = {g["session_id"]: g for g in map(json.loads, f) if g}
            with open(os.path.join(d, "extract_out", "features.csv"), newline="") as f:
                for r in csv.DictReader(f):
                    g = gt[r["client_id"]]
                    cols["family"].append(sd)
                    cols["arm"].append(arm)
                    cols["session_id"].append(r["client_id"])
                    cols["pair_id"].append(g["pair_id"])
                    cols["y"].append(g["y"])
                    cols["scenario"].append(g["scenario"])
                    rows.append([float(r[k]) for k in FEATURES])
    out = {k: np.array(v) for k, v in cols.items()}
    out["X"] = np.array(rows, dtype=float)
    out["y"] = out["y"].astype(int)
    return out


def make_pairs(data: dict) -> dict:
    """One row per identity: index of the benign session and of the malicious session (the two branches of the same pre-drift session)."""
    by = {}
    for i, p in enumerate(data["pair_id"]):
        by.setdefault(p, []).append(i)
    ben, mal, fam = [], [], []
    for p in sorted(by):
        idx = by[p]
        if len(idx) != 2 or sorted(data["y"][idx]) != [0, 1]:
            raise ValueError(f"pair {p} is not one benign + one malicious session: {idx}")
        b, m = (idx[0], idx[1]) if data["y"][idx[0]] == 0 else (idx[1], idx[0])
        ben.append(b)
        mal.append(m)
        fam.append(data["family"][b])
    mal = np.array(mal)
    return {"ben": np.array(ben), "mal": mal, "family": np.array(fam), "mal_scenario": data["scenario"][mal]}


# ---------------------------------------------------------------- novelty
class Novelty:
    """ln of the mean distance to the k nearest REFERENCE sessions in a standardized feature subspace.

    Reference = benign (S0 and B1) sessions of Pilot-v2 TRAIN-v2. Standardization: per feature mean and sample SD (ddof=1) of the reference set,
    SD floored (count features: 1.0, others 1e-6). Euclidean distance. Evaluated sessions are never part of the reference set.
    """

    def __init__(self, ref_X: np.ndarray, idx: list[int], k: int, sd_floor: dict[str, float]):
        self.idx, self.k = idx, k
        r = ref_X[:, idx]
        self.mu = r.mean(axis=0)
        floor = np.array([sd_floor.get(FEATURES[j], 1e-6) for j in idx])
        self.sd = np.maximum(r.std(axis=0, ddof=1), floor)
        self.ref = (r - self.mu) / self.sd

    def ln_n(self, X: np.ndarray) -> np.ndarray:
        z = (X[:, self.idx] - self.mu) / self.sd
        d = np.sqrt(((z[:, None, :] - self.ref[None, :, :]) ** 2).sum(axis=2))
        n = np.sort(d, axis=1)[:, :self.k].mean(axis=1)
        return np.log(np.maximum(n, 1e-6))


def build_novelty(train: dict, cfg: dict, which: str) -> Novelty:
    c = cfg["novelty"]
    benign = np.isin(train["scenario"], c["reference_scenarios"])
    idx = MAG_IDX if which == "primary" else ALL_IDX
    return Novelty(train["X"][benign], idx, c["k"], c["sd_floor"])


# ---------------------------------------------------------------- metrics
def pair_wins(score: np.ndarray, pairs: dict) -> np.ndarray:
    sm, sb = score[pairs["mal"]], score[pairs["ben"]]
    return (sm > sb).astype(float) + 0.5 * (sm == sb)


def caliper_keep(ln_n: np.ndarray, pairs: dict, c: float) -> np.ndarray:
    return np.abs(ln_n[pairs["mal"]] - ln_n[pairs["ben"]]) <= c


def auc_mw(score: np.ndarray, y: np.ndarray) -> float:
    r = rankdata(score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_family_index(cfg: dict, families: list, comp: str, B: int) -> np.ndarray:
    fams = sorted(set(families))
    n = len(fams)
    seed = cfg["analysis_seed"]
    return np.array([[int(draw(seed, f"{comp}/{b}", j) * n) for j in range(n)] for b in range(B)])


class FamilyStats:
    """Per-family sufficient statistics so that cluster-bootstrap replicates are cheap for PMP; AUC is recomputed from resampled sessions."""

    def __init__(self, data: dict, pairs: dict):
        self.data, self.pairs = data, pairs
        self.fams = sorted(set(data["family"]))
        self.sess_idx = {f: np.flatnonzero(data["family"] == f) for f in self.fams}
        self.pair_idx = {f: np.flatnonzero(pairs["family"] == f) for f in self.fams}

    def pmp_by_family(self, wins, keep):
        return np.array([(wins[self.pair_idx[f]] * keep[self.pair_idx[f]]).sum() for f in self.fams]), np.array([keep[self.pair_idx[f]].sum() for f in self.fams])

    def pmp_boot(self, wins, keep, boot_idx):
        s, n = self.pmp_by_family(wins, keep)
        num, den = s[boot_idx].sum(axis=1), n[boot_idx].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def auc_boot(self, score, boot_idx):
        y = self.data["y"]
        out = []
        for row in boot_idx:
            ii = np.concatenate([self.sess_idx[self.fams[j]] for j in row])
            out.append(auc_mw(score[ii], y[ii]))
        return np.array(out)


def summarize(point: float, boot: np.ndarray) -> dict:
    return {"estimate": float(point), "ci95_two_sided": [float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))],
            "lower_one_sided_97.5": float(np.nanpercentile(boot, 2.5)), "upper_one_sided_97.5": float(np.nanpercentile(boot, 97.5)),
            "boot_sd": float(np.nanstd(boot, ddof=1)), "n_boot": int(len(boot))}


# ---------------------------------------------------------------- detectors
def fit_detectors(train: dict, cal: dict, cfg: dict) -> dict:
    """D1 frozen RF+isotonic (Pilot-v2 config); D3 IsolationForest on benign TRAIN-v2; D4 logistic regression on TRAIN-v2. D2 (novelty) is computed separately."""
    pcfg = cfg["_pilot_v2_cfg"]
    rf, iso = pm.fit_pipeline(pcfg, train["X"], train["y"], cal["X"], cal["y"], pm.rs_state(pcfg, "rf/full"))
    d = cfg["detectors"]
    benign = train["y"] == 0
    iforest = IsolationForest(n_estimators=d["isolation_forest"]["n_estimators"], contamination="auto",
                              random_state=derived_seed(cfg["analysis_seed"], "iforest") % (2 ** 31)).fit(train["X"][benign])
    sc = StandardScaler().fit(train["X"])
    lr = LogisticRegression(C=d["logistic_regression"]["C"], max_iter=d["logistic_regression"]["max_iter"]).fit(sc.transform(train["X"]), train["y"])
    return {"D1_rf_isotonic": lambda X: pm.score(rf, iso, X), "D3_isolation_forest": lambda X: -iforest.score_samples(X),
            "D4_logistic_regression": lambda X: lr.predict_proba(sc.transform(X))[:, 1], "_rf": rf}


def ceiling_scores_lofo(data: dict, cfg: dict) -> np.ndarray:
    """Out-of-family RF probabilities: how well the 7 features can separate the twins at all (diagnostic ceiling, not a baseline)."""
    pcfg = cfg["_pilot_v2_cfg"]
    out = np.zeros(len(data["y"]))
    for f in sorted(set(data["family"])):
        te = data["family"] == f
        rf = pm._rf(pcfg, derived_seed(cfg["analysis_seed"], f"ceiling/{f}") % (2 ** 31)).fit(data["X"][~te], data["y"][~te])
        out[te] = rf.predict_proba(data["X"][te])[:, 1]
    return out


# ---------------------------------------------------------------- sample size (pre-specified rule)
def sample_size(dry_family_pmp: np.ndarray, mean_retained_pairs: float, cfg: dict) -> dict:
    s = cfg["sample_size"]
    df = len(dry_family_pmp) - 1
    sd_obs = float(np.std(dry_family_pmp, ddof=1)) if df >= 1 else float("nan")
    q = chi2.ppf(1 - s["sd_upper_confidence"], df)                     # lower chi-square quantile => upper confidence limit of the SD
    sd_up = sd_obs * math.sqrt(df / q) if df >= 1 else float("nan")
    floor = math.sqrt(0.25 / max(mean_retained_pairs, 1.0))           # sd of a mean of independent pairs at p = 0.5
    sd_used = max(sd_up if sd_up == sd_up else 0.0, floor)
    z_a, z_b = norm.ppf(1 - s["alpha_one_sided"]), norm.ppf(s["power"])
    f_power = math.ceil(((z_a + z_b) * sd_used / s["target_effect"]) ** 2)
    f_prec12 = math.ceil((norm.ppf(0.975) * sd_used / s["halfwidth_m12"]) ** 2)
    f_prec3 = math.ceil((norm.ppf(0.975) * sd_used / s["halfwidth_m3"]) ** 2)
    return {"family_pmp_dry": [float(x) for x in dry_family_pmp], "sd_obs": sd_obs, "sd_upper_limit": sd_up, "independent_pairs_floor": floor, "sd_used": sd_used,
            "mean_retained_pairs_per_family": mean_retained_pairs, "F_power": f_power, "F_precision_m12": f_prec12, "F_precision_m3": f_prec3,
            "F_m12": max(f_power, f_prec12, s["min_families"]), "F_m3": max(f_prec3, s["min_families"]),
            "rule": "F_m12 = max(F_power, F_precision_m12, min_families); F_m3 = max(F_precision_m3, min_families); SD_used = max(upper confidence limit of the SD across the dry-run families, sqrt(0.25 / mean retained pairs per family))"}
