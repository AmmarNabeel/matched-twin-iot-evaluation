"""Stage 3A point score s and run-cluster bootstrap width w. Configuration: configs/pilot_stage3a.json.

TRAIN -> Random Forest; CAL -> isotonic calibration of the RF score; s = isotonic(RF(x)) is a source-regime calibrated point-risk estimate.
VAL labels are never used here. Labels enter only through the TRAIN/CAL ground-truth join, after feature extraction.
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from . import features
from .pilot_quality import run_dirs
from .seeds import derived_seed, draw

FEATURES = list(features.FEATURE_NAMES)


def load_split(root: str, split: str) -> dict:
    """Arrays for one split. `scenario` is returned for DESCRIPTIVE use after prediction only; it must never be a model input."""
    rid, cid, run_seed, y, scen, rows = [], [], [], [], [], []
    for d in run_dirs(root, split):
        run_id = os.path.basename(d)
        with open(os.path.join(d, "meta.json")) as f:
            seed = json.load(f)["master_seed"]
        gt = {}
        with open(os.path.join(d, "ground_truth.jsonl")) as f:
            for ln in f:
                g = json.loads(ln)
                gt[g["session_id"]] = g
        with open(os.path.join(d, "extract_out", "features.csv"), newline="") as f:   # features were extracted blind; Y joined here
            for r in csv.DictReader(f):
                g = gt[r["client_id"]]
                rid.append(run_id)
                cid.append(r["client_id"])
                run_seed.append(seed)
                y.append(g["y"])
                scen.append(g["scenario"])
                rows.append([float(r[k]) for k in FEATURES])
    return {"run_id": np.array(rid), "session_id": np.array(cid), "run_seed": np.array(run_seed), "y": np.array(y, dtype=int),
            "scenario": np.array(scen), "X": np.array(rows, dtype=float)}


def _rf(cfg: dict, random_state: int) -> RandomForestClassifier:
    c = cfg["rf"]
    return RandomForestClassifier(n_estimators=c["n_estimators"], min_samples_leaf=c["min_samples_leaf"], max_features=c["max_features"],
                                  max_depth=c["max_depth"], class_weight=c["class_weight"], bootstrap=c["bootstrap"], n_jobs=c["n_jobs"],
                                  random_state=random_state)


def _iso(cfg: dict) -> IsotonicRegression:
    c = cfg["isotonic"]
    return IsotonicRegression(increasing=c["increasing"], y_min=c["y_min"], y_max=c["y_max"], out_of_bounds=c["out_of_bounds"])


def rs_state(cfg: dict, label: str) -> int:
    return derived_seed(cfg["analysis_seed"], label) % (2 ** 31)


def fit_pipeline(cfg: dict, X_train, y_train, X_cal, y_cal, random_state: int):
    rf = _rf(cfg, random_state).fit(X_train, y_train)
    iso = _iso(cfg).fit(rf.predict_proba(X_cal)[:, 1], y_cal)
    return rf, iso


def score(rf, iso, X) -> np.ndarray:
    return iso.predict(rf.predict_proba(X)[:, 1])


def resample_runs(cfg: dict, b: int, which: str, run_ids: np.ndarray) -> np.ndarray:
    """Row indices of a with-replacement resample of whole runs (all sessions of a selected run kept, duplicates kept)."""
    uniq = sorted(set(run_ids))
    n = len(uniq)
    picks = [uniq[int(draw(cfg["analysis_seed"], f"boot/{b}/{which}", j) * n)] for j in range(n)]
    return np.concatenate([np.flatnonzero(run_ids == r) for r in picks])


def bootstrap_probabilities(cfg: dict, train: dict, cal: dict, X_val) -> np.ndarray:
    """M x N_val matrix of calibrated probabilities from the run-cluster bootstrap pipeline."""
    M = cfg["bootstrap"]["M"]
    P = np.empty((M, len(X_val)))
    for b in range(M):
        it = resample_runs(cfg, b, "train", train["run_id"])
        ic = resample_runs(cfg, b, "cal", cal["run_id"])
        rf, iso = fit_pipeline(cfg, train["X"][it], train["y"][it], cal["X"][ic], cal["y"][ic], rs_state(cfg, f"rf/boot/{b}"))
        P[b] = score(rf, iso, X_val)
    return P


def width(P: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = np.quantile(P, kappa, axis=0), np.quantile(P, 1 - kappa, axis=0)
    return hi - lo, lo, hi
