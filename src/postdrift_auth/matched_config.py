"""Loader/validator for configs/matched_drift_v1.json and its lock file (written after the dry run, before any final data)."""
from __future__ import annotations

import hashlib
import json
import os

from . import pilot_config
from .seeds import MATCHED_RESERVATIONS, check_matched_reservations, split_of_seed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(ROOT, "configs", "matched_drift_v1.json")
LOCK_PATH = os.path.join(ROOT, "configs", "matched_drift_v1_lock.json")
# Data and result folders. The defaults follow the layout of this repository; the environment variables select other folders.
DATA_DIR = os.environ.get("MTE_DATA_DIR", os.path.join(ROOT, "data"))
SOURCE_ROOT = os.path.join(DATA_DIR, "source_regime")      # TRAIN and CAL runs of the source regime
RAW_ROOT = os.path.join(DATA_DIR, "matched_drift_v1")      # m12 and m3 twin families
OUT_ROOT = os.environ.get("MTE_OUT_DIR", os.path.join(ROOT, "outputs", "matched_drift_v1"))   # newly computed results
REF_ROOT = os.path.join(ROOT, "results", "matched_drift_v1")   # reference results shipped with the repository


def result_dir(name: str) -> str:
    """Folder `name` ("final", "stats_review", "exploratory") with newly computed results if it exists, otherwise the shipped reference results."""
    new = os.path.join(OUT_ROOT, name)
    return new if os.path.isdir(new) else os.path.join(REF_ROOT, name)
BLOCK_RESERVATION = {"dry_run": "matched_dev", "final_m12": "matched_final_m12", "final_m3": "matched_final_m3"}
STAGE_BLOCK = {"dry": "dry_run", "m12": "final_m12", "m3": "final_m3"}


class MatchedConfigError(ValueError):
    pass


def load(path: str = DEFAULT_PATH) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    cfg = json.loads(raw)
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()
    p2 = os.path.join(ROOT, cfg["pilot_v2_config"])
    cfg["_pilot_v2_cfg"] = pilot_config.load(p2)
    if cfg["_pilot_v2_cfg"]["_sha256"] != cfg["pilot_v2_config_sha256"]:
        raise MatchedConfigError("the Pilot-v2 config changed after the matched design was frozen")
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    check_matched_reservations()
    for name, blk in cfg["seed_blocks"].items():
        res = MATCHED_RESERVATIONS[BLOCK_RESERVATION[name]]
        lo, hi = res[blk["split"]]
        if not lo <= blk["base"] < hi or split_of_seed(blk["base"]) != blk["split"]:
            raise MatchedConfigError(f"{name}: base seed outside its reservation")
    if cfg["seed_blocks"]["dry_run"]["kind"] != "M12" or cfg["seed_blocks"]["final_m3"]["kind"] != "M3":
        raise MatchedConfigError("kinds: dry run must be M12; M3 only in the held-out block")
    if cfg["seed_blocks"]["dry_run"]["split"] in ("TEST-SEEN", "TEST-HELD-OUT"):
        raise MatchedConfigError("the dry run must not use TEST seeds")
    if cfg["novelty"]["min_retention"] <= 0 or cfg["novelty"]["k"] < 1:
        raise MatchedConfigError("invalid novelty settings")


def load_lock(cfg: dict, path: str = LOCK_PATH) -> dict:
    with open(path) as f:
        lock = json.load(f)
    if lock["config_sha256"] != cfg["_sha256"]:
        raise MatchedConfigError("lock file does not belong to this config")
    return lock


def family_seeds(cfg: dict, stage: str, lock: dict | None = None) -> list[int]:
    blk = cfg["seed_blocks"][STAGE_BLOCK[stage]]
    n = blk["n_families"]
    if n is None:
        if lock is None:
            raise MatchedConfigError(f"{stage}: number of families comes from the lock file")
        n = lock["F_m12"] if stage == "m12" else lock["F_m3"]
    lo, hi = MATCHED_RESERVATIONS[BLOCK_RESERVATION[STAGE_BLOCK[stage]]][blk["split"]]
    if blk["base"] + n > hi:
        raise MatchedConfigError(f"{stage}: {n} families do not fit in the reserved block")
    return [blk["base"] + i for i in range(n)]
