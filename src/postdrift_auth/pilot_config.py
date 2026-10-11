"""Loader and consistency checks for configs/pilot_stage3a.json (Stage 3A). The config cannot drift from the code constants."""
from __future__ import annotations

import hashlib
import json
import os

from . import features, scenarios
from .seeds import DEV_RESERVATIONS, SPLIT_RANGES, check_registry, check_reservations, split_of_seed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(ROOT, "configs", "pilot_stage3a.json")
PILOT_SPLITS = ("TRAIN", "CAL", "VAL")


class ConfigError(ValueError):
    pass


def load(path: str = DEFAULT_PATH) -> dict:
    with open(path, "rb") as f:
        raw = f.read()
    cfg = json.loads(raw)
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()
    cfg.setdefault("variant", "v1")                                  # the frozen Pilot-v1 config predates this key
    cfg.setdefault("output_root", "results/raw/pilot_stage3a")
    cfg.setdefault("results_dir", "results/pilot_stage3a")
    validate(cfg)
    return cfg


def run_seeds(cfg: dict, split: str) -> list[int]:
    s = cfg["splits"][split]
    return [s["seed_base"] + i for i in range(s["n_runs"])]


def validate(cfg: dict) -> None:
    check_registry()
    check_reservations()
    variant = cfg.get("variant", "v1")
    if variant not in scenarios.SPLIT_COMPOSITION_BY_VARIANT:
        raise ConfigError(f"unknown variant {variant!r}")
    reservation = DEV_RESERVATIONS["pilot_" + variant]
    if set(cfg["splits"]) != set(PILOT_SPLITS):
        raise ConfigError(f"pilot splits must be exactly {PILOT_SPLITS}, got {sorted(cfg['splits'])}")
    for forbidden in cfg["forbidden"]:
        if forbidden in cfg["splits"] or forbidden in str(cfg["splits"]):
            raise ConfigError(f"{forbidden} must not appear in the pilot")
    seen: set[int] = set()
    for split, s in cfg["splits"].items():
        quota = dict(scenarios.SPLIT_COMPOSITION_BY_VARIANT[variant][split][0])
        if s["quota"] != quota or sum(quota.values()) != s["sessions_per_run"]:
            raise ConfigError(f"{split}: quota in config {s['quota']} differs from the frozen composition {quota}")
        if "M3" in s["quota"]:
            raise ConfigError("M3 is forbidden in the pilot")
        seeds = run_seeds(cfg, split)
        for sd in seeds:
            if split_of_seed(sd) != split:
                raise ConfigError(f"seed {sd} is outside the {split} range")
            if not reservation[split][0] <= sd < reservation[split][1]:
                raise ConfigError(f"seed {sd} is outside the {variant} reservation {reservation[split]}")
        if seen & set(seeds):
            raise ConfigError("run seeds overlap between splits")
        seen |= set(seeds)
    if tuple(cfg["features"]["names"]) != features.FEATURE_NAMES or cfg["features"]["version"] != features.FEATURE_CONFIG["version"]:
        raise ConfigError("feature definition in config differs from the code")
    try:
        split_of_seed(cfg["analysis_seed"])
        raise ConfigError("analysis_seed must lie outside every split's seed range")
    except ValueError as e:
        if isinstance(e, ConfigError):
            raise
    if cfg["bootstrap"]["M"] < 1 or not 0 < cfg["bootstrap"]["kappa"] < 0.5:
        raise ConfigError("invalid bootstrap parameters")
    if "adequacy_gate" in cfg and cfg["adequacy_gate"].get("rule") != "brier_s_less_than_brier_constant_development_prevalence":
        raise ConfigError("unknown adequacy gate rule")
    if sorted(cfg["h1_verdict_rule"]) != sorted(["SUPPORTED", "NOT_SUPPORTED", "WEAK", "exactly_one"]):
        raise ConfigError("verdict rule incomplete")
