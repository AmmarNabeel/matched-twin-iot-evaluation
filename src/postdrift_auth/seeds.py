"""Seed ranges, split registry and derived draws.

Derived draws: BLAKE2b-64 over a canonical encoding of (run_seed, component, index); the top 53 bits give a double in
[0,1). Same construction as `accounting.stepup_uniform` but with its own domain tag. Independent of PYTHONHASHSEED.
"""
from __future__ import annotations

import hashlib
import math

_DOMAIN = b"postdrift_auth/derived-draw/v1|"

# (split, first seed, exclusive upper bound). Ranges are disjoint by construction; `check_registry` asserts it.
SPLIT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("TRAIN", 1_000_000, 2_000_000),
    ("CAL", 2_000_000, 3_000_000),
    ("VAL", 3_000_000, 4_000_000),
    ("TEST-SEEN", 4_000_000, 5_000_000),
    ("TEST-HELD-OUT", 5_000_000, 6_000_000),
    ("SMOKE", 9_000_000, 10_000_000),
)


def check_registry(ranges=SPLIT_RANGES) -> None:
    spans = sorted((lo, hi, name) for name, lo, hi in ranges)
    for lo, hi, name in spans:
        if not lo < hi:
            raise ValueError(f"empty range for {name}")
    for (lo1, hi1, n1), (lo2, hi2, n2) in zip(spans, spans[1:]):
        if hi1 > lo2:
            raise ValueError(f"seed ranges of {n1} and {n2} overlap")


def split_of_seed(seed: int) -> str:
    check_registry()
    for name, lo, hi in SPLIT_RANGES:
        if lo <= seed < hi:
            return name
    raise ValueError(f"seed {seed} belongs to no split")


def draw(run_seed: int, component: str, index: int) -> float:
    """Deterministic u in [0,1) from (run_seed, component, index)."""
    if isinstance(run_seed, bool) or not isinstance(run_seed, int):
        raise TypeError("run_seed must be an int")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise TypeError("index must be a non-negative int")
    payload = _DOMAIN + f"{run_seed}|{component}|{index}".encode("utf-8")
    n = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
    return (n >> 11) / float(1 << 53)


def uniform(run_seed: int, component: str, index: int, lo: float, hi: float) -> float:
    return lo + (hi - lo) * draw(run_seed, component, index)


def log_uniform(run_seed: int, component: str, index: int, lo: float, hi: float) -> float:
    return math.exp(uniform(run_seed, component, index, math.log(lo), math.log(hi)))


def derived_seed(run_seed: int, component: str) -> int:
    """64-bit integer identifying the stream of draws of one component (for logging)."""
    payload = _DOMAIN + f"{run_seed}|{component}|seed".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


# Development-pilot reservations: disjoint blocks INSIDE the TRAIN/CAL/VAL ranges. Pilot v1 is preserved; Pilot v2 uses fresh seeds.
# TEST-SEEN / TEST-HELD-OUT ranges are never reserved or touched by development pilots.
DEV_RESERVATIONS: dict[str, dict[str, tuple[int, int]]] = {
    "pilot_v1": {"TRAIN": (1_000_000, 1_000_100), "CAL": (2_000_000, 2_000_100), "VAL": (3_000_000, 3_000_100)},
    "pilot_v2": {"TRAIN": (1_100_000, 1_100_100), "CAL": (2_100_000, 2_100_100), "VAL": (3_100_000, 3_100_100)},
}


def check_reservations(res=None) -> None:
    res = DEV_RESERVATIONS if res is None else res
    spans = []
    for name, per_split in res.items():
        for split, (lo, hi) in per_split.items():
            if not lo < hi:
                raise ValueError(f"{name}/{split}: empty reservation")
            if split_of_seed(lo) != split or split_of_seed(hi - 1) != split:
                raise ValueError(f"{name}/{split}: reservation leaves the {split} range")
            if split.startswith("TEST"):
                raise ValueError("development pilots may not reserve TEST seeds")
            spans.append((lo, hi, f"{name}/{split}"))
    spans.sort()
    for (l1, h1, n1), (l2, h2, n2) in zip(spans, spans[1:]):
        if h1 > l2:
            raise ValueError(f"reservations {n1} and {n2} overlap")


# Matched-drift experiment (configs/matched_drift_v1.json). The dry run lives in a VAL-range block, the final families in fresh TEST-SEEN /
# TEST-HELD-OUT blocks. No earlier artifact exists in these blocks. Dry-run families must never enter the final analysis.
MATCHED_RESERVATIONS: dict[str, dict[str, tuple[int, int]]] = {
    "matched_dev": {"VAL": (3_200_000, 3_200_100)},
    "matched_final_m12": {"TEST-SEEN": (4_100_000, 4_100_200)},
    "matched_final_m3": {"TEST-HELD-OUT": (5_100_000, 5_100_200)},
}


def check_matched_reservations() -> None:
    spans = []
    for name, per in MATCHED_RESERVATIONS.items():
        for split, (lo, hi) in per.items():
            if not lo < hi or split_of_seed(lo) != split or split_of_seed(hi - 1) != split:
                raise ValueError(f"{name}/{split}: bad reservation")
            spans.append((lo, hi, name))
    for name, per in DEV_RESERVATIONS.items():
        for split, (lo, hi) in per.items():
            spans.append((lo, hi, name))
    spans.sort()
    for (l1, h1, n1), (l2, h2, n2) in zip(spans, spans[1:]):
        if h1 > l2:
            raise ValueError(f"reservations {n1} and {n2} overlap")
