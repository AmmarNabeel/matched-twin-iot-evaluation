"""The 7 frozen primary features. Pure function of a trace and (client_id, t_a).

ARCHITECTURAL SEPARATION: this module performs no I/O and knows nothing about labels, scenarios or roles. Its only inputs are
`Event` tuples (from the canonical PCAP trace) plus the identity's client_id and the trigger time t_a (seconds since run
start T0). Every event with t > t_a is discarded before any computation.

Interval conventions (one convention, applied everywhere):
  decision window   W = [t_a - 30, t_a]       closed on both ends   (an event exactly at t_a IS used)
  baseline          B = [t_a - 150, t_a - 30) closed left, OPEN right (an event exactly at t_a-30 is in W, not in B)
  history sets      H = [0, t_a - 30)         topics / SUBSCRIBE filters seen before the window (open right, like B)
  reconnect window  R = [t_a - 60, t_a]       closed on both ends
Conventions RATIFIED in Stage 2C (frozen, not to be changed after pilot results): sample standard deviation (ddof=1) for the CV;
max_gap_ratio = 0 when W has < 2 PUBLISHes (the edge case is reported by `extract_features_with_flags`); sub_novel_count counts
distinct filters; payload size = MQTT application payload length in bytes. The DUP flag is parsed by the trace layer for
diagnostics only; it is not a feature.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import NamedTuple, Sequence

WINDOW = 30.0
BASELINE = 150.0
RECONNECT = 60.0
FEATURE_NAMES = ("iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio",
                 "topic_novelty", "sub_novel_count", "reconnects_60s")
FEATURE_CONFIG = {
    "version": "v1.3/stage2C-7features",
    "names": list(FEATURE_NAMES),
    "window": "[t_a-30, t_a] closed", "baseline": "[t_a-150, t_a-30) right-open", "history": "[0, t_a-30) right-open",
    "reconnect": "[t_a-60, t_a] closed", "cv_ddof": 1, "cv_min_publishes": 3, "max_gap_if_lt2_publishes": 0.0,
    "rate_eps": 0.01, "payload": "mqtt application payload bytes", "sub_novel": "distinct filters in W not in H",
}


class Event(NamedTuple):
    t: float                 # seconds since run start T0
    kind: str                # "CONNECT" | "SUBSCRIBE" | "PUBLISH"
    client_id: str
    topic: str | None        # PUBLISH topic or SUBSCRIBE filter
    size: int | None         # PUBLISH payload bytes
    dup: bool | None         # PUBLISH DUP flag


class FeatureUndefined(ValueError):
    """The baseline needed by a ratio feature is empty; never silently replaced by a default."""


def config_hash() -> str:
    return hashlib.sha256(json.dumps(FEATURE_CONFIG, sort_keys=True).encode()).hexdigest()


def _mean(x: Sequence[float]) -> float:
    return sum(x) / len(x)


def extract_features(events: Sequence[Event], client_id: str, t_a: float) -> dict[str, float]:
    return extract_features_with_flags(events, client_id, t_a)[0]


def extract_features_with_flags(events: Sequence[Event], client_id: str, t_a: float) -> tuple[dict[str, float], list[str]]:
    """Features plus the list of edge-case conventions that were applied for this session (empty if none)."""
    flags: list[str] = []
    ev = sorted((e for e in events if e.client_id == client_id and e.t <= t_a), key=lambda e: e.t)  # nothing after t_a survives
    lo_w, lo_b = t_a - WINDOW, t_a - BASELINE
    win = [e for e in ev if lo_w <= e.t <= t_a]
    base = [e for e in ev if lo_b <= e.t < lo_w]
    hist = [e for e in ev if 0.0 <= e.t < lo_w]
    wp = [e for e in win if e.kind == "PUBLISH"]
    bp = [e for e in base if e.kind == "PUBLISH"]
    if len(bp) < 2:
        raise FeatureUndefined(f"baseline of {client_id} has {len(bp)} PUBLISHes; ratio features undefined")

    b_iat = [b.t - a.t for a, b in zip(bp, bp[1:])]
    b_mean_iat = _mean(b_iat)
    w_iat = [b.t - a.t for a, b in zip(wp, wp[1:])]

    if len(wp) == 0:
        flags.append("no_window_publishes")
    if len(wp) < 2:
        flags.append("max_gap_ratio_0_fewer_than_2_window_publishes")
    if 2 <= len(wp) < 3:
        flags.append("iat_cv_0_fewer_than_3_window_publishes")
    if len(wp) < 3:
        iat_cv = 0.0
    else:
        m = _mean(w_iat)
        iat_cv = math.sqrt(sum((x - m) ** 2 for x in w_iat) / (len(w_iat) - 1)) / m if m > 0 else 0.0
    max_gap_ratio = max(w_iat) / b_mean_iat if len(wp) >= 2 else 0.0
    log_rate_ratio = math.log((len(wp) / WINDOW + 0.01) / (len(bp) / (lo_w - lo_b) + 0.01))
    b_payload = _mean([e.size for e in bp])
    log_payload_ratio = math.log(_mean([e.size for e in wp]) / b_payload) if wp else 0.0
    h_topics = {e.topic for e in hist if e.kind == "PUBLISH"}
    topic_novelty = sum(1 for e in wp if e.topic not in h_topics) / len(wp) if wp else 0.0
    h_subs = {e.topic for e in hist if e.kind == "SUBSCRIBE"}
    sub_novel_count = float(len({e.topic for e in win if e.kind == "SUBSCRIBE" and e.topic not in h_subs}))
    reconnects = float(sum(1 for e in ev if e.kind == "CONNECT" and t_a - RECONNECT <= e.t <= t_a))
    out = dict(zip(FEATURE_NAMES, (iat_cv, max_gap_ratio, log_rate_ratio, log_payload_ratio, topic_novelty,
                                   sub_novel_count, reconnects)))
    if not all(math.isfinite(v) for v in out.values()):
        raise FeatureUndefined(f"non-finite feature for {client_id}: {out}")
    return out, flags
