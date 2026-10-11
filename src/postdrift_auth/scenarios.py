"""Logical scenario generation. Pure functions of (run_seed, scenario, client_id).

Nothing here touches Docker, the clock or the filesystem. The output plan is executed verbatim by testbed/device_client.py.
All parameter ranges are the FROZEN values of the study protocol; they are constants, not arguments.
"""
from __future__ import annotations

import hashlib
import json
import math

from .seeds import derived_seed, draw, log_uniform, uniform

SCENARIOS = ("S0", "B1", "B2", "M1", "M2", "M3")
MALICIOUS = {"S0": 0, "B1": 0, "B2": 0, "M1": 1, "M2": 1, "M3": 1}

DELTA = 30.0              # oracle trigger delay (19.3)
STABLE_END = 150.0        # onset tau = 150 + U(0,10)
ONSET_SPREAD = 10.0
PUB_TAIL = 3.0            # publishes planned until t_a + 3 s
END_TAIL = 5.0            # run (device) ends t_a + 5 s
PERIOD_RANGE = (0.5, 2.0)
SIZE_RANGE = (40.0, 200.0)
PERIOD_JITTER = 0.05
SIZE_JITTER = 0.10
STATUS_SHARE = 0.10
FIRST_PUB_OFFSET = 1.0


def topics(cid: str) -> dict[str, str]:
    return {"telemetry": f"dev/{cid}/telemetry", "status": f"dev/{cid}/status", "cmd": f"dev/{cid}/cmd",
            "diag": f"dev/{cid}/ext", "aux": f"dev/{cid}/ext"}  # one neutral name: topic names must not encode Y/scenario


def base_profile(run_seed: int, cid: str) -> dict:
    c = f"{cid}/base"
    return {"period": uniform(run_seed, c, 0, *PERIOD_RANGE), "size": uniform(run_seed, c, 1, *SIZE_RANGE)}


def onset(run_seed: int, comp: str) -> float:
    """tau = 150 + U(0,10) from the draw stream `comp` (the identity, or the container for B2 containers)."""
    return STABLE_END + uniform(run_seed, f"{comp}/onset", 0, 0.0, ONSET_SPREAD)


def scenario_params(run_seed: int, cid: str, scenario: str, netem_comp: str | None = None, stream: str | None = None) -> dict:
    """Frozen parameter draws for one session of the given scenario. `stream` renames the draw stream (twin runs: one per role)."""
    c = stream or f"{cid}/scn"
    if scenario == "S0":
        return {}
    if scenario == "B1":
        k = 0
        while True:  # |ln r| >= 0.1 by rejection on successive deterministic draws
            r = log_uniform(run_seed, c, k, 0.5, 2.0)
            if abs(math.log(r)) >= 0.1:
                break
            k += 1
        return {"period_mult": r, "payload_mult": log_uniform(run_seed, c, 100, 0.7, 1.5),
                "diag_topic": draw(run_seed, c, 101) < 0.3, "diag_share": 0.10}
    if scenario == "B2":
        nc = f"{netem_comp or cid}/netem"
        return {"delay_ms": uniform(run_seed, nc, 0, 100.0, 400.0), "jitter_ms": uniform(run_seed, nc, 1, 20.0, 100.0),
                "loss_pct": uniform(run_seed, nc, 2, 1.0, 8.0)}
    if scenario == "M1":
        n_sub = 0 if draw(run_seed, c, 2) < 0.3 else 1 + int(draw(run_seed, c, 3) * 3)
        return {"rate_mult": log_uniform(run_seed, c, 0, 0.5, 2.0), "payload_mult": log_uniform(run_seed, c, 1, 0.5, 2.0),
                "n_novel_subs": n_sub, "new_topic": draw(run_seed, c, 4) < 0.2, "new_topic_share": 0.10}
    if scenario == "M2":
        return {"rate_mult": log_uniform(run_seed, c, 0, 0.25, 4.0), "payload_mult": log_uniform(run_seed, c, 1, 0.5, 3.0),
                "new_topic": draw(run_seed, c, 2) < 0.6, "new_topic_share": uniform(run_seed, c, 3, 0.2, 0.5)}
    if scenario == "M3":
        return {"rate_mult": log_uniform(run_seed, c, 0, 0.8, 1.25), "payload_mult": log_uniform(run_seed, c, 1, 0.9, 1.2),
                "jitter_mult": uniform(run_seed, c, 2, 1.3, 2.0), "new_topic": draw(run_seed, c, 3) < 0.5,
                "new_topic_share": 0.05, "n_novel_subs": 0, "reconnect": False}
    raise ValueError(f"unknown scenario {scenario!r}")


def _regime(period, size, jitter=PERIOD_JITTER, extra_topic=None, extra_share=0.0):
    return {"period": period, "size": size, "jitter": jitter, "extra_topic": extra_topic, "extra_share": extra_share}


def _gen_pubs(run_seed: int, comp: str, cid: str, start: float, stop: float, tau: float, pre: dict, post: dict):
    """Publish schedule. Draw 4k..4k+3 belong to publish k, so the schedule is stateless and reproducible."""
    tp = topics(cid)
    pubs, t, k = [], start, 0
    while t <= stop:
        reg = pre if t < tau else post
        u_int, u_size, u_top, u_stat = (draw(run_seed, comp, 4 * k + j) for j in range(4))
        size = max(1, round(reg["size"] * (1 + SIZE_JITTER * (2 * u_size - 1))))
        if reg["extra_topic"] and u_top < reg["extra_share"]:
            topic = reg["extra_topic"]
        else:
            topic = tp["status"] if u_stat < STATUS_SHARE else tp["telemetry"]
        pubs.append([round(t, 6), topic, int(size)])
        t += reg["period"] * (1 + reg["jitter"] * (2 * u_int - 1))
        k += 1
    return pubs


def build_session(run_seed: int, cid: str, scenario: str, onset_comp: str | None = None, netem_comp: str | None = None,
                  scn_stream: str | None = None) -> dict:
    """All logical content of one identity-session: legit connection plan, optional attacker plan, netem.

    onset_comp / netem_comp: draw-stream owners. Default is the identity itself; B2 sessions inside a multi-device container use
    the container (one netem draw per container, and the container's common onset).
    """
    base = base_profile(run_seed, cid)
    P, S = base["period"], base["size"]
    tau = onset(run_seed, onset_comp or cid)
    t_a = tau + DELTA
    stop = t_a + PUB_TAIL
    prm = scenario_params(run_seed, cid, scenario, netem_comp, scn_stream)
    tp = topics(cid)
    pre = _regime(P, S)
    first = FIRST_PUB_OFFSET + draw(run_seed, f"{cid}/first", 0) * P
    legit = {"client_id": cid, "connect_at": 0.0, "no_reconnect": False, "subs": [[0.0, tp["cmd"]]], "end_at": t_a + END_TAIL}
    attacker = None
    netem = None
    if scenario in ("S0", "B2"):
        post = pre
        if scenario == "B2":
            netem = {"apply_at": tau, **prm}
    elif scenario == "B1":
        post = _regime(P * prm["period_mult"], S * prm["payload_mult"], extra_topic=tp["diag"] if prm["diag_topic"] else None,
                       extra_share=prm["diag_share"])
    elif scenario == "M2":
        post = _regime(P / prm["rate_mult"], S * prm["payload_mult"], extra_topic=tp["aux"] if prm["new_topic"] else None,
                       extra_share=prm["new_topic_share"])
    elif scenario == "M3":
        post = _regime(P / prm["rate_mult"], S * prm["payload_mult"], jitter=PERIOD_JITTER * prm["jitter_mult"],
                       extra_topic=tp["aux"] if prm["new_topic"] else None, extra_share=prm["new_topic_share"])
    elif scenario == "M1":
        post = pre
        legit["no_reconnect"] = True
        a_reg = _regime(P / prm["rate_mult"], S * prm["payload_mult"], extra_topic=tp["aux"] if prm["new_topic"] else None,
                        extra_share=prm["new_topic_share"])
        a_first = tau + 0.5 + draw(run_seed, f"{cid}/first", 1) * a_reg["period"]
        a_subs = [[tau, tp["cmd"]]] + [[round(tau + 0.2 * (j + 1), 6), f"dev/{cid}/ext{j + 1}"] for j in range(prm["n_novel_subs"])]
        attacker = {"client_id": cid, "connect_at": tau, "no_reconnect": False, "subs": a_subs, "end_at": t_a + END_TAIL,
                    "pubs": _gen_pubs(run_seed, f"{cid}/attacker", cid, a_first, stop, tau, a_reg, a_reg)}
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    legit_stop = tau - 1e-6 if scenario == "M1" else stop
    legit["pubs"] = _gen_pubs(run_seed, f"{cid}/sched", cid, first, legit_stop, tau, pre, post)
    return {"client_id": cid, "scenario": scenario, "y": MALICIOUS[scenario], "base": base, "tau": tau, "t_a": t_a,
            "params": prm, "legit": legit, "attacker": attacker, "netem": netem,
            "regimes": {"pre": pre, "post": post, "attacker": a_reg if scenario == "M1" else None}}


def build_run(run_seed: int, scenario: str) -> dict:
    """One smoke run: target session of `scenario` plus an S0 control session on a separate container."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    target = build_session(run_seed, "dev-target", scenario)
    control = build_session(run_seed, "dev-ctl", "S0")
    comps = [f"{c}/{s}" for c in ("dev-target", "dev-ctl") for s in ("base", "onset", "scn", "sched", "first", "netem")]
    comps += ["dev-target/attacker"]
    return {"run_seed": run_seed, "scenario": scenario, "sessions": [target, control],
            "derived_seeds": {c: derived_seed(run_seed, c) for c in comps}}


def plan_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---- 40-session VAL-like run (Stage 2C scale smoke; same composition as the VAL split, blueprint 19.8) ----
VAL_QUOTA = (("B2", 16), ("B1", 16), ("M1", 4), ("M2", 4))
SOURCE_QUOTA = (("S0", 32), ("M1", 4), ("M2", 4))      # TRAIN and CAL (clean links)
SPLIT_COMPOSITION = {"TRAIN": (SOURCE_QUOTA, False), "CAL": (SOURCE_QUOTA, False), "VAL": (VAL_QUOTA, True),
                     "SMOKE": (VAL_QUOTA, True)}      # (quota, background link variability)
DEVICES_PER_CONTAINER = 4
BG_PROBABILITY = 0.3
BG_NETEM = {"delay_ms": 50.0, "jitter_ms": 10.0, "loss_pct": 0.5}


# Pilot v2: the source regime also contains the KNOWN benign change family B1 (generator unchanged).
V2_SOURCE_QUOTA = (("S0", 24), ("B1", 8), ("M1", 4), ("M2", 4))
SPLIT_COMPOSITION_BY_VARIANT = {
    "v1": SPLIT_COMPOSITION,
    "v2": {"TRAIN": (V2_SOURCE_QUOTA, False), "CAL": (V2_SOURCE_QUOTA, False), "VAL": (VAL_QUOTA, True), "SMOKE": (VAL_QUOTA, True)},
}


def build_split_run(split: str, run_seed: int, variant: str = "v1") -> dict:
    """Run plan for a split with its frozen composition; the seed must lie in the split's registered range."""
    from .seeds import split_of_seed
    comp = SPLIT_COMPOSITION_BY_VARIANT[variant]
    if split not in comp:
        raise ValueError(f"no frozen composition for split {split!r}")
    if split_of_seed(run_seed) != split:
        raise ValueError(f"seed {run_seed} is not in the {split} range")
    quota, background = comp[split]
    return build_scale_run(run_seed, quota=quota, background=background)


def build_scale_run(run_seed: int, quota=VAL_QUOTA, background: bool = True) -> dict:
    """One run of 40 triggered identities in 10 device containers of 4 (+ one attacker container for M1).

    Grouping ('4 containers x 4 devices, one draw per container'): the 16 B2 identities fill c00-c03 (one netem
    draw and one common onset per container). The 24 other identities (B1, M1, M2) are assigned to c04-c09 by a seeded shuffle.
    Client ids are a seeded permutation of 0..39 so they carry no scenario information. Non-B2 containers (and the attacker
    container) get the static mild background netem with probability 0.3 each (independent of Y), applied from before t = 0.
    """
    slots = [scn for scn, n in quota for _ in range(n)]
    n = len(slots)
    id_perm = sorted(range(n), key=lambda k: draw(run_seed, "idperm", k))
    cids = {k: f"dev-{id_perm[k]:03d}" for k in range(n)}
    b2_slots = [k for k, s in enumerate(slots) if s == "B2"]
    other_slots = sorted((k for k, s in enumerate(slots) if s != "B2"), key=lambda k: draw(run_seed, "contperm", k))
    n_b2c = len(b2_slots) // DEVICES_PER_CONTAINER
    layout = {f"c{i:02d}": b2_slots[i * DEVICES_PER_CONTAINER:(i + 1) * DEVICES_PER_CONTAINER] for i in range(n_b2c)}
    for j in range(0, len(other_slots), DEVICES_PER_CONTAINER):
        layout[f"c{n_b2c + j // DEVICES_PER_CONTAINER:02d}"] = other_slots[j:j + DEVICES_PER_CONTAINER]
    containers, sessions, attackers = {}, [], []
    for cname, members in layout.items():
        is_b2 = slots[members[0]] == "B2"
        assert all((slots[k] == "B2") == is_b2 for k in members)
        sess = [build_session(run_seed, cids[k], slots[k], onset_comp=cname if is_b2 else None, netem_comp=cname if is_b2 else None)
                for k in members]
        sessions += sess
        bg = background and (not is_b2) and draw(run_seed, f"{cname}/bg", 0) < BG_PROBABILITY
        containers[cname] = {"kind": "B2" if is_b2 else "mixed", "client_ids": [s["client_id"] for s in sess], "bg_netem": BG_NETEM if bg else None,
                             "b2_netem": ({"apply_at": sess[0]["tau"], **sess[0]["params"]} if is_b2 else None),
                             "identities": [s["legit"] for s in sess]}
        attackers += [s["attacker"] for s in sess if s["attacker"]]
    if attackers:
        bg = background and draw(run_seed, "att/bg", 0) < BG_PROBABILITY
        containers["att"] = {"kind": "attacker", "client_ids": [a["client_id"] for a in attackers], "bg_netem": BG_NETEM if bg else None,
                             "b2_netem": None, "identities": attackers}
    comps = ["idperm", "contperm", "att/bg"] + [f"{c}/{x}" for c in layout for x in ("bg", "onset", "netem")]
    comps += [f"{s['client_id']}/{x}" for s in sessions for x in ("base", "scn", "sched", "first", "onset")]
    return {"run_seed": run_seed, "kind": "VAL-like-scale", "sessions": sessions, "containers": containers,
            "derived_seeds": {c: derived_seed(run_seed, c) for c in comps}}


# ---- Matched-drift twin families (configuration: configs/matched_drift_v1.json) ----
TWIN_KINDS = {"M12": ("M1", "M2"), "M3": ("M3",)}
TWIN_SESSIONS = 40


def build_twin_family(family_seed: int, kind: str) -> dict:
    """One family = two physical runs, A and B, built from the same family seed.

    Shared exactly (same draws): client ids, container layout, background link condition of every container, base profile, onset tau,
    trigger t_a, first publish, initial subscription and every publish before tau. Each identity is MALICIOUS in exactly one run and
    BENIGN (B1) in the other; its twin therefore has the same pre-drift session. Post-drift parameters of the benign and the malicious
    branch come from independent draw streams of the frozen B1 and M1/M2/M3 generators (the uniform stream behind the publish
    microstructure is shared). Inside every container two identities are malicious in A and the other two in B, so each run holds both
    labels in every container. M1 attackers connect from a separate attacker container per device container, with the same background link
    condition as that device container.
    """
    if kind not in TWIN_KINDS:
        raise ValueError(f"unknown twin kind {kind!r}")
    mal_types = TWIN_KINDS[kind]
    n = TWIN_SESSIONS
    id_perm = sorted(range(n), key=lambda k: draw(family_seed, "idperm", k))
    cids = {k: f"dev-{id_perm[k]:03d}" for k in range(n)}
    order = sorted(range(n), key=lambda k: draw(family_seed, "contperm", k))
    layout = {f"c{j // DEVICES_PER_CONTAINER:02d}": order[j:j + DEVICES_PER_CONTAINER] for j in range(0, n, DEVICES_PER_CONTAINER)}
    mal_in: dict[str, set] = {"A": set(), "B": set()}
    for cname, members in layout.items():
        ranked = sorted(members, key=lambda k: draw(family_seed, f"{cname}/role", k))
        mal_in["A"].update(ranked[:2])
        mal_in["B"].update(ranked[2:])
    mtype = {}
    for arm in ("A", "B"):
        ks = sorted(mal_in[arm], key=lambda k: draw(family_seed, f"{arm}/mtype", k))
        for i, k in enumerate(ks):
            mtype[k] = mal_types[0] if len(mal_types) == 1 or i < len(ks) // 2 else mal_types[1]
    bg = {cname: draw(family_seed, f"{cname}/bg", 0) < BG_PROBABILITY for cname in layout}
    arms = {}
    for arm in ("A", "B"):
        sessions, containers, attackers = [], {}, {}
        for cname, members in layout.items():
            sess = []
            for k in members:
                malicious = k in mal_in[arm]
                scn = mtype[k] if malicious else "B1"
                s = build_session(family_seed, cids[k], scn, scn_stream=f"{cids[k]}/scn/{'mal' if malicious else 'ben'}")
                s.update(family_seed=family_seed, arm=arm, pair_id=f"{family_seed}:{cids[k]}", role="malicious" if malicious else "benign")
                sess.append(s)
                if s["attacker"]:
                    attackers.setdefault(cname, []).append(s["attacker"])
            sessions += sess
            containers[cname] = {"kind": "mixed", "client_ids": [s["client_id"] for s in sess], "bg_netem": BG_NETEM if bg[cname] else None,
                                 "b2_netem": None, "identities": [s["legit"] for s in sess]}
        for cname, atts in attackers.items():
            containers[f"att-{cname}"] = {"kind": "attacker", "client_ids": [a["client_id"] for a in atts], "bg_netem": BG_NETEM if bg[cname] else None,
                                          "b2_netem": None, "identities": atts}
        comps = ["idperm", "contperm", f"{arm}/mtype"] + [f"{c}/{x}" for c in layout for x in ("bg", "role")]
        comps += [f"{s['client_id']}/{x}" for s in sessions for x in ("base", "sched", "first", "onset")]
        comps += [f"{s['client_id']}/scn/{'mal' if s['role'] == 'malicious' else 'ben'}" for s in sessions]
        arms[arm] = {"run_seed": family_seed, "kind": f"twin-{kind}", "arm": arm, "family_seed": family_seed, "sessions": sessions,
                     "containers": containers, "derived_seeds": {c: derived_seed(family_seed, c) for c in comps}}
    return {"family_seed": family_seed, "kind": kind, "arms": arms, "malicious_in": {a: sorted(cids[k] for k in v) for a, v in mal_in.items()}}


def twin_composition(kind: str) -> dict:
    if kind == "M12":
        return {"B1": 20, "M1": 10, "M2": 10}
    if kind == "M3":
        return {"B1": 20, "M3": 20}
    raise ValueError(kind)
