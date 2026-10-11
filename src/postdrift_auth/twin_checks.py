"""Automated invariants of a matched-drift twin family (see scenarios.build_twin_family). Returns a list of violations; empty means valid."""
from __future__ import annotations

from .scenarios import twin_composition


def _strip(sess: dict) -> dict:
    """Everything that must be identical between the two branches of one identity: the pre-drift session and its environment."""
    leg = sess["legit"]
    tau = sess["tau"]
    return {"client_id": sess["client_id"], "base": sess["base"], "tau": tau, "t_a": sess["t_a"], "connect_at": leg["connect_at"], "subs": leg["subs"],
            "end_at": leg["end_at"], "pre_pubs": [p for p in leg["pubs"] if p[0] < tau], "family_seed": sess["family_seed"], "pair_id": sess["pair_id"]}


def pair_invariants(fam: dict) -> list[str]:
    bad: list[str] = []
    A, B = fam["arms"]["A"], fam["arms"]["B"]
    kind = fam["kind"]
    mal_scn = {"M12": ("M1", "M2"), "M3": ("M3",)}[kind]
    sa = {s["client_id"]: s for s in A["sessions"]}
    sb = {s["client_id"]: s for s in B["sessions"]}
    if set(sa) != set(sb) or len(sa) != 40:
        return ["identity sets differ between arms or are not 40"]
    where = lambda plan: {cid: c for c, v in plan["containers"].items() if v["kind"] == "mixed" for cid in v["client_ids"]}  # noqa: E731
    wa, wb = where(A), where(B)
    for cid in sa:
        a, b = sa[cid], sb[cid]
        if _strip(a) != _strip(b):
            bad.append(f"{cid}: pre-drift session or its timing differs between the branches")
        if wa[cid] != wb[cid]:
            bad.append(f"{cid}: container differs between the runs")
        if {a["role"], b["role"]} != {"benign", "malicious"}:
            bad.append(f"{cid}: roles are not complementary ({a['role']}, {b['role']})")
        ben, mal = (a, b) if a["role"] == "benign" else (b, a)
        if ben["scenario"] != "B1" or ben["y"] != 0:
            bad.append(f"{cid}: benign branch is not B1/Y=0")
        if mal["scenario"] not in mal_scn or mal["y"] != 1:
            bad.append(f"{cid}: malicious branch is not {mal_scn}/Y=1")
    for arm, plan in (("A", A), ("B", B)):
        roles = [s["role"] for s in plan["sessions"]]
        if roles.count("malicious") != 20 or roles.count("benign") != 20:
            bad.append(f"run {arm}: not 20 malicious + 20 benign")
        comp = {}
        for s in plan["sessions"]:
            comp[s["scenario"]] = comp.get(s["scenario"], 0) + 1
        if comp != twin_composition(kind):
            bad.append(f"run {arm}: composition {comp} != {twin_composition(kind)}")
        for c, v in plan["containers"].items():
            if v["kind"] == "mixed":
                r = [next(s["role"] for s in plan["sessions"] if s["client_id"] == cid) for cid in v["client_ids"]]
                if r.count("malicious") != 2 or len(r) != 4:
                    bad.append(f"run {arm}/{c}: not 2 malicious + 2 benign in a container of 4")
    for c, v in A["containers"].items():
        if v["kind"] == "mixed" and v["bg_netem"] != B["containers"][c]["bg_netem"]:
            bad.append(f"{c}: background link condition differs between the runs")
    for arm, plan in (("A", A), ("B", B)):
        for c, v in plan["containers"].items():
            if v["kind"] == "attacker":
                dev = c[len("att-"):]
                if v["bg_netem"] != plan["containers"][dev]["bg_netem"]:
                    bad.append(f"run {arm}/{c}: attacker link condition differs from its device container")
    return bad
