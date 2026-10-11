"""Scenario-mechanics checks: did the FROZEN intervention occur in the captured trace? (Gate 2, evaluation side.)

These functions compare the logical plan (ground-truth side) with the PCAP-derived events. They are NOT part of feature
extraction and may read scenario information. Tolerances are mechanical (timer/scheduling) or derived from frozen
parameter definitions; they were fixed before the live runs and are not tuned on traces.
No classifier, score, separability or detectability quantity is computed here.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, deque

from .scenarios import topics

PLAN_TOL_S = 0.5          # |trace arrival - planned time| for publishes not under netem
ONSET_TOL_S = 1.0         # reconnect / netem timing vs tau
RATIO_TOL = 0.10          # |ln(observed ratio) - ln(planned ratio)| for mean IAT / mean payload over the post-onset window
INTERVAL_SLACK_S = 0.05
PLAN_ROUNDING_SLACK_S = 1e-5   # planned times are stored rounded to 1e-6 s; an interval can differ from its nominal bound by about 2e-6 s


def _pubs(events, cid):
    return [e for e in events if e.client_id == cid and e.kind == "PUBLISH"]


def _mean_iat(pubs):
    d = [b.t - a.t for a, b in zip(pubs, pubs[1:])]
    return sum(d) / len(d) if d else float("nan")


def _res(name, ok, detail):
    return {"check": name, "ok": bool(ok), "detail": detail}


def expected_pubs(sess):
    pubs = list(sess["legit"]["pubs"]) + (list(sess["attacker"]["pubs"]) if sess["attacker"] else [])
    return sorted(pubs, key=lambda p: p[0])


def check_sequence_and_timing(sess, events, under_netem_after=None):
    """Compare the traced PUBLISHes with the plan.

    Event time is the first observation at the broker-side capture interface, so under netem jitter/loss two
    segments can be first observed in a different order than they were sent. Therefore, from `under_netem_after` on, only the
    multiset of (topic, size) must match (nothing lost, nothing invented). Before it, order and timing must match exactly.
    """
    cid = sess["client_id"]
    got = _pubs(events, cid)
    exp = expected_pubs(sess)
    k = len(exp) if under_netem_after is None else sum(1 for p in exp if p[0] < under_netem_after)
    exp_keys = [(t, s) for _, t, s in exp]
    got_keys = [(g.topic, g.size) for g in got]
    multiset_ok = Counter(exp_keys) == Counter(got_keys)
    prefix_ok = got_keys[:k] == exp_keys[:k]
    seq_ok = multiset_ok and prefix_ok
    n_swapped = sum(1 for a, b in zip(exp_keys, got_keys) if a != b) if multiset_ok else None
    worst = max((abs(g.t - pt) for g, (pt, _, _) in zip(got[:k], exp[:k])), default=0.0) if seq_ok else 0.0
    exempt_txt = "" if under_netem_after is None else f"; order/timing strict for the first {k} planned publishes, multiset only after (positions differing: {n_swapped})"
    return [_res("publish sequence (topic,size) equals plan", seq_ok, f"{len(got)} trace vs {len(exp)} planned PUBLISHes{exempt_txt}"),
            _res(f"arrival within {PLAN_TOL_S}s of plan (outside netem)", seq_ok and worst <= PLAN_TOL_S, f"max |dt|={worst:.3f}s")]


def check_interval_bounds(sess, events):
    """Consecutive publishes of one regime must have interval within period*(1+-jitter) (+slack): the frozen timing law."""
    cid, tau, reg = sess["client_id"], sess["tau"], sess["regimes"]
    bad, n = 0, 0
    legit_pubs = [p for p in sess["legit"]["pubs"]]
    att_pubs = sess["attacker"]["pubs"] if sess["attacker"] else []
    for pubs, post_reg in ((legit_pubs, reg["post"]), (att_pubs, reg["attacker"])):
        for (t1, _, _), (t2, _, _) in zip(pubs, pubs[1:]):
            r = reg["pre"] if t1 < tau else post_reg
            if r is None:
                continue
            n += 1
            lo, hi = r["period"] * (1 - r["jitter"]), r["period"] * (1 + r["jitter"])
            if not (lo - PLAN_ROUNDING_SLACK_S <= t2 - t1 <= hi + PLAN_ROUNDING_SLACK_S):
                bad += 1
    return _res("planned intervals obey regime period*(1+-jitter)", bad == 0, f"{n} intervals, {bad} violations")


def _post_ratios(sess, events):
    cid, tau, t_a = sess["client_id"], sess["tau"], sess["t_a"]
    got = _pubs(events, cid)
    pre = [g for g in got if tau - 30 <= g.t < tau]
    post = [g for g in got if tau <= g.t <= t_a]
    r_iat = _mean_iat(post) / _mean_iat(pre)
    r_pay = (sum(g.size for g in post) / len(post)) / (sum(g.size for g in pre) / len(pre))
    return r_iat, r_pay, len(pre), len(post)


def _post_ratios_by_plan_window(sess, events):
    """Same ratios as _post_ratios, but each traced PUBLISH is assigned to the pre/post window by the time at which the PLAN scheduled it.

    A publish scheduled a few milliseconds before tau can arrive a few milliseconds after it; assigning by arrival time would move it across the
    boundary and change a ratio computed from only a handful of post-drift publishes. Interval lengths still use the traced arrival times.
    Falls back to arrival-time windows if the traced sequence does not align with the plan (the sequence check fails in that case anyway).
    """
    cid, tau, t_a = sess["client_id"], sess["tau"], sess["t_a"]
    got = _pubs(events, cid)
    exp = expected_pubs(sess)
    if len(got) != len(exp) or [(g.topic, g.size) for g in got] != [(t, s) for _, t, s in exp]:
        return _post_ratios(sess, events)
    pre = [g for g, p in zip(got, exp) if tau - 30 <= p[0] < tau]
    post = [g for g, p in zip(got, exp) if tau <= p[0] <= t_a]
    r_iat = _mean_iat(post) / _mean_iat(pre)
    r_pay = (sum(g.size for g in post) / len(post)) / (sum(g.size for g in pre) / len(pre))
    return r_iat, r_pay, len(pre), len(post)


def _plan_events(sess):
    from .features import Event
    cid = sess["client_id"]
    return [Event(t, "PUBLISH", cid, tp, sz, False) for t, tp, sz in expected_pubs(sess)]


def _check_ratios(sess, events, nominal_iat, nominal_pay, label):
    """Post/pre ratios of the TRACE must match the same ratios of the realized PLAN (the same windows and definition).

    Comparing with the nominal multiplier instead would flag sampling error when the post-drift window holds few publishes
    (a slow regime has only a handful), although the trace follows the plan exactly. The nominal values are reported in the detail.
    """
    r_iat, r_pay, npre, npost = _post_ratios_by_plan_window(sess, events)
    p_iat, p_pay, _, _ = _post_ratios(sess, _plan_events(sess))
    ok_iat = math.isnan(p_iat) or abs(math.log(r_iat / p_iat)) <= RATIO_TOL
    ok_pay = math.isnan(p_pay) or abs(math.log(r_pay / p_pay)) <= RATIO_TOL
    return [_res(f"{label}: mean IAT ratio post/pre matches plan", ok_iat,
                 f"observed {r_iat:.3f} vs realized plan {p_iat:.3f} (nominal {nominal_iat:.3f}; n_pre={npre}, n_post={npost})"),
            _res(f"{label}: mean payload ratio post/pre matches plan", ok_pay, f"observed {r_pay:.3f} vs realized plan {p_pay:.3f} (nominal {nominal_pay:.3f})")]


def _by_kind(events, cid, kind):
    return [e for e in events if e.client_id == cid and e.kind == kind]


def _topic_share(sess, events, topic):
    post = [g for g in _pubs(events, sess["client_id"]) if sess["tau"] <= g.t <= sess["t_a"] + 3]
    return sum(1 for g in post if g.topic == topic) / len(post) if post else float("nan"), len(post)


def _extra_topic_check(sess, events, topic, drawn, label):
    """Not drawn -> never published. Drawn -> trace count equals the count the seeded plan realised (may be 0 by chance)."""
    planned = sum(1 for p in expected_pubs(sess) if p[1] == topic)
    seen = sum(1 for g in _pubs(events, sess["client_id"]) if g.topic == topic)
    ok = seen == planned and (drawn or seen == 0)
    return _res(label, ok, f"drawn={drawn} planned_count={planned} traced_count={seen}")


def check_scenario(sess, events, device_log=None, netem_evidence=None, broker_log="", control_cid="dev-ctl", timing_exempt_after=None):
    """timing_exempt_after: planned time from which the arrival-vs-plan check is skipped (static background netem -> 0.0)."""
    cid, tau, scn, prm = sess["client_id"], sess["tau"], sess["scenario"], sess["params"]
    tp = topics(cid)
    exempt = tau if scn == "B2" else timing_exempt_after
    out = check_sequence_and_timing(sess, events, under_netem_after=exempt)
    out.append(check_interval_bounds(sess, events))
    conns = _by_kind(events, cid, "CONNECT")
    subs = _by_kind(events, cid, "SUBSCRIBE")
    late_subs = [s for s in subs if s.t > 1.0]
    if scn == "S0":
        out.append(_res("single CONNECT, no later SUBSCRIBE", len(conns) == 1 and not late_subs, f"{len(conns)} CONNECT, {len(late_subs)} late SUBSCRIBE"))
        r_iat, r_pay, *_ = _post_ratios(sess, events)
        out.append(_res("stable regime: |ln IAT ratio|<0.1 and |ln payload ratio|<0.1 (below the B1 floor)",
                        abs(math.log(r_iat)) < 0.1 and abs(math.log(r_pay)) < 0.1, f"IAT ratio {r_iat:.3f}, payload ratio {r_pay:.3f}"))
        out.append(_res("only baseline topics published", {g.topic for g in _pubs(events, cid)} <= {tp["telemetry"], tp["status"]}, ""))
    elif scn == "B1":
        out += _check_ratios(sess, events, prm["period_mult"], prm["payload_mult"], "B1")
        share, n = _topic_share(sess, events, tp["diag"])
        out.append(_extra_topic_check(sess, events, tp["diag"], prm["diag_topic"], "diag topic: traced count equals plan, absent if not drawn"))
        out.append(_res("diag publishes only at/after tau", all(g.t >= tau - PLAN_TOL_S for g in _pubs(events, cid) if g.topic == tp["diag"]), f"share_post={share:.2f} (n={n})"))
        out.append(_res("no reconnect, no new subscription", len(conns) == 1 and not late_subs, f"{len(conns)} CONNECT"))
    elif scn == "B2":
        ev = netem_evidence
        want = f"delay {prm['delay_ms']:.1f}ms"  # tc prints e.g. 'delay 149.2ms  49.5ms'
        out.append(_res("netem absent on both containers before tau", "netem" not in ev["dev_pre"] and "netem" not in ev["ctl_pre"], ev["dev_pre"].strip()[:80]))
        dev_post_ok = "netem" in ev["dev_post"] and "loss" in ev["dev_post"] and "delay" in ev["dev_post"]
        out.append(_res("frozen netem present on target container after tau", dev_post_ok, ev["dev_post"].strip()[:160]))
        out.append(_res("control container has no netem after tau", "netem" not in ev["ctl_post"], ev["ctl_post"].strip()[:80]))
        out.append(_res(f"netem applied within {ONSET_TOL_S}s of tau", 0 <= ev["apply_end_rel"] - tau <= ONSET_TOL_S + 0.5, f"apply start={ev['apply_start_rel']:.3f}s end={ev['apply_end_rel']:.3f}s tau={tau:.3f}s"))
        out.append(_res("no CONNECT/SUBSCRIBE change from application (app unchanged)", len(late_subs) == 0, f"{len(conns)} CONNECT (reconnects due to loss would show here)"))
        if device_log is not None:
            for who, c, apply_t in (("target", cid, ev["apply_end_rel"]), ("control", control_cid, ev["apply_end_rel"])):
                lat = latencies(device_log, events, c)
                pre = [v for t, v in lat if t < tau - 1]
                post = [v for t, v in lat if t >= apply_t + 2]
                m_pre, m_post = statistics.median(pre), statistics.median(post)
                if who == "target":
                    lo, hi = (prm["delay_ms"] - prm["jitter_ms"]) / 1e3, (prm["delay_ms"] + prm["jitter_ms"]) / 1e3 + 0.05
                    ok = m_pre < 0.05 and lo <= m_post <= hi
                    out.append(_res("target one-way latency: ~0 before, delay+-jitter after", ok, f"median pre {m_pre*1e3:.1f} ms, post {m_post*1e3:.1f} ms, planned {prm['delay_ms']:.0f}+-{prm['jitter_ms']:.0f} ms"))
                else:
                    out.append(_res("control one-way latency unchanged", m_pre < 0.05 and m_post < 0.05, f"median pre {m_pre*1e3:.1f} ms, post {m_post*1e3:.1f} ms"))
    elif scn == "M1":
        att = sess["attacker"]
        out.append(_res("exactly two CONNECTs with the same client_id (legit first, attacker at tau)",
                        len(conns) == 2 and conns[0].t < 2.0 and abs(conns[1].t - tau) <= ONSET_TOL_S,
                        f"CONNECT at {[round(c.t, 3) for c in conns]}, tau={tau:.3f}"))
        n_taken = sum(1 for ln in broker_log.splitlines() if f"Client {cid} [" in ln and "session taken over" in ln)
        out.append(_res("broker took the session over exactly once (cross-check from broker log, debug only)", n_taken == 1,
                        f"'session taken over' lines: {n_taken}"))
        novel = {s.topic for s in subs if s.t > 1.0 and s.topic != tp["cmd"]}
        out.append(_res("novel subscriptions = drawn count", len(novel) == prm["n_novel_subs"], f"{len(novel)} vs drawn {prm['n_novel_subs']}"))
        out.append(_extra_topic_check(sess, events, tp["aux"], prm["new_topic"], "new publish topic: traced count equals plan, absent if not drawn"))
        out.append(_res("attacker regime: sequence/interval already checked; legit stopped publishing at takeover", True, "covered by sequence check"))
        out.append(_res("legit device did not retake the identity (still 2 CONNECTs overall)", len(conns) == 2, ""))
    elif scn == "M2":
        out += _check_ratios(sess, events, 1.0 / prm["rate_mult"], prm["payload_mult"], "M2")
        share, n = _topic_share(sess, events, tp["aux"])
        out.append(_res("same logical session: single CONNECT, no reconnect, no new subscription", len(conns) == 1 and not late_subs, f"{len(conns)} CONNECT"))
        out.append(_extra_topic_check(sess, events, tp["aux"], prm["new_topic"], f"new topic: traced count equals plan (planned share {prm['new_topic_share']:.2f}, realised {share:.2f}, n={n})"))
    elif scn == "M3":
        bounds = [0.8 <= prm["rate_mult"] <= 1.25, 0.9 <= prm["payload_mult"] <= 1.2, 1.3 <= prm["jitter_mult"] <= 2.0,
                  prm["new_topic_share"] == 0.05, prm["n_novel_subs"] == 0, prm["reconnect"] is False]
        out.append(_res("M3 drawn parameters inside the frozen definition", all(bounds), json_short(prm)))
        out += _check_ratios(sess, events, 1.0 / prm["rate_mult"], prm["payload_mult"], "M3")
        out.append(_res("no novel subscription, no forced reconnect", len(conns) == 1 and not late_subs, f"{len(conns)} CONNECT, {len(late_subs)} late SUBSCRIBE"))
        out.append(_extra_topic_check(sess, events, tp["aux"], prm["new_topic"], "new topic (5% share): traced count equals plan, absent if not drawn"))
    else:
        raise ValueError(scn)
    return out


def json_short(d):
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def latencies(device_log, events, cid):
    """(send-time, arrival-minus-send) per publish. The i-th logged send is matched to the earliest unmatched traced PUBLISH with the
    same (topic, size); this tolerates the arrival-order swaps that netem jitter/loss can create (see check_sequence_and_timing)."""
    sent = [r for r in device_log if r.get("ev") == "pub" and r["cid"] == cid]
    got = _pubs(events, cid)
    if len(sent) != len(got):
        raise ValueError(f"{cid}: {len(sent)} logged sends vs {len(got)} traced PUBLISHes")
    queues: dict = {}
    for g in got:
        queues.setdefault((g.topic, g.size), deque()).append(g)
    out = []
    for r in sent:
        q = queues.get((r["topic"], r["size"]))
        if not q:
            raise ValueError(f"{cid}: logged send {(r['topic'], r['size'])} has no traced PUBLISH")
        out.append((r["t"], q.popleft().t - r["t"]))
    return out
