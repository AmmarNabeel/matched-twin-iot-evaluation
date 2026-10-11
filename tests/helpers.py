"""Test helper: the ideal PCAP-equivalent trace implied by a logical plan (no network jitter)."""
from postdrift_auth.features import Event


def ideal_events(sess, delay_after_tau=0.0):
    cid, tau = sess["client_id"], sess["tau"]
    ev = []
    for conn in (sess["legit"], sess["attacker"]):
        if conn is None:
            continue
        ev.append(Event(conn["connect_at"], "CONNECT", cid, None, None, None))
        ev += [Event(t, "SUBSCRIBE", cid, f, None, None) for t, f in conn["subs"]]
        for t, topic, size in conn["pubs"]:
            ev.append(Event(t + (delay_after_tau if t >= tau else 0.0), "PUBLISH", cid, topic, size, False))
    return sorted(ev, key=lambda e: e.t)


def ideal_device_log(sess):
    out = []
    for conn in (sess["legit"], sess["attacker"]):
        if conn is not None:
            out += [{"ev": "pub", "cid": sess["client_id"], "t": t, "plan_t": t, "topic": tp, "size": s} for t, tp, s in conn["pubs"]]
    return sorted(out, key=lambda r: r["t"])
