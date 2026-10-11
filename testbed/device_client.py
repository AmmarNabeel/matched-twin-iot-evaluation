"""Container-side executor of a precomputed logical plan (stdlib + paho-mqtt only). Contains no scenario logic.

usage: device_client.py PLAN.json OUT.jsonl
The plan is produced by the orchestrator from seeds; this process only follows absolute deadlines (t0 + t).
The OUT log is for debugging and mechanical cross-checks only, not a feature source.
"""
import json
import os
import sys
import threading
import time

import paho.mqtt.client as mqtt

plan = json.load(open(sys.argv[1]))
out = open(sys.argv[2], "w")
lock = threading.Lock()
T0 = plan["t0_epoch"]
_buf: list[str] = []


def log(**kw):
    """Events are buffered in memory; a separate thread writes them. A slow shared bind mount must never block a publishing thread."""
    kw["wall"] = time.time()
    line = json.dumps(kw)
    with lock:
        _buf.append(line)


def flush_log():
    with lock:
        lines, _buf[:] = _buf[:], []
    if lines:
        out.write("\n".join(lines) + "\n")
        out.flush()


def _flusher():
    while True:
        time.sleep(15)
        flush_log()


threading.Thread(target=_flusher, daemon=True).start()


def sleep_until(t_abs):
    while True:
        d = t_abs - time.time()
        if d <= 0:
            return
        time.sleep(min(d, 0.05))


def run_identity(idn):
    cid = idn["client_id"]
    state = {"stop": False}
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=cid, clean_session=True)
    if idn.get("no_reconnect"):
        c.reconnect_delay_set(min_delay=3600, max_delay=3600)  # displaced identity never retakes the session within a run
    else:
        c.reconnect_delay_set(min_delay=1, max_delay=2)

    def on_connect(cl, ud, flags, rc, props):
        log(ev="connected", cid=cid, rc=str(rc))

    def on_disconnect(cl, ud, flags, rc, props):
        log(ev="disconnected", cid=cid, rc=str(rc))

    c.on_connect = on_connect
    c.on_disconnect = on_disconnect
    sleep_until(T0 + idn["connect_at"])
    c.connect(plan["broker_host"], 1883, keepalive=60)
    c.loop_start()
    log(ev="connect_called", cid=cid, t=time.time() - T0)
    events = [(t, "sub", f) for t, f in idn["subs"]] + [(t, "pub", (tp, sz)) for t, tp, sz in idn["pubs"]]
    events.sort(key=lambda e: (e[0], e[1]))
    for t, kind, arg in events:
        sleep_until(T0 + t)
        if kind == "sub":
            c.subscribe(arg, qos=1)
            log(ev="sub", cid=cid, t=time.time() - T0, plan_t=t, filter=arg)
        else:
            tp, sz = arg
            c.publish(tp, b"x" * sz, qos=1)
            log(ev="pub", cid=cid, t=time.time() - T0, plan_t=t, topic=tp, size=sz)
    sleep_until(T0 + idn["end_at"])
    state["stop"] = True
    c.disconnect()
    time.sleep(0.3)  # let the DISCONNECT packet leave; the network thread is a daemon and is torn down by os._exit below


threads = [threading.Thread(target=run_identity, args=(i,), daemon=True) for i in plan["identities"]]
for th in threads:
    th.start()
for th in threads:
    th.join()
log(ev="done")
flush_log()
os._exit(0)
