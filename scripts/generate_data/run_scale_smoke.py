"""Stage 2C: ONE 40-triggered-session SMOKE run with the VAL-like composition (16 B1, 16 B2, 4 M1, 4 M2; no M3).

Engineering scale test only: no classifier, calibration, width, or hypothesis test exists. SMOKE seed range only.
Layout (see scenarios.build_scale_run): 4 B2 containers x 4 devices (one netem draw and common onset per container),
6 mixed containers x 4 devices (B1/M1/M2), one attacker container for the M1 takeovers, static background netem w.p. 0.3 per
non-B2 container. Traces = PCAP from a tcpdump sidecar in the broker namespace; features from a separate extraction process.

usage: python scripts/generate_data/run_scale_smoke.py [--allow-dirty]
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import run_smoke_gate2 as g2  # noqa: E402  (docker helpers shared with Gate 2)
from postdrift_auth import features, mechanics, scenarios  # noqa: E402
from postdrift_auth.provenance import git_state, require_clean  # noqa: E402
from postdrift_auth.seeds import split_of_seed  # noqa: E402
from postdrift_auth.trace import read_trace_csv, sha256_file  # noqa: E402

SEED = 9_000_100          # fixed a priori, SMOKE range
OUT_ROOT = os.path.join(ROOT, "results/raw/smoke_scale")
MARGIN = 60.0
B2_PRE_SNAPSHOT_LEAD_S = 6.0
sh, wait_until = g2.sh, g2.wait_until
NETEM_RE = re.compile(r"delay ([\d.]+)ms\s+([\d.]+)ms\s+loss ([\d.]+)%")


INFRA_RETRIES: list[dict] = []   # every failed `docker exec tc ...` probe, with stderr; saved to meta.json


def _exec(name, *cmd, tries=4, log=None):
    log = INFRA_RETRIES if log is None else log
    for k in range(tries):
        p = subprocess.run(["docker", "exec", name, *cmd], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout
        log.append({"container": name, "cmd": " ".join(cmd), "try": k, "rc": p.returncode, "stderr": p.stderr.strip()[:300], "t": time.time()})
        time.sleep(0.5)
    raise RuntimeError(f"docker exec {name} {' '.join(cmd)} failed {tries}x: {log[-1]}")


def tc_show(name, log=None):
    return _exec(name, "tc", "qdisc", "show", "dev", "eth0", log=log)


def tc_add(name, n, log=None):
    cmd = ("tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay", f"{n['delay_ms']:.1f}ms", f"{n['jitter_ms']:.1f}ms",
           "loss", f"{n['loss_pct']:.2f}%")
    p = subprocess.run(["docker", "exec", name, *cmd], capture_output=True, text=True)
    if p.returncode and "netem" not in tc_show(name, log):   # failed and not applied: one more attempt; never add twice
        (INFRA_RETRIES if log is None else log).append({"container": name, "cmd": " ".join(cmd), "try": 0, "rc": p.returncode, "stderr": p.stderr.strip()[:300], "t": time.time()})
        _exec(name, *cmd, log=log)


def parse_netem(txt):
    m = NETEM_RE.search(txt)
    return tuple(float(x) for x in m.groups()) if m else None


def unit_mb(s):
    v, u = re.match(r"([\d.]+)\s*([A-Za-z]+)", s.strip()).groups()
    return float(v) * {"B": 1 / 1048576, "kB": 1 / 1024, "KB": 1 / 1024, "KiB": 1 / 1024, "MiB": 1, "MB": 1, "GiB": 1024, "GB": 1024}[u]


def execute_run(split: str, seed: int, rdir: str, git: dict, docker_meta: dict, margin: float = MARGIN, variant: str = "v1",
                plan: dict | None = None, run_id: str | None = None, docker_tag: str | None = None, composition: dict | None = None,
                t0_epoch: float | None = None, stats_interval_s: float = 8.0) -> dict:
    """Run one 40-session episode for `split` with its frozen composition. Returns meta (also written to rdir/meta.json).

    Raises on infrastructure failure. Structural failures are reported in meta['summary'][...]['ok'] and meta['all_ok'].
    """
    SEED = seed
    MARGIN = margin  # noqa: N806
    RL: list[dict] = []   # this run's failed docker-exec probes (per run, not shared with parallel runs)
    show = lambda name: tc_show(name, RL)  # noqa: E731
    add = lambda name, n: tc_add(name, n, RL)  # noqa: E731
    wall0 = time.time()
    assert split_of_seed(SEED) == split
    run = plan if plan is not None else scenarios.build_split_run(split, SEED, variant)
    run_id = run_id or f"{split}-{SEED}"
    tag = docker_tag or f"pilot{SEED}"
    ein, eout = os.path.join(rdir, "extract_in"), os.path.join(rdir, "extract_out")
    for d in (rdir, ein, eout, os.path.join(rdir, "logs")):
        os.makedirs(d, exist_ok=True)
    sessions = {s["client_id"]: s for s in run["sessions"]}
    conts = run["containers"]
    cname = lambda c: f"{tag}-{c}"  # noqa: E731
    net, broker, cap = tag, cname("broker"), cname("cap")
    t_end = max(s["t_a"] for s in sessions.values()) + scenarios.END_TAIL + 3
    # control container for B2 latency/isolation reference: a clean mixed container
    clean = [c for c, v in conts.items() if v["kind"] == "mixed" and not v["bg_netem"]]
    has_b2 = any(v["b2_netem"] for v in conts.values())
    ctl_cont = clean[0] if clean else None
    ctl_cid = next((cid for cid in conts[ctl_cont]["client_ids"] if sessions[cid]["scenario"] in ("B1", "M2", "S0")), None) if ctl_cont else None
    if has_b2 and ctl_cid is None:
        raise RuntimeError("no clean control container available for B2 isolation checks")
    b2_evidence, stats, snapshots = {}, [], {}
    stop_stats = threading.Event()
    started = []
    sh("docker", "network", "create", net)
    try:
        sh("docker", "run", "-d", "--name", broker, "--network", net, "--network-alias", "broker",
           "-v", f"{ROOT}/testbed/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro", g2.IMG_BROKER)
        started.append(broker)
        # The capture is written to the container's own filesystem (not a shared bind mount, whose stalls made tcpdump drop packets) with a
        # 16 MiB capture buffer, and copied out after the run. tcpdump reports kernel drops on exit; they are checked below.
        sh("docker", "run", "-d", "--name", cap, "--network", f"container:{broker}", "--cap-add", "NET_ADMIN", "--cap-add", "NET_RAW",
           g2.IMG_TOOLS, "sh", "-c", "tcpdump -i eth0 -n -U -B 16384 -w /tmp/trace.pcap tcp port 1883")
        started.append(cap)
        time.sleep(3)
        t0 = t0_epoch if t0_epoch is not None else time.time() + MARGIN   # shared start time lets two runs start their containers in lockstep
        if t0 - time.time() < 20:
            raise RuntimeError(f"start time t0 is only {t0 - time.time():.1f} s away; containers would not be ready")
        for c, v in conts.items():
            json.dump({"t0_epoch": t0, "broker_host": "broker", "identities": v["identities"]}, open(os.path.join(rdir, f"plan_{c}.json"), "w"), indent=1)
            sh("docker", "run", "-d", "--name", cname(c), "--network", net, "--cap-add", "NET_ADMIN",
               "-v", f"{g2.PAHO}:/opt/py/paho:ro", "-v", f"{ROOT}/testbed/device_client.py:/app/device_client.py:ro", "-v", f"{rdir}:/run",
               "-e", "PYTHONPATH=/opt/py", g2.IMG_TOOLS, "python3", "/app/device_client.py", f"/run/plan_{c}.json", f"/run/logs/{c}.jsonl")
            started.append(cname(c))
        for c, v in conts.items():  # static background link variability, in place before t = 0
            if v["bg_netem"]:
                add(cname(c), v["bg_netem"])
        pre_t0 = {c: show(cname(c)) for c in conts}

        def b2_worker(c):
            n = conts[c]["b2_netem"]
            wait_until(t0 + n["apply_at"] - B2_PRE_SNAPSHOT_LEAD_S)   # early, so slow `docker exec` under load cannot delay the apply at tau
            ev = {"dev_pre": show(cname(c)), "ctl_pre": show(cname(ctl_cont))}
            wait_until(t0 + n["apply_at"])
            a0 = time.time()
            add(cname(c), n)
            a1 = time.time()
            time.sleep(1)
            ev.update(dev_post=show(cname(c)), ctl_post=show(cname(ctl_cont)), apply_start_rel=a0 - t0, apply_end_rel=a1 - t0)
            b2_evidence[c] = ev

        def stats_worker():
            while not stop_stats.is_set():
                try:
                    out = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}"],
                                         capture_output=True, text=True).stdout
                    rows = [l.split("|") for l in out.splitlines() if l.startswith(f"{tag}-")]
                    if rows:
                        stats.append({"t": time.time() - t0, "cpu_pct": sum(float(r[1].rstrip("%")) for r in rows),
                                      "mem_mb": sum(unit_mb(r[2].split("/")[0]) for r in rows), "n": len(rows)})
                except Exception as e:  # noqa: BLE001  (resource sampling must never break a run)
                    stats.append({"t": time.time() - t0, "sampling_error": repr(e)})
                stop_stats.wait(stats_interval_s)

        workers = [threading.Thread(target=b2_worker, args=(c,)) for c, v in conts.items() if v["b2_netem"]]
        st = threading.Thread(target=stats_worker)
        for w in workers + [st]:
            w.start()
        last_ta = {c: max(sessions[i]["t_a"] for i in v["client_ids"]) for c, v in conts.items()}
        snap_source = {}
        if has_b2:
            def snap_worker(c):  # final qdisc state, 1 s after the container's own last t_a (its devices end at t_a + 5)
                wait_until(t0 + last_ta[c] + 1)
                snapshots[c] = show(cname(c))
                snap_source[c] = "final"

            snappers = [threading.Thread(target=snap_worker, args=(c,)) for c in conts]
            for th in snappers:
                th.start()
            for th in snappers:
                th.join()
        else:
            # Static link profile only (applied before t0). Few `docker exec` calls, sequential, well before the earliest container exits,
            # background containers only and best effort; every other container is represented by its pre-t0 snapshot (recorded in snap_source).
            wait_until(t0 + min(last_ta.values()) - 15)
            for c, v in conts.items():
                if v["bg_netem"]:
                    try:
                        snapshots[c] = show(cname(c))
                        snap_source[c] = "final"
                    except Exception as e:  # noqa: BLE001
                        RL.append({"container": cname(c), "cmd": "final snapshot", "error": repr(e)[:200]})
            for c in conts:
                if c not in snapshots:
                    snapshots[c] = pre_t0[c]
                    snap_source[c] = "pre_t0"
        wait_until(t0 + t_end)
        for c in conts:
            sh("docker", "wait", cname(c))
        for w in workers:
            w.join()
        stop_stats.set()
        st.join()
        time.sleep(1.5)
        sh("docker", "stop", "-t", "3", cap)
        cap_logs = sh("docker", "logs", cap)
        tcpdump_stats = cap_logs.stdout + cap_logs.stderr
        sh("docker", "cp", f"{cap}:/tmp/trace.pcap", os.path.join(ein, "trace.pcap"))
        lg = sh("docker", "logs", broker)
        broker_log = lg.stdout + lg.stderr
        open(os.path.join(rdir, "logs", "broker.log"), "w").write(broker_log)
    finally:
        stop_stats.set()
        for c in started:
            sh("docker", "rm", "-f", c, check=False)
        sh("docker", "network", "rm", net, check=False)

    gt = os.path.join(rdir, "ground_truth.jsonl")
    cont_of = {cid: c for c, v in conts.items() if v["kind"] != "attacker" for cid in v["client_ids"]}
    with open(gt, "w") as f:
        for s in run["sessions"]:
            f.write(json.dumps({"session_id": s["client_id"], "scenario": s["scenario"], "y": s["y"], "tau": s["tau"], "t_a": s["t_a"],
                                "container": cont_of[s["client_id"]], "params": s["params"],
                                **({k: s[k] for k in ("family_seed", "arm", "pair_id") if k in s})}, sort_keys=True) + "\n")
    trig = os.path.join(ein, "trigger.jsonl")
    with open(trig, "w") as f:
        for s in run["sessions"]:
            f.write(json.dumps({"client_id": s["client_id"], "t_a": s["t_a"]}) + "\n")
    ex = subprocess.run([sys.executable, "-m", "postdrift_auth.extract_cli", "--pcap", os.path.join(ein, "trace.pcap"), "--t0", repr(t0),
                         "--trigger", trig, "--out-dir", eout], cwd=ein, env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
                        capture_output=True, text=True)
    if ex.returncode:
        raise RuntimeError(f"extraction failed:\n{ex.stderr}")
    events = read_trace_csv(os.path.join(eout, "trace_events.csv"))
    rep = json.load(open(os.path.join(eout, "extract_report.json")))
    feats = list(csv.DictReader(open(os.path.join(eout, "features.csv"))))
    dlog, done = [], {}
    for c in conts:
        rows = [json.loads(l) for l in open(os.path.join(rdir, "logs", f"{c}.jsonl"))]
        done[c] = any(r.get("ev") == "done" for r in rows)
        dlog += rows
    dlog.sort(key=lambda r: r.get("t", 0.0))

    # ---- per-session mechanics ----
    results = {}
    for cid, s in sessions.items():
        c = cont_of[cid]
        kw = dict(device_log=dlog, broker_log=broker_log)
        if s["scenario"] == "B2":
            kw.update(netem_evidence=b2_evidence[c], control_cid=ctl_cid)
        elif conts[c]["bg_netem"]:
            kw.update(timing_exempt_after=0.0)
        results[cid] = mechanics.check_scenario(s, events, **kw)
    failed = {cid: [x for x in r if not x["ok"]] for cid, r in results.items()}
    failed = {k: v for k, v in failed.items() if v}

    # ---- container-level netem state (F, G) ----
    cont_checks = []
    for c, v in conts.items():
        exp = v["b2_netem"] or v["bg_netem"]
        got = parse_netem(snapshots[c])
        if exp is None:
            ok = "netem" not in snapshots[c] and "netem" not in pre_t0[c]
            detail = "no netem expected; none found" if ok else f"UNEXPECTED: {snapshots[c].strip()[:100]}"
        else:
            ok = got is not None and abs(got[0] - exp["delay_ms"]) <= 1.0 and abs(got[1] - exp["jitter_ms"]) <= 0.06 and abs(got[2] - exp["loss_pct"]) <= 0.01
            if v["b2_netem"]:
                ok = ok and "netem" not in pre_t0[c]
            detail = f"expected {exp['delay_ms']:.1f}/{exp['jitter_ms']:.1f}ms/{exp['loss_pct']:.2f}%, tc shows {got}"
        cont_checks.append({"container": c, "kind": v["kind"], "expected": "b2" if v["b2_netem"] else ("background" if v["bg_netem"] else "clean"), "ok": ok, "detail": detail})
    # latency by container class (B1/M2 sessions only; M1 mixes two connections)
    lat, lat_errors = {}, []
    for cid, s in sessions.items():
        if s["scenario"] in ("B1", "M2", "S0"):
            try:
                l = mechanics.latencies(dlog, events, cid)
                lat[cid] = statistics.median(v for _, v in l)
            except ValueError as e:   # logged sends and traced PUBLISHes differ: a structural failure of this run, reported, not a crash
                lat_errors.append(f"{cid}: {e}")
    for c, v in conts.items():
        if v["kind"] == "mixed":
            ms = [lat[cid] for cid in v["client_ids"] if cid in lat]
            want_bg = bool(v["bg_netem"])
            ok = all((0.035 <= m <= 0.08) if want_bg else (m < 0.03) for m in ms)
            cont_checks.append({"container": c, "kind": "latency", "expected": "background ~50 ms" if want_bg else "clean (<30 ms)", "ok": ok,
                                "detail": f"median one-way latency per B1/M2 session (ms): {[round(m * 1e3, 1) for m in ms]}"})

    drops = re.search(r"(\d+) packets dropped by kernel", tcpdump_stats)
    n_dropped = int(drops.group(1)) if drops else None
    if lat_errors:
        cont_checks.append({"container": "all", "kind": "latency", "expected": "logged sends == traced PUBLISHes", "ok": False, "detail": "; ".join(lat_errors[:5])})
    # ---- acceptance A-K ----
    from collections import Counter
    quota = Counter(json.loads(l)["scenario"] for l in open(gt))
    want_quota = dict(composition) if composition is not None else dict(scenarios.SPLIT_COMPOSITION_BY_VARIANT[variant][split][0])
    n_conn = sum(1 for e in events if e.kind == "CONNECT")
    taken = sum(1 for l in broker_log.splitlines() if "session taken over" in l)
    m1_ids = [cid for cid, s in sessions.items() if s["scenario"] == "M1"]
    fin = all(math.isfinite(float(r[k])) for r in feats for k in features.FEATURE_NAMES)
    pcap = os.path.join(ein, "trace.pcap")
    du = int(subprocess.run(["du", "-sb", rdir], capture_output=True, text=True).stdout.split()[0])
    edges = Counter(e["edge_case"] for e in rep["edge_cases"])
    summary = {
        "A_all_40_sessions_completed": {"ok": not failed and not lat_errors and all(done.values()) and len(sessions) == 40, "containers_done": done, "sessions_with_failed_checks": {k: [x["check"] for x in v] for k, v in failed.items()}},
        "B_quota_exact": {"ok": dict(quota) == want_quota, "quota": dict(quota), "expected": want_quota},
        "C_feature_rows": {"ok": len(feats) == 40 and {r["client_id"] for r in feats} == set(sessions), "rows": len(feats)},
        "D_all_7_finite": {"ok": fin, "n_features": len(features.FEATURE_NAMES)},
        "E_attribution": {"ok": rep["diagnostics"]["unattributed_dropped"] == 0 and rep["n_distinct_client_ids_in_trace"] == 40 and n_conn == 40 + len(m1_ids),
                          "diagnostics": rep["diagnostics"], "distinct_client_ids_in_trace": rep["n_distinct_client_ids_in_trace"],
                          "connects_in_trace": n_conn, "connects_expected": 40 + len(m1_ids)},
        "F_b2_only_intended_containers": {"ok": all(c["ok"] for c in cont_checks if c["expected"] in ("b2", "clean")), "container_checks": cont_checks},
        "G_background_variability": {"ok": all(c["ok"] for c in cont_checks if c["expected"] != "b2"),
                                     "bg_containers": [c for c, v in conts.items() if v["bg_netem"]],
                                     "eligible_non_b2_containers": [c for c, v in conts.items() if v["kind"] != "B2"],
                                     "profile": scenarios.BG_NETEM, "probability": scenarios.BG_PROBABILITY},
        "H_m1_takeover": {"ok": n_conn == 40 + len(m1_ids) and taken == len(m1_ids) and not any(
            x for cid in m1_ids for x in failed.get(cid, [])), "takeover_log_lines": taken, "m1_sessions": m1_ids},
        "I_sizes_bytes": {"pcap": os.path.getsize(pcap), "raw_run_dir": du},
        "J_resources": {"wall_clock_s": time.time() - wall0, "samples": len(stats),
                        "peak_total_cpu_pct_all_containers": max((x["cpu_pct"] for x in stats if "cpu_pct" in x), default=None),
                        "peak_total_mem_mb_all_containers": max((x["mem_mb"] for x in stats if "mem_mb" in x), default=None),
                        "containers_sampled": max((x["n"] for x in stats if "n" in x), default=None), "sampling_errors": sum(1 for x in stats if "sampling_error" in x)},
        "K_edge_conventions": dict(edges),
        "M_capture_no_kernel_drops": {"ok": n_dropped == 0, "packets_dropped_by_kernel": n_dropped, "tcpdump_stats": tcpdump_stats.strip()[-300:]},
        "L_one_mqtt_message_per_segment": {"ok": rep["segment_audit"]["streams_with_byte_mismatch"] == 0 and rep["segment_audit"]["frames_with_pdu_len_mismatch"] == 0
                                           and rep["diagnostics"]["seq_overlap_conflicts"] == 0, "audit": rep["segment_audit"]},
    }
    all_ok = all(v.get("ok", True) for v in summary.values())
    meta = {"run_id": run_id, "split": split, "master_seed": SEED, "all_ok": all_ok, "composition": want_quota, "variant": variant, "derived_seeds": run["derived_seeds"], "git": git, "docker": docker_meta,
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "t0_epoch": t0,
            "containers": {c: {k: v[k] for k in ("kind", "client_ids", "bg_netem", "b2_netem")} for c, v in conts.items()},
            "infra_retries": RL, "netem_b2_evidence": b2_evidence, "netem_final_snapshots": snapshots, "netem_snapshot_source": snap_source, "feature_config_hash": rep["feature_config_hash"],
            "features_py_sha256": rep["features_py_sha256"], "pcap_sha256": rep["pcap_sha256"], "ground_truth_sha256": sha256_file(gt),
            "plan_sha256": scenarios.plan_hash(run), "extract_diagnostics": rep["diagnostics"], "n_events": rep["n_events"],
            "summary": summary, "per_session_checks": results, "resource_samples": stats}
    json.dump(meta, open(os.path.join(rdir, "meta.json"), "w"), indent=1, default=str)
    json.dump(summary, open(os.path.join(rdir, "summary.json"), "w"), indent=1, default=str)
    return meta


def main(allow_dirty: bool) -> int:
    git = git_state(ROOT)
    require_clean(git, "SMOKE", allow_dirty_smoke=allow_dirty)
    meta = execute_run("SMOKE", SEED, os.path.join(OUT_ROOT, f"SMOKE-SCALE40-{SEED}"), git, g2.docker_info())
    for k, v in meta["summary"].items():
        print(k, "OK" if v.get("ok", True) else "FAIL", json.dumps({a: b for a, b in v.items() if a != "ok"}, default=str)[:700])
    return 0


if __name__ == "__main__":
    sys.exit(main("--allow-dirty" in sys.argv))
