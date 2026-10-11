"""Stage 2B Gate 2: six-scenario SMOKE run (generator + canonical-trace + feature smoke test). SMOKE seeds only.

Architecture per run (own Docker network): mosquitto broker; tcpdump sidecar in the broker's network namespace writing a PCAP to a
mounted path (canonical trace); device container `dev` (target identity), `ctl` (S0 control identity on a separate container),
`att` (attacker, M1 only). Plans are computed from seeds before anything starts. netem is applied to `dev` at tau by `docker exec`.
Features are extracted by a SEPARATE process that is given only PCAP + T0 + {client_id, t_a}; ground truth is written by this
orchestrator to a different place and read only by the mechanics checker.

usage: python scripts/generate_data/run_smoke_gate2.py [SCENARIO ...]   (default: all six, in parallel)
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from postdrift_auth import features, mechanics, scenarios  # noqa: E402
from postdrift_auth.provenance import git_state as _git_state  # noqa: E402
from postdrift_auth.seeds import derived_seed, split_of_seed  # noqa: E402
from postdrift_auth.trace import read_trace_csv, sha256_file  # noqa: E402

IMG_TOOLS = "nicolaka/netshoot"
IMG_BROKER = "eclipse-mosquitto:2"
PAHO = os.path.join(ROOT, ".venv/lib/python3.12/site-packages/paho")
OUT_ROOT = os.path.join(ROOT, "results/raw/smoke_gate2")
SMOKE_SEEDS = {s: 9_000_000 + i for i, s in enumerate(scenarios.SCENARIOS)}  # fixed a priori, never re-chosen
STARTUP_MARGIN = 30.0


def sh(*args, check=True, capture=True):
    return subprocess.run(list(args), check=check, capture_output=capture, text=True)


def docker_info():
    v = json.loads(sh("docker", "version", "--format", "{{json .}}").stdout)
    imgs = {i: sh("docker", "image", "inspect", i, "--format", "{{.Id}}").stdout.strip() for i in (IMG_TOOLS, IMG_BROKER)}
    return {"client": v["Client"]["Version"], "server": v["Server"]["Version"], "images": imgs,
            "tshark": sh("tshark", "--version").stdout.splitlines()[0],
            "paho_mqtt": [d for d in os.listdir(os.path.dirname(PAHO)) if d.startswith("paho_mqtt")]}


def git_state():
    return _git_state(ROOT)


def wait_until(t_abs):
    while (d := t_abs - time.time()) > 0:
        time.sleep(min(d, 0.05))


def run_one(scenario: str, docker_meta: dict, git_meta: dict) -> dict:
    wall_start = time.time()
    seed = SMOKE_SEEDS[scenario]
    assert split_of_seed(seed) == "SMOKE"
    run_id = f"SMOKE-{scenario}-{seed}"
    rdir = os.path.join(OUT_ROOT, run_id)
    ein, eout = os.path.join(rdir, "extract_in"), os.path.join(rdir, "extract_out")
    for d in (rdir, ein, eout, os.path.join(rdir, "logs")):
        os.makedirs(d, exist_ok=True)
    run = scenarios.build_run(seed, scenario)
    target, control = run["sessions"]
    net, pre = f"smoke{seed}", f"smoke{seed}"
    names = {"broker": f"{pre}-broker", "cap": f"{pre}-cap", "dev": f"{pre}-dev", "ctl": f"{pre}-ctl", "att": f"{pre}-att"}
    t_end = max(s["t_a"] for s in run["sessions"]) + scenarios.END_TAIL + 3
    sh("docker", "network", "create", net)
    started = []
    try:
        sh("docker", "run", "-d", "--name", names["broker"], "--network", net, "--network-alias", "broker",
           "-v", f"{ROOT}/testbed/mosquitto.conf:/mosquitto/config/mosquitto.conf:ro", IMG_BROKER)
        started.append(names["broker"])
        sh("docker", "run", "-d", "--name", names["cap"], "--network", f"container:{names['broker']}", "--cap-add", "NET_ADMIN",
           "--cap-add", "NET_RAW", "-v", f"{ein}:/out", IMG_TOOLS, "sh", "-c", "tcpdump -i eth0 -n -U -w /out/trace.pcap tcp port 1883")
        started.append(names["cap"])
        time.sleep(3)
        t0 = time.time() + STARTUP_MARGIN
        containers = [("dev", [target["legit"]]), ("ctl", [control["legit"]])]
        if target["attacker"]:
            containers.append(("att", [target["attacker"]]))
        for cname, idents in containers:
            plan = {"t0_epoch": t0, "broker_host": "broker", "identities": idents}
            json.dump(plan, open(os.path.join(rdir, f"plan_{cname}.json"), "w"), indent=1)
            sh("docker", "run", "-d", "--name", names[cname], "--network", net, "--cap-add", "NET_ADMIN",
               "-v", f"{PAHO}:/opt/py/paho:ro", "-v", f"{ROOT}/testbed/device_client.py:/app/device_client.py:ro",
               "-v", f"{rdir}:/run", "-e", "PYTHONPATH=/opt/py", IMG_TOOLS, "python3", "/app/device_client.py",
               f"/run/plan_{cname}.json", f"/run/logs/{cname}.jsonl")
            started.append(names[cname])
        netem_ev = None
        if target["netem"]:
            n = target["netem"]
            wait_until(t0 + target["tau"] - 0.5)
            ev = {k: sh("docker", "exec", names[c], "tc", "qdisc", "show", "dev", "eth0").stdout for k, c in (("dev_pre", "dev"), ("ctl_pre", "ctl"))}
            wait_until(t0 + target["tau"])
            a0 = time.time()
            sh("docker", "exec", names["dev"], "tc", "qdisc", "add", "dev", "eth0", "root", "netem", "delay",
               f"{n['delay_ms']:.1f}ms", f"{n['jitter_ms']:.1f}ms", "loss", f"{n['loss_pct']:.2f}%")
            a1 = time.time()
            time.sleep(1)
            ev.update({k: sh("docker", "exec", names[c], "tc", "qdisc", "show", "dev", "eth0").stdout for k, c in (("dev_post", "dev"), ("ctl_post", "ctl"))})
            ev.update(apply_start_rel=a0 - t0, apply_end_rel=a1 - t0, netem_cmd=n)
            netem_ev = ev
        wait_until(t0 + t_end)
        for cname, _ in containers:
            sh("docker", "wait", names[cname])
        time.sleep(1.5)
        sh("docker", "stop", "-t", "3", names["cap"])
        lg = sh("docker", "logs", names["broker"])
        broker_log = lg.stdout + lg.stderr
        open(os.path.join(rdir, "logs", "broker.log"), "w").write(broker_log)
    finally:
        for c in started:
            sh("docker", "rm", "-f", c, check=False)
        sh("docker", "network", "rm", net, check=False)

    # ground truth: written by the orchestrator, outside extract_in, never given to the extractor
    gt_path = os.path.join(rdir, "ground_truth.jsonl")
    with open(gt_path, "w") as f:
        for s in run["sessions"]:
            f.write(json.dumps({"session_id": s["client_id"], "scenario": s["scenario"], "y": s["y"], "tau": s["tau"],
                                "t_a": s["t_a"], "params": s["params"]}, sort_keys=True) + "\n")
    trig = os.path.join(ein, "trigger.jsonl")
    with open(trig, "w") as f:
        for s in run["sessions"]:
            f.write(json.dumps({"client_id": s["client_id"], "t_a": s["t_a"]}) + "\n")
    t0_str = repr(t0)
    ex = subprocess.run([sys.executable, "-m", "postdrift_auth.extract_cli", "--pcap", os.path.join(ein, "trace.pcap"),
                         "--t0", t0_str, "--trigger", trig, "--out-dir", eout], cwd=ein, env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
                        capture_output=True, text=True)
    if ex.returncode:
        raise RuntimeError(f"{run_id}: extraction failed:\n{ex.stderr}")
    events = read_trace_csv(os.path.join(eout, "trace_events.csv"))
    dev_log = [json.loads(line) for line in open(os.path.join(rdir, "logs", "dev.jsonl"))]
    ctl_log = [json.loads(line) for line in open(os.path.join(rdir, "logs", "ctl.jsonl"))]
    checks = {"target": mechanics.check_scenario(target, events, device_log=dev_log + ctl_log, netem_evidence=netem_ev, broker_log=broker_log),
              "control": mechanics.check_scenario(control, events)}
    feats = list(__import__("csv").DictReader(open(os.path.join(eout, "features.csv"))))
    finite = all(__import__("math").isfinite(float(r[k])) for r in feats for k in features.FEATURE_NAMES)
    rep = json.load(open(os.path.join(eout, "extract_report.json")))
    meta = {
        "run_id": run_id, "split": split_of_seed(seed), "master_seed": seed, "derived_seeds": run["derived_seeds"],
        "scenario": scenario, "sessions": [{k: s[k] for k in ("client_id", "scenario", "y", "base", "tau", "t_a", "params", "netem")} for s in run["sessions"]],
        "t0_epoch": t0, "netem_applied": netem_ev, "git": git_meta, "docker": docker_meta,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "wall_clock_s": time.time() - wall_start,
        "feature_config_hash": rep["feature_config_hash"], "features_py_sha256": rep["features_py_sha256"],
        "pcap_sha256": rep["pcap_sha256"], "ground_truth_sha256": sha256_file(gt_path),
        "plan_sha256": scenarios.plan_hash(run), "extract_diagnostics": rep["diagnostics"], "n_events": rep["n_events"],
        "features_all_finite": finite, "n_feature_rows": len(feats), "mechanics": checks,
    }
    json.dump(meta, open(os.path.join(rdir, "meta.json"), "w"), indent=1, default=str)
    return meta


if __name__ == "__main__":
    which = sys.argv[1:] or list(scenarios.SCENARIOS)
    docker_meta, git_meta = docker_info(), git_state()
    t_all = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(which)) as ex:
        futs = {s: ex.submit(run_one, s, docker_meta, git_meta) for s in which}
        for s, f in futs.items():
            try:
                m = f.result()
                bad = [c for part in m["mechanics"].values() for c in part if not c["ok"]]
                print(f"{s}: wall {m['wall_clock_s']:.0f}s events={m['n_events']} finite={m['features_all_finite']} failed_checks={len(bad)}")
                for c in bad:
                    print("   FAIL", c["check"], "|", c["detail"])
            except Exception as e:  # noqa: BLE001
                print(f"{s}: ERROR {e}")
    print(f"total wall {time.time() - t_all:.0f}s")
