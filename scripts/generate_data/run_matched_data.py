"""Matched-drift data generation: twin families (two physical runs per family) from configs/matched_drift_v1.json.

usage: python scripts/generate_data/run_matched_data.py --stage dry|m12|m3
Stages m12 and m3 need configs/matched_drift_v1_lock.json (written after the dry run). Refuses a dirty tree (every stage uses a registered split
range). A failed attempt is renamed <run>.FAILED_attemptN and kept; each run is retried once with the same seed. Earlier bookkeeping files are kept.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime
import json
import os
import sys
import threading
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import run_scale_smoke as rs  # noqa: E402
import run_smoke_gate2 as g2  # noqa: E402
from postdrift_auth import matched_config as mc, scenarios  # noqa: E402
from postdrift_auth.provenance import git_state, require_clean  # noqa: E402
from postdrift_auth.seeds import split_of_seed  # noqa: E402
from postdrift_auth.twin_checks import pair_invariants  # noqa: E402


SYNC_MARGIN_S = 150.0   # start-up window shared by the two runs of a family (infrastructure setting, recorded in the manifest)


def preserve(rdir: str) -> str:
    n = 1
    while os.path.exists(f"{rdir}.FAILED_attempt{n}"):
        n += 1
    dest = f"{rdir}.FAILED_attempt{n}"
    os.rename(rdir, dest)
    return dest


def _all_ok(path):
    with open(path) as f:
        return json.load(f).get("all_ok") is True


def do_run(label, split, kind, seed, arm, plan, git, docker_meta, out, margin, t0_first=None):
    rid = f"{label}-{seed}-{arm}"
    rdir = os.path.join(out, rid)
    rec = {"run_id": rid, "seed": seed, "arm": arm, "attempts": [], "status": None}
    if os.path.exists(os.path.join(rdir, "meta.json")) and _all_ok(os.path.join(rdir, "meta.json")):
        rec["status"] = "reused_existing_ok"
        return rec
    for attempt in (1, 2):
        if os.path.exists(rdir):
            rec["attempts"].append({"before_attempt": attempt, "preserved_as": preserve(rdir)})
        t = time.time()
        try:
            # first attempt of a lockstep pair uses the pair's shared start time; a retry runs alone with its own margin
            meta = rs.execute_run(split, seed, rdir, git, docker_meta, margin=margin, plan=plan, run_id=rid, docker_tag=f"mdr{seed}{arm.lower()}",
                                  composition=scenarios.twin_composition(kind), t0_epoch=t0_first if attempt == 1 else None, stats_interval_s=30.0)
            bad = [k for k, v in meta["summary"].items() if not v.get("ok", True)]
            rec["attempts"].append({"attempt": attempt, "wall_s": time.time() - t, "all_ok": meta["all_ok"], "failed_summary_keys": bad})
            if meta["all_ok"]:
                rec["status"] = "ok"
                return rec
        except Exception:  # noqa: BLE001
            rec["attempts"].append({"attempt": attempt, "wall_s": time.time() - t, "exception": traceback.format_exc()[-1500:]})
    if os.path.exists(rdir):
        rec["attempts"].append({"after_attempt": 2, "preserved_as": preserve(rdir)})
    rec["status"] = "FAILED"
    return rec


def main(stage: str) -> int:
    cfg = mc.load()
    lock = mc.load_lock(cfg) if stage in ("m12", "m3") else None
    blk = cfg["seed_blocks"][mc.STAGE_BLOCK[stage]]
    git = git_state(ROOT)
    require_clean(git, blk["split"])
    seeds = mc.family_seeds(cfg, stage, lock)
    assert all(split_of_seed(s) == blk["split"] for s in seeds)
    fams = {sd: scenarios.build_twin_family(sd, blk["kind"]) for sd in seeds}
    for sd, fam in fams.items():
        v = pair_invariants(fam)
        if v:
            raise SystemExit(f"pair invariants violated for family {sd}: {v[:3]}")
    out = os.path.join(mc.RAW_ROOT, stage)
    os.makedirs(out, exist_ok=True)
    for fname in ("matched_manifest.json", "run_index.json", "load_log.json"):
        old = os.path.join(out, fname)
        if os.path.exists(old):
            k = 1
            while os.path.exists(f"{old}.invocation{k}"):
                k += 1
            os.rename(old, f"{old}.invocation{k}")
    docker_meta = g2.docker_info()
    parallel = cfg["execution"]["parallel_runs"]
    jobs = [(sd, arm) for sd in seeds for arm in ("A", "B")]
    manifest = {"stage": stage, "config_sha256": cfg["_sha256"], "lock_sha256": (__import__("hashlib").sha256(open(mc.LOCK_PATH, "rb").read()).hexdigest() if lock else None),
                "git": git, "docker": docker_meta, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "parallel_runs": parallel,
                "n_families": len(seeds), "family_seeds": seeds, "scheduling": {"mode": "lockstep pairs", "sync_margin_s": SYNC_MARGIN_S, "stats_interval_s": 30.0, "note": "introduced after load-induced failures of the staggered two-run schedule; no scientific setting changed"}}
    json.dump(manifest, open(os.path.join(out, "matched_manifest.json"), "w"), indent=1)
    load_log, stop = [], threading.Event()

    def sampler():
        t0 = time.time()
        while not stop.is_set():
            load_log.append({"t": time.time() - t0, "load1": os.getloadavg()[0]})
            stop.wait(15)

    th = threading.Thread(target=sampler)
    th.start()
    wall0 = time.time()
    recs = []
    # Lockstep scheduling (infrastructure only): the two arms of a family start their containers together and share one start time T0, so the
    # start-up burst of one run never hits the other run while its devices are active. Already valid runs are reused.
    lockstep_margin = SYNC_MARGIN_S
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        for sd in seeds:
            t0_pair = time.time() + lockstep_margin
            futs = [ex.submit(do_run, blk["label"], blk["split"], blk["kind"], sd, arm, fams[sd]["arms"][arm], git, docker_meta, out,
                              cfg["execution"]["margin_s"], t0_pair) for arm in ("A", "B")]
            for f in cf.as_completed(futs):
                r = f.result()
                recs.append(r)
                print(r["run_id"], r["status"], f"{len(recs)}/{len(jobs)}", flush=True)
                json.dump(sorted(recs, key=lambda r: r["run_id"]), open(os.path.join(out, "run_index.json"), "w"), indent=1)
    stop.set()
    th.join()
    manifest.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), wall_clock_s=time.time() - wall0,
                    peak_load1=max((x["load1"] for x in load_log), default=None), n_cpus=os.cpu_count(),
                    statuses={s: sum(1 for r in recs if r["status"] == s) for s in {r["status"] for r in recs}})
    json.dump(manifest, open(os.path.join(out, "matched_manifest.json"), "w"), indent=1)
    json.dump(load_log, open(os.path.join(out, "load_log.json"), "w"))
    print(manifest["statuses"], f"wall {manifest['wall_clock_s']:.0f}s", f"peak load1 {manifest['peak_load1']}")
    return 0 if all(r["status"] in ("ok", "reused_existing_ok") for r in recs) else 1


if __name__ == "__main__":
    st = sys.argv[sys.argv.index("--stage") + 1]
    sys.exit(main(st))
