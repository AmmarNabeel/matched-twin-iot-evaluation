"""Stage 3A pre-modelling data-quality checks. Any structural failure stops the pipeline before a classifier is fitted.

Reads raw run directories only (meta.json, summary, features.csv, trace_events.csv, ground_truth.jsonl, PCAP hash). Nothing is repaired
or filtered: the report states pass/fail per check and lists the preserved FAILED attempts.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

from . import features, scenarios
from .seeds import check_registry, split_of_seed
from .trace import read_trace_csv, sha256_file

REQUIRED_META = ("git", "docker", "derived_seeds", "pcap_sha256", "ground_truth_sha256", "plan_sha256", "feature_config_hash",
                 "features_py_sha256", "timestamp_utc", "master_seed", "split", "t0_epoch", "containers")


def run_dirs(root: str, split: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(root, split, f"{split}-*")) if os.path.isdir(d) and ".FAILED_attempt" not in d)


def failed_attempt_dirs(root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "*", "*.FAILED_attempt*")))


def _json(path):
    with open(path) as f:
        return json.load(f)


def _check(name, ok, detail=""):
    return {"check": name, "ok": bool(ok), "detail": detail}


def _read_features(rdir):
    with open(os.path.join(rdir, "extract_out", "features.csv"), newline="") as f:
        return list(csv.DictReader(f))


def _read_gt(rdir):
    with open(os.path.join(rdir, "ground_truth.jsonl")) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def check_quality(root: str, cfg: dict, expected_commit: str | None = None) -> dict:
    checks = []
    check_registry()
    manifest_path = os.path.join(root, "pilot_manifest.json")
    manifest = _json(manifest_path) if os.path.exists(manifest_path) else {}
    commit = expected_commit or manifest.get("git", {}).get("commit")
    # every invocation of the runner stores its own manifest (older ones are kept as pilot_manifest.json.invocationN)
    accepted = {commit} if expected_commit else set()
    for mp in glob.glob(os.path.join(root, "pilot_manifest.json*")):
        accepted.add(_json(mp).get("git", {}).get("commit"))
    accepted.discard(None)
    all_seeds, all_run_ids, pcap_hashes, keys = [], [], [], []
    per_run = []
    for split, sc in cfg["splits"].items():
        want_seeds = [sc["seed_base"] + i for i in range(sc["n_runs"])]
        dirs = run_dirs(root, split)
        got_seeds = []
        for d in dirs:
            m = _json(os.path.join(d, "meta.json"))
            got_seeds.append(m["master_seed"])
        checks.append(_check(f"{split}: expected number of runs ({sc['n_runs']}) and exact seeds", sorted(got_seeds) == want_seeds,
                             f"found {len(dirs)} run directories, seeds {sorted(got_seeds)[:3]}..."))
        for d in dirs:
            rid = os.path.basename(d)
            m = _json(os.path.join(d, "meta.json"))
            fr, gt = _read_features(d), _read_gt(d)
            quota = {}
            for g in gt:
                quota[g["scenario"]] = quota.get(g["scenario"], 0) + 1
            summ = m["summary"]
            seg = summ["L_one_mqtt_message_per_segment"]["audit"]
            rep = _json(os.path.join(d, "extract_out", "extract_report.json"))
            events = read_trace_csv(os.path.join(d, "extract_out", "trace_events.csv"))
            ta = {g["session_id"]: g["t_a"] for g in gt}
            # future-event leakage: features from the trace truncated at each session's own t_a must equal the stored ones
            leak = 0
            for r in fr:
                cut = [e for e in events if e.t <= ta[r["client_id"]]]
                vec = features.extract_features(cut, r["client_id"], ta[r["client_id"]])
                leak += any(repr(vec[k]) != r[k] and float(r[k]) != vec[k] for k in features.FEATURE_NAMES)
            finite = all(math.isfinite(float(r[k])) for r in fr for k in features.FEATURE_NAMES)
            b2_checks = [c for cid, cs in m["per_session_checks"].items() for c in cs if "netem" in c["check"] or "latency" in c["check"]]
            m1_sessions = [g["session_id"] for g in gt if g["scenario"] == "M1"]
            m1_ok = all(c["ok"] for cid in m1_sessions for c in m["per_session_checks"][cid])
            pcap = os.path.join(d, "extract_in", "trace.pcap")
            gtp = os.path.join(d, "ground_truth.jsonl")
            per_run.append({
                "run": rid, "split": split,
                "n_feature_rows": len(fr), "n_gt_rows": len(gt), "quota_ok": quota == sc["quota"], "quota": quota,
                "seed_in_split": split_of_seed(m["master_seed"]) == split and m["split"] == split, "finite": finite, "future_leak_rows": leak,
                "unattributed": rep["diagnostics"]["unattributed_dropped"], "distinct_ids": rep["n_distinct_client_ids_in_trace"],
                "segment_mismatch": seg["streams_with_byte_mismatch"] + seg["frames_with_pdu_len_mismatch"] + rep["diagnostics"]["seq_overlap_conflicts"],
                "b2_ok": all(c["ok"] for c in b2_checks) and summ["F_b2_only_intended_containers"]["ok"],
                "n_b2_sessions": quota.get("B2", 0), "n_b2_containers": sum(1 for c in m["containers"].values() if c["b2_netem"]),
                "m1_ok": m1_ok and summ["H_m1_takeover"]["ok"], "n_m1": len(m1_sessions),
                "summary_all_ok": m["all_ok"] is True and all(v.get("ok", True) for v in summ.values()),
                "provenance_ok": all(k in m for k in REQUIRED_META) and m["git"]["git_dirty"] is False and (not accepted or m["git"]["commit"] in accepted)
                                 and m["pcap_sha256"] == sha256_file(pcap) and m["ground_truth_sha256"] == sha256_file(gtp)
                                 and m["feature_config_hash"] == features.config_hash(),
                "forbidden_content": any(g["scenario"] == "M3" for g in gt),
                "edge_cases": rep["edge_cases"], "wall_clock_s": summ["J_resources"]["wall_clock_s"],
            })
            all_seeds.append(m["master_seed"])
            all_run_ids.append(rid)
            pcap_hashes.append(m["pcap_sha256"])
            keys += [(rid, g["session_id"]) for g in gt]
    P = lambda f: [r for r in per_run if not f(r)]  # noqa: E731  (runs violating a condition)
    n_total = sum(sc["n_runs"] for sc in cfg["splits"].values())
    checks += [
        _check("total runs", len(per_run) == n_total, f"{len(per_run)} of {n_total}"),
        _check("40 sessions per run (feature rows and ground-truth rows)", not P(lambda r: r["n_feature_rows"] == 40 and r["n_gt_rows"] == 40), [r["run"] for r in P(lambda r: r["n_feature_rows"] == 40 and r["n_gt_rows"] == 40)]),
        _check("exact scenario quotas in every run", not P(lambda r: r["quota_ok"]), [r["run"] for r in P(lambda r: r["quota_ok"])]),
        _check("no seed overlap (all run seeds distinct and in their own split range)", len(set(all_seeds)) == len(all_seeds) and not P(lambda r: r["seed_in_split"]), ""),
        _check("no run/session leakage (unique run ids, unique (run, session) keys, unique PCAP hashes)",
               len(set(all_run_ids)) == len(all_run_ids) and len(set(keys)) == len(keys) and len(set(pcap_hashes)) == len(pcap_hashes), ""),
        _check("all 7 features finite", not P(lambda r: r["finite"]), [r["run"] for r in P(lambda r: r["finite"])]),
        _check("no future-event leakage (features unchanged when the trace is cut at each session's t_a)", not P(lambda r: r["future_leak_rows"] == 0), ""),
        _check("no unattributed TCP streams; 40 distinct client ids per run", not P(lambda r: r["unattributed"] == 0 and r["distinct_ids"] == 40), ""),
        _check("no violated one-MQTT-message-per-segment assumption (byte audit, overlap conflicts)", not P(lambda r: r["segment_mismatch"] == 0), [r["run"] for r in P(lambda r: r["segment_mismatch"] == 0)]),
        _check("all B2 interventions applied correctly (VAL: 4 containers x 4 sessions, netem and latency checks)",
               not P(lambda r: r["b2_ok"]) and all((r["n_b2_containers"] == 4 and r["n_b2_sessions"] == 16) for r in per_run if r["split"] == "VAL")
               and all(r["n_b2_sessions"] == 0 for r in per_run if r["split"] != "VAL"), ""),
        _check("all M1 takeovers correct (4 per run)", not P(lambda r: r["m1_ok"] and r["n_m1"] == 4), [r["run"] for r in P(lambda r: r["m1_ok"] and r["n_m1"] == 4)]),
        _check("all per-run mechanics/summary checks passed", not P(lambda r: r["summary_all_ok"]), [r["run"] for r in P(lambda r: r["summary_all_ok"])]),
        _check("run hashes and provenance present and consistent (clean tree, same commit, PCAP/ground-truth hashes recomputed)", not P(lambda r: r["provenance_ok"]), [r["run"] for r in P(lambda r: r["provenance_ok"])]),
        _check("no M3 and no TEST content", not any(r["forbidden_content"] for r in per_run) and not glob.glob(os.path.join(root, "TEST*")), ""),
    ]
    edge = {}
    for r in per_run:
        for e in r["edge_cases"]:
            edge[e["edge_case"]] = edge.get(e["edge_case"], 0) + 1
    return {"passed": all(c["ok"] for c in checks), "checks": checks, "n_runs": len(per_run), "commit": commit, "accepted_commits": sorted(accepted),
            "edge_case_counts": edge, "failed_attempt_dirs_preserved": [os.path.relpath(p, root) for p in failed_attempt_dirs(root)],
            "per_run": [{k: v for k, v in r.items() if k != "edge_cases"} for r in per_run]}
