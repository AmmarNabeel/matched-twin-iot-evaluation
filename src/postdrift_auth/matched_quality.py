"""Data-quality gate for matched twin runs. Any structural failure stops the analysis; nothing is repaired or filtered."""
from __future__ import annotations

import glob
import json
import math
import os

from . import features, scenarios
from .matched import run_dir
from .trace import read_trace_csv, sha256_file
from .twin_checks import pair_invariants


def _j(p):
    with open(p) as f:
        return json.load(f)


def _chk(name, ok, detail=""):
    return {"check": name, "ok": bool(ok), "detail": detail}


def check_matched_quality(root: str, label: str, seeds: list[int], kind: str, manifests_glob: str, forbidden_seeds: set[int] = frozenset()) -> dict:
    checks, per_run, hashes, commits_seen = [], [], [], set()
    accepted = set()
    for mp in glob.glob(manifests_glob):
        accepted.add(_j(mp).get("git", {}).get("commit"))
    accepted.discard(None)
    comp = scenarios.twin_composition(kind)
    inv_bad, plan_bad = {}, {}
    for sd in seeds:
        fam = scenarios.build_twin_family(sd, kind)
        v = pair_invariants(fam)
        if v:
            inv_bad[sd] = v[:3]
        for arm in ("A", "B"):
            d = run_dir(root, label, sd, arm)
            if not os.path.exists(os.path.join(d, "meta.json")):
                per_run.append({"run": f"{label}-{sd}-{arm}", "missing": True})
                continue
            m = _j(os.path.join(d, "meta.json"))
            gt = [json.loads(l) for l in open(os.path.join(d, "ground_truth.jsonl")) if l.strip()]
            fr = list(__import__("csv").DictReader(open(os.path.join(d, "extract_out", "features.csv"))))
            rep = _j(os.path.join(d, "extract_out", "extract_report.json"))
            events = read_trace_csv(os.path.join(d, "extract_out", "trace_events.csv"))
            ta = {g["session_id"]: g["t_a"] for g in gt}
            leak = 0
            for r in fr:
                cut = [e for e in events if e.t <= ta[r["client_id"]]]
                vec = features.extract_features(cut, r["client_id"], ta[r["client_id"]])
                leak += any(repr(vec[k]) != r[k] and float(r[k]) != vec[k] for k in features.FEATURE_NAMES)
            quota = {}
            for g in gt:
                quota[g["scenario"]] = quota.get(g["scenario"], 0) + 1
            seg = m["summary"]["L_one_mqtt_message_per_segment"]["audit"]
            plan_ok = m["plan_sha256"] == scenarios.plan_hash(fam["arms"][arm])
            roles_ok = sum(g["y"] for g in gt) == 20 and len(gt) == 40
            pcap, gtp = os.path.join(d, "extract_in", "trace.pcap"), os.path.join(d, "ground_truth.jsonl")
            commits_seen.add(m["git"]["commit"])
            hashes.append(m["pcap_sha256"])
            per_run.append({
                "run": f"{label}-{sd}-{arm}", "n_features": len(fr), "quota_ok": quota == comp, "labels_ok": roles_ok, "plan_reproduces": plan_ok,
                "finite": all(math.isfinite(float(r[k])) for r in fr for k in features.FEATURE_NAMES), "future_leak_rows": leak,
                "unattributed": rep["diagnostics"]["unattributed_dropped"], "distinct_ids": rep["n_distinct_client_ids_in_trace"],
                "segment_mismatch": seg["streams_with_byte_mismatch"] + seg["frames_with_pdu_len_mismatch"] + rep["diagnostics"]["seq_overlap_conflicts"],
                "m1_ok": m["summary"]["H_m1_takeover"]["ok"], "all_ok": m["all_ok"] is True and all(x.get("ok", True) for x in m["summary"].values()),
                "provenance_ok": m["git"]["git_dirty"] is False and (not accepted or m["git"]["commit"] in accepted) and m["pcap_sha256"] == sha256_file(pcap)
                                 and m["ground_truth_sha256"] == sha256_file(gtp) and m["feature_config_hash"] == features.config_hash(),
                "edge_cases": rep["edge_cases"], "wall_s": m["summary"]["J_resources"]["wall_clock_s"]})
    present = [r for r in per_run if not r.get("missing")]
    bad = lambda key, f=lambda v: v: [r["run"] for r in present if not f(r[key])]  # noqa: E731
    checks += [
        _chk("all runs present (2 per family, exact seeds)", len(present) == 2 * len(seeds), f"{len(present)} of {2 * len(seeds)}"),
        _chk("no dry-run or other forbidden family seed is used", not (set(seeds) & set(forbidden_seeds)), ""),
        _chk("pair invariants hold for every family (shared pre-drift session, complementary roles, 20/20, 2/2 per container, equal link conditions)", not inv_bad, str(inv_bad)[:300]),
        _chk("exact composition in every run", not bad("quota_ok"), bad("quota_ok")),
        _chk("20 malicious + 20 benign in every run", not bad("labels_ok"), bad("labels_ok")),
        _chk("stored plan hash reproduces from the seed", not bad("plan_reproduces"), bad("plan_reproduces")),
        _chk("40 feature rows per run", not bad("n_features", lambda v: v == 40), bad("n_features", lambda v: v == 40)),
        _chk("all 7 features finite", not bad("finite"), bad("finite")),
        _chk("no future-event leakage", not bad("future_leak_rows", lambda v: v == 0), ""),
        _chk("no unattributed TCP streams; 40 distinct client ids", all(r["unattributed"] == 0 and r["distinct_ids"] == 40 for r in present), ""),
        _chk("one-MQTT-message-per-segment assumption holds (byte audit)", not bad("segment_mismatch", lambda v: v == 0), bad("segment_mismatch", lambda v: v == 0)),
        _chk("M1 takeover mechanics correct", not bad("m1_ok"), bad("m1_ok")),
        _chk("all per-run mechanics checks passed", not bad("all_ok"), bad("all_ok")),
        _chk("provenance present and consistent (clean tree, accepted commits, recomputed hashes)", not bad("provenance_ok"), bad("provenance_ok")),
        _chk("unique PCAP hashes", len(set(hashes)) == len(hashes), ""),
    ]
    edge = {}
    for r in present:
        for e in r["edge_cases"]:
            edge[e["edge_case"]] = edge.get(e["edge_case"], 0) + 1
    return {"passed": all(c["ok"] for c in checks), "checks": checks, "n_runs": len(present), "commits": sorted(commits_seen), "accepted_commits": sorted(accepted),
            "edge_case_counts": edge, "failed_attempt_dirs_preserved": sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, f"{label}-*.FAILED_attempt*"))),
            "per_run": [{k: v for k, v in r.items() if k != "edge_cases"} for r in per_run]}
