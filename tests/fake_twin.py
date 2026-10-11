"""Synthetic raw tree for matched twin runs in the REAL on-disk format (test support only; no Docker)."""
import csv
import hashlib
import json
import os

from postdrift_auth import features, scenarios
from postdrift_auth.features import extract_features
from postdrift_auth.trace import write_trace_csv
from tests.helpers import ideal_events

COMMIT = "c" * 40


def build_stage(root, label, seeds, kind):
    os.makedirs(root, exist_ok=True)
    json.dump({"git": {"commit": COMMIT, "git_dirty": False}}, open(os.path.join(root, "matched_manifest.json"), "w"))
    for sd in seeds:
        fam = scenarios.build_twin_family(sd, kind)
        for arm in ("A", "B"):
            plan = fam["arms"][arm]
            d = os.path.join(root, f"{label}-{sd}-{arm}")
            os.makedirs(os.path.join(d, "extract_in"))
            os.makedirs(os.path.join(d, "extract_out"))
            events, gt, rows = [], [], []
            for s in plan["sessions"]:
                events += ideal_events(s)
            events.sort(key=lambda e: e.t)
            for s in plan["sessions"]:
                vec = extract_features(events, s["client_id"], s["t_a"])
                rows.append({"client_id": s["client_id"], "t_a": s["t_a"], **{k: repr(v) for k, v in vec.items()}})
                gt.append({"session_id": s["client_id"], "scenario": s["scenario"], "y": s["y"], "tau": s["tau"], "t_a": s["t_a"], "params": s["params"],
                           "family_seed": sd, "arm": arm, "pair_id": s["pair_id"]})
            with open(os.path.join(d, "ground_truth.jsonl"), "w") as f:
                f.write("".join(json.dumps(g, sort_keys=True) + "\n" for g in gt))
            open(os.path.join(d, "extract_in", "trace.pcap"), "wb").write(f"fake {label}-{sd}-{arm}".encode())
            write_trace_csv(events, os.path.join(d, "extract_out", "trace_events.csv"))
            with open(os.path.join(d, "extract_out", "features.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["client_id", "t_a", *features.FEATURE_NAMES])
                w.writeheader()
                w.writerows(rows)
            audit = {"streams": 50, "tcp_payload_bytes": 1, "pdu_bytes": 1, "streams_with_byte_mismatch": 0, "frames_with_pdu_len_mismatch": 0, "mismatching_streams": {}}
            json.dump({"diagnostics": {"unattributed_dropped": 0, "seq_overlap_conflicts": 0, "duplicate_segments_dropped": 0}, "n_distinct_client_ids_in_trace": 40,
                       "segment_audit": audit, "edge_cases": []}, open(os.path.join(d, "extract_out", "extract_report.json"), "w"))
            sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
            summ = {"H_m1_takeover": {"ok": True}, "J_resources": {"wall_clock_s": 1.0}, "L_one_mqtt_message_per_segment": {"ok": True, "audit": audit}}
            meta = {"run_id": f"{label}-{sd}-{arm}", "master_seed": sd, "all_ok": True, "git": {"commit": COMMIT, "git_dirty": False}, "docker": {}, "derived_seeds": plan["derived_seeds"],
                    "pcap_sha256": sha(os.path.join(d, "extract_in", "trace.pcap")), "ground_truth_sha256": sha(os.path.join(d, "ground_truth.jsonl")),
                    "plan_sha256": scenarios.plan_hash(plan), "feature_config_hash": features.config_hash(), "summary": summ}
            json.dump(meta, open(os.path.join(d, "meta.json"), "w"))
