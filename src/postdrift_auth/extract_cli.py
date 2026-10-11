"""Feature-extraction process. Inputs: PCAP, run-start T0, trigger file {client_id, t_a}. NO label/scenario input exists.

usage: python -m postdrift_auth.extract_cli --pcap P --t0 T0 --trigger trigger.jsonl --out-dir D
Writes D/trace_events.csv, D/features.csv, D/extract_report.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess

from . import features
from .trace import parse_pcap_audited, sha256_file, write_trace_csv


def run(pcap: str, t0: str, trigger_path: str, out_dir: str) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    events, diag, audit = parse_pcap_audited(pcap, t0)
    write_trace_csv(events, os.path.join(out_dir, "trace_events.csv"))
    with open(trigger_path) as f:
        triggers = [json.loads(line) for line in f if line.strip()]
    rows, edge = [], []
    for tr in triggers:
        vec, flags = features.extract_features_with_flags(events, tr["client_id"], tr["t_a"])
        rows.append({"client_id": tr["client_id"], "t_a": tr["t_a"], **vec})
        edge += [{"client_id": tr["client_id"], "edge_case": fl} for fl in flags]
    with open(os.path.join(out_dir, "features.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["client_id", "t_a", *features.FEATURE_NAMES])
        w.writeheader()
        for r in rows:
            w.writerow({k: repr(v) if isinstance(v, float) else v for k, v in r.items()})
    tshark_v = subprocess.run(["tshark", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    report = {"pcap_sha256": sha256_file(pcap), "t0": t0, "n_events": len(events), "diagnostics": diag, "segment_audit": audit, "edge_cases": edge, "n_triggered_sessions": len(triggers),
              "n_distinct_client_ids_in_trace": len({e.client_id for e in events}), "tshark": tshark_v,
              "feature_config_hash": features.config_hash(), "features_py_sha256": sha256_file(features.__file__)}
    with open(os.path.join(out_dir, "extract_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--t0", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    run(a.pcap, a.t0, a.trigger, a.out_dir)
