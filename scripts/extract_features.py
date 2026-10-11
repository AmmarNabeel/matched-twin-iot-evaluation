"""Feature extraction from a PCAP (needs tshark on the PATH).

usage: python scripts/extract_features.py                      (runs the example in examples/pcap_example and compares with the stored features)
       python scripts/extract_features.py --pcap P --trigger T --t0 EPOCH --out-dir D

The extraction process gets only the PCAP, the run start time T0 and the trigger file (client id and trigger time). It has no access to labels.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import extract_cli, features  # noqa: E402

EX = os.path.join(ROOT, "examples", "pcap_example")


def read(path):
    with open(path, newline="") as f:
        return {r["client_id"]: [float(r[k]) for k in features.FEATURE_NAMES] for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", default=os.path.join(EX, "trace.pcap"))
    ap.add_argument("--trigger", default=os.path.join(EX, "trigger.jsonl"))
    ap.add_argument("--t0", default=open(os.path.join(EX, "T0.txt")).read().strip())
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "outputs", "pcap_example"))
    a = ap.parse_args()
    extract_cli.run(a.pcap, a.t0, a.trigger, a.out_dir)
    print("features written to", os.path.join(a.out_dir, "features.csv"))
    if a.pcap == os.path.join(EX, "trace.pcap"):
        new, ref = read(os.path.join(a.out_dir, "features.csv")), read(os.path.join(EX, "expected_features.csv"))
        worst = max(abs(x - y) for k in ref for x, y in zip(new[k], ref[k]))
        print(f"{len(ref)} sessions; largest difference to the stored features: {worst:.2e}")
        sys.exit(0 if set(new) == set(ref) and worst < 1e-9 else 1)


if __name__ == "__main__":
    main()
