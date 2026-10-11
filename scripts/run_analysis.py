"""Locked matched-twin analysis, run from the released feature tables.

usage: python scripts/run_analysis.py [--verify] [--check-data]

Reads the per-run feature tables and ground-truth files in data/ (see docs/data_description.md), fits the frozen detectors on the source regime,
computes novelty, matching, PMP, pooled AUC and the family-clustered bootstrap intervals for the M1/M2 families (confirmatory) and the M3 families
(descriptive stress test), and writes the results to outputs/matched_drift_v1/final/.

This is the same analysis code as in the study (src/postdrift_auth/matched_analysis.py). Two steps of the original runner are left out because they need
files that are not released: the raw-data quality gate (PCAP hashes, container logs) and the check for a clean git tree. The pair structure and the
number of runs per family are still checked below.

--check-data only checks that all runs and files are present (a quick dry run) and exits.
--verify compares the new results with the reference results in results/matched_drift_v1/final/ (see scripts/verify_results.py).
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from postdrift_auth import matched as M, matched_analysis as ma, matched_config as mc, matched_report as mr  # noqa: E402


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def write_scores(path, arrays):
    """One row per session: ids, label, novelty (primary and seven-feature) and the score of every detector."""
    d, ln_p, ln_s = arrays["data"], arrays["ln_p"], arrays["ln_s"]
    names = list(arrays["scores"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "arm", "session_id", "pair_id", "y", "scenario_descriptive_only", "ln_N_primary", "ln_N_7feature", *names])
        for i in range(len(d["y"])):
            w.writerow([d["family"][i], d["arm"][i], d["session_id"][i], d["pair_id"][i], d["y"][i], d["scenario"][i], repr(float(ln_p[i])), repr(float(ln_s[i])),
                        *[repr(float(arrays["scores"][n][i])) for n in names]])


def check_runs(root, label, seeds):
    """Every family must have both runs (arm A and arm B) with a feature table and a ground-truth file."""
    for sd in seeds:
        for arm in ("A", "B"):
            d = M.run_dir(root, label, sd, arm)
            for part in ("ground_truth.jsonl", os.path.join("extract_out", "features.csv")):
                if not os.path.exists(os.path.join(d, part)):
                    sys.exit(f"missing {os.path.join(d, part)}; see docs/data_description.md")


def main():
    cfg = mc.load()
    lock = mc.load_lock(cfg)
    if "--check-data" in sys.argv:
        for stage, label in (("m12", "MFIN"), ("m3", "MM3")):
            seeds = mc.family_seeds(cfg, stage, lock)
            check_runs(os.path.join(mc.RAW_ROOT, stage), label, seeds)
            print(f"{stage}: {len(seeds)} families, {2 * len(seeds)} runs present")
        return
    src = ma.Source(cfg)
    out = os.path.join(mc.OUT_ROOT, "final")
    for stage, label, kind, tag in (("m12", "MFIN", "M12", "primary_m1m2"), ("m3", "MM3", "M3", "stress_m3")):
        root = os.path.join(mc.RAW_ROOT, stage)
        seeds = mc.family_seeds(cfg, stage, lock)
        check_runs(root, label, seeds)
        data = M.load_families(root, label, seeds)
        res, arrays = ma.analyze_families(cfg, lock, src, data, tag, kind)
        res["decision"] = ma.decision(cfg, res) if stage == "m12" else None
        dump(os.path.join(out, f"{tag}_results.json"), res)
        write_scores(os.path.join(out, f"{tag}_session_scores.csv"), arrays)
        mr.write_tables(res, os.path.join(out, "tables"), tag)
        d1 = res["detectors"]["D1_rf_isotonic"]["PMP_primary"]
        print(f"{tag}: {res['n_families']} families, {res['n_pairs']} pairs, retention {100 * res['retention']['primary']:.1f}%, "
              f"D1 PMP {d1['estimate']:.3f} (95% CI {d1['ci95_two_sided'][0]:.3f} to {d1['ci95_two_sided'][1]:.3f}), decision: {res['decision']['label'] if res['decision'] else 'none (descriptive stress test)'}")
    print("results written to", out)


if __name__ == "__main__":
    main()
    if "--verify" in sys.argv and "--check-data" not in sys.argv:
        import verify_results
        sys.exit(verify_results.main(["final"]))
