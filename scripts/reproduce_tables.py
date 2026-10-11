"""Regenerate the result tables of the paper from the saved results (no model is refitted).

usage: python scripts/reproduce_tables.py

Reads results/matched_drift_v1/{final,stats_review} (or the copies in outputs/ if you recomputed them with run_analysis.py and the check scripts)
and writes CSV tables to outputs/matched_drift_v1/tables/. The numbers are the ones printed in the paper:

  table4_primary_detectors.csv       Table 4: PMP and pooled AUC of the four scores, M1/M2 families
  table5_caliper_m12.csv             Table 5: caliper sensitivity, M1/M2 families (descriptive)
  tableS4_caliper_full.csv           Supplementary Table S4: the same with the M3 families and the smallest family size
  tableS17_m3_detectors.csv          Supplementary Table S17: the four scores on the held-out M3 families (descriptive)
  retention.csv, by_malicious_type.csv, by_structural_signature.csv, decision.csv   numbers quoted in Sections 4.3 to 4.5
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import matched_config as mc  # noqa: E402

NAMES = {"D1_rf_isotonic": "D1 (frozen RF + isotonic)", "D2_novelty_only": "D2 (novelty only)", "D3_isolation_forest": "D3 (Isolation Forest)", "D4_logistic_regression": "D4 (logistic regression)"}
OUT = os.path.join(mc.OUT_ROOT, "tables")


def load(folder, name):
    with open(os.path.join(mc.result_dir(folder), name)) as f:
        return json.load(f)


def est(s, key="ci95_two_sided"):
    return f"{s['estimate']:.3f} [{s[key][0]:.3f}, {s[key][1]:.3f}]"


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def detector_rows(res):
    return [[NAMES[k], est(d["PMP_primary"]), est(d["AUC_pooled"]), est(d["Delta_match"]), f"{d['PMP_all_pairs']['estimate']:.3f}", f"{d['PMP_secondary_7feature_novelty']['estimate']:.3f}"]
            for k, d in res["detectors"].items()]


def main():
    p, m3 = load("final", "primary_m1m2_results.json"), load("final", "stress_m3_results.json")
    cal = load("stats_review", "caliper_sensitivity.json")
    cols = ["Detector", "PMP, matched pairs [95% CI]", "AUC, pooled [95% CI]", "Delta_match = AUC - PMP [95% CI]", "PMP, all pairs", "PMP, 7-feature novelty"]
    write("table4_primary_detectors.csv", cols, detector_rows(p))
    write("tableS17_m3_detectors.csv", cols, detector_rows(m3))

    rows5 = [[f"{float(k):g} x c = {v['caliper']:.3f}", f"{v['retained_pairs']} ({100 * v['retention']:.1f}%)", *[f"{v['PMP'][d]['estimate']:.3f}" for d in ("D1_rf_isotonic", "D2_novelty_only", "D3_isolation_forest")], f"{v['D1_minus_D2']:+.3f}"]
             for k, v in cal["m12"]["calipers"].items()]
    write("table5_caliper_m12.csv", ["Caliper", "Retained pairs", "PMP D1", "PMP D2", "PMP D3", "D1 - D2"], rows5)
    rowsS4 = [[lab, f"{float(k):g} x c = {v['caliper']:.3f}", f"{v['retained_pairs']} ({100 * v['retention']:.1f}%)", v["min_pairs_per_family"], *[f"{v['PMP'][d]['estimate']:.3f}" for d in ("D1_rf_isotonic", "D2_novelty_only", "D3_isolation_forest")], f"{v['D1_minus_D2']:+.3f}"]
              for st, lab in (("m12", "M1/M2"), ("m3", "M3 (descriptive)")) for k, v in cal[st]["calipers"].items()]
    write("tableS4_caliper_full.csv", ["Set", "Caliper", "Retained pairs", "Fewest pairs in a family", "PMP D1", "PMP D2", "PMP D3", "D1 - D2"], rowsS4)

    write("retention.csv", ["Set", "Pairs", "Retained (primary novelty)", "Retention, primary", "Retention, 7-feature novelty"],
          [[tag, r["n_pairs"], r["n_pairs"] - r["retention"]["unmatched_pairs"], f"{r['retention']['primary']:.4f}", f"{r['retention']['secondary_7feature']:.4f}"] for tag, r in (("M1/M2", p), ("M3", m3))])
    d1 = p["detectors"]["D1_rf_isotonic"]
    write("by_malicious_type.csv", ["Malicious type", "Pairs", "Retained", "PMP of D1 [95% CI]"],
          [[t, p["retention"]["by_malicious_type"][t]["pairs"], p["retention"]["by_malicious_type"][t]["retained"], est(d1["by_malicious_type"][t])] for t in ("M1", "M2")])
    write("by_structural_signature.csv", ["Structural change of the malicious twin", "Retained pairs", "PMP of D1 [95% CI]"],
          [[k, v["retained_pairs"], est(v)] for k, v in d1["by_structural_signature"].items()])
    dec = p["decision"]
    write("decision.csv", ["Criterion", "Value", "Holds"], [
        ["P1: lower 97.5% bound of D1 PMP > 0.5", f"{p['P1_D1_PMP_gt_0.5']['lower_97.5']:.4f}", p["P1_D1_PMP_gt_0.5"]["holds"]],
        ["P2: lower 97.5% bound of PMP(D1) - PMP(D2) > 0", f"{p['P2_D1_minus_D2_gt_0']['lower_one_sided_97.5']:.4f}", p["P2_D1_minus_D2_gt_0"]["holds"]],
        ["Retention at least the minimum", f"{dec['retention']:.4f}", dec["retention_ok"]],
        ["Ceiling check", f"{dec['ceiling_lower_97.5']:.4f}", dec["ceiling_ok"]],
        ["Decision", dec["label"], ""]])


if __name__ == "__main__":
    main()
