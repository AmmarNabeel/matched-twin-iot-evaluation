# Reproduction guide

This guide follows the order of the paper's analysis. Run all commands from the repository root with the environment from `requirements.txt`.
New results are written to `outputs/matched_drift_v1/`. The reference results in `results/matched_drift_v1/` are not changed.

Timings below were measured on a laptop with 4 CPU cores (no GPU needed).

## 0. Check the data

```bash
python scripts/run_analysis.py --check-data
```

Checks that all 74 + 40 twin runs are present, with a feature table and a ground-truth file each. Takes a second.

## 1. Locked analysis (feature tables to results)

```bash
python scripts/run_analysis.py --verify
```

What it does, in the order of the paper (Methods 3.13 to 3.15):

1. Fits the frozen detectors on the source-regime data: random forest with isotonic calibration (D1), novelty-only score (D2), Isolation Forest (D3), logistic regression (D4).
2. Computes the novelty N of every session (mean distance to the 10 nearest benign reference sessions, four magnitude features) and takes ln N.
3. Forms the twin pairs and keeps the pairs with a ln N difference of at most the caliper (0.514).
4. Computes the paired malicious-over-benign win probability (PMP), the pooled AUC and the gap between them.
5. Runs the family-clustered percentile bootstrap (10,000 replicates, seed from the lock file) and the decision criteria P1 and P2 for the M1/M2 families.
6. Repeats steps 2 to 5 for the held-out M3 families (descriptive).

Writes `final/primary_m1m2_results.json`, `final/stress_m3_results.json`, the session scores and the result tables. `--verify` then compares every file with the reference
(tolerance 1e-9). About 75 seconds.

## 2. Statistical checks and supplementary analyses

```bash
python scripts/stat_review.py            # alternative intervals, family sign tests, ties, paired contrasts
python scripts/caliper_sensitivity.py    # descriptive caliper sensitivity (tighter calipers)
python scripts/novelty7_check.py         # novelty on all seven features as a second comparator
python scripts/exploratory_mechanism.py  # exploratory: feature-group permutation, auxiliary models, M3 swaps
```

`stat_review.py`, `caliper_sensitivity.py` and `novelty7_check.py` run in a few seconds, `exploratory_mechanism.py` in about 30 seconds. The first three read the
session scores from step 1 if you ran it, and the reference scores otherwise.

## 3. Tables and figures

```bash
python scripts/reproduce_tables.py
python scripts/reproduce_figures.py
python scripts/verify_results.py final stats_review exploratory tables figures
```

| Output | Paper |
|---|---|
| `tables/table4_primary_detectors.csv` | Table 4 |
| `tables/table5_caliper_m12.csv` | Table 5 |
| `tables/tableS4_caliper_full.csv` | Supplementary Table S4 |
| `tables/tableS17_m3_detectors.csv` | Supplementary Table S17 |
| `tables/retention.csv`, `by_malicious_type.csv`, `by_structural_signature.csv`, `decision.csv` | numbers in Sections 4.3 to 4.5 |
| `figures/fig1_framework` to `fig5_m3_stress` | Figures 1 to 5 |
| `figures/figS1_family_pmp`, `figS2_feature_distributions` | Supplementary Figures S1 and S2 |

Figures are written as PDF, SVG and PNG. They are compared through `figures/figure_data.json`, which holds every plotted value, because the image files differ in metadata.
The other supplementary tables of the paper (failed attempts, harness history, leakage checklist) come from audit files that are not released.

## 4. Independent re-implementation

```bash
python scripts/independent_check.py
```

Recomputes the primary quantities (hashing, forest and isotonic fit, novelty, pairing, PMP, AUC, bootstrap) without importing the project package, and compares 51 quantities
with the stored results (largest allowed difference 1e-9). About 10 seconds.

## 5. Feature extraction from a PCAP

```bash
python scripts/extract_features.py
```

Needs `tshark` (4.6.4 was used). Parses `examples/pcap_example/trace.pcap`, computes the features of the 40 sessions and compares them with the stored feature table.
For your own capture: `python scripts/extract_features.py --pcap P --trigger T --t0 EPOCH --out-dir D`. The trigger file has one JSON line `{"client_id": ..., "t_a": ...}` per session,
and `EPOCH` is the capture start time that the run harness recorded.

## 6. Data generation (not part of the verification)

`scripts/generate_data/run_matched_data.py` generates the twin families in Docker containers (Mosquitto, Python clients, `tc netem` for the background link) from the configuration
and the seeds, and `testbed/` holds the client and broker configuration. It needs Docker with permission to use `tc`, a clean git working tree (the runner records the commit hash)
and several hours for the full data set. This path was not repeated for the release. Because the clock and the container scheduler are involved, a
regeneration reproduces the planned scenario parameters exactly (they come from the seeds) but not the realised timing.

## 7. Unit tests

```bash
python -m unittest discover -s tests -t .
```

About 115 tests: feature definitions, trace parsing on synthetic captures, scenario generators, seed handling, twin checks, and the equations of the analysis
recomputed from the saved session scores. About 40 seconds.

## What reproduces, and what does not

| Result | From | Status |
|---|---|---|
| All numbers of the locked analysis (Tables 4 and 5, Section 4) | feature tables | reproduced, differences below 1e-9 |
| Statistical checks, caliper sensitivity, seven-feature novelty | feature tables | reproduced |
| Exploratory mechanism analysis, Figures 4 and 5b, 5c | feature tables | reproduced |
| Figures 1 to 5, S1, S2 | results and feature tables | reproduced (compared through the plotted values) |
| PCAP parsing | example run only | reproduced for one run; the other captures are not released |
| Raw-data quality gate, failed-attempt and harness audits | raw files | not reproducible from this release |
| Data generation | Docker testbed | not repeated for the release |

The random forest and the isotonic fit depend on the library versions. The pinned versions in `requirements.txt` reproduce the reference numbers. With other versions, small differences are possible.
