# Matched-twin evaluation of post-authentication trust scores

Code, configuration and derived results for a measurement study of how a post-authentication trust score ranks malicious and benign
behavioural change in a software-emulated MQTT testbed.

## What this repository is

IoT services that re-check a device during a session often turn its behaviour into a trust score. A score of this kind can rise
simply because the device has moved away from its usual behaviour, for benign reasons as well as malicious ones. We ask whether a trust score
ranks a malicious change above a benign change when both are equally novel.

The study is an evaluation, not a new detector. Each identity has a *twin*: both twins share the same pre-drift session, and after the onset
one follows a benign reconfiguration (B1) and the other a malicious scenario (M1 or M2; M3 is held out as a stress test). Pairs whose novelty differs by
at most a fixed caliper are kept. The main measure is the paired win probability (PMP): the share of kept pairs in which the malicious twin gets the higher
risk score. It is reported next to the pooled AUC, with family-clustered bootstrap intervals. Four standard scores are evaluated: a frozen random forest with
isotonic calibration (D1), a novelty-only score (D2), an Isolation Forest (D3) and a logistic regression (D4). No new detector, algorithm or trust model is proposed.

Everything runs on an emulator (Docker containers with Mosquitto and Python clients). The attacks and benign changes are defined by our generators.
The results say nothing about real devices or real traffic.

## Relationship to the paper

- Title: *Matched-Twin Evaluation of Post-Authentication Trust Scores Under Behavioural Drift: An Emulated MQTT Testbed Study for IoT Service Monitoring*
- Journal: IEEE Transactions on Network and Service Management (submitted; no decision yet).
- Citation: `Citation information will be updated after publication.`. A `CITATION.cff` file is included with placeholders.

The numbers in the paper's results tables come from the files in `results/`. Section "Reproducing the results" shows which command regenerates which table or figure.

## Repository contents

```
configs/       frozen configuration of the study and its lock file (caliper, number of families, seeds)
src/           the Python package postdrift_auth (see the module list below)
scripts/       entry points: analysis, tables, figures, checks, feature extraction, data generation
data/          derived per-run feature tables and ground-truth files (no raw captures)
results/       reference results of the study: final analysis, statistical checks, exploratory analysis, tables, figures
examples/      one complete run (PCAP and trigger file) for testing the feature extraction
docs/          reproduction guide and data description
testbed/       the MQTT client and broker configuration used to generate the data
tests/         unit tests of the package
```

Modules of `src/postdrift_auth/`:

| Module | Content |
|---|---|
| `scenarios.py`, `seeds.py`, `mechanics.py` | scenario generators (S0, B1, B2, M1, M2, M3), seed derivation, planned behaviour of each session |
| `trace.py`, `features.py`, `extract_cli.py` | PCAP parsing and the seven session features (feature extraction) |
| `pilot_model.py`, `pilot_config.py`, `pilot_quality.py` | source-regime data, the detectors D1 to D4, their configuration |
| `matched.py` | novelty, caliper matching, pair win indicator (PMP), AUC, family bootstrap |
| `matched_analysis.py`, `matched_config.py`, `matched_report.py` | the locked analysis, configuration loading, result tables |
| `matched_quality.py`, `twin_checks.py`, `provenance.py` | data-quality gate and twin checks (need raw files), run provenance |

## Installation

Python 3.12 is the tested version. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`environment.yml` is an equivalent conda environment. The package versions in `requirements.txt` are the ones used for the paper. Other versions are likely
to work, but the random forest and the isotonic fit are only guaranteed to reproduce the reference numbers within a tolerance of 1e-9 with the pinned versions.

Feature extraction from PCAP files needs `tshark` (version 4.6.4 was used). Data generation needs Docker. Neither is needed for the analysis.

## Reproducing the results

All commands are run from the repository root. Output goes to `outputs/matched_drift_v1/`. The reference results are in `results/matched_drift_v1/` and are never overwritten.

| Step | Command | Writes | Needs |
|---|---|---|---|
| Locked analysis: novelty, matching, PMP, pooled AUC, bootstrap intervals | `python scripts/run_analysis.py --verify` | `outputs/.../final/` | released feature tables |
| Result tables | `python scripts/reproduce_tables.py` | `outputs/.../tables/` | `results/` or the new outputs |
| Figures | `python scripts/reproduce_figures.py` | `outputs/.../figures/` | `results/` or the new outputs, feature tables |
| Statistical checks (alternative intervals, sign tests, ties) | `python scripts/stat_review.py` | `outputs/.../stats_review/` | outputs of the analysis |
| Caliper sensitivity | `python scripts/caliper_sensitivity.py` | `outputs/.../stats_review/` | outputs of the analysis |
| Seven-feature novelty comparator | `python scripts/novelty7_check.py` | `outputs/.../stats_review/` | outputs of the analysis |
| Exploratory mechanism analysis | `python scripts/exploratory_mechanism.py` | `outputs/.../exploratory/` | feature tables |
| Compare with the reference | `python scripts/verify_results.py` | prints a report | the outputs above |
| Independent re-implementation | `python scripts/independent_check.py` | `outputs/.../independent/` | feature tables |
| Feature extraction from a PCAP | `python scripts/extract_features.py` | `outputs/pcap_example/` | `tshark` |
| Unit tests | `python -m unittest discover -s tests -t .` | | |

The recommended order for a full re-run is: `run_analysis.py`, `stat_review.py`, `caliper_sensitivity.py`, `novelty7_check.py`, `exploratory_mechanism.py`,
`reproduce_tables.py`, `reproduce_figures.py`, `verify_results.py`. `docs/reproduction_guide.md` explains each step and how long it takes.

Scripts read the new outputs if they exist and the reference results otherwise, so `reproduce_tables.py` and `reproduce_figures.py` also run directly after cloning.

### What is reproduced, and from what

The analysis starts from the per-run feature tables (`data/`). These are the output of the feature extraction. Running the analysis from them reproduces the
paper's results. It does not repeat the PCAP parsing or the data generation, because the raw captures are not part of this repository (see below).
`extract_features.py` shows the parsing step on one example run. The independent re-implementation (`independent_check.py`) does not import the project package.

## Data availability

| Item | In this repository |
|---|---|
| Source code, configuration, lock file, seeds | yes |
| Per-run feature tables and ground-truth files (source regime, 37 M1/M2 families, 20 M3 families) | yes, in `data/` |
| Derived results: scores, results files, tables, figure values | yes, in `results/` |
| One example run with PCAP and trigger file | yes, in `examples/pcap_example/` |
| Raw PCAP captures of all runs, container logs, per-run plan and summary files (about 1.1 GB) | **no** |
| Records of failed attempts and the audit files of the study | **no** |
| Manuscript | **no** |

The raw captures are not released here. `Raw packet captures are not publicly released due to storage constraints.
Derived feature tables and results required to reproduce the analysis are
provided.`.
`docs/data_description.md` describes every file format.

## Limitations

The data come from a software emulator. The attacks and the benign changes are defined by the study's own generators, and the detectors are trained on data
from the same generators. No hardware or real network traffic is involved. The results should not be read as a statement about deployed systems.

## License

MIT, see `LICENSE`.