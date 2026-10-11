# Data description

## Design of the data

The data come from a software-emulated MQTT testbed: Docker containers with a Mosquitto broker and Python publish/subscribe clients (`testbed/`). A *session* is one
client identity during one run. Each run has 40 sessions. Network traffic is captured at the broker side, and the seven session features are computed
from the capture in a window that ends at the trigger time `t_a` (30 s after the onset `tau` of the change).

A *family* is two runs from one seed (arm A and arm B). Every identity appears in both runs. It is malicious in one run and benign (B1) in the other, and both
twins share the pre-drift session. The two members of a pair are therefore the same identity at the same trigger time.

| Set | Folder | Families | Runs | Sessions | Pairs |
|---|---|---|---|---|---|
| Source regime, TRAIN | `data/source_regime/TRAIN/` | 10 runs | 10 | 400 | |
| Source regime, CAL | `data/source_regime/CAL/` | 6 runs | 6 | 240 | |
| M1/M2 twin families (confirmatory) | `data/matched_drift_v1/m12/` | 37 | 74 | 2,960 | 1,480 |
| M3 twin families (held-out stress test) | `data/matched_drift_v1/m3/` | 20 | 40 | 1,600 | 800 |

The source regime (TRAIN and CAL runs of an earlier development pilot) is used to fit the detectors and the novelty reference. It is not part of the evaluated data.
The three development dry-run families that fixed the caliper are not included. Their only trace in the release is the caliper in `configs/matched_drift_v1_lock.json`.

## A) Released files

### `data/**/<run>/ground_truth.jsonl`

One JSON line per session: `session_id` (client id), `pair_id` (family seed and identity), `arm` (A or B), `family_seed`, `scenario`
(S0, B1, B2, M1, M2 or M3), `y` (1 malicious, 0 benign), `container`, `tau`, `t_a` and `params` (the drawn scenario parameters).
The labels were joined to the features only after the extraction. `scenario` is used for description after scoring, never as a model input.

### `data/**/<run>/extract_out/features.csv`

One row per session: `client_id`, `t_a` and the seven features.

| Feature | Meaning |
|---|---|
| `iat_cv` | coefficient of variation of PUBLISH inter-arrival times in the decision window |
| `max_gap_ratio` | longest PUBLISH gap in the decision window relative to the baseline mean gap |
| `log_rate_ratio` | log ratio of the PUBLISH rate in the decision window to the pre-drift baseline |
| `log_payload_ratio` | log ratio of the mean payload size to the baseline |
| `topic_novelty` | share of PUBLISH events in the decision window sent to topics that had no PUBLISH event in the history window |
| `sub_novel_count` | number of new SUBSCRIBE filters |
| `reconnects_60s` | number of CONNECT events of the identity in the 60 s before `t_a` |

The first four are the *magnitude* features (used for the novelty). The last three are the *structural* features. The exact definitions are in `src/postdrift_auth/features.py`.

### `data/**/<run>/meta.json`

A trimmed copy of the run metadata: run id, split, master seed, composition, the feature configuration hash and the SHA-256 hashes of the PCAP,
ground-truth and plan files of the full run. The hashes allow a later check of the raw captures against this release. The full metadata (container, Docker and
git details) is not included.

### `results/matched_drift_v1/`

| Folder | Content | Produced by |
|---|---|---|
| `final/` | results of the locked analysis (`*_results.json`), score of every detector for every session (`*_session_scores.csv`), data-quality reports of the generation, result tables | `run_analysis.py` |
| `stats_review/` | alternative intervals, family sign tests, ties, caliper sensitivity, seven-feature novelty comparator | `stat_review.py`, `caliper_sensitivity.py`, `novelty7_check.py` |
| `exploratory/` | permutation importance, auxiliary models, feature distributions, counterfactual swaps (exploratory) | `exploratory_mechanism.py` |
| `tables/` | the CSV tables of the paper's results | `reproduce_tables.py` |
| `figures/` | the paper's figures (PDF and PNG) and the plotted values (`figure_data.json`) | `reproduce_figures.py` |

`*_data_quality_report.json` records the 15 checks that the original generation harness ran on the raw files (PCAP hashes, container logs, pair integrity).
They cannot be re-run without the raw files.

### Configuration

`configs/matched_drift_v1.json` is the frozen configuration of the study, and `configs/matched_drift_v1_lock.json` fixes the values chosen after the dry run (the
number of families, the caliper). `configs/pilot_stage3a_v2.json` is the configuration of the source regime and the detectors. The three files are used
as they were during the study (their hashes are checked by the code), so the configuration contains some references to the original development
folders (for example `results/raw/...`) and to a protocol document that is not part of this release. The code takes the data location from `data/`
or from the environment variable `MTE_DATA_DIR`.

Seeds: the analysis seed is 7,000,000. Family seeds are 4,100,000 to 4,100,036 (M1/M2), 5,100,000 to 5,100,019 (M3). The source regime uses 1,100,000 to 1,100,009 (TRAIN)
and 2,100,000 to 2,100,005 (CAL). The bootstrap resampling indices are derived from the analysis seed, so the intervals are reproducible.

## B) Files not included

| Item | Reason |
|---|---|
| Raw PCAP captures of all runs (about 1.1 GB) | size |
| Container logs, per-run plan and summary files, run-level resource samples | size, and they contain local environment details |
| Records of failed and repeated attempts, retry and harness audits | internal audit material |
| Manuscript | not part of the code release |

`[AUTHORS TO DECIDE: whether the raw captures are archived (for example on Zenodo) or shared on request.]` Until then the raw-data quality gate, the PCAP parsing of
all runs and the data generation cannot be repeated from this release. They can be checked on the example run in `examples/pcap_example/`, and the SHA-256 hashes in `meta.json`
identify the captures that were used.

## Scope

All sessions come from the emulator. The attack and benign scenarios are defined by the generators in `src/postdrift_auth/scenarios.py`. The data say how the four evaluated scores behave
on these scenarios. They do not describe real devices or real traffic.
