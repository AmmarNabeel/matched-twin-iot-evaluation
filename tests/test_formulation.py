"""Equation-to-code tests: every numbered equation of Section 3.14 is checked against the implementation used for the locked analysis."""
import csv
import json
import math
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from postdrift_auth import features, matched as M, matched_config as mc  # noqa: E402


def toy_pairs(n_fam=3, per=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_fam * per
    fam = np.repeat(np.arange(n_fam), per)
    pairs = {"ben": np.arange(n), "mal": np.arange(n, 2 * n), "family": fam, "mal_scenario": np.array(["M1"] * n)}
    data = {"family": np.concatenate([fam, fam]), "y": np.concatenate([np.zeros(n, int), np.ones(n, int)])}
    return data, pairs, rng


class EquationTests(unittest.TestCase):
    def test_eq3_paired_win_counts_ties_as_one_half(self):
        _, pairs, _ = toy_pairs(1, 3)
        s = np.array([0.2, 0.5, 0.7, 0.3, 0.5, 0.1])  # ben = first three, mal = last three
        np.testing.assert_allclose(M.pair_wins(s, pairs), [1.0, 0.5, 0.0])

    def test_eq2_caliper_is_inclusive_and_uses_absolute_difference(self):
        _, pairs, _ = toy_pairs(1, 3)
        ln = np.array([0.0, 0.0, 0.0, 0.5, -0.5, 0.6])
        np.testing.assert_array_equal(M.caliper_keep(ln, pairs, 0.5), [True, True, False])

    def test_eq4_pmp_is_pair_weighted_ratio_of_family_sums(self):
        data, pairs, rng = toy_pairs(3, 5)
        w = rng.choice([0.0, 0.5, 1.0], size=15)
        keep = rng.random(15) < 0.6
        fs = M.FamilyStats(data, pairs)
        s, n = fs.pmp_by_family(w, keep)
        self.assertAlmostEqual(s.sum() / n.sum(), w[keep].mean())          # pooled over retained pairs, every pair weighted equally
        self.assertNotAlmostEqual(float(np.mean(s / n)), float(w[keep].mean()), places=6)  # not the family-weighted mean

    def test_eq6_bootstrap_resamples_family_totals_not_pairs(self):
        data, pairs, rng = toy_pairs(3, 5)
        w = rng.choice([0.0, 1.0], size=15)
        keep = np.ones(15, bool)
        fs = M.FamilyStats(data, pairs)
        s, n = fs.pmp_by_family(w, keep)
        boot = np.array([[0, 0, 0], [1, 2, 2]])
        got = fs.pmp_boot(w, keep, boot)
        np.testing.assert_allclose(got, [s[0] / n[0], (s[1] + 2 * s[2]) / (n[1] + 2 * n[2])])

    def test_eq1_novelty_uses_sample_sd_k_nearest_mean_distance_and_natural_log(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(size=(30, 7))
        nov = M.Novelty(ref, [0, 1], 3, {})
        x = rng.normal(size=(4, 7))
        mu, sd = ref[:, [0, 1]].mean(0), ref[:, [0, 1]].std(0, ddof=1)
        z, zr = (x[:, [0, 1]] - mu) / sd, (ref[:, [0, 1]] - mu) / sd
        want = [math.log(np.sort(np.linalg.norm(zr - zi, axis=1))[:3].mean()) for zi in z]
        np.testing.assert_allclose(nov.ln_n(x), want)

    def test_eq1_count_feature_floor_applies_to_standard_deviation(self):
        ref = np.zeros((20, 7))
        ref[:, 5] = np.r_[np.zeros(19), 0.01]  # tiny spread of a count feature
        nov = M.Novelty(ref, [5], 2, {"sub_novel_count": 1.0})
        self.assertEqual(float(nov.sd[0]), 1.0)

    def test_eq7_auc_uses_all_sessions_ties_half_and_differs_from_pmp(self):
        data, pairs, _ = toy_pairs(2, 2)
        s = np.array([0.1, 0.9, 0.2, 0.3, 0.4, 0.5, 0.2, 0.8])  # ben = 0..3, mal = 4..7
        y = data["y"]
        brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
        self.assertAlmostEqual(M.auc_mw(s, y), brute)                   # cross-pair and cross-family comparisons are included
        self.assertNotAlmostEqual(M.auc_mw(s, y), float(M.pair_wins(s, pairs).mean()))

    def test_eq8_structural_features_are_the_three_structural_ones(self):
        names = [features.FEATURE_NAMES[j] for j in M.STRUCT_IDX]
        self.assertEqual(names, ["topic_novelty", "sub_novel_count", "reconnects_60s"])
        self.assertEqual([features.FEATURE_NAMES[j] for j in M.MAG_IDX], ["iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio"])

    def test_feature_definitions_on_a_small_trace(self):
        E = features.Event
        ta = 200.0
        ev = [E(t, "CONNECT", "c", None, None, None) for t in (0.0, 150.0)]
        ev += [E(float(t), "PUBLISH", "c", "a/x", 10, False) for t in range(80, 170, 10)]       # baseline (I_base = [50, 170)): 9 publishes
        ev += [E(t, "PUBLISH", "c", "a/x", 20, False) for t in (175.0, 185.0, 195.0)]            # decision window (I_dec = [170, 200]): 3 publishes
        ev += [E(190.0, "PUBLISH", "c", "a/new", 20, False), E(180.0, "SUBSCRIBE", "c", "f/+", None, None)]
        f = features.extract_features(ev, "c", ta)
        self.assertAlmostEqual(f["topic_novelty"], 1 / 4)   # 4 PUBLISHes in I_dec, one to a topic absent from I_hist
        self.assertEqual(f["sub_novel_count"], 1.0)
        self.assertEqual(f["reconnects_60s"], 1.0)  # only the CONNECT at 150 s lies in I_rec = [140, 200]
        self.assertAlmostEqual(f["log_rate_ratio"], math.log((4 / 30 + 0.01) / (9 / 120 + 0.01)))
        self.assertAlmostEqual(f["log_payload_ratio"], math.log(20 / 10))

    def test_saved_results_reproduce_from_saved_scores(self):
        """PMP, retention, D1-D2 contrast and pooled AUC of the locked JSON files are recomputed from the saved session scores (no raw data needed)."""
        cfg = mc.load()
        lock = mc.load_lock(cfg)
        for tag in ("primary_m1m2", "stress_m3"):
            with open(os.path.join(mc.result_dir("final"), f"{tag}_results.json")) as fh:
                res = json.load(fh)
            with open(os.path.join(mc.result_dir("final"), f"{tag}_session_scores.csv"), newline="") as fh:
                rows = list(csv.DictReader(fh))
            by = {}
            for r in rows:
                by.setdefault(r["pair_id"], {})[int(r["y"])] = r
            ln_m = np.array([float(v[1]["ln_N_primary"]) for v in by.values()])
            ln_b = np.array([float(v[0]["ln_N_primary"]) for v in by.values()])
            keep = np.abs(ln_m - ln_b) <= lock["caliper_primary"]
            self.assertAlmostEqual(float(keep.mean()), res["retention"]["primary"], places=12)
            wins = {}
            for d in ("D1_rf_isotonic", "D2_novelty_only", "D3_isolation_forest", "D4_logistic_regression"):
                sm = np.array([float(v[1][d]) for v in by.values()])
                sb = np.array([float(v[0][d]) for v in by.values()])
                w = (sm > sb) + 0.5 * (sm == sb)
                wins[d] = w
                self.assertAlmostEqual(float(w[keep].mean()), res["detectors"][d]["PMP_primary"]["estimate"], places=12)
                y = np.array([int(r["y"]) for r in rows])
                self.assertAlmostEqual(M.auc_mw(np.array([float(r[d]) for r in rows]), y), res["detectors"][d]["AUC_pooled"]["estimate"], places=12)
            self.assertAlmostEqual(float(wins["D1_rf_isotonic"][keep].mean() - wins["D2_novelty_only"][keep].mean()), res["P2_D1_minus_D2_gt_0"]["estimate"], places=12)


if __name__ == "__main__":
    unittest.main()
