import math
import random
import unittest

from postdrift_auth.features import (Event, FEATURE_CONFIG, FeatureUndefined, FEATURE_NAMES, config_hash, extract_features,
                                     extract_features_with_flags)

CID, TA = "d", 200.0
B_LO, W_LO = TA - 150, TA - 30  # 50, 170


def pub(t, topic="tel", size=80, dup=False, cid=CID):
    return Event(t, "PUBLISH", cid, topic, size, dup)


def base_events():
    ev = [Event(0.0, "CONNECT", CID, None, None, None), Event(0.5, "SUBSCRIBE", CID, "cmd", None, None)]
    ev += [pub(60.0 + 10 * i, "tel" if i % 2 else "stat") for i in range(11)]  # 60..160: 11 pubs, mean IAT 10, payload 80
    return ev


def window_events():
    return [pub(170.0, "tel", 50), pub(175.0, "tel", 50, dup=True), pub(190.0, "new", 150), pub(200.0, "new", 150)]


class HandCalculatedTests(unittest.TestCase):
    def test_all_eight_features(self):
        ev = base_events() + window_events()
        ev += [Event(180.0, "SUBSCRIBE", CID, "cmd", None, None), Event(181.0, "SUBSCRIBE", CID, "x", None, None),
               Event(182.0, "SUBSCRIBE", CID, "x", None, None), Event(183.0, "SUBSCRIBE", CID, "y", None, None)]
        ev += [Event(139.999, "CONNECT", CID, None, None, None), Event(150.0, "CONNECT", CID, None, None, None)]
        f = extract_features(ev, CID, TA)
        self.assertEqual(tuple(f), FEATURE_NAMES)
        # window intervals 5, 15, 10: mean 10, sample sd 5
        self.assertAlmostEqual(f["iat_cv"], 0.5)
        self.assertAlmostEqual(f["max_gap_ratio"], 15 / 10)
        self.assertAlmostEqual(f["log_rate_ratio"], math.log((4 / 30 + 0.01) / (11 / 120 + 0.01)))
        self.assertAlmostEqual(f["log_payload_ratio"], math.log(100 / 80))
        self.assertAlmostEqual(f["topic_novelty"], 0.5)
        self.assertEqual(f["sub_novel_count"], 2.0)       # distinct novel filters x, y ("cmd" is in history)
        self.assertEqual(f["reconnects_60s"], 1.0)        # 150 is inside [140,200]; 139.999 is not

    def test_edge_cases_defaults(self):
        f = extract_features(base_events() + [pub(180.0)], CID, TA)
        self.assertEqual(f["iat_cv"], 0.0)                 # < 3 window publishes
        self.assertEqual(f["max_gap_ratio"], 0.0)          # < 2 window publishes
        f0 = extract_features(base_events(), CID, TA)      # no window publishes at all
        self.assertEqual((f0["iat_cv"], f0["log_payload_ratio"], f0["topic_novelty"]), (0.0, 0.0, 0.0))
        self.assertAlmostEqual(f0["log_rate_ratio"], math.log(0.01 / (11 / 120 + 0.01)))

    def test_undefined_baseline_is_an_error_not_a_default(self):
        with self.assertRaises(FeatureUndefined):
            extract_features([pub(180.0), pub(185.0)], CID, TA)


class SevenFeatureLockTests(unittest.TestCase):
    def test_exactly_seven_primary_features_without_dup(self):
        self.assertEqual(FEATURE_NAMES, ("iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio", "topic_novelty",
                                         "sub_novel_count", "reconnects_60s"))
        f = extract_features(base_events() + window_events(), CID, TA)
        self.assertEqual(len(f), 7)
        self.assertNotIn("dup_frac", f)

    def test_dup_flag_cannot_influence_any_feature(self):
        a = extract_features(base_events() + window_events(), CID, TA)
        nodup = [e._replace(dup=False) if e.kind == "PUBLISH" else e for e in base_events() + window_events()]
        self.assertEqual(a, extract_features(nodup, CID, TA))

    def test_edge_case_flags_are_reported(self):
        _, fl = extract_features_with_flags(base_events() + [pub(180.0)], CID, TA)
        self.assertIn("max_gap_ratio_0_fewer_than_2_window_publishes", fl)
        _, fl0 = extract_features_with_flags(base_events(), CID, TA)
        self.assertIn("no_window_publishes", fl0)
        _, fl2 = extract_features_with_flags(base_events() + [pub(180.0), pub(190.0)], CID, TA)
        self.assertEqual(fl2, ["iat_cv_0_fewer_than_3_window_publishes"])
        _, none = extract_features_with_flags(base_events() + window_events(), CID, TA)
        self.assertEqual(none, [])

    def test_distinct_filters_and_ddof1_are_frozen(self):
        self.assertEqual(FEATURE_CONFIG["cv_ddof"], 1)
        self.assertIn("distinct", FEATURE_CONFIG["sub_novel"])


class WindowBoundaryTests(unittest.TestCase):
    """Convention: W=[t_a-30,t_a] closed; B=[t_a-150,t_a-30) right-open; H=[0,t_a-30) right-open; R=[t_a-60,t_a] closed."""

    def rate_n(self, events):  # number of window publishes implied by log_rate_ratio
        f = extract_features(events, CID, TA)
        b_rate = 11 / 120 + 0.01
        return round((math.exp(f["log_rate_ratio"]) * b_rate - 0.01) * 30)

    def test_window_left_endpoint_inclusive(self):
        self.assertEqual(self.rate_n(base_events() + [pub(W_LO)]), 1)
        self.assertEqual(self.rate_n(base_events() + [pub(W_LO - 1e-6)]), 0)

    def test_window_right_endpoint_inclusive_and_future_excluded(self):
        self.assertEqual(self.rate_n(base_events() + [pub(TA)]), 1)
        self.assertEqual(self.rate_n(base_events() + [pub(TA + 1e-6)]), 0)

    def test_baseline_left_inclusive_right_open(self):
        # extra baseline-region publish at exactly t_a-30 must NOT enter B (it belongs to W)
        ev = base_events()
        with_edge = extract_features(ev + [pub(W_LO, "tel", 80)], CID, TA)
        self.assertAlmostEqual(with_edge["max_gap_ratio"], 0.0)  # single window pub; B unchanged (mean IAT still 10)
        # event at exactly t_a-150 is inside B; just before it is outside
        inside = [pub(B_LO, "tel", 80)] + [pub(60.0 + 10 * i) for i in range(11)]
        outside = [pub(B_LO - 1e-6, "tel", 80)] + [pub(60.0 + 10 * i) for i in range(11)]
        a = extract_features(inside + window_events(), CID, TA)
        b = extract_features(outside + window_events(), CID, TA)
        self.assertAlmostEqual(a["log_rate_ratio"], math.log((4 / 30 + 0.01) / (12 / 120 + 0.01)))  # 12 baseline pubs
        self.assertAlmostEqual(b["log_rate_ratio"], math.log((4 / 30 + 0.01) / (11 / 120 + 0.01)))  # 11 baseline pubs

    def test_history_sets_right_open(self):
        # a topic first seen exactly at t_a-30 is NOT history: that publish is novel
        ev = base_events() + [pub(W_LO, "fresh")]
        self.assertEqual(extract_features(ev, CID, TA)["topic_novelty"], 1.0)
        ev = base_events() + [pub(W_LO - 1e-6, "fresh"), pub(W_LO + 1, "fresh")]
        self.assertEqual(extract_features(ev, CID, TA)["topic_novelty"], 0.0)
        sub_hist = base_events() + [Event(W_LO - 1e-6, "SUBSCRIBE", CID, "s", None, None), Event(W_LO + 1, "SUBSCRIBE", CID, "s", None, None)]
        self.assertEqual(extract_features(sub_hist, CID, TA)["sub_novel_count"], 0.0)
        sub_edge = base_events() + [Event(W_LO, "SUBSCRIBE", CID, "s", None, None)]
        self.assertEqual(extract_features(sub_edge, CID, TA)["sub_novel_count"], 1.0)

    def test_reconnect_window(self):
        for t, n in ((TA - 60, 1), (TA - 60 - 1e-6, 0), (TA, 1), (TA + 1e-6, 0)):
            ev = base_events() + [Event(t, "CONNECT", CID, None, None, None)]
            self.assertEqual(extract_features(ev, CID, TA)["reconnects_60s"], float(n), msg=t)

    def test_events_of_other_identities_ignored(self):
        ev = base_events() + window_events()
        other = [pub(185.0, "zzz", 999, cid="other"), Event(190.0, "CONNECT", "other", None, None, None)]
        self.assertEqual(extract_features(ev, CID, TA), extract_features(ev + other, CID, TA))


class NoFutureLeakageTests(unittest.TestCase):
    def test_future_events_cannot_change_the_vector(self):
        ev = base_events() + window_events()
        ref = extract_features(ev, CID, TA)
        rng = random.Random(0)
        future = []
        for _ in range(60):
            t = TA + rng.uniform(1e-6, 300)
            kind = rng.choice(["PUBLISH", "SUBSCRIBE", "CONNECT"])
            future.append(Event(t, kind, CID, f"f{rng.randint(0, 3)}" if kind != "CONNECT" else None,
                                rng.randint(1, 999) if kind == "PUBLISH" else None, rng.random() < 0.5 if kind == "PUBLISH" else None))
        self.assertEqual(extract_features(ev + future, CID, TA), ref)
        self.assertEqual(extract_features(future + ev, CID, TA), ref)  # order independent too

    def test_future_event_in_baseline_gap_does_not_matter_even_when_malformed(self):
        ev = base_events() + window_events()
        bad = Event(TA + 5, "PUBLISH", CID, None, None, None)  # would crash any code that touched it
        self.assertEqual(extract_features(ev + [bad], CID, TA), extract_features(ev, CID, TA))


class DeterminismTests(unittest.TestCase):
    def test_same_input_same_output(self):
        ev = base_events() + window_events()
        self.assertEqual(extract_features(ev, CID, TA), extract_features(list(reversed(ev)), CID, TA))
        self.assertEqual(config_hash(), config_hash())
        self.assertEqual(len(config_hash()), 64)


if __name__ == "__main__":
    unittest.main()
