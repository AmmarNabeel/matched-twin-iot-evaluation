import math
import unittest

from postdrift_auth import scenarios as sc
from postdrift_auth.seeds import split_of_seed

SMOKE = [9_000_000 + i for i in range(200)]  # SMOKE range only; used for generator-parameter checks, never traces


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_plan_for_every_scenario(self):
        for i, s in enumerate(sc.SCENARIOS):
            a, b = sc.build_run(9_000_000 + i, s), sc.build_run(9_000_000 + i, s)
            self.assertEqual(a, b)
            self.assertEqual(sc.plan_hash(a), sc.plan_hash(b))

    def test_different_seed_different_plan(self):
        self.assertNotEqual(sc.plan_hash(sc.build_run(9_000_000, "B1")), sc.plan_hash(sc.build_run(9_000_001, "B1")))

    def test_derived_seeds_logged_and_stable(self):
        a = sc.build_run(9_000_003, "M1")["derived_seeds"]
        self.assertEqual(a, sc.build_run(9_000_003, "M1")["derived_seeds"])
        self.assertIn("dev-target/scn", a)


class FrozenRangeTests(unittest.TestCase):
    def test_timeline_and_base_profile(self):
        for seed in SMOKE:
            s = sc.build_session(seed, "dev-target", "S0")
            self.assertTrue(150 <= s["tau"] < 160)
            self.assertAlmostEqual(s["t_a"] - s["tau"], 30.0)
            self.assertTrue(0.5 <= s["base"]["period"] < 2.0 and 40 <= s["base"]["size"] < 200)
            self.assertEqual(s["params"], {})
            self.assertEqual(s["regimes"]["pre"], s["regimes"]["post"])

    def test_b1_ranges(self):
        for seed in SMOKE:
            p = sc.build_session(seed, "dev-target", "B1")["params"]
            self.assertTrue(0.5 <= p["period_mult"] <= 2.0 and abs(math.log(p["period_mult"])) >= 0.1)
            self.assertTrue(0.7 <= p["payload_mult"] <= 1.5)

    def test_b2_ranges_and_app_unchanged(self):
        for seed in SMOKE:
            s = sc.build_session(seed, "dev-target", "B2")
            n = s["netem"]
            self.assertTrue(100 <= n["delay_ms"] < 400 and 20 <= n["jitter_ms"] < 100 and 1 <= n["loss_pct"] < 8)
            self.assertEqual(n["apply_at"], s["tau"])
            self.assertEqual(s["regimes"]["pre"], s["regimes"]["post"])
            self.assertIsNone(s["attacker"])

    def test_m1_structure(self):
        counts = set()
        for seed in SMOKE:
            s = sc.build_session(seed, "dev-target", "M1")
            a, p = s["attacker"], s["params"]
            self.assertEqual(a["client_id"], s["legit"]["client_id"])
            self.assertEqual(a["connect_at"], s["tau"])
            self.assertTrue(s["legit"]["no_reconnect"] and all(t < s["tau"] for t, _, _ in s["legit"]["pubs"]))
            self.assertTrue(all(t >= s["tau"] for t, _, _ in a["pubs"]))
            self.assertTrue(0.5 <= p["rate_mult"] <= 2.0 and 0.5 <= p["payload_mult"] <= 2.0 and p["n_novel_subs"] in (0, 1, 2, 3))
            self.assertEqual(len(a["subs"]) - 1, p["n_novel_subs"])
            counts.add(p["n_novel_subs"])
        self.assertEqual(counts, {0, 1, 2, 3})

    def test_m2_structure(self):
        seen = set()
        for seed in SMOKE:
            s = sc.build_session(seed, "dev-target", "M2")
            p = s["params"]
            self.assertTrue(0.25 <= p["rate_mult"] <= 4.0 and 0.5 <= p["payload_mult"] <= 3.0 and 0.2 <= p["new_topic_share"] < 0.5)
            self.assertIsNone(s["attacker"])
            self.assertEqual(len(s["legit"]["subs"]), 1)  # same connection, no recon
            aux = sc.topics("dev-target")["aux"]
            seen.add((p["new_topic"], any(t == aux for _, t, _ in s["legit"]["pubs"])))
        self.assertEqual(seen, {(True, True), (False, False)})  # new topic appears iff drawn; both branches exist

    def test_m3_is_low_rate_mimicry(self):
        for seed in SMOKE:
            s = sc.build_session(seed, "dev-target", "M3")
            p = s["params"]
            self.assertTrue(0.8 <= p["rate_mult"] <= 1.25 and 0.9 <= p["payload_mult"] <= 1.2 and 1.3 <= p["jitter_mult"] <= 2.0)
            self.assertEqual((p["new_topic_share"], p["n_novel_subs"], p["reconnect"]), (0.05, 0, False))
            self.assertIsNone(s["attacker"])
            self.assertEqual(len(s["legit"]["subs"]), 1)
            self.assertFalse(s["legit"]["no_reconnect"])
            self.assertAlmostEqual(s["regimes"]["post"]["jitter"], 0.05 * p["jitter_mult"])

    def test_labels(self):
        self.assertEqual({k: sc.MALICIOUS[k] for k in sc.SCENARIOS}, {"S0": 0, "B1": 0, "B2": 0, "M1": 1, "M2": 1, "M3": 1})

    def test_unknown_scenario(self):
        with self.assertRaises(ValueError):
            sc.build_run(9_000_000, "M9")


class ScheduleTests(unittest.TestCase):
    def test_pre_onset_schedule_identical_across_scenarios(self):
        """Before tau every scenario shares the identity's base behaviour (same seed -> same pre-onset publishes)."""
        seed = 9_000_007
        ref = sc.build_session(seed, "dev-target", "S0")
        for scn in ("B1", "B2", "M2", "M3", "M1"):
            s = sc.build_session(seed, "dev-target", scn)
            n = [p for p in s["legit"]["pubs"] if p[0] < s["tau"]]
            self.assertEqual(n, [p for p in ref["legit"]["pubs"] if p[0] < ref["tau"]])

    def test_interval_law(self):
        for scn in ("S0", "B1", "M2", "M3"):
            s = sc.build_session(9_000_002, "dev-target", scn)
            for (t1, _, _), (t2, _, _) in zip(s["legit"]["pubs"], s["legit"]["pubs"][1:]):
                r = s["regimes"]["pre"] if t1 < s["tau"] else s["regimes"]["post"]
                self.assertTrue(r["period"] * (1 - r["jitter"]) - 1e-6 <= t2 - t1 <= r["period"] * (1 + r["jitter"]) + 1e-6)

    def test_publishes_stop_after_trigger_tail(self):
        s = sc.build_session(9_000_002, "dev-target", "S0")
        self.assertLessEqual(s["legit"]["pubs"][-1][0], s["t_a"] + sc.PUB_TAIL)
        self.assertEqual(s["legit"]["end_at"], s["t_a"] + sc.END_TAIL)

    def test_seeds_used_by_smoke_are_smoke(self):
        self.assertTrue(all(split_of_seed(s) == "SMOKE" for s in SMOKE))


if __name__ == "__main__":
    unittest.main()
