import unittest

from postdrift_auth import mechanics, scenarios as sc
from postdrift_auth.features import extract_features
from tests.helpers import ideal_device_log, ideal_events


def failed(checks):
    return [c for c in checks if not c["ok"]]


class IdealTraceMechanicsTests(unittest.TestCase):
    """The checker accepts the ideal trace of each plan and rejects tampered ones (deterministic mechanics)."""

    def test_non_netem_scenarios_pass_on_ideal_trace(self):
        for i, scn in enumerate(sc.SCENARIOS):
            if scn == "B2":
                continue
            s = sc.build_session(9_000_000 + i, "dev-target", scn)
            log = "Client dev-target [1.2.3.4:5] disconnected: session taken over." if scn == "M1" else ""
            self.assertEqual(failed(mechanics.check_scenario(s, ideal_events(s), broker_log=log)), [], msg=scn)

    def test_pass_on_many_smoke_seeds_covering_both_topic_branches(self):
        for scn in ("B1", "M1", "M2", "M3"):
            for seed in range(9_000_100, 9_000_130):
                s = sc.build_session(seed, "dev-target", scn)
                log = "Client dev-target [1.2.3.4:5] disconnected: session taken over." if scn == "M1" else ""
                self.assertEqual(failed(mechanics.check_scenario(s, ideal_events(s), broker_log=log)), [], msg=(scn, seed))

    def test_b2_netem_mechanics(self):
        s = sc.build_session(9_000_002, "dev-target", "B2")
        ctl = sc.build_session(9_000_002, "dev-ctl", "S0")
        delay = s["params"]["delay_ms"] / 1e3
        ev = ideal_events(s, delay_after_tau=delay) + ideal_events(ctl)
        log = ideal_device_log(s) + ideal_device_log(ctl)
        n = s["netem"]
        evidence = {"dev_pre": "qdisc noqueue 0: root refcnt 2\n", "ctl_pre": "qdisc noqueue 0: root refcnt 2\n",
                    "dev_post": f"qdisc netem 8001: root refcnt 2 limit 1000 delay {n['delay_ms']}ms {n['jitter_ms']}ms loss {n['loss_pct']}%\n",
                    "ctl_post": "qdisc noqueue 0: root refcnt 2\n", "apply_start_rel": s["tau"] + 0.01, "apply_end_rel": s["tau"] + 0.2}
        self.assertEqual(failed(mechanics.check_scenario(s, ev, device_log=log, netem_evidence=evidence)), [])
        bad = dict(evidence, ctl_post=evidence["dev_post"])        # netem leaked to the control container
        self.assertTrue(failed(mechanics.check_scenario(s, ev, device_log=log, netem_evidence=bad)))
        late = dict(evidence, apply_end_rel=s["tau"] + 9)         # applied too late
        self.assertTrue(failed(mechanics.check_scenario(s, ev, device_log=log, netem_evidence=late)))
        nodelay = ideal_events(s) + ideal_events(ctl)             # netem "applied" but no latency in the trace
        self.assertTrue(failed(mechanics.check_scenario(s, nodelay, device_log=log, netem_evidence=evidence)))

    def test_m1_requires_second_connect_at_tau_and_broker_takeover(self):
        s = sc.build_session(9_000_003, "dev-target", "M1")
        ev = ideal_events(s)
        self.assertTrue(failed(mechanics.check_scenario(s, ev, broker_log="")))                     # no takeover line
        storm = "Client dev-target [1.2.3.4:5] disconnected: session taken over.\n" * 3
        self.assertTrue(failed(mechanics.check_scenario(s, ev, broker_log=storm)))                  # reconnect storm
        no_att = [e for e in ev if not (e.kind == "CONNECT" and e.t > 1)]
        self.assertTrue(failed(mechanics.check_scenario(s, no_att, broker_log="Client dev-target [1.2.3.4:5] disconnected: session taken over.")))

    def test_m2_and_m3_fail_if_a_reconnect_or_novel_subscription_appears(self):
        from postdrift_auth.features import Event
        for scn, seed in (("M2", 9_000_004), ("M3", 9_000_005)):
            s = sc.build_session(seed, "dev-target", scn)
            ev = ideal_events(s)
            self.assertEqual(failed(mechanics.check_scenario(s, ev)), [])
            rec = ev + [Event(s["tau"], "CONNECT", "dev-target", None, None, None)]
            self.assertTrue(failed(mechanics.check_scenario(s, rec)), msg=scn)
            sub = ev + [Event(s["tau"] + 1, "SUBSCRIBE", "dev-target", "novel/x", None, None)]
            self.assertTrue(failed(mechanics.check_scenario(s, sub)), msg=scn)

    def test_tampered_sequence_is_rejected(self):
        s = sc.build_session(9_000_001, "dev-target", "B1")
        ev = ideal_events(s)
        k = next(i for i, e in enumerate(ev) if e.kind == "PUBLISH")
        self.assertTrue(failed(mechanics.check_scenario(s, ev[:k] + ev[k + 1:])))

    def test_latencies_alignment_error(self):
        s = sc.build_session(9_000_002, "dev-target", "B2")
        with self.assertRaises(ValueError):
            mechanics.latencies(ideal_device_log(s)[:-1], ideal_events(s), "dev-target")


class NetemReorderRegressionTests(unittest.TestCase):
    """Stage 3A: B2 netem jitter/loss can swap the first-observation order of two segments. Not a lost or invented message."""

    def _b2(self):
        s = sc.build_session(9_000_002, "dev-target", "B2")
        return s, ideal_events(s, delay_after_tau=s["params"]["delay_ms"] / 1e3)

    @staticmethod
    def _swap(ev, i, j):
        a, b = ev[i], ev[j]
        ev = list(ev)
        ev[i], ev[j] = a._replace(topic=b.topic, size=b.size), b._replace(topic=a.topic, size=a.size)   # times stay in order
        return ev

    def _post_pub_indices(self, s, ev):
        idx = [k for k, e in enumerate(ev) if e.kind == "PUBLISH" and e.t >= s["tau"] + 5]
        # two adjacent publishes that differ in (topic, size) so the swap is observable
        for a, b in zip(idx, idx[1:]):
            if (ev[a].topic, ev[a].size) != (ev[b].topic, ev[b].size):
                return a, b
        raise AssertionError("no distinguishable pair")

    def test_swap_after_onset_is_accepted_under_b2(self):
        s, ev = self._b2()
        a, b = self._post_pub_indices(s, ev)
        got = mechanics.check_sequence_and_timing(s, self._swap(ev, a, b), under_netem_after=s["tau"])
        self.assertEqual(failed(got), [], msg=got)
        self.assertIn("positions differing: 2", got[0]["detail"])

    def test_same_swap_is_rejected_when_no_netem_is_allowed(self):
        s, ev = self._b2()
        a, b = self._post_pub_indices(s, ev)
        self.assertTrue(failed(mechanics.check_sequence_and_timing(s, self._swap(ev, a, b), under_netem_after=None)))

    def test_swap_before_onset_is_rejected(self):
        s, ev = self._b2()
        idx = [k for k, e in enumerate(ev) if e.kind == "PUBLISH" and e.t < s["tau"] - 5]
        a, b = next((a, b) for a, b in zip(idx, idx[1:]) if (ev[a].topic, ev[a].size) != (ev[b].topic, ev[b].size))
        self.assertTrue(failed(mechanics.check_sequence_and_timing(s, self._swap(ev, a, b), under_netem_after=s["tau"])))

    def test_lost_or_invented_or_altered_messages_are_still_rejected(self):
        s, ev = self._b2()
        k = next(i for i, e in enumerate(ev) if e.kind == "PUBLISH" and e.t > s["tau"] + 5)
        self.assertTrue(failed(mechanics.check_sequence_and_timing(s, ev[:k] + ev[k + 1:], under_netem_after=s["tau"])))     # lost
        self.assertTrue(failed(mechanics.check_sequence_and_timing(s, ev + [ev[k]], under_netem_after=s["tau"])))           # duplicated
        self.assertTrue(failed(mechanics.check_sequence_and_timing(s, ev[:k] + [ev[k]._replace(size=ev[k].size + 1)] + ev[k + 1:], under_netem_after=s["tau"])))

    def test_latency_pairing_survives_swapped_arrival_order(self):
        s, ev = self._b2()
        a, b = self._post_pub_indices(s, ev)
        lat = mechanics.latencies(ideal_device_log(s), self._swap(ev, a, b), "dev-target")
        self.assertEqual(len(lat), len(ideal_device_log(s)))


class IdealTraceFeatureDeterminismTests(unittest.TestCase):
    def test_feature_vector_identical_for_same_seed_and_finite_for_all_scenarios(self):
        for i, scn in enumerate(sc.SCENARIOS):
            seed = 9_000_000 + i
            vecs = []
            for _ in range(2):
                s = sc.build_session(seed, "dev-target", scn)
                vecs.append(extract_features(ideal_events(s), "dev-target", s["t_a"]))
            self.assertEqual(vecs[0], vecs[1], msg=scn)
            self.assertEqual(len(vecs[0]), 7)

    def test_features_use_nothing_after_t_a_on_a_full_scenario_trace(self):
        s = sc.build_session(9_000_003, "dev-target", "M1")
        ev = ideal_events(s)
        cut = [e for e in ev if e.t <= s["t_a"]]
        self.assertEqual(extract_features(ev, "dev-target", s["t_a"]), extract_features(cut, "dev-target", s["t_a"]))


if __name__ == "__main__":
    unittest.main()


class CheckerRefinementRegressionTests(unittest.TestCase):
    """Matched-drift final stage: deterministic false alarms of the plan/trace comparison (not live-system problems)."""

    def test_rounding_slack_in_the_planned_interval_check(self):
        import copy
        s = copy.deepcopy(sc.build_session(9_000_001, "dev-target", "B1"))
        reg = s["regimes"]["pre"]
        hi = reg["period"] * (1 + reg["jitter"])
        s["tau"] = 1e6                                              # every interval belongs to the pre-drift regime
        s["legit"]["pubs"] = [[round(k * (hi + 1.5e-6), 6), "dev/dev-target/telemetry", 100] for k in range(20)]
        s["attacker"] = None
        self.assertTrue(mechanics.check_interval_bounds(s, [])["ok"])       # 1.5 microseconds beyond the bound = planned-time rounding
        self.assertGreater(1.5e-6, 1e-9)                                     # the old slack (1e-9) would have flagged it
        s["legit"]["pubs"] = [[round(k * (hi + 5e-5), 6), "dev/dev-target/telemetry", 100] for k in range(20)]
        self.assertFalse(mechanics.check_interval_bounds(s, [])["ok"])      # a real excess is still caught

    def test_publish_scheduled_just_before_tau_but_arriving_after_it(self):
        s = sc.build_session(9_000_004, "dev-target", "M2")
        ev = ideal_events(s)
        k = max(i for i, e in enumerate(ev) if e.kind == "PUBLISH" and e.t < s["tau"])
        ev[k] = ev[k]._replace(t=s["tau"] + 0.004)                      # arrives 4 ms after tau although planned before it
        self.assertEqual(failed([c for c in mechanics.check_scenario(s, ev) if "ratio" in c["check"]]), [])
        by_arrival = mechanics._post_ratios(s, ev)
        by_plan = mechanics._post_ratios_by_plan_window(s, ev)
        self.assertNotEqual(by_arrival[3], by_plan[3])                    # the two window rules really differ on this trace

    def test_ratio_is_compared_with_the_realized_plan_not_the_nominal_multiplier(self):
        s = sc.build_session(9_000_004, "dev-target", "M2")
        s["params"] = dict(s["params"], rate_mult=s["params"]["rate_mult"] * 3.0)   # nominal multiplier no longer describes the realized plan
        self.assertEqual(failed([c for c in mechanics.check_scenario(s, ideal_events(s)) if "ratio" in c["check"]]), [])

    def test_a_trace_that_departs_from_the_plan_still_fails_the_ratio_check(self):
        s = sc.build_session(9_000_004, "dev-target", "M2")
        ev = ideal_events(s)
        late = [e._replace(t=e.t * 1.4 - 0.4 * s["tau"]) if e.kind == "PUBLISH" and e.t >= s["tau"] else e for e in ev]   # post-drift slows down by 40%
        self.assertTrue(failed(mechanics.check_scenario(s, late)))
