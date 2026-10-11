import copy
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from postdrift_auth import matched as M, matched_config as mc, scenarios as sc, seeds, twin_checks
from postdrift_auth.matched_quality import check_matched_quality
from tests.fake_twin import build_stage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = mc.load()


class PairConstructionInvariantTests(unittest.TestCase):
    """The frozen pair invariants, as automated tests (correction 2)."""

    def test_invariants_hold_for_many_families_of_both_kinds(self):
        for kind in ("M12", "M3"):
            for fs in range(3_200_000, 3_200_060):
                self.assertEqual(twin_checks.pair_invariants(sc.build_twin_family(fs, kind)), [], msg=(kind, fs))

    def test_shared_before_drift_and_differ_only_after(self):
        fam = sc.build_twin_family(3_200_007, "M12")
        A = {s["client_id"]: s for s in fam["arms"]["A"]["sessions"]}
        B = {s["client_id"]: s for s in fam["arms"]["B"]["sessions"]}
        n_diff_post = 0
        for cid in A:
            a, b = A[cid], B[cid]
            for key in ("base", "tau", "t_a"):
                self.assertEqual(a[key], b[key])
            self.assertEqual(a["legit"]["subs"], b["legit"]["subs"])
            self.assertEqual(a["legit"]["connect_at"], b["legit"]["connect_at"])
            pre = lambda s: [p for p in s["legit"]["pubs"] if p[0] < s["tau"]]  # noqa: E731
            self.assertEqual(pre(a), pre(b))
            self.assertGreater(len(pre(a)), 50)
            post = lambda s: [p for p in s["legit"]["pubs"] if p[0] >= s["tau"]]  # noqa: E731
            n_diff_post += post(a) != post(b)
            self.assertNotEqual(a["role"], b["role"])
        self.assertGreater(n_diff_post, 25)                     # the branches do differ after the onset

    def test_each_run_holds_both_roles_in_every_container(self):
        fam = sc.build_twin_family(3_200_011, "M12")
        for arm, plan in fam["arms"].items():
            roles = {s["client_id"]: s["role"] for s in plan["sessions"]}
            for c, v in plan["containers"].items():
                if v["kind"] == "mixed":
                    self.assertEqual(sorted(roles[i] for i in v["client_ids"]), ["benign", "benign", "malicious", "malicious"])
            self.assertEqual(sum(r == "malicious" for r in roles.values()), 20)

    def test_every_identity_is_malicious_in_exactly_one_run(self):
        fam = sc.build_twin_family(3_200_013, "M12")
        mal_a = {s["client_id"] for s in fam["arms"]["A"]["sessions"] if s["role"] == "malicious"}
        mal_b = {s["client_id"] for s in fam["arms"]["B"]["sessions"] if s["role"] == "malicious"}
        self.assertFalse(mal_a & mal_b)
        self.assertEqual(len(mal_a | mal_b), 40)

    def test_background_link_condition_shared_including_attacker_containers(self):
        fam = sc.build_twin_family(3_200_017, "M12")
        for c, v in fam["arms"]["A"]["containers"].items():
            if v["kind"] == "mixed":
                self.assertEqual(v["bg_netem"], fam["arms"]["B"]["containers"][c]["bg_netem"])
        for arm in "AB":
            for c, v in fam["arms"][arm]["containers"].items():
                if v["kind"] == "attacker":
                    self.assertEqual(v["bg_netem"], fam["arms"][arm]["containers"][c[4:]]["bg_netem"])

    def test_type_balance_and_m3_family(self):
        fam = sc.build_twin_family(3_200_019, "M12")
        for arm in "AB":
            cnt = {}
            for s in fam["arms"][arm]["sessions"]:
                cnt[s["scenario"]] = cnt.get(s["scenario"], 0) + 1
            self.assertEqual(cnt, {"B1": 20, "M1": 10, "M2": 10})
        f3 = sc.build_twin_family(3_200_019, "M3")
        self.assertTrue(all(s["scenario"] in ("B1", "M3") for a in "AB" for s in f3["arms"][a]["sessions"]))
        self.assertTrue(all(v["kind"] != "attacker" for a in "AB" for v in f3["arms"][a]["containers"].values()))   # M3 has no reconnect

    def test_frozen_generators_unchanged_by_the_new_stream_argument(self):
        # default streams reproduce the pre-existing plans (Pilot hashes are tested elsewhere); a role-specific stream gives different post-drift draws
        a = sc.build_session(3_200_000, "dev-001", "M2")
        b = sc.build_session(3_200_000, "dev-001", "M2", scn_stream="dev-001/scn/mal")
        self.assertEqual(a["legit"]["pubs"][:50], b["legit"]["pubs"][:50])
        self.assertNotEqual(a["params"], b["params"])

    def test_invariant_checker_detects_violations(self):
        fam = sc.build_twin_family(3_200_021, "M12")
        bad = copy.deepcopy(fam)
        cid = bad["arms"]["A"]["sessions"][0]["client_id"]
        bad["arms"]["A"]["sessions"][0]["legit"]["pubs"][3][2] += 1                    # pre-drift event differs
        self.assertTrue(twin_checks.pair_invariants(bad))
        bad = copy.deepcopy(fam)
        bad["arms"]["A"]["sessions"][0]["tau"] += 0.5                                    # onset differs
        self.assertTrue(twin_checks.pair_invariants(bad))
        bad = copy.deepcopy(fam)
        for s in bad["arms"]["B"]["sessions"]:
            s["role"] = "benign" if s["role"] == "malicious" else "malicious"        # roles no longer complementary
        self.assertTrue(twin_checks.pair_invariants(bad))
        bad = copy.deepcopy(fam)
        c = next(c for c, v in bad["arms"]["A"]["containers"].items() if v["kind"] == "mixed")
        bad["arms"]["A"]["containers"][c]["bg_netem"] = {"delay_ms": 1.0, "jitter_ms": 1.0, "loss_pct": 1.0}
        self.assertTrue(twin_checks.pair_invariants(bad))


class SeedBlockTests(unittest.TestCase):
    def test_blocks_are_fresh_disjoint_and_dry_run_separate(self):
        seeds.check_matched_reservations()
        dry = range(*seeds.MATCHED_RESERVATIONS["matched_dev"]["VAL"])
        m12 = range(*seeds.MATCHED_RESERVATIONS["matched_final_m12"]["TEST-SEEN"])
        m3 = range(*seeds.MATCHED_RESERVATIONS["matched_final_m3"]["TEST-HELD-OUT"])
        self.assertFalse(set(dry) & set(m12) or set(dry) & set(m3) or set(m12) & set(m3))
        self.assertEqual(seeds.split_of_seed(dry[0]), "VAL")
        self.assertEqual(seeds.split_of_seed(m12[0]), "TEST-SEEN")
        self.assertEqual(seeds.split_of_seed(m3[0]), "TEST-HELD-OUT")
        for name, per in seeds.DEV_RESERVATIONS.items():            # earlier pilots: no overlap with the matched blocks
            for lo, hi in per.values():
                for blk in (dry, m12, m3):
                    self.assertFalse(lo < blk[-1] and blk[0] < hi)

    def test_config_is_frozen_and_complete(self):
        n = CFG["novelty"]
        self.assertEqual((n["k"], n["caliper_sd_multiple"], n["min_retention"]), (10, 1.0, 0.30))
        self.assertEqual(n["primary_features"], ["iat_cv", "max_gap_ratio", "log_rate_ratio", "log_payload_ratio"])
        self.assertEqual(n["reference_scenarios"], ["S0", "B1"])
        pa = CFG["primary_analysis"]
        self.assertEqual((pa["n_boot"], pa["alpha_one_sided"]), (10000, 0.025))
        self.assertIn("ties", pa)
        self.assertEqual(CFG["seed_blocks"]["final_m12"]["n_families"], None)      # comes from the lock file only
        self.assertEqual(CFG["seed_blocks"]["dry_run"]["n_families"], 3)
        self.assertNotIn("TEST", CFG["seed_blocks"]["dry_run"]["split"])

    def test_changed_pilot_config_is_rejected(self):
        c = copy.deepcopy(CFG)
        c["pilot_v2_config_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            json.dump({k: v for k, v in c.items() if not k.startswith("_")}, open(p, "w"))
            with self.assertRaises(mc.MatchedConfigError):
                mc.load(p)


class NoveltyAndMetricTests(unittest.TestCase):
    def test_novelty_matches_a_brute_force_definition(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(60, 7)) * np.array([1, 2, 0.5, 1, 1, 0, 0]) + np.array([0, 1, 0, 0, 0, 0, 0])
        nov = M.Novelty(ref, M.MAG_IDX, 10, {"sub_novel_count": 1.0, "reconnects_60s": 1.0})
        x = rng.normal(size=(5, 7))
        got = nov.ln_n(x)
        mu, sd = ref[:, M.MAG_IDX].mean(0), ref[:, M.MAG_IDX].std(0, ddof=1)
        for i in range(5):
            d = np.sqrt((((x[i, M.MAG_IDX] - mu) / sd - (ref[:, M.MAG_IDX] - mu) / sd) ** 2).sum(1))
            self.assertAlmostEqual(got[i], float(np.log(np.sort(d)[:10].mean())))

    def test_sd_floor_for_count_features_in_the_full_novelty(self):
        ref = np.zeros((30, 7))
        ref[:, :4] = np.random.default_rng(1).normal(size=(30, 4))
        nov = M.Novelty(ref, M.ALL_IDX, 5, {"sub_novel_count": 1.0, "reconnects_60s": 1.0})
        x1, x0 = np.zeros((1, 7)), np.zeros((1, 7))
        x1[0, 6] = 1.0                                     # one reconnect = one unit of distance, not infinity
        self.assertTrue(np.isfinite(nov.ln_n(x1)[0]))
        self.assertGreater(nov.ln_n(x1)[0], nov.ln_n(x0)[0])

    def test_pair_wins_ties_and_ratio_estimator(self):
        pairs = {"ben": np.array([0, 1, 2, 3]), "mal": np.array([4, 5, 6, 7]), "family": np.array([1, 1, 2, 2]), "mal_scenario": np.array(["M1"] * 4)}
        s = np.array([.2, .2, .5, .5, .9, .2, .1, .5])
        w = M.pair_wins(s, pairs)
        self.assertEqual(w.tolist(), [1.0, 0.5, 0.0, 0.5])
        keep = np.array([True, True, False, True])
        self.assertAlmostEqual(w[keep].mean(), 2.0 / 3)

    def test_caliper_and_unmatched_pairs(self):
        pairs = {"ben": np.array([0, 1]), "mal": np.array([2, 3]), "family": np.array([1, 1]), "mal_scenario": np.array(["M2", "M2"])}
        ln = np.array([0.0, 0.0, 0.4, 2.0])
        self.assertEqual(M.caliper_keep(ln, pairs, 0.5).tolist(), [True, False])
        self.assertEqual(M.caliper_keep(ln, pairs, 0.4).tolist(), [True, False])
        self.assertEqual(M.caliper_keep(ln, pairs, 0.39).tolist(), [False, False])

    def test_auc_matches_scikit_learn_with_ties(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 200)
        s = np.round(rng.random(200), 1)
        self.assertAlmostEqual(M.auc_mw(s, y), roc_auc_score(y, s))

    def test_family_bootstrap_resamples_whole_families(self):
        fams = np.repeat([10, 11, 12, 13], 20)
        boot = M.boot_family_index(CFG, list(fams), "t", 50)
        self.assertEqual(boot.shape, (50, 4))
        self.assertTrue(((boot >= 0) & (boot < 4)).all())
        self.assertTrue((boot == M.boot_family_index(CFG, list(fams), "t", 50)).all())
        data = {"family": fams, "y": np.tile([0, 1], 40)}
        pairs = {"ben": np.arange(0, 80, 2), "mal": np.arange(1, 80, 2), "family": fams[::2], "mal_scenario": np.array(["M1"] * 40)}
        fs = M.FamilyStats(data, pairs)
        w = np.random.default_rng(3).random(40)
        keep = np.ones(40, dtype=bool)
        b = fs.pmp_boot(w, keep, boot)
        for i in (0, 7, 33):                                   # replicate = ratio over the resampled families
            idx = np.concatenate([fs.pair_idx[fs.fams[j]] for j in boot[i]])
            self.assertAlmostEqual(b[i], w[idx].mean())

    def test_sample_size_rule(self):
        ss = M.sample_size(np.array([0.70, 0.74, 0.66]), 25.0, CFG)
        self.assertAlmostEqual(ss["independent_pairs_floor"], 0.1)
        self.assertGreaterEqual(ss["sd_used"], 0.1)
        self.assertEqual(ss["F_m12"], max(ss["F_power"], ss["F_precision_m12"], 20))
        self.assertEqual(ss["F_power"], int(np.ceil(((1.959964 + 1.281552) * ss["sd_used"] / 0.10) ** 2)))
        self.assertGreaterEqual(ss["F_m3"], 20)
        wide = M.sample_size(np.array([0.4, 0.8, 0.6]), 25.0, CFG)
        self.assertGreater(wide["F_m12"], ss["F_m12"])                       # more variable families need more families


class QualityGateOnSyntheticTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.root = os.path.join(cls.tmp, "dry")
        cls.seeds = [3_200_000, 3_200_001]
        build_stage(cls.root, "MDEV", cls.seeds, "M12")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def q(self, root=None, seeds_=None, forbidden=frozenset()):
        r = root or self.root
        return check_matched_quality(r, "MDEV", seeds_ or self.seeds, "M12", os.path.join(r, "matched_manifest.json*"), forbidden)

    def test_passes_on_a_valid_tree(self):
        rep = self.q()
        self.assertTrue(rep["passed"], msg=[c for c in rep["checks"] if not c["ok"]])
        self.assertEqual(rep["n_runs"], 4)

    def test_dry_run_seed_in_final_set_is_rejected(self):
        rep = self.q(forbidden={3_200_001})
        self.assertFalse(rep["passed"])
        self.assertTrue(any("forbidden" in c["check"] for c in rep["checks"] if not c["ok"]))

    def test_tampered_roles_missing_run_and_leakage_are_detected(self):
        with tempfile.TemporaryDirectory() as d:
            r = os.path.join(d, "dry")
            shutil.copytree(self.root, r)
            gp = os.path.join(r, "MDEV-3200000-A", "ground_truth.jsonl")
            rows = [json.loads(l) for l in open(gp)]
            rows[0]["y"] = 1 - rows[0]["y"]
            open(gp, "w").write("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
            rep = self.q(r)
            self.assertFalse(rep["passed"])
            shutil.rmtree(os.path.join(r, "MDEV-3200001-B"))
            self.assertTrue(any("present" in c["check"] for c in self.q(r)["checks"] if not c["ok"]))

    def test_end_to_end_analysis_on_synthetic_tree(self):
        pilot_root = os.path.join(ROOT, CFG["source_data"]["root"])
        if not os.path.isdir(os.path.join(pilot_root, "TRAIN")):
            self.skipTest("Pilot-v2 source data not present")
        from postdrift_auth import matched_analysis as ma
        src = ma.Source(CFG)
        data = M.load_families(self.root, "MDEV", self.seeds)
        lock = {"caliper_primary": 1.0, "caliper_secondary": 1.0}
        cfg = copy.deepcopy(CFG)
        res, _ = ma.analyze_families(cfg, lock, src, data, "synthetic", "M12", B=40)
        self.assertEqual(res["n_pairs"], 80)
        for d in res["detectors"].values():
            self.assertIn("PMP_primary", d)
            self.assertIn("Delta_match", d)
            self.assertAlmostEqual(d["Delta_match"]["estimate"], d["AUC_pooled"]["estimate"] - d["PMP_primary"]["estimate"]
                                   if d["PMP_primary"]["estimate"] == d["PMP_primary"]["estimate"] else d["Delta_match"]["estimate"])
        self.assertIn(ma.decision(cfg, res)["label"], ("SUPPORTED", "NOT_SUPPORTED", "INCONCLUSIVE"))


if __name__ == "__main__":
    unittest.main()


class ReportWriterTests(unittest.TestCase):
    def test_tables_and_figures_from_a_synthetic_result(self):
        pilot_root = os.path.join(ROOT, CFG["source_data"]["root"])
        if not os.path.isdir(os.path.join(pilot_root, "TRAIN")):
            self.skipTest("Pilot-v2 source data not present")
        from postdrift_auth import matched_analysis as ma, matched_report as mr
        with tempfile.TemporaryDirectory() as d:
            build_stage(os.path.join(d, "dry"), "MDEV", [3_200_000, 3_200_001], "M12")
            data = M.load_families(os.path.join(d, "dry"), "MDEV", [3_200_000, 3_200_001])
            res, arrays = ma.analyze_families(CFG, {"caliper_primary": 1.0, "caliper_secondary": 1.0}, ma.Source(CFG), data, "t", "M12", B=30)
            files = mr.write_tables(res, os.path.join(d, "t"), "t") + mr.write_figures(res, arrays, os.path.join(d, "f"), "t")
            self.assertTrue(all(os.path.getsize(f) > 0 for f in files))
            self.assertGreaterEqual(len(files), 8)


class SchedulingTests(unittest.TestCase):
    def test_execute_run_accepts_a_shared_start_time_and_runner_uses_lockstep(self):
        import inspect
        import sys
        sys.path.insert(0, os.path.join(ROOT, "scripts", "generate_data"))
        import run_scale_smoke as rs
        sig = inspect.signature(rs.execute_run)
        self.assertIn("t0_epoch", sig.parameters)
        self.assertIn("stats_interval_s", sig.parameters)
        src = open(os.path.join(ROOT, "scripts", "generate_data", "run_matched_data.py")).read()
        self.assertIn("SYNC_MARGIN_S", src)
        self.assertIn("t0_pair", src)

    def test_start_time_too_close_is_rejected_before_any_container_work(self):
        import sys
        import time
        sys.path.insert(0, os.path.join(ROOT, "scripts", "generate_data"))
        import run_scale_smoke as rs
        from postdrift_auth import scenarios as s2
        fam = s2.build_twin_family(3_200_050, "M12")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                rs.execute_run("VAL", 3_200_050, os.path.join(d, "r"), {"commit": "x", "git_dirty": False}, {}, plan=fam["arms"]["A"], run_id="x", docker_tag="mdrtestx",
                               composition=s2.twin_composition("M12"), t0_epoch=time.time() + 1.0)
        import subprocess
        self.assertEqual(subprocess.run(["docker", "ps", "-aq", "--filter", "name=mdrtestx"], capture_output=True, text=True).stdout.strip(), "")


class DeviceClientLoggingTests(unittest.TestCase):
    """Root cause of the Stage-final stalls: per-event synchronous writes to a shared bind mount froze all containers at once."""

    def test_log_does_not_write_to_the_file_in_the_publishing_path(self):
        src = open(os.path.join(ROOT, "testbed", "device_client.py")).read()
        body = src[src.index("def log(**kw):"):src.index("def flush_log")]
        self.assertIn("_buf.append", body)
        self.assertNotIn("out.write", body)
        self.assertNotIn("buffering=1", src)
        self.assertIn("threading.Thread(target=_flusher, daemon=True)", src)
        self.assertIn("flush_log()\nos._exit(0)", src)

    def test_run_scale_smoke_uses_sparse_sequential_snapshots_without_b2(self):
        src = open(os.path.join(ROOT, "scripts", "generate_data", "run_scale_smoke.py")).read()
        self.assertIn('snap_source[c] = "pre_t0"', src)
        self.assertIn("if has_b2:", src)


class CaptureIntegrityTests(unittest.TestCase):
    """Packet capture must not depend on the shared bind mount, and kernel drops must be an explicit check."""

    def test_capture_is_local_to_the_container_and_drops_are_checked(self):
        src = open(os.path.join(ROOT, "scripts", "generate_data", "run_scale_smoke.py")).read()
        self.assertIn("-w /tmp/trace.pcap", src)
        self.assertNotIn("-w /out/trace.pcap", src)
        self.assertIn("-B 16384", src)
        self.assertIn('"docker", "cp", f"{cap}:/tmp/trace.pcap"', src)
        self.assertIn("M_capture_no_kernel_drops", src)
        self.assertIn("packets dropped by kernel", src)
        self.assertIn("lat_errors", src)           # logged-sends vs traced-PUBLISH mismatch is a reported failure, not a crash


class AuditScriptsTests(unittest.TestCase):
    def test_independent_reproduction_script_imports_no_project_analysis_code(self):
        import ast
        tree = ast.parse(open(os.path.join(ROOT, "scripts", "independent_check.py")).read())
        mods = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        self.assertNotIn("postdrift_auth", mods)

      # raw files are opened only through the read helpers (jl, csv readers)
