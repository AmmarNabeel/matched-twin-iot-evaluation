import os
import subprocess
import sys
import unittest

from postdrift_auth import seeds


class SeedTests(unittest.TestCase):
    def test_registry_is_disjoint(self):
        seeds.check_registry()
        spans = sorted((lo, hi) for _, lo, hi in seeds.SPLIT_RANGES)
        for (_, h1), (l2, _) in zip(spans, spans[1:]):
            self.assertLessEqual(h1, l2)

    def test_registry_detects_overlap(self):
        with self.assertRaises(ValueError):
            seeds.check_registry((("A", 0, 10), ("B", 5, 20)))

    def test_split_of_seed_and_boundaries(self):
        self.assertEqual(seeds.split_of_seed(1_000_000), "TRAIN")
        self.assertEqual(seeds.split_of_seed(1_999_999), "TRAIN")
        self.assertEqual(seeds.split_of_seed(2_000_000), "CAL")
        self.assertEqual(seeds.split_of_seed(9_000_000), "SMOKE")
        self.assertEqual(seeds.split_of_seed(9_000_005), "SMOKE")
        with self.assertRaises(ValueError):
            seeds.split_of_seed(7_000_000)

    def test_smoke_seeds_are_smoke_only(self):
        for i in range(6):
            self.assertEqual(seeds.split_of_seed(9_000_000 + i), "SMOKE")

    def test_draw_pinned_values_and_range(self):
        self.assertEqual(seeds.draw(9_000_000, "c", 0), 0.8502238102654889)
        self.assertEqual(seeds.draw(1, "dev/x", 7), 0.9570741745383472)
        for i in range(200):
            self.assertTrue(0.0 <= seeds.draw(9_000_001, "c", i) < 1.0)

    def test_draw_depends_on_every_argument(self):
        base = seeds.draw(9_000_000, "c", 3)
        self.assertNotEqual(base, seeds.draw(9_000_001, "c", 3))
        self.assertNotEqual(base, seeds.draw(9_000_000, "d", 3))
        self.assertNotEqual(base, seeds.draw(9_000_000, "c", 4))

    def test_draw_independent_of_hash_randomization(self):
        code = "import sys; sys.path.insert(0,'src'); from postdrift_auth.seeds import draw; print(repr(draw(9000000,'c',0)))"
        outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env={**os.environ, "PYTHONHASHSEED": h}).stdout.strip() for h in ("0", "1", "12345")}
        self.assertEqual(outs, {"0.8502238102654889"})

    def test_type_checks(self):
        with self.assertRaises(TypeError):
            seeds.draw(True, "c", 0)
        with self.assertRaises(TypeError):
            seeds.draw(1, "c", -1)


if __name__ == "__main__":
    unittest.main()
