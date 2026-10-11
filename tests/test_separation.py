import ast
import builtins
import inspect
import io
import os
import subprocess
import sys
import tempfile
import unittest

from postdrift_auth import extract_cli, features, trace
from postdrift_auth.features import Event
from tests.pcap_synth import PcapWriter, mqtt_connect, mqtt_publish

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
EXTRACTION_MODULES = ("features", "trace", "extract_cli")
FORBIDDEN_IMPORTS = {"scenarios", "mechanics", "accounting", "seeds", "costs", "policy"}
FORBIDDEN_WORDS = ("ground_truth", "scenario", "groundtruth", "attacker", "malicious")


def _code_strings_and_imports(path):
    tree = ast.parse(open(path).read())
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and node.body and isinstance(node.body[0], ast.Expr) \
                and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            docstring_nodes.add(id(node.body[0].value))
    strings, imports, names = [], set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_nodes:
            strings.append(node.value)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[-1])
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            imports.update(a.name.split(".")[-1] for a in node.names)
        elif isinstance(node, (ast.Name, ast.Attribute)):
            names.add(node.id if isinstance(node, ast.Name) else node.attr)
    return strings, imports, names


class ArchitecturalSeparationTests(unittest.TestCase):
    def test_extraction_modules_do_not_import_label_side_code(self):
        for m in EXTRACTION_MODULES:
            _, imports, _ = _code_strings_and_imports(os.path.join(SRC, "postdrift_auth", f"{m}.py"))
            self.assertFalse(imports & FORBIDDEN_IMPORTS, msg=f"{m} imports {imports & FORBIDDEN_IMPORTS}")

    def test_extraction_modules_mention_no_label_or_scenario_in_code(self):
        for m in EXTRACTION_MODULES:
            strings, _, names = _code_strings_and_imports(os.path.join(SRC, "postdrift_auth", f"{m}.py"))
            for s in strings + sorted(names):
                for w in FORBIDDEN_WORDS:
                    self.assertNotIn(w, s.lower(), msg=f"{m}: {s!r}")

    def test_features_module_performs_no_io(self):
        _, imports, names = _code_strings_and_imports(os.path.join(SRC, "postdrift_auth", "features.py"))
        self.assertFalse(imports & {"os", "io", "subprocess", "pathlib", "csv", "shutil", "sys"})
        self.assertFalse(names & {"open", "read", "write"})

    def test_signatures_admit_no_label_input(self):
        self.assertEqual(list(inspect.signature(features.extract_features).parameters), ["events", "client_id", "t_a"])
        self.assertEqual(list(inspect.signature(extract_cli.run).parameters), ["pcap", "t0", "trigger_path", "out_dir"])
        self.assertEqual(Event._fields, ("t", "kind", "client_id", "topic", "size", "dup"))


class RuntimeGroundTruthSeparationTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.ein, self.eout = os.path.join(self.root, "extract_in"), os.path.join(self.root, "extract_out")
        os.makedirs(self.ein)
        self.gt = os.path.join(self.root, "ground_truth.jsonl")
        open(self.gt, "w").write('{"session_id":"A","scenario":"M1","y":1}\n')
        w = PcapWriter()
        w.add(1000.0, 40001, mqtt_connect("A"))
        for i in range(14):
            w.add(1000.0 + 60 + 10 * i, 40001, mqtt_publish("t/a", 20, msgid=i + 1))
        w.write(os.path.join(self.ein, "trace.pcap"))
        open(os.path.join(self.ein, "trigger.jsonl"), "w").write('{"client_id":"A","t_a":200.0}\n')

    def _guarded_run(self):
        touched = []
        real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

        def spy(real):
            def f(path, *a, **k):
                touched.append(os.fspath(path) if not isinstance(path, int) else str(path))
                return real(path, *a, **k)
            return f
        builtins.open, io.open, os.open = spy(real_open), spy(real_io_open), spy(real_os_open)
        try:
            rows = extract_cli.run(os.path.join(self.ein, "trace.pcap"), "1000.0", os.path.join(self.ein, "trigger.jsonl"), self.eout)
        finally:
            builtins.open, io.open, os.open = real_open, real_io_open, real_os_open
        return rows, touched

    def test_extractor_never_opens_ground_truth(self):
        rows, touched = self._guarded_run()
        self.assertTrue(rows and touched)
        self.assertFalse([p for p in touched if "ground_truth" in p], msg=touched)

    def test_guard_would_catch_a_leaky_reader(self):
        """Control: the same spy flags a function that does read ground truth, so the test above is not vacuous."""
        touched, real = [], builtins.open
        builtins.open = lambda p, *a, **k: (touched.append(os.fspath(p)), real(p, *a, **k))[1]
        try:
            open(self.gt).close()
        finally:
            builtins.open = real
        self.assertTrue([p for p in touched if "ground_truth" in p])

    def test_extraction_succeeds_with_ground_truth_unreadable_and_in_a_separate_process(self):
        os.chmod(self.gt, 0)
        try:
            p = subprocess.run([sys.executable, "-m", "postdrift_auth.extract_cli", "--pcap", "trace.pcap", "--t0", "1000.0",
                                "--trigger", "trigger.jsonl", "--out-dir", self.eout], cwd=self.ein,
                               env={**os.environ, "PYTHONPATH": SRC}, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, msg=p.stderr)
            self.assertTrue(os.path.exists(os.path.join(self.eout, "features.csv")))
        finally:
            os.chmod(self.gt, 0o600)

    def test_feature_output_has_no_label_column(self):
        rows, _ = self._guarded_run()
        self.assertEqual(set(rows[0]), {"client_id", "t_a", *features.FEATURE_NAMES})


if __name__ == "__main__":
    unittest.main()
