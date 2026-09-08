import json
import tempfile
import unittest
from pathlib import Path

from studio.diagnostics import depth_rejection_message


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        outputs = Path(__file__).resolve().parents[2] / "outputs"
        outputs.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="studio_diagnostic_test_", dir=outputs)
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.report = self.folder / "pipeline" / "depth" / "depth_acceptance.json"
        self.report.parent.mkdir(parents=True)

    def test_numeric_rejection_is_explained_without_changing_report(self):
        self.report.write_text(json.dumps({"status": "rejected", "surface": {
            "mode": "single_plane", "hole_to_ring_plane_residual_p95": .094,
            "effective_max_hole_to_ring_plane_p95": .08014473}}))
        before = self.report.read_bytes()
        message = depth_rejection_message(self.folder)
        self.assertIn("Image generation succeeded", message)
        self.assertIn("0.094000 exceeds 0.080145", message)
        self.assertIn("Retrying unchanged inputs may fail again", message)
        self.assertEqual(before, self.report.read_bytes())

    def test_arbitrary_strings_and_nonfinite_values_are_not_echoed(self):
        for value in ["SECRET_OR_SIGNED_URL", True, None, float("nan"), float("inf"), -1, 1e20]:
            self.report.write_text(json.dumps({"status": "rejected", "failures": ["SECRET_OR_SIGNED_URL"],
                "surface": {"mode": "single_plane", "hole_to_ring_plane_residual_p95": value,
                            "effective_max_hole_to_ring_plane_p95": .08}}))
            message = depth_rejection_message(self.folder)
            self.assertNotIn("SECRET_OR_SIGNED_URL", message)
            self.assertNotIn("exceeds", message)

    def test_missing_invalid_large_or_accepted_reports_have_no_override(self):
        self.assertIsNone(depth_rejection_message(self.folder))
        for raw in [b"{", b"[]", b"null", b"x" * 65537,
                    b'{"status":"accepted"}', b'{"status":"rejected","surface":null}']:
            self.report.write_bytes(raw)
            if b'"rejected"' in raw:
                self.assertIn("step 3 rejected", depth_rejection_message(self.folder))
            else:
                self.assertIsNone(depth_rejection_message(self.folder))
