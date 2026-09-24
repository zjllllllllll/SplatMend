import unittest

import numpy as np

from studio.color_correction import _solve_laplace, correct_rgb
from scipy.ndimage import distance_transform_edt


class ColorCorrectionTests(unittest.TestCase):
    def test_laplace_interpolates_opposing_offsets_without_a_meeting_seam(self):
        mask = np.zeros((41, 41), dtype=bool)
        mask[1:-1, 1:-1] = True
        center = np.zeros_like(mask)
        center[20, 20] = True
        boundary = np.zeros((41, 41, 3), dtype=np.float32)
        boundary[..., 0] = np.linspace(-10, 10, 41)
        field, report = _solve_laplace(mask, center, boundary)
        # A linear ramp is the analytic harmonic solution for these boundaries.
        np.testing.assert_allclose(field[mask], boundary[mask], atol=1e-6)
        self.assertLess(report["relative_solver_residual"], 1e-8)

    def test_edge_matches_original_then_fades_inward_without_changing_outside(self):
        source = np.empty((256, 256, 4), dtype=np.uint8)
        source[..., :3] = (120, 110, 95)
        source[..., 3] = 255
        source[64:192, 64:192, 3] = 0
        repaired = np.full((256, 256, 3), (175, 135, 125), dtype=np.uint8)

        corrected, report = correct_rgb(source, repaired, radius=32, blur_sigma=12)

        np.testing.assert_array_equal(corrected[0, 0], source[0, 0, :3])
        near = np.abs(corrected[65, 128].astype(int) - source[65, 128, :3].astype(int)).sum()
        deep = np.abs(corrected[128, 128].astype(int) - source[128, 128, :3].astype(int)).sum()
        self.assertLess(near, 8)
        self.assertGreater(deep, near)
        np.testing.assert_array_equal(corrected[128, 128], repaired[128, 128])
        self.assertGreater(report["center_distance_pixels"], report["radius_pixels"])
        self.assertEqual(report["outside_changed_pixels"], 0)

    def test_dark_fringe_inside_reference_offset_cannot_tint_the_patch(self):
        source = np.empty((256, 256, 4), dtype=np.uint8)
        source[..., :3] = (120, 110, 95)
        source[..., 3] = 255
        source[64:192, 64:192, 3] = 0
        mask = source[..., 3] == 0
        repaired = np.full((256, 256, 3), (175, 135, 125), dtype=np.uint8)
        distance = distance_transform_edt(~mask)
        dark_source = source.copy()
        dark_source[(distance > 0) & (distance < 30), :3] = 1

        expected, _ = correct_rgb(source, repaired)
        actual, report = correct_rgb(dark_source, repaired)

        np.testing.assert_array_equal(actual[mask], expected[mask])
        np.testing.assert_array_equal(actual[~mask], dark_source[..., :3][~mask])
        self.assertEqual(report["reference_inner_distance_pixels"], 30)
        self.assertEqual(report["reference_outer_distance_pixels"], 180)


if __name__ == "__main__":
    unittest.main()
