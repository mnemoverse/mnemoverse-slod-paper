"""Smoke tests for Poincare ball operations.

These tests validate basic properties that must hold for any
correct implementation of the Poincare ball model.
"""

from __future__ import annotations

import numpy as np
import pytest


class TestPoincareBasicProperties:
    """Test mathematical properties of the Poincare ball."""

    def test_points_inside_ball(self, points_2d_ball):
        """All fixture points must lie inside the unit ball."""
        norms = np.linalg.norm(points_2d_ball, axis=1)
        assert np.all(norms < 1.0), f"Points outside ball: max norm = {norms.max()}"

    def test_origin_is_valid(self, origin_2d):
        """Origin (0, 0) is inside the ball."""
        assert np.linalg.norm(origin_2d) < 1.0

    def test_distance_symmetry(self, points_2d_ball):
        """d_H(x, y) == d_H(y, x) for all pairs."""
        from slod.core.poincare import poincare_distance

        for i in range(len(points_2d_ball)):
            for j in range(i + 1, len(points_2d_ball)):
                d_ij = poincare_distance(points_2d_ball[i], points_2d_ball[j])
                d_ji = poincare_distance(points_2d_ball[j], points_2d_ball[i])
                assert d_ij == pytest.approx(d_ji, abs=1e-6)

    def test_distance_positive(self, points_2d_ball):
        """d_H(x, y) >= 0, and d_H(x, x) == 0."""
        from slod.core.poincare import poincare_distance

        for i in range(len(points_2d_ball)):
            d_ii = poincare_distance(points_2d_ball[i], points_2d_ball[i])
            assert d_ii == pytest.approx(0.0, abs=1e-6)

            for j in range(i + 1, len(points_2d_ball)):
                d_ij = poincare_distance(points_2d_ball[i], points_2d_ball[j])
                assert d_ij > 0.0

    def test_triangle_inequality(self, points_2d_ball):
        """d_H(a, c) <= d_H(a, b) + d_H(b, c) for all triples."""
        from slod.core.poincare import poincare_distance

        n = len(points_2d_ball)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if i == j or j == k or i == k:
                        continue
                    d_ik = poincare_distance(points_2d_ball[i], points_2d_ball[k])
                    d_ij = poincare_distance(points_2d_ball[i], points_2d_ball[j])
                    d_jk = poincare_distance(points_2d_ball[j], points_2d_ball[k])
                    assert d_ik <= d_ij + d_jk + 1e-6, (
                        f"Triangle inequality violated: d({i},{k})={d_ik:.6f} > "
                        f"d({i},{j})={d_ij:.6f} + d({j},{k})={d_jk:.6f}"
                    )

    def test_exp_log_roundtrip(self, points_2d_ball, origin_2d):
        """Exp_x(Log_x(y)) == y (roundtrip property)."""
        from slod.core.poincare import exp_map, log_map

        x = origin_2d
        for y in points_2d_ball[:5]:
            v = log_map(x, y)
            y_recovered = exp_map(x, v)
            np.testing.assert_allclose(y_recovered, y, atol=1e-5)
