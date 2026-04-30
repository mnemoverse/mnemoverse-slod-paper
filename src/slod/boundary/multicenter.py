"""Multi-center extension for multi-modal scales.

When K*(sigma) > 1 at a detected boundary, a single Frechet mean is lossy.
Multi-center SLoD: Phi^MC_sigma = {(mu_j, pi_j)}_{j=1}^K

Uses Riemannian k-means on the Poincare ball:
1. Assign each v_i to nearest center: argmin_j d_H(v_i, mu_j)
2. Update centers as weighted Frechet means of assigned points
3. Convergence guaranteed (geodesically convex objective)

Distortion improvement: O(log(n/K)) vs O(log n) for single center.
"""

from __future__ import annotations
