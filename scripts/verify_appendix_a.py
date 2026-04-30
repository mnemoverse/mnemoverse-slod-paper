"""Numerical sanity check for Lemma A.2 of the SLoD paper Appendix A.

Refutes the original (and reviewer-proposed) bounds on a star graph K_{1,n}
at the hub focus near sigma = 0, and confirms the corrected bound
||dot w||_1 <= 2 ||L||_{1->1}.

Run:
    python scripts/verify_appendix_a.py [n]

Output for n=100 (default):
    ||L||_{1->1}                        = 11.0000  (= 1 + sqrt(n))
    lambda_max(L)                       = 2.0000
    ||u(0)||_1 (heat trace)             = 1.0000
    ||dot w(0)||_1 (analytic)           = 20.0000  (= 2 sqrt(n))
    ||dot w(eps)||_1 (numerical)        ~ 19.9600  (sigma = 1e-4)

    Bounds at hub focus near sigma=0:
      Original Lemma A.2 (lambda_max=2):              VIOLATED  20 > 2
      Reviewer's 2*lambda_max/Sigma_min = 4:          VIOLATED  20 > 4
      Corrected 2*||L||_{1->1} = 22:                  HOLDS     20 <= 22

The example confirms that the spectral norm lambda_max is insufficient
for the simplex-l^1 Lipschitz constant; the induced l^1 -> l^1 norm
||L||_{1->1} = max_j sum_i |L_{ij}| is the correct object on
graphs with non-trivial degree variance.

The kNN graphs SLoD uses in practice have OR-symmetrized degree ratio
<= 2, so ||L||_{1->1} <= 1 + sqrt(2) and L_w <= 2(1 + sqrt(2)) ~ 4.83.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.linalg import expm


def build_star_normalized_laplacian(n: int) -> np.ndarray:
    """Symmetric normalized Laplacian of K_{1,n} (hub = node 0)."""
    size = n + 1
    W = np.zeros((size, size))
    W[0, 1:] = 1
    W[1:, 0] = 1
    d = W.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
    return np.eye(size) - D_inv_sqrt @ W @ D_inv_sqrt


def induced_1_to_1_norm(M: np.ndarray) -> float:
    """Operator norm M : l^1 -> l^1 = max_j sum_i |M_{ij}| (max abs col sum)."""
    return float(np.max(np.abs(M).sum(axis=0)))


def w_dot_at_focus(L: np.ndarray, focus: int, sigma: float) -> tuple[np.ndarray, float]:
    """Compute (w_dot, ||w_dot||_1) at given sigma, focus = node index."""
    size = L.shape[0]
    e = np.zeros(size)
    e[focus] = 1.0
    u = expm(-sigma * L) @ e
    Z = u.sum()
    u_dot = -L @ u
    Z_dot = u_dot.sum()
    w = u / Z
    w_dot = u_dot / Z - w * Z_dot / Z
    return w_dot, float(np.abs(w_dot).sum())


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    L = build_star_normalized_laplacian(n)
    L1to1 = induced_1_to_1_norm(L)
    lambda_max = float(np.linalg.eigvalsh(L).max())

    print(f"Star graph K_{{1,{n}}}, hub focus")
    print(f"  ||L||_{{1->1}}                    = {L1to1:.4f}  (= 1 + sqrt({n}) = {1 + np.sqrt(n):.4f})")
    print(f"  lambda_max(L)                       = {lambda_max:.4f}")

    sigma_eps = 1e-4
    w_dot_eps, norm_eps = w_dot_at_focus(L, focus=0, sigma=sigma_eps)
    _, norm_zero = w_dot_at_focus(L, focus=0, sigma=0.0)

    print(f"  ||dot w(0)||_1 (sigma=0, analytic)  = {norm_zero:.4f}  (= 2 sqrt({n}) = {2*np.sqrt(n):.4f})")
    print(f"  ||dot w({sigma_eps:.0e})||_1 (numerical)    = {norm_eps:.4f}")
    print()

    print(f"Bounds at hub focus near sigma = 0 (taking ||dot w||_1 = {norm_zero:.2f}):")

    bound_original = lambda_max
    bound_reviewer = 2.0 * lambda_max / 1.0  # Sigma_min = 1 at sigma = 0
    bound_corrected = 2.0 * L1to1

    def status(actual: float, bound: float) -> str:
        return "HOLDS  " if actual <= bound else "VIOLATED"

    print(f"  Original Lemma A.2: lambda_max          = {bound_original:.2f}  {status(norm_zero, bound_original)}  ({norm_zero:.2f} {'<=' if norm_zero <= bound_original else '>'} {bound_original:.2f})")
    print(f"  Reviewer's 2*lambda_max/Sigma_min       = {bound_reviewer:.2f}  {status(norm_zero, bound_reviewer)}  ({norm_zero:.2f} {'<=' if norm_zero <= bound_reviewer else '>'} {bound_reviewer:.2f})")
    print(f"  Corrected 2*||L||_{{1->1}}               = {bound_corrected:.2f}  {status(norm_zero, bound_corrected)}  ({norm_zero:.2f} {'<=' if norm_zero <= bound_corrected else '>'} {bound_corrected:.2f})")


if __name__ == "__main__":
    main()
