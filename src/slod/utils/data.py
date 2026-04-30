"""Data generation and loading for experiments.

- HSBM (Hierarchical Stochastic Block Model): synthetic 3-level hierarchy
- Poincare embedding: contrastive loss (Nickel & Kiela 2017) via geoopt
- Hyperbolic MDS (legacy): embed graph distances via Sammon stress
"""

from __future__ import annotations

import geoopt
import networkx as nx
import numpy as np
import scipy.sparse
import torch

from slod.core.poincare import project_to_ball


def generate_hsbm(
    n_nodes: int = 1024,
    n_macro: int = 2,
    n_meso_per_macro: int = 4,
    n_micro_per_meso: int = 8,
    r: float = 20.0,
    seed: int = 42,
) -> tuple[nx.Graph, dict[str, np.ndarray]]:
    """Generate a 3-level Hierarchical Stochastic Block Model.

    Paper: Section 5.1, Experiment 1.

    Hierarchy: n_macro → n_meso_per_macro → n_micro_per_meso communities.
    Default: 2 macro → 8 meso → 64 micro, 1024 nodes (16 per micro).

    Edge probabilities (paper spec):
        - Same micro-community:        p_within = 2*r / N
        - Same meso, different micro:   p_meso   = 8 / N
        - Same macro, different meso:   p_macro  = 2 / N
        - Different macro:              p_between = 0.5 / N

    At r=20: p_within=40/N, matching paper's stated values.

    Args:
        n_nodes: Total number of nodes.
        n_macro: Number of macro-level communities.
        n_meso_per_macro: Meso communities per macro community.
        n_micro_per_meso: Micro communities per meso community.
        r: Within-community scaling factor. Higher r = clearer hierarchy.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (graph, labels_dict) where labels_dict has keys
        "macro", "meso", "micro", each mapping to shape (n_nodes,) array.
    """
    if n_macro < 1 or n_meso_per_macro < 1 or n_micro_per_meso < 1:
        raise ValueError(
            "Hierarchy parameters must be positive integers: "
            f"n_macro={n_macro}, n_meso_per_macro={n_meso_per_macro}, "
            f"n_micro_per_meso={n_micro_per_meso}."
        )
    if n_nodes < 1:
        raise ValueError(f"`n_nodes` must be positive; got {n_nodes}.")
    if r <= 0:
        raise ValueError(f"`r` must be positive; got {r}.")

    total_meso = n_macro * n_meso_per_macro
    total_micro = total_meso * n_micro_per_meso

    if n_nodes % total_micro != 0:
        raise ValueError(
            f"`n_nodes` ({n_nodes}) must be divisible by "
            f"total micro-communities ({total_micro} = "
            f"{n_macro}*{n_meso_per_macro}*{n_micro_per_meso})."
        )

    nodes_per_micro = n_nodes // total_micro
    sizes = [nodes_per_micro] * total_micro

    # Edge probabilities from paper Section 5.1
    n = n_nodes
    p_within = min(2.0 * r / n, 0.99)
    p_meso = min(8.0 / n, 0.99)
    p_macro = min(2.0 / n, 0.99)
    p_between = min(0.5 / n, 0.99)

    # Build (total_micro x total_micro) probability matrix
    probs = np.full((total_micro, total_micro), p_between)

    for mi in range(total_micro):
        # Which meso and macro does micro-community mi belong to?
        meso_i = mi // n_micro_per_meso
        macro_i = meso_i // n_meso_per_macro

        for mj in range(mi, total_micro):
            meso_j = mj // n_micro_per_meso
            macro_j = meso_j // n_meso_per_macro

            if mi == mj:
                probs[mi, mj] = p_within
            elif meso_i == meso_j:
                probs[mi, mj] = p_meso
                probs[mj, mi] = p_meso
            elif macro_i == macro_j:
                probs[mi, mj] = p_macro
                probs[mj, mi] = p_macro
            # else: p_between (already set)

    graph = nx.stochastic_block_model(sizes, probs.tolist(), seed=seed)

    # Build label arrays
    macro_labels = np.empty(n_nodes, dtype=np.int64)
    meso_labels = np.empty(n_nodes, dtype=np.int64)
    micro_labels = np.empty(n_nodes, dtype=np.int64)

    for mi in range(total_micro):
        meso_id = mi // n_micro_per_meso
        macro_id = meso_id // n_meso_per_macro
        start = mi * nodes_per_micro
        end = start + nodes_per_micro
        macro_labels[start:end] = macro_id
        meso_labels[start:end] = meso_id
        micro_labels[start:end] = mi

    labels = {"macro": macro_labels, "meso": meso_labels, "micro": micro_labels}
    return graph, labels


def graph_to_laplacian(graph: nx.Graph) -> scipy.sparse.csr_matrix:
    """Convert networkx graph to sparse graph Laplacian."""
    adjacency = nx.adjacency_matrix(graph).astype(np.float64)
    degree = scipy.sparse.diags(np.array(adjacency.sum(axis=1)).flatten())
    return (degree - adjacency).tocsr()


def poincare_embed(
    graph: nx.Graph,
    dim: int = 10,
    epochs: int = 50,
    lr: float = 1.0,
    burn_in: int = 10,
    burn_in_lr: float = 0.1,
    n_negative: int = 10,
    seed: int = 42,
    verbose: bool = True,
    device: str | None = None,
) -> np.ndarray:
    """Embed graph into Poincare ball B^d via contrastive loss.

    Implements Nickel & Kiela (2017) ranking loss using geoopt:
        L(u,v) = d_H(u,v)^2 + log sum_{v' in {v}+Neg} exp(-d_H(u,v')^2)

    This is cross-entropy where the positive edge should have the
    smallest distance among positive + negative samples.

    Key differences from hyperbolic_mds():
        - Contrastive (ranking) loss instead of Sammon stress (reconstruction)
        - Batched training per-epoch instead of full-batch O(N^2)
        - Burn-in period with low LR for stable early training
        - Scales to N=80K+ (WordNet); MDS fails at N>500

    Args:
        graph: Input graph with integer node labels 0..n-1. Must be connected.
        dim: Embedding dimension (10 is standard for hierarchical graphs).
        epochs: Number of full passes over edges.
        lr: Learning rate after burn-in (default 1.0 per Nickel & Kiela).
        burn_in: Number of burn-in epochs with reduced LR.
        burn_in_lr: Learning rate during burn-in phase.
        n_negative: Number of negative samples per positive edge.
        seed: Random seed for reproducibility.

    Returns:
        Points array of shape (N, dim) inside B^d.
    """
    n = graph.number_of_nodes()

    # Device resolution: default CPU (backward-compat with existing callers).
    # For large graphs (N > 10K), callers should pass device="cuda".
    if device is None:
        device = "cpu"
    dev = torch.device(device)

    # Pre-compute edge tensors for batched training
    edge_list = list(graph.edges())
    edge_src = torch.tensor([e[0] for e in edge_list], dtype=torch.long, device=dev)
    edge_dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long, device=dev)
    n_edges = len(edge_list)

    torch.manual_seed(seed)
    # rng_gen stays on CPU (torch.Generator CPU works for all devices — generated
    # indices are transferred to the parameter device implicitly via indexing).
    rng_gen = torch.Generator().manual_seed(seed)

    ball = geoopt.PoincareBall(c=1.0)
    # Generate init on CPU with the seeded generator for cross-device
    # reproducibility, then move to the target device. This ensures that
    # seed=X on CPU and seed=X on CUDA produce the same initial points.
    init = (torch.randn(n, dim, dtype=torch.float64) * 0.1).to(dev)
    points = geoopt.ManifoldParameter(ball.projx(init), manifold=ball)

    optimizer = geoopt.optim.RiemannianAdam([points], lr=burn_in_lr)

    total_epochs = burn_in + epochs
    for epoch in range(total_epochs):
        # Switch LR after burn-in
        if epoch == burn_in:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        optimizer.zero_grad()

        # Positive distances: all edges at once — shape (|E|,)
        d_pos = ball.dist(points[edge_src], points[edge_dst])

        # Negative samples: (|E|, n_negative) random node indices
        # Generate on CPU with the seeded generator for determinism, then
        # move to the parameter device for indexing.
        neg_idx = torch.randint(0, n, (n_edges, n_negative), generator=rng_gen).to(dev)

        # Negative distances: expand source to match negatives
        src_expanded = edge_src.unsqueeze(1).expand(-1, n_negative).reshape(-1)
        neg_flat = neg_idx.reshape(-1)
        d_neg = ball.dist(points[src_expanded], points[neg_flat]).reshape(
            n_edges, n_negative
        )

        # Cross-entropy loss (Nickel & Kiela 2017)
        # logits = -d^2; target = column 0 (positive)
        all_dists_sq = torch.cat(
            [d_pos.unsqueeze(1).pow(2), d_neg.pow(2)], dim=1
        )
        logits = -all_dists_sq
        target = torch.zeros(n_edges, dtype=torch.long, device=dev)
        loss = torch.nn.functional.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        # Log progress
        if verbose and (epoch % max(1, total_epochs // 10) == 0 or epoch == total_epochs - 1):
            norms = points.data.norm(dim=1)
            phase = "burn-in" if epoch < burn_in else "train"
            print(
                f"    [{phase}] epoch {epoch:4d}/{total_epochs} "
                f"loss={loss.item():.4f} "
                f"norm=[{norms.min():.3f}, {norms.max():.3f}]"
            )

    # Extract numpy, clamp to ball interior
    result = points.detach().cpu().numpy()
    for i in range(n):
        result[i] = project_to_ball(result[i], eps=1e-5)

    return result


def hyperbolic_mds(
    graph: nx.Graph,
    dim: int = 10,
    max_iter: int = 500,
    lr: float = 0.05,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """Embed graph into Poincare ball B^d via Riemannian stress minimization.

    Minimizes Sammon stress on the Poincare ball:
        L = sum_{i<j} w_ij * (d_H(x_i, x_j) - delta_ij)^2

    where delta_ij = shortest path distance in graph,
    w_ij = 1 / max(delta_ij, 1)^2 (Sammon weighting: emphasizes short distances).

    Uses geoopt.ManifoldParameter + geoopt.optim.RiemannianAdam.

    Note:
        Graph should be connected for meaningful results. Disconnected pairs
        get fallback distance N (number of nodes), which degrades Sammon stress
        quality. Callers should extract the largest connected component and
        relabel nodes to 0..n-1 before calling this function.

    Args:
        graph: Input graph with integer node labels 0..n-1.
        dim: Embedding dimension.
        max_iter: Optimization iterations.
        lr: Learning rate for RiemannianAdam (0.05 works well for N<=1024).
        seed: Random seed.
        device: "cpu" (default, backward-compat) or "cuda" for GPU acceleration
            on RTX-class hardware. Init is generated on CPU then moved to the
            target device, mirroring poincare_embed's pattern, so seed=X gives
            the same starting point on either device.

    Returns:
        Points array of shape (N, dim) inside B^d.
    """
    n = graph.number_of_nodes()
    dev = torch.device(device)

    # Compute shortest path distances
    sp_dict = dict(nx.all_pairs_shortest_path_length(graph))
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = sp_dict.get(i, {}).get(j, n)

    # Target distances and Sammon weights (upper triangle only for loss)
    target = torch.tensor(dist_matrix, dtype=torch.float64, device=dev)
    weight = 1.0 / torch.clamp(target, min=1.0) ** 2
    # Zero out diagonal (self-distances)
    weight.fill_diagonal_(0.0)
    # Use upper triangle to avoid double-counting
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=1)

    # Initialize on the Poincare ball. Init generated on CPU with seeded RNG
    # then moved to device — same idiom as poincare_embed, ensures cross-device
    # determinism (seed=X gives identical starting points on cpu and cuda).
    torch.manual_seed(seed)
    ball = geoopt.PoincareBall(c=1.0)
    init = (torch.randn(n, dim, dtype=torch.float64) * 0.01).to(dev)
    points = geoopt.ManifoldParameter(ball.projx(init), manifold=ball)

    optimizer = geoopt.optim.RiemannianAdam([points], lr=lr)

    for _epoch in range(max_iter):
        optimizer.zero_grad()

        # Pairwise hyperbolic distances
        dists = ball.dist(points.unsqueeze(1), points.unsqueeze(0))  # (N, N)

        # Sammon stress (upper triangle)
        residuals = (dists - target) ** 2 * weight
        loss = residuals[mask].sum()

        loss.backward()
        optimizer.step()

    # Extract numpy, safety clamp to ball interior
    result = points.detach().cpu().numpy()
    for i in range(n):
        result[i] = project_to_ball(result[i], eps=1e-5)

    return result


def euclidean_mds(
    graph: nx.Graph,
    dim: int = 10,
    max_iter: int = 500,
    lr: float = 0.05,
    seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """Euclidean Sammon MDS baseline for ablation (mirrors hyperbolic_mds API).

    Same Sammon-weighted stress minimization as hyperbolic_mds, but uses
    plain L2 distance in R^d instead of the Poincaré-ball metric. The only
    geometric difference is the distance function — optimizer (Adam), loss
    (Sammon stress on upper-triangle with 1/delta^2 weighting), iteration
    budget, and seed are identical, so any ablation gap between the two
    variants is attributable to hyperbolic vs Euclidean geometry, not to
    optimization differences.

    Args:
        graph: Input graph with integer node labels 0..n-1. Should be
            connected; disconnected pairs get fallback distance N.
        dim: Embedding dimension.
        max_iter: Optimization iterations (default matches hyperbolic_mds).
        lr: Learning rate for Adam.
        seed: Random seed.
        device: "cpu" (default) or "cuda". Same cross-device determinism
            convention as hyperbolic_mds.

    Returns:
        Points array of shape (N, dim) in R^d. NOT constrained to any ball.
    """
    n = graph.number_of_nodes()
    dev = torch.device(device)

    # Shortest-path distances (identical to hyperbolic_mds).
    sp_dict = dict(nx.all_pairs_shortest_path_length(graph))
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = sp_dict.get(i, {}).get(j, n)

    target = torch.tensor(dist_matrix, dtype=torch.float64, device=dev)
    weight = 1.0 / torch.clamp(target, min=1.0) ** 2
    weight.fill_diagonal_(0.0)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=dev), diagonal=1)

    torch.manual_seed(seed)
    # Init scale matches hyperbolic_mds (also 0.01) to remove a 10× silent
    # asymmetry that was found by the methodology review on 2026-04-25.
    # The Sammon stress is highly init-sensitive when weights = 1/d^2; matched
    # init is required for the §6.3 "Euclidean vs Poincaré" comparison to be
    # methodologically sound.
    # Init on CPU with seeded RNG then move — cross-device determinism.
    points = (torch.randn(n, dim, dtype=torch.float64) * 0.01).to(dev)
    points.requires_grad_(True)

    optimizer = torch.optim.Adam([points], lr=lr)

    for _epoch in range(max_iter):
        optimizer.zero_grad()
        diffs = points.unsqueeze(1) - points.unsqueeze(0)  # (N, N, d)
        dists = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-12)
        residuals = (dists - target) ** 2 * weight
        loss = residuals[mask].sum()
        loss.backward()
        optimizer.step()

    return points.detach().cpu().numpy()
