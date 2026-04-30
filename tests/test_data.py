"""Tests for data generation: HSBM and hyperbolic MDS."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
from scipy.stats import spearmanr

from slod.core.poincare import poincare_distance
from slod.utils.data import generate_hsbm, hyperbolic_mds, poincare_embed


class TestGenerateHsbm:
    """Tests for 3-level Hierarchical SBM generator."""

    def test_returns_correct_types(self):
        """Returns (nx.Graph, dict) with macro/meso/micro keys."""
        graph, labels = generate_hsbm(n_nodes=64, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=2)
        assert isinstance(graph, nx.Graph)
        assert isinstance(labels, dict)
        assert set(labels.keys()) == {"macro", "meso", "micro"}

    def test_node_count(self):
        """Graph has exactly n_nodes nodes."""
        graph, _ = generate_hsbm(n_nodes=128, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=4)
        assert graph.number_of_nodes() == 128

    def test_label_shapes(self):
        """Each label array has correct shape and number of unique values."""
        n = 64
        _, labels = generate_hsbm(
            n_nodes=n, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=2
        )
        assert labels["macro"].shape == (n,)
        assert labels["meso"].shape == (n,)
        assert labels["micro"].shape == (n,)
        assert len(np.unique(labels["macro"])) == 2
        assert len(np.unique(labels["meso"])) == 4   # 2*2
        assert len(np.unique(labels["micro"])) == 8   # 2*2*2

    def test_label_consistency(self):
        """Same micro → same meso → same macro (hierarchy is nested)."""
        _, labels = generate_hsbm(
            n_nodes=128, n_macro=2, n_meso_per_macro=4, n_micro_per_meso=2, r=20.0
        )
        # For each pair of nodes sharing same micro label,
        # they must share meso and macro labels
        for micro_id in np.unique(labels["micro"]):
            mask = labels["micro"] == micro_id
            assert len(np.unique(labels["meso"][mask])) == 1
            assert len(np.unique(labels["macro"][mask])) == 1

        # For each pair sharing same meso label, they must share macro
        for meso_id in np.unique(labels["meso"]):
            mask = labels["meso"] == meso_id
            assert len(np.unique(labels["macro"][mask])) == 1

    def test_within_density_higher(self):
        """Within micro-community density exceeds between-macro density."""
        graph, labels = generate_hsbm(
            n_nodes=256, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=4, r=20.0, seed=42
        )
        adj = nx.adjacency_matrix(graph).toarray()
        n = 256

        # Count within-micro edges
        within_edges = 0
        within_pairs = 0
        between_edges = 0
        between_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                if labels["micro"][i] == labels["micro"][j]:
                    within_edges += adj[i, j]
                    within_pairs += 1
                elif labels["macro"][i] != labels["macro"][j]:
                    between_edges += adj[i, j]
                    between_pairs += 1

        within_density = within_edges / max(within_pairs, 1)
        between_density = between_edges / max(between_pairs, 1)

        # Within should be substantially higher than between
        assert within_density > between_density * 2, (
            f"within_density={within_density:.4f}, between_density={between_density:.4f}"
        )

    def test_small_graph(self):
        """Works with minimal hierarchy: 64 nodes, 2*2*2=8 micro."""
        graph, _ = generate_hsbm(
            n_nodes=64, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=2, r=10.0
        )
        assert graph.number_of_nodes() == 64
        assert graph.number_of_edges() > 0

    def test_invalid_nodes_raises(self):
        """n_nodes not divisible by total micro → ValueError."""
        with pytest.raises(ValueError, match="divisible"):
            generate_hsbm(n_nodes=100, n_macro=2, n_meso_per_macro=4, n_micro_per_meso=8)

    def test_negative_params_raises(self):
        """Zero or negative hierarchy params → ValueError."""
        with pytest.raises(ValueError, match="positive"):
            generate_hsbm(n_nodes=64, n_macro=0)
        with pytest.raises(ValueError, match="positive"):
            generate_hsbm(n_nodes=64, n_meso_per_macro=-1)
        with pytest.raises(ValueError, match="positive"):
            generate_hsbm(n_nodes=0)
        with pytest.raises(ValueError, match="positive"):
            generate_hsbm(n_nodes=64, r=-1.0)

    def test_deterministic(self):
        """Same seed produces identical graph."""
        kwargs = dict(n_nodes=64, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=2, seed=123)
        g1, l1 = generate_hsbm(**kwargs)
        g2, l2 = generate_hsbm(**kwargs)
        assert nx.utils.graphs_equal(g1, g2)
        np.testing.assert_array_equal(l1["micro"], l2["micro"])


class TestHyperbolicMds:
    """Tests for Riemannian hyperbolic MDS embedding."""

    @pytest.fixture
    def small_graph(self):
        """Small connected graph for fast tests."""
        return nx.barabasi_albert_graph(20, 3, seed=42)

    def test_output_shape(self, small_graph):
        """Output has shape (n_nodes, dim)."""
        points = hyperbolic_mds(small_graph, dim=5, max_iter=50, seed=42)
        assert points.shape == (20, 5)

    def test_all_points_inside_ball(self, small_graph):
        """All points have ||x|| < 1."""
        points = hyperbolic_mds(small_graph, dim=5, max_iter=50, seed=42)
        norms = np.linalg.norm(points, axis=1)
        assert np.all(norms < 1.0), f"Max norm: {norms.max()}"

    def test_deterministic(self, small_graph):
        """Same seed produces identical embedding."""
        p1 = hyperbolic_mds(small_graph, dim=5, max_iter=50, seed=42)
        p2 = hyperbolic_mds(small_graph, dim=5, max_iter=50, seed=42)
        np.testing.assert_allclose(p1, p2, atol=1e-10)

    def test_small_graph(self):
        """Works on a tiny graph (10 nodes)."""
        g = nx.path_graph(10)
        points = hyperbolic_mds(g, dim=3, max_iter=30, seed=42)
        assert points.shape == (10, 3)
        assert np.all(np.linalg.norm(points, axis=1) < 1.0)

    @pytest.mark.slow
    def test_spearman_on_hsbm(self):
        """Spearman(d_H_embedding, graph_distance) > 0.7 on HSBM r=20.

        This is the quality gate from EMBEDDING_STRATEGY.md.
        Uses 256-node HSBM in B^10 with 3000 iterations (lr=0.2).
        Only considers reachable pairs (largest connected component).
        """
        graph, _ = generate_hsbm(
            n_nodes=256, n_macro=2, n_meso_per_macro=2, n_micro_per_meso=4, r=20.0, seed=42
        )
        # Use largest connected component to avoid infinite distance pairs
        if not nx.is_connected(graph):
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()
            graph = nx.convert_node_labels_to_integers(graph)

        n = graph.number_of_nodes()
        points = hyperbolic_mds(graph, dim=10, max_iter=3000, lr=0.2, seed=42)

        # Compute graph shortest-path distances
        sp_dict = dict(nx.all_pairs_shortest_path_length(graph))
        graph_dists = []
        embed_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                graph_dists.append(sp_dict[i][j])
                embed_dists.append(poincare_distance(points[i], points[j]))

        rho, _ = spearmanr(graph_dists, embed_dists)
        assert rho > 0.7, f"Spearman rho={rho:.3f}, expected > 0.7"


class TestPoincareEmbed:
    """Tests for contrastive Poincare embedding (Nickel & Kiela 2017)."""

    @pytest.fixture
    def small_tree(self):
        """Small binary tree for fast tests."""
        return nx.balanced_tree(r=2, h=3)  # 15 nodes

    def test_output_shape(self, small_tree):
        """Output has shape (n_nodes, dim)."""
        points = poincare_embed(small_tree, dim=5, epochs=10, burn_in=2, seed=42)
        assert points.shape == (15, 5)

    def test_all_points_inside_ball(self, small_tree):
        """All points have ||x|| < 1."""
        points = poincare_embed(small_tree, dim=5, epochs=10, burn_in=2, seed=42)
        norms = np.linalg.norm(points, axis=1)
        assert np.all(norms < 1.0), f"Max norm: {norms.max()}"

    def test_deterministic(self, small_tree):
        """Same seed produces identical embedding."""
        p1 = poincare_embed(small_tree, dim=5, epochs=10, burn_in=2, seed=42)
        p2 = poincare_embed(small_tree, dim=5, epochs=10, burn_in=2, seed=42)
        np.testing.assert_allclose(p1, p2, atol=1e-10)

    @pytest.mark.slow
    def test_spearman_on_tree(self):
        """Spearman > 0.7 on binary tree (ideal for hyperbolic space).

        Contrastive loss excels on trees, which are the natural
        geometry of hyperbolic space (delta-hyperbolicity = 0).
        """
        tree = nx.balanced_tree(r=2, h=4)  # 31 nodes
        n = tree.number_of_nodes()
        points = poincare_embed(tree, dim=5, epochs=200, burn_in=20, seed=42)

        sp_dict = dict(nx.all_pairs_shortest_path_length(tree))
        graph_dists = []
        embed_dists = []
        for i in range(n):
            for j in range(i + 1, n):
                graph_dists.append(sp_dict[i][j])
                embed_dists.append(poincare_distance(points[i], points[j]))

        rho, _ = spearmanr(graph_dists, embed_dists)
        assert rho > 0.7, f"Spearman rho={rho:.3f}, expected > 0.7 on tree"
