"""WordNet noun hierarchy loader for Experiment 2.

Loads WordNet 3.0 noun synsets as a directed graph (child → parent via hypernyms)
with metadata for depth, leaves, and ancestor chains.

Requires: nltk with wordnet corpus downloaded.
    python -c "import nltk; nltk.download('wordnet')"
"""

from __future__ import annotations

from collections import deque

import networkx as nx
import numpy as np


def load_wordnet_noun_graph() -> tuple[nx.DiGraph, dict]:
    """Load WordNet noun hierarchy as a directed graph.

    Edges: child → parent (direct hypernym relations only, NOT transitive closure).
    This preserves hop-1 structure needed for poincare_embed().

    Returns:
        Tuple of (dag, metadata) where:
            dag: nx.DiGraph with integer nodes (0..N-1), edges child→parent.
            metadata: dict with keys:
                - synset_names: list[str] of synset names (e.g., "dog.n.01")
                - depths: np.ndarray of max_depth() for each node
                - leaves: np.ndarray of leaf node indices
                - node_to_idx: dict[str, int] mapping synset name → index
                - idx_to_synset: dict[int, str] mapping index → synset name
                - n_nodes: int
                - max_depth: int
    """
    from nltk.corpus import wordnet as wn

    # Collect all noun synsets
    synsets = list(wn.all_synsets(pos="n"))
    synset_names = [s.name() for s in synsets]
    node_to_idx = {name: i for i, name in enumerate(synset_names)}
    idx_to_synset = {i: name for name, i in node_to_idx.items()}

    n = len(synsets)

    # Build directed graph: child → parent (direct hypernyms)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))

    for synset in synsets:
        child_idx = node_to_idx[synset.name()]
        for hypernym in synset.hypernyms():
            parent_idx = node_to_idx[hypernym.name()]
            dag.add_edge(child_idx, parent_idx)

    # Compute depths (max_depth for each synset)
    depths = np.array([s.max_depth() for s in synsets], dtype=np.int32)

    # Find leaf nodes: synsets with no hyponyms (true WordNet leaves)
    true_leaves = []
    for i, synset in enumerate(synsets):
        if not synset.hyponyms():
            true_leaves.append(i)
    leaves = np.array(true_leaves, dtype=np.int32)

    return dag, {
        "synset_names": synset_names,
        "depths": depths,
        "leaves": leaves,
        "node_to_idx": node_to_idx,
        "idx_to_synset": idx_to_synset,
        "n_nodes": n,
        "max_depth": int(depths.max()),
    }


def load_wordnet_subtree(
    max_depth: int = 5,
) -> tuple[nx.DiGraph, dict]:
    """Load a depth-limited subtree of the WordNet noun hierarchy.

    Keeps only synsets with max_depth() <= max_depth, then reindexes
    to contiguous 0..N-1. Much smaller than full graph (~82K → ~3-5K
    for max_depth=5).

    Args:
        max_depth: Maximum synset depth to include (inclusive).

    Returns:
        Same format as load_wordnet_noun_graph().
    """
    from nltk.corpus import wordnet as wn

    # Collect synsets within depth limit
    synsets = [s for s in wn.all_synsets(pos="n") if s.max_depth() <= max_depth]
    synset_names = [s.name() for s in synsets]
    node_to_idx = {name: i for i, name in enumerate(synset_names)}
    idx_to_synset = {i: name for name, i in node_to_idx.items()}
    n = len(synsets)

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))

    for synset in synsets:
        child_idx = node_to_idx[synset.name()]
        for hypernym in synset.hypernyms():
            if hypernym.name() in node_to_idx:
                parent_idx = node_to_idx[hypernym.name()]
                dag.add_edge(child_idx, parent_idx)

    depths = np.array([s.max_depth() for s in synsets], dtype=np.int32)

    true_leaves = []
    for i, synset in enumerate(synsets):
        # A leaf in the subtree: either no hyponyms at all,
        # or all hyponyms are outside the depth limit
        hyponyms_in_tree = [
            h for h in synset.hyponyms() if h.name() in node_to_idx
        ]
        if not hyponyms_in_tree:
            true_leaves.append(i)
    leaves = np.array(true_leaves, dtype=np.int32)

    actual_max = int(depths.max()) if n > 0 else 0

    return dag, {
        "synset_names": synset_names,
        "depths": depths,
        "leaves": leaves,
        "node_to_idx": node_to_idx,
        "idx_to_synset": idx_to_synset,
        "n_nodes": n,
        "max_depth": actual_max,
    }


def wordnet_to_undirected(dag: nx.DiGraph) -> nx.Graph:
    """Convert WordNet DAG to undirected graph for poincare_embed().

    poincare_embed() expects an undirected graph since it uses edge pairs
    symmetrically in the contrastive loss.

    Args:
        dag: Directed graph from load_wordnet_noun_graph().

    Returns:
        Undirected version of the graph.
    """
    return dag.to_undirected()


def get_ancestors(dag: nx.DiGraph, node_idx: int) -> list[int]:
    """Get all ancestors of a node in the WordNet DAG.

    In our child→parent graph, ancestors are reachable via outgoing edges
    (following hypernym chains upward).

    Args:
        dag: Directed graph with child→parent edges.
        node_idx: Index of the query node.

    Returns:
        List of ancestor indices, ordered by distance (nearest first).
    """
    ancestors = []
    visited = set()
    queue = deque([node_idx])
    visited.add(node_idx)

    while queue:
        current = queue.popleft()
        for parent in dag.successors(current):
            if parent not in visited:
                visited.add(parent)
                ancestors.append(parent)
                queue.append(parent)

    return ancestors


def select_stratified_leaves(
    leaf_indices: np.ndarray,
    depths: np.ndarray,
    n: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Select n leaves stratified by depth for balanced coverage.

    Divides leaves into depth bins and samples proportionally from each bin.

    Args:
        leaf_indices: Array of leaf node indices.
        depths: Array of depths for ALL nodes (indexed by node index).
        n: Number of leaves to select.
        seed: Random seed.

    Returns:
        Array of selected leaf indices, shape up to (n,).
        May return fewer than n if insufficient leaves are available.
    """
    rng = np.random.RandomState(seed)
    leaf_depths = depths[leaf_indices]

    # Create depth bins (quantile-based for even distribution)
    n_bins = min(10, len(np.unique(leaf_depths)))
    bin_edges = np.percentile(leaf_depths, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1  # Include max depth

    selected = []
    samples_per_bin = max(1, n // n_bins)

    for i in range(n_bins):
        mask = (leaf_depths >= bin_edges[i]) & (leaf_depths < bin_edges[i + 1])
        bin_leaves = leaf_indices[mask]
        if len(bin_leaves) == 0:
            continue
        k = min(samples_per_bin, len(bin_leaves))
        chosen = rng.choice(bin_leaves, size=k, replace=False)
        selected.extend(chosen.tolist())

    # If we have fewer than n, sample remaining randomly
    remaining = n - len(selected)
    if remaining > 0:
        available = np.setdiff1d(leaf_indices, selected)
        if len(available) > 0:
            extra = rng.choice(available, size=min(remaining, len(available)), replace=False)
            selected.extend(extra.tolist())

    # If we have more than n, trim
    if len(selected) > n:
        selected = rng.choice(selected, size=n, replace=False).tolist()

    return np.array(selected, dtype=np.int32)
