"""Synthetic control graph generators for BPU experiments.

Each generator produces a directed graph as a scipy CSR sparse matrix
with weights normalized to [0, 1]. These serve as controlled baselines
to test whether biological wiring topology confers computational
advantages over matched random topologies.
"""

import numpy as np
import scipy.sparse as sp


def _normalize_weights(adj_sparse):
    """Normalize sparse matrix values to [0, 1] range.

    If the matrix has no nonzero entries or all entries are equal,
    returns the matrix unchanged (or with all ones).
    """
    if adj_sparse.nnz == 0:
        return adj_sparse
    data = adj_sparse.data.copy().astype(np.float64)
    vmin, vmax = data.min(), data.max()
    if vmax - vmin < 1e-12:
        data[:] = 1.0
    else:
        data = (data - vmin) / (vmax - vmin)
    result = adj_sparse.copy().astype(np.float64)
    result.data = data
    return result


def _ensure_csr(adj):
    """Ensure the adjacency matrix is in CSR format."""
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    return adj.tocsr()


def _match_edge_count(adj_sparse, target_edges, rng):
    """Match an off-diagonal directed graph to an exact edge count."""
    adj = _ensure_csr(adj_sparse)
    n = adj.shape[0]
    max_edges = n * (n - 1)
    if target_edges < 0 or target_edges > max_edges:
        raise ValueError(
            f"target_edges must be between 0 and {max_edges}, got {target_edges}"
        )

    coo = adj.tocoo()
    edge_weights = {
        (int(row), int(col)): float(weight)
        for row, col, weight in zip(coo.row, coo.col, coo.data)
        if row != col
    }

    if len(edge_weights) > target_edges:
        edges = list(edge_weights)
        remove_count = len(edge_weights) - target_edges
        for index in rng.choice(len(edges), size=remove_count, replace=False):
            edge_weights.pop(edges[int(index)])

    while len(edge_weights) < target_edges:
        remaining = target_edges - len(edge_weights)
        batch_size = max(1024, remaining * 2)
        rows = rng.randint(0, n, size=batch_size)
        cols = rng.randint(0, n, size=batch_size)
        weights = rng.uniform(0.0, 1.0, size=batch_size)
        for row, col, weight in zip(rows, cols, weights):
            edge = (int(row), int(col))
            if row != col and edge not in edge_weights:
                edge_weights[edge] = float(weight)
                if len(edge_weights) == target_edges:
                    break

    if not edge_weights:
        return sp.csr_matrix((n, n), dtype=np.float64)

    edges = list(edge_weights)
    rows = np.fromiter((edge[0] for edge in edges), dtype=np.int64)
    cols = np.fromiter((edge[1] for edge in edges), dtype=np.int64)
    weights = np.fromiter(
        (edge_weights[edge] for edge in edges), dtype=np.float64
    )
    return sp.csr_matrix((weights, (rows, cols)), shape=(n, n))


def _assign_weight_distribution(adj_sparse, source_weights, rng):
    """Assign an exact shuffled source-weight multiset to a graph topology."""
    coo = _ensure_csr(adj_sparse).tocoo()
    weights = np.asarray(source_weights, dtype=np.float64).copy()
    if coo.nnz != len(weights):
        raise ValueError(
            f"topology has {coo.nnz} edges but weight source has {len(weights)}"
        )
    rng.shuffle(weights)
    return sp.csr_matrix((weights, (coo.row, coo.col)), shape=coo.shape)


def erdos_renyi(N, density, seed=42):
    """Generate a directed Erdos-Renyi random graph.

    Each possible directed edge (i -> j, i != j) is included independently
    with probability = density. Edge weights are uniform random in (0, 1].

    Args:
        N: Number of nodes.
        density: Edge probability (expected density of adjacency matrix).
        seed: Random seed for reproducibility.

    Returns:
        scipy CSR sparse matrix (N, N), weights in [0, 1].
    """
    rng = np.random.RandomState(seed)
    # Number of possible directed edges (excluding self-loops)
    n_possible = N * (N - 1)
    n_edges = int(round(density * n_possible))

    # Sample edge indices without replacement
    rows = []
    cols = []
    # Efficient: sample flat indices, convert to (row, col) excluding diagonal
    flat_indices = rng.choice(n_possible, size=n_edges, replace=False)
    for idx in flat_indices:
        r = idx // (N - 1)
        c = idx % (N - 1)
        if c >= r:
            c += 1  # skip diagonal
        rows.append(r)
        cols.append(c)

    weights = rng.uniform(0.0, 1.0, size=n_edges)
    adj = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    return _normalize_weights(adj)


def barabasi_albert(N, density=None, seed=42):
    """Generate a directed Barabasi-Albert scale-free graph.

    Uses preferential attachment to grow the graph from an initial
    clique. The resulting undirected edges are randomly oriented to
    produce a directed graph.

    Args:
        N: Number of nodes.
        density: Target density. If None, uses m = max(1, int(sqrt(N)))
            for the BA parameter (edges added per new node).
        seed: Random seed.

    Returns:
        scipy CSR sparse matrix (N, N), weights in [0, 1].
    """
    rng = np.random.RandomState(seed)

    if density is not None:
        target_edges = int(round(density * N * (N - 1)))
        # This implementation creates approximately m*N directed edges.
        m = max(1, int(round(target_edges / N)))
        m = min(m, N - 1)
    else:
        m = max(1, int(np.sqrt(N)))
        m = min(m, N - 1)

    # Start with a fully connected clique of m+1 nodes
    adj_lil = sp.lil_matrix((N, N), dtype=np.float64)
    initial_nodes = list(range(min(m + 1, N)))
    for i in initial_nodes:
        for j in initial_nodes:
            if i != j:
                adj_lil[i, j] = rng.uniform(0.0, 1.0)

    # Degree array for preferential attachment
    degree = np.zeros(N)
    for node in initial_nodes:
        degree[node] = len(initial_nodes) - 1

    # Grow graph by preferential attachment
    for new_node in range(len(initial_nodes), N):
        if degree[:new_node].sum() == 0:
            # Uniform attachment if no edges yet
            probs = np.ones(new_node) / new_node
        else:
            probs = degree[:new_node] / degree[:new_node].sum()

        targets = rng.choice(
            new_node, size=min(m, new_node), replace=False, p=probs
        )

        for t in targets:
            w = rng.uniform(0.0, 1.0)
            # Randomly orient the edge
            if rng.rand() < 0.5:
                adj_lil[new_node, t] = w
            else:
                adj_lil[t, new_node] = w
            degree[new_node] += 1
            degree[t] += 1

    result = adj_lil.tocsr()
    if density is not None:
        result = _match_edge_count(result, target_edges, rng)
    return _normalize_weights(result)


def watts_strogatz(N, density=None, p=0.1, seed=42):
    """Generate a directed Watts-Strogatz small-world graph.

    Starts with a ring lattice where each node connects to its K nearest
    neighbors, then rewires each edge with probability p. The undirected
    edges are randomly oriented to produce a directed graph.

    Args:
        N: Number of nodes.
        density: Target density. If None, K = max(2, int(sqrt(N))).
        p: Rewiring probability. Default 0.1.
        seed: Random seed.

    Returns:
        scipy CSR sparse matrix (N, N), weights in [0, 1].
    """
    rng = np.random.RandomState(seed)

    if density is not None:
        target_edges = int(round(density * N * (N - 1)))
        # The ring contributes N*K edges before random orientation.
        K = max(1, int(round(target_edges / N)))
    else:
        K = max(1, int(np.sqrt(N)) // 2)

    K = min(K, (N - 1) // 2)  # Can't exceed half the ring
    if K < 1:
        K = 1

    # Build ring lattice (undirected edge list)
    edges = set()
    for i in range(N):
        for j in range(1, K + 1):
            neighbor = (i + j) % N
            edges.add((min(i, neighbor), max(i, neighbor)))

    # Rewire
    edges_list = list(edges)
    new_edges = set()
    for u, v in edges_list:
        if rng.rand() < p:
            # Rewire v to a random node
            candidates = [
                c for c in range(N) if c != u
                and (min(u, c), max(u, c)) not in new_edges
                and (min(u, c), max(u, c)) not in edges
            ]
            if candidates:
                w = rng.choice(candidates)
                new_edges.add((min(u, w), max(u, w)))
            else:
                new_edges.add((u, v))  # Keep original
        else:
            new_edges.add((u, v))

    # Orient edges randomly to make directed
    adj_lil = sp.lil_matrix((N, N), dtype=np.float64)
    for u, v in new_edges:
        w = rng.uniform(0.0, 1.0)
        if rng.rand() < 0.5:
            adj_lil[u, v] = w
        else:
            adj_lil[v, u] = w

    result = adj_lil.tocsr()
    if density is not None:
        result = _match_edge_count(result, target_edges, rng)
    return _normalize_weights(result)


def degree_preserved_shuffle(adj_sparse, seed=42):
    """Rewire edges while preserving in- and out-degree sequences.

    Uses the edge-swap (Maslov-Sneppen) algorithm: repeatedly pick two
    edges (a->b, c->d) and swap to (a->d, c->b), rejecting swaps that
    create self-loops or multi-edges. Performs 10 * nnz swaps.

    Args:
        adj_sparse: scipy sparse matrix (N, N) to shuffle.
        seed: Random seed.

    Returns:
        scipy CSR sparse matrix (N, N), same density and degree sequence,
        weights in [0, 1].
    """
    rng = np.random.RandomState(seed)
    adj = _ensure_csr(adj_sparse).copy()
    coo = adj.tocoo()
    rows = coo.row.copy()
    cols = coo.col.copy()
    weights = coo.data.copy()
    n_edges = len(rows)
    N = adj.shape[0]

    if n_edges < 2:
        return _normalize_weights(adj)

    # Build edge set for O(1) lookup
    edge_set = set(zip(rows.tolist(), cols.tolist()))
    n_swaps = 10 * n_edges

    for _ in range(n_swaps):
        # Pick two random edges
        i1 = rng.randint(n_edges)
        i2 = rng.randint(n_edges)
        if i1 == i2:
            continue

        a, b = rows[i1], cols[i1]
        c, d = rows[i2], cols[i2]

        # Proposed swap: a->d, c->b
        if a == d or c == b:  # self-loop
            continue
        if (a, d) in edge_set or (c, b) in edge_set:  # multi-edge
            continue

        # Perform swap
        edge_set.discard((a, b))
        edge_set.discard((c, d))
        edge_set.add((a, d))
        edge_set.add((c, b))

        cols[i1] = d
        cols[i2] = b

    result = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))
    return _normalize_weights(result)


def generate_all_controls(bio_adj, seed=42):
    """Generate all four control graphs matched to a biological adjacency.

    Computes the density from the biological graph and generates:
    1. Erdos-Renyi random graph (same density)
    2. Barabasi-Albert scale-free graph (matched density)
    3. Watts-Strogatz small-world graph (matched density)
    4. Degree-preserved shuffle of the biological graph

    Args:
        bio_adj: scipy sparse matrix or numpy ndarray (N, N).
        seed: Random seed.

    Returns:
        dict with keys 'erdos_renyi', 'barabasi_albert', 'watts_strogatz',
        'degree_preserved', each mapping to a scipy CSR sparse matrix.
    """
    bio_adj = _ensure_csr(bio_adj)
    bio_adj.eliminate_zeros()
    N = bio_adj.shape[0]
    n_possible = N * (N - 1) if N > 1 else 1
    density = bio_adj.nnz / n_possible

    topologies = {
        "erdos_renyi": erdos_renyi(N, density, seed=seed),
        "barabasi_albert": barabasi_albert(N, density=density, seed=seed + 1),
        "watts_strogatz": watts_strogatz(N, density=density, seed=seed + 2),
        "degree_preserved": degree_preserved_shuffle(bio_adj, seed=seed + 3),
    }
    return {
        name: _assign_weight_distribution(
            adjacency,
            bio_adj.data,
            np.random.RandomState(seed + 100 + index),
        )
        for index, (name, adjacency) in enumerate(topologies.items())
    }
