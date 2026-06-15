import numpy as np
import scipy.sparse as sp
import umap


def as_float32_dense(x):
    if sp.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def random_sketch_indices(n_obs, sketch_n, seed=42):
    rng = np.random.default_rng(seed)
    sketch_n = min(int(sketch_n), int(n_obs))
    return rng.choice(n_obs, size=sketch_n, replace=False).astype(np.int64)


def balanced_sketch_indices(obs, n_obs, sketch_n, seed=42):
    """
    Simple balanced sketch by FCS_File x metaclustering.
    Falls back to random if those columns are missing.
    """
    if ("FCS_File" not in obs.columns) or ("metaclustering" not in obs.columns):
        return random_sketch_indices(n_obs, sketch_n, seed=seed)

    rng = np.random.default_rng(seed)
    sketch_n = min(int(sketch_n), int(n_obs))

    groups = obs.groupby(["FCS_File", "metaclustering"], sort=False).indices
    if len(groups) == 0:
        return random_sketch_indices(n_obs, sketch_n, seed=seed)

    per_group = max(1, sketch_n // len(groups))
    picks = []

    for idx in groups.values():
        idx = np.asarray(idx, dtype=np.int64)
        k = min(per_group, idx.size)
        if k > 0:
            picks.append(rng.choice(idx, size=k, replace=False))

    sketch_idx = np.concatenate(picks) if len(picks) else np.empty(0, dtype=np.int64)

    # Top up to target (simple top-up)
    if sketch_idx.size < sketch_n:
        need = sketch_n - sketch_idx.size
        extra = rng.choice(n_obs, size=need, replace=False).astype(np.int64)
        sketch_idx = np.unique(np.concatenate([sketch_idx, extra]))

    # Trim if too large
    if sketch_idx.size > sketch_n:
        sketch_idx = rng.choice(sketch_idx, size=sketch_n, replace=False)

    rng.shuffle(sketch_idx)
    return sketch_idx.astype(np.int64)


def fit_on_sketch_transform_all_in_memory(
    fsom,
    sketch_n=7_500_000,
    chunk_size=500_000,
    seed=42,
    balanced=True,
    n_neighbors=30,
    min_dist=0.05,
    spread=1.0,
):
    """
    Returns full UMAP coordinates for all cells in memory.
    """
    # Handle both FlowSOM and AnnData inputs
    if hasattr(fsom, "get_cell_data"):
        ad = fsom.get_cell_data()
    else:
        # Already AnnData
        ad = fsom
    n_obs = ad.n_obs

    # Use only clustering markers if available
    if "cols_used" in ad.var.columns:
        marker_mask = ad.var["cols_used"].to_numpy().astype(bool)
        X = ad.X[:, marker_mask]
    else:
        X = ad.X

    if balanced:
        sketch_idx = balanced_sketch_indices(ad.obs, n_obs, sketch_n, seed=seed)
    else:
        sketch_idx = random_sketch_indices(n_obs, sketch_n, seed=seed)

    X_sketch = as_float32_dense(X[sketch_idx, :])

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        transform_seed=seed,
        low_memory=True,
        n_jobs=-1,
        spread=spread,
    )
    reducer.fit(X_sketch)

    # Full coordinates for all cells
    umap_coords = np.empty((n_obs, 2), dtype=np.float32)

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        X_chunk = as_float32_dense(X[start:end, :])
        umap_coords[start:end, :] = reducer.transform(X_chunk).astype(np.float32)
        print(f"Transformed {start:,} to {end:,}")

    # Attach to AnnData in memory
    ad.obsm["X_umap"] = umap_coords
    return ad, reducer, sketch_idx


# ---------------------------
# Example usage
# ---------------------------
# ad, reducer, sketch_idx = fit_on_sketch_transform_all_in_memory(
#     fsom,
#     sketch_n=2_000_000,
#     chunk_size=500_000,
#     seed=42,
#     balanced=True,
# )
# print(ad.obsm["X_umap"].shape)  # (all_cells, 2)
