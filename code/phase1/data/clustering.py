"""
phase1/data/clustering.py

Downloads the Bitext customer-support dataset, filters to IT-proxy categories,
embeds ticket text with sentence-transformers, reduces dimensions with UMAP,
clusters with HDBSCAN, and saves a ticket-level CSV with cluster assignments.

Checkpoints:
  - embeddings.pkl       : raw sentence-transformer embeddings
  - umap_projection.pkl  : UMAP-reduced coordinates
  - clustering_results.csv: final ticket-level CSV (skips all above steps if present)
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public entry point ────────────────────────────────────────────────────────

def run_clustering(cfg: dict) -> pd.DataFrame:
    """
    Full clustering pipeline.

    Parameters
    ----------
    cfg : dict
        Loaded and path-resolved phase1_config.yaml.

    Returns
    -------
    pd.DataFrame
        Ticket-level DataFrame with columns:
        ticket_id, ticket_details, cluster_id,
        ticket_rank_in_cluster, distance_to_centroid.
        Noise points (HDBSCAN label -1) are excluded.
    """
    from phase1.data.schema import (
        TICKET_ID, TICKET_DETAILS, CLUSTER_ID,
        TICKET_RANK, DISTANCE_TO_CENTROID,
        FILE_CLUSTERED_CSV,
    )

    paths      = cfg["paths"]
    ckpt_cfg   = cfg["checkpoints"]
    ds_cfg     = cfg["dataset"]
    clust_cfg  = cfg["clustering"]
    k          = cfg["top_k"]
    use_ckpt   = ckpt_cfg["use_checkpoints"]

    ckpt_dir   = Path(paths["checkpoints"])
    out_dir    = Path(paths["data_processed"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    clustering_csv = out_dir / FILE_CLUSTERED_CSV
    embeddings_pkl = ckpt_dir / ckpt_cfg["embeddings_file"].split("/")[-1]
    umap_pkl       = ckpt_dir / ckpt_cfg["umap_file"].split("/")[-1]

    # ── Fast path: clustering CSV already exists ──────────────────────────────
    if use_ckpt and clustering_csv.exists():
        logger.info(f"[clustering] Checkpoint found: {clustering_csv}. Skipping.")
        return pd.read_csv(clustering_csv)

    # ── Step 1: Load and filter Bitext dataset ────────────────────────────────
    tickets_df = _load_bitext(ds_cfg)

    texts = tickets_df["text"].tolist()
    logger.info(f"[clustering] {len(texts)} tickets loaded.")

    # ── Step 2: Embed with sentence-transformers ──────────────────────────────
    if use_ckpt and embeddings_pkl.exists():
        logger.info(f"[clustering] Loading embeddings from checkpoint: {embeddings_pkl}")
        with open(embeddings_pkl, "rb") as f:
            embeddings = pickle.load(f)
    else:
        embeddings = _embed_texts(texts, clust_cfg["embedding_model"])
        with open(embeddings_pkl, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"[clustering] Embeddings saved to {embeddings_pkl}")

    # ── Step 3: UMAP dimensionality reduction ─────────────────────────────────
    if use_ckpt and umap_pkl.exists():
        logger.info(f"[clustering] Loading UMAP projection from checkpoint: {umap_pkl}")
        with open(umap_pkl, "rb") as f:
            umap_data = pickle.load(f)
        umap_embeddings = umap_data["projection"]
    else:
        umap_embeddings = _run_umap(embeddings, clust_cfg["umap"])
        umap_data = {"projection": umap_embeddings}
        with open(umap_pkl, "wb") as f:
            pickle.dump(umap_data, f)
        logger.info(f"[clustering] UMAP projection saved to {umap_pkl}")

    # ── Step 4: HDBSCAN clustering ────────────────────────────────────────────
    labels = _run_hdbscan(umap_embeddings, clust_cfg["hdbscan"])

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    logger.info(
        f"[clustering] HDBSCAN found {n_clusters} clusters. "
        f"Noise points discarded: {n_noise}"
    )
    if n_clusters < 5:
        logger.warning(
            "[clustering] Fewer than 5 clusters found. "
            "Try lowering min_cluster_size in the config."
        )

    # ── Step 5: Compute centroid distances and ranks ───────────────────────────
    result_df = _build_result_df(tickets_df, umap_embeddings, labels)

    # ── Save ──────────────────────────────────────────────────────────────────
    result_df.to_csv(clustering_csv, index=False)
    logger.info(
        f"[clustering] Saved {len(result_df)} tickets across {n_clusters} clusters "
        f"to {clustering_csv}"
    )
    return result_df


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_bitext(ds_cfg: dict) -> pd.DataFrame:
    """
    Download the Bitext dataset from HuggingFace and return a filtered DataFrame
    with columns: text (ticket text), original_category.

    Filters to the categories in ds_cfg['it_categories'] and samples
    ds_cfg['n_samples'] rows.
    """
    from datasets import load_dataset

    logger.info(f"[clustering] Downloading dataset: {ds_cfg['name']}")
    raw = load_dataset(ds_cfg["name"], split="train")
    df  = raw.to_pandas()

    logger.info(f"[clustering] Raw dataset: {len(df)} rows")
    logger.info(f"[clustering] Available categories: {sorted(df['category'].unique().tolist())}")

    # Filter to configured IT-proxy categories
    target_cats = [c.upper() for c in ds_cfg["it_categories"]]
    filtered = df[df["category"].str.upper().isin(target_cats)].copy()

    if len(filtered) == 0:
        logger.warning(
            f"[clustering] No rows matched categories {target_cats}. "
            "Using all categories as fallback."
        )
        filtered = df.copy()

    # Sample
    n = min(ds_cfg["n_samples"], len(filtered))
    sampled = filtered.sample(n=n, random_state=42).reset_index(drop=True)
    logger.info(
        f"[clustering] After filter + sample: {len(sampled)} rows "
        f"from categories {sampled['category'].unique().tolist()}"
    )

    # Standardise output
    text_col = ds_cfg.get("text_column", "instruction")
    sampled["text"] = sampled[text_col].astype(str).str.strip()
    sampled["original_category"] = sampled["category"].astype(str)

    return sampled[["text", "original_category"]].copy()


def _embed_texts(texts: list[str], model_name: str) -> np.ndarray:
    """Embed a list of texts using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"[clustering] Embedding {len(texts)} texts with {model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    logger.info(f"[clustering] Embeddings shape: {embeddings.shape}")
    return embeddings


def _run_umap(embeddings: np.ndarray, umap_cfg: dict) -> np.ndarray:
    """Reduce embedding dimensions with UMAP."""
    import umap

    logger.info(
        f"[clustering] Running UMAP: {embeddings.shape[1]}D → "
        f"{umap_cfg['n_components']}D ..."
    )
    reducer = umap.UMAP(
        n_components=umap_cfg["n_components"],
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg["random_state"],
        verbose=False,
    )
    projected = reducer.fit_transform(embeddings)
    logger.info(f"[clustering] UMAP projection shape: {projected.shape}")
    return projected


def _run_hdbscan(umap_embeddings: np.ndarray, hdbscan_cfg: dict) -> np.ndarray:
    """Cluster UMAP-projected embeddings with HDBSCAN."""
    import hdbscan

    logger.info(
        f"[clustering] Running HDBSCAN "
        f"(min_cluster_size={hdbscan_cfg['min_cluster_size']}) ..."
    )
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdbscan_cfg["min_cluster_size"],
        min_samples=hdbscan_cfg["min_samples"],
        metric=hdbscan_cfg["metric"],
        cluster_selection_method=hdbscan_cfg["cluster_selection_method"],
    )
    labels = clusterer.fit_predict(umap_embeddings)
    return labels


def _build_result_df(
    tickets_df: pd.DataFrame,
    umap_embeddings: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Build the final ticket-level DataFrame with cluster assignments,
    centroid distances, and within-cluster ranks.

    Filters out noise points (label == -1).
    """
    from phase1.data.schema import (
        TICKET_ID, TICKET_DETAILS, CLUSTER_ID,
        TICKET_RANK, DISTANCE_TO_CENTROID,
    )

    df = pd.DataFrame({
        TICKET_ID:           range(len(tickets_df)),
        TICKET_DETAILS:      tickets_df["text"].values,
        CLUSTER_ID:          labels,
        "umap_embedding":    list(umap_embeddings),
    })

    # Remove noise points
    df = df[df[CLUSTER_ID] != -1].copy()
    df[CLUSTER_ID] = df[CLUSTER_ID].astype(int)

    # Compute centroid distances
    distances = []
    for _, row in df.iterrows():
        cluster_mask   = df[CLUSTER_ID] == row[CLUSTER_ID]
        cluster_vecs   = np.array(df.loc[cluster_mask, "umap_embedding"].tolist())
        centroid       = cluster_vecs.mean(axis=0)
        dist           = float(np.linalg.norm(row["umap_embedding"] - centroid))
        distances.append(dist)

    df[DISTANCE_TO_CENTROID] = distances

    # Assign within-cluster rank (rank 1 = closest to centroid)
    df[TICKET_RANK] = (
        df.groupby(CLUSTER_ID)[DISTANCE_TO_CENTROID]
        .rank(method="first", ascending=True)
        .astype(int)
    )

    df = df.drop(columns=["umap_embedding"])
    df = df.sort_values([CLUSTER_ID, TICKET_RANK]).reset_index(drop=True)

    return df[[TICKET_ID, TICKET_DETAILS, CLUSTER_ID, TICKET_RANK, DISTANCE_TO_CENTROID]]
