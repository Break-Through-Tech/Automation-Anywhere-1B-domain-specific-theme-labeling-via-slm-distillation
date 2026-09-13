"""
phase1/data/preprocessing.py

Groups the ticket-level clustered CSV by cluster_id and selects the top-k
most representative tickets per cluster (lowest distance to centroid).

Input:  ticket-level CSV (output of clustering.py)
Output:
  - grouped cluster-level CSV (intermediate, used by labeling step)
  - the same ticket-level CSV is passed through unchanged for the labeled output
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def run_preprocessing(cfg: dict, clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group tickets by cluster and select top-k.

    Parameters
    ----------
    cfg : dict
        Loaded and path-resolved phase1_config.yaml.
    clustered_df : pd.DataFrame
        Output of clustering.py — ticket-level with cluster assignments.

    Returns
    -------
    pd.DataFrame
        Cluster-level DataFrame with one row per cluster, containing:
        cluster_id, cluster_size, domain,
        top_{k}_ticket_ids (JSON string),
        top_{k}_ticket_details (JSON string).
    """
    from phase1.data.schema import (
        TICKET_ID, TICKET_DETAILS, CLUSTER_ID, CLUSTER_SIZE,
        TICKET_RANK, DISTANCE_TO_CENTROID, DOMAIN,
        FILE_GROUPED_CSV,
        top_k_ids_col, top_k_details_col,
    )

    k        = cfg["top_k"]
    domain   = cfg["dataset"]["domain"]
    out_dir  = Path(cfg["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / FILE_GROUPED_CSV

    logger.info(
        f"[preprocessing] Grouping {len(clustered_df)} tickets into clusters "
        f"with top-k={k} ..."
    )

    # Filter to top-k tickets per cluster (already ranked in clustering.py)
    top_k_df = clustered_df[clustered_df[TICKET_RANK] <= k].copy()

    # Compute cluster sizes from the full clustered_df
    cluster_sizes = (
        clustered_df.groupby(CLUSTER_ID)
        .size()
        .rename(CLUSTER_SIZE)
        .reset_index()
    )

    # Group top-k tickets by cluster
    grouped = (
        top_k_df
        .sort_values([CLUSTER_ID, TICKET_RANK])
        .groupby(CLUSTER_ID)
        .agg(
            **{
                top_k_ids_col(k):     (TICKET_ID,     lambda x: json.dumps(x.tolist())),
                top_k_details_col(k): (TICKET_DETAILS, lambda x: json.dumps(x.tolist())),
            }
        )
        .reset_index()
    )

    # Merge cluster sizes
    grouped = grouped.merge(cluster_sizes, on=CLUSTER_ID, how="left")

    # Flag clusters with fewer than k tickets (context will be shorter)
    small_clusters = grouped[
        grouped[top_k_ids_col(k)].apply(lambda x: len(json.loads(x))) < k
    ]
    if len(small_clusters) > 0:
        logger.warning(
            f"[preprocessing] {len(small_clusters)} clusters have fewer than "
            f"k={k} tickets. They will use all available tickets."
        )

    grouped[DOMAIN] = domain

    # Reorder columns
    cols = [CLUSTER_ID, CLUSTER_SIZE, DOMAIN, top_k_ids_col(k), top_k_details_col(k)]
    grouped = grouped[cols].sort_values(CLUSTER_ID).reset_index(drop=True)

    grouped.to_csv(out_path, index=False)

    logger.info(
        f"[preprocessing] {len(grouped)} clusters saved to {out_path}\n"
        f"  Cluster size range: "
        f"{grouped[CLUSTER_SIZE].min()}–{grouped[CLUSTER_SIZE].max()}\n"
        f"  Top-k={k} tickets per cluster sent to LLM."
    )

    _log_sample(grouped, k)
    return grouped


def _log_sample(grouped_df: pd.DataFrame, k: int) -> None:
    """Log a sample cluster for visual sanity-check."""
    from phase1.data.schema import CLUSTER_ID, top_k_details_col

    sample = grouped_df.sample(1, random_state=0).iloc[0]
    tickets = json.loads(sample[top_k_details_col(k)])
    logger.info(
        f"\n[preprocessing] Sample cluster {sample[CLUSTER_ID]}:\n"
        + "\n".join(f"  Ticket {i+1}: {t[:120]}..." for i, t in enumerate(tickets[:3]))
    )
