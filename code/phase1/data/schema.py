"""
phase1/data/schema.py
Single source of truth for all CSV column names and output field names.
Import these constants everywhere — never hardcode column name strings.
"""


# ── Raw ticket-level CSV columns ──────────────────────────────────────────────
TICKET_ID            = "ticket_id"
TICKET_DETAILS       = "ticket_details"
CLUSTER_ID           = "cluster_id"
CLUSTER_SIZE         = "cluster_size"
TICKET_RANK          = "ticket_rank_in_cluster"
DISTANCE_TO_CENTROID = "distance_to_centroid"
DOMAIN               = "domain"

# ── Grouped cluster-level CSV columns (intermediate, used for LLM input) ─────
def top_k_ids_col(k: int) -> str:
    """Column holding a JSON-encoded list of top-k ticket IDs per cluster."""
    return f"top_{k}_ticket_ids"


def top_k_details_col(k: int) -> str:
    """Column holding a JSON-encoded list of top-k ticket texts per cluster."""
    return f"top_{k}_ticket_details"


# ── Labeled CSV columns (final output, ticket-level) ─────────────────────────
def cluster_name_col(model_id: str, prompt_id: str) -> str:
    """
    Generate the column name for a teacher-generated cluster label.

    Example:
        cluster_name_col("claude-haiku-4-5", "P1")
        → "cluster_name_claude_haiku_4_5_P1"

    model_id and prompt_id are sanitised so the column name is safe for
    pandas, CSV, and most downstream tools.
    """
    safe_model = (
        model_id
        .replace("/", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(" ", "_")
    )
    return f"cluster_name_{safe_model}_{prompt_id}"


def get_all_cluster_name_cols(model_id: str, prompt_ids: list[str]) -> list[str]:
    """Return all label column names for a given model and list of prompt IDs."""
    return [cluster_name_col(model_id, pid) for pid in prompt_ids]


# ── Prediction output fields (JSONL records) ──────────────────────────────────
PRED_CLUSTER_ID      = "cluster_id"
PRED_PROMPT_ID       = "prompt_id"
PRED_MODEL           = "model"
PRED_FINE_TUNED      = "fine_tuned"
PRED_GENERATED_LABEL = "generated_label"

# ── Evaluation result fields ──────────────────────────────────────────────────
EVAL_COSINE_SAME_PROMPT  = "cosine_sim_same_prompt"
EVAL_COSINE_MULTI_REF    = "cosine_sim_multi_ref"
EVAL_BERTSCORE_SAME      = "bertscore_f1_same_prompt"
EVAL_BERTSCORE_MULTI     = "bertscore_f1_multi_ref"
EVAL_ROUGEL_SAME         = "rougeL_same_prompt"
EVAL_ROUGEL_MULTI        = "rougeL_multi_ref"

# ── Business eval fields ──────────────────────────────────────────────────────
BIZ_LABEL_LATENCY_S      = "label_generation_latency_s"
BIZ_INFERENCE_LATENCY_S  = "inference_latency_s"
BIZ_FINETUNING_TIME_S    = "finetuning_time_s"

# ── File names (relative to configured output dirs) ───────────────────────────
FILE_CLUSTERED_CSV        = "bitext_clustered.csv"
FILE_GROUPED_CSV          = "bitext_grouped.csv"
FILE_LABELED_CSV          = "bitext_labeled.csv"
FILE_LABELED_COPY         = "labeled_data.csv"        # copy in run dir for self-contained reruns
FILE_CLUSTER_SPLITS       = "cluster_splits.json"      # {cluster_id: "train"/"val"/"test"}
FILE_TRAIN_JSONL          = "train.jsonl"
FILE_VAL_JSONL            = "val.jsonl"
FILE_TEST_JSONL           = "test.jsonl"
FILE_TEACHER_PREDS        = "teacher_predictions.jsonl"
FILE_BASELINE_PREDS       = "baseline_predictions.jsonl"
FILE_FINETUNED_PREDS      = "finetuned_predictions.jsonl"

# ── Per-model evaluation files (one per model, written independently) ─────────
# tag is one of: "teacher", "baseline", "finetuned"
def nonllm_file(tag: str) -> str:
    """Non-LLM metrics file for a model.  Columns: cluster_id, prompt_id, model, generated_label, nonllm_metrics (JSON)."""
    return f"nonllm_{tag}.csv"

def llm_file(tag: str) -> str:
    """LLM judge metrics file for a model.  Columns: cluster_id, prompt_id, model, generated_label, reference_label, llm_metrics (JSON)."""
    return f"llm_{tag}.csv"

def latency_file(tag: str) -> str:
    """Per-model inference latency.  Columns: cluster_id, prompt_id, latency_s."""
    return f"latency_{tag}.csv"

# ── Teacher latency (in data/processed/, shared across runs) ──────────────────
FILE_TEACHER_LATENCY      = "teacher_latency_records.csv"

# ── Combined / pivot files (written by combine.py, whenever per-model files exist) ─
FILE_LATENCY_PIVOT        = "latency_by_cluster.csv"     # wide pivot: one row per cluster
FILE_NONLLM_PIVOT         = "nonllm_by_cluster.csv"       # per (cluster, prompt): non-LLM metrics
FILE_LLM_PIVOT            = "llm_by_cluster.csv"           # per (cluster, prompt): judge scores
FILE_LABELS_PIVOT         = "labels_by_cluster.csv"        # per (cluster, prompt): actual label text
FILE_METRICS_SUMMARY      = "metrics_summary.csv"           # mean non-LLM metrics per (split, model)
FILE_JUDGE_SUMMARY        = "judge_summary.csv"             # mean LLM judge scores per (split, model)
FILE_BUSINESS_EVAL        = "business_eval.csv"

# ── Prompt IDs ────────────────────────────────────────────────────────────────
PROMPT_IDS = ["P1", "P2", "P3", "P4", "P5"]
