"""
phase1/evaluation/business_eval.py

Tracks and reports business-relevant performance metrics:
  - Label generation latency (per cluster, per prompt, via frontier LLM)
  - SLM inference latency (per cluster, per prompt)
  - Total fine-tuning wall-clock time
  - Estimated cost comparison (teacher LLM API vs SLM inference)

Usage
-----
business_eval = BusinessEvaluator(cfg)

# In labeling step:
business_eval.record_label_latency(cluster_id, prompt_id, elapsed_s)

# In fine-tuning step:
business_eval.record_finetuning_time(elapsed_s)

# In inference step:
business_eval.record_inference_latency(cluster_id, prompt_id, elapsed_s)

# At the end of the pipeline:
business_eval.save(output_path)
business_eval.log_summary()
"""

import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Approximate cost constants (USD per 1M tokens as of 2026)
# Update these if pricing changes.
COST_PER_1M_TOKENS = {
    "claude-haiku-4-5":   0.25,    # Anthropic Haiku
    "gpt-4o-mini":         0.15,    # OpenAI GPT-4o-mini
    "gpt-5.1":            10.00,    # placeholder for Phase 2
}
# Typical tokens per label generation call
AVG_INPUT_TOKENS_PER_CALL  = 500   # system prompt + k tickets
AVG_OUTPUT_TOKENS_PER_CALL = 20    # label is 5–15 words


class BusinessEvaluator:
    """Collects and reports business metrics throughout the pipeline."""

    def __init__(self, cfg: dict) -> None:
        self.cfg       = cfg
        self.model_id  = cfg["teacher_llm"]["model"]
        self.slm_id    = cfg["student_slm"]["model_id"]

        self._label_records:      list[dict] = []
        self._inference_records:  list[dict] = []
        self._finetuning_time_s:  float | None = None
        self._model_load_times:   dict[str, float] = {}   # tag → seconds

    # ── Recording methods ─────────────────────────────────────────────────────

    def record_model_load_time(self, tag: str, elapsed_s: float) -> None:
        """
        Record how long it took to load a model into memory.
        tag: "baseline" | "finetuned"
        Kept separate from per-label inference latency in all reports.
        """
        self._model_load_times[tag] = elapsed_s
        logger.info(f"[business_eval] Model load ({tag}): {elapsed_s:.1f}s")

    def record_label_latency(
        self, cluster_id: int, prompt_id: str, elapsed_s: float
    ) -> None:
        self._label_records.append({
            "cluster_id": cluster_id,
            "prompt_id":  prompt_id,
            "elapsed_s":  elapsed_s,
            "source":     "teacher_llm",
        })

    def record_finetuning_time(self, elapsed_s: float) -> None:
        self._finetuning_time_s = elapsed_s
        logger.info(f"[business_eval] Fine-tuning time: {elapsed_s / 60:.1f} min")

    def record_inference_latency(
        self, cluster_id: int, prompt_id: str, elapsed_s: float, fine_tuned: bool
    ) -> None:
        self._inference_records.append({
            "cluster_id": cluster_id,
            "prompt_id":  prompt_id,
            "elapsed_s":  elapsed_s,
            "fine_tuned": fine_tuned,
            "source":     "slm_finetuned" if fine_tuned else "slm_baseline",
        })

    @contextmanager
    def time_inference(self, cluster_id: int, prompt_id: str, fine_tuned: bool):
        """Context manager for timing a single inference call."""
        t0 = time.time()
        yield
        self.record_inference_latency(cluster_id, prompt_id, time.time() - t0, fine_tuned)

    # ── Per-model latency file ─────────────────────────────────────────────────

    def save_latency_file(self, tag: str, output_path: str) -> None:
        """
        Save inference latency records for one model as a flat CSV.

        tag: "teacher" | "baseline" | "finetuned"
        Columns: cluster_id | prompt_id | latency_s

        This file is read by combine.py to build the latency pivot.
        """
        import pandas as pd
        from pathlib import Path

        if tag == "teacher":
            records = self._label_records
            rows    = [
                {"cluster_id": r["cluster_id"], "prompt_id": r["prompt_id"], "latency_s": r["elapsed_s"]}
                for r in records
            ]
        else:
            fine_tuned = (tag == "finetuned")
            rows = [
                {"cluster_id": r["cluster_id"], "prompt_id": r["prompt_id"], "latency_s": r["elapsed_s"]}
                for r in self._inference_records
                if r["fine_tuned"] == fine_tuned
            ]

        if not rows:
            logger.debug(f"[business_eval] No latency records for tag='{tag}' — skipping file.")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        mean_lat = sum(r["latency_s"] for r in rows) / len(rows)
        logger.info(
            f"[business_eval] Latency file ({tag}) → {output_path}  "
            f"({len(rows)} records, mean {mean_lat:.3f}s)"
        )

    # ── Teacher latency persistence ────────────────────────────────────────────

    def save_teacher_latency_records(self, output_path: str) -> None:
        """
        Persist teacher (label generation) latency records to a shared file
        in data/processed/ so future eval-only runs can read them back.
        """
        import pandas as pd
        from pathlib import Path

        if not self._label_records:
            logger.debug("[business_eval] No teacher latency records to persist.")
            return

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._label_records)
        df.to_csv(output_path, index=False)
        logger.info(
            f"[business_eval] Teacher latency records persisted → {output_path} "
            f"({len(df)} records)"
        )

    def load_teacher_latency_from_file(self, file_path: str) -> bool:
        """
        Load teacher latency records from disk into self._label_records.
        Returns True if loaded successfully, False if file not found.
        Logs clearly whether records came from runtime or disk.
        """
        import pandas as pd
        from pathlib import Path

        if self._label_records:
            n = len(self._label_records)
            mean = sum(r["elapsed_s"] for r in self._label_records) / n
            logger.info(
                f"[business_eval] Teacher latency: computed at runtime — "
                f"{n} API calls, mean {mean:.2f}s"
            )
            return True

        path = Path(file_path)
        if not path.exists():
            logger.warning(
                f"[business_eval] Teacher latency: not available "
                f"(label generation skipped and no records file at {path})."
            )
            return False

        df = pd.read_csv(path)
        self._label_records = df.to_dict("records")
        n    = len(self._label_records)
        mean = sum(r["elapsed_s"] for r in self._label_records) / n if n else 0
        logger.info(
            f"[business_eval] Teacher latency: loaded from file "
            f"[{path}] — {n} calls, mean {mean:.2f}s"
        )
        return True

    # ── Reporting ─────────────────────────────────────────────────────────────

    def save(self, output_path: str) -> pd.DataFrame:
        """Save all collected metrics to a CSV file."""
        rows = []

        # Label generation stats
        if self._label_records:
            label_df = pd.DataFrame(self._label_records)
            rows.append({
                "metric":            "label_gen_latency_mean_s",
                "value":             label_df["elapsed_s"].mean(),
                "unit":              "seconds per call",
                "description":       f"Mean latency per LLM call ({self.model_id})",
            })
            rows.append({
                "metric":            "label_gen_latency_p95_s",
                "value":             label_df["elapsed_s"].quantile(0.95),
                "unit":              "seconds per call",
                "description":       "95th percentile latency",
            })
            rows.append({
                "metric":            "label_gen_total_calls",
                "value":             len(label_df),
                "unit":              "API calls",
                "description":       "Total teacher LLM API calls made",
            })

        # SLM inference stats
        if self._inference_records:
            inf_df = pd.DataFrame(self._inference_records)
            for fine_tuned in [False, True]:
                subset = inf_df[inf_df["fine_tuned"] == fine_tuned]
                tag    = "finetuned" if fine_tuned else "baseline"
                if len(subset) > 0:
                    rows.append({
                        "metric":      f"slm_inference_latency_mean_s_{tag}",
                        "value":       subset["elapsed_s"].mean(),
                        "unit":        "seconds per cluster per prompt",
                        "description": f"Mean SLM inference latency ({tag})",
                    })
                    rows.append({
                        "metric":      f"slm_throughput_labels_per_min_{tag}",
                        "value":       60.0 / subset["elapsed_s"].mean(),
                        "unit":        "labels per minute",
                        "description": f"SLM throughput ({tag})",
                    })

        # Fine-tuning time
        if self._finetuning_time_s is not None:
            rows.append({
                "metric":      "finetuning_wall_time_s",
                "value":       self._finetuning_time_s,
                "unit":        "seconds",
                "description": f"Total LoRA/QLoRA fine-tuning time ({self.slm_id})",
            })
            rows.append({
                "metric":      "finetuning_wall_time_min",
                "value":       self._finetuning_time_s / 60,
                "unit":        "minutes",
                "description": f"Total fine-tuning time in minutes",
            })

        # Cost comparison
        cost_rows = self._compute_cost_rows()
        rows.extend(cost_rows)

        df = pd.DataFrame(rows)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"[business_eval] Business metrics saved to {output_path}")
        return df

    def log_summary(self) -> None:
        """Print a human-readable summary to the log."""
        label_records = self._label_records
        inf_records   = self._inference_records

        label_mean = (
            sum(r["elapsed_s"] for r in label_records) / len(label_records)
            if label_records else 0
        )
        baseline_inf = [r for r in inf_records if not r["fine_tuned"]]
        ft_inf       = [r for r in inf_records if r["fine_tuned"]]

        baseline_mean = (
            sum(r["elapsed_s"] for r in baseline_inf) / len(baseline_inf)
            if baseline_inf else 0
        )
        ft_mean = (
            sum(r["elapsed_s"] for r in ft_inf) / len(ft_inf)
            if ft_inf else 0
        )

        n_calls     = len(label_records)
        est_cost    = _estimate_cost(self.model_id, n_calls)
        speedup     = label_mean / ft_mean if ft_mean > 0 else float("inf")

        # Model loading times (separate from per-label latency)
        load_lines = ""
        if self._model_load_times:
            load_lines = "\n  Model loading time (one-time, not in per-label latency):\n"
            for tag, t in self._model_load_times.items():
                load_lines += f"    {tag:<18}: {t:.1f}s\n"

        logger.info(
            f"\n{'=' * 55}\n"
            f"  BUSINESS EVALUATION SUMMARY\n"
            f"{'=' * 55}\n"
            f"  Teacher LLM ({self.model_id}):\n"
            f"    Mean call latency:   {label_mean:.2f}s\n"
            f"    Total API calls:     {n_calls}\n"
            f"    Estimated cost:      ${est_cost:.4f} USD\n"
            f"\n"
            f"  Student SLM ({self.slm_id}):{load_lines}"
            f"    Baseline inference:  {baseline_mean:.3f}s / label\n"
            f"    Fine-tuned inference:{ft_mean:.3f}s / label\n"
            f"    Inference cost:      ~$0.00 (local/Colab)\n"
            f"    Speed vs teacher:    {speedup:.1f}x faster\n"
        )
        if self._finetuning_time_s:
            logger.info(
                f"  Fine-tuning time:    {self._finetuning_time_s / 60:.1f} min\n"
                f"{'=' * 55}"
            )

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_cost_rows(self) -> list[dict]:
        n_calls  = len(self._label_records)
        est_cost = _estimate_cost(self.model_id, n_calls)
        return [
            {
                "metric":      "teacher_llm_estimated_cost_usd",
                "value":       est_cost,
                "unit":        "USD",
                "description": f"Estimated API cost for {n_calls} label generation calls",
            },
            {
                "metric":      "slm_inference_estimated_cost_usd",
                "value":       0.0,
                "unit":        "USD",
                "description": "SLM inference cost (local GPU/Colab = ~$0)",
            },
        ]


def _estimate_cost(model_id: str, n_calls: int) -> float:
    """Rough cost estimate in USD."""
    rate = COST_PER_1M_TOKENS.get(model_id, 1.0)  # fallback $1/1M
    total_tokens = n_calls * (AVG_INPUT_TOKENS_PER_CALL + AVG_OUTPUT_TOKENS_PER_CALL)
    return (total_tokens / 1_000_000) * rate
