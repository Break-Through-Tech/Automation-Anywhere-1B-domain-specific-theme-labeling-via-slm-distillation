"""Rejudge saved predictions on a fixed split; never load or train the SLM."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


TAGS = ("teacher", "baseline", "finetuned")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_inputs(run_dir, split):
    evaluation_dir = run_dir / "evaluation"
    labels = read_csv(evaluation_dir / "labels_by_cluster.csv")
    selected = [row for row in labels if row["split"] == split]
    if not selected:
        raise ValueError(f"No {split} rows found in the saved labels_by_cluster.csv.")

    keys = {(int(row["cluster_id"]), row["prompt_id"]) for row in selected}
    if len(keys) != len(selected):
        raise ValueError("Duplicate cluster/prompt keys in the saved split.")

    cluster_ids = {cid for cid, _ in keys}
    for row in labels:
        if int(row["cluster_id"]) in cluster_ids and row["split"] != split:
            raise ValueError("A cluster appears in multiple saved splits.")

    predictions = {}
    for tag in TAGS:
        path = evaluation_dir / f"{tag}_predictions.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        chosen = [
            row for row in rows
            if (int(row["cluster_id"]), row["prompt_id"]) in keys
        ]
        found = [(int(row["cluster_id"]), row["prompt_id"]) for row in chosen]
        if len(found) != len(set(found)) or set(found) != keys:
            raise ValueError(f"{tag}: missing or duplicate predictions for the saved split.")
        if any(not str(row.get("generated_label", "")).strip() for row in chosen):
            raise ValueError(f"{tag}: empty generated label.")
        predictions[tag] = sorted(
            chosen, key=lambda row: (int(row["cluster_id"]), row["prompt_id"])
        )

    tickets = read_csv(run_dir / "labeled_data.csv")
    tickets = [row for row in tickets if int(row["cluster_id"]) in cluster_ids]
    if {int(row["cluster_id"]) for row in tickets} != cluster_ids:
        raise ValueError("The saved labeled_data.csv is missing selected clusters.")
    return selected, tickets, predictions, cluster_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--judge-mode", choices=["reference", "reference_free"],
                        default="reference_free")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check input coverage only; no files written or API calls.")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    selected, tickets, predictions, cluster_ids = prepare_inputs(run_dir, args.split)

    print(f"Source: {run_dir}")
    print(f"Saved split: {args.split}; clusters: {len(cluster_ids)}")
    for tag in TAGS:
        print(f"{tag}: {len(predictions[tag])} saved predictions")
    calls = len(selected) * (3 if args.judge_mode == "reference_free" else 2)
    print(f"Judge mode: {args.judge_mode}; expected judge calls: {calls}")
    if args.dry_run:
        print("Input checks passed. No training, inference, or judge calls performed.")
        return

    # Imports are delayed so --dry-run needs only Python's standard library.
    import logging
    import pandas as pd
    import yaml
    from phase1.evaluation.llm_judge import run_llm_judge
    from phase1.evaluation.combine import combine_llm_by_cluster, combine_llm_summary

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    with Path(args.config).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["evaluation"]["judge_mode"] = args.judge_mode
    # Every input file below contains only the selected split; sample all rows.
    cfg["evaluation"]["judge_llm"]["n_samples"] = len(selected)

    top_k = int(cfg["top_k"])
    inputs = []
    for cid in sorted(cluster_ids):
        cluster_tickets = sorted(
            [row for row in tickets
             if int(row["cluster_id"]) == cid
             and int(row["ticket_rank_in_cluster"]) <= top_k],
            key=lambda row: int(row["ticket_rank_in_cluster"]),
        )
        if not cluster_tickets:
            raise ValueError(f"No top-k tickets for cluster {cid}.")
        for row in sorted(
            [r for r in selected if int(r["cluster_id"]) == cid],
            key=lambda r: r["prompt_id"],
        ):
            inputs.append({
                "cluster_id": cid,
                "prompt_id": row["prompt_id"],
                "tickets": [r["ticket_details"] for r in cluster_tickets],
                "teacher_label": row["teacher_label"],
            })
    fingerprint = hashlib.sha256(
        json.dumps(inputs, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out_dir = run_dir / "reevaluations" / f"{stamp}_{args.judge_mode}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "_cache").mkdir()
    pd.DataFrame(selected).to_csv(out_dir / "labels_by_cluster.csv", index=False)
    labeled_df = pd.DataFrame(tickets)
    for col in ("cluster_id", "ticket_rank_in_cluster"):
        labeled_df[col] = labeled_df[col].astype(int)
    labeled_df.to_csv(out_dir / "labeled_data.csv", index=False)
    with (out_dir / "config_used.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    manifest = {
        "source_run": str(run_dir),
        "split": args.split,
        "cluster_ids": sorted(cluster_ids),
        "n_predictions_per_model": len(selected),
        "judge_mode": args.judge_mode,
        "top_k": top_k,
        "evaluation_input_sha256": fingerprint,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (out_dir / "evaluation_inputs.json").write_text(
        json.dumps(inputs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for tag, rows in predictions.items():
        with (out_dir / f"{tag}_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"New results folder: {out_dir}", flush=True)
    print(f"Evaluation input fingerprint: {fingerprint}", flush=True)
    dims = (
        ("faithfulness", "specificity", "coherence")
        if args.judge_mode == "reference_free"
        else ("faithfulness", "specificity", "equivalence")
    )
    for tag in TAGS:
        result = run_llm_judge(
            cfg=cfg,
            labeled_df=labeled_df,
            predictions_path=str(out_dir / f"{tag}_predictions.jsonl"),
            fine_tuned=(tag == "finetuned"),
            output_path=str(out_dir / f"judge_scores_{tag}.csv"),
            eval_label=tag,
            tag=tag,
        )
        if len(result) != len(selected):
            raise RuntimeError(f"{tag}: not every selected prediction was judged.")
        if not result[list(dims)].isin([1, 2, 3, 4, 5]).all().all():
            raise RuntimeError(
                f"{tag}: invalid judge scores detected. Inspect {out_dir / '_cache'}; "
                "these results should not be treated as a completed comparison."
            )

    split_map = {cid: args.split for cid in cluster_ids}
    combine_llm_by_cluster(out_dir, out_dir / "_cache", split_map)
    combine_llm_summary(out_dir, out_dir / "_cache", split_map)
    summary_path = out_dir / "judge_summary.csv"
    summary = pd.read_csv(summary_path)
    summary = summary[summary["split"] == args.split]
    summary.to_csv(summary_path, index=False)
    print("\n" + summary.to_string(index=False))
    print(f"Saved: {summary_path}")
    print("This run reused saved labels; it did not measure new model latency.")


if __name__ == "__main__":
    main()


