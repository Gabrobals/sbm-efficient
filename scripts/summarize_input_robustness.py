#!/usr/bin/env python3
"""
Aggregate input robustness results (B2) across seeds.

Reads metrics.json files from all completed runs and aggregates
robustness_input results per model/test/level.

Output: results/summaries/mnist_input_robustness.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List


def safe_stdev(values: List[float]) -> float:
    """Standard deviation, returns 0.0 if fewer than 2 values."""
    if len(values) < 2:
        return 0.0
    return stdev(values)


def load_input_robustness_from_runs(
    runs_dir: Path, task_filter: str = "mnist"
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Load robustness_input from all matching run directories.

    Returns:
        { model: { (test, level): [ {accuracy, precision, ...}, ... ] } }
    """
    results: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        if task_filter and f"_{task_filter}_" not in run_dir.name:
            continue

        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except json.JSONDecodeError:
            print(f"[WARN] Invalid JSON in {metrics_path}", file=sys.stderr)
            continue

        robustness = metrics.get("robustness_input")
        if not robustness:
            continue

        model = metrics.get("model", "unknown")
        evaluations = robustness.get("evaluations", [])

        for ev in evaluations:
            test_name = ev.get("test", "unknown")
            level = ev.get("level", 0.0)
            key = f"{test_name}:{level}"
            results[model][key].append(ev)

    return results


def aggregate_input_robustness(
    results: Dict[str, Dict[str, List[Dict[str, Any]]]]
) -> Dict[str, Any]:
    """
    Aggregate robustness metrics across seeds.

    Returns JSON-serializable summary structure.
    """
    output: Dict[str, Any] = {"models": {}}

    for model, test_data in sorted(results.items()):
        model_summary: Dict[str, Any] = {"per_test_level": {}}

        for key, seed_results in sorted(test_data.items()):
            test_name, level_str = key.split(":", 1)
            level = float(level_str)

            accs = [r.get("accuracy", 0.0) for r in seed_results]
            precs = [r.get("precision", 0.0) for r in seed_results]
            recs = [r.get("recall", 0.0) for r in seed_results]
            f1s = [r.get("f1", 0.0) for r in seed_results]
            prec_micros = [r.get("precision_micro", 0.0) for r in seed_results]
            rec_micros = [r.get("recall_micro", 0.0) for r in seed_results]
            f1_micros = [r.get("f1_micro", 0.0) for r in seed_results]
            flops = [r.get("flops_executed", 0) for r in seed_results]
            latencies = [r.get("latency_ms", 0.0) for r in seed_results]
            k_means = [r.get("k_mean", 0.0) for r in seed_results]
            k_stds = [r.get("k_std", 0.0) for r in seed_results]

            # Degradation aggregates
            degr_accs = [r.get("degradation_pct", {}).get("accuracy_pct", 0.0) for r in seed_results]
            degr_f1s = [r.get("degradation_pct", {}).get("f1_pct", 0.0) for r in seed_results]
            degr_f1_micros = [r.get("degradation_pct", {}).get("f1_micro_pct", 0.0) for r in seed_results]

            model_summary["per_test_level"][key] = {
                "test": test_name,
                "level": level,
                "n_seeds": len(seed_results),
                "accuracy_mean": mean(accs),
                "accuracy_std": safe_stdev(accs),
                "precision_mean": mean(precs),
                "precision_std": safe_stdev(precs),
                "recall_mean": mean(recs),
                "recall_std": safe_stdev(recs),
                "f1_mean": mean(f1s),
                "f1_std": safe_stdev(f1s),
                "precision_micro_mean": mean(prec_micros),
                "precision_micro_std": safe_stdev(prec_micros),
                "recall_micro_mean": mean(rec_micros),
                "recall_micro_std": safe_stdev(rec_micros),
                "f1_micro_mean": mean(f1_micros),
                "f1_micro_std": safe_stdev(f1_micros),
                "flops_executed_mean": mean(flops),
                "flops_executed_std": safe_stdev(flops),
                "latency_ms_mean": mean(latencies),
                "latency_ms_std": safe_stdev(latencies),
                "k_mean_mean": mean(k_means),
                "k_mean_std": safe_stdev(k_means),
                "k_std_mean": mean(k_stds),
                "degradation_accuracy_pct_mean": mean(degr_accs),
                "degradation_accuracy_pct_std": safe_stdev(degr_accs),
                "degradation_f1_pct_mean": mean(degr_f1s),
                "degradation_f1_pct_std": safe_stdev(degr_f1s),
                "degradation_f1_micro_pct_mean": mean(degr_f1_micros),
                "degradation_f1_micro_pct_std": safe_stdev(degr_f1_micros),
            }

        output["models"][model] = model_summary

    return output


def main():
    parser = argparse.ArgumentParser(description="Aggregate B2 input robustness results across seeds")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("results/runs"),
        help="Directory containing run subdirectories",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mnist",
        help="Task filter (e.g., mnist, xor)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results/summaries/{task}_input_robustness.json)",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists():
        print(f"[ERROR] Runs directory not found: {runs_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output
    if output_path is None:
        summaries_dir = runs_dir.parent / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        output_path = summaries_dir / f"{args.task}_input_robustness.json"
    output_path = output_path.resolve()

    print(f"Loading input robustness data from: {runs_dir}")
    print(f"Task filter: {args.task}")

    results = load_input_robustness_from_runs(runs_dir, task_filter=args.task)

    if not results:
        print("[WARN] No input robustness data found in any runs.", file=sys.stderr)
        sys.exit(0)

    print(f"Found input robustness data for models: {list(results.keys())}")

    summary = aggregate_input_robustness(results)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Summary written to: {output_path}")

    # Print quick overview
    for model, model_data in summary["models"].items():
        print(f"\n  {model}:")
        for key, data in model_data["per_test_level"].items():
            print(
                f"    {key}: acc={100*data['accuracy_mean']:.2f}% +/- {100*data['accuracy_std']:.2f}%, "
                f"f1={100*data['f1_mean']:.2f}% +/- {100*data['f1_std']:.2f}%, "
                f"degr_acc={data['degradation_accuracy_pct_mean']:.2f}%"
            )


if __name__ == "__main__":
    main()
