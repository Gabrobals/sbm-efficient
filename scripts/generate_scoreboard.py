#!/usr/bin/env python3
"""
Generate B2 scoreboard JSON from input robustness summary.
Usage: python -m scripts.generate_scoreboard --task mnist
"""

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone


def load_input_robustness(task: str, summaries_dir: Path) -> dict:
    path = summaries_dir / f"{task}_input_robustness.json"
    if not path.exists():
        raise FileNotFoundError(f"Input robustness summary not found: {path}")
    with open(path) as f:
        return json.load(f)


def compute_wins_losses(data: dict, baseline: str = "static_topk", candidate: str = "sbm_adaptive_k") -> dict:
    baseline_data = data["models"].get(baseline, {}).get("per_test_level", {})
    candidate_data = data["models"].get(candidate, {}).get("per_test_level", {})
    
    if not baseline_data or not candidate_data:
        raise ValueError(f"Missing model data. Found: {list(data['models'].keys())}")
    
    wins, losses, ties = 0, 0, 0
    details = []
    
    perturbations = set(baseline_data.keys()) & set(candidate_data.keys())
    
    for perturb in sorted(perturbations):
        b_acc = baseline_data[perturb].get("accuracy_mean", 0)
        c_acc = candidate_data[perturb].get("accuracy_mean", 0)
        
        diff = c_acc - b_acc
        if diff > 0.005:
            winner = "candidate"
            wins += 1
        elif diff < -0.005:
            winner = "baseline"
            losses += 1
        else:
            winner = "tie"
            ties += 1
        
        details.append({
            "perturbation": perturb,
            "baseline_acc": round(b_acc, 4),
            "candidate_acc": round(c_acc, 4),
            "diff_pct": round(diff * 100, 2),
            "winner": winner
        })
    
    return {"wins": wins, "losses": losses, "ties": ties, "details": details}


def get_clean_metrics(data: dict, model: str) -> dict:
    model_data = data["models"].get(model, {}).get("per_test_level", {})
    for key in ["gaussian:0.0", "salt_pepper:0.0", "occlusion:0.0"]:
        if key in model_data:
            return {
                "accuracy": model_data[key].get("accuracy_mean", 0),
                "f1": model_data[key].get("f1_mean", 0),
                "flops": model_data[key].get("flops_executed_mean", 0),
                "k_mean": model_data[key].get("k_mean_mean", 0)
            }
    return {}


def generate_scoreboard(task: str, summaries_dir: Path, baseline: str = "static_topk", candidate: str = "sbm_adaptive_k") -> dict:
    data = load_input_robustness(task, summaries_dir)
    comparison = compute_wins_losses(data, baseline, candidate)
    baseline_metrics = get_clean_metrics(data, baseline)
    candidate_metrics = get_clean_metrics(data, candidate)
    
    flops_gain, k_gain = 0, 0
    if baseline_metrics.get("flops") and candidate_metrics.get("flops"):
        flops_gain = round((1 - candidate_metrics["flops"] / baseline_metrics["flops"]) * 100, 1)
    if baseline_metrics.get("k_mean") and candidate_metrics.get("k_mean"):
        k_gain = round((1 - candidate_metrics["k_mean"] / baseline_metrics["k_mean"]) * 100, 1)
    
    acc_delta = round((candidate_metrics.get("accuracy", 0) - baseline_metrics.get("accuracy", 0)) * 100, 2)
    
    return {
        "task": task,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_model": baseline,
        "candidate_model": candidate,
        "headline": {
            "compute_gain_flops_pct": flops_gain,
            "compute_gain_kmean_pct": k_gain,
            "robustness_wins_count": comparison["wins"],
            "robustness_losses_count": comparison["losses"],
            "robustness_ties_count": comparison["ties"],
            "baseline_final_accuracy": round(baseline_metrics.get("accuracy", 0), 4),
            "candidate_final_accuracy": round(candidate_metrics.get("accuracy", 0), 4),
            "accuracy_delta_pct": acc_delta
        },
        "perturbation_details": comparison["details"]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate B2 scoreboard")
    parser.add_argument("--task", required=True, help="Task name (mnist, fashion_mnist)")
    parser.add_argument("--summaries-dir", default="results/summaries", help="Path to summaries directory")
    parser.add_argument("--output", default=None, help="Output path (default: {summaries_dir}/{task}_b2_scoreboard.json)")
    args = parser.parse_args()
    
    summaries_dir = Path(args.summaries_dir)
    
    scoreboard = generate_scoreboard(args.task, summaries_dir)
    
    output_path = Path(args.output) if args.output else summaries_dir / f"{args.task}_b2_scoreboard.json"
    
    with open(output_path, "w") as f:
        json.dump(scoreboard, f, indent=2)
    
    print(f"[OK] Scoreboard generated: {output_path}")
    h = scoreboard["headline"]
    print(f"  Task: {args.task}")
    print(f"  Compute gain (FLOPs): {h['compute_gain_flops_pct']}%")
    print(f"  Compute gain (K): {h['compute_gain_kmean_pct']}%")
    print(f"  Robustness: {h['robustness_wins_count']} wins / {h['robustness_losses_count']} losses / {h['robustness_ties_count']} ties")
    print(f"  Accuracy: baseline={h['baseline_final_accuracy']:.4f}, candidate={h['candidate_final_accuracy']:.4f}, delta={h['accuracy_delta_pct']}%")


if __name__ == "__main__":
    main()
