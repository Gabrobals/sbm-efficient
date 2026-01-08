"""Update mnist_comparison.json with noise robustness section.

Merges data from:
- results/aggregated_results.json (clean multi-seed aggregates)
- results/summaries/mnist_noise_robustness.json (noise robustness summary)
- results/summaries/mnist_comparison.json (existing comparison, if any)

Outputs updated results/summaries/mnist_comparison.json with a "robustness" section.

ASCII-only; standard library only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
AGG_PATH = ROOT / "results" / "aggregated_results.json"
NOISE_PATH = ROOT / "results" / "summaries" / "mnist_noise_robustness.json"
COMPARISON_PATH = ROOT / "results" / "summaries" / "mnist_comparison.json"

# Map from noise summary model keys to comparison keys
MODEL_KEY_MAP = {
    "static_topk": "mnist_static_topk",
    "sbm_adaptive_k": "mnist_sbm_adaptive_k",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_per_sigma(model_block: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert aggregates dict to per_sigma list sorted by sigma."""
    aggregates = model_block.get("aggregates", {})
    sigmas = model_block.get("sigmas", [])
    
    per_sigma: List[Dict[str, Any]] = []
    for sigma in sorted(sigmas):
        # Key in aggregates is string like "0", "0.05", etc.
        sigma_key = f"{sigma:.2f}".rstrip("0").rstrip(".")
        if sigma_key == "":
            sigma_key = "0"
        
        agg = aggregates.get(sigma_key, {})
        entry = {
            "sigma": sigma,
            "accuracy_mean": agg.get("accuracy_mean"),
            "accuracy_std": agg.get("accuracy_std"),
            "precision_mean": agg.get("precision_mean"),
            "precision_std": agg.get("precision_std"),
            "recall_mean": agg.get("recall_mean"),
            "recall_std": agg.get("recall_std"),
            "f1_mean": agg.get("f1_mean"),
            "f1_std": agg.get("f1_std"),
            "precision_micro_mean": agg.get("precision_micro_mean"),
            "precision_micro_std": agg.get("precision_micro_std"),
            "recall_micro_mean": agg.get("recall_micro_mean"),
            "recall_micro_std": agg.get("recall_micro_std"),
            "f1_micro_mean": agg.get("f1_micro_mean"),
            "f1_micro_std": agg.get("f1_micro_std"),
            "fp_total_mean": agg.get("fp_total_mean"),
            "fn_total_mean": agg.get("fn_total_mean"),
            "flops_executed_mean": agg.get("flops_executed_mean"),
            "k_mean_mean": agg.get("k_mean_mean"),
            "active_modules_mean_mean": agg.get("active_modules_mean_mean"),
            "n_runs": agg.get("n_runs"),
            "degradation_pct": agg.get("degradation_pct", {}),
        }
        per_sigma.append(entry)
    
    return per_sigma


def main() -> int:
    # Load existing comparison or create minimal
    comparison = _load_json(COMPARISON_PATH)
    if not comparison:
        comparison = {
            "task": "mnist",
            "models": [],
            "notes": "Auto-generated comparison file.",
        }
    
    # Load noise robustness summary
    noise_summary = _load_json(NOISE_PATH)
    if not noise_summary:
        print(f"[WARN] Noise summary not found: {NOISE_PATH}")
        print("[INFO] Run scripts/summarize_noise_robustness.py first.")
        return 1
    
    # Extract sigmas from first model found
    all_sigmas: List[float] = []
    noise_models = noise_summary.get("models", {})
    for model_key, model_block in noise_models.items():
        if "sigmas" in model_block:
            all_sigmas = sorted(set(all_sigmas) | set(model_block.get("sigmas", [])))
    
    # Build robustness section
    robustness: Dict[str, Any] = {
        "method": "eval_logit_noise",
        "sigmas": all_sigmas,
        "models": {},
    }
    
    for short_key, full_key in MODEL_KEY_MAP.items():
        model_block = noise_models.get(short_key)
        if model_block:
            robustness["models"][full_key] = {
                "per_sigma": _build_per_sigma(model_block),
            }
    
    # Update comparison with robustness section
    comparison["robustness"] = robustness
    
    # Write back
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMPARISON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    
    print(f"[OK] written: {COMPARISON_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
