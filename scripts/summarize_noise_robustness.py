"""Summarize MNIST noise robustness across runs.

- Scans results/runs/*/metrics.json
- Supports models: static_topk, sbm_adaptive_k
- Aggregates per noise_sigma (includes baseline sigma=0.0)
- Computes mean/std for accuracy/precision/recall/F1 (macro+micro), fp/fn, flops, latency_ms, active_modules_mean, k_mean
- Computes degradation_pct vs sigma=0 baseline per run, then aggregates mean/std

ASCII-only; standard library only (no numpy).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, DefaultDict

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "results" / "runs"
OUT_PATH = ROOT / "results" / "summaries" / "mnist_noise_robustness.json"
TARGET_MODELS = {"static_topk", "sbm_adaptive_k"}
TARGET_TASK = "mnist"

METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "precision_micro",
    "recall_micro",
    "f1_micro",
    "fp_total",
    "fn_total",
    "flops_executed",
    "latency_ms",
    "active_modules_mean",
    "k_mean",
]


def _degradation(base: float, val: float) -> float:
    if base == 0 or val is None or math.isnan(val):
        return 0.0
    return float((base - val) / abs(base) * 100.0)


def _safe_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def _extract_metrics(obj: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in METRIC_KEYS:
        # k_mean fallback to active_modules_mean when missing
        if k == "k_mean" and k not in obj:
            out[k] = _safe_float(obj.get("active_modules_mean"))
        else:
            out[k] = _safe_float(obj.get(k))
    return out


def _mean_std(values: List[float]) -> Tuple[float, float]:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return float("nan"), float("nan")
    mean = sum(clean) / len(clean)
    if len(clean) == 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in clean) / len(clean)
    return mean, math.sqrt(variance)


def _aggregate(per_run: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    for k in METRIC_KEYS:
        vals = [_safe_float(run.get(k)) for run in per_run]
        m, s = _mean_std(vals)
        agg[f"{k}_mean"] = m
        agg[f"{k}_std"] = s

    # degradation_pct if present
    if per_run and "degradation_pct" in per_run[0]:
        degr_keys = list(per_run[0]["degradation_pct"].keys())
        degr_block: Dict[str, Any] = {}
        for dk in degr_keys:
            vals = [_safe_float(run.get("degradation_pct", {}).get(dk)) for run in per_run]
            m, s = _mean_std(vals)
            degr_block[f"{dk}_mean"] = m
            degr_block[f"{dk}_std"] = s
        agg["degradation_pct"] = degr_block

    agg["n_runs"] = len(per_run)
    return agg


def _collect_runs() -> Dict[str, DefaultDict[float, List[Dict[str, Any]]]]:
    collected: Dict[str, DefaultDict[float, List[Dict[str, Any]]]] = {}
    if not RUNS_DIR.exists():
        print(f"[WARN] RUNS_DIR not found: {RUNS_DIR}")
        return collected

    for run_dir in RUNS_DIR.iterdir():
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        model = str(metrics.get("model", ""))
        task = str(metrics.get("task", ""))
        if model not in TARGET_MODELS or task != TARGET_TASK:
            continue

        final = metrics.get("final", {})
        base = _extract_metrics(final)
        base["degradation_pct"] = {f"{k}_pct": 0.0 for k in METRIC_KEYS if k != "flops_executed"}
        # Include baseline sigma=0.0
        collected.setdefault(model, defaultdict(list))[0.0].append(base)

        noise = metrics.get("noise", {}) if isinstance(metrics.get("noise", {}), dict) else {}
        evaluations = noise.get("evaluations", []) if isinstance(noise.get("evaluations", []), list) else []
        for ev in evaluations:
            if not isinstance(ev, dict):
                continue
            try:
                sigma = float(ev.get("noise_sigma", None))
            except Exception:
                continue
            ev_metrics = _extract_metrics(ev)
            # Compute degradation vs this run's baseline
            degr: Dict[str, float] = {}
            for mk in METRIC_KEYS:
                degr_key = f"{mk}_pct"
                degr[degr_key] = _degradation(base.get(mk, 0.0), ev_metrics.get(mk, 0.0))
            ev_metrics["degradation_pct"] = degr
            collected.setdefault(model, defaultdict(list))[sigma].append(ev_metrics)

    return collected


def build_summary() -> Dict[str, Any]:
    collected = _collect_runs()

    summary: Dict[str, Any] = {
        "task": TARGET_TASK,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {},
    }

    for model, sigma_map in collected.items():
        model_block: Dict[str, Any] = {"aggregates": {}}
        for sigma in sorted(sigma_map.keys()):
            per_run = sigma_map[sigma]
            # Format sigma key: 0.0 -> "0", 0.05 -> "0.05", etc.
            sigma_key = f"{sigma:.2f}".rstrip("0").rstrip(".")
            if sigma_key == "":
                sigma_key = "0"
            model_block["aggregates"][sigma_key] = _aggregate(per_run)
        model_block["sigmas"] = sorted(sigma_map.keys())
        summary["models"][model] = model_block

    return summary


def main() -> int:
    print(f"[INFO] Scanning runs in: {RUNS_DIR}")
    summary = build_summary()

    n_models = len(summary.get("models", {}))
    if n_models == 0:
        print("[WARN] No matching runs found (static_topk or sbm_adaptive_k for mnist)")
        print("[INFO] Writing empty summary anyway...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Written: {OUT_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
