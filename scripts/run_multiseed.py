# scripts/run_multiseed.py
"""
Multi-seed runner for SBM-Efficient.

Purpose:
- Run ONE experiment config across multiple seeds (default: 42..46)
- Call the standard entrypoint: `python -m src.experiments.run --config <yaml>`
- Collect per-run `metrics.json`
- Aggregate mean/std and save to `results/aggregated_results.json`

Usage (PowerShell):
  & ".venv/Scripts/python.exe" -m scripts.run_multiseed --config configs/sbm_adaptive_k_mnist.yaml
  & ".venv/Scripts/python.exe" -m scripts.run_multiseed --config configs/sbm_adaptive_k_mnist.yaml --seeds 42,43,44,45,46
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]


def _parse_seeds(seeds_str: str) -> List[int]:
    parts = [p.strip() for p in seeds_str.split(",") if p.strip()]
    seeds: List[int] = []
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            if b_i < a_i:
                raise ValueError(f"Invalid seed range: {p}")
            seeds.extend(list(range(a_i, b_i + 1)))
        else:
            seeds.append(int(p))
    if not seeds:
        raise ValueError("No seeds provided.")
    return seeds


def _safe_float(x: Any) -> float:
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return 0.0


def _safe_int(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if hasattr(x, "item"):
        try:
            return int(x.item())
        except Exception:
            pass
    try:
        return int(x)
    except Exception:
        return 0


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.pstdev(xs))


def _load_yaml_or_fallback_text(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Try to load YAML using PyYAML. If not available, return (None, raw_text).
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("Config YAML root is not a dict.")
        return data, text
    except Exception:
        return None, text


def _write_yaml(data: Dict[str, Any], out_path: Path) -> None:
    """
    Write YAML using PyYAML (preferred). If PyYAML not available, write as text JSON-ish is not acceptable,
    so we fallback to a minimal seed line replace elsewhere.
    """
    import yaml  # type: ignore

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _set_seed_in_yaml(config_path: Path, seed: int, work_dir: Path) -> Path:
    """
    Create a seed-specific temporary YAML under work_dir and return its path.
    Works with PyYAML if present; otherwise uses a conservative text-based seed replacement.
    """
    data, raw = _load_yaml_or_fallback_text(config_path)

    tmp_path = work_dir / f"{config_path.stem}__seed{seed}.yaml"

    if data is not None:
        # Ensure run.seed exists
        run = data.get("run")
        if not isinstance(run, dict):
            run = {}
            data["run"] = run
        run["seed"] = int(seed)
        _write_yaml(data, tmp_path)
        return tmp_path

    # Fallback text mode: replace the first occurrence of "seed: <num>" under run: block if possible
    # If not found, append under run:.
    lines = raw.splitlines()

    # Try to locate run: block indentation
    run_idx = None
    run_indent = ""
    for i, line in enumerate(lines):
        if re.match(r"^\s*run\s*:\s*$", line):
            run_idx = i
            run_indent = re.match(r"^(\s*)run\s*:\s*$", line).group(1)  # type: ignore
            break

    if run_idx is None:
        # No run block: append one
        lines.append("")
        lines.append("run:")
        lines.append(f"  seed: {seed}")
        tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp_path

    # Determine child indentation (two spaces more than run indent by convention)
    child_indent = run_indent + "  "

    # Find end of run block
    end_idx = len(lines)
    for j in range(run_idx + 1, len(lines)):
        # A new top-level key (same indent as run) ends the block
        if re.match(rf"^{re.escape(run_indent)}\S", lines[j]) and not re.match(
            rf"^{re.escape(run_indent)}\s", lines[j]
        ):
            # rare, ignore
            pass
        if re.match(rf"^{re.escape(run_indent)}[A-Za-z0-9_\-]+\s*:\s*", lines[j]) and not re.match(
            rf"^{re.escape(child_indent)}", lines[j]
        ):
            end_idx = j
            break

    # Replace seed line if present
    seed_replaced = False
    for k in range(run_idx + 1, end_idx):
        if re.match(rf"^{re.escape(child_indent)}seed\s*:\s*", lines[k]):
            lines[k] = f"{child_indent}seed: {seed}"
            seed_replaced = True
            break

    if not seed_replaced:
        # Insert seed line right after run:
        lines.insert(run_idx + 1, f"{child_indent}seed: {seed}")

    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def _run_one(config_path: Path, seed: int, python_exe: str) -> Tuple[bool, str, Optional[Path], str]:
    """
    Run one seed. Returns:
      (ok, run_id, run_dir, error_message)
    """
    cmd = [
        python_exe,
        "-m",
        "src.experiments.run",
        "--config",
        str(config_path),
    ]

    # Use project root as cwd
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    out = proc.stdout or ""
    # Parse Run ID / Run dir / Metrics saved
    run_id = "unknown"
    run_dir = None

    m_id = re.search(r"Run ID:\s*([^\r\n]+)", out)
    if m_id:
        run_id = m_id.group(1).strip()

    m_dir = re.search(r"Run dir:\s*([^\r\n]+)", out)
    if m_dir:
        run_dir = Path(m_dir.group(1).strip())

    m_metrics = re.search(r"Metrics saved to:\s*([^\r\n]+metrics\.json)", out)
    if m_metrics:
        metrics_path = Path(m_metrics.group(1).strip())
        run_dir = metrics_path.parent

    if proc.returncode != 0:
        # Provide tail of output as error
        tail = "\n".join(out.splitlines()[-25:])
        return False, run_id, run_dir, f"Non-zero exit code {proc.returncode}\n{tail}"

    # Sometimes the training prints "[FAIL] ..." but returns 0 (rare). Detect common fail marker.
    if "[FAIL]" in out or "Training failed" in out:
        tail = "\n".join(out.splitlines()[-25:])
        return False, run_id, run_dir, tail

    return True, run_id, run_dir, ""


def _read_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _aggregate(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate over runs (seeds). Assumes each metrics has a 'final' dict.
    """
    acc = [_safe_float(m.get("final", {}).get("accuracy")) for m in metrics_list]
    flops = [_safe_int(m.get("final", {}).get("flops_executed")) for m in metrics_list]
    lat = [_safe_float(m.get("final", {}).get("latency_ms")) for m in metrics_list]
    active = [_safe_float(m.get("final", {}).get("active_modules_mean")) for m in metrics_list]
    ent = [_safe_float(m.get("final", {}).get("entropy_mean")) for m in metrics_list]

    k_mean = [_safe_float(m.get("final", {}).get("k_mean")) for m in metrics_list]
    k_std = [_safe_float(m.get("final", {}).get("k_std")) for m in metrics_list]

    prec = [_safe_float(m.get("final", {}).get("precision")) for m in metrics_list]
    rec = [_safe_float(m.get("final", {}).get("recall")) for m in metrics_list]
    f1 = [_safe_float(m.get("final", {}).get("f1")) for m in metrics_list]
    prec_micro = [_safe_float(m.get("final", {}).get("precision_micro")) for m in metrics_list]
    rec_micro = [_safe_float(m.get("final", {}).get("recall_micro")) for m in metrics_list]
    f1_micro = [_safe_float(m.get("final", {}).get("f1_micro")) for m in metrics_list]

    # Params: assume constant, take first
    params = _safe_int(metrics_list[0].get("training", {}).get("parameter_count"))

    acc_m, acc_s = _mean_std(acc)
    lat_m, lat_s = _mean_std(lat)
    active_m, active_s = _mean_std(active)
    ent_m, ent_s = _mean_std(ent)

    # FLOPs are ints; std meaningful but keep float
    flops_f = [float(x) for x in flops]
    flops_m, flops_s = _mean_std(flops_f)

    k_mean_m, k_mean_s = _mean_std(k_mean)
    k_std_m, k_std_s = _mean_std(k_std)

    prec_m, prec_s = _mean_std(prec)
    rec_m, rec_s = _mean_std(rec)
    f1_m, f1_s = _mean_std(f1)

    prec_micro_m, prec_micro_s = _mean_std(prec_micro)
    rec_micro_m, rec_micro_s = _mean_std(rec_micro)
    f1_micro_m, f1_micro_s = _mean_std(f1_micro)

    return {
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "flops_mean": int(round(flops_m)),
        "flops_std": flops_s,
        "latency_mean": lat_m,
        "latency_std": lat_s,
        "active_modules_mean": active_m,
        "active_modules_std": active_s,
        "entropy_mean": ent_m,
        "entropy_std": ent_s,
        "k_mean": k_mean_m,
        "k_mean_std": k_mean_s,
        "k_std": k_std_m,
        "k_std_std": k_std_s,
        "precision_mean": prec_m,
        "precision_std": prec_s,
        "recall_mean": rec_m,
        "recall_std": rec_s,
        "f1_mean": f1_m,
        "f1_std": f1_s,
        "precision_micro_mean": prec_micro_m,
        "precision_micro_std": prec_micro_s,
        "recall_micro_mean": rec_micro_m,
        "recall_micro_std": rec_micro_s,
        "f1_micro_mean": f1_micro_m,
        "f1_micro_std": f1_micro_s,
        "params": params,
        "n_seeds": len(metrics_list),
    }


def _aggregate_noise(metrics_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Aggregate noise robustness evaluations across seeds.

    Expects metrics entries to include noise.evaluations list with per-sigma metrics.
    Returns None when no noise section is present.
    """

    sigma_buckets: Dict[float, List[Dict[str, Any]]] = {}
    baseline_sigma: Optional[float] = None

    for m in metrics_list:
        noise = m.get("noise") if isinstance(m, dict) else None
        if not isinstance(noise, dict):
            continue

        if baseline_sigma is None:
            try:
                baseline_sigma = float(noise.get("baseline_sigma", 0.0))
            except Exception:
                baseline_sigma = 0.0

        for ev in noise.get("evaluations", []):
            if not isinstance(ev, dict):
                continue
            try:
                sigma = float(ev.get("noise_sigma", 0.0))
            except Exception:
                sigma = 0.0
            sigma_buckets.setdefault(sigma, []).append(ev)

    if not sigma_buckets:
        return None

    evaluations: List[Dict[str, Any]] = []

    def _collect(key: str, bucket: List[Dict[str, Any]]) -> Tuple[float, float]:
        values = [_safe_float(ev.get(key)) for ev in bucket]
        return _mean_std(values)

    def _collect_int(key: str, bucket: List[Dict[str, Any]]) -> Tuple[float, float]:
        values = [_safe_int(ev.get(key)) for ev in bucket]
        return _mean_std([float(v) for v in values])

    for sigma in sorted(sigma_buckets.keys()):
        bucket = sigma_buckets[sigma]

        acc_m, acc_s = _collect("accuracy", bucket)
        prec_m, prec_s = _collect("precision", bucket)
        rec_m, rec_s = _collect("recall", bucket)
        f1_m, f1_s = _collect("f1", bucket)

        prec_mi_m, prec_mi_s = _collect("precision_micro", bucket)
        rec_mi_m, rec_mi_s = _collect("recall_micro", bucket)
        f1_mi_m, f1_mi_s = _collect("f1_micro", bucket)

        fp_m, fp_s = _collect_int("fp_total", bucket)
        fn_m, fn_s = _collect_int("fn_total", bucket)

        active_m, active_s = _collect("active_modules_mean", bucket)
        entropy_m, entropy_s = _collect("entropy_mean", bucket)
        flops_m, flops_s = _collect_int("flops_executed", bucket)
        lat_m, lat_s = _collect("latency_ms", bucket)
        k_mean_m, k_mean_s = _collect("k_mean", bucket)

        def _collect_degr(dkey: str) -> Tuple[float, float]:
            values = []
            for ev in bucket:
                degr = ev.get("degradation_pct", {}) if isinstance(ev, dict) else {}
                values.append(_safe_float(degr.get(dkey)))
            return _mean_std(values)

        degr_block = {
            "accuracy_pct_mean": 0.0,
            "accuracy_pct_std": 0.0,
            "precision_pct_mean": 0.0,
            "precision_pct_std": 0.0,
            "recall_pct_mean": 0.0,
            "recall_pct_std": 0.0,
            "f1_pct_mean": 0.0,
            "f1_pct_std": 0.0,
            "precision_micro_pct_mean": 0.0,
            "precision_micro_pct_std": 0.0,
            "recall_micro_pct_mean": 0.0,
            "recall_micro_pct_std": 0.0,
            "f1_micro_pct_mean": 0.0,
            "f1_micro_pct_std": 0.0,
        }

        for key in [
            "accuracy_pct",
            "precision_pct",
            "recall_pct",
            "f1_pct",
            "precision_micro_pct",
            "recall_micro_pct",
            "f1_micro_pct",
        ]:
            mean_val, std_val = _collect_degr(key)
            degr_block[f"{key}_mean"] = mean_val
            degr_block[f"{key}_std"] = std_val

        evaluations.append(
            {
                "noise_sigma": sigma,
                "n_runs": len(bucket),
                "accuracy_mean": acc_m,
                "accuracy_std": acc_s,
                "precision_mean": prec_m,
                "precision_std": prec_s,
                "recall_mean": rec_m,
                "recall_std": rec_s,
                "f1_mean": f1_m,
                "f1_std": f1_s,
                "precision_micro_mean": prec_mi_m,
                "precision_micro_std": prec_mi_s,
                "recall_micro_mean": rec_mi_m,
                "recall_micro_std": rec_mi_s,
                "f1_micro_mean": f1_mi_m,
                "f1_micro_std": f1_mi_s,
                "fp_total_mean": fp_m,
                "fp_total_std": fp_s,
                "fn_total_mean": fn_m,
                "fn_total_std": fn_s,
                "active_modules_mean_mean": active_m,
                "active_modules_mean_std": active_s,
                "entropy_mean_mean": entropy_m,
                "entropy_mean_std": entropy_s,
                "flops_executed_mean": flops_m,
                "flops_executed_std": flops_s,
                "latency_ms_mean": lat_m,
                "latency_ms_std": lat_s,
                "k_mean_mean": k_mean_m,
                "k_mean_std": k_mean_s,
                "degradation_pct": degr_block,
            }
        )

    return {
        "baseline_sigma": 0.0 if baseline_sigma is None else baseline_sigma,
        "evaluations": evaluations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one experiment config across multiple seeds and aggregate results.")
    parser.add_argument("--config", required=True, help="Path to a single YAML config file (e.g., configs/sbm_adaptive_k_mnist.yaml)")
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Comma list and/or ranges, e.g. '42,43,44,45,46' or '42-46'")
    parser.add_argument("--python", default=str(ROOT / ".venv" / "Scripts" / "python.exe"), help="Python executable to use")
    parser.add_argument("--out", default=str(ROOT / "results" / "aggregated_results.json"), help="Output JSON path for aggregated results")
    args = parser.parse_args()

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    if not config_path.exists():
        print(f"[FAIL] Config not found: {config_path}")
        return 2

    seeds = _parse_seeds(args.seeds)

    out_path = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTI-SEED RUNNER")
    print("=" * 80)
    print(f"Config: {config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path}")
    print(f"Seeds: {seeds}")
    print(f"Total runs: {len(seeds)}")
    print("")

    # Use a temporary directory for seed-specific YAMLs
    with tempfile.TemporaryDirectory(prefix="sbm_multiseed_") as td:
        tmp_dir = Path(td)

        per_seed_metrics: List[Dict[str, Any]] = []
        failures: List[Dict[str, Any]] = []

        for seed in seeds:
            cfg_seed = _set_seed_in_yaml(config_path, seed, tmp_dir)

            print(f"[*] Running seed {seed}...")
            ok, run_id, run_dir, err = _run_one(cfg_seed, seed, args.python)

            if not ok:
                print(f"  [FAIL] seed {seed}")
                if err:
                    print("  ---- error (tail) ----")
                    print(err)
                    print("  ----------------------")
                failures.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "run_dir": str(run_dir) if run_dir else None,
                        "error": err or "Unknown failure",
                    }
                )
                continue

            if run_dir is None:
                print(f"  [FAIL] seed {seed}: could not detect run_dir from stdout.")
                failures.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "run_dir": None,
                        "error": "Could not detect run_dir from stdout (expected 'Run dir:' or 'Metrics saved to:')",
                    }
                )
                continue

            # run_dir might be relative; resolve from project root
            run_dir_abs = (ROOT / run_dir).resolve() if not run_dir.is_absolute() else run_dir.resolve()

            try:
                metrics = _read_metrics(run_dir_abs)
                per_seed_metrics.append(metrics)
                acc = _safe_float(metrics.get("final", {}).get("accuracy"))
                print(f"  [OK] seed {seed} acc={100.0 * acc:.2f}%  run_dir={run_dir}")
            except Exception as e:
                print(f"  [FAIL] seed {seed}: could not read metrics.json: {e}")
                failures.append(
                    {
                        "seed": seed,
                        "run_id": run_id,
                        "run_dir": str(run_dir_abs),
                        "error": f"Could not read metrics.json: {e}",
                    }
                )

        print("")
        print("=" * 80)
        print("FINAL RESULTS (mean ± std over seeds)")
        print("=" * 80)

        if not per_seed_metrics:
            print("[FAIL] No successful runs. Nothing to aggregate.")
            # Still write failures to out
            payload = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "config": str(config_path),
                "seeds": seeds,
                "successes": 0,
                "failures": failures,
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[>] Failure report saved to {out_path}")
            return 3

        agg = _aggregate(per_seed_metrics)

        noise_agg = _aggregate_noise(per_seed_metrics)
        if noise_agg:
            agg["noise"] = noise_agg

        # Include run metadata and per-seed pointers
        model_name = str(per_seed_metrics[0].get("model", per_seed_metrics[0].get("run", {}).get("model", "unknown")))
        task_name = str(per_seed_metrics[0].get("task", per_seed_metrics[0].get("run", {}).get("task", "unknown")))

        summary_key = f"{task_name}_{model_name}"

        # Merge with existing aggregated file if present
        merged: Dict[str, Any] = {"results": {}}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if isinstance(existing, dict) and isinstance(existing.get("results"), dict):
                    merged = existing
            except Exception:
                pass

        if "results" not in merged or not isinstance(merged.get("results"), dict):
            merged["results"] = {}

        agg["seeds"] = seeds
        agg["config"] = str(config_path)
        agg["successes"] = len(per_seed_metrics)
        agg["failures"] = failures
        agg["model"] = model_name
        agg["task"] = task_name

        merged["results"][summary_key] = agg
        merged["timestamp"] = datetime.utcnow().isoformat() + "Z"

        print(f"Task/Model key: {summary_key}")
        print(f"Accuracy: {agg['accuracy_mean']:.6f} ± {agg['accuracy_std']:.6f}")
        print(f"Latency_ms: {agg['latency_mean']:.6f} ± {agg['latency_std']:.6f}")
        print(f"FLOPs: {agg['flops_mean']} ± {agg['flops_std']:.6f}")
        print(f"ActiveModules: {agg['active_modules_mean']:.6f} ± {agg['active_modules_std']:.6f}")
        print(f"Entropy: {agg['entropy_mean']:.6f} ± {agg['entropy_std']:.6f}")
        print(f"k_mean: {agg.get('k_mean', 0.0):.6f} ± {agg.get('k_mean_std', 0.0):.6f}")
        print(f"Precision: {agg.get('precision_mean', 0.0):.6f} ± {agg.get('precision_std', 0.0):.6f}")
        print(f"Recall: {agg.get('recall_mean', 0.0):.6f} ± {agg.get('recall_std', 0.0):.6f}")
        print(f"F1: {agg.get('f1_mean', 0.0):.6f} ± {agg.get('f1_std', 0.0):.6f}")
        print(f"Params: {agg['params']}")

        out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print("")
        print(f"[>] Aggregated results saved to {out_path}")

        # Return non-zero if any failures occurred (so CI can flag it), but keep it mild.
        return 0 if not failures else 4


if __name__ == "__main__":
    raise SystemExit(main())
