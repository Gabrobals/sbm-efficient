# scripts/analyze_entropy_buckets.py
"""
STEP 6A.2.a - Entropy bucket analysis (SBM project)

Goal:
- Analyze per-sample behavior as a function of routing entropy H(x)
- Produce bucket tables with: count, accuracy, k_mean, flops_mean, etc.

Input:
- One or more run directories under results/runs/<run_id>/
- Each run must contain:
  - metrics.json
  - OPTIONAL (but recommended): per-sample stats file, one of:
      * per_sample_stats.jsonl
      * per_sample_stats.csv
      * analysis/per_sample_stats.jsonl
      * analysis/per_sample_stats.csv

Expected per-sample fields (minimum):
- entropy: float
- correct: int (0/1)
- k: float or int (active modules for that sample)  [optional but recommended]
- flops: int                                      [optional]
- latency_ms: float                               [optional]

If per-sample file is missing, the script will stop with a clear message.
This is intentional: bucket analysis must be evidence-based.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class SampleStat:
    entropy: float
    correct: int
    k: Optional[float] = None
    flops: Optional[int] = None
    latency_ms: Optional[float] = None


@dataclass
class BucketAgg:
    bucket: str
    n: int
    acc: float
    entropy_mean: float
    entropy_std: float
    k_mean: Optional[float]
    k_std: Optional[float]
    flops_mean: Optional[float]
    flops_std: Optional[float]
    latency_mean_ms: Optional[float]
    latency_std_ms: Optional[float]


# -----------------------------
# Helpers
# -----------------------------
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _try_load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_per_sample_file(run_dir: Path) -> Optional[Path]:
    candidates = [
        run_dir / "per_sample_stats.jsonl",
        run_dir / "per_sample_stats.csv",
        run_dir / "analysis" / "per_sample_stats.jsonl",
        run_dir / "analysis" / "per_sample_stats.csv",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _load_samples_jsonl(path: Path) -> List[SampleStat]:
    out: List[SampleStat] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                entropy = float(obj["entropy"])
                correct = int(obj["correct"])
                k = float(obj["k"]) if "k" in obj and obj["k"] is not None else None
                flops = int(obj["flops"]) if "flops" in obj and obj["flops"] is not None else None
                latency_ms = float(obj["latency_ms"]) if "latency_ms" in obj and obj["latency_ms"] is not None else None
                out.append(SampleStat(entropy=entropy, correct=correct, k=k, flops=flops, latency_ms=latency_ms))
            except Exception as e:
                raise RuntimeError(f"Failed parsing JSONL at {path} line {line_no}: {e}") from e
    return out


def _load_samples_csv(path: Path) -> List[SampleStat]:
    out: List[SampleStat] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"entropy", "correct"}
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        missing = required - set(reader.fieldnames)
        if missing:
            raise RuntimeError(f"CSV missing required columns {sorted(missing)}: {path}")

        for row_no, row in enumerate(reader, start=2):
            try:
                entropy = float(row["entropy"])
                correct = int(row["correct"])
                k = float(row["k"]) if "k" in row and row["k"] not in (None, "", "null") else None
                flops = int(float(row["flops"])) if "flops" in row and row["flops"] not in (None, "", "null") else None
                latency_ms = float(row["latency_ms"]) if "latency_ms" in row and row["latency_ms"] not in (None, "", "null") else None
                out.append(SampleStat(entropy=entropy, correct=correct, k=k, flops=flops, latency_ms=latency_ms))
            except Exception as e:
                raise RuntimeError(f"Failed parsing CSV at {path} row {row_no}: {e}") from e
    return out


def load_samples_for_run(run_dir: Path) -> Tuple[Dict, List[SampleStat], Path]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Missing metrics.json in run_dir: {run_dir}")

    metrics = _try_load_json(metrics_path)
    ps_path = _find_per_sample_file(run_dir)

    if ps_path is None:
        raise RuntimeError(
            "Per-sample stats file not found.\n"
            f"Run: {run_dir}\n"
            "Expected one of:\n"
            "  - per_sample_stats.jsonl / .csv\n"
            "  - analysis/per_sample_stats.jsonl / .csv\n"
            "\n"
            "This analysis requires per-sample entropy. Please export per-sample stats during evaluation."
        )

    if ps_path.suffix.lower() == ".jsonl":
        samples = _load_samples_jsonl(ps_path)
    elif ps_path.suffix.lower() == ".csv":
        samples = _load_samples_csv(ps_path)
    else:
        raise RuntimeError(f"Unsupported per-sample file type: {ps_path}")

    return metrics, samples, ps_path


def bucketize(samples: List[SampleStat], edges: List[float]) -> List[Tuple[str, List[SampleStat]]]:
    """
    edges: sorted list, e.g. [0.3, 0.6, 1.0]
    buckets:
      (-inf, e0), [e0, e1), [e1, e2), [e2, +inf)
    """
    edges = sorted(edges)
    buckets: List[Tuple[str, List[SampleStat]]] = []
    bounds: List[Tuple[float, float]] = []

    prev = float("-inf")
    for e in edges:
        bounds.append((prev, e))
        prev = e
    bounds.append((prev, float("inf")))

    for lo, hi in bounds:
        label = f"H in [{lo:.3f}, {hi:.3f})" if math.isfinite(hi) else f"H in [{lo:.3f}, +inf)"
        buckets.append((label, []))

    for s in samples:
        placed = False
        for i, (lo, hi) in enumerate(bounds):
            if s.entropy >= lo and s.entropy < hi:
                buckets[i][1].append(s)
                placed = True
                break
        if not placed:
            buckets[-1][1].append(s)

    return buckets


def aggregate_bucket(label: str, items: List[SampleStat]) -> BucketAgg:
    n = len(items)
    if n == 0:
        return BucketAgg(
            bucket=label, n=0, acc=float("nan"),
            entropy_mean=float("nan"), entropy_std=float("nan"),
            k_mean=None, k_std=None,
            flops_mean=None, flops_std=None,
            latency_mean_ms=None, latency_std_ms=None,
        )

    corrects = [s.correct for s in items]
    acc = sum(corrects) / n

    entropies = [s.entropy for s in items]
    entropy_mean = _mean(entropies)
    entropy_std = _std(entropies)

    ks = [s.k for s in items if s.k is not None]
    flops = [float(s.flops) for s in items if s.flops is not None]
    lats = [s.latency_ms for s in items if s.latency_ms is not None]

    k_mean = _mean(ks) if ks else None
    k_std = _std(ks) if ks else None

    flops_mean = _mean(flops) if flops else None
    flops_std = _std(flops) if flops else None

    lat_mean = _mean(lats) if lats else None
    lat_std = _std(lats) if lats else None

    return BucketAgg(
        bucket=label, n=n, acc=acc,
        entropy_mean=entropy_mean, entropy_std=entropy_std,
        k_mean=k_mean, k_std=k_std,
        flops_mean=flops_mean, flops_std=flops_std,
        latency_mean_ms=lat_mean, latency_std_ms=lat_std,
    )


def save_csv(rows: List[BucketAgg], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket", "n", "acc",
        "entropy_mean", "entropy_std",
        "k_mean", "k_std",
        "flops_mean", "flops_std",
        "latency_mean_ms", "latency_std_ms"
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def save_json(payload: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=str, default="results/runs", help="Root folder containing run directories.")
    ap.add_argument("--match", type=str, default="*", help="Glob for run dir names, e.g. '*mnist_sbm_adaptive_k*'")
    ap.add_argument("--entropy-bins", type=str, default="0.3,0.6,1.0",
                    help="Comma-separated entropy bin edges, e.g. '0.3,0.6,1.0'")
    ap.add_argument("--out-dir", type=str, default="results/summaries/entropy_buckets",
                    help="Output directory for summaries.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)

    edges = [float(x.strip()) for x in args.entropy_bins.split(",") if x.strip()]

    run_dirs = sorted([p for p in runs_root.glob(args.match) if p.is_dir()])
    if not run_dirs:
        print(f"[FAIL] No run directories found under {runs_root} matching: {args.match}")
        return 2

    all_results = []

    for run_dir in run_dirs:
        try:
            metrics, samples, ps_path = load_samples_for_run(run_dir)
        except Exception as e:
            print(f"[SKIP] {run_dir.name}: {e}")
            continue

        run_id = metrics.get("run_id", run_dir.name)
        model = metrics.get("model", metrics.get("run", {}).get("model", "unknown"))
        task = metrics.get("task", metrics.get("run", {}).get("task", "unknown"))
        seed = metrics.get("seed", metrics.get("run", {}).get("seed", None))

        buckets = bucketize(samples, edges)
        bucket_rows = [aggregate_bucket(label, items) for (label, items) in buckets]

        payload = {
            "run_id": run_id,
            "task": task,
            "model": model,
            "seed": seed,
            "per_sample_source": str(ps_path).replace("\\", "/"),
            "entropy_bins": edges,
            "buckets": [asdict(r) for r in bucket_rows],
            "final": metrics.get("final", {}),
        }

        out_json = out_dir / f"{run_id}_entropy_buckets.json"
        out_csv = out_dir / f"{run_id}_entropy_buckets.csv"

        save_json(payload, out_json)
        save_csv(bucket_rows, out_csv)

        print(f"[OK] {run_dir.name}")
        print(f"     per-sample: {ps_path}")
        print(f"     out json : {out_json}")
        print(f"     out csv  : {out_csv}")

        all_results.append(payload)

    if not all_results:
        print("[FAIL] No runs produced bucket outputs (missing per-sample stats in all runs).")
        return 3

    # Combined summary
    combined = {
        "runs_root": str(runs_root).replace("\\", "/"),
        "match": args.match,
        "entropy_bins": edges,
        "n_runs_analyzed": len(all_results),
        "runs": [{"run_id": r["run_id"], "task": r["task"], "model": r["model"], "seed": r["seed"]} for r in all_results]
    }
    save_json(combined, out_dir / "_index.json")
    print(f"[OK] Index saved: {out_dir / '_index.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
