"""
Compare experiment results across all runs.

Usage:
    python -m scripts.compare_results
"""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def format_results_table(results: List[Dict]) -> str:
    """Format results as a nice table."""
    if not results:
        return "No results found."
    
    # Header
    header = (
        f"{'Task':<12} {'Model':<16} {'Accuracy':>10} {'Entropy':>10} "
        f"{'Params':>12} {'Latency':>10}"
    )
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    # Sort by task then accuracy
    results.sort(key=lambda x: (x['task'], -x['accuracy']))
    
    current_task = None
    for r in results:
        if current_task and r['task'] != current_task:
            lines.append(separator)
        current_task = r['task']
        
        acc_str = f"{r['accuracy']*100:.2f}%"
        entropy_str = f"{r['entropy']:.4f}" if r['entropy'] else "-"
        params_str = f"{r['params']:,}"
        latency_str = f"{r['latency']:.3f}ms" if r['latency'] else "-"
        
        line = (
            f"{r['task']:<12} {r['model']:<16} {acc_str:>10} {entropy_str:>10} "
            f"{params_str:>12} {latency_str:>10}"
        )
        lines.append(line)
    
    lines.append(separator)
    
    return "\n".join(lines)


def main():
    runs_dir = Path("results/runs")
    
    if not runs_dir.exists():
        print("No results directory found.")
        return
    
    results = []
    
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        
        metrics = load_metrics(run_dir)
        if metrics is None:
            continue
        
        # Extract key info
        result = {
            "run_id": metrics.get("run_id", run_dir.name),
            "task": metrics.get("task", "unknown"),
            "model": metrics.get("model", "unknown"),
            "accuracy": metrics.get("final", {}).get("accuracy", 0),
            "entropy": metrics.get("final", {}).get("entropy_mean", 0),
            "params": metrics.get("training", {}).get("parameter_count", 0),
            "latency": metrics.get("final", {}).get("latency_ms", 0),
            "flops": metrics.get("final", {}).get("flops_executed", 0),
        }
        results.append(result)
    
    print("\n" + "=" * 80)
    print("SBM-EFFICIENT EXPERIMENT RESULTS")
    print("=" * 80 + "\n")
    
    print(format_results_table(results))
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY BY TASK")
    print("=" * 80 + "\n")
    
    tasks = set(r['task'] for r in results)
    for task in sorted(tasks):
        task_results = [r for r in results if r['task'] == task]
        best = max(task_results, key=lambda x: x['accuracy'])
        
        print(f"📊 {task.upper()}")
        print(f"   Best model: {best['model']} ({best['accuracy']*100:.2f}%)")
        
        # Compare SBM vs random
        sbm = next((r for r in task_results if r['model'] == 'sbm'), None)
        random = next((r for r in task_results if r['model'] == 'random_routing'), None)
        
        if sbm and random:
            delta = (sbm['accuracy'] - random['accuracy']) * 100
            print(f"   SBM vs Random: {delta:+.2f}% improvement")
        
        baseline = next((r for r in task_results if r['model'] == 'baseline'), None)
        if baseline:
            print(f"   Baseline: {baseline['accuracy']*100:.2f}%")
        
        print()
    
    # Export to JSON
    export_path = Path("results/comparison.json")
    with open(export_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"📁 Results exported to {export_path}")


if __name__ == "__main__":
    main()
