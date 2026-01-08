"""
Validation schema and functions for experiment metrics JSON files.

Exit codes:
    0 = OK
    3 = metrics invalid
    4 = consistency invalid
    5 = missing files
"""

from typing import Any, Dict, List, Tuple
import json
from pathlib import Path


REQUIRED_ROOT_FIELDS = {
    "run_id": str,
    "timestamp": str,
    "git": dict,
    "config_path": str,
    "task": str,
    "model": str,
    "seed": int,
}

REQUIRED_GIT_FIELDS = {
    "sha": str,
    "dirty": bool,
}

REQUIRED_FINAL_FIELDS = {
    "accuracy": (int, float),
    "loss": (int, float),
    "flops_executed": int,
    "latency_ms": (int, float),
    "active_modules_mean": (int, float),
    "entropy_mean": (int, float),
}

# Classification metrics required for Fase A (all models)
CLASSIFICATION_FIELDS = {
    "precision": (int, float),
    "recall": (int, float),
    "f1": (int, float),
    "precision_micro": (int, float),
    "recall_micro": (int, float),
    "f1_micro": (int, float),
    "tp_total": int,
    "fp_total": int,
    "fn_total": int,
    "tn_total": int,
    "confusion_matrix": list,
}

# Noise robustness (optional section "noise")
NOISE_EVAL_FIELDS = {
    "noise_sigma": (int, float),
    "accuracy": (int, float),
    "loss": (int, float),
    "flops_executed": int,
    "active_modules_mean": (int, float),
    "entropy_mean": (int, float),
    "latency_ms": (int, float),
    "precision": (int, float),
    "recall": (int, float),
    "f1": (int, float),
    "precision_micro": (int, float),
    "recall_micro": (int, float),
    "f1_micro": (int, float),
    "tp_total": int,
    "fp_total": int,
    "fn_total": int,
    "tn_total": int,
    "confusion_matrix": list,
}

NOISE_DEGRADATION_FIELDS = {
    "accuracy_pct": (int, float),
    "precision_pct": (int, float),
    "recall_pct": (int, float),
    "f1_pct": (int, float),
    "precision_micro_pct": (int, float),
    "recall_micro_pct": (int, float),
    "f1_micro_pct": (int, float),
}

# Input robustness (optional section "robustness_input")
INPUT_ROBUST_BASELINE_FIELDS = {
    "accuracy": (int, float),
    "precision": (int, float),
    "recall": (int, float),
    "f1": (int, float),
    "precision_micro": (int, float),
    "recall_micro": (int, float),
    "f1_micro": (int, float),
}

INPUT_ROBUST_EVAL_FIELDS = {
    "test": str,  # "gaussian" | "salt_pepper" | "occlusion" | "inversion"
    "level": (int, float),
    "accuracy": (int, float),
    "loss": (int, float),
    "flops_executed": int,
    "active_modules_mean": (int, float),
    "entropy_mean": (int, float),
    "latency_ms": (int, float),
    "precision": (int, float),
    "recall": (int, float),
    "f1": (int, float),
    "precision_micro": (int, float),
    "recall_micro": (int, float),
    "f1_micro": (int, float),
    "tp_total": int,
    "fp_total": int,
    "fn_total": int,
    "tn_total": int,
    "k_mean": (int, float),
    "k_std": (int, float),
}

# Adaptive-K specific required fields (only when model == "sbm_adaptive_k")
ADAPTIVE_FINAL_FIELDS = {
    "k_mean": (int, float),
    "k_std": (int, float),
    "k_histogram": dict,
    "k_histogram_pct": dict,
}

REQUIRED_PROFILE_FIELDS = {
    "warmup_steps": int,
    "timed_steps": int,
    "latency_p50_ms": (int, float),
    "latency_p90_ms": (int, float),
    "latency_p99_ms": (int, float),
}


def validate_metrics_schema(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate metrics JSON against required schema.
    
    Args:
        metrics: Parsed metrics dictionary
        
    Returns:
        (is_valid, errors) tuple
    """
    errors = []
    
    # C1.1 - Root fields
    for field, expected_type in REQUIRED_ROOT_FIELDS.items():
        if field not in metrics:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(metrics[field], expected_type):
            errors.append(
                f"{field} must be {expected_type.__name__}, "
                f"got: {type(metrics[field]).__name__}"
            )
    
    # C1.2 - Git metadata
    if "git" in metrics:
        git = metrics["git"]
        for field, expected_type in REQUIRED_GIT_FIELDS.items():
            if field not in git:
                errors.append(f"Missing required field: git.{field}")
            elif not isinstance(git[field], expected_type):
                errors.append(
                    f"git.{field} must be {expected_type.__name__}, "
                    f"got: {type(git[field]).__name__}"
                )
    
    # C1.3 - Final metrics
    if "final" not in metrics:
        errors.append("Missing required section: final")
    else:
        final = metrics["final"]
        for field, expected_types in REQUIRED_FINAL_FIELDS.items():
            if field not in final:
                errors.append(f"Missing required field: final.{field}")
            elif not isinstance(final[field], expected_types):
                type_names = (expected_types.__name__ if isinstance(expected_types, type) 
                             else " or ".join(t.__name__ for t in expected_types))
                errors.append(
                    f"final.{field} must be {type_names}, "
                    f"got: {type(final[field]).__name__}"
                )
            # Non-negative validation for certain fields
            elif field in {"flops_executed", "latency_ms", "active_modules_mean", "entropy_mean"}:
                if final[field] < 0:
                    errors.append(f"final.{field} must be >= 0, got: {final[field]}")

        # Classification metrics (required for all models per Fase A)
        for field, expected_types in CLASSIFICATION_FIELDS.items():
            if field not in final:
                errors.append(f"Missing required field: final.{field}")
            elif not isinstance(final[field], expected_types):
                type_names = (expected_types.__name__ if isinstance(expected_types, type)
                             else " or ".join(t.__name__ for t in expected_types))
                errors.append(
                    f"final.{field} must be {type_names}, "
                    f"got: {type(final[field]).__name__}"
                )
            elif field in {"precision", "recall", "f1", "precision_micro", "recall_micro", "f1_micro"}:
                if not (0.0 <= final[field] <= 1.0):
                    errors.append(f"final.{field} should be in [0,1], got: {final[field]}")
            elif field in {"tp_total", "fp_total", "fn_total", "tn_total"}:
                if final[field] < 0:
                    errors.append(f"final.{field} must be >= 0, got: {final[field]}")

        # Confusion matrix sanity (if present)
        cm = final.get("confusion_matrix")
        if isinstance(cm, list):
            # Must be square
            rows = len(cm)
            if rows == 0:
                errors.append("final.confusion_matrix must not be empty")
            else:
                for row in cm:
                    if not isinstance(row, list) or len(row) != rows:
                        errors.append("final.confusion_matrix must be square (list of equal-length lists)")
                        break
        else:
            errors.append("final.confusion_matrix must be a list")

        # Adaptive-K required fields (only when applicable)
        if metrics.get("model") == "sbm_adaptive_k":
            for field, expected_types in ADAPTIVE_FINAL_FIELDS.items():
                if field not in final:
                    errors.append(f"Missing required field: final.{field} (adaptive_k)")
                elif not isinstance(final[field], expected_types):
                    type_names = (expected_types.__name__ if isinstance(expected_types, type)
                                 else " or ".join(t.__name__ for t in expected_types))
                    errors.append(
                        f"final.{field} must be {type_names}, "
                        f"got: {type(final[field]).__name__}"
                    )
    
    # C1.4 - Profile metrics
    if "profile" not in metrics:
        errors.append("Missing required section: profile")
    else:
        profile = metrics["profile"]
        for field, expected_types in REQUIRED_PROFILE_FIELDS.items():
            if field not in profile:
                errors.append(f"Missing required field: profile.{field}")
            elif not isinstance(profile[field], expected_types):
                type_names = (expected_types.__name__ if isinstance(expected_types, type)
                             else " or ".join(t.__name__ for t in expected_types))
                errors.append(
                    f"profile.{field} must be {type_names}, "
                    f"got: {type(profile[field]).__name__}"
                )
            # Non-negative for latency metrics
            elif "latency" in field and profile[field] < 0:
                errors.append(f"profile.{field} must be >= 0, got: {profile[field]}")

    # C1.5 - Optional noise robustness section
    if "noise" in metrics:
        noise = metrics.get("noise")
        if not isinstance(noise, dict):
            errors.append("noise must be a dict when present")
        else:
            if "baseline_sigma" not in noise:
                errors.append("Missing required field: noise.baseline_sigma")
            elif not isinstance(noise.get("baseline_sigma"), (int, float)):
                errors.append(
                    f"noise.baseline_sigma must be int or float, got: {type(noise.get('baseline_sigma')).__name__}"
                )

            evaluations = noise.get("evaluations")
            if evaluations is None:
                errors.append("Missing required field: noise.evaluations")
            elif not isinstance(evaluations, list):
                errors.append("noise.evaluations must be a list")
            elif len(evaluations) == 0:
                errors.append("noise.evaluations must not be empty")
            else:
                for idx, ev in enumerate(evaluations):
                    prefix = f"noise.evaluations[{idx}]"
                    if not isinstance(ev, dict):
                        errors.append(f"{prefix} must be a dict")
                        continue

                    for field, expected_types in NOISE_EVAL_FIELDS.items():
                        if field not in ev:
                            errors.append(f"Missing required field: {prefix}.{field}")
                            continue
                        if not isinstance(ev[field], expected_types):
                            type_names = (
                                expected_types.__name__
                                if isinstance(expected_types, type)
                                else " or ".join(t.__name__ for t in expected_types)
                            )
                            errors.append(
                                f"{prefix}.{field} must be {type_names}, got: {type(ev[field]).__name__}"
                            )
                            continue

                        if field in {
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "precision_micro",
                            "recall_micro",
                            "f1_micro",
                        } and not (0.0 <= ev[field] <= 1.0):
                            errors.append(f"{prefix}.{field} should be in [0,1], got: {ev[field]}")

                        if field in {
                            "flops_executed",
                            "active_modules_mean",
                            "entropy_mean",
                            "latency_ms",
                        } and ev[field] < 0:
                            errors.append(f"{prefix}.{field} must be >= 0, got: {ev[field]}")

                        if field in {"tp_total", "fp_total", "fn_total", "tn_total"} and ev[field] < 0:
                            errors.append(f"{prefix}.{field} must be >= 0, got: {ev[field]}")

                    # Per-sigma confusion matrix must be square when present
                    cm = ev.get("confusion_matrix")
                    if isinstance(cm, list):
                        rows = len(cm)
                        if rows == 0:
                            errors.append(f"{prefix}.confusion_matrix must not be empty")
                        else:
                            for row in cm:
                                if not isinstance(row, list) or len(row) != rows:
                                    errors.append(
                                        f"{prefix}.confusion_matrix must be square (list of equal-length lists)"
                                    )
                                    break
                    else:
                        errors.append(f"{prefix}.confusion_matrix must be a list")

                    # Degradation block required
                    degr = ev.get("degradation_pct")
                    if degr is None:
                        errors.append(f"Missing required field: {prefix}.degradation_pct")
                    elif not isinstance(degr, dict):
                        errors.append(f"{prefix}.degradation_pct must be a dict")
                    else:
                        for dfield, expected_types in NOISE_DEGRADATION_FIELDS.items():
                            if dfield not in degr:
                                errors.append(f"Missing required field: {prefix}.degradation_pct.{dfield}")
                            elif not isinstance(degr[dfield], expected_types):
                                type_names = (
                                    expected_types.__name__
                                    if isinstance(expected_types, type)
                                    else " or ".join(t.__name__ for t in expected_types)
                                )
                                errors.append(
                                    f"{prefix}.degradation_pct.{dfield} must be {type_names}, got: {type(degr[dfield]).__name__}"
                                )

    # C1.6 - Optional input robustness section (B2)
    if "robustness_input" in metrics:
        robust_input = metrics.get("robustness_input")
        if not isinstance(robust_input, dict):
            errors.append("robustness_input must be a dict when present")
        else:
            # Baseline block
            baseline = robust_input.get("baseline")
            if baseline is None:
                errors.append("Missing required field: robustness_input.baseline")
            elif not isinstance(baseline, dict):
                errors.append("robustness_input.baseline must be a dict")
            else:
                for field, expected_types in INPUT_ROBUST_BASELINE_FIELDS.items():
                    if field not in baseline:
                        errors.append(f"Missing required field: robustness_input.baseline.{field}")
                    elif not isinstance(baseline[field], expected_types):
                        type_names = (
                            expected_types.__name__
                            if isinstance(expected_types, type)
                            else " or ".join(t.__name__ for t in expected_types)
                        )
                        errors.append(
                            f"robustness_input.baseline.{field} must be {type_names}, got: {type(baseline[field]).__name__}"
                        )
                    elif field in {"accuracy", "precision", "recall", "f1", "precision_micro", "recall_micro", "f1_micro"}:
                        if not (0.0 <= baseline[field] <= 1.0):
                            errors.append(f"robustness_input.baseline.{field} should be in [0,1], got: {baseline[field]}")

            # Evaluations list
            evaluations = robust_input.get("evaluations")
            if evaluations is None:
                errors.append("Missing required field: robustness_input.evaluations")
            elif not isinstance(evaluations, list):
                errors.append("robustness_input.evaluations must be a list")
            elif len(evaluations) == 0:
                errors.append("robustness_input.evaluations must not be empty")
            else:
                valid_tests = {"gaussian", "salt_pepper", "occlusion", "inversion"}
                for idx, ev in enumerate(evaluations):
                    prefix = f"robustness_input.evaluations[{idx}]"
                    if not isinstance(ev, dict):
                        errors.append(f"{prefix} must be a dict")
                        continue

                    for field, expected_types in INPUT_ROBUST_EVAL_FIELDS.items():
                        if field not in ev:
                            errors.append(f"Missing required field: {prefix}.{field}")
                            continue
                        if not isinstance(ev[field], expected_types):
                            type_names = (
                                expected_types.__name__
                                if isinstance(expected_types, type)
                                else " or ".join(t.__name__ for t in expected_types)
                            )
                            errors.append(
                                f"{prefix}.{field} must be {type_names}, got: {type(ev[field]).__name__}"
                            )
                            continue

                        if field == "test" and ev[field] not in valid_tests:
                            errors.append(f"{prefix}.test must be one of {valid_tests}, got: {ev[field]}")

                        if field in {
                            "accuracy",
                            "precision",
                            "recall",
                            "f1",
                            "precision_micro",
                            "recall_micro",
                            "f1_micro",
                        } and not (0.0 <= ev[field] <= 1.0):
                            errors.append(f"{prefix}.{field} should be in [0,1], got: {ev[field]}")

                        if field in {
                            "flops_executed",
                            "active_modules_mean",
                            "entropy_mean",
                            "latency_ms",
                        } and ev[field] < 0:
                            errors.append(f"{prefix}.{field} must be >= 0, got: {ev[field]}")

                        if field in {"tp_total", "fp_total", "fn_total", "tn_total"} and ev[field] < 0:
                            errors.append(f"{prefix}.{field} must be >= 0, got: {ev[field]}")

                        if field == "level" and ev[field] < 0:
                            errors.append(f"{prefix}.level must be >= 0, got: {ev[field]}")

                    # Degradation block required
                    degr = ev.get("degradation_pct")
                    if degr is None:
                        errors.append(f"Missing required field: {prefix}.degradation_pct")
                    elif not isinstance(degr, dict):
                        errors.append(f"{prefix}.degradation_pct must be a dict")
                    else:
                        for dfield, expected_types in NOISE_DEGRADATION_FIELDS.items():
                            if dfield not in degr:
                                errors.append(f"Missing required field: {prefix}.degradation_pct.{dfield}")
                            elif not isinstance(degr[dfield], expected_types):
                                type_names = (
                                    expected_types.__name__
                                    if isinstance(expected_types, type)
                                    else " or ".join(t.__name__ for t in expected_types)
                                )
                                errors.append(
                                    f"{prefix}.degradation_pct.{dfield} must be {type_names}, got: {type(degr[dfield]).__name__}"
                                )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_metrics_consistency(metrics: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate internal consistency of metrics.
    
    Args:
        metrics: Parsed metrics dictionary
        config: Parsed config dictionary
        
    Returns:
        (is_valid, errors) tuple
    """
    errors = []
    
    model = metrics.get("model", "")
    final = metrics.get("final", {})
    
    # C2.1 - Baseline should use all modules
    if model == "baseline":
        if "sbm" in config:
            experts_num = config["sbm"].get("experts_num", 0)
            active_mean = final.get("active_modules_mean", 0)
            
            # Allow some tolerance for floating point
            if experts_num > 0 and abs(active_mean - experts_num) > 0.1:
                errors.append(
                    f"Model 'baseline' should have active_modules_mean ≈ experts_num "
                    f"({experts_num}), got: {active_mean}"
                )
    
    # C2.2 - Sparse routing models should use K modules
    if model in {"sbm", "random_routing", "static_topk"}:
        if "sbm" in config:
            experts_top_k = config["sbm"].get("experts_top_k", 0)
            active_mean = final.get("active_modules_mean", 0)
            
            # Allow some tolerance (mean might vary slightly)
            if experts_top_k > 0 and abs(active_mean - experts_top_k) > 0.5:
                errors.append(
                    f"Model '{model}' with experts_top_k={experts_top_k} "
                    f"should have active_modules_mean ≈ {experts_top_k}, got: {active_mean}"
                )

    # C2.2b - Adaptive-K coherence checks
    if model == "sbm_adaptive_k":
        k_mean = final.get("k_mean")
        k_std = final.get("k_std")
        k_hist = final.get("k_histogram", {})
        k_hist_pct = final.get("k_histogram_pct", {})
        active_mean = final.get("active_modules_mean", 0)

        if k_mean is None or k_std is None:
            errors.append("Adaptive-K requires final.k_mean and final.k_std")
        else:
            if abs(active_mean - k_mean) > 0.25:
                errors.append(
                    f"Adaptive-K active_modules_mean should match k_mean (tolerance 0.25); "
                    f"got active_modules_mean={active_mean}, k_mean={k_mean}"
                )

        if not k_hist:
            errors.append("Adaptive-K requires non-empty final.k_histogram")
        else:
            # Ensure keys are strings and counts are non-negative
            for k_str, count in k_hist.items():
                if not isinstance(k_str, str):
                    errors.append("k_histogram keys must be strings")
                    break
                if count < 0:
                    errors.append(f"k_histogram[{k_str}] must be >= 0")

        if not k_hist_pct:
            errors.append("Adaptive-K requires final.k_histogram_pct with percentages")
        else:
            pct_sum = sum(k_hist_pct.values())
            if pct_sum > 0 and abs(pct_sum - 100.0) > 5.0:
                errors.append(
                    f"k_histogram_pct should sum near 100 (±5); got {pct_sum:.2f}"
                )

    # C2.5 - Classification metrics coherence
    final = metrics.get("final", {})
    cm = final.get("confusion_matrix")
    if isinstance(cm, list) and cm:
        try:
            total_cm = sum(sum(int(x) for x in row) for row in cm)
            if total_cm <= 0:
                errors.append("confusion_matrix sum must be > 0")
        except Exception:
            errors.append("confusion_matrix contains non-numeric values")
    
    # C2.3 - Profiling enabled implies measurements exist
    if "profiling" in config and config["profiling"].get("enabled", False):
        if final.get("flops_executed", 0) == 0:
            errors.append("Profiling enabled but final.flops_executed is 0 or missing")
        
        if final.get("latency_ms", 0) == 0:
            errors.append("Profiling enabled but final.latency_ms is 0 or missing")
    
    # C2.4 - Latency percentiles should be ordered
    profile = metrics.get("profile", {})
    p50 = profile.get("latency_p50_ms", 0)
    p90 = profile.get("latency_p90_ms", 0)
    p99 = profile.get("latency_p99_ms", 0)
    
    if p50 > 0 and p90 > 0 and p50 > p90:
        errors.append(f"latency_p50_ms ({p50}) should be <= latency_p90_ms ({p90})")
    
    if p90 > 0 and p99 > 0 and p90 > p99:
        errors.append(f"latency_p90_ms ({p90}) should be <= latency_p99_ms ({p99})")

    # C2.6 - Noise robustness consistency (when present)
    noise = metrics.get("noise")
    if isinstance(noise, dict) and noise.get("evaluations"):
        baseline_sigma = noise.get("baseline_sigma", 0.0)
        if baseline_sigma != 0.0:
            errors.append(f"noise.baseline_sigma expected 0.0 for reference, got {baseline_sigma}")

        base_acc = float(final.get("accuracy", 0.0))
        base_prec = float(final.get("precision", 0.0))
        base_rec = float(final.get("recall", 0.0))
        base_f1 = float(final.get("f1", 0.0))
        base_prec_mi = float(final.get("precision_micro", 0.0))
        base_rec_mi = float(final.get("recall_micro", 0.0))
        base_f1_mi = float(final.get("f1_micro", 0.0))
        base_flops = final.get("flops_executed", 0)

        def _deg(base: float, noisy: float) -> float:
            if base == 0:
                return 0.0
            return (base - noisy) / abs(base) * 100.0

        for ev in noise.get("evaluations", []):
            if not isinstance(ev, dict):
                continue

            ev_flops = ev.get("flops_executed", base_flops)
            if isinstance(ev_flops, (int, float)) and isinstance(base_flops, (int, float)):
                if abs(float(ev_flops) - float(base_flops)) > 1.0:
                    errors.append(
                        f"noise sigma={ev.get('noise_sigma')} flops_executed should match baseline ({base_flops}), got {ev_flops}"
                    )

            degr = ev.get("degradation_pct", {})
            if isinstance(degr, dict):
                checks = [
                    ("accuracy_pct", _deg(base_acc, float(ev.get("accuracy", 0.0)))),
                    ("precision_pct", _deg(base_prec, float(ev.get("precision", 0.0)))),
                    ("recall_pct", _deg(base_rec, float(ev.get("recall", 0.0)))),
                    ("f1_pct", _deg(base_f1, float(ev.get("f1", 0.0)))),
                    ("precision_micro_pct", _deg(base_prec_mi, float(ev.get("precision_micro", 0.0)))),
                    ("recall_micro_pct", _deg(base_rec_mi, float(ev.get("recall_micro", 0.0)))),
                    ("f1_micro_pct", _deg(base_f1_mi, float(ev.get("f1_micro", 0.0)))),
                ]
                for key, expected in checks:
                    actual = degr.get(key)
                    if actual is None:
                        continue
                    try:
                        if abs(float(actual) - float(expected)) > 1e-3:
                            errors.append(
                                f"noise sigma={ev.get('noise_sigma')} degradation_pct.{key} mismatch (got {actual}, expected {expected:.6f})"
                            )
                    except Exception:
                        errors.append(
                            f"noise sigma={ev.get('noise_sigma')} degradation_pct.{key} not comparable to expected value"
                        )
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_run_directory(run_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate complete run directory structure.
    
    Args:
        run_dir: Path to run directory (results/runs/<run_id>/)
        
    Returns:
        (is_valid, errors) tuple
    """
    errors = []
    
    # C3.1 - Check required files exist
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.yaml"
    
    if not metrics_path.exists():
        errors.append(f"Missing required file: {metrics_path}")
        return False, errors
    
    if not config_path.exists():
        errors.append(f"Missing required file: {config_path}")
        return False, errors
    
    # Load files
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"JSON parsing error in metrics.json: {e}"]
    except Exception as e:
        return False, [f"Error loading metrics.json: {e}"]
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"YAML parsing error in config.yaml: {e}"]
    except Exception as e:
        return False, [f"Error loading config.yaml: {e}"]
    
    # Validate schema
    schema_valid, schema_errors = validate_metrics_schema(metrics)
    if not schema_valid:
        errors.extend(schema_errors)
    
    # Validate consistency
    consistency_valid, consistency_errors = validate_metrics_consistency(metrics, config)
    if not consistency_valid:
        errors.extend(consistency_errors)
    
    is_valid = len(errors) == 0
    return is_valid, errors


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python -m src.common.validate_metrics <run_directory>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    
    if not run_dir.exists():
        print(f"[FAIL] RUN DIRECTORY NOT FOUND: {run_dir}")
        sys.exit(5)
    
    is_valid, errors = validate_run_directory(run_dir)
    
    if is_valid:
        print("[OK] METRICS VALID")
        sys.exit(0)
    else:
        print("[FAIL] METRICS INVALID")
        for error in errors:
            print(f"  - {error}")
        
        # Determine exit code based on error type
        if any("Missing required file" in e for e in errors):
            sys.exit(5)
        elif any("consistency" in e.lower() or "should" in e.lower() for e in errors):
            sys.exit(4)
        else:
            sys.exit(3)
