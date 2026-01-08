"""
Validation schema and functions for experiment config YAML files.

Exit codes:
    0 = OK
    2 = config invalid
"""

from typing import Any, Dict, List, Tuple
import yaml


VALID_TASKS = {"xor", "mnist", "fashion_mnist", "cifar10"}
VALID_MODELS = {"baseline", "sbm", "random_routing", "static_topk", "sbm_adaptive_k"}
VALID_OPTIMIZERS = {"adamw", "sgd"}
VALID_SCHEDULES = {"linear", "cosine", "step"}
VALID_DEVICES = {"cuda", "cpu"}
VALID_PRECISIONS = {"fp32", "amp"}


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate experiment config against schema.
    
    Args:
        config: Parsed YAML config dictionary
        
    Returns:
        (is_valid, errors) tuple where errors is list of error messages
    """
    errors = []
    
    # A2.1 - Run section
    if "run" not in config:
        errors.append("Missing required section: run")
        return False, errors
    
    run = config["run"]
    
    if "task" not in run:
        errors.append("run.task is required")
    elif run["task"] not in VALID_TASKS:
        errors.append(f"run.task must be one of {VALID_TASKS}, got: {run['task']}")
    
    if "model" not in run:
        errors.append("run.model is required")
    elif run["model"] not in VALID_MODELS:
        errors.append(f"run.model must be one of {VALID_MODELS}, got: {run['model']}")
    
    if "seed" not in run:
        errors.append("run.seed is required")
    elif not isinstance(run["seed"], int):
        errors.append(f"run.seed must be int, got: {type(run['seed']).__name__}")
    
    # A2.2 - Hardware section
    if "hardware" in config:
        hw = config["hardware"]
        if "device" in hw and hw["device"] not in VALID_DEVICES:
            errors.append(f"hardware.device must be one of {VALID_DEVICES}, got: {hw['device']}")
        if "precision" in hw and hw["precision"] not in VALID_PRECISIONS:
            errors.append(f"hardware.precision must be one of {VALID_PRECISIONS}, got: {hw['precision']}")
    
    # A2.3 - Data section
    if "data" not in config:
        errors.append("Missing required section: data")
    else:
        data = config["data"]
        if "batch_size" not in data:
            errors.append("data.batch_size is required")
        elif not isinstance(data["batch_size"], int) or data["batch_size"] <= 0:
            errors.append(f"data.batch_size must be positive int, got: {data.get('batch_size')}")
    
    # A2.4 - Train section
    if "train" not in config:
        errors.append("Missing required section: train")
    else:
        train = config["train"]
        if "epochs" not in train:
            errors.append("train.epochs is required")
        elif not isinstance(train["epochs"], int) or train["epochs"] <= 0:
            errors.append(f"train.epochs must be positive int, got: {train.get('epochs')}")
        
        if "optimizer" in train and train["optimizer"] not in VALID_OPTIMIZERS:
            errors.append(f"train.optimizer must be one of {VALID_OPTIMIZERS}, got: {train['optimizer']}")
    
    # A2.5 - SBM section (if SBM family model)
    model = run.get("model", "")
    if model in {"sbm", "random_routing", "static_topk", "sbm_adaptive_k"}:
        if "sbm" not in config:
            errors.append(f"Model {model} requires 'sbm' section in config")
        else:
            sbm = config["sbm"]
            
            # Experts validation
            if "experts_num" not in sbm:
                errors.append("sbm.experts_num is required")
            elif not isinstance(sbm["experts_num"], int) or sbm["experts_num"] < 2:
                errors.append(f"sbm.experts_num must be >= 2, got: {sbm.get('experts_num')}")
            
            if "experts_top_k" not in sbm:
                errors.append("sbm.experts_top_k is required")
            elif not isinstance(sbm["experts_top_k"], int) or sbm["experts_top_k"] < 1:
                errors.append(f"sbm.experts_top_k must be >= 1, got: {sbm.get('experts_top_k')}")
            
            # Cross-validation: K <= N
            if "experts_num" in sbm and "experts_top_k" in sbm:
                if sbm["experts_top_k"] > sbm["experts_num"]:
                    errors.append(
                        f"sbm.experts_top_k ({sbm['experts_top_k']}) must be <= "
                        f"sbm.experts_num ({sbm['experts_num']})"
                    )
            
            # Decoherence schedule validation
            if "decoherence_tau" in sbm:
                tau = sbm["decoherence_tau"]
                if not isinstance(tau, dict):
                    errors.append("sbm.decoherence_tau must be a dictionary")
                else:
                    if "start" not in tau:
                        errors.append("sbm.decoherence_tau.start is required")
                    elif not isinstance(tau["start"], (int, float)) or tau["start"] <= 0:
                        errors.append(f"sbm.decoherence_tau.start must be > 0, got: {tau.get('start')}")
                    
                    if "end" not in tau:
                        errors.append("sbm.decoherence_tau.end is required")
                    elif not isinstance(tau["end"], (int, float)) or tau["end"] <= 0:
                        errors.append(f"sbm.decoherence_tau.end must be > 0, got: {tau.get('end')}")
                    
                    # start >= end validation
                    if "start" in tau and "end" in tau:
                        if tau["start"] < tau["end"]:
                            errors.append(
                                f"sbm.decoherence_tau.start ({tau['start']}) must be >= "
                                f"sbm.decoherence_tau.end ({tau['end']})"
                            )
                    
                    if "schedule" in tau and tau["schedule"] not in VALID_SCHEDULES:
                        errors.append(
                            f"sbm.decoherence_tau.schedule must be one of {VALID_SCHEDULES}, "
                            f"got: {tau['schedule']}"
                        )
            
            # Entropy lambda validation
            if "entropy_lambda" in sbm:
                ent = sbm["entropy_lambda"]
                if not isinstance(ent, dict):
                    errors.append("sbm.entropy_lambda must be a dictionary")
                else:
                    if "value" not in ent:
                        errors.append("sbm.entropy_lambda.value is required")
                    elif not isinstance(ent["value"], (int, float)) or ent["value"] < 0:
                        errors.append(f"sbm.entropy_lambda.value must be >= 0, got: {ent.get('value')}")

            # Adaptive-K requires max K not exceeding experts_top_k sizing
            if model == "sbm_adaptive_k" and "experts_top_k" in sbm and "adaptive_k" in config:
                k_values = config["adaptive_k"].get("k_values", [])
                if k_values:
                    max_k = max(k_values)
                    if sbm["experts_top_k"] < max_k:
                        errors.append(
                            f"sbm.experts_top_k ({sbm['experts_top_k']}) must be >= max(adaptive_k.k_values) ({max_k})"
                        )

    if model == "sbm_adaptive_k":
        if "adaptive_k" not in config:
            errors.append("Model sbm_adaptive_k requires 'adaptive_k' section")
        else:
            adaptive = config["adaptive_k"]
            k_values = adaptive.get("k_values")
            h_thresholds = adaptive.get("h_thresholds")

            if not isinstance(k_values, list) or not k_values:
                errors.append("adaptive_k.k_values must be a non-empty list")
            else:
                if any(not isinstance(k, int) or k <= 0 for k in k_values):
                    errors.append("adaptive_k.k_values must contain positive integers")
                if k_values != sorted(k_values):
                    errors.append("adaptive_k.k_values must be ascending")

            if not isinstance(h_thresholds, list) or not h_thresholds:
                errors.append("adaptive_k.h_thresholds must be a non-empty list")
            else:
                if any(not isinstance(h, (int, float)) or h <= 0 for h in h_thresholds):
                    errors.append("adaptive_k.h_thresholds must contain positive numbers")
                if h_thresholds != sorted(h_thresholds):
                    errors.append("adaptive_k.h_thresholds must be ascending")

            if isinstance(k_values, list) and isinstance(h_thresholds, list):
                if len(h_thresholds) != len(k_values) - 1:
                    errors.append(
                        "adaptive_k.h_thresholds length must be len(k_values) - 1"
                    )
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_config_file(config_path: str) -> Tuple[bool, List[str]]:
    """
    Load and validate config YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        (is_valid, errors) tuple
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        return False, [f"Config file not found: {config_path}"]
    except yaml.YAMLError as e:
        return False, [f"YAML parsing error: {e}"]
    except Exception as e:
        return False, [f"Error loading config: {e}"]
    
    return validate_config(config)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python -m src.common.validate_config <config.yaml>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    is_valid, errors = validate_config_file(config_path)
    
    if is_valid:
        print("[OK] CONFIG VALID")
        sys.exit(0)
    else:
        print("[FAIL] CONFIG INVALID")
        for error in errors:
            print(f"  - {error}")
        sys.exit(2)
