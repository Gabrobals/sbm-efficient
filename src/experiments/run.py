"""
Main experiment runner for SBM-Efficient.

Usage:
    python -m src.experiments.run --config <config.yaml> [--validate-only]

Exit codes:
    0 = success
    1 = pre-flight failed
    2 = config invalid
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime
import yaml

from src.experiments.preflight import run_preflight
from src.common.seed import set_seed


def setup_run_directory(run_dir: Path, config_path: Path) -> Path:
    """
    Create run directory and copy config.
    
    Args:
        run_dir: Path to run directory
        config_path: Path to original config file
        
    Returns:
        Path to stdout.log file
    """
    # Create run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config to run directory
    config_dest = run_dir / "config.yaml"
    shutil.copy(config_path, config_dest)
    
    # Create stdout.log
    stdout_log = run_dir / "stdout.log"
    stdout_log.touch()
    
    return stdout_log


def run_training(config: dict, run_dir: Path, run_id: str):
    """
    Execute training based on model type.
    
    Args:
        config: Experiment configuration
        run_dir: Path to run output directory
        run_id: Run identifier
    """
    from src.data.loaders import get_data_loaders
    from src.models.baseline import create_baseline_model
    from src.models.sbm_model import create_sbm_model, create_sbm_adaptive_k_model
    from src.training.loops import train_baseline
    from src.training.sbm_loops import train_sbm
    
    # Set seed for reproducibility
    seed = config["run"]["seed"]
    set_seed(seed)
    
    task = config["run"]["task"]
    model_type = config["run"]["model"]
    
    # Get data loaders
    batch_size = config.get("data", {}).get("batch_size", 128)
    num_workers = config.get("data", {}).get("num_workers", 0)
    
    print(f"\nLoading {task} dataset...")
    train_loader, test_loader = get_data_loaders(
        task=task,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model based on type
    if model_type == "baseline":
        print(f"\nCreating baseline model for {task}...")
        model = create_baseline_model(task, config)
        print(f"  Parameters: {model.count_parameters():,}")
        
        # Train baseline
        metrics = train_baseline(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            run_dir=run_dir,
            run_id=run_id
        )
        
    elif model_type in ["full", "random_routing", "static_topk", "sbm"]:
        print(f"\nCreating {model_type} model for {task}...")
        model = create_sbm_model(task, config)
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Experts: {model.num_experts}, Top-K: {model.top_k}")
        
        # Train SBM
        metrics = train_sbm(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            run_dir=run_dir,
            run_id=run_id
        )

    elif model_type == "sbm_adaptive_k":
        print(f"\nCreating {model_type} model for {task}...")
        model = create_sbm_adaptive_k_model(task, config)
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Experts: {model.num_experts}, K_values: {model.k_values}")

        metrics = train_sbm(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            config=config,
            run_dir=run_dir,
            run_id=run_id
        )
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="SBM-Efficient Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Validate only (no training)
    python -m src.experiments.run --config configs/baseline_mnist.yaml --validate-only
    
    # Run full experiment
    python -m src.experiments.run --config configs/sbm_mnist.yaml
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run pre-flight checks only, do not train"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    
    # Run pre-flight checks
    passed, run_id, metadata = run_preflight(config_path)
    
    if not passed:
        print("\n[FAIL] Pre-flight failed. Aborting.")
        sys.exit(1)
    
    # Setup run directory
    run_dir = Path(metadata["run_dir"])
    stdout_log = setup_run_directory(run_dir, config_path)
    
    print(f"\nRun directory created: {run_dir}")
    print(f"   - config.yaml copied")
    print(f"   - stdout.log created")
    
    # If validate-only, exit here
    if args.validate_only:
        print("\n" + "=" * 60)
        print("[OK] VALIDATE-ONLY MODE: Pre-flight passed, no training executed")
        print("=" * 60)
        print(f"\nRun ID: {run_id}")
        print(f"Run dir: {run_dir}")
        print("\nTo run full training, remove --validate-only flag.")
        sys.exit(0)
    
    # ========================================
    # TRAINING
    # ========================================
    config = metadata["config"]
    
    try:
        metrics = run_training(config, run_dir, run_id)
        
        # Run post-flight validation
        print("\n" + "=" * 60)
        print("POST-FLIGHT VALIDATION")
        print("=" * 60)
        
        from src.experiments.postflight import run_postflight
        valid = run_postflight(run_dir)
        
        if valid:
            print(f"\n[OK] Experiment completed successfully!")
            print(f"   Run ID: {run_id}")
            print(f"   Run dir: {run_dir}")
            sys.exit(0)
        else:
            print(f"\n[WARN] Experiment completed but validation failed.")
            sys.exit(4)
            
    except Exception as e:
        print(f"\n[FAIL] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
