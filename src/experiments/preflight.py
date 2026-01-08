"""
Pre-flight checks before starting experiment run.

Validates:
- Config schema
- Git status
- Environment setup
- Output directories

Exit codes:
    0 = all checks passed
    1 = pre-flight failed
"""

import sys
import subprocess
from pathlib import Path
from typing import Tuple, List
import yaml

from src.common.validate_config import validate_config


def check_git_status() -> Tuple[bool, str, bool]:
    """
    Check git repository status.
    
    Returns:
        (has_git, sha, is_dirty) tuple
    """
    try:
        # Get short SHA
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        sha = result.stdout.strip()
        
        # Check if working directory is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        is_dirty = len(result.stdout.strip()) > 0
        
        return True, sha, is_dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, "unknown", False


def check_pytorch() -> Tuple[bool, str]:
    """
    Check PyTorch installation.
    
    Returns:
        (is_available, error_message) tuple
    """
    try:
        import torch
        return True, ""
    except ImportError as e:
        return False, f"PyTorch not installed: {e}"


def check_device(requested_device: str) -> Tuple[bool, str]:
    """
    Check if requested device is available.
    
    Args:
        requested_device: "cuda" or "cpu"
        
    Returns:
        (is_available, error_message) tuple
    """
    try:
        import torch
        
        if requested_device == "cuda":
            if not torch.cuda.is_available():
                return False, "CUDA requested but not available"
        elif requested_device == "cpu":
            pass  # CPU always available
        else:
            return False, f"Unknown device: {requested_device}"
        
        return True, ""
    except Exception as e:
        return False, f"Error checking device: {e}"


def check_output_directory(out_dir: Path) -> Tuple[bool, str]:
    """
    Check if output directory exists and is writable.
    
    Args:
        out_dir: Path to output directory
        
    Returns:
        (is_ok, error_message) tuple
    """
    if not out_dir.exists():
        return False, f"Output directory does not exist: {out_dir}"
    
    if not out_dir.is_dir():
        return False, f"Output path is not a directory: {out_dir}"
    
    # Try to create a test file to check writability
    test_file = out_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
        return True, ""
    except Exception as e:
        return False, f"Output directory not writable: {e}"


def generate_run_id(config: dict, git_sha: str) -> str:
    """
    Generate run ID from config and git status.
    
    Format: {date}_{task}_{model}_seed{seed}_{gitShortSha}
    
    Args:
        config: Experiment config dictionary
        git_sha: Git commit short SHA
        
    Returns:
        Run ID string
    """
    from datetime import datetime
    
    date = datetime.now().strftime("%Y-%m-%d")
    task = config["run"]["task"]
    model = config["run"]["model"]
    seed = config["run"]["seed"]
    
    return f"{date}_{task}_{model}_seed{seed}_{git_sha}"


def run_preflight(config_path: Path) -> Tuple[bool, str, dict]:
    """
    Run all pre-flight checks.
    
    Args:
        config_path: Path to experiment config YAML
        
    Returns:
        (passed, run_id, metadata) tuple
    """
    errors = []
    
    print("=" * 60)
    print("PRE-FLIGHT CHECKS")
    print("=" * 60)
    
    # A1.1 - Git status
    print("\n[A1] Git & Environment")
    has_git, git_sha, is_dirty = check_git_status()
    if has_git:
        status = "[WARN] DIRTY" if is_dirty else "[OK] CLEAN"
        print(f"  Git SHA: {git_sha} {status}")
        if is_dirty:
            errors.append("Working directory has uncommitted changes (git dirty)")
    else:
        print("  [WARN] Git not available (SHA will be 'unknown')")
        git_sha = "unknown"
        is_dirty = False
    
    # A1.2 - PyTorch
    pytorch_ok, pytorch_error = check_pytorch()
    if pytorch_ok:
        print("  [OK] PyTorch installed")
    else:
        print(f"  [FAIL] {pytorch_error}")
        errors.append(pytorch_error)
        return False, "", {}
    
    # A2 - Config validation
    print("\n[A2] Config Validation")
    if not config_path.exists():
        print(f"  [ERR] Config file not found: {config_path}")
        errors.append(f"Config file not found: {config_path}")
        return False, "", {}
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"  [ERR] Error loading config: {e}")
        errors.append(f"Error loading config: {e}")
        return False, "", {}
    
    config_valid, config_errors = validate_config(config)
    if config_valid:
        print("  [OK] Config schema valid")
    else:
        print("  [FAIL] Config schema invalid:")
        for err in config_errors:
            print(f"     - {err}")
        errors.extend(config_errors)
        return False, "", {}
    
    # A1.3 - Device check
    requested_device = config.get("hardware", {}).get("device", "cpu")
    device_ok, device_error = check_device(requested_device)
    if device_ok:
        print(f"  [OK] Device '{requested_device}' available")
    else:
        print(f"  [FAIL] {device_error}")
        errors.append(device_error)
    
    # A3 - Output directory
    print("\n[A3] Output Directory")
    out_dir = Path(config["run"].get("out_dir", "results/runs"))
    out_dir_ok, out_dir_error = check_output_directory(out_dir)
    if out_dir_ok:
        print(f"  [OK] Output directory OK: {out_dir}")
    else:
        print(f"  [FAIL] {out_dir_error}")
        errors.append(out_dir_error)
    
    # Generate run ID
    run_id = generate_run_id(config, git_sha)
    run_dir = out_dir / run_id
    
    print(f"\n[A3] Run ID: {run_id}")
    print(f"      Run dir: {run_dir}")
    
    if run_dir.exists():
        print(f"  [WARN] Run directory already exists (will be overwritten)")
    
    # Summary
    print("\n" + "=" * 60)
    if len(errors) == 0:
        print("[OK] ALL PRE-FLIGHT CHECKS PASSED")
        print("=" * 60)
        
        metadata = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "git_sha": git_sha,
            "git_dirty": is_dirty,
            "config": config,
        }
        
        return True, run_id, metadata
    else:
        print("[FAIL] PRE-FLIGHT FAILED")
        print("=" * 60)
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
        print("\nFix errors and try again.")
        
        return False, "", {}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.experiments.preflight <config.yaml>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    passed, run_id, metadata = run_preflight(config_path)
    
    if passed:
        sys.exit(0)
    else:
        sys.exit(1)
