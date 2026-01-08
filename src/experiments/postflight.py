"""
Post-flight validation after experiment run.

Validates:
- metrics.json exists and is valid
- Internal consistency with config
- Run quality criteria

Exit codes:
    0 = validation passed
    3 = metrics schema invalid
    4 = consistency invalid
    5 = missing files
"""

import sys
from pathlib import Path

from src.common.validate_metrics import validate_run_directory


def run_postflight(run_dir: Path) -> bool:
    """
    Run post-flight validation on completed run.
    
    Args:
        run_dir: Path to run directory (results/runs/<run_id>/)
        
    Returns:
        True if validation passed, False otherwise
    """
    print("=" * 60)
    print("POST-FLIGHT VALIDATION")
    print("=" * 60)
    print(f"\nRun directory: {run_dir}")
    
    if not run_dir.exists():
        print(f"\n[FAIL] RUN DIRECTORY NOT FOUND: {run_dir}")
        return False
    
    is_valid, errors = validate_run_directory(run_dir)
    
    if is_valid:
        print("\n" + "=" * 60)
        print("[OK] POST-FLIGHT VALIDATION PASSED")
        print("=" * 60)
        print("\nRun is VALID and ready for analysis.")
        return True
    else:
        print("\n" + "=" * 60)
        print("[FAIL] POST-FLIGHT VALIDATION FAILED")
        print("=" * 60)
        print("\nErrors:")
        for err in errors:
            print(f"  - {err}")
        
        print("\n[WARN] Run is INVALID and should not be used for comparison.")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.experiments.postflight <run_directory>")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    
    passed = run_postflight(run_dir)
    
    if passed:
        sys.exit(0)
    else:
        # Determine specific exit code based on error type
        is_valid, errors = validate_run_directory(run_dir)
        if any("Missing required file" in e for e in errors):
            sys.exit(5)
        elif any("consistency" in e.lower() or "should" in e.lower() for e in errors):
            sys.exit(4)
        else:
            sys.exit(3)
