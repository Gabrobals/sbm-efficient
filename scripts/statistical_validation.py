#!/usr/bin/env python3
"""
Statistical Validation Script for Adaptive-K Paper

This script implements the statistical tests required by the peer review:
1. TOST (Two One-Sided Tests) for non-inferiority
2. Power analysis for sample size determination
3. Bootstrap confidence intervals

Usage:
    python scripts/statistical_validation.py --generate-example
    python scripts/statistical_validation.py --data results/ppl_comparison.json

References:
    - Schuirmann (1987) "A comparison of the Two One-Sided Tests Procedure and the Power Approach"
    - Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy import stats


@dataclass
class TOSTResult:
    """Result from Two One-Sided Tests."""
    lower_test_stat: float
    upper_test_stat: float
    lower_pvalue: float
    upper_pvalue: float
    combined_pvalue: float  # max of the two
    margin: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    non_inferior: bool
    alpha: float
    n_samples: int


def tost_paired(
    baseline: np.ndarray,
    treatment: np.ndarray,
    margin: float,
    alpha: float = 0.05
) -> TOSTResult:
    """
    Two One-Sided Tests (TOST) for paired samples.
    
    Tests non-inferiority: treatment is not worse than baseline by more than margin.
    
    H0_lower: μ_diff ≤ -margin (treatment much worse)
    H0_upper: μ_diff ≥ +margin (treatment much better - not usually of interest)
    
    For non-inferiority, we only care about the lower bound:
    H0: μ_treatment - μ_baseline ≤ -margin (treatment is inferior)
    H1: μ_treatment - μ_baseline > -margin (treatment is non-inferior)
    
    Args:
        baseline: Baseline measurements (e.g., perplexity with K=2)
        treatment: Treatment measurements (e.g., perplexity with Adaptive-K)
        margin: Non-inferiority margin (e.g., 1% of baseline mean)
        alpha: Significance level (default 0.05)
    
    Returns:
        TOSTResult with all statistics
    """
    if len(baseline) != len(treatment):
        raise ValueError("Baseline and treatment must have same length")
    
    n = len(baseline)
    diff = treatment - baseline  # Positive means treatment is worse (higher PPL)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # For non-inferiority with perplexity (lower is better):
    # We test if treatment PPL is at most `margin` higher than baseline
    # H0: μ_diff >= margin (inferior)
    # H1: μ_diff < margin (non-inferior)
    
    # Lower bound test (treatment not too much worse)
    t_lower = (mean_diff - (-margin)) / se_diff  # Should be positive if non-inferior
    p_lower = stats.t.cdf(t_lower, df=n-1)  # One-sided (we want this to be high)
    
    # Upper bound test (usually not of interest for non-inferiority)
    t_upper = (mean_diff - margin) / se_diff
    p_upper = 1 - stats.t.cdf(t_upper, df=n-1)
    
    # For non-inferiority, we use the lower test
    # p_lower is the p-value for "treatment is non-inferior"
    # We reject non-inferiority if p_lower > alpha
    
    # Actually, standard TOST uses: p = max(p_lower, p_upper) for equivalence
    # For one-sided non-inferiority, we just use p_lower
    combined_pvalue = 1 - p_lower  # Convert to standard p-value interpretation
    
    # Confidence interval
    t_crit = stats.t.ppf(1 - alpha, df=n-1)
    ci_lower = mean_diff - t_crit * se_diff
    ci_upper = mean_diff + t_crit * se_diff
    
    # Non-inferior if upper bound of CI is below margin
    non_inferior = ci_upper < margin
    
    return TOSTResult(
        lower_test_stat=t_lower,
        upper_test_stat=t_upper,
        lower_pvalue=p_lower,
        upper_pvalue=p_upper,
        combined_pvalue=combined_pvalue,
        margin=margin,
        mean_diff=mean_diff,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        non_inferior=non_inferior,
        alpha=alpha,
        n_samples=n
    )


def power_analysis_paired_ttest(
    effect_size: float = 0.35,
    alpha: float = 0.025,
    power: float = 0.95
) -> int:
    """
    Calculate required sample size for paired t-test.
    
    Uses the formula: n = 2 * ((z_α + z_β) / d)²
    
    Args:
        effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
        alpha: Significance level (0.025 for TOST = 0.05 two-sided)
        power: Desired power (1 - β)
    
    Returns:
        Required sample size (per group for paired samples = total n)
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    
    # For paired t-test
    n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    
    return int(np.ceil(n))


def bootstrap_ci(
    data: np.ndarray,
    statistic_func=np.mean,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.
    
    Args:
        data: Input data
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        ci_level: Confidence level (default: 0.95)
        seed: Random seed for reproducibility
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    
    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile method
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    point_estimate = statistic_func(data)
    
    return point_estimate, ci_lower, ci_upper


def generate_example_data(
    n_samples: int = 1500,
    baseline_mean: float = 3.84,
    effect_size: float = 0.008,  # 0.8% increase
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate example perplexity data for demonstration.
    
    Args:
        n_samples: Number of sequences
        baseline_mean: Mean baseline perplexity
        effect_size: Relative increase for treatment (0.008 = 0.8%)
        seed: Random seed
    
    Returns:
        (baseline_ppl, adaptive_ppl) arrays
    """
    rng = np.random.RandomState(seed)
    
    # Baseline perplexity (log-normal distribution is common for PPL)
    baseline = rng.lognormal(
        mean=np.log(baseline_mean),
        sigma=0.1,
        size=n_samples
    )
    
    # Adaptive-K perplexity (slightly higher on average)
    # Model: adaptive = baseline * (1 + effect) + noise
    treatment = baseline * (1 + effect_size) + rng.normal(0, 0.01, size=n_samples)
    
    return baseline, treatment


def run_full_analysis(
    baseline: np.ndarray,
    treatment: np.ndarray,
    model_name: str = "Unknown Model",
    margin_pct: float = 1.0,  # 1% non-inferiority margin
    alpha: float = 0.05
):
    """
    Run complete statistical analysis as required by peer review.
    
    Args:
        baseline: Baseline perplexity values
        treatment: Adaptive-K perplexity values
        model_name: Name for reporting
        margin_pct: Non-inferiority margin as percentage of baseline
        alpha: Significance level
    """
    print("=" * 70)
    print(f"Statistical Validation Report: {model_name}")
    print("=" * 70)
    
    n = len(baseline)
    baseline_mean = np.mean(baseline)
    treatment_mean = np.mean(treatment)
    margin = baseline_mean * margin_pct / 100
    
    print(f"\n1. Sample Statistics")
    print("-" * 40)
    print(f"   Sample size: n = {n}")
    print(f"   Baseline PPL: {baseline_mean:.4f} (SD: {np.std(baseline):.4f})")
    print(f"   Adaptive-K PPL: {treatment_mean:.4f} (SD: {np.std(treatment):.4f})")
    print(f"   Difference: {treatment_mean - baseline_mean:+.4f} ({(treatment_mean - baseline_mean)/baseline_mean*100:+.2f}%)")
    print(f"   Non-inferiority margin: {margin:.4f} ({margin_pct}% of baseline)")
    
    # TOST
    print(f"\n2. TOST Non-Inferiority Test")
    print("-" * 40)
    tost = tost_paired(baseline, treatment, margin=margin, alpha=alpha)
    
    print(f"   Null hypothesis: Adaptive-K is inferior (ΔPPL ≥ {margin:.4f})")
    print(f"   Alternative: Adaptive-K is non-inferior (ΔPPL < {margin:.4f})")
    print(f"   Test statistic: t = {tost.lower_test_stat:.4f}")
    print(f"   P-value: {tost.combined_pvalue:.6f}")
    print(f"   {100*(1-alpha)}% CI for difference: [{tost.ci_lower:.4f}, {tost.ci_upper:.4f}]")
    print(f"   Conclusion: {'✅ Non-inferiority CONFIRMED' if tost.non_inferior else '❌ Non-inferiority NOT confirmed'}")
    print(f"              (p < {alpha} = {tost.combined_pvalue < alpha})")
    
    # Bootstrap CI
    print(f"\n3. Bootstrap Confidence Intervals")
    print("-" * 40)
    diff = treatment - baseline
    point, ci_lo, ci_hi = bootstrap_ci(diff, n_bootstrap=10000)
    print(f"   Mean difference: {point:.4f}")
    print(f"   95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"   Percentile method with 10,000 resamples")
    
    # Effect size
    print(f"\n4. Effect Size")
    print("-" * 40)
    cohens_d = (treatment_mean - baseline_mean) / np.std(diff)
    print(f"   Cohen's d: {cohens_d:.4f}")
    print(f"   Interpretation: {'negligible' if abs(cohens_d) < 0.2 else 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'}")
    
    # Power analysis
    print(f"\n5. Power Analysis (retrospective)")
    print("-" * 40)
    required_n = power_analysis_paired_ttest(
        effect_size=0.35,
        alpha=alpha/2,  # One-sided
        power=0.95
    )
    print(f"   For detecting d=0.35 with α={alpha}, power=0.95:")
    print(f"   Required n: {required_n}")
    print(f"   Actual n: {n}")
    print(f"   Status: {'✅ Adequately powered' if n >= required_n else '⚠️ May be underpowered'}")
    
    print("\n" + "=" * 70)
    
    return {
        "model": model_name,
        "n": n,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "mean_diff": tost.mean_diff,
        "pct_diff": (treatment_mean - baseline_mean) / baseline_mean * 100,
        "margin": margin,
        "tost_pvalue": tost.combined_pvalue,
        "ci_lower": tost.ci_lower,
        "ci_upper": tost.ci_upper,
        "non_inferior": tost.non_inferior,
        "cohens_d": cohens_d,
        "required_n": required_n,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Statistical validation for Adaptive-K paper"
    )
    parser.add_argument(
        "--generate-example",
        action="store_true",
        help="Generate and analyze example data"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to JSON file with actual PPL data"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Non-inferiority margin as percentage (default: 1.0%%)"
    )
    
    args = parser.parse_args()
    
    if args.generate_example:
        print("Generating example data for demonstration...")
        print()
        
        # Calculate required sample size first
        required_n = power_analysis_paired_ttest(
            effect_size=0.35,
            alpha=0.025,
            power=0.95
        )
        print(f"Required sample size for TOST (d=0.35, α=0.05, power=0.95): n = {required_n}")
        print()
        
        # Generate data with adequate sample size
        baseline, treatment = generate_example_data(
            n_samples=max(1500, required_n),
            baseline_mean=3.84,  # Mixtral baseline PPL
            effect_size=0.008,   # 0.8% increase
        )
        
        result = run_full_analysis(
            baseline, treatment,
            model_name="Mixtral 8x7B (Example Data)",
            margin_pct=args.margin
        )
        
        # Save example data
        example_data = {
            "model": "Mixtral 8x7B",
            "baseline_ppl": baseline.tolist(),
            "adaptive_ppl": treatment.tolist(),
            "metadata": {
                "n_samples": len(baseline),
                "baseline_mean": float(np.mean(baseline)),
                "adaptive_mean": float(np.mean(treatment)),
                "margin_pct": args.margin,
            },
            "results": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in result.items()}
        }
        
        output_path = "results/statistical_validation_example.json"
        import os
        os.makedirs("results", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(example_data, f, indent=2)
        print(f"\nExample data saved to: {output_path}")
        
    elif args.data:
        with open(args.data, "r") as f:
            data = json.load(f)
        
        baseline = np.array(data["baseline_ppl"])
        treatment = np.array(data["adaptive_ppl"])
        model_name = data.get("model", "Unknown")
        
        run_full_analysis(
            baseline, treatment,
            model_name=model_name,
            margin_pct=args.margin
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
