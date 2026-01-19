#!/usr/bin/env python3
"""
MoE Cost Analyzer & Enterprise Savings Calculator

Analyze profiling results and project enterprise-scale cost savings
with Adaptive-K routing implementation.

Usage:
    # Analyze existing profiling results
    python scripts/moe_cost_analyzer.py --input workspace/moe_profiling_results.json

    # Generate enterprise projections
    python scripts/moe_cost_analyzer.py --scenario enterprise --tokens-daily 100M
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime


# ============================================================================
# PRICING DATA (as of January 2026)
# ============================================================================

MODEL_PRICING = {
    # DeepSeek Models
    "deepseek-v3.1": {
        "provider": "together",
        "input_per_1m": 0.60,
        "output_per_1m": 1.25,
        "experts": 256,
        "active_k": 8,
    },
    "deepseek-v3": {
        "provider": "together",
        "input_per_1m": 1.25,
        "output_per_1m": 1.25,
        "experts": 256,
        "active_k": 8,
    },
    "deepseek-chat": {
        "provider": "deepseek",
        "input_per_1m": 0.14,
        "output_per_1m": 0.28,
        "experts": 256,
        "active_k": 8,
    },
    
    # Mixtral Models
    "mixtral-8x7b": {
        "provider": "openrouter",
        "input_per_1m": 0.24,
        "output_per_1m": 0.24,
        "experts": 8,
        "active_k": 2,
    },
    "mixtral-8x22b": {
        "provider": "openrouter",
        "input_per_1m": 0.65,
        "output_per_1m": 0.65,
        "experts": 8,
        "active_k": 2,
    },
    
    # Qwen Models
    "qwen3-235b-moe": {
        "provider": "together",
        "input_per_1m": 0.65,
        "output_per_1m": 3.00,
        "experts": 128,
        "active_k": 22,
    },
    "qwen3-coder-480b": {
        "provider": "together",
        "input_per_1m": 2.00,
        "output_per_1m": 2.00,
        "experts": 160,
        "active_k": 35,
    },
    
    # Cogito MoE
    "cogito-109b-moe": {
        "provider": "together",
        "input_per_1m": 0.18,
        "output_per_1m": 0.59,
        "experts": 64,
        "active_k": 8,
    },
    "cogito-671b-moe": {
        "provider": "together",
        "input_per_1m": 1.25,
        "output_per_1m": 1.25,
        "experts": 256,
        "active_k": 8,
    },
}

# Validated Adaptive-K savings from our experiments
VALIDATED_SAVINGS = {
    "mixtral-8x7b": 0.31,      # 31.0%
    "qwen-moe": 0.324,          # 32.4% (Qwen1.5-MoE)
    "olmoe": 0.247,             # 24.7%
    # Estimates for untested models (conservative)
    "deepseek-v3": 0.35,        # ~35% estimated
    "qwen3-235b-moe": 0.30,     # ~30% estimated
    "default": 0.30,            # Default conservative estimate
}


@dataclass
class CostProjection:
    """Cost projection for a scenario"""
    scenario: str
    model: str
    daily_tokens: int
    baseline_daily_cost: float
    adaptive_k_daily_cost: float
    daily_savings: float
    monthly_savings: float
    annual_savings: float
    savings_percent: float


def calculate_baseline_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost without Adaptive-K"""
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        print(f"Warning: No pricing for {model}, using deepseek-chat")
        pricing = MODEL_PRICING["deepseek-chat"]
    
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    
    return input_cost + output_cost


def estimate_adaptive_k_savings(model: str, k_distribution: Optional[Dict] = None) -> float:
    """Estimate savings percentage with Adaptive-K"""
    # Use validated savings if available
    for key, savings in VALIDATED_SAVINGS.items():
        if key in model.lower():
            return savings
    
    # If we have K distribution from profiling, calculate directly
    if k_distribution:
        total = sum(k_distribution.values())
        if total > 0:
            pricing = MODEL_PRICING.get(model, {})
            baseline_k = pricing.get("active_k", 8)
            
            # Map K estimates to actual K values
            k_mapping = {
                "low": max(1, baseline_k // 4),
                "medium": baseline_k // 2,
                "high": baseline_k,
            }
            
            adaptive_compute = sum(
                k_distribution.get(level, 0) * k_mapping.get(level, baseline_k)
                for level in k_mapping
            )
            baseline_compute = total * baseline_k
            
            return 1 - (adaptive_compute / baseline_compute)
    
    return VALIDATED_SAVINGS["default"]


def project_enterprise_costs(
    model: str,
    daily_tokens: int,
    input_ratio: float = 0.3,  # 30% input, 70% output typical
    k_distribution: Optional[Dict] = None,
) -> CostProjection:
    """Project enterprise-scale costs"""
    
    input_tokens = int(daily_tokens * input_ratio)
    output_tokens = int(daily_tokens * (1 - input_ratio))
    
    # Baseline cost
    baseline_daily = calculate_baseline_cost(model, input_tokens, output_tokens)
    
    # Adaptive-K savings
    savings_pct = estimate_adaptive_k_savings(model, k_distribution)
    
    # Note: Adaptive-K saves on compute, which correlates with output cost
    # We apply savings primarily to output (where experts are activated)
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["deepseek-chat"])
    input_cost = (input_tokens / 1_000_000) * pricing["input_per_1m"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_1m"]
    
    adaptive_output_cost = output_cost * (1 - savings_pct)
    adaptive_daily = input_cost + adaptive_output_cost
    
    daily_savings = baseline_daily - adaptive_daily
    
    return CostProjection(
        scenario=f"{daily_tokens/1_000_000:.0f}M tokens/day",
        model=model,
        daily_tokens=daily_tokens,
        baseline_daily_cost=baseline_daily,
        adaptive_k_daily_cost=adaptive_daily,
        daily_savings=daily_savings,
        monthly_savings=daily_savings * 30,
        annual_savings=daily_savings * 365,
        savings_percent=savings_pct * 100,
    )


def analyze_profiling_results(filepath: str) -> Dict:
    """Analyze profiling results file"""
    with open(filepath) as f:
        data = json.load(f)
    
    analysis = {
        "source": filepath,
        "timestamp": data.get("timestamp"),
        "models_analyzed": [],
        "projections": [],
    }
    
    for model_key, profile in data.get("results", {}).items():
        summary = profile.get("summary", {})
        k_dist = summary.get("k_distribution")
        
        model_name = profile.get("model_name", model_key.split("/")[-1])
        
        # Project for common enterprise scenarios
        for daily_tokens in [10_000_000, 100_000_000, 1_000_000_000]:  # 10M, 100M, 1B
            proj = project_enterprise_costs(
                model_name, 
                daily_tokens, 
                k_distribution=k_dist
            )
            analysis["projections"].append({
                "model": model_name,
                "scenario": proj.scenario,
                "baseline_daily": f"${proj.baseline_daily_cost:.2f}",
                "adaptive_k_daily": f"${proj.adaptive_k_daily_cost:.2f}",
                "daily_savings": f"${proj.daily_savings:.2f}",
                "monthly_savings": f"${proj.monthly_savings:.2f}",
                "annual_savings": f"${proj.annual_savings:,.2f}",
                "savings_percent": f"{proj.savings_percent:.1f}%",
            })
        
        analysis["models_analyzed"].append(model_name)
    
    return analysis


def generate_enterprise_report(scenarios: List[str] = None):
    """Generate comprehensive enterprise cost report"""
    
    if scenarios is None:
        scenarios = ["startup", "growth", "enterprise", "hyperscale"]
    
    # Define scenario parameters
    scenario_params = {
        "startup": {"daily_tokens": 10_000_000, "description": "10M tokens/day"},
        "growth": {"daily_tokens": 100_000_000, "description": "100M tokens/day"},
        "enterprise": {"daily_tokens": 500_000_000, "description": "500M tokens/day"},
        "hyperscale": {"daily_tokens": 1_000_000_000, "description": "1B tokens/day"},
    }
    
    report = {
        "generated": datetime.now().isoformat(),
        "scenarios": {},
    }
    
    for scenario in scenarios:
        params = scenario_params.get(scenario, scenario_params["enterprise"])
        scenario_results = {
            "description": params["description"],
            "models": {},
        }
        
        for model in MODEL_PRICING:
            proj = project_enterprise_costs(model, params["daily_tokens"])
            scenario_results["models"][model] = {
                "baseline_monthly": proj.baseline_daily_cost * 30,
                "adaptive_k_monthly": proj.adaptive_k_daily_cost * 30,
                "monthly_savings": proj.monthly_savings,
                "annual_savings": proj.annual_savings,
                "savings_percent": proj.savings_percent,
            }
        
        report["scenarios"][scenario] = scenario_results
    
    return report


def print_report(report: Dict):
    """Print formatted report"""
    print("\n" + "="*80)
    print("ADAPTIVE-K ENTERPRISE COST ANALYSIS")
    print("="*80)
    print(f"Generated: {report['generated']}\n")
    
    for scenario_name, scenario_data in report.get("scenarios", {}).items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name.upper()} ({scenario_data['description']})")
        print("="*60)
        
        print(f"\n{'Model':<20} {'Baseline/mo':>12} {'Adaptive-K':>12} {'Savings/mo':>12} {'Savings':>8}")
        print("-"*70)
        
        for model, data in scenario_data["models"].items():
            print(f"{model:<20} ${data['baseline_monthly']:>10,.0f} ${data['adaptive_k_monthly']:>10,.0f} "
                  f"${data['monthly_savings']:>10,.0f} {data['savings_percent']:>7.1f}%")
        
        # Total row
        total_baseline = sum(d["baseline_monthly"] for d in scenario_data["models"].values())
        total_adaptive = sum(d["adaptive_k_monthly"] for d in scenario_data["models"].values())
        total_savings = sum(d["monthly_savings"] for d in scenario_data["models"].values())
        avg_savings = total_savings / total_baseline * 100 if total_baseline > 0 else 0
        
        print("-"*70)
        print(f"{'AVERAGE':<20} ${total_baseline/len(scenario_data['models']):>10,.0f} "
              f"${total_adaptive/len(scenario_data['models']):>10,.0f} "
              f"${total_savings/len(scenario_data['models']):>10,.0f} {avg_savings:>7.1f}%")


def print_single_projection(proj: CostProjection):
    """Print single model projection"""
    print(f"\n{'='*60}")
    print(f"COST PROJECTION: {proj.model}")
    print(f"Scenario: {proj.scenario}")
    print("="*60)
    
    print(f"\n  Daily Tokens: {proj.daily_tokens:,}")
    print(f"  Baseline Daily Cost: ${proj.baseline_daily_cost:.2f}")
    print(f"  Adaptive-K Daily Cost: ${proj.adaptive_k_daily_cost:.2f}")
    print(f"\n  Daily Savings: ${proj.daily_savings:.2f}")
    print(f"  Monthly Savings: ${proj.monthly_savings:,.2f}")
    print(f"  Annual Savings: ${proj.annual_savings:,.2f}")
    print(f"\n  Savings Percentage: {proj.savings_percent:.1f}%")
    
    # Visual
    savings_bar = "=" * int(proj.savings_percent / 2)
    print(f"\n  [{savings_bar}] {proj.savings_percent:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="MoE Cost Analyzer & Enterprise Savings Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--input", help="Profiling results JSON to analyze")
    parser.add_argument("--model", help="Specific model to project")
    parser.add_argument("--tokens-daily", type=int, default=100_000_000,
                        help="Daily token volume (default: 100M)")
    parser.add_argument("--scenario", choices=["startup", "growth", "enterprise", "hyperscale", "all"],
                        help="Enterprise scenario to analyze")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\nAvailable Models:")
        print("-"*60)
        for model, info in MODEL_PRICING.items():
            print(f"  {model:<20} ${info['input_per_1m']:.2f}/$1M in | ${info['output_per_1m']:.2f}/$1M out")
        return 0
    
    if args.input:
        # Analyze profiling results
        analysis = analyze_profiling_results(args.input)
        print(json.dumps(analysis, indent=2))
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"\nSaved to: {args.output}")
    
    elif args.scenario:
        # Generate enterprise report
        scenarios = None if args.scenario == "all" else [args.scenario]
        report = generate_enterprise_report(scenarios)
        print_report(report)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved to: {args.output}")
    
    elif args.model:
        # Single model projection
        proj = project_enterprise_costs(args.model, args.tokens_daily)
        print_single_projection(proj)
    
    else:
        # Default: show all scenarios
        report = generate_enterprise_report()
        print_report(report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
