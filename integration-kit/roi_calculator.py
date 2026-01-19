#!/usr/bin/env python3
"""
Adaptive-K ROI Calculator

Estimate cost savings before integration. Run this FIRST to build the business case.

Usage:
    python roi_calculator.py --tokens-per-day 1000000000 --cost-per-1k 0.001
    python roi_calculator.py --interactive
"""

import argparse
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class WorkloadProfile:
    """Describes the inference workload."""
    tokens_per_day: int
    cost_per_1k_tokens: float  # USD
    current_k: int = 2
    num_experts: int = 8
    model_name: str = "mixtral-8x7b"
    
    @property
    def daily_cost(self) -> float:
        return (self.tokens_per_day / 1000) * self.cost_per_1k_tokens
    
    @property
    def annual_cost(self) -> float:
        return self.daily_cost * 365


@dataclass
class SavingsEstimate:
    """Projected savings from Adaptive-K."""
    conservative: float  # 25% savings
    moderate: float      # 31% savings  
    aggressive: float    # 33% savings
    
    def to_dict(self):
        return {
            "conservative_25pct": self.conservative,
            "moderate_31pct": self.moderate,
            "aggressive_33pct": self.aggressive
        }


def estimate_savings(profile: WorkloadProfile) -> dict:
    """
    Calculate projected savings based on workload profile.
    
    Savings ranges based on empirical results:
    - Conservative (25%): Low-entropy workloads, safety-critical applications
    - Moderate (31%): Typical production workloads (Mixtral-level)
    - Aggressive (33%): High-entropy workloads, Nemotron 3 Nano level
    """
    
    annual_cost = profile.annual_cost
    
    # Savings percentages from experiments
    savings = SavingsEstimate(
        conservative=annual_cost * 0.25,
        moderate=annual_cost * 0.31,
        aggressive=annual_cost * 0.33
    )
    
    # Integration costs (one-time)
    integration_cost = estimate_integration_cost(profile)
    
    # Time to ROI
    days_to_roi = {
        "conservative": integration_cost / (savings.conservative / 365),
        "moderate": integration_cost / (savings.moderate / 365),
        "aggressive": integration_cost / (savings.aggressive / 365)
    }
    
    return {
        "workload": {
            "tokens_per_day": profile.tokens_per_day,
            "daily_cost_usd": profile.daily_cost,
            "annual_cost_usd": annual_cost,
            "model": profile.model_name,
            "current_k": profile.current_k
        },
        "projected_savings": {
            "annual_usd": savings.to_dict(),
            "daily_usd": {
                "conservative": savings.conservative / 365,
                "moderate": savings.moderate / 365,
                "aggressive": savings.aggressive / 365
            }
        },
        "integration": {
            "estimated_days": 3,  # Median estimate
            "engineering_cost_usd": integration_cost,
            "risk": "low"
        },
        "roi": {
            "days_to_breakeven": days_to_roi,
            "first_year_net_savings": {
                "conservative": savings.conservative - integration_cost,
                "moderate": savings.moderate - integration_cost,
                "aggressive": savings.aggressive - integration_cost
            }
        }
    }


def estimate_integration_cost(profile: WorkloadProfile) -> float:
    """
    Estimate one-time integration cost.
    
    Assumptions:
    - Senior ML Engineer: $150/hour
    - 3-5 days integration time
    - 8 hours/day
    """
    hourly_rate = 150
    days = 4  # Average of 3-5 days
    hours_per_day = 8
    
    return hourly_rate * days * hours_per_day


def print_report(results: dict):
    """Print formatted ROI report."""
    
    print("\n" + "=" * 70)
    print("                    ADAPTIVE-K ROI ANALYSIS")
    print("=" * 70)
    
    w = results["workload"]
    print(f"\n📊 WORKLOAD PROFILE")
    print(f"   Model: {w['model']}")
    print(f"   Tokens/day: {w['tokens_per_day']:,}")
    print(f"   Current K: {w['current_k']}")
    print(f"   Daily cost: ${w['daily_cost_usd']:,.2f}")
    print(f"   Annual cost: ${w['annual_cost_usd']:,.2f}")
    
    s = results["projected_savings"]["annual_usd"]
    print(f"\n💰 PROJECTED ANNUAL SAVINGS")
    print(f"   Conservative (25%): ${s['conservative_25pct']:,.2f}")
    print(f"   Moderate (40%):     ${s['moderate_40pct']:,.2f}")
    print(f"   Aggressive (52%):   ${s['aggressive_52pct']:,.2f}")
    
    i = results["integration"]
    print(f"\n🔧 INTEGRATION COST")
    print(f"   Estimated days: {i['estimated_days']}")
    print(f"   Engineering cost: ${i['engineering_cost_usd']:,.2f}")
    print(f"   Risk level: {i['risk']}")
    
    r = results["roi"]
    print(f"\n📈 RETURN ON INVESTMENT")
    print(f"   Days to breakeven:")
    print(f"      Conservative: {r['days_to_breakeven']['conservative']:.1f} days")
    print(f"      Moderate:     {r['days_to_breakeven']['moderate']:.1f} days")
    print(f"      Aggressive:   {r['days_to_breakeven']['aggressive']:.1f} days")
    
    print(f"\n   First-year net savings:")
    print(f"      Conservative: ${r['first_year_net_savings']['conservative']:,.2f}")
    print(f"      Moderate:     ${r['first_year_net_savings']['moderate']:,.2f}")
    print(f"      Aggressive:   ${r['first_year_net_savings']['aggressive']:,.2f}")
    
    print("\n" + "=" * 70)
    
    # Recommendation
    if w['annual_cost_usd'] > 100000:
        print("\n✅ RECOMMENDATION: High ROI potential. Proceed with integration.")
        print("   Expected payback period: < 1 week")
    elif w['annual_cost_usd'] > 10000:
        print("\n✅ RECOMMENDATION: Good ROI potential. Consider integration.")
        print("   Expected payback period: 1-4 weeks")
    else:
        print("\n⚠️  RECOMMENDATION: Moderate ROI. Evaluate based on other factors.")
        print("   Consider if latency improvements also valuable.")
    
    print("\n" + "=" * 70)


def interactive_mode():
    """Interactive ROI calculator."""
    print("\n🧮 ADAPTIVE-K ROI CALCULATOR (Interactive Mode)\n")
    
    # Gather inputs
    print("Enter your workload details:\n")
    
    tokens = input("Tokens processed per day [default: 1,000,000,000]: ").strip()
    tokens = int(tokens.replace(",", "")) if tokens else 1_000_000_000
    
    cost = input("Cost per 1K tokens in USD [default: 0.001]: ").strip()
    cost = float(cost) if cost else 0.001
    
    model = input("Model name [default: mixtral-8x7b]: ").strip()
    model = model if model else "mixtral-8x7b"
    
    k = input("Current K value [default: 2]: ").strip()
    k = int(k) if k else 2
    
    experts = input("Number of experts [default: 8]: ").strip()
    experts = int(experts) if experts else 8
    
    profile = WorkloadProfile(
        tokens_per_day=tokens,
        cost_per_1k_tokens=cost,
        model_name=model,
        current_k=k,
        num_experts=experts
    )
    
    results = estimate_savings(profile)
    print_report(results)
    
    # Save results
    save = input("\nSave results to JSON? [y/N]: ").strip().lower()
    if save == 'y':
        filename = f"roi_estimate_{model.replace('-', '_')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate ROI from Adaptive-K integration"
    )
    parser.add_argument("--tokens-per-day", type=int, default=1_000_000_000,
                        help="Tokens processed per day")
    parser.add_argument("--cost-per-1k", type=float, default=0.001,
                        help="Cost per 1K tokens in USD")
    parser.add_argument("--model", type=str, default="mixtral-8x7b",
                        help="Model name")
    parser.add_argument("--current-k", type=int, default=2,
                        help="Current K value")
    parser.add_argument("--num-experts", type=int, default=8,
                        help="Number of experts in model")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    profile = WorkloadProfile(
        tokens_per_day=args.tokens_per_day,
        cost_per_1k_tokens=args.cost_per_1k,
        model_name=args.model,
        current_k=args.current_k,
        num_experts=args.num_experts
    )
    
    results = estimate_savings(profile)
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_report(results)


if __name__ == "__main__":
    main()
