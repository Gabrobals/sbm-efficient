#!/usr/bin/env python3
"""
Adaptive-K A/B Testing Framework

Safe production rollout with statistical validation.
Compare Adaptive-K vs baseline with proper significance testing.

Usage:
    python ab_test_framework.py --demo
"""

import time
import random
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import argparse


@dataclass
class ExperimentConfig:
    """Configuration for A/B test."""
    name: str = "adaptive_k_rollout"
    control_group: str = "baseline"
    treatment_group: str = "adaptive_k"
    traffic_split: float = 0.10  # 10% to treatment initially
    min_samples: int = 1000
    max_samples: int = 100000
    confidence_level: float = 0.95
    mde: float = 0.01  # Minimum detectable effect (1% perplexity)


@dataclass
class GroupMetrics:
    """Metrics for a single group."""
    name: str
    samples: int = 0
    total_latency: float = 0
    total_ppl: float = 0
    total_k: float = 0
    latencies: List[float] = field(default_factory=list)
    perplexities: List[float] = field(default_factory=list)
    
    def add_sample(self, latency: float, ppl: float, k: float):
        self.samples += 1
        self.total_latency += latency
        self.total_ppl += ppl
        self.total_k += k
        self.latencies.append(latency)
        self.perplexities.append(ppl)
    
    @property
    def mean_latency(self) -> float:
        return self.total_latency / self.samples if self.samples > 0 else 0
    
    @property
    def mean_ppl(self) -> float:
        return self.total_ppl / self.samples if self.samples > 0 else 0
    
    @property
    def mean_k(self) -> float:
        return self.total_k / self.samples if self.samples > 0 else 0
    
    @property
    def std_latency(self) -> float:
        if self.samples < 2:
            return 0
        mean = self.mean_latency
        variance = sum((x - mean) ** 2 for x in self.latencies) / (self.samples - 1)
        return math.sqrt(variance)
    
    @property
    def std_ppl(self) -> float:
        if self.samples < 2:
            return 0
        mean = self.mean_ppl
        variance = sum((x - mean) ** 2 for x in self.perplexities) / (self.samples - 1)
        return math.sqrt(variance)


class ABTestExperiment:
    """
    A/B testing framework for safe Adaptive-K rollout.
    
    Features:
    - Gradual traffic ramping
    - Statistical significance testing
    - Automatic rollback on quality regression
    - Detailed metrics tracking
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.start_time = time.time()
        
        self.control = GroupMetrics(name=config.control_group)
        self.treatment = GroupMetrics(name=config.treatment_group)
        
        self.current_split = config.traffic_split
        self.phase = "initial"  # initial, ramping, full, concluded
        self.conclusion = None
        
    def assign_group(self, request_id: str) -> str:
        """
        Assign request to control or treatment group.
        Uses consistent hashing for stable assignment.
        """
        # Consistent hash based on request ID
        hash_val = hash(request_id) % 100
        
        if hash_val < self.current_split * 100:
            return self.config.treatment_group
        return self.config.control_group
    
    def record_sample(
        self,
        group: str,
        latency_ms: float,
        perplexity: float,
        k: float
    ):
        """Record a sample for the specified group."""
        if group == self.config.treatment_group:
            self.treatment.add_sample(latency_ms, perplexity, k)
        else:
            self.control.add_sample(latency_ms, perplexity, k)
        
        # Check for automatic actions
        self._check_guardrails()
        self._check_ramp_up()
    
    def _check_guardrails(self):
        """Check quality guardrails and trigger rollback if needed."""
        if self.treatment.samples < 100:
            return
        
        # Check if treatment perplexity is significantly worse
        ppl_diff = (self.treatment.mean_ppl - self.control.mean_ppl) / self.control.mean_ppl
        
        # Rollback if >5% perplexity increase
        if ppl_diff > 0.05:
            self.phase = "rolled_back"
            self.conclusion = "ROLLBACK: Treatment perplexity too high"
            print(f"\n⚠️ AUTOMATIC ROLLBACK: Treatment PPL {ppl_diff*100:.1f}% higher than control")
    
    def _check_ramp_up(self):
        """Check if we should ramp up traffic."""
        if self.phase != "initial" and self.phase != "ramping":
            return
        
        if self.treatment.samples < 500:
            return
        
        # Check statistical significance
        is_sig, p_value = self._compute_significance()
        
        if is_sig:
            # If significant improvement, ramp up
            if self.treatment.mean_ppl <= self.control.mean_ppl * 1.02:  # Within 2%
                old_split = self.current_split
                self.current_split = min(self.current_split * 2, 1.0)
                self.phase = "ramping" if self.current_split < 1.0 else "full"
                print(f"\n📈 RAMPING UP: {old_split*100:.0f}% → {self.current_split*100:.0f}%")
    
    def _compute_significance(self) -> Tuple[bool, float]:
        """
        Compute statistical significance using two-sample t-test.
        Returns (is_significant, p_value)
        """
        n1, n2 = self.control.samples, self.treatment.samples
        
        if n1 < 30 or n2 < 30:
            return False, 1.0
        
        # Welch's t-test for perplexity
        m1, m2 = self.control.mean_ppl, self.treatment.mean_ppl
        s1, s2 = self.control.std_ppl, self.treatment.std_ppl
        
        if s1 == 0 or s2 == 0:
            return False, 1.0
        
        se = math.sqrt(s1**2/n1 + s2**2/n2)
        t_stat = (m1 - m2) / se
        
        # Approximate p-value (two-tailed)
        # Using normal approximation for large samples
        p_value = 2 * (1 - self._normal_cdf(abs(t_stat)))
        
        is_significant = p_value < (1 - self.config.confidence_level)
        
        return is_significant, p_value
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def get_results(self) -> dict:
        """Get current experiment results."""
        is_sig, p_value = self._compute_significance()
        
        savings = 0
        if self.control.mean_k > 0:
            savings = (self.control.mean_k - self.treatment.mean_k) / self.control.mean_k
        
        latency_improvement = 0
        if self.control.mean_latency > 0:
            latency_improvement = (self.control.mean_latency - self.treatment.mean_latency) / self.control.mean_latency
        
        ppl_change = 0
        if self.control.mean_ppl > 0:
            ppl_change = (self.treatment.mean_ppl - self.control.mean_ppl) / self.control.mean_ppl
        
        return {
            "experiment": self.config.name,
            "phase": self.phase,
            "traffic_split": self.current_split,
            "duration_seconds": time.time() - self.start_time,
            "control": {
                "name": self.control.name,
                "samples": self.control.samples,
                "mean_latency_ms": self.control.mean_latency,
                "mean_perplexity": self.control.mean_ppl,
                "mean_k": self.control.mean_k,
            },
            "treatment": {
                "name": self.treatment.name,
                "samples": self.treatment.samples,
                "mean_latency_ms": self.treatment.mean_latency,
                "mean_perplexity": self.treatment.mean_ppl,
                "mean_k": self.treatment.mean_k,
            },
            "analysis": {
                "compute_savings": f"{savings*100:.1f}%",
                "latency_improvement": f"{latency_improvement*100:.1f}%",
                "perplexity_change": f"{ppl_change*100:.2f}%",
                "is_significant": is_sig,
                "p_value": p_value,
                "confidence_level": self.config.confidence_level,
            },
            "recommendation": self._get_recommendation(is_sig, ppl_change, savings),
        }
    
    def _get_recommendation(self, is_sig: bool, ppl_change: float, savings: float) -> str:
        """Generate recommendation based on results."""
        if self.phase == "rolled_back":
            return "ROLLBACK: Quality degradation detected. Investigate and recalibrate."
        
        if self.treatment.samples < self.config.min_samples:
            return f"COLLECTING DATA: Need {self.config.min_samples - self.treatment.samples} more samples."
        
        if not is_sig:
            return "NOT SIGNIFICANT: Continue experiment to gather more data."
        
        if ppl_change > 0.02:
            return "CAUTION: Perplexity increase >2%. Consider recalibrating thresholds."
        
        if savings > 0.25:
            return f"✅ SHIP IT: {savings*100:.0f}% savings with acceptable quality. Recommend full rollout."
        
        return "MARGINAL: Savings <25%. Consider if worth the complexity."
    
    def print_report(self):
        """Print formatted experiment report."""
        r = self.get_results()
        
        print("\n" + "=" * 70)
        print(f"              A/B TEST REPORT: {r['experiment']}")
        print("=" * 70)
        
        print(f"\n📊 EXPERIMENT STATUS")
        print(f"   Phase: {r['phase']}")
        print(f"   Traffic split: {r['traffic_split']*100:.0f}% treatment")
        print(f"   Duration: {r['duration_seconds']:.0f}s")
        
        print(f"\n📈 GROUP COMPARISON")
        print(f"   {'Metric':<20} {'Control':<15} {'Treatment':<15} {'Diff':<15}")
        print(f"   {'-'*60}")
        
        c, t = r['control'], r['treatment']
        
        print(f"   {'Samples':<20} {c['samples']:<15,} {t['samples']:<15,}")
        print(f"   {'Mean Latency (ms)':<20} {c['mean_latency_ms']:<15.1f} {t['mean_latency_ms']:<15.1f} {r['analysis']['latency_improvement']}")
        print(f"   {'Mean Perplexity':<20} {c['mean_perplexity']:<15.2f} {t['mean_perplexity']:<15.2f} {r['analysis']['perplexity_change']}")
        print(f"   {'Mean K':<20} {c['mean_k']:<15.2f} {t['mean_k']:<15.2f} {r['analysis']['compute_savings']}")
        
        print(f"\n🔬 STATISTICAL ANALYSIS")
        print(f"   Significant: {'Yes ✓' if r['analysis']['is_significant'] else 'No'}")
        print(f"   p-value: {r['analysis']['p_value']:.4f}")
        print(f"   Confidence: {r['analysis']['confidence_level']*100:.0f}%")
        
        print(f"\n📋 RECOMMENDATION")
        print(f"   {r['recommendation']}")
        
        print("\n" + "=" * 70)


def simulate_experiment():
    """Run simulated A/B test."""
    config = ExperimentConfig(
        name="adaptive_k_v1_rollout",
        traffic_split=0.10,
        min_samples=500,
    )
    
    experiment = ABTestExperiment(config)
    
    print("\n🧪 Starting A/B Test Simulation")
    print(f"   Initial traffic split: {config.traffic_split*100:.0f}% treatment")
    
    # Simulate requests
    for i in range(5000):
        request_id = f"req_{i}"
        group = experiment.assign_group(request_id)
        
        if group == "baseline":
            # Baseline: fixed K=2
            latency = random.gauss(80, 15)
            ppl = random.gauss(3.84, 0.05)
            k = 2.0
        else:
            # Treatment: Adaptive-K
            # 60% get K=1, 40% get K=2
            if random.random() < 0.60:
                latency = random.gauss(60, 10)
                k = 1.0
            else:
                latency = random.gauss(80, 15)
                k = 2.0
            ppl = random.gauss(3.87, 0.05)  # Slightly higher PPL
        
        experiment.record_sample(group, latency, ppl, k)
        
        # Print progress
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i+1:,} requests...")
    
    experiment.print_report()
    
    # Save results
    results = experiment.get_results()
    with open("ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n📁 Results saved to ab_test_results.json")


def main():
    parser = argparse.ArgumentParser(description="Adaptive-K A/B Testing Framework")
    parser.add_argument("--demo", action="store_true", help="Run simulation demo")
    
    args = parser.parse_args()
    
    if args.demo:
        simulate_experiment()
    else:
        print("Usage:")
        print("  python ab_test_framework.py --demo  # Run simulation")
        print("\nFor production, import ABTestExperiment:")
        print("  from ab_test_framework import ABTestExperiment, ExperimentConfig")
        print("  experiment = ABTestExperiment(ExperimentConfig())")


if __name__ == "__main__":
    main()
