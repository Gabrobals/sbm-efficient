#!/usr/bin/env python3
"""
Cerebras Cloud API Profiling

Profile Cerebras Inference API to establish baseline metrics.
Cerebras offers OpenAI-compatible API with ultra-fast inference.

Usage:
    # Sign up at https://cloud.cerebras.ai
    export CEREBRAS_API_KEY="your-key"
    python scripts/cerebras_api_profiling.py

Models available:
    - llama3.1-8b
    - llama-3.3-70b  
    - qwen-3-32b
    - glm-4.7
"""

import os
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime
import statistics

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai not installed. Run: pip install openai")


@dataclass
class CerebrasProfile:
    """Profile for a single request"""
    prompt: str
    category: str
    model: str
    response: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_ms: float = 0.0
    timestamp: str = ""


@dataclass 
class ProfilingResults:
    """Aggregated results"""
    provider: str = "cerebras"
    model: str = ""
    timestamp: str = ""
    profiles: List[CerebrasProfile] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)


# Benchmark prompts - varied complexity
BENCHMARK_PROMPTS = [
    # Simple (expected low latency, high tok/s)
    {"prompt": "What is 2 + 2?", "category": "simple"},
    {"prompt": "Say hello in French.", "category": "simple"},
    {"prompt": "What color is grass?", "category": "simple"},
    
    # Medium (factual retrieval)
    {"prompt": "Explain photosynthesis in 2 sentences.", "category": "medium"},
    {"prompt": "What are the three laws of thermodynamics?", "category": "medium"},
    {"prompt": "Describe the water cycle briefly.", "category": "medium"},
    
    # Complex (reasoning required)
    {
        "prompt": "A farmer has 17 sheep. All but 9 run away. How many are left? Explain your reasoning.",
        "category": "reasoning"
    },
    {
        "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "category": "reasoning"
    },
    
    # Code generation
    {
        "prompt": "Write a Python function to check if a number is prime.",
        "category": "code"
    },
    {
        "prompt": "Write a Python function to reverse a linked list.",
        "category": "code"
    },
    
    # Long output (stress test throughput)
    {
        "prompt": "Write a detailed 500-word essay about the future of artificial intelligence.",
        "category": "long_output"
    },
]


def create_cerebras_client() -> Optional[OpenAI]:
    """Create Cerebras client (OpenAI-compatible)"""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("Error: CEREBRAS_API_KEY not set")
        print("Sign up at https://cloud.cerebras.ai to get API key")
        return None
    
    return OpenAI(
        base_url="https://api.cerebras.ai/v1",
        api_key=api_key
    )


def profile_request(
    client: OpenAI,
    prompt: str,
    category: str,
    model: str = "llama3.1-8b"
) -> CerebrasProfile:
    """Profile a single request"""
    
    profile = CerebrasProfile(
        prompt=prompt,
        category=category,
        model=model,
        timestamp=datetime.now().isoformat()
    )
    
    try:
        start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        
        end_time = time.perf_counter()
        
        profile.response = response.choices[0].message.content
        profile.tokens_input = response.usage.prompt_tokens
        profile.tokens_output = response.usage.completion_tokens
        profile.latency_ms = (end_time - start_time) * 1000
        
        if profile.latency_ms > 0:
            profile.tokens_per_second = (profile.tokens_output / profile.latency_ms) * 1000
            
    except Exception as e:
        print(f"Error profiling: {e}")
        profile.response = f"ERROR: {e}"
    
    return profile


def run_profiling(
    model: str = "llama3.1-8b",
    num_runs: int = 1,
    output_file: Optional[str] = None
) -> ProfilingResults:
    """Run full profiling suite"""
    
    if not HAS_OPENAI:
        print("Please install openai: pip install openai")
        return ProfilingResults()
    
    client = create_cerebras_client()
    if not client:
        return ProfilingResults()
    
    results = ProfilingResults(
        model=model,
        timestamp=datetime.now().isoformat()
    )
    
    print(f"\n{'='*60}")
    print(f"Cerebras Cloud API Profiling")
    print(f"Model: {model}")
    print(f"Runs per prompt: {num_runs}")
    print(f"{'='*60}\n")
    
    for prompt_data in BENCHMARK_PROMPTS:
        prompt = prompt_data["prompt"]
        category = prompt_data["category"]
        
        print(f"[{category}] {prompt[:50]}...")
        
        for run in range(num_runs):
            profile = profile_request(client, prompt, category, model)
            results.profiles.append(profile)
            
            print(f"  Run {run+1}: {profile.tokens_per_second:.1f} tok/s, "
                  f"{profile.latency_ms:.0f}ms, "
                  f"{profile.tokens_output} tokens")
    
    # Compute summary statistics
    results.summary = compute_summary(results.profiles)
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total requests: {len(results.profiles)}")
    print(f"Average tok/s: {results.summary['avg_tokens_per_second']:.1f}")
    print(f"Average latency: {results.summary['avg_latency_ms']:.0f}ms")
    print(f"P50 latency: {results.summary['p50_latency_ms']:.0f}ms")
    print(f"P99 latency: {results.summary['p99_latency_ms']:.0f}ms")
    
    print(f"\nBy category:")
    for cat, stats in results.summary.get('by_category', {}).items():
        print(f"  {cat}: {stats['avg_tokens_per_second']:.1f} tok/s, "
              f"{stats['avg_latency_ms']:.0f}ms avg")
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    return results


def compute_summary(profiles: List[CerebrasProfile]) -> Dict:
    """Compute summary statistics"""
    
    if not profiles:
        return {}
    
    latencies = [p.latency_ms for p in profiles if p.latency_ms > 0]
    tok_rates = [p.tokens_per_second for p in profiles if p.tokens_per_second > 0]
    
    summary = {
        "total_requests": len(profiles),
        "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
        "p50_latency_ms": statistics.median(latencies) if latencies else 0,
        "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
        "avg_tokens_per_second": statistics.mean(tok_rates) if tok_rates else 0,
        "max_tokens_per_second": max(tok_rates) if tok_rates else 0,
    }
    
    # By category
    categories = set(p.category for p in profiles)
    summary["by_category"] = {}
    
    for cat in categories:
        cat_profiles = [p for p in profiles if p.category == cat]
        cat_latencies = [p.latency_ms for p in cat_profiles if p.latency_ms > 0]
        cat_rates = [p.tokens_per_second for p in cat_profiles if p.tokens_per_second > 0]
        
        summary["by_category"][cat] = {
            "count": len(cat_profiles),
            "avg_latency_ms": statistics.mean(cat_latencies) if cat_latencies else 0,
            "avg_tokens_per_second": statistics.mean(cat_rates) if cat_rates else 0,
        }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cerebras Cloud API Profiling")
    parser.add_argument(
        "--model", 
        default="llama3.1-8b",
        choices=["llama3.1-8b", "llama-3.3-70b", "qwen-3-32b", "glm-4.7"],
        help="Model to profile"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs per prompt"
    )
    parser.add_argument(
        "--output",
        default="results/cerebras_profile.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    run_profiling(
        model=args.model,
        num_runs=args.runs,
        output_file=args.output
    )


if __name__ == "__main__":
    main()
