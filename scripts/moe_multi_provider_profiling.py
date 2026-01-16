#!/usr/bin/env python3
"""
Multi-Provider MoE Profiling for Adaptive-K Validation

Comprehensive profiling across ALL available MoE models via API:
- DeepSeek-V3 / V3.1 (671B, 256 experts, top-8)
- Qwen3 MoE variants (235B, 480B)
- Cogito MoE (109B, 671B)
- Mixtral (via compatible endpoints)

Usage:
    # Single provider test
    python scripts/moe_multi_provider_profiling.py --provider together --model deepseek-v3.1

    # Full benchmark across all available
    python scripts/moe_multi_provider_profiling.py --full

    # With specific API key
    export TOGETHER_API_KEY="your-key"
    python scripts/moe_multi_provider_profiling.py --provider together

Environment Variables:
    TOGETHER_API_KEY   - Together.ai API key
    OPENROUTER_API_KEY - OpenRouter API key  
    DEEPSEEK_API_KEY   - DeepSeek direct API key
"""

import os
import sys
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
import statistics

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ============================================================================
# MODEL REGISTRY - All MoE models available via API
# ============================================================================

MOE_MODELS = {
    # Together.ai models
    "together": {
        "deepseek-v3.1": {
            "model_id": "deepseek-ai/DeepSeek-V3.1",
            "experts": 256,
            "active_experts": 8,
            "total_params": "671B",
            "price_input": 0.60,
            "price_output": 1.25,
        },
        "deepseek-v3": {
            "model_id": "deepseek-ai/DeepSeek-V3-0324",
            "experts": 256,
            "active_experts": 8,
            "total_params": "671B",
            "price_input": 1.25,
            "price_output": 1.25,
        },
        "qwen3-235b-moe": {
            "model_id": "Qwen/Qwen3-235B-A22B-FP8",
            "experts": 128,
            "active_experts": 22,
            "total_params": "235B",
            "price_input": 0.65,
            "price_output": 3.00,
        },
        "qwen3-coder-480b": {
            "model_id": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
            "experts": 160,
            "active_experts": 35,
            "total_params": "480B",
            "price_input": 2.00,
            "price_output": 2.00,
        },
        "qwen3-next-80b": {
            "model_id": "Qwen/Qwen3-Next-80B-A3B-Instruct",
            "experts": 64,
            "active_experts": 3,
            "total_params": "80B",
            "price_input": 0.15,
            "price_output": 1.50,
        },
        "cogito-109b-moe": {
            "model_id": "cogito/cogito-v2-109b-moe",
            "experts": 64,
            "active_experts": 8,
            "total_params": "109B",
            "price_input": 0.18,
            "price_output": 0.59,
        },
        "cogito-671b-moe": {
            "model_id": "cogito/cogito-v2-671b-moe",
            "experts": 256,
            "active_experts": 8,
            "total_params": "671B",
            "price_input": 1.25,
            "price_output": 1.25,
        },
    },
    
    # OpenRouter models
    "openrouter": {
        "deepseek-chat": {
            "model_id": "deepseek/deepseek-chat",
            "experts": 256,
            "active_experts": 8,
            "total_params": "671B",
            "price_input": 0.14,
            "price_output": 0.28,
        },
        "mixtral-8x7b": {
            "model_id": "mistralai/mixtral-8x7b-instruct",
            "experts": 8,
            "active_experts": 2,
            "total_params": "46.7B",
            "price_input": 0.24,
            "price_output": 0.24,
        },
        "mixtral-8x22b": {
            "model_id": "mistralai/mixtral-8x22b-instruct",
            "experts": 8,
            "active_experts": 2,
            "total_params": "141B",
            "price_input": 0.65,
            "price_output": 0.65,
        },
        "qwen-moe": {
            "model_id": "qwen/qwen-2.5-72b-instruct",  # Dense but for comparison
            "experts": 1,
            "active_experts": 1,
            "total_params": "72B",
            "price_input": 0.23,
            "price_output": 0.40,
        },
    },
    
    # DeepSeek direct API
    "deepseek": {
        "deepseek-chat": {
            "model_id": "deepseek-chat",
            "experts": 256,
            "active_experts": 8,
            "total_params": "671B",
            "price_input": 0.14,
            "price_output": 0.28,
        },
        "deepseek-reasoner": {
            "model_id": "deepseek-reasoner",
            "experts": 256,
            "active_experts": 8,
            "total_params": "671B",
            "price_input": 0.55,
            "price_output": 2.19,
        },
    },
}

PROVIDER_CONFIG = {
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "env_key": "DEEPSEEK_API_KEY",
    },
}


# ============================================================================
# BENCHMARK PROMPTS - Stratified by complexity
# ============================================================================

BENCHMARK_PROMPTS = {
    "trivial": [
        {"prompt": "What is 2+2?", "expected_k": 1},
        {"prompt": "Say hello.", "expected_k": 1},
        {"prompt": "What color is the sky?", "expected_k": 1},
    ],
    "simple": [
        {"prompt": "Translate 'hello world' to French.", "expected_k": 2},
        {"prompt": "What is the capital of Japan?", "expected_k": 2},
        {"prompt": "List 3 primary colors.", "expected_k": 2},
    ],
    "medium": [
        {"prompt": "Explain photosynthesis in 2 sentences.", "expected_k": 4},
        {"prompt": "What are the main differences between HTTP and HTTPS?", "expected_k": 4},
        {"prompt": "Write a Python function to reverse a string.", "expected_k": 4},
    ],
    "complex": [
        {"prompt": "Implement quicksort in Python with comments explaining each step.", "expected_k": 6},
        {"prompt": "Explain the CAP theorem and give a real-world example for each trade-off.", "expected_k": 6},
        {"prompt": "Compare REST vs GraphQL APIs with pros/cons and use cases.", "expected_k": 6},
    ],
    "expert": [
        {
            "prompt": "Design a distributed cache system that handles 1M QPS with < 5ms p99 latency. Include consistency strategy and failure handling.",
            "expected_k": 8
        },
        {
            "prompt": "Write a Python implementation of a lock-free concurrent queue using CAS operations, then analyze its correctness under the memory model.",
            "expected_k": 8
        },
        {
            "prompt": "Explain the mathematical proof of why transformers can approximate any sequence-to-sequence function, referencing the universal approximation theorem.",
            "expected_k": 8
        },
    ],
}


@dataclass
class ModelProfile:
    """Results for a single model"""
    provider: str
    model_name: str
    model_id: str
    model_info: Dict
    profiles: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    timestamp: str = ""


@dataclass
class BenchmarkResults:
    """Full benchmark results"""
    timestamp: str
    models_tested: List[str] = field(default_factory=list)
    results: Dict[str, ModelProfile] = field(default_factory=dict)
    comparison: Dict = field(default_factory=dict)


class MoEProfiler:
    """Multi-provider MoE profiler"""
    
    def __init__(self, provider: str, api_key: Optional[str] = None):
        if not HAS_OPENAI:
            raise ImportError("openai package required. Install: pip install openai")
        
        self.provider = provider
        config = PROVIDER_CONFIG.get(provider)
        if not config:
            raise ValueError(f"Unknown provider: {provider}")
        
        # Get API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(config["env_key"], "")
        
        if not self.api_key:
            raise ValueError(f"API key required. Set {config['env_key']} or use --api-key")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=config["base_url"]
        )
        
        self.available_models = MOE_MODELS.get(provider, {})
    
    def profile_prompt(self, model_id: str, prompt: str, expected_k: int) -> Dict:
        """Profile a single prompt"""
        result = {
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "expected_k": expected_k,
            "success": False,
            "tokens_input": 0,
            "tokens_output": 0,
            "latency_ms": 0,
            "tokens_per_second": 0,
            "estimated_k": "unknown",
            "error": None,
        }
        
        try:
            start = time.perf_counter()
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7,
            )
            latency_ms = (time.perf_counter() - start) * 1000
            
            result["success"] = True
            result["tokens_input"] = response.usage.prompt_tokens if response.usage else 0
            result["tokens_output"] = response.usage.completion_tokens if response.usage else 0
            result["latency_ms"] = latency_ms
            
            if result["tokens_output"] > 0 and latency_ms > 0:
                result["tokens_per_second"] = (result["tokens_output"] / latency_ms) * 1000
            
            # Estimate K from performance characteristics
            result["estimated_k"] = self._estimate_k(result, expected_k)
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def _estimate_k(self, result: Dict, expected_k: int) -> str:
        """Estimate K based on response patterns"""
        tps = result["tokens_per_second"]
        
        # Heuristic: faster TPS often correlates with fewer active experts
        # This is model/provider dependent
        if tps > 60:
            return "low"
        elif tps > 35:
            return "medium"
        else:
            return "high"
    
    def profile_model(self, model_name: str, quick: bool = False) -> ModelProfile:
        """Profile a specific model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available for {self.provider}")
        
        model_info = self.available_models[model_name]
        model_id = model_info["model_id"]
        
        profile = ModelProfile(
            provider=self.provider,
            model_name=model_name,
            model_id=model_id,
            model_info=model_info,
            timestamp=datetime.now().isoformat()
        )
        
        print(f"\n{'='*60}")
        print(f"Profiling: {model_name}")
        print(f"Model ID: {model_id}")
        print(f"Architecture: {model_info['total_params']} / {model_info['experts']} experts / top-{model_info['active_experts']}")
        print(f"{'='*60}")
        
        # Select prompts
        categories = ["trivial", "simple", "medium", "complex", "expert"]
        if quick:
            categories = ["trivial", "medium", "expert"]
        
        for category in categories:
            prompts = BENCHMARK_PROMPTS[category]
            if quick:
                prompts = prompts[:1]  # Just first prompt
            
            print(f"\n[{category.upper()}]")
            for p in prompts:
                result = self.profile_prompt(model_id, p["prompt"], p["expected_k"])
                result["category"] = category
                profile.profiles.append(result)
                
                if result["success"]:
                    print(f"  {result['latency_ms']:6.0f}ms | {result['tokens_per_second']:5.1f} tok/s | K~{result['estimated_k']}")
                else:
                    print(f"  ERROR: {result['error'][:50]}")
                    profile.errors.append(result["error"])
                
                time.sleep(0.3)  # Rate limiting
        
        # Compute summary
        profile.summary = self._compute_summary(profile)
        
        return profile
    
    def _compute_summary(self, profile: ModelProfile) -> Dict:
        """Compute summary statistics"""
        successful = [p for p in profile.profiles if p["success"]]
        
        if not successful:
            return {"error": "No successful profiles"}
        
        tps_list = [p["tokens_per_second"] for p in successful if p["tokens_per_second"] > 0]
        latency_list = [p["latency_ms"] for p in successful]
        
        k_dist = {"low": 0, "medium": 0, "high": 0}
        for p in successful:
            k = p.get("estimated_k", "unknown")
            if k in k_dist:
                k_dist[k] += 1
        
        # By category
        by_category = {}
        for p in successful:
            cat = p.get("category", "unknown")
            if cat not in by_category:
                by_category[cat] = {"tps": [], "latency": []}
            by_category[cat]["tps"].append(p["tokens_per_second"])
            by_category[cat]["latency"].append(p["latency_ms"])
        
        for cat in by_category:
            by_category[cat] = {
                "avg_tps": statistics.mean(by_category[cat]["tps"]) if by_category[cat]["tps"] else 0,
                "avg_latency_ms": statistics.mean(by_category[cat]["latency"]) if by_category[cat]["latency"] else 0,
            }
        
        # Estimate Adaptive-K savings
        model_info = profile.model_info
        baseline_k = model_info.get("active_experts", 8)
        
        # Estimate adaptive K usage
        adaptive_k_sum = k_dist["low"] * 1 + k_dist["medium"] * (baseline_k // 2) + k_dist["high"] * baseline_k
        baseline_sum = len(successful) * baseline_k
        
        savings_pct = ((baseline_sum - adaptive_k_sum) / baseline_sum * 100) if baseline_sum > 0 else 0
        
        return {
            "total_prompts": len(profile.profiles),
            "successful": len(successful),
            "errors": len(profile.errors),
            "avg_tps": statistics.mean(tps_list) if tps_list else 0,
            "avg_latency_ms": statistics.mean(latency_list) if latency_list else 0,
            "k_distribution": k_dist,
            "by_category": by_category,
            "adaptive_k_savings_estimate": round(savings_pct, 1),
            "baseline_experts": baseline_k,
        }


def print_comparison(results: BenchmarkResults):
    """Print comparison table"""
    print(f"\n{'='*80}")
    print("MULTI-MODEL COMPARISON")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'Model':<25} {'Provider':<12} {'Avg TPS':>10} {'Latency':>10} {'Savings':>10}")
    print("-" * 80)
    
    for key, profile in results.results.items():
        s = profile.summary
        if "error" in s:
            print(f"{profile.model_name:<25} {profile.provider:<12} {'ERROR':<30}")
        else:
            print(f"{profile.model_name:<25} {profile.provider:<12} {s['avg_tps']:>10.1f} {s['avg_latency_ms']:>8.0f}ms {s['adaptive_k_savings_estimate']:>9.1f}%")
    
    print("-" * 80)
    
    # Best performer
    best_savings = max(
        (k, v.summary.get("adaptive_k_savings_estimate", 0))
        for k, v in results.results.items()
        if "error" not in v.summary
    )
    print(f"\nBest Adaptive-K Savings: {best_savings[0]} ({best_savings[1]:.1f}%)")


def list_available_models():
    """Print available models"""
    print("\n" + "="*60)
    print("AVAILABLE MOE MODELS FOR PROFILING")
    print("="*60)
    
    for provider, models in MOE_MODELS.items():
        env_key = PROVIDER_CONFIG[provider]["env_key"]
        has_key = "SET" if os.environ.get(env_key) else "NOT SET"
        
        print(f"\n[{provider.upper()}] (API Key: {has_key})")
        print("-" * 40)
        
        for name, info in models.items():
            print(f"  {name:<20} {info['total_params']:>6} | {info['experts']}E top-{info['active_experts']} | ${info['price_input']:.2f}/$1M")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-provider MoE profiling for Adaptive-K validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/moe_multi_provider_profiling.py --list
  python scripts/moe_multi_provider_profiling.py --provider together --model deepseek-v3.1
  python scripts/moe_multi_provider_profiling.py --provider together --all --quick
        """
    )
    
    parser.add_argument("--provider", choices=list(PROVIDER_CONFIG.keys()),
                        help="API provider")
    parser.add_argument("--model", help="Specific model to profile")
    parser.add_argument("--all", action="store_true", help="Profile all models for provider")
    parser.add_argument("--full", action="store_true", help="Full benchmark across all providers")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer prompts)")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--api-key", help="API key (overrides environment)")
    parser.add_argument("--output", default="workspace/moe_profiling_results.json",
                        help="Output file path")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return 0
    
    if not args.provider and not args.full:
        parser.print_help()
        print("\n[!] Specify --provider or use --list to see available models")
        return 1
    
    results = BenchmarkResults(timestamp=datetime.now().isoformat())
    
    try:
        if args.full:
            # Profile across all providers with available keys
            for provider in PROVIDER_CONFIG:
                env_key = PROVIDER_CONFIG[provider]["env_key"]
                if os.environ.get(env_key):
                    try:
                        profiler = MoEProfiler(provider)
                        for model_name in profiler.available_models:
                            try:
                                profile = profiler.profile_model(model_name, quick=args.quick)
                                results.results[f"{provider}/{model_name}"] = profile
                                results.models_tested.append(f"{provider}/{model_name}")
                            except Exception as e:
                                print(f"Error profiling {model_name}: {e}")
                    except Exception as e:
                        print(f"Error initializing {provider}: {e}")
                else:
                    print(f"[SKIP] {provider} - no API key ({env_key})")
        
        elif args.provider:
            profiler = MoEProfiler(args.provider, api_key=args.api_key)
            
            if args.all:
                # Profile all models for this provider
                for model_name in profiler.available_models:
                    try:
                        profile = profiler.profile_model(model_name, quick=args.quick)
                        results.results[f"{args.provider}/{model_name}"] = profile
                        results.models_tested.append(f"{args.provider}/{model_name}")
                    except Exception as e:
                        print(f"Error profiling {model_name}: {e}")
            
            elif args.model:
                profile = profiler.profile_model(args.model, quick=args.quick)
                results.results[f"{args.provider}/{args.model}"] = profile
                results.models_tested.append(f"{args.provider}/{args.model}")
            
            else:
                print("Specify --model or --all")
                return 1
        
        # Print comparison if multiple models
        if len(results.results) > 1:
            print_comparison(results)
        elif len(results.results) == 1:
            key = list(results.results.keys())[0]
            profile = results.results[key]
            s = profile.summary
            
            print(f"\n{'='*60}")
            print(f"RESULTS: {profile.model_name}")
            print(f"{'='*60}")
            print(f"  Prompts: {s['successful']}/{s['total_prompts']} successful")
            print(f"  Avg TPS: {s['avg_tps']:.1f}")
            print(f"  Avg Latency: {s['avg_latency_ms']:.0f}ms")
            print(f"\n  K Distribution:")
            for k, v in s['k_distribution'].items():
                print(f"    {k}: {v}")
            print(f"\n  Adaptive-K Savings Estimate: {s['adaptive_k_savings_estimate']:.1f}%")
        
        # Save results
        output_data = {
            "timestamp": results.timestamp,
            "models_tested": results.models_tested,
            "results": {k: asdict(v) for k, v in results.results.items()},
        }
        
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")
        
    except ValueError as e:
        print(f"\nError: {e}")
        list_available_models()
        return 1
    except ImportError as e:
        print(f"\nError: {e}")
        print("Install: pip install openai")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
