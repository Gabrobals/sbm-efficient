"""
Profile entropy distribution of LLM outputs.

This script analyzes the entropy distribution of a model's output logits
to understand how to set adaptive K thresholds.

Usage:
    python profile_entropy.py --model meta-llama/Llama-3-8B --dataset wikitext
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm

# Will be imported conditionally
transformers = None
datasets = None


def load_dependencies():
    """Lazy load heavy dependencies."""
    global transformers, datasets
    import transformers as tf
    import datasets as ds
    transformers = tf
    datasets = ds


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute entropy of logit distribution."""
    probs = torch.softmax(logits, dim=dim)
    log_probs = torch.log(probs + 1e-9)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    return entropy


def profile_model(
    model_name: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 100,
    max_tokens_per_sample: int = 512,
    device: str = "cuda",
    output_path: Optional[str] = None,
) -> dict:
    """
    Profile entropy distribution of a model.
    
    Args:
        model_name: HuggingFace model name
        dataset_name: Dataset to use
        dataset_config: Dataset configuration
        split: Dataset split
        max_samples: Maximum number of samples to process
        max_tokens_per_sample: Maximum tokens per sample
        device: Device to use
        output_path: Path to save results (optional)
        
    Returns:
        Dict with entropy statistics and histogram
    """
    load_dependencies()
    
    print(f"Loading model: {model_name}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    
    print(f"Loading dataset: {dataset_name}/{dataset_config}")
    dataset = datasets.load_dataset(dataset_name, dataset_config, split=split)
    
    all_entropies = []
    token_positions = []
    
    print(f"Processing {min(max_samples, len(dataset))} samples...")
    
    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(dataset, total=min(max_samples, len(dataset)))):
            if i >= max_samples:
                break
            
            text = sample.get("text", sample.get("content", ""))
            if not text or len(text) < 50:
                continue
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_tokens_per_sample,
            ).to(device)
            
            if inputs.input_ids.shape[1] < 10:
                continue
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits[0]  # [seq_len, vocab_size]
            
            # Compute entropy for each position
            entropies = compute_entropy(logits)
            
            # Store results (skip first token which is often special)
            for pos, ent in enumerate(entropies[1:].cpu().numpy()):
                all_entropies.append(float(ent))
                token_positions.append(pos + 1)
    
    all_entropies = np.array(all_entropies)
    
    # Compute statistics
    results = {
        "model": model_name,
        "dataset": f"{dataset_name}/{dataset_config}",
        "num_tokens": len(all_entropies),
        "statistics": {
            "mean": float(np.mean(all_entropies)),
            "std": float(np.std(all_entropies)),
            "min": float(np.min(all_entropies)),
            "max": float(np.max(all_entropies)),
            "median": float(np.median(all_entropies)),
            "p10": float(np.percentile(all_entropies, 10)),
            "p25": float(np.percentile(all_entropies, 25)),
            "p75": float(np.percentile(all_entropies, 75)),
            "p90": float(np.percentile(all_entropies, 90)),
        },
        "histogram": {},
        "suggested_thresholds": {},
    }
    
    # Compute histogram
    bins = np.linspace(0, 8, 41)  # 0 to 8 in 0.2 increments
    hist, bin_edges = np.histogram(all_entropies, bins=bins)
    results["histogram"] = {
        "bins": bin_edges.tolist(),
        "counts": hist.tolist(),
        "normalized": (hist / len(all_entropies)).tolist(),
    }
    
    # Suggest thresholds for different target distributions
    for name, ratios in [
        ("balanced", [0.25, 0.25, 0.25, 0.25]),
        ("conservative", [0.15, 0.25, 0.35, 0.25]),
        ("aggressive", [0.35, 0.30, 0.20, 0.15]),
    ]:
        thresholds = []
        cumsum = 0.0
        for ratio in ratios[:-1]:
            cumsum += ratio
            idx = int(cumsum * len(all_entropies))
            sorted_ent = np.sort(all_entropies)
            thresholds.append(float(sorted_ent[min(idx, len(sorted_ent) - 1)]))
        results["suggested_thresholds"][name] = thresholds
    
    # Print summary
    print("\n" + "=" * 60)
    print("ENTROPY PROFILE RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Tokens analyzed: {results['num_tokens']:,}")
    print(f"\nStatistics:")
    for key, value in results["statistics"].items():
        print(f"  {key}: {value:.4f}")
    print(f"\nSuggested Thresholds:")
    for name, thresholds in results["suggested_thresholds"].items():
        print(f"  {name}: {thresholds}")
    print("=" * 60)
    
    # Save results
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile LLM entropy distribution")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Model name or path")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        help="Dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum samples to process")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum tokens per sample")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    
    args = parser.parse_args()
    
    profile_model(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.max_samples,
        max_tokens_per_sample=args.max_tokens,
        device=args.device,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
