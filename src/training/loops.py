"""
Training loops for SBM-Efficient experiments.

Provides:
- Standard training loop with logging
- Metrics calculation
- JSON output conforming to schema
"""

import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import subprocess

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.seed import set_seed
from src.common.device import get_device


def get_git_info() -> Dict[str, Any]:
    """Get git commit info for reproducibility."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
        
        dirty = len(subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True
        ).stdout.strip()) > 0
        
        return {"sha": sha, "dirty": dirty}
    except:
        return {"sha": "unknown", "dirty": False}


def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> optim.Optimizer:
    """Create optimizer from config."""
    train_config = config.get("train", {})
    opt_name = train_config.get("optimizer", "adamw").lower()
    lr = train_config.get("lr", 0.001)
    weight_decay = train_config.get("weight_decay", 0.0)
    
    if opt_name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


def _init_confusion(num_classes: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)


def _update_confusion(cm: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    idx = target * num_classes + preds
    counts = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += counts.view(num_classes, num_classes)
    return cm


def _classification_from_confusion(cm: torch.Tensor, eps: float = 1e-9) -> Dict[str, Any]:
    cm = cm.to(torch.float64)
    tp = cm.diag()
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - (tp + fp + fn)

    precision_per = tp / torch.clamp(tp + fp, min=eps)
    recall_per = tp / torch.clamp(tp + fn, min=eps)
    f1_per = 2.0 * precision_per * recall_per / torch.clamp(precision_per + recall_per, min=eps)

    precision = precision_per.mean().item()
    recall = recall_per.mean().item()
    f1 = f1_per.mean().item()

    tp_sum = tp.sum().item()
    fp_sum = fp.sum().item()
    fn_sum = fn.sum().item()
    tn_sum = tn.sum().item()

    precision_micro = tp_sum / max(tp_sum + fp_sum, eps)
    recall_micro = tp_sum / max(tp_sum + fn_sum, eps)
    f1_micro = (
        2.0 * precision_micro * recall_micro / max(precision_micro + recall_micro, eps)
        if (precision_micro + recall_micro) > 0
        else 0.0
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "f1_micro": float(f1_micro),
        "tp": int(tp_sum),
        "fp": int(fp_sum),
        "fn": int(fn_sum),
        "tn": int(tn_sum),
        "confusion_matrix": cm.to(torch.int64).cpu().tolist(),
    }


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        (avg_loss, accuracy) tuple
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ascii=True)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100. * correct / total:.1f}%"
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_confusion: bool = False,
    noise_sigma: float = 0.0,
) -> Tuple[float, float, torch.Tensor | None]:
    """
    Evaluate model on test set.
    
    Returns:
        (avg_loss, accuracy) tuple
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    num_classes = None
    cm = None

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)

        if noise_sigma and noise_sigma > 0.0:
            logits = logits + torch.randn_like(logits) * float(noise_sigma)

        loss = criterion(logits, target)
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if num_classes is None:
            num_classes = getattr(model, "num_classes", None) or output.size(1)
            if compute_confusion:
                cm = _init_confusion(num_classes, device)
        if compute_confusion and cm is not None:
            cm = _update_confusion(cm, pred, target, num_classes)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy, cm


def measure_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup_steps: int = 20,
    timed_steps: int = 50,
    use_cuda_events: bool = False
) -> Dict[str, float]:
    """
    Measure inference latency.
    
    Returns:
        Dict with p50, p90, p99 latencies in ms
    """
    model.eval()
    sample_input = sample_input.to(device)
    
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(sample_input)
    
    # Timed runs
    if use_cuda_events and device.type == "cuda":
        torch.cuda.synchronize()
        
        for _ in range(timed_steps):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.no_grad():
                _ = model(sample_input)
            end_event.record()
            
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(timed_steps):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(sample_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    latencies = sorted(latencies)
    n = len(latencies)
    
    return {
        "latency_p50_ms": latencies[int(n * 0.5)],
        "latency_p90_ms": latencies[int(n * 0.9)],
        "latency_p99_ms": latencies[int(n * 0.99)] if n > 100 else latencies[-1]
    }


def count_flops_simple(model: nn.Module, input_shape: tuple) -> int:
    """
    Simple FLOPs counter for baseline models.
    
    This is a simplified counter that estimates FLOPs based on layer types.
    For accurate measurement, use torch.profiler or custom hooks.
    
    Returns:
        Estimated FLOPs for one forward pass
    """
    total_flops = 0
    
    def count_linear(layer, input_size):
        # FLOPs for Linear: 2 * in_features * out_features (multiply-add)
        return 2 * layer.in_features * layer.out_features
    
    def count_conv2d(layer, input_size):
        # FLOPs for Conv2d: 2 * Cout * Cin * K^2 * Hout * Wout
        out_h = input_size[2] // (layer.stride[0] if hasattr(layer, 'stride') else 1)
        out_w = input_size[3] // (layer.stride[1] if hasattr(layer, 'stride') else 1)
        return 2 * layer.out_channels * layer.in_channels * \
               layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            total_flops += count_linear(module, None)
        elif isinstance(module, nn.Conv2d):
            # Approximate based on typical sizes
            total_flops += count_conv2d(module, (1, module.in_channels, 28, 28))
    
    return total_flops


def train_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    run_dir: Path,
    run_id: str
) -> Dict[str, Any]:
    """
    Full training loop for baseline model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader
        config: Full experiment config
        run_dir: Path to run output directory
        run_id: Run identifier
        
    Returns:
        Metrics dictionary conforming to schema
    """
    # Setup
    device = get_device(config.get("hardware", {}).get("device", "cpu"))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    
    train_config = config.get("train", {})
    epochs = train_config.get("epochs", 20)
    
    profiling_config = config.get("profiling", {})
    do_profiling = profiling_config.get("enabled", True)
    
    # Training history
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    print(f"\nTraining on {device} for {epochs} epochs...")
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss, test_acc, _ = evaluate(model, test_loader, criterion, device, compute_confusion=False)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch:3d}: "
              f"Train Loss={train_loss:.4f}, "
              f"Test Loss={test_loss:.4f}, "
              f"Test Acc={100*test_acc:.2f}%")
    
    # Profiling
    profile_results = {
        "warmup_steps": 0,
        "timed_steps": 0,
        "latency_p50_ms": 0.0,
        "latency_p90_ms": 0.0,
        "latency_p99_ms": 0.0
    }
    
    flops_executed = 0
    latency_ms = 0.0
    
    if do_profiling:
        print("\nRunning profiling...")
        
        warmup = profiling_config.get("warmup_steps", 20)
        timed = profiling_config.get("timed_steps", 50)
        use_cuda = profiling_config.get("cuda_events", False) and device.type == "cuda"
        
        # Get sample input
        sample_data, _ = next(iter(test_loader))
        sample_input = sample_data[:1]  # Single sample for latency
        
        latency_results = measure_latency(
            model, sample_input, device,
            warmup_steps=warmup,
            timed_steps=timed,
            use_cuda_events=use_cuda
        )
        
        profile_results.update({
            "warmup_steps": warmup,
            "timed_steps": timed,
            **latency_results
        })
        
        latency_ms = latency_results["latency_p50_ms"]
        
        # Estimate FLOPs
        flops_executed = count_flops_simple(model, sample_data.shape)
        
        print(f"  Latency p50: {latency_results['latency_p50_ms']:.3f} ms")
        print(f"  Latency p90: {latency_results['latency_p90_ms']:.3f} ms")
        print(f"  Estimated FLOPs: {flops_executed:,}")
    
    # Final evaluation with confusion for metrics
    final_loss, final_acc, cm = evaluate(model, test_loader, criterion, device, compute_confusion=True)

    cls_stats = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "precision_micro": 0.0,
        "recall_micro": 0.0,
        "f1_micro": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "confusion_matrix": [],
    }
    if cm is not None:
        cls_stats = _classification_from_confusion(cm)

    # Build metrics.json
    # For baseline: all modules active, no entropy
    sbm_config = config.get("sbm", {})
    experts_num = sbm_config.get("experts_num", 16)
    
    metrics = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "config_path": str(run_dir / "config.yaml"),
        "task": config["run"]["task"],
        "model": config["run"]["model"],
        "seed": config["run"]["seed"],
        
        "final": {
            "accuracy": float(final_acc),
            "loss": float(final_loss),
            "flops_executed": flops_executed,
            "latency_ms": latency_ms,
            "active_modules_mean": float(experts_num),  # Baseline uses all
            "entropy_mean": 0.0,  # No routing entropy for baseline
            # Classification metrics (Fase A)
            "precision": float(cls_stats.get("precision", 0.0)),
            "recall": float(cls_stats.get("recall", 0.0)),
            "f1": float(cls_stats.get("f1", 0.0)),
            "precision_micro": float(cls_stats.get("precision_micro", 0.0)),
            "recall_micro": float(cls_stats.get("recall_micro", 0.0)),
            "f1_micro": float(cls_stats.get("f1_micro", 0.0)),
            "tp_total": int(cls_stats.get("tp", 0)),
            "fp_total": int(cls_stats.get("fp", 0)),
            "fn_total": int(cls_stats.get("fn", 0)),
            "tn_total": int(cls_stats.get("tn", 0)),
            "confusion_matrix": cls_stats.get("confusion_matrix", []),
        },
        
        "profile": profile_results,
        
        # Extra info (not required but useful)
        "training": {
            "epochs": epochs,
            "final_train_loss": train_losses[-1] if train_losses else 0.0,
            "loss_std": float(torch.tensor(train_losses).std()) if train_losses else 0.0,
            "parameter_count": model.count_parameters()
        }
    }
    
    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n[OK] Training complete!")
    print(f"   Final accuracy: {100*metrics['final']['accuracy']:.2f}%")
    print(f"   Metrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    print("Training loop module loaded successfully.")
    print("Use via src/experiments/run.py")
