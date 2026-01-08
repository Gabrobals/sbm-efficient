# src/training/sbm_loops.py
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.device import get_device
from src.training.loops import (
    get_git_info,
    create_optimizer,
    measure_latency,
)


# =============================================================================
# Input Perturbation Functions (B2: Input Robustness)
# =============================================================================

def _apply_gaussian_noise(x: torch.Tensor, sigma: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Add gaussian noise to input and clamp to [0, 1]."""
    if sigma <= 0.0:
        return x
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def _apply_salt_pepper(x: torch.Tensor, prob: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply salt and pepper noise: with prob p, pixel becomes 0 or 1."""
    if prob <= 0.0:
        return x
    # Generate uniform random for each pixel
    rand = torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    # Salt (white): rand < prob/2 => set to 1
    # Pepper (black): prob/2 <= rand < prob => set to 0
    salt_mask = rand < (prob / 2.0)
    pepper_mask = (rand >= (prob / 2.0)) & (rand < prob)
    out = x.clone()
    out[salt_mask] = 1.0
    out[pepper_mask] = 0.0
    return out


def _apply_occlusion(x: torch.Tensor, ratio: float, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Apply rectangular occlusion mask with area = ratio * total pixels."""
    if ratio <= 0.0:
        return x
    # x shape: (B, C, H, W) or (B, H*W) for flattened
    out = x.clone()
    if x.dim() == 4:
        B, C, H, W = x.shape
        total_pixels = H * W
        occlude_pixels = int(total_pixels * ratio)
        if occlude_pixels <= 0:
            return out
        # Determine rectangle dimensions (roughly square)
        side = int(math.sqrt(occlude_pixels))
        if side <= 0:
            side = 1
        rect_h = min(side, H)
        rect_w = min(occlude_pixels // max(rect_h, 1), W)
        if rect_w <= 0:
            rect_w = 1
        # Random position (deterministic with generator)
        if generator is not None:
            top = int(torch.randint(0, max(H - rect_h + 1, 1), (1,), generator=generator).item())
            left = int(torch.randint(0, max(W - rect_w + 1, 1), (1,), generator=generator).item())
        else:
            top = torch.randint(0, max(H - rect_h + 1, 1), (1,)).item()
            left = torch.randint(0, max(W - rect_w + 1, 1), (1,)).item()
        out[:, :, top:top+rect_h, left:left+rect_w] = 0.0
    elif x.dim() == 2:
        # Flattened: (B, D) - apply mask to portion of features
        B, D = x.shape
        occlude_count = int(D * ratio)
        if occlude_count <= 0:
            return out
        if generator is not None:
            start = int(torch.randint(0, max(D - occlude_count + 1, 1), (1,), generator=generator).item())
        else:
            start = torch.randint(0, max(D - occlude_count + 1, 1), (1,)).item()
        out[:, start:start+occlude_count] = 0.0
    return out


def _apply_inversion(x: torch.Tensor) -> torch.Tensor:
    """Invert input: x -> 1 - x."""
    return 1.0 - x


def get_tau_schedule(
    schedule_type: str,
    tau_start: float,
    tau_end: float,
    current_epoch: int,
    total_epochs: int,
) -> float:
    """
    Compute routing temperature tau for current epoch.

    schedule_type: "constant", "linear", "cosine"
    current_epoch: 1-indexed
    """
    if schedule_type == "constant":
        return float(tau_start)

    progress = (current_epoch - 1) / max(total_epochs - 1, 1)

    if schedule_type == "linear":
        return float(tau_start + (tau_end - tau_start) * progress)

    if schedule_type == "cosine":
        import math

        return float(
            tau_end + (tau_start - tau_end) * (1.0 + math.cos(math.pi * progress)) / 2.0
        )

    return float(tau_start)


def _to_float(x: Any) -> float:
    """Convert tensors / numpy scalars / python numbers to float safely."""
    if x is None:
        return 0.0
    if isinstance(x, (float, int)):
        return float(x)
    if hasattr(x, "item"):
        try:
            return float(x.item())
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return 0.0


def _to_int(x: Any) -> int:
    """Convert to int safely."""
    if x is None:
        return 0
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if hasattr(x, "item"):
        try:
            return int(x.item())
        except Exception:
            pass
    try:
        return int(x)
    except Exception:
        return 0


def _degradation_pct(base: float, noisy: float) -> float:
    """Compute percent degradation relative to a baseline value."""
    if base == 0:
        return 0.0
    return float((base - noisy) / abs(base) * 100.0)


def _init_confusion(num_classes: int, device: torch.device) -> torch.Tensor:
    """Create zero confusion matrix on given device."""
    return torch.zeros((num_classes, num_classes), dtype=torch.long, device=device)


def _update_confusion(
    cm: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Accumulate confusion matrix using bincount (works for multi-class)."""
    # idx = target * C + preds
    idx = target * num_classes + preds
    bin_counts = torch.bincount(idx, minlength=num_classes * num_classes)
    cm += bin_counts.view(num_classes, num_classes)
    return cm


def _classification_from_confusion(cm: torch.Tensor, eps: float = 1e-9) -> Dict[str, Any]:
    """Derive precision/recall/F1 and totals from confusion matrix."""
    cm = cm.to(torch.float64)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp
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


def train_epoch_sbm(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    tau: float = 1.0,
    lambda_entropy: float = 0.0,
) -> Tuple[float, float, float, float]:
    """
    Train one epoch for SBM-style models.

    Expectations:
    - model(data, tau=...) -> logits tensor [B, C]
    - model.get_routing_stats() -> dict with optional keys:
        "entropy", "active_modules", "flops_executed"
      where entropy/active_modules may be python numbers or tensors.
    """
    model.train()

    total_ce_loss = 0.0
    correct = 0
    total = 0

    entropy_sum = 0.0
    active_sum = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, ascii=True)

    for data, target in pbar:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        logits = model(data, tau=tau)

        stats = model.get_routing_stats() if hasattr(model, "get_routing_stats") else {}
        entropy_val = _to_float(stats.get("entropy", 0.0))
        active_val = _to_float(stats.get("active_modules", 0.0))

        ce_loss = criterion(logits, target)

        # Entropy regularization (consistent with your existing behavior):
        # total_loss = CE + lambda * entropy
        total_loss = ce_loss + (float(lambda_entropy) * entropy_val)

        total_loss.backward()
        optimizer.step()

        total_ce_loss += ce_loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        entropy_sum += entropy_val
        active_sum += active_val
        num_batches += 1

        pbar.set_postfix(
            {
                "loss": f"{ce_loss.item():.4f}",
                "acc": f"{100.0 * correct / max(total, 1):.1f}%",
                "tau": f"{tau:.3f}",
                "H": f"{entropy_val:.3f}",
            }
        )

    avg_loss = total_ce_loss / max(len(train_loader), 1)
    accuracy = correct / max(total, 1)
    avg_entropy = entropy_sum / max(num_batches, 1)
    avg_active = active_sum / max(num_batches, 1)

    return avg_loss, accuracy, avg_entropy, avg_active


@torch.no_grad()
def evaluate_sbm(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tau: float = 1.0,
    compute_confusion: bool = False,
    noise_sigma: float = 0.0,
) -> Tuple[float, float, float, float, int, torch.Tensor | None, float]:
    """
    Evaluate SBM-style model.

        Returns:
            (avg_loss, accuracy, avg_entropy, avg_active, flops_sum, confusion or None, latency_ms)
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    entropy_sum = 0.0
    active_sum = 0.0
    flops_sum = 0

    num_batches = 0
    latency_total_ms = 0.0

    # Determine num_classes lazily
    num_classes = None
    cm = None

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        start = time.perf_counter()
        logits = model(data, tau=tau)
        latency_total_ms += (time.perf_counter() - start) * 1000.0

        if noise_sigma and noise_sigma > 0.0:
            logits = logits + torch.randn_like(logits) * float(noise_sigma)
        loss = criterion(logits, target)

        if num_classes is None:
            num_classes = getattr(model, "num_classes", None) or logits.size(1)
            if compute_confusion:
                cm = _init_confusion(num_classes, device)

        stats = model.get_routing_stats() if hasattr(model, "get_routing_stats") else {}
        entropy_val = _to_float(stats.get("entropy", 0.0))
        active_val = _to_float(stats.get("active_modules", 0.0))
        flops_val = _to_int(stats.get("flops_executed", 0))

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if compute_confusion and cm is not None:
            cm = _update_confusion(cm, pred, target, num_classes)

        entropy_sum += entropy_val
        active_sum += active_val
        flops_sum += flops_val
        num_batches += 1

    avg_loss = total_loss / max(len(test_loader), 1)
    accuracy = correct / max(total, 1)
    avg_entropy = entropy_sum / max(num_batches, 1)
    avg_active = active_sum / max(num_batches, 1)
    avg_latency_ms = latency_total_ms / max(num_batches, 1)

    return avg_loss, accuracy, avg_entropy, avg_active, flops_sum, cm, avg_latency_ms


@torch.no_grad()
def evaluate_sbm_with_input_perturbation(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tau: float = 1.0,
    perturbation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float, float, float, int, torch.Tensor | None, float]:
    """
    Evaluate SBM-style model with optional input perturbation.

    Args:
        perturbation_fn: Function to apply to input data before forward pass.
                         If None, no perturbation is applied.

    Returns:
        (avg_loss, accuracy, avg_entropy, avg_active, flops_sum, confusion, latency_ms)
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    entropy_sum = 0.0
    active_sum = 0.0
    flops_sum = 0

    num_batches = 0
    latency_total_ms = 0.0

    num_classes = None
    cm = None

    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)

        # Apply input perturbation if provided
        if perturbation_fn is not None:
            data = perturbation_fn(data)

        start = time.perf_counter()
        logits = model(data, tau=tau)
        latency_total_ms += (time.perf_counter() - start) * 1000.0

        loss = criterion(logits, target)

        if num_classes is None:
            num_classes = getattr(model, "num_classes", None) or logits.size(1)
            cm = _init_confusion(num_classes, device)

        stats = model.get_routing_stats() if hasattr(model, "get_routing_stats") else {}
        entropy_val = _to_float(stats.get("entropy", 0.0))
        active_val = _to_float(stats.get("active_modules", 0.0))
        flops_val = _to_int(stats.get("flops_executed", 0))

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        if cm is not None:
            cm = _update_confusion(cm, pred, target, num_classes)

        entropy_sum += entropy_val
        active_sum += active_val
        flops_sum += flops_val
        num_batches += 1

    avg_loss = total_loss / max(len(test_loader), 1)
    accuracy = correct / max(total, 1)
    avg_entropy = entropy_sum / max(num_batches, 1)
    avg_active = active_sum / max(num_batches, 1)
    avg_latency_ms = latency_total_ms / max(num_batches, 1)

    return avg_loss, accuracy, avg_entropy, avg_active, flops_sum, cm, avg_latency_ms


def train_sbm(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: Dict[str, Any],
    run_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """
    Full training loop for SBM-style models with routing.

    Compatible with:
    - baseline
    - random_routing
    - static_topk
    - sbm
    - sbm_adaptive_k
    """
    # Device
    device = get_device(config.get("hardware", {}).get("device", "cpu"))
    model = model.to(device)

    # Optional: reset Adaptive-K stats
    if hasattr(model, "reset_k_stats"):
        model.reset_k_stats()

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)

    train_config = config.get("train", {})
    epochs = int(train_config.get("epochs", 20))

    sbm_config = config.get("sbm", {})

    # Tau config: nested legacy vs flat
    if "decoherence_tau" in sbm_config:
        tau_cfg = sbm_config.get("decoherence_tau", {})
        tau_start = float(tau_cfg.get("start", 2.0))
        tau_end = float(tau_cfg.get("end", 0.5))
        tau_schedule = str(tau_cfg.get("schedule", "linear"))
    else:
        tau_start = float(sbm_config.get("tau_start", 2.0))
        tau_end = float(sbm_config.get("tau_end", 0.5))
        tau_schedule = str(sbm_config.get("tau_schedule", "cosine"))

    # Entropy lambda: nested vs flat
    if "entropy_lambda" in sbm_config and isinstance(sbm_config["entropy_lambda"], dict):
        lambda_entropy = float(sbm_config["entropy_lambda"].get("value", 0.01))
    else:
        lambda_entropy = float(sbm_config.get("lambda_entropy", 0.01))

    profiling_config = config.get("profiling", {})
    do_profiling = bool(profiling_config.get("enabled", True))

    # Histories
    train_losses: List[float] = []
    test_losses: List[float] = []
    test_accuracies: List[float] = []
    entropies: List[float] = []
    active_modules: List[float] = []

    # Header (ASCII-only)
    print(f"\nTraining SBM on {device} for {epochs} epochs...")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Routing type: {getattr(model, 'routing_type', 'unknown')}")

    # Routing info (safe for Adaptive-K)
    if hasattr(model, "top_k"):
        print(f"Experts: {model.num_experts}, Top-K: {model.top_k}")
    elif hasattr(model, "adaptive_k"):
        k_values = getattr(model.adaptive_k, "k_values", [])
        print(f"Experts: {model.num_experts}, Adaptive-K values: {k_values}")
    else:
        print(f"Experts: {getattr(model, 'num_experts', 'unknown')}")

    print(f"Temperature schedule: {tau_schedule} ({tau_start} -> {tau_end})")
    print(f"Entropy regularization lambda: {lambda_entropy}")

    # Train loop
    for epoch in range(1, epochs + 1):
        tau = get_tau_schedule(tau_schedule, tau_start, tau_end, epoch, epochs)

        tr_loss, tr_acc, tr_entropy, tr_active = train_epoch_sbm(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            tau=tau,
            lambda_entropy=lambda_entropy,
        )

        te_loss, te_acc, te_entropy, te_active, _, _, _ = evaluate_sbm(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            tau=tau,
            compute_confusion=False,
        )

        train_losses.append(float(tr_loss))
        test_losses.append(float(te_loss))
        test_accuracies.append(float(te_acc))
        entropies.append(float(te_entropy))
        active_modules.append(float(te_active))

        print(
            f"Epoch {epoch:3d}: "
            f"tau={tau:.3f}, "
            f"Train Loss={tr_loss:.4f}, "
            f"Test Acc={100.0 * te_acc:.2f}%, "
            f"H={te_entropy:.3f}, "
            f"K_eff={te_active:.1f}"
        )

    # Profiling
    profile_results = {
        "warmup_steps": 0,
        "timed_steps": 0,
        "latency_p50_ms": 0.0,
        "latency_p90_ms": 0.0,
        "latency_p99_ms": 0.0,
    }
    latency_ms = 0.0

    if do_profiling:
        print("\nRunning profiling...")

        warmup = int(profiling_config.get("warmup_steps", 20))
        timed = int(profiling_config.get("timed_steps", 50))
        use_cuda = bool(profiling_config.get("cuda_events", False)) and getattr(device, "type", "") == "cuda"

        sample_data, _ = next(iter(test_loader))
        sample_input = sample_data[:1]

        # IMPORTANT:
        # measure_latency expects an nn.Module and calls model.eval().
        # So we pass the model and temporarily patch forward to force tau=tau_end.
        old_forward = model.forward

        def _patched_forward(x, *args, **kwargs):
            kwargs = dict(kwargs)
            kwargs["tau"] = tau_end
            return old_forward(x, *args, **kwargs)

        model.forward = _patched_forward
        try:
            latency_results = measure_latency(
                model,
                sample_input,
                device,
                warmup_steps=warmup,
                timed_steps=timed,
                use_cuda_events=use_cuda,
            )
        finally:
            model.forward = old_forward

        profile_results.update(
            {
                "warmup_steps": warmup,
                "timed_steps": timed,
                **latency_results,
            }
        )
        latency_ms = float(latency_results.get("latency_p50_ms", 0.0))

        print(f"  Latency p50: {profile_results['latency_p50_ms']:.3f} ms")
        print(f"  Latency p90: {profile_results['latency_p90_ms']:.3f} ms")
        print(f"  Latency p99: {profile_results['latency_p99_ms']:.3f} ms")

    # Final evaluation (reset stats so Adaptive-K stats reflect test set)
    if hasattr(model, "reset_k_stats"):
        model.reset_k_stats()

    final_loss, final_acc, final_entropy, final_active, final_flops_sum, cm, final_latency_ms = evaluate_sbm(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        tau=tau_end,
        compute_confusion=True,
    )
    final_flops = final_flops_sum // max(len(test_loader), 1)
    if latency_ms == 0.0:
        latency_ms = float(final_latency_ms)

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

    # Adaptive-K stats if provided by model.policy.get_stats() (preferred) or model.adaptive_k.get_stats()
    k_stats = {
        "k_mean": float(final_active),
        "k_std": 0.0,
        "k_histogram": {},
        "k_histogram_pct": {},
    }
    if hasattr(model, "policy") and hasattr(model.policy, "get_stats"):
        try:
            k_stats = model.policy.get_stats()
        except Exception:
            pass
    elif hasattr(model, "adaptive_k") and hasattr(model.adaptive_k, "get_stats"):
        try:
            k_stats = model.adaptive_k.get_stats()
        except Exception:
            pass

    # Keep histories aligned with final evaluation
    if test_losses:
        test_losses[-1] = float(final_loss)
    if test_accuracies:
        test_accuracies[-1] = float(final_acc)
    if entropies:
        entropies[-1] = float(final_entropy)
    if active_modules:
        active_modules[-1] = float(final_active)

    # Optional noise robustness sweep (evaluation-only, no training changes)
    eval_cfg = config.get("evaluation", {}) if isinstance(config.get("evaluation", {}), dict) else {}
    noise_sigmas = [float(x) for x in eval_cfg.get("noise_sigmas", [])]

    # Support explicit noise_eval block (recommended)
    noise_eval_cfg = eval_cfg.get("noise_eval") if isinstance(eval_cfg.get("noise_eval"), dict) else None
    if noise_eval_cfg:
        enabled = bool(noise_eval_cfg.get("enabled", False))
        if enabled:
            override_sigmas = noise_eval_cfg.get("sigmas", noise_sigmas)
            noise_sigmas = [float(x) for x in override_sigmas]
        else:
            noise_sigmas = []

    # Deduplicate and sort for stability
    noise_sigmas = sorted({float(s) for s in noise_sigmas if float(s) > 0.0})
    noise_evaluations: List[Dict[str, Any]] = []

    # Baseline reference for degradation calculations
    base_final = {
        "accuracy": float(final_acc),
        "precision": float(cls_stats.get("precision", 0.0)),
        "recall": float(cls_stats.get("recall", 0.0)),
        "f1": float(cls_stats.get("f1", 0.0)),
        "precision_micro": float(cls_stats.get("precision_micro", 0.0)),
        "recall_micro": float(cls_stats.get("recall_micro", 0.0)),
        "f1_micro": float(cls_stats.get("f1_micro", 0.0)),
    }

    if noise_sigmas:
        print("\nRunning noise robustness evaluation (logit noise)...")

    for sigma in noise_sigmas:
        if hasattr(model, "reset_k_stats"):
            model.reset_k_stats()

        n_loss, n_acc, n_entropy, n_active, n_flops_sum, n_cm, _n_latency_ms = evaluate_sbm(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            tau=tau_end,
            compute_confusion=True,
            noise_sigma=sigma,
        )

        n_flops = n_flops_sum // max(len(test_loader), 1)
        n_cls_stats = _classification_from_confusion(n_cm) if n_cm is not None else {
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

        # Adaptive-K stats per-noise pass (fallback to active mean if unavailable)
        n_k_stats = {
            "k_mean": float(n_active),
            "k_std": 0.0,
            "k_histogram": {},
            "k_histogram_pct": {},
        }
        if hasattr(model, "policy") and hasattr(model.policy, "get_stats"):
            try:
                n_k_stats = model.policy.get_stats()
            except Exception:
                pass
        elif hasattr(model, "adaptive_k") and hasattr(model.adaptive_k, "get_stats"):
            try:
                n_k_stats = model.adaptive_k.get_stats()
            except Exception:
                pass

        degradation = {
            "accuracy_pct": _degradation_pct(base_final["accuracy"], n_acc),
            "precision_pct": _degradation_pct(base_final["precision"], n_cls_stats.get("precision", 0.0)),
            "recall_pct": _degradation_pct(base_final["recall"], n_cls_stats.get("recall", 0.0)),
            "f1_pct": _degradation_pct(base_final["f1"], n_cls_stats.get("f1", 0.0)),
            "precision_micro_pct": _degradation_pct(base_final["precision_micro"], n_cls_stats.get("precision_micro", 0.0)),
            "recall_micro_pct": _degradation_pct(base_final["recall_micro"], n_cls_stats.get("recall_micro", 0.0)),
            "f1_micro_pct": _degradation_pct(base_final["f1_micro"], n_cls_stats.get("f1_micro", 0.0)),
        }

        noise_evaluations.append(
            {
                "noise_sigma": float(sigma),
                "accuracy": float(n_acc),
                "loss": float(n_loss),
                "flops_executed": int(n_flops),
                "active_modules_mean": float(n_active),
                "entropy_mean": float(n_entropy),
                # For noise sweeps, reuse profiling latency (p50) to avoid mixing wall-clock noise eval timing
                "latency_ms": float(latency_ms),
                "precision": float(n_cls_stats.get("precision", 0.0)),
                "recall": float(n_cls_stats.get("recall", 0.0)),
                "f1": float(n_cls_stats.get("f1", 0.0)),
                "precision_micro": float(n_cls_stats.get("precision_micro", 0.0)),
                "recall_micro": float(n_cls_stats.get("recall_micro", 0.0)),
                "f1_micro": float(n_cls_stats.get("f1_micro", 0.0)),
                "tp_total": int(n_cls_stats.get("tp", 0)),
                "fp_total": int(n_cls_stats.get("fp", 0)),
                "fn_total": int(n_cls_stats.get("fn", 0)),
                "tn_total": int(n_cls_stats.get("tn", 0)),
                "confusion_matrix": n_cls_stats.get("confusion_matrix", []),
                "k_mean": float(n_k_stats.get("k_mean", n_active)),
                "k_std": float(n_k_stats.get("k_std", 0.0)),
                "k_histogram": n_k_stats.get("k_histogram", {}),
                "k_histogram_pct": n_k_stats.get("k_histogram_pct", {}),
                "degradation_pct": degradation,
            }
        )

    # =============================================================================
    # B2: Input Robustness Sweep (evaluation-only, no training changes)
    # =============================================================================
    robustness_input_cfg = config.get("robustness_input", {})
    input_robust_enabled = bool(robustness_input_cfg.get("enabled", False))
    input_robust_evaluations: List[Dict[str, Any]] = []

    if input_robust_enabled:
        print("\nRunning input robustness evaluation...")

        gaussian_sigmas = [float(x) for x in robustness_input_cfg.get("gaussian_sigmas", [])]
        salt_pepper_probs = [float(x) for x in robustness_input_cfg.get("salt_pepper_probs", [])]
        occlusion_ratios = [float(x) for x in robustness_input_cfg.get("occlusion_ratios", [])]
        do_inversion = bool(robustness_input_cfg.get("inversion", False))

        # Seed generator for reproducibility
        seed = config["run"]["seed"]
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        def _run_input_robust_eval(
            test_name: str,
            level: float,
            perturb_fn: Optional[Callable[[torch.Tensor], torch.Tensor]],
        ) -> Dict[str, Any]:
            """Run single input robustness evaluation and return metrics dict."""
            if hasattr(model, "reset_k_stats"):
                model.reset_k_stats()

            r_loss, r_acc, r_entropy, r_active, r_flops_sum, r_cm, _r_lat = evaluate_sbm_with_input_perturbation(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                tau=tau_end,
                perturbation_fn=perturb_fn,
            )

            r_flops = r_flops_sum // max(len(test_loader), 1)
            r_cls = _classification_from_confusion(r_cm) if r_cm is not None else {
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "precision_micro": 0.0, "recall_micro": 0.0, "f1_micro": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "confusion_matrix": [],
            }

            r_k_stats = {"k_mean": float(r_active), "k_std": 0.0}
            if hasattr(model, "policy") and hasattr(model.policy, "get_stats"):
                try:
                    r_k_stats = model.policy.get_stats()
                except Exception:
                    pass
            elif hasattr(model, "adaptive_k") and hasattr(model.adaptive_k, "get_stats"):
                try:
                    r_k_stats = model.adaptive_k.get_stats()
                except Exception:
                    pass

            degr = {
                "accuracy_pct": _degradation_pct(base_final["accuracy"], r_acc),
                "precision_pct": _degradation_pct(base_final["precision"], r_cls.get("precision", 0.0)),
                "recall_pct": _degradation_pct(base_final["recall"], r_cls.get("recall", 0.0)),
                "f1_pct": _degradation_pct(base_final["f1"], r_cls.get("f1", 0.0)),
                "precision_micro_pct": _degradation_pct(base_final["precision_micro"], r_cls.get("precision_micro", 0.0)),
                "recall_micro_pct": _degradation_pct(base_final["recall_micro"], r_cls.get("recall_micro", 0.0)),
                "f1_micro_pct": _degradation_pct(base_final["f1_micro"], r_cls.get("f1_micro", 0.0)),
            }

            return {
                "test": test_name,
                "level": float(level),
                "accuracy": float(r_acc),
                "loss": float(r_loss),
                "flops_executed": int(r_flops),
                "active_modules_mean": float(r_active),
                "entropy_mean": float(r_entropy),
                "latency_ms": float(latency_ms),
                "precision": float(r_cls.get("precision", 0.0)),
                "recall": float(r_cls.get("recall", 0.0)),
                "f1": float(r_cls.get("f1", 0.0)),
                "precision_micro": float(r_cls.get("precision_micro", 0.0)),
                "recall_micro": float(r_cls.get("recall_micro", 0.0)),
                "f1_micro": float(r_cls.get("f1_micro", 0.0)),
                "tp_total": int(r_cls.get("tp", 0)),
                "fp_total": int(r_cls.get("fp", 0)),
                "fn_total": int(r_cls.get("fn", 0)),
                "tn_total": int(r_cls.get("tn", 0)),
                "k_mean": float(r_k_stats.get("k_mean", r_active)),
                "k_std": float(r_k_stats.get("k_std", 0.0)),
                "degradation_pct": degr,
            }

        # Gaussian noise on input
        for sigma in gaussian_sigmas:
            print(f"  [gaussian] sigma={sigma}")
            if sigma == 0.0:
                perturb = None
            else:
                perturb = lambda x, s=sigma: _apply_gaussian_noise(x, s, gen)
            input_robust_evaluations.append(_run_input_robust_eval("gaussian", sigma, perturb))

        # Salt & pepper noise
        for prob in salt_pepper_probs:
            print(f"  [salt_pepper] prob={prob}")
            if prob == 0.0:
                perturb = None
            else:
                perturb = lambda x, p=prob: _apply_salt_pepper(x, p, gen)
            input_robust_evaluations.append(_run_input_robust_eval("salt_pepper", prob, perturb))

        # Occlusion
        for ratio in occlusion_ratios:
            print(f"  [occlusion] ratio={ratio}")
            if ratio == 0.0:
                perturb = None
            else:
                perturb = lambda x, r=ratio: _apply_occlusion(x, r, gen)
            input_robust_evaluations.append(_run_input_robust_eval("occlusion", ratio, perturb))

        # Inversion
        if do_inversion:
            print("  [inversion]")
            input_robust_evaluations.append(_run_input_robust_eval("inversion", 1.0, _apply_inversion))

        print(f"  Input robustness: {len(input_robust_evaluations)} tests completed")

    # SBM info block (SAFE: do not require model.top_k for Adaptive-K)
    sbm_block: Dict[str, Any] = {
        "num_experts": getattr(model, "num_experts", None),
        "routing_type": getattr(model, "routing_type", "unknown"),
        "tau_start": float(tau_start),
        "tau_end": float(tau_end),
        "tau_schedule": str(tau_schedule),
        "lambda_entropy": float(lambda_entropy),
        "entropy_history": entropies,
        "active_modules_history": active_modules,
    }
    if hasattr(model, "top_k"):
        sbm_block["top_k"] = int(model.top_k)
    elif hasattr(model, "adaptive_k"):
        sbm_block["adaptive_k_values"] = getattr(model.adaptive_k, "k_values", [])
        sbm_block["adaptive_k_thresholds"] = getattr(model.adaptive_k, "h_thresholds", [])

    # Metrics (schema-friendly)
    metrics: Dict[str, Any] = {
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
            "flops_executed": int(final_flops),
            "latency_ms": float(latency_ms),
            "active_modules_mean": float(final_active),
            "entropy_mean": float(final_entropy),
            "noise_sigma": 0.0,  # Baseline (no noise) reference
            "k_mean": float(k_stats.get("k_mean", 0.0)),
            "k_std": float(k_stats.get("k_std", 0.0)),
            "k_histogram": k_stats.get("k_histogram", {}),
            "k_histogram_pct": k_stats.get("k_histogram_pct", {}),
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
        "sbm": sbm_block,
        "training": {
            "epochs": int(epochs),
            "final_train_loss": float(train_losses[-1]) if train_losses else 0.0,
            "loss_std": float(torch.tensor(train_losses).std().item()) if train_losses else 0.0,
            "parameter_count": int(model.count_parameters()),
        },
    }

    if noise_evaluations:
        metrics["noise"] = {
            "baseline_sigma": 0.0,
            "evaluations": noise_evaluations,
        }

    if input_robust_evaluations:
        metrics["robustness_input"] = {
            "baseline": {
                "accuracy": float(base_final["accuracy"]),
                "precision": float(base_final["precision"]),
                "recall": float(base_final["recall"]),
                "f1": float(base_final["f1"]),
                "precision_micro": float(base_final["precision_micro"]),
                "recall_micro": float(base_final["recall_micro"]),
                "f1_micro": float(base_final["f1_micro"]),
            },
            "evaluations": input_robust_evaluations,
        }

    # Save metrics
    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n[OK] Training complete!")
    print(f"   Final accuracy: {100.0 * metrics['final']['accuracy']:.2f}%")
    print(f"   Final entropy: {metrics['final']['entropy_mean']:.4f}")
    print(f"   Metrics saved to: {metrics_path}")

    return metrics
