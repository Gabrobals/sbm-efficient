# Adaptive-K Installation Guide

## Quick Start (5 minutes)

### 1. Install the SDK

```bash
pip install adaptive-k-routing
```

Or with all optional dependencies:
```bash
pip install adaptive-k-routing[cuda,vllm]
```

### 2. Set Your License Key

**Option A: Environment Variable (Recommended)**
```bash
# Linux/Mac
export ADAPTIVE_K_LICENSE="your-license-key-here"

# Windows PowerShell
$env:ADAPTIVE_K_LICENSE = "your-license-key-here"

# Windows CMD
set ADAPTIVE_K_LICENSE=your-license-key-here
```

**Option B: In Code**
```python
from adaptive_k import AdaptiveKRouter

router = AdaptiveKRouter(
    num_experts=8,
    license_key="your-license-key-here"
)
```

### 3. Verify Installation

```bash
adaptive-k license --validate
```

Expected output for Professional license:
```
✅ License Valid
   Company: Your Company Name
   Tier: professional
   Expires: 2027-01-15
   Days Remaining: 365
   Features: cuda_kernels, vllm_integration, tensorrt_integration, priority_support
```

---

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Application                          │
├─────────────────────────────────────────────────────────────┤
│                    adaptive-k-routing                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Router    │  │  Licensing  │  │   Integrations      │  │
│  │ (Adaptive-K)│  │  Validator  │  │ (vLLM, TensorRT...) │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     PyTorch / CUDA                           │
└─────────────────────────────────────────────────────────────┘
```

### License Validation Flow

1. **At Startup**: When you create `AdaptiveKRouter`, it checks:
   - Environment variable `ADAPTIVE_K_LICENSE`
   - Or `license_key` parameter
   
2. **Validation**: The SDK decodes and verifies:
   - Key format (Base64 payload + signature)
   - Signature validity
   - Expiration date
   
3. **Feature Unlocking**: Based on tier:
   - **Community** (no key): Base features, Apache 2.0
   - **Professional**: CUDA kernels, vLLM, TensorRT integration
   - **Enterprise**: Custom optimizations, SLA, redistribution rights

4. **Expiration**: When license expires:
   - SDK continues to work in Community mode
   - Commercial features are disabled
   - Warning message shown at startup

---

## Usage Examples

### Basic Usage (Community Tier)

```python
from adaptive_k import AdaptiveKRouter
import torch

# No license needed for community features
router = AdaptiveKRouter(num_experts=8, top_k_range=(1, 4))

# Route tokens
hidden_states = torch.randn(32, 512, 768)  # batch, seq, hidden
indices, weights = router(hidden_states)
```

### Professional/Enterprise Usage

```python
import os
os.environ["ADAPTIVE_K_LICENSE"] = "eyJ..."  # Your license key

from adaptive_k import AdaptiveKRouter

# Professional features now available
router = AdaptiveKRouter(
    num_experts=8,
    top_k_range=(1, 4),
    use_cuda_kernels=True,      # Professional feature
    enable_tensorrt=True         # Professional feature
)
```

### With vLLM Integration

```python
from adaptive_k.integrations import vLLMAdaptiveRouter

# Requires Professional or Enterprise license
router = vLLMAdaptiveRouter(
    num_experts=8,
    model_name="mistral-7b",
    calibration_samples=1000
)
```

### Check License Status Programmatically

```python
from adaptive_k.licensing import LicenseValidator

validator = LicenseValidator()
info = validator.validate()

print(f"Tier: {info.tier}")
print(f"Valid: {info.valid}")
print(f"Expires: {info.expires}")
print(f"Features: {info.features}")
```

---

## Tier Comparison

| Feature | Community | Professional | Enterprise |
|---------|-----------|--------------|------------|
| Base Routing | ✅ | ✅ | ✅ |
| Calibration | ✅ | ✅ | ✅ |
| CLI Tools | ✅ | ✅ | ✅ |
| CUDA Kernels | ❌ | ✅ | ✅ |
| vLLM Integration | ❌ | ✅ | ✅ |
| TensorRT Integration | ❌ | ✅ | ✅ |
| HuggingFace Integration | ❌ | ✅ | ✅ |
| Priority Support | ❌ | ✅ | ✅ |
| No Attribution Required | ❌ | ✅ | ✅ |
| Custom Optimization | ❌ | ❌ | ✅ |
| SLA Guarantee | ❌ | ❌ | ✅ |
| Redistribution Rights | ❌ | ❌ | ✅ |

---

## Troubleshooting

### "Invalid license key" Error

1. Check key is correctly copied (no extra spaces)
2. Verify with portal: https://adaptive-k.vertexdata.it/portal
3. Check expiration date

### "License expired" Warning

Your license has expired. Options:
1. Continue using Community features (free)
2. Renew at https://adaptive-k.vertexdata.it or contact amministrazione@vertexdata.it

### Environment Variable Not Found

Make sure the variable is set in the same session:

```bash
# Check if set (Linux/Mac)
echo $ADAPTIVE_K_LICENSE

# Check if set (Windows PowerShell)
echo $env:ADAPTIVE_K_LICENSE
```

### CUDA Features Not Working

1. Ensure CUDA is installed: `nvidia-smi`
2. Install with CUDA support: `pip install adaptive-k-routing[cuda]`
3. Verify Professional/Enterprise license is active

---

## Support

- **Documentation**: https://github.com/VertexData/SBM-Efficient
- **License Issues**: amministrazione@vertexdata.it
- **Technical Support** (Professional/Enterprise): priority queue via email
- **Portal**: https://adaptive-k.vertexdata.it/portal

---

## License Key Format

For technical reference, license keys use this format:

```
{base64_payload}.{signature}
```

Where payload contains:
```json
{
  "company": "Company Name",
  "tier": "professional",
  "expires": "2027-01-15",
  "email": "user@company.com",
  "issued": "2026-01-15"
}
```

The signature is a SHA256 hash (first 16 chars) ensuring key integrity.
