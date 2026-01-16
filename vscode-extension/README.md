# Adaptive-K Toolkit for VS Code

[![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://marketplace.visualstudio.com/items?itemName=vertexdata.adaptive-k-toolkit)
[![Downloads](https://img.shields.io/badge/downloads-1k+-green)](https://marketplace.visualstudio.com/items?itemName=vertexdata.adaptive-k-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**MoE cost estimation, entropy analysis, and Adaptive-K routing tools for LLM developers.**

![Dashboard Screenshot](images/dashboard-preview.png)

## Features

### 📊 Real-time Cost Estimation
- Token counting in status bar
- Automatic cost calculation
- Savings projection with Adaptive-K

### 🔍 MoE API Detection
- Detects OpenAI, Together.ai, DeepSeek calls
- Highlights optimization opportunities
- Shows potential savings per call

### 📈 Interactive Dashboard
- ROI calculator
- Model comparison
- Usage statistics

### 💡 Code Snippets
- Adaptive-K SDK setup (Python/TypeScript)
- Entropy-based routing
- Cost tracking integration

## Installation

### From VS Code Marketplace
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "Adaptive-K Toolkit"
4. Click Install

### From VSIX
```bash
code --install-extension adaptive-k-toolkit-0.1.0.vsix
```

## Quick Start

1. **Open any Python/TypeScript file** with LLM API calls
2. **Check the status bar** (bottom right) for token count and savings estimate
3. **Use Command Palette** (Ctrl+Shift+P):
   - `Adaptive-K: Estimate MoE Cost` - Detailed cost analysis
   - `Adaptive-K: Open Dashboard` - Interactive ROI calculator
   - `Adaptive-K: Analyze Current File` - Find optimization opportunities

## Configuration

Open Settings (Ctrl+,) and search for "Adaptive-K":

| Setting | Default | Description |
|---------|---------|-------------|
| `adaptive-k.defaultModel` | `deepseek-v3` | MoE model for estimation |
| `adaptive-k.savingsRate` | `35` | Expected savings percentage |
| `adaptive-k.showStatusBar` | `true` | Show status bar item |
| `adaptive-k.dailyTokenBudget` | `1000000` | Daily token budget |

## Supported Models

| Model | Experts | Expected Savings | Status |
|-------|---------|------------------|--------|
| DeepSeek-V3 | 256 | 35% | ✅ Validated |
| Mixtral 8x7B | 8 | 52.5% | ✅ Validated |
| Qwen1.5-MoE | 60 | 32.4% | ✅ Validated |
| OLMoE 1B-7B | 64 | 24.7% | ✅ Validated |
| Qwen3-235B | 128 | 30% | 🔄 Estimated |

## Snippets

Type these prefixes and press Tab:

### Python
- `adaptive-k-setup` - Basic SDK setup
- `entropy-routing` - Entropy-based expert selection
- `cost-estimation` - Cost calculation helper
- `batch-processing` - Batch with per-sample K
- `together-adaptive` - Together.ai integration
- `adaptive-k-metrics` - Metrics logging

### TypeScript
- `adaptive-k-setup-ts` - TypeScript SDK setup
- `openai-adaptive` - OpenAI with cost tracking
- `together-client` - Together.ai client
- `cost-calc` - Cost calculator class

## API Detection

The extension detects these API patterns:

```python
# OpenAI
client.chat.completions.create(...)

# Together.ai
together.complete(...)

# DeepSeek
deepseek_client.chat(...)

# HuggingFace
pipeline("text-generation", ...)

# vLLM
LLM(model="...")
```

## Commands

| Command | Shortcut | Description |
|---------|----------|-------------|
| Estimate MoE Cost | - | Show cost analysis |
| Open Dashboard | - | Interactive calculator |
| Analyze Current File | - | Detect MoE calls |
| Insert SDK Snippet | - | Quick snippet insertion |

## Resources

- [Landing Page](https://adaptive-k.vertexdata.it)
- [Documentation](https://adaptive-k.vertexdata.it/docs)
- [PyPI Package](https://pypi.org/project/adaptive-k-sdk/)
- [GitHub Repository](https://github.com/Gabrobals/sbm-efficient)
- [Technical Whitepaper](https://adaptive-k.vertexdata.it/whitepaper.html)

## Contributing

Contributions welcome! Please see our [Contributing Guide](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with ❤️ by [VertexData](https://vertexdata.it)**

*Adaptive-K: Reduce MoE inference costs by 25-52% with entropy-based dynamic routing.*
