# Cerebras Strategic Approach

**Date**: January 19, 2026
**Status**: 🟡 Planning Phase
**Priority**: HIGH (timing critico per deal OpenAI)

---

## Executive Summary

Cerebras ha firmato un deal da **$10B con OpenAI** per low-latency inference (Q1 2026). Adaptive-K può amplificare il valore dell'hardware Cerebras riducendo i token necessari per reasoning tasks.

**Value proposition**: `Hardware speed × Software efficiency = Multiplicative gains`

---

## 1. Perché Cerebras è Strategico

### 1.1 Il Deal OpenAI

| Dettaglio | Valore |
|-----------|--------|
| Valore contratto | >$10 miliardi |
| Durata | 3 anni (2026-2028) |
| Capacità | 750 megawatt |
| Focus | Low-latency inference per reasoning models |
| Deployment | Q1 2026 (ADESSO) |

### 1.2 Perché cercano Adaptive-K

OpenAI usa Cerebras per **reasoning models** (o1, futuri):
- Generano **migliaia di token** per risposta
- Ogni token ridotto = risparmio moltiplicato
- Cerebras fa token veloci, Adaptive-K fa **meno token necessari**

**Math**:
```
Cerebras: 20x più veloce
Adaptive-K: 40% meno compute
Combinato: 20x × 1.67 = 33x improvement
```

### 1.3 Architettura Hardware Compatibile

| Feature WSE-3 | Rilevanza Adaptive-K |
|---------------|----------------------|
| 21 PB/s memory bandwidth | Routing dinamico senza bottleneck |
| SLAC cores (Sparse Linear Algebra) | Ottimizzati per sparsity |
| Dataflow execution | Elimina multiply-by-zero |
| 900k AI cores | Parallel expert execution |

---

## 2. Chi Contattare

### 2.1 Decision Makers

| Nome | Ruolo | LinkedIn | Priorità |
|------|-------|----------|----------|
| **Hagay Lupesko** | SVP for AI Inference | [LinkedIn](https://linkedin.com/in/hagay-lupesko) | ⭐⭐⭐ TOP |
| **Andrew Feldman** | CEO & Co-founder | [LinkedIn](https://linkedin.com/in/andrewdfeldman) | ⭐⭐ (molto tecnico) |
| **Dhiraj Mallik** | VP of Products | [LinkedIn](https://linkedin.com/in/dhirajmallik) | ⭐⭐ |
| **Sean Lie** | CTO & Co-founder | [LinkedIn](https://linkedin.com/in/seanlie) | ⭐⭐ |

### 2.2 Technical Team

| Team | Contatto via |
|------|--------------|
| ML Research | publications@cerebras.net |
| Developer Relations | devrel@cerebras.net |
| Partnerships | partnerships@cerebras.net |

### 2.3 Channels

| Canale | Approccio |
|--------|-----------|
| **LinkedIn** | Direct message a Hagay Lupesko |
| **GitHub** | Issue su cerebras/modelzoo |
| **Email** | Technical whitepaper a partnerships@ |
| **Conference** | GTC 2026 (Marzo) - se presentano |

---

## 3. Pitch Strategy

### 3.1 One-liner

> "Cerebras ha risolto l'hardware bottleneck per inference. Adaptive-K risolve il compute waste nel routing. Insieme, redefiniamo l'economics dei reasoning models."

### 3.2 Value Proposition per OpenAI Deal

```
Deal attuale:           $10B per 3 anni
Token generati/giorno:  ~100B (stima)
Costo per token:        $0.00001

Con Adaptive-K (40% reduction):
Savings/giorno:         $400,000
Savings/anno:           $146M
Savings su 3 anni:      $438M
```

**ROI per Cerebras**: Offrire Adaptive-K aumenta il valore percepito senza costi hardware aggiuntivi.

### 3.3 Technical Fit

| Cerebras Tech | Adaptive-K Integration |
|---------------|------------------------|
| Model Zoo (PyTorch) | Drop-in router replacement |
| CS-3 inference API | Entropy computation hook |
| Sparse execution | Native support per dynamic K |

---

## 4. Materiale da Preparare

### 4.1 Technical Whitepaper (2-3 pagine) ✅ COMPLETATO

**File**: `outreach/cerebras/ADAPTIVE_K_CEREBRAS_WHITEPAPER.md`

Contenuto:
1. Problem: MoE fixed-K routing waste
2. Solution: Entropy-guided dynamic K
3. Results: 30-52% validated savings
4. Cerebras Fit: WSE-3 architecture alignment
5. Projected Impact: $150-400M savings on OpenAI deal
6. Integration Roadmap: 8-week plan

### 4.2 API Profiling Script ✅ COMPLETATO

**File**: `scripts/cerebras_api_profiling.py`

```bash
# Sign up at https://cloud.cerebras.ai
export CEREBRAS_API_KEY="your-key"
python scripts/cerebras_api_profiling.py --model llama3.1-8b --output results/cerebras_profile.json
```

### 4.3 Benchmark Request

Chiedere accesso a:
- Cerebras Cloud API (inference)
- Model Zoo fork permission
- WSE-3 profiling tools

### 4.3 Proof of Concept Scope

1. Fork Cerebras Model Zoo
2. Implement Adaptive-K in their PyTorch stack
3. Benchmark su Llama models (già supportati)
4. Show metrics: FLOPs reduction, throughput, accuracy

---

## 5. Outreach Timeline

| Data | Azione | Deliverable |
|------|--------|-------------|
| Jan 20-22 | Write technical whitepaper | `WHITEPAPER.md` |
| Jan 23 | LinkedIn research su Hagay Lupesko | Notes |
| Jan 24 | Send LinkedIn connection request | Message draft |
| Jan 25-26 | If connected: Send whitepaper | Follow-up |
| Jan 27+ | Email fallback se no response | partnerships@cerebras.net |
| Feb | Follow up based on response | TBD |

---

## 6. LinkedIn Message Draft (Hagay Lupesko)

```
Hi Hagay,

Congratulations on the OpenAI partnership - the WSE-3 architecture is 
impressive for low-latency inference.

I've developed Adaptive-K, an entropy-guided routing method that reduces 
MoE compute by 30-52% (validated on Mixtral, Qwen-MoE, OLMoE). For 
reasoning models generating thousands of tokens, this could translate 
to significant cost savings.

Given Cerebras' native sparsity support and the OpenAI focus on reasoning 
workloads, I believe there's strong synergy.

I've prepared a short technical brief - would you be open to a 15-min call 
to discuss potential integration with the Model Zoo?

Best,
Gabriele Balsamo
DOI: 10.5281/zenodo.18282008
```

---

## 7. Email Template (partnerships@)

**Subject**: Adaptive-K Routing: 30-52% MoE Compute Reduction for WSE-3

```
Dear Cerebras Partnerships Team,

I'm reaching out regarding a potential technical collaboration that could 
enhance the value of your OpenAI inference deployment.

**The Opportunity**:
Adaptive-K is an entropy-guided routing method that dynamically selects 
the number of MoE experts based on input complexity. Validated results:

| Model | Compute Reduction | Quality Impact |
|-------|-------------------|----------------|
| Mixtral 8x7B | 31.0% | +0.8% PPL |
| Qwen-MoE | 32.4% | +0.3% PPL |
| OLMoE-1B-7B | 24.7% | +0.5% PPL |

**Why Cerebras**:
- WSE-3's native sparsity support (SLAC cores) can accelerate dynamic routing
- 21 PB/s bandwidth eliminates memory bottleneck for adaptive selection
- Reasoning workloads (OpenAI o1) benefit most from "right-sized" compute

**Resources**:
- DOI: https://doi.org/10.5281/zenodo.18282008
- GitHub: https://github.com/Gabrobals/sbm-efficient
- TensorRT-LLM PR: https://github.com/NVIDIA/TensorRT-LLM/pull/10672

I've prepared a technical whitepaper specific to Cerebras integration. 
Would someone from your ML team be interested in reviewing it?

Best regards,
Gabriele Balsamo
amministrazione@vertexdata.it
```

---

## 8. Risk Assessment

| Rischio | Probabilità | Mitigazione |
|---------|-------------|-------------|
| Ignorano l'outreach | Alta | Multi-channel (LinkedIn + Email + GitHub) |
| Già hanno soluzione interna | Media | Focus su "complementary" not "replacement" |
| NDA richiesto per discussione | Media | Valutare con avvocato prima di firmare |
| Lungo ciclo decisionale | Alta | Pazienza, follow-up mensile |

---

## 9. Success Metrics

| Outcome | Valore |
|---------|--------|
| Response from technical team | ⭐ Good |
| Request for whitepaper/demo | ⭐⭐ Great |
| Call scheduled | ⭐⭐⭐ Excellent |
| PoC collaboration | 🏆 Win |

---

## 10. Notes

**Timing critico**: Il deal OpenAI è in fase di deployment (Q1 2026 = adesso). 
Cerebras ha incentivo a mostrare valore aggiunto velocemente.

**Competitive angle**: Nvidia sta spingendo Nemotron 3 con reasoning optimization.
Cerebras ha bisogno di differenziatori software, non solo hardware speed.

**Pre-IPO**: Cerebras valuation $22B, probabilmente IPO 2026. 
Motivati a mostrare innovation per investor story.
