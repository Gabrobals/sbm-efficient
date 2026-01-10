# SBM-Efficient – Experimental Protocol

---

## 1. Obiettivo

Validare empiricamente che **SBM-Efficient** (routing sparso + misura + decoerenza) fornisce un vantaggio misurabile rispetto a baseline classiche in termini di:

1. **Efficienza computazionale** (FLOPs eseguiti, latenza)
2. **Accuratezza** (test accuracy)
3. **Stabilità** (variance della loss durante training)
4. **Robustezza** (performance sotto rumore/occlusione)

**Non è sufficiente** ottenere accuracy migliore: serve dimostrare che il modello è più efficiente **a parità di capacità** (parametri comparabili).

---

## 2. Task sequence (obbligatoria)

### 2.1 Phase 1: Synthetic controlled tasks

**Task**: XOR generalizzato, compositional reasoning

**Obiettivo**: Verificare che il routing sparso apprende pattern compositivi meglio di baseline dense.

**Metriche chiave**:
- Accuracy su test set out-of-distribution
- FLOPs per forward pass
- Active modules per sample

**Baseline**:
- MLP denso (tutti i moduli attivi)
- Random routing (K moduli random)
- Static Top-K (sempre gli stessi K moduli)

---

### 2.2 Phase 2: Small vision

**Tasks**:
- MNIST (28×28, 10 classi)
- Fashion-MNIST (28×28, 10 classi)
- CIFAR-10 (32×32, 10 classi)

**Obiettivo**: Misurare efficienza su task realistici ma piccoli.

**Metriche chiave**:
- Test accuracy
- FLOPs totali per epoch
- Latency per batch (p50/p90/p99)
- Memory footprint (se rilevante)

**Baseline**: stesse di Phase 1

---

### 2.3 Phase 3: Robustness tests

**Test**:
1. Noise injection (Gaussian, salt-and-pepper)
2. Occlusion (maschere random)
3. Inversion (1 - x)

**Obiettivo**: Verificare che SBM mantiene performance migliori sotto degradazione input.

**Metriche chiave**:
- Accuracy degradation rispetto a clean data
- Active modules variance (stabilità routing)

---

## 3. Baseline variants (obbligatorie)

Ogni esperimento deve includere **tutte** queste varianti:

1. **`baseline`** (full compute)
   - Tutti i moduli attivi (K = N)
   - Nessuna entropia penalty
   - Nessuna decoerenza

2. **`random_routing`**
   - K moduli scelti random per ogni sample
   - Stessa architettura di SBM
   - Serve per dimostrare che il routing learnable ha senso

3. **`static_topk`**
   - Sempre gli stessi K moduli (determinati pre-training o random seed)
   - Dimostra necessità di routing dinamico

4. **`sbm`** (proposta principale)
   - Top-K learnable basato su misura
   - Decoherence schedule (τ da alto a basso)
   - Entropy regularization (λ > 0)

---

## 4. Hyperparameters grid (iniziale)

### 4.1 Architettura

| Parametro | Valori iniziali |
|-----------|-----------------|
| N (experts_num) | 8, 16 |
| K (experts_top_k) | 2, 4 |
| Hidden dim per expert | 128, 256 |

### 4.2 Training

| Parametro | Valori |
|-----------|--------|
| Optimizer | AdamW |
| Learning rate | 1e-3, 1e-4 |
| Weight decay | 0.0, 1e-5 |
| Batch size | 64 (XOR), 128 (vision) |
| Epochs | 50 (synthetic), 20 (vision) |

### 4.3 SBM-specific

| Parametro | Valori |
|-----------|--------|
| τ start | 2.0, 2.5 |
| τ end | 0.5, 0.7 |
| τ schedule | linear, cosine |
| λ (entropy) | 0.01, 0.02, 0.03 |

---

## 5. Mandatory metrics (per ogni run)

Ogni `metrics.json` DEVE contenere:

1. **Test accuracy** (finale, media su ultimi 5 epochs)
2. **Training stability**: std(loss) su training epochs
3. **FLOPs executed**: somma FLOPs reali su test set completo
4. **Latency**: p50/p90/p99 su profiling batch
5. **Active modules mean**: media moduli attivi per sample
6. **Entropy mean**: media entropia routing su test set

**Run senza questi campi sono invalidi.**

---

## 6. Ablation studies (obbligatori)

### 6.1 Phase ablation

**Test**: SBM con φ = 0 (no phase)

**Obiettivo**: Misurare contributo informativo della fase (se applicabile in implementazione SBM-Efficient).

### 6.2 Orthogonality ablation

**Test**: SBM con operatori standard (non Cayley transform)

**Obiettivo**: Verificare che vincolo UᵀU = I migliora stabilità.

### 6.3 Observable type ablation

**Test**: Full observable vs. low-rank vs. mixture

**Obiettivo**: Determinare trade-off capacità/parametri.

---

## 7. Multi-seed evaluation

**Obbligatorio**: Ogni configurazione deve essere eseguita con almeno **5 seeds diversi**.

Report finale deve includere:
- Mean ± std per tutte le metriche
- Worst-case e best-case per accuracy
- Variance plot per FLOPs (verificare consistenza routing)

---

## 8. Profiling protocol

### 8.1 FLOPs measurement

**Approccio**:
- Wrapper su layers che accumula FLOPs quando chiamati
- Reset contatore a inizio batch
- Log FLOPs totali per batch e media per sample

**Verifica**:
- Baseline (K=N) deve avere FLOPs massimi
- SBM (K=2) deve avere FLOPs ≈ 2/N × baseline
- Random routing deve avere FLOPs simili a SBM

### 8.2 Latency measurement

**Setup**:
- Warmup: 20 steps
- Timed: 50 steps
- CUDA events se `cuda_events: true`

**Report**:
- p50/p90/p99 latency in ms
- Latency vs. FLOPs scatter plot (verificare correlazione)

---

## 9. Success criteria

SBM è considerato **valido** se:

1. **Efficienza**: FLOPs(SBM) < 0.3 × FLOPs(baseline) con K=2, N=8
2. **Accuracy**: Accuracy(SBM) ≥ 0.95 × Accuracy(baseline) su task clean
3. **Robustness**: Degradation(SBM) ≤ Degradation(baseline) su task noisy
4. **Stabilità**: std(loss) SBM ≤ std(loss) baseline
5. **Learnable routing**: Accuracy(SBM) > Accuracy(random_routing) con significatività statistica

**Se anche uno di questi criteri fallisce**, il modello necessita revisione.

---

## 10. Failure modes da documentare

Se SBM fallisce, documentare:

1. **Collapse**: tutti i samples usano stessi K moduli (entropia → 0 troppo presto)
2. **Instability**: variance alta routing durante test
3. **Underfitting**: accuracy baseline >> SBM (capacità insufficiente)
4. **No efficiency gain**: FLOPs(SBM) ≈ FLOPs(baseline) (routing non funziona)

Ogni failure mode richiede analisi dettagliata in `results/summaries/`.

---

## 11. Output structure

Ogni esperimento produce:

```text
results/runs/<run_id>/
├── metrics.json         # Schema obbligatorio
├── config.yaml          # Copia config usato
├── stdout.log           # Log completo
└── plots/               # Opzionale: loss curves, routing heatmaps
```

---

## 12. Comparative analysis

Alla fine di ogni phase, generare:

```text
results/summaries/<task>_comparison.json

{
  "task": "mnist",
  "models": {
    "baseline": {"accuracy": 0.98, "flops": 1e6, ...},
    "random_routing": {"accuracy": 0.92, "flops": 2.5e5, ...},
    "static_topk": {"accuracy": 0.94, "flops": 2.5e5, ...},
    "sbm": {"accuracy": 0.97, "flops": 2.2e5, ...}
  },
  "winner": "sbm",
  "efficiency_gain": 4.5,
  "accuracy_retention": 0.99
}
---

## Phase 2 — Adaptive-K Extensions (Not Implemented)

Questa sezione definisce estensioni concettuali pianificate.  
Non sono implementate in codice e **non fanno parte dei risultati Phase 1**.

Servono a:
- fissare direzioni di ricerca future,
- evitare scope creep,
- rendere esplicito cosa è core vs esplorativo.

### C1 — Risk-aware Adaptive-K
- Idea: aumentare K quando cresce l’incertezza (entropia routing o incertezza logits).
- Ipotesi: migliore robustezza sotto input degradati a FLOPs simili.
- Stato: **non implementato**.

### C2 — Budget-controlled Adaptive-K
- Idea: imporre un target su k_mean o FLOPs_mean tramite controllo duale.
- Ipotesi: frontiera Pareto accuracy ↔ compute controllabile.
- Stato: **non implementato**.

### C3 — Curriculum-based Adaptive-K
- Idea: esplorazione iniziale (K alto) → sfruttamento finale (K basso).
- Ipotesi: riduzione varianza tra seed e maggiore stabilità.
- Stato: **non implementato**.

*Experimental Protocol – SBM-Efficient*
