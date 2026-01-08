# SBM-Adaptive-K — Conceptual Extension (Research Track)

---

## Status

⚠️ **NOT IMPLEMENTED — Research only**

SBM-Adaptive-K **non fa parte del core SBM-Efficient**  
e **non è incluso nei benchmark Phase 1**.

Questo documento serve a:
- fissare una direzione di ricerca futura,
- separare concetto, implementazione e validazione,
- evitare scope creep sul core SBM.

---

## 0. Scopo

SBM-Adaptive-K esplora il **controllo dinamico del compute**:
invece di usare un Top-K fisso, il modello seleziona **K(x)** in funzione
dell’incertezza del routing o della difficoltà dell’input.

Obiettivo teorico:
studiare se un budget computazionale adattivo può migliorare
il tradeoff **accuracy ↔ compute / latency** rispetto a:

- Random routing (K fisso)
- Static Top-K (K fisso)
- SBM learnable con K fisso

SBM-Adaptive-K è considerato valido **solo se** riduce FLOPs/latency
**realmente eseguiti**, mantenendo accuracy comparabile.

---

## 1. Definizioni

### 1.1 Experts bank

Dato un insieme di N experts, il router produce logits:

z(x) ∈ ℝⁿ  
p(x) = softmax(z(x) / τ)

---

### 1.2 Entropia del routing

H(x) = − Σᵢ pᵢ(x) log(pᵢ(x) + ε)

Proprietà:
- H(x) ≈ 0 → routing certo (one-hot)
- H(x) alto → routing incerto (input ambiguo)

---

### 1.3 K dinamico

Adaptive-K introduce una policy:

K(x) = f(H(x))

con K(x) ∈ {K₁, K₂, …, Kₘ},  
dove K₁ < K₂ < … < Kₘ ≤ N.

---

## 2. Famiglie di politiche Adaptive-K

### 2.1 Threshold policy (v1 — raccomandata)

Policy deterministica basata su soglie:

K(x) =
- K_min se H(x) < h₁
- K_mid se h₁ ≤ H(x) < h₂
- K_max se H(x) ≥ h₂

Esempio:
- K_min = 1
- K_mid = 2
- K_max = 4

Vantaggi:
- semplice
- robusta
- FLOPs realmente ridotti
- facile da validare

---

### 2.2 Budget-controlled policy (v2 — futura)

Obiettivo:
E[K(x)] ≤ K_budget

Meccanismo:
- le soglie h₁, h₂ vengono adattate online
- controllo sul compute medio

⚠️ **Non implementata finché v1 non è validata**

---

### 2.3 Differentiable expected compute (v3 — opzionale)

Gating continuo gᵢ(x) ∈ [0,1]

⚠️ Alto rischio di anti-pattern:
- può degenerare in full compute mascherato

Implementare **solo se v1/v2 falliscono**.

---

## 3. Regola fondamentale (anti-pattern)

🚫 **È vietato** calcolare tutti gli experts e poi mascherare.

Con Adaptive-K:
- devono essere eseguiti **solo** gli experts selezionati

Se questo vincolo non è rispettato:
- FLOPs e latency sono **invalidi**
- la run va marcata **INVALID**

---

## 4. Metriche richieste (quando implementato)

### 4.1 Metriche aggiuntive

- final.k_mean
- final.k_std
- final.k_histogram
- final.flops_executed
- final.latency_ms

### 4.2 Coerenza attesa

- active_modules_mean ≈ k_mean
- flops_executed ↓ quando k_mean ↓
- latency ↓ (non necessariamente lineare su CPU)

---

## 5. Criteri di accettazione (future)

Per MNIST:

- Accuracy ≥ static_topk_k2 − 0.3%
- FLOPs_mean ≤ 0.75 × static_topk_k2
- k_mean < 2.0 (con K = {1,2,4})
- Nessun collasso su K_max
- Multi-seed (≥5) stabile

Se uno fallisce → Adaptive-K **non entra nel core**.

---

## 6. Implementazione (non attiva)

⚠️ Questa sezione **non implica implementazione immediata**.

Componenti previsti:
- AdaptiveKPolicy (routing)
- Sparse execution per-sample
- Logging K / entropy
- Estensione validate_metrics

L’implementazione è consentita **solo dopo decisione esplicita**.

---

## 7. Non-goals

Adaptive-K **non deve**:
- aumentare il numero di parametri
- introdurre branching non deterministico
- complicare il training loop core SBM
- rendere i benchmark non confrontabili

---

## 8. Decision gate

Adaptive-K entra nel codice **solo se**:
- documentazione Phase 1 è conclusa
- risultati SBM core sono stabili
- c’è evidenza empirica di beneficio compute

In caso contrario:
Adaptive-K resta **ricerca pura**.

---

*SBM-Adaptive-K — Conceptual research document*
