# SBM / SBM-H / SBM-Adaptive-K — Project Context & Roadmap (for Opus)

## 0. Scopo di questo file
Questo documento serve a fornire a Opus **contesto completo e unificato** sul progetto SBM, includendo:
- lavoro già svolto (con risultati numerici),
- stato di maturità tecnologica (TRL),
- roadmap vincolante step-by-step,
- cosa implementare ora e cosa **NON** implementare.

Opus deve attenersi a questa roadmap senza introdurre variazioni architetturali non richieste.

---

## 1. Visione del progetto (high level)

**SBM (Superposed Bit Model)** è un **sistema di controllo adattivo del compute**, non una tecnica di compressione.

Obiettivo:
- ridurre FLOPs e latenza **realmente eseguiti**
- mantenendo accuracy comparabile
- tramite routing basato su incertezza (entropia)
- senza dipendere da hardware specifico
- e senza degradare deterministicamente l’output (modalità controllabili)

SBM si colloca come **meta-framework di efficienza**, che governa modelli esistenti.

---

## 2. Componenti del progetto (concettuali)

### 2.1 SBM-Efficient (core)
- Routing learnable
- Top-K sparsificazione
- Temperature schedule (tau)
- Entropy regularization
- Sparse execution reale (no masking)

### 2.2 SBM-Adaptive-K (estensione attiva)
- K(x) dinamico in funzione dell’incertezza
- Policy v1: threshold su entropia
- Obiettivo: miglior tradeoff accuracy ↔ FLOPs

### 2.3 SBM-H (ricerca concettuale, non prioritaria ora)
- Embedding qubit-like / Hilbert space
- Born rule come misura
- NON è quantum computing fisico
- Serve come possibile **inductive bias**, non come driver immediato di efficienza

⚠️ **Nota vincolante**  
SBM-H non va integrato ora nel core finché:
- Adaptive-K non dimostra vantaggio netto di FLOPs su benchmark standard.

---

## 3. Stato attuale del progetto (LAVORO GIÀ FATTO)

### 3.1 Training & risultati validati (Fase A = DONE)

Multi-seed (5 seed) su MNIST completati e validati per:

- `baseline` (full compute)
- `random_routing`
- `static_topk` (K=2)
- `sbm` (K fisso)
- `sbm_adaptive_k` (K dinamico, v1 threshold)

**Aggregato MNIST (static_topk vs adaptive_k, 5 seed):**
- static_topk: acc=0.97964 ±0.00382, FLOPs=2,073,924, k_mean=2.0, F1=0.97951
- adaptive_k: acc=0.97986 ±0.00268, FLOPs=1,721,688 (−17%), k_mean=1.66, F1=0.97968

Conclusioni Fase A:
- Accuracy/precision/recall/F1 macro e micro allineate; nessun aumento FP/FN.
- Riduzione compute reale (~17%) con routing adattivo (k std > 0, entropy non collassa).
- Adaptive-K v1 diventa baseline ufficiale per controllo computazionale.

Tutti i run hanno:
- `metrics.json` con campi Fase A (precision/recall/F1, confusion matrix, k metrics)
- pre-flight e post-flight PASS
- schema coerente

---

## 4. Stato di maturità (TRL)

- **TRL 3–4**: validazione sperimentale in ambiente di laboratorio → COMPLETATA
- **Obiettivo prossimo**: **TRL 6**
  - prototipo dimostrato in ambiente rilevante
  - benchmark su dataset standard pubblici
  - metriche ML complete (non solo accuracy)

---

## 5. Problema emerso → cosa manca ora

Finora:
- focus su accuracy + FLOPs + latency

Mancano ancora:
- precision / recall / F1
- analisi falsi positivi / falsi negativi
- test di **robustezza logica** (iniezione di falla)
- validazione su dataset pubblici esterni (Hugging Face)

Questi sono **obbligatori** per passare a TRL 5–6.

---

## 6. Roadmap vincolante (STEP-BY-STEP)

### FASE A — Metriche ML complete (COMPLETATA)

- Precision/recall/F1 (macro + micro) loggate in metrics.json.
- Confusion matrix e FP/FN espliciti.
- Multi-seed 5x MNIST con mean±std, nessuna regressione di accuracy.
- Risultato: Adaptive-K v1 conferma stessa accuracy di static_topk con −17% FLOPs.

---

### FASE B — Test di “falla logica” (robustezza)

**Obiettivo**: simulare errore interno controllato.

**B1 — Iniezione di errore**
Esempi:
- rumore sulle logits
- perturbazione sull’entropia
- dropout strutturato sugli expert

**B2 — Misura risposta**
- variazione accuracy
- variazione precision/recall
- variazione k_mean
- stabilità routing

**B3 — Confronto**
- baseline vs static_topk vs sbm vs adaptive_k
- Adaptive-K deve degradare **meno o uguale** a static_topk

---

### FASE C — Dataset Hugging Face (TRL 6)

⚠️ **Vincolo importante**  
Usare Hugging Face **non come dataset singolo**, ma come **harness standard**.

**C1 — HF Loader**
- File: `src/data/hf_loader.py`
- Dataset configurabili via YAML
- Cache + checksum

**C2 — Dataset configs**
- `configs/datasets/*.yaml`
- mapping split / label / preprocessing

**C3 — Benchmark pubblico**
- almeno 1 dataset HF (vision o NLP)
- stessi modelli
- multi-seed
- report completo

---

## 7. Cosa NON fare (per ora)

❌ Non integrare:
- SBM-H nel training loop
- quantum gates reali
- Bloch/Cayley nel routing
- policy Adaptive-K v2 (budget) o v3 (differentiable)

Finché:
- v1 Adaptive-K non mostra vantaggio netto e stabile.

---

## 8. Output atteso (standard)

Ogni run:
results/runs/<run_id>/
├── metrics.json
├── config.yaml
├── stdout.log
└── plots/ (opzionale)


Ogni confronto:
results/summaries/<task>_comparison.json


---

## 9. Perché tutto questo è importante (business)

SBM punta a:
- riduzione 30–50% costi inferenza
- mantenendo accuracy invariata (verificata anche su FP/FN)
- senza rifare i modelli
- senza lock-in hardware

Mercati target:
- AI SaaS
- Enterprise inference
- Cloud cost optimization
- Runtime orchestration

---

## 10. Istruzione finale per Opus

Procedere **solo** seguendo l’ordine:
1. FASE A (DONE)
2. FASE B (NEXT)
3. FASE C (ON HOLD)

Ogni step:
- deve passare validazione
- deve produrre output riproducibile
- deve essere compatibile con VS Code workflow

Nessuna deviazione architetturale senza approvazione esplicita.

