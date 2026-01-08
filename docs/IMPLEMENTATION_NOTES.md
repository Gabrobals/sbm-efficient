# SBM-Efficient – Implementation Notes

---

## 0. Scopo

Questo documento è la specifica operativa per implementare SBM-Efficient in **Python/PyTorch** in modo:

* riproducibile;
* misurabile (FLOPs reali, latenza, parametri attivi);
* compatibile con i documenti:
  * `SBM_EFFICIENT_CONCEPT.md`
  * `SBM_EFFICIENT_ARCHITECTURE.md`
  * `SBM_EFFICIENT_EXPERIMENTS.md`

Nessuna implementazione è considerata "corretta" se viola le metriche o le baseline obbligatorie.

---

## 1. Repository structure (da creare subito)

Struttura standard (minima ma completa):

```text
sbm-efficient/
├── README.md
├── requirements.txt
├── pyproject.toml                  # opzionale
├── .gitignore
│
├── docs/
│   ├── SBM_EFFICIENT_CONCEPT.md
│   ├── SBM_EFFICIENT_ARCHITECTURE.md
│   ├── SBM_EFFICIENT_EXPERIMENTS.md
│   └── IMPLEMENTATION_NOTES.md
│
├── configs/
│   ├── baseline_mnist.yaml
│   ├── sbm_mnist.yaml
│   ├── baseline_xor.yaml
│   └── sbm_xor.yaml
│
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── seed.py
│   │   ├── device.py
│   │   ├── io.py
│   │   └── metrics.py
│   │
│   ├── data/
│   │   ├── loaders.py
│   │   └── transforms.py
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── sbm_state.py
│   │   ├── sbm_measure.py
│   │   ├── sbm_experts.py
│   │   └── sbm_model.py
│   │
│   ├── training/
│   │   ├── train_baseline.py
│   │   ├── train_sbm.py
│   │   └── loops.py
│   │
│   ├── profiling/
│   │   ├── flops.py
│   │   ├── latency.py
│   │   └── profiler.py
│   │
│   └── experiments/
│       ├── run.py
│       └── registry.py
│
├── results/
│   ├── runs/                        # un folder per run
│   └── summaries/
│
└── scripts/
    ├── run_baseline.sh
    └── run_sbm.sh
```

Nota: tutta la logica eseguibile sta in `src/`. I file in `scripts/` sono solo comandi.

---

## 2. Naming conventions (rigide)

### 2.1 Esperimenti / run_id

Formato run_id:

```
{date}_{task}_{model}_{seed}_{gitShortSha}

Esempio:
2026-01-04_mnist_sbm_seed42_a1b2c3d
```

### 2.2 Config keys

* snake_case
* nessuna abbreviazione ambigua

Esempi ammessi:

* `decoherence_tau`
* `entropy_lambda`
* `experts_num`
* `experts_top_k`

---

## 3. YAML config template (riproducibilità)

Template base (da usare per tutti i `.yaml`):

```yaml
run:
  task: "mnist"                 # xor | mnist | fashion_mnist | cifar10
  model: "sbm"                  # baseline | sbm | random_routing | static_topk
  seed: 42
  out_dir: "results/runs"
  notes: ""

hardware:
  device: "cuda"                # cuda | cpu
  precision: "fp32"             # fp32 | amp

data:
  batch_size: 128
  num_workers: 4
  shuffle: true

train:
  epochs: 20
  lr: 0.001
  weight_decay: 0.0
  optimizer: "adamw"            # adamw | sgd

sbm:
  experts_num: 16
  experts_top_k: 2
  decoherence_tau:
    start: 2.0
    end: 0.7
    schedule: "linear"          # linear | cosine | step
  entropy_lambda:
    value: 0.02
  routing:
    mode: "softmax"             # softmax (default)

profiling:
  enabled: true
  measure_flops: true
  measure_latency: true
  warmup_steps: 20
  timed_steps: 50
  cuda_events: true

logging:
  log_every_steps: 50
  save_checkpoints: false
  save_model: false
```

Regola: ogni run deve essere completamente definita dal config + commit git.

---

## 4. JSON logging schema (obbligatorio)

Ogni run deve produrre `results/runs/<run_id>/metrics.json` con questo schema minimo:

```json
{
  "run_id": "2026-01-04_mnist_sbm_seed42_a1b2c3d",
  "timestamp": "2026-01-04T01:20:00+01:00",
  "git": {"sha": "a1b2c3d", "dirty": false},
  "config_path": "configs/sbm_mnist.yaml",
  "task": "mnist",
  "model": "sbm",
  "seed": 42,

  "final": {
    "accuracy": 0.0,
    "loss": 0.0,
    "flops_executed": 0,
    "latency_ms": 0.0,
    "active_modules_mean": 0.0,
    "entropy_mean": 0.0
  },

  "profile": {
    "warmup_steps": 20,
    "timed_steps": 50,
    "latency_p50_ms": 0.0,
    "latency_p90_ms": 0.0,
    "latency_p99_ms": 0.0
  }
}
```

Obbligatori (non negoziabili):

* `accuracy`, `loss`
* `flops_executed` (reale, misurato)
* `latency_ms` (wall-clock o CUDA events)
* `active_modules_mean`
* `entropy_mean`

Run senza questi campi sono **invalidi**.

---

## 5. Baseline variants (mapping diretto al protocollo)

Deve esistere una variante eseguibile per ciascuna baseline richiesta da `SBM_EFFICIENT_EXPERIMENTS.md`:

* `baseline` (full compute)
* `random_routing` (stesso K, moduli scelti random)
* `static_topk` (sempre gli stessi K)
* `sbm` (misura + decoerenza)

Tutte devono condividere:

* stesso feature extractor
* stessi experts
* stessi iperparametri di training

---

## 6. Routing e Top-K: regole implementative

### 6.1 Efficienza reale

Il routing deve:

* eseguire SOLO i moduli attivi
* evitare di calcolare output di tutti gli experts e poi mascherarli

Anti-pattern vietato:

* `outputs = [f_i(x) for i in range(N)]` seguito da maschera Top-K

Pattern ammesso:

* selezione indici Top-K
* esecuzione loop solo sugli indici

### 6.2 Differenziabilità

Il Top-K puro non è differenziabile. L'implementazione iniziale deve usare:

* `softmax(s/τ)` come pesi;
* Top-K come selezione di moduli;
* gradienti passano nel routing tramite i pesi dei moduli selezionati.

Non introdurre straight-through estimators nella fase iniziale.

---

## 7. Decoerenza: schedule e penalità

### 7.1 Entropia

Entropia calcolata su `p = softmax(s/τ)`:

$$
H(p) = -\sum_i p_i \log(p_i + \epsilon)
$$

Loss totale:

$$
\mathcal{L} = \mathcal{L}_{task} + \lambda H(p)
$$

### 7.2 Tau schedule

`τ` deve essere loggato per epoch.

Regola:

* `τ` decresce durante il training (collasso progressivo)

---

## 8. Profiling (FLOPs e latency)

### 8.1 FLOPs reali

Obiettivo: misurare le operazioni realmente eseguite dai moduli attivi.

Approccio consigliato:

* wrapper per layers `Linear/Conv` che accumula FLOPs quando vengono chiamati
* contatore resettato a inizio batch

Nota: per la fase iniziale, è accettabile un contatore deterministico basato su:

* shapes runtime
* chiamate effettive

A patto che rifletta correttamente il routing (solo moduli attivi).

### 8.2 Latency

Misurare:

* p50/p90/p99 su `timed_steps`
* con warmup

Se `cuda_events: true`, usare CUDA events per timing su GPU.

---

## 9. Riproducibilità

Obbligatorio:

* seed globale
* seed per dataloader workers
* log di versione pacchetti (opzionale ma consigliato)

Output run folder:

```text
results/runs/<run_id>/
├── metrics.json
├── config.yaml                # copia del config usato
└── stdout.log
```

---

## 10. Workflow Git (anche se lavori "da solo")

Branching consigliato:

* `main`: stabile
* `feature/baseline`
* `feature/sbm-routing`
* `feature/sbm-decoherence`
* `feature/profiling`

Regola:

* ogni merge su `main` deve mantenere i test/esperimenti lanciabili.

---

## 11. "Instruction to Opus" (da incollare)

> Implementa SBM-Efficient seguendo rigidamente:
>
> * `docs/SBM_EFFICIENT_ARCHITECTURE.md`
> * `docs/SBM_EFFICIENT_EXPERIMENTS.md`
> * `docs/IMPLEMENTATION_NOTES.md`
>
> Non introdurre nuove varianti o concetti non documentati.
>
> Output richiesto:
>
> 1. struttura directory come definita;
> 2. runner `src/experiments/run.py` che accetta `--config <yaml>`;
> 3. baseline variants: baseline/full, random_routing, static_topk, sbm;
> 4. logging `metrics.json` conforme allo schema;
> 5. profiling FLOPs e latency che rispetta il routing (solo moduli attivi).

---

*Documento di lavoro – Implementation Notes SBM-Efficient*
