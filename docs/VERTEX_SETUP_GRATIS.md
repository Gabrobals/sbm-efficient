# VERTEX-RESEARCH - Setup GRATIS 🆓

> Sistema di automazione a **ZERO COSTI** usando strumenti gratuiti.

## 💰 Costi: €0/mese

| Strumento | Uso | Costo |
|-----------|-----|-------|
| GitHub Actions | Automazione | ✅ Gratis (repo pubblici) |
| Google Sheets | Database | ✅ Gratis |
| Gmail SMTP | Notifiche | ✅ Gratis |
| JSON locale | Backup | ✅ Gratis |

---

## 🚀 Setup (15 minuti)

### Step 1: Configura Gmail App Password

1. Vai su https://myaccount.google.com/security
2. Attiva **2-Factor Authentication** (se non già attivo)
3. Cerca "App password" e crea una nuova password
4. Scegli "Mail" e "Other (custom name)" → "VERTEX"
5. **Copia la password** (16 caratteri senza spazi)

### Step 2: Configura GitHub Secrets

Vai su GitHub → Repository → Settings → Secrets → Actions

Aggiungi questi secrets:

| Nome | Valore |
|------|--------|
| `GMAIL_USER` | tua.email@gmail.com |
| `GMAIL_APP_PASSWORD` | la password di 16 caratteri |
| `RESEARCHER_NAME` | Il tuo nome |

### Step 3: Configura Environment Locale (Opzionale)

Crea file `.env` nella root del progetto:

```env
# VERTEX-RESEARCH Configuration
GMAIL_USER=tua.email@gmail.com
GMAIL_APP_PASSWORD=xxxx-xxxx-xxxx-xxxx
RESEARCHER_NAME=Gabriele
```

### Step 4: Verifica Setup

```bash
# Test del sistema
python -m src.utils.vertex_simple project new "Test Project"

# Dovresti vedere:
# ✅ Project created: PROJ-20260128-test-project
#    Name: Test Project
#    Phase: Phase 0: Problem Identification
```

---

## 📋 Comandi Disponibili

### Progetti

```bash
# Crea nuovo progetto
python -m src.utils.vertex_simple project new "Adaptive-K Research"

# Lista progetti
python -m src.utils.vertex_simple project list

# Avanza alla prossima fase
python -m src.utils.vertex_simple project advance
```

### Ipotesi

```bash
# Registra ipotesi
python -m src.utils.vertex_simple hypothesis register \
    --statement "Adaptive-K riduce FLOPs del 30% mantenendo accuracy >98%" \
    --conditions "MNIST task, 16 experts, batch size 64" \
    --falsification "FLOPs reduction < 25% OR accuracy < 97%" \
    --thresholds '{"min_flops_reduction": 0.25, "min_accuracy": 0.97}'

# Valuta ipotesi
python -m src.utils.vertex_simple hypothesis evaluate \
    --id "H20260128120000" \
    --results '{"flops_reduction": 0.33, "accuracy": 0.984}'
```

### Esperimenti

```bash
# Avvia esperimento
python -m src.utils.vertex_simple experiment start \
    --config '{"model": "sbm_adaptive_k", "task": "mnist", "seed": 42}'

# Checkpoint (durante l'esperimento)
python -m src.utils.vertex_simple experiment checkpoint \
    --id "EXP-20260128-1200-abc123" \
    --progress 50 \
    --metrics '{"accuracy": 0.95, "loss": 0.15}'

# Completa esperimento
python -m src.utils.vertex_simple experiment complete \
    --id "EXP-20260128-1200-abc123" \
    --results '{"accuracy": 0.984, "flops_mean": 0.33, "latency_ms": 12.5}'
```

### Daily Log

```bash
# Registra progresso giornaliero
python -m src.utils.vertex_simple daily \
    --notes "Completato Phase 2. Implementato adaptive-K router." \
    --blockers "GPU OOM su batch > 128" \
    --hours 4.5
```

### Report

```bash
# Report settimanale
python -m src.utils.vertex_simple report weekly

# Daily standup
python -m src.utils.vertex_simple report standup
```

### Gate Review

```bash
# Review di fase prima di procedere
python -m src.utils.vertex_simple gate review --phase 3
```

---

## 🤖 Automazione con GitHub Actions

Le GitHub Actions sono già configurate:

| Workflow | Quando | Cosa fa |
|----------|--------|---------|
| `vertex-daily.yml` | Ogni giorno 9:00 | Invia standup email |
| `vertex-weekly.yml` | Ogni venerdì 17:00 | Invia report settimanale |

### Trigger Manuale

Puoi anche triggerare manualmente:
1. Vai su GitHub → Actions
2. Seleziona il workflow
3. Click "Run workflow"

---

## 📁 Dove Trovo i Dati?

```
results/
└── vertex/
    ├── current_state.json      # Progetto corrente
    ├── projects/               # Tutti i progetti
    │   └── PROJ-*.json
    ├── hypotheses/             # Ipotesi registrate
    │   └── H*.json
    ├── experiments/            # Esperimenti
    │   └── EXP-*.json
    └── daily/                  # Log giornalieri
        ├── 2026-01-28.json
        └── 2026-01-28.md
```

---

## 🔧 Opzionale: Google Sheets Sync

Se vuoi anche sincronizzare con Google Sheets:

1. Crea un Google Sheet con questi fogli:
   - `Projects`
   - `Hypotheses`
   - `Experiments`
   - `Daily Log`

2. Crea credenziali Service Account:
   - Vai su https://console.cloud.google.com
   - Crea progetto → Enable Sheets API
   - Create credentials → Service Account
   - Scarica JSON e salvalo come `service_account.json`

3. Condividi lo Sheet con l'email del service account

4. Aggiungi lo Sheet ID al `.env`:
   ```env
   VERTEX_SHEET_ID=1abc123...
   ```

---

## ❓ Troubleshooting

### "Email disabled (no credentials)"
→ Configura `GMAIL_USER` e `GMAIL_APP_PASSWORD` nel `.env`

### "Authentication failed"  
→ Verifica che la App Password sia corretta (16 caratteri, senza trattini)

### "No project selected"
→ Prima crea un progetto con `project new`

---

## 🎯 Workflow Giornaliero Tipico

```bash
# Mattina: vedi standup (automatico via email)

# Durante il giorno: registra progresso
python -m src.utils.vertex_simple daily --notes "..."

# Prima di esperimento: registra ipotesi
python -m src.utils.vertex_simple hypothesis register --statement "..."

# Avvia esperimento
python -m src.utils.vertex_simple experiment start --config '{...}'

# A fine giornata: rivedi
python -m src.utils.vertex_simple project list

# Fine settimana: report automatico via email
```

---

**Costo totale: €0/mese** 💪
