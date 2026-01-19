# VERTEX-RESEARCH: Automazione Semplice (Zero Cost)

## Overview

Soluzione **100% gratuita** per automatizzare il protocollo VERTEX-RESEARCH senza n8n.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARCHITETTURA SEMPLICE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │   GitHub     │     │   Google     │     │    Email     │   │
│   │   Actions    │────►│   Sheets     │────►│   (Gmail)    │   │
│   │   (gratis)   │     │   (gratis)   │     │   (gratis)   │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│          │                    │                    │            │
│          │                    │                    │            │
│          └────────────────────┼────────────────────┘            │
│                               │                                  │
│                    ┌──────────▼──────────┐                      │
│                    │   vertex_simple.py  │                      │
│                    │   (CLI locale)      │                      │
│                    └─────────────────────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Costi

| Componente | Costo |
|------------|-------|
| GitHub Actions | **€0** (repo pubblico) |
| Google Sheets | **€0** |
| Gmail SMTP | **€0** |
| **TOTALE** | **€0/mese** |

---

## Setup in 10 Minuti

### Step 1: Crea Google Sheet

1. Vai su https://sheets.google.com
2. Crea nuovo foglio chiamato "VERTEX Research Tracker"
3. Crea questi fogli (tabs):
   - `Projects` (colonne: ID, Name, Phase, Status, Created, Notes)
   - `Hypotheses` (colonne: ID, Project, Statement, Conditions, Falsification, Status, Registered)
   - `Experiments` (colonne: ID, Project, Config, Status, Started, Completed, Results)
   - `Daily Log` (colonne: Date, Project, Phase, Notes, Blockers)

4. **Rendi pubblico per lettura**: File → Condividi → Chiunque con il link può visualizzare

5. Copia l'ID del foglio dall'URL:
   ```
   https://docs.google.com/spreadsheets/d/QUESTO_E_L_ID/edit
   ```

### Step 2: Configura Variabili

Crea file `.env` nella root del progetto:

```bash
# .env
VERTEX_SHEET_ID=il_tuo_sheet_id
GMAIL_USER=tua_email@gmail.com
GMAIL_APP_PASSWORD=password_app_16_caratteri
RESEARCHER_NAME=Gabriele Balsamo
```

> **Per Gmail App Password**: 
> 1. Vai su https://myaccount.google.com/apppasswords
> 2. Genera password per "Mail" su "Windows Computer"
> 3. Copia i 16 caratteri (senza spazi)

### Step 3: Installa dipendenze

```bash
pip install gspread google-auth python-dotenv
```

---

## Comandi CLI

```bash
# Nuovo progetto
python -m src.utils.vertex_simple project new "Adaptive-K Speculative Decoding"

# Registra ipotesi
python -m src.utils.vertex_simple hypothesis register \
  --statement "Entropy-based draft length improves spec decoding by >20%" \
  --conditions "LLM with softmax output" \
  --falsification "Speedup < 10%"

# Inizia esperimento
python -m src.utils.vertex_simple experiment start \
  --config '{"model": "llama-3-8b", "threshold": 1.0}'

# Log giornaliero
python -m src.utils.vertex_simple daily \
  --notes "Completato Phase 2, iniziato design esperimenti" \
  --blockers "Nessuno"

# Report settimanale (invia email)
python -m src.utils.vertex_simple report weekly

# Gate review
python -m src.utils.vertex_simple gate review --phase 3

# Completa esperimento
python -m src.utils.vertex_simple experiment complete \
  --results '{"savings": 0.31, "ppl_change": 0.008}'
```

---

## GitHub Actions (Automazioni Gratuite)

### Daily Standup (ogni mattina alle 9:00)

File: `.github/workflows/vertex-daily.yml`

Cosa fa:
- Legge progetti attivi da Google Sheets
- Genera summary con lo stato
- Invia email con priorità del giorno

### Weekly Report (ogni venerdì alle 17:00)

File: `.github/workflows/vertex-weekly.yml`

Cosa fa:
- Genera report settimanale
- Calcola metriche (esperimenti completati, papers letti, etc.)
- Invia email con PDF summary

### Experiment Monitor (ogni ora)

File: `.github/workflows/vertex-monitor.yml`

Cosa fa:
- Controlla se ci sono esperimenti in corso
- Verifica se sono passate più di 24h senza update
- Invia alert se qualcosa sembra bloccato

---

## Struttura File Locali

```
results/
├── vertex/
│   ├── projects/
│   │   └── PROJ-20260119-adaptive-spec.json
│   ├── hypotheses/
│   │   └── H20260119143022.json
│   ├── experiments/
│   │   └── EXP-20260119-1430-abc123.json
│   └── daily/
│       └── 2026-01-19.md
```

Tutto viene salvato sia su Google Sheets che localmente (backup).

---

## Template Email Automatiche

### Daily Standup

```
Subject: 🔬 VERTEX Daily - 19 Gennaio 2026

Buongiorno Gabriele!

📊 STATO PROGETTI
─────────────────
• Adaptive-K Speculative: Phase 4 (Design)
• Information Flow Monitor: Phase 1 (Literature Review)

🎯 PRIORITÀ OGGI
─────────────────
1. Completare design esperimenti per speculative decoding
2. Leggere 3 paper su mutual information in LLMs

⚠️ BLOCKERS
─────────────────
Nessun blocker attivo.

Buon lavoro!
```

### Weekly Report

```
Subject: 📈 VERTEX Weekly Report - Week 3, 2026

RIEPILOGO SETTIMANALE
════════════════════════

📊 METRICHE
• Esperimenti completati: 3
• Paper letti: 12
• Ipotesi registrate: 2
• Gate reviews passati: 1

🔬 PROGRESSI PER PROGETTO

Adaptive-K Speculative Decoding
├── Phase: 4 → 5 (implementazione)
├── Esperimenti: 2 completati
└── Risultati: 28% speedup medio

📋 PROSSIMA SETTIMANA
• Completare implementazione vLLM
• Benchmark su MT-Bench
• Prima draft paper

────────────────────────────────
Report generato automaticamente
VERTEX-RESEARCH Protocol v1.0
```

---

## FAQ

### Come funziona senza n8n?

GitHub Actions sostituisce n8n per le automazioni scheduled. Python script gestisce la logica. Google Sheets è il database. Tutto gratis!

### Posso usarlo offline?

Sì! I file JSON locali funzionano sempre. Google Sheets si sincronizza quando sei online.

### E se GitHub Actions non basta?

Per automazioni più complesse, puoi sempre aggiungere n8n dopo. Ma per il 90% dei casi, questa soluzione è sufficiente.

---

## Comandi Rapidi

```bash
# Alias da aggiungere a .bashrc/.zshrc
alias vx="python -m src.utils.vertex_simple"
alias vxp="vx project"
alias vxh="vx hypothesis" 
alias vxe="vx experiment"
alias vxd="vx daily"
alias vxr="vx report"
```

Con gli alias:
```bash
vxp new "Nuovo Progetto"
vxh register --statement "..."
vxe start --config '{...}'
vxd --notes "Oggi ho fatto X"
vxr weekly
```
