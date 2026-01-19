# 🚀 n8n + VERTEX-RESEARCH: Guida Pratica

> Impara n8n costruendo l'automazione per il tuo protocollo di ricerca!

## 📋 Indice

1. [Cos'è n8n](#cosè-n8n)
2. [Setup in 5 minuti](#setup-in-5-minuti)
3. [Primo Workflow](#primo-workflow-daily-standup)
4. [Workflow Completi](#workflow-completi)
5. [Integrazione con VERTEX](#integrazione-con-vertex)

---

## Cos'è n8n

n8n è un tool di automazione visuale (come Zapier, ma open source e self-hosted).

**Concetti base:**

| Termine | Significato |
|---------|-------------|
| **Workflow** | Una sequenza di azioni automatiche |
| **Node** | Un singolo step (es. "invia email", "chiama API") |
| **Trigger** | L'evento che avvia il workflow (orario, webhook, etc.) |
| **Credential** | Credenziali salvate (API keys, password) |

---

## Setup in 5 minuti

### Opzione A: Docker (Consigliata)

```bash
cd n8n
docker-compose up -d
```

Apri http://localhost:5678
- User: `vertex`
- Password: `research2026`

### Opzione B: npx (Senza Docker)

```bash
npx n8n
```

Apri http://localhost:5678

### Opzione C: n8n Cloud (Gratis per iniziare)

1. Vai su https://n8n.io
2. "Start Free"
3. 5 workflow gratis inclusi

---

## Primo Workflow: Daily Standup

Costruiamo insieme il tuo primo workflow! 🎯

### Step 1: Crea Nuovo Workflow

1. Click "+" in alto a destra
2. Nome: "VERTEX Daily Standup"

### Step 2: Aggiungi Trigger (Orario)

1. Click "+" → Cerca "Schedule"
2. Aggiungi **Schedule Trigger**
3. Configura:
   - Mode: `Cron`
   - Cron Expression: `0 9 * * *` (ogni giorno alle 9:00)

### Step 3: Chiama API VERTEX

1. Click "+" dopo il trigger
2. Cerca "HTTP Request"
3. Configura:
   - Method: `GET`
   - URL: `http://vertex-api:8000/report/standup`
   
### Step 4: Invia Email

1. Click "+" dopo HTTP Request
2. Cerca "Gmail" (o "Send Email")
3. Configura:
   - To: `tua.email@gmail.com`
   - Subject: `🔬 VERTEX Daily - {{ $now.format('yyyy-MM-dd') }}`
   - Body: `{{ $json.standup }}`

### Step 5: Attiva!

1. Toggle "Active" in alto a destra → ON
2. 🎉 Il workflow gira ogni giorno alle 9!

---

## Workflow Completi

### Workflow 1: Nuovo Progetto

```
[Webhook] → [Create Project API] → [GitHub Issue] → [Slack/Email]
```

**Import questo JSON in n8n:**

```json
{
  "name": "VERTEX-01-New-Project",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "new-project"
      },
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://vertex-api:8000/project/new",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "name",
              "value": "={{ $json.body.name }}"
            }
          ]
        }
      },
      "name": "Create Project",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "resource": "issue",
        "operation": "create",
        "owner": "Gabrobals",
        "repository": "sbm-efficient",
        "title": "=🔬 Research: {{ $json.project.name }}",
        "body": "=## New Project\n\n**ID:** {{ $json.project.id }}\n**Phase:** {{ $json.project.phase_name }}\n\n### Checklist Phase 0\n- [ ] Problem statement\n- [ ] Significance\n- [ ] Scope\n- [ ] Success criteria"
      },
      "name": "GitHub Issue",
      "type": "n8n-nodes-base.github",
      "position": [650, 300]
    },
    {
      "parameters": {
        "fromEmail": "vertex@research.com",
        "toEmail": "gabriele@balsamo.dev",
        "subject": "=🚀 New Project: {{ $json.project.name }}",
        "text": "=Project {{ $json.project.id }} created!\n\nPhase: {{ $json.project.phase_name }}"
      },
      "name": "Send Email",
      "type": "n8n-nodes-base.emailSend",
      "position": [850, 300]
    }
  ],
  "connections": {
    "Webhook": { "main": [[{ "node": "Create Project", "type": "main", "index": 0 }]] },
    "Create Project": { "main": [[{ "node": "GitHub Issue", "type": "main", "index": 0 }]] },
    "GitHub Issue": { "main": [[{ "node": "Send Email", "type": "main", "index": 0 }]] }
  }
}
```

**Per importare:**
1. n8n → Settings (⚙️) → Import from File/URL
2. Incolla il JSON

---

### Workflow 2: Experiment Monitor

Monitora esperimenti in corso e notifica quando finiscono.

```
[Schedule 5min] → [Check Experiments] → [IF running] → [Notify Progress]
                                      → [IF complete] → [Evaluate Hypothesis]
```

```json
{
  "name": "VERTEX-02-Experiment-Monitor",
  "nodes": [
    {
      "parameters": {
        "rule": { "interval": [{ "field": "minutes", "minutesInterval": 5 }] }
      },
      "name": "Every 5 Minutes",
      "type": "n8n-nodes-base.scheduleTrigger",
      "position": [250, 300]
    },
    {
      "parameters": {
        "url": "http://vertex-api:8000/experiments?status=running"
      },
      "name": "Get Running",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [{ "value1": "={{ $json.experiments.length }}", "operation": "larger", "value2": 0 }]
        }
      },
      "name": "Has Running?",
      "type": "n8n-nodes-base.if",
      "position": [650, 300]
    },
    {
      "parameters": {
        "channel": "#research",
        "text": "=🔬 {{ $json.experiments.length }} experiments running"
      },
      "name": "Slack Update",
      "type": "n8n-nodes-base.slack",
      "position": [850, 200]
    }
  ],
  "connections": {
    "Every 5 Minutes": { "main": [[{ "node": "Get Running", "type": "main", "index": 0 }]] },
    "Get Running": { "main": [[{ "node": "Has Running?", "type": "main", "index": 0 }]] },
    "Has Running?": { "main": [[{ "node": "Slack Update", "type": "main", "index": 0 }], []] }
  }
}
```

---

### Workflow 3: Gate Review Automation

Quando un progetto è pronto per la gate review, invia checklist via email.

```
[Webhook: gate-request] → [Get Checklist] → [Send Form Email] → [Wait Response] → [Submit Review]
```

```json
{
  "name": "VERTEX-03-Gate-Review",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "gate-review-request"
      },
      "name": "Gate Request",
      "type": "n8n-nodes-base.webhook",
      "position": [250, 300]
    },
    {
      "parameters": {
        "functionCode": "const phase = $input.item.json.body.phase;\nconst checklists = {\n  0: ['Problem statement defined', 'Significance established', 'Scope bounded'],\n  1: ['Literature review complete', 'Gap analysis done', '10+ papers reviewed'],\n  2: ['Math framework defined', 'Assumptions documented'],\n  3: ['Hypotheses falsifiable', 'Thresholds quantified'],\n  4: ['Experiment matrix complete', 'Controls defined'],\n  5: ['Code committed', 'Tests passing', 'Seeds fixed'],\n  6: ['All seeds complete (5+)', 'Logs archived'],\n  7: ['Criteria checked', 'Stats tests done'],\n  8: ['Results interpreted', 'Visualizations created'],\n  9: ['Paper draft complete', 'Proofread']\n};\nreturn { json: { phase, items: checklists[phase] || [] } };"
      },
      "name": "Get Checklist",
      "type": "n8n-nodes-base.function",
      "position": [450, 300]
    },
    {
      "parameters": {
        "fromEmail": "vertex@research.com",
        "toEmail": "gabriele@balsamo.dev",
        "subject": "=🔍 Gate Review - Phase {{ $json.phase }}",
        "html": "=<h2>Gate Review Checklist</h2><ul>{{ $json.items.map(i => '<li>' + i + '</li>').join('') }}</ul><p>Reply with YES/NO for each item.</p>"
      },
      "name": "Send Checklist",
      "type": "n8n-nodes-base.emailSend",
      "position": [650, 300]
    }
  ],
  "connections": {
    "Gate Request": { "main": [[{ "node": "Get Checklist", "type": "main", "index": 0 }]] },
    "Get Checklist": { "main": [[{ "node": "Send Checklist", "type": "main", "index": 0 }]] }
  }
}
```

---

### Workflow 4: Weekly Report

```
[Schedule Friday 17:00] → [Get Stats] → [Format Report] → [Email + Slack]
```

```json
{
  "name": "VERTEX-04-Weekly-Report",
  "nodes": [
    {
      "parameters": {
        "rule": { "interval": [{ "field": "cronExpression", "expression": "0 17 * * 5" }] }
      },
      "name": "Friday 17:00",
      "type": "n8n-nodes-base.scheduleTrigger",
      "position": [250, 300]
    },
    {
      "parameters": { "url": "http://vertex-api:8000/report/weekly" },
      "name": "Get Report",
      "type": "n8n-nodes-base.httpRequest",
      "position": [450, 300]
    },
    {
      "parameters": {
        "fromEmail": "vertex@research.com",
        "toEmail": "gabriele@balsamo.dev",
        "subject": "=📈 VERTEX Weekly Report",
        "text": "={{ $json.report }}"
      },
      "name": "Email Report",
      "type": "n8n-nodes-base.emailSend",
      "position": [650, 300]
    }
  ],
  "connections": {
    "Friday 17:00": { "main": [[{ "node": "Get Report", "type": "main", "index": 0 }]] },
    "Get Report": { "main": [[{ "node": "Email Report", "type": "main", "index": 0 }]] }
  }
}
```

---

## Integrazione con VERTEX

### Architettura Completa

```
┌─────────────────────────────────────────────────────────────┐
│                         TUO PC                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐        ┌─────────────┐                     │
│  │   n8n       │◄──────►│  VERTEX API │                     │
│  │  :5678      │  HTTP  │   :8000     │                     │
│  └──────┬──────┘        └──────┬──────┘                     │
│         │                      │                             │
│         │                      ▼                             │
│         │               ┌─────────────┐                     │
│         │               │ vertex_     │                     │
│         │               │ simple.py   │                     │
│         │               └──────┬──────┘                     │
│         │                      │                             │
│         │                      ▼                             │
│         │               ┌─────────────┐                     │
│         │               │ results/    │                     │
│         │               │ vertex/     │                     │
│         │               └─────────────┘                     │
│         │                                                    │
│         ▼                                                    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              EXTERNAL SERVICES                          │ │
│  ├──────────┬──────────┬──────────┬──────────┬───────────┤ │
│  │  Gmail   │  Slack   │  GitHub  │  Notion  │  Sheets   │ │
│  └──────────┴──────────┴──────────┴──────────┴───────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### API Endpoints

L'API VERTEX espone questi endpoint che n8n può chiamare:

| Endpoint | Method | Descrizione |
|----------|--------|-------------|
| `/project/new` | POST | Crea progetto |
| `/projects` | GET | Lista progetti |
| `/hypothesis/register` | POST | Registra ipotesi |
| `/experiment/start` | POST | Avvia esperimento |
| `/experiment/complete` | POST | Completa esperimento |
| `/daily` | POST | Log giornaliero |
| `/report/weekly` | GET | Report settimanale |
| `/gate/review` | POST | Sottometti gate review |

### Eventi da n8n a VERTEX

n8n può anche inviare eventi al VERTEX:

```bash
# Esempio: n8n chiama webhook quando riceve risposta email
POST http://vertex-api:8000/webhook/n8n
{
  "action": "daily_reminder",
  "data": {}
}
```

---

## 🎓 Esercizi per Imparare

### Livello 1: Base
1. ✅ Crea workflow "Daily Standup" (fatto sopra)
2. Aggiungi un nodo Slack al workflow
3. Testa il webhook manualmente con curl

### Livello 2: Intermedio
4. Crea workflow "Experiment Alert" che notifica quando un esperimento supera 1 ora
5. Aggiungi condizioni IF/ELSE basate su metriche
6. Integra con GitHub per creare issue automatiche

### Livello 3: Avanzato
7. Crea un workflow che analizza i log e suggerisce ottimizzazioni
8. Integra con Notion per dashboard automatica
9. Crea workflow che valuta automaticamente le ipotesi

---

## 🔧 Troubleshooting

### n8n non si connette all'API VERTEX

```bash
# Verifica che l'API sia running
curl http://localhost:8000/status

# Se usi Docker, usa il nome del container
# http://vertex-api:8000 (non localhost!)
```

### Webhook non risponde

```bash
# Testa il webhook
curl -X POST http://localhost:5678/webhook/new-project \
  -H "Content-Type: application/json" \
  -d '{"name": "Test Project"}'
```

### Email non arrivano

1. Verifica credenziali Gmail in n8n
2. Usa App Password (non password normale)
3. Controlla spam folder

---

## 📚 Risorse

- [n8n Documentation](https://docs.n8n.io/)
- [n8n Community](https://community.n8n.io/)
- [Workflow Templates](https://n8n.io/workflows/)

---

*Ora hai tutto per automatizzare la tua ricerca con n8n! 🚀*
