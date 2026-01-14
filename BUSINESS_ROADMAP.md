# Adaptive-K Business Roadmap

> **Vertex Data** - Piano strategico per la commercializzazione di Adaptive-K  
> Ultimo aggiornamento: 14 Gennaio 2026

---

## 🎯 Vision

Diventare lo standard de-facto per l'ottimizzazione dell'inferenza su modelli Mixture-of-Experts, riducendo i costi computazionali del 30-50% senza perdita di qualità.

---

## 📊 Executive Summary

| Metrica | Valore |
|---------|--------|
| Risparmio compute provato | 30-50% |
| Modelli validati | Mixtral, Qwen-MoE, OLMoE |
| Status IP | Open source + PR TensorRT-LLM |
| SDK | ✅ **LIVE su PyPI** |
| Revenue target Y1 | €50-100K |

---

## ⚡ PROSSIME AZIONI (Questa Settimana)

| # | Azione | Impatto | Tempo |
|---|--------|---------|-------|
| 1 | **LinkedIn Post** - Annuncio SDK su PyPI | Visibilità, primi utenti | 30 min |
| 2 | **Email outreach** - 10 aziende che usano MoE | Primi lead consulenza | 2h |
| 3 | **GitHub README** - Link a PyPI + landing page | SEO, credibilità | 30 min |
| 4 | **Hacker News / Reddit** - Post su r/MachineLearning | Traffico, feedback | 1h |

### Post LinkedIn (Bozza)

```
🚀 Just published adaptive-k on PyPI!

Reduce MoE inference costs by 30-50% with entropy-guided dynamic expert selection.

Works with:
✅ Mixtral 8x7B (52% savings)
✅ Qwen-MoE (32% savings)  
✅ OLMoE-1B-7B (25% savings)

pip install adaptive-k

3 lines of code to optimize your MoE model.

Paper + TensorRT-LLM PR in comments 👇

#AI #MachineLearning #LLM #Optimization
```

---

## 🚀 Fasi di Sviluppo

### FASE 1: Consulenza (NOW - Q1 2026)
**Status: ✅ ATTIVO**

**Obiettivo**: Validare il mercato e generare primi ricavi

| Servizio | Prezzo | Timeline |
|----------|--------|----------|
| Proof of Concept | €2,500 | 1-2 settimane |
| Full Implementation | €8,000 | 4-6 settimane |
| Enterprise | Custom | Flessibile |

**Deliverables completati**:
- [x] Landing page live: https://adaptive-k.vertexdata.it
- [x] Contact form funzionante (Web3Forms)
- [x] Google Analytics (G-J9R2C0TPW7) + Search Console
- [x] SEO ottimizzato (IT/EN multilingua)
- [x] Sezione Pricing trasparente
- [x] Sezione "Cosa Facciamo" (servizi senza prezzi)

**KPI Fase 1**:
- [ ] 3 clienti consulenza entro Q1 2026
- [ ] €15K revenue da consulenza
- [ ] Feedback per feature SDK

---

### FASE 2: SDK Python (Q2 2026)
**Status: ✅ COMPLETATO - LIVE SU PYPI**

**Obiettivo**: Prodotto scalabile vendibile come licenza

#### 2.1 SDK Pubblicato ✅

```bash
pip install adaptive-k
```

**PyPI**: https://pypi.org/project/adaptive-k/

#### 2.2 Struttura SDK (Implementata)

```
sdk/
├── adaptive_k/
│   ├── __init__.py        # ✅ Package exports
│   ├── router.py          # ✅ AdaptiveKRouter core
│   ├── calibration.py     # ✅ Auto-calibrazione soglie
│   └── cli.py             # ✅ Command line interface
├── tests/
│   └── test_router.py     # ✅ Test suite pytest
├── pyproject.toml         # ✅ Package config PyPI
├── LICENSE                 # ✅ Apache 2.0
└── README.md              # ✅ Documentazione
```

#### 2.3 API Implementata

```python
from adaptive_k import AdaptiveKRouter

# Preset per modelli noti
router = AdaptiveKRouter.from_pretrained("mixtral-8x7b")

# Routing con metriche
indices, weights, metrics = router.route(logits, return_metrics=True)
print(f"Savings: {metrics.compute_savings:.1%}")

# Statistiche cumulative
print(router.stats)
# {'tokens_processed': 1234567, 'average_savings': 0.472}
```

#### 2.4 CLI Implementata

```bash
adaptive-k calibrate --model mixtral-8x7b --dataset wikitext-2
adaptive-k benchmark --model mixtral-8x7b --compare baseline
adaptive-k export --format tensorrt --output config.json
```

#### 2.5 Modello Business SDK (Open Core)

| Tier | Prezzo | Features |
|------|--------|----------|
| **Community** | **FREE** (PyPI) | Core routing, tutti modelli |
| **Consulenza** | €2,500+ | Calibrazione custom, integrazione |
| **Enterprise** | €5,000+/anno | SLA, supporto dedicato |

**KPI Fase 2**:
- [x] SDK v0.1.1 pubblicato su PyPI
- [ ] 100+ downloads primo mese
- [ ] 5 richieste consulenza da utenti SDK
- [ ] Documentazione esempi avanzati

---

### FASE 3: SaaS Platform (Q4 2026)
**Status: 🔮 FUTURO**

**Obiettivo**: Revenue ricorrente, scala infinita

#### 3.1 Architettura SaaS

```
┌─────────────────────────────────────────────────────────────┐
│                    ADAPTIVE-K CLOUD                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Web App │───▶│ API Gateway │───▶│ Optimization Engine │  │
│  │ (Next)  │    │ (FastAPI)   │    │ (Python + CUDA)     │  │
│  └─────────┘    └─────────────┘    └─────────────────────┘  │
│       │              │                      │               │
│       ▼              ▼                      ▼               │
│  ┌─────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │ Auth    │    │ Usage       │    │ Model Registry      │  │
│  │ (Clerk) │    │ Tracking    │    │ (S3 + PostgreSQL)   │  │
│  └─────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

#### 3.2 Features Dashboard

- **Real-time metrics**: Tokens, savings, latency
- **Model management**: Upload, calibrate, deploy
- **Cost calculator**: Proiezione risparmi
- **API keys**: Gestione accessi
- **Team collaboration**: Multi-user workspace
- **Billing**: Stripe integration

#### 3.3 Pricing SaaS

| Tier | Prezzo | Tokens/mese | Features |
|------|--------|-------------|----------|
| **Free** | €0 | 1M | 1 modello, basic metrics |
| **Starter** | €49/mese | 10M | 3 modelli, API access |
| **Pro** | €199/mese | 100M | Unlimited, priority |
| **Enterprise** | Custom | Unlimited | On-prem, SLA, dedicated |

**KPI Fase 3**:
- [ ] 100 utenti free tier
- [ ] 20 clienti paganti
- [ ] €5K MRR (Monthly Recurring Revenue)
- [ ] 99.9% uptime

---

### FASE 4: Partnership & Licensing (2027+)
**Status: 🎯 STRATEGICO**

**Potenziali partner**:
- **NVIDIA**: Già PR su TensorRT-LLM, espandere collaborazione
- **AWS**: SageMaker integration
- **Azure**: Azure ML plugin
- **Google Cloud**: Vertex AI integration
- **Hugging Face**: Official integration

**Modello licensing**:
- Royalty su token processati
- White-label per cloud provider
- OEM per hardware AI

---

## 💰 Proiezioni Finanziarie

### Anno 1 (2026)

| Fonte | Q1 | Q2 | Q3 | Q4 | Totale |
|-------|-----|-----|-----|-----|--------|
| Consulenza | €15K | €20K | €15K | €10K | €60K |
| SDK Licenze | - | €5K | €10K | €15K | €30K |
| SaaS | - | - | - | €2K | €2K |
| **Totale** | €15K | €25K | €25K | €27K | **€92K** |

### Anno 2 (2027)

| Fonte | Target |
|-------|--------|
| Consulenza | €50K (ridotto, focus prodotto) |
| SDK Licenze | €100K |
| SaaS MRR | €10K/mese = €120K |
| Partnership | €50K |
| **Totale** | **€320K** |

---

## 🛠️ Tech Stack

### SDK
- **Language**: Python 3.9+
- **Dependencies**: PyTorch, Transformers, numpy
- **Testing**: pytest, tox
- **CI/CD**: GitHub Actions
- **Distribution**: PyPI, conda-forge

### SaaS
- **Frontend**: Next.js 14, TailwindCSS, shadcn/ui
- **Backend**: FastAPI, PostgreSQL, Redis
- **Auth**: Clerk
- **Payments**: Stripe
- **Hosting**: Vercel (frontend), Railway/Fly.io (backend)
- **Monitoring**: Sentry, Posthog

### Infrastructure
- **GPU**: Lambda Labs, RunPod (on-demand)
- **Storage**: S3/R2
- **CDN**: Cloudflare

---

## 📋 Best Practices

### Development
1. **Semantic versioning**: MAJOR.MINOR.PATCH
2. **Changelog**: Mantenere CHANGELOG.md aggiornato
3. **Testing**: Coverage minimo 80%
4. **Documentation**: Docstrings + Sphinx/MkDocs

### Business
1. **Customer first**: Ogni feature nasce da feedback reale
2. **Open core**: Core open source, premium features a pagamento
3. **Content marketing**: Blog post tecnici, LinkedIn, Twitter
4. **Community**: Discord/Slack per utenti SDK

### Legal
1. **Licenza SDK**: Apache 2.0 (community) + Commercial (pro)
2. **Privacy**: GDPR compliant
3. **Terms of Service**: Per SaaS
4. **P.IVA**: IT18354371009 (già attiva)

---

## 📅 Timeline Dettagliata

```
2026
────────────────────────────────────────────────────────────
GEN  FEB  MAR  APR  MAG  GIU  LUG  AGO  SET  OTT  NOV  DIC
 │    │    │    │    │    │    │    │    │    │    │    │
 ├────┴────┤    ├────┴────┴────┤    ├────┴────┴────┴────┤
 │ FASE 1  │    │   FASE 2     │    │      FASE 3       │
 │Consulenza│    │    SDK       │    │       SaaS        │
 └─────────┘    └──────────────┘    └───────────────────┘
      │              │                      │
      ▼              ▼                      ▼
   3 clienti     SDK v1.0 PyPI         Dashboard MVP
   €15K rev      10 licenze Pro        100 free users
```

---

## 🎯 Next Actions (Questa Settimana)

- [ ] Finalizzare landing page multilingua
- [ ] Creare struttura cartella `sdk/`
- [ ] Setup repository PyPI (test.pypi.org)
- [ ] Scrivere README SDK
- [ ] Primo blog post LinkedIn

---

## 📞 Contatti

**Vertex Data**  
Email: amministrazione@vertexdata.it  
Website: https://vertexdata.it  
Adaptive-K: https://adaptive-k.vertexdata.it  
GitHub: https://github.com/Gabrobals/sbm-efficient

---

*Documento confidenziale - Vertex Data © 2026*
