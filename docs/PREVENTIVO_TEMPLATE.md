# PREVENTIVO / PROPOSTA DI SERVIZI

---

## DATI FORNITORE

**Gabriel Ballerini**
Partita IVA: IT18354371009
ATECO: 62.01.00 - Produzione di software non connesso all'edizione
Email: gabriele.ballerini@gmail.com
LinkedIn: linkedin.com/in/gabriele-ballerini

---

## DATI CLIENTE

| Campo | Valore |
|-------|--------|
| Ragione Sociale | [NOME AZIENDA] |
| Partita IVA / CF | [P.IVA] |
| Indirizzo | [INDIRIZZO] |
| Referente | [NOME REFERENTE] |
| Email | [EMAIL] |

---

## OGGETTO DEL PREVENTIVO

**Servizio:** [SELEZIONARE]
- [ ] Feasibility Assessment
- [ ] Implementation Package
- [ ] Expert Consulting
- [ ] Custom Solution

**Descrizione:** Implementazione della metodologia Adaptive-K per l'ottimizzazione dell'inferenza di modelli Mixture-of-Experts (MoE), con riduzione stimata dei costi computazionali del 30-50%.

---

## DETTAGLIO SERVIZI

### Opzione A: Feasibility Assessment
| Descrizione | Importo |
|-------------|---------|
| Analisi infrastruttura MoE esistente | € 800 |
| Profilazione entropy routing su dataset cliente | € 800 |
| Report savings stimati con raccomandazioni | € 600 |
| Roadmap implementazione | € 300 |
| **Totale (IVA esclusa)** | **€ 2.500** |

**Durata stimata:** 1-2 settimane
**Deliverables:**
- Report PDF con analisi tecnica
- Stima risparmio computazionale
- Roadmap implementazione prioritizzata

---

### Opzione B: Implementation Package
| Descrizione | Importo |
|-------------|---------|
| Feasibility Assessment (incluso) | € 2.500 |
| Sviluppo codice Adaptive-K custom | € 3.000 |
| Calibrazione threshold per dataset cliente | € 1.000 |
| Integrazione pipeline inferenza | € 1.500 |
| Benchmark e testing | € 1.000 |
| Documentazione tecnica | € 500 |
| **Totale (IVA esclusa)** | **€ 9.500** |

**Durata stimata:** 4-6 settimane
**Deliverables:**
- Codice production-ready
- Benchmark report con metriche
- Documentazione tecnica
- 30 giorni supporto post-implementazione

---

### Opzione C: Expert Consulting
| Descrizione | Importo |
|-------------|---------|
| Tariffa giornaliera | € 1.000/giorno |
| Tariffa mezza giornata (4h) | € 600 |
| Tariffa oraria | € 150/ora |

**Attività incluse:**
- Review architettura
- Code review
- Performance tuning
- Training team
- Supporto on-call

---

## CONDIZIONI ECONOMICHE

**Modalità di pagamento:**
- 40% all'accettazione del preventivo
- 30% a metà progetto (milestone)
- 30% alla consegna finale

**Metodi di pagamento:**
- Bonifico bancario
- PayPal Business

**Termini:** 15 giorni data fattura

---

## ESCLUSIONI

Non sono inclusi nel presente preventivo:
- Costi infrastruttura cloud (GPU, storage, etc.)
- Licenze software di terze parti
- Attività non espressamente indicate
- Modifiche scope dopo accettazione (quotate separatamente)

---

## VALIDITÀ

Il presente preventivo è valido per **30 giorni** dalla data di emissione.

---

## NOTE LEGALI

- I prezzi si intendono IVA esclusa (22%)
- Il lavoro svolto rimane proprietà intellettuale del Cliente
- È garantita la riservatezza su dati e codice del Cliente
- Per controversie è competente il Foro di [CITTÀ]

---

## FIRME

| Fornitore | Cliente |
|-----------|---------|
| Gabriele Balsamo | [NOME] |
| Data: ____________ | Data: ____________ |
| Firma: ____________ | Firma: ____________ |

---

**Preventivo N°:** [ANNO]-[NUMERO]
**Data emissione:** [DATA]

---

# ALLEGATO A: SPECIFICHE TECNICHE ADAPTIVE-K

## Metodologia

L'ottimizzazione Adaptive-K si basa sulla selezione dinamica del numero di esperti (K) nei modelli Mixture-of-Experts, guidata dall'entropia del routing.

**Principio:** Quando l'entropia H del router è bassa (distribuzione concentrata), il router è "sicuro" della scelta e un singolo esperto è sufficiente. Quando H è alta (distribuzione uniforme), più esperti sono necessari.

**Formula:**
```
H = -Σ p_i × log(p_i)

K = 1  se H < τ₁
K = 2  se τ₁ ≤ H < τ₂
K = 4  se H ≥ τ₂
```

## Risultati Validati

| Modello | Risparmio Compute | Accuracy Relativa |
|---------|-------------------|-------------------|
| Mixtral 8x7B | 31.0% | 99.8% |
| Qwen-MoE | 32.4% | 99.9% |
| OLMoE-1B-7B | 24.7% | 99.7% |

## Riferimenti

- Paper: [Adaptive-K Paper](https://adaptive-k.vercel.app/paper.html)
- Codice: [GitHub](https://github.com/Gabrobals/sbm-efficient)
- PR TensorRT-LLM: [#10672](https://github.com/NVIDIA/TensorRT-LLM/pull/10672)
