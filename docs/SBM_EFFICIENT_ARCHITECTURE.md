# SBM-H — Superposed Bit Model in Hilbert Space

> Documento tecnico unico (teoria + architettura + implementazioni)

---

## 1. Obiettivo del modello

Lo **SBM-H (Superposed Bit Model in Hilbert Space)** è un modello di Machine Learning **quantum‑inspired**, ma completamente **classico**, che sostituisce il paradigma:

- vettori + pesi + ReLU

con:

- **stati normalizzati** (spazio di Hilbert)
- **operatori** (dinamica conservativa)
- **misure energetiche** (non‑linearità fisicamente coerente)

Il modello è progettato per migliorare:
- stabilità numerica
- gestione dell’ambiguità
- generalizzazione su dati rumorosi / continui

---

## 2. Fondamenti matematici

### 2.1 Stato

Uno stato informativo è un vettore normalizzato:

ψ ∈ ℝᵈ ,  ||ψ||₂ = 1

Non rappresenta un valore puntuale, ma una **configurazione di informazione**.

---

### 2.2 Encoding SBM (Superposed Bits)

Per ogni feature scalata xᵢ ∈ [0,1]:

αᵢ = cos(πxᵢ / 2)
βᵢ = sin(πxᵢ / 2)

Lo stato complessivo è:

ψ = normalize([α₁, β₁, α₂, β₂, …, αₙ, βₙ])

Dimensione dello stato:

d = 2n

---

### 2.3 Dinamica (Layer)

L’evoluzione dello stato è lineare e conservativa:

ψ' = U ψ

con:

UᵀU = I

Gli operatori U sono **ortogonali** (unitari nel caso complesso).

#### Parametrizzazione (Cayley Transform)

Si apprende una matrice libera M e si costruisce:

A = M − Mᵀ  (anti‑simmetrica)
U = (I − A)⁻¹ (I + A)

Questo garantisce UᵀU = I.

---

### 2.4 Residual & Normalization

Per stabilità:

ψ ← normalize(ψ + λ ψ_skip)

---

## 3. Non‑linearità tramite misura (cuore del modello)

Nel SBM‑H **non esistono ReLU o soglie**.

La non‑linearità nasce dalla **misura**:

ℓ = ⟨ψ | O | ψ⟩ = ψᵀ O ψ

O è un **operatore osservabile** (simmetrico).

---

## 4. Teste di misura (Observables)

### 4.1 Osservabili pieni

Per ogni classe k:

Oₖ = ½(Sₖ + Sₖᵀ)
ℓₖ = ψᵀ Oₖ ψ

Catturano **termini incrociati** ψᵢψⱼ (interferenza).

---

### 4.2 Osservabili low‑rank (consigliato)

Per ogni classe k:

Oₖ = Lₖ Lₖᵀ ,   Lₖ ∈ ℝᵈˣʳ

Logit:

ℓₖ = || Lₖᵀ ψ ||²

Vantaggi:
- meno parametri
- PSD garantito
- migliore generalizzazione

---

### 4.3 Mixture of Observables (multimodalità)

Per ogni classe k e componente m:

Oₖₘ = Lₖₘ Lₖₘᵀ

Energia:

eₖₘ(ψ) = || Lₖₘᵀ ψ ||²

Gating:

gₘ(ψ) = softmax(G(ψ))

Logit finale:

ℓₖ = Σₘ gₘ(ψ) · eₖₘ(ψ)

Interpretazione:
- il modello seleziona dinamicamente **quale osservabile** usare
- ideale per dati con **regimi multipli**

---

## 5. Regolarizzazioni

### 5.1 Osservabili

Penalità consigliata:

λ · ||L||²_F

Controlla la capacità senza distruggere la struttura.

---

### 5.2 Gating (opzionale)

Entropia del gating:

H(g) = − Σ gₘ log gₘ

- λ > 0 → miscela più morbida
- λ < 0 → selezione più netta

---

## 6. Architettura completa

1. Encoding SBM → ψ₀
2. Stack di blocchi ortogonali (Cayley + residual)
3. Testa di misura:
   - low‑rank
   - oppure mixture of low‑rank
4. Softmax finale sui logit

---

## 7. Confronto concettuale con ML classico

| ML Classico | SBM‑H |
|-----------|------|
| Vettori ℝⁿ | Stati normalizzati |
| Pesi scalari | Operatori |
| ReLU distruttive | Misura energetica |
| Feature engineering | Interferenza nativa |
| Overfitting comune | Regolarizzazione geometrica |

---

## 8. Quando SBM‑H è superiore

- dati rumorosi
- decisioni graduali / ambigue
- confini curvi
- multimodalità
- stabilità più importante della sola accuracy

Non è pensato per:
- OCR / vision brute force
- problemi puramente discreti

---

## 9. Implementazioni disponibili

Nel progetto sono state implementate:

- SBM‑H base (stati + operatori)
- Testa full observable
- Testa low‑rank observable
- Mixture of low‑rank observables
- Benchmark contro MLP

Tutto eseguibile su CPU/GPU standard.

---

## 10. Evoluzioni future rigorose

- Stati misti (density matrix ρ)
- Ottimizzazione Riemanniana (natural gradient)
- Osservabili tempo‑dipendenti
- Applicazioni a time‑series e decision systems

---

## 11. Posizionamento

SBM‑H **non sostituisce universalmente il ML classico**, ma:

> lo estende dove il paradigma vettoriale è concettualmente rigido.

È un modello **quantum‑inspired corretto**, matematicamente difendibile, implementabile oggi e pubblicabile.

---

**Fine documento**

