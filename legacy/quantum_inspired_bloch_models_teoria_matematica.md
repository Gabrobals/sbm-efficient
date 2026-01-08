# ⚠️ DEPRECATED – ARCHIVED

**Questo documento è stato superato e sostituito integralmente da:**
- `docs/SBM_EFFICIENT_CONCEPT.md`
- `docs/SBM_EFFICIENT_ARCHITECTURE.md`
- `docs/SBM_EFFICIENT_BUSINESS_CASE.md`

**Il contenuto è mantenuto esclusivamente come audit trail storico.**
**Non utilizzare come riferimento attivo.**

---

# Modelli Quantum‑Inspired basati su Qubit (Bloch) per Machine Learning e AI Generativa

## 1. Premessa epistemologica

Questo documento descrive una **famiglia di modelli di Machine Learning e AI Generativa ispirati alla matematica del qubit**, senza ricorrere a calcolo quantistico reale o hardware quantistico.

L’obiettivo non è simulare un computer quantistico, ma **importare strutture matematiche, vincoli geometrici e regole probabilistiche** della meccanica quantistica all’interno di modelli neurali classici (eseguibili in Python/PyTorch).

Questa classe di modelli rientra nella categoria *quantum‑inspired machine learning*.

---

## 2. Il qubit come oggetto matematico

### 2.1 Stato puro di un qubit

Uno stato puro di qubit è un vettore complesso normalizzato in uno spazio di Hilbert bidimensionale:

\[
|\psi\rangle = \alpha |0\rangle + \beta |1\rangle,
\quad \alpha, \beta \in \mathbb{C},
\quad |\alpha|^2 + |\beta|^2 = 1
\]

Poiché una fase globale non è osservabile, ogni qubit puro può essere parametrizzato con **due gradi di libertà reali**.

### 2.2 Parametrizzazione di Bloch

Ogni stato puro di qubit può essere rappresentato tramite due angoli:

\[
\alpha = \cos(\tfrac{\theta}{2}), \quad
\beta = e^{i\phi}\sin(\tfrac{\theta}{2})
\]

con:
- \(\theta \in [0, \pi]\)
- \(\phi \in [ -\pi, \pi ]\)

---

## 3. Sfera di Bloch e osservabili

### 3.1 Coordinate sulla sfera di Bloch

Le aspettazioni degli operatori di Pauli forniscono una rappresentazione reale tridimensionale:

\[
x = \langle \psi | \sigma_x | \psi \rangle = \sin\theta \cos\phi
\]
\[
y = \langle \psi | \sigma_y | \psi \rangle = \sin\theta \sin\phi
\]
\[
z = \langle \psi | \sigma_z | \psi \rangle = \cos\theta
\]

Il vettore \((x,y,z)\) soddisfa:

\[
x^2 + y^2 + z^2 = 1
\]

quindi vive sulla **sfera unitaria**.

---

## 4. Bloch Embedding per Machine Learning

### 4.1 Definizione

Dato un token o una feature discreta \(t\), definiamo un embedding composto da \(K\) qubit indipendenti.

Per ogni qubit \(k\):

\[
(\theta_{t,k}, \phi_{t,k}) \rightarrow (x_{t,k}, y_{t,k}, z_{t,k})
\]

L’embedding finale è:

\[
E(t) = [x_{t,1}, y_{t,1}, z_{t,1}, \dots, x_{t,K}, y_{t,K}, z_{t,K}] \in \mathbb{R}^{3K}
\]

### 4.2 Proprietà

- **Norma controllata** (vincolo geometrico)
- **Spazio compatto** → regolarizzazione implicita
- **Separazione naturale tra ampiezza (θ) e fase (φ)**

---

## 5. Fase come variabile latente informativa

Nel contesto ML:

- \(\theta\) governa l’ampiezza (quanto un’informazione è presente)
- \(\phi\) governa la fase (come l’informazione interagisce con altre)

La fase introduce:
- interferenza costruttiva/distruttiva
- capacità di rappresentare ambiguità
- segnali periodici e ciclici

L’ablation \(\phi = 0\) permette di testare empiricamente il contributo informativo della fase.

---

## 6. Modelli generativi e Born Rule

### 6.1 Regola di Born

In meccanica quantistica la probabilità non è parametrizzata direttamente, ma deriva dal modulo quadro dell’ampiezza:

\[
p(x) = |\psi(x)|^2
\]

### 6.2 Born‑Inspired Generative Model

Un modello generativo può essere definito come:

\[
\psi_\theta(x) \in \mathbb{C}
\]
\[
p_\theta(x) = \frac{|\psi_\theta(x)|^2}{Z},
\quad Z = \sum_x |\psi_\theta(x)|^2
\]

Questa struttura consente **interferenza tra percorsi generativi**, impossibile nei mixture model classici.

---

## 7. Interferenza e somma di ampiezze

Se un output \(x\) può essere generato da più percorsi latenti \(k\):

\[
\psi(x) = \sum_k a_k e^{i\phi_k}
\]

la probabilità risultante è:

\[
p(x) = |\psi(x)|^2 = \sum_k a_k^2 + \sum_{k \neq j} a_k a_j \cos(\phi_k - \phi_j)
\]

Il secondo termine è **interferenza**, assente nei modelli probabilistici classici.

---

## 8. Attention e rappresentazioni complesse

### 8.1 Attention complessa

Query, Key e Value possono essere estesi a numeri complessi:

\[
Q, K, V \in \mathbb{C}^d
\]

Score tipico:

\[
\text{score}(i,j) = \Re(Q_i \cdot K_j^*)
\]

Oppure forme più generali che includono dipendenza esplicita dalla fase.

### 8.2 Interpretazione

- la magnitudine misura l’allineamento
- la fase modula l’interazione

---

## 9. Rumore e canali quantistici (analogia)

### 9.1 Depolarizing channel

Un canale di depolarizzazione aumenta l’entropia dello stato:

\[
\rho \rightarrow (1-p)\rho + p\frac{I}{2}
\]

### 9.2 Analogia con diffusion models

- forward process: aumento entropia
- reverse process: denoising condizionato

Questa analogia consente di definire **diffusion‑like models su stati Bloch**.

---

## 10. Ipotesi scientifiche testabili

1. I vincoli di Bloch migliorano la generalizzazione rispetto a embedding reali liberi
2. La fase \(\phi\) contiene informazione utile
3. L’interferenza migliora la modellazione di distribuzioni multimodali
4. La geometria compatta riduce instabilità di training

---

## 11. Criteri di validazione empirica

- confronto a parità di parametri
- ablation sistematiche
- multi‑seed evaluation
- metriche standard (accuracy, NLL, stability)

---

## 12. Posizionamento concettuale

Questi modelli:
- **non sono quantum computing**
- **non richiedono hardware quantistico**
- **sono compatibili con PyTorch / GPU classiche**

Sono modelli **geometrici e probabilistici avanzati**, ispirati alla struttura matematica della meccanica quantistica.

---

## 13. Prossime estensioni

- Phase‑Attention Transformer
- Born‑Diffusion Models
- Interpretabilità su Bloch space
- Applicazioni a NLP, segnali, dati finanziari

---

## 14. Conclusione

Il qubit, visto come oggetto matematico e geometrico, fornisce:
- vincoli utili
- nuove variabili latenti (fase)
- una semantica probabilistica alternativa

che possono essere **sfruttati in modo rigoroso** nel Machine Learning moderno.

