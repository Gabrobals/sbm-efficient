# Eduly Setup per Animazioni Adaptive-K

## Panoramica

[Eduly](https://github.com/eduly-ai/eduly) è un agente di coding Manim che trasforma qualsiasi contenuto in animazioni stile 3Blue1Brown.

## Requisiti

1. **Python 3.12+**
2. **uv package manager**: `pip install uv`
3. **LaTeX** (per rendering testo Manim):
   - Windows: [MiKTeX](https://miktex.org/download)
   - Mac: `brew install --cask mactex`
4. **Manim**: `pip install manim`
5. **Google AI API Key** (Gemini)

## Setup

```bash
# 1. Clona il repository (già fatto)
cd C:\Users\ottic\Desktop\eduly

# 2. Installa dipendenze
uv sync

# 3. Configura API key
# Modifica .env con la tua Gemini API key:
# GOOGLE_API_KEY=your_key_here
```

## Utilizzo per Paper Adaptive-K

### Script Python

```python
import pathlib
from google import genai
from eduly import EdulyBreakdownClient, EdulyAnimationClient
from langchain_google_genai import ChatGoogleGenerativeAI

# Inizializza client Gemini
gemini_client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

# Step 1: Breakdown del paper in topic atomici
breakdown_client = EdulyBreakdownClient(gemini_client)

breakdown, _ = breakdown_client.breakdown(
    file_path=pathlib.Path("../landing-page/public/paper.html"),
    model="gemini-2.5-flash-preview",
    thinking_level="high"
)

print(f"Document: {breakdown.document_title}")
for i, topic in enumerate(breakdown.topics):
    print(f"  Topic {i}: {topic.name}")

# Step 2: Genera storyboard per ogni topic
storyboards = {}
for topic in breakdown.topics:
    storyboard, _ = breakdown_client.storyboard(
        topic=topic,
        model="gemini-2.5-flash-preview",
        thinking_level="high"
    )
    storyboards[topic.name] = storyboard

# Step 3: Genera animazioni Manim
langchain_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro-preview",
    temperature=1.0,
)

animation_client = EdulyAnimationClient(
    langchain_model=langchain_model,
    agent_workspace_path="./adaptive_k_animations/"
)

# Anima topic specifici
key_topics = [
    "Mixture-of-Experts Architecture",
    "Shannon Entropy in Routing",
    "Adaptive-K Algorithm",
    "Multiplicative Savings"
]

for topic_name in key_topics:
    if topic_name in storyboards:
        result = animation_client.animate_single(
            breakdown=breakdown,
            storyboard=storyboards[topic_name],
            topic_index=0,
            max_iterations=5
        )
        if result.success:
            print(f"✅ Video: {result.video_path}")
        else:
            print(f"❌ Failed: {result.error_message}")
```

## Topic Suggeriti per Animazioni

1. **Entropia di Routing** - Visualizzare come H bassa/alta → K piccolo/grande
2. **Architettura MoE** - 8/16/128 esperti, routing sparso
3. **Algoritmo Adaptive-K** - Flow diagram animato
4. **Risparmi Moltiplicativi** - 0.69 × 0.687 × 0.65 = 0.31

## Output Atteso

Eduly genererà video MP4 1080×1920 (formato mobile) per ogni topic:
- `moe_architecture.mp4` (~5 min)
- `entropy_routing.mp4` (~5 min)
- `adaptive_k_algorithm.mp4` (~5 min)
- `multiplicative_savings.mp4` (~5 min)

## Integrazione con Landing Page

I video possono essere hostati su:
- YouTube (embedded nel paper)
- Vimeo
- Cloudflare Stream
- Self-hosted con HTML5 video

## Costi Stimati

- **Gemini 2.5 Flash**: ~$0.001/1K input tokens
- **Gemini 2.5 Pro**: ~$0.01/1K input tokens
- **Stima totale per 4 animazioni**: ~$2-5

## Troubleshooting

### Errore LaTeX
```bash
# Windows - assicurati MiKTeX sia nel PATH
# Verifica: pdflatex --version
```

### Errore Manim
```bash
pip install manim --upgrade
manim --version
```

### Errore API Key
```bash
# Verifica che .env sia configurato
cat .env
# Deve contenere: GOOGLE_API_KEY=ai...
```

## Link Utili

- [Documentazione Eduly](https://github.com/eduly-ai/eduly)
- [Manim Community](https://docs.manim.community/)
- [Google AI Studio](https://aistudio.google.com/) per API key
