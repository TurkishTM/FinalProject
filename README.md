# 🩺 Maternal Health Risk — Hybrid Expert System

> An AI-powered web application that predicts pregnancy risk level using a **two-stage neural network chain** combined with a **clinical certainty-factor expert system**, with an integrated **LLM chatbot** for interactive explanations.

---

## What It Does

This system takes 6 patient vitals and produces a grounded, explainable risk diagnosis:

```
Patient Vitals (Age, BP, Blood Sugar, Temp, Heart Rate)
        │
        ▼
   NN₁ (Stage-1 MLP)  →  initial probability vector
        │
        ▼
   NN₂ (Stage-2 MLP)  →  refined probability vector
        │
        ▼
   Expert System (Certainty Factors)  →  fires 5 clinical rules
        │
        ▼
   Final Verdict: LOW / MID / HIGH risk  +  full chain of evidence
```

The verdict is accompanied by a full chain-of-evidence breakdown and an AI assistant (Qwen2.5-1.5B) that can answer follow-up questions grounded in the actual diagnosis.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | scikit-learn MLPClassifier (×2), joblib |
| Expert System | Certainty Factor rules (custom Python) |
| LLM | Qwen2.5-1.5B-Instruct (local, HuggingFace transformers) |
| Backend | Flask 3.x, Server-Sent Events (SSE) streaming |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Dataset | [UCI Maternal Health Risk](https://archive.ics.uci.edu/dataset/863/maternal+health+risk) (1,014 patients, Bangladesh, 2023) |

---

## Project Structure

```
FinalProject/
├── data/
│   └── Maternal Health Risk Data Set.csv   ← UCI dataset
│
├── models/                                 ← trained artefacts (all required)
│   ├── nn1.joblib                          ← Stage-1 MLP
│   ├── nn2.joblib                          ← Stage-2 MLP
│   ├── scaler.joblib                       ← fitted StandardScaler
│   ├── label_encoder.joblib                ← int ↔ risk-label mapping
│   └── preprocess_meta.joblib              ← skewed columns, CF rules, thresholds
│
├── interface/                              ← Flask web application
│   ├── app.py                              ← routes + SSE streaming
│   ├── predictor.py                        ← loads models, runs diagnose()
│   ├── llm_engine.py                       ← loads Qwen, streams tokens
│   ├── knowledge_base.py                   ← clinical facts + system prompt builder
│   ├── static/
│   │   ├── css/main.css
│   │   └── js/
│   │       ├── diagnosis.js
│   │       └── chat.js
│   └── templates/
│       ├── base.html
│       ├── index.html
│       ├── diagnosis.html
│       └── report.html
│
├── LLM/                                    ← NOT in repo (see setup below)
│   └── Qwen2.5-1.5B-Instruct/
│
├── final_project_explained.ipynb           ← full walkthrough with markdown
├── final_project.ipynb                     ← clean demo notebook
├── requirements.txt
└── DOCUMENTATION.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/maternal-health-risk-expert-system.git
cd maternal-health-risk-expert-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the LLM weights

The Qwen2.5-1.5B-Instruct weights (~3 GB) are not included in this repo. Download them with:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-1.5B-Instruct',
    local_dir='LLM/Qwen2.5-1.5B-Instruct'
)
"
```

> This requires `huggingface_hub` which is included in `requirements.txt`.  
> The download takes a few minutes depending on your connection.

### 4. Run the web interface

```bash
cd interface
python app.py
```

Open your browser at **http://127.0.0.1:5000**

> **Note:** Flask takes ~10–20 seconds to start because the Qwen model is loaded into memory at startup. This is intentional — loading per-request would be unusable.

---

## GPU vs CPU

The system runs on both GPU and CPU automatically:

| Hardware | Behaviour |
|----------|-----------|
| NVIDIA GPU (CUDA) | Qwen loads in `float16`, fast inference (~2–5s per response) |
| CPU only | Qwen loads in `float32`, slower inference (~20–60s per response) |

No configuration needed — `llm_engine.py` detects `torch.cuda.is_available()` automatically.

---

## API Reference

### `POST /api/diagnose`

Runs the full NN₁ → NN₂ → Expert System pipeline.

**Request body:**
```json
{
  "Age": 38,
  "SystolicBP": 145,
  "DiastolicBP": 95,
  "BS": 13,
  "BodyTemp": 100,
  "HeartRate": 92
}
```

**Response:**
```json
{
  "verdict": "Likely HIGH risk",
  "final_cf": 0.9872,
  "nn_cf": 0.8213,
  "nn_probs": { "high risk": 0.8213, "mid risk": 0.1124, "low risk": 0.0663 },
  "rules_fired": ["high_bp", "high_sugar", "fever", "fast_heart", "older_mom"],
  "chain": [
    { "name": "NN_chain P(high)", "cf": 0.8213, "running": 0.8213 },
    { "name": "rule: high_bp",    "cf": 0.70,   "running": 0.9464 }
  ]
}
```

### `POST /api/chat`

Streams an LLM response as Server-Sent Events. Accepts the full conversation history plus optional patient context.

**Request body:**
```json
{
  "messages": [{ "role": "user", "content": "Why is this patient high risk?" }],
  "patient": { "Age": 38, "SystolicBP": 145, "..." : "..." },
  "result":  { "verdict": "Likely HIGH risk", "..." : "..." }
}
```

**Response:** `text/event-stream` — tokens arrive as `data: <token>\n\n`, terminated by `data: [DONE]\n\n`.

### `GET /report?data=<base64>`

Returns a printable HTML report for a patient. Pass the patient + result as a base64-encoded JSON blob.

---

## Expert System — Clinical Rules

| Rule | Condition | Certainty Factor |
|------|-----------|:---:|
| `high_bp` | SystolicBP ≥ 140 **or** DiastolicBP ≥ 90 | 0.70 |
| `high_sugar` | Blood Sugar ≥ 11 mmol/L | 0.65 |
| `fast_heart` | Heart Rate ≥ 90 bpm | 0.50 |
| `fever` | Body Temp ≥ 100 °F | 0.40 |
| `older_mom` | Age ≥ 35 years | 0.30 |

Rules combine with: **`total_cf = total_cf + cf × (1 − total_cf)`**

| Final CF | Verdict |
|----------|---------|
| ≥ 0.70 | 🔴 Likely HIGH risk |
| 0.40 – 0.69 | 🟡 Possibly MID risk |
| < 0.40 | 🟢 Likely LOW risk |

---

## Training the Models

If you want to retrain the models from scratch instead of using the saved `.joblib` files:

```bash
jupyter notebook final_project.ipynb
# Cell → Run All
```

This will overwrite the files in `models/` with freshly trained versions.  
The explained version with full markdown commentary is in `final_project_explained.ipynb`.

---

## Reference Papers

1. **Togunwa et al. (2023)** — *Frontiers in AI* — ANN + Random Forest hybrid, 94.88% accuracy. [`PDF`](papers/paper1_Togunwa_2023_FrontiersAI.pdf)
2. **Saleem et al. (2024)** — *Scientific Reports* — Quad-Ensemble framework. [`PDF`](papers/paper2_Saleem_2024_NatureSR.pdf)
3. **Jamel et al. (2024)** — *PeerJ Computer Science* — MLP + ExtraTree + PCA, 98.25% accuracy.

---

## Disclaimer

This project is an academic coursework submission. It is **not** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## Course

**Expert Systems** — Final Practical Project  
Computer Science, 2025–2026