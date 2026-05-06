# 🩺 Maternal Health Risk — Web Interface Plan
### Full Blueprint: Expert System UI + Qwen2.5-1.5B Chatbot

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Tech Stack](#3-tech-stack)
4. [Design System](#4-design-system)
5. [Pages & Layout](#5-pages--layout)
6. [Backend API (Flask)](#6-backend-api-flask)
7. [LLM Integration — Qwen2.5-1.5B-Instruct](#7-llm-integration--qwen25-15b-instruct)
8. [Knowledge Base & System Prompt](#8-knowledge-base--system-prompt)
9. [Chat Architecture](#9-chat-architecture)
10. [Frontend Implementation](#10-frontend-implementation)
11. [Data Flow — End to End](#11-data-flow--end-to-end)
12. [Step-by-Step Build Order](#12-step-by-step-build-order)
13. [requirements.txt](#13-requirementstxt)

---

## 1. Project Overview

The website wraps two distinct systems into a single, coherent interface:

| System | What it does | How the user interacts |
|--------|-------------|------------------------|
| **Hybrid Expert System** | Takes 6 patient vitals → runs NN₁ → NN₂ → CF rules → verdict (`low / mid / high risk`) | A clean input form on the **Diagnosis** page |
| **Qwen2.5-1.5B Chatbot** | Answers follow-up questions about the diagnosis, maternal health, and the system's reasoning | A floating chat panel available on every page |

The two systems **share state**: when a diagnosis is completed, the chatbot automatically knows the patient's vitals and the full chain-of-evidence, so the user can ask *"why high risk?"* and get a grounded answer.

---

## 2. Folder Structure

```
FinalProject/
├── data/
├── models/                        ← the 5 saved .joblib files
├── papers/
├── LLM/
│   └── Qwen2.5-1.5B-Instruct/    ← local model weights
│
├── interface/                     ← NEW: everything web-related lives here
│   ├── app.py                     ← Flask application entry point
│   ├── predictor.py               ← loads models, exposes diagnose()
│   ├── llm_engine.py              ← loads Qwen, exposes generate()
│   ├── knowledge_base.py          ← static clinical knowledge + prompt builder
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── main.css
│   │   ├── js/
│   │   │   ├── diagnosis.js       ← form logic + result rendering
│   │   │   └── chat.js            ← chat panel logic
│   │   └── icons/                 ← SVG icons (heart, pulse, etc.)
│   │
│   └── templates/
│       ├── base.html              ← shared nav + chat panel shell
│       ├── index.html             ← landing / home page
│       └── diagnosis.html         ← input form + results panel
│
├── final_project_explained.ipynb
├── final_project.ipynb
├── DOCUMENTATION.md
└── requirements.txt
```

---

## 3. Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| **Backend framework** | Flask 3.x | Lightweight; no ORM needed; easy `joblib` integration |
| **ML inference** | `joblib` + `scikit-learn` | Already used in the notebooks |
| **LLM inference** | `transformers` + `torch` | Standard HuggingFace pipeline for Qwen |
| **Templating** | Jinja2 (built into Flask) | Simple, no build step |
| **CSS** | Plain CSS with custom properties | Zero framework overhead; full design control |
| **JS** | Vanilla ES6 (`fetch`, `async/await`) | No npm, no bundler — runs straight in the browser |
| **Streaming** | Server-Sent Events (SSE) | Streams LLM tokens to the chat in real time |

> **No React, no Vue, no webpack.** The goal is a lightweight system that runs on the same machine as the Qwen model without any build pipeline.

---

## 4. Design System

### 4.1 Color Palette

```css
:root {
  /* Backgrounds */
  --bg-base:        #F8F9FB;   /* off-white page background */
  --bg-surface:     #FFFFFF;   /* cards, panels */
  --bg-subtle:      #F0F4F8;   /* input fields, table rows */

  /* Brand */
  --brand-rose:     #E05C7A;   /* primary accent — warm, medical but not alarming */
  --brand-rose-dim: #F2C4CE;   /* hover states, progress fill */
  --brand-teal:     #2BBFB3;   /* secondary accent — trust, calm */

  /* Risk colours — used only in result badges */
  --risk-high:      #E05C7A;   /* rose */
  --risk-mid:       #F4A740;   /* amber */
  --risk-low:       #2BBFB3;   /* teal */

  /* Text */
  --text-primary:   #1A1D23;   /* headings */
  --text-secondary: #5A6172;   /* body, labels */
  --text-muted:     #9AA3B2;   /* placeholders, disabled */

  /* Borders */
  --border:         #E4E7EF;
  --border-focus:   #2BBFB3;

  /* Chat */
  --chat-user-bg:   #E8F5F4;   /* user bubble */
  --chat-bot-bg:    #F8F9FB;   /* bot bubble */
  --chat-border:    #E4E7EF;
}
```

### 4.2 Typography

```css
/* Google Fonts import (add to base.html <head>) */
/* @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=DM+Serif+Display&display=swap'); */

:root {
  --font-sans:  'Inter', system-ui, sans-serif;   /* UI, labels, body */
  --font-serif: 'DM Serif Display', Georgia, serif; /* hero headings only */

  --text-xs:   0.75rem;   /* 12px */
  --text-sm:   0.875rem;  /* 14px */
  --text-base: 1rem;      /* 16px */
  --text-lg:   1.125rem;  /* 18px */
  --text-xl:   1.25rem;   /* 20px */
  --text-2xl:  1.5rem;    /* 24px */
  --text-3xl:  1.875rem;  /* 30px */
}
```

### 4.3 Spacing & Shape

```css
:root {
  --radius-sm:  6px;
  --radius-md:  12px;
  --radius-lg:  20px;
  --radius-xl:  28px;
  --radius-full: 9999px;

  --shadow-sm: 0 1px 3px rgba(0,0,0,.06), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,.08);
  --shadow-lg: 0 8px 32px rgba(0,0,0,.10);
}
```

### 4.4 Visual Principles

- **Lots of white space.** Minimum `2rem` padding inside every card.
- **No decorative borders** on most components — separation is done with subtle box-shadows and background-color differences.
- **Risk badge** is the only bright colour on the results page; everything else stays neutral so the badge pops immediately.
- **Icons** are thin-line SVGs (Heroicons or Lucide style) — never filled/blobby.
- **Animations** are only on the chat message entry (fade-up, 200 ms) and the result card reveal (slide-down, 300 ms). Nowhere else.

---

## 5. Pages & Layout

### 5.1 Navigation Bar (`base.html`)

```
┌──────────────────────────────────────────────────────────────┐
│  🩺 MaternaAI          Home    Diagnose    About             │
│                                             [Chat  💬]       │
└──────────────────────────────────────────────────────────────┘
```

- Fixed top, `height: 60px`, white background, `box-shadow: var(--shadow-sm)`.
- Brand name in `var(--font-serif)`, nav links in `var(--font-sans)` 500 weight.
- **[Chat 💬]** button in the top-right always opens/closes the chat panel.

---

### 5.2 Home Page (`index.html`)

```
┌──────────────────────────────────────────────────────────────┐
│                          HERO                                │
│   Maternal Health Risk Assessment                            │
│   AI-powered. Evidence-based. Explainable.                   │
│                                                              │
│           [ → Start Diagnosis ]                              │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│  HOW IT WORKS — 3 icon cards in a row                        │
│  [🧠 Neural Network] [📋 Expert Rules] [💬 Ask the AI]       │
└──────────────────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────────────────┐
│  DATA — small stat strip                                     │
│  1,014 patients ·  3 risk classes  · 2-stage neural chain    │
└──────────────────────────────────────────────────────────────┘
```

---

### 5.3 Diagnosis Page (`diagnosis.html`)

This is the core page. It is split into **two columns** on desktop, one column on mobile.

```
┌────────────────────────┬───────────────────────────────────┐
│  INPUT FORM            │   RESULT PANEL (hidden until POST)│
│                        │                                   │
│  Age           [  35 ] │  ┌─ Verdict Badge ─────────────┐ │
│  Systolic BP   [ 145 ] │  │  🔴  LIKELY HIGH RISK         │ │
│  Diastolic BP  [  95 ] │  │  Certainty: 0.8821           │ │
│  Blood Sugar   [  13 ] │  └──────────────────────────────┘ │
│  Body Temp (°F)[ 100 ] │                                   │
│  Heart Rate    [  88 ] │  NN₂ Probabilities               │
│                        │  ████████ high  0.78             │
│  [ Run Diagnosis ]     │  ███      mid   0.15             │
│                        │  █        low   0.07             │
│                        │                                   │
│                        │  Chain of Evidence               │
│                        │  NN chain     cf=0.78  ──→ 0.78  │
│                        │  + high_bp    cf=0.70  ──→ 0.93  │
│                        │  + high_sugar cf=0.65  ──→ 0.98  │
│                        │                                   │
│                        │  [ 💬 Ask AI about this result ] │
└────────────────────────┴───────────────────────────────────┘
```

**Input field design:**
- Labels above each field, small, `var(--text-sm)`, `var(--text-secondary)`.
- Inputs with `border-radius: var(--radius-sm)`, `border: 1px solid var(--border)`.
- On focus, border turns `var(--border-focus)` with a subtle teal glow.
- Each field has a small hint below it: e.g. *"Normal: 90–140 mmHg"*.

**Result panel design:**
- Slides in from the right (or drops in below on mobile) after the API returns.
- Verdict badge takes the full width of the panel, coloured by `var(--risk-*)`.
- Progress bars for the three class probabilities (animated fill on reveal).
- Chain-of-evidence is a vertical timeline: each row shows the rule name, its CF, and the running total.

---

### 5.4 Chat Panel (global overlay)

```
┌──────────────────────────────────┐
│  MaternaAI Assistant          ✕ │
├──────────────────────────────────┤
│                                  │
│  [Bot] Hello! I'm ready to       │
│  discuss the diagnosis or any    │
│  maternal health questions.      │
│                                  │
│        [User] Why is BP risky?   │
│                                  │
│  [Bot] High systolic BP (≥140)   │
│  is linked to preeclampsia, a    │
│  condition that...               │
│  ▍ (streaming)                   │
│                                  │
├──────────────────────────────────┤
│  [Type a message...]    [Send →] │
└──────────────────────────────────┘
```

- **Fixed position** bottom-right, `width: 380px`, `height: 560px`.
- Slides up from the bottom-right corner when opened.
- Bot messages align left, user messages align right.
- Streaming is shown with a blinking cursor `▍` at the end of the current token.
- A subtle **"Diagnosis context loaded"** badge appears at the top of the chat when a patient result is active.

---

## 6. Backend API (Flask)

### `interface/app.py`

```python
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from predictor import diagnose_patient
from llm_engine import stream_response
from knowledge_base import build_system_prompt

app = Flask(__name__)

# ── Pages ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose')
def diagnose_page():
    return render_template('diagnosis.html')

# ── API: run the expert system ─────────────────────────────────────────
@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    """
    Receives JSON: {Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate}
    Returns JSON:  {verdict, final_cf, nn_cf, nn_probs, rules_fired, chain}
    """
    patient = request.get_json(force=True)
    result  = diagnose_patient(patient)
    return jsonify(result)


# ── API: chat with LLM ─────────────────────────────────────────────────
@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    Receives JSON: {
        messages: [{role, content}, ...],   # full conversation so far
        patient:  { vitals dict } | null,   # null if no diagnosis done yet
        result:   { diagnosis result } | null
    }
    Streams SSE tokens back.
    """
    body        = request.get_json(force=True)
    messages    = body.get('messages', [])
    patient     = body.get('patient')
    result      = body.get('result')
    system_prompt = build_system_prompt(patient, result)

    def token_stream():
        for token in stream_response(system_prompt, messages):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(token_stream()),
        content_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
```

---

### `interface/predictor.py`

```python
"""
Loads the 5 saved artefacts once at import time.
Exposes diagnose_patient(patient_dict) → result_dict.
"""
import numpy as np
import pandas as pd
import joblib

# ── Load artefacts (relative to FinalProject root) ─────────────────────
NN1    = joblib.load('../models/nn1.joblib')
NN2    = joblib.load('../models/nn2.joblib')
SCALER = joblib.load('../models/scaler.joblib')
LE     = joblib.load('../models/label_encoder.joblib')
META   = joblib.load('../models/preprocess_meta.joblib')

FEATURE_ORDER  = META['feature_order']
SKEWED_COLS    = META['skewed_columns']
CF_RULES       = META['cf_rules']
HIGH_IDX       = META['high_idx']
THRESHOLDS     = META['thresholds']


def _combine_cf(cf1: float, cf2: float) -> float:
    return cf1 + cf2 * (1 - cf1)


def _check_rules(patient: dict) -> dict:
    return {
        'high_bp':    patient['SystolicBP'] >= 140 or patient['DiastolicBP'] >= 90,
        'high_sugar': patient['BS']         >= 11,
        'fever':      patient['BodyTemp']   >= 100,
        'fast_heart': patient['HeartRate']  >= 90,
        'older_mom':  patient['Age']        >= 35,
    }


def diagnose_patient(patient: dict) -> dict:
    # 1. Preprocess
    row = pd.DataFrame([patient])[FEATURE_ORDER].astype(float).copy()
    for col in SKEWED_COLS:
        row[col] = np.sign(row[col]) * np.log1p(row[col].abs())
    scaled = SCALER.transform(row)

    # 2. NN chain
    p1     = NN1.predict_proba(scaled)
    stage2 = np.hstack([scaled, p1])
    p2     = NN2.predict_proba(stage2)[0]
    nn_cf  = float(p2[HIGH_IDX])

    # 3. Expert system
    total_cf  = nn_cf
    chain     = [{'name': 'NN_chain P(high)', 'cf': round(nn_cf, 4),
                  'running': round(nn_cf, 4)}]
    rules_fired = []

    for rule_name, triggered in _check_rules(patient).items():
        if triggered:
            cf       = CF_RULES[rule_name]
            total_cf = _combine_cf(total_cf, cf)
            rules_fired.append(rule_name)
            chain.append({'name': f'rule: {rule_name}',
                          'cf': round(cf, 4),
                          'running': round(total_cf, 4)})

    # 4. Verdict
    if   total_cf >= THRESHOLDS['high']: verdict = 'Likely HIGH risk'
    elif total_cf >= THRESHOLDS['mid']:  verdict = 'Possibly MID risk'
    else:                                verdict = 'Likely LOW risk'

    return {
        'verdict':     verdict,
        'final_cf':    round(total_cf, 4),
        'nn_cf':       round(nn_cf, 4),
        'nn_probs':    {cls: round(float(p), 4)
                        for cls, p in zip(LE.classes_, p2)},
        'rules_fired': rules_fired,
        'chain':       chain,
    }
```

---

## 7. LLM Integration — Qwen2.5-1.5B-Instruct

### `interface/llm_engine.py`

```python
"""
Loads Qwen2.5-1.5B-Instruct from the local snapshots directory.
Exposes stream_response(system_prompt, messages) as a generator of string tokens.
"""
import threading
from queue import Queue
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
import torch

# ── Model path ────────────────────────────────────────────────────────
SNAPSHOT = (
    '../LLM/Qwen2.5-1.5B-Instruct/snapshots/'
    '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
)

# ── Load once at startup ──────────────────────────────────────────────
print("[LLM] Loading tokenizer…")
TOKENIZER = AutoTokenizer.from_pretrained(SNAPSHOT, trust_remote_code=True)

print("[LLM] Loading model…")
MODEL = AutoModelForCausalLM.from_pretrained(
    SNAPSHOT,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map='auto' if torch.cuda.is_available() else 'cpu',
    trust_remote_code=True,
)
MODEL.eval()
print("[LLM] Ready.")

# ── Generation config ─────────────────────────────────────────────────
GENERATION_KWARGS = {
    'max_new_tokens':  512,
    'temperature':     0.7,
    'top_p':           0.9,
    'repetition_penalty': 1.1,
    'do_sample':       True,
}


def stream_response(system_prompt: str, messages: list):
    """
    Yields string tokens one at a time (Server-Sent Events source).

    messages: list of {role: 'user'|'assistant', content: str}
    """
    # Build the chat template
    chat = [{'role': 'system', 'content': system_prompt}] + messages

    input_ids = TOKENIZER.apply_chat_template(
        chat,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors='pt',
    ).to(MODEL.device)

    streamer = TextIteratorStreamer(
        TOKENIZER,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        **GENERATION_KWARGS,
    )

    # Run generation in a background thread so we can yield tokens
    thread = threading.Thread(target=MODEL.generate, kwargs=gen_kwargs)
    thread.start()

    for token in streamer:
        yield token

    thread.join()
```

**Key decisions:**
- The model is loaded **once** when Flask starts — not per-request. This is essential; loading a 1.5B model takes ~5 seconds.
- `TextIteratorStreamer` runs generation in a background thread and exposes tokens via a Python iterator, which Flask's `stream_with_context` then pipes to the browser as SSE.
- `torch.float16` on GPU cuts VRAM usage roughly in half; `float32` on CPU still works but is slower.
- `max_new_tokens: 512` keeps responses focused and snappy.

---

## 8. Knowledge Base & System Prompt

### `interface/knowledge_base.py`

This module holds all the static clinical facts that control what the LLM can say, and assembles the `system_prompt` that is prepended to every conversation.

```python
"""
Two responsibilities:
  1. KNOWLEDGE BASE — a dictionary of clinical facts the LLM should always know.
  2. build_system_prompt(patient, result) — assembles the dynamic system prompt
     injecting the current patient + diagnosis context when available.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STATIC KNOWLEDGE BASE
# The LLM is instructed to draw only from these facts when making clinical
# statements. This prevents hallucination of statistics not grounded in the
# project's data.
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {

    "system_description": """
This system is a Hybrid Expert System for Maternal Health Risk Assessment.
It was trained on the UCI Maternal Health Risk dataset (1,014 patients from
rural Bangladesh, donated 2023). It classifies pregnancy risk into three levels:
low risk, mid risk, or high risk, based on 6 vital signs.
""",

    "pipeline": """
The pipeline has three stages:
1. NN₁ (Stage-1 MLP): reads the 6 scaled vitals, outputs P(low), P(mid), P(high).
2. NN₂ (Stage-2 MLP): reads NN₁'s probabilities CONCATENATED with the scaled
   vitals (9 inputs total), and produces a refined probability vector.
3. Expert System: takes NN₂'s P(high risk) as a seed certainty factor (CF),
   then fires up to 5 clinical rules. Each rule that triggers adds its CF using
   the combination formula: total_cf = total_cf + cf × (1 − total_cf).
   Final verdict is thresholded: ≥0.70 → HIGH, 0.40–0.69 → MID, <0.40 → LOW.
""",

    "features": {
        "Age":         "Mother's age in years. Risk increases above 35 (older_mom rule, CF=0.30).",
        "SystolicBP":  "Upper blood pressure (mmHg). ≥140 triggers high_bp rule (CF=0.70). Normal: 90–120.",
        "DiastolicBP": "Lower blood pressure (mmHg). ≥90 triggers high_bp rule (CF=0.70). Normal: 60–80.",
        "BS":          "Blood sugar (mmol/L). ≥11 triggers high_sugar rule (CF=0.65). Normal fasting: 3.9–5.5.",
        "BodyTemp":    "Body temperature (°F). ≥100 triggers fever rule (CF=0.40). Normal: 97–99.",
        "HeartRate":   "Heart rate (bpm). ≥90 triggers fast_heart (tachycardia) rule (CF=0.50). Normal: 60–100.",
    },

    "cf_rules": {
        "high_bp":    {"threshold": "SystolicBP ≥ 140 OR DiastolicBP ≥ 90", "cf": 0.70,
                       "clinical_meaning": "Hypertension in pregnancy is a major risk factor for preeclampsia and eclampsia."},
        "high_sugar": {"threshold": "BS ≥ 11 mmol/L", "cf": 0.65,
                       "clinical_meaning": "Elevated blood sugar in pregnancy is associated with gestational diabetes, which raises risk of complications for mother and baby."},
        "fever":      {"threshold": "BodyTemp ≥ 100°F", "cf": 0.40,
                       "clinical_meaning": "Fever during pregnancy can indicate infection (e.g. UTI, sepsis) that elevates maternal risk."},
        "fast_heart": {"threshold": "HeartRate ≥ 90 bpm", "cf": 0.50,
                       "clinical_meaning": "Tachycardia may signal anaemia, infection, or cardiovascular stress, all of which are risk factors during pregnancy."},
        "older_mom":  {"threshold": "Age ≥ 35 years", "cf": 0.30,
                       "clinical_meaning": "Advanced maternal age is associated with higher rates of gestational hypertension, diabetes, and chromosomal abnormalities."},
    },

    "verdict_interpretation": {
        "Likely HIGH risk":   "The patient's vitals suggest a high probability of serious pregnancy complications. Immediate medical review is strongly recommended.",
        "Possibly MID risk":  "The patient shows some concerning indicators. Regular monitoring and follow-up with a healthcare provider is advised.",
        "Likely LOW risk":    "The patient's vitals are mostly within normal ranges. Routine prenatal care is still important.",
    },

    "cf_formula": "total_cf = total_cf + cf × (1 − total_cf). This ensures no single piece of evidence can push certainty above 1, and each additional rule has a diminishing effect.",

    "dataset_source": "UCI Maternal Health Risk dataset — 1,014 de-identified patients from rural Bangladesh health centres, 2023. Features: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate. Target: low / mid / high risk.",

    "important_disclaimer": "This system is a student academic project. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.",
}


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_system_prompt(patient: dict | None, result: dict | None) -> str:
    """
    Assembles the full system prompt sent to Qwen before every inference call.

    If patient and result are provided (a diagnosis has been run), the prompt
    includes the patient's exact vitals and the full chain-of-evidence so the
    LLM can give grounded, specific answers.
    """

    base = f"""You are MaternaAI, a helpful medical assistant embedded in a Maternal Health Risk Expert System.

ABOUT THE SYSTEM:
{KNOWLEDGE_BASE['system_description']}

PIPELINE:
{KNOWLEDGE_BASE['pipeline']}

CLINICAL RULES IN THIS SYSTEM:
- high_bp (CF=0.70):    {KNOWLEDGE_BASE['cf_rules']['high_bp']['threshold']}
  Meaning: {KNOWLEDGE_BASE['cf_rules']['high_bp']['clinical_meaning']}
- high_sugar (CF=0.65): {KNOWLEDGE_BASE['cf_rules']['high_sugar']['threshold']}
  Meaning: {KNOWLEDGE_BASE['cf_rules']['high_sugar']['clinical_meaning']}
- fever (CF=0.40):       {KNOWLEDGE_BASE['cf_rules']['fever']['threshold']}
  Meaning: {KNOWLEDGE_BASE['cf_rules']['fever']['clinical_meaning']}
- fast_heart (CF=0.50): {KNOWLEDGE_BASE['cf_rules']['fast_heart']['threshold']}
  Meaning: {KNOWLEDGE_BASE['cf_rules']['fast_heart']['clinical_meaning']}
- older_mom (CF=0.30):  {KNOWLEDGE_BASE['cf_rules']['older_mom']['threshold']}
  Meaning: {KNOWLEDGE_BASE['cf_rules']['older_mom']['clinical_meaning']}

CERTAINTY FACTOR FORMULA:
{KNOWLEDGE_BASE['cf_formula']}

YOUR BEHAVIOUR RULES:
1. Be warm, clear, and non-alarming. Never be blunt about risk without explaining what it means.
2. When explaining the diagnosis, always ground your answer in the patient's ACTUAL vitals and which rules fired.
3. Never invent statistics not present in this prompt. If unsure, say so.
4. If asked about anything unrelated to maternal health or this system, politely redirect.
5. Keep answers concise — 3 to 6 sentences unless the user asks for more detail.
6. Always close with: remind the user this is an academic project and not medical advice.

IMPORTANT DISCLAIMER:
{KNOWLEDGE_BASE['important_disclaimer']}
"""

    # ── Inject live patient context if available ──────────────────────────
    if patient and result:
        rules_fired_str = ', '.join(result.get('rules_fired', [])) or 'none'
        chain_lines = '\n'.join(
            f"  {i+1}. {step['name']} → CF={step['cf']}, running total={step['running']}"
            for i, step in enumerate(result.get('chain', []))
        )
        probs = result.get('nn_probs', {})

        patient_context = f"""

CURRENT PATIENT CONTEXT (use this to answer follow-up questions):
  Age:          {patient.get('Age')} years
  SystolicBP:   {patient.get('SystolicBP')} mmHg
  DiastolicBP:  {patient.get('DiastolicBP')} mmHg
  Blood Sugar:  {patient.get('BS')} mmol/L
  Body Temp:    {patient.get('BodyTemp')} °F
  Heart Rate:   {patient.get('HeartRate')} bpm

DIAGNOSIS RESULT:
  Verdict:      {result.get('verdict')}
  Final CF:     {result.get('final_cf')} (scale 0–1, higher = more certain of high risk)
  NN₂ seed CF:  {result.get('nn_cf')}
  Rules fired:  {rules_fired_str}

NN₂ PROBABILITY BREAKDOWN:
  P(high risk) = {probs.get('high risk', 'n/a')}
  P(mid risk)  = {probs.get('mid risk', 'n/a')}
  P(low risk)  = {probs.get('low risk', 'n/a')}

CHAIN OF EVIDENCE:
{chain_lines}

INTERPRETATION:
  {KNOWLEDGE_BASE['verdict_interpretation'].get(result.get('verdict', ''), '')}
"""
        return base + patient_context

    # No diagnosis yet
    return base + "\n\nNo diagnosis has been run yet. Help the user understand what inputs they need to provide and what the system does.\n"
```

---

## 9. Chat Architecture

### How conversation state is managed

The browser owns the conversation history. On every user message:

```
Browser                                Flask /api/chat              Qwen
  │                                        │                          │
  │── POST { messages, patient, result } ──►│                          │
  │                                        │── build_system_prompt() ──►│
  │                                        │── stream_response() ──────►│
  │                                        │◄── token, token, token ────│
  │◄── SSE: data: token ───────────────────│                          │
  │◄── SSE: data: token ───────────────────│                          │
  │◄── SSE: data: [DONE] ──────────────────│                          │
  │                                        │                          │
  │  append assistant message to history   │                          │
```

The `messages` array sent to the backend always contains the **full conversation** (user + assistant turns). Flask reconstructs the full chat context for Qwen on every call. This is stateless — Flask stores nothing.

### Context injection: when the user clicks "Ask AI about this result"

When the user clicks the result panel button, `chat.js` does two things:
1. Sets `window.diagnosisContext = { patient, result }`.
2. Opens the chat panel and sends an initial **synthetic bot message**: *"I can see the diagnosis for this patient. What would you like to know?"*

From that point, every `POST /api/chat` includes the `patient` and `result` objects, so Qwen always has the full clinical context in its system prompt.

---

## 10. Frontend Implementation

### `static/js/diagnosis.js`

```javascript
// Key responsibilities:
// 1. Intercept form submit, POST JSON to /api/diagnose
// 2. Animate the result panel into view
// 3. Render verdict badge, probability bars, chain-of-evidence timeline
// 4. Wire "Ask AI" button to chat.js

const form    = document.getElementById('diagnosis-form');
const panel   = document.getElementById('result-panel');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const patient = {
    Age:         parseFloat(form.age.value),
    SystolicBP:  parseFloat(form.systolic.value),
    DiastolicBP: parseFloat(form.diastolic.value),
    BS:          parseFloat(form.bs.value),
    BodyTemp:    parseFloat(form.bodytemp.value),
    HeartRate:   parseFloat(form.heartrate.value),
  };

  // Show loading state
  panel.innerHTML = '<div class="spinner"></div>';
  panel.classList.add('visible');

  const res    = await fetch('/api/diagnose', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patient),
  });
  const result = await res.json();

  // Store for chat.js
  window.diagnosisContext = { patient, result };

  renderResult(result);
});


function renderResult(r) {
  const riskClass = r.verdict.includes('HIGH') ? 'high'
                  : r.verdict.includes('MID')  ? 'mid' : 'low';

  const probBars = Object.entries(r.nn_probs)
    .map(([label, p]) => `
      <div class="prob-row">
        <span class="prob-label">${label}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill ${label.split(' ')[0]}"
               style="--target-width: ${(p * 100).toFixed(1)}%"></div>
        </div>
        <span class="prob-value">${p.toFixed(4)}</span>
      </div>
    `).join('');

  const chainRows = r.chain.map((step, i) => `
    <div class="chain-row">
      <span class="chain-index">${i + 1}</span>
      <span class="chain-name">${step.name}</span>
      <span class="chain-cf">cf=${step.cf.toFixed(3)}</span>
      <span class="chain-running">→ ${step.running.toFixed(4)}</span>
    </div>
  `).join('');

  panel.innerHTML = `
    <div class="verdict-badge risk-${riskClass}">
      <span class="verdict-icon">${riskClass === 'high' ? '🔴' : riskClass === 'mid' ? '🟡' : '🟢'}</span>
      <span class="verdict-text">${r.verdict}</span>
      <span class="verdict-cf">Certainty Factor: ${r.final_cf}</span>
    </div>

    <section class="result-section">
      <h3>Neural Network Probabilities</h3>
      <div class="prob-bars">${probBars}</div>
    </section>

    <section class="result-section">
      <h3>Chain of Evidence</h3>
      <div class="chain-timeline">${chainRows}</div>
    </section>

    <button class="ask-ai-btn" onclick="openChatWithContext()">
      💬 Ask AI about this result
    </button>
  `;

  // Animate probability bars
  requestAnimationFrame(() => {
    document.querySelectorAll('.prob-bar-fill').forEach(bar => {
      bar.style.width = bar.style.getPropertyValue('--target-width');
    });
  });
}
```

---

### `static/js/chat.js`

```javascript
// Key responsibilities:
// 1. Toggle chat panel open/close
// 2. Render message history
// 3. POST to /api/chat with full history + optional diagnosis context
// 4. Stream SSE tokens into the current bot message bubble

let messages = [];  // [{role, content}]

async function sendMessage(userText) {
  messages.push({ role: 'user', content: userText });
  renderMessages();

  const ctx    = window.diagnosisContext || {};
  const res    = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages: messages,
      patient:  ctx.patient || null,
      result:   ctx.result  || null,
    }),
  });

  // Placeholder for the streaming bot message
  messages.push({ role: 'assistant', content: '' });
  const botIdx = messages.length - 1;

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value);
    // Parse SSE lines
    for (const line of chunk.split('\n')) {
      if (line.startsWith('data: ')) {
        const token = line.slice(6);
        if (token === '[DONE]') break;
        messages[botIdx].content += token;
        renderMessages();        // re-render on each token
      }
    }
  }
}

function openChatWithContext() {
  document.getElementById('chat-panel').classList.add('open');
  if (window.diagnosisContext && messages.length === 0) {
    // Synthetic opening bot message
    messages.push({
      role: 'assistant',
      content: `I can see the diagnosis for this patient — **${window.diagnosisContext.result.verdict}** (CF = ${window.diagnosisContext.result.final_cf}). What would you like to know?`
    });
    renderMessages();
  }
}
```

---

## 11. Data Flow — End to End

```
User fills form
      │
      ▼
diagnosis.js  ──── POST /api/diagnose ────► predictor.py
                                                 │
                  { vitals }                     │  preprocess → NN₁ → NN₂ → CF rules
                                                 ▼
                  { verdict, final_cf,       diagnose_patient()
                    nn_probs, chain, ... }        │
                                                 ▼
              ◄──── JSON response ───────── Flask returns result

              ┌─────────────────────────────────┐
              │  Result panel renders in browser │
              └─────────────────────────────────┘
                            │
              User clicks "Ask AI about this result"
                            │
              chat.js sets window.diagnosisContext
                            │
              User types a question
                            │
              chat.js ── POST /api/chat ────────► knowledge_base.py
                         { messages,              build_system_prompt(patient, result)
                           patient,                     │
                           result }                     ▼
                                               llm_engine.py
                                               Qwen.generate() in thread
                                                     │
                                               TextIteratorStreamer
                                                     │ tokens
              chat.js renders tokens ◄──── SSE stream ─────────────┘
              in real time
```

---

## 12. Step-by-Step Build Order

Follow this order to avoid dependency hell:

### Phase 1 — Backend core (no UI)

1. Copy `interface/` skeleton (empty files).
2. Write `predictor.py` — test it with `python -c "from predictor import diagnose_patient; print(diagnose_patient({...}))"`.
3. Write `knowledge_base.py` — test `build_system_prompt(None, None)` prints cleanly.
4. Write `llm_engine.py` — run `python llm_engine.py` standalone to confirm the model loads and generates text.
5. Write `app.py` with just the `/api/diagnose` route. Test with `curl` or Postman.

### Phase 2 — LLM streaming

6. Add `/api/chat` route to `app.py`.
7. Test SSE streaming with `curl -N http://localhost:5000/api/chat -d '...'`.

### Phase 3 — Templates & CSS

8. Write `base.html` (nav + chat panel shell + `<slot>`).
9. Write `main.css` with the design tokens from Section 4.
10. Write `index.html` (hero + how-it-works cards).
11. Write `diagnosis.html` (form + empty result panel).

### Phase 4 — JavaScript

12. Write `diagnosis.js` — wire form submit to `/api/diagnose`, render result.
13. Write `chat.js` — wire chat input to `/api/chat`, SSE streaming, context injection.
14. Test the full end-to-end flow.

### Phase 5 — Polish

15. Add input validation (numeric, range hints below each field).
16. Add mobile layout (single column, chat panel full-screen on mobile).
17. Add loading spinner states.
18. Add a `<noscript>` fallback message.

---

## 13. requirements.txt

Add these to (or merge with) the existing `requirements.txt`:

```txt
# Existing (ML)
numpy
pandas
scikit-learn
joblib
matplotlib
seaborn

# Web server
flask>=3.0

# LLM
transformers>=4.40
torch>=2.2
accelerate>=0.28
sentencepiece
```

Run the server with:
```bash
cd FinalProject/interface
python app.py
# → http://127.0.0.1:5000
```

> **Note on startup time:** The first request after starting Flask will be fast (model is pre-loaded), but Flask itself takes ~10–20 seconds to start because Qwen is loaded at import time. This is the correct trade-off — per-request loading would be unusable.

---

## Quick Reference — API Contract

### `POST /api/diagnose`

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
    { "name": "rule: high_bp",    "cf": 0.70,   "running": 0.9464 },
    { "name": "rule: high_sugar", "cf": 0.65,   "running": 0.9812 },
    { "name": "rule: fever",      "cf": 0.40,   "running": 0.9887 },
    { "name": "rule: fast_heart", "cf": 0.50,   "running": 0.9944 },
    { "name": "rule: older_mom",  "cf": 0.30,   "running": 0.9961 }
  ]
}
```

---

### `POST /api/chat`

**Request body:**
```json
{
  "messages": [
    { "role": "user", "content": "Why is this patient high risk?" }
  ],
  "patient": { "Age": 38, "SystolicBP": 145, ... },
  "result":  { "verdict": "Likely HIGH risk", "final_cf": 0.9872, ... }
}
```

**Response:** `text/event-stream`
```
data: Because
data:  this
data:  patient's
data:  systolic
data:  BP
data:  is
data:  145...
data: [DONE]
```

---

*End of plan. Total estimated implementation time: 6–10 hours following the phase order above.*
