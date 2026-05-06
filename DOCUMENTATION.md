# Project Documentation — Maternal Health Risk Expert System

## What This Project Does

It predicts the **risk level during pregnancy** (low / mid / high) for a
patient using a hybrid system that combines:

- A **neural network** (MLPClassifier) trained on real patient data, and
- A **rule-based expert system** that uses certainty factors from the course
  material.

The two parts **complete each other**: the neural network gives a
data-driven probability, the rules add domain-knowledge evidence, and both
are combined into a single final diagnosis.

---

## Folder Structure

```
Final project/
├── final_project.ipynb              ← short student-style notebook
├── final_project_explained.ipynb    ← longer notebook with full markdown explanations
├── DOCUMENTATION.md                 ← this file
├── requirements.txt                 ← Python packages needed
├── data/
│   └── Maternal Health Risk Data Set.csv
├── models/
│   ├── mlp_model.joblib             ← trained neural network
│   ├── scaler.joblib                ← fitted StandardScaler
│   └── label_encoder.joblib         ← fitted LabelEncoder
└── papers/
    ├── paper1_Togunwa_2023_FrontiersAI.pdf
    ├── paper2_Saleem_2024_NatureSR.pdf
    └── references.md
```

---

## The Pipeline

```
   Patient input (Age, BP, BS, BodyTemp, HeartRate)
                   │
                   ▼
              StandardScaler           ← models/scaler.joblib
                   │
                   ▼
            MLPClassifier              ← models/mlp_model.joblib
                   │
        predict_proba → P(high risk)
                   │
                   ▼
        Expert System (CF rules)
                   │
        combine all CFs with formula
                   │
                   ▼
       Final diagnosis (low / mid / high)
```

---

## What Each Part Does

### `data/Maternal Health Risk Data Set.csv`
1014 rows from rural Bangladesh health centres, 6 numeric features:

| Feature | Unit |
|---|---|
| Age | years |
| SystolicBP | mmHg |
| DiastolicBP | mmHg |
| BS (blood sugar) | mmol/L |
| BodyTemp | °F |
| HeartRate | bpm |

Target column `RiskLevel`: `low risk`, `mid risk`, or `high risk`.

### The two notebooks
| File | Purpose |
|---|---|
| `final_project.ipynb` | Short, student-style. Just code + section titles. The one to actually demo. |
| `final_project_explained.ipynb` | Same pipeline but every step has a markdown cell explaining what is happening and why. Useful when reading the project later or showing to someone unfamiliar. |

Both notebooks produce the same model and the same diagnoses — they only
differ in how much explanation surrounds the code.

### The 3 files in `models/`

**They are NOT three different models.** They are the three pieces that
together make up one trained system. To make a prediction you need all three:

| File | What it is | Why it's saved |
|---|---|---|
| `mlp_model.joblib` | The trained neural network (MLPClassifier with the winning architecture) | The actual classifier — turns scaled features into a class prediction |
| `scaler.joblib` | A fitted `StandardScaler` | The neural network only works on scaled inputs. The same scaling applied during training (mean and std of the training set) must be re-applied to any new patient before the network can read them. |
| `label_encoder.joblib` | A fitted `LabelEncoder` | The network outputs integers 0 / 1 / 2. The encoder maps those back to `'high risk'` / `'low risk'` / `'mid risk'`. |

**Loading the saved system to predict on a new patient:**
```python
import joblib, pandas as pd

mlp     = joblib.load('models/mlp_model.joblib')
scaler  = joblib.load('models/scaler.joblib')
le      = joblib.load('models/label_encoder.joblib')

patient = {'Age': 38, 'SystolicBP': 145, 'DiastolicBP': 95,
           'BS': 13, 'BodyTemp': 98, 'HeartRate': 88}

row     = pd.DataFrame([patient])
scaled  = scaler.transform(row)
pred_id = mlp.predict(scaled)[0]
label   = le.inverse_transform([pred_id])[0]

print(label)   # → 'high risk'
```

If you save only `mlp_model.joblib` and try to predict on a new patient
without the scaler, the prediction will be wrong (the network will see
unscaled inputs that look very different from what it was trained on).

### `papers/`
The 3 reference research papers (2023–2024) that use this same UCI dataset.
PDFs of papers 1 and 2 are bundled; paper 3 is link-only because the
publisher's CDN blocks scripted downloads. Full citations live in
`papers/references.md`.

---

## Why We Made These Choices

### Why the Maternal Health Risk dataset?
- **Recent**: donated to UCI in 2023 → fits the 2023–2026 era requirement.
- **Easy to understand**: 6 features that anyone (not just a doctor) recognises:
  blood pressure, blood sugar, temperature, heart rate, age. No biomarker
  abbreviations to look up.
- **Right size**: 1014 rows — large enough for a real train/test split, small
  enough to train an MLP in seconds on a laptop.
- **Lesser-known**: not Pima Diabetes, not Heart Disease, not Wisconsin
  Breast Cancer (which everyone uses in tutorials).
- **Has rules a student can write**: clinical danger zones (e.g. BP ≥ 140/90)
  are common knowledge, so the expert-system rule base writes itself.

### Why MLPClassifier (and not a deeper neural network)?
- **It's part of scikit-learn** — no extra TensorFlow or PyTorch dependency.
- The dataset only has 6 features and 1014 rows; a deep network would just
  overfit. A small MLP (a few neurons in 1–2 hidden layers) is the right
  size.
- The `predict_proba` method gives the probability per class, which is what
  the expert system needs as its first piece of evidence.

### Why a hybrid (NN + rules) and not just one?
- A pure neural network is a **black box** — it gives a prediction but no
  explanation. Doctors cannot trust it without justification.
- Pure rules can only encode what we **already know** — they can't catch
  patterns hidden in the data.
- Combining them: the network finds patterns, the rules give explanations,
  the certainty-factor formula merges both kinds of evidence into one
  number that has a clinical meaning.

### Why certainty factors (CFs)?
- The course material uses CFs (see `Expert_System_Certainty_Factors.pdf`
  in the previous project's folder). The combination formula
  `total_cf = total_cf + cf × (1 − total_cf)` is what the project is
  expected to demonstrate.
- CFs are simple to implement (a single line of math) but powerful — they
  let several pieces of evidence accumulate without any one piece pushing
  confidence above 1.

### Why save 3 files instead of 1?
- The neural network alone cannot make predictions on new patients —
  it needs scaled inputs and label decoding. The scaler and label encoder
  hold the **state from training time** (the mean/std the scaler learned,
  the integer↔label mapping the encoder built). Without saving them, that
  state would be lost when the notebook closes.
- This is standard practice for any sklearn pipeline: you save the model
  and every preprocessor that touched the data.

---

## How to Run

```
pip install -r requirements.txt
jupyter notebook final_project.ipynb
```

Then click **Cell → Run All**. The notebook will:
1. Load the dataset from `data/`
2. Train the MLP (a few seconds)
3. Run the hybrid diagnosis on three example patients
4. Save the model files into `models/` (overwriting the existing ones)

---

## Reference Papers

All three are listed with full citations and URLs in
[`papers/references.md`](papers/references.md). Two are bundled as PDFs in
the `papers/` folder.
