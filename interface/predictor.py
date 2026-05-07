"""
Loads the 5 saved artefacts once at import time.
Exposes diagnose_patient(patient_dict) → result_dict.
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Base path relative to this file
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'

# ── Load artefacts ─────────────────────────────────────────────────────
NN1    = joblib.load(MODELS_DIR / 'nn1.joblib')
NN2    = joblib.load(MODELS_DIR / 'nn2.joblib')
SCALER = joblib.load(MODELS_DIR / 'scaler.joblib')
LE     = joblib.load(MODELS_DIR / 'label_encoder.joblib')
META   = joblib.load(MODELS_DIR / 'preprocess_meta.joblib')

FEATURE_ORDER  = META['feature_order']
SKEWED_COLS    = META['skewed_columns']
HIGH_IDX       = META['high_idx']

# ── CF helper functions ────────────────────────────────────────────────

def nn_prob_to_cf(p_high: float, max_nn_cf: float = 0.50) -> float:
    """
    Converts NN probability to an initial CF contribution.
    max_nn_cf = 0.50 means the NN alone can contribute at most 0.50,
    leaving the remaining 0.50 for clinical rules to fill.

    Mapping: P(high)=0.0 → CF=-0.50 (evidence against high risk)
             P(high)=0.5 → CF= 0.00 (neutral, no evidence either way)
             P(high)=1.0 → CF=+0.50 (moderate evidence for high risk)
    """
    return (p_high - 0.5) * max_nn_cf * 2   # linear mapping: [-0.50, +0.50]


def combine_cf(total_cf: float, new_cf: float) -> float:
    """
    MYCIN certainty factor combination for two pieces of evidence.
    Handles positive and negative CF values correctly.
    CFs range from -1.0 (certain false) to +1.0 (certain true).
    """
    if total_cf >= 0 and new_cf >= 0:
        return total_cf + new_cf * (1 - total_cf)
    elif total_cf <= 0 and new_cf <= 0:
        return total_cf + new_cf * (1 + total_cf)
    else:
        return (total_cf + new_cf) / (1 - min(abs(total_cf), abs(new_cf)))


# ── Clinical rules (positive + negative) ───────────────────────────────

CF_RULES = [
    # --- Positive rules (evidence FOR high risk) ---
    {
        "name":      "high_bp",
        "cf":        0.70,
        "condition": lambda p: p["SystolicBP"] >= 140 or p["DiastolicBP"] >= 90,
        "label":     "Stage 2 hypertension (≥140/90 mmHg)",
    },
    {
        "name":      "borderline_bp",
        "cf":        0.25,
        "condition": lambda p: (130 <= p["SystolicBP"] < 140) or (80 <= p["DiastolicBP"] < 90),
        "label":     "Stage 1 hypertension (130–139 / 80–89 mmHg)",
    },
    {
        "name":      "high_sugar",
        "cf":        0.65,
        "condition": lambda p: p["BS"] >= 11,
        "label":     "High blood sugar (≥11 mmol/L)",
    },
    {
        "name":      "hypoglycemia",
        "cf":        0.45,
        "condition": lambda p: p["BS"] < 3.9,
        "label":     "Hypoglycemia (BS < 3.9 mmol/L) — dangerous in pregnancy",
    },
    {
        "name":      "fast_heart",
        "cf":        0.50,
        "condition": lambda p: p["HeartRate"] >= 90,
        "label":     "Elevated heart rate (≥90 bpm)",
    },
    {
        "name":      "fever",
        "cf":        0.40,
        "condition": lambda p: p["BodyTemp"] >= 100.4,
        "label":     "Fever (≥100.4°F / 38°C)",
    },
    {
        "name":      "older_mom",
        "cf":        0.30,
        "condition": lambda p: p["Age"] >= 35,
        "label":     "Advanced maternal age (≥35 years)",
    },

    # --- Negative rules (evidence AGAINST high risk) ---
    {
        "name":      "normal_bp",
        "cf":        -0.40,
        "condition": lambda p: p["SystolicBP"] < 120 and p["DiastolicBP"] < 80,
        "label":     "Normal blood pressure (<120/80 mmHg)",
    },
    {
        "name":      "normal_sugar",
        "cf":        -0.35,
        "condition": lambda p: 3.9 <= p["BS"] <= 7.0,
        "label":     "Normal blood sugar (3.9–7.0 mmol/L)",
    },
    {
        "name":      "borderline_sugar",
        "cf":        0.25,
        "condition": lambda p: 7.0 < p["BS"] < 11.0,
        "label":     "Borderline blood sugar (7.0–11.0 mmol/L) — pre-diabetic range",
    },
    {
        "name":      "normal_heart",
        "cf":        -0.25,
        "condition": lambda p: 60 <= p["HeartRate"] <= 80,
        "label":     "Normal heart rate (60–80 bpm)",
    },
    {
        "name":      "normal_temp",
        "cf":        -0.20,
        "condition": lambda p: 97.0 <= p["BodyTemp"] <= 99.0,
        "label":     "Normal body temperature (97–99°F)",
    },
]


# ── Critical vital thresholds ─────────────────────────────────────────────
CRITICAL_THRESHOLDS = [
    ("HeartRate",    ">=", 130),   # Severe tachycardia
    ("SystolicBP",   ">=", 160),   # Severe hypertension
    ("DiastolicBP",  ">=", 110),   # Severe DBP hypertension
    ("BodyTemp",     ">=", 103.0), # High fever
    ("BS",           ">=", 20.0),  # Severe hyperglycemia
    ("BS",           "<",  2.5),   # Severe hypoglycemia
]

def is_critical_patient(patient: dict) -> tuple:
    """Returns (is_critical, list_of_triggered_thresholds)."""
    triggered = []
    for feat, op, threshold in CRITICAL_THRESHOLDS:
        val = patient.get(feat, 0)
        if op == ">=" and val >= threshold:
            triggered.append(f"{feat}={val} (critical threshold: ≥{threshold})")
        elif op == "<" and val < threshold:
            triggered.append(f"{feat}={val} (critical threshold: <{threshold})")
    return len(triggered) > 0, triggered


def determine_verdict(total_cf: float, is_critical: bool, critical_reasons: list) -> dict:
    """
    Determines final verdict. Critical override takes absolute priority.
    """
    # Hard override for critical vitals
    if is_critical:
        return {
            "verdict":          "🔴 HIGH risk — Critical vital signs detected",
            "verdict_source":   "critical_override",
            "critical_reasons": critical_reasons,
            "cf_ignored":       True,
            "cf_value":         round(total_cf, 4),
            "note":             (
                "Final verdict was determined by critical vital sign thresholds, "
                "not the certainty factor chain. The CF chain result is shown for "
                "reference only."
            )
        }

    # Normal CF-based verdict
    if total_cf >= 0.50:
        label = "🔴 Likely HIGH risk"
    elif total_cf >= 0.10:
        label = "🟡 Possibly MID risk"
    elif total_cf >= -0.10:
        label = "⚪ Uncertain — borderline"
    else:
        label = "🟢 Likely LOW risk"

    return {
        "verdict":        label,
        "verdict_source": "cf_chain",
        "cf_value":       round(total_cf, 4),
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
    p_high = float(p2[HIGH_IDX])

    # 3. Check for critical vitals (BEFORE CF chain)
    is_critical, critical_reasons = is_critical_patient(patient)

    # 4. Expert system — seed CF from NN probability (normalized)
    total_cf = nn_prob_to_cf(p_high)
    nn_cf_entry = {"name": "NN P(high)", "cf": round(total_cf, 4),
                   "running": round(total_cf, 4)}
    chain = [nn_cf_entry]
    rules_fired = []

    for rule in CF_RULES:
        if rule["condition"](patient):
            cf = rule["cf"]

            # Suppress negative rules when patient is critical
            if is_critical and cf < 0:
                rules_fired.append(rule["name"])
                chain.append({
                    "name": f"rule: {rule['name']}",
                    "cf": round(cf, 4),
                    "running": round(total_cf, 4),
                    "suppressed": True,
                    "suppression_reason": "Suppressed: critical vital signs override reassuring indicators"
                })
                continue  # skip combining — don't change total_cf

            total_cf = combine_cf(total_cf, cf)
            rules_fired.append(rule["name"])
            chain.append({"name": f"rule: {rule['name']}",
                          "cf": round(cf, 4),
                          "running": round(total_cf, 4)})

    # 5. Determine verdict (with critical override)
    verdict_info = determine_verdict(total_cf, is_critical, critical_reasons)

    return {
        "verdict":           verdict_info["verdict"],
        "verdict_source":    verdict_info.get("verdict_source", "cf_chain"),
        "final_cf":          verdict_info.get("cf_value", round(total_cf, 4)),
        "nn_prob_high":      round(p_high, 4),
        "nn_probs":          {cls: round(float(p), 4)
                             for cls, p in zip(LE.classes_, p2)},
        "rules_fired":       rules_fired,
        "chain":             chain,
        "is_critical":       is_critical,
        "critical_reasons":  critical_reasons,
        "cf_ignored":        verdict_info.get("cf_ignored", False),
        "override_note":     verdict_info.get("note"),
    }
