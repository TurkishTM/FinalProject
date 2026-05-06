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
