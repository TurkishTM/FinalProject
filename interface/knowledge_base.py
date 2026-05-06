"""
Two responsibilities:
  1. KNOWLEDGE BASE — a dictionary of clinical facts the LLM should always know.
  2. build_system_prompt(patient, result) — assembles the dynamic system prompt
     injecting the current patient + diagnosis context when available.
"""

# ─────────────────────────────────────────────────────────────────────────────
# STATIC KNOWLEDGE BASE
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
   then fires up to 10 clinical rules (5 high-value danger rules + 5 low-value
   danger rules). Each rule that triggers adds its CF using the combination
   formula: total_cf = total_cf + cf × (1 − total_cf).
   Final verdict is thresholded: ≥0.70 → HIGH, 0.40–0.69 → MID, <0.40 → LOW.
""",

    "features": {
        "Age":         "Mother's age in years. Risk increases above 35 (older_mom rule, CF=0.30) or below 14 (very_young rule, CF=0.60).",
        "SystolicBP":  "Upper blood pressure (mmHg). ≥140 triggers high_bp (CF=0.70). ≤80 triggers low_bp (CF=0.50). Normal: 90–120.",
        "DiastolicBP": "Lower blood pressure (mmHg). ≥90 triggers high_bp (CF=0.70). ≤50 triggers low_bp (CF=0.50). Normal: 60–80.",
        "BS":          "Blood sugar (mmol/L). ≥11 triggers high_sugar (CF=0.65). ≤2.5 triggers low_sugar (CF=0.45). Normal fasting: 3.9–5.5.",
        "BodyTemp":    "Body temperature (°F). ≥100 triggers fever (CF=0.40). ≤96 triggers hypothermia (CF=0.40). Normal: 97–99.",
        "HeartRate":   "Heart rate (bpm). ≥90 triggers fast_heart (CF=0.50). ≤50 triggers slow_heart (CF=0.55). Normal: 60–100.",
    },

    "cf_rules": {
        # ── High-value danger rules ───────────────────────────────────────
        "high_bp": {
            "threshold": "SystolicBP ≥ 140 OR DiastolicBP ≥ 90",
            "cf": 0.70,
            "clinical_meaning": "Hypertension in pregnancy is a major risk factor for preeclampsia and eclampsia."
        },
        "high_sugar": {
            "threshold": "BS ≥ 11 mmol/L",
            "cf": 0.65,
            "clinical_meaning": "Elevated blood sugar in pregnancy is associated with gestational diabetes, which raises risk of complications for mother and baby."
        },
        "fast_heart": {
            "threshold": "HeartRate ≥ 90 bpm",
            "cf": 0.50,
            "clinical_meaning": "Tachycardia may signal anaemia, infection, or cardiovascular stress, all of which are risk factors during pregnancy."
        },
        "fever": {
            "threshold": "BodyTemp ≥ 100°F",
            "cf": 0.40,
            "clinical_meaning": "Fever during pregnancy can indicate infection (e.g. UTI, sepsis) that elevates maternal risk."
        },
        "older_mom": {
            "threshold": "Age ≥ 35 years",
            "cf": 0.30,
            "clinical_meaning": "Advanced maternal age is associated with higher rates of gestational hypertension, diabetes, and chromosomal abnormalities."
        },

        # ── Low-value danger rules ────────────────────────────────────────
        "very_young": {
            "threshold": "Age ≤ 14 years",
            "cf": 0.60,
            "clinical_meaning": "Adolescent pregnancy under age 14 carries significantly elevated obstetric risk including eclampsia, obstructed labour, and severe anaemia."
        },
        "slow_heart": {
            "threshold": "HeartRate ≤ 50 bpm",
            "cf": 0.55,
            "clinical_meaning": "Bradycardia in pregnancy can indicate cardiac dysfunction, medication overdose, or autonomic instability — all serious maternal risks."
        },
        "low_bp": {
            "threshold": "SystolicBP ≤ 80 OR DiastolicBP ≤ 50",
            "cf": 0.50,
            "clinical_meaning": "Hypotension during pregnancy may signal haemorrhage, septic shock, or severe dehydration, all of which threaten mother and fetus."
        },
        "low_sugar": {
            "threshold": "BS ≤ 2.5 mmol/L",
            "cf": 0.45,
            "clinical_meaning": "Severe hypoglycaemia starves the fetus of glucose and can cause maternal unconsciousness or seizures."
        },
        "hypothermia": {
            "threshold": "BodyTemp ≤ 96°F",
            "cf": 0.40,
            "clinical_meaning": "Subnormal body temperature may indicate sepsis, prolonged cold exposure, or severe metabolic disturbance during pregnancy."
        },
    },

    "verdict_interpretation": {
        "Likely HIGH risk":  "The patient's vitals suggest a high probability of serious pregnancy complications. Immediate medical review is strongly recommended.",
        "Possibly MID risk": "The patient shows some concerning indicators. Regular monitoring and follow-up with a healthcare provider is advised.",
        "Likely LOW risk":   "The patient's vitals are mostly within normal ranges. Routine prenatal care is still important.",
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

    r = KNOWLEDGE_BASE['cf_rules']

    base = f"""You are MaternaAI, a helpful medical assistant embedded in a Maternal Health Risk Expert System.

ABOUT THE SYSTEM:
{KNOWLEDGE_BASE['system_description']}

PIPELINE:
{KNOWLEDGE_BASE['pipeline']}

CLINICAL RULES IN THIS SYSTEM (10 rules total — 5 high-value, 5 low-value):

HIGH-VALUE DANGER RULES:
- high_bp    (CF=0.70): {r['high_bp']['threshold']}
  Meaning: {r['high_bp']['clinical_meaning']}
- high_sugar (CF=0.65): {r['high_sugar']['threshold']}
  Meaning: {r['high_sugar']['clinical_meaning']}
- fast_heart (CF=0.50): {r['fast_heart']['threshold']}
  Meaning: {r['fast_heart']['clinical_meaning']}
- fever      (CF=0.40): {r['fever']['threshold']}
  Meaning: {r['fever']['clinical_meaning']}
- older_mom  (CF=0.30): {r['older_mom']['threshold']}
  Meaning: {r['older_mom']['clinical_meaning']}

LOW-VALUE DANGER RULES:
- very_young  (CF=0.60): {r['very_young']['threshold']}
  Meaning: {r['very_young']['clinical_meaning']}
- slow_heart  (CF=0.55): {r['slow_heart']['threshold']}
  Meaning: {r['slow_heart']['clinical_meaning']}
- low_bp      (CF=0.50): {r['low_bp']['threshold']}
  Meaning: {r['low_bp']['clinical_meaning']}
- low_sugar   (CF=0.45): {r['low_sugar']['threshold']}
  Meaning: {r['low_sugar']['clinical_meaning']}
- hypothermia (CF=0.40): {r['hypothermia']['threshold']}
  Meaning: {r['hypothermia']['clinical_meaning']}

CERTAINTY FACTOR FORMULA:
{KNOWLEDGE_BASE['cf_formula']}

YOUR BEHAVIOUR RULES:
1. Be warm, clear, and non-alarming. Never be blunt about risk without explaining what it means.
2. When explaining the diagnosis, always ground your answer in the patient's ACTUAL vitals and which rules fired.
3. Never invent statistics not present in this prompt. If unsure, say so.
4. If asked about anything unrelated to maternal health or this system, politely redirect.
5. Keep answers concise — 3 to 6 sentences unless the user asks for more detail.
6. Always close with: remind the user this is an academic project and not medical advice.
7. If both a high-value AND low-value rule fired (e.g. fast_heart AND slow_heart),
   flag this as likely a data entry error and ask the user to double-check the inputs.

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