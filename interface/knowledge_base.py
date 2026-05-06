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
