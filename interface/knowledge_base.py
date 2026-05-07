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
3. Expert System: converts NN₂'s P(high risk) to a normalized seed CF via
   nn_prob_to_cf() (maps P=0→CF=-0.50, P=0.5→CF=0, P=1→CF=+0.50), then fires
   up to 10 clinical rules (6 positive + 4 negative). Each rule that triggers
   combines its CF using the MYCIN formula (handles both positive and negative CFs).
   Final verdict is thresholded on the -1 to +1 scale:
   ≥0.50 → HIGH, 0.10–0.49 → MID, -0.10–0.09 → UNCERTAIN, <-0.10 → LOW.
""",

    "features": {
        "Age":         "Mother's age in years. Risk increases above 35 (older_mom rule, CF=+0.30). Normal: 18–34.",
        "SystolicBP":  "Upper blood pressure (mmHg). ≥140 triggers high_bp (CF=+0.70); 130–139 triggers borderline_bp (CF=+0.25); <120 triggers normal_bp (CF=−0.40).",
        "DiastolicBP": "Lower blood pressure (mmHg). ≥90 triggers high_bp (CF=+0.70); 80–89 triggers borderline_bp (CF=+0.25); <80 triggers normal_bp (CF=−0.40).",
        "BS":          "Blood sugar (mmol/L). ≥11 triggers high_sugar (CF=+0.65); <6.1 triggers normal_sugar (CF=−0.35). Normal fasting: 3.9–5.5.",
        "BodyTemp":    "Body temperature (°F). ≥100.4 triggers fever (CF=+0.40); 97–99 triggers normal_temp (CF=−0.20). Normal: 97–99.",
        "HeartRate":   "Heart rate (bpm). ≥90 triggers fast_heart (CF=+0.50); 60–80 triggers normal_heart (CF=−0.25). Normal: 60–100.",
    },

    "cf_rules": {
        # ── Positive rules (evidence FOR high risk) ──────────────────────
        "high_bp": {
            "threshold": "SystolicBP ≥ 140 OR DiastolicBP ≥ 90",
            "cf": 0.70,
            "clinical_meaning": "Hypertension in pregnancy is a major risk factor for preeclampsia and eclampsia."
        },
        "borderline_bp": {
            "threshold": "130 ≤ SystolicBP < 140 OR 80 ≤ DiastolicBP < 90",
            "cf": 0.25,
            "clinical_meaning": "Stage 1 hypertension in pregnancy can progress to more severe hypertension and warrants monitoring."
        },
        "high_sugar": {
            "threshold": "BS ≥ 11 mmol/L",
            "cf": 0.65,
            "clinical_meaning": "Elevated blood sugar in pregnancy is associated with gestational diabetes, which raises risk of complications for mother and baby."
        },
        "hypoglycemia": {
            "threshold": "BS < 3.9 mmol/L",
            "cf": 0.45,
            "clinical_meaning": "Hypoglycemia (low blood sugar) in pregnancy is dangerous and can cause fetal distress and maternal complications."
        },
        "borderline_sugar": {
            "threshold": "7.0 < BS < 11.0 mmol/L",
            "cf": 0.25,
            "clinical_meaning": "Blood sugar in the pre-diabetic range suggests impaired glucose tolerance, which increases risk of gestational diabetes."
        },
        "fast_heart": {
            "threshold": "HeartRate ≥ 90 bpm",
            "cf": 0.50,
            "clinical_meaning": "Tachycardia may signal anaemia, infection, or cardiovascular stress, all of which are risk factors during pregnancy."
        },
        "fever": {
            "threshold": "BodyTemp ≥ 100.4°F (38°C)",
            "cf": 0.40,
            "clinical_meaning": "Fever during pregnancy can indicate infection (e.g. UTI, sepsis) that elevates maternal risk."
        },
        "older_mom": {
            "threshold": "Age ≥ 35 years",
            "cf": 0.30,
            "clinical_meaning": "Advanced maternal age is associated with higher rates of gestational hypertension, diabetes, and chromosomal abnormalities."
        },

        # ── Negative rules (evidence AGAINST high risk) ──────────────────
        "normal_bp": {
            "threshold": "SystolicBP < 120 AND DiastolicBP < 80",
            "cf": -0.40,
            "clinical_meaning": "Normal blood pressure reduces the likelihood of hypertensive disorders of pregnancy."
        },
        "normal_sugar": {
            "threshold": "3.9 ≤ BS ≤ 7.0 mmol/L",
            "cf": -0.35,
            "clinical_meaning": "Normal blood sugar makes gestational diabetes unlikely, reducing risk."
        },
        "normal_heart": {
            "threshold": "60 ≤ HeartRate ≤ 80 bpm",
            "cf": -0.25,
            "clinical_meaning": "A normal resting heart rate suggests good cardiovascular health during pregnancy."
        },
        "normal_temp": {
            "threshold": "97°F ≤ BodyTemp ≤ 99°F",
            "cf": -0.20,
            "clinical_meaning": "Normal body temperature makes infection or sepsis less likely."
        },
    },

    "verdict_interpretation": {
        "🔴 Likely HIGH risk":      "The patient's vitals suggest a high probability of serious pregnancy complications. Immediate medical review is strongly recommended.",
        "🟡 Possibly MID risk":     "The patient shows some concerning indicators. Regular monitoring and follow-up with a healthcare provider is advised.",
        "⚪ Uncertain — borderline": "The evidence is mixed or inconclusive. Further assessment and monitoring are recommended.",
        "🟢 Likely LOW risk":       "The patient's vitals are mostly within normal ranges. Routine prenatal care is still important.",
    },

    "cf_formula": "MYCIN combination: if both CFs ≥ 0 → total_cf + new_cf × (1 − total_cf); if both ≤ 0 → total_cf + new_cf × (1 + total_cf); else → (total_cf + new_cf) / (1 − min(|total_cf|, |new_cf|)). The NN seed CF is mapped via nn_prob_to_cf(): P(high)=0→CF=−0.50, P(high)=0.5→CF=0, P(high)=1→CF=+0.50.",

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

CLINICAL RULES IN THIS SYSTEM (12 rules total — 8 positive, 4 negative):

POSITIVE RULES (evidence FOR high risk):
- high_bp       (CF=+0.70): {r['high_bp']['threshold']}
  Meaning: {r['high_bp']['clinical_meaning']}
- borderline_bp (CF=+0.25): {r['borderline_bp']['threshold']}
  Meaning: {r['borderline_bp']['clinical_meaning']}
- high_sugar    (CF=+0.65): {r['high_sugar']['threshold']}
  Meaning: {r['high_sugar']['clinical_meaning']}
- hypoglycemia  (CF=+0.45): {r['hypoglycemia']['threshold']}
  Meaning: {r['hypoglycemia']['clinical_meaning']}
- borderline_sugar (CF=+0.25): {r['borderline_sugar']['threshold']}
  Meaning: {r['borderline_sugar']['clinical_meaning']}
- fast_heart    (CF=+0.50): {r['fast_heart']['threshold']}
  Meaning: {r['fast_heart']['clinical_meaning']}
- fever         (CF=+0.40): {r['fever']['threshold']}
  Meaning: {r['fever']['clinical_meaning']}
- older_mom     (CF=+0.30): {r['older_mom']['threshold']}
  Meaning: {r['older_mom']['clinical_meaning']}

NEGATIVE RULES (evidence AGAINST high risk):
- normal_bp     (CF=−0.40): {r['normal_bp']['threshold']}
  Meaning: {r['normal_bp']['clinical_meaning']}
- normal_sugar  (CF=−0.35): {r['normal_sugar']['threshold']}
  Meaning: {r['normal_sugar']['clinical_meaning']}
- normal_heart  (CF=−0.25): {r['normal_heart']['threshold']}
  Meaning: {r['normal_heart']['clinical_meaning']}
- normal_temp   (CF=−0.20): {r['normal_temp']['threshold']}
  Meaning: {r['normal_temp']['clinical_meaning']}

CERTAINTY FACTOR FORMULA:
{KNOWLEDGE_BASE['cf_formula']}

VERDICT THRESHOLDS (CF scale: −1 to +1):
  CF ≥ 0.50       → 🔴 Likely HIGH risk
  0.10 ≤ CF < 0.50 → 🟡 Possibly MID risk
  −0.10 ≤ CF < 0.10 → ⚪ Uncertain — borderline
  CF < −0.10      → 🟢 Likely LOW risk

YOUR BEHAVIOUR RULES:
1. Keep answers BRIEF — 2 to 4 sentences maximum. Be concise and direct.
2. Be warm, clear, and non-alarming. Never be blunt about risk without explaining what it means.
3. When explaining the diagnosis, ground your answer in the patient's ACTUAL vitals and which rules fired.
4. Never invent statistics not present in this prompt. If unsure, say so.
5. If asked about anything unrelated to maternal health or this system, politely redirect.
6. Always close with: remind the user this is an academic project and not medical advice.
7. Negative rules (normal_bp, normal_sugar, etc.) reduce the CF, making a LOW risk verdict more likely.

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
  Final CF:     {result.get('final_cf')} (scale −1 to +1, positive = evidence for high risk)
  NN P(high):   {result.get('nn_prob_high')}
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