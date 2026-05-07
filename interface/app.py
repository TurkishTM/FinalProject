import json
import base64
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from predictor import diagnose_patient
from llm_engine import stream_response, generate_sync, LLM_AVAILABLE, llm_lock
from knowledge_base import build_system_prompt, KNOWLEDGE_BASE
from explainer import generate_explanation
import os

app = Flask(__name__)
# Enable CORS for localhost
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:*", "http://127.0.0.1:*"]}})

# ── Pages ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diagnose')
def diagnose_page():
    return render_template('diagnosis.html')

# ── API: run the expert system ─────────────────────────────────────────
# ── Clinical validation rules ─────────────────────────────────────────────
VALIDATION_RULES = {
    'Age': {
        'min': 15, 'max': 55,
        'error': 'Age must be between 15 and 55 for a maternal patient. '
                 'Age {val} is outside the valid maternal range.'
    },
    'SystolicBP': {
        'min': 70, 'max': 200,
        'error': 'Systolic BP {val} mmHg is outside the plausible range (70–200).'
    },
    'DiastolicBP': {
        'min': 40, 'max': 150,
        'error': 'Diastolic BP {val} mmHg is outside the plausible range (40–150).'
    },
    'BS': {
        'min': 2.0, 'max': 30.0,
        'error': 'Blood Sugar {val} mmol/L is outside the plausible range (2–30).'
    },
    'BodyTemp': {
        'min': 95.0, 'max': 106.0,
        'error': 'Body Temperature {val}°F is outside the plausible range (95–106).'
    },
    'HeartRate': {
        'min': 40, 'max': 200,
        'error': 'Heart Rate {val} bpm is outside the plausible range (40–200).'
    },
}

def validate_patient(patient: dict) -> list:
    """
    Returns a list of error strings.
    Empty list = patient is valid.
    """
    errors = []

    for field, rules in VALIDATION_RULES.items():
        val = patient.get(field)
        if val is None:
            errors.append(f"Missing required field: {field}")
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            errors.append(f"{field}: must be a number, got {type(val).__name__}")
            continue
        if val < rules['min'] or val > rules['max']:
            errors.append(rules['error'].format(val=val))

    # Cross-field check: DBP must be lower than SBP
    sbp = patient.get('SystolicBP')
    dbp = patient.get('DiastolicBP')
    if sbp is not None and dbp is not None:
        try:
            sbp, dbp = float(sbp), float(dbp)
            if dbp >= sbp:
                errors.append(
                    f"DiastolicBP ({dbp}) must be lower than SystolicBP ({sbp}). "
                    f"These values are physiologically impossible together."
                )
        except (ValueError, TypeError):
            pass

    return errors


@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    """
    Receives JSON: {Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate}
    Returns JSON:  {verdict, final_cf, nn_prob_high, nn_probs, rules_fired, chain}
    """
    try:
        patient = request.get_json(force=True)

        # Validate input
        errors = validate_patient(patient)
        if errors:
            return jsonify({
                'error':   'Input validation failed',
                'details': errors,
                'code':    'INVALID_INPUT'
            }), 400

        # Convert to float
        required_fields = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        for field in required_fields:
            patient[field] = float(patient[field])

        result  = diagnose_patient(patient)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── API: template-based explanation ────────────────────────────────────
@app.route('/api/explain', methods=['POST'])
def api_explain():
    """
    Receives JSON: {verdict, rules_fired, final_cf, nn_prob_high}
    Returns JSON: {explanation: markdown string}
    """
    try:
        data = request.get_json(force=True)
        explanation = generate_explanation(
            verdict      = data.get('verdict', ''),
            rules_fired  = data.get('rules_fired', []),
            final_cf     = data.get('final_cf', 0.0),
            nn_prob_high = data.get('nn_prob_high', 0.5),
        )
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
    if not LLM_AVAILABLE:
        return jsonify({'error': 'LLM model is currently unavailable or failed to load.'}), 503

    if not llm_lock.acquire(blocking=False):
        return jsonify({'error': 'The AI is currently processing another request. Please try again in a moment.'}), 429

    try:
        body        = request.get_json(force=True)
        messages    = body.get('messages', [])
        patient     = body.get('patient')
        result      = body.get('result')
        system_prompt = build_system_prompt(patient, result)

        def token_stream():
            try:
                for token in stream_response(system_prompt, messages):
                    # SSE format
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                llm_lock.release()

        return Response(
            stream_with_context(token_stream()),
            content_type='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',
                'Connection': 'keep-alive',
            }
        )
    except Exception as e:
        llm_lock.release()
        return jsonify({'error': str(e)}), 500

# ── API: Full Result Report ────────────────────────────────────────────
@app.route('/report', methods=['GET'])
def generate_report():
    data_b64 = request.args.get('data')
    if not data_b64:
        return "Missing data parameter", 400

    try:
        # Decode URL-safe base64 (frontend uses btoa with URL-safe replacements)
        # Restore standard base64 format
        data_b64_standard = data_b64.replace('-', '+').replace('_', '/')
        # Add padding if needed
        padding = 4 - len(data_b64_standard) % 4
        if padding != 4:
            data_b64_standard += '=' * padding

        # Decode base64
        json_str = base64.urlsafe_b64decode(data_b64_standard).decode('utf-8')
        data = json.loads(json_str)
        patient = data.get('patient', {})
        result = data.get('result', {})
    except Exception as e:
        return f"Invalid data format: {str(e)}", 400

    # Ensure we have data
    if not patient or not result:
        return "Patient or result data is empty", 400

    # Generate LLM summary
    prompt = "Write a 3-paragraph clinical summary of this patient's diagnosis result using the provided context. Be professional and clear."
    summary = "LLM unavailable."
    if LLM_AVAILABLE:
        system_prompt = build_system_prompt(patient, result)
        summary = generate_sync(system_prompt, [{'role': 'user', 'content': prompt}])

    return render_template(
        'report.html',
        patient=patient,
        result=result,
        summary=summary,
        kb=KNOWLEDGE_BASE
    )

if __name__ == '__main__':
    # Use environment variable for port or default to 5000
    port = int(os.environ.get('PORT', 5000))
    # Threaded=True is important for streaming
    app.run(debug=True, port=port, threaded=True, use_reloader=False)
