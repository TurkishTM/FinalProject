import json
import base64
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from predictor import diagnose_patient
from llm_engine import stream_response, generate_sync, LLM_AVAILABLE, llm_lock
from knowledge_base import build_system_prompt, KNOWLEDGE_BASE
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

@app.route('/about')
def about_page():
    return render_template('index.html', scroll_to='about')

# ── API: run the expert system ─────────────────────────────────────────
@app.route('/api/diagnose', methods=['POST'])
def api_diagnose():
    """
    Receives JSON: {Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate}
    Returns JSON:  {verdict, final_cf, nn_cf, nn_probs, rules_fired, chain}
    """
    try:
        patient = request.get_json(force=True)
        # Ensure all fields are present and numeric
        required_fields = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
        
        for field in required_fields:
            if field not in patient:
                return jsonify({'error': f'Missing field: {field}'}), 400
            try:
                val = float(patient[field])
                patient[field] = val
            except ValueError:
                return jsonify({'error': f'Invalid number for field: {field}'}), 400
        
        result  = diagnose_patient(patient)
        return jsonify(result)
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
        # Decode base64 URL-safe JSON blob
        json_str = base64.urlsafe_b64decode(data_b64).decode('utf-8')
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
