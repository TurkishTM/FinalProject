// 🩺 MaternaAI — Diagnosis Logic

const form = document.getElementById('diagnosis-form');
const panel = document.getElementById('result-panel');
const submitBtn = document.getElementById('submit-btn');

// Updated ranges to match backend validation
const RANGES = {
    Age: { min: 15, max: 55, label: 'Age (maternal range)' },
    SystolicBP: { min: 70, max: 200, label: 'Systolic BP' },
    DiastolicBP: { min: 40, max: 150, label: 'Diastolic BP' },
    BS: { min: 2.0, max: 30.0, label: 'Blood Sugar' },
    BodyTemp: { min: 95.0, max: 106.0, label: 'Body Temperature' },
    HeartRate: { min: 40, max: 200, label: 'Heart Rate' }
};

function clearErrors() {
    document.querySelectorAll('.error-msg').forEach(e => e.remove());
    document.querySelectorAll('.form-group input').forEach(i => i.style.borderColor = '');
}

function showError(inputName, msg) {
    const input = form.querySelector(`[name="${inputName}"]`);
    input.style.borderColor = 'var(--brand-rose)';
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-msg';
    errorDiv.style.color = 'var(--brand-rose)';
    errorDiv.style.fontSize = 'var(--text-xs)';
    errorDiv.style.marginTop = '4px';
    errorDiv.innerText = msg;
    input.parentElement.appendChild(errorDiv);
}

function validatePatient(patient) {
    clearErrors();
    let valid = true;

    // Check for NaN
    for (const [key, val] of Object.entries(patient)) {
        if (isNaN(val)) {
            showError(key, 'Must be a valid number');
            valid = false;
        }
    }

    // Check ranges
    for (const [key, val] of Object.entries(patient)) {
        if (!isNaN(val)) {
            if (val < RANGES[key].min || val > RANGES[key].max) {
                showError(key, `${RANGES[key].label}: must be between ${RANGES[key].min} and ${RANGES[key].max}`);
                valid = false;
            }
        }
    }

    // Cross-field check: DBP < SBP
    if (!isNaN(patient.SystolicBP) && !isNaN(patient.DiastolicBP)) {
        if (patient.DiastolicBP >= patient.SystolicBP) {
            showError('DiastolicBP', 'Diastolic BP must be lower than Systolic BP');
            valid = false;
        }
    }

    return valid;
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect data
    const formData = new FormData(form);
    const patient = {
        Age: parseInt(formData.get('Age')),
        SystolicBP: parseInt(formData.get('SystolicBP')),
        DiastolicBP: parseInt(formData.get('DiastolicBP')),
        BS: parseFloat(formData.get('BS')),
        BodyTemp: parseFloat(formData.get('BodyTemp')),
        HeartRate: parseInt(formData.get('HeartRate')),
    };

    if (!validatePatient(patient)) return;

    // UI Loading state
    submitBtn.disabled = true;
    submitBtn.innerText = 'Analyzing...';
    panel.innerHTML = '<div class="spinner"></div>';
    panel.classList.add('visible');

    try {
        const response = await fetch('/api/diagnose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(patient),
        });

        const result = await response.json();

        // Handle validation errors from backend
        if (!response.ok) {
            if (result.code === 'INVALID_INPUT' && result.details) {
                // Show validation errors
                panel.innerHTML = `
                    <div style="background: #fff5f5; border: 2px solid var(--brand-rose); padding: 1rem; border-radius: 8px;">
                        <h3 style="color: var(--brand-rose); margin-top: 0;">Input Validation Error</h3>
                        <ul style="margin: 0.5rem 0; padding-left: 1.5rem;">
                            ${result.details.map(d => `<li>${d}</li>`).join('')}
                        </ul>
                    </div>
                `;
            } else {
                throw new Error(result.error || 'API request failed');
            }
            submitBtn.disabled = false;
            submitBtn.innerText = 'Run Diagnosis';
            return;
        }

        // Store context globally for chat.js
        window.diagnosisContext = { patient, result };
        
        // Show context badge in chat header if it exists
        const badge = document.getElementById('context-badge');
        if (badge) badge.style.display = 'block';
        
        // Notify chat panel to clear history and show welcome for new context
        if (typeof resetChatForNewContext === 'function') {
            resetChatForNewContext();
        }

        renderResult(result);
    } catch (err) {
        panel.innerHTML = `<div class="result-section" style="color: var(--risk-high)">Error: ${err.message}</div>`;
    } finally {
        submitBtn.disabled = false;
        submitBtn.innerText = 'Run Diagnosis';
    }
});

const ALL_RULES = [
    // Positive rules (evidence FOR high risk)
    { key: 'high_bp',       label: 'High Blood Pressure',     threshold: 'SBP ≥ 140 or DBP ≥ 90',  sign: '+' },
    { key: 'borderline_bp', label: 'Stage 1 Hypertension',    threshold: 'SBP 130–139 or DBP 80–89', sign: '+' },
    { key: 'high_sugar',    label: 'High Blood Sugar',        threshold: 'BS ≥ 11 mmol/L',          sign: '+' },
    { key: 'fast_heart',    label: 'Tachycardia',             threshold: 'HR ≥ 90 bpm',             sign: '+' },
    { key: 'fever',         label: 'Fever',                   threshold: 'Temp ≥ 100.4°F',          sign: '+' },
    { key: 'older_mom',     label: 'Advanced Maternal Age',   threshold: 'Age ≥ 35',                sign: '+' },
    // Negative rules (evidence AGAINST high risk)
    { key: 'normal_bp',     label: 'Normal Blood Pressure',   threshold: 'SBP < 120 and DBP < 80',  sign: '−' },
    { key: 'normal_sugar',  label: 'Normal Blood Sugar',      threshold: 'BS < 6.1 mmol/L',         sign: '−' },
    { key: 'normal_heart',  label: 'Normal Heart Rate',       threshold: 'HR 60–80 bpm',            sign: '−' },
    { key: 'normal_temp',   label: 'Normal Temperature',      threshold: 'Temp 97–99°F',            sign: '−' },
];

function renderResult(r) {
    const v = r.verdict.toLowerCase();
    const riskClass = v.includes('high')       ? 'high'
                    : v.includes('mid')        ? 'mid'
                    : v.includes('uncertain')  ? 'uncertain'
                    : 'low';

    const probBars = Object.entries(r.nn_probs)
        .map(([label, p]) => {
            const cls = label.toLowerCase().includes('high') ? 'high'
                      : label.toLowerCase().includes('mid')  ? 'mid' : 'low';
            return `
                <div class="prob-row">
                    <span class="prob-label">${label}</span>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill ${cls}" style="--target-width: ${(p * 100).toFixed(1)}%"></div>
                    </div>
                    <span class="prob-value">${(p*100).toFixed(1)}%</span>
                </div>
            `;
        }).join('');

    let chainHTML = '';
    // Seed
    const seed = r.chain[0];
    chainHTML += `
        <div class="chain-row">
            <span class="chain-index" style="background: var(--brand-teal); color: white;">0</span>
            <span class="chain-name">${seed.name}</span>
            <span class="chain-cf">cf=${seed.cf.toFixed(3)}</span>
            <span class="chain-running" style="font-weight: bold;">→ ${seed.running.toFixed(4)}</span>
        </div>
    `;

    // Rules
    let chainIndex = 1;
    for (const ruleObj of ALL_RULES) {
        const ruleName = ruleObj.key || ruleObj;
        const firedStep = r.chain.find(step => step.name === `rule: ${ruleName}`);
        if (firedStep) {
            chainHTML += `
                <div class="chain-row" style="border-left: 2px solid var(--brand-teal); padding-left: 1rem; margin-left: 11px;">
                    <span class="chain-index" style="background: var(--brand-rose); color: white;">${chainIndex++}</span>
                    <span class="chain-name">${firedStep.name}</span>
                    <span class="chain-cf">cf=${firedStep.cf.toFixed(3)}</span>
                    <span class="chain-running" style="font-weight: bold;">→ ${firedStep.running.toFixed(4)}</span>
                </div>
            `;
        } else {
            chainHTML += `
                <div class="chain-row" style="border-left: 2px dashed var(--border); padding-left: 1rem; margin-left: 11px; opacity: 0.5;">
                    <span class="chain-index" style="background: var(--bg-subtle); color: var(--text-muted);">-</span>
                    <span class="chain-name" style="text-decoration: line-through;">rule: ${ruleName}</span>
                    <span class="chain-cf"></span>
                    <span class="chain-running">→ DID NOT FIRE</span>
                </div>
            `;
        }
    }

    panel.innerHTML = `
        <div class="verdict-badge risk-${riskClass}" style="width: 100%;">
            <span class="verdict-text">${r.verdict}</span>
            <span class="verdict-cf">Combined Certainty Factor: ${r.final_cf.toFixed(4)}</span>
        </div>

        <section class="result-section">
            <h3>Neural Network Chain</h3>
            <p style="font-size: var(--text-xs); color: var(--text-muted); margin-bottom: 1rem;">
                Baseline probabilities from Stage-2 MLP.
            </p>
            <div class="prob-bars">${probBars}</div>
        </section>

        <section class="result-section">
            <h3>Chain of Evidence</h3>
            <p style="font-size: var(--text-xs); color: var(--text-muted); margin-bottom: 1rem;">
                Combined Certainty Factor (CF) using expert clinical rules.
            </p>
            <div class="chain-timeline" style="display: flex; flex-direction: column; gap: 0.5rem;">${chainHTML}</div>
        </section>

        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
            <button class="ask-ai-btn" style="flex: 1;" onclick="openChatWithContext()">
                💬 Ask AI about this result
            </button>
            <button class="ask-ai-btn" style="flex: 1; border-color: var(--text-secondary); color: var(--text-secondary);" onclick="downloadReport()">
                📄 Download Full Report
            </button>
        </div>
    `;

    // Trigger probability bar animations
    setTimeout(() => {
        document.querySelectorAll('.prob-bar-fill').forEach(bar => {
            bar.style.width = bar.style.getPropertyValue('--target-width');
        });
    }, 50);
}

function downloadReport() {
    if (!window.diagnosisContext) return;
    const base64Data = btoa(unescape(encodeURIComponent(JSON.stringify(window.diagnosisContext))));
    // Replace URL-unsafe characters
    const urlSafe = base64Data.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    window.open(`/report?data=${urlSafe}`, '_blank');
}

window.openChatWithContext = function() {
    if (typeof toggleChat === 'function') {
        const chatPanel = document.getElementById('chat-panel');
        if (!chatPanel.classList.contains('open')) {
            toggleChat();
        }
        if (typeof injectSyntheticMessage === 'function') {
            injectSyntheticMessage();
        }
    }
};
