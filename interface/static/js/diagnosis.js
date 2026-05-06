// 🩺 MaternaAI — Diagnosis Logic

const form = document.getElementById('diagnosis-form');
const panel = document.getElementById('result-panel');
const submitBtn = document.getElementById('submit-btn');

const RANGES = {
    Age: { min: 10, max: 70 },
    SystolicBP: { min: 50, max: 250 },
    DiastolicBP: { min: 30, max: 180 },
    BS: { min: 1, max: 30 },
    BodyTemp: { min: 90, max: 115 },
    HeartRate: { min: 30, max: 200 }
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
    for (const [key, val] of Object.entries(patient)) {
        if (isNaN(val)) {
            showError(key, 'Must be a valid number');
            valid = false;
        } else if (val < RANGES[key].min || val > RANGES[key].max) {
            showError(key, `Value must be between ${RANGES[key].min} and ${RANGES[key].max}`);
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

        if (!response.ok) throw new Error('API request failed');

        const result = await response.json();

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
    // existing
    { key: 'high_bp',    label: 'High Blood Pressure',  threshold: 'SBP ≥ 140 or DBP ≥ 90' },
    { key: 'high_sugar', label: 'High Blood Sugar',     threshold: 'BS ≥ 11 mmol/L' },
    { key: 'fever',      label: 'Fever',                threshold: 'Temp ≥ 100°F' },
    { key: 'fast_heart', label: 'Tachycardia',          threshold: 'HR ≥ 90 bpm' },
    { key: 'older_mom',  label: 'Advanced Maternal Age',threshold: 'Age ≥ 35' },
    { key: 'slow_heart', label: 'Bradycardia',          threshold: 'HR ≤ 50 bpm' },
    { key: 'low_bp',     label: 'Low Blood Pressure',   threshold: 'SBP ≤ 80 or DBP ≤ 50' },
    { key: 'low_sugar',  label: 'Hypoglycaemia',        threshold: 'BS ≤ 2.5 mmol/L' },
    { key: 'hypothermia',label: 'Hypothermia',          threshold: 'Temp ≤ 96°F' },
    { key: 'very_young', label: 'Adolescent Pregnancy', threshold: 'Age ≤ 14' },
];

function renderResult(r) {
    const riskClass = r.verdict.toLowerCase().includes('high') ? 'high'
                    : r.verdict.toLowerCase().includes('mid')  ? 'mid' : 'low';

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
