// 🤖 MaternaAI — Chat Logic

const chatPanel = document.getElementById('chat-panel');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatForm = document.getElementById('chat-form');

let messageHistory = []; // [{role, content}]

function toggleChat() {
    chatPanel.classList.toggle('open');
    if (chatPanel.classList.contains('open')) {
        chatInput.focus();
    }
}

function parseMarkdown(text) {
    let html = text;
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Inline code
    html = html.replace(/`(.*?)`/g, '<code style="background: rgba(0,0,0,0.05); padding: 2px 4px; border-radius: 3px;">$1</code>');
    // Bullets (simple heuristic: line starts with - or *)
    html = html.replace(/^(?:-|\*)\s+(.*)$/gm, '<ul><li style="margin-left: 20px;">$1</li></ul>');
    // Cleanup multiple adjacent ul tags
    html = html.replace(/<\/ul>\n<ul>/g, '\n');
    // Newlines to br
    html = html.replace(/\n/g, '<br>');
    return html;
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

window.resetChatForNewContext = function() {
    messageHistory = [];
    chatMessages.innerHTML = '';
};

async function handleChatSubmit(e) {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';
    addMessage('user', text);

    // Prepare request
    const payload = {
        messages: messageHistory,
        patient: window.diagnosisContext?.patient || null,
        result: window.diagnosisContext?.result || null
    };

    // Add empty bot message for streaming
    const botMsgDiv = document.createElement('div');
    botMsgDiv.className = 'message bot';
    botMsgDiv.innerHTML = '<span class="content"></span><span class="streaming-cursor"></span>';
    chatMessages.appendChild(botMsgDiv);
    scrollToBottom();
    
    const contentSpan = botMsgDiv.querySelector('.content');
    const cursor = botMsgDiv.querySelector('.streaming-cursor');

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullContent = '';
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep the last incomplete chunk in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data.trim() === '[DONE]') break;
                    
                    fullContent += data;
                    contentSpan.innerHTML = parseMarkdown(fullContent);
                    scrollToBottom();
                }
            }
        }
        
        // Finalize message
        messageHistory.push({ role: 'assistant', content: fullContent });
        cursor.remove();

    } catch (err) {
        cursor.remove();
        const errDiv = document.createElement('div');
        errDiv.style.color = 'var(--brand-rose)';
        errDiv.style.marginTop = '8px';
        errDiv.style.fontSize = 'var(--text-xs)';
        errDiv.textContent = `Error: ${err.message}`;
        botMsgDiv.appendChild(errDiv);
        scrollToBottom();
    }
}

function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.innerHTML = parseMarkdown(content);
    chatMessages.appendChild(div);
    scrollToBottom();

    if (role === 'user') {
        messageHistory.push({ role, content });
    }
    return div;
}

// Global helper called by diagnosis.js
window.injectSyntheticMessage = function() {
    if (window.diagnosisContext && messageHistory.length === 0) {
        const verdict = window.diagnosisContext.result.verdict;
        const cf = window.diagnosisContext.result.final_cf.toFixed(4);
        
        const content = `I can see the diagnosis for this patient — **${verdict}** (Combined CF = ${cf}). What would you like to know about this result?`;
        
        addMessage('assistant', content);
        messageHistory.push({ role: 'assistant', content: content });
    }
};
