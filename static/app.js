let currentAbortController = null;

async function handleSendMessage() {
    const inputField = document.getElementById('user-input');
    const prompt = inputField.value.trim();
    if (!prompt) return;

    inputField.value = '';
    appendMessage('user', prompt);

    currentAbortController = new AbortController();
    
    document.getElementById('stop-btn').classList.remove('hidden');
    document.getElementById('send-btn').classList.add('hidden');

    const responseDiv = appendMessage('assistant', '');

    try {
        const response = await fetch('/api/chat/stream', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: document.getElementById('model-selector').value,
                prompt: prompt,
                web_search: document.getElementById('web-search-toggle').checked
            }),
            signal: currentAbortController.signal
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            fullText += decoder.decode(value, { stream: true });
            responseDiv.innerText = fullText;
            responseDiv.scrollTop = responseDiv.scrollHeight;
        }
    } catch (error) {
        if (error.name === 'AbortError') {
            responseDiv.innerText += "\n\n*[Generation stopped by user]*";
        } else {
            console.error("Stream error:", error);
        }
    } finally {
        resetStreamState();
    }
}

function stopGeneration() {
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
    }
    resetStreamState();
}

function resetStreamState() {
    document.getElementById('stop-btn').classList.add('hidden');
    document.getElementById('send-btn').classList.remove('hidden');
}

document.getElementById('send-btn').addEventListener('click', handleSendMessage);
document.getElementById('stop-btn').addEventListener('click', stopGeneration);

function appendMessage(role, text) {
    const container = document.getElementById('chat-messages');
    const msgDiv = document.createElement('div');
    msgDiv.className = `p-4 rounded-xl max-w-2xl text-sm ${role === 'user' ? 'bg-indigo-600/20 ml-auto border border-indigo-500/30' : 'bg-slate-900 border border-slate-800'}`;
    msgDiv.innerText = text;
    container.appendChild(msgDiv);
    container.scrollTop = container.scrollHeight;
    return msgDiv;
}