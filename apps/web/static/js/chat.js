const conversationsUl = document.getElementById('conversations');
const messagesDiv = document.getElementById('messages');
const chatForm = document.getElementById('chatForm');
const textInput = document.getElementById('textInput');
const micBtn = document.getElementById('micBtn');
const useRagCb = document.getElementById('useRag');
const convIdInput = document.getElementById('conversationId');

async function fetchConversations() {
    const res = await fetch('/api/conversations');
    const data = await res.json();
    conversationsUl.innerHTML = '';
    data.forEach(c => {
        const li = document.createElement('li');
        li.textContent = `${c.title} — ${new Date(c.created_at).toLocaleString()}`;
        li.style.cursor = 'pointer';
        li.onclick = async () => {
            convIdInput.value = c.id;
            await loadHistory(c.id);
        };
        conversationsUl.appendChild(li);
    });
}

async function loadHistory(conversationId) {
    const res = await fetch(`/api/history/${conversationId}`);
    const data = await res.json();
    messagesDiv.innerHTML = '';
    data.forEach(m => addMessage(m.role, m.content, m.audio_path));
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addMessage(role, text, audioUrl) {
    const b = document.createElement('div');
    b.className = `msg ${role}`;
    b.textContent = text || '';
    if (audioUrl) {
        const a = document.createElement('div');
        a.className = 'audio-bubble';
        a.innerHTML = `🔊 <a href="${audioUrl}" target="_blank">audio</a>`;
        b.appendChild(document.createElement('br'));
        b.appendChild(a);
    }
    messagesDiv.appendChild(b);
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = (textInput.value || '').trim();
    if (!text) return;
    addMessage('user', text);
    const conversation_id = parseInt(convIdInput.value, 10);
    textInput.value = '';
    const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ conversation_id, text, use_rag: useRagCb.checked })
    });
    const data = await res.json();
    addMessage('assistant', data.reply || '(no reply)');
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
});

// Simple hold-to-record -> upload
let mediaRecorder, chunks = [];
let recording = false;

micBtn.addEventListener('mousedown', async () => {
    if (!navigator.mediaDevices) return alert('No media devices API');
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    chunks = [];
    mediaRecorder.ondataavailable = e => chunks.push(e.data);
    mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        const fd = new FormData();
        fd.append('audio', blob, 'speech.webm');
        fd.append('conversation_id', parseInt(convIdInput.value, 10));
        fd.append('use_rag', useRagCb.checked ? 'true' : 'false');
        const res = await fetch('/api/say', { method: 'POST', body: fd });
        const data = await res.json();
        // show transcript as user message (+audio bubble)
        addMessage('user', data.transcript || '(no speech)', data.audio_url);
        // show assistant reply
        addMessage('assistant', data.reply || '(no reply)');
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    };
    mediaRecorder.start();
    recording = true;
    micBtn.textContent = '⏺️ Recording...';
});

['mouseup', 'mouseleave'].forEach(ev => {
    micBtn.addEventListener(ev, () => {
        if (recording && mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            recording = false;
            micBtn.textContent = '🎤';
        }
    });
});

(async function init() {
    await fetchConversations();
    await loadHistory(parseInt(convIdInput.value, 10));
})();
