// Real-time voice conversation functionality

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioContext = null;
let analyser = null;
let visualizerAnimationId = null;
let conversationHistory = [];

// Toggle voice conversation
async function toggleVoiceConversation() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

// Start recording
async function startRecording() {
    try {
        // Request microphone permission
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 44100
            }
        });

        // Create MediaRecorder
        const options = { mimeType: 'audio/webm;codecs=opus' };
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm';
        }

        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            await sendVoiceToBackend(audioBlob);

            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };

        // Start recording
        mediaRecorder.start();
        isRecording = true;

        // Update UI
        updateStatus('listening', 'Listening... Speak now');
        updateVoiceButton(true);

        // Start visualizer
        startVisualizer(stream);

        AppUtils.showToast('Recording started. Click again to stop.', 'success');

    } catch (error) {
        console.error('Error starting recording:', error);
        AppUtils.showToast('Microphone access denied or not available.', 'danger');
        updateStatus('ready', 'Ready to start conversation');
    }
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;

        // Update UI
        updateStatus('processing', 'Processing your question...');
        updateVoiceButton(false);

        // Stop visualizer
        stopVisualizer();
    }
}

// Send voice audio to backend
async function sendVoiceToBackend(audioBlob) {
    try {
        // Add placeholder for user turn
        addConversationTurn('user', 'Processing your question...', null);

        const formData = new FormData();
        formData.append('file', audioBlob, 'question.webm');

        updateStatus('processing', 'Querying documents and generating response...');

        const response = await fetch('/api/voice/conversation', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Voice conversation failed');
        }

        // Get audio response
        const audioResponseBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioResponseBlob);

        // Update conversation display
        updateLastUserTurn('[Spoken question]');
        addConversationTurn('assistant', '[AI is speaking...]', audioUrl);

        // Update status
        updateStatus('speaking', 'AI is speaking...');

        // Auto-play response
        const audio = new Audio(audioUrl);

        audio.onended = () => {
            updateStatus('ready', 'Ready for next question');
            AppUtils.showToast('Click microphone to ask another question', 'info');
        };

        audio.onerror = () => {
            updateStatus('ready', 'Ready to start conversation');
            AppUtils.showToast('Error playing audio response', 'danger');
        };

        await audio.play();

    } catch (error) {
        console.error('Error in voice conversation:', error);
        updateStatus('ready', 'Ready to start conversation');
        AppUtils.showToast(`Voice conversation failed: ${error.message}`, 'danger');

        // Remove placeholder
        removeLastConversationTurn();
    }
}

// Update status indicator
function updateStatus(state, text) {
    const statusIcon = document.getElementById('statusIcon');
    const statusText = document.getElementById('statusText');

    // Remove all state classes
    statusIcon.className = 'status-icon';

    // Add specific state class
    if (state === 'listening') {
        statusIcon.classList.add('listening');
        statusIcon.innerHTML = '<i class="fas fa-circle text-warning"></i>';
    } else if (state === 'processing') {
        statusIcon.classList.add('processing');
        statusIcon.innerHTML = '<i class="fas fa-circle text-info"></i>';
    } else if (state === 'speaking') {
        statusIcon.classList.add('speaking');
        statusIcon.innerHTML = '<i class="fas fa-circle text-success"></i>';
    } else {
        statusIcon.innerHTML = '<i class="fas fa-circle text-secondary"></i>';
    }

    statusText.textContent = text;
}

// Update voice button appearance
function updateVoiceButton(recording) {
    const button = document.getElementById('voiceButton');
    const icon = document.getElementById('voiceIcon');
    const status = document.getElementById('voiceStatus');
    const hint = document.getElementById('voiceHint');
    const pulseRing = document.getElementById('pulseRing');

    if (recording) {
        button.classList.add('recording');
        icon.className = 'fas fa-stop fa-3x';
        status.textContent = 'Recording...';
        status.className = 'badge bg-danger fs-6';
        hint.textContent = 'Click to stop recording and send';
        pulseRing.style.display = 'block';
    } else {
        button.classList.remove('recording');
        icon.className = 'fas fa-microphone fa-3x';
        status.textContent = 'Press to Start';
        status.className = 'badge bg-secondary fs-6';
        hint.textContent = 'Click the microphone to start speaking';
        pulseRing.style.display = 'none';
    }
}

// Start audio visualizer
function startVisualizer(stream) {
    const canvas = document.getElementById('waveformCanvas');
    const container = document.getElementById('waveformContainer');
    const canvasContext = canvas.getContext('2d');

    container.style.display = 'block';
    canvas.width = canvas.offsetWidth;
    canvas.height = 80;

    // Create audio context
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function draw() {
        visualizerAnimationId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        canvasContext.fillStyle = '#f8f9fa';
        canvasContext.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            barHeight = (dataArray[i] / 255) * canvas.height;

            const gradient = canvasContext.createLinearGradient(0, 0, 0, canvas.height);
            gradient.addColorStop(0, '#ffc107');
            gradient.addColorStop(1, '#ff9800');

            canvasContext.fillStyle = gradient;
            canvasContext.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

            x += barWidth + 1;
        }
    }

    draw();
}

// Stop visualizer
function stopVisualizer() {
    const container = document.getElementById('waveformContainer');
    container.style.display = 'none';

    if (visualizerAnimationId) {
        cancelAnimationFrame(visualizerAnimationId);
        visualizerAnimationId = null;
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
}

// Add conversation turn to display
function addConversationTurn(role, text, audioUrl) {
    const display = document.getElementById('conversationDisplay');

    // Remove initial prompt
    const initialPrompt = display.querySelector('.text-center.text-muted');
    if (initialPrompt) {
        initialPrompt.remove();
    }

    const turn = document.createElement('div');
    turn.className = `conversation-turn ${role}`;
    turn.dataset.role = role;

    const icon = role === 'user' ? 'fa-user' : 'fa-robot';
    const label = role === 'user' ? 'You' : 'AI Assistant';

    let audioHtml = '';
    if (audioUrl && role === 'assistant') {
        audioHtml = `
            <div class="turn-audio">
                <audio controls class="w-100">
                    <source src="${audioUrl}" type="audio/mpeg">
                    Your browser does not support audio playback.
                </audio>
            </div>
        `;
    }

    turn.innerHTML = `
        <div class="turn-header">
            <i class="fas ${icon}"></i>
            <span>${label}</span>
            <small class="text-muted ms-auto">${AppUtils.formatTimestamp()}</small>
        </div>
        <div class="turn-content">${text}</div>
        ${audioHtml}
    `;

    display.appendChild(turn);
    display.scrollTop = display.scrollHeight;

    // Add to history
    conversationHistory.push({
        role: role,
        text: text,
        audioUrl: audioUrl,
        timestamp: new Date().toISOString()
    });
}

// Update last user turn with actual text
function updateLastUserTurn(text) {
    const display = document.getElementById('conversationDisplay');
    const userTurns = display.querySelectorAll('.conversation-turn.user');
    if (userTurns.length > 0) {
        const lastTurn = userTurns[userTurns.length - 1];
        const content = lastTurn.querySelector('.turn-content');
        if (content) {
            content.textContent = text;
        }
    }
}

// Remove last conversation turn (for error handling)
function removeLastConversationTurn() {
    const display = document.getElementById('conversationDisplay');
    const turns = display.querySelectorAll('.conversation-turn');
    if (turns.length > 0) {
        turns[turns.length - 1].remove();
        conversationHistory.pop();
    }
}

// Clear conversation
function clearConversation() {
    if (!confirm('Clear the entire conversation?')) {
        return;
    }

    const display = document.getElementById('conversationDisplay');
    display.innerHTML = `
        <div class="text-center text-muted">
            <i class="fas fa-comment-dots fa-2x mb-2"></i>
            <p>Your conversation will appear here</p>
            <small>The AI will respond to your questions using your uploaded documents</small>
        </div>
    `;

    conversationHistory = [];
    updateStatus('ready', 'Ready to start conversation');
    AppUtils.showToast('Conversation cleared', 'success');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check for secure context (required for microphone access)
    const isSecureContext = window.isSecureContext || location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';

    if (!isSecureContext) {
        AppUtils.showToast(
            'Voice features require HTTPS or localhost. Current URL uses HTTP from ' + location.hostname,
            'danger'
        );
        document.getElementById('voiceButton').disabled = true;
        updateStatus('error', 'Insecure context - HTTPS required');
        console.error('Voice features blocked: Page must be served over HTTPS or from localhost for microphone access');
        return;
    }

    // Check for browser support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        AppUtils.showToast('Voice features not supported in this browser', 'warning');
        document.getElementById('voiceButton').disabled = true;
        updateStatus('error', 'Voice features not supported');
    }

    // Check for MediaRecorder support
    if (!window.MediaRecorder) {
        AppUtils.showToast('Audio recording not supported in this browser', 'warning');
        document.getElementById('voiceButton').disabled = true;
        updateStatus('error', 'Recording not supported');
    }
});
