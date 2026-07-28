// Chat functionality

let conversationHistory = [];

// Send message on Enter key
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Send message to backend
async function sendMessage() {
    const input = document.getElementById('messageInput');
    const query = input.value.trim();

    if (!query) return;

    // Clear input
    input.value = '';

    // Add user message to chat
    addMessage('user', query);

    // Show loading indicator
    showLoading(true);

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Response data:', data);
        console.log('Search stats:', data.search_stats);

        if (data.error) {
            throw new Error(data.error);
        }

        // Add assistant response to chat with search stats
        addMessage('assistant', data.answer, data.citations, data.retrieved_chunks, data.search_stats);

        // Update conversation history
        conversationHistory.push({
            query,
            answer: data.answer,
            citations: data.citations,
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('Error sending message:', error);
        addMessage('assistant', `Error: ${error.message}`, [], []);
        AppUtils.showToast(`Failed to get response: ${error.message}`, 'danger');
    } finally {
        showLoading(false);
    }
}

// Add message to chat UI
function addMessage(role, content, citations = [], chunks = [], searchStats = null) {
    const messagesDiv = document.getElementById('chatMessages');

    // Remove initial prompt if exists
    const initialPrompt = messagesDiv.querySelector('.text-center.text-muted');
    if (initialPrompt) {
        initialPrompt.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;

    let citationsHtml = '';
    if (citations && citations.length > 0) {
        citationsHtml = '<div class="citation">';
        citationsHtml += '<strong><i class="fas fa-quote-left"></i> Sources:</strong>';
        citations.forEach(c => {
            citationsHtml += `
                <div class="citation-item">
                    <i class="fas fa-file-pdf text-danger"></i> 
                    ${c.source_filename} 
                    ${c.page_number ? `(Page ${c.page_number})` : ''} 
                    - Chunk ${c.chunk_index}
                </div>
            `;
        });
        citationsHtml += '</div>';
    }

    // Build search stats HTML for assistant messages
    let searchStatsHtml = '';
    if (role === 'assistant' && searchStats) {
        let statsText = '';
        if (searchStats.mode === 'hybrid') {
            statsText = `Hybrid: ${searchStats.semantic_count} semantic + ${searchStats.keyword_count} keyword → ${searchStats.after_dedup || '?'} unique`;
        } else if (searchStats.mode === 'keyword') {
            statsText = `Keyword: ${searchStats.keyword_count} results → ${searchStats.after_dedup || '?'} unique`;
        } else {
            statsText = `Semantic: ${searchStats.semantic_count} results → ${searchStats.after_dedup || '?'} unique`;
        }
        searchStatsHtml = `
            <div class="search-stats">
                <i class="fas fa-search"></i> ${statsText}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-bubble">
            <div class="message-content">${escapeHtml(content)}</div>
            ${role === 'assistant' ? searchStatsHtml : ''}
            ${role === 'assistant' ? citationsHtml : ''}
        </div>
        <div class="message-timestamp">${AppUtils.formatTimestamp()}</div>
    `;

    messagesDiv.appendChild(messageDiv);

    // Update retrieved chunks panel
    if (chunks && chunks.length > 0) {
        updateRetrievedChunks(chunks);
    }

    // Scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Update retrieved chunks panel
function updateRetrievedChunks(chunks) {
    const chunksContent = document.getElementById('chunksContent');

    let html = '<div class="accordion" id="chunksAccordion">';

    chunks.forEach((chunk, index) => {
        const metadata = chunk.metadata || {};
        html += `
            <div class="accordion-item">
                <h2 class="accordion-header" id="heading${index}">
                    <button class="accordion-button ${index !== 0 ? 'collapsed' : ''}" 
                            type="button" 
                            data-bs-toggle="collapse" 
                            data-bs-target="#collapse${index}">
                        <strong>Chunk ${index + 1}:</strong>&nbsp;
                        ${metadata.source_filename || 'Unknown'} 
                        ${metadata.page_number ? `(Page ${metadata.page_number})` : ''}
                    </button>
                </h2>
                <div id="collapse${index}" 
                     class="accordion-collapse collapse ${index === 0 ? 'show' : ''}" 
                     data-bs-parent="#chunksAccordion">
                    <div class="accordion-body">
                        <pre class="mb-0 text-break" style="white-space: pre-wrap;">${escapeHtml(chunk.document)}</pre>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    chunksContent.innerHTML = html;
}

// Show/hide loading indicator
function showLoading(show) {
    const indicator = document.getElementById('loadingIndicator');
    indicator.style.display = show ? 'block' : 'none';
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initialize chat on page load
document.addEventListener('DOMContentLoaded', () => {
    // Focus on input
    document.getElementById('messageInput').focus();

    // Load retrieval mode and update badge
    loadRetrievalMode();

    // Add welcome message
    const messagesDiv = document.getElementById('chatMessages');
    if (!messagesDiv.querySelector('.chat-message')) {
        // Keep initial prompt
    }
});

// Fetch and display retrieval mode
async function loadRetrievalMode() {
    console.log('Loading retrieval mode config...');
    try {
        const response = await fetch('/api/config');
        console.log('Config response status:', response.status);
        if (response.ok) {
            const config = await response.json();
            console.log('Config loaded:', config);
            updateModeBadge(config.retrieval_mode);
        } else {
            console.error('Config response not OK:', response.status);
            updateModeBadge('error');
        }
    } catch (error) {
        console.error('Failed to load config:', error);
        updateModeBadge('unknown');
    }
}

// Update mode badge in header
function updateModeBadge(mode) {
    const badge = document.getElementById('retrievalModeBadge');
    if (!badge) return;

    const icons = {
        'hybrid': '🔀',
        'keyword': '🔤',
        'semantic': '🧠'
    };
    const labels = {
        'hybrid': 'Hybrid',
        'keyword': 'Keyword',
        'semantic': 'Semantic'
    };
    const colors = {
        'hybrid': 'bg-success',
        'keyword': 'bg-warning text-dark',
        'semantic': 'bg-info'
    };

    badge.innerHTML = `${icons[mode] || '❓'} ${labels[mode] || mode}`;
    badge.className = `badge ${colors[mode] || 'bg-secondary'}`;
}
