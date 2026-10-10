// SmallLLM Web Interface - Complete Chat UI

// Global State
let state = {
    chats: [],
    currentChatId: 0,
    config: {
        apiUrl: localStorage.getItem('apiUrl') || 'http://localhost:5000',
        theme: localStorage.getItem('theme') || 'system',
        model: localStorage.getItem('model') || 'small',
        temperature: parseFloat(localStorage.getItem('temperature')) || 0.7,
        maxTokens: parseInt(localStorage.getItem('maxTokens')) || 100,
        saveChats: localStorage.getItem('saveChats') !== 'false'
    },
    apiConnected: false
};

// DOM Elements
const sidebar = document.getElementById('sidebar');
const mainContent = document.getElementById('main');
const chatMessages = document.getElementById('chat-messages');
const chatArea = document.getElementById('chat-area');
const chatWelcome = document.getElementById('chat-welcome');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const globalStatusDot = document.getElementById('global-status-dot');
const globalStatusText = document.getElementById('global-status-text');
const currentChatTitle = document.getElementById('current-chat-title');
const selectedModel = document.getElementById('selected-model');
const temperatureSlider = document.getElementById('temperature');
const temperatureValue = document.getElementById('temp-value');
const maxTokensSlider = document.getElementById('max-tokens');
const maxTokensValue = document.getElementById('tokens-value');
const modelDropdown = document.getElementById('model-dropdown');
const toastContainer = document.getElementById('toast-container');
const settingsModal = document.getElementById('settings-modal');
const chatHistory = document.getElementById('chat-history');

// Initialize
function init() {
    loadState();
    applyTheme();
    setupEventListeners();
    checkApiStatus();
    loadChats();
    updateUI();
    userInput.focus();
}

// Load State from LocalStorage
function loadState() {
    const savedState = localStorage.getItem('smallllm-state');
    if (savedState) {
        try {
            const parsed = JSON.parse(savedState);
            state.config = { ...state.config, ...parsed.config };
        } catch (e) {
            console.error('Error loading state:', e);
        }
    }
}

// Save State to LocalStorage
function saveState() {
    localStorage.setItem('smallllm-state', JSON.stringify({
        config: state.config
    }));
}

// Apply Theme
function applyTheme() {
    const html = document.documentElement;
    html.removeAttribute('data-theme');
    
    if (state.config.theme === 'dark') {
        html.setAttribute('data-theme', 'dark');
    } else if (state.config.theme === 'light') {
        // Light is default
    } else {
        // System theme
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
            html.setAttribute('data-theme', 'dark');
        }
    }
}

// Setup Event Listeners
function setupEventListeners() {
    // Send button
    sendButton.addEventListener('click', sendMessage);
    
    // Input keypress
    userInput.addEventListener('keydown', handleInputKey);
    
    // Input auto-resize
    userInput.addEventListener('input', autoResize);
    
    // Temperature slider
    temperatureSlider.addEventListener('input', () => {
        state.config.temperature = parseFloat(temperatureSlider.value);
        temperatureValue.textContent = state.config.temperature;
        localStorage.setItem('temperature', state.config.temperature);
    });
    
    // Max tokens slider
    maxTokensSlider.addEventListener('input', () => {
        state.config.maxTokens = parseInt(maxTokensSlider.value);
        maxTokensValue.textContent = state.config.maxTokens;
        localStorage.setItem('maxTokens', state.config.maxTokens);
    });
    
    // Model dropdown toggle
    document.querySelector('.model-selector').addEventListener('click', toggleModelDropdown);
    
    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.model-selector')) {
            modelDropdown.classList.remove('active');
        }
    });
    
    // Theme change from system preferences
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
        if (state.config.theme === 'system') {
            applyTheme();
        }
    });
    
    // Close modal on outside click
    settingsModal.addEventListener('click', (e) => {
        if (e.target === settingsModal) {
            closeSettings();
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + , for settings
        if ((e.ctrlKey || e.metaKey) && e.key === ',') {
            e.preventDefault();
            openSettings();
        }
        
        // Ctrl/Cmd + N for new chat
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            newChat();
        }
        
        // Escape to close dropdown
        if (e.key === 'Escape') {
            modelDropdown.classList.remove('active');
        }
    });
}

// Handle Input Key
function handleInputKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

// Auto Resize Textarea
function autoResize() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 200) + 'px';
}

// Toggle Sidebar
function toggleSidebar() {
    sidebar.classList.toggle('active');
    mainContent.classList.toggle('sidebar-collapsed');
}

// Toggle Model Dropdown
function toggleModelDropdown() {
    modelDropdown.classList.toggle('active');
}

// Select Model
function selectModel(model) {
    state.config.model = model;
    localStorage.setItem('model', model);
    
    const modelNames = {
        'tiny': 'SmallLLM: Tiny',
        'small': 'SmallLLM: Small',
        'medium': 'SmallLLM: Medium'
    };
    
    selectedModel.textContent = modelNames[model] || `SmallLLM: ${model}`;
    modelDropdown.classList.remove('active');
    
    showToast('✅ Modello selezionato: ' + model, 'success');
}

// Update UI
function updateUI() {
    // Update sliders
    temperatureSlider.value = state.config.temperature;
    temperatureValue.textContent = state.config.temperature;
    maxTokensSlider.value = state.config.maxTokens;
    maxTokensValue.textContent = state.config.maxTokens;
    
    // Update model selector
    const modelNames = {
        'tiny': 'SmallLLM: Tiny',
        'small': 'SmallLLM: Small',
        'medium': 'SmallLLM: Medium'
    };
    selectedModel.textContent = modelNames[state.config.model] || `SmallLLM: ${state.config.model}`;
    
    // Update send button state
    sendButton.disabled = !state.apiConnected || !userInput.value.trim();
    
    // Update API status
    statusDot.className = state.apiConnected ? 'status-dot' : 'status-dot error';
    statusText.textContent = state.apiConnected ? 'API connessa' : 'API non connessa';
    globalStatusDot.className = state.apiConnected ? 'status-dot' : 'status-dot error';
    globalStatusText.textContent = state.apiConnected ? 'Locale' : 'Non connessa';
}

// Check API Status
async function checkApiStatus() {
    try {
        const response = await fetch(`${state.config.apiUrl}/api/health`);
        const data = await response.json();
        
        if (data.status === 'healthy') {
            state.apiConnected = true;
            showToast('✅ API connessa con successo!', 'success');
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        state.apiConnected = false;
        console.error('API connection error:', error);
    }
    
    updateUI();
}

// Show Toast Notification
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        info: 'fa-info-circle'
    };
    
    toast.innerHTML = `<i class="fas ${icons[type] || 'fa-info-circle'}"></i> <span>${message}</span>`;
    toastContainer.appendChild(toast);
    
    // Remove after 3 seconds
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Send Message
async function sendMessage() {
    const text = userInput.value.trim();
    
    if (!text || !state.apiConnected) {
        return;
    }
    
    // Clear input
    userInput.value = '';
    autoResize();
    
    // Add user message
    const userMessageId = addMessage(text, 'user');
    
    // Hide welcome screen
    chatWelcome.style.display = 'none';
    
    // Show typing indicator
    const typingIndicator = showTypingIndicator();
    
    // Disable input
    sendButton.disabled = true;
    userInput.disabled = true;
    
    try {
        // Call API
        const response = await fetch(`${state.config.apiUrl}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: text,
                max_tokens: state.config.maxTokens,
                temperature: state.config.temperature,
                config: state.config.model
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Remove typing indicator
        typingIndicator.remove();
        
        // Add AI response
        const aiMessageId = addMessage(data.response, 'ai');
        
        // Save to current chat
        if (state.saveChats) {
            saveMessageToCurrentChat(text, data.response);
        }
        
        showToast('✅ Risposta ricevuta!', 'success');
        
    } catch (error) {
        // Remove typing indicator
        typingIndicator.remove();
        
        // Show error
        showToast(`❌ Errore: ${error.message}`, 'error');
        addMessage(`Errore: ${error.message}`, 'system');
    } finally {
        // Re-enable input
        sendButton.disabled = false;
        userInput.disabled = false;
        userInput.focus();
        updateUI();
    }
}

// Add Message to Chat
function addMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    // Format text (basic markdown support)
    const formattedText = formatMessage(text);
    messageDiv.innerHTML = `<pre>${formattedText}</pre>`;
    
    // Add copy button for AI messages
    if (type === 'ai') {
        const copyBtn = document.createElement('button');
        copyBtn.className = 'copy-btn';
        copyBtn.innerHTML = '<i class="fas fa-copy"></i>';
        copyBtn.onclick = () => copyToClipboard(text, copyBtn);
        
        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'message-actions';
        actionsDiv.appendChild(copyBtn);
        
        messageDiv.appendChild(actionsDiv);
    }
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    scrollToBottom();
    
    return messageDiv;
}

// Show Typing Indicator
function showTypingIndicator() {
    const div = document.createElement('div');
    div.className = 'typing-indicator';
    div.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    chatMessages.appendChild(div);
    scrollToBottom();
    return div;
}

// Format Message (basic markdown)
function formatMessage(text) {
    // Escape HTML
    let formatted = escapeHtml(text);
    
    // Bold: **text** -> <strong>text</strong>
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic: *text* -> <em>text</em>
    formatted = formatted.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Code: `text` -> <code>text</code>
    formatted = formatted.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Code blocks: ```code``` -> <pre><code>code</code></pre>
    formatted = formatted.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Headers
    formatted = formatted.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    
    // Lists
    formatted = formatted.replace(/^\- (.*$)/gm, '<li>$1</li>');
    formatted = formatted.replace(/^\* (.*$)/gm, '<li>$1</li>');
    formatted = formatted.replace(/^\d+\. (.*$)/gm, '<li>$1</li>');
    
    // New lines
    formatted = formatted.replace(/\n\n/g, '</p><p>');
    formatted = formatted.replace(/\n/g, '<br>');
    
    // Wrap in paragraph
    if (!formatted.startsWith('<h') && !formatted.startsWith('<p') && !formatted.startsWith('<ul') && !formatted.startsWith('<pre')) {
        formatted = `<p>${formatted}</p>`;
    }
    
    return formatted;
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Copy to Clipboard
function copyToClipboard(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        button.innerHTML = '<i class="fas fa-check"></i>';
        button.classList.add('copied');
        showToast('✅ Copiato negli appunti!', 'success');
        
        setTimeout(() => {
            button.innerHTML = '<i class="fas fa-copy"></i>';
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        showToast('❌ Errore nel copiare', 'error');
        console.error('Copy error:', err);
    });
}

// Scroll to Bottom
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Use Example
function useExample(text) {
    userInput.value = text;
    userInput.focus();
    sendMessage();
}

// Open Settings
function openSettings() {
    document.getElementById('api-url').value = state.config.apiUrl;
    
    // Set active theme button
    const themeButtons = document.querySelectorAll('.theme-btn');
    themeButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.getAttribute('onclick').includes(`'${state.config.theme}'`)) {
            btn.classList.add('active');
        }
    });
    
    document.getElementById('save-chats').checked = state.config.saveChats;
    
    settingsModal.classList.add('active');
}

// Close Settings
function closeSettings() {
    settingsModal.classList.remove('active');
}

// Select Theme
function selectTheme(theme) {
    state.config.theme = theme;
    localStorage.setItem('theme', theme);
    
    // Update active button
    const themeButtons = document.querySelectorAll('.theme-btn');
    themeButtons.forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    applyTheme();
}

// Save Settings
function saveSettings() {
    const newApiUrl = document.getElementById('api-url').value;
    const saveChats = document.getElementById('save-chats').checked;
    
    state.config.apiUrl = newApiUrl;
    state.config.saveChats = saveChats;
    
    localStorage.setItem('apiUrl', newApiUrl);
    localStorage.setItem('saveChats', saveChats);
    
    saveState();
    closeSettings();
    
    // Recheck API connection
    checkApiStatus();
    
    showToast('✅ Impostazioni salvate!', 'success');
}

// Test API Connection
async function testApiConnection() {
    const btn = event.target.closest('button');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Test...';
    btn.disabled = true;
    
    try {
        await checkApiStatus();
        if (state.apiConnected) {
            showToast('✅ Connessione API funzionante!', 'success');
        } else {
            showToast('❌ Connessione API fallita', 'error');
        }
    } catch (error) {
        showToast(`❌ Errore: ${error.message}`, 'error');
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// Chat Management
function newChat() {
    // Create new chat
    const newChatId = Date.now();
    state.currentChatId = newChatId;
    state.chats.push({
        id: newChatId,
        title: 'Nuova Chat',
        messages: [],
        createdAt: new Date().toISOString()
    });
    
    // Update chat history
    updateChatHistory();
    
    // Clear messages
    chatMessages.innerHTML = '';
    chatWelcome.style.display = 'block';
    
    // Update current chat title
    currentChatTitle.textContent = 'Nuova Chat';
    
    // Save chats
    if (state.config.saveChats) {
        saveChats();
    }
    
    // Close sidebar on mobile
    if (window.innerWidth <= 1024) {
        sidebar.classList.remove('active');
        mainContent.classList.remove('sidebar-collapsed');
    }
    
    userInput.focus();
}

// Load Chat
function loadChat(chatId) {
    const chat = state.chats.find(c => c.id == chatId);
    
    if (!chat) return;
    
    state.currentChatId = chatId;
    
    // Update current chat title
    currentChatTitle.textContent = chat.title || 'Nuova Chat';
    
    // Load messages
    chatMessages.innerHTML = '';
    chatWelcome.style.display = chat.messages.length === 0 ? 'block' : 'none';
    
    chat.messages.forEach(msg => {
        addMessage(msg.text, msg.type);
    });
    
    // Update active chat in history
    updateChatHistory();
    
    // Close sidebar on mobile
    if (window.innerWidth <= 1024) {
        sidebar.classList.remove('active');
        mainContent.classList.remove('sidebar-collapsed');
    }
    
    scrollToBottom();
    userInput.focus();
}

// Delete Chat
function deleteChat(chatId, event) {
    if (event) {
        event.stopPropagation();
    }
    
    if (state.chats.length <= 1) {
        showToast('❌ Non puoi eliminare l\'ultima chat', 'error');
        return;
    }
    
    state.chats = state.chats.filter(c => c.id != chatId);
    
    // If deleting current chat, switch to first one
    if (state.currentChatId == chatId) {
        state.currentChatId = state.chats[0].id;
        loadChat(state.currentChatId);
    }
    
    updateChatHistory();
    
    if (state.config.saveChats) {
        saveChats();
    }
    
    showToast('✅ Chat eliminata!', 'success');
}

// Update Chat History
function updateChatHistory() {
    chatHistory.innerHTML = '';
    
    state.chats.forEach(chat => {
        const item = document.createElement('div');
        item.className = `history-item ${chat.id === state.currentChatId ? 'active' : ''}`;
        item.dataset.chatId = chat.id;
        item.onclick = () => loadChat(chat.id);
        
        const title = chat.title || (chat.messages[0] ? chat.messages[0].text.substring(0, 30) + '...' : 'Nuova Chat');
        const preview = chat.messages.length > 0 ? 
            (chat.messages[chat.messages.length - 1].text.substring(0, 40) + (chat.messages[chat.messages.length - 1].text.length > 40 ? '...' : '')) :
            'Inizia una nuova conversazione';
        
        item.innerHTML = `
            <i class="fas fa-comment"></i>
            <div class="history-content">
                <span class="history-title">${escapeHtml(title)}</span>
                <span class="history-preview">${escapeHtml(preview)}</span>
            </div>
            <button class="delete-btn" onclick="event.stopPropagation(); deleteChat(${chat.id}, event)" title="Elimina">
                <i class="fas fa-trash"></i>
            </button>
        `;
        
        chatHistory.appendChild(item);
    });
}

// Save Message to Current Chat
function saveMessageToCurrentChat(userText, aiText) {
    const chat = state.chats.find(c => c.id === state.currentChatId);
    
    if (!chat) return;
    
    // If first message, update title
    if (chat.messages.length === 0) {
        chat.title = userText.substring(0, 30);
        currentChatTitle.textContent = chat.title;
        updateChatHistory();
    }
    
    chat.messages.push({
        text: userText,
        type: 'user',
        timestamp: new Date().toISOString()
    });
    
    chat.messages.push({
        text: aiText,
        type: 'ai',
        timestamp: new Date().toISOString()
    });
    
    // Save chats
    saveChats();
    updateChatHistory();
}

// Save All Chats
function saveChats() {
    localStorage.setItem('smallllm-chats', JSON.stringify(state.chats));
}

// Load Chats
function loadChats() {
    const savedChats = localStorage.getItem('smallllm-chats');
    
    if (savedChats) {
        try {
            state.chats = JSON.parse(savedChats);
            
            // If no chats or invalid, create default
            if (!state.chats || state.chats.length === 0) {
                state.chats = [{
                    id: Date.now(),
                    title: 'Nuova Chat',
                    messages: []
                }];
            }
            
            // Load first chat
            state.currentChatId = state.chats[0].id;
            loadChat(state.currentChatId);
            
        } catch (e) {
            console.error('Error loading chats:', e);
            // Create default chat
            state.chats = [{
                id: Date.now(),
                title: 'Nuova Chat',
                messages: []
            }];
            state.currentChatId = state.chats[0].id;
        }
    } else {
        // Create default chat
        state.chats = [{
            id: Date.now(),
            title: 'Nuova Chat',
            messages: []
        }];
        state.currentChatId = state.chats[0].id;
    }
    
    updateChatHistory();
}

// Clear All Chats
function clearAllChats() {
    if (confirm('Sei sicuro di voler eliminare TUTTE le chat? Questa azione non può essere annullata.')) {
        state.chats = [{
            id: Date.now(),
            title: 'Nuova Chat',
            messages: []
        }];
        state.currentChatId = state.chats[0].id;
        
        localStorage.removeItem('smallllm-chats');
        
        // Clear UI
        chatMessages.innerHTML = '';
        chatWelcome.style.display = 'block';
        currentChatTitle.textContent = 'Nuova Chat';
        updateChatHistory();
        
        showToast('✅ Tutte le chat sono state eliminate!', 'success');
        closeSettings();
    }
}

// Update Temperature
function updateTemperature(value) {
    state.config.temperature = parseFloat(value);
    temperatureValue.textContent = value;
    localStorage.setItem('temperature', value);
}

// Update Max Tokens
function updateMaxTokens(value) {
    state.config.maxTokens = parseInt(value);
    maxTokensValue.textContent = value;
    localStorage.setItem('maxTokens', value);
}

// Periodic API Status Check
setInterval(checkApiStatus, 60000); // Check every minute

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', init);

// Add CSS for toast animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
    
    .history-content {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-width: 0;
    }
    
    .history-title {
        font-weight: 500;
        font-size: 0.875rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    .history-preview {
        font-size: 0.75rem;
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
`;
document.head.appendChild(style);
