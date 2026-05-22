// 椎間盤分割系統 JavaScript 功能

// 處理圖片上傳和分析
function processImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showStatus('請先選擇一張圖片', 'error');
        return;
    }
    
    // 檢查文件類型
    if (!file.type.startsWith('image/')) {
        showStatus('請選擇有效的圖片文件', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    // 顯示載入狀態
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    showStatus('', '');
    
    // 禁用按鈕防止重複提交
    const button = document.querySelector('button');
    button.disabled = true;
    button.textContent = '處理中...';
    
    // 發送請求
    fetch('/process', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // 隱藏載入狀態
        document.getElementById('loading').style.display = 'none';
        
        // 恢復按鈕狀態
        button.disabled = false;
        button.textContent = '開始分析';
        
        if (data.success) {
            // 顯示結果
            document.getElementById('originalImg').src = 'data:image/png;base64,' + data.original;
            document.getElementById('firstImg').src = 'data:image/png;base64,' + data.first;
            document.getElementById('secondImg').src = 'data:image/png;base64,' + data.second;
            document.getElementById('results').style.display = 'block';
            showStatus('分析完成！結果已保存為 _result.jpg 和 _result_disc.jpg', 'success');
        } else {
            showStatus('處理錯誤：' + data.error, 'error');
        }
    })
    .catch(error => {
        // 隱藏載入狀態並恢復按鈕
        document.getElementById('loading').style.display = 'none';
        button.disabled = false;
        button.textContent = '開始分析';
        
        showStatus('網路錯誤：' + error.message, 'error');
    });
}

    // 狀態管理
    function updateStatus(message) {
        statusDiv.textContent = message;
    }

    function showStatus() {
        statusDiv.style.display = 'block';
    }

    function hideStatus() {
        statusDiv.style.display = 'none';
    }

    // 聊天機器人功能
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const sendButton = document.getElementById('send-button');

    // 發送聊天消息
    function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        // 添加用戶消息到聊天區域
        addMessageToChat(message, 'user');
        
        // 清空輸入框
        chatInput.value = '';
        
        // 禁用發送按鈕
        sendButton.disabled = true;
        sendButton.textContent = '發送中...';
        
        // 顯示打字指示器
        showTypingIndicator();

        // 發送請求到後端
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // 移除打字指示器
            removeTypingIndicator();
            
            // 添加機器人回覆
            if (data.response) {
                addMessageToChat(data.response, 'bot');
            } else if (data.error) {
                addMessageToChat('抱歉，處理您的請求時出現錯誤。', 'bot');
            }
        })
        .catch(error => {
            console.error('Chat error:', error);
            removeTypingIndicator();
            addMessageToChat('抱歉，無法連接到聊天服務。', 'bot');
        })
        .finally(() => {
            // 重新啟用發送按鈕
            sendButton.disabled = false;
            sendButton.textContent = '發送';
        });
    }

    // 添加消息到聊天區域
    function addMessageToChat(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        messageDiv.textContent = message;
        
        chatMessages.appendChild(messageDiv);
        
        // 滾動到最新消息
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 顯示打字指示器
    function showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message typing-indicator';
        typingDiv.id = 'typing-indicator';
        typingDiv.textContent = '正在輸入...';
        
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // 移除打字指示器
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    // 綁定聊天事件
    if (sendButton) {
        sendButton.addEventListener('click', sendChatMessage);
    }

    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendChatMessage();
            }
        });
    }

// 文件拖拽功能
function initDragAndDrop() {
    const uploadArea = document.querySelector('.upload-area');
    
    // 拖拽進入
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#007bff';
        uploadArea.style.backgroundColor = '#f8f9fa';
    });
    
    // 拖拽離開
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'transparent';
    });
    
    // 放置文件
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'transparent';
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            document.getElementById('imageInput').files = files;
            // 顯示已選擇的文件名
            showStatus(`已選擇文件: ${files[0].name}`, 'info');
        }
    });
}

// 文件選擇變化處理
function handleFileChange() {
    const fileInput = document.getElementById('imageInput');
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            showStatus(`已選擇文件: ${file.name}`, 'info');
        }
    });
}

// 圖片預覽功能
function previewImage(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        // 可以在這裡添加圖片預覽功能
        console.log('圖片已載入:', e.target.result);
    };
    reader.readAsDataURL(file);
}

// 鍵盤快捷鍵支援
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl + Enter 或 Cmd + Enter 執行分析
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            processImage();
        }
        
        // Escape 清除狀態
        if (e.key === 'Escape') {
            showStatus('', '');
        }
    });
}

// 頁面載入完成後初始化
document.addEventListener('DOMContentLoaded', function() {
    // 初始化拖拽功能
    initDragAndDrop();
    
    // 初始化文件選擇處理
    handleFileChange();
    
    // 初始化鍵盤快捷鍵
    initKeyboardShortcuts();
    
    // 顯示歡迎訊息
    showStatus('系統已準備就緒，請上傳醫學影像進行分析', 'info');
    
    console.log('椎間盤分割系統已初始化完成');
});
