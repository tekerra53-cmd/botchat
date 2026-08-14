document.addEventListener('DOMContentLoaded', function() {
  const chatContainer = document.getElementById('chat-container');
  const messageInput = document.getElementById('message-input');
  const sendBtn = document.getElementById('send-btn');
  const clearBtn = document.getElementById('clear-chat');

  // Keep chat text flowing left-to-right so mobile keyboards/browsers do not
  // auto-switch the composer or user bubbles into RTL mode.
  messageInput.setAttribute('dir', 'ltr');
  messageInput.setAttribute('inputmode', 'text');

  // Initialize send button state
  function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText;
    sendBtn.style.opacity = hasText ? '1' : '0.5';
  }

  messageInput.addEventListener('input', updateSendButton);
  updateSendButton();

  function createMessageElement(content, isUser = false, sources = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;

    const avatarDiv = document.createElement('div');
    avatarDiv.className = 'message-avatar';
    avatarDiv.innerHTML = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.setAttribute('dir', 'ltr');
    contentDiv.textContent = content;

    if (!isUser && Array.isArray(sources) && sources.length > 0) {
      const metaDiv = document.createElement('div');
      metaDiv.className = 'message-meta';

      const sourceBadge = document.createElement('span');
      sourceBadge.className = 'source-badge';
      sourceBadge.textContent = sources.length === 1 ? 'Source' : `Sources (${sources.length})`;
      sourceBadge.setAttribute('role', 'button');
      sourceBadge.setAttribute('tabindex', '0');
      sourceBadge.setAttribute('aria-expanded', 'false');
      sourceBadge.setAttribute('aria-label', 'Toggle sources');

      const sourceList = document.createElement('div');
      sourceList.className = 'source-list';
      sourceList.setAttribute('aria-hidden', 'true');

      const chips = document.createElement('div');
      chips.className = 'source-chips';
      sources.forEach(src => {
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        chip.textContent = src;
        chips.appendChild(chip);
      });
      sourceList.appendChild(chips);

      const copyBtn = document.createElement('button');
      copyBtn.type = 'button';
      copyBtn.className = 'copy-sources';
      copyBtn.textContent = 'Copy sources';

      const toggleSources = (forceOpen = null) => {
        // Auto-collapse other source lists
        document.querySelectorAll('.source-list').forEach(list => {
          if (list !== sourceList) {
            list.classList.remove('is-open');
            list.setAttribute('aria-hidden', 'true');
          }
        });
        document.querySelectorAll('.source-badge').forEach(badge => {
          if (badge !== sourceBadge) {
            badge.setAttribute('aria-expanded', 'false');
          }
        });

        const isCurrentlyOpen = sourceList.classList.contains('is-open');
        const shouldOpen = forceOpen === null ? !isCurrentlyOpen : forceOpen;
        if (shouldOpen) {
          sourceList.classList.add('is-open');
          sourceList.setAttribute('aria-hidden', 'false');
        } else {
          sourceList.classList.remove('is-open');
          sourceList.setAttribute('aria-hidden', 'true');
        }
        sourceBadge.setAttribute('aria-expanded', String(shouldOpen));
      };

      copyBtn.addEventListener('click', async () => {
        const text = sources.join(', ');
        try {
          if (navigator.clipboard && navigator.clipboard.writeText) {
            await navigator.clipboard.writeText(text);
          } else {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.opacity = '0';
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            ta.remove();
          }
          copyBtn.textContent = 'Copied!';
          setTimeout(() => {
            copyBtn.textContent = 'Copy sources';
          }, 1200);
        } catch (err) {
          console.error('Copy failed:', err);
          copyBtn.textContent = 'Copy failed';
          setTimeout(() => {
            copyBtn.textContent = 'Copy sources';
          }, 1200);
        }
      });

      sourceBadge.addEventListener('click', () => toggleSources(null));
      sourceBadge.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleSources(null);
        }
      });

      metaDiv.addEventListener('mouseenter', () => toggleSources(true));
      metaDiv.addEventListener('mouseleave', () => toggleSources(false));

      metaDiv.appendChild(sourceBadge);
      metaDiv.appendChild(copyBtn);
      metaDiv.appendChild(sourceList);
      contentDiv.appendChild(metaDiv);
    }

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    return messageDiv;
  }

  function addMessage(content, isUser = false, sources = null) {
    const messageElement = createMessageElement(content, isUser, sources);
    chatContainer.appendChild(messageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Add a small delay for smooth animation
    setTimeout(() => {
      messageElement.style.opacity = '1';
    }, 10);
  }

  function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
      <div class="message-avatar">
        <i class="fas fa-robot"></i>
      </div>
      <div class="message-content">
        <span>UniBot is typing</span>
        <div class="typing-dots">
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
          <div class="typing-dot"></div>
        </div>
      </div>
    `;
    chatContainer.appendChild(typingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
      typingIndicator.remove();
    }
  }

  function sendMessage(message = null) {
    const query = message || messageInput.value.trim();
    if (!query) return;

    // Hide welcome message after first interaction
    const welcomeMessage = document.querySelector('.welcome-message');
    if (welcomeMessage) {
      welcomeMessage.style.display = 'none';
    }

    addMessage(query, true);
    messageInput.value = '';

    updateSendButton();
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    showTypingIndicator();

    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: query })
    })
    .then(res => res.json())
    .then(data => {
      hideTypingIndicator();
      addMessage(data.response, false, data.sources);
    })
    .catch(err => {
      hideTypingIndicator();
      addMessage('Sorry, I encountered an error. Please try again later.');
      console.error('Chat error:', err);
    })
    .finally(() => {
      sendBtn.disabled = false;
      sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
      updateSendButton();
    });
  }

  // Event listeners
  sendBtn.addEventListener('click', () => sendMessage());

  messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  // Quick question buttons
  document.addEventListener('click', (e) => {
    if (e.target.classList.contains('quick-btn')) {
      const question = e.target.dataset.question;
      messageInput.value = question;
      updateSendButton();
      sendMessage(question);
    }
  });

  // Clear chat functionality
  clearBtn.addEventListener('click', () => {
    // Keep only the welcome message
    const welcomeMessage = document.querySelector('.welcome-message');
    chatContainer.innerHTML = '';
    if (welcomeMessage) {
      chatContainer.appendChild(welcomeMessage);
      welcomeMessage.style.display = 'flex';
    }

    // Show success message
    const successMsg = createMessageElement('Chat cleared! How can I help you today?', false);
    chatContainer.appendChild(successMsg);
  });

  // Auto-focus input on page load
  messageInput.focus();

  // Add some helpful keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to clear chat
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      clearBtn.click();
    }

    // Escape to clear input
    if (e.key === 'Escape') {
      messageInput.value = '';
      updateSendButton();
      messageInput.focus();
    }
  });

  // Add smooth scrolling for better UX
  chatContainer.addEventListener('scroll', () => {
    // Could add logic to load more messages if needed
  });
});
