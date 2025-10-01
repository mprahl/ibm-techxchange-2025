/* Basic, dependency-free client for Yoda Speak via OpenAI-compatible chat */
(function () {
  const input = document.getElementById('inputText');
  const output = document.getElementById('outputText');
  const status = document.getElementById('status');
  const btn = document.getElementById('translateBtn');

  const API_ENDPOINT = '/api/chat';

  function setBusy(isBusy) {
    btn.disabled = isBusy;
    btn.textContent = isBusy ? 'Translating…' : 'Translate';
  }

  function setStatus(msg, isError = false) {
    status.textContent = msg || '';
    status.style.color = isError ? '#ff8a8a' : '#a7b7c2';
  }

  async function translate() {
    const text = input.value.trim();
    if (!text) {
      setStatus('Enter text to translate.');
      return;
    }

    setBusy(true);
    setStatus('Contacting Yoda…');
    output.value = '';

    try {
      const payload = {
        temperature: 0.0,
        max_tokens: 64,
        messages: [
          {
            role: 'system',
            content:
              'You are a translation engine. Reply with only the translation of the original sentence.',
          },
          { role: 'user', content: `Translate to Yoda speak: ${text}` },
        ],
      };

      const headers = { 'Content-Type': 'application/json' };
      const res = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers,
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText}`);
      }

      const data = await res.json();
      const content = data?.choices?.[0]?.message?.content ?? '';
      if (!content) {
        setStatus('No content returned. Try again or adjust max tokens.', true);
        return;
      }
      output.value = content.replace(/^\s+|\s+$/g, '');
      setStatus('Done.');
    } catch (err) {
      console.error(err);
      setStatus(String(err.message || err), true);
    } finally {
      setBusy(false);
    }
  }

  btn.addEventListener('click', translate);
  input.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'enter') {
      translate();
    }
  });
})();


