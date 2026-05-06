# Bugs Fixed

This document logs all the bugs found and fixed across the codebase during the system completion pass.

### 1. SSE Stream Chunk Truncation
- **Where:** `chat.js` (`handleChatSubmit`)
- **Bug:** `[DONE]` inside chunked Server-Sent Events (SSE) responses could be split across packets or chunk boundaries, leading to early termination or lost tokens.
- **Fix:** Switched to buffering incomplete lines using `buffer += decoder.decode(value, { stream: true })`. Processed complete `\n\n` delimited messages and appended the leftover chunk back to the buffer.

### 2. Stale Chat Context Across Diagnoses
- **Where:** `chat.js` & `diagnosis.js`
- **Bug:** When running a new diagnosis, the chat kept the old `messageHistory`, resulting in Qwen receiving mixed histories.
- **Fix:** Added a `window.resetChatForNewContext()` hook to clear messages before opening chat, and pushed a synthetic bot message dynamically to greet the user with the new diagnosis info.

### 3. Missing Chat API Error Handling
- **Where:** `chat.js`
- **Bug:** A non-200 status (e.g., from `503 Service Unavailable` or `429 Too Many Requests`) would cause the chat to silently hang.
- **Fix:** Added a try/catch block with a specific check for `!response.ok` that extracts the JSON error or HTTP status and appends a visible red error bubble to the DOM.

### 4. No Auto-Scroll in Chat
- **Where:** `chat.js`
- **Bug:** Chat panel did not scroll to the bottom after rendering new tokens.
- **Fix:** Added a `scrollToBottom` function using `chatMessages.scrollTop = chatMessages.scrollHeight` and invoked it during SSE chunk rendering and new message additions.

### 5. Markdown Parsing Missing
- **Where:** `chat.js`
- **Bug:** Model responses were displayed as raw text, ignoring Qwen's Markdown formatting.
- **Fix:** Implemented a lightweight `parseMarkdown()` function using Regex to handle bold (`**text**`), italics (`*text*`), inline code (`` `code` ``), and bullet lists (`- item` or `* item`).

### 6. Missing CORS Header
- **Where:** `app.py`
- **Bug:** Frontend API calls would fail if hosted on different domains, or sometimes even across localhost loops.
- **Fix:** Installed and imported `flask_cors` and wrapped the Flask app to enable CORS for localhost/127.0.0.1.

### 7. Backend Input Validation
- **Where:** `app.py`
- **Bug:** Server trusted the frontend blindly and passed payload directly to `predictor.py`, crashing on missing or non-numeric keys.
- **Fix:** Iterated over required fields ensuring keys existed and could be parsed as `float`. Returned `400 Bad Request` gracefully otherwise.

### 8. Frontend Input Validation
- **Where:** `diagnosis.js`
- **Bug:** Form accepted out-of-bounds metrics resulting in invalid/meaningless predictions.
- **Fix:** Created a `validatePatient(patient)` function checking values against defined medically plausible ranges. Extracted styling into inline `.error-msg` divs to flag problematic inputs visually to the user.

### 9. Hardcoded Joblib Paths
- **Where:** `predictor.py`
- **Bug:** `joblib.load('../models/...')` crashed if the Flask app wasn't launched from inside the `interface` directory specifically.
- **Fix:** Used Python's `pathlib.Path` to dynamically infer the `models` folder relative to the script location (`Path(__file__).parent.parent / 'models'`).

### 10. Thread Safety in Generation
- **Where:** `llm_engine.py` & `app.py`
- **Bug:** Concurrent requests accessing `MODEL.generate()` directly could corrupt generation or cause OOM errors.
- **Fix:** Implemented a `threading.Lock()`. If the lock is held, `app.py` instantly rejects the request with a `429 Too Many Requests` response.

### 11. Graceful Model Load Fallback
- **Where:** `llm_engine.py`
- **Bug:** Missing model weights would crash the entire Flask server at startup, preventing the basic expert system UI from running.
- **Fix:** Wrapped `AutoModelForCausalLM` loading in a try/except block, setting a global `LLM_AVAILABLE` flag, and checked this flag in `app.py` to route around failures.

### 12. Streaming Disconnect Leaks
- **Where:** `llm_engine.py` & `app.py`
- **Bug:** A disconnected client would leave the lock locked forever, breaking the bot.
- **Fix:** Handled via Python's `try...finally` block encapsulating the generator in `app.py` to ensure `llm_lock.release()` is always called, even if `GeneratorExit` is raised due to a closed client connection.

### 13. System Prompt String Interpolation Issue
- **Where:** `knowledge_base.py`
- **Bug:** Using escaped quotes (`\"\"\"`) inside f-strings on older Python versions or complex parsing caused a `SyntaxError: unexpected character after line continuation character`.
- **Fix:** Avoided backslash escaping completely and used clean multi-line `"""` strings.

### 14. OS Error 1455: Paging File Too Small (OOM)
- **Where:** `app.py`
- **Bug:** Starting the Flask server with `debug=True` enabled the Werkzeug auto-reloader, which spawns a second subprocess. Loading a 1.5B LLM model twice exceeded the system's memory and page file capacity.
- **Fix:** Added `use_reloader=False` to `app.run()` to prevent the duplicate memory allocation while retaining debug logging.
