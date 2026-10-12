# GitHub Deployment

For the temporary Gemini connectivity test, follow
[RENDER_DIAGNOSTICS.md](RENDER_DIAGNOSTICS.md) after uploading this package.

1. Extract the ZIP locally.
2. Open https://github.com/rhishidhina-blip/ecosignal_ask_lichen-lens
3. Replace matching files in the repository ROOT (including main.py).
4. Commit to the main branch.
5. Confirm Render deployment has completed.
6. In https://ecosignal-ask-lichen-lens.onrender.com/docs test POST /assistant/ask with:

```json
{"question":"What do lichens primarily bioindicate about air quality?","user_id":"cac-answer-quality-test"}
```

Do not copy any GEMINI_API_KEY into GitHub. Retain your existing Render environment settings.
