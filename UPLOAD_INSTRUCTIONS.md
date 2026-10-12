# Update your existing repository

1. Extract this ZIP. Upload the extracted files into the **root** of your existing `ecosignal_ask_lichen-lens` GitHub repository, replacing same-name files. Do not upload the ZIP as one file.
2. Commit to `main`; allow the existing Render service to redeploy.
3. In **Render > Environment**, check these values (Render Dashboard values for an existing service may override `render.yaml`):

- `ECOSIGNAL_CHAT_MODEL=gemini-3.1-flash-lite`
- `ECOSIGNAL_CHAT_FALLBACK_MODELS=gemini-3.8-flash`
- `ECOSIGNAL_GEMINI_TIMEOUT_MS=30000`
- `ECOSIGNAL_GEMINI_CONCURRENCY=1`
- `ECOSIGNAL_GEMINI_MAX_ATTEMPTS=2`
- `ECOSIGNAL_GEMINI_BACKOFF_SECONDS=1.0`
- `ECOSIGNAL_GEMINI_QUEUE_TIMEOUT_SECONDS=8`

Keep `GEMINI_API_KEY` stored privately in Render.

4. Test `https://ecosignal-ask-lichen-lens.onrender.com/docs` > POST `/assistant/ask` with:

```json
{"question":"What are soredia in lichens?","user_id":"cac-validation-test"}
```

5. Watch Render logs for `GEMINI_OK`, `GEMINI_ERROR`, `GEMINI_BACKOFF` and `CHAT_RESULT`.

Notes: Each failed 503/504 can now take up to 30 seconds per model plus backoff, so a two-model attempt may take around a minute. A frontend timeout shorter than this should be increased separately. The concurrency limit is per server process, not global across scaled instances. API rate limits and outages still require fallback.

Context-aware scientific validation: rejects only clear unsupported affirmative claims; permits negations and limitation explanations.
