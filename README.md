# EcoSignal — Ask LichenLens / Render deployment

## What's inside
- `main.py`: extracted from the user's latest 61-cell notebook, **with the same 191 built-in curated knowledge records**, retrieval, source validation, FastAPI routes, and deterministic offline responses.
- Gemini structured-response retry and alternative model selection are integrated into `AskLichenLens.ask()`, not merely appended as a test cell.
- `requirements.txt`: portable Python web service dependencies (no Google Colab or localtunnel).
- `render.yaml`: optional Render Blueprint; the dashboard instructions below work without it.
- `.gitignore`: keep databases and keys outside Git.

## Deploy using the GitHub / Render dashboards
1. Create a **private** GitHub repo, e.g. `ecosignal-ask-lichenlens`. Upload **main.py, requirements.txt, render.yaml, README.md, .gitignore** into the repository root (not in a nested folder or ZIP).
2. Render > **New** > **Web Service** > Git Provider: GitHub > choose the repository.
3. Choose Runtime **Python**, a suitable region and instance plan; choose deployment branch `main`.
4. Build command: `pip install -r requirements.txt`
5. Start command: `uvicorn main:api --host 0.0.0.0 --port $PORT`
6. Set these environment variables: `GEMINI_API_KEY` (secret), `ECOSIGNAL_CHAT_MODEL=gemini-3.8-flash`, `ECOSIGNAL_CHAT_FALLBACK_MODELS=gemini-3.5-flash,gemini-3.1-flash-lite`, `ECOSIGNAL_ALLOWED_ORIGINS=https://YOUR-EXACT-WEBSITE-DOMAIN`, `ECOSIGNAL_ENABLE_EMBEDDINGS=0`. Use **Python 3.11** (configure `PYTHON_VERSION=3.11.11` if your Render service supports this setting).
7. Deploy and test `https://<your-render-domain>/assistant/health`, `https://<your-render-domain>/docs`, and `POST /assistant/ask` with body `{"question":"What are soredia?","user_id":"test"}`.
8. Change your website base API URL to `https://<your-render-domain>`; its chatbot sends POST requests to `/assistant/ask`.

## Important limitations
- The notebook's new **LocalTunnel** cell is only a temporary Colab tunnel and is *not* needed by Render; Render supplies HTTPS hosting directly.
- Database files and extra Drive knowledge files are **not** included: no saved scans, missions, repeat scans or additional knowledge files until these are securely provisioned. Health reports `production_db_found` to help check setup.
- Audit SQLite stored in the container's local filesystem may disappear on redeploy/restart. Use a persistent store for final audit retention.
- The built-in rate limiter is per-process, not an abuse-proof service-wide rate limiter. CORS is not authentication. Do not expose unrestricted private scan records or spend unlimited Gemini quota publicly.
- If Gemini is temporarily overloaded, the service retries once and then tries configured fallback models; if all fail, it returns a deterministic answer based on its knowledge bank. It does not guarantee successful AI responses or overcome API quota restrictions.
- Live hosting, DNS, billing, Google Drive DB imports, and Gemini calls have not been verified in this environment.


## GitHub upload (current deployment)
See `UPLOAD_INSTRUCTIONS.md`. Python is pinned to 3.11.11 in `.python-version`; do not use the default Python 3.14 runtime.
