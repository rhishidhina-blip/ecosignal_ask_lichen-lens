# Upload this entire folder to your existing GitHub repository

Repository: https://github.com/rhishidhina-blip/ecosignal_ask_lichen-lens

1. Extract the ZIP locally.
2. In the GitHub repository, upload **all seven root-level files** from the extracted folder using Add file > Upload files. Replace existing same-named files and commit to main. The `.python-version` and `.gitignore` files may be hidden by your operating system; enable hidden files if needed.
3. Existing Render service should automatically deploy. If not, use Manual Deploy > Deploy latest commit.
4. Keep your existing Render service Environment settings. This file supplies Render Blueprint defaults but **updating render.yaml alone does not necessarily modify environment variables on an existing manually-created service**. In Render > Environment, set `ECOSIGNAL_CHAT_MODEL=gemini-3.1-flash-lite`, `ECOSIGNAL_CHAT_FALLBACK_MODELS=gemini-3.8-flash`, `ECOSIGNAL_ALLOWED_ORIGINS=https://ecosignal-app.dhina-srini.chatgpt.site`, and leave your secret `GEMINI_API_KEY` as-is.
5. Keep your existing Render build command `python -m pip install --only-binary=:all: -r requirements.txt` and start command `uvicorn main:api --host 0.0.0.0 --port $PORT`.
6. Verify https://ecosignal-ask-lichen-lens.onrender.com/assistant/health and POST `/assistant/ask` from `/docs`.

The ZIP is a complete **backend** repository package. The existing EcoSignal website is a separate site project. Its React UI changes must be deployed in that site's source and cannot be applied just by uploading this backend repository ZIP.

No API key or private scan database is included. Real scan-specific answers still require connecting production database securely.
