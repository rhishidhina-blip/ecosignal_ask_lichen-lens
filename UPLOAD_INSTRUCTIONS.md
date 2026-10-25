# EcoSignal Ask LichenLens — observation and image context v1

Completed continuation of the supplied draft, integrated against Site v27 source 91e92fdf13366be33b18d5733c277ffd451a1a34. This backend package is prepared and tested; it is NOT deployed to Render by this handoff.

## Deployment — existing Render service
1. Extract this ZIP. Copy its CONTENTS into the existing Ask LichenLens backend GitHub repository root, replacing matching files. Preserve tests/ folders; do not flatten. This is the chatbot repository, not the adaptive planner repository.
2. Commit. Let the existing Render service deploy that commit (or select Manual Deploy > Deploy latest commit). Do not create another service or change the Site audience.
3. Keep the existing secret GEMINI_API_KEY and working ECOSIGNAL_CHAT_MODEL. Do not paste keys into chat, code or GitHub. No new secret or training is required. If ECOSIGNAL_CHAT_FALLBACK_MODELS exists, retain only model names your account supports. The package default is unchanged gemini-3.1-flash-lite; the Site direct path retains its existing separate model setting.
4. Build: python -m pip install --only-binary=:all: -r requirements.txt
5. Start: uvicorn main:api --host 0.0.0.0 --port $PORT
6. Health: GET https://ecosignal-ask-lichen-lens.onrender.com/assistant/health must show image_context_supported:true and observation_context_version:observation-context-v1.
7. Test a text-only question, then a saved Site observation with Photograph selected. The Site requires an exact receipt for all selected fields; an old backend is rejected visibly rather than dropping attachments.

Existing Site setting ECOSIGNAL_CHATBOT_URL remains https://ecosignal-ask-lichen-lens.onrender.com/assistant/ask. Existing Codespaces image-analysis URL/token, AirNow, planner and Site access are unrelated and must remain unchanged.

## Recommended existing service settings
ECOSIGNAL_ALLOWED_ORIGINS=https://ecosignal-app.dhina-srini.chatgpt.site
ECOSIGNAL_ENABLE_EMBEDDINGS=0
ECOSIGNAL_ENABLE_PUBLIC_AUDIT=0
ECOSIGNAL_ENABLE_GEMINI_DIAGNOSTICS=0
ECOSIGNAL_GEMINI_TIMEOUT_MS=18000
ECOSIGNAL_GEMINI_MAX_ATTEMPTS=2
ECOSIGNAL_GEMINI_BUDGET_SECONDS=40
ECOSIGNAL_GEMINI_QUEUE_TIMEOUT_SECONDS=3
ECOSIGNAL_GEMINI_BACKOFF_SECONDS=1.0
ECOSIGNAL_GEMINI_CONCURRENCY=1

An existing service does not necessarily apply render.yaml environment changes automatically. Keep your working Python runtime; this package was tested on Python 3.12 and declares .python-version=3.12 for new checkouts. Do not add a new paid service.

