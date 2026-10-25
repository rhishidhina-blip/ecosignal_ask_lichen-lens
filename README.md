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

## Contract and privacy
POST /assistant/ask accepts the existing text/history contract plus observation_context with observation_id, seven explicit selection booleans, and exactly the corresponding chosen fields: image, coverage, growth_form, satellite, geomorphology, elevation, location. Unknown/unselected fields and invalid values are rejected. Attachments override automatic stored-record assembly: the backend never merges deselected fields or downloads an arbitrary image URL. Without explicit context, the Site forwards no sample ID or implicit sample object.

Images are validated using Pillow, EXIF-oriented, resized to at most 1280px and reencoded without embedded metadata; MIME must match magic, decoded input <=700KB and <=16M pixels. The Site separately decodes/reencodes JPEGs using bounded jpeg-js decoding and a vendored ESM/Uint8Array encoder adaptation; original licences are retained. Request body limit is 1,200,000 UTF-8 bytes at both server boundaries. Validation errors exclude raw input/base64. Gemini receives actual binary SDK inline_data, not an image URL or a textual claim of attachment.

A successful response has context_receipt={version:'observation-context-v1',observation_id:...,included:[...exact category order...],image_received:boolean}. The receipt acknowledges accepted input, not scientific correctness or proof of every generated statement. Site presentation strips internal flags/model diagnostics. Context requests bypass the general-answer cache. Interpretation and answer share one Gemini generation; bounded retries remain. Existing audit stores questions/answers and evidence, but not attached image bytes. Keep audit private. Diagnostic scripts are optional one-shot CLI tools; no diagnostic HTTP route is added.

Only explicitly selected coordinates go to Gemini. A photograph may visually reveal its location despite metadata removal. Site satellite/terrain lookups use existing same-origin actual data. Elevation is an optional explicitly clicked Open-Meteo lookup; it uses Copernicus GLO-90 2021, 90m DEM, not a current survey. Open-Meteo free endpoint is for noncommercial usage and has availability/rate limits; configure commercial access separately if usage changes. Model visual descriptions never replace trained coverage/form outputs. Susceptibility is uncalibrated and the nearby 20m sample is not an exact-point 1m prediction. Current AQI remains in the existing AirNow panel; chat does not silently choose default coordinates or send unselected readings.

## Tests
Install requirements, then python -m unittest discover -s tests -v.
39 distinct Python checks passed: 7 multimodal/API tests (including 127 selection combinations), 7 diagnostic tests and 25 nested semantic regressions (15 top-level unittest cases include the semantic-suite wrapper). All provider outputs are explicit synthetic software fixtures; actual Google SDK request models are validated without billable calls. Tests do not create observed lichen data or validate scientific accuracy.

The existing live Render text path was checked: HTTP200, grounded_llm, four citations, 6.8s. Its health did NOT advertise the new context capability. That verifies the old text backend only. The new package still requires deployment. No live direct-Sites Gemini call was possible because no GEMINI_API_KEY is configured there. Both new paths passed injected-provider transport/privacy tests. Full real-observation and exact mobile-device QA remain pending.
