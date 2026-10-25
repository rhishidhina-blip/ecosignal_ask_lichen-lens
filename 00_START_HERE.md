# EcoSignal chatbot: text and observation data only

This ZIP is an update package, not a deployed change. Source baseline: EcoSignal v28, commit 9ac604128d4f018dcaf9ea20660ac373642fe547. It preserves that version’s text-chat implementation; it is not a verified byte-for-byte restoration of a separate older chatbot checkpoint.

## Apply to the existing Render chatbot repository
1. Back up your current repository or create a branch.
2. Copy the CONTENTS of Render_Backend/ into the root of your existing Ask LichenLens repository. Replace matching files, including tests/test_multimodal.py. Do not upload the Sites_Patch folder into the Python backend.
3. Keep the existing GEMINI_API_KEY secret. Set the environment entries below. Existing Render environment values override Python defaults; replacing render.yaml alone does not update a manually configured service.
4. Commit and deploy the new commit on your existing Render service.
5. Start command remains: uvicorn main:api --host 0.0.0.0 --port $PORT
6. Health route remains /assistant/health. Expect image_context_supported=false and data_context_supported=true.

Required Render settings:
- ECOSIGNAL_CHAT_MODEL=gemini-3.1-flash-lite
- ECOSIGNAL_CHAT_FALLBACK_MODELS= (remove this entry or leave empty)
- ECOSIGNAL_GEMINI_MAX_ATTEMPTS=2
- ECOSIGNAL_GEMINI_BACKOFF_SECONDS=8
- ECOSIGNAL_GEMINI_BUDGET_SECONDS=40

No API key is included. Keep the same public chatbot endpoint and the same Site runtime configuration.

## Apply the Sites patch
Sites_Patch/ contains only the changed frontend/server files and the replacement chat transport regression test. Merge those files into the CURRENT EcoSignal source, preserving relative paths. Build and publish that Site version with its existing audience. A Render deploy alone does not remove the photograph checkbox from the website.

The Site source bundle still needs its original package.json, assets, and unchanged modules. Sites_Patch is an overlay, not a standalone app. Its app/lib/ files belong to the Site, not the Render repository.

For a Sites chat, attach this ZIP and use HANDOFF_TO_SITES_CHAT.md. No live Site or Render service has been changed by creation of this ZIP.

## Behavior retained
- Normal questions, natural follow-ups, last three completed question/answer pairs and conversation IDs.
- The same 192 reviewed knowledge chunks, local lexical retrieval, source citations and scientific guards.
- Existing sample/mission context lookup, suggested questions, caching, audit handling, rate limits and review actions.
- Selected coverage and growth form results, satellite indices, terrain susceptibility, elevation and optional coordinates, including source/date/confidence fields.
- Explicit field selection, exact context receipts, and a fresh conversation when shared data changes.
- Existing retry error behavior; it does not silently fabricate a successful answer when Gemini fails.

## Changes
Chat has no photograph option or preview. The chat panel no longer prepares an image. The Site proxy and Python API reject supplied photographs, even if other data is selected. Direct Gemini and Render requests contain text only. The backend advertises no image support. The retry delay defaults to eight seconds before one further attempt using the same configured model.

LichenLens photo analysis, saved photographs, coverage masks, image models, research workspace, maps, AQI, satellite, unified score and planner are outside this patch and remain in the existing app.

## Important limit
Removing image support does not prove or resolve the observed Gemini provider 503 failures. Those failures occurred for ordinary text questions too. The eight-second backoff may help transient overload but cannot guarantee a successful provider response. Live Gemini, Render and authenticated browser testing must follow deployment.

## Validation
See VALIDATION.md. Tests use synthetic provider responses and make no live model calls.
