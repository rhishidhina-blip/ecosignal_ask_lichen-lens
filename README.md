# EcoSignal Ask LichenLens — GitHub-ready reliability update

This is a complete backend deployment, preserving the 191 built-in reviewed knowledge records, routing fixes, short retrieval context, cache for successful general educational replies, relaxed citation placement, source-ID verification, and deterministic fallback.

**New in this version:** 30-second configurable Gemini SDK timeout; default one concurrent Gemini call per worker; at most two generation attempts (primary + alternate); bounded exponential backoff with jitter after retryable failures. 429 quota failures do not trigger repetitive attempts of the same model.

See `UPLOAD_INSTRUCTIONS.md` for deployment steps and Render environment settings. The production scan database is not included in this repository package. Paid API billing can raise quotas, but cannot guarantee that 503 or 504 errors will disappear.
