# EcoSignal Ask LichenLens — Answer Quality Update

This is the complete GitHub-ready backend based on the context-aware validator deployment.

New improvements:
- The Gemini system instructions require a direct answer before uncertainty and methodological caveats.
- Bioindicator questions prioritize the reviewed NPS lichen-bioindication knowledge record in lexical retrieval, rather than leading with regulatory caveats.
- The backend normalizes Markdown-wrapped citation IDs to clean `[SOURCE-ID]` markers and continues checking that citations correspond to retrieved knowledge.
- Existing token limits, response caching, retries/backoff, concurrency control, scientific validation and observation-specific routing are retained.

Deployment: extract into the existing GitHub repository root, replace matching files and commit to main. Keep the same Render API key and environment variables. Do not upload the ZIP itself to the repository.

Website: source link rendering is controlled by the frontend. The backend now avoids emitting Markdown citation hyperlinks inside answer text, while returning `sources` separately. If the site's source rendering inserts links itself, frontend styling may still need a separate update.

After deployment, test the exact question: "What do lichens primarily bioindicate about air quality?" and a multi-part bioindicator question. Verify `grounded_llm`, accuracy and source relevance. This is not a guarantee against Gemini 503/504 failures.
