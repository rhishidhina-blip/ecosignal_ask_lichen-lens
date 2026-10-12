# One-time Render Gemini diagnostic

This package preserves the Answer Quality improvements and adds semantic
follow-up interpretation; requirements are unchanged. The diagnostic is a standalone command, disabled by default,
with no HTTP route. It uses only the existing Render `GEMINI_API_KEY`; it never
prints the key, provider text, request headers, raw exception or response body.

## Deploy and run once

1. Extract the ZIP and upload its contents to the existing GitHub repository
   root, replacing matching files. Commit. Do not upload the ZIP itself.
2. In Render > Environment, retain the existing `GEMINI_API_KEY`. Set
   `ECOSIGNAL_CHAT_MODEL=gemini-3.1-flash-lite` and temporarily set
   `ECOSIGNAL_ENABLE_GEMINI_DIAGNOSTICS=1`. No key belongs in GitHub or a command.
   Existing Render services may need this environment change manually; editing
   `render.yaml` alone does not necessarily update an existing service.
3. In Render > Settings, temporarily change Start Command to:

   ```sh
   python gemini_diagnostic.py && uvicorn main:api --host 0.0.0.0 --port $PORT
   ```

   To also compare the full EcoSignal request, use:

   ```sh
   python gemini_diagnostic.py --full && uvicorn main:api --host 0.0.0.0 --port $PORT
   ```

4. Deploy and look in Render Logs for `ECOSIGNAL_GEMINI_DIAGNOSTIC`. The single
   JSON line contains separate durations and outcomes. Diagnostic failures
   return exit code 0 so the web service still starts. Startup is delayed until
   the diagnostic finishes; prefer the Render Shell command below if available.
5. Immediately set `ECOSIGNAL_ENABLE_GEMINI_DIAGNOSTICS=0`, restore the original
   Start Command below, and deploy again:

   ```sh
   uvicorn main:api --host 0.0.0.0 --port $PORT
   ```

The startup command runs once **per process startup**, not once forever. Leaving
it enabled will incur calls again on restarts and new instances. Do not leave it
enabled or add it to health checks, build commands, scheduled jobs or frontend
code. No diagnostic token is required because there is no publicly callable
endpoint. The Blueprint keeps diagnostics disabled and the original start
command by default.

If Render Shell is available, leave Start Command unchanged, temporarily enable
the flag, and manually run `python gemini_diagnostic.py` or
`python gemini_diagnostic.py --full` there. Disable the flag afterward.

## What each test measures

- **Minimal:** exactly one SDK generation attempt against
  `gemini-3.1-flash-lite`, fixed prompt `Reply with only OK.`, output limit 64
  tokens, 15-second HTTP timeout, no retrieval, schema, embeddings, retry or
  fallback model. `success` means nonempty text; `expected_text_match` reports
  whether it was precisely OK without publishing it.
- **Full (optional):** fixed question “What do lichens primarily bioindicate
  about air quality?” through the current backend's `assistant.ask`, including
  retrieval, system policy, structured generation, validation and deterministic
  fallback. It starts with an empty answer cache and does not write an audit
  event. This is a general question with no mission/sample, not an HTTP or
  observation-specific test. Embeddings are disabled only in this process to
  avoid additional calls; this matches the supplied Blueprint's lexical mode.
  It uses your configured primary/fallback models, production timeout,
  application retry/backoff and concurrency settings. SDK automatic retries are
  disabled so each listed provider attempt corresponds to one generation call.
  `request_duration_s` excludes backend initialization; `duration_s` includes it.

Minimal costs one generation call; full may add up to twice the configured application
attempt limit (interpretation plus answer) (clamped by the backend to 1–3, with at most two configured models).
Full normally uses up to four attempts across two stages and a 30-second HTTP timeout per attempt;
startup may take over two minutes. Timeouts are HTTP limits, not a strict total
wall-clock deadline. Calls may still be charged if the client times out.

## Read the results

| Result | Meaning / next check |
| --- | --- |
| `disabled` / `missing_api_key` | No calls made. Check the flag / Render secret. |
| `authentication_or_permission` (401/403) | Check the existing key's project, API permissions and restrictions. |
| `quota_or_rate_limit` (429) | Check Google quota/billing; do not repeatedly redeploy. |
| `model_or_endpoint_not_found` (404) | Check model availability for this key and API. |
| `provider_unavailable` (5xx) | Provider rejected this attempt; compare timings and retry later manually. |
| `timeout` / `transport_error` | The Render-side request timed out or transport failed. |
| Minimal success + full `grounded_llm` | Both tested paths worked at that time. |
| Minimal success + full `generation_unavailable` / `interpretation_unavailable` | Inspect `provider_attempts`: full schema/prompt/model calls failed, returned empty text, or failed parsing. Provider success plus generation failure suggests parsing or postprocessing. |
| Minimal success + full `deterministic_fallback` | Generation did not yield an accepted grounded answer; provider success suggests validation or routing. |
| `internal_or_sdk_error` / `diagnostic_internal_error` | Local initialization/SDK error; raw details are intentionally suppressed. |

`http_status` is an allowlisted integer or null. A successful provider response
can still fail schema parsing or scientific validation. No single result proves
the cause of a prior failure or guarantees future availability.

Google's documented model code and generate-content examples:
https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite

## Local verification (no key, no billable calls)

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python gemini_diagnostic.py
```

Tests use fake provider responses and a temporary data directory. Live Render
connectivity and quota can only be checked after deploying with your secret.
