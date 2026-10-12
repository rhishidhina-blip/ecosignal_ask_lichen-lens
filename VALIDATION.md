# Local validation — October 12, 2026

Passed: `python -m unittest discover -s tests -v` — 11 tests.

Verified with Python 3.12.14 on Windows, Google GenAI 2.29.0, and the package's
requirements (including its pinned NumPy, pandas, SciPy and scikit-learn).
The existing Render Blueprint still specifies Python 3.11.11; that exact Linux
deployment environment was not executed locally.

Checks cover disabled/missing-key behavior without provider calls, fixed minimal
prompt and output cap, SDK retry/timeout options, client cleanup, sanitized
outcomes for auth/quota/provider/transport failures, empty responses, stdout /
stderr / log suppression, nonblocking diagnostic failure, the real backend's
structured plan and validator, cache bypass, optional full application retries,
no diagnostic audit event, unchanged HTTP route behavior, and no diagnostic
route. Provider generation responses were faked; no billable calls were made.

`main.py` and `requirements.txt` are byte-for-byte identical to
`EcoSignal_Ask_LichenLens_Answer_Quality_GitHub.zip`. All original files remain.
The archive is checked for integrity, expected files, excluded runtime data and
cache files, and recognizable Google API key literals.

The disabled diagnostic command also ran successfully and reported `disabled`.
Live Render networking, secret permissions, model access, duration and quota
remain to be tested using the instructions in `RENDER_DIAGNOSTICS.md`.
