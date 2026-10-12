# Local validation — October 12, 2026

Passed: 15 semantic pipeline checks and 7 diagnostic checks (22 total), executed
through `python -m unittest discover -s tests -v` (one wrapper runs the 15 checks).
Provider responses were explicit test fixtures; no billable calls were made.

Checks cover unfamiliar wording reaching the model, both conversation roles,
resolved retrieval, ambiguity, unrelated topics, unavailable/invalid interpretation,
stored-record requirements, prompt/secret protection, scientific assertions,
citations, cache isolation and reuse, HTTP contracts, audit-chain integrity,
diagnostic privacy, timeout/retry options and tracing both generation stages.

The notebook code cells were executed in order locally, excluding the Colab
installation command. Optional live generation, export and server remained off.
Outputs, execution counts and stale widget state are cleared. The backend core
matches main.py; its embedded ZIP is checked for integrity and matching SHA-256.

Tested on Windows Python 3.12.14 with NumPy 2.2.6, pandas 2.2.3, SciPy 1.15.3,
scikit-learn 1.6.1, Google GenAI 2.29.0, Pydantic 2.14.1 and FastAPI 0.143.0.
requirements.txt retains the earlier scientific pins. The exact Render Linux
Python 3.11.11 environment and Colab Drive were not executed here.

The keyword topic router and phrase-based context matcher are removed. Retrieval
ranking and targeted safety checks do not decide whether wording is accepted.
There is no public diagnostic endpoint. Live interpretation quality, provider
quota and latency remain to be checked with the existing private key. Optional
live examples are included and off by default. No keys or raw provider exceptions
are saved in outputs.
