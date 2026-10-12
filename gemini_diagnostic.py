"""Explicit, one-shot Render diagnostic. Never imported by the web application."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import json
import logging
import os
import re
import time
from types import SimpleNamespace

MODEL = 'gemini-3.1-flash-lite'
QUESTION = 'What do lichens primarily bioindicate about air quality?'


def duration(started):
    return round(time.monotonic() - started, 3)


def safe_model(value):
    return value if isinstance(value, str) and re.fullmatch(r'gemini-[a-z0-9.-]{1,70}', value) else None


def safe_error(exc):
    # Never stringify exceptions, request URLs, response bodies or headers.
    code = getattr(exc, 'code', None)
    code = code if type(code) is int and 100 <= code <= 599 else None
    if code in (401, 403):
        category = 'authentication_or_permission'
    elif code == 429:
        category = 'quota_or_rate_limit'
    elif code == 404:
        category = 'model_or_endpoint_not_found'
    elif code and code >= 500:
        category = 'provider_unavailable'
    elif code == 400:
        category = 'invalid_request'
    elif isinstance(exc, TimeoutError):
        category = 'timeout'
    else:
        import httpx
        category = 'timeout' if isinstance(exc, httpx.TimeoutException) else (
            'transport_error' if isinstance(exc, httpx.TransportError) else 'internal_or_sdk_error')
    return {'outcome': category, 'http_status': code}


@contextmanager
def private_output():
    # The existing backend can print local knowledge-file parse errors. Suppress
    # all imported/backend/SDK output; publish only our allowlisted JSON report.
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, 'w') as sink, redirect_stdout(sink), redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous)


def new_client(key, timeout_ms):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, vertexai=False, http_options=types.HttpOptions(
        timeout=timeout_ms, retry_options=types.HttpRetryOptions(attempts=1)))


def close_client(client):
    try:
        client.close()
    except Exception:
        pass


def minimal_probe(key):
    started = time.monotonic()
    client = None
    result = {'model': MODEL, 'timeout_ms': 15000, 'sdk_attempt_limit': 1}
    try:
        client = new_client(key, 15000)
        response = client.models.generate_content(
            model=MODEL, contents='Reply with only OK.',
            config={'temperature': 0, 'max_output_tokens': 64})
        text = response.text or ''
        result.update(outcome='success' if text.strip() else 'empty_response',
                      text_present=bool(text.strip()), expected_text_match=text.strip() == 'OK')
    except Exception as exc:
        result.update(safe_error(exc))
    finally:
        if client is not None:
            close_client(client)
    result['duration_s'] = duration(started)
    return result


class TracedClient:
    """Record provider calls without changing retrieval, planning or validation."""
    def __init__(self, client):
        self.client = client
        self.attempts = []
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **kwargs):
        started = time.monotonic()
        item = {'model': safe_model(kwargs.get('model'))}
        try:
            response = self.client.models.generate_content(**kwargs)
            item.update(outcome='success' if response.text else 'empty_response')
            return response
        except Exception as exc:
            item.update(safe_error(exc))
            raise
        finally:
            item['duration_s'] = duration(started)
            self.attempts.append(item)


def full_probe(key):
    started = time.monotonic()
    result = {'provider_attempts': [], 'cache_bypassed': True, 'audit_written': False}
    client = None
    # Only this short-lived process is affected. Avoid any embedding calls while
    # importing the backend. Production's next process keeps its own settings.
    embeddings = os.environ.get('ECOSIGNAL_ENABLE_EMBEDDINGS')
    os.environ['ECOSIGNAL_ENABLE_EMBEDDINGS'] = '0'
    try:
        import main
        timeout_ms = max(15000, min(60000, int(os.getenv('ECOSIGNAL_GEMINI_TIMEOUT_MS', '30000'))))
        client = new_client(key, timeout_ms)
        traced = TracedClient(client)
        # Standalone process starts with a fresh cache; clear it explicitly too.
        main.assistant._answer_cache.clear()
        main.assistant.client = traced
        result.update(primary_model=safe_model(main.SETTINGS.chat_model), timeout_ms=timeout_ms,
                      sdk_attempt_limit=1, retrieval='lexical')
        call_started = time.monotonic()
        try:
            reply = main.assistant.ask(main.AskRequest(question=QUESTION,
                user_id='render-gemini-diagnostic'), audit=False)
            flags = set(reply.safety_flags)
            result.update(mode=reply.mode, selected_model=safe_model(reply.model),
                outcome='grounded_llm' if reply.mode == 'grounded_llm' else (
                    'generation_unavailable' if 'LLM_GENERATION_UNAVAILABLE' in flags else 'deterministic_fallback'),
                source_count=len(reply.sources),
                generation_unavailable='LLM_GENERATION_UNAVAILABLE' in flags)
        finally:
            result['request_duration_s'] = duration(call_started)
            result['provider_attempts'] = traced.attempts
    except Exception as exc:
        result.update(safe_error(exc))
    finally:
        if embeddings is None:
            os.environ.pop('ECOSIGNAL_ENABLE_EMBEDDINGS', None)
        else:
            os.environ['ECOSIGNAL_ENABLE_EMBEDDINGS'] = embeddings
        if client is not None:
            close_client(client)
    result['duration_s'] = duration(started)
    return result


def run(include_full=False):
    report = {'diagnostic_version': 1, 'minimal': {'outcome': 'skipped'},
              'full': {'outcome': 'not_requested'}}
    if os.getenv('ECOSIGNAL_ENABLE_GEMINI_DIAGNOSTICS') != '1':
        report['outcome'] = 'disabled'
        return report
    key = os.getenv('GEMINI_API_KEY', '').strip()
    if not key:
        report['outcome'] = 'missing_api_key'
        return report
    report['outcome'] = 'completed'
    with private_output():
        report['minimal'] = minimal_probe(key)
        if include_full:
            report['full'] = full_probe(key)
    return report


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Also run the fixed EcoSignal question once.')
    args = parser.parse_args()
    try:
        report = run(args.full)
    except Exception:
        # A diagnostic must never prevent normal service startup or leak details.
        report = {'diagnostic_version': 1, 'outcome': 'diagnostic_internal_error'}
    print('ECOSIGNAL_GEMINI_DIAGNOSTIC ' + json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
