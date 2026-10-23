import contextlib
import gc
import importlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gemini_diagnostic as diag

SECRET = 'PRIVATE_SENTINEL_NEVER_PUBLISH'


class ProviderError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(SECRET)


class FakeClient:
    def __init__(self, text='OK', error=None):
        self.text, self.error, self.calls = text, error, []
        self.models = SimpleNamespace(generate_content=self.generate_content)
        self.closed = False

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        print(SECRET)
        print(SECRET, file=sys.stderr)
        logging.error(SECRET)
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)

    def close(self):
        self.closed = True


class DiagnosticTests(unittest.TestCase):
    def test_disabled_and_missing_key_never_create_client(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(diag, 'new_client') as factory:
            self.assertEqual(diag.run(True)['outcome'], 'disabled')
            os.environ['ECOSIGNAL_ENABLE_GEMINI_DIAGNOSTICS'] = '1'
            self.assertEqual(diag.run(True)['outcome'], 'missing_api_key')
            factory.assert_not_called()

    def test_minimal_single_fixed_call_private_output_and_close(self):
        client = FakeClient(SECRET)
        output = io.StringIO()
        with patch.dict(os.environ, {'ECOSIGNAL_ENABLE_GEMINI_DIAGNOSTICS': '1', 'GEMINI_API_KEY': SECRET}), \
             patch.object(diag, 'new_client', return_value=client) as factory, \
             contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            report = diag.run()
        factory.assert_called_once_with(SECRET, 15000)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]['model'], 'gemini-3.1-flash-lite')
        self.assertEqual(client.calls[0]['contents'], 'Reply with only OK.')
        self.assertEqual(client.calls[0]['config']['max_output_tokens'], 64)
        self.assertEqual(report['minimal']['outcome'], 'success')
        self.assertFalse(report['minimal']['expected_text_match'])
        self.assertGreaterEqual(report['minimal']['duration_s'], 0)
        self.assertNotIn(SECRET, json.dumps(report) + output.getvalue())
        self.assertEqual(report['full']['outcome'], 'not_requested')
        self.assertTrue(client.closed)

    def test_error_classification_and_redaction(self):
        import httpx
        cases = [(ProviderError(403), 'authentication_or_permission'),
                 (ProviderError(429), 'quota_or_rate_limit'),
                 (ProviderError(404), 'model_or_endpoint_not_found'),
                 (ProviderError(400), 'invalid_request'),
                 (ProviderError(503), 'provider_unavailable'),
                 (httpx.ReadTimeout(SECRET), 'timeout'),
                 (httpx.ConnectError(SECRET), 'transport_error'),
                 (ValueError(SECRET), 'internal_or_sdk_error')]
        for error, expected in cases:
            with self.subTest(expected=expected), patch.object(diag, 'new_client', return_value=FakeClient(error=error)), diag.private_output():
                result = diag.minimal_probe(SECRET)
            self.assertEqual(result['outcome'], expected)
            self.assertNotIn(SECRET, json.dumps(result))
        self.assertIsNone(diag.safe_error(ProviderError(SECRET))['http_status'])

    def test_empty_response(self):
        with patch.object(diag, 'new_client', return_value=FakeClient(text='')), diag.private_output():
            self.assertEqual(diag.minimal_probe(SECRET)['outcome'], 'empty_response')

    def test_sdk_options_validate_without_network(self):
        # Validate against the installed Google SDK, not a mocked schema.
        client = diag.new_client(SECRET, 15000)
        self.assertEqual(client._api_client._http_options.timeout, 15000)
        self.assertEqual(client._api_client._http_options.retry_options.attempts, 1)
        diag.close_client(client)

    def test_cli_failure_is_sanitized_and_does_not_block_startup(self):
        output = io.StringIO()
        with patch('sys.argv', ['gemini_diagnostic.py']), patch.object(diag, 'run', side_effect=ValueError(SECRET)), contextlib.redirect_stdout(output):
            self.assertEqual(diag.cli(), 0)
        self.assertNotIn(SECRET, output.getvalue())
        self.assertIn('diagnostic_internal_error', output.getvalue())

    def test_full_diagnostic_reports_single_combined_attempt(self):
        import main
        class FullFake(FakeClient):
            def generate_content(self, **kwargs):
                self.calls.append(kwargs)
                value = json.loads(kwargs['contents'])
                retrieved = value['approved_sources']
                payload = {'interpretation': {'intent': 'lichen_education',
                    'standalone_question': diag.QUESTION, 'requires_stored_observation': False,
                    'is_followup': False},
                    'answer': 'Lichens can indicate environmental conditions.',
                    'uncertainty': 'A photo cannot measure pollutant concentration.',
                    'recommended_action': 'none',
                    'cited_source_ids': [retrieved[0]['source_id']]}
                return SimpleNamespace(text=json.dumps(payload))
        fake = FullFake()
        saved = main.assistant.client
        try:
            with patch.object(diag, 'new_client', return_value=fake), diag.private_output():
                result = diag.full_probe(SECRET)
            self.assertEqual(result['outcome'], 'grounded_llm', result)
            self.assertEqual(len(result['provider_attempts']), 1)
            self.assertNotIn(SECRET, json.dumps(result))
        finally:
            main.assistant.client = saved
            main.assistant._answer_cache.clear()


if __name__ == '__main__':
    unittest.main()
