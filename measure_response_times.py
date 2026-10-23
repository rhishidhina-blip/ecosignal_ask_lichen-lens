"""Measure the deployed ask endpoint; report timings without saving answer text."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True, help='Exact HTTPS /assistant/ask or website /api/ask URL')
    parser.add_argument('--count', type=int, default=5, choices=range(1, 11))
    parser.add_argument('--output', default='live-response-times.json')
    args = parser.parse_args()
    if not args.url.startswith('https://'):
        parser.error('Use your deployed HTTPS endpoint.')
    rows = []
    questions = ['What do lichens bioindicate about air quality?',
                 'What are foliose, fruticose, and crustose lichens?',
                 'What are the limits of interpreting a lichen photo?']
    for i in range(args.count):
        headers = {'Content-Type': 'application/json'}
        token = os.getenv('ECOSIGNAL_BENCHMARK_TOKEN')
        if token:
            headers['Authorization'] = 'Bearer ' + token
        request = urllib.request.Request(args.url, headers=headers,
            data=json.dumps({'question': questions[i % len(questions)],
                             'user_id': 'response-time-check'}).encode())
        started = time.perf_counter()
        try:
            response = urllib.request.urlopen(request, timeout=75)
        except urllib.error.HTTPError as exc:
            response = exc
        except (OSError, TimeoutError):
            rows.append({'request': i+1, 'http_status': None, 'outcome': 'transport_error',
                         'elapsed_ms': round((time.perf_counter()-started)*1000, 3)})
            continue
        with response:
            raw = response.read(65537)
            try:
                body = json.loads(raw) if len(raw) <= 65536 else {}
            except (ValueError, UnicodeError):
                body = {}
            rows.append({'request': i+1, 'http_status': response.status,
                'outcome': body.get('status', 'http_success' if response.status == 200 else 'http_error'),
                'elapsed_ms': round((time.perf_counter()-started)*1000, 3),
                'server_timing': response.headers.get('Server-Timing'),
                'backend_response_time_ms': body.get('response_time_ms')})
        time.sleep(1)
    elapsed = [r['elapsed_ms'] for r in rows]
    report = {'measurement': 'live HTTP including network and response body',
        'cache_note': 'Repeated questions may use the successful-answer cache; cold starts remain included.',
        'requests': rows, 'median_ms': round(statistics.median(elapsed), 3),
        'max_ms': max(elapsed), 'http_200_count': sum(r['http_status'] == 200 for r in rows)}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
