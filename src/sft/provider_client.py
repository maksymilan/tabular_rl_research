"""Shared configuration helpers for external teacher and audit clients."""

from __future__ import annotations

from pathlib import Path
import json
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_api_config(path: Path | None = None) -> tuple[str, str]:
    """Load the local API key and base URL without logging either value."""
    config_path = path or PROJECT_ROOT / "api.md"
    api_key = ""
    base_url = ""
    with config_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            elif line.startswith("BASE_URL="):
                base_url = line.split("=", 1)[1].strip()
    return api_key, base_url


def request_official_deepseek_json(
    payload: dict, *, timeout: int = 180, config_path: Path | None = None
) -> dict:
    """One explicit official request, no retries, redirects or provider fallback.

    Return the untouched JSON response for caller-owned validation/auditing.
    Unlike the archived generation client this has no protocol/tool dependencies.
    Never include credentials, request headers or response bodies in exceptions.
    """
    api_key, base_url = load_api_config(config_path)
    if not api_key or base_url.rstrip('/') != 'https://api.deepseek.com':
        raise ValueError('official DeepSeek credentials/endpoint required')
    if not str(payload.get('model', '')).startswith('deepseek-'):
        raise ValueError('explicit DeepSeek model required')

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    request = urllib.request.Request(
        'https://api.deepseek.com/chat/completions',
        data=json.dumps(payload, ensure_ascii=False).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + api_key},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'official DeepSeek HTTP {exc.code}; no retry or fallback') from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError('official DeepSeek transport failure; no retry or fallback') from None
    except (ValueError, UnicodeDecodeError):
        raise RuntimeError('official DeepSeek returned invalid JSON') from None
    if not isinstance(result, dict):
        raise RuntimeError('official DeepSeek response must be a JSON object')
    return result
