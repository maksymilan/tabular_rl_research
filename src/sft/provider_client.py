"""Shared configuration helpers for external teacher and audit clients."""

from __future__ import annotations

from pathlib import Path


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
