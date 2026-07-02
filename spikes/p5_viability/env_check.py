# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail fast with a friendly message when a required provider API key is unset,
instead of a deep bmlib/provider stack trace part-way through a paid run."""

from __future__ import annotations

import os

# provider prefix (from a "provider:model" string) -> required env var.
# Local providers (e.g. ollama) need no key and are simply absent here.
_PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY"}


def require_api_keys(models: list[str]) -> None:
    """Raise SystemExit with a clear message if any of ``models`` needs a
    provider API key that isn't set in the environment."""
    missing: dict[str, str] = {}
    for model in models:
        provider = model.split(":", 1)[0]
        env = _PROVIDER_ENV.get(provider)
        if env and not os.environ.get(env):
            missing.setdefault(env, provider)
    if missing:
        details = "; ".join(f"{env} (for {prov}: models)" for env, prov in missing.items())
        first = next(iter(missing))
        raise SystemExit(
            f"Missing required API key(s): {details}.\n"
            f"Set it in your shell, e.g.:  export {first}=sk-...\n"
            "(`uv run` inherits your environment; there is no .env support.)"
        )
