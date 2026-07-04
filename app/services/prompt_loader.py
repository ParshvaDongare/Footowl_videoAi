"""Utility for loading version-controlled prompt templates from app/prompts/*.md."""
from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
_cache: dict[str, str] = {}


def load_prompt(name: str) -> str:
    """Return the text content of ``app/prompts/{name}.md``.

    Results are cached after the first read so repeated calls are free.
    The prompt name should be the filename stem, e.g. ``"intent"`` for
    ``app/prompts/intent.md``.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
    """
    if name in _cache:
        return _cache[name]
    path = _PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    _cache[name] = text
    return text


def clear_cache() -> None:
    """Clear the in-memory prompt cache (useful for testing)."""
    _cache.clear()
