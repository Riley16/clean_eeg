"""Jinja rendering for LLM prompts + per-key context values.

Two entry points:

* `render_string(template_str, variables)` — render a single template.
* `render_context(context, variables)` — render each *value* in a
  {key: template_str} dict against the same variables; non-string values
  pass through unchanged.

`StrictUndefined` is used — a template referencing a missing variable
raises `jinja2.UndefinedError` rather than silently producing an empty
string. That catches config errors early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2


def _env() -> jinja2.Environment:
    return jinja2.Environment(
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
    )


def render_string(template: str, variables: dict[str, Any]) -> str:
    """Render `template` as a Jinja source string against `variables`."""
    return _env().from_string(template).render(**variables)


def render_context(
    context: dict[str, Any], variables: dict[str, Any]
) -> dict[str, Any]:
    """Render each string value in `context` as a Jinja template. Non-string
    values pass through untouched.

    Preserves iteration order (dict insertion order) so downstream hashing
    is deterministic."""
    out: dict[str, Any] = {}
    for k, v in context.items():
        out[k] = render_string(v, variables) if isinstance(v, str) else v
    return out


def load_prompt(path: str | Path) -> str:
    """Read a prompt template file from disk verbatim."""
    return Path(path).read_text(encoding="utf-8")


class PromptRegistry:
    """Look up prompt templates by short name.

    A prompt name (e.g. `"generic_phi"`) resolves to the file
    `{root}/{name}.jinja`. Templates are cached in memory after first
    read. An absolute or relative path is passed through unchanged, so
    schemas can reference custom prompts that live outside the registry
    root.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: dict[str, str] = {}

    def get(self, name_or_path: str) -> str:
        """Load a prompt by short name or by path. Cached after first read."""
        if name_or_path in self._cache:
            return self._cache[name_or_path]
        as_path = Path(name_or_path)
        if as_path.is_absolute() or as_path.suffix == ".jinja":
            path = as_path
        else:
            path = self.root / f"{name_or_path}.jinja"
        text = path.read_text(encoding="utf-8")
        self._cache[name_or_path] = text
        return text

    def has(self, name: str) -> bool:
        try:
            self.get(name)
            return True
        except FileNotFoundError:
            return False
