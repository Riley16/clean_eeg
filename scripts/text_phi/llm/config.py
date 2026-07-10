"""LLM runtime configuration.

Config JSON shape:

    {
      "server_type": "ollama",
      "server_url":  "http://localhost:11434/v1",
      "models": {
        "phi_detector":    "qwen2.5:7b-instruct",
        "record_reviewer": "qwen2.5:7b-instruct"
      },
      "cache_path":       "temp/llm_cache.sqlite",
      "seed":             42,
      "temperature":      0.0,
      "timeout_seconds":  60,
      "max_retries":      3,
      "api_key":          null
    }

The `server_type` value maps to how the client formats the structured-output
request body (`ServerType.OLLAMA` uses top-level `format`; `ServerType.VLLM`
uses `extra_body.guided_json`; others use `response_format`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ServerType(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"
    OPENAI = "openai"
    LM_STUDIO = "lm_studio"


class ConfigError(ValueError):
    """The config JSON is malformed or missing required fields."""


@dataclass(frozen=True)
class LLMConfig:
    server_type: ServerType
    server_url: str
    models: dict[str, str]
    cache_path: Path | None = None
    seed: int = 42
    temperature: float = 0.0
    timeout_seconds: float = 60.0
    max_retries: int = 3
    api_key: str | None = None

    # ---------- construction ----------

    @classmethod
    def load(cls, path: str | Path) -> "LLMConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "LLMConfig":
        for req in ("server_type", "server_url", "models"):
            if req not in raw:
                raise ConfigError(f"llm config missing required key {req!r}")

        try:
            server_type = ServerType(raw["server_type"])
        except ValueError as e:
            raise ConfigError(
                f"server_type must be one of {[s.value for s in ServerType]}; "
                f"got {raw['server_type']!r}"
            ) from e

        if not isinstance(raw["server_url"], str) or not raw["server_url"].strip():
            raise ConfigError("server_url must be a non-empty string")

        models = raw["models"]
        if not isinstance(models, dict) or not models:
            raise ConfigError("models must be a non-empty {hint: model_name} map")
        for k, v in models.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ConfigError(f"models entries must be str→str; got {k!r}: {v!r}")

        cache_path_raw = raw.get("cache_path")
        cache_path = Path(cache_path_raw) if cache_path_raw else None

        return cls(
            server_type=server_type,
            server_url=raw["server_url"].rstrip("/"),
            models=dict(models),
            cache_path=cache_path,
            seed=int(raw.get("seed", 42)),
            temperature=float(raw.get("temperature", 0.0)),
            timeout_seconds=float(raw.get("timeout_seconds", 60.0)),
            max_retries=int(raw.get("max_retries", 3)),
            api_key=raw.get("api_key"),
        )

    # ---------- accessors ----------

    def resolve_model(self, hint: str) -> str:
        """Map a logical hint like `"phi_detector"` to the concrete model
        name declared in the config. Raises if the hint isn't registered."""
        if hint not in self.models:
            raise ConfigError(
                f"model hint {hint!r} not registered in config. "
                f"Known hints: {sorted(self.models.keys())}"
            )
        return self.models[hint]
