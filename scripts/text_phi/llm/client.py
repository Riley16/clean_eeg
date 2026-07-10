"""OpenAI-compatible HTTP client for local LLM servers.

Server-type-aware structured output:

* `ServerType.OLLAMA`  — top-level `format: <schema>`
* `ServerType.VLLM`    — `extra_body: {"guided_json": <schema>}`
* `ServerType.OPENAI` / `ServerType.LM_STUDIO` — `response_format:
  {"type": "json_schema", "json_schema": {...}}`

Retries with exponential backoff on 429 / 5xx / timeout, up to
`config.max_retries`. Non-retryable HTTP errors raise `LLMHTTPError`.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from .config import LLMConfig, ServerType


class LLMHTTPError(RuntimeError):
    """The LLM server returned a non-retryable HTTP error."""


class LLMClient:
    """Sync client over the OpenAI-compatible chat-completions endpoint.

    Injecting a custom `httpx.MockTransport` is the recommended way to
    unit-test callers of this client without a running LLM server.
    """

    def __init__(
        self,
        config: LLMConfig,
        transport: httpx.BaseTransport | None = None,
    ):
        self.config = config
        self._client = httpx.Client(
            base_url=config.server_url,
            timeout=config.timeout_seconds,
            transport=transport,
        )

    # ---------- primary API ----------

    def chat(
        self,
        messages: list[dict[str, str]],
        model_hint: str = "phi_detector",
        response_schema: dict[str, Any] | None = None,
        seed: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Send a chat-completions request. Returns the parsed JSON body.

        Structured output: pass a JSON-schema `dict` as `response_schema`
        and the model output will be constrained to match (server permitting).
        """
        model = self.config.resolve_model(model_hint)
        body = self._build_body(
            messages=messages,
            model=model,
            response_schema=response_schema,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        headers = self._build_headers()

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.post(
                    "/chat/completions", json=body, headers=headers
                )
            except httpx.TimeoutException as e:
                last_exc = e
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
            except httpx.RequestError as e:
                # Connection refused / DNS / etc. — likely worth a retry too.
                last_exc = e
                if attempt < self.config.max_retries:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.config.max_retries:
                time.sleep(min(2 ** attempt, 30))
                continue

            raise LLMHTTPError(
                f"{resp.status_code} {resp.reason_phrase}: {resp.text[:500]}"
            )

        # Unreachable in practice — retries either return or raise.
        raise LLMHTTPError(f"exhausted retries; last error: {last_exc!r}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------- body assembly ----------

    def _build_body(
        self,
        messages: list[dict[str, str]],
        model: str,
        response_schema: dict[str, Any] | None,
        seed: int | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "seed": seed if seed is not None else self.config.seed,
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        if response_schema is None:
            return body

        st = self.config.server_type
        if st == ServerType.OLLAMA:
            body["format"] = response_schema
        elif st == ServerType.VLLM:
            body["extra_body"] = {"guided_json": response_schema}
        else:  # OPENAI, LM_STUDIO
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema},
            }
        return body

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers
