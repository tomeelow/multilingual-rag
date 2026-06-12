"""LLM client: OpenAI (default) or Anthropic, selected by LLM_PROVIDER.

Models are pinned via env (.env.example), never "latest". All calls are
cache-first (src/pipeline/cache.py); streaming replays cached text on a hit
and fills the cache on a miss.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.config import env
from src.pipeline.cache import LLMCache, request_key


def credentials_available(provider: str | None = None) -> bool:
    provider = provider or env("LLM_PROVIDER", "openai")
    if provider == "openai":
        return bool(env("OPENAI_API_KEY") or env("OPENAI_ADMIN_KEY"))
    if provider == "anthropic":
        return bool(env("ANTHROPIC_API_KEY"))
    return False


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    cached: bool = False


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        cache: LLMCache | None = None,
    ) -> None:
        self.provider = provider or env("LLM_PROVIDER", "openai")
        if self.provider == "openai":
            self.model = model or env("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
        elif self.provider == "anthropic":
            self.model = model or env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        else:
            raise ValueError(f"unknown LLM_PROVIDER: {self.provider}")
        self.cache = cache or LLMCache()
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            if self.provider == "openai":
                from openai import OpenAI

                self._client = OpenAI()
            else:
                from anthropic import Anthropic

                self._client = Anthropic()
        return self._client

    def _key(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        return request_key(
            {
                "provider": self.provider,
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

    def complete(
        self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResult:
        key = self._key(messages, max_tokens, temperature)
        if hit := self.cache.get(key):
            return LLMResult(**hit, cached=True)
        result = self._raw_complete(messages, max_tokens, temperature)
        stored = {k: v for k, v in result.__dict__.items() if k != "cached"}
        self.cache.set(key, stored)
        return result

    def stream(
        self, messages: list[dict], max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[str]:
        """Yields text deltas; replays from cache on a hit."""
        key = self._key(messages, max_tokens, temperature)
        if hit := self.cache.get(key):
            text = hit["text"]
            for i in range(0, len(text), 24):  # replay in small chunks
                yield text[i : i + 24]
            return
        parts: list[str] = []
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        for delta in self._raw_stream(messages, max_tokens, temperature, usage):
            parts.append(delta)
            yield delta
        self.cache.set(
            key,
            {"text": "".join(parts), "model": self.model, **usage},
        )

    # --- provider-specific transport ---------------------------------------

    def _raw_complete(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResult:
        logger.debug("LLM call: {} {} ({} messages)", self.provider, self.model, len(messages))
        if self.provider == "openai":
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature
            )
            return LLMResult(
                text=r.choices[0].message.content or "",
                input_tokens=r.usage.prompt_tokens,
                output_tokens=r.usage.completion_tokens,
                model=self.model,
            )
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        r = self.client.messages.create(
            model=self.model,
            system=system or None,
            messages=rest,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return LLMResult(
            text=r.content[0].text,
            input_tokens=r.usage.input_tokens,
            output_tokens=r.usage.output_tokens,
            model=self.model,
        )

    def _raw_stream(
        self, messages: list[dict], max_tokens: int, temperature: float, usage: dict[str, int]
    ) -> Iterator[str]:
        logger.debug("LLM stream: {} {}", self.provider, self.model)
        if self.provider == "openai":
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )
            for event in stream:
                if event.usage:
                    usage["input_tokens"] = event.usage.prompt_tokens
                    usage["output_tokens"] = event.usage.completion_tokens
                if event.choices and event.choices[0].delta.content:
                    yield event.choices[0].delta.content
            return
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        with self.client.messages.stream(
            model=self.model,
            system=system or None,
            messages=rest,
            max_tokens=max_tokens,
            temperature=temperature,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
            usage["input_tokens"] = final.usage.input_tokens
            usage["output_tokens"] = final.usage.output_tokens


def get_llm() -> LLMClient:
    return LLMClient()
