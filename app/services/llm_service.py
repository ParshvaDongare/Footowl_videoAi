from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.settings import SETTINGS
from app.utils import ensure_dir

try:
    from google import genai
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - optional dependency
    genai = None
    genai_types = None


T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMService:
    api_key: str | None = None
    base_url: str | None = None
    provider: str | None = None
    cache_path: Path | None = None

    def __post_init__(self) -> None:
        self.provider = (self.provider or SETTINGS.llm_provider or os.getenv("LLM_PROVIDER") or "").strip().lower()
        self.cache_path = ensure_dir(Path(self.cache_path or SETTINGS.llm_cache_path))
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        self.base_url = self.base_url or os.getenv("OPENAI_BASE_URL") or None
        self.call_budget = int(os.getenv("GEMINI_CALL_BUDGET", str(SETTINGS.gemini_call_budget)))
        self.call_count = 0
        self._depleted = False

        if self.provider == "openai" or (not self.provider and os.getenv("OPENAI_API_KEY")):
            self.provider = "openai"
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url) if self.api_key else None
        elif self.provider == "gemini" or (not self.provider and os.getenv("GEMINI_API_KEY")):
            self.provider = "gemini"
            self._client = genai.Client(api_key=self.api_key) if self.api_key and genai else None
        else:
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None and not self._depleted

    def generate_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.2,
        image_paths: list[str] | None = None,
        max_output_tokens: int = 1024,
    ) -> T | None:
        if not self.available:
            return None
        cache_key = self._cache_key(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=image_paths or [],
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        cached = self._read_cache(cache_key)
        if cached:
            parsed = self._parse_response(cached, response_model)
            if parsed is not None:
                return parsed

        try:
            if self.call_count >= self.call_budget:
                self._depleted = True
                return None
            if self.provider == "gemini":
                content = self._generate_json_gemini(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=image_paths or [],
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    response_model=response_model,
                )
            else:
                content = self._generate_json_openai(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                )
            if not content:
                return None
            self.call_count += 1
            self._write_cache(cache_key, content)
            return self._parse_response(content, response_model)
        except Exception as exc:  # pragma: no cover - provider/network failures
            if self._looks_like_quota_error(exc):
                self._depleted = True
            return None

    def _generate_json_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
    ) -> str | None:
        if not self._client:
            return None
        response = self._client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"

    def _generate_json_gemini(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float,
        max_output_tokens: int,
        response_model: type[T],
    ) -> str | None:
        if not self._client or not genai_types:
            return None
        contents: list[Any] = [user_prompt]
        for image_path in image_paths:
            from PIL import Image

            with Image.open(image_path) as img:
                contents.append(img.convert("RGB").copy())

        config = genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=response_model.model_json_schema(),
        )
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        text = getattr(response, "text", None) or getattr(response, "output_text", None) or ""
        return text or "{}"

    def _parse_response(self, content: str, response_model: type[T]) -> T | None:
        try:
            return response_model.model_validate_json(content)
        except ValidationError:
            try:
                data = json.loads(content)
                return response_model.model_validate(data)
            except Exception:
                return None

    def _cache_key(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        payload = {
            "provider": self.provider,
            "model": model,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_fingerprint": [self._fingerprint_image(path) for path in image_paths],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return digest

    def _fingerprint_image(self, path: str) -> dict[str, Any]:
        file = Path(path)
        stat = file.stat()
        return {
            "path": str(file),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    def _cache_file(self, key: str) -> Path:
        return self.cache_path / f"{key}.json"

    def _read_cache(self, key: str) -> str | None:
        path = self._cache_file(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write_cache(self, key: str, content: str) -> None:
        path = self._cache_file(key)
        path.write_text(content, encoding="utf-8")

    def _looks_like_quota_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return "resource_exhausted" in text or "429" in text or "quota" in text or "rate limit" in text
