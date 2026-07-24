"""
OpenAI & LLM Game Localization Provider.
Uses system prompts customized for game contexts, retaining placeholders, tone, and brevity.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional
import urllib.request
import urllib.error

from fgg.providers.base import BaseTranslator


SYSTEM_PROMPT = """You are an expert game localization translator.
Translate game text strings from {source_lang} to {target_lang}.
STRICT RULES:
1. Preserve all placeholders like __PH_0__, __PH_1__, {clr:red}, [J], %s, etc. EXACTLY as they appear.
2. Keep game terminology natural, punchy, and appropriate for video games.
3. Return ONLY a JSON list of translated strings matching the order of input strings.
"""


class OpenAIProvider(BaseTranslator):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        source_lang: str = "ru",
        max_retries: int = 3,
    ) -> None:
        super().__init__(source_lang=source_lang, max_retries=max_retries)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.base_url = base_url.rstrip("/")

    def translate_batch(self, texts: List[str], target_lang: str) -> List[Optional[str]]:
        if not self.api_key:
            print("  [WARN] OPENAI_API_KEY not set, falling back to None")
            return [None] * len(texts)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(
                        source_lang=self.source_lang, target_lang=target_lang
                    ),
                },
                {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict) and "translations" in parsed:
                        return parsed["translations"]
            except Exception as exc:
                print(f"    [ERR OpenAI] Batch retry {attempt}: {exc}")
                time.sleep(self.retry_delay * attempt)

        return [None] * len(texts)

    def translate_single(self, text: str, target_lang: str) -> Optional[str]:
        res = self.translate_batch([text], target_lang)
        return res[0] if res else None
