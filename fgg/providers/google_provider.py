"""
Google Translate Provider using deep_translator.
"""

from __future__ import annotations

import time
from typing import Optional
from deep_translator import GoogleTranslator
from fgg.providers.base import BaseTranslator


class GoogleProvider(BaseTranslator):
    def translate_single(self, text: str, target_lang: str) -> Optional[str]:
        if not text.strip():
            return text

        translator = GoogleTranslator(source=self.source_lang, target=target_lang)
        for attempt in range(1, self.max_retries + 1):
            try:
                res = translator.translate(text)
                if isinstance(res, str) and res.strip():
                    return res
            except Exception as exc:
                if attempt == self.max_retries:
                    print(f"    [ERR Google] {text[:30]}... -> {exc}")
            
            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        return None
