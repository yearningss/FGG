"""
Offline Mock Translator Provider for testing and dry runs.
"""

from __future__ import annotations

from typing import Optional
from fgg.providers.base import BaseTranslator


class MockProvider(BaseTranslator):
    def translate_single(self, text: str, target_lang: str) -> Optional[str]:
        if not text.strip():
            return text
        return f"[{target_lang.upper()}] {text}"
