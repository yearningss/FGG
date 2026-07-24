"""
Base Translation Provider Interface.
"""

from __future__ import annotations

from typing import List, Optional


class BaseTranslator:
    def __init__(self, source_lang: str = "ru", max_retries: int = 3, retry_delay: float = 2.0) -> None:
        self.source_lang = source_lang
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def translate_single(self, text: str, target_lang: str) -> Optional[str]:
        """Translates a single text string."""
        raise NotImplementedError

    def translate_batch(self, texts: List[str], target_lang: str) -> List[Optional[str]]:
        """Translates a batch of text strings."""
        return [self.translate_single(t, target_lang) for t in texts]
