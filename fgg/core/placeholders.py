"""
Smart Tag Protection & Placeholder Preservation Engine.
Protects game tags, variables, key icons, and control codes during translation.
"""

from __future__ import annotations

import re
from typing import List, Tuple


# Default regex patterns for common game localization placeholders
DEFAULT_PATTERNS = [
    r"\{clr:[^}]*\}",          # {clr:red}, {clr:green}
    r"\{clr/\}",               # {clr/}
    r"\{[a-zA-Z0-9_\-.]+\}",   # {player_name}, {0}, {item_id}
    r"\[[a-zA-Z0-9_\-.\s/]+\]", # [J], [M], [G], [KEY_SPACE], [b]text[/b]
    r"%\d*\$?[sdfg]",          # %s, %d, %1$s
    r"\$[a-zA-Z0-9_]+",        # $player_name
    r"\\n|\\t|\\r",            # Escaped newlines and tabs
    r"</?[a-zA-Z0-9]+\b[^>]*>",# HTML tags <b>, <color=#fff>
]


class PlaceholderEngine:
    def __init__(self, custom_patterns: List[str] | None = None) -> None:
        patterns = list(DEFAULT_PATTERNS)
        if custom_patterns:
            patterns.extend(custom_patterns)
        
        # Combine patterns into a single regex
        self.regex = re.compile("|".join(f"({p})" for p in patterns))

    def extract(self, text: str) -> Tuple[str, List[str]]:
        """
        Replaces all detected placeholders with unique token markers.
        Returns (protected_text, list_of_placeholders).
        """
        placeholders: List[str] = []

        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            placeholders.append(token)
            return f"__PH_{len(placeholders) - 1}__"

        protected_text = self.regex.sub(_replace, text)
        return protected_text, placeholders

    def restore(self, text: str, placeholders: List[str]) -> str:
        """
        Restores extracted placeholders back into the translated text,
        handling space insertion by translation engines (e.g. '__ PH _ 0 __').
        """
        restored = text
        for index, placeholder in enumerate(placeholders):
            target_marker = f"__PH_{index}__"
            
            # Direct match
            if target_marker in restored:
                restored = restored.replace(target_marker, placeholder)
                continue
            
            # Fuzzy match for whitespace added by Google Translate or LLM
            fuzzy_pattern = re.compile(
                r"__\s*PH\s*_\s*" + str(index) + r"\s*__", re.IGNORECASE
            )
            restored = fuzzy_pattern.sub(placeholder, restored)

        return restored
