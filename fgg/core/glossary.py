"""
Glossary & Translation Memory (TM) Engine.
Manages terminology rules and persistent caching of translated pairs.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple


class GlossaryManager:
    def __init__(self, glossary_file: Path | None = None) -> None:
        self.rules: Dict[str, Dict[str, str]] = {}
        if glossary_file and glossary_file.exists():
            try:
                data = json.loads(glossary_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.rules = data
            except Exception:
                pass

    def apply_pre_translation(self, text: str, target_lang: str) -> str:
        """Apply explicit terminology substitutions if defined for language."""
        lang_rules = self.rules.get(target_lang, {})
        for src_term, target_term in lang_rules.items():
            text = text.replace(src_term, target_term)
        return text


class TranslationMemory:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS translations (
                    source_lang TEXT,
                    target_lang TEXT,
                    source_text TEXT,
                    translated_text TEXT,
                    PRIMARY KEY (source_lang, target_lang, source_text)
                )
                """
            )
            conn.commit()

    def get(self, source_lang: str, target_lang: str, source_text: str) -> Optional[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT translated_text FROM translations WHERE source_lang=? AND target_lang=? AND source_text=?",
                (source_lang, target_lang, source_text),
            )
            row = cursor.fetchone()
            return row[0] if row else None

    def set(self, source_lang: str, target_lang: str, source_text: str, translated_text: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO translations (source_lang, target_lang, source_text, translated_text)
                VALUES (?, ?, ?, ?)
                """,
                (source_lang, target_lang, source_text, translated_text),
            )
            conn.commit()
