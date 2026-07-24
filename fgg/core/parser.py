"""
Multi-format Localization File Parser.
Supports key = "value" text files, JSON, CSV/TSV, and INI formats.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

ASSIGNMENT_RE = re.compile(r'^(\s*([^#\s][^=\s]*)\s*=\s*)"(.*)"(\s*)$')


@dataclass
class LocalizationEntry:
    id: str                  # Unique key identifier
    raw_key: str             # Original key name
    original_value: str      # Source text string
    translated_value: str    # Translated text string (if available)
    line_index: int = -1     # Line index in source file
    prefix: str = ""         # Prefix for reconstruction (key = ")
    suffix: str = ""         # Suffix for reconstruction (")


class BaseParser:
    def parse(self, file_path: Path) -> Tuple[List[LocalizationEntry], List[str]]:
        raise NotImplementedError

    def export(
        self,
        file_path: Path,
        entries: List[LocalizationEntry],
        original_lines: List[str],
        lang_code: str,
        lang_name: str,
    ) -> None:
        raise NotImplementedError


class KeyValueParser(BaseParser):
    """Parses standard game localization files (key = "value")."""

    def parse(self, file_path: Path) -> Tuple[List[LocalizationEntry], List[str]]:
        raw_content = file_path.read_text(encoding="utf-8")
        lines = raw_content.splitlines(keepends=True)
        entries: List[LocalizationEntry] = []

        for idx, line in enumerate(lines):
            match = ASSIGNMENT_RE.match(line.rstrip("\r\n"))
            if not match:
                continue

            prefix, key, value, suffix = match.groups()
            # Skip system language header keys
            if key in {"lang", "lang_name"} or not value.strip():
                continue

            entries.append(
                LocalizationEntry(
                    id=key,
                    raw_key=key,
                    original_value=value,
                    translated_value="",
                    line_index=idx,
                    prefix=prefix,
                    suffix=suffix,
                )
            )

        return entries, lines

    def export(
        self,
        file_path: Path,
        entries: List[LocalizationEntry],
        original_lines: List[str],
        lang_code: str,
        lang_name: str,
    ) -> None:
        result_lines = list(original_lines)

        # Update language metadata headers if present
        for idx, line in enumerate(result_lines):
            stripped = line.strip()
            if stripped.startswith("lang ="):
                result_lines[idx] = f'lang = "{lang_code}"\n'
            elif stripped.startswith("lang_name ="):
                result_lines[idx] = f'lang_name = "{lang_name}"\n'

        entry_map = {e.id: e for e in entries}

        for idx, line in enumerate(result_lines):
            match = ASSIGNMENT_RE.match(line.rstrip("\r\n"))
            if not match:
                continue
            _, key, _, _ = match.groups()
            if key in entry_map and entry_map[key].translated_value:
                entry = entry_map[key]
                val = entry.translated_value
                result_lines[idx] = f'{entry.prefix}"{val}"{entry.suffix}\n'

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("".join(result_lines), encoding="utf-8")


class JsonParser(BaseParser):
    """Parses JSON localization files."""

    def parse(self, file_path: Path) -> Tuple[List[LocalizationEntry], List[str]]:
        raw_text = file_path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        entries: List[LocalizationEntry] = []

        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    entries.append(
                        LocalizationEntry(id=k, raw_key=k, original_value=v, translated_value="")
                    )

        return entries, [raw_text]

    def export(
        self,
        file_path: Path,
        entries: List[LocalizationEntry],
        original_lines: List[str],
        lang_code: str,
        lang_name: str,
    ) -> None:
        result_dict = {}
        for entry in entries:
            result_dict[entry.raw_key] = entry.translated_value or entry.original_value

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2), encoding="utf-8")


def get_parser_for_file(file_path: Path) -> BaseParser:
    """Detects appropriate parser by file extension."""
    if file_path.suffix.lower() == ".json":
        return JsonParser()
    return KeyValueParser()
