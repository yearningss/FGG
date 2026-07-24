"""
FGG Core Translation Engine.
Coordinates parsers, placeholders, translation memory, multi-threading, and QA checks.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from fgg.core.glossary import GlossaryManager, TranslationMemory
from fgg.core.parser import LocalizationEntry, get_parser_for_file
from fgg.core.placeholders import PlaceholderEngine
from fgg.core.qa import QAReport, QAValidator
from fgg.providers.base import BaseTranslator
from fgg.providers.google_provider import GoogleProvider
from fgg.providers.mock_provider import MockProvider


@dataclass
class EngineStats:
    total_entries: int = 0
    translated: int = 0
    reused_existing: int = 0
    reused_tm: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0


class TranslationEngine:
    def __init__(
        self,
        provider: BaseTranslator | None = None,
        source_lang: str = "ru",
        workers: int = 5,
        request_delay: float = 0.5,
        use_tm: bool = True,
        tm_path: Path | None = None,
        glossary_path: Path | None = None,
    ) -> None:
        self.provider = provider or GoogleProvider(source_lang=source_lang)
        self.source_lang = source_lang
        self.workers = max(1, workers)
        self.request_delay = request_delay
        self.ph_engine = PlaceholderEngine()
        self.qa_validator = QAValidator(self.ph_engine)
        self.glossary = GlossaryManager(glossary_path)
        self.tm = TranslationMemory(tm_path or Path(__file__).parent.parent / ".fgg_cache.sqlite") if use_tm else None

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        target_lang: str,
        lang_name: str = "",
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> Tuple[EngineStats, QAReport]:
        start_time = time.time()
        parser = get_parser_for_file(input_file)
        entries, orig_lines = parser.parse(input_file)
        stats = EngineStats(total_entries=len(entries))

        existing_entries: Dict[str, str] = {}
        if not force and output_file.exists():
            ex_parsed, _ = parser.parse(output_file)
            existing_entries = {e.id: e.original_value for e in ex_parsed if e.original_value}

        work_items: List[Tuple[LocalizationEntry, str, List[str]]] = []

        for entry in entries:
            # 1. Reuse existing translation in file if present
            if not force and entry.id in existing_entries and existing_entries[entry.id] != entry.original_value:
                entry.translated_value = existing_entries[entry.id]
                stats.reused_existing += 1
                continue

            # 2. Check Translation Memory (TM)
            if self.tm and not force:
                cached = self.tm.get(self.source_lang, target_lang, entry.original_value)
                if cached:
                    entry.translated_value = cached
                    stats.reused_tm += 1
                    continue

            # Apply pre-translation glossary rules
            text_to_translate = self.glossary.apply_pre_translation(entry.original_value, target_lang)
            protected_text, placeholders = self.ph_engine.extract(text_to_translate)
            work_items.append((entry, protected_text, placeholders))

        completed_count = stats.reused_existing + stats.reused_tm
        total_items = len(entries)

        if progress_callback:
            progress_callback(completed_count, total_items, f"Started translation for {target_lang}")

        if work_items:
            def _worker_task(item: Tuple[LocalizationEntry, str, List[str]]) -> Tuple[LocalizationEntry, Optional[str], List[str]]:
                entry, prot_text, placeholders = item
                trans = self.provider.translate_single(prot_text, target_lang)
                if self.request_delay > 0:
                    time.sleep(self.request_delay)
                return entry, trans, placeholders

            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = [executor.submit(_worker_task, item) for item in work_items]
                for future in as_completed(futures):
                    entry, trans_text, placeholders = future.result()
                    completed_count += 1

                    if trans_text is None:
                        entry.translated_value = entry.original_value  # Fallback to source
                        stats.failed += 1
                    else:
                        restored = self.ph_engine.restore(trans_text, placeholders)
                        entry.translated_value = restored
                        stats.translated += 1
                        if self.tm:
                            self.tm.set(self.source_lang, target_lang, entry.original_value, restored)

                    if progress_callback:
                        progress_callback(
                            completed_count,
                            total_items,
                            f"Translated key: {entry.raw_key}",
                        )

        # Export completed file
        parser.export(output_file, entries, orig_lines, target_lang, lang_name or target_lang.upper())

        # QA Check
        qa_pairs = [(e.raw_key, e.original_value, e.translated_value) for e in entries]
        qa_report = self.qa_validator.validate(qa_pairs)

        stats.elapsed_seconds = round(time.time() - start_time, 2)
        return stats, qa_report
