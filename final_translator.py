"""
Reliable localization translator for FGG.

The script keeps the localization file structure, protects game placeholders
like {clr:red}, can resume from existing translated files, and does not treat
failed API calls as successful translations.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from deep_translator import GoogleTranslator


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "оригинал" / "rus.txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "переводы"

LANGUAGES: Dict[str, Tuple[str, str]] = {
    "en": ("en", "ENGLISH"),
    "de": ("de", "DEUTSCH"),
    "fr": ("fr", "FRANCAIS"),
    "es": ("es", "ESPANOL"),
    "it": ("it", "ITALIANO"),
    "pt": ("pt", "PORTUGUES"),
    "pl": ("pl", "POLSKI"),
    "tr": ("tr", "TURKCE"),
    "ja": ("ja", "日本語"),
    "ko": ("ko", "한국어"),
    "zh-cn": ("zh-CN", "简体中文"),
    "be": ("be", "БЕЛАРУСКАЯ"),
    "bg": ("bg", "БЪЛГАРСКИ"),
    "kk": ("kk", "ҚАЗАҚША"),
    "ky": ("ky", "КЫРГЫЗЧА"),
    "mk": ("mk", "МАКЕДОНСКИ"),
    "mn": ("mn", "МОНГОЛ"),
    "sr": ("sr", "СРПСКИ"),
    "tg": ("tg", "ТОҶИКӢ"),
    "tt": ("tt", "ТАТАРЧА"),
    "uk": ("uk", "УКРАЇНСЬКА"),
}


ASSIGNMENT_RE = re.compile(r'^(\s*([^#\s][^=\s]*)\s*=\s*)"(.*)"(\s*)$')
PLACEHOLDER_RE = re.compile(r"\{[^}]*\}")


@dataclass(frozen=True)
class Entry:
    line_index: int
    key: str
    prefix: str
    value: str
    suffix: str


@dataclass
class TranslationStats:
    translated: int = 0
    reused: int = 0
    skipped: int = 0
    failed: int = 0


class FinalTranslator:
    def __init__(
        self,
        source_lang: str = "ru",
        max_retries: int = 4,
        retry_delay: float = 4.0,
        request_delay: float = 1.5,
        batch_size: int = 1,
        workers: int = 1,
    ) -> None:
        self.source_lang = source_lang
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self.batch_size = max(1, batch_size)
        self.workers = max(1, workers)

    def extract_placeholders(self, text: str) -> Tuple[str, List[str]]:
        placeholders: List[str] = []

        def replace_with_marker(match: re.Match[str]) -> str:
            placeholders.append(match.group(0))
            return f"__PH_{len(placeholders) - 1}__"

        return PLACEHOLDER_RE.sub(replace_with_marker, text), placeholders

    def restore_placeholders(self, text: str, placeholders: Sequence[str]) -> str:
        for index, placeholder in enumerate(placeholders):
            text = text.replace(f"__PH_{index}__", placeholder)
            text = text.replace(f"__ PH _ {index} __", placeholder)
        return text

    def parse_entries(self, lines: Sequence[str]) -> List[Entry]:
        entries: List[Entry] = []
        for index, line in enumerate(lines):
            match = ASSIGNMENT_RE.match(line.rstrip("\n"))
            if not match:
                continue

            prefix, key, value, suffix = match.groups()
            if key in {"lang", "lang_name"} or not value.strip():
                continue

            entries.append(Entry(index, key, prefix, value, suffix))

        return entries

    def read_existing_values(self, output_file: Path) -> Dict[str, str]:
        if not output_file.exists():
            return {}

        existing: Dict[str, str] = {}
        for line in output_file.read_text(encoding="utf-8").splitlines():
            match = ASSIGNMENT_RE.match(line)
            if match:
                _, key, value, _ = match.groups()
                existing[key] = value

        return existing

    def has_real_translation(self, key: str, source_value: str, existing: Dict[str, str]) -> bool:
        translated = existing.get(key)
        return bool(translated and translated.strip() and translated != source_value)

    def make_batches(self, items: Sequence[Tuple[Entry, str]]) -> Iterable[List[Tuple[Entry, str]]]:
        for index in range(0, len(items), self.batch_size):
            yield list(items[index : index + self.batch_size])

    def translate_one(self, translator: GoogleTranslator, text: str) -> Optional[str]:
        for attempt in range(1, self.max_retries + 1):
            try:
                result = translator.translate(text)
                if isinstance(result, str) and result.strip():
                    return result
            except Exception as exc:
                print(f"    [RETRY] {attempt}/{self.max_retries}: {str(exc)[:120]}")

            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        return None

    def translate_batch(
        self,
        translator: GoogleTranslator,
        batch: Sequence[Tuple[Entry, str]],
    ) -> List[Optional[str]]:
        results: List[Optional[str]] = []
        for _, text in batch:
            results.append(self.translate_one(translator, text))
            time.sleep(self.request_delay)
        return results

    def translate_entry(
        self,
        target_lang: str,
        item: Tuple[Entry, str],
    ) -> Tuple[Entry, Optional[str]]:
        entry, text = item
        translator = GoogleTranslator(source=self.source_lang, target=target_lang)
        translated = self.translate_one(translator, text)
        if self.request_delay:
            time.sleep(self.request_delay)
        return entry, translated

    def translate_file(
        self,
        input_file: Path,
        output_file: Path,
        target_lang: str,
        lang_code: str,
        lang_name: str,
        force: bool = False,
    ) -> bool:
        print(f"\n{'=' * 70}")
        print(f">>> Перевод на {lang_name} ({lang_code})")
        print(f"{'=' * 70}")

        source_lines = input_file.read_text(encoding="utf-8").splitlines(keepends=True)
        result_lines = list(source_lines)
        entries = self.parse_entries(source_lines)
        existing = self.read_existing_values(output_file)
        stats = TranslationStats()
        failed_keys: List[str] = []

        for index, line in enumerate(result_lines):
            stripped = line.strip()
            if stripped.startswith("lang ="):
                result_lines[index] = f'lang = "{lang_code}"\n'
            elif stripped.startswith("lang_name ="):
                result_lines[index] = f'lang_name = "{lang_name}"\n'

        work_items: List[Tuple[Entry, str]] = []
        placeholder_map: Dict[str, List[str]] = {}

        for entry in entries:
            if not force and self.has_real_translation(entry.key, entry.value, existing):
                result_lines[entry.line_index] = f'{entry.prefix}"{existing[entry.key]}"{entry.suffix}\n'
                stats.reused += 1
                continue

            protected_text, placeholders = self.extract_placeholders(entry.value)
            placeholder_map[entry.key] = placeholders
            work_items.append((entry, protected_text))

        total = len(work_items)
        done = 0
        completed_keys = set()

        if self.workers == 1:
            translator = GoogleTranslator(source=self.source_lang, target=target_lang)
            for batch in self.make_batches(work_items):
                translated_batch = self.translate_batch(translator, batch)
                for (entry, _), translated in zip(batch, translated_batch):
                    done += 1
                    if translated is None:
                        failed_keys.append(entry.key)
                        stats.failed += 1
                        result_lines[entry.line_index] = (
                            f'{entry.prefix}"{entry.value}"{entry.suffix}\n'
                        )
                        continue

                    translated = self.restore_placeholders(translated, placeholder_map[entry.key])
                    result_lines[entry.line_index] = f'{entry.prefix}"{translated}"{entry.suffix}\n'
                    stats.translated += 1

                print(
                    f"  [PROGRESS] {done}/{total} новых, "
                    f"{stats.reused} взято из готового файла, ошибок: {stats.failed}"
                )
                time.sleep(self.request_delay)
        else:
            print(f"  [INFO] Быстрый режим: workers={self.workers}")
            executor = ThreadPoolExecutor(max_workers=self.workers)
            pending = set()
            work_iter = iter(work_items)
            interrupted = False

            def submit_more() -> None:
                while len(pending) < self.workers * 2:
                    try:
                        item = next(work_iter)
                    except StopIteration:
                        return
                    pending.add(executor.submit(self.translate_entry, target_lang, item))

            submit_more()
            try:
                while pending:
                    for future in as_completed(pending):
                        pending.remove(future)
                        break

                    entry, translated = future.result()
                    completed_keys.add(entry.key)
                    done += 1
                    if translated is None:
                        failed_keys.append(entry.key)
                        stats.failed += 1
                        result_lines[entry.line_index] = (
                            f'{entry.prefix}"{entry.value}"{entry.suffix}\n'
                        )
                    else:
                        translated = self.restore_placeholders(
                            translated, placeholder_map[entry.key]
                        )
                        result_lines[entry.line_index] = (
                            f'{entry.prefix}"{translated}"{entry.suffix}\n'
                        )
                        stats.translated += 1

                    if done % 20 == 0 or done == total:
                        print(
                            f"  [PROGRESS] {done}/{total} новых, "
                            f"{stats.reused} взято из готового файла, ошибок: {stats.failed}"
                        )
                    submit_more()
            except KeyboardInterrupt:
                interrupted = True
                print("\n  [STOP] Остановлено пользователем, сохраняю готовые строки...")
                for future in pending:
                    future.cancel()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)

            if interrupted:
                failed_keys.extend(
                    entry.key
                    for entry, _ in work_items
                    if entry.key not in completed_keys and entry.key not in failed_keys
                )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text("".join(result_lines), encoding="utf-8")

        failed_file = output_file.with_suffix(".failed.txt")
        if failed_keys:
            failed_file.write_text("\n".join(failed_keys) + "\n", encoding="utf-8")
            print(f"  [WARN] Не переведено ключей: {len(failed_keys)} ({failed_file.name})")
        elif failed_file.exists():
            failed_file.unlink()

        print(f"  [DONE] Новых переводов: {stats.translated}")
        print(f"  [DONE] Использовано готовых: {stats.reused}")
        print(f"  [SAVED] {output_file}")
        return stats.failed == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate FGG localization files.")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=list(LANGUAGES.keys()),
        help="Language keys to translate, for example: --langs de en uk",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Retranslate existing values")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-delay", type=float, default=4.0)
    parser.add_argument("--request-delay", type=float, default=1.5)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel translation workers. Try 3-5 for speed; lower it if Google returns errors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="How many strings to process before printing progress. Keep 1 for reliability.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_file = args.input
    output_dir = args.output_dir

    if not input_file.exists():
        print(f"[ERROR] Файл не найден: {input_file}")
        return 1

    selected_languages = []
    for lang_key in args.langs:
        if lang_key not in LANGUAGES:
            print(f"[ERROR] Неизвестный язык: {lang_key}")
            print(f"[INFO] Доступные языки: {', '.join(LANGUAGES)}")
            return 1
        selected_languages.append(lang_key)

    translator = FinalTranslator(
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        request_delay=args.request_delay,
        batch_size=args.batch_size,
        workers=args.workers,
    )

    print("\n" + "=" * 70)
    print(">>> АВТОМАТИЧЕСКИЙ ПЕРЕВОДЧИК")
    print("=" * 70)
    print(f"[INPUT] {input_file}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[LANGS] {', '.join(selected_languages)}")
    print(f"[WORKERS] {args.workers}")
    print("=" * 70)

    started_at = time.time()
    failed_languages: List[str] = []

    for lang_key in selected_languages:
        target_lang, lang_name = LANGUAGES[lang_key]
        output_file = output_dir / f"{lang_key}.txt"
        ok = translator.translate_file(
            input_file=input_file,
            output_file=output_file,
            target_lang=target_lang,
            lang_code=lang_key,
            lang_name=lang_name,
            force=args.force,
        )
        if not ok:
            failed_languages.append(lang_key)

    elapsed = int(time.time() - started_at)
    print("\n" + "=" * 70)
    print(">>> ГОТОВО")
    print("=" * 70)
    print(f"[TIME] {elapsed // 60} мин {elapsed % 60} сек")
    if failed_languages:
        print(f"[WARN] Есть неполные языки: {', '.join(failed_languages)}")
        return 2

    print("[SUCCESS] Все выбранные языки переведены без ошибок")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
