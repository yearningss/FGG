"""
FGG Advanced Command Line Interface (CLI) 2.0.
Features ASCII Art header, rich terminal progress bars, QA summaries, and AI Multi-Model support.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fgg.core.engine import TranslationEngine
from fgg.providers.google_provider import GoogleProvider
from fgg.providers.mock_provider import MockProvider
from fgg.providers.ai_provider import UnifiedAITranslator
from fgg.web.app import start_web_server

ASCII_BANNER = r"""
  FGG  --  Fast Global Translator
  v2.0 -- High Performance Game Localization Engine (AI Multi-Model Enabled)
"""

LANGUAGES_BY_CATEGORY: Dict[str, Dict[str, Tuple[str, str]]] = {
    "Латиница (Европа & Мир)": {
        "en": ("en", "ENGLISH"),
        "de": ("de", "DEUTSCH"),
        "fr": ("fr", "FRANCAIS"),
        "es": ("es", "ESPANOL"),
        "it": ("it", "ITALIANO"),
        "pt": ("pt", "PORTUGUES"),
        "pl": ("pl", "POLSKI"),
        "tr": ("tr", "TURKCE"),
        "nl": ("nl", "NEDERLANDS"),
        "sv": ("sv", "SVENSKA"),
        "cs": ("cs", "CESTINA"),
        "hu": ("hu", "MAGYAR"),
        "ro": ("ro", "ROMANA"),
        "da": ("da", "DANSK"),
        "no": ("no", "NORSK"),
        "fi": ("fi", "SUOMI"),
    },
    "Иероглифы & Азия": {
        "ja": ("ja", "JAPANESE"),
        "zh-cn": ("zh-CN", "SIMPLIFIED CHINESE"),
        "zh-tw": ("zh-TW", "TRADITIONAL CHINESE"),
        "ko": ("ko", "KOREAN"),
        "th": ("th", "THAI"),
        "vi": ("vi", "VIETNAMESE"),
        "hi": ("hi", "HINDI"),
    },
    "Кириллица": {
        "uk": ("uk", "UKRAINIAN"),
        "be": ("be", "BELARUSIAN"),
        "bg": ("bg", "BULGARIAN"),
        "kk": ("kk", "KAZAKH"),
        "ky": ("ky", "KYRGYZ"),
        "sr": ("sr", "SERBIAN"),
        "mn": ("mn", "MONGOLIAN"),
        "tg": ("tg", "TAJIK"),
        "tt": ("tt", "TATAR"),
    },
    "Арабское письмо": {
        "ar": ("ar", "ARABIC"),
        "fa": ("fa", "PERSIAN"),
        "he": ("he", "HEBREW"),
    },
}

LANGUAGES: Dict[str, Tuple[str, str]] = {}
for cat_langs in LANGUAGES_BY_CATEGORY.values():
    LANGUAGES.update(cat_langs)


def run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="FGG 2.0 -- High Performance Game Localization Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python final_translator.py --langs de en ja --provider ai --ai-model deepseek-v3",
    )

    parser.add_argument("--langs", nargs="+", default=list(LANGUAGES.keys()), help="Languages to translate")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent.parent / "оригинал" / "rus.txt")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent.parent / "переводы")
    parser.add_argument("--force", action="store_true", help="Force re-translation of existing keys")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel worker threads")
    parser.add_argument("--provider", choices=["google", "ai", "mock"], default="google", help="Translation backend")
    
    # AI CLI arguments
    parser.add_argument("--ai-model", type=str, default="openai-gpt4o-mini", help="AI model key (gpt4o, claude, gemini, deepseek, ollama)")
    parser.add_argument("--api-key", type=str, default="", help="AI provider API key")
    parser.add_argument("--base-url", type=str, default="", help="Custom base URL endpoint")
    parser.add_argument("--ai-prompt", type=str, default="", help="Custom AI localization prompt")

    parser.add_argument("--web", action="store_true", help="Launch FGG Web UI Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Web UI port")

    args = parser.parse_args()

    print(ASCII_BANNER)

    if args.web:
        start_web_server(port=args.port, open_browser=True)
        return 0

    if not args.input.exists():
        print(f"[ERROR] Source file not found: {args.input}")
        return 1

    # Select Provider
    if args.provider == "ai":
        def _cli_ai_log(msg: str):
            print(f"  {msg}")

        provider = UnifiedAITranslator(
            model_key=args.ai_model,
            api_key=args.api_key,
            base_url=args.base_url,
            custom_prompt=args.ai_prompt,
            log_callback=_cli_ai_log,
        )
    elif args.provider == "mock":
        provider = MockProvider()
    else:
        provider = GoogleProvider()

    engine = TranslationEngine(provider=provider, workers=args.workers)

    print(f"[*] Input File : {args.input}")
    print(f"[*] Output Dir : {args.output_dir}")
    print(f"[*] Threads    : {args.workers}")
    print(f"[*] Provider   : {args.provider.upper()} ({args.ai_model if args.provider == 'ai' else ''})")
    print("=" * 65)

    failed_langs = []

    for lang in args.langs:
        if lang not in LANGUAGES:
            print(f"[WARN] Unknown language code: {lang}")
            continue

        target_code, lang_name = LANGUAGES[lang]
        out_file = args.output_dir / f"{lang}.txt"

        print(f"\n[>] Translating to {lang_name} ({lang})...")

        def _on_progress(current: int, total: int, msg: str):
            pct = int((current / total) * 100) if total > 0 else 100
            bar = "#" * (pct // 5) + "-" * (20 - (pct // 5))
            sys.stdout.write(f"\r  [{bar}] {pct}% ({current}/{total})")
            sys.stdout.flush()

        stats, qa = engine.process_file(
            input_file=args.input,
            output_file=out_file,
            target_lang=target_code,
            lang_name=lang_name,
            force=args.force,
            progress_callback=_on_progress,
        )

        sys.stdout.write("\n")
        print(f"  [DONE] Translated: {stats.translated} | Reused: {stats.reused_existing + stats.reused_tm} | Failed: {stats.failed} ({stats.elapsed_seconds}s)")
        
        if qa.issues:
            print(f"  [QA] Warnings: {len(qa.issues)} potential tag mismatches detected")

        if stats.failed > 0:
            failed_langs.append(lang)

    print("\n" + "=" * 65)
    print("[SUCCESS] ALL TRANSLATION TASKS COMPLETED!")
    print("=" * 65)

    return 0 if not failed_langs else 2


if __name__ == "__main__":
    sys.exit(run_cli())
