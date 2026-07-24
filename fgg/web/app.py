"""
FGG Standalone Web Server & REST API.
Provides an interactive web dashboard for non-technical users and game localizers.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
import urllib.parse

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

WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
PROJECT_ROOT = WEB_DIR.parent.parent


class FGGHTTPRequestHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed = urllib.parse.urlparse(path)
        clean_path = parsed.path
        if clean_path == "/" or clean_path == "/index.html":
            return str(STATIC_DIR / "index.html")
        return str(STATIC_DIR / clean_path.lstrip("/"))

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            payload = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            payload = {}

        if parsed_url.path == "/api/translate":
            input_file = payload.get("input_file") or str(PROJECT_ROOT / "оригинал" / "rus.txt")
            output_dir = payload.get("output_dir") or str(PROJECT_ROOT / "переводы")
            langs = payload.get("langs", ["en"])
            workers = payload.get("workers", 5)
            provider_type = payload.get("provider", "google")
            
            # AI & Fine-Tuning options
            ai_model = payload.get("ai_model", "openai-gpt4o-mini")
            api_key = payload.get("api_key", "")
            base_url = payload.get("base_url", "")
            custom_prompt = payload.get("custom_prompt", "")
            temperature = float(payload.get("temperature", 0.2))
            top_p = float(payload.get("top_p", 1.0))
            max_tokens = int(payload.get("max_tokens", 4096))
            reasoning_effort = payload.get("reasoning_effort", "medium")
            freq_penalty = float(payload.get("frequency_penalty", 0.0))
            pres_penalty = float(payload.get("presence_penalty", 0.0))

            inp_path = Path(input_file)
            out_dir = Path(output_dir)

            if not inp_path.exists():
                self._send_json({"success": False, "error": f"File not found: {inp_path}"}, status=400)
                return

            ai_logs: List[str] = []

            def _ai_log_cb(msg: str):
                ai_logs.append(msg)

            # Select Provider
            if provider_type == "ai":
                provider = UnifiedAITranslator(
                    model_key=ai_model,
                    api_key=api_key,
                    base_url=base_url,
                    custom_prompt=custom_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    frequency_penalty=freq_penalty,
                    presence_penalty=pres_penalty,
                    log_callback=_ai_log_cb,
                )
            elif provider_type == "mock":
                provider = MockProvider()
            else:
                provider = GoogleProvider()

            engine = TranslationEngine(provider=provider, workers=workers)

            results = []
            for lang in langs:
                out_file = out_dir / f"{lang}.txt"
                stats, qa = engine.process_file(
                    input_file=inp_path,
                    output_file=out_file,
                    target_lang=lang,
                    force=payload.get("force", False),
                )
                results.append(
                    {
                        "lang": lang,
                        "translated": stats.translated,
                        "reused": stats.reused_existing + stats.reused_tm,
                        "failed": stats.failed,
                        "elapsed": stats.elapsed_seconds,
                        "qa_issues": len(qa.issues),
                        "output_file": str(out_file),
                    }
                )

            self._send_json({"success": True, "results": results, "ai_logs": ai_logs})
        else:
            self.send_error(404, "Endpoint not found")


def start_web_server(port: int = 8080, open_browser: bool = True) -> None:
    server_address = ("", port)
    httpd = HTTPServer(server_address, FGGHTTPRequestHandler)
    print(f"\n=======================================================")
    print(f"[>] FGG Web Dashboard is running at http://localhost:{port}")
    print(f"=======================================================\n")

    if open_browser:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web Server...")
        httpd.server_close()


if __name__ == "__main__":
    start_web_server()
