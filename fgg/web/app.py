"""
FGG Standalone Web Server & REST API.
Provides an interactive web dashboard for non-technical users and game localizers.
Includes AI Model Ping & Latency Diagnostic Benchmark endpoint (/api/ping_ai).
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
import urllib.parse
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fgg.core.engine import TranslationEngine
from fgg.providers.google_provider import GoogleProvider
from fgg.providers.mock_provider import MockProvider
from fgg.providers.ai_provider import UnifiedAITranslator, AI_MODELS

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

        if parsed_url.path == "/api/ping_ai":
            # Diagnostic AI Ping & Connection Benchmark
            model_key = payload.get("ai_model", "gpt-4o-mini")
            api_key = payload.get("api_key", "")
            base_url = payload.get("base_url", "")

            preset = AI_MODELS.get(model_key, AI_MODELS.get("gpt-4o-mini", AI_MODELS["custom"]))
            actual_model = preset["model"]
            target_url = (base_url or preset["base_url"]).rstrip("/")
            provider_type = preset["provider"]

            start_t = time.time()
            http_code = 0
            ping_ms = 0
            model_confirmed = ""
            error_msg = ""

            try:
                # Prepare a lightweight test request
                if provider_type == "anthropic":
                    req_url = f"{target_url}/messages"
                    data = json.dumps({
                        "model": actual_model,
                        "max_tokens": 5,
                        "messages": [{"role": "user", "content": "ping"}]
                    }).encode("utf-8")
                    headers = {
                        "Content-Type": "application/json",
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01"
                    }
                elif provider_type == "gemini":
                    req_url = f"{target_url}/models/{actual_model}:generateContent?key={api_key}"
                    data = json.dumps({
                        "contents": [{"parts": [{"text": "ping"}]}],
                        "generationConfig": {"maxOutputTokens": 5}
                    }).encode("utf-8")
                    headers = {"Content-Type": "application/json"}
                else:
                    req_url = f"{target_url}/chat/completions"
                    data = json.dumps({
                        "model": actual_model,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 5
                    }).encode("utf-8")
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key or 'ollama'}"
                    }

                req = urllib.request.Request(req_url, data=data, headers=headers)
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=12) as resp:
                    ping_ms = int((time.time() - t0) * 1000)
                    http_code = resp.status
                    res_body = json.loads(resp.read().decode("utf-8"))
                    model_confirmed = res_body.get("model", actual_model)

                self._send_json({
                    "success": True,
                    "model_key": model_key,
                    "model_name": preset["name"],
                    "model_id": model_confirmed,
                    "ping_ms": ping_ms,
                    "status_code": http_code,
                    "endpoint": target_url,
                    "status": "🟢 Модель активна и доступна"
                })
            except urllib.error.HTTPError as he:
                ping_ms = int((time.time() - start_t) * 1000)
                error_body = he.read().decode("utf-8", errors="ignore")[:200]
                self._send_json({
                    "success": False,
                    "model_key": model_key,
                    "ping_ms": ping_ms,
                    "status_code": he.code,
                    "endpoint": target_url,
                    "error": f"HTTP {he.code}: {he.reason}",
                    "details": error_body,
                    "status": "🔴 Ошибка ответа сервера"
                })
            except Exception as exc:
                ping_ms = int((time.time() - start_t) * 1000)
                self._send_json({
                    "success": False,
                    "model_key": model_key,
                    "ping_ms": ping_ms,
                    "status_code": 0,
                    "endpoint": target_url,
                    "error": str(exc),
                    "status": "🔴 Не удалось подключиться к серверу"
                })

        elif parsed_url.path == "/api/translate":
            input_file = payload.get("input_file") or str(PROJECT_ROOT / "оригинал" / "rus.txt")
            output_dir = payload.get("output_dir") or str(PROJECT_ROOT / "переводы")
            langs = payload.get("langs", ["en"])
            workers = payload.get("workers", 5)
            provider_type = payload.get("provider", "google")
            
            ai_model = payload.get("ai_model", "gpt-4o-mini")
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
