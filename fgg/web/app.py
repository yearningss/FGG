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
from fgg.providers.ai_provider import UnifiedAITranslator, AI_MODELS, normalize_base_url

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
            model_key = payload.get("ai_model", "gpt-4o-mini")
            custom_model_name = payload.get("custom_model_name", "")
            api_key = payload.get("api_key", "")
            raw_base_url = payload.get("base_url", "")

            preset = AI_MODELS.get(model_key, AI_MODELS.get("gpt-4o-mini", AI_MODELS["custom"]))
            actual_model = custom_model_name.strip() or preset["model"]
            preset_base_url = preset["base_url"]
            provider_type = preset["provider"]

            is_custom_router = bool(raw_base_url and raw_base_url.rstrip("/") != preset_base_url.rstrip("/"))
            effective_provider = "openai" if is_custom_router else provider_type
            target_url = normalize_base_url(raw_base_url or preset_base_url, is_openai_compatible=(effective_provider == "openai"))

            start_t = time.time()
            ping_ms = 0

            def _try_request(url: str, is_openai: bool, is_anthropic: bool, is_gemini: bool):
                if is_anthropic:
                    req_url = f"{url}/messages"
                    data = json.dumps({"model": actual_model, "max_tokens": 5, "messages": [{"role": "user", "content": "ping"}]}).encode("utf-8")
                    headers = {"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
                elif is_gemini:
                    req_url = f"{url}/models/{actual_model}:generateContent?key={api_key}"
                    data = json.dumps({"contents": [{"parts": [{"text": "ping"}]}], "generationConfig": {"maxOutputTokens": 5}}).encode("utf-8")
                    headers = {"Content-Type": "application/json"}
                else:
                    req_url = f"{url}/chat/completions"
                    data = json.dumps({"model": actual_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 5}).encode("utf-8")
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key or 'ollama'}"}

                req = urllib.request.Request(req_url, data=data, headers=headers)
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=15) as resp:
                    p_ms = int((time.time() - t0) * 1000)
                    r_body = json.loads(resp.read().decode("utf-8"))
                    m_id = r_body.get("model", actual_model)
                    return resp.status, p_ms, m_id, req_url

            try:
                is_a = (effective_provider == "anthropic")
                is_g = (effective_provider == "gemini")
                is_o = (effective_provider == "openai")
                
                status_code, ping_ms, model_id, final_url = _try_request(target_url, is_o, is_a, is_g)
                self._send_json({
                    "success": True,
                    "model_key": model_key,
                    "model_name": preset["name"],
                    "model_id": model_id,
                    "ping_ms": ping_ms,
                    "status_code": status_code,
                    "endpoint": final_url,
                    "status": "🟢 Модель активна и соединение установлены"
                })
            except urllib.error.HTTPError as he:
                if (effective_provider != "openai"):
                    try:
                        fallback_url = normalize_base_url(raw_base_url or preset_base_url, is_openai_compatible=True)
                        status_code, ping_ms, model_id, final_url = _try_request(fallback_url, True, False, False)
                        self._send_json({
                            "success": True,
                            "model_key": model_key,
                            "model_name": preset["name"],
                            "model_id": model_id,
                            "ping_ms": ping_ms,
                            "status_code": status_code,
                            "endpoint": final_url,
                            "status": "🟢 Модель доступна (через OpenAI-совместимый прокси)"
                        })
                        return
                    except Exception:
                        pass

                ping_ms = int((time.time() - start_t) * 1000)
                error_body = he.read().decode("utf-8", errors="ignore")[:200]
                status_desc = "🔴 Ошибка авторизации (401 - Проверьте API Ключ)" if he.code == 401 else f"🔴 Ошибка сервера ({he.code})"
                self._send_json({
                    "success": False,
                    "model_key": model_key,
                    "ping_ms": ping_ms,
                    "status_code": he.code,
                    "endpoint": target_url,
                    "error": f"HTTP {he.code}: {he.reason}",
                    "details": error_body,
                    "status": status_desc
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
            custom_model_name = payload.get("custom_model_name", "")
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
                    custom_model_name=custom_model_name,
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
