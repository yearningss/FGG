"""
Unified Multi-Model AI Translation Engine for FGG 2.0.
Supports OpenAI, Anthropic, Gemini, DeepSeek, Mistral, Llama, Qwen, and Ollama/vLLM/OpenRouter.
Includes granular fine-tuning parameters (temperature, top_p, penalties, max_tokens, reasoning_effort)
and real-time AI reasoning & stream logs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, List, Optional, Dict
import urllib.request
import urllib.error

from fgg.providers.base import BaseTranslator

DEFAULT_GAME_PROMPT = """You are an expert video game localization translator.
Translate game text strings from {source_lang} to {target_lang}.
STRICT GAME LOCALIZATION RULES:
1. Preserve all placeholder tokens like __PH_0__, __PH_1__, {clr:red}, [J], %s, %d, {0} EXACTLY as they appear.
2. Keep game terminology natural, punchy, and appropriate for the game genre.
3. Return ONLY a valid JSON list of translated strings matching the exact length and order of the input array.
Example Input: ["Привет, [J]! {clr:red}Опасность{clr/}"]
Example Output: ["Hello, [J]! {clr:red}Danger{clr/}"]
"""

# Extensive AI Model Catalog
AI_MODELS: Dict[str, Dict[str, str]] = {
    # OPENAI
    "openai-gpt4o": {"name": "OpenAI GPT-4o", "provider": "openai", "model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
    "openai-gpt4o-mini": {"name": "OpenAI GPT-4o mini", "provider": "openai", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "openai-o3-mini": {"name": "OpenAI o3-mini (Reasoning)", "provider": "openai", "model": "o3-mini", "base_url": "https://api.openai.com/v1"},
    "openai-o1-mini": {"name": "OpenAI o1-mini", "provider": "openai", "model": "o1-mini", "base_url": "https://api.openai.com/v1"},
    "openai-gpt4-turbo": {"name": "OpenAI GPT-4 Turbo", "provider": "openai", "model": "gpt-4-turbo", "base_url": "https://api.openai.com/v1"},

    # ANTHROPIC
    "anthropic-claude-35-sonnet": {"name": "Anthropic Claude 3.5 Sonnet", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "base_url": "https://api.anthropic.com/v1"},
    "anthropic-claude-35-haiku": {"name": "Anthropic Claude 3.5 Haiku", "provider": "anthropic", "model": "claude-3-5-haiku-20241022", "base_url": "https://api.anthropic.com/v1"},
    "anthropic-claude-3-opus": {"name": "Anthropic Claude 3 Opus", "provider": "anthropic", "model": "claude-3-opus-20240229", "base_url": "https://api.anthropic.com/v1"},
    "anthropic-claude-3-haiku": {"name": "Anthropic Claude 3 Haiku", "provider": "anthropic", "model": "claude-3-haiku-20240307", "base_url": "https://api.anthropic.com/v1"},

    # GOOGLE GEMINI
    "gemini-20-flash-exp": {"name": "Google Gemini 2.0 Flash (Experimental)", "provider": "gemini", "model": "gemini-2.0-flash-exp", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-20-thinking": {"name": "Google Gemini 2.0 Thinking (Reasoning)", "provider": "gemini", "model": "gemini-2.0-flash-thinking-exp-1219", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-15-pro": {"name": "Google Gemini 1.5 Pro", "provider": "gemini", "model": "gemini-1.5-pro", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-15-flash": {"name": "Google Gemini 1.5 Flash", "provider": "gemini", "model": "gemini-1.5-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta"},

    # DEEPSEEK
    "deepseek-v3": {"name": "DeepSeek V3", "provider": "deepseek", "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"},
    "deepseek-r1": {"name": "DeepSeek R1 (Reasoning)", "provider": "deepseek", "model": "deepseek-reasoner", "base_url": "https://api.deepseek.com/v1"},

    # MISTRAL
    "mistral-large": {"name": "Mistral Large 2", "provider": "openai", "model": "mistral-large-latest", "base_url": "https://api.mistral.ai/v1"},
    "mistral-small": {"name": "Mistral Small", "provider": "openai", "model": "mistral-small-latest", "base_url": "https://api.mistral.ai/v1"},

    # META LLAMA & QWEN (via OpenRouter or Ollama)
    "meta-llama-33-70b": {"name": "Meta Llama 3.3 70B", "provider": "openai", "model": "meta-llama/llama-3.3-70b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen-25-72b": {"name": "Qwen 2.5 72B Instruct", "provider": "openai", "model": "qwen/qwen-2.5-72b-instruct", "base_url": "https://openrouter.ai/api/v1"},

    # LOCAL / CUSTOM
    "local-ollama": {"name": "Local Ollama / vLLM (localhost:11434)", "provider": "openai", "model": "llama3.2", "base_url": "http://localhost:11434/v1"},
    "openrouter": {"name": "OpenRouter (All AI Models)", "provider": "openai", "model": "auto", "base_url": "https://openrouter.ai/api/v1"},
    "custom": {"name": "Custom Model (User-defined Endpoint)", "provider": "openai", "model": "custom", "base_url": "http://localhost:11434/v1"},
}


class UnifiedAITranslator(BaseTranslator):
    def __init__(
        self,
        model_key: str = "openai-gpt4o-mini",
        api_key: str = "",
        base_url: str = "",
        custom_model_name: str = "",
        custom_prompt: str = "",
        temperature: float = 0.2,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "medium",
        source_lang: str = "ru",
        max_retries: int = 3,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(source_lang=source_lang, max_retries=max_retries)
        self.preset = AI_MODELS.get(model_key, AI_MODELS["openai-gpt4o-mini"])
        self.provider_type = self.preset["provider"]
        self.model = custom_model_name or self.preset["model"]
        self.base_url = (base_url or self.preset["base_url"]).rstrip("/")
        self.api_key = api_key or os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.custom_prompt = custom_prompt

        # Granular tuning parameters
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.frequency_penalty = float(frequency_penalty)
        self.presence_penalty = float(presence_penalty)
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        self.log_callback = log_callback

    def _log(self, message: str) -> None:
        if self.log_callback:
            self.log_callback(message)

    def translate_batch(self, texts: List[str], target_lang: str) -> List[Optional[str]]:
        if not texts:
            return []

        prompt_template = self.custom_prompt or DEFAULT_GAME_PROMPT
        sys_prompt = prompt_template.format(
            source_lang=self.source_lang, target_lang=target_lang
        )

        self._log(
            f"🤖 [AI Query] Model: {self.model} | Lang: '{target_lang}' | "
            f"Temp: {self.temperature} | TopP: {self.top_p} | MaxTokens: {self.max_tokens}"
        )

        if self.provider_type == "anthropic":
            return self._call_anthropic(sys_prompt, texts)
        elif self.provider_type == "gemini":
            return self._call_gemini(sys_prompt, texts)
        else:
            return self._call_openai_compatible(sys_prompt, texts)

    def _call_openai_compatible(self, sys_prompt: str, texts: List[str]) -> List[Optional[str]]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(texts, ensure_ascii=False)},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        # Add penalties if supported
        if self.frequency_penalty != 0.0:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            payload["presence_penalty"] = self.presence_penalty

        # Reasoning effort for o1/o3-mini/R1 models if applicable
        if "o1" in self.model or "o3" in self.model or "reasoner" in self.model:
            payload["reasoning_effort"] = self.reasoning_effort

        # JSON mode
        if "ollama" not in self.base_url and "reasoner" not in self.model and "o1" not in self.model:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key or 'ollama'}",
        }

        url = f"{self.base_url}/chat/completions"
        return self._send_http_request(url, payload, headers, texts)

    def _call_anthropic(self, sys_prompt: str, texts: List[str]) -> List[Optional[str]]:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system": sys_prompt,
            "messages": [
                {"role": "user", "content": f"Translate this JSON array to target language:\n{json.dumps(texts, ensure_ascii=False)}"}
            ],
        }

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        url = f"{self.base_url}/messages"
        return self._send_http_request(url, payload, headers, texts, is_anthropic=True)

    def _call_gemini(self, sys_prompt: str, texts: List[str]) -> List[Optional[str]]:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"{sys_prompt}\n\nJSON array to translate:\n{json.dumps(texts, ensure_ascii=False)}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": self.top_p,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        headers = {"Content-Type": "application/json"}
        return self._send_http_request(url, payload, headers, texts, is_gemini=True)

    def _send_http_request(
        self,
        url: str,
        payload: dict,
        headers: dict,
        original_texts: List[str],
        is_anthropic: bool = False,
        is_gemini: bool = False,
    ) -> List[Optional[str]]:
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers)

        for attempt in range(1, self.max_retries + 1):
            try:
                start_t = time.time()
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw_response = resp.read().decode("utf-8")
                    duration = round(time.time() - start_t, 2)
                    res_json = json.loads(raw_response)

                    raw_text = ""
                    if is_anthropic:
                        raw_text = res_json["content"][0]["text"]
                    elif is_gemini:
                        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    else:
                        msg = res_json["choices"][0]["message"]
                        if "reasoning_content" in msg and msg["reasoning_content"]:
                            self._log(f"🧠 [DeepSeek R1 Reasoning] {msg['reasoning_content'][:300]}...")
                        raw_text = msg["content"]

                    self._log(f"⚡ [AI Output ({duration}s)] snippet: {raw_text[:200]}...")

                    parsed_array = self._parse_json_result(raw_text)
                    if parsed_array and len(parsed_array) == len(original_texts):
                        return parsed_array
                    elif parsed_array:
                        self._log(f"⚠️ [AI Warning] Length mismatch! Expected {len(original_texts)}, got {len(parsed_array)}.")
                        return parsed_array[:len(original_texts)] + [None] * (len(original_texts) - len(parsed_array))

            except Exception as exc:
                self._log(f"❌ [AI Retry {attempt}/{self.max_retries}] Exception: {str(exc)[:150]}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)

        return [None] * len(original_texts)

    def _parse_json_result(self, raw_text: str) -> Optional[List[str]]:
        try:
            data = json.loads(raw_text.strip())
            if isinstance(data, list):
                return [str(x) for x in data]
            elif isinstance(data, dict):
                for val in data.values():
                    if isinstance(val, list):
                        return [str(x) for x in val]
        except Exception:
            pass

        import re
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return [str(x) for x in data]
            except Exception:
                pass
        return None

    def translate_single(self, text: str, target_lang: str) -> Optional[str]:
        res = self.translate_batch([text], target_lang)
        return res[0] if res else None
