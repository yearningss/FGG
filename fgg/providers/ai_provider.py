"""
Unified Multi-Model AI Translation Engine for FGG 2.0.
Supports all major AI model providers, reasoning engines, local LLMs, and API routers.
Includes auto-routing for custom endpoints like agentrouter.org, OpenRouter, and Ollama.
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

def normalize_base_url(url: str, is_openai_compatible: bool = True) -> str:
    url = url.rstrip("/")
    if is_openai_compatible and not url.endswith("/v1") and not url.endswith("/v1beta") and not url.endswith("/v2"):
        if "anthropic.com" not in url and "generativelanguage.googleapis.com" not in url:
            url = f"{url}/v1"
    return url

# Ultimate AI Model Catalog
AI_MODELS: Dict[str, Dict[str, str]] = {
    # --- OpenAI ---
    "gpt-5": {"name": "GPT-5", "provider": "openai", "model": "gpt-5", "base_url": "https://api.openai.com/v1"},
    "gpt-5-thinking": {"name": "GPT-5 Thinking", "provider": "openai", "model": "gpt-5-thinking", "base_url": "https://api.openai.com/v1"},
    "gpt-5-thinking-mini": {"name": "GPT-5 Thinking Mini", "provider": "openai", "model": "gpt-5-thinking-mini", "base_url": "https://api.openai.com/v1"},
    "gpt-5-mini": {"name": "GPT-5 Mini", "provider": "openai", "model": "gpt-5-mini", "base_url": "https://api.openai.com/v1"},
    "gpt-5-nano": {"name": "GPT-5 Nano", "provider": "openai", "model": "gpt-5-nano", "base_url": "https://api.openai.com/v1"},
    "gpt-4.1": {"name": "GPT-4.1", "provider": "openai", "model": "gpt-4.1", "base_url": "https://api.openai.com/v1"},
    "gpt-4.1-mini": {"name": "GPT-4.1 Mini", "provider": "openai", "model": "gpt-4.1-mini", "base_url": "https://api.openai.com/v1"},
    "gpt-4.1-nano": {"name": "GPT-4.1 Nano", "provider": "openai", "model": "gpt-4.1-nano", "base_url": "https://api.openai.com/v1"},
    "gpt-4o": {"name": "GPT-4o", "provider": "openai", "model": "gpt-4o", "base_url": "https://api.openai.com/v1"},
    "gpt-4o-mini": {"name": "GPT-4o Mini", "provider": "openai", "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "gpt-4-turbo": {"name": "GPT-4 Turbo", "provider": "openai", "model": "gpt-4-turbo", "base_url": "https://api.openai.com/v1"},
    "gpt-4": {"name": "GPT-4", "provider": "openai", "model": "gpt-4", "base_url": "https://api.openai.com/v1"},
    "gpt-4-32k": {"name": "GPT-4 32K", "provider": "openai", "model": "gpt-4-32k", "base_url": "https://api.openai.com/v1"},
    "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "provider": "openai", "model": "gpt-3.5-turbo", "base_url": "https://api.openai.com/v1"},
    "o1": {"name": "o1", "provider": "openai", "model": "o1", "base_url": "https://api.openai.com/v1"},
    "o1-pro": {"name": "o1 Pro", "provider": "openai", "model": "o1-pro", "base_url": "https://api.openai.com/v1"},
    "o1-mini": {"name": "o1 Mini", "provider": "openai", "model": "o1-mini", "base_url": "https://api.openai.com/v1"},
    "o3": {"name": "o3", "provider": "openai", "model": "o3", "base_url": "https://api.openai.com/v1"},
    "o3-pro": {"name": "o3 Pro", "provider": "openai", "model": "o3-pro", "base_url": "https://api.openai.com/v1"},
    "o3-mini": {"name": "o3 Mini", "provider": "openai", "model": "o3-mini", "base_url": "https://api.openai.com/v1"},
    "o4-mini": {"name": "o4 Mini", "provider": "openai", "model": "o4-mini", "base_url": "https://api.openai.com/v1"},

    # --- Anthropic Claude ---
    "claude-opus-4-8": {"name": "Claude Opus 4.8", "provider": "anthropic", "model": "claude-opus-4.8", "base_url": "https://api.anthropic.com/v1"},
    "claude-opus-4.8": {"name": "Claude Opus 4.8", "provider": "anthropic", "model": "claude-opus-4.8", "base_url": "https://api.anthropic.com/v1"},
    "claude-sonnet-4.8": {"name": "Claude Sonnet 4.8", "provider": "anthropic", "model": "claude-sonnet-4.8", "base_url": "https://api.anthropic.com/v1"},
    "claude-haiku-4.8": {"name": "Claude Haiku 4.8", "provider": "anthropic", "model": "claude-haiku-4.8", "base_url": "https://api.anthropic.com/v1"},
    "claude-opus-4.1": {"name": "Claude Opus 4.1", "provider": "anthropic", "model": "claude-opus-4.1", "base_url": "https://api.anthropic.com/v1"},
    "claude-opus-4": {"name": "Claude Opus 4", "provider": "anthropic", "model": "claude-opus-4", "base_url": "https://api.anthropic.com/v1"},
    "claude-sonnet-4": {"name": "Claude Sonnet 4", "provider": "anthropic", "model": "claude-sonnet-4", "base_url": "https://api.anthropic.com/v1"},
    "claude-3.7-sonnet": {"name": "Claude 3.7 Sonnet", "provider": "anthropic", "model": "claude-3-7-sonnet", "base_url": "https://api.anthropic.com/v1"},
    "claude-3.5-sonnet": {"name": "Claude 3.5 Sonnet", "provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "base_url": "https://api.anthropic.com/v1"},
    "claude-3.5-haiku": {"name": "Claude 3.5 Haiku", "provider": "anthropic", "model": "claude-3-5-haiku-20241022", "base_url": "https://api.anthropic.com/v1"},
    "claude-3-opus": {"name": "Claude 3 Opus", "provider": "anthropic", "model": "claude-3-opus-20240229", "base_url": "https://api.anthropic.com/v1"},
    "claude-3-sonnet": {"name": "Claude 3 Sonnet", "provider": "anthropic", "model": "claude-3-sonnet-20240229", "base_url": "https://api.anthropic.com/v1"},
    "claude-3-haiku": {"name": "Claude 3 Haiku", "provider": "anthropic", "model": "claude-3-haiku-20240307", "base_url": "https://api.anthropic.com/v1"},

    # --- Google Gemini ---
    "gemini-2.5-pro": {"name": "Gemini 2.5 Pro", "provider": "gemini", "model": "gemini-2.5-pro", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "provider": "gemini", "model": "gemini-2.5-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.5-flash-lite": {"name": "Gemini 2.5 Flash Lite", "provider": "gemini", "model": "gemini-2.5-flash-lite", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.0-flash": {"name": "Gemini 2.0 Flash", "provider": "gemini", "model": "gemini-2.0-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.0-flash-lite": {"name": "Gemini 2.0 Flash Lite", "provider": "gemini", "model": "gemini-2.0-flash-lite", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.0-flash-preview": {"name": "Gemini 2.0 Flash Preview", "provider": "gemini", "model": "gemini-2.0-flash-preview", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.0-flash-experimental": {"name": "Gemini 2.0 Flash Experimental", "provider": "gemini", "model": "gemini-2.0-flash-exp", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.0-thinking": {"name": "Gemini 2.0 Thinking", "provider": "gemini", "model": "gemini-2.0-flash-thinking", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-2.0-thinking-experimental": {"name": "Gemini 2.0 Thinking Experimental", "provider": "gemini", "model": "gemini-2.0-flash-thinking-exp-1219", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-1.5-pro": {"name": "Gemini 1.5 Pro", "provider": "gemini", "model": "gemini-1.5-pro", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-1.5-flash": {"name": "Gemini 1.5 Flash", "provider": "gemini", "model": "gemini-1.5-flash", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemini-1.5-flash-8b": {"name": "Gemini 1.5 Flash-8B", "provider": "gemini", "model": "gemini-1.5-flash-8b", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemma-3": {"name": "Gemma 3", "provider": "gemini", "model": "gemma-3", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "gemma-3n": {"name": "Gemma 3n", "provider": "gemini", "model": "gemma-3n", "base_url": "https://generativelanguage.googleapis.com/v1beta"},

    # --- DeepSeek ---
    "deepseek-v3": {"name": "DeepSeek V3", "provider": "deepseek", "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"},
    "deepseek-v3.1": {"name": "DeepSeek V3.1", "provider": "deepseek", "model": "deepseek-v3.1", "base_url": "https://api.deepseek.com/v1"},
    "deepseek-r1": {"name": "DeepSeek R1", "provider": "deepseek", "model": "deepseek-reasoner", "base_url": "https://api.deepseek.com/v1"},
    "deepseek-r1-0528": {"name": "DeepSeek R1-0528", "provider": "deepseek", "model": "deepseek-r1-0528", "base_url": "https://api.deepseek.com/v1"},
    "deepseek-coder-v2": {"name": "DeepSeek Coder V2", "provider": "deepseek", "model": "deepseek-coder", "base_url": "https://api.deepseek.com/v1"},
    "deepseek-janus-pro": {"name": "DeepSeek Janus Pro", "provider": "deepseek", "model": "deepseek-janus-pro", "base_url": "https://api.deepseek.com/v1"},

    # --- Meta Llama ---
    "llama-4-maverick": {"name": "Llama 4 Maverick", "provider": "openai", "model": "meta-llama/llama-4-maverick", "base_url": "https://openrouter.ai/api/v1"},
    "llama-4-scout": {"name": "Llama 4 Scout", "provider": "openai", "model": "meta-llama/llama-4-scout", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.3-70b-instruct": {"name": "Llama 3.3 70B Instruct", "provider": "openai", "model": "meta-llama/llama-3.3-70b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.2-90b-vision": {"name": "Llama 3.2 90B Vision", "provider": "openai", "model": "meta-llama/llama-3.2-90b-vision-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.2-11b-vision": {"name": "Llama 3.2 11B Vision", "provider": "openai", "model": "meta-llama/llama-3.2-11b-vision-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.2-3b": {"name": "Llama 3.2 3B", "provider": "openai", "model": "meta-llama/llama-3.2-3b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.2-1b": {"name": "Llama 3.2 1B", "provider": "openai", "model": "meta-llama/llama-3.2-1b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.1-405b-instruct": {"name": "Llama 3.1 405B Instruct", "provider": "openai", "model": "meta-llama/llama-3.1-405b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.1-70b-instruct": {"name": "Llama 3.1 70B Instruct", "provider": "openai", "model": "meta-llama/llama-3.1-70b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3.1-8b-instruct": {"name": "Llama 3.1 8B Instruct", "provider": "openai", "model": "meta-llama/llama-3.1-8b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3-70b-instruct": {"name": "Llama 3 70B Instruct", "provider": "openai", "model": "meta-llama/llama-3-70b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "llama-3-8b-instruct": {"name": "Llama 3 8B Instruct", "provider": "openai", "model": "meta-llama/llama-3-8b-instruct", "base_url": "https://openrouter.ai/api/v1"},

    # --- Qwen ---
    "qwen3-235b": {"name": "Qwen3 235B", "provider": "openai", "model": "qwen/qwen3-235b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen3-32b": {"name": "Qwen3 32B", "provider": "openai", "model": "qwen/qwen3-32b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen3-30b-a3b": {"name": "Qwen3 30B-A3B", "provider": "openai", "model": "qwen/qwen3-30b-a3b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen3-14b": {"name": "Qwen3 14B", "provider": "openai", "model": "qwen/qwen3-14b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen3-8b": {"name": "Qwen3 8B", "provider": "openai", "model": "qwen/qwen3-8b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen3-4b": {"name": "Qwen3 4B", "provider": "openai", "model": "qwen/qwen3-4b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen3-1.7b": {"name": "Qwen3 1.7B", "provider": "openai", "model": "qwen/qwen3-1.7b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen3-0.6b": {"name": "Qwen3 0.6B", "provider": "openai", "model": "qwen/qwen3-0.6b", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-72b-instruct": {"name": "Qwen2.5 72B Instruct", "provider": "openai", "model": "qwen/qwen-2.5-72b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-32b-instruct": {"name": "Qwen2.5 32B Instruct", "provider": "openai", "model": "qwen/qwen-2.5-32b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-14b-instruct": {"name": "Qwen2.5 14B Instruct", "provider": "openai", "model": "qwen/qwen-2.5-14b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-7b-instruct": {"name": "Qwen2.5 7B Instruct", "provider": "openai", "model": "qwen/qwen-2.5-7b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-3b-instruct": {"name": "Qwen2.5 3B Instruct", "provider": "openai", "model": "qwen/qwen-2.5-3b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-1.5b-instruct": {"name": "Qwen2.5 1.5B Instruct", "provider": "openai", "model": "qwen/qwen-2.5-1.5b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-coder-32b": {"name": "Qwen2.5 Coder 32B", "provider": "openai", "model": "qwen/qwen-2.5-coder-32b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-coder-14b": {"name": "Qwen2.5 Coder 14B", "provider": "openai", "model": "qwen/qwen-2.5-coder-14b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2.5-vl": {"name": "Qwen2.5 VL", "provider": "openai", "model": "qwen/qwen-2.5-vl-72b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "qwen2-vl": {"name": "Qwen2 VL", "provider": "openai", "model": "qwen/qwen-2-vl-72b-instruct", "base_url": "https://openrouter.ai/api/v1"},

    # --- Mistral AI ---
    "magistral-medium": {"name": "Magistral Medium", "provider": "openai", "model": "mistralai/magistral-medium", "base_url": "https://api.mistral.ai/v1"},
    "magistral-small": {"name": "Magistral Small", "provider": "openai", "model": "mistralai/magistral-small", "base_url": "https://api.mistral.ai/v1"},
    "mistral-large-2": {"name": "Mistral Large 2", "provider": "openai", "model": "mistral-large-latest", "base_url": "https://api.mistral.ai/v1"},
    "mistral-large": {"name": "Mistral Large", "provider": "openai", "model": "mistral-large", "base_url": "https://api.mistral.ai/v1"},
    "mistral-medium-3": {"name": "Mistral Medium 3", "provider": "openai", "model": "mistral-medium-3", "base_url": "https://api.mistral.ai/v1"},
    "mistral-small-3": {"name": "Mistral Small 3", "provider": "openai", "model": "mistral-small-3", "base_url": "https://api.mistral.ai/v1"},
    "mistral-small-3.1": {"name": "Mistral Small 3.1", "provider": "openai", "model": "mistral-small-3.1", "base_url": "https://api.mistral.ai/v1"},
    "pixtral-large": {"name": "Pixtral Large", "provider": "openai", "model": "pixtral-large-latest", "base_url": "https://api.mistral.ai/v1"},
    "pixtral-12b": {"name": "Pixtral 12B", "provider": "openai", "model": "pixtral-12b", "base_url": "https://api.mistral.ai/v1"},
    "ministral-8b": {"name": "Ministral 8B", "provider": "openai", "model": "ministral-8b-latest", "base_url": "https://api.mistral.ai/v1"},
    "ministral-3b": {"name": "Ministral 3B", "provider": "openai", "model": "ministral-3b-latest", "base_url": "https://api.mistral.ai/v1"},
    "codestral": {"name": "Codestral", "provider": "openai", "model": "codestral-latest", "base_url": "https://api.mistral.ai/v1"},
    "devstral": {"name": "Devstral", "provider": "openai", "model": "devstral", "base_url": "https://api.mistral.ai/v1"},
    "mixtral-8x22b": {"name": "Mixtral 8x22B", "provider": "openai", "model": "open-mixtral-8x22b", "base_url": "https://api.mistral.ai/v1"},
    "mixtral-8x7b": {"name": "Mixtral 8x7B", "provider": "openai", "model": "open-mixtral-8x7b", "base_url": "https://api.mistral.ai/v1"},
    "mistral-nemo": {"name": "Mistral Nemo", "provider": "openai", "model": "open-mistral-nemo", "base_url": "https://api.mistral.ai/v1"},
    "mistral-7b": {"name": "Mistral 7B", "provider": "openai", "model": "open-mistral-7b", "base_url": "https://api.mistral.ai/v1"},

    # --- xAI (Grok) ---
    "grok-4": {"name": "Grok 4", "provider": "openai", "model": "grok-4", "base_url": "https://api.x.ai/v1"},
    "grok-4-heavy": {"name": "Grok 4 Heavy", "provider": "openai", "model": "grok-4-heavy", "base_url": "https://api.x.ai/v1"},
    "grok-3": {"name": "Grok 3", "provider": "openai", "model": "grok-3", "base_url": "https://api.x.ai/v1"},
    "grok-3-mini": {"name": "Grok 3 Mini", "provider": "openai", "model": "grok-3-mini", "base_url": "https://api.x.ai/v1"},
    "grok-2": {"name": "Grok 2", "provider": "openai", "model": "grok-2", "base_url": "https://api.x.ai/v1"},
    "grok-2-mini": {"name": "Grok 2 Mini", "provider": "openai", "model": "grok-2-mini", "base_url": "https://api.x.ai/v1"},

    # --- Moonshot AI (Kimi) ---
    "kimi-k2": {"name": "Kimi K2", "provider": "openai", "model": "kimi-k2", "base_url": "https://api.moonshot.cn/v1"},
    "kimi-k1.5": {"name": "Kimi K1.5", "provider": "openai", "model": "kimi-k1.5", "base_url": "https://api.moonshot.cn/v1"},

    # --- Zhipu AI (GLM) ---
    "glm-4.5": {"name": "GLM-4.5", "provider": "openai", "model": "glm-4.5", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "glm-4.5-air": {"name": "GLM-4.5 Air", "provider": "openai", "model": "glm-4.5-air", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "glm-4-plus": {"name": "GLM-4 Plus", "provider": "openai", "model": "glm-4-plus", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "glm-4-air": {"name": "GLM-4 Air", "provider": "openai", "model": "glm-4-air", "base_url": "https://open.bigmodel.cn/api/paas/v4"},
    "glm-4-flash": {"name": "GLM-4 Flash", "provider": "openai", "model": "glm-4-flash", "base_url": "https://open.bigmodel.cn/api/paas/v4"},

    # --- MiniMax ---
    "minimax-m1": {"name": "MiniMax M1", "provider": "openai", "model": "minimax-m1", "base_url": "https://api.minimax.chat/v1"},
    "minimax-text-01": {"name": "MiniMax Text-01", "provider": "openai", "model": "minimax-text-01", "base_url": "https://api.minimax.chat/v1"},

    # --- Cohere ---
    "command-a": {"name": "Command A", "provider": "openai", "model": "command-a", "base_url": "https://api.cohere.com/v2"},
    "command-r-plus": {"name": "Command R+", "provider": "openai", "model": "command-r-plus", "base_url": "https://api.cohere.com/v2"},
    "command-r": {"name": "Command R", "provider": "openai", "model": "command-r", "base_url": "https://api.cohere.com/v2"},

    # --- AI21 ---
    "jamba-large": {"name": "Jamba Large", "provider": "openai", "model": "jamba-1.5-large", "base_url": "https://api.ai21.com/v1"},
    "jamba-mini": {"name": "Jamba Mini", "provider": "openai", "model": "jamba-1.5-mini", "base_url": "https://api.ai21.com/v1"},

    # --- Microsoft ---
    "phi-4": {"name": "Phi-4", "provider": "openai", "model": "microsoft/phi-4", "base_url": "https://openrouter.ai/api/v1"},
    "phi-4-mini": {"name": "Phi-4 Mini", "provider": "openai", "model": "microsoft/phi-4-mini", "base_url": "https://openrouter.ai/api/v1"},
    "phi-4-multimodal": {"name": "Phi-4 Multimodal", "provider": "openai", "model": "microsoft/phi-4-multimodal", "base_url": "https://openrouter.ai/api/v1"},
    "phi-3.5-mini": {"name": "Phi-3.5 Mini", "provider": "openai", "model": "microsoft/phi-3.5-mini-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "phi-3-medium": {"name": "Phi-3 Medium", "provider": "openai", "model": "microsoft/phi-3-medium-4k-instruct", "base_url": "https://openrouter.ai/api/v1"},

    # --- IBM Granite ---
    "granite-3.3-8b": {"name": "Granite 3.3 8B", "provider": "openai", "model": "ibm/granite-3.3-8b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "granite-3.3-2b": {"name": "Granite 3.3 2B", "provider": "openai", "model": "ibm/granite-3.3-2b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "granite-vision": {"name": "Granite Vision", "provider": "openai", "model": "ibm/granite-vision-3.1-2b-preview", "base_url": "https://openrouter.ai/api/v1"},
    "granite-code": {"name": "Granite Code", "provider": "openai", "model": "ibm/granite-34b-code-instruct", "base_url": "https://openrouter.ai/api/v1"},

    # --- NVIDIA ---
    "nemotron-ultra": {"name": "Nemotron Ultra", "provider": "openai", "model": "nvidia/llama-3.1-nemotron-70b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "nemotron-70b": {"name": "Nemotron 70B", "provider": "openai", "model": "nvidia/nemotron-4-340b-instruct", "base_url": "https://openrouter.ai/api/v1"},
    "nemotron-nano": {"name": "Nemotron Nano", "provider": "openai", "model": "nvidia/nemotron-nano", "base_url": "https://openrouter.ai/api/v1"},

    # --- Local / Self-Hosted ---
    "ollama": {"name": "Ollama (Local)", "provider": "openai", "model": "llama3.2", "base_url": "http://localhost:11434/v1"},
    "vllm": {"name": "vLLM (Local)", "provider": "openai", "model": "default", "base_url": "http://localhost:8000/v1"},
    "llama.cpp": {"name": "llama.cpp (Local Server)", "provider": "openai", "model": "default", "base_url": "http://localhost:8080/v1"},
    "lm-studio": {"name": "LM Studio (Local)", "provider": "openai", "model": "default", "base_url": "http://localhost:1234/v1"},
    "text-gen-webui": {"name": "Text Generation WebUI", "provider": "openai", "model": "default", "base_url": "http://localhost:5000/v1"},
    "koboldcpp": {"name": "KoboldCpp", "provider": "openai", "model": "default", "base_url": "http://localhost:5001/v1"},
    "localai": {"name": "LocalAI", "provider": "openai", "model": "default", "base_url": "http://localhost:8080/v1"},

    # --- Routers / Aggregators ---
    "openrouter": {"name": "OpenRouter", "provider": "openai", "model": "auto", "base_url": "https://openrouter.ai/api/v1"},
    "together-ai": {"name": "Together AI", "provider": "openai", "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "base_url": "https://api.together.xyz/v1"},
    "fireworks-ai": {"name": "Fireworks AI", "provider": "openai", "model": "accounts/fireworks/models/llama-v3p1-70b-instruct", "base_url": "https://api.fireworks.ai/inference/v1"},
    "groq": {"name": "Groq LPU", "provider": "openai", "model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    "cerebras": {"name": "Cerebras Fast AI", "provider": "openai", "model": "llama3.1-70b", "base_url": "https://api.cerebras.ai/v1"},
    "sambanova": {"name": "SambaNova Systems", "provider": "openai", "model": "Meta-Llama-3.1-70B-Instruct", "base_url": "https://api.sambanova.ai/v1"},
    "replicate": {"name": "Replicate", "provider": "openai", "model": "meta/meta-llama-3-70b-instruct", "base_url": "https://api.replicate.com/v1"},
    "huggingface": {"name": "Hugging Face Inference", "provider": "openai", "model": "meta-llama/Meta-Llama-3-70B-Instruct", "base_url": "https://api-inference.huggingface.co/v1"},
    "azure-openai": {"name": "Azure OpenAI Service", "provider": "openai", "model": "gpt-4o", "base_url": "https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT"},
    "vertex-ai": {"name": "Google Vertex AI", "provider": "gemini", "model": "gemini-1.5-pro", "base_url": "https://generativelanguage.googleapis.com/v1beta"},
    "amazon-bedrock": {"name": "Amazon Bedrock (OpenAI Adapter)", "provider": "openai", "model": "anthropic.claude-3-5-sonnet", "base_url": "http://localhost:8080/v1"},
    "cloudflare-workers-ai": {"name": "Cloudflare Workers AI", "provider": "openai", "model": "@cf/meta/llama-3.1-70b-instruct", "base_url": "https://api.cloudflare.com/client/v4/accounts/YOUR_ACCOUNT/ai/v1"},
    "custom": {"name": "Custom Endpoint (OpenAI-compatible)", "provider": "openai", "model": "custom-model", "base_url": "http://localhost:11434/v1"},
}


class UnifiedAITranslator(BaseTranslator):
    def __init__(
        self,
        model_key: str = "gpt-4o-mini",
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
        self.preset = AI_MODELS.get(model_key, AI_MODELS.get("gpt-4o-mini", AI_MODELS["custom"]))
        self.raw_base_url = base_url or self.preset["base_url"]
        self.api_key = api_key or os.environ.get("AI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self.custom_prompt = custom_prompt
        self.model = custom_model_name or self.preset["model"]

        # Check if user is using a custom router/proxy (like agentrouter, openrouter, custom endpoint)
        self.is_custom_router = bool(base_url and base_url.rstrip("/") != self.preset["base_url"].rstrip("/"))
        self.provider_type = "openai" if self.is_custom_router else self.preset["provider"]
        self.base_url = normalize_base_url(self.raw_base_url, is_openai_compatible=(self.provider_type == "openai"))

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
            f"🤖 [AI Query] Model: '{self.model}' | Endpoint: '{self.base_url}' | "
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

        if self.frequency_penalty != 0.0:
            payload["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            payload["presence_penalty"] = self.presence_penalty

        if any(k in self.model for k in ["o1", "o3", "o4", "reasoner", "thinking"]):
            payload["reasoning_effort"] = self.reasoning_effort

        if "ollama" not in self.base_url and not any(k in self.model for k in ["reasoner", "o1", "o3", "thinking"]):
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
                            self._log(f"🧠 [AI Reasoning] {msg['reasoning_content'][:300]}...")
                        raw_text = msg["content"]

                    self._log(f"⚡ [AI Output ({duration}s)] snippet: {raw_text[:200]}...")

                    parsed_array = self._parse_json_result(raw_text)
                    if parsed_array and len(parsed_array) == len(original_texts):
                        return parsed_array
                    elif parsed_array:
                        self._log(f"⚠️ [AI Warning] Length mismatch! Expected {len(original_texts)}, got {len(parsed_array)}.")
                        return parsed_array[:len(original_texts)] + [None] * (len(original_texts) - len(parsed_array))

            except urllib.error.HTTPError as he:
                self._log(f"❌ [AI HTTP Error {he.code}] {he.reason}")
                # If custom router failed on Anthropic/Gemini path, fallback to OpenAI chat completions
                if is_anthropic or is_gemini:
                    self._log("🔄 Attempting OpenAI-compatible proxy fallback (/v1/chat/completions)...")
                    fallback_url = normalize_base_url(self.raw_base_url) + "/chat/completions"
                    fb_payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": f"{payload.get('system', '')}\n\n{json.dumps(original_texts, ensure_ascii=False)}"}],
                        "temperature": self.temperature
                    }
                    fb_req = urllib.request.Request(
                        fallback_url,
                        data=json.dumps(fb_payload).encode("utf-8"),
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
                    )
                    try:
                        with urllib.request.urlopen(fb_req, timeout=60) as fb_resp:
                            fb_json = json.loads(fb_resp.read().decode("utf-8"))
                            fb_text = fb_json["choices"][0]["message"]["content"]
                            return self._parse_json_result(fb_text) or [None] * len(original_texts)
                    except Exception as fb_exc:
                        self._log(f"❌ [Fallback Error] {fb_exc}")

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
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
