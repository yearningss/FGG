# FGG 2.0 — High-Performance Game Localization & AI Translation Suite

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![AI Models](https://img.shields.io/badge/AI%20Models-100%2B-purple.svg)](#supported-ai-models)

**FGG 2.0** is an enterprise-grade video game localization toolkit designed for indie developers, translation studios, and modders. It features an automated placeholder protection engine, SQLite Translation Memory, real-time AI stream reasoning logs, ping latency diagnostics, and support for over 100+ AI models across major providers and local servers.

---

## 🌟 Key Features

* **🛡️ Smart Placeholder Protection (`PlaceholderEngine`)**  
  Safeguards color tags (`{clr:red}`), key icons (`[J]`), format parameters (`%s`, `%d`, `{0}`), control sequences (`\n`), and HTML/BBCode during translation.

* **🤖 100+ AI Models & Custom Proxy Routers**  
  Built-in support for OpenAI (GPT-5, o3, o1, GPT-4o), Anthropic (Claude Opus 4.8 / 4.1, Sonnet), Google Gemini (2.5 / 2.0), DeepSeek (V3, R1), Grok 4, Llama 4, Qwen 3, Mistral, and local servers (Ollama, vLLM, LM Studio). Fully compatible with custom API routers like `agentrouter.org` and OpenRouter.

* **📡 AI Ping & Connection Latency Benchmark**  
  Real-time connection testing tool displaying server ping (latency in ms), HTTP status codes, verified model IDs, and endpoint routing.

* **💾 SQLite Translation Memory & Terminology Glossary**  
  Caches translated segments locally in `.fgg_cache.sqlite` to prevent duplicate API calls, reduce costs, and ensure glossary term consistency.

* **🔍 Quality Assurance (QA) Engine**  
  Validates tag parity, bracket balance, quotes, and empty output post-translation.

* **🌐 Standalone Web Dashboard & CLI Interface**  
  Includes both an interactive GitHub Dark-style Web UI and a command-line interface.

---

## 🚀 Quick Start

### 1. Requirements & Installation

```bash
git clone https://github.com/yearningss/FGG.git
cd FGG
pip install deep_translator
```

### 2. Launch the Web UI

Run the single-click web launcher:

```bash
python web_launcher.py
```

Open **`http://localhost:8080`** in your browser.

### 3. Command Line Interface (CLI)

```bash
python final_translator.py --input оригинал/rus.txt --out переводы --langs en de ja zh-cn --provider google --workers 5
```

---

## 🎛️ Configuration & Fine-Tuning

### AI Hyperparameters
- **Temperature**: Controls creativity (`0.0` for deterministic game localization, `1.0` for creative dialog).
- **Top P**: Nucleus sampling filter.
- **Max Output Tokens**: Up to 8192 tokens per batch.
- **Reasoning Effort**: Low / Medium / High for DeepSeek R1, OpenAI o1, o3, and Thinking models.

---

## 📁 Repository Structure

```
FGG/
├── fgg/
│   ├── core/
│   │   ├── engine.py          # Multi-threaded translation coordinator
│   │   ├── glossary.py        # Glossary rules & SQLite Translation Memory
│   │   ├── parser.py          # KeyValue and JSON localization parsers
│   │   ├── placeholders.py    # Game tag protection & restoration
│   │   └── qa.py              # Localization Quality Assurance validator
│   ├── providers/
│   │   ├── ai_provider.py     # Multi-model AI engine (100+ models)
│   │   ├── google_provider.py # Free Google Translate provider
│   │   └── mock_provider.py   # Dry-run testing provider
│   ├── web/
│   │   ├── app.py             # REST API & static web server
│   │   └── static/
│   │       └── index.html     # Web UI Dashboard
│   └── cli.py                 # CLI entry point
├── final_translator.py        # Main CLI entry script
├── web_launcher.py            # Web UI launcher script
└── README.md                  # Documentation
```

---

## 📄 License

MIT License. Free for commercial and non-commercial game localization projects.
