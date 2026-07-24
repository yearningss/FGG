# ⚡ FGG 2.0 — Fast Global Translator

<p align="center">
  <img src="https://img.shields.io/badge/Version-2.0.0-00f2fe?style=for-the-badge&logo=python" alt="Version 2.0">
  <img src="https://img.shields.io/badge/Python-3.8+-7928ca?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-ff0080?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/Game_Localization-Engine-00f2fe?style=for-the-badge" alt="Game Localization">
</p>

**FGG 2.0 (Fast Global Translator)** — это мощный, высокоскоростной и надежный движок автоматической локализации видеоигр и модов. 

Поддерживает сохранение всех игровых тегов (`{clr:red}`, `[J]`, `%s`, `\n`), параллельную многопоточность, перевод через **Google Translate**, **OpenAI GPT-4o / LLM**, встроенный **Web UI Dashboard** в браузере, систему кэширования (Translation Memory) и автоматическую проверку ошибок (QA Validator).

---

## 🔥 Ключевые возможности FGG 2.0

* 🌐 **Многоязычный автоматический перевод**: Поддержка 20+ основных мировых языков.
* 🛡️ **Защита игровых тегов и переменных**: Игровые конструкции `{clr:red}`, `{clr/}`, `[J]`, `%s`, `%d`, `{0}`, `\n`, `\t` и BBCode не ломаются при переводе.
* 🖥️ **Встроенный Web UI Dashboard**: Графический интерфейс прямо в браузере — запуск перевода в один клик, выбор языков, отслеживание в реальном времени.
* 🤖 **AI & LLM Поддержка (GPT-4o)**: Возможность перевода через нейросети с учётом игрового контекста и терминологии.
* 💾 **Translation Memory & Glossaries**: Повторные и схожие строки берутся из локальной базы данных SQLite, экономя время и трафик.
* 🔍 **QA Validator (Проверка качества)**: Автоматическая проверка перевода на пропущенные теги, несбалансированные скобки или кавычки.
* ⚡ **Высокая скорость (Multi-threading)**: Настраиваемый многопоточный режим (Workers 5-15) для мгновенной обработки больших файлов локализаций.

---

## 🚀 Быстрый запуск

### 1. Веб-интерфейс (Web UI) в браузере:

Запустите веб-панель одной командой:

```bash
python web_launcher.py
```
Или через CLI:
```bash
python final_translator.py --web
```
Интерфейс откроется по адресу `http://localhost:8080`.

---

### 2. Запуск через командную строку (CLI):

#### 🇩🇪 Перевод на немецкий язык (Google Translate):
```bash
python final_translator.py --langs de
```

#### ⚡ Быстрый многопоточный перевод на несколько языков:
```bash
python final_translator.py --langs de en fr es ja --workers 10
```

#### 🤖 Перевод через OpenAI GPT-4o:
```bash
python final_translator.py --langs de --provider openai --openai-key "YOUR_API_KEY"
```

#### 🧪 Тестовый режим (Mock / Dry Run):
```bash
python final_translator.py --langs de --provider mock
```

---

## 🎮 Поддерживаемые форматы и теги

FGG автоматически распознаёт и сохраняет:
* Формат локализаций `key = "value"` (GameMaker / Hotline Miami / Pizza Tower / Fangames).
* `.json` файлы конфигураций и локализаций.
* Теги цветов и стилей: `{clr:red}`, `{clr/}`, `[color=#fff]`
* Кнопки и управления: `[J]`, `[M]`, `[G]`, `[SPACE]`
* Переменные кода: `%s`, `%d`, `{player_name}`, `{0}`
* Служебные символы: `\n`, `\t`, `\r`

Пример:
```text
# Исходный текст:
tutorial1 = "Нажми [J], чтобы сделать прыжок в {clr:red}бездну{clr/}!"

# После перевода на немецкий:
tutorial1 = "Drücken Sie [J], um einen Sprung in {clr:red}Abgrund{clr/} zu machen!"
```

---

## 🛠️ Параметры CLI

| Флаг | Описание | Значение по умолчанию |
| :--- | :--- | :--- |
| `--langs` | Список кодов языков (`de`, `en`, `fr`, `es`, `ja`, `zh-cn`...) | Все доступные |
| `--input` | Путь к исходному файлу локализации | `оригинал/rus.txt` |
| `--output-dir` | Папка для сохранения готовых переводов | `переводы/` |
| `--workers` | Количество параллельных потоков | `5` |
| `--provider` | Движок перевода (`google`, `openai`, `mock`) | `google` |
| `--force` | Принудительный переперевод всех строк | `False` |
| `--web` | Запуск Web UI панели | `False` |

---

## 🏗️ Структура проекта

```text
FGG/
├── fgg/                       # Ядро системы FGG 2.0
│   ├── core/                  # Парсеры, теги, QA, кэш и движок
│   │   ├── engine.py          # Основной контроллер перевода
│   │   ├── placeholders.py    # Защита игровых тегов и переменных
│   │   ├── parser.py          # Парсер файлов (key=val, json, etc)
│   │   ├── qa.py              # QA-валидатор ошибок
│   │   └── glossary.py        # Глоссарий и Translation Memory
│   ├── providers/             # Провайдеры (Google, OpenAI GPT, Mock)
│   ├── web/                   # Standalone Web Dashboard
│   └── cli.py                 # CLI интерфейс 2.0
├── оригинал/                  # Исходные локализации
│   └── rus.txt
├── переводы/                  # Сгенерированные переводы
├── final_translator.py        # Главная точка входа (CLI)
└── web_launcher.py            # Быстрый запуск Web UI
```

---

## 📝 Лицензия

Проект распространяется под лицензией MIT.
Разработано с 💖 для геймдева и сообщества локализаторов игр.
