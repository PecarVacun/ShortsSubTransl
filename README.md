# Subtitle localization pipeline

A modular pipeline that extracts screenplay text from a text-based PDF, asks a
local Ollama model for screenplay metadata and character cards, translates SRT
cues with DeepL, then validates line length, line count and cue timing.

## Setup

Requires Python 3.9 or later, a running Ollama server, and a DeepL API key.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
cp .env.example .env
```

Set `DEEPL_API_KEY` in `.env`, then download the configured local model:

```sh
ollama pull llama3.1:70b-instruct-q4_K_M
python -m ui.web_app
```

The web UI accepts a video as optional workflow context, but video content is
not inspected. Script PDFs must contain extractable text; scanned PDFs need an
external OCR step. The script is processed in bounded chunks and every chunk's
analysis is retained. An incomplete model response or invalid JSON stops the
run instead of silently omitting script content.

DeepL receives screenplay analysis as **advisory context**. It is not a
terminology glossary and does not guarantee exact character-name or term
choices. The pipeline translates cues one at a time; large SRT files may
therefore take time and consume DeepL quota.

Output is written next to the source SRT as `<name>.<LANG>.srt`. The formatter
normalizes incidental line breaks, wraps at 38 characters without splitting
words, limits cues to two lines, and extends cues shorter than 1000 ms only
when doing so does not overlap the next cue. Invalid source overlaps,
unbreakable words, overlong text, or impossible timing are reported as errors;
text is never truncated. No output file is written until all cues pass
formatting validation.

## Modules

- `config.py`: environment-backed runtime settings.
- `services/`: PDF extraction, local Ollama analysis and DeepL translation.
- `core/`: UI-independent subtitle formatting and pipeline orchestration.
- `ui/web_app.py`: Gradio components and event wiring only.

Run tests with `python -m pytest`.
