"""DeepL translation adapter."""

import json

import deepl


class TranslationError(RuntimeError):
    """Translation failed or its request cannot be safely submitted."""


class TranslatorService:
    """Translate subtitles with advisory context, not an enforced glossary."""

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise TranslationError("DEEPL_API_KEY is missing from the environment or .env.")
        self.client = deepl.DeepLClient(api_key)

    def translate(self, text: str, target_lang: str, glossary_context: str) -> str:
        """Translate one cue; DeepL context is advisory, not a glossary."""
        if not text.strip() or not target_lang.strip():
            raise TranslationError("Subtitle text and target language must not be empty.")
        request_size = len(
            json.dumps([text, glossary_context], ensure_ascii=True).encode("utf-8")
        )
        if request_size > 100_000:
            raise TranslationError(
                "Subtitle and glossary context exceed the safe DeepL request limit."
            )
        try:
            result = self.client.translate_text(
                text, target_lang=target_lang, context=glossary_context
            )
        except deepl.DeepLException as exc:
            raise TranslationError(f"DeepL translation failed: {exc}") from exc
        if not result.text or not result.text.strip():
            raise TranslationError("DeepL returned an empty translation.")
        return result.text
