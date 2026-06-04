"""Language detection and translation helpers."""

from __future__ import annotations

from langdetect import LangDetectException, detect

from src.config import LANGUAGES, Language


class UnsupportedLanguageError(ValueError):
    """Raised when a detected or requested language is not configured."""


def detect_language_code(text: str) -> str:
    """Detect an ISO 639 language code for free-form user text.

    Args:
        text: User-provided text.

    Returns:
        A supported ISO language code. Defaults to English for very short text.

    Raises:
        UnsupportedLanguageError: If detection succeeds but the language is not configured.
    """

    normalized = text.strip()
    if len(normalized) < 4:
        return "en"

    try:
        detected = detect(normalized).lower()
    except LangDetectException as exc:
        raise UnsupportedLanguageError(
            "Could not detect the input language. Please choose a language manually."
        ) from exc

    if detected in LANGUAGES:
        return detected

    if detected == "zh":
        return "zh-cn"

    raise UnsupportedLanguageError(
        f"Detected language '{detected}' is not in the supported language list."
    )


def get_language(language_code: str) -> Language:
    """Return configured language metadata for an ISO code."""

    try:
        return LANGUAGES[language_code]
    except KeyError as exc:
        raise UnsupportedLanguageError(
            f"Language '{language_code}' is not supported by this app."
        ) from exc


def translate_text(
    text: str,
    source_language: Language,
    target_language: Language,
    translator,
    max_length: int = 512,
) -> str:
    """Translate text with an NLLB-compatible Hugging Face translation pipeline."""

    if source_language.iso_639_1 == target_language.iso_639_1:
        return text

    translated = translator(
        text,
        src_lang=source_language.nllb_code,
        tgt_lang=target_language.nllb_code,
        max_length=max_length,
    )
    return translated[0]["translation_text"]
