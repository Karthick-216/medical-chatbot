"""Configuration values used by the Streamlit medical chatbot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Language:
    """User-facing language metadata and NLLB translation code."""

    name: str
    iso_639_1: str
    nllb_code: str


APP_TITLE: Final[str] = "Multilingual Medical Support Chatbot"
APP_ICON: Final[str] = "🩺"
MAX_INPUT_CHARACTERS: Final[int] = int(os.getenv("MAX_INPUT_CHARACTERS", "2000"))
DEFAULT_GENERATION_MODEL: Final[str] = os.getenv("MEDICAL_LLM_MODEL", "google/flan-t5-small")
DEFAULT_TRANSLATION_MODEL: Final[str] = os.getenv(
    "TRANSLATION_MODEL", "facebook/nllb-200-distilled-600M"
)

LANGUAGES: Final[dict[str, Language]] = {
    "auto": Language("Auto detect", "auto", "auto"),
    "en": Language("English", "en", "eng_Latn"),
    "es": Language("Spanish", "es", "spa_Latn"),
    "fr": Language("French", "fr", "fra_Latn"),
    "de": Language("German", "de", "deu_Latn"),
    "it": Language("Italian", "it", "ita_Latn"),
    "pt": Language("Portuguese", "pt", "por_Latn"),
    "hi": Language("Hindi", "hi", "hin_Deva"),
    "bn": Language("Bengali", "bn", "ben_Beng"),
    "ta": Language("Tamil", "ta", "tam_Taml"),
    "te": Language("Telugu", "te", "tel_Telu"),
    "mr": Language("Marathi", "mr", "mar_Deva"),
    "gu": Language("Gujarati", "gu", "guj_Gujr"),
    "kn": Language("Kannada", "kn", "kan_Knda"),
    "ml": Language("Malayalam", "ml", "mal_Mlym"),
    "ur": Language("Urdu", "ur", "urd_Arab"),
    "ar": Language("Arabic", "ar", "arb_Arab"),
    "zh-cn": Language("Chinese (Simplified)", "zh-cn", "zho_Hans"),
    "ja": Language("Japanese", "ja", "jpn_Jpan"),
    "ko": Language("Korean", "ko", "kor_Hang"),
    "ru": Language("Russian", "ru", "rus_Cyrl"),
}

SUPPORTED_LANGUAGE_OPTIONS: Final[list[str]] = list(LANGUAGES)
EMERGENCY_KEYWORDS: Final[tuple[str, ...]] = (
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "severe bleeding",
    "stroke",
    "suicide",
    "unconscious",
    "seizure",
    "poisoning",
    "anaphylaxis",
)
MEDICAL_DISCLAIMER: Final[str] = (
    "This chatbot provides general health education only and is not a substitute "
    "for professional medical diagnosis, treatment, or emergency care. If symptoms "
    "are severe, worsening, or urgent, contact a licensed clinician or emergency services."
)
SYSTEM_SAFETY_PROMPT: Final[str] = """
You are a careful medical support assistant. Provide general educational guidance,
possible self-care steps, and questions the user may discuss with a clinician.
Do not diagnose, prescribe medication, or claim certainty. Mention urgent red flags
when relevant and recommend professional medical evaluation when appropriate.
Keep the answer concise, empathetic, and easy to understand.
""".strip()
