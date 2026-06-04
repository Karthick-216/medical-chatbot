"""Core orchestration for multilingual medical question answering."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    EMERGENCY_KEYWORDS,
    LANGUAGES,
    MEDICAL_DISCLAIMER,
    SYSTEM_SAFETY_PROMPT,
)
from src.language import detect_language_code, get_language, translate_text


@dataclass(frozen=True)
class ChatbotResponse:
    """Structured response returned by the chatbot pipeline."""

    detected_language_code: str
    detected_language_name: str
    english_question: str
    english_answer: str
    localized_answer: str
    emergency_notice: str | None


def validate_question(question: str, max_characters: int) -> str:
    """Validate and normalize user input before model inference."""

    cleaned = " ".join(question.split())
    if not cleaned:
        raise ValueError("Please enter a medical question before submitting.")
    if len(cleaned) > max_characters:
        raise ValueError(
            f"Please keep the question under {max_characters:,} characters."
        )
    return cleaned


def build_medical_prompt(question: str) -> str:
    """Build the instruction prompt sent to the generation model."""

    return (
        f"{SYSTEM_SAFETY_PROMPT}\n\n"
        f"User question: {question}\n\n"
        "Answer with these sections: 1) General information, 2) What the user can do now, "
        "3) When to seek medical care."
    )


def detect_emergency_context(english_question: str) -> str | None:
    """Return an urgent-care notice if the question contains red-flag terms."""

    lowered = english_question.lower()
    if any(keyword in lowered for keyword in EMERGENCY_KEYWORDS):
        return (
            "Potential urgent symptoms detected. If this is happening now, call local "
            "emergency services immediately or go to the nearest emergency department."
        )
    return None


def generate_english_answer(question: str, generator, max_new_tokens: int = 256) -> str:
    """Generate an English medical guidance answer from a text-to-text model."""

    prompt = build_medical_prompt(question)
    generated = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        truncation=True,
    )
    answer = generated[0]["generated_text"].strip()
    if MEDICAL_DISCLAIMER.lower() not in answer.lower():
        answer = f"{answer}\n\nDisclaimer: {MEDICAL_DISCLAIMER}"
    return answer


def answer_medical_question(
    question: str,
    source_language_code: str,
    translator,
    generator,
    max_characters: int,
) -> ChatbotResponse:
    """Translate, answer, and localize a medical support question."""

    cleaned_question = validate_question(question, max_characters)
    detected_code = (
        detect_language_code(cleaned_question)
        if source_language_code == "auto"
        else source_language_code
    )
    source_language = get_language(detected_code)
    english_language = LANGUAGES["en"]

    english_question = translate_text(
        cleaned_question,
        source_language=source_language,
        target_language=english_language,
        translator=translator,
    )
    emergency_notice = detect_emergency_context(english_question)
    english_answer = generate_english_answer(english_question, generator)

    if emergency_notice:
        english_answer = f"{emergency_notice}\n\n{english_answer}"

    localized_answer = translate_text(
        english_answer,
        source_language=english_language,
        target_language=source_language,
        translator=translator,
        max_length=768,
    )

    return ChatbotResponse(
        detected_language_code=detected_code,
        detected_language_name=source_language.name,
        english_question=english_question,
        english_answer=english_answer,
        localized_answer=localized_answer,
        emergency_notice=emergency_notice,
    )
