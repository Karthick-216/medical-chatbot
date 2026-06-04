"""Streamlit entry point for the multilingual medical support chatbot."""

from __future__ import annotations

import streamlit as st

from src.config import (
    APP_ICON,
    APP_TITLE,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_TRANSLATION_MODEL,
    LANGUAGES,
    MAX_INPUT_CHARACTERS,
    MEDICAL_DISCLAIMER,
)
from src.language import detect_language_code
from src.medical_chatbot import answer_medical_question, validate_question
from src.model_registry import load_generation_pipeline, load_translation_pipeline


@st.cache_resource(show_spinner="Loading multilingual translation model...")
def get_translator(model_id: str):
    """Cache the translation model across Streamlit reruns."""

    return load_translation_pipeline(model_id)


@st.cache_resource(show_spinner="Loading medical answer generation model...")
def get_generator(model_id: str):
    """Cache the answer-generation model across Streamlit reruns."""

    return load_generation_pipeline(model_id)


def render_sidebar() -> tuple[str, str, str]:
    """Render configuration controls and return selected runtime options."""

    st.sidebar.header("Settings")
    language_code = st.sidebar.selectbox(
        "Question language",
        options=list(LANGUAGES),
        format_func=lambda code: LANGUAGES[code].name,
        help="Use auto detect unless you already know the input language.",
    )
    translation_model = st.sidebar.text_input(
        "Translation model",
        value=DEFAULT_TRANSLATION_MODEL,
        help="NLLB-compatible Hugging Face model ID.",
    )
    generation_model = st.sidebar.text_input(
        "Medical generation model",
        value=DEFAULT_GENERATION_MODEL,
        help="Use your fine-tuned Hugging Face model ID for production deployments.",
    )

    st.sidebar.divider()
    st.sidebar.caption("Supported languages")
    st.sidebar.write(
        ", ".join(language.name for code, language in LANGUAGES.items() if code != "auto")
    )
    return language_code, translation_model, generation_model


def render_header() -> None:
    """Render the main app header and safety disclaimer."""

    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.info(MEDICAL_DISCLAIMER, icon="⚕️")
    st.markdown(
        "Ask a health-related question in a supported language. The app translates "
        "your question to English, generates a safety-aware response, and translates "
        "the answer back to your language."
    )


def render_chatbot() -> None:
    """Render the chatbot form and model outputs."""

    language_code, translation_model, generation_model = render_sidebar()

    with st.form("medical_question_form", clear_on_submit=False):
        question = st.text_area(
            "Your medical question",
            height=160,
            max_chars=MAX_INPUT_CHARACTERS,
            placeholder="Example: I have had a sore throat and mild fever for two days. What should I do?",
        )
        submitted = st.form_submit_button("Get guidance", type="primary")

    if not submitted:
        return

    try:
        cleaned_question = validate_question(question, MAX_INPUT_CHARACTERS)
        if language_code == "auto":
            resolved_language_code = detect_language_code(cleaned_question)
        else:
            resolved_language_code = language_code
        translator = None
        if resolved_language_code != "en":
            translator = get_translator(translation_model)

        generator = get_generator(generation_model)
        with st.spinner("Translating and generating a response..."):
            response = answer_medical_question(
                question=cleaned_question,
                source_language_code=resolved_language_code,
                translator=translator,
                generator=generator,
                max_characters=MAX_INPUT_CHARACTERS,
            )
    except Exception as exc:
        st.error(f"Unable to generate a response: {exc}", icon="🚧")
        st.stop()

    if response.emergency_notice:
        st.error(response.emergency_notice, icon="🚨")

    st.subheader("Response")
    st.write(response.localized_answer)

    with st.expander("Translation and model details"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected language", response.detected_language_name)
            st.text_area("Question translated to English", response.english_question, height=120)
        with col2:
            st.text_area("English model answer", response.english_answer, height=220)


def main() -> None:
    """Run the Streamlit application."""

    render_header()
    render_chatbot()


if __name__ == "__main__":
    main()
