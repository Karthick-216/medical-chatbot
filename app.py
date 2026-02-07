import streamlit as st
from langdetect import detect
from transformers import pipeline

# ---------------- CORE LOGIC ----------------

def detect_language(text: str) -> str:
    """
    Detect the language of the input text.
    """
    return detect(text)


# Load translation model once
translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M"
)

def translate_to_english(text: str, src_lang: str) -> str:
    """
    Translate input text to English.
    """
    if src_lang == "en":
        return text

    result = translator(
        text,
        src_lang=src_lang,
        tgt_lang="eng_Latn"
    )
    return result[0]["translation_text"]


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Multilingual Medical Chatbot")
st.title("🩺 Multilingual Medical Support Chatbot")

user_input = st.text_area("Enter your medical question")

if st.button("Translate to English"):
    if user_input.strip():
        lang = detect_language(user_input)
        translated_text = translate_to_english(user_input, lang)

        st.write(f"Detected language: {lang}")
        st.success(translated_text)
    else:
        st.warning("Please enter some text")
