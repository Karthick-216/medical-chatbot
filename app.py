import streamlit as st
from langdetect import detect

st.title("🩺 Multilingual Medical Chatbot")

user_input = st.text_area("Enter your medical question")

if st.button("Detect Language"):
    if user_input.strip():
        lang = detect(user_input)
        st.success(f"Detected language code: {lang}")
    else:
        st.warning("Please enter some text")
