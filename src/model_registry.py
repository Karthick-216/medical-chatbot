"""Model loading utilities for Hugging Face pipelines."""

from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


def load_translation_pipeline(model_id: str):
    """Load a multilingual translation pipeline.

    The default model is NLLB 200 distilled, which covers many common languages
    and keeps routing logic simple for a portfolio-scale Streamlit deployment.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("translation", model=model, tokenizer=tokenizer)


def load_generation_pipeline(model_id: str):
    """Load a text-to-text model for medical support answer generation."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)
