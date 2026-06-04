"""Model loading utilities for Hugging Face translation and generation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


@dataclass
class NllbTranslator:
    """Callable NLLB translator compatible with the app's translation helper.

    Recent Transformers versions may not register a generic ``translation`` pipeline
    task for NLLB models. This wrapper avoids that registry issue by calling the
    tokenizer and seq2seq model directly while preserving a pipeline-like return
    shape: ``[{"translation_text": "..."}]``.
    """

    model_id: str
    device: str | None = None

    def __post_init__(self) -> None:
        """Load the tokenizer and model on CPU or GPU."""

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _target_token_id(self, target_language: str) -> int:
        """Return the decoder BOS token ID for the target NLLB language."""

        language_map = getattr(self.tokenizer, "lang_code_to_id", {})
        token_id = language_map.get(target_language)
        if token_id is not None:
            return token_id

        token_id = self.tokenizer.convert_tokens_to_ids(target_language)
        if token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Target language '{target_language}' is not supported by {self.model_id}."
            )
        return token_id

    def __call__(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 512,
        **_: object,
    ) -> list[dict[str, str]]:
        """Translate text from ``src_lang`` to ``tgt_lang``."""

        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.inference_mode():
            generated_tokens = self.model.generate(
                **encoded,
                forced_bos_token_id=self._target_token_id(tgt_lang),
                max_length=max_length,
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]
        return [{"translation_text": translated}]


def load_translation_pipeline(model_id: str) -> NllbTranslator:
    """Load an NLLB-compatible multilingual translator.

    The default model is NLLB 200 distilled, which covers many common languages
    and keeps routing logic simple for a portfolio-scale Streamlit deployment.
    """

    return NllbTranslator(model_id=model_id)


def load_generation_pipeline(model_id: str):
    """Load a text-to-text model for medical support answer generation."""

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)
