"""Fine-tune a text-to-text medical support model with Hugging Face Transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


def parse_args() -> argparse.Namespace:
    """Parse fine-tuning arguments."""

    parser = argparse.ArgumentParser(description="Fine-tune a medical QA model.")
    parser.add_argument("--model-id", default="google/flan-t5-small", help="Base model ID.")
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--validation-file", default="data/processed/validation.jsonl")
    parser.add_argument("--output-dir", default="models/medical-flan-t5-small")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=256)
    return parser.parse_args()


def tokenize_records(examples, tokenizer, max_source_length: int, max_target_length: int):
    """Tokenize prompt and response fields for seq2seq training."""

    model_inputs = tokenizer(
        examples["prompt"],
        max_length=max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["response"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_metrics(tokenizer):
    """Create a ROUGE metric callback for validation."""

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return rouge.compute(predictions=decoded_predictions, references=decoded_labels)

    return compute_metrics


def main() -> None:
    """Run supervised fine-tuning and save the resulting model."""

    args = parse_args()
    data_files = {"train": args.train_file, "validation": args.validation_file}
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    tokenized = dataset.map(
        tokenize_records,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": args.max_source_length,
            "max_target_length": args.max_target_length,
        },
        remove_columns=dataset["train"].column_names,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        fp16=False,
        logging_steps=25,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=build_metrics(tokenizer),
    )
    trainer.train()

    output_dir = Path(args.output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved fine-tuned model to {output_dir}")


if __name__ == "__main__":
    main()
