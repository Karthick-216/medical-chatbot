# 🩺 Multilingual Medical Support Chatbot

A professional Streamlit application that accepts health questions in multiple languages, translates them to English, generates safety-aware medical guidance with a Hugging Face text-generation model, and translates the response back to the user's original language.

> **Medical disclaimer:** This project provides general health education only. It is not a substitute for professional medical diagnosis, treatment, or emergency care.

## Project objectives

- Detect or select the user's language.
- Translate non-English questions into English using a multilingual Hugging Face model.
- Generate a medically cautious response with a domain-specific or fine-tuned LLM.
- Translate the answer back to the user's language.
- Provide a clean, deployable Streamlit interface with robust validation and error feedback.
- Include reproducible data-preparation and fine-tuning scripts.

## Business use cases

1. **Patient support:** 24/7 multilingual answers for common health questions.
2. **Public health awareness:** Accessible vaccination, hygiene, and prevention guidance.
3. **Hospital and clinic assistance:** Help users understand departments, appointments, and FAQs.
4. **Symptom pre-screening:** Collect symptoms and summarize topics to discuss with a clinician.
5. **Medical education:** Explain common medical terms in simple language.
6. **Rural and remote access:** Reduce language barriers for first-level health information.

## Features

- Streamlit UI with language selector, model configuration, and expandable translation details.
- Automatic language detection with manual override.
- NLLB-based multilingual translation route through English.
- Configurable generation model via sidebar or `MEDICAL_LLM_MODEL` environment variable.
- Emergency keyword detection for red-flag symptoms.
- Built-in medical disclaimer and conservative response prompt.
- Modular Python package for maintainability and testing.
- Fine-tuning workflow using Hugging Face Datasets and Transformers.

## Supported languages

The default configuration includes English, Spanish, French, German, Italian, Portuguese, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Urdu, Arabic, Simplified Chinese, Japanese, Korean, and Russian.

Language support is defined in [`src/config.py`](src/config.py). You can add more languages by registering the ISO code and the corresponding NLLB language code.

## Repository structure

```text
medical-chatbot/
├── app.py                     # Streamlit application entry point
├── requirements.txt           # Runtime and training dependencies
├── src/
│   ├── config.py              # App settings, language map, disclaimer, safety prompt
│   ├── language.py            # Language detection and translation helpers
│   ├── medical_chatbot.py     # End-to-end chatbot orchestration
│   └── model_registry.py      # Hugging Face model loading utilities
├── scripts/
│   ├── prepare_dataset.py     # Download and normalize medical QA data
│   └── fine_tune.py           # Fine-tune a seq2seq model
└── .streamlit/config.toml     # Streamlit theme and server defaults
```

## Setup

### 1. Clone the repository

```bash
git clone <your-public-repo-url>
cd medical-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, enter a medical question, and click **Get guidance**.

## Model configuration

The app defaults to:

- Translation: `facebook/nllb-200-distilled-600M`
- Generation: `google/flan-t5-small`

For production, replace the generation model with your fine-tuned medical model:

```bash
export MEDICAL_LLM_MODEL="your-org/medical-support-flan-t5"
streamlit run app.py
```

You can also change both model IDs in the Streamlit sidebar.

## Fine-tuning workflow

### 1. Prepare data

The preparation script can download a public Hugging Face medical QA dataset and normalize it into JSONL files with `prompt` and `response` fields.

```bash
python scripts/prepare_dataset.py \
  --dataset medalpaca/medical_meadow_medqa \
  --split train \
  --max-rows 5000 \
  --output-dir data/processed
```

Example dataset options to evaluate:

- `medalpaca/medical_meadow_medqa`
- MedQuAD-style medical question-answering datasets available through Hugging Face
- PubMedQA-derived datasets
- HealthCareMagic or other open medical conversational QA datasets, subject to their licenses

If a dataset cannot be redistributed, keep it out of Git and document the source, license, and download command.

### 2. Fine-tune

```bash
python scripts/fine_tune.py \
  --model-id google/flan-t5-small \
  --train-file data/processed/train.jsonl \
  --validation-file data/processed/validation.jsonl \
  --output-dir models/medical-flan-t5-small \
  --epochs 1 \
  --batch-size 4
```

### 3. Use the fine-tuned model

Local model directory:

```bash
export MEDICAL_LLM_MODEL="models/medical-flan-t5-small"
streamlit run app.py
```

Hugging Face Hub model:

```bash
export MEDICAL_LLM_MODEL="your-hf-username/medical-flan-t5-small"
streamlit run app.py
```

## Deployment

### Hugging Face Spaces

1. Create a new Space with the **Streamlit** SDK.
2. Upload this repository or connect the GitHub repo.
3. Keep `app.py` and `requirements.txt` at the repository root.
4. Add secrets or environment variables for private models if needed.
5. Set `MEDICAL_LLM_MODEL` to your fine-tuned model ID for production.

### Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Create a Streamlit Cloud app from the repository.
3. Select `app.py` as the entry point.
4. Add environment variables in app settings if using a custom/private model.

### AWS

A simple deployment path is an EC2 instance or container service:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

For production, place the service behind HTTPS, configure authentication if needed, and monitor latency and model memory usage.

## Quality and evaluation

Suggested evaluation checks:

- **Translation quality:** BLEU/COMET or manual bilingual review for supported language pairs.
- **Answer relevance:** ROUGE/BERTScore against validation QA references.
- **Medical safety:** Clinician review for hallucinations, unsafe advice, diagnosis, and missing red flags.
- **Latency:** Measure cold-start and per-request latency for translation and generation.
- **Usability:** Test UI clarity with multilingual users.

## Coding standards

- Python modules use functional blocks and docstrings.
- Business logic is separated from the Streamlit UI.
- Secrets, data, and model artifacts are excluded by `.gitignore`.
- Code is intended to follow PEP 8 naming and readability conventions.

## Important limitations

- The default generation model is lightweight for easy local demonstration; it is not clinically validated.
- The app provides education and triage-style awareness, not diagnosis or treatment.
- Translation errors can affect the generated answer. Use clinician review before any real-world healthcare deployment.
- Always follow dataset and model licenses when training and deploying.
