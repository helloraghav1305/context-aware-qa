## Context-Aware Question-Answering System
A transformer-based Question Answering (QA) system fine-tuned on the Natural Questions dataset to generate relevant answers from document context. This project leverages the power of Google's FLAN-T5 model to understand and answer questions using surrounding context.

## Project Highlights
- Fine-tuned **`flan-t5-small`** using Hugging Face Transformers and Datasets.
- Extracts structured context from the Natural Questions dataset.
- Preprocesses document tokens to build a context-rich input prompt for QA generation.
- Generates human-readable answers from contexts using trained model.
- Clean, modular Python codebase for both training and inference.

## Project Structure
| File / Folder | Description |
|---------------|-------------|
| `app.py` | Script for loading model/tokenizer and performing inference on a question + context |
| `train.py` | End-to-end script: loads Natural Questions subset, preprocesses data, fine-tunes the model |
| `requirements.txt` | List of Python packages needed for training and inference |
| `.gitignore` | Prevents loading unnecessary files |
| `README.md` | Project documentation (this file) |

## Dataset: Natural Questions (Google AI)
- Source: Hugging Face Datasets- `natural_questions`
- Format: Complex documents with tokenized HTML content and annotations for short/long answers
- Subset: First 3000 samples extracted and saved as `small_nq.json` for faster training/testing
- Preprocessing includes:
- Removing HTML tags
- Extracting answer candidates from token positions
- Constructing a text-based context for T5 input

## Attribution
- Model: [FLAN-T5-small](https://huggingface.co/google/flan-t5-small) (Apache-2.0 License)
- Dataset: [Natural Questions] (Original Source: Google Research). Accessed via Hugging Face Datasets 'natural_questions'. For licensing and detailed information, please refer to the original source.









