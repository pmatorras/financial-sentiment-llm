# Financial Sentiment LLM

Fine-tuning lightweight LLMs for financial sentiment analysis using LoRA/QLoRA.

## Project Goal

Learn LLM fine-tuning techniques by building a financial sentiment classifier, then integrate sentiment features into equity selection pipeline ([financial-ML](link-to-your-repo)).

## Tech Stack

- Python 3.10+
- HuggingFace Transformers, PEFT, TRL
- PyTorch
- LoRA/QLoRA for parameter-efficient fine-tuning

## Current Status

**In Progress** - Phase 1: Dataset & Baseline Evaluation

See [PROJECT.md](PROJECT.md) for detailed roadmap.

## Project Structure
```bash
├── src/
│ └── financial_sentiment_llm/ # Main package
│ ├── data.py # Dataset loading and preprocessing
│ ├── train.py # Training pipeline
│ ├── evaluate.py # Evaluation metrics
│ └── inference.py # Sentiment scoring
├── notebooks/ # Exploration notebooks
├── configs/ # Model configurations
├── pyproject.toml # Package metadata (modern)
└── PROJECT.md # Detailed roadmap
```

## Resources

- Dataset: [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
- [PROJECT.md](PROJECT.md) - Detailed roadmap and progress tracking

---

**Note**: This is a learning project to develop production-grade LLM skills. Results and documentation will be updated as the project progresses.
