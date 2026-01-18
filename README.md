---
title: Financial Sentiment Demo
emoji: 🏃
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
app_port: 7860
---

# Financial Sentiment LLM

Fine-tuning lightweight LLMs for financial sentiment analysis using FinBERT.

## Project Goal

Learn LLM fine-tuning techniques by building a financial sentiment classifier, then integrate sentiment features into equity selection pipeline ([financial-ML](https://github.com/pmatorras/financial-ML)).

**Built with**: Python 3.10 - PyTorch - Hugging Face Transformers - Pandas


## Current Status
**Phase 2 In Progress** - Multi-Task Architecture Validated (2026-01-03)

- **Test Accuracy:** 85.0%
- **Model:** FinBERT with Multi-Task Head (Classification + Regression)
- **Key Finding:** Treating FiQA scores as regression targets improved accuracy on that dataset by **+15%** (from 65% to 80%).
- **Data cleaning:** Several cleaning/filtering steps were tested, but did not consistently improve accuracy when training/testing on the same distribution; cleaning remains available via flags but is disabled by default (see [EXPERIMENTS.md](EXPERIMENTS.md)).

**Data sources**

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (33% weight)
- [Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) (33% weight)
- [FiQA Sentiment](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification) (34% weight)

**Next Steps** - Optimize Hyperparameters & Analyze Errors

See [PROJECT.md](PROJECT.md) for detailed results and roadmap.

## Model Approach \& Performance

To handle the diverse nature of financial text, this project implements a **Multi-Task FinBERT** architecture. Unlike standard classifiers, this model shares a BERT backbone with two task-specific heads:

1. **Classification Head:** Predicts Negative/Neutral/Positive (for news/tweets).
2. **Regression Head:** Predicts continuous sentiment scores (for FiQA).

## Validation Results
*Phase 2 (Jan 2026)*

These results use a **Multi-Task Architecture** (Classification + Regression) to better handle the continuous sentiment scores in the FiQA dataset. 

This architecture significantly outperformed our **Single-Task Baseline** (standard classification). By training on continuous scores (Regression), the model learns sentiment intensity, yielding a **+15%** accuracy boost on the challenging FiQA dataset.
| Metric | Value |
|--------|-------|
| Overall Accuracy | 85.0% |
| Macro F1-Score | 0.84 |

### Performance by Dataset Source

| Dataset | Accuracy | Samples | Style |
|---------|----------|---------|-------|
| FinancialPhraseBank | 92.1% | 343 | Professional news |
| Twitter Financial | 81.2% | 499 | Social media |
| FiQA Forums | 80.2% | 116 | Retail discussions|

### Performance by Sentiment Class

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.73 | 0.87 | 0.79 |
| Neutral | 0.88 | 0.84 | 0.86 |
| Positive | 0.88 | 0.85 | 0.86 |

### Why this Architecture?
- **Capturing Nuance:** Standard classification throws away the difference between "slightly negative" and "very negative." The regression head forces the model to learn this nuance.
- **Multi-Task Loss:** We combine Cross-Entropy (for classes) and MSE (for scores) to handle diverse data formats simultaneously.
- **Robust Training:** Includes Early Stopping (patience=3) to prevent overfitting and fixed random seeds for Reproducibility.
- **Loss weighting:** Multi-task loss uses Cross-Entropy (classification) + weighted MSE (regression); defaults are rescaled to `1/10` for implementation convenience while preserving the original 1:10 ratio.



## Project Structure
```bash
├── src/
│   └── finsentiment/
│       ├── cli/                  # CLI entrypoints
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── parser.py
│       ├── datasets/             # Data loading + splitting + dataset class
│       │   ├── __init__.py
│       │   ├── load.py           # Download/load HF datasets → pandas
│       │   ├── preprocessing.py  # Split / balance / combine datasets
│       │   ├── sentiment.py      # Dataset wrapper (task_type-aware)
│       │   └── clean_data.py     # Optional cleaning utilities (default OFF)
│       ├── modeling/             # Model definition(s)
│       │   ├── __init__.py
│       │   └── bert.py           # FinancialSentimentModel (cls + reg heads)
│       ├── training/             # Training loop(s)
│       │   ├── __init__.py
│       │   └── trainer.py
│       ├── evaluation/           # Metrics
│       │   ├── __init__.py
│       │   └── metrics.py
│       ├── config.py             # Global configuration
│       ├── main.py               # Application entry point
│       └── __main__.py           # python -m finsentiment
├── data/
│ └── raw/ # Auto-downloaded datasets
├── models/ # Saved checkpoints
├── notebooks/ # Exploratory analysis
├── PROJECT.md # Detailed roadmap & progress
└── README.md # This file
```
> **Note**: the codebase is now organized as a unified pipeline (multi-task capable), rather than maintaining parallel “single vs multi” modules.

## Installation

### Install PyTorch

Choose based on your hardware:

**GPU (NVIDIA CUDA):**
```bash
# GPU version
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU version  
pip install torch --index-url https://download.pytorch.org/whl/cpu

```
### Install Project

```bash
# Clone and setup
git clone https://github.com/pmatorras/financial-sentiment-llm.git
cd financial-sentiment-llm
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -e . 
pip install -e ".[dev]" #with dev dependencies
```

### Verify Setup
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Usage

### Training
```bash
# Train baseline model (Downloads datasets automatically on first run, ~10MB total)
python -m finsentiment train # Defaults to Multi-Task architecture. Use --model-type multi for baseline.

```
**Training Time:**
- **GPU (RTX 4050):** ~4 minutes (3 epochs)
- **CPU (Intel Core Ultra 7):** ~50 minutes (3 epochs)

### Evaluation
```bash
#Evaluate trained model on test set
python -m finsentiment evaluate # Defaults to Multi-Task architecture. Use --model-type multi for baseline.
```



## Resources

- [PROJECT.md](PROJECT.md) - Detailed roadmap and progress tracking
- [FinBERT Paper](https://arxiv.org/abs/1908.10063) - Financial domain BERT
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft) - For upcoming LoRA implementation

---

**Note**: This is a learning project to develop production-grade LLM and NLP skills. Documentation and results are continuously updated as experimentation progresses.
