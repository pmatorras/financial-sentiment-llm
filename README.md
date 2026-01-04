# Financial Sentiment LLM

Fine-tuning lightweight LLMs for financial sentiment analysis using LoRA/QLoRA.

## Project Goal

Learn LLM fine-tuning techniques by building a financial sentiment classifier, then integrate sentiment features into equity selection pipeline ([financial-ML](https://github.com/pmatorras/financial-ML)).

## Tech Stack

- Python 3.10+
- HuggingFace Transformers, Datasets
- PyTorch (with CUDA support)
- Multi-dataset training approach
- Future: LoRA/QLoRA for parameter-efficient fine-tuning


## Current Status
**Phase 1 Complete** - Baseline Model Established (2025-12-17)

- **Test Accuracy:** 85.1%
- **Model:** BERT fine-tuned on 3 financial sentiment datasets
- **Key Finding:** Excellent performance on professional news (99%), challenges with forum discussions (35%)

**In Progress** - Phase 2: Model Optimization & Fine-tuning


See [PROJECT.md](PROJECT.md) for detailed results and roadmap.

## Quick Results
*Phase 2 Validation (Jan 2026)*

These results use a **Multi-Task Architecture** (Classification + Regression) to better handle the continuous sentiment scores in the FiQA dataset.
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

## Model Architecture (Multi-Task)

To address the diverse nature of financial text, we use a shared **FinBERT** backbone with two task-specific heads:

1.  **Classification Head:** Predicts Negative/Neutral/Positive (Standard Cross-Entropy Loss). Used for *PhraseBank* and *Twitter*.
2.  **Regression Head:** Predicts continuous sentiment scores (-1.0 to 1.0). Used for *FiQA*.

**Why?**
*   Standard classification throws away the nuance of "slightly negative" vs "very negative."
*   By training on the continuous scores (Regression), the model learns a richer sentiment representation, which boosted FiQA accuracy by **15%**.


## Project Structure
```bash
├── src/
│ ├── config.py # Paths and hyperparameters
│ ├── preprocessing.py # Dataset loading & preprocessing
│ ├── dataset.py # PyTorch Dataset wrapper
│ ├── model.py # Model architecture
│ ├── train.py # Training loop
│ ├── evaluate.py # Evaluation metrics
│ └── main.py # Training entry point
├── data/
│ └── raw/ # Auto-downloaded datasets
├── models/ # Saved checkpoints
├── notebooks/ # Exploratory analysis
├── PROJECT.md # Detailed roadmap & progress
└── README.md # This file
```


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
python -m financial_sentiment_llm.main
```
**Training Time:**
- **GPU (RTX 4050):** ~4 minutes (3 epochs)
- **CPU (Intel Core Ultra 7):** ~50 minutes (3 epochs)

### Evaluation
```bash
#Evaluate trained model on test set
python -m financial_sentiment_llm.evaluate
```


## Datasets

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (33% weight)
- [Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) (33% weight)
- [FiQA Sentiment](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification) (34% weight)

## Resources

- [PROJECT.md](PROJECT.md) - Detailed roadmap and progress tracking
- [FinBERT Paper](https://arxiv.org/abs/1908.10063) - Financial domain BERT
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft) - For upcoming LoRA implementation

---

**Note**: This is a learning project to develop production-grade LLM and NLP skills. Documentation and results are continuously updated as experimentation progresses.
