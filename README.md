# Financial Sentiment LLM

Fine-tuning lightweight LLMs for financial sentiment analysis using LoRA/QLoRA.

## Project Goal

Learn LLM fine-tuning techniques by building a financial sentiment classifier, then integrate sentiment features into equity selection pipeline ([financial-ML](https://github.com/pmatorras/financial-ML)).

## Tech Stack

- Python 3.10+
- HuggingFace Transformers, Datasets
- PyTorch (with CUDA support)
- Multi-dataset training approach (Classification + Regression Multi-Task)
- Future: LoRA/QLoRA for parameter-efficient fine-tuning

## Current Status
**Phase 2 In Progress** - Multi-Task Architecture Validated (2026-01-03)

- **Test Accuracy:** 85.0%
- **Model:** FinBERT with Multi-Task Head (Classification + Regression)
- **Key Finding:** Treating FiQA scores as regression targets improved accuracy on that dataset by **+15%** (from 65% to 80%).

**Next Steps** - Optimize Hyperparameters & Analyze Errors

See [PROJECT.md](PROJECT.md) for detailed results and roadmap.

## Datasets

- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (Target: 33%, Actual: Limited by size)
- [Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) (Target: 33%)
- [FiQA Sentiment](https://huggingface.co/datasets/TheFinAI/fiqa-sentiment-classification) (Target: 34%, Actual: Limited by size)

## Quick Results (Multi-Task Model)

| Metric | Value |
|--------|-------|
| Overall Accuracy | 85.0% |
| Macro F1-Score | 0.84 |

### Performance by Dataset Source

| Dataset | Accuracy | Samples | Style | Status |
|---------|----------|---------|-------|--------|
| FinancialPhraseBank | 92.1% | 343 | Professional news | Excellent |
| Twitter Financial | 81.2% | 499 | Social media | Robust |
| FiQA Forums | 80.2% | 116 | Retail discussions | **Fixed (+15%)** |

### Performance by Sentiment Class

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.73 | 0.87 | 0.79 |
| Neutral | 0.88 | 0.84 | 0.86 |
| Positive | 0.88 | 0.85 | 0.86 |

## Project Structure
```bash
├── src/
│ ├── config.py # Paths and hyperparameters
│ ├── preprocessing.py # Dataset loading & preprocessing (Split-before-Balance)
│ ├── dataset.py # PyTorch Dataset wrapper
│ ├── model.py # Model architecture (MultiTaskModel)
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
# Defaults to Multi-Task architecture (use --model-type single for baseline)
python -m financial_sentiment_llm.main
```
**Training Time:**
- **GPU (RTX 4050):** ~4 minutes (3 epochs)
- **CPU (Intel Core Ultra 7):** ~50 minutes (3 epochs)

### Evaluation
```bash
#Evaluate trained model on test set
# Defaults to Multi-Task architecture (use --model-type single for baseline)
python -m financial_sentiment_llm.evaluate
```


## Resources

- [PROJECT.md](PROJECT.md) - Detailed roadmap and progress tracking
- [FinBERT Paper](https://arxiv.org/abs/1908.10063) - Financial domain BERT
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft) - For upcoming LoRA implementation

---

**Note**: This is a learning project to develop production-grade LLM and NLP skills. Documentation and results are continuously updated as experimentation progresses.
