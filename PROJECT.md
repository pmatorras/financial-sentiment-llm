# Financial Sentiment LLM

## Objective
Fine-tune a lightweight LLM using LoRA/QLoRA for financial sentiment analysis, then integrate sentiment features into financial-ML pipeline.

## Learning Goals
- Understand LoRA/QLoRA parameter-efficient fine-tuning
- Implement HuggingFace Transformers training pipeline
- Design evaluation framework for NLP tasks
- Build production-ready inference system
- Automate ML feature generation with GitHub Actions

---

## Phase 1: Baseline Model
**Date Completed:** 2025-12-17

Established baseline performance with multi-dataset training approach.

### Setup & Implementation
- ✅ Multi-dataset approach: FinancialPhraseBank (60%), Twitter (15%), FiQA (25%)
- ✅ Percentile-based preprocessing for FiQA's skewed distribution
- ✅ BERT fine-tuning pipeline (3 epochs, lr=2e-5, batch_size=32)
- ✅ Reproducible training with seed=42
- ✅ Comprehensive evaluation framework with per-source metrics

### Results

**Overall Performance:**
- Test Accuracy: **86.0%**
- Macro F1: **0.86**

**Per-Class Metrics:**
| Class    | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| Negative | 0.74      | 0.96   | 0.84     |
| Neutral  | 0.95      | 0.82   | 0.88     |
| Positive | 0.92      | 0.79   | 0.85     |

**Confusion Matrix:**
```bash
          negative  neutral  positive
negative       307        6         8
neutral         44      268        14
positive        62        7       265
```

**Per-Source Performance:**
| Dataset         | Accuracy | Samples | Notes |
|-----------------|----------|---------|-------|
| PhraseBank      | 98.88%   | 624     | Excellent - professional news |
| Twitter         | 78.12%   | 224     | Good - social media style |
| FiQA            | 36.09%   | 133        | Poor - domain mismatch |

#### Key Findings

 **Strengths:**
- Strong overall performance (86%)
- Excellent on FinancialPhraseBank (professional news style)
- Good generalization to Twitter (social media style)
- High precision on neutral (0.95) and positive (0.92) classes
- Very high recall on negative class (0.96) - catches almost all negative sentiment

 **Limitations Identified:**
1. **FiQA performance is poor (36%)** - Major domain mismatch issue
   - Root cause: Long forum posts vs. short news sentences
   - Model biased toward PhraseBank's distribution (60% training weight)
   - Requires different approach (flagged for Phase 2)

2. **Class imbalance in predictions:**
   - Negative class: High recall (0.96) but lower precision (0.74)
     - 62 positive samples misclassified as negative
     - 44 neutral samples misclassified as negative
   - Positive class: High precision (0.92) but lower recall (0.79)
     - Model conservative about predicting positive

3. **Training efficiency:**
   - GPU: ~4 minutes (RTX 4050)
   - CPU: ~50 minutes (Intel Core Ultra 7)

#### Baseline Model Details
- **Architecture:** bert-base-uncased (110M parameters)
- **Model saved:** `models/sentiment_model_v2_percentile.pt`

**Phase 1 Success Criteria Met:** Baseline established with reproducible results

---

## Phase 2: Model Optimization
**Status:** In Progress (Jan 19, 2026)
> ⚠ **NOTE**: Early experiments in this section (Dec 19) contain data leakage (inflated scores). See [ Data Leakage Fix & Multi-Task Validation](#data-leakage-fix--multi-task-validation) for the first scientifically valid results.

**Drop FiQA Dataset**
- Tested training without FiQA (PhraseBank + Twitter only)
- Result: 92% overall accuracy (up from 86%)
- Confirmed FiQA is dragging down performance

**Increase FiQA Weight**
- Tested 50% FiQA weight (vs. 25% baseline)
- Result: No improvement, FiQA stayed at 36%
- Confirmed problem is dataset quality, not exposure

**FinBERT Migration**
- Switched to `ProsusAI/finbert` (financial domain pretrained)
- Result: 99.52% on PhraseBank, FiQA unchanged
- Domain pretraining helps professional news, not forum text

**Early Stopping**
- Implemented validation-based early stopping (patience=3)
- Prevents overfitting, reduces training time 40%
- Best model automatically selected at epoch 2-3

**Current Baseline:** FinBERT + early stopping
- Overall: 86% | PhraseBank: 99.52% | Twitter: 78.57% | FiQA: 34.59%
- Positive recall: 0.95 

### Data Leakage Fix & Multi-Task Validation
**Date:** Jan 03, 2026
**Goal:** Establish valid baseline (no leakage) and validate Multi-Task hypothesis.

**1. Leakage Fix:**
- Refactored pipeline (Split → Balance) to remove duplicate samples from Test set.

**2. Architecture Comparison (Post-Fix):**
Comparing Single-Task (Classification) vs Multi-Task (Class + Regression) on the fixed data.

| Model | FiQA Accuracy | Overall Accuracy | Note |
|-------|---------------|------------------|------|
| **Single-Task** | 65.5% | 83.0% | Struggles with continuous scores |
| **Multi-Task** | **80.2%** | **85.0%** | **+15% on FiQA** |

**Conclusion:** Multi-Task is the superior architecture.

**Loss weighting (implementation detail):**
- Multi-task objective: `classification_weight * CrossEntropy + regression_weight * MSE`.
- Original validated baseline used `0.5 / 5.0`.  
- Defaults moved to `classification_weight=1.0`, `regression_weight=10.0` primarily for implementation simplicity when running the unified “classification-only” path (so classification uses the standard scaling without needing special-case logic), while preserving the same loss ratio. Retraining showed comparable evaluation metrics.


> **Data cleaning note:** Cleaning/filters were evaluated but did not consistently improve accuracy when training and testing on the same distribution, so cleaning remains available via flags but is disabled by default. (See [EXPERIMENTS.md](EXPERIMENTS.md).)

**Data Pipeline Refactoring (Issue #26)**
- Refactored sampling logic to enforce explicit dataset weights
- Discovered and fixed bottleneck in weight enforcement that limited training data
- Established fixed train/test splits (prevents evaluation variance across experiments)
- Validated optimal configuration: 2:1 Twitter/PhraseBank ratio, no class balancing
- Result: Maintains 85% accuracy with scientifically rigorous methodology
> See [EXPERIMENTS.md](EXPERIMENTS.md#dataset-weighting-and-pipeline-refactoring) for detailed information.

**Model Selection (Completed Jan 20, 2026):**
- Validated FinBERT vs BERT vs DistilBERT on fixed multi-task pipeline.
- **Winner:** FinBERT (85% accuracy, +3% on professional news).
- **Decision:** Proceed with FinBERT as the base model. DistilBERT efficiency gains did not materialize (training was slower), and accuracy drop (-5% on news) was too high.

**Current Focus:** Implement LoRA (Low-Rank Adaptation) on FinBERT.
- Goal: Match full fine-tuning performance (85%) with <1% trainable parameters.
- Hypothesis: LoRA will reduce checkpoint size and allow faster iteration.


### Potential Future Improvements

**Multi-Task Learning** -> Done
- Dual-head architecture: classification + regression
- Use FiQA continuous scores (avoid percentile binning)
- Goal: Improve FiQA to 50%+ or document exclusion

**Advanced Techniques:**
- Learning rate scheduling (warmup + decay)
- LoRA/QLoRA parameter-efficient fine-tuning
- Class weight balancing

**Analysis:**
- Error analysis on FiQA misclassifications
- Attention visualization for failing examples

**Alternative Models:** -> Done
- DistilBERT (faster, smaller)
- Separate FiQA-specific model (if multi-task fails)

**Success Criteria for Phase 2:**
- Overall accuracy ≥ 87%
- FiQA accuracy ≥ 60% (or documented decision to exclude)
- Training time < 30min with GPU optimization
**Success**: Fine-tuned model beats baseline significantly (target: 85%+ accuracy)

## Phase 3: Deploy & Integrate
Make it useful for financial-ML
- Build inference pipeline
- Generate sentiment scores for stock data
- Integrate into financial-ML as features
- Measure if it actually helps predictions

**Success**: Sentiment features flowing into financial-ML pipeline

## Phase 4: Automate (Optional)
Only if Phase 3 proves valuable
- GitHub Actions for daily sentiment updates
- Productionize the pipeline



---

## Key Resources
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft)
- [Financial PhraseBank dataset](https://huggingface.co/datasets/takala/financial_phrasebank)
- [Philipp Schmid: Fine-tune LLMs in 2025](https://www.philschmid.de/fine-tune-llms-in-2025)

---

## Progress Log

**2025-12-17**: Phase 1 completed
- Implemented multi-dataset training pipeline
- Achieved 85% test accuracy baseline
- Identified FiQA domain mismatch as key challenge
- Ready for Phase 2 experimentation

**2025-12-16**: Repository created, roadmap defined
**2026-01-03**: Data Leakage Fix \& Multi-Task Architecture

- **Critical Fix:** Discovered and fixed data leakage (upsampling before train/test split)
- Implemented Multi-Task Learning (dual-head: classification + regression)
- **Breakthrough:** FiQA accuracy jumped from 65% → 80% (+15%) using regression head
- Established scientifically valid baseline: 85% overall accuracy

**2026-01-12**: Data Cleaning Ablation Study

- Tested aggressive data cleaning (filtering short/noisy samples)
- **Finding:** Cleaning did not improve generalization on matched distributions
- **Decision:** Kept cleaning utilities but disabled by default

**2026-01-19**: Dataset Pipeline Refactoring

- Implemented fixed train/val/test splits (70/15/15) per source
- Established explicit dataset weighting controls (Twitter:PhraseBank = 2:1)
- Decoupled classification and regression sampling pipelines
- **Result:** Reproducible 85% baseline with fixed test set (1,896 samples)

**2026-01-20**: Model Selection Completed
- Benchmarked FinBERT, BERT-Base, and DistilBERT.
- Confirmed FinBERT + Multi-Task is the superior configuration (85% accuracy).
- Refactored CLI to support dynamic model selection and logging.

