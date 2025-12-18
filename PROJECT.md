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

## Phase 2: Fine-tune
**Priority 1 - Address FiQA Performance:**
- [ ] Exp 2.1: Drop FiQA, train only on PhraseBank+Twitter (cleaner baseline)
- [ ] Exp 2.2: Increase FiQA weight to 40-50% (more exposure)
- [ ] Exp 2.3: Train separate FiQA-specific model (domain adaptation)
- [ ] Exp 2.4: Multi-task learning with regression head for FiQA scores

**Priority 2 - Better Base Model:**
- [ ] Exp 2.5: Switch to FinBERT (`ProsusAI/finbert`) - financial domain pretrained
- [ ] Exp 2.6: Try DistilBERT (faster training, smaller model)

**Priority 3 - Training Improvements:**
- [ ] Exp 2.7: Early stopping to prevent overfitting
- [ ] Exp 2.8: Learning rate scheduling (warmup + decay)
- [ ] Exp 2.9: Class weights instead of oversampling
- [ ] Exp 2.10: Apply LoRA/QLoRA for parameter-efficient fine-tuning

**Priority 4 - Analysis & Understanding:**
- [ ] Error analysis: Manual review of FiQA misclassifications
- [ ] Confusion matrix deep-dive per data source
- [ ] Attention visualization for failing examples

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
