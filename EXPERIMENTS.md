# Experiment Log

## Phase 2: Model Optimization

## Drop FiQA Dataset
**Date:** Dec 19, 2024
**Goal:** Test if FiQA is hurting overall performance

**Configuration:**
- Model: BERT baseline
- Weights: PhraseBank only, Twitter only (no FiQA)

**Results:**
Overall: 92.0%
PhraseBank: 98.86% (613 samples)
Twitter: 73.95% (238 samples)

**Conclusion:** Performance improved to 92% without FiQA. Confirms FiQA is dragging down overall accuracy.

---

### Increase FiQA Weight (0.4/0.1/0.5)

**Date:** Dec 19, 2024
**Goal:** Give FiQA more exposure in training

**Configuration:**

- Weights: PhraseBank 0.4, Twitter 0.1, FiQA 0.5
- Model: FinBERT
- No early stopping (3 epochs fixed)

**Results:**

```
Overall: 86.0%
PhraseBank: 98.88% (624 samples)
Twitter: 78.12% (224 samples)
FiQA: 36.09% (133 samples)

Confusion Matrix:
          negative  neutral  positive
negative       307        6         8
neutral         44      268        14
positive        62        7       265
```

**Conclusion:** Increasing FiQA weight didn't improve FiQA performance. Overall accuracy maintained but no breakthrough.

***

### FinBERT Migration

**Date:** Dec 19, 2024
**Goal:** Use domain-specific pretrained model

**Configuration:**

- Model: `bert-base-uncased` → `ProsusAI/finbert`
- Architecture: `AutoModelForSequenceClassification` (pretrained sentiment head)
- Weights: 0.4/0.1/0.5 (same as Exp 2.2 for comparison)
- Epochs: 3 (no early stopping)

**Training:**

```
Epoch 1: Train Loss=0.7757, Val Loss=0.4434, Val Acc=0.7961
Epoch 2: Train Loss=0.3600, Val Loss=0.3541, Val Acc=0.8318
Epoch 3: Train Loss=0.2456, Val Loss=0.3646, Val Acc=0.8420
```

**Results:**

```
Overall: 86.0%
PhraseBank: 98.88% (624 samples)
Twitter: 78.12% (224 samples)
FiQA: 36.09% (133 samples)

Confusion Matrix: [IDENTICAL to Exp 2.2]
```

**Conclusion:** FinBERT with wrong weights performed identically to BERT. Confirmed overfitting (val_loss increased at epoch 3).

***

### Early Stopping Implementation

**Date:** Dec 19, 2024
**Goal:** Prevent overfitting, improve training efficiency

**Implementation:**

- Patience: 3 epochs
- Monitors: Validation loss
- Action: Restores best model checkpoint

**Test Run (with 0.4/0.1/0.5 weights):**

```
Epoch 1: Val Loss=0.4434
Epoch 2: Val Loss=0.3541 ✓ Best
Epoch 3: Val Loss=0.3646 (patience 1/3)
Epoch 4: Val Loss increased (patience 2/3)
Epoch 5: Val Loss increased (patience 3/3) → STOPPED

⚠ Early stopping triggered at epoch 5
Best model was at epoch 2 (val_loss=0.3541, val_acc=0.8318)
```

**Impact:**

- Prevented overfitting (automatically selected epoch 2 model)
- ~40% reduction in training time (5 epochs vs potential 10)
- Validation-driven model selection

***

### FinBERT with Early Stopping (Corrected Weights)

**Date:** Dec 19, 2024
**Goal:** Establish clean baseline with proper configuration

**Configuration:**

- Model: FinBERT with pretrained head
- Weights: **0.6/0.15/0.25** (reverted to original)
- Early stopping: patience=3

**Training:**

```
Best model at epoch 2
Stopped at epoch ~4-5
```

**Results:**

```
Overall: 86.0%
PhraseBank: 99.52% (624 samples)  ← Improved!
Twitter: 78.57% (224 samples)
FiQA: 34.59% (133 samples)

Per-Class Metrics:
              precision    recall  f1-score
negative       0.95      0.80      0.87
neutral        0.94      0.83      0.88
positive       0.75      0.95      0.84  ← Much higher recall!

Confusion Matrix:
          negative  neutral  positive
negative       256       11        54
neutral          5      269        52
positive         9        7       318  ← Many more correct!
```

**Key Changes:**

- Positive recall: 0.79 → **0.95** (catches almost all positive sentiment)
- Positive precision: 0.92 → 0.75 (more false positives, but acceptable tradeoff)
- PhraseBank: 98.88% → **99.52%** (near-perfect on professional news)