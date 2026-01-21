# Experiment Log

## Phase 2: Model Optimization
>WARNING (Jan 03, 2026):
Results in this section (Dec 19 experiments) were obtained using a preprocessing pipeline that contained Data Leakage (Upsampling was applied before the Train/Test split).
While the accuracy on FiQA remained low (36%), the PhraseBank/Twitter scores may be inflated due to duplicate samples appearing in both sets.
See [Multi-Task Learning & Data Leakage Fix](#multi-task-learning--data-leakage-fix) section for the corrected baseline.

### Drop FiQA Dataset
**Date:** Dec 19, 2024 \
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

**Date:** Dec 19, 2024 \
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

**Date:** Dec 19, 2024 \
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

***

### Multi-Task Learning & Data Leakage Fix

**Date:** Jan 03, 2026
**Goal:** Fix methodological errors (leakage) and improve FiQA performance using regression.

**1. The Methodological Fix (Data Leakage):**
*   **Issue:** Previous pipeline upsampled data *before* splitting train/test. This caused "clones" (identical samples) to appear in both sets.
*   **Fix:** Refactored pipeline to Split → Stratify → Balance (Train Only).
*   **Impact:** Test set is now purely unseen data. "Real" baseline established.

**2. Multi-Task Architecture:**
*   **Hypothesis:** FiQA is regression data (continuous scores). Forcing it into classification bins loses information.
*   **Solution:** Multi-Task Head (Classification + Regression).
*   **Loss:** `CrossEntropy + 5 * MSE` (Regression weighted higher).

**Results (Post-Fix):**

| Dataset | Single-Task (Baseline) | Multi-Task (Final) | Delta |
|---------|------------------------|--------------------|-------|
| **FiQA** | 65.52% | **80.17%** | **+14.6%** |
| PhraseBank | 93.29% | 92.13% | -1.1% |
| Twitter | 79.76% | 81.16% | +1.4% |
| **Overall** | 83.0% | **85.0%** | +2.0% |

**Conclusion:**
*   **Leakage Fixed:** Single-task FiQA score dropped to 65% (realistic baseline).
*   **Hypothesis Confirmed:** Multi-task learning drastically improved FiQA performance (+15%) by respecting the continuous nature of the data.
*   **Robustness:** The model now generalises to complex financial text instead of just memorizing it.


***
### Data cleaning ablation (clean vs not-clean)

**Date:** Jan 12, 2026 \
**Goal:** Verify whether dataset “cleaning” (e.g., filtering short texts) improves real generalization, vs. just changing the evaluation distribution.

**Setup (important):**

- Two dataset “recipes” were used:
    - **Unclean**: full dataset (no filtering).
    - **Clean**: filtered dataset (e.g., short/noisy items removed).
- Key lesson: results can look better/worse depending on whether the **model was evaluated on the same recipe it was trained on** (matched distribution) vs. on a filtered subset (mismatched distribution).

**Results (multi-task evaluation):**


| Train recipe | Test recipe | Overall acc | PhraseBank | Twitter | FiQA | Test N |
| :-- | :-- | --: | --: | --: | --: | --: |
| Clean | Clean | 0.85 | 94.87% | 78.69% | 80.81% | 952 |
| Unclean | Clean | 0.85 | 96.58% | 76.49% | 89.90% | 952 |
| Unclean | Unclean | 0.84 | 92.42% | 80.16% | 79.31% | 958 |

**Interpretation:**

- The “Unclean → Clean” jump on FiQA (89.90%) is likely a **subset effect**: the cleaned test set is not the same distribution as the full unclean test set, and appears easier (or at least different).
- When comparing matched distributions (Clean→Clean vs Unclean→Unclean), cleaning does **not** deliver a clear, reliable improvement (0.85 vs 0.84 overall; FiQA ~80 either way).
- Conclusion: data cleaning here risks “chasing shadows” (changing which examples exist) rather than improving the model.

**Decision:**

- Keep the cleaning utilities/flags for reproducibility and future targeted filtering, but default them to **off** (not used in the main pipeline), since no consistent improvement was observed on matched train/test distributions.

***

### Dataset Weighting and Pipeline Refactoring

**Date:** Jan 19, 2026 \
**Goal:** Refactor sampling logic to support explicit dataset weighting and establish fixed train/test splits for reproducible experiments.

**Motivation:**

- Previous pipeline had implicit/unclear weighting behavior.
- Test set composition varied between experiments (makes comparisons unreliable).
- Need scientific rigor: fixed test set + explicit control over source representation.

**The Refactoring:**

* **Fixed Splits:** Changed pipeline to split **each source** into train/val/test (70/15/15) **before** applying any sampling. This ensures test set is identical across all experiments.
* **Explicit Weighting:** Implemented clear weight configuration in `DATASET_REGISTRY` to control source representation.
* **Decoupled Pipelines:** Split logic into two independent streams:
    * Classification Bucket: PhraseBank + Twitter (weighted relative to each other).
    * Regression Bucket: FiQA (independent).
* **Discovered Issue:** Initial implementation created a bottleneck where classification sources were artificially limited by the regression source size. Fixed by making pipelines truly independent.

**Experiment: Weight \& Balance Comparison**

Tested configurations on fixed test set (1,896 samples: 340 PhraseBank / 1,432 Twitter / 124 FiQA):


| Config | Overall | PhraseBank | Twitter | FiQA | Train N |
| :-- | --: | --: | --: | --: | --: |
| 1:1 + Balanced | 76.0% | 96.8% | 71.0% | 75.0% | 5,300 |
| 2:1 + Balanced | 84.0% | 95.3% | 82.1% | 75.8% | 8,400 |
| **2:1 + Unbalanced** | **85.0%** | **95.6%** | **82.8%** | **76.6%** | **8,400** |

**Key Findings:**

* **Optimal Weight Ratio:** 2:1 (Twitter:PhraseBank) provides best balance. 1:1 starved the model of Twitter diversity (Twitter Neutral recall dropped to 0.30).
* **Class Balancing Hurts:** Artificial oversampling degraded metrics (-1% overall). Natural positive-skewed distribution matches real-world priors.
* **Fixed Splits Enable Comparison:** Can now reliably compare experiments without test set variance.

**Latest Configuration:**

```python
# Dataset weights (DATASET_REGISTRY)
phrasebank: 1.0
twitter: 2.0
fiqa: 1.0

# Pipeline settings
- Split: 70% train / 15% val / 15% test (per-source, before sampling)
- Class balancing: OFF
- Sampling: Decoupled (classification/regression independent)
```

**Conclusion:** Refactored pipeline with scientific rigor. Validated 85% overall accuracy (95.6% PhraseBank, 82.8% Twitter) with reproducible methodology and explicit dataset control.

***

### Baseline Model Comparison (FinBERT vs BERT vs DistilBERT)

**Date:** Jan 20, 2026
**Goal:** Measure the pure impact of financial domain pre-training vs generic pre-training on the fixed, leak-free dataset.

**Configuration:**
- **Architecture:** Multi-Task (Class + Regression) for all models.
- **Dataset:** Fixed split (2:1 Twitter/PhraseBank ratio).
- **Training:** Same hyperparameters (LR=2e-5, Batch=32) for all.

**Results:**

| Model | Overall | PhraseBank (News) | Twitter (Social) | FiQA (Forum) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FinBERT** | **85.0%** | **95.6%** | **82.8%** | **76.6%** | **Best overall & on professional text** |
| BERT-Base | 83.0% | 92.7% | 81.9% | 75.0% | Strong baseline, but lags in domain tasks |
| DistilBERT | 82.0% | 90.3% | 80.7% | 75.0% | Efficient, but significant drop on PhraseBank (-5%) |

**Key Findings:**
1.  **Domain Value Confirmed:** FinBERT outperforms generic BERT by **+2.0% overall**. The gap is most dramatic on **PhraseBank (+2.9%)**, proving that financial pre-training captures professional financial nuance better than generic pre-training.
2.  **Efficiency Trade-off:** DistilBERT is ~40% smaller but drops **5.3% accuracy** on professional news compared to FinBERT.
    *   *Observation:* Training time was unexpectedly 2x longer for DistilBERT in this environment, negating its theoretical speed advantage (likely an implementation/dataloader bottleneck).
3.  **Architecture vs Pre-training:** The FiQA scores are clustered closely (75-76%) across all three models. This reinforces that **Multi-Task architecture** (treating FiQA as regression) is the primary driver of performance here, not the underlying pre-trained weights.

**Conclusion:**
**FinBERT** remains the superior choice. The accuracy gain on professional text is significant enough to justify the standard model size over DistilBERT.

***

### LoRA Implementation \& Tuning

**Date:** Jan 21, 2026 \
**Goal:** Implement Parameter-Efficient Fine-Tuning (LoRA) to reduce training cost and storage size while maintaining FinBERT's performance.

**Configurations Tested:**
All LoRA models used the same frozen FinBERT backbone.

1. **`finbert-lora` (Baseline):** Rank `r=8`, Alpha `16`, Targets `["query", "value"]`.
2. **`finbert-lora-r64` (High Capacity):** Rank `r=64`, Alpha `128`, Targets `["query", "key", "value", "dense"]`.
3. **`finbert-lora-tuned` (Balanced):** Rank `r=16`, Alpha `32`, Targets `["query", "key", "value", "dense"]`.
4. **`finbert-lora-tuned-weighted`:** Same as Balanced but with `regression_weight=20.0`.
5. **`finbert-lora-r64-weighted` (Max Resources):** Rank `r=64`, `regression_weight=20.0`.

**Results:**


| Model | Rank | Weight | Epochs | Overall Acc | PhraseBank | Twitter | FiQA (Reg) | Speed/Epoch |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| **Full FinBERT** | N/A | 10.0 | 16 | **85.4%** | 95.9% | **83.3%** | **81.5%** | ~105s |
| LoRA Baseline | 8 | 10.0 | 15 | 70.0% | 89.4% | 65.2% | 71.8% | **~55s** |
| LoRA High-Cap | 64 | 10.0 | 9 | 85.0% | 95.3% | **84.7%** | 66.1% | ~100s |
| **LoRA Tuned** | **16** | **10.0** | **10** | **83.2%** | **97.1%** | 80.5% | 72.6% | **~80s** |
| LoRA Tuned+W | 16 | 20.0 | 8 | 78.0% | 94.1% | 74.1% | 75.0% | ~80s |
| LoRA Max Res | 64 | 20.0 | 9 | 84.4% | **97.7%** | 82.5% | 68.6% | ~100s |

**Key Findings:**

1. **Classification Parity:** For pure classification tasks, LoRA is effectively equivalent to Full Fine-Tuning. The `r=16` model outperformed FinBERT on News (PhraseBank), and the `r=64` model outperformed FinBERT on Social Media (Twitter).
2. **Multi-Head Divergence:** The Multi-Head architecture reveals a critical limitation of LoRA. While it handles the discrete classification heads easily, it consistently fails to match Full FinBERT on the continuous regression head (FiQA).
    * **Low Rank (`r=16`)**: Underfits the regression complexity (72% vs 81%).
    * **High Rank (`r=64`)**: Overfits the small regression dataset (66% vs 81%).
    * **Weighting**: Aggressive loss weighting (`20.0`) failed to close this gap and instead destabilized the classification tasks.
3. **Efficiency Analysis:**
    * **Storage:** LoRA is a clear winner (5MB vs 420MB).
    * **Time:** The training speed advantage diminishes as Rank increases. At `r=64`, LoRA is nearly as slow as full fine-tuning, negating the time benefit.

**Conclusion:**
LoRA is a viable replacement for FinBERT **only if the use case is restricted to Classification**. In these scenarios, it delivers comparable or superior accuracy with massive storage savings. However, for a true **Multi-Task** system that relies on high-quality Regression output (e.g., FiQA scores), LoRA's constrained parameter space is insufficient to model the continuous output distribution effectively. For robust Multi-Head performance, **Full Fine-Tuning remains the required approach.**