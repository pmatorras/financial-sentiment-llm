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

## High-Level Roadmap

### Phase 1: Learn & Baseline (Current)
Get comfortable with the tools and establish baseline performance
- Setup environment and load Financial PhraseBank dataset
- Exploratory data analysis
- Run zero-shot baseline model (Phi-2 or LLaMA-3-8B)
- Implement evaluation metrics

**Success**: Understand the data, have baseline accuracy number

### Phase 2: Fine-tune
Apply LoRA/QLoRA and improve performance
- Configure and train with LoRA
- Hyperparameter experimentation
- Compare against baseline

**Success**: Fine-tuned model beats baseline significantly (target: 85%+ accuracy)

### Phase 3: Deploy & Integrate
Make it useful for financial-ML
- Build inference pipeline
- Generate sentiment scores for stock data
- Integrate into financial-ML as features
- Measure if it actually helps predictions

**Success**: Sentiment features flowing into financial-ML pipeline

### Phase 4: Automate (Optional)
Only if Phase 3 proves valuable
- GitHub Actions for daily sentiment updates
- Productionize the pipeline

---

## 📋 Current Sprint

**Focus**: Phase 1 - Dataset & Baseline

- [ ] Setup repo structure and install dependencies
- [ ] Load Financial PhraseBank from HuggingFace
- [ ] Basic EDA: class distribution, sample texts
- [ ] Train/val/test split
- [ ] Run zero-shot inference with pre-trained model
- [ ] Calculate baseline accuracy

**Next steps will be defined after completing this sprint**

---

## 📚 Key Resources
- [HuggingFace PEFT docs](https://huggingface.co/docs/peft)
- [Financial PhraseBank dataset](https://huggingface.co/datasets/takala/financial_phrasebank)
- [Philipp Schmid: Fine-tune LLMs in 2025](https://www.philschmid.de/fine-tune-llms-in-2025)

---

## 📝 Progress Log
**2025-12-16**: Repository created, roadmap defined
