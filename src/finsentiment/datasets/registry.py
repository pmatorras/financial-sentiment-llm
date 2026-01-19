DATASET_REGISTRY = [
    {
        'name': 'phrasebank',
        'hf_path': 'mteb/FinancialPhrasebankClassification',
        'task_type': 'classification',
        'relative_weight': 1.0,
    },
    {
        'name': 'twitter',
        'hf_path': 'zeroshot/twitter-financial-news-sentiment',
        'task_type': 'classification',
        'relative_weight': 2.0,
    },
    {
        'name': 'fiqa',
        'hf_path': 'TheFinAI/fiqa-sentiment-classification',
        'task_type': 'regression',  # or determined by multi_task flag
        'relative_weight': 1.0,
    },
]
