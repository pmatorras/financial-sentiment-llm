"""Dataset loading and preprocessing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from finsentiment.datasets.load import (
    #load_phrasebank,
    #load_twitter,
    load_fiqa,
    load_any_dataset
)


def balance_dataset(df, target_col='label'):
    """Balance dataset classes via oversampling."""
    class_counts = df[target_col].value_counts()
    max_count = class_counts.max()
    
    balanced_dfs = []
    for label in df[target_col].unique():
        class_df = df[df[target_col] == label]
        class_df_upsampled = resample(
            class_df,
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        balanced_dfs.append(class_df_upsampled)
    
    return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1, random_state=42)

def prepare_combined_dataset(weights=None, seed=42, multi_task=False):
    """
    Load, balance (train only), and combine all datasets.
    Returns train/val/test splits.
    """
    if weights is None:
        #weights = {'phrasebank': 0.0, 'twitter': 0.0, 'fiqa': 1.0}
        weights = {'phrasebank': 0.33, 'twitter': 0.33, 'fiqa': 0.34}

    print("Loading datasets...")
    #phrasebank = load_phrasebank()
    #twitter = load_twitter()
    phrasebank = load_any_dataset(dataset_name='phrasebank', dataset_path='mteb/FinancialPhrasebankClassification', task_type='classification')
    twitter = load_any_dataset(dataset_name='twitter', dataset_path= 'zeroshot/twitter-financial-news-sentiment', task_type='classification')

    fiqa = load_fiqa(multi_task=multi_task)
    fiqa_type = 'regression' if multi_task else 'classification'
    #fiqa = load_any_dataset(dataset_name='fiqa', dataset_path='TheFinAI/fiqa-sentiment-classification', task_type=fiqa_type )

    # Sample raw data according to weights
    total_samples = 10000
    pb_size = int(total_samples * weights['phrasebank'])
    tw_size = int(total_samples * weights['twitter'])
    fq_size = int(total_samples * weights['fiqa'])
    
    pb_sample = phrasebank.sample(n=min(pb_size, len(phrasebank)), 
                                   random_state=seed)
    tw_sample = twitter.sample(n=min(tw_size, len(twitter)), 
                                random_state=seed)
    fq_sample = fiqa.sample(n=min(fq_size, len(fiqa)), 
                             random_state=seed)
    
    # Combine
    #combined = pd.concat([pb_sample, tw_sample], ignore_index=True)
    combined = pd.concat([pb_sample, tw_sample, fq_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split
    train_df, temp_df = train_test_split(combined, test_size=0.3, 
                                          random_state=seed, stratify=combined['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                        random_state=seed, stratify=temp_df['label'])
    
    print("Balancing datasets...")
    train_df = balance_dataset(train_df)
    
    # Clean up - keep only necessary columns
    required_cols = ['text', 'label', 'source']
    if multi_task:
        required_cols.extend(['score', 'task_type'])
    # Add continuous_score if it exists and has values
    if 'continuous_score' in combined.columns and combined['continuous_score'].notna().any():
        required_cols.append('continuous_score')
    
    train_df = train_df[required_cols].copy()
    val_df = val_df[required_cols].copy()
    test_df = test_df[required_cols].copy()
    
    print(f"\nDataset prepared:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
