"""Dataset loading and preprocessing utilities."""

import pandas as pd
from finsentiment.config import RAW_DATA_DIR
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from datasets import load_dataset

def load_phrasebank():
    """Load Financial PhraseBank dataset."""
    file_path = RAW_DATA_DIR / "phrasebank.csv"
    
    if file_path.exists():
        # Load from local CSV
        df = pd.read_csv(file_path)
    else:
        # Download from HuggingFace and save
        print(f"Downloading Financial PhraseBank to {file_path}...")
        dataset = load_dataset("mteb/FinancialPhrasebankClassification")
        df = pd.DataFrame(dataset['train'])
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"✓ Saved to {file_path}")
    
    df['task_type'] = 'classification'
    df['score'] = 0.0
    df['source'] = 'phrasebank'
    return df

def load_twitter():
    
    """Load Twitter financial sentiment dataset."""
    file_path = RAW_DATA_DIR / "twitter.csv"
    
    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        print(f"Downloading Twitter dataset to {file_path}...")
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")
        df = pd.DataFrame(dataset['train'])
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"✓ Saved to {file_path}")
    
    df['task_type'] = 'classification'
    df['score'] = 0.0
    df['source'] = 'twitter'
    return df

def load_fiqa(multi_task=False):
    """Load FiQA dataset with continuous scores."""
    file_path = RAW_DATA_DIR / "fiqa.csv"
    
    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        print(f"Downloading FiQA dataset to {file_path}...")
        dataset = load_dataset("TheFinAI/fiqa-sentiment-classification")
        df = pd.DataFrame(dataset['train'])
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False)
        print(f"✓ Saved to {file_path}")
    
    if multi_task:
        df['task_type'] = 'regression'
        if 'score' in df.columns:
            df['label'] = df['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))
            lower_threshold = df['score'].quantile(0.33)
            upper_threshold = df['score'].quantile(0.67)
            print(f"DEBUG FiQA Quantiles: 33%={lower_threshold:.4f}, 67%={upper_threshold:.4f}")
    else:
        df['task_type'] = 'classification'
        # Convert continuous score to discrete label
        if 'score' in df.columns:
            lower_threshold = df['score'].quantile(0.33)
            upper_threshold = df['score'].quantile(0.67)

            df['label'] = df['score'].apply(
                lambda x: 0 if x < lower_threshold else (2 if x > upper_threshold else 1)
            )
    
    df['source'] = 'fiqa'
    return df

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
    Load, balance, and combine all datasets.
    Returns train/val/test splits.
    """
    if weights is None:
        weights = {'phrasebank': 0.0, 'twitter': 0.0, 'fiqa': 1.0}
        #weights = {'phrasebank': 0.6, 'twitter': 0.15, 'fiqa': 0.25}

    print("Loading datasets...")
    phrasebank = load_phrasebank()
    twitter = load_twitter()
    fiqa = load_fiqa(multi_task=multi_task)
    
    print("Balancing datasets...")
    phrasebank_bal = balance_dataset(phrasebank)
    twitter_bal = balance_dataset(twitter)
    fiqa_bal = balance_dataset(fiqa)
    
    # Sample according to weights
    total_samples = 10000
    pb_size = int(total_samples * weights['phrasebank'])
    tw_size = int(total_samples * weights['twitter'])
    fq_size = int(total_samples * weights['fiqa'])
    
    pb_sample = phrasebank_bal.sample(n=min(pb_size, len(phrasebank_bal)), 
                                       replace=True, random_state=seed)
    tw_sample = twitter_bal.sample(n=min(tw_size, len(twitter_bal)), 
                                    replace=True, random_state=seed)
    fq_sample = fiqa_bal.sample(n=min(fq_size, len(fiqa_bal)), 
                                 replace=True, random_state=seed)
    
    # Combine
    #combined = pd.concat([pb_sample, tw_sample], ignore_index=True)
    combined = pd.concat([pb_sample, tw_sample, fq_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split
    train_df, temp_df = train_test_split(combined, test_size=0.3, 
                                          random_state=seed, stratify=combined['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                        random_state=seed, stratify=temp_df['label'])
    
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
