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
    
    df['source'] = 'phrasebank'
    df['task_type'] = 'classification'
    df['score'] = 0.0 # Placeholder
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
    
    df['source'] = 'twitter'
    df['task_type'] = 'classification'
    df['score'] = 0.0 # Placeholder
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
    
    # Use explicit thresholds for labels (needed for stratification/balancing)
    # Avoids 33% quantile forcing which mislabels neutral events
    if 'score' in df.columns:
        df['label'] = df['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))

    if multi_task:
        df['task_type'] = 'regression'
    else:
        df['task_type'] = 'classification'
    
    df['source'] = 'fiqa'
    return df

def balance_dataset(df, target_col='label'):
    """Balance dataset classes via oversampling."""
    if df.empty:
        return df
        
    class_counts = df[target_col].value_counts()
    if class_counts.empty:
        return df
        
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
        weights = {'phrasebank': 0.33, 'twitter': 0.33, 'fiqa': 0.34}
    
    print("Loading datasets...")
    phrasebank = load_phrasebank()
    twitter = load_twitter()
    fiqa = load_fiqa(multi_task=multi_task)
    
    # Sample raw data first
    total_samples = 10000
    pb_size = min(int(total_samples * weights['phrasebank']), len(phrasebank))
    tw_size = min(int(total_samples * weights['twitter']), len(twitter))
    fq_size = min(int(total_samples * weights['fiqa']), len(fiqa))
    
    pb_sample = phrasebank.sample(n=pb_size, random_state=seed)
    tw_sample = twitter.sample(n=tw_size, random_state=seed)
    fq_sample = fiqa.sample(n=fq_size, random_state=seed)
    
    # Combine
    combined = pd.concat([pb_sample, tw_sample, fq_sample], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split BEFORE balancing (Fixes data leakage)
    train_df, temp_df = train_test_split(combined, test_size=0.3, 
                                          random_state=seed, stratify=combined['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, 
                                        random_state=seed, stratify=temp_df['label'])
    
    print("Balancing datasets...")
    # Balance only the training set
    train_df = balance_dataset(train_df)
    
    # Clean up - keep only necessary columns
    required_cols = ['text', 'label', 'source', 'task_type']
    if multi_task and 'score' in combined.columns:
        required_cols.append('score')

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
