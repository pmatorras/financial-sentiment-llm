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
    
    if multi_task:
        df['task_type'] = 'regression'
        # Explicit thresholds for label creation (used for stratification/balancing)
        if 'score' in df.columns:
            df['label'] = df['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))
    else:
        df['task_type'] = 'classification'
        # Convert continuous score to discrete label
        # UPDATED: Use explicit thresholds instead of quantiles to avoid zero-spike issues
        if 'score' in df.columns:
            df['label'] = df['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))
    
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
    Prepare combined dataset with proper train/val/test split.
    
    CRITICAL SCIENTIFIC FIX: 
    1. We generate 'labels' for FiQA (even if regression) ONLY to enable Stratified Splitting and Balancing.
    2. We Split BEFORE Balancing to prevent Data Leakage (identical samples in Train/Test).
    3. We use explicit thresholds (-0.1, 0.1) for FiQA to respect the sentiment physics, 
       rather than forcing a 33% split which mislabels neutral events.
    """
    if weights is None:
        # Default weights favoring high-quality data (FiQA/PB) while using Twitter for volume.
        weights = {'phrasebank': 0.33, 'twitter': 0.33, 'fiqa': 0.34}
    
    # 1. Load raw (imbalanced) datasets
    phrasebank = load_phrasebank()
    twitter = load_twitter()
    fiqa = load_fiqa(multi_task=multi_task)
    
    # --- FIQA SPECIAL HANDLING ---
    # We need labels for stratification and balancing, even if the task is regression.
    if multi_task and 'score' in fiqa.columns:
        print("\\n--- FiQA Physics Check ---")
        # Generate labels based on Explicit Thresholds (Science) vs Quantiles (Distribution)
        # We stick to the Explicit Thresholds (-0.1, 0.1) for truth, but print quantiles for awareness.
        lower_q = fiqa['score'].quantile(0.33)
        upper_q = fiqa['score'].quantile(0.67)
        print(f"DEBUG: Natural Data Quantiles -> 33%: {lower_q:.4f}, 67%: {upper_q:.4f}")
        print(f"DEBUG: Using Explicit Thresholds -> Negative < -0.1, Positive > 0.1")
        
        # Apply the explicit thresholds to create the stratification label
        fiqa['label'] = fiqa['score'].apply(lambda x: 0 if x < -0.1 else (2 if x > 0.1 else 1))
    
    # 2. Sample according to weights (keeping raw/imbalanced distribution)
    total_samples = 10000
    pb_raw = phrasebank.sample(
        n=min(int(total_samples * weights['phrasebank']), len(phrasebank)), 
        random_state=seed
    )
    tw_raw = twitter.sample(
        n=min(int(total_samples * weights['twitter']), len(twitter)), 
        random_state=seed
    )
    fq_raw = fiqa.sample(
        n=min(int(total_samples * weights['fiqa']), len(fiqa)), 
        random_state=seed
    )
    
    # 3. Combine raw datasets
    combined_raw = pd.concat([pb_raw, tw_raw, fq_raw], ignore_index=True)
    
    print(f"\\nRaw Dataset Statistics:")
    print(f"Total samples: {len(combined_raw)}")
    if 'label' in combined_raw.columns:
        print("Label distribution (Imbalanced / Authentic):")
        print(combined_raw['label'].value_counts().sort_index())
    
    # 4. SPLIT FIRST (Stratified)
    # This prevents 'Twin Leakage' where a copy of a sample ends up in both Train and Test.
    stratify_col = combined_raw['label'] if 'label' in combined_raw.columns else None
    
    train_df, temp_df = train_test_split(
        combined_raw, 
        test_size=0.3, 
        random_state=seed, 
        stratify=stratify_col
    )
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=seed, 
        stratify=temp_df['label'] if 'label' in temp_df.columns else None
    )
    
    print(f"\\nAfter Split (Before Balancing):")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 5. BALANCE ONLY TRAINING DATA
    # We upsample the Training set to help the model learn rare classes.
    # We LEAVE Validation/Test sets alone to measure performance on real-world distributions.
    print("\\nBalancing Training Set (Upsampling minority classes)...")
    train_df = balance_dataset(train_df)
    
    print(f"\\nFinal Dataset Shapes:")
    print(f"Train: {len(train_df)} (Balanced)")
    print(f"Val:   {len(val_df)} (Authentic Imbalance)")
    print(f"Test:  {len(test_df)} (Authentic Imbalance)")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
