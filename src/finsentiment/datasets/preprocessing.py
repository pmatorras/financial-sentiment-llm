"""Dataset loading and preprocessing utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from finsentiment.datasets.load import load_any_dataset
from finsentiment.datasets.registry import DATASET_REGISTRY

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

def prepare_combined_dataset(weights=None, seed=42, multi_task=False, clean_data=False):
    """
    Load, balance (train only), and combine all datasets.
    Returns train/val/test splits.
    """
    if weights is None:
        weights = {src['name']: src['relative_weight'] for src in DATASET_REGISTRY}
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

    print("Loading datasets...")

    # Load all sources via registry
    loaded_dfs = {}
    for src in DATASET_REGISTRY:
        task = src['task_type']
        # Special case: fiqa switches based on multi_task flag
        if src['name'] == 'fiqa':
            task = 'regression' if multi_task else 'classification'
        
        loaded_dfs[src['name']] = load_any_dataset(
            dataset_name=src['name'],
            dataset_path=src['hf_path'],
            task_type=task,
            clean_data=clean_data
        )

    samples = []
    # For each source, compute max samples we could draw given its weight
    max_per_source = {name: len(loaded_dfs[name]) / weights[name] for name in weights}
    for name, df in loaded_dfs.items():
        print(f"{name}: {len(df)} rows available, weight={weights[name]}, can support total={(len(df) / weights[name]):.0f}")
    # The bottleneck determines total_samples
    total_samples = int(min(max_per_source.values()))

    # Now sample each source proportionally
    for name, weight in weights.items():
        size = int(total_samples * weight)
        df = loaded_dfs[name]
        sampled = df.sample(n=size, random_state=seed)  # no min() needed anymore
        samples.append(sampled)
        
    # Combine
    combined = pd.concat(samples, ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split
    train_df, temp_df = train_test_split(combined, test_size=0.3, random_state=seed, stratify=combined['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['label'])
    
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
