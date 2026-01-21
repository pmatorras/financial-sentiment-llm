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
    source_tasks = {}
    for src in DATASET_REGISTRY:
        task = src['task_type']
        # Special case: fiqa switches based on multi_task flag
        if src['name'] == 'fiqa':
            task = 'regression' if multi_task else 'classification'
        source_tasks[src['name']] = task
        loaded_dfs[src['name']] = load_any_dataset(
            dataset_name=src['name'],
            dataset_path=src['hf_path'],
            task_type=task,
            clean_data=clean_data
        )
    # Split each source into train/val/test BEFORE sampling (for consistency across tests)
    train_pool = {}
    val_pool = {}
    test_pool = {}
    
    for name, df in loaded_dfs.items():
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed, stratify=df['label'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df['label'])
        train_pool[name] = train_df
        val_pool[name] = val_df
        test_pool[name] = test_df

    samples = []
    # Split sources by task type
    cls_sources = [n for n, t in source_tasks.items() if t == 'classification']
    reg_sources= [n for n, t in source_tasks.items() if t == 'regression']

    # --- Pipeline 1: Classification ---
    if cls_sources:
        # Re-normalize weights for just these sources
        cls_sub_weights = {k: weights[k] for k in cls_sources}
        total_cls_weight = sum(cls_sub_weights.values())
        
        # Calculate limit based on weakest link
        cls_maxes = [len(train_pool[k]) / (cls_sub_weights[k]/total_cls_weight) for k in cls_sources]
        cls_limit = int(min(cls_maxes))
        
        for name in cls_sources:
            # Calculate proportion relative to this group
            relative_w = cls_sub_weights[name] / total_cls_weight
            size = int(cls_limit * relative_w)
            samples.append(train_pool[name].sample(n=size, random_state=seed))
            print(f"  > Classification ({name}): {size} samples")

    # --- Pipeline 2: Regression ---
    if reg_sources:
        # Same logic for regression group
        reg_sub_weights = {k: weights[k] for k in reg_sources}
        total_reg_weight = sum(reg_sub_weights.values())
        
        reg_maxes = [len(train_pool[k]) / (reg_sub_weights[k]/total_reg_weight) for k in reg_sources]
        reg_limit = int(min(reg_maxes))
        
        for name in reg_sources:
            relative_w = reg_sub_weights[name] / total_reg_weight
            size = int(reg_limit * relative_w)
            samples.append(train_pool[name].sample(n=size, random_state=seed))
            print(f"  > Regression ({name}):     {size} samples")
        
    # Combine sampled train data
    train_df = pd.concat(samples, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    # Combine full val/test pools
    val_df = pd.concat(val_pool.values(), ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_pool.values(), ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

    
    print("Balancing datasets...")
    #train_df = balance_dataset(train_df)
    
    # Clean up - keep only necessary columns
    required_cols = ['text', 'label', 'source']
    if multi_task:
        required_cols.extend(['score', 'task_type'])
        
    # Add continuous_score if it exists and has values
    if 'continuous_score' in train_df.columns and train_df['continuous_score'].notna().any():
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
