"""Dataset loading and preprocessing utilities."""

from datasets import load_dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import pandas as pd


def load_financial_phrasebank(split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)):
    """
    Load Financial PhraseBank dataset and create train/val/test splits.
    
    Args:
        split_ratios: Tuple of (train, val, test) ratios. Default (0.7, 0.15, 0.15)
        
    Returns:
        Dictionary with 'train', 'val', 'test' splits
    """
    # Load dataset from HuggingFace
    # Use 'sentences_allagree' config for highest quality labels
    dataset = load_dataset("mteb/FinancialPhrasebankClassification")
    
    # Convert to pandas for easier manipulation
    df = pd.DataFrame(dataset['train'])
    
    # First split: train vs (val + test)
    train_ratio, val_ratio, test_ratio = split_ratios
    train_df, temp_df = train_test_split(
        df, 
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=df['label']  # Maintain class balance
    )
    
    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_test_ratio),
        random_state=42,
        stratify=temp_df['label']
    )
    
    print(f"Dataset loaded:")
    print(f"  Train: {len(train_df)} examples")
    print(f"  Val:   {len(val_df)} examples")
    print(f"  Test:  {len(test_df)} examples")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }


def get_label_mapping():
    """Return mapping of label indices to names."""
    return {
        0: "negative",
        1: "neutral", 
        2: "positive"
    }


if __name__ == "__main__":
    # Quick test
    splits = load_financial_phrasebank()
    print("\nClass distribution:")
    for split_name, split_df in splits.items():
        print(f"\n{split_name.upper()}:")
        print(split_df['label'].value_counts().sort_index())
