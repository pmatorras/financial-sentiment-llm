"""
Data cleaning functions for financial sentiment datasets.
Filters garbage, truncation, and low-quality samples.
"""
import pandas as pd
from finsentiment.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def is_garbage(text):
    """Check if text is likely garbage (fragment, too short, etc.)."""
    if not isinstance(text, str):
        return True
    
    text = text.strip()
    
    # Too short
    if len(text.split()) < 5:
        return True
    
    # Too much punctuation (likely a fragment)
    if len(text) > 0:
        punct_ratio = sum(c in '.,;:!?()[]{}' for c in text) / len(text)
        if punct_ratio > 0.3:
            return True
    
    # Ends with incomplete punctuation patterns
    if text.endswith((',', ';', ':')):
        # Check if it's actually a sentence or just a fragment
        if len(text.split()) < 10:
            return True
    
    return False

def clean_dataset(dataset_name = 'phrasebank'):
    """Clean a given dataset."""
    print(f"\n [X/Y] Cleaning {dataset_name}...")
    
    # Load from raw
    raw_path = RAW_DATA_DIR / f"{dataset_name}.csv"
    clean_path = PROCESSED_DATA_DIR / f"{dataset_name}_clean.csv" 
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run training first to download datasets.")
        return 0
    
    df = pd.read_csv(raw_path)
    original_count = len(df)
    
    # Filter garbage (check 'text' column)
    df['is_garbage'] = df['text'].apply(is_garbage)
    df_clean = df[~df['is_garbage']].copy()
    df_clean = df_clean.drop(columns=['is_garbage'])
    
    removed = original_count - len(df_clean)
    print(f"      Removed {removed} garbage samples ({removed/original_count*100:.1f}%)")
    print(f"      Kept {len(df_clean)} samples")
    
    # Save
    df_clean.to_csv(clean_path, index=False)
    print(f"      Saved to {clean_path}")
    
    return removed

