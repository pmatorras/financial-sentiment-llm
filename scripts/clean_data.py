"""
scripts/clean_data.py
Remove garbage samples (short sentences, fragments) from datasets.
Saves cleaned versions to data/processed/

Usage:
    python scripts/clean_data.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
from finsentiment.config import RAW_DATA_DIR, DATA_DIR


PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


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


def clean_phrasebank():
    """Clean Financial PhraseBank dataset."""
    print("\n[1/3] Cleaning PhraseBank...")
    
    # Load from raw
    file_path = RAW_DATA_DIR / "phrasebank.csv"
    if not file_path.exists():
        print(f"ERROR: {file_path} not found. Run training first to download datasets.")
        return 0
    
    df = pd.read_csv(file_path)
    original_count = len(df)
    
    # Filter garbage (check 'text' column)
    df['is_garbage'] = df['text'].apply(is_garbage)
    df_clean = df[~df['is_garbage']].copy()
    df_clean = df_clean.drop(columns=['is_garbage'])
    
    removed = original_count - len(df_clean)
    print(f"      Removed {removed} garbage samples ({removed/original_count*100:.1f}%)")
    print(f"      Kept {len(df_clean)} samples")
    
    # Save
    output_path = PROCESSED_DIR / "phrasebank_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"      Saved to {output_path}")
    
    return removed


def clean_twitter():
    """Clean Twitter Financial News dataset."""
    print("\n[2/3] Cleaning Twitter...")
    
    # Load from raw
    file_path = RAW_DATA_DIR / "twitter.csv"
    if not file_path.exists():
        print(f"ERROR: {file_path} not found. Run training first to download datasets.")
        return 0
    
    df = pd.read_csv(file_path)
    original_count = len(df)
    
    # Filter garbage
    df['is_garbage'] = df['text'].apply(is_garbage)
    df_clean = df[~df['is_garbage']].copy()
    df_clean = df_clean.drop(columns=['is_garbage'])
    
    removed = original_count - len(df_clean)
    print(f"      Removed {removed} garbage samples ({removed/original_count*100:.1f}%)")
    print(f"      Kept {len(df_clean)} samples")
    
    # Save
    output_path = PROCESSED_DIR / "twitter_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"      Saved to {output_path}")
    
    return removed


def clean_fiqa():
    """Clean FiQA dataset."""
    print("\n[3/3] Cleaning FiQA...")
    
    # Load from raw
    file_path = RAW_DATA_DIR / "fiqa.csv"
    if not file_path.exists():
        print(f"ERROR: {file_path} not found. Run training first to download datasets.")
        return 0
    
    df = pd.read_csv(file_path)
    original_count = len(df)
    
    # Filter garbage (FiQA uses 'text' column)
    df['is_garbage'] = df['text'].apply(is_garbage)
    df_clean = df[~df['is_garbage']].copy()
    df_clean = df_clean.drop(columns=['is_garbage'])
    
    removed = original_count - len(df_clean)
    print(f"      Removed {removed} garbage samples ({removed/original_count*100:.1f}%)")
    print(f"      Kept {len(df_clean)} samples")
    
    # Save
    output_path = PROCESSED_DIR / "fiqa_clean.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"      Saved to {output_path}")
    
    return removed


def main():
    print("="*60)
    print("Data Cleaning Pipeline")
    print("="*60)
    
    total_removed = 0
    total_removed += clean_phrasebank()
    total_removed += clean_twitter()
    total_removed += clean_fiqa()
    
    print(f"\n{'='*60}")
    print(f"✓ Cleaning complete!")
    print(f"Total garbage removed: {total_removed} samples")
    print(f"Cleaned datasets saved to: {PROCESSED_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
