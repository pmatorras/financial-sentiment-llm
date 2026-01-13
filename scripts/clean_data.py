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
from finsentiment.datasets.clean_data import clean_phrasebank, clean_twitter, clean_fiqa
from finsentiment.config import PROCESSED_DATA_DIR
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
    print(f"Cleaned datasets saved to: {PROCESSED_DATA_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
