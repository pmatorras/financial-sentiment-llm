"""
scripts/apply_relabels.py
Apply manual label corrections from data/relabel.csv to processed datasets.

Usage:
    1. Create data/relabel.csv with corrections
    2. Run: python scripts/apply_relabels.py
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import pandas as pd
from finsentiment.config import DATA_DIR


PROCESSED_DIR = DATA_DIR / "processed"
RELABEL_PATH = DATA_DIR / "relabel.csv"


def apply_relabels():
    """Apply relabeling corrections to cleaned datasets."""
    
    # Check if relabel file exists
    if not RELABEL_PATH.exists():
        print(f"ERROR: Relabel file not found at {RELABEL_PATH}")
        print("\nCreate data/relabel.csv with format:")
        print("source,text,old_label,new_label")
        print('twitter,"Sample text...",2,0')
        return
    
    # Load relabel instructions
    relabels = pd.read_csv(RELABEL_PATH)
    print(f"Loaded {len(relabels)} relabeling instructions")
    
    # Map source to filename
    source_map = {
        'phrasebank': 'phrasebank_clean.csv',
        'twitter': 'twitter_clean.csv',
        'fiqa': 'fiqa_clean.csv'
    }
    
    # Group relabels by source
    relabels_by_source = relabels.groupby('source')
    
    total_applied = 0
    
    for source, group in relabels_by_source:
        if source not in source_map:
            print(f"WARNING: Unknown source '{source}', skipping...")
            continue
        
        filename = source_map[source]
        filepath = PROCESSED_DIR / filename
        
        if not filepath.exists():
            print(f"WARNING: {filepath} not found. Run clean_data.py first.")
            continue
        
        print(f"\n[{source}] Loading {filename}...")
        df = pd.read_csv(filepath)
        
        # Determine text column name
        text_col = 'text'
        label_col = 'label' if source == 'twitter' else 'sentiment_score' if source == 'fiqa' else 'label'
        
        applied = 0
        for _, row in group.iterrows():
            # Find matching rows
            mask = df[text_col] == row['text']
            
            if mask.sum() == 0:
                print(f"  WARNING: Text not found: {row['text'][:50]}...")
                continue
            
            # Apply relabel
            df.loc[mask, label_col] = row['new_label']
            applied += mask.sum()
        
        print(f"  Applied {applied} label corrections")
        total_applied += applied
        
        # Save updated file
        df.to_csv(filepath, index=False)
        print(f"  Updated {filepath}")
    
    print(f"\n{'='*60}")
    print(f"✓ Relabeling complete!")
    print(f"Total corrections applied: {total_applied}")
    print(f"{'='*60}\n")


def main():
    print("="*60)
    print("Apply Manual Relabeling")
    print("="*60)
    
    apply_relabels()


if __name__ == "__main__":
    main()
