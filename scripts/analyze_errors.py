"""
scripts/analyze_errors.py
Comprehensive error analysis across all misclassification types.
Saves results to data/analysis/ folder.

Usage:
    python scripts/analyze_errors.py
"""

import sys
from pathlib import Path

# Add src to path for imports
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd

from finsentiment.config import MODEL_NAME, BATCH_SIZE, get_model_path, DATA_DIR
from finsentiment.datasets.preprocessing import prepare_combined_dataset
from finsentiment.evaluation.metrics import evaluate_model
from finsentiment.datasets import get_dataset_class
from finsentiment.modeling import get_model_class


# Setup analysis output directory
ANALYSIS_DIR = DATA_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


def get_source_breakdown(errors_df):
    """Get compact source breakdown string."""
    if len(errors_df) == 0:
        return ""
    
    source_counts = {
        'P': len(errors_df[errors_df['source'] == 'phrasebank']),
        'T': len(errors_df[errors_df['source'] == 'twitter']),
        'F': len(errors_df[errors_df['source'] == 'fiqa'])
    }
    
    # Only include non-zero sources
    parts = [f"{count}{label}" for label, count in source_counts.items() if count > 0]
    return f"({', '.join(parts)})" if parts else ""


def main():
    print("="*60)
    print("Comprehensive Error Analysis")
    print("="*60)
    
    # Load test data
    print("\n[1/4] Loading test data...")
    data_splits = prepare_combined_dataset(multi_task=True)
    test_df = data_splits['test'].copy()
    print(f"      Loaded {len(test_df)} test samples")
    
    # Setup model
    print("\n[2/4] Loading Multi-Task model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    DatasetClass = get_dataset_class(multi_task=True)
    test_dataset = DatasetClass(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    checkpoint_path = get_model_path('multi')
    if not checkpoint_path.exists():
        print(f"ERROR: Model not found at {checkpoint_path}")
        print("Please train the multi-task model first: python -m finsentiment train")
        return
    
    ModelClass = get_model_class(multi_task=True)
    model = ModelClass()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"      Using device: {device}")
    
    # Run inference
    print("\n[3/4] Running inference...")
    results = evaluate_model(model, test_loader, device, is_multi_task=True)
    test_df['prediction'] = results['predictions']
    
    # Extract all error types
    print("\n[4/4] Analyzing error patterns...")
    
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Define error categories
    error_types = {
        'critical': [
            (0, 2, 'neg_to_pos'),  # Negative → Positive
            (2, 0, 'pos_to_neg'),  # Positive → Negative
        ],
        'moderate': [
            (0, 1, 'neg_to_neu'),  # Negative → Neutral
            (1, 0, 'neu_to_neg'),  # Neutral → Negative
            (1, 2, 'neu_to_pos'),  # Neutral → Positive
            (2, 1, 'pos_to_neu'),  # Positive → Neutral
        ]
    }
    
    print(f"\n{'='*60}")
    print("ERROR SUMMARY")
    print(f"{'='*60}\n")
    
    saved_files = []
    
    # Process critical errors
    print("CRITICAL ERRORS (Sentiment Flips):")
    for true_label, pred_label, name in error_types['critical']:
        errors = test_df[(test_df['label'] == true_label) & 
                        (test_df['prediction'] == pred_label)]
        count = len(errors)
        breakdown = get_source_breakdown(errors)
        
        print(f"  {label_map[true_label].capitalize()} → {label_map[pred_label].capitalize()}: "
              f"{count:3d} samples {breakdown}")
        
        if count > 0:
            filename = f"errors_{name}.csv"
            filepath = ANALYSIS_DIR / filename
            errors[['source', 'text', 'label', 'prediction']].to_csv(filepath, index=False)
            saved_files.append(filename)
    
    # Process moderate errors
    print("\nMODERATE ERRORS (Partial Misses):")
    for true_label, pred_label, name in error_types['moderate']:
        errors = test_df[(test_df['label'] == true_label) & 
                        (test_df['prediction'] == pred_label)]
        count = len(errors)
        breakdown = get_source_breakdown(errors)
        
        print(f"  {label_map[true_label].capitalize()} → {label_map[pred_label].capitalize()}: "
              f"{count:3d} samples {breakdown}")
        
        if count > 0:
            filename = f"errors_{name}.csv"
            filepath = ANALYSIS_DIR / filename
            errors[['source', 'text', 'label', 'prediction']].to_csv(filepath, index=False)
            saved_files.append(filename)
    
    # Overall stats
    total_errors = len(test_df[test_df['label'] != test_df['prediction']])
    accuracy = (len(test_df) - total_errors) / len(test_df)
    
    print(f"\n{'='*60}")
    print(f"Total Errors: {total_errors}/{len(test_df)} ({(1-accuracy)*100:.1f}%)")
    print(f"Accuracy: {accuracy*100:.1f}%")
    print(f"\n✓ Saved {len(saved_files)} error files to: {ANALYSIS_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
