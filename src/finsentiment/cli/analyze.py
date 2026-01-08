"""Error analysis command handler."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd

from finsentiment.config import MODEL_NAME, BATCH_SIZE, get_model_path
from finsentiment.datasets.preprocessing import prepare_combined_dataset
from finsentiment.evaluation.metrics import evaluate_model
from finsentiment.datasets import get_dataset_class
from finsentiment.modeling import get_model_class

def execute(args):
    """
    Execute analysis command.
    
    Args:
        args: Parsed command-line arguments
    """
    if args.type == 'false_positives':
        analyze_false_positives(args)
    else:
        print(f"Unknown analysis type: {args.type}")

def analyze_false_positives(args):
    """Analyze False Positive errors (Negative -> Positive)."""
    is_multi_task = (args.model_type == 'multi')
    
    print(f"\n{'='*60}")
    print(f"Analyzing False Positives (GT: Negative -> Pred: Positive)")
    print(f"Model Type: {args.model_type}")
    print(f"{'='*60}")
    
    # Load Data
    print("\nLoading test data...")
    data_splits = prepare_combined_dataset(multi_task=is_multi_task)
    test_df = data_splits['test'].copy()
    
    # Setup Model & Data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    DatasetClass = get_dataset_class(multi_task=is_multi_task)
    test_dataset = DatasetClass(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load Checkpoint
    checkpoint_path = args.checkpoint or get_model_path(args.model_type)
    print(f"Loading model from: {checkpoint_path}")
    
    ModelClass = get_model_class(multi_task=is_multi_task)
    model = ModelClass()
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    except FileNotFoundError:
        print(f"Error: Could not find model at {checkpoint_path}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Inference
    print("Running inference...")
    results = evaluate_model(model, test_loader, device, is_multi_task=is_multi_task)
    test_df['prediction'] = results['predictions']
    
    # Filter Errors: Label=0 (Negative) -> Pred=2 (Positive)
    errors = test_df[
        (test_df['label'] == 0) & 
        (test_df['prediction'] == 2)
    ]
    
    print(f"\nFound {len(errors)} False Positive samples.")
    print("-" * 60)
    
    # Display Results
    for idx, row in errors.iterrows():
        print(f"\n[Sample #{idx}] Source: {row['source']}")
        print(f"Text: {row['text']}")
        if 'score' in row:
            print(f"Original Score: {row['score']}")
            
    # Save Results
    output_file = "analysis_false_positives.csv"
    errors.to_csv(output_file, index=False)
    print(f"\nSaved detailed list to {output_file}")
