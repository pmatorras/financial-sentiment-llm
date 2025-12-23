"""Evaluate command handler."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finsentiment.config import MODEL_NAME, BATCH_SIZE, get_model_path
from finsentiment.datasets.preprocessing import prepare_combined_dataset
from finsentiment.datasets.dataset_single import FinancialSentimentDataset
from finsentiment.modeling.single_task import FinancialSentimentModel
from finsentiment.evaluation.metrics import evaluate_model, print_evaluation_results


def execute(args):
    """
    Execute evaluate command.
    
    Args:
        args: Parsed command-line arguments with:
            - model_type: 'single' or 'multi'
            - checkpoint: Path to model checkpoint
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {args.model_type}-task model")
    print(f"{'='*60}")
    
    # Load data
    print("\nLoading test data...")
    data_splits = prepare_combined_dataset()
    test_df = data_splits['test']
    
    # Setup tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = FinancialSentimentDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    checkpoint_path = args.checkpoint or get_model_path(args.model_type)

    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model = FinancialSentimentModel()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    # Print results
    print_evaluation_results(results['labels'], results['predictions'], test_df)
    
    print(f"\n{'='*60}")
    print("✓ Evaluation complete!")
    print(f"{'='*60}\n")
