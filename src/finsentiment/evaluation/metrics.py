"""Model evaluation utilities."""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model to evaluate
        test_loader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        dict: Contains predictions, labels, and metrics
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return {
        'predictions': all_preds,
        'labels': all_labels
    }


def print_evaluation_results(labels, predictions, test_df):
    """
    Print formatted evaluation results.
    
    Args:
        labels: Ground truth labels
        predictions: Model predictions
        test_df: Test dataframe with source information
    """
    label_names = ['negative', 'neutral', 'positive']
    
    # Overall metrics
    print("\n" + "="*60)
    print("=== Overall Test Set Performance ===")
    print("="*60)
    print(classification_report(labels, predictions, target_names=label_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)
    
    # Per-source performance
    print("\n" + "="*60)
    print("=== Performance by Data Source ===")
    print("="*60)
    test_df['prediction'] = predictions
    
    for source in ['phrasebank', 'twitter', 'fiqa']:
        source_df = test_df[test_df['source'] == source]
        if len(source_df) > 0:
            accuracy = (source_df['prediction'] == source_df['label']).mean()
            print(f"{source.capitalize():12s}: {accuracy:.2%} ({len(source_df)} samples)")
