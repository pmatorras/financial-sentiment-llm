"""Model evaluation utilities."""

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def evaluate_model(model, test_loader, device='cuda', is_multi_task=False):
    """
    Evaluate model on test set.
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
            
            if is_multi_task:
                # --- Multi-Task Logic ---
                task_types = batch['task_type']
                
                # Prepare a tensor to hold the final class predictions for this batch
                batch_preds = torch.zeros(len(input_ids), dtype=torch.long)
                
                # Identify which samples in the batch are classification tasks
                cls_mask = torch.tensor([t == 'classification' for t in task_types])
                
                if cls_mask.any():
                    # Forward pass only for classification samples
                    logits = model(
                        input_ids[cls_mask], 
                        attention_mask[cls_mask], 
                        task_type='classification'
                    )
                    # Standard Argmax to get class 0, 1, or 2
                    batch_preds[cls_mask] = torch.argmax(logits, dim=1).cpu()
                
                # Identify which samples are regression tasks
                reg_mask = torch.tensor([t == 'regression' for t in task_types])
                
                if reg_mask.any():
                    # Forward pass only for regression samples
                    scores = model(
                        input_ids[reg_mask], 
                        attention_mask[reg_mask], 
                        task_type='regression'
                    )
                    scores = scores.cpu()
                    # Convert continuous scores to discrete labels [0, 1, 2]                    
                    reg_labels = torch.zeros_like(scores, dtype=torch.long)
                    
                    # Default is 0 (Negative), so we only update for Neutral and Positive
                    #reg_labels[scores > 0.395] = 2       # Positive
                    #reg_labels[(scores >= 0.0) & (scores <= 0.395)] = 1  # Neutral
                    reg_labels[scores > 0.1] = 2       # Positive
                    reg_labels[(scores >= -0.1) & (scores <= 0.1)] = 1  # Neutral
                    batch_preds[reg_mask] = reg_labels
                
                predictions = batch_preds

            else:
                # Single-task Logic (Original)
                logits = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1).cpu()
            
            all_preds.extend(predictions.numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return {'predictions': all_preds, 'labels': all_labels}


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
            # Overall accuracy for this source
            accuracy = (source_df['prediction'] == source_df['label']).mean()
            
            # Per-class metrics for this source
            source_labels = source_df['label'].values
            source_preds = source_df['prediction'].values
            
            print(f"\n{source.upper()} ({len(source_df)} samples) - Overall: {accuracy:.2%}")
            
            # Generate classification report just for this source
            try:
                report = classification_report(
                    source_labels, 
                    source_preds, 
                    target_names=label_names,
                    output_dict=True,
                    zero_division=0
                )
                
                # Print per-class breakdown
                print(f"  Class Breakdown:")
                for label in label_names:
                    if label in report:
                        metrics = report[label]
                        support = int(metrics['support'])
                        print(f"    {label:8s}: P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1-score']:.2f} (n={support})")
                        
            except Exception as e:
                print(f"  Could not generate detailed metrics: {e}")
