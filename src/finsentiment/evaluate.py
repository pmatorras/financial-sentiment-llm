"""Evaluate model on test set."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

from finsentiment.config import MODEL_PATH, MODEL_NAME
from finsentiment.preprocessing import prepare_combined_dataset
from finsentiment.dataset import FinancialSentimentDataset
from finsentiment.model import FinancialSentimentModel

def evaluate_model():
    # Load data
    data_splits = prepare_combined_dataset()
    test_df = data_splits['test']
    
    # Setup
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = FinancialSentimentDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Load model
    model = FinancialSentimentModel()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    # Predictions
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
    
    # Metrics
    label_names = ['negative', 'neutral', 'positive']
    print("\n=== Test Set Evaluation ===")
    print(classification_report(all_labels, all_preds, target_names=label_names))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)
    
    # Per-source performance
    print("\n=== Performance by Data Source ===")
    test_df['prediction'] = all_preds
    for source in ['phrasebank', 'twitter', 'fiqa']:
        source_df = test_df[test_df['source'] == source]
        if len(source_df) > 0:
            accuracy = (source_df['prediction'] == source_df['label']).mean()
            print(f"{source}: {accuracy:.2%} ({len(source_df)} samples)")

if __name__ == "__main__":
    evaluate_model()
