"""Main training script."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from financial_sentiment_llm.config import MODELS_DIR
from financial_sentiment_llm.preprocessing import prepare_combined_dataset
from financial_sentiment_llm.dataset import FinancialSentimentDataset
from financial_sentiment_llm.model import FinancialSentimentModel
from financial_sentiment_llm.train import train_model

def main():
    # Prepare data
    data_splits = prepare_combined_dataset()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    train_dataset = FinancialSentimentDataset(data_splits['train'], tokenizer)
    val_dataset = FinancialSentimentDataset(data_splits['val'], tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = FinancialSentimentModel()
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_model(model, train_loader, val_loader, device=device, epochs=3)
    model_path = MODELS_DIR / 'sentiment_model.pt'

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Training complete! Model saved to {model_path}")

if __name__ == "__main__":
    main()
