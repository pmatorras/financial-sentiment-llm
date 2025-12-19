"""Main training script."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from financial_sentiment_llm.config import MODEL_PATH, MODEL_NAME, set_seed  
from financial_sentiment_llm.preprocessing import prepare_combined_dataset
from financial_sentiment_llm.dataset import FinancialSentimentDataset
from financial_sentiment_llm.model import FinancialSentimentModel
from financial_sentiment_llm.train import train_model

def main():
    # Set seed for reproducibility
    set_seed()

    # Prepare data
    data_splits = prepare_combined_dataset()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
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
    print(f"\n Using device: {device}")  # ADD THIS LINE
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")  # ADD THIS LINE
    train_model(model, train_loader, val_loader, device=device, epochs=3)

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n✓ Training complete! Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
