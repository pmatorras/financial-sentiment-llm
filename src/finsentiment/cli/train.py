"""Train command handler."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finsentiment.config import (
    MODEL_NAME,
    NUM_EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    MODELS_DIR,
    set_seed,
    get_model_path
)
from finsentiment.datasets.preprocessing import prepare_combined_dataset
from finsentiment.datasets.dataset_single import FinancialSentimentDataset
from finsentiment.modeling.single_task import FinancialSentimentModel
from finsentiment.training.trainer_single import train_model


def execute(args):
    """
    Execute train command.
    
    Args:
        args: Parsed command-line arguments with:
            - model_type: 'single' or 'multi'
            - epochs: Number of training epochs
            - batch_size: Batch size for training
            - lr: Learning rate
    """
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = FinancialSentimentModel()
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = train_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=PATIENCE,
    )
    
    # Save model
    model_path = get_model_path(args.model_type)
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Training complete! Model saved to {model_path}")
