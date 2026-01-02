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
from finsentiment.datasets import prepare_combined_dataset, get_dataset_class
from finsentiment.modeling import get_model_class
from finsentiment.training import get_trainer_function

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
    # Get configuration and components
    is_multi_task = (args.model_type == 'multi')
    print(f"Running on: Mode={args.model_type}, MultiTask={is_multi_task}")

    DatasetClass = get_dataset_class(multi_task=is_multi_task)
    ModelClass = get_model_class(multi_task=is_multi_task)
    train_fn = get_trainer_function(multi_task=is_multi_task)
    # Prepare data
    set_seed()
    data_splits = prepare_combined_dataset(multi_task=is_multi_task)
    
    # Load tokenizer


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    
    train_dataset = DatasetClass(data_splits['train'], tokenizer)
    val_dataset = DatasetClass(data_splits['val'], tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    model = ModelClass()
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = train_fn(
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
