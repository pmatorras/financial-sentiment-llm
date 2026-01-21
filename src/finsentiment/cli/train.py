"""Train command handler."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from finsentiment.config import (
    SEED,
    PATIENCE,
    CLEAN_DATA_DEFAULT,
    CLAS_WEIGHT,
    REGR_WEIGHT,
    set_seed,
    get_model_path,
    get_model_config
)
from finsentiment.datasets import FinancialSentimentDataset, prepare_combined_dataset
from finsentiment.modeling import FinancialSentimentModel, LoRAFinancialSentimentModel
from finsentiment.training import train_multi_task_model

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
    model_config = get_model_config(args.model)
    lora_title = ' (LORA)' if model_config['lora_config'] else ''
    model_name = model_config['base_model']
    model_path = get_model_path(model_name=args.model, model_type=args.model_type)
    print(f"Running on: Mode={args.model_type}, MultiTask={is_multi_task}")
    print(f"Base Model: {model_name} {lora_title}")
    print(f"Model will be saved to: {model_path}")

    # Prepare data
    set_seed()
    data_splits = prepare_combined_dataset(seed=SEED, multi_task=is_multi_task, clean_data=CLEAN_DATA_DEFAULT)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    train_dataset = FinancialSentimentDataset(data_splits['train'], tokenizer)
    val_dataset = FinancialSentimentDataset(data_splits['val'], tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Initialize model
    if model_config['lora_config']:
        model = LoRAFinancialSentimentModel(
            model_name=model_name,
            lora_config=model_config['lora_config']
        )
    else:
        model = FinancialSentimentModel(model_name=model_name)    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = train_multi_task_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        patience=PATIENCE,
        classification_weight=CLAS_WEIGHT,
        regression_weight=REGR_WEIGHT,
        debug=args.debug
    )
    
    # Save model
    if model_config['lora_config']:
        adapter_path = model_path.parent / f"{args.model}_{args.model_type}_adapter"
        model.encoder.save_pretrained(adapter_path)
        torch.save({
            'classifier': model.classifier.state_dict(),
            'regressor': model.regressor.state_dict()
        }, model_path)
        print(f"\n✓ Training complete! LoRA adapter saved to {adapter_path}, heads to {model_path}")
    else:
        torch.save(model.state_dict(), model_path)
        print(f"\n✓ Training complete! Model saved to {model_path}")
