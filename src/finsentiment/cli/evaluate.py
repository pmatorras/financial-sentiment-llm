"""Evaluate command handler."""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import datetime
from peft import PeftModel
from finsentiment.config import (
    BATCH_SIZE, 
    get_model_path,
    get_model_config
)
from finsentiment.datasets.preprocessing import prepare_combined_dataset
from finsentiment.evaluation.metrics import evaluate_model, print_evaluation_results, print_and_log

from finsentiment.datasets import FinancialSentimentDataset
from finsentiment.modeling import FinancialSentimentModel,LoRAFinancialSentimentModel

def execute(args):
    """
    Execute evaluate command.
    
    Args:
        args: Parsed command-line arguments with:
            - model_type: 'single' or 'multi'
            - checkpoint: Path to model checkpoint
    """
    log_file = f"logs/evaluation_{args.model}_{args.model_type}.txt" if args.write_log else None
    is_multi_task = (args.model_type == 'multi')
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

    print_and_log(f"\n{'='*60}", log_file)
    print_and_log(f"Evaluating {args.model_type}-task {args.model} model", log_file)
    print_and_log(timestamp, log_file)
    print_and_log(f"{'='*60}", log_file)

    # Load data
    print_and_log("\nLoading test data...", log_file)
    data_splits = prepare_combined_dataset(multi_task=is_multi_task)
    test_df = data_splits['test']
    model_config = get_model_config(args.model)
    model_name = model_config['base_model']
    # Setup tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = FinancialSentimentDataset(test_df, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    checkpoint_path = args.checkpoint or get_model_path(model_name=args.model, model_type=args.model_type)

    # Load model
    print_and_log(f"Loading model from: {checkpoint_path}", log_file)
    if model_config['lora_config']:
        # Initialize STANDARD Base Model (No LoRA yet)

        print_and_log("Initializing base model for LoRA attachment...", log_file)
        model = FinancialSentimentModel(model_name=model_name)
        
        # Attach the Trained LoRA Adapter
        adapter_path = checkpoint_path.parent / f"{args.model}_{args.model_type}_adapter"
        print_and_log(f"Loading LoRA adapters from: {adapter_path}", log_file)
        
        # Base weights frozen, Adapter weights loaded from disk
        model.encoder = PeftModel.from_pretrained(
            model.encoder, 
            str(adapter_path)
        )

        # Load Task Heads
        heads_state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.classifier.load_state_dict(heads_state_dict['classifier'])
        model.regressor.load_state_dict(heads_state_dict['regressor'])
    else:
        model = FinancialSentimentModel(model_name=model_name)    
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print_and_log(f"Using device: {device}", log_file)
    
    # Evaluate
    print_and_log("\nRunning evaluation...", log_file)
    results = evaluate_model(model, test_loader, device, is_multi_task=is_multi_task)
    ground_truth = test_df['label'].values
    # Print results
    print_evaluation_results(ground_truth, results['predictions'], test_df, log_file=log_file)
    
    print_and_log(f"\n{'='*60}", log_file)
    print_and_log("✓ Evaluation complete!", log_file)
    print_and_log(f"{'='*60}\n", log_file)
    print(f"Results saved to {log_file}")