"""Argument parser setup."""
import argparse
from finsentiment.config import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE
)

def create_parser():
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        prog='finsentiment',
        description='Financial sentiment analysis with LLMs'
    )
    
    # Global arguments
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model-type', choices=['single', 'multi'], default='multi')
    train_parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    train_parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    train_parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('--model-type', choices=['single', 'multi'], default='single')
    eval_parser.add_argument('--checkpoint', default=None, help='Model checkpoint')

    # Analyze (New Subcommand)
    analyze_parser = subparsers.add_parser('analyze', help='Run error analysis')
    analyze_parser.add_argument('--type', default='false_positives', help='Analysis type to run')
    analyze_parser.add_argument('--model-type', choices=['single', 'multi'], default='multi')
    analyze_parser.add_argument('--checkpoint', default=None, help='Model checkpoint')
    
    return parser
